"""A Python host for web/soar/raymarch.wgsl — the one renderer core.

There used to be two implementations of the same look: a numba kernel here
and the WGSL the browser runs. Two implementations of one thing diverge, and
they did — periodic domains and distance LOD landed in the shader and never
came back to the CPU, so `witness` could not render what soar renders. This
module deletes the second implementation instead of maintaining it: it drives
the browser's own shader through wgpu-native, so there is exactly one place
the look is defined.

What that costs: rendering now needs a GPU. That is deliberate.

The parity that matters is the 368-byte uniform block. The shader is shared
by construction, so the only thing that can drift is how the host fills it in
— which is why tests/test_uniform_parity.py diffs this module's packing
against the browser's own packUniforms running under node.

Conventions inherited from the shader, all easy to get subtly wrong:
  * FOV is HORIZONTAL (raymarch.wgsl's file header says vertical; it is
    stale, the struct comment and fs_main agree it is horizontal).
  * The volume texture is indexed width=z, height=y, depth=x, which means a
    C-order (nx+2, ny+2, nz+2) numpy array is already in the right byte
    order. Do not transpose.
  * Relative camera z is anchored to the physical surface, not the AABB
    floor: rel_z = -1 is sea level.
  * Booleans travel as 0.0/1.0 floats and are tested with `> 0.5`.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from . import look

# The shader and ocean tiles live here in the package — the one copy. The
# browser reaches the same files through symlinks at web/soar/raymarch.wgsl
# and web/soar/ocean, so there is still exactly one renderer core and the
# look cannot drift between the Python host and the web app.
SOAR_DIR = Path(__file__).resolve().parent / "soar"
SHADER_PATH = SOAR_DIR / "raymarch.wgsl"
OCEAN_DIR = SOAR_DIR / "ocean"
# The night-city surface tile. Same byte format and same binding as the ocean
# tile (rgba16float mip chain, binding 3) — under CITY the shader reads it as
# a building-density field instead of a water normal field, so exactly one of
# the two is ever loaded. Its meta names the cell size "cell_m" where the
# ocean's names it "dx_m"; the keys "n" and "mips" are common to both.
CITY_DIR = SOAR_DIR / "city"
# The Sonoran desert tile (DESERT, 2026-09-23): terrain height, its two
# slopes and a material channel, same byte format and same binding again.
DESERT_DIR = SOAR_DIR / "desert"

# What the world stands on. "ocean" and "desert" are day scenes over the
# same sky; "city" is the night city. Each compiles its own specialization
# and binds its own tile at binding 3 — see `specialize` and SoarRenderer.
SURFACES = ("ocean", "desert", "city")
SURFACE_DIRS = {"ocean": OCEAN_DIR, "desert": DESERT_DIR, "city": CITY_DIR}
# The meta.json schema each surface tile must carry: the layout the shader
# decodes it with. Mirrors web/soar/constants.js SURFACE_TILE_SCHEMAS (pinned
# by tests/test_city_lots.py); a tile of another schema is refused, because
# the city's v1 tile decoded as v2 raises a different city without an error.
SURFACE_TILE_SCHEMAS = {
    "ocean": "cloudyview.web.ocean.v2",
    "desert": "cloudyview.web.desert.v2",
    "city": "cloudyview.web.city.v2",
}


def check_surface(surface: str) -> str:
    """The surface name, or a ValueError naming the ones that exist."""
    if surface not in SURFACES:
        raise ValueError(
            f"surface must be one of {', '.join(SURFACES)}; got {surface!r}.")
    return surface


def surface_cell_m(surface: str, meta: dict) -> float:
    """A surface tile's texel size from its meta.json.

    The one place the key difference lives: the ocean names it "dx_m" (a
    sampling step of a wave field), the city and the desert "cell_m".
    """
    key = "dx_m" if check_surface(surface) == "ocean" else "cell_m"
    if key not in meta:
        raise KeyError(f"the {surface} tile's meta.json has no {key!r}.")
    return float(meta[key])


# Channels per texel of each surface-tile format. The ocean and the city are
# rgba16float; the desert is rg16float (height and lightness).
SURFACE_FORMAT_CHANNELS = {"rgba16float": 4, "rg16float": 2}


def surface_level_files(meta: dict) -> list:
    """The file of each mip level: named in the meta, or the legacy scheme."""
    if "files" in meta:
        return list(meta["files"])
    return [f"fif_mip{level}.bin" for level in range(int(meta["mips"]))]


def decode_surface_level(raw: bytes, side: int, meta: dict) -> bytes:
    """One stored mip level -> the interleaved texel bytes the texture wants.

    A level is stored either as the texels themselves, or (encoding
    "planar-byte-shuffle", the desert's) as each channel's plane with every
    half float's low bytes before its high bytes — the arrangement that lets
    gzip see a height field's smoothness. `raw` is already decompressed.
    web/soar/scene.js decodeSurfaceLevel is the twin.
    """
    channels = SURFACE_FORMAT_CHANNELS[meta.get("format", "rgba16float")]
    expected = side * side * channels * 2
    if len(raw) != expected:
        raise RuntimeError(
            f"a {side}x{side} {meta.get('format', 'rgba16float')} level is "
            f"{expected} bytes; this one is {len(raw)}.")
    encoding = meta.get("encoding", "interleaved")
    if encoding == "interleaved":
        return raw
    if encoding != "planar-byte-shuffle":
        raise RuntimeError(f"unknown surface-tile encoding {encoding!r}.")
    b = np.frombuffer(raw, np.uint8).reshape(channels, 2, side * side)
    return np.ascontiguousarray(b.transpose(2, 0, 1)).tobytes()
def surface_period_m(surface: str, meta: dict) -> float:
    """The period the surface frame folds at (SceneState.surface_period_m).

    The city and the desert are pure functions of their tile, so they fold at
    the tile. The ocean's swell trains are periodic over several tiles
    (raymarch.wgsl SWELL_PERIOD_M), and its meta.json says how many metres.
    """
    key = "surface_period_m" if check_surface(surface) == "ocean" \
        else "tile_extent_m"
    if key not in meta:
        raise KeyError(f"the {surface} tile's meta.json has no {key!r}.")
    return float(meta[key])


# The surfaces whose shader reads the scene clock, and so whose clock runs:
# the sea (its waves) and the city (its steam). The desert reads none, and a
# running clock there would only keep a parked view from converging.
CLOCK_SURFACES = ("ocean", "city")


def surface_clock_period_s(surface: str, meta: dict) -> Optional[float]:
    """The period a surface's clock wraps at (SceneState.clock_period_s), from
    its meta.json — where its animation repeats exactly, so the wrap is
    invisible and float32 never loses the cadence — or None for a surface
    whose clock is always 0."""
    if check_surface(surface) not in CLOCK_SURFACES:
        return None
    if "clock_period_s" not in meta:
        raise KeyError(f"the {surface} tile's meta.json has no "
                       "'clock_period_s'.")
    return float(meta["clock_period_s"])


@functools.lru_cache(maxsize=1)
def _sea_levels(tile_dir: str):
    """(meta, per-level float16 bytes [t][y][x][rgba]) for the moving sea.

    Decoded and mip-filtered once per process (cloudyview/sea_tile.py) — the
    renderers a test session builds all upload the same ~90 MB.
    """
    from . import sea_tile
    meta, planes = sea_tile.read(Path(tile_dir), SURFACE_TILE_SCHEMAS["ocean"])
    nt = int(meta["frames"])
    chains = [sea_tile.mip_chain(sea_tile.decode_frame(planes, t, meta))
              for t in range(nt)]
    if len(chains[0]) != int(meta["mips"]):
        raise RuntimeError(
            f"the sea decodes to {len(chains[0])} mip levels; its meta.json "
            f"says {meta['mips']}.")
    levels = [np.stack([chain[k] for chain in chains]).astype(np.float16)
              .tobytes() for k in range(len(chains[0]))]
    return meta, levels


def read_shader() -> str:
    """The shader source, or a diagnosis of why there isn't any."""
    if not SHADER_PATH.exists():
        raise FileNotFoundError(
            f"the ray-marching shader is missing: {SHADER_PATH}\n"
            "It ships inside the package (cloudyview/soar/), so this "
            "install is broken — reinstall from PyPI, or work from a "
            "checkout:\n"
            "    git clone https://github.com/thomasdewitt/cloudyview\n"
            "    cd cloudyview && uv sync\n"
            "glimpse and behold do not need the shader and still work.")
    return SHADER_PATH.read_text()

# Row 24 (the 25th) is the surface offset the water samples and the city
# frame are derived from — see raymarch.wgsl's surface_offset. Rows 25..46
# are the desert tile's own numbers (raymarch.wgsl desert_tile, written by
# desert_tile_rows; zero off the desert).
DESERT_TILE_ROW0 = 25
DESERT_TILE_ROWS = 22
UNIFORM_ROWS = DESERT_TILE_ROW0 + DESERT_TILE_ROWS
UNIFORM_NBYTES = UNIFORM_ROWS * 16          # 752; the scene key compares all of it

DEG = math.pi / 180.0

# --- defaults, mirroring web/soar/constants.js -----------------------------

DEFAULT_SUN_AZIMUTH = 20.0
DEFAULT_SUN_ELEVATION = 55.0
DEFAULT_EXPOSURE = 4.0
DEFAULT_G_HG = 0.76
DEFAULT_AMBIENT_STRENGTH = 0.15
DEFAULT_OCEAN_REFLECTANCE = (0.0020, 0.0045, 0.0126)
DEFAULT_GRADIENT_SHADING_STRENGTH = 1.50
DEFAULT_GRADIENT_COARSE_WEIGHT = 0.65
DEFAULT_GRADIENT_COARSE_RADIUS_M = 500.0
DEFAULT_DEEP_SHADOW_MS_SUPPRESSION = 0.90
DEFAULT_AMBIENT_OCCLUSION_STRENGTH = 1.00
DEFAULT_AMBIENT_OCCLUSION_FLOOR = 0.24
DEFAULT_BOUNCE_DEPTH_ATTENUATION = 0.80
APP_LIGHT_MARCH_LOD_DEGREES = 1.4
APP_VIEW_STEP_LOD_DEGREES = 0.6
# What multiplies both for a render that is not trying to hold a framerate:
# an offline still, a video frame, anything witness draws. Matches the
# browser's CAPTURE_LOD_STRENGTH and its high/max tier default, so a terminal
# render, an in-app capture and a high-tier flight all march the same.
#
# Angular LOD only ever grows the step FLOOR (dt = max(dt_base, t*tan(theta))),
# so a smaller strength costs time and can never coarsen anything: at 0.5 the
# far field is marched twice as finely as the tuned constants alone ask for.
DEFAULT_LOD_STRENGTH = 0.5
# Whether the aerial haze thins with height. Mirrors
# web/soar/constants.js DEFAULT_HAZE_HEIGHT_DEPENDENT; see ViewState below.
DEFAULT_HAZE_HEIGHT_DEPENDENT = False
STEP_VOXEL_FACTOR = 2.0
DEFAULT_MAX_LIGHT_STEPS = 512

# Tone map. 1.4 is what the numba renderer used; 2.66 is what soar shows and
# is now the default, because soar is the renderer.
TONE_MAP_GAMMA_WITNESS = 1.4
TONE_MAP_GAMMA_AS_FLOWN = 3.08
DEFAULT_TONE_MAP_GAMMA = 1.66
TONE_MAP_GAMMA_LIMITS = (1.0, 4.0)

# The extended-Reinhard white point: the exposed radiance that maps to 1.0.
# It was a shader const until it became the second thing worth reaching for
# after gamma — it is what decides whether a sunlit face is white or is a
# bright grey it cannot escape. The default is the tuned 15.0, so nothing
# moves unless a caller asks. Below 4 the whole picture clips (plain
# Reinhard's own ceiling behaviour); above 40 the curve is within a fraction
# of a percent of linear-through-the-shoulder and the slider does nothing.
DEFAULT_TONE_MAP_WHITE_POINT = 15.0
TONE_MAP_WHITE_POINT_LIMITS = (4.0, 40.0)

# Display-space contrast about mid-grey, applied after the gamma encode.
# 1.0 is exactly the identity (see raymarch.wgsl's tone_map for why the
# arithmetic is written the way it is). The range is deliberately narrow:
# past 1.6 the sky posterizes and the shadows block up, and below 0.5 the
# picture is fog.
DEFAULT_CONTRAST = 1.0
CONTRAST_LIMITS = (0.5, 1.6)

MIN_SUN_ELEVATION_DEG = 0.5
STILL_ACCUMULATE_FRAMES = 64

# --- night, for CITY -------------------------------------------------------
#
# The five spectral rows carry a sun's radiance and the sky it lights, all
# derived by look._spectral_lighting_colors from an air-mass model of a
# DAYTIME atmosphere. None of that applies at night: there is no air mass to
# redden a moon that is 4e-6 of the sun's irradiance, and the sky's own
# radiance is not scattered moonlight at all. So city mode does not scale the
# daytime pipeline down — it packs measured-ish night values straight into the
# same five rows, and look's daytime path is not called.
#
# The row layout does not move: these go where cloud_sun / ambient / horizon /
# bloom / disc go, so the shader reads the same slots either way.
NIGHT_MOON_CLOUD_COLOR = (0.34, 0.42, 0.60)     # row 13.xyz, direct beam
NIGHT_AMBIENT_TINT = (0.020, 0.026, 0.044)      # row 14.xyz, ambient fill
NIGHT_SKY_HORIZON = (0.012, 0.014, 0.022)       # row 15.xyz, haze/horizon
NIGHT_MOON_BLOOM = (0.20, 0.22, 0.26)           # row 16.xyz, circum-lunar
NIGHT_MOON_DISC = (4.0, 4.2, 4.6)               # row 17.xyz, the disc itself


# --- validation, ported from uniforms.js -----------------------------------

def _unit_interval(name: str, value: float) -> float:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]; got {value}.")
    return float(value)


def _non_negative(name: str, value: float) -> float:
    if not (float(value) >= 0.0) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite and >= 0; got {value}.")
    return float(value)


def _lod_degrees(name: str, value: float) -> float:
    if not (0.0 <= float(value) < 45.0):
        raise ValueError(f"{name} must be in [0, 45) degrees; got {value}.")
    return float(value)


def direction_from_azimuth_elevation(azimuth_deg: float,
                                     elevation_deg: float) -> np.ndarray:
    """Meteorological bearing + elevation to a unit vector (east, north, up)."""
    az_internal = (90.0 - (azimuth_deg % 360.0)) % 360.0
    el = elevation_deg * DEG
    a = az_internal * DEG
    d = np.array([math.cos(el) * math.cos(a),
                  math.cos(el) * math.sin(a),
                  math.sin(el)], dtype=np.float64)
    return d / np.linalg.norm(d)


def camera_basis(azimuth_deg: float,
                 elevation_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward/right/up for a met azimuth and elevation.

    `right` is the closed form rather than cross(forward, up): the cross
    product degenerates within a couple of degrees of straight up or down and
    snaps the horizon over. Two different angles appear here on purpose —
    `forward` uses the internal 90-az convention, `right` the raw azimuth.
    """
    forward = direction_from_azimuth_elevation(azimuth_deg, elevation_deg)
    a = azimuth_deg * DEG
    right = np.array([math.cos(a), -math.sin(a), 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up


def effective_light_transfer_split(strength: float, elevation_deg: float) -> float:
    """Fade the split out as the sun rises; zero at and above the cutoff."""
    _unit_interval("light_transfer_split_strength", strength)
    full = look.LIGHT_TRANSFER_FULL_ELEVATION_DEG
    cutoff = look.LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG
    if elevation_deg >= cutoff - 1e-6:      # the slack is load-bearing
        return 0.0
    if elevation_deg > full:
        m = (cutoff - elevation_deg) / (cutoff - full)
        return strength * m * m * (3.0 - 2.0 * m)
    return strength


# --- the uniform block -----------------------------------------------------

@dataclass
class SceneState:
    """Everything about the field, independent of where the camera is.

    `surface` is what the world stands on: "ocean", "desert" or "city". The
    desert and the city each compile in through their own constant (DESERT,
    CITY — see `specialize`). Neither adds uniform rows — the block is the
    same 752 bytes in every mode — they REINTERPRET a few of them, and the
    fields keep their ocean names because the slots are literally the
    ocean's. Under "city":

      * `ocean_z` is the ground plane; `ocean_reflectance` is unused and packs
        as zeros.
      * `ocean_fif_dx` is the city cell size in metres and `ocean_tile_extent`
        the tile's extent in metres, both from the tile's meta.json
        ("cell_m" / "tile_extent_m"); `ocean_max_lod` is its top mip.
      * `ocean_enabled` still means "draw the surface", and must be True — a
        city with the surface off is an empty ground plane, so pack_uniforms
        raises rather than render it.

    Under "desert" the same four row-9 numbers describe the desert tile,
    `ocean_z` is the datum its heights stand on, and row 8's other three
    slots carry the tile's height range and its largest band-pass excursion
    (`desert_relief_m`, from the tile's meta: height_min_m, height_max_m,
    band_max_m) — what the shader's slab tests are built from. The ocean's haze extinction (row 19.y) still applies —
    it is the aerial perspective over whatever ground there is.

    The ViewState's ocean realism knobs (glint, slope draw, mip bias) are
    simply ignored by the shader under CITY and DESERT; they are not
    validated away.
    """
    bmin: Sequence[float]
    bmax: Sequence[float]
    dt_view: float
    dt_light: float
    periodic: bool = True
    ocean_z: float = 0.0
    ocean_reflectance: Sequence[float] = DEFAULT_OCEAN_REFLECTANCE
    ocean_fif_dx: float = 0.4
    ocean_tile_extent: float = 102.4
    ocean_enabled: bool = True
    ocean_max_lod: int = 8
    nested: bool = False
    nest_bmin: Optional[Sequence[float]] = None
    nest_bmax: Optional[Sequence[float]] = None
    dt_view_nest: float = 0.0
    dt_light_nest: float = 0.0
    surface: str = "ocean"
    # Where the city tile sits in world space (m, a whole number of blocks;
    # row 8.yz). The default parks the tile's megatower district — its
    # wildest cascade excursion — under the TWP-ICE subvolume's central
    # cameras, which is where the harness flies.
    city_tile_offset_m: Sequence[float] = (75330.0, -17010.0)
    # Row 24: explicit override of the surface offset — where the surface
    # tile sits in the world this render draws (the shader derives the city
    # frame and the water uv as world minus row 24, folded into
    # [0, tile extent)). None means the scene's static offset:
    # city_tile_offset_m under CITY, the origin otherwise — which folds to
    # exactly 0.0 for a day scene, keeping the water samples bit-identical
    # to a shader with no surface frame at all. witness's --surface-origin
    # lands here so a reproduction command from a flight whose camera folds
    # moved the offset carries the value that flight rendered with.
    surface_offset_m: Optional[Sequence[float]] = None
    # Row 8.yzw under DESERT — see above. Required for the desert, and
    # refused anywhere else: without it the terrain collapses onto the datum
    # plane and still looks plausible, which is the failure worth a raise.
    desert_relief_m: Optional[Sequence[float]] = None
    # Rows 25..46 under DESERT: the tile's own tables (desert_tile_rows).
    # Required and refused alike, and for the same reason: without them the
    # trace steps with no slope bound and the far field has no colour.
    desert_tile_rows: Optional[Sequence[Sequence[float]]] = None
    # The period row 24 folds at. None is the tile extent — the city's and
    # the desert's rule, since each is a pure function of its tile. The
    # ocean's swell repeats over several tiles and a fold at the tile would
    # jump it, so an ocean scene states its own (surface_period_m(), from
    # the sea's meta.json).
    surface_period_m: Optional[float] = None
    # The surface's period in time (its meta.json clock_period_s,
    # surface_clock_period_s()): row 24.z packs the scene clock modulo it, so
    # float32 never loses the cadence. Required over the sea and the city
    # whenever the clock is not 0 (a still at clock 0 needs nothing); unused
    # over the desert, whose clock is 0. Mirrors uniforms.js.
    clock_period_s: Optional[float] = None


@dataclass
class ViewState:
    """Everything about this particular frame."""
    camera_position: Sequence[float]        # absolute metres
    azimuth: float
    elevation: float
    fov: float                              # HORIZONTAL degrees
    output_size: Tuple[int, int]
    render_size: Tuple[int, int]
    jitter: bool = True
    sun_azimuth: float = DEFAULT_SUN_AZIMUTH
    sun_elevation: float = DEFAULT_SUN_ELEVATION
    exposure: float = DEFAULT_EXPOSURE
    g_hg: float = DEFAULT_G_HG
    ambient_strength: float = DEFAULT_AMBIENT_STRENGTH
    gradient_shading_strength: float = DEFAULT_GRADIENT_SHADING_STRENGTH
    gradient_coarse_weight: float = DEFAULT_GRADIENT_COARSE_WEIGHT
    gradient_coarse_radius_m: float = DEFAULT_GRADIENT_COARSE_RADIUS_M
    deep_shadow_ms_suppression: float = DEFAULT_DEEP_SHADOW_MS_SUPPRESSION
    ambient_occlusion_strength: float = DEFAULT_AMBIENT_OCCLUSION_STRENGTH
    ambient_occlusion_floor: float = DEFAULT_AMBIENT_OCCLUSION_FLOOR
    bounce_depth_attenuation: float = DEFAULT_BOUNCE_DEPTH_ATTENUATION
    spectral_lighting_strength: float = look.SPECTRAL_LIGHTING_STRENGTH
    low_sun_sky_field_strength: float = look.LOW_SUN_SKY_FIELD_STRENGTH
    light_transfer_split_strength: float = look.LIGHT_TRANSFER_SPLIT_STRENGTH
    aerial_perspective_strength: float = look.AERIAL_PERSPECTIVE_STRENGTH
    # One number for the whole aerosol story: aerial extinction, the sky's
    # horizon wedge, the circumsolar lobe, the haze over the sea. There is
    # deliberately no per-term override — see look.DEFAULT_HAZE.
    haze: float = look.DEFAULT_HAZE
    # Whether that aerosol thins with height. Off by default (Thomas,
    # 2026-08-15): the exponential profile is the physical one, but a
    # 2.5 km scale height means an upward ray leaves the haze entirely, and
    # nothing then bounds how far it marches except the range ceiling. Uniform
    # haze puts the same extinction at every altitude, which is unphysical and
    # caps every ray at one distance — the cheapest performance lever in the
    # renderer. See raymarch.wgsl's sky_disc.w.
    haze_height_dependent: bool = DEFAULT_HAZE_HEIGHT_DEPENDENT
    ocean_realism: float = look.OCEAN_REALISM
    ocean_mip_bias: float = look.OCEAN_MIP_BIAS
    ocean_glint_strength: float = look.OCEAN_GLINT_STRENGTH
    ocean_glint_roughness: float = look.OCEAN_GLINT_ROUGHNESS
    ocean_slope_draw_fraction: float = look.OCEAN_SLOPE_DRAW_FRACTION
    ocean_sky_shadow_floor: float = look.OCEAN_SKY_SHADOW_FLOOR
    cone_stencil_theta_deg: float = look.CONE_STENCIL_THETA_DEG
    # The app's angles scaled by DEFAULT_LOD_STRENGTH — see above. Given as
    # resolved angles rather than as a strength because the shader wants
    # tan(theta) and there is no reason for two representations to exist down
    # here; callers that think in strength (witness's --lod) multiply.
    light_march_lod_degrees: float = (APP_LIGHT_MARCH_LOD_DEGREES
                                      * DEFAULT_LOD_STRENGTH)
    view_step_lod_degrees: float = (APP_VIEW_STEP_LOD_DEGREES
                                    * DEFAULT_LOD_STRENGTH)
    tone_map_gamma: float = DEFAULT_TONE_MAP_GAMMA
    tone_map_white_point: float = DEFAULT_TONE_MAP_WHITE_POINT
    contrast: float = DEFAULT_CONTRAST
    frame_index: int = 0
    subpixel: bool = False
    jitter_scale: float = 1.0
    # Read tau_sun for cloud samples from the baked cache instead of marching
    # to the sun per sample. Only meaningful after
    # SoarRenderer.bake_light_cache has run for this scene and sun; the flag
    # alone against a stale or absent cache reads the dummy zero texture and
    # renders every cloud unshadowed.
    light_cache: bool = False
    # The vertical sky-visibility march (row 23.z inverts it): off means
    # every consumer of t_sky sees a fully open sky, and buried samples
    # brighten. A cost/look toggle, on by default.
    sky_probe: bool = True
    # The simplified desert (raymarch.wgsl DESERT_LITE): a specialization of
    # a desert scene, keyed in the app on the tier (DESERT_LITE_TIERS) —
    # witness --quality sets it the same way. It has no refine stage.
    desert_lite: bool = False
    # How far out the full desert walks the explicit shrubs (raymarch.wgsl
    # DESERT_FLORA_CUT, a scale on the shrubs' cut), keyed in the app
    # on the tier (DESERT_FLORA_CUT_BY_TIER); 1.0 is the shader's own value.
    desert_flora_cut: float = 1.0
    # The scene clock (s, row 24.z): where the moving sea is in its motion.
    # The browser advances it while flying; a still names it (witness
    # --time) and a track carries it per sample.
    scene_time: float = 0.0


def pack_uniforms(state: SceneState, view: ViewState) -> np.ndarray:
    """The 752-byte block, row for row with web/soar/uniforms.js.

    Returns (47, 4) float32. Every unwritten slot stays zero, which is what
    the shader's unused components and the absent nest rows require.

    Under the city the layout is untouched and a handful of rows change
    MEANING — the surface rows describe the city tile rather than the sea, the
    sun row is the moon, and the five spectral rows carry the night palette
    instead of look's daytime air-mass model. See SceneState.
    """
    _unit_interval("jitter_scale", view.jitter_scale)
    _unit_interval("spectral_lighting_strength", view.spectral_lighting_strength)
    _unit_interval("low_sun_sky_field_strength", view.low_sun_sky_field_strength)
    _unit_interval("ocean_realism", view.ocean_realism)
    _unit_interval("ocean_sky_shadow_floor", view.ocean_sky_shadow_floor)
    _non_negative("aerial_perspective_strength", view.aerial_perspective_strength)
    _non_negative("ocean_glint_strength", view.ocean_glint_strength)
    _non_negative("ocean_glint_roughness", view.ocean_glint_roughness)
    _unit_interval("ocean_slope_draw_fraction", view.ocean_slope_draw_fraction)
    if not (look.HAZE_MIN <= view.haze <= look.HAZE_MAX):
        raise ValueError(
            f"haze must be in [{look.HAZE_MIN}, {look.HAZE_MAX}]; "
            f"got {view.haze}.")
    _lod_degrees("light_march_lod_degrees", view.light_march_lod_degrees)
    _lod_degrees("view_step_lod_degrees", view.view_step_lod_degrees)
    if not (0.0 <= view.cone_stencil_theta_deg < 90.0):
        raise ValueError("cone_stencil_theta_deg must be in [0, 90); got "
                         f"{view.cone_stencil_theta_deg}.")
    lo, hi = TONE_MAP_GAMMA_LIMITS
    if not (lo <= view.tone_map_gamma <= hi):
        raise ValueError(f"tone_map_gamma must be in [{lo}, {hi}]; got "
                         f"{view.tone_map_gamma}.")
    lo, hi = TONE_MAP_WHITE_POINT_LIMITS
    if not (lo <= view.tone_map_white_point <= hi):
        raise ValueError(f"tone_map_white_point must be in [{lo}, {hi}]; got "
                         f"{view.tone_map_white_point}.")
    lo, hi = CONTRAST_LIMITS
    if not (lo <= view.contrast <= hi):
        raise ValueError(f"contrast must be in [{lo}, {hi}]; got "
                         f"{view.contrast}.")
    # Not a clamp: a periodic light march exits only through the domain top,
    # so a sun at or below the horizon has no exit and the picture is wrong
    # in a way that still looks plausible.
    if state.periodic and view.sun_elevation <= 0.0:
        raise ValueError(
            "A periodic domain needs the sun above the horizon; got "
            f"sun_elevation={view.sun_elevation}. The light march exits "
            f"through the domain top, so keep it >= {MIN_SUN_ELEVATION_DEG}.")
    # The surface flag is what draws the city; with it off the ground plane is
    # empty and the scene is a night sky over nothing, which looks like a
    # renderer bug rather than a choice.
    check_surface(state.surface)
    city = state.surface == "city"
    desert = state.surface == "desert"
    if desert and state.desert_relief_m is None:
        raise ValueError(
            "the desert needs its relief (SceneState.desert_relief_m, from the "
            "tile's meta: soar_host.desert_relief_m(renderer.surface_meta)); "
            "without it every ray lands on the flat datum.")
    if not desert and state.desert_relief_m is not None:
        raise ValueError("desert_relief_m is the desert's; this scene is the "
                         f"{state.surface}.")
    if desert and state.desert_tile_rows is None:
        raise ValueError(
            "the desert needs its tile's tables (SceneState.desert_tile_rows, "
            "from soar_host.desert_tile_rows(renderer.surface_meta)).")
    if not desert and state.desert_tile_rows is not None:
        raise ValueError("desert_tile_rows is the desert's; this scene is the "
                         f"{state.surface}.")
    if desert and not state.ocean_enabled:
        raise ValueError(
            "the desert needs the surface enabled: ocean_enabled is the "
            "surface flag (row 9.z) and the desert is drawn regardless.")
    if city and not state.ocean_enabled:
        raise ValueError(
            "city mode needs the surface enabled: ocean_enabled is the "
            "surface flag under CITY (row 9.z) and it is False, which would "
            "render an empty ground plane.")

    forward, right, up = camera_basis(view.azimuth, view.elevation)
    # Under CITY this is the MOON: same azimuth/elevation fields, same
    # packing, same row — only the light it stands for is different.
    sun = direction_from_azimuth_elevation(view.sun_azimuth, view.sun_elevation)
    if city:
        cloud_sun, ambient, horizon, bloom, disc = (
            NIGHT_MOON_CLOUD_COLOR, NIGHT_AMBIENT_TINT, NIGHT_SKY_HORIZON,
            NIGHT_MOON_BLOOM, NIGHT_MOON_DISC)
        # Both of these are daytime-sun machinery: the low-sun sky field
        # reddens a setting sun's sky, and the light-transfer split fades in
        # as that sun drops. Neither has a night meaning, so both are off.
        low_sun_sky = 0.0
        split = 0.0
    else:
        cloud_sun, ambient, horizon, bloom, disc = look._spectral_lighting_colors(
            tuple(sun), look.SUN_COLOR, view.spectral_lighting_strength)
        low_sun_sky = view.low_sun_sky_field_strength
        split = effective_light_transfer_split(
            view.light_transfer_split_strength, view.sun_elevation)

    out_w, out_h = view.output_size
    w, h = view.render_size

    u = np.zeros((UNIFORM_ROWS, 4), dtype=np.float32)
    u[0] = (*view.camera_position, math.tan(view.fov * DEG * 0.5))
    u[1] = (*forward, out_w / out_h)
    u[2] = (*right, view.exposure)
    u[3] = (*up, 1.0 if view.jitter else 0.0)
    u[4] = (*sun, float(view.frame_index))
    u[5] = (*state.bmin, state.dt_view)
    u[6] = (*state.bmax, state.dt_light)
    u[7] = (w, h, view.g_hg, view.ambient_strength)
    # Row 8 is the surface: sea level plus the water's reflectance, or — under
    # CITY — the ground plane and the tile's world offset (which district
    # sits under this field's cameras).
    # Under DESERT the datum and the tile's relief: the desert has no
    # reflectance row, its albedos are properties of the materials and live
    # in the shader.
    if city:
        u[8] = (state.ocean_z, *state.city_tile_offset_m, 0.0)
    elif desert:
        u[8] = (state.ocean_z, *state.desert_relief_m)
        rows = np.asarray(state.desert_tile_rows, dtype=np.float32)
        if rows.shape != (DESERT_TILE_ROWS, 4):
            raise ValueError(
                f"desert_tile_rows is {rows.shape}; the shader reads "
                f"({DESERT_TILE_ROWS}, 4).")
        u[DESERT_TILE_ROW0:DESERT_TILE_ROW0 + DESERT_TILE_ROWS] = rows
    else:
        u[8] = (state.ocean_z, *state.ocean_reflectance)
    # Row 9 needs no branch: cell size, tile extent, surface flag and top mip
    # are the same four numbers about whichever tile is bound.
    u[9] = (state.ocean_fif_dx, state.ocean_tile_extent,
            1.0 if state.ocean_enabled else 0.0, float(state.ocean_max_lod))
    u[10] = (1.0 if view.subpixel else 0.0, view.jitter_scale, view.haze,
             view.tone_map_white_point)
    u[11] = (view.gradient_shading_strength, view.deep_shadow_ms_suppression,
             view.ambient_occlusion_strength, view.bounce_depth_attenuation)
    u[12] = (view.gradient_coarse_weight, view.gradient_coarse_radius_m,
             view.ambient_occlusion_floor,
             math.tan(view.cone_stencil_theta_deg * DEG))
    u[13] = (*cloud_sun, low_sun_sky)
    u[14] = (*ambient, split)
    u[15] = (*horizon, view.aerial_perspective_strength)
    u[16] = (*bloom, look.aerial_beta_per_km(view.haze) * 1e-3)
    u[17] = (*disc, look.AERIAL_SCALE_HEIGHT_M
                    if view.haze_height_dependent else 0.0)
    u[18] = (view.ocean_realism, view.ocean_mip_bias,
             view.ocean_glint_strength, view.ocean_glint_roughness)
    u[19] = (view.ocean_slope_draw_fraction,
             look.ocean_haze_extinction_per_km(view.haze) * 1e-3,
             view.ocean_sky_shadow_floor, view.contrast)
    # The view-step angle floors at a QUARTER of the render's pixel angle:
    # the accumulation is jittered, so a converged image resolves sub-pixel,
    # and a floor at exactly one pixel dissolves the very detail
    # accumulation would have antialiased. An exact 0 stays 0 — the "LOD
    # off, fixed dt" sentinel. Twin of the packer in uniforms.js
    # (test_uniform_parity pins them to each other).
    pixel_tan = 2.0 * math.tan(view.fov * DEG * 0.5) / w
    if desert and not view.desert_lite and pixel_tan < DESERT_MIN_PIXEL_TAN:
        raise ValueError(
            f"a pixel of {pixel_tan:.3g} rad is finer than the desert's "
            f"flora can walk to ({DESERT_MIN_PIXEL_TAN:.3g}): widen the "
            "field of view or lower the render width.")
    view_step_tan = math.tan(view.view_step_lod_degrees * DEG)
    if view_step_tan > 0.0:
        view_step_tan = max(view_step_tan, 0.25 * pixel_tan)
    u[20] = (1.0 if state.periodic else 0.0,
             math.tan(view.light_march_lod_degrees * DEG),
             view_step_tan,
             view.tone_map_gamma)
    if state.nested:
        u[21] = (*state.nest_bmin, state.dt_view_nest)
        u[22] = (*state.nest_bmax, state.dt_light_nest)
    u[23] = (1.0 if view.light_cache else 0.0, 0.0,
             0.0 if view.sky_probe else 1.0, 0.0)
    # Row 24: the surface offset, folded into [0, surface period) — see
    # SceneState.surface_offset_m and the raymarch.wgsl surface_offset
    # comment — and the scene clock. witness has no flight, so the offset is
    # always the static one folded; for a day scene that is fold(0) = exactly
    # 0.0 (the shader then subtracts a bit-neutral zero). Fold spelled
    # exactly as the browser's (uniforms.js foldTile): one fmod, then a
    # conditional add.
    tile = (state.surface_period_m if state.surface_period_m is not None
            else state.ocean_tile_extent)
    if not math.isfinite(view.scene_time):
        raise ValueError(f"scene_time must be finite; got {view.scene_time}.")

    def _fold_tile(v: float) -> float:
        r = math.fmod(float(v), tile)
        return r + tile if r < 0 else r

    base = (state.surface_offset_m if state.surface_offset_m is not None
            else (state.city_tile_offset_m if city else (0.0, 0.0)))
    # The clock only where the shader reads it (CLOCK_SURFACES: the sea's
    # waves, the city's steam; as part of the browser's scene key a running
    # clock would keep a parked desert from converging while a track
    # records), wrapped at the surface's own period the way uniforms.js
    # wraps it — in float64, before the narrowing to float32.
    clock = 0.0
    if not desert and view.scene_time != 0.0:
        if state.clock_period_s is None or not state.clock_period_s > 0:
            raise ValueError(
                f"a {'city' if city else 'sea'} scene with a running clock "
                "needs clock_period_s (the surface tile's meta.json "
                "clock_period_s; surface_clock_period_s()).")
        clock = math.fmod(float(view.scene_time), state.clock_period_s)
        if clock < 0:
            clock += state.clock_period_s
    u[24] = (_fold_tile(base[0]), _fold_tile(base[1]), clock, 0.0)
    return u


# --- coordinates -----------------------------------------------------------

def camera_world_origin(rel, bmin, bmax) -> np.ndarray:
    """Relative camera position to absolute metres.

    x and y span the AABB over [-1, 1]; z does NOT — it is anchored to the
    physical surface, so rel_z = -1 is sea level rather than the box floor.
    An elevated domain therefore keeps its real altitude, and treating z like
    x and y is the classic way to put the camera underground.
    """
    bmin = np.asarray(bmin, np.float64)
    bmax = np.asarray(bmax, np.float64)
    return np.array([
        bmin[0] + (rel[0] + 1.0) * 0.5 * (bmax[0] - bmin[0]),
        bmin[1] + (rel[1] + 1.0) * 0.5 * (bmax[1] - bmin[1]),
        (rel[2] + 1.0) * 0.5 * bmax[2],
    ], dtype=np.float64)


# --- the renderer ----------------------------------------------------------

_ACCUM_SHADER = """
struct AccumUniforms { prev_weight: f32, sample_weight: f32, _p0: f32, _p1: f32 };
@group(0) @binding(0) var<uniform> au: AccumUniforms;
@group(0) @binding(1) var sample_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_tex: texture_2d<f32>;
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(frag_pos.xy);
    let s = textureLoad(sample_tex, xy, 0);
    if (au.prev_weight <= 0.0) { return vec4<f32>(s.rgb, 1.0); }
    let prev = textureLoad(prev_tex, xy, 0);
    return vec4<f32>(prev.rgb * au.prev_weight + s.rgb * au.sample_weight, 1.0);
}
"""

# The march's own output for one frame: half precision is plenty for a single
# sample (its round-off is ~0.03 8-bit levels, unbiased, and it averages away).
SAMPLE_FORMAT = "rgba16float"
# The running mean, which is a *feedback* loop: this frame's output is next
# frame's input, so its round-off does not average away, it integrates. In
# half precision the loop drifts systematically darker with frame count
# (~10 levels from 1 to 1024 frames — see the journal, iter_012). Full
# precision costs one texture's worth of bandwidth and removes it entirely.
ACCUM_FORMAT = "rgba32float"
# The surface pre-pass target: (radiance, t) of the traced city or desert
# under each render pixel. t runs to 1e30 on a miss and to tens of km on a
# hit, so it needs float32; fs_main only ever textureLoad()s it.
SURFACE_FORMAT = "rgba32float"
# The surface stages each traced surface runs before the march, in order:
# raymarch.wgsl fragment entries, each after the first reading the one
# before at group 1 binding 1, the last writing the surface the march reads.
# The city: its boxes, the facade shell over them (fs_surface_refine), the
# street (fs_surface_street: furniture, wet reflection, steam, kept out of
# the earlier stages' registers). The desert: its ground
# (fs_surface_desert, which also writes the optical depths it lit it with,
# TAUS_FORMAT), the explicit plants' far forms in front of it
# (fs_surface_flora: their walk, kept out of the ground's registers, and
# the far cover, lit from those depths), then its near plants
# (fs_surface_refine). A
# surface or a stage joins
# here and in web/soar/constants.js SURFACE_STAGES, its twin (pinned by
# tests/test_surface_stages.py); the hosts derive everything else from it.
SURFACE_STAGES = {
    "city": ("fs_surface", "fs_surface_refine", "fs_surface_street"),
    "desert": ("fs_surface_desert", "fs_surface_flora", "fs_surface_refine"),
}
# Under DESERT_LITE the desert's later stages are compiled out and skipped,
# and its one stage is the shared first stage.
SURFACE_STAGES_LITE = ("fs_surface",)
# The stages that also write the veil (a second attachment, VEIL_FORMAT):
# the first, and the city's street.
VEIL_STAGES = ("fs_surface", "fs_surface_street")
# The full desert's first stage writes, in the veil's place (the desert has
# none, and the two would pass the 32 bytes a pixel's attachments may hold),
# the optical depths it lit its ground with, (tau_sun, tau_up) as f32 so the
# flora stage rebuilds the very same light (raymarch.wgsl
# desert_light_from); the flora stage reads them at group 1 binding 3.
TAUS_STAGE = "fs_surface_desert"
TAUS_READER = "fs_surface_flora"
TAUS_FORMAT = "rg32float"
# Derived: the scenes whose opaque surface is traced by the stages rather
# than in the march, and those with a refinement or a street stage.
SURFACE_PASS_SURFACES = tuple(SURFACE_STAGES)
SURFACE_REFINE_SURFACES = tuple(
    s for s, e in SURFACE_STAGES.items() if "fs_surface_refine" in e)
# The tiers that fly the simplified desert (raymarch.wgsl DESERT_LITE: no
# refine stage, each desert section's Low drops compiled out). The twin of
# web/soar/constants.js DESERT_LITE_TIERS; witness --quality sets
# ViewState.desert_lite from it.
DESERT_LITE_TIERS = ("low", "minimal")
# The finest pixel the full desert can render: its refine stage walks the
# plants out to DESERT_FLORA_REFINE_FP_M / pixel angle, which must lie
# inside the explicit walk's cell budget, DF_EXPLICIT_MAX_T (raymarch.wgsl;
# 0.025 m / (0.65 * 96 * 16 m)) — past it, plants inside the reach would go
# undrawn. About 5.5 degrees of field across 3840 pixels. The twin of
# web/soar/constants.js DESERT_MIN_PIXEL_TAN (tests/test_desert_pixel_limit.py
# pins both to the shader).
DESERT_MIN_PIXEL_TAN = 0.025 / (0.65 * 96 * 16.0)
# The shrubs' explicit cut per tier (raymarch.wgsl DESERT_FLORA_CUT): Medium
# walks the explicit shrubs about half as far as High; the lite tiers ignore
# it.
# The twin of web/soar/constants.js DESERT_FLORA_CUT_BY_TIER; witness
# --quality sets ViewState.desert_flora_cut from it.
DESERT_FLORA_CUT_BY_TIER = {"max": 1.0, "high": 1.0, "medium": 0.5,
                            "low": 1.0, "minimal": 1.0}

SURFACE_STREET_SURFACES = tuple(
    s for s, e in SURFACE_STAGES.items() if "fs_surface_street" in e)
# The veil the first and last surface stages write beside the surface
# (raymarch.wgsl veil_pack): half-float radiance and opacity and an f32
# depth, packed as uints so no float path touches the bits.
VEIL_FORMAT = "rgba32uint"


def surface_stage_entries(surface: str, desert_lite: bool = False) -> tuple:
    """The surface stages (raymarch.wgsl entries) a view of `surface` runs,
    in order; none over the sea."""
    stages = SURFACE_STAGES.get(check_surface(surface), ())
    if desert_lite:
        if surface != "desert":
            raise ValueError("desert_lite is a desert specialization; the "
                             f"surface is {surface!r}.")
        return SURFACE_STAGES_LITE
    return stages


def surface_stage_plan(n: int) -> list:
    """Where each of `n` chained stages writes and what it reads, as
    (writes, reads) with "final" the surface the march reads and "a"/"b"
    the two intermediates the chain alternates between (None: the first
    stage reads nothing). renderer.js surfaceStagePlan is the twin."""
    plan = []
    for i in range(n):
        writes = "final" if i == n - 1 else "ab"[i % 2]
        reads = None if i == 0 else "ab"[(i - 1) % 2]
        plan.append((writes, reads))
    return plan


# Section stripping (raymarch.wgsl's `// @section` markers). Each surface's
# code is fenced in one or more section blocks plus one stub block that
# re-declares, with neutral bodies, exactly the symbols shared code calls
# into it. A module keeps the compiled surface's sections and the OTHER
# surfaces' stubs, so it carries no code of a surface it cannot show: dead
# code of another surface had been moving this one's codegen (a dead desert
# call moved ocean pixels; the desert's tables moved the city's surface pass
# 128 -> 168 registers). renderer.js stripSections is the twin, and
# tests/test_wgsl_sections.py pins the two to the same bytes.
SECTION_OF_SURFACE = {"ocean": "sea", "city": "city", "desert": "desert"}
_SECTION_MARKER = re.compile(r"^// @(section|end|stub|end-stub) (\w+)\s*$")


def strip_sections(source: str, surface: str) -> str:
    """`source` with every section but `surface`'s blanked, and `surface`'s
    stub blanked. Lines are blanked, not removed, so a compile error's line
    number is still the file's. A source with no markers (a revision from
    before them, which soar_shader_ab replays) is returned as it is; one
    with any must be complete: every surface has at least one section and
    exactly one stub, blocks do not nest, and each closes what it opened."""
    keep = SECTION_OF_SURFACE[check_surface(surface)]
    lines = source.split("\n")
    marks = [(i, m.group(1), m.group(2)) for i, l in enumerate(lines)
             for m in [_SECTION_MARKER.match(l)] if m]
    if not marks:
        return source
    names = set(SECTION_OF_SURFACE.values())
    seen = {n: {"section": 0, "stub": 0} for n in names}
    out, open_ = list(lines), None          # open_: (kind, name, line)
    for i, kind, name in marks:
        if name not in names:
            raise RuntimeError(f"raymarch.wgsl line {i + 1}: unknown section "
                               f"{name!r}; the sections are {sorted(names)}.")
        if kind in ("section", "stub"):
            if open_ is not None:
                raise RuntimeError(
                    f"raymarch.wgsl line {i + 1}: @{kind} {name} opens inside "
                    f"@{open_[0]} {open_[1]} (line {open_[2] + 1}); blocks "
                    "do not nest.")
            open_ = (kind, name, i)
            seen[name][kind] += 1
            continue
        want = "section" if kind == "end" else "stub"
        if open_ is None or open_[:2] != (want, name):
            raise RuntimeError(f"raymarch.wgsl line {i + 1}: @{kind} {name} "
                               f"closes nothing open as @{want} {name}.")
        start = open_[2]
        drop = (name != keep) if want == "section" else (name == keep)
        if drop:
            for j in range(start + 1, i):
                out[j] = ""
        open_ = None
    if open_ is not None:
        raise RuntimeError(f"raymarch.wgsl line {open_[2] + 1}: "
                           f"@{open_[0]} {open_[1]} is never closed.")
    for name, n in sorted(seen.items()):
        if n["section"] < 1 or n["stub"] != 1:
            raise RuntimeError(
                f"raymarch.wgsl: {name} has {n['section']} section block(s) "
                f"and {n['stub']} stub block(s); a marked shader needs at "
                "least one and exactly one.")
    return "\n".join(out)


def specialize(source: str, *, periodic: bool, nested: bool,
               max_light_steps: int, tone_map: bool = True,
               surface: str = "ocean", light_cache: bool = False,
               sky_probe: bool = True, desert_lite: bool = False,
               desert_flora_cut: float = 1.0) -> str:
    """Bake the compile-time constants into the shader source.

    Textual replacement rather than WGSL `override` because MAX_LIGHT_STEPS
    bounds a loop. Each sentinel must appear exactly once — a miss renders
    the wrong thing at full speed and says nothing.

    `surface` turns on at most one of CITY and DESERT, and each is only
    rewritten when it is being turned ON: its declared value is already
    `false`, so the ocean is a no-op rewrite of both. Turning one on demands
    its declaration, like every other sentinel here.

    `light_cache` and `sky_probe` are the lighting method (raymarch.wgsl
    LIGHT_CACHE, SKY_PROBE): the same two flags pack_uniforms writes into
    row 23, compiled in because a run-time branch on them cost every sample
    its untaken side. A shader revision from before they were constants
    (2026-09-23) declares neither and reads row 23 instead —
    tools/soar_shader_ab.py replays such revisions through this host — so
    for it there is nothing to rewrite. Any revision that declares one must
    declare both, exactly once, like every other sentinel.
    """
    if not (1 <= int(max_light_steps) <= 512):
        raise ValueError(f"max_light_steps must be in [1, 512]; got {max_light_steps}.")
    source = strip_sections(source, surface)
    swaps = [
        ("const PERIODIC_DOMAIN: bool = true;",
         f"const PERIODIC_DOMAIN: bool = {'true' if periodic else 'false'};"),
        ("const NESTED: bool = false;",
         f"const NESTED: bool = {'true' if nested else 'false'};"),
        ("const MAX_LIGHT_STEPS: i32 = 512;",
         f"const MAX_LIGHT_STEPS: i32 = {int(max_light_steps)};"),
        ("const TONE_MAP: bool = true;",
         f"const TONE_MAP: bool = {'true' if tone_map else 'false'};"),
    ]
    if surface == "city":
        swaps.append(("const CITY: bool = false;",
                      "const CITY: bool = true;"))
    if surface == "desert":
        swaps.append(("const DESERT: bool = false;",
                      "const DESERT: bool = true;"))
    # The simplified desert: demanded, like CITY and DESERT, only when on.
    if desert_lite:
        if surface != "desert":
            raise ValueError("desert_lite is a desert specialization; the "
                             f"surface is {surface!r}.")
        swaps.append(("const DESERT_LITE: bool = false;",
                      "const DESERT_LITE: bool = true;"))
    # The flora's explicit cut, the same way: demanded only when it departs
    # from the declared 1.0, and only ever over a desert.
    cut = float(desert_flora_cut)
    if not (math.isfinite(cut) and cut > 0.0):
        raise ValueError(f"desert_flora_cut must be positive; got {cut}.")
    if cut != 1.0:
        if surface != "desert":
            raise ValueError("desert_flora_cut is a desert specialization; "
                             f"the surface is {surface!r}.")
        swaps.append(("const DESERT_FLORA_CUT: f32 = 1.0;",
                      f"const DESERT_FLORA_CUT: f32 = {cut!r};"))
    if "const LIGHT_CACHE: bool" in source:
        swaps += [
            ("const LIGHT_CACHE: bool = false;",
             f"const LIGHT_CACHE: bool = {'true' if light_cache else 'false'};"),
            ("const SKY_PROBE: bool = true;",
             f"const SKY_PROBE: bool = {'true' if sky_probe else 'false'};"),
        ]
    for sentinel, replacement in swaps:
        if source.count(sentinel) != 1:
            raise RuntimeError(
                f"Expected exactly one occurrence of {sentinel!r} in "
                f"{SHADER_PATH.name}; found {source.count(sentinel)}. The "
                "shader and this host have drifted apart.")
        source = source.replace(sentinel, replacement)
    return source


class SoarRenderer:
    """A device, the specialized shader, and one uploaded field.

    Deliberately stateful and reusable: prebake renders hundreds of frames of
    one field, and re-creating the device or re-uploading a 400 MB volume per
    frame would dominate everything else.
    """

    def __init__(self, *, periodic: bool = True, nested: bool = False,
                 max_light_steps: int = DEFAULT_MAX_LIGHT_STEPS,
                 tone_map: bool = True, surface: str = "ocean", device=None,
                 shader_source: Optional[str] = None):
        """`shader_source` overrides the checked-in shader.

        It exists so two revisions of raymarch.wgsl can be held against each
        other on one device, in raw float, without a checkout — which is how
        a change claiming to preserve the image gets to prove it (see
        tools/soar_shader_ab.py). Everything else reads the file.

        `surface` is what the world stands on — "ocean", "desert", or the
        night "city". The desert and the city each compile their own constant
        into the shader and bind their own tile at binding 3 in place of the
        ocean's. It is a property of the renderer rather than of a frame
        because both of those are fixed at construction. The SceneState
        passed to each render must agree — set `SceneState.surface` to match.
        """
        import wgpu                                    # local: GPU only on use
        self.wgpu = wgpu
        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        self.device = device
        self.periodic = bool(periodic)
        self.nested = bool(nested)
        self.max_light_steps = int(max_light_steps)
        self.tone_map = bool(tone_map)
        self.surface = check_surface(surface)

        # One module per specialization (see specialize and _spec), compiled
        # when a render first asks for it. The declared default lighting —
        # the live march with the sky probe, Max's method — is compiled here
        # so that a shader that does not compile fails at construction, not
        # at the first frame.
        self._shader_source = (shader_source if shader_source is not None
                               else read_shader())
        self._modules = {}
        self._pipelines = {}
        self._module_for(self._spec(light_cache=False, sky_probe=True))
        self._accum_module = device.create_shader_module(code=_ACCUM_SHADER)

        self._uniform_buf = device.create_buffer(
            size=UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._accum_buf = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._vol_sampler = device.create_sampler(
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge", mag_filter="linear", min_filter="linear")
        # The periodic read of the same volume. Texture axes: u indexes field
        # z (never periodic), v field y, w field x. Repeat on the two lateral
        # axes is what makes filtering exact across the wrap seam with no
        # ghost ring uploaded — see raymarch.wgsl sample_level.
        self._vol_wrap_sampler = device.create_sampler(
            address_mode_u="clamp-to-edge", address_mode_v="repeat",
            address_mode_w="repeat", mag_filter="linear", min_filter="linear")
        self._ocean_sampler = device.create_sampler(
            address_mode_u="repeat", address_mode_v="repeat",
            mag_filter="linear", min_filter="linear", mipmap_filter="linear")

        self._vol_tex = None
        self._nest_tex = self._dummy_volume()
        # Binding 7 starts as the same always-bound zero dummy the nest uses;
        # bake_light_cache replaces it with a real cache.
        self._light_tau_tex = self._dummy_volume()
        # Binding 8 (ice-detection mode, browser-only) is never read here —
        # row 23.w stays 0 — but the layout requires it.
        self._ice_frac_tex = self._dummy_volume()
        # Group 1's stand-in: what the march reads there in a scene with no
        # traced surface.
        self._surface_dummy = self.device.create_texture(
            size=(1, 1, 1), dimension="2d", format=SURFACE_FORMAT,
            usage=self.wgpu.TextureUsage.TEXTURE_BINDING)
        self._veil_dummy = self.device.create_texture(
            size=(1, 1, 1), dimension="2d", format=VEIL_FORMAT,
            usage=self.wgpu.TextureUsage.TEXTURE_BINDING)
        self._light_cache_dims = None
        # Binding 3 is the city's or the desert's tile; binding 9 the moving
        # sea. Each scene fills the one it reads and binds a stand-in in the
        # other, so one layout serves every surface.
        if self.surface == "ocean":
            self._ocean_tex = self._dummy_surface()
            self._sea_tex = self._load_sea()
        else:
            self._ocean_tex = self._load_surface_tile()
            self._sea_tex = self._dummy_surface()
        self._targets = None
        self._bind_group = None
        self._build_pipelines()

    # -- resources ---------------------------------------------------------

    def _dummy_volume(self):
        """Binding 5 must always be bound, nest or no nest."""
        wgpu = self.wgpu
        tex = self.device.create_texture(
            size=(1, 1, 1), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": tex}, np.zeros(1, np.float16).tobytes(),
            {"bytes_per_row": 2, "rows_per_image": 1}, (1, 1, 1))
        return tex

    def _dummy_surface(self):
        """A 1x1 zero rgba16float stand-in: binding 3 over the ocean (viewed
        as 2d), binding 9 under the city and the desert (as a 1-layer array)."""
        wgpu = self.wgpu
        tex = self.device.create_texture(
            size=(1, 1, 1), dimension="2d", format="rgba16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": tex}, np.zeros(4, np.float16).tobytes(),
            {"bytes_per_row": 8, "rows_per_image": 1}, (1, 1, 1))
        return tex

    def _load_sea(self):
        """The moving sea at binding 9: a 2D array, one layer per frame.

        `self.surface_meta` is the sea's meta.json — cell size, tile extent,
        top mip and the surface period, which SceneState has to be told
        about (surface_cell_m, surface_period_m).
        """
        wgpu = self.wgpu
        meta, levels = _sea_levels(str(OCEAN_DIR))
        self.surface_meta = meta
        n, nt = int(meta["n"]), int(meta["frames"])
        tex = self.device.create_texture(
            size=(n, n, nt), dimension="2d", format="rgba16float",
            mip_level_count=len(levels),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for level, data in enumerate(levels):
            side = max(1, n >> level)
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": level}, data,
                {"bytes_per_row": side * 8, "rows_per_image": side},
                (side, side, nt))
        return tex

    def _load_surface_tile(self):
        """The tile bound at binding 3: pre-built mips, level by level.

        Which tile depends on the surface: the desert's eroded terrain
        (rg16float) or the city density field (rgba16float). One binding,
        one loader; each tile's meta names its format, files, encoding and
        compression (decode_surface_level). `self.surface_meta` is the loaded meta.json, and is where a
        caller reads the cell size and tile extent that SceneState has to be
        told about (surface_cell_m).
        """
        import gzip
        import json
        wgpu = self.wgpu
        tile_dir = SURFACE_DIRS[self.surface]
        meta = json.loads((tile_dir / "meta.json").read_text())
        expected = SURFACE_TILE_SCHEMAS[self.surface]
        if meta.get("schema") != expected:
            raise RuntimeError(
                f"{tile_dir}/meta.json is schema {meta.get('schema')!r}; this "
                f"renderer reads {expected!r}. Re-run "
                "tools/export_web_assets.py, or update the shader with it.")
        n, mips = int(meta["n"]), int(meta["mips"])
        fmt = meta.get("format", "rgba16float")
        texel_bytes = SURFACE_FORMAT_CHANNELS[fmt] * 2
        self.surface_meta = meta
        tex = self.device.create_texture(
            size=(n, n, 1), dimension="2d", format=fmt,
            mip_level_count=mips,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for level, name in enumerate(surface_level_files(meta)):
            side = max(1, n >> level)
            raw = (tile_dir / name).read_bytes()
            if meta.get("compression") == "gzip":
                raw = gzip.decompress(raw)
            try:
                data = decode_surface_level(raw, side, meta)
            except RuntimeError as exc:
                raise RuntimeError(f"{tile_dir.name}/{name}: {exc}") from None
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": level}, data,
                {"bytes_per_row": side * texel_bytes, "rows_per_image": side},
                (side, side, 1))
        return tex

    def upload_volume(self, sigma: np.ndarray) -> None:
        """Upload the extinction field.

        `sigma` is (nx, ny, nz) — the field itself, with no border of any
        kind. A C-order array of that shape is already z-fastest, which is
        exactly the byte order a texture of size (nz, ny, nx) wants, so there
        is no transpose here and adding one is a bug.

        There is nothing to pad and nothing to wrap. Both boundary behaviours
        the ghost ring used to carry — the linear taper into zero, and the
        periodic lateral wrap — are the shader's now (raymarch.wgsl
        sample_level), computed from the same texel this uploads. That is what
        keeps a 2048-cell axis inside a 2048-texel limit, and it retires the
        one surface the two hosts had already drifted on.
        """
        wgpu = self.wgpu
        data = np.ascontiguousarray(sigma, dtype=np.float16)
        nx, ny, nz = data.shape
        limit = self.device.limits.get("max-texture-dimension-3d", 2048)
        if max(nx, ny, nz) > limit:
            raise ValueError(
                f"Volume {nx}x{ny}x{nz} exceeds this device's 3D texture "
                f"limit of {limit} on one axis.")
        if self._vol_tex is not None:
            self._vol_tex.destroy()
        self._vol_tex = self.device.create_texture(
            size=(nz, ny, nx), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        # queue.write_texture, never copy_buffer_to_texture: the 256-byte
        # bytes_per_row rule applies to buffer copies, and nz*2 almost never
        # satisfies it.
        self.device.queue.write_texture(
            {"texture": self._vol_tex}, data.tobytes(),
            {"bytes_per_row": nz * 2, "rows_per_image": ny}, (nz, ny, nx))
        self._bind_group = None

    def upload_nest(self, sigma: np.ndarray) -> None:
        """Upload the finer level bound at binding 5.

        Always tapered to zero at its edges, even in a periodic domain: that
        taper is how the nest blends out into the coarse field around it,
        which is a different thing from the outer domain's wrap. The shader
        applies it; like the outer level, this is the bare (nx, ny, nz) field.
        """
        if not self.nested:
            raise RuntimeError(
                "This renderer was built with nested=False, so the shader has "
                "NESTED baked to false and would ignore the nest.")
        wgpu = self.wgpu
        data = np.ascontiguousarray(sigma, dtype=np.float16)
        nx, ny, nz = data.shape
        if self._nest_tex is not None:
            self._nest_tex.destroy()
        self._nest_tex = self.device.create_texture(
            size=(nz, ny, nx), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture(
            {"texture": self._nest_tex}, data.tobytes(),
            {"bytes_per_row": nz * 2, "rows_per_image": ny}, (nz, ny, nx))
        self._bind_group = None

    # -- pipelines ---------------------------------------------------------

    def _build_pipelines(self):
        wgpu = self.wgpu
        d = self.device
        tex3d = {"sample_type": "float", "view_dimension": "3d"}
        self._ray_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex3d},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float", "view_dimension": "2d"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
            {"binding": 5, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex3d},
            {"binding": 6, "visibility": wgpu.ShaderStage.FRAGMENT,
             "sampler": {"type": "filtering"}},
            {"binding": 7, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex3d},
            {"binding": 8, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": tex3d},
            {"binding": 9, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float",
                         "view_dimension": "2d-array"}},
        ])
        self._ray_pipeline_layout = d.create_pipeline_layout(
            bind_group_layouts=[self._ray_layout])
        # Group 1: the surface pre-pass's targets (raymarch.wgsl) — the final
        # surface, read by the march; the previous stage's, read by each later
        # stage (fs_surface_refine, fs_surface_street); the veil, read by
        # the march; and the desert's optical depths (TAUS_FORMAT), read by
        # its flora stage. textureLoad()ed only, so the float formats'
        # unfilterability is exactly right. The passes that read none of
        # them (the bake, the first surface stage) are laid out without it.
        unfilterable = {"sample_type": "unfilterable-float",
                        "view_dimension": "2d"}
        self._surface_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "uint", "view_dimension": "2d"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": unfilterable},
        ])
        self._march_pipeline_layout = d.create_pipeline_layout(
            bind_group_layouts=[self._ray_layout, self._surface_layout])

        self._accum_layout = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT,
             "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "float", "view_dimension": "2d"}},
            # The accumulator is rgba32float, which core WebGPU declares
            # unfilterable. This pass only ever textureLoad()s it, so that is
            # exactly right — and stating it here means a future sampled read
            # fails validation loudly instead of silently needing a feature.
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT,
             "texture": {"sample_type": "unfilterable-float",
                         "view_dimension": "2d"}},
        ])
        self._accum_pipeline = d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self._accum_layout]),
            vertex={"module": self._accum_module, "entry_point": "vs_main"},
            fragment={"module": self._accum_module, "entry_point": "fs_main",
                      "targets": [{"format": ACCUM_FORMAT}]},
            primitive={"topology": "triangle-list"})

    def _spec(self, *, light_cache: bool, sky_probe: bool,
              desert_lite: bool = False,
              desert_flora_cut: float = 1.0) -> dict:
        """Everything specialize() takes, from this renderer's attributes as
        they stand now plus a lighting method. Every module and pipeline
        cache here keys on the whole of it, so an attribute changed after
        construction can never be served a module compiled for the old
        value."""
        return dict(periodic=self.periodic, nested=self.nested,
                    max_light_steps=self.max_light_steps,
                    tone_map=self.tone_map, surface=self.surface,
                    light_cache=bool(light_cache), sky_probe=bool(sky_probe),
                    desert_lite=bool(desert_lite),
                    desert_flora_cut=float(desert_flora_cut))

    def _module_for(self, spec: dict):
        key = tuple(sorted(spec.items()))
        module = self._modules.get(key)
        if module is None:
            module = self.device.create_shader_module(
                code=specialize(self._shader_source, **spec))
            self._modules[key] = module
        return module

    def _pipeline_for(self, entry_point: str, target_format, spec: dict,
                      layout=None):
        """`layout` defaults to the scene group alone; the march, which also
        reads the surface pre-pass's target, passes _march_pipeline_layout.
        `target_format` is one format, or a tuple of them for an entry point
        with several outputs (the surface stages that write the veil)."""
        formats = ((target_format,) if isinstance(target_format, str)
                   else tuple(target_format))
        key = (entry_point, formats, tuple(sorted(spec.items())))
        pipeline = self._pipelines.get(key)
        if pipeline is None:
            module = self._module_for(spec)
            pipeline = self.device.create_render_pipeline(
                layout=layout or self._ray_pipeline_layout,
                vertex={"module": module, "entry_point": "vs_main"},
                fragment={"module": module, "entry_point": entry_point,
                          "targets": [{"format": f} for f in formats]},
                primitive={"topology": "triangle-list"})
            self._pipelines[key] = pipeline
        return pipeline

    def _ray_pipeline_for(self, view: "ViewState"):
        """The march pipeline for the method `view` asks for — the same two
        flags pack_uniforms writes into row 23, so the row and the compiled
        method cannot disagree."""
        return self._pipeline_for("fs_main", SAMPLE_FORMAT, self._view_spec(view),
                                  layout=self._march_pipeline_layout)

    def _view_spec(self, view: "ViewState") -> dict:
        """The specialization `view` asks for: its lighting method and, over
        a desert, its detail."""
        return self._spec(light_cache=view.light_cache,
                          sky_probe=view.sky_probe,
                          desert_lite=view.desert_lite,
                          desert_flora_cut=view.desert_flora_cut)

    def _bake_pipeline(self):
        """The sun-tau bake: one field-x plane of the cache per draw, into an
        r16float 2D target the caller copies into the 3D cache texture.
        fs_bake_light reads neither lighting constant, so it is built on the
        declared default method (as the browser's is)."""
        return self._pipeline_for("fs_bake_light", "r16float", self._spec(
            light_cache=False, sky_probe=True))

    def _stage_entries(self, view: "ViewState") -> tuple:
        """The surface stages `view` runs (SURFACE_STAGES), in order."""
        return surface_stage_entries(self.surface, view.desert_lite)

    def _stage_pipeline_for(self, entry: str, first: bool,
                            view: "ViewState"):
        """One surface stage's pipeline on `view`'s specialization: the
        first reads nothing at group 1, a later one the stage before (the
        march's layout); a VEIL_STAGES entry writes the veil too, and
        TAUS_STAGE the desert's optical depths."""
        if entry in VEIL_STAGES:
            formats = (SURFACE_FORMAT, VEIL_FORMAT)
        elif entry == TAUS_STAGE:
            formats = (SURFACE_FORMAT, TAUS_FORMAT)
        else:
            formats = SURFACE_FORMAT
        return self._pipeline_for(
            entry, formats, self._view_spec(view),
            layout=None if first else self._march_pipeline_layout)

    def _named_stage_for(self, entry: str, view: "ViewState"):
        entries = self._stage_entries(view)
        if entry not in entries:
            return None
        return self._stage_pipeline_for(entry, entries.index(entry) == 0,
                                        view)

    def _surface_pipeline_for(self, view: "ViewState"):
        """The first surface stage (fs_surface), or None in a scene with no
        traced surface."""
        return self._named_stage_for("fs_surface", view)

    def _refine_pipeline_for(self, view: "ViewState"):
        """The refinement (fs_surface_refine), or None where the view has
        none (the sea; the simplified desert compiles it out)."""
        return self._named_stage_for("fs_surface_refine", view)

    def _street_pipeline_for(self, view: "ViewState"):
        """The city's street stage (fs_surface_street), or None."""
        return self._named_stage_for("fs_surface_street", view)

    def _ensure_targets(self, size, view: Optional["ViewState"] = None):
        """This size's render targets, with the surface intermediates
        `view`'s stage chain alternates between (surface_stage_plan) — or,
        with no view, the longest chain this surface can run. Grow-only, as
        renderer.js _ensureSurfaceTarget: a set that already holds enough
        keeps them, so a Low render (the simplified desert's one stage)
        never makes the two it does not use — nor the optical depths'
        target (TAUS_STAGE), which only the full desert's chain writes."""
        wgpu = self.wgpu
        entries = (self._stage_entries(view) if view is not None
                   else SURFACE_STAGES.get(self.surface, ()))
        n_mid = min(max(len(entries) - 1, 0), 2)
        if self._targets is None or self._targets["size"] != size:
            w, h = size
            has = self.surface in SURFACE_PASS_SURFACES
            surface = self._make_target(size, SURFACE_FORMAT) if has else None
            veil = self._make_target(size, VEIL_FORMAT) if has else None
            self._targets = {
                "size": size, "sample": self._make_target(size, SAMPLE_FORMAT),
                "accum": [self._make_target(size, ACCUM_FORMAT),
                          self._make_target(size, ACCUM_FORMAT)],
                "surface": surface, "veil": veil,
                "plain": None, "mid": None, "taus": None,
                # Group 1 for a later stage reading the plain or the middle
                # surface (made with them); no stage binds what it writes.
                "after_plain": None, "after_mid": None,
                # Group 1 for the march: this size's final surface and veil.
                "surface_group": self._group1(
                    surface or self._surface_dummy, self._surface_dummy,
                    veil or self._veil_dummy, self._surface_dummy),
            }
        t = self._targets
        grown = False
        if TAUS_STAGE in entries and t["taus"] is None:
            t["taus"] = self._make_target(size, TAUS_FORMAT)
            grown = True
        for name, after, needed in (("plain", "after_plain", n_mid >= 1),
                                    ("mid", "after_mid", n_mid >= 2)):
            if needed and t[name] is None:
                t[name] = self._make_target(size, SURFACE_FORMAT)
            if t[name] is not None and (t[after] is None or grown):
                # A later stage's group: what it reads, and the optical
                # depths wherever they exist (the flora stage reads them).
                t[after] = self._group1(self._surface_dummy, t[name],
                                        self._veil_dummy,
                                        t["taus"] or self._surface_dummy)
        return t

    def _make_target(self, size, fmt):
        wgpu = self.wgpu
        w, h = size
        return self.device.create_texture(
            size=(w, h, 1), dimension="2d", format=fmt,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_SRC))

    def _group1(self, final, prev, veil, taus):
        return self.device.create_bind_group(
            layout=self._surface_layout, entries=[
                {"binding": 0, "resource": final.create_view()},
                {"binding": 1, "resource": prev.create_view()},
                {"binding": 2, "resource": veil.create_view()},
                {"binding": 3, "resource": taus.create_view()}])

    def _surface_stages(self, targets, view: "ViewState"):
        """The surface stages this view runs, in order, as (pipeline, group 1
        or None, the textures it writes): SURFACE_STAGES chained by
        surface_stage_plan — each later stage reads the one before at group
        1 binding 1 through the two intermediates ("plain", "mid"), the last
        writes the surface the march reads, a VEIL_STAGES entry writes
        the veil beside it, and TAUS_STAGE the optical depths TAUS_READER
        reads (group 1 binding 3). A stage without the target it needs
        raises rather than read a stand-in."""
        entries = self._stage_entries(view)
        slot = {"final": (targets["surface"], None),
                "a": (targets["plain"], targets["after_plain"]),
                "b": (targets["mid"], targets["after_mid"])}
        stages = []
        for k, (entry, (writes, reads)) in enumerate(
                zip(entries, surface_stage_plan(len(entries)))):
            out = [slot[writes][0]]
            if entry in VEIL_STAGES:
                out.append(targets["veil"])
            if entry in (TAUS_STAGE, TAUS_READER) and targets["taus"] is None:
                raise RuntimeError(
                    f"{entry} needs the desert's optical-depth target, and "
                    "this size's targets have none: _ensure_targets was "
                    "not given this view.")
            if entry == TAUS_STAGE:
                out.append(targets["taus"])
            stages.append((self._stage_pipeline_for(entry, k == 0, view),
                           None if reads is None else slot[reads][1], out))
        return stages

    def _ensure_bind_group(self):
        if self._bind_group is not None:
            return self._bind_group
        if self._vol_tex is None:
            raise RuntimeError("No volume uploaded; call upload_volume() first.")
        self._bind_group = self.device.create_bind_group(
            layout=self._ray_layout, entries=[
                {"binding": 0, "resource": {"buffer": self._uniform_buf,
                                            "offset": 0, "size": UNIFORM_NBYTES}},
                {"binding": 1, "resource": self._vol_tex.create_view()},
                {"binding": 2, "resource": self._vol_sampler},
                {"binding": 3, "resource": self._ocean_tex.create_view()},
                {"binding": 4, "resource": self._ocean_sampler},
                {"binding": 5, "resource": self._nest_tex.create_view()},
                {"binding": 6, "resource": self._vol_wrap_sampler},
                {"binding": 7, "resource": self._light_tau_tex.create_view()},
                {"binding": 8, "resource": self._ice_frac_tex.create_view()},
                {"binding": 9, "resource": self._sea_tex.create_view(
                    dimension="2d-array")},
            ])
        return self._bind_group

    def _check_scene(self, state: SceneState) -> None:
        """The surface, periodicity and nest are compiled in at construction.

        A SceneState that disagrees would pack the night palette into a shader
        reading it as sunlight (or the reverse), or describe one tile to a
        shader holding another, and render something plausible and wrong,
        which is the failure mode worth a loud error.
        """
        if state.surface != self.surface:
            raise RuntimeError(
                f"This renderer was built for the {self.surface} — its "
                "specialization is compiled and its tile bound — but the "
                f"scene says surface={state.surface!r}. Build a renderer for "
                "the scene you mean to render.")
        # Periodicity and the nest are compiled in the same way (the
        # PERIODIC_DOMAIN and NESTED constants); the scene's rows 20.x and
        # 21-22 describing something else would be packed and ignored.
        if bool(state.periodic) != self.periodic or bool(state.nested) != self.nested:
            raise RuntimeError(
                f"This renderer was compiled with periodic={self.periodic}, "
                f"nested={self.nested}, but the scene says "
                f"periodic={bool(state.periodic)}, nested={bool(state.nested)}. "
                "Build a renderer for the scene you mean to render.")

    # -- the sun-tau cache ---------------------------------------------------

    def bake_light_cache(self, state: SceneState, view: ViewState, *,
                         divisor: int = 2) -> Tuple[int, int, int]:
        """Bake sun optical depth for every cache voxel toward `view`'s sun.

        `divisor` is the linear resolution reduction against the volume grid
        (1 = one cache texel per voxel). The bake is one fs_bake_light draw
        per field-x plane, copied into the 3D cache; after it returns,
        rendering with view.light_cache=True reads the cache instead of
        marching to the sun per cloud sample. Re-bake whenever the sun, the
        field, or the divisor changes — this host is the offline path, so
        staleness is the caller's to manage, and the browser manages its own.

        The bake's marches step at `state.dt_light` (no distance-LOD
        coarsening — dt_floor is 0). Pass the finest configuration you will
        render with: the browser always bakes at High's light step so its
        cache never goes stale on a tier change, and a caller comparing
        against a coarser tier should still usually do the same, because a
        parked view converges to High's sampling on every tier.

        Returns the cache dims (nx, ny, nz) in field axis order.
        """
        wgpu = self.wgpu
        if self._vol_tex is None:
            raise RuntimeError("No volume uploaded; call upload_volume() first.")
        self._check_scene(state)
        if not (1 <= int(divisor) <= 8):
            raise ValueError(f"divisor must be in [1, 8]; got {divisor}.")
        divisor = int(divisor)
        # Volume texture is (w=nz, h=ny, d=nx); the cache mirrors it.
        vz, vy, vx = self._vol_tex.size
        cz = max(1, -(-vz // divisor))
        cy = max(1, -(-vy // divisor))
        cx = max(1, -(-vx // divisor))

        self._light_tau_tex.destroy()
        self._light_tau_tex = self.device.create_texture(
            size=(cz, cy, cx), dimension="3d", format="r16float",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self._light_cache_dims = (cx, cy, cz)
        self._bind_group = None
        bind_group = self._ensure_bind_group()

        slice_tex = self.device.create_texture(
            size=(cz, cy, 1), dimension="2d", format="r16float",
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.COPY_SRC))
        slice_view = slice_tex.create_view()

        base = pack_uniforms(state, view)
        bake_pipeline = self._bake_pipeline()
        for ix in range(cx):
            u = base.copy()
            u[23, 1] = float(ix)
            self.device.queue.write_buffer(self._uniform_buf, 0, u.tobytes())
            enc = self.device.create_command_encoder()
            rp = enc.begin_render_pass(color_attachments=[{
                "view": slice_view, "load_op": "clear", "store_op": "store",
                "clear_value": (0.0, 0.0, 0.0, 1.0)}])
            rp.set_pipeline(bake_pipeline)
            rp.set_bind_group(0, bind_group)
            rp.draw(3)
            rp.end()
            enc.copy_texture_to_texture(
                {"texture": slice_tex, "mip_level": 0, "origin": (0, 0, 0)},
                {"texture": self._light_tau_tex, "mip_level": 0,
                 "origin": (0, 0, ix)},
                (cz, cy, 1))
            self.device.queue.submit([enc.finish()])
        return (cx, cy, cz)

    # -- rendering ---------------------------------------------------------

    def render(self, state: SceneState, view: ViewState, *,
               frames: int = STILL_ACCUMULATE_FRAMES) -> np.ndarray:
        """Accumulate `frames` passes at a fixed camera and read the result.

        This is the offline path, not the interactive one: the scene key never
        changes across passes, so the weights are a plain uniform average and
        the result is exactly mean(sample_0 .. sample_{frames-1}). Pass 0 is
        unjittered in subpixel; every later pass is jittered. Getting that
        backwards — all passes jittered, or none — is the likeliest silent
        divergence from the browser.

        Returns (h, w, 3) float64 in [0, 1], already tone-mapped by the
        shader. Read back from the float accumulator rather than an 8-bit
        blit, so nothing is quantised on the way out.
        """
        if frames < 1:
            raise ValueError(f"frames must be >= 1; got {frames}.")
        self._check_scene(state)
        w, h = view.render_size
        targets = self._ensure_targets((w, h), view)
        bind_group = self._ensure_bind_group()
        sample_view = targets["sample"].create_view()
        ray_pipeline = self._ray_pipeline_for(view)
        surface_stages = self._surface_stages(targets, view)

        accum_index = 0
        for i in range(frames):
            pass_view = ViewState(**{**view.__dict__,
                                     "frame_index": view.frame_index + i,
                                     "subpixel": i >= 1,
                                     "jitter_scale": 1.0})
            u = pack_uniforms(state, pass_view)
            self.device.queue.write_buffer(self._uniform_buf, 0, u.tobytes())
            prev_w = 0.0 if i == 0 else i / (i + 1.0)
            samp_w = 1.0 if i == 0 else 1.0 / (i + 1.0)
            self.device.queue.write_buffer(
                self._accum_buf, 0,
                np.array([prev_w, samp_w, 0.0, 0.0], np.float32).tobytes())

            prev_tex = targets["accum"][accum_index]
            out_tex = targets["accum"][1 - accum_index]
            accum_bg = self.device.create_bind_group(
                layout=self._accum_layout, entries=[
                    {"binding": 0, "resource": {"buffer": self._accum_buf,
                                                "offset": 0, "size": 16}},
                    {"binding": 1, "resource": sample_view},
                    {"binding": 2, "resource": prev_tex.create_view()},
                ])

            enc = self.device.create_command_encoder()
            # The surface under this frame's rays, for the march below: same
            # uniforms, same frame, same pixels, stage by stage.
            for pipeline, group1, writes in surface_stages:
                sp = enc.begin_render_pass(color_attachments=[
                    {"view": tex.create_view(), "load_op": "clear",
                     "store_op": "store",
                     "clear_value": ((0.0, 0.0, 0.0, 1e30) if k == 0
                                     else (0, 0, 0, 0))}
                    for k, tex in enumerate(writes)])
                sp.set_pipeline(pipeline)
                sp.set_bind_group(0, bind_group)
                if group1 is not None:
                    sp.set_bind_group(1, group1)
                sp.draw(3)
                sp.end()
            rp = enc.begin_render_pass(color_attachments=[{
                "view": sample_view, "load_op": "clear", "store_op": "store",
                "clear_value": (0.0, 0.0, 0.0, 1.0)}])
            rp.set_pipeline(ray_pipeline)
            rp.set_bind_group(0, bind_group)
            rp.set_bind_group(1, targets["surface_group"])
            rp.draw(3)
            rp.end()

            ap = enc.begin_render_pass(color_attachments=[{
                "view": out_tex.create_view(), "load_op": "clear",
                "store_op": "store", "clear_value": (0.0, 0.0, 0.0, 1.0)}])
            ap.set_pipeline(self._accum_pipeline)
            ap.set_bind_group(0, accum_bg)
            ap.draw(3)
            ap.end()
            self.device.queue.submit([enc.finish()])
            accum_index = 1 - accum_index

        return self._read_back(targets["accum"][accum_index], w, h)

    def _read_back(self, texture, w: int, h: int) -> np.ndarray:
        """copy_texture_to_buffer does enforce the 256-byte row rule."""
        wgpu = self.wgpu
        dtype = {"rgba16float": np.float16, "rgba32float": np.float32}[ACCUM_FORMAT]
        bytes_per_texel = 4 * np.dtype(dtype).itemsize
        row = math.ceil(w * bytes_per_texel / 256) * 256
        buf = self.device.create_buffer(
            size=row * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": buf, "offset": 0, "bytes_per_row": row,
             "rows_per_image": h},
            (w, h, 1))
        self.device.queue.submit([enc.finish()])
        buf.map_sync(wgpu.MapMode.READ)
        raw = np.frombuffer(bytearray(buf.read_mapped()), dtype)
        buf.unmap()
        img = raw.reshape(h, row // np.dtype(dtype).itemsize)[:, :w * 4]
        img = img.reshape(h, w, 4)
        return np.ascontiguousarray(img[:, :, :3], dtype=np.float64)


def desert_relief_m(meta: dict) -> Tuple[float, float, float]:
    """Row 8.yzw under DESERT, from the desert tile's meta.json."""
    return (float(meta["height_min_m"]), float(meta["height_max_m"]),
            float(meta["band_max_m"]))


# The per-level tables raymarch.wgsl's desert_tile reads, in row order.
_DESERT_LEVEL_TABLES = ("slope_max", "envelope_m", "slope_variance_removed")


def desert_tile_rows(meta: dict) -> list:
    """Rows 25..46 under DESERT (raymarch.wgsl desert_tile), from the desert
    tile's meta.json: its per-level tables four levels to a row, then its
    per-level colour correction a level to a row, then (bounce albedo,
    band_slope_max). scene.js desertTileRows is the twin."""
    mips = int(meta["mips"])
    if mips != 12:
        raise ValueError(f"the desert tile has {mips} mips; the shader's "
                         "tables are laid out for 12.")
    rows = []
    for key in _DESERT_LEVEL_TABLES:
        table = [float(v) for v in meta[key]]
        if len(table) != mips:
            raise ValueError(f"meta {key} has {len(table)} levels, not {mips}.")
        rows += [table[i:i + 4] for i in range(0, mips, 4)]
    correction = meta["albedo_correction"]
    if len(correction) != mips:
        raise ValueError(f"meta albedo_correction has {len(correction)} "
                         f"levels, not {mips}.")
    rows += [[*map(float, rgb), 0.0] for rgb in correction]
    rows.append([*map(float, meta["bounce_albedo"]),
                 float(meta["band_slope_max"])])
    assert len(rows) == DESERT_TILE_ROWS
    return rows


def desert_scene_fields(meta: dict) -> dict:
    """The SceneState fields a desert takes from its tile's meta.json."""
    return {"desert_relief_m": desert_relief_m(meta),
            "desert_tile_rows": desert_tile_rows(meta)}
