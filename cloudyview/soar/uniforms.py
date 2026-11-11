"""The shader's uniform block: what a scene and a view are, and how they pack.

SceneState is the field and the surface under it; ViewState is one frame's
camera, sun and look. pack_uniforms turns the pair into the 26 rows of four
float32 that shader/uniforms.wgsl declares. The browser packs the same block
in web/soar/uniforms.js, and tests/test_uniform_parity.py compares the two
byte for byte, so the arithmetic here is written to match the JavaScript:
the order of operations, where float64 narrows to float32, and which
functions are used (sqrt and fmod are correctly rounded everywhere; pow is
not).

Conventions the shader fixes:
  * FOV is HORIZONTAL.
  * Relative camera z is anchored to the physical surface, not the bottom of
    the data: rel_z = -1 is sea level (camera_world_origin).
  * Booleans travel as 0.0/1.0 floats and are tested with `> 0.5`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import look
from .surfaces import check_surface
from .tiers import CITY_ROOF_DENSITY_RANGE


# Rows of vec4<f32> in the block (shader/uniforms.wgsl lists what each holds).
UNIFORM_ROWS = 21
UNIFORM_NBYTES = UNIFORM_ROWS * 16          # 336
# Where two slots sit in the block, as (row, column), for the passes that
# write them into a packed block themselves.
SLOT_LENS_DT = (16, 3)
SLOT_BAKE_SLICE = (20, 3)

DEG = math.pi / 180.0

# --- defaults; web/soar/constants.js holds the browser's copies -------------

DEFAULT_SUN_AZIMUTH = 20.0
DEFAULT_SUN_ELEVATION = 55.0
DEFAULT_EXPOSURE = 4.0
DEFAULT_OCEAN_REFLECTANCE = (0.0020, 0.0045, 0.0126)
APP_LIGHT_MARCH_LOD_DEGREES = 1.4
APP_VIEW_STEP_LOD_DEGREES = 0.6
# The two angles above are the app's; this multiplies both for a render that
# is not trying to hold a framerate (a still, a video frame). It is the
# browser's capture strength and its high/max tier default, so a terminal
# render, an in-app capture and a high-tier flight all march the same.
#
# Angular LOD only ever grows the step FLOOR (dt = max(dt_base, t*tan(theta))),
# so a smaller strength costs time and can never coarsen anything: at 0.5 the
# far field is marched twice as finely as the two angles alone ask for.
DEFAULT_LOD_STRENGTH = 0.5
# Whether the aerial haze thins with height; see ViewState.
DEFAULT_HAZE_HEIGHT_DEPENDENT = False
# The march step, in units of the field's smallest voxel.
STEP_VOXEL_FACTOR = 2.0

# The display gamma. TONE_MAP_GAMMA_WITNESS is the gamma of witness.tone_map
# and of behold's encode, for a caller who wants a soar render on that scale.
DEFAULT_TONE_MAP_GAMMA = 1.66
TONE_MAP_GAMMA_WITNESS = 1.4
TONE_MAP_GAMMA_LIMITS = (1.0, 4.0)

# The extended-Reinhard white point: the exposed radiance that maps to 1.0.
# It decides whether a sunlit cloud face reaches white or stays a bright
# grey. Below 4 the whole picture clips; above 40 the curve is within a
# fraction of a percent of linear through the shoulder and the number does
# nothing.
DEFAULT_TONE_MAP_WHITE_POINT = 15.0
TONE_MAP_WHITE_POINT_LIMITS = (4.0, 40.0)

# Display-space contrast about mid-grey, applied after the gamma encode.
# 1.0 is exactly the identity (shader/tonemap.wgsl's tone_map says why the
# arithmetic is written the way it is). Past 1.6 the sky posterizes and the
# shadows block up; below 0.5 the picture is fog.
DEFAULT_CONTRAST = 1.0
CONTRAST_LIMITS = (0.5, 1.6)

# A periodic domain's light march leaves only through the domain top, so the
# sun must be above the horizon. pack_uniforms raises for a sun at or below
# 0 degrees, and its message asks for at least this.
MIN_SUN_ELEVATION_DEG = 0.5


# --- night, for the city ----------------------------------------------------
#
# Rows 13-17 carry the sun's radiance and the sky it lights, which by day come
# from look._spectral_lighting_colors, an air-mass model of a DAYTIME
# atmosphere. None of that applies at night: there is no air mass to redden a
# moon that is 4e-6 of the sun's irradiance, and the night sky's own radiance
# is not scattered moonlight at all. So over the city these night values are
# packed straight into the same five rows, and the daytime model is not
# called.
NIGHT_MOON_CLOUD_COLOR = (0.34, 0.42, 0.60)     # u.cloud_sun: the direct beam
NIGHT_AMBIENT_TINT = (0.020, 0.026, 0.044)      # u.ambient_tint: the ambient fill
NIGHT_SKY_HORIZON = (0.012, 0.014, 0.022)       # u.sky_horizon: haze and horizon
NIGHT_MOON_BLOOM = (0.20, 0.22, 0.26)           # u.sky_bloom: the circum-lunar bloom
NIGHT_MOON_DISC = (4.0, 4.2, 4.6)               # u.sky_disc: the disc itself


# --- validation, the same checks as uniforms.js ------------------------------

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
    """Meteorological bearing + elevation to a unit vector (east, north, up).

    camera.py has a function of the same name that computes the same vector
    with numpy's cos and sin. This one uses math's, which is what
    tests/test_uniform_parity.py holds byte for byte against the browser's
    Math.cos and Math.sin. The two agree to the bit where numpy calls the C
    library, but numpy may use its own vectorized cos and sin on other
    machines, so the uniform block keeps its own.
    """
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



class _Unstated:
    """SceneState.clock_period_s before its caller has said what the
    surface's clock is — distinct from None, which says it has none."""

    def __repr__(self):
        return "UNSTATED"


UNSTATED = _Unstated()

# --- the uniform block -----------------------------------------------------

@dataclass
class SceneState:
    """Everything about the field, independent of where the camera is.

    `surface` is what the world stands on: "ocean" or "city". The city adds
    no uniform rows. The block is the same 336 bytes over either surface; the
    city REINTERPRETS a few rows, and the fields keep their ocean names
    because the slots are the ocean's. Under "city":

      * `ocean_z` is the ground plane; `ocean_reflectance` is unused and packs
        as zeros.
      * `ocean_fif_dx` is the city cell size in metres and `ocean_tile_extent`
        the tile's extent in metres, both from the tile's meta.json
        ("cell_m" / "tile_extent_m"); `ocean_max_lod` is its top mip.
      * `ocean_enabled` still means "draw the surface", and must be True — a
        city with the surface off is an empty ground plane, so pack_uniforms
        raises rather than render it.

    The ViewState's ocean knobs (glint, resolved slope, mip bias) are still
    validated and packed over the city; its shader module does not read them.
    """
    bmin: Sequence[float]
    bmax: Sequence[float]
    dt_view: float
    dt_light: float
    periodic: bool = True
    ocean_z: float = 0.0
    ocean_reflectance: Sequence[float] = DEFAULT_OCEAN_REFLECTANCE
    ocean_fif_dx: float = 0.1
    ocean_tile_extent: float = 102.4
    ocean_enabled: bool = True
    ocean_max_lod: int = 10
    nested: bool = False
    nest_bmin: Optional[Sequence[float]] = None
    nest_bmax: Optional[Sequence[float]] = None
    dt_view_nest: float = 0.0
    dt_light_nest: float = 0.0
    surface: str = "ocean"
    # Where the city tile sits in world space (m, a whole number of blocks):
    # the surface offset of a city scene that states none. The default puts
    # the tile's tallest district under the
    # central cameras of the TWP-ICE test field, where
    # tools/night_city_harness.py flies.
    city_tile_offset_m: Sequence[float] = (75330.0, -17010.0)
    # u.surface_offset: where the surface tile sits in the world this render
    # draws.
    # The shader takes the city frame and the water uv as world minus this,
    # folded into [0, period). None means the scene's static offset:
    # city_tile_offset_m over the city, and the origin over the sea, which
    # folds to exactly 0.0 so the shader subtracts a zero. A flight in the
    # browser moves the offset each time its camera folds; witness's
    # --surface-origin puts a flight's value here so a still reproduces it.
    surface_offset_m: Optional[Sequence[float]] = None
    # The period the surface offset folds at (surfaces.surface_period_m).
    # None is the
    # tile extent.
    surface_period_m: Optional[float] = None
    # The surface's clock (surfaces.surface_clock_period_s). A period:
    # u.scene_clock is the scene clock modulo it, so float32 keeps the
    # clock's resolution however long the scene has run (the city). None:
    # nothing on the surface moves, and u.scene_clock is 0 at any scene time
    # (the ocean).
    # UNSTATED until a caller says which, and a scene_time other than 0 with
    # it unstated raises. uniforms.js has the same three cases (undefined /
    # null / number).
    clock_period_s: object = UNSTATED


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
    spectral_lighting_strength: float = look.SPECTRAL_LIGHTING_STRENGTH
    light_transfer_split_strength: float = look.LIGHT_TRANSFER_SPLIT_STRENGTH
    # One number for all the aerosol: aerial extinction, the sky's horizon
    # whitening, the circumsolar lobe, the haze over the sea. There is
    # deliberately no per-term override (see look.HAZE_ANCHOR).
    haze: float = look.DEFAULT_HAZE
    # Whether that aerosol thins with height. The exponential profile is the
    # physical one, but with a 2.5 km scale height an upward ray leaves the
    # haze entirely, and nothing then bounds how far it marches except the
    # range ceiling. Uniform haze puts the same extinction at every altitude,
    # which is unphysical and ends every ray at about the same distance, so
    # it is much cheaper. Off by default. Packed as u.haze_scale_height.
    haze_height_dependent: bool = DEFAULT_HAZE_HEIGHT_DEPENDENT
    # The app's angles scaled by DEFAULT_LOD_STRENGTH. These are the angles
    # themselves, because the shader wants tan(theta); a caller that thinks
    # in strength (witness's lod) multiplies.
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
    # Read the sun's optical depth at each cloud sample from the baked light
    # cache instead of marching to the sun (the LIGHT_CACHE switch). The
    # marched sun has a penumbra and the cache's does not.
    light_cache: bool = False
    # The scene clock (s): where the city's steam is in its motion.
    # The browser advances it while flying; a still states it (witness
    # --time) and a track carries it per sample.
    scene_time: float = 0.0
    # The city's roof clutter density (1 is the density the roofs
    # were tuned at, tiers.QUALITY_PRESETS' city_roof_density the tiers'). Packed over
    # the city only.
    city_roof_density: float = 1.0


def pack_uniforms(state: SceneState, view: ViewState) -> np.ndarray:
    """The uniform block, row for row with web/soar/uniforms.js and with the
    Uniforms struct in shader/uniforms.wgsl, which says what each slot is.

    Returns (21, 4) float32. Every unwritten slot stays zero, which is what
    the absent nest rows require.

    Over the city the layout is the same and a handful of rows change
    MEANING: the surface rows describe the city tile rather than the sea, the
    sun row is the moon, and rows 9-13 carry the night colours instead of
    look's daytime air-mass model. See SceneState.
    """
    _unit_interval("jitter_scale", view.jitter_scale)
    _unit_interval("spectral_lighting_strength", view.spectral_lighting_strength)
    if not (look.HAZE_MIN <= view.haze <= look.HAZE_MAX):
        raise ValueError(
            f"haze must be in [{look.HAZE_MIN}, {look.HAZE_MAX}]; "
            f"got {view.haze}.")
    _lod_degrees("light_march_lod_degrees", view.light_march_lod_degrees)
    _lod_degrees("view_step_lod_degrees", view.view_step_lod_degrees)
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
    # An error, not a clamp: a periodic light march exits only through the
    # domain top, so a sun at or below the horizon has no exit and the
    # picture is wrong in a way that still looks plausible.
    if state.periodic and view.sun_elevation <= 0.0:
        raise ValueError(
            "A periodic domain needs the sun above the horizon; got "
            f"sun_elevation={view.sun_elevation}. The light march exits "
            "through the domain top.")
    # The surface flag is what draws the city; with it off the scene is a
    # night sky over an empty ground plane, which looks like a renderer bug.
    check_surface(state.surface)
    city = state.surface == "city"
    if city and not state.ocean_enabled:
        raise ValueError(
            "city mode needs the surface enabled: ocean_enabled is the "
            "surface flag under CITY and it is False, which would "
            "render an empty ground plane.")

    forward, right, up = camera_basis(view.azimuth, view.elevation)
    # Over the city this is the MOON: the same fields, packing and row.
    sun = direction_from_azimuth_elevation(view.sun_azimuth, view.sun_elevation)
    if city:
        cloud_sun, ambient, horizon, bloom, disc = (
            NIGHT_MOON_CLOUD_COLOR, NIGHT_AMBIENT_TINT, NIGHT_SKY_HORIZON,
            NIGHT_MOON_BLOOM, NIGHT_MOON_DISC)
        # The light-transfer split fades in as the sun drops. It means
        # nothing at night, so it is off.
        split = 0.0
    else:
        cloud_sun, ambient, horizon, bloom, disc = look._spectral_lighting_colors(
            tuple(sun), look.SUN_COLOR, view.spectral_lighting_strength)
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
    if state.nested:
        u[7] = (*state.nest_bmin, state.dt_view_nest)
        u[8] = (*state.nest_bmax, state.dt_light_nest)
    u[9] = (*cloud_sun, split)
    u[10] = (*ambient, view.haze)
    u[11] = (*horizon, look.aerial_beta_per_km(view.haze) * 1e-3)
    u[12] = (*bloom, look.AERIAL_SCALE_HEIGHT_M
                     if view.haze_height_dependent else 0.0)
    u[13] = (*disc, look.ocean_haze_extinction_per_km(view.haze) * 1e-3)
    # The surface: the water's reflectance (none over the city) and its z.
    if city:
        u[14] = (0.0, 0.0, 0.0, state.ocean_z)
    else:
        u[14] = (*state.ocean_reflectance, state.ocean_z)
    # Cell size, tile extent, surface flag and top mip of whichever tile is
    # bound.
    u[15] = (state.ocean_fif_dx, state.ocean_tile_extent,
             1.0 if state.ocean_enabled else 0.0, float(state.ocean_max_lod))
    # The surface offset folded into [0, surface period) (see
    # SceneState.surface_offset_m, and surface_offset in shader/uniforms.wgsl)
    # and the scene clock. The fold is written exactly as the browser's
    # (uniforms.js foldTile): one fmod, then a conditional add.
    tile = (state.surface_period_m if state.surface_period_m is not None
            else state.ocean_tile_extent)
    if not math.isfinite(view.scene_time):
        raise ValueError(f"scene_time must be finite; got {view.scene_time}.")

    def _fold_tile(v: float) -> float:
        r = math.fmod(float(v), tile)
        return r + tile if r < 0 else r

    base = (state.surface_offset_m if state.surface_offset_m is not None
            else (state.city_tile_offset_m if city else (0.0, 0.0)))
    # The clock is packed only for a surface with a period (the city's
    # steam), wrapped the way uniforms.js wraps it: in float64, before the
    # narrowing to float32. A surface with no clock packs 0 whatever the
    # scene time, so the browser's accumulation never restarts for it.
    clock = 0.0
    period = state.clock_period_s
    if view.scene_time != 0.0:
        if period is UNSTATED:
            raise ValueError(
                "a running clock needs the surface's clock_period_s — its "
                "meta.json's, via surface_clock_period_s(): a period, or None "
                "for a surface with no clock.")
        if period is not None:
            if not period > 0:
                raise ValueError(
                    f"clock_period_s must be positive or None; got {period!r}.")
            clock = math.fmod(float(view.scene_time), period)
            if clock < 0:
                clock += period
    u[16] = (_fold_tile(base[0]), _fold_tile(base[1]), clock, 0.0)
    u[17] = (w, h, 1.0 if view.subpixel else 0.0, view.jitter_scale)
    u[18] = (view.tone_map_white_point, view.tone_map_gamma, view.contrast,
             0.0)
    # The view-step angle is at least a QUARTER of the render's pixel angle.
    # The accumulation is jittered, so a converged image resolves detail
    # below one pixel, and a floor of a whole pixel would step over the
    # detail the accumulation is there to antialias. An exact 0 stays 0: it
    # means LOD off, a fixed step.
    pixel_tan = 2.0 * math.tan(view.fov * DEG * 0.5) / w
    view_step_tan = math.tan(view.view_step_lod_degrees * DEG)
    if view_step_tan > 0.0:
        view_step_tan = max(view_step_tan, 0.25 * pixel_tan)
    # The roof clutter density is the city's; zero over the sea, whose shader
    # module does not read it.
    roof_density = 0.0
    if city:
        lo, hi = CITY_ROOF_DENSITY_RANGE
        if not lo <= view.city_roof_density <= hi:
            raise ValueError(
                f"city_roof_density must be in [{lo}, {hi}] (the far "
                f"mean's measured range); got {view.city_roof_density}.")
        roof_density = view.city_roof_density
    u[19] = (math.tan(view.light_march_lod_degrees * DEG), view_step_tan,
             roof_density, 0.0)
    u[20] = (1.0 if state.periodic else 0.0,
             1.0 if view.light_cache else 0.0, 0.0, 0.0)
    return u


# --- coordinates -----------------------------------------------------------

def camera_world_origin(rel, bmin, bmax) -> np.ndarray:
    """Relative camera position to absolute metres.

    x and y span the AABB over [-1, 1]; z does NOT — it is anchored to the
    physical surface, so rel_z = -1 is sea level rather than the box floor.
    A domain that starts above the ground therefore keeps its real altitude;
    treating z like x and y would put the camera underground.
    """
    bmin = np.asarray(bmin, np.float64)
    bmax = np.asarray(bmax, np.float64)
    return np.array([
        bmin[0] + (rel[0] + 1.0) * 0.5 * (bmax[0] - bmin[0]),
        bmin[1] + (rel[1] + 1.0) * 0.5 * (bmax[1] - bmin[1]),
        (rel[2] + 1.0) * 0.5 * bmax[2],
    ], dtype=np.float64)
