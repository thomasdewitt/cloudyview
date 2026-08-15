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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from . import look

WEB_SOAR = Path(__file__).resolve().parents[1] / "web" / "soar"
SHADER_PATH = WEB_SOAR / "raymarch.wgsl"
OCEAN_DIR = WEB_SOAR / "ocean"


def read_shader() -> str:
    """The shader source, or a diagnosis of why there isn't any.

    `web/soar/` sits beside the package rather than inside it, so this
    resolves only in a source checkout. `pip install`, from PyPI or from git,
    copies just `cloudyview/` and leaves the shader behind — at which point
    the bare FileNotFoundError names a path under site-packages and reads like
    a bug in the library. Say what actually happened instead.
    """
    if not SHADER_PATH.exists():
        raise FileNotFoundError(
            f"the ray-marching shader is missing: {SHADER_PATH}\n"
            "cloudyview renders from web/soar/raymarch.wgsl, which lives "
            "beside the package rather than inside it, so witness and soar "
            "need a source checkout:\n"
            "    git clone https://github.com/thomasdewitt/cloudyview\n"
            "    cd cloudyview && uv sync\n"
            "A pip install of the package alone does not carry the shader. "
            "glimpse and behold do not need it and still work.")
    return SHADER_PATH.read_text()

UNIFORM_ROWS = 23
UNIFORM_NBYTES = UNIFORM_ROWS * 16          # 368; the scene key compares all of it

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
    """Everything about the field, independent of where the camera is."""
    bmin: Sequence[float]
    bmax: Sequence[float]
    dt_view: float
    dt_light: float
    periodic: bool = True
    ocean_z: float = 0.0
    ocean_reflectance: Sequence[float] = DEFAULT_OCEAN_REFLECTANCE
    ocean_fif_dx: float = 0.2
    ocean_tile_extent: float = 102.4
    ocean_enabled: bool = True
    ocean_max_lod: int = 9
    nested: bool = False
    nest_bmin: Optional[Sequence[float]] = None
    nest_bmax: Optional[Sequence[float]] = None
    dt_view_nest: float = 0.0
    dt_light_nest: float = 0.0


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


def pack_uniforms(state: SceneState, view: ViewState) -> np.ndarray:
    """The 368-byte block, row for row with web/soar/uniforms.js.

    Returns (23, 4) float32. Every unwritten slot stays zero, which is what
    the shader's unused components and the absent nest rows require.
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

    forward, right, up = camera_basis(view.azimuth, view.elevation)
    sun = direction_from_azimuth_elevation(view.sun_azimuth, view.sun_elevation)
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
    u[7] = (w, h, view.g_hg, view.ambient_strength)
    u[8] = (state.ocean_z, *state.ocean_reflectance)
    u[9] = (state.ocean_fif_dx, state.ocean_tile_extent,
            1.0 if state.ocean_enabled else 0.0, float(state.ocean_max_lod))
    u[10] = (1.0 if view.subpixel else 0.0, view.jitter_scale, view.haze,
             view.tone_map_white_point)
    u[11] = (view.gradient_shading_strength, view.deep_shadow_ms_suppression,
             view.ambient_occlusion_strength, view.bounce_depth_attenuation)
    u[12] = (view.gradient_coarse_weight, view.gradient_coarse_radius_m,
             view.ambient_occlusion_floor,
             math.tan(view.cone_stencil_theta_deg * DEG))
    u[13] = (*cloud_sun, view.low_sun_sky_field_strength)
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
    u[20] = (1.0 if state.periodic else 0.0,
             math.tan(view.light_march_lod_degrees * DEG),
             math.tan(view.view_step_lod_degrees * DEG),
             view.tone_map_gamma)
    if state.nested:
        u[21] = (*state.nest_bmin, state.dt_view_nest)
        u[22] = (*state.nest_bmax, state.dt_light_nest)
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


def specialize(source: str, *, periodic: bool, nested: bool,
               max_light_steps: int, tone_map: bool = True) -> str:
    """Bake the three compile-time constants into the shader source.

    Textual replacement rather than WGSL `override` because MAX_LIGHT_STEPS
    bounds a loop. Each sentinel must appear exactly once — a miss renders
    the wrong thing at full speed and says nothing.
    """
    if not (1 <= int(max_light_steps) <= 512):
        raise ValueError(f"max_light_steps must be in [1, 512]; got {max_light_steps}.")
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
                 tone_map: bool = True, device=None,
                 shader_source: Optional[str] = None):
        """`shader_source` overrides the checked-in shader.

        It exists so two revisions of raymarch.wgsl can be held against each
        other on one device, in raw float, without a checkout — which is how
        a change claiming to preserve the image gets to prove it (see
        tools/soar_shader_ab.py). Everything else reads the file.
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

        source = specialize(shader_source if shader_source is not None
                            else read_shader(), periodic=self.periodic,
                            nested=self.nested,
                            max_light_steps=self.max_light_steps,
                            tone_map=self.tone_map)
        self._ray_module = device.create_shader_module(code=source)
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
        self._ocean_tex = self._load_ocean()
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

    def _load_ocean(self):
        """The FIF normal tile: 10 pre-renormalised mips, uploaded level by level.

        Not GPU-generated mips — the shader hand-blends two levels because a
        single hardware trilinear fetch's half-texel offset only fits level 0.
        """
        import json
        wgpu = self.wgpu
        meta = json.loads((OCEAN_DIR / "meta.json").read_text())
        n, mips = int(meta["n"]), int(meta["mips"])
        self.ocean_meta = meta
        tex = self.device.create_texture(
            size=(n, n, 1), dimension="2d", format="rgba16float",
            mip_level_count=mips,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        for level in range(mips):
            side = max(1, n >> level)
            raw = (OCEAN_DIR / f"fif_mip{level}.bin").read_bytes()
            expected = side * side * 4 * 2
            if len(raw) != expected:
                raise RuntimeError(
                    f"ocean/fif_mip{level}.bin is {len(raw)} bytes, expected "
                    f"{expected} for a {side}x{side} rgba16float level.")
            self.device.queue.write_texture(
                {"texture": tex, "mip_level": level}, raw,
                {"bytes_per_row": side * 8, "rows_per_image": side},
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
        ])
        self._ray_pipeline = d.create_render_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[self._ray_layout]),
            vertex={"module": self._ray_module, "entry_point": "vs_main"},
            fragment={"module": self._ray_module, "entry_point": "fs_main",
                      "targets": [{"format": SAMPLE_FORMAT}]},
            primitive={"topology": "triangle-list"})

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

    def _ensure_targets(self, size):
        wgpu = self.wgpu
        if self._targets is not None and self._targets["size"] == size:
            return self._targets
        w, h = size
        usage = (wgpu.TextureUsage.RENDER_ATTACHMENT
                 | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC)
        make = lambda fmt: self.device.create_texture(
            size=(w, h, 1), dimension="2d", format=fmt, usage=usage)
        self._targets = {"size": size, "sample": make(SAMPLE_FORMAT),
                         "accum": [make(ACCUM_FORMAT), make(ACCUM_FORMAT)]}
        return self._targets

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
            ])
        return self._bind_group

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
        w, h = view.render_size
        targets = self._ensure_targets((w, h))
        bind_group = self._ensure_bind_group()
        sample_view = targets["sample"].create_view()

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
            rp = enc.begin_render_pass(color_attachments=[{
                "view": sample_view, "load_op": "clear", "store_op": "store",
                "clear_value": (0.0, 0.0, 0.0, 1.0)}])
            rp.set_pipeline(self._ray_pipeline)
            rp.set_bind_group(0, bind_group)
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
