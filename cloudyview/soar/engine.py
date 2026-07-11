"""wgpu volume-render engine for the interactive fly-through (spike).

Holds the extinction volume resident on the GPU as a 3D texture (uploaded
exactly once — per-frame re-upload was the #1 measured bottleneck, see
docs/architecture.md) and renders frames from `cloudyview.Camera` viewpoints
with the WGSL raymarcher in `raymarch.wgsl`.

Two consumers:
- offscreen rendering / benchmarking (`render`, `benchmark`)
- the windowed app (`app.py`), which reuses the pipeline against the
  swapchain texture format.
"""

from dataclasses import dataclass
from pathlib import Path
import struct
from time import perf_counter
from typing import Optional, Tuple

import numpy as np

try:
    import wgpu
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "cloudyview.soar requires wgpu. "
        "Install the interactive extra: uv sync --extra interactive "
        "(or: pip install 'cloudyview[interactive]')."
    ) from e

from .. import optical_depth
from ..camera import Camera
from ..cloudfield import CloudField
from ..angles import direction_from_azimuth_elevation

# Witness and Soar share this dependency-light look module. Importing the
# interactive engine must not import Witness (and therefore Numba) at startup.
from ..look import (
    SUN_COLOR as WITNESS_SUN_COLOR,
    AERIAL_BETA_PER_KM,
    CONE_STENCIL_THETA_DEG,
    AERIAL_PERSPECTIVE_STRENGTH,
    AERIAL_SCALE_HEIGHT_M,
    LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG,
    LIGHT_TRANSFER_FULL_ELEVATION_DEG,
    LIGHT_TRANSFER_SPLIT_STRENGTH,
    LOW_SUN_SKY_FIELD_STRENGTH,
    OCEAN_GLINT_ROUGHNESS,
    OCEAN_GLINT_ROUGHNESS_PER_LOD,
    OCEAN_GLINT_STRENGTH,
    OCEAN_HAZE_EXTINCTION_PER_KM,
    OCEAN_MIP_BIAS,
    OCEAN_REALISM,
    OCEAN_SKY_SHADOW_FLOOR,
    SPECTRAL_LIGHTING_STRENGTH,
    _spectral_lighting_colors,
)

SHADER_PATH = Path(__file__).parent / "raymarch.wgsl"
LEGACY_SHADER_PATH = Path(__file__).parent / "raymarch_legacy.wgsl"

# Defaults mirroring witness.py / config.py.
DEFAULT_SUN_AZIMUTH = 20.0
DEFAULT_SUN_ELEVATION = 55.0
DEFAULT_EXPOSURE = 4.0
DEFAULT_G_HG = 0.76
DEFAULT_AMBIENT_STRENGTH = 0.12
DEFAULT_OCEAN_REFLECTANCE = (0.0020, 0.0045, 0.0126)  # witness.py:104-106
DEFAULT_GRADIENT_SHADING_STRENGTH = 1.50
DEFAULT_GRADIENT_COARSE_WEIGHT = 0.65
DEFAULT_GRADIENT_COARSE_RADIUS_M = 500.0
DEFAULT_DEEP_SHADOW_MS_SUPPRESSION = 0.90
DEFAULT_AMBIENT_OCCLUSION_STRENGTH = 1.00
DEFAULT_AMBIENT_OCCLUSION_FLOOR = 0.24
DEFAULT_BOUNCE_DEPTH_ATTENUATION = 0.80
# Spectral time-of-day lighting / low-sun sky field / light-transfer split /
# aerial perspective (witness realism package, iter_002..iter_008). Defaults
# come straight from the witness tuning block; 0.0 is the documented exact
# legacy path for each mechanism.
DEFAULT_SPECTRAL_LIGHTING_STRENGTH = SPECTRAL_LIGHTING_STRENGTH
DEFAULT_LOW_SUN_SKY_FIELD_STRENGTH = LOW_SUN_SKY_FIELD_STRENGTH
DEFAULT_LIGHT_TRANSFER_SPLIT_STRENGTH = LIGHT_TRANSFER_SPLIT_STRENGTH
DEFAULT_AERIAL_PERSPECTIVE_STRENGTH = AERIAL_PERSPECTIVE_STRENGTH
# Ocean realism (witness iter_004/009/011): footprint-filtered normal mips,
# spectral GGX sun glint, sky-field haze, per-term cloud shadowing.
# ocean_realism=0.0 is the untouched legacy ocean shader.
DEFAULT_OCEAN_REALISM = OCEAN_REALISM
DEFAULT_OCEAN_MIP_BIAS = OCEAN_MIP_BIAS
DEFAULT_OCEAN_GLINT_STRENGTH = OCEAN_GLINT_STRENGTH
DEFAULT_OCEAN_GLINT_ROUGHNESS = OCEAN_GLINT_ROUGHNESS
DEFAULT_OCEAN_GLINT_ROUGHNESS_PER_LOD = OCEAN_GLINT_ROUGHNESS_PER_LOD
DEFAULT_OCEAN_HAZE_EXTINCTION_PER_KM = OCEAN_HAZE_EXTINCTION_PER_KM
DEFAULT_OCEAN_SKY_SHADOW_FLOOR = OCEAN_SKY_SHADOW_FLOOR
# Cone stencil (witness iter_001): the coarse gradient-shading radius
# subtends a fixed angle at the camera (radius = distance * tan(theta)).
# Exact 0.0 selects the legacy fixed-radius coarse stencil.
DEFAULT_CONE_STENCIL_THETA_DEG = CONE_STENCIL_THETA_DEG
# Exact pre-port soar ambient tint (stale vs witness iter_010's retuned
# (0.19, 0.225, 0.30)). The master kill combination reproducing the pre-port
# frame bit-for-bit is: spectral_lighting_strength=0.0,
# low_sun_sky_field_strength=0.0, light_transfer_split_strength=0.0,
# aerial_perspective_strength=0.0, ocean_realism=0.0,
# cone_stencil_theta_deg=0.0, ambient_tint=PRE_PORT_AMBIENT_TINT.
PRE_PORT_AMBIENT_TINT = (0.22, 0.23, 0.28)
# Periodic-domain march caps — keep in exact sync with raymarch.wgsl
# (PERIODIC_AIR_TAU_CUTOFF / PERIODIC_MAX_WRAPS there).
PERIODIC_AIR_TAU_CUTOFF = 3.912023005428146  # -ln(0.02)
PERIODIC_MAX_WRAPS = 2.0
STEP_VOXEL_FACTOR = 2.0  # dt = min voxel dimension * this (witness value)
DEFAULT_MAX_LIGHT_STEPS = 512  # keep in sync with both WGSL modules
# 0.45 was approved pre-realism; with the ported look Thomas flagged the
# trailing smear ('turn down the time-blur') but 0.72 read 'a bit
# speckled' — 0.58 is Thomas's requested midpoint (2026-07-10).
DEFAULT_MOTION_BLEND_ALPHA = 0.58
DEFAULT_MOTION_BLEND_REFERENCE_FPS = 60.0
DEFAULT_MOTION_JITTER_SCALE = 0.65
DEFAULT_MOTION_RESET_ANGLE_DEGREES = 8.0
DEFAULT_MOTION_RESET_TRANSLATION_FRACTION = 0.05

_UNIFORM_NBYTES = 21 * 16  # 21 vec4<f32> (rows documented in write_uniforms)
_ACCUM_UNIFORM_NBYTES = 16  # 4 f32s
_DEFAULT_FIF_NORMALS = None


@dataclass(frozen=True)
class QualityPreset:
    """One named interactive performance preset.

    ``step_factor`` is measured in minimum-volume voxels. Potato's moving
    settings are deliberately temporary: once the camera stops it uses the
    exact High sampling settings so temporal accumulation converges a clean
    still instead of merely averaging a coarse flight frame.
    """

    name: str
    label: str
    render_scale: float
    step_factor: float
    max_light_steps: int


QUALITY_PRESETS = {
    "high": QualityPreset("high", "High", 1.0, 2.0, 512),
    "medium": QualityPreset("medium", "Medium", 0.75, 2.5, 384),
    "low": QualityPreset("low", "Low", 0.60, 3.0, 256),
    "potato": QualityPreset(
        "potato", "Potato — smooth stills, rough flight", 0.25, 4.0, 128
    ),
}
QUALITY_TIER_NAMES = tuple(QUALITY_PRESETS)
DEFAULT_QUALITY_TIER = "high"
MIN_RENDER_SCALE = 0.25
MAX_RENDER_SCALE = 1.0


def render_target_size(
    size: Tuple[int, int], render_scale: float
) -> Tuple[int, int]:
    """Deterministic scaled target size, rounded to the nearest pixel."""
    scale = _validate_finite_float("render_scale", render_scale)
    if not MIN_RENDER_SCALE <= scale <= MAX_RENDER_SCALE:
        raise ValueError(
            f"render_scale must be in [{MIN_RENDER_SCALE}, "
            f"{MAX_RENDER_SCALE}]; got {scale!r}."
        )
    w, h = (int(size[0]), int(size[1]))
    if w < 1 or h < 1:
        raise ValueError(f"render size must be positive; got {(w, h)!r}.")
    return (
        max(1, int(np.floor(w * scale + 0.5))),
        max(1, int(np.floor(h * scale + 0.5))),
    )


def choose_quality_tier(
    frame_times_ms: dict[str, float], *, target_ms: float = 1000.0 / 60.0
) -> str:
    """Choose the highest measured tier at or below ``target_ms``."""
    target_ms = _validate_positive_float("target_ms", target_ms)
    missing = [name for name in QUALITY_TIER_NAMES if name not in frame_times_ms]
    if missing:
        raise ValueError(f"Missing benchmark times for: {', '.join(missing)}.")
    for name in QUALITY_TIER_NAMES:
        value = _validate_positive_float(
            f"frame_times_ms[{name!r}]", frame_times_ms[name]
        )
        if value <= target_ms:
            return name
    return "potato"


_ACCUM_SHADER = """
struct AccumUniforms {
    prev_weight: f32,
    sample_weight: f32,
    _pad0: f32,
    _pad1: f32,
};

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
    let sample = textureLoad(sample_tex, xy, 0);
    if (au.prev_weight <= 0.0) {
        return vec4<f32>(sample.rgb, 1.0);
    }
    let prev = textureLoad(prev_tex, xy, 0);
    return vec4<f32>(
        prev.rgb * au.prev_weight + sample.rgb * au.sample_weight,
        1.0
    );
}
"""

_PRESENT_SHADER = """
@group(0) @binding(0) var src_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(frag_pos.xy);
    let src = textureLoad(src_tex, xy, 0);
    return vec4<f32>(src.rgb, 1.0);
}
"""

_UPSCALE_SHADER = """
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    var out: VertexOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * (x + 1.0), 0.5 * (1.0 - y));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(
        textureSampleLevel(src_tex, src_samp, in.uv, 0.0).rgb,
        1.0
    );
}
"""


def _volume_aabb(field: CloudField) -> Tuple[np.ndarray, np.ndarray]:
    """Absolute-meter AABB with half-cell padding (matches witness())."""
    x = np.asarray(field.x, dtype=np.float64)
    y = np.asarray(field.y, dtype=np.float64)
    z = np.asarray(field.z, dtype=np.float64)
    dx_half = 0.5 * abs(x[1] - x[0])
    dy_half = 0.5 * abs(y[1] - y[0])
    dz_lo_half = 0.5 * abs(z[1] - z[0])
    dz_hi_half = 0.5 * abs(z[-1] - z[-2])
    bmin = np.array([x.min() - dx_half, y.min() - dy_half, z.min() - dz_lo_half])
    bmax = np.array([x.max() + dx_half, y.max() + dy_half, z.max() + dz_hi_half])
    return bmin, bmax


def _validate_finite_float(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return value


def _validate_unit_interval(name: str, value: float) -> float:
    value = _validate_finite_float(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {value!r}.")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    value = _validate_finite_float(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0; got {value!r}.")
    return value


def _effective_light_transfer_split(
    strength: float, sun_elevation_deg: float
) -> float:
    """Elevation fade of the light-transfer split (witness._render_levels).

    Full strength at/below LIGHT_TRANSFER_FULL_ELEVATION_DEG, smoothstepped to
    exactly 0.0 at LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG so the approved
    high-sun look is untouched. Mirrors witness.py's per-frame precompute
    (do NOT change one without the other).
    """
    strength = _validate_finite_float("light_transfer_split_strength", strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            f"light_transfer_split_strength must be in [0, 1]; got {strength!r}."
        )
    if sun_elevation_deg >= LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG - 1e-6:
        return 0.0
    if sun_elevation_deg > LIGHT_TRANSFER_FULL_ELEVATION_DEG:
        low_sun_mix = (
            (LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG - sun_elevation_deg)
            / (LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG
               - LIGHT_TRANSFER_FULL_ELEVATION_DEG)
        )
        low_sun_mix = low_sun_mix * low_sun_mix * (3.0 - 2.0 * low_sun_mix)
        return strength * low_sun_mix
    return strength


def periodic_march_cap_m(
    cam_z: float,
    direction,
    bmin,
    bmax,
    *,
    aerial_perspective_strength: float = DEFAULT_AERIAL_PERSPECTIVE_STRENGTH,
) -> float:
    """Python mirror of the shader's ``periodic_march_cap`` (raymarch.wgsl).

    Distance at which camera->sample clear-air transmittance through the
    aerial-perspective exponential atmosphere drops to ~2%, bounded by the
    PERIODIC_MAX_WRAPS horizontal-travel ceiling. Used by the app to decide
    whether a periodic view marches past the finite Mitsuba volume (the
    behold hand-off notice); keep in exact sync with the WGSL.
    """
    direction = np.asarray(direction, dtype=np.float64)
    cap = np.inf
    h_len = float(np.hypot(direction[0], direction[1]))
    if h_len > 1e-8:
        extent = np.asarray(bmax, dtype=np.float64) - np.asarray(
            bmin, dtype=np.float64
        )
        cap = PERIODIC_MAX_WRAPS * max(extent[0], extent[1]) / h_len
    if aerial_perspective_strength > 0.0:
        beta0 = AERIAL_BETA_PER_KM * 1e-3
        scale_h = AERIAL_SCALE_HEIGHT_M
        z0 = max(float(cam_z), 0.0)
        mu = float(direction[2])
        tau_cap = PERIODIC_AIR_TAU_CUTOFF / aerial_perspective_strength
        e0 = np.exp(-z0 / scale_h)
        if abs(mu) > 1e-6:
            a = e0 - tau_cap * mu / (beta0 * scale_h)
            if a > 0.0:
                t_sol = (-scale_h * np.log(a) - z0) / mu
                if t_sol > 0.0:
                    cap = min(cap, float(t_sol))
        else:
            cap = min(cap, float(tau_cap / (beta0 * e0)))
    return float(cap)


def camera_world_origin(camera: Camera, bmin, bmax) -> np.ndarray:
    """Relative camera position -> absolute meters (matches witness()).

    x, y: rel [-1, 1] spans the AABB. z is anchored to the physical surface
    (rel -1 -> z=0, rel +1 -> top of the data domain), so elevated domains
    keep their real altitude.
    """
    rel = camera.position
    return np.array([
        bmin[0] + (rel[0] + 1.0) * 0.5 * (bmax[0] - bmin[0]),
        bmin[1] + (rel[1] + 1.0) * 0.5 * (bmax[1] - bmin[1]),
        (rel[2] + 1.0) * 0.5 * bmax[2],
    ])


def _ghost_face_arrays(sigma: np.ndarray) -> dict:
    """Periodic x/y ghost-border faces for the padded volume texture.

    The padded texture is (w=nz+2, h=ny+2, d=nx+2); texture depth indexes x
    and texture rows index y (see the upload swizzle note in __init__). The
    x faces are single depth slices of shape (ny+2, nz+2); the y faces span
    every depth slice with shape (nx+2, 1, nz+2) so one write_texture call
    covers the whole row. Corner texels wrap in both x and y (they are the
    trilinear support for samples near a domain corner); the z ghost columns
    stay zero — the vertical taper is not periodic.
    """
    nx, ny, nz = sigma.shape
    dtype = sigma.dtype
    x_lo = np.zeros((ny + 2, nz + 2), dtype=dtype)  # texture depth 0
    x_hi = np.zeros((ny + 2, nz + 2), dtype=dtype)  # texture depth nx+1
    x_lo[1:-1, 1:-1] = sigma[-1]
    x_lo[0, 1:-1] = sigma[-1, -1]
    x_lo[-1, 1:-1] = sigma[-1, 0]
    x_hi[1:-1, 1:-1] = sigma[0]
    x_hi[0, 1:-1] = sigma[0, -1]
    x_hi[-1, 1:-1] = sigma[0, 0]
    y_lo = np.zeros((nx + 2, 1, nz + 2), dtype=dtype)  # texture row 0
    y_hi = np.zeros((nx + 2, 1, nz + 2), dtype=dtype)  # texture row ny+1
    y_lo[1:-1, 0, 1:-1] = sigma[:, -1]
    y_lo[0, 0, 1:-1] = sigma[-1, -1]
    y_lo[-1, 0, 1:-1] = sigma[0, -1]
    y_hi[1:-1, 0, 1:-1] = sigma[:, 0]
    y_hi[0, 0, 1:-1] = sigma[-1, 0]
    y_hi[-1, 0, 1:-1] = sigma[0, 0]
    return {"x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi}


def _default_fif_normals():
    """Generate and cache the witness-default FIF normal map once per process."""
    global _DEFAULT_FIF_NORMALS
    if _DEFAULT_FIF_NORMALS is None:
        from ..ocean_fif import generate_fif_normals
        # Defaults are the witness ocean tile parameters: N/dx/outer scale/etc.
        # live in ocean_fif.py:16-24 and generate_fif_normals() uses them.
        _DEFAULT_FIF_NORMALS = generate_fif_normals(verbose=False)
    return _DEFAULT_FIF_NORMALS


def _validate_fif_normals(fif_normals):
    nx, ny, nz, dx = fif_normals
    nx = np.ascontiguousarray(nx, dtype=np.float32)
    ny = np.ascontiguousarray(ny, dtype=np.float32)
    nz = np.ascontiguousarray(nz, dtype=np.float32)
    if nx.ndim != 2 or nx.shape != ny.shape or nx.shape != nz.shape:
        raise ValueError(
            "fif_normals must contain matching 2D nx/ny/nz arrays; "
            f"got {nx.shape}, {ny.shape}, {nz.shape}."
        )
    if nx.shape[0] != nx.shape[1]:
        raise ValueError(
            "The witness FIF sampler assumes a square periodic tile; "
            f"got {nx.shape}."
        )
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"FIF dx must be positive and finite; got {dx!r}.")
    return nx, ny, nz, float(dx)


def _build_fif_normal_mips(base: np.ndarray) -> list[np.ndarray]:
    """Average and renormalize a periodic normal-map mip chain on the CPU."""
    mips = [np.ascontiguousarray(base, dtype=np.float32)]
    cur = mips[0]
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h, w, _ = cur.shape
        nh = max(1, h // 2)
        nw = max(1, w // 2)
        y0 = (np.arange(nh) * 2) % h
        y1 = (y0 + 1) % h
        x0 = (np.arange(nw) * 2) % w
        x1 = (x0 + 1) % w
        down = (
            cur[y0[:, None], x0[None, :], :3]
            + cur[y1[:, None], x0[None, :], :3]
            + cur[y0[:, None], x1[None, :], :3]
            + cur[y1[:, None], x1[None, :], :3]
        ) * np.float32(0.25)
        length = np.linalg.norm(down, axis=-1, keepdims=True)
        down = down / np.maximum(length, np.float32(1e-12))
        level = np.empty((nh, nw, 4), dtype=np.float32)
        level[..., :3] = down
        level[..., 3] = 1.0
        mips.append(np.ascontiguousarray(level))
        cur = mips[-1]
    return mips


class InteractiveRenderer:
    """Resident-volume WGSL raymarcher for a single CloudField.

    Parameters
    ----------
    field : CloudField
        The loaded cloud volume (see :func:`cloudyview.load`).
    extinction_multiplier : float
        Scales the physical extinction field (witness config default 1.0).
    periodic : bool
        Tile the volume horizontally (SAM LES domains are doubly periodic
        in x/y): density sampling wraps, the view march never exits
        sideways, and the light march exits through the domain top. On by
        default; turn off for subvolume cutouts (which are not physically
        periodic and would show seams). Off reproduces the finite-box
        behavior bit-for-bit. Toggle later with :meth:`set_periodic`.
    device : wgpu.GPUDevice, optional
        Reuse an existing device (the windowed app shares one with its
        canvas). Must have the ``float32-filterable`` feature.
    quality_tier : {"high", "medium", "low", "potato"}
        Initial explicit preset. The engine default is High so library callers
        retain the reference behavior; the window app's CLI defaults to auto.
    volume_fp16 : bool
        Store extinction in r16float instead of reference r32float. This is a
        load-time choice because the engine deliberately keeps no second full
        CPU copy for live re-upload.
    """

    _ACCUM_FORMAT = wgpu.TextureFormat.rgba16float

    def __init__(
        self,
        field: CloudField,
        *,
        extinction_multiplier: float = 1.0,
        periodic: bool = True,
        ocean_enabled: bool = True,
        ocean_z: float = 0.0,
        ocean_reflectance: Tuple[float, float, float] = DEFAULT_OCEAN_REFLECTANCE,
        fif_normals: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = None,
        motion_accumulation: bool = True,
        motion_blend_alpha: float = DEFAULT_MOTION_BLEND_ALPHA,
        motion_blend_reference_fps: float = DEFAULT_MOTION_BLEND_REFERENCE_FPS,
        motion_jitter_scale: float = DEFAULT_MOTION_JITTER_SCALE,
        motion_reset_angle_degrees: float = DEFAULT_MOTION_RESET_ANGLE_DEGREES,
        motion_reset_translation_m: Optional[float] = None,
        quality_tier: str = DEFAULT_QUALITY_TIER,
        volume_fp16: bool = False,
        device=None,
    ):
        self.field = field
        self.bmin, self.bmax = _volume_aabb(field)
        self.periodic = bool(periodic)
        self.ocean_enabled = bool(ocean_enabled)
        self.ocean_z = float(ocean_z)
        self.ocean_reflectance = tuple(float(c) for c in ocean_reflectance)
        self.volume_fp16 = bool(volume_fp16)
        self.motion_accumulation = bool(motion_accumulation)
        self.motion_blend_alpha = _validate_unit_interval(
            "motion_blend_alpha", motion_blend_alpha
        )
        self.motion_blend_reference_fps = _validate_positive_float(
            "motion_blend_reference_fps", motion_blend_reference_fps
        )
        self.motion_jitter_scale = _validate_unit_interval(
            "motion_jitter_scale", motion_jitter_scale
        )
        self.motion_reset_angle_degrees = _validate_positive_float(
            "motion_reset_angle_degrees", motion_reset_angle_degrees
        )
        if motion_reset_translation_m is None:
            horizontal_extent = min(
                float(self.bmax[0] - self.bmin[0]),
                float(self.bmax[1] - self.bmin[1]),
            )
            motion_reset_translation_m = (
                DEFAULT_MOTION_RESET_TRANSLATION_FRACTION * horizontal_extent
            )
        self.motion_reset_translation_m = _validate_positive_float(
            "motion_reset_translation_m", motion_reset_translation_m
        )

        nx, ny, nz = field.shape
        extent = self.bmax - self.bmin
        voxel = np.array([extent[0] / nx, extent[1] / ny, extent[2] / nz])
        self._min_voxel_m = float(voxel.min())
        self.quality_tier = DEFAULT_QUALITY_TIER
        self.render_scale = 1.0
        self.step_factor = STEP_VOXEL_FACTOR
        self.max_light_steps = DEFAULT_MAX_LIGHT_STEPS
        self._camera_moving = False
        self.set_quality_tier(quality_tier, camera_moving=False)

        if device is None:
            device = request_device()
        self.device = device
        if "float32-filterable" not in device.features:
            raise RuntimeError(
                "wgpu device lacks the 'float32-filterable' feature required "
                "for hardware trilinear sampling of float32 textures "
                "(density in fp32 mode and the ocean normal map). Refusing "
                "to fall back to nearest-neighbor."
            )

        # --- Extinction volume -> resident 3D texture (uploaded ONCE). ---
        iwc = field.iwc
        if iwc is not None and float(np.max(iwc)) < 1e-6:
            iwc = None
        sigma = optical_depth.compute_extinction_field(
            field.lwc, field.z, re=10.0, iwc=iwc, re_ice=30.0
        )
        if extinction_multiplier != 1.0:
            sigma = sigma * np.float32(extinction_multiplier)
        sigma_dtype = np.float16 if self.volume_fp16 else np.float32
        sigma = np.ascontiguousarray(sigma, dtype=sigma_dtype)

        # Bake witness's ghost-zero boundary into the resident texture. The
        # public AABB remains the unpadded level extent, where witness samples
        # with gx = (p - bmin) / dx and dx = (bmax - bmin) / N. Padding shifts
        # original voxel i to padded texel i+1; the shader maps
        #     texel = gx + 1, texcoord = (texel + 0.5) / (N + 2),
        # so p=bmin+i*dx still lands exactly on original sigma[i], while
        # gx in [-1,0) and [N-1,N) filters against the zero ghost texels.
        # In periodic mode the x/y ghost texels are instead filled from the
        # OPPOSITE faces so hardware trilinear filtering is exact across the
        # wrap seam (the shader wraps its sample coordinate into [0, N));
        # z keeps the ghost-zero taper. The border alone is rewritten on
        # set_periodic() so toggling never re-uploads the volume.
        sigma_padded = np.zeros(
            (nx + 2, ny + 2, nz + 2), dtype=sigma_dtype
        )
        sigma_padded[1:-1, 1:-1, 1:-1] = sigma
        self._ghost_faces = _ghost_face_arrays(sigma)
        if self.periodic:
            sigma_padded[0, :, :] = self._ghost_faces["x_lo"]
            sigma_padded[-1, :, :] = self._ghost_faces["x_hi"]
            sigma_padded[:, 0, :] = self._ghost_faces["y_lo"][:, 0, :]
            sigma_padded[:, -1, :] = self._ghost_faces["y_hi"][:, 0, :]

        # Zero-reshuffle upload: a C-order (nx, ny, nz) array already has z
        # fastest, so it maps directly onto a texture with width=nz,
        # height=ny, depth=nx. The shader swizzles sample coords to match.
        max_dim = self.device.limits["max-texture-dimension-3d"]
        if max(sigma_padded.shape) > max_dim:
            raise ValueError(
                f"Padded volume {nx + 2}x{ny + 2}x{nz + 2} exceeds the "
                f"device's 3D texture "
                f"limit ({max_dim}); bricking/LOD is out of scope for the "
                "spike (docs/architecture.md)."
            )
        # COPY_SRC so tests can read the resident texels back and verify the
        # ghost-border content (periodic wrap vs ghost zero); free otherwise.
        self.volume_texture_format = (
            wgpu.TextureFormat.r16float
            if self.volume_fp16
            else wgpu.TextureFormat.r32float
        )
        self._texture = self.device.create_texture(
            label="cloud-sigma",
            size=(nz + 2, ny + 2, nx + 2),
            format=self.volume_texture_format,
            dimension="3d",
            usage=(wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_DST
                   | wgpu.TextureUsage.COPY_SRC),
        )
        self.device.queue.write_texture(
            {"texture": self._texture},
            sigma_padded,
            {
                "bytes_per_row": (nz + 2) * sigma_padded.dtype.itemsize,
                "rows_per_image": ny + 2,
            },
            (nz + 2, ny + 2, nx + 2),
        )
        self.volume_nbytes = sigma_padded.nbytes

        if fif_normals is None:
            fif_normals = _default_fif_normals()
        fif_nx, fif_ny, fif_nz, fif_dx = _validate_fif_normals(fif_normals)
        self.ocean_fif_normals = (fif_nx, fif_ny, fif_nz, fif_dx)
        self.ocean_fif_dx = fif_dx
        self.ocean_tile_extent = float(fif_nx.shape[0]) * fif_dx
        ocean_normals = np.empty((*fif_nx.shape, 4), dtype=np.float32)
        ocean_normals[..., 0] = fif_nx
        ocean_normals[..., 1] = fif_ny
        ocean_normals[..., 2] = fif_nz
        ocean_normals[..., 3] = 1.0
        ocean_mips = _build_fif_normal_mips(ocean_normals)
        fif_n = fif_nx.shape[0]
        max_dim_2d = self.device.limits["max-texture-dimension-2d"]
        if fif_n > max_dim_2d:
            raise ValueError(
                f"FIF normal tile {fif_n}x{fif_n} exceeds the device's 2D "
                f"texture limit ({max_dim_2d})."
            )
        self._ocean_texture = self.device.create_texture(
            label="ocean-fif-normals",
            size=(fif_n, fif_n, 1),
            mip_level_count=len(ocean_mips),
            format=wgpu.TextureFormat.rgba32float,
            dimension="2d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        for level, mip in enumerate(ocean_mips):
            h, w, _ = mip.shape
            self.device.queue.write_texture(
                {"texture": self._ocean_texture, "mip_level": level},
                mip,
                {"bytes_per_row": w * 4 * 4, "rows_per_image": h},
                (w, h, 1),
            )
        self.ocean_mip_count = len(ocean_mips)
        self.ocean_max_lod = float(len(ocean_mips) - 1)
        self.ocean_nbytes = sum(mip.nbytes for mip in ocean_mips)

        # TODO(occupancy-grid): build a coarse boolean brick grid from
        # `sigma` here and bind it in the shader for empty-space skipping —
        # the known next lever for full-domain (1024^2) interactivity.

        self._sampler = self.device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
        )
        self._ocean_sampler = self.device.create_sampler(
            address_mode_u="repeat",
            address_mode_v="repeat",
            mag_filter="linear",
            min_filter="linear",
            mipmap_filter="linear",
        )
        self._uniform_buf = self.device.create_buffer(
            size=_UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._shader_source = SHADER_PATH.read_text()
        self._legacy_shader_source = LEGACY_SHADER_PATH.read_text()
        self._shader = self.device.create_shader_module(code=self._shader_source)
        # A separate, untouched pre-periodic module is deliberate: even a
        # pipeline-specialized branch in the combined module changed register
        # allocation/arithmetic enough to move a few output bytes on Vulkan.
        # OFF therefore uses the exact legacy instruction graph and carries
        # zero periodic runtime cost; ON is free to optimize independently.
        self._legacy_shader = self.device.create_shader_module(
            code=self._legacy_shader_source
        )
        self._shader_modules = {
            (True, DEFAULT_MAX_LIGHT_STEPS): self._shader,
            (False, DEFAULT_MAX_LIGHT_STEPS): self._legacy_shader,
        }
        self._accum_uniform_buf = self.device.create_buffer(
            size=_ACCUM_UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._accum_shader = self.device.create_shader_module(
            code=_ACCUM_SHADER
        )
        self._present_shader = self.device.create_shader_module(
            code=_PRESENT_SHADER
        )
        self._upscale_shader = self.device.create_shader_module(
            code=_UPSCALE_SHADER
        )
        self._upscale_sampler = self.device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
        )
        self._bind_group_layout = self.device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": "float",
                    "view_dimension": "3d",
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": "float",
                    "view_dimension": "2d",
                },
            },
            {
                "binding": 4,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
        ])
        self._accum_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
            ]
        )
        self._present_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
            ]
        )
        self._upscale_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        self._bind_group = self.device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._uniform_buf,
                                            "offset": 0,
                                            "size": _UNIFORM_NBYTES}},
                {"binding": 1, "resource": self._texture.create_view()},
                {"binding": 2, "resource": self._sampler},
                {"binding": 3, "resource": self._ocean_texture.create_view()},
                {"binding": 4, "resource": self._ocean_sampler},
            ],
        )
        self._pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._accum_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._accum_bind_group_layout]
        )
        self._present_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._present_bind_group_layout]
        )
        self._upscale_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._upscale_bind_group_layout]
        )
        self._pipelines = {}  # (target format, periodic, light steps) -> pipeline
        self._accum_pipeline = None
        self._present_pipelines = {}  # target format -> render pipeline
        self._upscale_pipelines = {}  # target format -> render pipeline

        # Offscreen target cache: (w, h) -> texture.
        self._offscreen = None
        self._scaled_sample = None
        self._accum_targets = None
        self._accum_key = None
        self._accum_count = 0
        self._accum_index = 0
        self._accum_motion = False
        self._accum_last_origin = None
        self._accum_last_forward = None
        self._last_motion_delta = None
        self._last_motion_reset = False
        self._current_uniform_key = None
        self._current_uniform_size = None
        self._current_output_size = None
        self._current_uniform = None
        self._current_jitter = False
        self._current_subpixel = False
        self._current_jitter_scale = 1.0

        # The HUD minimap: static glimpse albedo texture + tiny per-frame
        # camera/FOV uniform. Created with the renderer so the map is loaded
        # once alongside the resident volume.
        from .hud import MinimapHUD
        self._hud = MinimapHUD(self)

        # The flying subject (bird.py), created on first use. Offscreen
        # rendering opts in with render(..., bird=True); the windowed app
        # drives it every frame.
        self._bird = None

    @property
    def hud(self):
        """The :class:`~cloudyview.soar.hud.MinimapHUD` for this renderer."""
        return self._hud

    @property
    def bird(self):
        """The :class:`~cloudyview.soar.bird.Bird` for this renderer (lazy)."""
        if self._bird is None:
            from .bird import Bird
            self._bird = Bird(self)
        return self._bird

    # ------------------------------------------------------------------
    # Interactive quality
    # ------------------------------------------------------------------

    @property
    def dt_view(self) -> float:
        return self._min_voxel_m * self.step_factor

    @property
    def dt_light(self) -> float:
        return self.dt_view

    @property
    def flight_render_scale(self) -> float:
        """Render scale used while moving (the slider-facing value)."""
        return self._flight_render_scale

    @property
    def quality_is_custom(self) -> bool:
        preset = QUALITY_PRESETS[self.quality_tier]
        return self._flight_render_scale != preset.render_scale

    def _apply_effective_quality(self) -> None:
        preset = QUALITY_PRESETS[self.quality_tier]
        effective = preset
        if self.quality_tier == "potato" and not self._camera_moving:
            effective = QUALITY_PRESETS["high"]
        new_values = (
            1.0 if effective.name == "high" and self.quality_tier == "potato"
            else self._flight_render_scale,
            effective.step_factor,
            effective.max_light_steps,
        )
        old_values = (
            getattr(self, "render_scale", None),
            getattr(self, "step_factor", None),
            getattr(self, "max_light_steps", None),
        )
        self.render_scale, self.step_factor, self.max_light_steps = new_values
        if new_values != old_values and hasattr(self, "_accum_key"):
            self.reset_accumulation()

    def set_quality_tier(
        self, quality_tier: str, *, camera_moving: Optional[bool] = None
    ) -> None:
        """Apply a named preset; Potato restores High when stationary."""
        quality_tier = str(quality_tier).lower()
        if quality_tier not in QUALITY_PRESETS:
            raise ValueError(
                f"Unknown quality tier {quality_tier!r}; expected one of "
                f"{', '.join(QUALITY_TIER_NAMES)}."
            )
        if camera_moving is not None:
            self._camera_moving = bool(camera_moving)
        self.quality_tier = quality_tier
        self._flight_render_scale = QUALITY_PRESETS[quality_tier].render_scale
        self._apply_effective_quality()

    def set_render_scale(self, render_scale: float) -> None:
        """Override the selected preset's moving render scale."""
        render_target_size((1, 1), render_scale)  # shared validation
        self._flight_render_scale = float(render_scale)
        self._apply_effective_quality()

    def set_camera_moving(self, moving: bool) -> None:
        """Select Potato's flight or converged-still sampling settings."""
        moving = bool(moving)
        if moving == self._camera_moving:
            return
        self._camera_moving = moving
        self._apply_effective_quality()

    # ------------------------------------------------------------------
    # Periodic domain
    # ------------------------------------------------------------------

    def _write_ghost_border(self, periodic: bool) -> None:
        """Rewrite the x/y ghost-border texels of the resident volume.

        Periodic fills them from the opposite faces (exact trilinear wrap);
        non-periodic restores the witness ghost-zero taper. Only the four
        border faces are uploaded — the interior never moves.
        """
        nx, ny, nz = self.field.shape
        faces = self._ghost_faces
        if not periodic:
            faces = {name: np.zeros_like(arr) for name, arr in faces.items()}
        queue = self.device.queue
        row_bytes = (nz + 2) * self._ghost_faces["x_lo"].dtype.itemsize
        queue.write_texture(
            {"texture": self._texture, "origin": (0, 0, 0)},
            np.ascontiguousarray(faces["x_lo"]),
            {"bytes_per_row": row_bytes, "rows_per_image": ny + 2},
            (nz + 2, ny + 2, 1),
        )
        queue.write_texture(
            {"texture": self._texture, "origin": (0, 0, nx + 1)},
            np.ascontiguousarray(faces["x_hi"]),
            {"bytes_per_row": row_bytes, "rows_per_image": ny + 2},
            (nz + 2, ny + 2, 1),
        )
        queue.write_texture(
            {"texture": self._texture, "origin": (0, 0, 0)},
            np.ascontiguousarray(faces["y_lo"]),
            {"bytes_per_row": row_bytes, "rows_per_image": 1},
            (nz + 2, 1, nx + 2),
        )
        queue.write_texture(
            {"texture": self._texture, "origin": (0, ny + 1, 0)},
            np.ascontiguousarray(faces["y_hi"]),
            {"bytes_per_row": row_bytes, "rows_per_image": 1},
            (nz + 2, 1, nx + 2),
        )

    def set_periodic(self, periodic: bool) -> None:
        """Switch horizontal tiling on/off; rewrites only the ghost border.

        The periodic flag is part of the uniform scene-identity key, so the
        temporal accumulation history resets on the next frame by itself.
        """
        periodic = bool(periodic)
        if periodic == self.periodic:
            return
        self._write_ghost_border(periodic)
        self.periodic = periodic

    # ------------------------------------------------------------------
    # Pipeline / uniforms
    # ------------------------------------------------------------------

    def _shader_for(self, periodic: bool, max_light_steps: int):
        key = (bool(periodic), int(max_light_steps))
        if key in self._shader_modules:
            return self._shader_modules[key]
        if not 1 <= key[1] <= DEFAULT_MAX_LIGHT_STEPS:
            raise ValueError(
                "max_light_steps must be between 1 and "
                f"{DEFAULT_MAX_LIGHT_STEPS}; got {key[1]}."
            )
        source = self._shader_source if key[0] else self._legacy_shader_source
        sentinel = "const MAX_LIGHT_STEPS: i32 = 512;"
        if source.count(sentinel) != 1:
            raise RuntimeError(
                "WGSL light-step specialization sentinel is missing or "
                "ambiguous; refusing to build a mismatched tier shader."
            )
        source = source.replace(
            sentinel, f"const MAX_LIGHT_STEPS: i32 = {key[1]};"
        )
        module = self.device.create_shader_module(code=source)
        self._shader_modules[key] = module
        return module

    def pipeline_for(self, target_format: str):
        """Render pipeline for a given color-target format (cached)."""
        key = (target_format, self.periodic, self.max_light_steps)
        if key not in self._pipelines:
            shader = self._shader_for(self.periodic, self.max_light_steps)
            fragment = {
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": target_format}],
            }
            self._pipelines[key] = self.device.create_render_pipeline(
                layout=self._pipeline_layout,
                vertex={"module": shader, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment=fragment,
            )
        return self._pipelines[key]

    def _accum_pipeline_for(self):
        """Running-average pipeline for the fixed accumulation texture format."""
        if self._accum_pipeline is None:
            self._accum_pipeline = self.device.create_render_pipeline(
                layout=self._accum_pipeline_layout,
                vertex={"module": self._accum_shader, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment={
                    "module": self._accum_shader,
                    "entry_point": "fs_main",
                    "targets": [{"format": self._ACCUM_FORMAT}],
                },
            )
        return self._accum_pipeline

    def _present_pipeline_for(self, target_format: str):
        """Present the accumulated scene into the caller's target format."""
        if target_format not in self._present_pipelines:
            self._present_pipelines[target_format] = (
                self.device.create_render_pipeline(
                    layout=self._present_pipeline_layout,
                    vertex={
                        "module": self._present_shader,
                        "entry_point": "vs_main",
                    },
                    primitive={"topology": "triangle-list"},
                    fragment={
                        "module": self._present_shader,
                        "entry_point": "fs_main",
                        "targets": [{"format": target_format}],
                    },
                )
            )
        return self._present_pipelines[target_format]

    def _upscale_pipeline_for(self, target_format: str):
        """Bilinear scaled-present pipeline; High never takes this path."""
        if target_format not in self._upscale_pipelines:
            self._upscale_pipelines[target_format] = (
                self.device.create_render_pipeline(
                    layout=self._upscale_pipeline_layout,
                    vertex={
                        "module": self._upscale_shader,
                        "entry_point": "vs_main",
                    },
                    primitive={"topology": "triangle-list"},
                    fragment={
                        "module": self._upscale_shader,
                        "entry_point": "fs_main",
                        "targets": [{"format": target_format}],
                    },
                )
            )
        return self._upscale_pipelines[target_format]

    def write_uniforms(
        self,
        camera: Camera,
        size: Tuple[int, int],
        *,
        jitter: bool = True,
        sun_azimuth: float = DEFAULT_SUN_AZIMUTH,
        sun_elevation: float = DEFAULT_SUN_ELEVATION,
        exposure: float = DEFAULT_EXPOSURE,
        g_hg: float = DEFAULT_G_HG,
        ambient_strength: float = DEFAULT_AMBIENT_STRENGTH,
        gradient_shading_strength: float = DEFAULT_GRADIENT_SHADING_STRENGTH,
        gradient_coarse_weight: float = DEFAULT_GRADIENT_COARSE_WEIGHT,
        gradient_coarse_radius_m: float = DEFAULT_GRADIENT_COARSE_RADIUS_M,
        deep_shadow_ms_suppression: float = DEFAULT_DEEP_SHADOW_MS_SUPPRESSION,
        ambient_occlusion_strength: float = DEFAULT_AMBIENT_OCCLUSION_STRENGTH,
        ambient_occlusion_floor: float = DEFAULT_AMBIENT_OCCLUSION_FLOOR,
        bounce_depth_attenuation: float = DEFAULT_BOUNCE_DEPTH_ATTENUATION,
        spectral_lighting_strength: float = DEFAULT_SPECTRAL_LIGHTING_STRENGTH,
        low_sun_sky_field_strength: float = DEFAULT_LOW_SUN_SKY_FIELD_STRENGTH,
        light_transfer_split_strength: float = (
            DEFAULT_LIGHT_TRANSFER_SPLIT_STRENGTH
        ),
        aerial_perspective_strength: float = DEFAULT_AERIAL_PERSPECTIVE_STRENGTH,
        ambient_tint: Optional[Tuple[float, float, float]] = None,
        ocean_realism: float = DEFAULT_OCEAN_REALISM,
        ocean_mip_bias: float = DEFAULT_OCEAN_MIP_BIAS,
        ocean_glint_strength: float = DEFAULT_OCEAN_GLINT_STRENGTH,
        ocean_glint_roughness: float = DEFAULT_OCEAN_GLINT_ROUGHNESS,
        ocean_glint_roughness_per_lod: float = (
            DEFAULT_OCEAN_GLINT_ROUGHNESS_PER_LOD
        ),
        ocean_haze_extinction_per_km: float = (
            DEFAULT_OCEAN_HAZE_EXTINCTION_PER_KM
        ),
        ocean_sky_shadow_floor: float = DEFAULT_OCEAN_SKY_SHADOW_FLOOR,
        cone_stencil_theta_deg: float = DEFAULT_CONE_STENCIL_THETA_DEG,
        frame_index: int = 0,
        subpixel: bool = False,
        jitter_scale: float = 1.0,
    ) -> None:
        """Pack the uniform block and enqueue the (tiny) per-frame upload.

        Realism-package strengths (spectral lighting, low-sun sky field,
        light-transfer split, aerial perspective) default to the witness
        tuning-block values; 0.0 selects each mechanism's exact legacy
        arithmetic, matching witness's documented gate semantics.
        ``ambient_tint`` overrides the spectral ambient color (used only to
        reproduce the pre-port frame; see PRE_PORT_AMBIENT_TINT).
        """
        output_w, output_h = (int(size[0]), int(size[1]))
        w, h = render_target_size((output_w, output_h), self.render_scale)
        jitter_scale = _validate_unit_interval("jitter_scale", jitter_scale)
        spectral_lighting_strength = _validate_unit_interval(
            "spectral_lighting_strength", spectral_lighting_strength
        )
        low_sun_sky_field_strength = _validate_unit_interval(
            "low_sun_sky_field_strength", low_sun_sky_field_strength
        )
        aerial_perspective_strength = _validate_finite_float(
            "aerial_perspective_strength", aerial_perspective_strength
        )
        if aerial_perspective_strength < 0.0:
            raise ValueError(
                "aerial_perspective_strength must be >= 0; got "
                f"{aerial_perspective_strength!r}."
            )
        ocean_realism = _validate_unit_interval("ocean_realism", ocean_realism)
        ocean_mip_bias = _validate_finite_float(
            "ocean_mip_bias", ocean_mip_bias
        )
        for name, value in (
            ("ocean_glint_strength", ocean_glint_strength),
            ("ocean_glint_roughness", ocean_glint_roughness),
            ("ocean_glint_roughness_per_lod", ocean_glint_roughness_per_lod),
            ("ocean_haze_extinction_per_km", ocean_haze_extinction_per_km),
        ):
            value = _validate_finite_float(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0; got {value!r}.")
        ocean_sky_shadow_floor = _validate_unit_interval(
            "ocean_sky_shadow_floor", ocean_sky_shadow_floor
        )
        cone_stencil_theta_deg = _validate_finite_float(
            "cone_stencil_theta_deg", cone_stencil_theta_deg
        )
        if not 0.0 <= cone_stencil_theta_deg < 90.0:
            raise ValueError(
                "cone_stencil_theta_deg must be in [0, 90); got "
                f"{cone_stencil_theta_deg!r}."
            )
        cone_stencil_tan_theta = float(
            np.tan(np.deg2rad(cone_stencil_theta_deg))
        )
        if self.periodic and float(sun_elevation) <= 0.0:
            raise ValueError(
                "Periodic domains require the sun above the horizon (the "
                "light march exits only through the domain top); got "
                f"sun_elevation={sun_elevation!r}. Disable periodic "
                "(set_periodic(False)) for below-horizon suns."
            )
        origin = camera_world_origin(camera, self.bmin, self.bmax)
        forward, right, up = camera.basis()
        sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)
        tan_half_fov = np.tan(np.deg2rad(camera.fov) * 0.5)

        # Per-frame CPU precompute, exactly as witness._render_levels does it:
        # spectral beam/fill/sky colors from the sun's air mass, and the
        # elevation-faded light-transfer split strength.
        (cloud_sun, spectral_ambient, sky_horizon,
         sky_bloom, sky_disc) = _spectral_lighting_colors(
            tuple(float(c) for c in sun),
            WITNESS_SUN_COLOR,
            spectral_lighting_strength,
        )
        if ambient_tint is None:
            ambient_tint = spectral_ambient
        else:
            ambient_tint = tuple(
                _validate_finite_float("ambient_tint", c) for c in ambient_tint
            )
            if len(ambient_tint) != 3:
                raise ValueError(
                    f"ambient_tint must have 3 components; got {ambient_tint!r}."
                )
        light_transfer_eff = _effective_light_transfer_split(
            light_transfer_split_strength, float(sun_elevation)
        )

        u = np.zeros((21, 4), dtype=np.float32)
        u[0] = [*origin, tan_half_fov]
        u[1] = [*forward, output_w / output_h]
        u[2] = [*right, exposure]
        u[3] = [*up, 1.0 if jitter else 0.0]
        u[4] = [*sun, float(frame_index)]
        u[5] = [*self.bmin, self.dt_view]
        u[6] = [*self.bmax, self.dt_light]
        u[7] = [w, h, g_hg, ambient_strength]
        u[8] = [self.ocean_z, *self.ocean_reflectance]
        u[9] = [
            self.ocean_fif_dx,
            self.ocean_tile_extent,
            1.0 if self.ocean_enabled else 0.0,
            self.ocean_max_lod,
        ]
        # Row 10: sampling flags only (excluded from scene identity).
        u[10] = [1.0 if subpixel else 0.0, jitter_scale, 0.0, 0.0]
        # Rows 11-12: Cb realism look parameters (scene identity).
        u[11] = [
            gradient_shading_strength,
            deep_shadow_ms_suppression,
            ambient_occlusion_strength,
            bounce_depth_attenuation,
        ]
        u[12] = [
            gradient_coarse_weight,
            gradient_coarse_radius_m,
            ambient_occlusion_floor,
            cone_stencil_tan_theta,
        ]
        # Rows 13-17: witness realism package, per-frame spectral precompute
        # (scene identity). Colors are the _spectral_lighting_colors outputs;
        # at strength 0 (or the 55-degree reference sun) they equal the legacy
        # constants exactly, which is what keeps the WGSL legacy paths exact.
        u[13] = [*cloud_sun, low_sun_sky_field_strength]
        u[14] = [*ambient_tint, light_transfer_eff]
        u[15] = [*sky_horizon, aerial_perspective_strength]
        u[16] = [*sky_bloom, AERIAL_BETA_PER_KM * 1e-3]  # w: beta0 in m^-1
        u[17] = [*sky_disc, AERIAL_SCALE_HEIGHT_M]
        # Rows 18-19: ocean realism (scene identity). Haze extinction is
        # packed in m^-1 like the cloud aerial beta0.
        u[18] = [
            ocean_realism,
            ocean_mip_bias,
            ocean_glint_strength,
            ocean_glint_roughness,
        ]
        u[19] = [
            ocean_glint_roughness_per_lod,
            ocean_haze_extinction_per_km * 1e-3,
            ocean_sky_shadow_floor,
            0.0,
        ]
        # Row 20: periodic domain (scene identity — toggling it must reset
        # the temporal accumulation, and it does via the key below). Driven
        # by renderer state, never per-call: the shader flag must match the
        # ghost-border texel content baked by set_periodic().
        u[20] = [1.0 if self.periodic else 0.0, 0.0, 0.0, 0.0]
        key = u.copy()
        key[4, 3] = 0.0  # frame_index varies jitter seeds, not scene identity
        key[10] = 0.0  # sampling flags are not scene identity
        self._current_uniform_key = key.tobytes() + struct.pack(
            "<III", self.max_light_steps, output_w, output_h
        )
        self._current_uniform_size = (w, h)
        self._current_output_size = (output_w, output_h)
        self._current_uniform = u
        self._current_jitter = bool(jitter)
        self._current_subpixel = bool(subpixel)
        self._current_jitter_scale = jitter_scale
        self.device.queue.write_buffer(self._uniform_buf, 0, u.tobytes())

    def _set_current_subpixel(self, enabled: bool) -> None:
        """Flip only the subpixel sampling flag in the already-packed uniforms."""
        self._set_current_sampling(subpixel=enabled)

    def _set_current_sampling(
        self,
        *,
        subpixel: Optional[bool] = None,
        jitter_scale: Optional[float] = None,
    ) -> None:
        """Flip sampling flags in the already-packed uniforms."""
        if self._current_uniform is None:
            return
        changed = False
        if subpixel is not None:
            subpixel = bool(subpixel)
            if self._current_subpixel != subpixel:
                self._current_uniform[10, 0] = 1.0 if subpixel else 0.0
                self._current_subpixel = subpixel
                changed = True
        if jitter_scale is not None:
            jitter_scale = _validate_unit_interval("jitter_scale", jitter_scale)
            if self._current_jitter_scale != jitter_scale:
                self._current_uniform[10, 1] = jitter_scale
                self._current_jitter_scale = jitter_scale
                changed = True
        if changed:
            self.device.queue.write_buffer(
                self._uniform_buf, 0, self._current_uniform.tobytes()
            )

    def _encode_raymarch_pass(self, command_encoder, target_view,
                              target_format: str, timestamp_writes=None) -> None:
        """Encode the fullscreen raymarch pass into an existing encoder."""
        desc = {
            "color_attachments": [{
                "view": target_view,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.0, 0.0, 0.0, 1.0),
            }],
        }
        if timestamp_writes is not None:
            desc["timestamp_writes"] = timestamp_writes
        rpass = command_encoder.begin_render_pass(**desc)
        rpass.set_pipeline(self.pipeline_for(target_format))
        rpass.set_bind_group(0, self._bind_group)
        rpass.draw(3)
        rpass.end()

    def _scaled_sample_target(self, size: Tuple[int, int]):
        if self._scaled_sample is None or self._scaled_sample["size"] != tuple(size):
            tex = self.device.create_texture(
                label="soar-scaled-sample",
                size=(size[0], size[1], 1),
                format=self._ACCUM_FORMAT,
                usage=(
                    wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.TEXTURE_BINDING
                ),
            )
            self._scaled_sample = {
                "size": tuple(size),
                "texture": tex,
                "view": tex.create_view(),
            }
        return self._scaled_sample

    def reset_accumulation(self) -> None:
        """Drop the temporal history; the next jittered frame seeds it."""
        self._accum_key = None
        self._accum_count = 0
        self._accum_index = 0
        self._accum_motion = False
        self._accum_last_origin = None
        self._accum_last_forward = None
        self._last_motion_delta = None
        self._last_motion_reset = False

    def encode_present_last(self, command_encoder, target_view,
                            target_format: str) -> bool:
        """Re-present the last accumulated frame without marching the volume.

        Used while a behold render owns the GPU: the app keeps its window
        alive by blitting the frozen frame under the progress overlay at
        near-zero cost. Returns False when no accumulated frame exists
        (e.g. jitter off, or nothing rendered yet) — the caller must then
        encode a normal pass.
        """
        if (
            self._accum_targets is None
            or self._accum_count < 1
            or self._current_uniform_size is None
            or self._accum_targets["size"] != tuple(self._current_uniform_size)
        ):
            return False
        self._encode_present_pass(
            command_encoder,
            self._accum_targets["accum_views"][self._accum_index],
            target_view,
            target_format,
            upscale=(self._current_uniform_size != self._current_output_size),
        )
        return True

    def _accum_target(self, size: Tuple[int, int]):
        if self._accum_targets is None or self._accum_targets["size"] != tuple(size):
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
            sample = self.device.create_texture(
                label="soar-temporal-sample",
                size=(size[0], size[1], 1),
                format=self._ACCUM_FORMAT,
                usage=usage,
            )
            accum = [
                self.device.create_texture(
                    label=f"soar-temporal-accum-{i}",
                    size=(size[0], size[1], 1),
                    format=self._ACCUM_FORMAT,
                    usage=usage,
                )
                for i in range(2)
            ]
            self._accum_targets = {
                "size": tuple(size),
                "sample": sample,
                "sample_view": sample.create_view(),
                "accum": accum,
                "accum_views": [tex.create_view() for tex in accum],
            }
            self.reset_accumulation()
        return self._accum_targets

    def _encode_accum_blend_pass(
        self,
        command_encoder,
        target,
        *,
        prev_weight: float,
        sample_weight: float,
        next_count: int,
    ) -> None:
        weights = np.array(
            [prev_weight, sample_weight, 0.0, 0.0], dtype=np.float32
        )
        self.device.queue.write_buffer(
            self._accum_uniform_buf, 0, weights.tobytes()
        )

        prev_index = self._accum_index
        dst_index = 1 - self._accum_index
        bind_group = self.device.create_bind_group(
            layout=self._accum_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._accum_uniform_buf,
                        "offset": 0,
                        "size": _ACCUM_UNIFORM_NBYTES,
                    },
                },
                {"binding": 1, "resource": target["sample_view"]},
                {"binding": 2, "resource": target["accum_views"][prev_index]},
            ],
        )
        rpass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": target["accum_views"][dst_index],
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.0, 0.0, 0.0, 1.0),
            }]
        )
        rpass.set_pipeline(self._accum_pipeline_for())
        rpass.set_bind_group(0, bind_group)
        rpass.draw(3)
        rpass.end()
        self._accum_index = dst_index
        self._accum_count = int(next_count)

    def _encode_present_pass(
        self,
        command_encoder,
        src_view,
        target_view,
        target_format: str,
        *,
        upscale: bool = False,
    ) -> None:
        if upscale:
            layout = self._upscale_bind_group_layout
            entries = [
                {"binding": 0, "resource": src_view},
                {"binding": 1, "resource": self._upscale_sampler},
            ]
            pipeline = self._upscale_pipeline_for(target_format)
        else:
            layout = self._present_bind_group_layout
            entries = [{"binding": 0, "resource": src_view}]
            pipeline = self._present_pipeline_for(target_format)
        bind_group = self.device.create_bind_group(
            layout=layout,
            entries=entries,
        )
        rpass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": target_view,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.0, 0.0, 0.0, 1.0),
            }]
        )
        rpass.set_pipeline(pipeline)
        rpass.set_bind_group(0, bind_group)
        rpass.draw(3)
        rpass.end()

    def _current_camera_motion_basis(self):
        if self._current_uniform is None:
            return None, None
        origin = np.asarray(self._current_uniform[0, :3], dtype=np.float64)
        forward = np.asarray(self._current_uniform[1, :3], dtype=np.float64)
        return origin, forward

    def _motion_delta_exceeds_reset(
        self,
        origin: np.ndarray,
        forward: np.ndarray,
        *,
        translation_threshold_m: float,
        angle_threshold_degrees: float,
    ) -> bool:
        if self._accum_last_origin is None or self._accum_last_forward is None:
            self._last_motion_delta = None
            return True
        translation = float(np.linalg.norm(origin - self._accum_last_origin))
        cos_angle = float(
            np.clip(np.dot(forward, self._accum_last_forward), -1.0, 1.0)
        )
        angle = float(np.degrees(np.arccos(cos_angle)))
        self._last_motion_delta = {
            "translation_m": translation,
            "angle_degrees": angle,
        }
        return (
            translation > translation_threshold_m
            or angle > angle_threshold_degrees
        )

    def _motion_alpha_for_dt(
        self,
        alpha_per_reference_frame: float,
        *,
        reference_fps: float,
        delta_seconds: Optional[float],
    ) -> float:
        alpha = _validate_unit_interval(
            "motion_blend_alpha", alpha_per_reference_frame
        )
        reference_fps = _validate_positive_float(
            "motion_blend_reference_fps", reference_fps
        )
        if delta_seconds is None:
            return alpha
        delta_seconds = _validate_finite_float("motion_delta_seconds", delta_seconds)
        if delta_seconds <= 0.0 or alpha in (0.0, 1.0):
            return alpha
        equivalent_frames = delta_seconds * reference_fps
        return float(1.0 - (1.0 - alpha) ** equivalent_frames)

    def encode_pass(self, command_encoder, target_view, target_format: str,
                    timestamp_writes=None, accumulate: Optional[bool] = None,
                    *, motion_accumulation: Optional[bool] = None,
                    motion_blend_alpha: Optional[float] = None,
                    motion_blend_reference_fps: Optional[float] = None,
                    motion_jitter_scale: Optional[float] = None,
                    motion_reset_angle_degrees: Optional[float] = None,
                    motion_reset_translation_m: Optional[float] = None,
                    motion_delta_seconds: Optional[float] = None) -> None:
        """Encode the scene pass, with temporal averaging for jittered views.

        Static frames use a true running average. Small camera deltas use a
        no-reprojection exponential blend at ``motion_blend_alpha`` new-frame
        weight, specified per frame at ``motion_blend_reference_fps`` unless
        ``motion_delta_seconds`` is supplied.
        """
        if accumulate is None:
            accumulate = True
        if (
            not accumulate
            or not self._current_jitter
            or self._current_uniform_key is None
            or self._current_uniform_size is None
        ):
            self._set_current_subpixel(False)
            if not accumulate or not self._current_jitter:
                self.reset_accumulation()
            scaled = self._current_uniform_size != self._current_output_size
            if scaled:
                sample = self._scaled_sample_target(self._current_uniform_size)
                self._encode_raymarch_pass(
                    command_encoder,
                    sample["view"],
                    self._ACCUM_FORMAT,
                    timestamp_writes,
                )
                self._encode_present_pass(
                    command_encoder,
                    sample["view"],
                    target_view,
                    target_format,
                    upscale=True,
                )
            else:
                self._encode_raymarch_pass(
                    command_encoder, target_view, target_format, timestamp_writes
                )
            return

        target = self._accum_target(self._current_uniform_size)
        origin, forward = self._current_camera_motion_basis()
        current_key = self._current_uniform_key

        use_motion = (
            self.motion_accumulation
            if motion_accumulation is None
            else bool(motion_accumulation)
        )
        alpha_ref = (
            self.motion_blend_alpha
            if motion_blend_alpha is None
            else _validate_unit_interval("motion_blend_alpha", motion_blend_alpha)
        )
        reference_fps = (
            self.motion_blend_reference_fps
            if motion_blend_reference_fps is None
            else _validate_positive_float(
                "motion_blend_reference_fps", motion_blend_reference_fps
            )
        )
        jitter_scale_motion = (
            self.motion_jitter_scale
            if motion_jitter_scale is None
            else _validate_unit_interval("motion_jitter_scale", motion_jitter_scale)
        )
        reset_angle = (
            self.motion_reset_angle_degrees
            if motion_reset_angle_degrees is None
            else _validate_positive_float(
                "motion_reset_angle_degrees", motion_reset_angle_degrees
            )
        )
        reset_translation = (
            self.motion_reset_translation_m
            if motion_reset_translation_m is None
            else _validate_positive_float(
                "motion_reset_translation_m", motion_reset_translation_m
            )
        )

        prev_count = self._accum_count
        next_count = prev_count + 1
        if prev_count == 0:
            prev_weight = 0.0
            sample_weight = 1.0
        else:
            prev_weight = prev_count / next_count
            sample_weight = 1.0 / next_count
        subpixel = prev_count >= 1
        jitter_scale = 1.0
        self._last_motion_reset = False

        if self._accum_key is None:
            self._accum_key = current_key
            self._accum_motion = False
            subpixel = False
        elif self._accum_key == current_key:
            if self._accum_motion:
                # The motion buffer is deliberately smeared. Once the camera
                # stops, discard it so static accumulation is again a true
                # running average for the fixed view.
                self._accum_count = 0
                self._accum_index = 0
                self._accum_motion = False
                prev_weight = 0.0
                sample_weight = 1.0
                next_count = 1
                subpixel = False
            else:
                self._accum_motion = False
        else:
            alpha = self._motion_alpha_for_dt(
                alpha_ref,
                reference_fps=reference_fps,
                delta_seconds=motion_delta_seconds,
            )
            reset_for_jump = self._motion_delta_exceeds_reset(
                origin,
                forward,
                translation_threshold_m=reset_translation,
                angle_threshold_degrees=reset_angle,
            )
            can_blend_motion = use_motion and alpha < 1.0 and not reset_for_jump
            self._accum_key = current_key
            if can_blend_motion:
                prev_weight = 1.0 - alpha
                sample_weight = alpha
                next_count = 1
                subpixel = True
                jitter_scale = jitter_scale_motion
                self._accum_motion = True
            else:
                self._accum_count = 0
                self._accum_index = 0
                prev_weight = 0.0
                sample_weight = 1.0
                next_count = 1
                subpixel = False
                self._accum_motion = False
                self._last_motion_reset = True

        self._set_current_sampling(subpixel=subpixel, jitter_scale=jitter_scale)
        self._encode_raymarch_pass(
            command_encoder,
            target["sample_view"],
            self._ACCUM_FORMAT,
            timestamp_writes,
        )
        self._encode_accum_blend_pass(
            command_encoder,
            target,
            prev_weight=prev_weight,
            sample_weight=sample_weight,
            next_count=next_count,
        )
        self._encode_present_pass(
            command_encoder,
            target["accum_views"][self._accum_index],
            target_view,
            target_format,
            upscale=(self._current_uniform_size != self._current_output_size),
        )
        self._accum_last_origin = origin
        self._accum_last_forward = forward

    # ------------------------------------------------------------------
    # Offscreen rendering
    # ------------------------------------------------------------------

    _OFFSCREEN_FORMAT = wgpu.TextureFormat.rgba8unorm

    def _offscreen_target(self, size: Tuple[int, int]):
        if self._offscreen is None or self._offscreen["size"] != tuple(size):
            tex = self.device.create_texture(
                size=(size[0], size[1], 1),
                format=self._OFFSCREEN_FORMAT,
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.COPY_SRC),
            )
            self._offscreen = {"size": tuple(size), "texture": tex,
                               "view": tex.create_view()}
        return self._offscreen

    def render(self, camera: Optional[Camera] = None,
               size: Tuple[int, int] = (960, 540), *,
               bird: bool = False, bird_time: float = 0.0,
               bird_pose: Optional[dict] = None, hud: bool = False,
               accumulate_frames: int = 1,
               accumulate: Optional[bool] = None,
               motion_accumulation: Optional[bool] = None,
               motion_blend_alpha: Optional[float] = None,
               motion_blend_reference_fps: Optional[float] = None,
               motion_jitter_scale: Optional[float] = None,
               motion_reset_angle_degrees: Optional[float] = None,
               motion_reset_translation_m: Optional[float] = None,
               motion_delta_seconds: Optional[float] = None,
               **kwargs) -> np.ndarray:
        """Render one frame offscreen and read it back.

        Parameters
        ----------
        bird : bool
            Draw the flying subject (default off: existing renders, tests
            and benchmarks are bird-free). Offscreen the bird cruises in a
            deterministic pose; `bird_time` (s) sets the wingbeat phase and
            `bird_pose` may override {"bank", "pitch"} (deg) and
            {"flap_phase"} (rad) — see :meth:`bird.Bird.set_static`.
        hud : bool
            Draw the minimap overlay (default off so parity renders and
            benchmarks remain HUD-free unless explicitly requested).
        accumulate_frames : int
            Number of static jittered frames to accumulate before readback.
            The frame index is advanced for each internal frame so jitter
            samples decorrelate. Single-frame and jitter-off renders use the
            direct path. Overlays (bird, hud) are drawn only on the final
            frame so they stay crisp over the converged volume.
        accumulate : bool, optional
            Force use of the temporal path even for a single frame. This is
            how offscreen motion-sequence tests/evaluation exercise the same
            accumulation path that the windowed app uses.

        Returns
        -------
        ndarray (height, width, 3), uint8
            Tone-mapped RGB (tone map + gamma run in-shader).
        """
        if camera is None:
            camera = Camera()
        accumulate_frames = int(accumulate_frames)
        if accumulate_frames < 1:
            raise ValueError(
                f"accumulate_frames must be >= 1; got {accumulate_frames}."
            )
        frame_index0 = int(kwargs.pop("frame_index", 0))
        target = self._offscreen_target(size)
        use_accumulation = (
            accumulate_frames > 1 if accumulate is None else bool(accumulate)
        )
        for i in range(accumulate_frames):
            self.write_uniforms(
                camera, size, frame_index=frame_index0 + i, **kwargs
            )
            enc = self.device.create_command_encoder()
            self.encode_pass(
                enc, target["view"], self._OFFSCREEN_FORMAT,
                accumulate=use_accumulation,
                motion_accumulation=motion_accumulation,
                motion_blend_alpha=motion_blend_alpha,
                motion_blend_reference_fps=motion_blend_reference_fps,
                motion_jitter_scale=motion_jitter_scale,
                motion_reset_angle_degrees=motion_reset_angle_degrees,
                motion_reset_translation_m=motion_reset_translation_m,
                motion_delta_seconds=motion_delta_seconds,
            )
            if i == accumulate_frames - 1:
                if bird:
                    origin = camera_world_origin(camera, self.bmin, self.bmax)
                    self.bird.set_static(origin, camera, bird_time,
                                         **(bird_pose or {}))
                    self.bird.write_uniforms(
                        origin, camera, size,
                        sun_azimuth=kwargs.get("sun_azimuth",
                                               DEFAULT_SUN_AZIMUTH),
                        sun_elevation=kwargs.get("sun_elevation",
                                                 DEFAULT_SUN_ELEVATION),
                        exposure=kwargs.get("exposure", DEFAULT_EXPOSURE),
                        ambient_strength=kwargs.get("ambient_strength",
                                                    DEFAULT_AMBIENT_STRENGTH),
                    )
                    self.bird.encode_pass(enc, target["view"],
                                          self._OFFSCREEN_FORMAT, size)
                if hud:
                    self.hud.write_uniforms(camera, size)
                    self.hud.encode_pass(enc, target["view"],
                                         self._OFFSCREEN_FORMAT)
            self.device.queue.submit([enc.finish()])
        data = self.device.queue.read_texture(
            {"texture": target["texture"]},
            {"bytes_per_row": size[0] * 4, "rows_per_image": size[1]},
            (size[0], size[1], 1),
        )
        img = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 4)
        return img[:, :, :3].copy()

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(self, camera: Optional[Camera] = None,
                  size: Tuple[int, int] = (960, 540), *,
                  n_warmup: int = 5, n_frames: int = 30,
                  azimuth_step: float = 0.4, bird: bool = False,
                  hud: bool = False,
                  **kwargs) -> dict:
        """Steady-state per-frame timing.

        The camera azimuth is nudged `azimuth_step` deg/frame (same protocol
        as temp/benchmarks-2026-07-07) so no frame is trivially cached.
        GPU time comes from timestamp queries when the device has
        'timestamp-query'; wall time is measured around submit+sync always.
        With `bird=True` and/or `hud=True`, the extra overlay pass(es) are
        encoded too and the GPU interval spans the full frame.

        Returns a dict with per-frame arrays and summary stats (ms).
        """
        if camera is None:
            camera = Camera()
        target = self._offscreen_target(size)

        has_ts = "timestamp-query" in self.device.features
        if has_ts:
            query_set = self.device.create_query_set(type="timestamp", count=2)
            resolve_buf = self.device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
            )

        def frame(i: int, timed: bool):
            cam = Camera(position=camera.position,
                         azimuth=camera.azimuth + azimuth_step * i,
                         elevation=camera.elevation, fov=camera.fov)
            self.write_uniforms(cam, size, frame_index=i, **kwargs)
            if bird:
                origin = camera_world_origin(cam, self.bmin, self.bmax)
                self.bird.set_static(origin, cam, t=i / 60.0)
                self.bird.write_uniforms(
                    origin, cam, size,
                    sun_azimuth=kwargs.get("sun_azimuth",
                                           DEFAULT_SUN_AZIMUTH),
                    sun_elevation=kwargs.get("sun_elevation",
                                             DEFAULT_SUN_ELEVATION),
                    exposure=kwargs.get("exposure", DEFAULT_EXPOSURE),
                    ambient_strength=kwargs.get("ambient_strength",
                                                DEFAULT_AMBIENT_STRENGTH),
                )
            if hud:
                self.hud.write_uniforms(cam, size)
            enc = self.device.create_command_encoder()
            ts = None
            if timed and has_ts:
                ts = {"query_set": query_set,
                      "beginning_of_pass_write_index": 0,
                      "end_of_pass_write_index": 1}
            if bird or hud:
                # Timestamp interval spans volume + overlays: begin on the
                # volume pass, end on the last overlay pass.
                ts_begin, ts_end = None, None
                if ts is not None:
                    ts_begin = {"query_set": query_set,
                                "beginning_of_pass_write_index": 0}
                    ts_end = {"query_set": query_set,
                              "end_of_pass_write_index": 1}
                self.encode_pass(enc, target["view"], self._OFFSCREEN_FORMAT,
                                 ts_begin)
                if bird:
                    self.bird.encode_pass(enc, target["view"],
                                          self._OFFSCREEN_FORMAT, size,
                                          ts_end if not hud else None)
                if hud:
                    self.hud.encode_pass(enc, target["view"],
                                         self._OFFSCREEN_FORMAT, ts_end)
            else:
                self.encode_pass(enc, target["view"], self._OFFSCREEN_FORMAT,
                                 ts)
            if timed and has_ts:
                enc.resolve_query_set(query_set, 0, 2, resolve_buf, 0)
            t0 = perf_counter()
            self.device.queue.submit([enc.finish()])
            gpu_ms = None
            if timed and has_ts:
                stamps = np.frombuffer(
                    self.device.queue.read_buffer(resolve_buf), dtype=np.uint64
                )
                gpu_ms = float(stamps[1] - stamps[0]) / 1e6
            else:
                # read_buffer above already syncs; otherwise force one.
                self.device.queue.read_texture(
                    {"texture": target["texture"]},
                    {"bytes_per_row": size[0] * 4, "rows_per_image": size[1]},
                    (size[0], size[1], 1),
                )
            wall_ms = (perf_counter() - t0) * 1e3
            return gpu_ms, wall_ms

        for i in range(n_warmup):
            frame(i, timed=False)

        gpu_times, wall_times = [], []
        for i in range(n_frames):
            gpu_ms, wall_ms = frame(n_warmup + i, timed=True)
            wall_times.append(wall_ms)
            if gpu_ms is not None:
                gpu_times.append(gpu_ms)

        wall = np.array(wall_times)
        result = {
            "size": tuple(size),
            "n_frames": n_frames,
            "wall_ms_mean": float(wall.mean()),
            "wall_ms_std": float(wall.std()),
            "wall_ms_min": float(wall.min()),
            "wall_ms_max": float(wall.max()),
            "timestamps_used": has_ts,
        }
        if gpu_times:
            gpu = np.array(gpu_times)
            result.update({
                "gpu_ms_mean": float(gpu.mean()),
                "gpu_ms_std": float(gpu.std()),
                "gpu_ms_min": float(gpu.min()),
                "gpu_ms_max": float(gpu.max()),
            })
        return result


def request_device():
    """Request a high-performance adapter/device with the features we need.

    Raises RuntimeError (not a fallback) when float32-filterable is missing;
    the fp32 ocean normal texture requires it even with an fp16 volume.
    'timestamp-query' is added opportunistically for benchmarking.
    """
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    features = ["float32-filterable"]
    if "float32-filterable" not in adapter.features:
        raise RuntimeError(
            "GPU adapter does not support 'float32-filterable' (required for "
            "hardware trilinear sampling of the renderer's float32 textures)."
        )
    if "timestamp-query" in adapter.features:
        features.append("timestamp-query")
    return adapter.request_device_sync(required_features=features)
