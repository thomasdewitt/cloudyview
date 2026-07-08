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

from pathlib import Path
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

SHADER_PATH = Path(__file__).parent / "raymarch.wgsl"

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
STEP_VOXEL_FACTOR = 2.0  # dt = min voxel dimension * this (witness value)
DEFAULT_MOTION_BLEND_ALPHA = 0.45
DEFAULT_MOTION_BLEND_REFERENCE_FPS = 60.0
DEFAULT_MOTION_JITTER_SCALE = 0.65
DEFAULT_MOTION_RESET_ANGLE_DEGREES = 8.0
DEFAULT_MOTION_RESET_TRANSLATION_FRACTION = 0.05

_UNIFORM_NBYTES = 13 * 16  # 13 vec4<f32>
_ACCUM_UNIFORM_NBYTES = 16  # 4 f32s
_DEFAULT_FIF_NORMALS = None

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
    device : wgpu.GPUDevice, optional
        Reuse an existing device (the windowed app shares one with its
        canvas). Must have the ``float32-filterable`` feature.
    """

    _ACCUM_FORMAT = wgpu.TextureFormat.rgba16float

    def __init__(
        self,
        field: CloudField,
        *,
        extinction_multiplier: float = 1.0,
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
        device=None,
    ):
        self.field = field
        self.bmin, self.bmax = _volume_aabb(field)
        self.ocean_enabled = bool(ocean_enabled)
        self.ocean_z = float(ocean_z)
        self.ocean_reflectance = tuple(float(c) for c in ocean_reflectance)
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
        self.dt_view = float(voxel.min()) * STEP_VOXEL_FACTOR
        self.dt_light = self.dt_view  # witness shadow march uses the same factor

        if device is None:
            device = request_device()
        self.device = device
        if "float32-filterable" not in device.features:
            raise RuntimeError(
                "wgpu device lacks the 'float32-filterable' feature required "
                "for hardware trilinear sampling of the r32float density "
                "texture. Refusing to fall back to nearest-neighbor."
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
        sigma = np.ascontiguousarray(sigma, dtype=np.float32)

        # Bake witness's ghost-zero boundary into the resident texture. The
        # public AABB remains the unpadded level extent, where witness samples
        # with gx = (p - bmin) / dx and dx = (bmax - bmin) / N. Padding shifts
        # original voxel i to padded texel i+1; the shader maps
        #     texel = gx + 1, texcoord = (texel + 0.5) / (N + 2),
        # so p=bmin+i*dx still lands exactly on original sigma[i], while
        # gx in [-1,0) and [N-1,N) filters against the zero ghost texels.
        sigma_padded = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float32)
        sigma_padded[1:-1, 1:-1, 1:-1] = sigma

        # Zero-reshuffle upload: a C-order (nx, ny, nz) array already has z
        # fastest, so it maps directly onto a texture with width=nz,
        # height=ny, depth=nx. The shader swizzles sample coords to match.
        # TODO(fp16): optional float16 texture to halve resident memory.
        max_dim = self.device.limits["max-texture-dimension-3d"]
        if max(sigma_padded.shape) > max_dim:
            raise ValueError(
                f"Padded volume {nx + 2}x{ny + 2}x{nz + 2} exceeds the "
                f"device's 3D texture "
                f"limit ({max_dim}); bricking/LOD is out of scope for the "
                "spike (docs/architecture.md)."
            )
        self._texture = self.device.create_texture(
            label="cloud-sigma",
            size=(nz + 2, ny + 2, nx + 2),
            format=wgpu.TextureFormat.r32float,
            dimension="3d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": self._texture},
            sigma_padded,
            {"bytes_per_row": (nz + 2) * 4, "rows_per_image": ny + 2},
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
        self._shader = self.device.create_shader_module(
            code=SHADER_PATH.read_text()
        )
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
        self._pipelines = {}  # target format -> render pipeline
        self._accum_pipeline = None
        self._present_pipelines = {}  # target format -> render pipeline

        # Offscreen target cache: (w, h) -> texture.
        self._offscreen = None
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
    # Pipeline / uniforms
    # ------------------------------------------------------------------

    def pipeline_for(self, target_format: str):
        """Render pipeline for a given color-target format (cached)."""
        if target_format not in self._pipelines:
            self._pipelines[target_format] = self.device.create_render_pipeline(
                layout=self._pipeline_layout,
                vertex={"module": self._shader, "entry_point": "vs_main"},
                primitive={"topology": "triangle-list"},
                fragment={
                    "module": self._shader,
                    "entry_point": "fs_main",
                    "targets": [{"format": target_format}],
                },
            )
        return self._pipelines[target_format]

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
        frame_index: int = 0,
        subpixel: bool = False,
        jitter_scale: float = 1.0,
    ) -> None:
        """Pack the uniform block and enqueue the (tiny) per-frame upload."""
        w, h = size
        jitter_scale = _validate_unit_interval("jitter_scale", jitter_scale)
        origin = camera_world_origin(camera, self.bmin, self.bmax)
        forward, right, up = camera.basis()
        sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)
        tan_half_fov = np.tan(np.deg2rad(camera.fov) * 0.5)

        u = np.zeros((13, 4), dtype=np.float32)
        u[0] = [*origin, tan_half_fov]
        u[1] = [*forward, w / h]
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
            0.0,
        ]
        key = u.copy()
        key[4, 3] = 0.0  # frame_index varies jitter seeds, not scene identity
        key[10] = 0.0  # sampling flags are not scene identity
        self._current_uniform_key = key.tobytes()
        self._current_uniform_size = tuple(size)
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

    def _encode_present_pass(self, command_encoder, src_view, target_view,
                             target_format: str) -> None:
        bind_group = self.device.create_bind_group(
            layout=self._present_bind_group_layout,
            entries=[{"binding": 0, "resource": src_view}],
        )
        rpass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": target_view,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.0, 0.0, 0.0, 1.0),
            }]
        )
        rpass.set_pipeline(self._present_pipeline_for(target_format))
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

    Raises RuntimeError (not a fallback) when float32-filterable is missing.
    'timestamp-query' is added opportunistically for benchmarking.
    """
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    features = ["float32-filterable"]
    if "float32-filterable" not in adapter.features:
        raise RuntimeError(
            "GPU adapter does not support 'float32-filterable' (required for "
            "hardware trilinear sampling of the r32float density texture)."
        )
    if "timestamp-query" in adapter.features:
        features.append("timestamp-query")
    return adapter.request_device_sync(required_features=features)
