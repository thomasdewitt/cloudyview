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
STEP_VOXEL_FACTOR = 2.0  # dt = min voxel dimension * this (witness value)

_UNIFORM_NBYTES = 8 * 16  # 8 vec4<f32>


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

    def __init__(
        self,
        field: CloudField,
        *,
        extinction_multiplier: float = 1.0,
        device=None,
    ):
        self.field = field
        self.bmin, self.bmax = _volume_aabb(field)

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

        # Zero-reshuffle upload: a C-order (nx, ny, nz) array already has z
        # fastest, so it maps directly onto a texture with width=nz,
        # height=ny, depth=nx. The shader swizzles sample coords to match.
        # TODO(fp16): optional float16 texture to halve resident memory.
        max_dim = self.device.limits["max-texture-dimension-3d"]
        if max(nx, ny, nz) > max_dim:
            raise ValueError(
                f"Volume {nx}x{ny}x{nz} exceeds the device's 3D texture "
                f"limit ({max_dim}); bricking/LOD is out of scope for the "
                "spike (docs/architecture.md)."
            )
        self._texture = self.device.create_texture(
            label="cloud-sigma",
            size=(nz, ny, nx),
            format=wgpu.TextureFormat.r32float,
            dimension="3d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": self._texture},
            sigma,
            {"bytes_per_row": nz * 4, "rows_per_image": ny},
            (nz, ny, nx),
        )
        self.volume_nbytes = sigma.nbytes

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
        self._uniform_buf = self.device.create_buffer(
            size=_UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._shader = self.device.create_shader_module(
            code=SHADER_PATH.read_text()
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
        ])
        self._bind_group = self.device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._uniform_buf,
                                            "offset": 0,
                                            "size": _UNIFORM_NBYTES}},
                {"binding": 1, "resource": self._texture.create_view()},
                {"binding": 2, "resource": self._sampler},
            ],
        )
        self._pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._pipelines = {}  # target format -> render pipeline

        # Offscreen target cache: (w, h) -> texture.
        self._offscreen = None

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
        frame_index: int = 0,
    ) -> None:
        """Pack the uniform block and enqueue the (tiny) per-frame upload."""
        w, h = size
        origin = camera_world_origin(camera, self.bmin, self.bmax)
        forward, right, up = camera.basis()
        sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)
        tan_half_fov = np.tan(np.deg2rad(camera.fov) * 0.5)

        u = np.empty((8, 4), dtype=np.float32)
        u[0] = [*origin, tan_half_fov]
        u[1] = [*forward, w / h]
        u[2] = [*right, exposure]
        u[3] = [*up, 1.0 if jitter else 0.0]
        u[4] = [*sun, float(frame_index)]
        u[5] = [*self.bmin, self.dt_view]
        u[6] = [*self.bmax, self.dt_light]
        u[7] = [w, h, g_hg, ambient_strength]
        self.device.queue.write_buffer(self._uniform_buf, 0, u.tobytes())

    def encode_pass(self, command_encoder, target_view, target_format: str,
                    timestamp_writes=None) -> None:
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
               size: Tuple[int, int] = (960, 540), **kwargs) -> np.ndarray:
        """Render one frame offscreen and read it back.

        Returns
        -------
        ndarray (height, width, 3), uint8
            Tone-mapped RGB (tone map + gamma run in-shader).
        """
        if camera is None:
            camera = Camera()
        target = self._offscreen_target(size)
        self.write_uniforms(camera, size, **kwargs)
        enc = self.device.create_command_encoder()
        self.encode_pass(enc, target["view"], self._OFFSCREEN_FORMAT)
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
                  azimuth_step: float = 0.4, **kwargs) -> dict:
        """Steady-state per-frame timing.

        The camera azimuth is nudged `azimuth_step` deg/frame (same protocol
        as temp/benchmarks-2026-07-07) so no frame is trivially cached.
        GPU time comes from timestamp queries when the device has
        'timestamp-query'; wall time is measured around submit+sync always.

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
            enc = self.device.create_command_encoder()
            ts = None
            if timed and has_ts:
                ts = {"query_set": query_set,
                      "beginning_of_pass_write_index": 0,
                      "end_of_pass_write_index": 1}
            self.encode_pass(enc, target["view"], self._OFFSCREEN_FORMAT, ts)
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
