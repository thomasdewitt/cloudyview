"""Heads-up-display minimap overlay for the soar fly-through.

The map image is static: a ``cv.glimpse(field)`` two-stream albedo map,
colorized with the same sky-blue -> white convention as ``basic_render`` and
uploaded once as a resident 2D texture. Per frame, only a small uniform block
is updated with the minimap rectangle and camera/FOV overlay geometry.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import wgpu

from ..angles import azimuth_met_to_internal_deg, direction_from_azimuth_elevation
from ..camera import Camera
from ..glimpse import glimpse

SHADER_PATH = Path(__file__).parent / "hud.wgsl"

_UNIFORM_NBYTES = 6 * 16  # 6 vec4<f32>
_SKY_BLUE = np.array([0x3A, 0x4A, 0xA6], dtype=np.float32) / 255.0
_WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)

MAP_HEIGHT_FRAC = 0.22
MAP_MAX_WIDTH_FRAC = 0.34
MAP_MARGIN_FRAC = 0.025
MAP_OPACITY = 0.74


def colorize_glimpse_albedo(albedo: np.ndarray) -> np.ndarray:
    """Return an RGBA8 sky-blue -> white visualization of glimpse albedo."""
    a = np.clip(np.asarray(albedo, dtype=np.float32), 0.0, 1.0)
    rgb = _SKY_BLUE + (_WHITE - _SKY_BLUE) * a[..., None]
    rgba = np.empty((*a.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.round(rgb * 255.0).astype(np.uint8)
    rgba[..., 3] = 255
    return np.ascontiguousarray(rgba)


def _unit_xy(direction_3d: np.ndarray, fallback_angle_rad: float) -> np.ndarray:
    """Project a 3D direction to unit top-down XY, with an azimuth fallback."""
    xy = direction_3d[:2]
    norm_xy = np.linalg.norm(xy)
    if norm_xy < 1e-10:
        return np.array([
            np.cos(fallback_angle_rad),
            np.sin(fallback_angle_rad),
        ])
    return xy / norm_xy


def _camera_overlay_geometry(camera: Camera, image_shape: Tuple[int, int],
                             render_aspect: float) -> dict:
    """Port of glimpse._build_camera_overlay for HUD uniform generation.

    Coordinates returned here use map UV space: x/east and y/north both map
    from 0..1 across the loaded field. The shader flips y when converting to
    screen pixels because framebuffer coordinates are top-down.
    """
    ny, nx = image_shape
    nxm1 = max(nx - 1, 1)
    nym1 = max(ny - 1, 1)
    cam_x = ((camera.position[0] + 1.0) * 0.5) * nxm1
    cam_y = ((camera.position[1] + 1.0) * 0.5) * nym1
    cam_uv = (cam_x / nxm1, cam_y / nym1)

    half_vfov_deg = 0.5 * float(camera.fov)
    includes_zenith = (90.0 - float(camera.elevation)) <= half_vfov_deg
    includes_nadir = (90.0 + float(camera.elevation)) <= half_vfov_deg

    if includes_zenith or includes_nadir:
        return {
            "camera_uv": cam_uv,
            "circle_radius_px": nx / 10.0,
        }

    az_internal_rad = np.deg2rad(azimuth_met_to_internal_deg(camera.azimuth))
    half_vfov = np.deg2rad(camera.fov * 0.5)
    half_hfov = np.arctan(np.tan(half_vfov) * render_aspect)

    forward = direction_from_azimuth_elevation(
        camera.azimuth, camera.elevation
    )
    # Analytic horizontal right vector (see Camera.basis): continuous
    # through straight up/down, no up-reference flip.
    az_rad = np.deg2rad(camera.azimuth)
    right = np.array([np.cos(az_rad), -np.sin(az_rad), 0.0])

    left_dir = forward - np.tan(half_hfov) * right
    right_dir = forward + np.tan(half_hfov) * right
    left_dir /= np.linalg.norm(left_dir)
    right_dir /= np.linalg.norm(right_dir)

    left_xy = _unit_xy(left_dir, az_internal_rad - half_hfov)
    right_xy = _unit_xy(right_dir, az_internal_rad + half_hfov)

    ray_length = 1.5 * max(nx, ny)
    left_end = (
        (cam_x + ray_length * left_xy[0]) / nxm1,
        (cam_y + ray_length * left_xy[1]) / nym1,
    )
    right_end = (
        (cam_x + ray_length * right_xy[0]) / nxm1,
        (cam_y + ray_length * right_xy[1]) / nym1,
    )

    return {
        "camera_uv": cam_uv,
        "fov_endpoints": [left_end, right_end],
    }


class MinimapHUD:
    """GPU resources for the static map texture and live camera overlay."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.device = renderer.device

        albedo = glimpse(renderer.field)
        self.albedo_shape = tuple(int(v) for v in albedo.shape)
        self.image = colorize_glimpse_albedo(albedo)
        self.map_nbytes = self.image.nbytes
        ny, nx = self.albedo_shape

        max_dim = self.device.limits["max-texture-dimension-2d"]
        if max(nx, ny) > max_dim:
            raise ValueError(
                f"HUD minimap {nx}x{ny} exceeds the device's 2D texture "
                f"limit ({max_dim})."
            )

        self._texture = self.device.create_texture(
            label="hud-minimap",
            size=(nx, ny, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            dimension="2d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": self._texture},
            self.image,
            {"bytes_per_row": nx * 4, "rows_per_image": ny},
            (nx, ny, 1),
        )
        self._sampler = self.device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
        )
        self._ubuf = self.device.create_buffer(
            label="hud-uniforms",
            size=_UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
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
                "texture": {"sample_type": "float", "view_dimension": "2d"},
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
                {"binding": 0, "resource": {"buffer": self._ubuf,
                                            "offset": 0,
                                            "size": _UNIFORM_NBYTES}},
                {"binding": 1, "resource": self._texture.create_view()},
                {"binding": 2, "resource": self._sampler},
            ],
        )
        self._pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._shader = self.device.create_shader_module(
            label="hud-minimap", code=SHADER_PATH.read_text()
        )
        self._pipelines = {}
        self._last_state = None

    def rect_for_size(self, size: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Return top-right minimap rectangle as ``(x, y, w, h)`` pixels."""
        screen_w, screen_h = (float(size[0]), float(size[1]))
        ny, nx = self.albedo_shape
        aspect = nx / ny
        margin = max(8.0, round(screen_h * MAP_MARGIN_FRAC))

        map_h = max(24.0, round(screen_h * MAP_HEIGHT_FRAC))
        map_w = round(map_h * aspect)
        max_w = max(24.0, screen_w * MAP_MAX_WIDTH_FRAC)
        if map_w > max_w:
            scale = max_w / map_w
            map_w *= scale
            map_h *= scale

        avail_w = max(24.0, screen_w - 2.0 * margin)
        avail_h = max(24.0, screen_h - 2.0 * margin)
        if map_w > avail_w:
            scale = avail_w / map_w
            map_w *= scale
            map_h *= scale
        if map_h > avail_h:
            scale = avail_h / map_h
            map_w *= scale
            map_h *= scale

        return (screen_w - margin - map_w, margin, map_w, map_h)

    def nest_map_uv(self):
        """Nested field's horizontal footprint in map UV, or None.

        The map spans the OUTER field's x/y extent, and the nest is required
        to lie inside it, so the fractions are always in [0, 1]. Vertical
        extent is not shown: the minimap is a top-down plan view, and a
        nest's z range has no place to go on it.
        """
        renderer = self.renderer
        if not getattr(renderer, "nested", False):
            return None
        bmin = np.asarray(renderer.bmin, dtype=np.float64)
        bmax = np.asarray(renderer.bmax, dtype=np.float64)
        extent = bmax - bmin
        if np.any(extent[:2] <= 0.0):
            return None
        lo = (np.asarray(renderer.nest_bmin, dtype=np.float64) - bmin) / extent
        hi = (np.asarray(renderer.nest_bmax, dtype=np.float64) - bmin) / extent
        return (
            float(np.clip(lo[0], 0.0, 1.0)), float(np.clip(lo[1], 0.0, 1.0)),
            float(np.clip(hi[0], 0.0, 1.0)), float(np.clip(hi[1], 0.0, 1.0)),
        )

    def nest_pixel_rect(self, size: Tuple[int, int]):
        """The nest outline in screen pixels ``(x, y, w, h)``, or None."""
        nest_uv = self.nest_map_uv()
        if nest_uv is None:
            return None
        rect = self.rect_for_size(size)
        u0, v0, u1, v1 = nest_uv
        # uv_to_screen flips y; sort so the result is a positive-extent box.
        xs = sorted((rect[0] + u0 * rect[2], rect[0] + u1 * rect[2]))
        ys = sorted((
            rect[1] + (1.0 - v0) * rect[3],
            rect[1] + (1.0 - v1) * rect[3],
        ))
        return (xs[0], ys[0], xs[1] - xs[0], ys[1] - ys[0])

    def marker_pixel(self, camera: Camera, size: Tuple[int, int]) -> Tuple[float, float]:
        """Return the marker center in screen pixels for tests/diagnostics."""
        rect = self.rect_for_size(size)
        overlay = _camera_overlay_geometry(
            camera, self.albedo_shape, size[0] / size[1]
        )
        cam_u, cam_v = overlay["camera_uv"]
        return (rect[0] + cam_u * rect[2], rect[1] + (1.0 - cam_v) * rect[3])

    def write_uniforms(self, camera: Camera, size: Tuple[int, int]) -> None:
        """Pack minimap layout + camera/FOV overlay and enqueue upload."""
        screen_w, screen_h = (float(size[0]), float(size[1]))
        rect = self.rect_for_size(size)
        ny, nx = self.albedo_shape
        overlay = _camera_overlay_geometry(
            camera, self.albedo_shape, screen_w / screen_h
        )
        cam_u, cam_v = overlay["camera_uv"]

        min_side = min(rect[2], rect[3])
        marker_radius = max(3.0, min_side * 0.028)
        line_width = max(1.25, min_side * 0.010)
        border_width = max(1.0, min_side * 0.008)
        halo_width = max(0.85, line_width * 0.75)

        mode = 0.0
        circle_radius = 0.0
        endpoints = overlay.get("fov_endpoints")
        if endpoints is None:
            mode = 1.0
            nxm1 = max(nx - 1, 1)
            circle_radius = (
                float(overlay["circle_radius_px"]) * rect[2] / nxm1
            )
            left_u, left_v, right_u, right_v = 0.0, 0.0, 0.0, 0.0
        else:
            (left_u, left_v), (right_u, right_v) = endpoints

        nest_uv = self.nest_map_uv()
        u = np.empty((6, 4), dtype=np.float32)
        u[0] = [screen_w, screen_h, MAP_OPACITY, marker_radius]
        u[1] = [*rect]
        u[2] = [cam_u, cam_v, mode, circle_radius]
        u[3] = [left_u, left_v, right_u, right_v]
        u[4] = [
            line_width, border_width, halo_width,
            0.0 if nest_uv is None else 1.0,
        ]
        u[5] = [0.0, 0.0, 0.0, 0.0] if nest_uv is None else list(nest_uv)
        self.device.queue.write_buffer(self._ubuf, 0, u.tobytes())

        self._last_state = {
            "size": tuple(size),
            "rect": rect,
            "camera_uv": (cam_u, cam_v),
            "mode": mode,
            "circle_radius": circle_radius,
            "endpoints": endpoints,
            "marker_pixel": (
                rect[0] + cam_u * rect[2],
                rect[1] + (1.0 - cam_v) * rect[3],
            ),
            "nest_uv": nest_uv,
        }

    def _pipeline_for(self, target_format: str):
        if target_format not in self._pipelines:
            self._pipelines[target_format] = \
                self.device.create_render_pipeline(
                    label="hud-minimap",
                    layout=self._pipeline_layout,
                    vertex={
                        "module": self._shader,
                        "entry_point": "vs_main",
                    },
                    primitive={"topology": "triangle-list"},
                    fragment={
                        "module": self._shader,
                        "entry_point": "fs_main",
                        "targets": [{
                            "format": target_format,
                            "blend": {
                                "color": {
                                    "src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                                "alpha": {
                                    "src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                            },
                        }],
                    },
                )
        return self._pipelines[target_format]

    def encode_pass(self, command_encoder, target_view, target_format: str,
                    timestamp_writes=None) -> None:
        """Encode the HUD pass over an already-rendered frame."""
        desc = {
            "color_attachments": [{
                "view": target_view,
                "load_op": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            }],
        }
        if timestamp_writes is not None:
            desc["timestamp_writes"] = timestamp_writes
        rpass = command_encoder.begin_render_pass(**desc)
        rpass.set_pipeline(self._pipeline_for(target_format))
        rpass.set_bind_group(0, self._bind_group)
        if self._last_state is not None:
            screen_w, screen_h = self._last_state["size"]
            x, y, w, h = self._last_state["rect"]
            pad = 3.0
            x0 = max(0, int(np.floor(x - pad)))
            y0 = max(0, int(np.floor(y - pad)))
            x1 = min(int(screen_w), int(np.ceil(x + w + pad)))
            y1 = min(int(screen_h), int(np.ceil(y + h + pad)))
            if x1 > x0 and y1 > y0:
                rpass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0)
        rpass.draw(3)
        rpass.end()
