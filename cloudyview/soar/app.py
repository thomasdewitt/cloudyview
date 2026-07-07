"""Windowed fly-through app (glfw via rendercanvas, wgpu-py's gui stack).

Controls:
    W/S         move forward/back along the view direction
    A/D         strafe left/right
    Space       move up
    LShift / C  move down
    mouse       look (cursor is captured, video-game style)
    Tab         release / recapture the mouse
    scroll      movement speed (exponential)
    1/2/3/4     toggle cumulonimbus realism terms (gradient/MS/AO/bounce)
    J           toggle jittered ray starts (A/B the banding fix)
    B           toggle the bird (the flying subject leading the camera)
    M           toggle the minimap
    F           toggle fullscreen/windowed
    ESC         pause menu (releases the mouse)

Pause menu:
    R / click   resume and recapture the mouse
    F           toggle fullscreen/windowed
    ESC / Q     quit

The window title shows a running fps readout and the current camera state
in cv.Camera terms plus the cumulonimbus realism gate bitfield (e.g.
cb:1010), so a good viewpoint and lighting A/B can be transcribed straight
into a witness/behold/soar render call.
"""

from time import perf_counter

import numpy as np

from ..camera import Camera
from ..cloudfield import CloudField
from .engine import (
    DEFAULT_AMBIENT_STRENGTH,
    DEFAULT_AMBIENT_OCCLUSION_STRENGTH,
    DEFAULT_BOUNCE_DEPTH_ATTENUATION,
    DEFAULT_DEEP_SHADOW_MS_SUPPRESSION,
    DEFAULT_EXPOSURE,
    DEFAULT_GRADIENT_SHADING_STRENGTH,
    DEFAULT_SUN_AZIMUTH,
    DEFAULT_SUN_ELEVATION,
    InteractiveRenderer,
    request_device,
)

DEFAULT_SPEED = 60.0        # m/s, comfortable for the 25 km dev domain
MOUSE_SENS = 0.12           # degrees per pixel
SPEED_WHEEL_FACTOR = 1.25   # per wheel notch
OCEAN_FLOOR_MARGIN_M = 2.0

ACTION_PAUSE = "pause"
ACTION_RESUME = "resume"
ACTION_QUIT = "quit"
ACTION_TOGGLE_FULLSCREEN = "toggle_fullscreen"

CONTROL_SUMMARY = (
    "Controls: W/S forward/back, A/D strafe, Space up, LShift/C down, mouse look "
    "(Tab releases, click recaptures), scroll speed, "
    "1 gradient, 2 MS floor, 3 ambient AO, 4 bounce attenuation, "
    "J jitter toggle, B bird toggle, M minimap toggle, "
    "F fullscreen/window, ESC pause menu; "
    "paused: R/click resume, F fullscreen/window, ESC/Q quit"
)

CB_DEFAULT_STRENGTHS = (
    DEFAULT_GRADIENT_SHADING_STRENGTH,
    DEFAULT_DEEP_SHADOW_MS_SUPPRESSION,
    DEFAULT_AMBIENT_OCCLUSION_STRENGTH,
    DEFAULT_BOUNCE_DEPTH_ATTENUATION,
)

_PAUSE_OVERLAY_SHADER = """
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32)
        -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index) - 1);
    let y = f32(i32(vertex_index & 1u) * 2 - 1);
    return vec4<f32>(x * 3.0, y * 3.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.42);
}
"""


def _normalized_key(key: str) -> str:
    return key.lower() if len(key) == 1 else key


def _control_action_for_key(paused: bool, key: str) -> str | None:
    """Pure pause/fullscreen key state machine for the window shell."""
    normalized = _normalized_key(key)
    if paused:
        if key == "Escape" or normalized == "q":
            return ACTION_QUIT
        if normalized == "r":
            return ACTION_RESUME
        if normalized == "f":
            return ACTION_TOGGLE_FULLSCREEN
        return None
    if key == "Escape":
        return ACTION_PAUSE
    if normalized == "f":
        return ACTION_TOGGLE_FULLSCREEN
    return None


def _clamp_position_above_ocean(
    position, margin: float = OCEAN_FLOOR_MARGIN_M
) -> np.ndarray:
    """Return a copy of a world-space camera position held above z=0 ocean."""
    clamped = np.asarray(position, dtype=np.float64).copy()
    clamped[2] = max(clamped[2], margin)
    return clamped


class FlyThroughApp:
    def __init__(self, field: CloudField, *, size=(1280, 720),
                 extinction_multiplier: float = 1.0,
                 max_fps: float = 120.0):
        # Import here so offscreen use never needs glfw / a display.
        from rendercanvas.glfw import RenderCanvas, loop

        self._loop = loop
        # "continuous" honors max_fps; uncapped ("fastest", vsync off) burns
        # a full GPU rendering ~4000 fps nobody can see.
        self.canvas = RenderCanvas(
            title="cloudyview", size=size, update_mode="continuous",
            max_fps=max_fps, vsync=True,
        )
        self._ensure_resizable()

        device = request_device()
        self.renderer = InteractiveRenderer(
            field, extinction_multiplier=extinction_multiplier, device=device
        )

        self.context = self.canvas.get_context("wgpu")
        self.format = self.context.get_preferred_format(device.adapter)
        self.context.configure(device=device, format=self.format)

        # Camera state: world meters + met angles. Start at the default
        # witness viewpoint.
        cam0 = Camera()
        from .engine import camera_world_origin
        self.position = camera_world_origin(
            cam0, self.renderer.bmin, self.renderer.bmax)
        self.position = _clamp_position_above_ocean(self.position)
        self.azimuth = cam0.azimuth
        self.elevation = cam0.elevation
        self.fov = cam0.fov

        self.speed = DEFAULT_SPEED
        self.jitter = True
        self.cb_enabled = [True, True, True, True]
        self.bird_enabled = True
        self.minimap_enabled = True
        self._keys = set()
        self._last_pointer = None   # None -> ignore next move (capture jump guard)
        self._captured = False
        self._paused = False
        self._fullscreen = False
        self._windowed_bounds = None
        self._pause_overlay_pipelines = {}
        self._capture_mouse(True)
        self._last_time = perf_counter()
        self._frame_index = 0
        self._fps_acc = []
        self._fps_last_title = 0.0

        self.canvas.add_event_handler(self._on_event,
                                      "key_down", "key_up",
                                      "pointer_down", "pointer_up",
                                      "pointer_move", "wheel")
        self.canvas.request_draw(self._draw)

    # ------------------------------------------------------------------

    def _ensure_resizable(self):
        """Keep the GLFW window user-resizable even if backend defaults shift."""
        import glfw

        glfw.set_window_attrib(self.canvas._window, glfw.RESIZABLE, glfw.TRUE)

    def _capture_mouse(self, capture: bool):
        """Game-style pointer capture via glfw (rendercanvas has no
        pointer-lock API; the glfw window handle is the supported way in)."""
        import glfw

        window = self.canvas._window
        if capture:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            if glfw.raw_mouse_motion_supported():
                glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.TRUE)
        else:
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
            if glfw.raw_mouse_motion_supported():
                glfw.set_input_mode(window, glfw.RAW_MOUSE_MOTION, glfw.FALSE)
        self._captured = capture
        self._last_pointer = None

    def _paused_title(self, fps: float | None = None) -> str:
        target = "windowed" if self._fullscreen else "fullscreen"
        fps_part = "" if fps is None else f"  {fps:5.1f} fps"
        return (
            f"cloudyview PAUSED{fps_part}  frame={self._frame_index}  "
            f"cb:{self._cb_bits()}  R/click resume  F {target}  ESC/Q quit"
        )

    def _cb_bits(self) -> str:
        enabled = getattr(self, "cb_enabled", (True, True, True, True))
        return "".join("1" if value else "0" for value in enabled)

    def _cb_strength_kwargs(self) -> dict:
        strengths = [
            value if enabled else 0.0
            for enabled, value in zip(self.cb_enabled, CB_DEFAULT_STRENGTHS)
        ]
        return {
            "gradient_shading_strength": strengths[0],
            "deep_shadow_ms_suppression": strengths[1],
            "ambient_occlusion_strength": strengths[2],
            "bounce_depth_attenuation": strengths[3],
        }

    def _set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        self._keys.clear()
        if paused:
            self._capture_mouse(False)
            self.canvas.set_title(self._paused_title())
        else:
            self._capture_mouse(True)
            self._last_time = perf_counter()
        self.canvas.request_draw()

    def _current_monitor(self):
        import glfw

        window = self.canvas._window
        if window is None:
            raise RuntimeError("Cannot choose a fullscreen monitor: window closed.")

        monitor = glfw.get_window_monitor(window)
        if monitor is not None:
            return monitor

        monitors = glfw.get_monitors()
        if not monitors:
            raise RuntimeError("Cannot enter fullscreen: GLFW found no monitors.")

        wx, wy = glfw.get_window_pos(window)
        ww, wh = glfw.get_window_size(window)
        wcx, wcy = wx + ww * 0.5, wy + wh * 0.5

        best = None
        best_overlap = -1
        best_distance = float("inf")
        for candidate in monitors:
            mode = glfw.get_video_mode(candidate)
            if mode is None:
                continue
            mx, my = glfw.get_monitor_pos(candidate)
            mw, mh = int(mode.width), int(mode.height)
            overlap_w = max(0, min(wx + ww, mx + mw) - max(wx, mx))
            overlap_h = max(0, min(wy + wh, my + mh) - max(wy, my))
            overlap = overlap_w * overlap_h
            mcx, mcy = mx + mw * 0.5, my + mh * 0.5
            distance = (wcx - mcx) ** 2 + (wcy - mcy) ** 2
            if overlap > best_overlap or (
                overlap == best_overlap and distance < best_distance
            ):
                best = candidate
                best_overlap = overlap
                best_distance = distance

        if best is None:
            raise RuntimeError("Cannot enter fullscreen: no video mode found.")
        return best

    def _toggle_fullscreen(self) -> None:
        import glfw

        window = self.canvas._window
        if window is None:
            raise RuntimeError("Cannot toggle fullscreen: window closed.")

        if self._fullscreen:
            if self._windowed_bounds is None:
                raise RuntimeError(
                    "Cannot restore windowed mode: previous bounds missing."
                )
            x, y, w, h = self._windowed_bounds
            glfw.set_window_monitor(window, None, x, y, w, h, 0)
            self._fullscreen = False
        else:
            x, y = glfw.get_window_pos(window)
            w, h = glfw.get_window_size(window)
            self._windowed_bounds = (int(x), int(y), int(w), int(h))
            monitor = self._current_monitor()
            mode = glfw.get_video_mode(monitor)
            if mode is None:
                raise RuntimeError("Cannot enter fullscreen: no video mode found.")
            glfw.set_window_monitor(
                window, monitor, 0, 0,
                int(mode.width), int(mode.height), int(mode.refresh_rate),
            )
            self._fullscreen = True

        self.canvas._determine_size()
        if self._paused:
            self.canvas.set_title(self._paused_title())
        self.canvas.request_draw()

    def camera(self) -> Camera:
        """Current viewpoint as a cv.Camera (relative-coordinate position)."""
        bmin, bmax = self.renderer.bmin, self.renderer.bmax
        rel = (
            2.0 * (self.position[0] - bmin[0]) / (bmax[0] - bmin[0]) - 1.0,
            2.0 * (self.position[1] - bmin[1]) / (bmax[1] - bmin[1]) - 1.0,
            2.0 * self.position[2] / bmax[2] - 1.0,
        )
        return Camera(position=rel, azimuth=self.azimuth,
                      elevation=self.elevation, fov=self.fov)

    def _on_event(self, event):
        etype = event["event_type"]
        if etype == "key_down":
            key = event["key"]
            action = _control_action_for_key(self._paused, key)
            if action == ACTION_PAUSE:
                self._set_paused(True)
            elif action == ACTION_RESUME:
                self._set_paused(False)
            elif action == ACTION_QUIT:
                self.canvas.close()
            elif action == ACTION_TOGGLE_FULLSCREEN:
                self._toggle_fullscreen()
            elif self._paused:
                return
            elif key in ("j", "J"):
                self.jitter = not self.jitter
            elif key in ("b", "B"):
                self.bird_enabled = not self.bird_enabled
            elif key in ("m", "M"):
                self.minimap_enabled = not self.minimap_enabled
            elif key in ("1", "2", "3", "4"):
                self.cb_enabled[int(key) - 1] = not self.cb_enabled[int(key) - 1]
            elif key == "Tab":
                self._capture_mouse(not self._captured)
            else:
                self._keys.add(key.lower() if len(key) == 1 else key)
        elif etype == "key_up":
            key = event["key"]
            self._keys.discard(key.lower() if len(key) == 1 else key)
        elif etype == "pointer_down":
            if self._paused:
                self._set_paused(False)
            elif not self._captured:
                self._capture_mouse(True)   # click back in to recapture
        elif self._paused:
            return
        elif etype == "pointer_move" and self._captured:
            if self._last_pointer is None:
                self._last_pointer = (event["x"], event["y"])
                return
            dx = event["x"] - self._last_pointer[0]
            dy = event["y"] - self._last_pointer[1]
            self._last_pointer = (event["x"], event["y"])
            self.azimuth = (self.azimuth + dx * MOUSE_SENS) % 360.0
            self.elevation = float(np.clip(
                self.elevation - dy * MOUSE_SENS, -89.0, 89.0))
        elif etype == "wheel":
            notches = -event.get("dy", 0.0) / 100.0
            self.speed = float(np.clip(
                self.speed * SPEED_WHEEL_FACTOR ** notches, 0.5, 5000.0))

    def _move(self, dt: float):
        if self._paused:
            return

        cam = Camera(azimuth=self.azimuth, elevation=self.elevation,
                     fov=self.fov)
        forward, right, _up = cam.basis()

        step = np.zeros(3)
        if "w" in self._keys:
            step += forward
        if "s" in self._keys:
            step -= forward
        if "d" in self._keys:
            step += right
        if "a" in self._keys:
            step -= right
        if " " in self._keys:
            step += np.array([0.0, 0.0, 1.0])
        if "c" in self._keys or "Shift" in self._keys:
            step -= np.array([0.0, 0.0, 1.0])
        if np.any(step):
            self.position = self.position + step * (self.speed * dt)
        self.position = _clamp_position_above_ocean(self.position)

    def _pause_overlay_pipeline_for(self, target_format: str):
        import wgpu

        if target_format not in self._pause_overlay_pipelines:
            shader = self.renderer.device.create_shader_module(
                label="pause-menu-dim", code=_PAUSE_OVERLAY_SHADER
            )
            self._pause_overlay_pipelines[target_format] = (
                self.renderer.device.create_render_pipeline(
                    label="pause-menu-dim",
                    layout="auto",
                    vertex={"module": shader, "entry_point": "vs_main"},
                    primitive={"topology": "triangle-list"},
                    fragment={
                        "module": shader,
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
            )
        return self._pause_overlay_pipelines[target_format]

    def _encode_pause_overlay(self, command_encoder, target_view) -> None:
        import wgpu

        rpass = command_encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        rpass.set_pipeline(self._pause_overlay_pipeline_for(self.format))
        rpass.draw(3)
        rpass.end()

    def _draw(self):
        now = perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if not self._paused:
            self._move(min(dt, 0.1))
        self.position = _clamp_position_above_ocean(self.position)

        w, h = self.canvas.get_physical_size()
        self.renderer.write_uniforms(
            self.camera(), (w, h), jitter=self.jitter,
            frame_index=self._frame_index,
            **self._cb_strength_kwargs())
        self._frame_index += 1

        texture = self.context.get_current_texture()
        view = texture.create_view()
        enc = self.renderer.device.create_command_encoder()
        self.renderer.encode_pass(enc, view, self.format)
        if self.bird_enabled:
            bird = self.renderer.bird
            bird.update(
                0.0 if self._paused else dt,
                self.position, self.azimuth, self.elevation,
            )
            bird.write_uniforms(
                self.position, self.camera(), (w, h),
                sun_azimuth=DEFAULT_SUN_AZIMUTH,
                sun_elevation=DEFAULT_SUN_ELEVATION,
                exposure=DEFAULT_EXPOSURE,
                ambient_strength=DEFAULT_AMBIENT_STRENGTH,
            )
            bird.encode_pass(enc, view, self.format, (w, h))
        if self.minimap_enabled:
            self.renderer.hud.write_uniforms(self.camera(), (w, h))
            self.renderer.hud.encode_pass(enc, view, self.format)
        if self._paused:
            self._encode_pause_overlay(enc, view)
        self.renderer.device.queue.submit([enc.finish()])

        self._fps_acc.append(dt)
        if now - self._fps_last_title > 0.5 and self._fps_acc:
            fps = 1.0 / (sum(self._fps_acc) / len(self._fps_acc))
            if self._paused:
                self.canvas.set_title(self._paused_title(fps))
            else:
                cam = self.camera()
                self.canvas.set_title(
                    f"cloudyview  {fps:5.1f} fps  "
                    f"pos=({cam.position[0]:+.2f},{cam.position[1]:+.2f},"
                    f"{cam.position[2]:+.2f}) az={cam.azimuth:.0f} "
                    f"el={cam.elevation:.0f} speed={self.speed:.0f}m/s "
                    f"cb:{self._cb_bits()} "
                    f"jitter={'on' if self.jitter else 'OFF'} "
                    f"map={'on' if self.minimap_enabled else 'OFF'}"
                )
            self._fps_acc = []
            self._fps_last_title = now

        self.canvas.request_draw()

    def run(self):
        self._loop.run()


def run_app(field: CloudField, **kwargs):
    """Open the fly-through window for a loaded CloudField (blocks)."""
    FlyThroughApp(field, **kwargs).run()
