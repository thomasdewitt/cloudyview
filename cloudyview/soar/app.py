"""Windowed fly-through app (glfw via rendercanvas, wgpu-py's gui stack).

Controls:
    W/A/S/D     move forward/left/back/right (horizontal)
    Space       move up
    LShift / C  move down
    mouse       look (cursor is captured, video-game style)
    Tab         release / recapture the mouse
    scroll      movement speed (exponential)
    J           toggle jittered ray starts (A/B the banding fix)
    ESC         quit

The window title shows a running fps readout and the current camera state
in cv.Camera terms, so a good viewpoint can be transcribed straight into a
witness/behold render call.
"""

from time import perf_counter

import numpy as np

from ..camera import Camera
from ..cloudfield import CloudField
from .engine import InteractiveRenderer, request_device

DEFAULT_SPEED = 60.0        # m/s, comfortable for the 25 km dev domain
MOUSE_SENS = 0.12           # degrees per pixel
SPEED_WHEEL_FACTOR = 1.25   # per wheel notch


class FlyThroughApp:
    def __init__(self, field: CloudField, *, size=(1280, 720),
                 extinction_multiplier: float = 1.0):
        # Import here so offscreen use never needs glfw / a display.
        from rendercanvas.glfw import RenderCanvas, loop

        self._loop = loop
        self.canvas = RenderCanvas(
            title="cloudyview", size=size, update_mode="fastest", vsync=False
        )

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
        self.azimuth = cam0.azimuth
        self.elevation = cam0.elevation
        self.fov = cam0.fov

        self.speed = DEFAULT_SPEED
        self.jitter = True
        self._keys = set()
        self._last_pointer = None   # None -> ignore next move (capture jump guard)
        self._captured = False
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
        self._captured = capture
        self._last_pointer = None

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
            if key == "Escape":
                self.canvas.close()
            elif key in ("j", "J"):
                self.jitter = not self.jitter
            elif key == "Tab":
                self._capture_mouse(not self._captured)
            else:
                self._keys.add(key.lower() if len(key) == 1 else key)
        elif etype == "key_up":
            key = event["key"]
            self._keys.discard(key.lower() if len(key) == 1 else key)
        elif etype == "pointer_down" and not self._captured:
            self._capture_mouse(True)   # click back in to recapture
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
        cam = Camera(azimuth=self.azimuth, elevation=self.elevation,
                     fov=self.fov)
        forward, right, _up = cam.basis()
        # Horizontal-plane movement (game-style): flatten forward.
        fwd_h = np.array([forward[0], forward[1], 0.0])
        n = np.linalg.norm(fwd_h)
        fwd_h = fwd_h / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])

        step = np.zeros(3)
        if "w" in self._keys:
            step += fwd_h
        if "s" in self._keys:
            step -= fwd_h
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

    def _draw(self):
        now = perf_counter()
        dt = now - self._last_time
        self._last_time = now
        self._move(min(dt, 0.1))

        w, h = self.canvas.get_physical_size()
        self.renderer.write_uniforms(
            self.camera(), (w, h), jitter=self.jitter,
            frame_index=self._frame_index)
        self._frame_index += 1

        texture = self.context.get_current_texture()
        enc = self.renderer.device.create_command_encoder()
        self.renderer.encode_pass(enc, texture.create_view(), self.format)
        self.renderer.device.queue.submit([enc.finish()])

        self._fps_acc.append(dt)
        if now - self._fps_last_title > 0.5 and self._fps_acc:
            fps = 1.0 / (sum(self._fps_acc) / len(self._fps_acc))
            cam = self.camera()
            self.canvas.set_title(
                f"cloudyview  {fps:5.1f} fps  "
                f"pos=({cam.position[0]:+.2f},{cam.position[1]:+.2f},"
                f"{cam.position[2]:+.2f}) az={cam.azimuth:.0f} "
                f"el={cam.elevation:.0f} speed={self.speed:.0f}m/s "
                f"jitter={'on' if self.jitter else 'OFF'}"
            )
            self._fps_acc = []
            self._fps_last_title = now

        self.canvas.request_draw()

    def run(self):
        self._loop.run()


def run_app(field: CloudField, **kwargs):
    """Open the fly-through window for a loaded CloudField (blocks)."""
    FlyThroughApp(field, **kwargs).run()
