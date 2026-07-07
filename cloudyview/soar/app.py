"""Windowed fly-through app (glfw via rendercanvas, wgpu-py's gui stack).

Controls:
    W/S         move forward/back along the view direction
    A/D         strafe left/right
    Space       move up
    LShift / C  move down
    mouse       look (cursor is captured, video-game style)
    Tab         release / recapture the mouse
    scroll      movement speed (exponential)
    J           toggle jittered ray starts (A/B the banding fix)
    B           toggle the bird (the flying subject leading the camera)
    M           toggle the minimap
    F           toggle fullscreen/windowed
    F12         save a PNG screenshot with render metadata
    ESC         pause menu (releases the mouse)

Pause menu:
    R / click   resume and recapture the mouse
    O           open a new .nc file (optionally a split ice .nc)
    G           render current view in behold (then 1=min, 2=low,
                3=medium, 4=high; ESC backs out)
    F           toggle fullscreen/windowed
    ESC / Q     quit

The window title shows a running fps readout and the current camera state
in cv.Camera terms, so a good viewpoint can be transcribed straight into a
witness/behold render call.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shlex
from time import perf_counter

import numpy as np

from ..camera import Camera
from ..cloudfield import CloudField, load as load_cloud_field
from ..render_metadata import build_render_metadata, embed_metadata
from .engine import (
    DEFAULT_AMBIENT_STRENGTH,
    DEFAULT_EXPOSURE,
    DEFAULT_SUN_AZIMUTH,
    DEFAULT_SUN_ELEVATION,
    InteractiveRenderer,
    camera_world_origin,
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
ACTION_OPEN_FILE = "open_file"
ACTION_OPEN_ICE_YES = "open_ice_yes"
ACTION_OPEN_ICE_NO = "open_ice_no"
ACTION_RENDER_MENU = "render_menu"
ACTION_RENDER_BEHOLD = "render_behold"
ACTION_MENU_BACK = "menu_back"
ACTION_SCREENSHOT = "screenshot"

MENU_MAIN = "main"
MENU_OPEN_ICE_PROMPT = "open_ice_prompt"
MENU_RENDER_QUALITY = "render_quality"

BEHOLD_QUALITIES_BY_KEY = {
    "1": "min",
    "2": "low",
    "3": "medium",
    "4": "high",
}

CONTROL_SUMMARY = (
    "Controls: W/S forward/back, A/D strafe, Space up, LShift/C down, mouse look "
    "(Tab releases, click recaptures), scroll speed, "
    "J jitter toggle, B bird toggle, M minimap toggle, "
    "F fullscreen/window, F12 screenshot, ESC pause menu; "
    "paused: R/click resume, O open file, G behold render, "
    "F fullscreen/window, ESC/Q quit"
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


@dataclass(frozen=True)
class MenuTransition:
    action: str | None
    next_state: str | None = None
    quality: str | None = None


def _menu_transition(
    paused: bool, menu_state: str, key: str
) -> MenuTransition:
    """Pure key state machine for flight, pause menu, and pause submenus."""
    normalized = _normalized_key(key)
    if not paused:
        if key == "Escape":
            return MenuTransition(ACTION_PAUSE, MENU_MAIN)
        if key == "F12":
            return MenuTransition(ACTION_SCREENSHOT)
        if normalized == "f":
            return MenuTransition(ACTION_TOGGLE_FULLSCREEN)
        return MenuTransition(None)

    if menu_state == MENU_MAIN:
        if key == "Escape" or normalized == "q":
            return MenuTransition(ACTION_QUIT, MENU_MAIN)
        if normalized == "r":
            return MenuTransition(ACTION_RESUME, MENU_MAIN)
        if normalized == "f":
            return MenuTransition(ACTION_TOGGLE_FULLSCREEN, MENU_MAIN)
        if normalized == "o":
            return MenuTransition(ACTION_OPEN_FILE, MENU_MAIN)
        if normalized == "g":
            return MenuTransition(ACTION_RENDER_MENU, MENU_RENDER_QUALITY)
        return MenuTransition(None, MENU_MAIN)

    if menu_state == MENU_OPEN_ICE_PROMPT:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        if normalized == "q":
            return MenuTransition(ACTION_QUIT, MENU_OPEN_ICE_PROMPT)
        if normalized == "y":
            return MenuTransition(ACTION_OPEN_ICE_YES, MENU_OPEN_ICE_PROMPT)
        if normalized == "n":
            return MenuTransition(ACTION_OPEN_ICE_NO, MENU_OPEN_ICE_PROMPT)
        return MenuTransition(None, MENU_OPEN_ICE_PROMPT)

    if menu_state == MENU_RENDER_QUALITY:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        if normalized == "q":
            return MenuTransition(ACTION_QUIT, MENU_RENDER_QUALITY)
        quality = BEHOLD_QUALITIES_BY_KEY.get(normalized)
        if quality is not None:
            return MenuTransition(
                ACTION_RENDER_BEHOLD, MENU_RENDER_QUALITY, quality
            )
        return MenuTransition(None, MENU_RENDER_QUALITY)

    return MenuTransition(None, menu_state)


def _control_action_for_key(
    paused: bool, key: str, menu_state: str = MENU_MAIN
) -> str | None:
    """Backward-compatible action-only view of the menu state machine."""
    return _menu_transition(paused, menu_state, key).action


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
                 max_fps: float = 120.0,
                 camera: Camera | None = None):
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

        self._extinction_multiplier = float(extinction_multiplier)
        self.sun_azimuth = DEFAULT_SUN_AZIMUTH
        self.sun_elevation = DEFAULT_SUN_ELEVATION

        device = request_device()
        self.renderer = self._create_renderer(field, device=device)

        self.context = self.canvas.get_context("wgpu")
        self.format = self.context.get_preferred_format(device.adapter)
        self.context.configure(device=device, format=self.format)

        # Camera state: world meters + met angles. Start at the default
        # witness viewpoint.
        self._reset_camera_to_default(camera)

        self.speed = DEFAULT_SPEED
        self.jitter = True
        self.bird_enabled = True
        self.minimap_enabled = True
        self._keys = set()
        self._last_pointer = None   # None -> ignore next move (capture jump guard)
        self._captured = False
        self._paused = False
        self._menu_state = MENU_MAIN
        self._pending_open_path = None
        self._rendering = False
        self._fullscreen = False
        self._windowed_bounds = None
        self._pause_overlay_pipelines = {}
        self._title_flash_text = None
        self._title_flash_until = 0.0
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

    def _create_renderer(self, field: CloudField, *, device=None, previous=None):
        """Create the field-resident renderer, reusing the app GPU device."""
        kwargs = {
            "extinction_multiplier": self._extinction_multiplier,
            "device": device,
        }
        if previous is not None:
            kwargs.update({
                "ocean_enabled": previous.ocean_enabled,
                "ocean_z": previous.ocean_z,
                "ocean_reflectance": previous.ocean_reflectance,
                "fif_normals": previous.ocean_fif_normals,
            })
        return InteractiveRenderer(field, **kwargs)

    def _reset_camera_to_default(self, camera: Camera | None = None) -> None:
        """Reset the app camera to the shared default or supplied ``cv.Camera``."""
        cam0 = camera or Camera()
        self.position = camera_world_origin(
            cam0, self.renderer.bmin, self.renderer.bmax)
        self.position = _clamp_position_above_ocean(self.position)
        self.azimuth = cam0.azimuth
        self.elevation = cam0.elevation
        self.fov = cam0.fov

    def _install_field(self, field: CloudField) -> None:
        """Swap to a freshly loaded field and rebuild field-specific GPU state."""
        previous = self.renderer
        self.renderer = self._create_renderer(
            field, device=previous.device, previous=previous
        )
        self._reset_camera_to_default()
        self._frame_index = 0
        self.renderer.reset_accumulation()

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
            f"R/click resume  O open file  G behold  F {target}  ESC/Q quit"
        )

    def _open_ice_title(self) -> str:
        filename = (
            Path(self._pending_open_path).name if self._pending_open_path else ""
        )
        return (
            f"cloudyview OPEN FILE  {filename}  "
            "pick separate ice file?  Y yes  N no  ESC back"
        )

    def _render_quality_title(self) -> str:
        return (
            "cloudyview BEHOLD QUALITY  "
            "1 min  2 low  3 medium  4 high  ESC back"
        )

    def _menu_title(self, fps: float | None = None) -> str:
        state = getattr(self, "_menu_state", MENU_MAIN)
        if state == MENU_OPEN_ICE_PROMPT:
            return self._open_ice_title()
        if state == MENU_RENDER_QUALITY:
            return self._render_quality_title()
        return self._paused_title(fps)

    def _set_menu_state(self, state: str) -> None:
        self._menu_state = state
        if self._paused:
            self.canvas.set_title(self._menu_title())

    def _flash_title(self, title: str, *, seconds: float = 2.5) -> None:
        self._title_flash_text = title
        self._title_flash_until = perf_counter() + seconds
        self.canvas.set_title(title)

    def _title_flash_active(self, now: float | None = None) -> bool:
        if now is None:
            now = perf_counter()
        return bool(
            getattr(self, "_title_flash_text", None)
            and now < getattr(self, "_title_flash_until", 0.0)
        )

    def _set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        self._keys.clear()
        if paused:
            self._menu_state = MENU_MAIN
            self._capture_mouse(False)
            self.canvas.set_title(self._menu_title())
        else:
            self._menu_state = MENU_MAIN
            self._pending_open_path = None
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
            self.canvas.set_title(self._menu_title())
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

    def _ask_netcdf_file(self, *, title: str, initialdir: str | None = None):
        """Open the required native Tk file dialog and return a selected path."""
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as e:  # pragma: no cover - platform packaging only
            raise RuntimeError(
                "Native file dialogs require tkinter. Install python3-tkinter "
                "for your Python, then restart soar."
            ) from e

        root = tk.Tk()
        root.withdraw()
        try:
            return filedialog.askopenfilename(
                parent=root,
                title=title,
                initialdir=initialdir,
                filetypes=[
                    ("NetCDF files", "*.nc"),
                    ("All files", "*"),
                ],
            )
        finally:
            root.destroy()

    def _start_open_file(self) -> None:
        self.canvas.set_title("cloudyview OPEN FILE  choose liquid .nc")
        path = self._ask_netcdf_file(title="Open CloudyView NetCDF")
        if not path:
            self._pending_open_path = None
            self._set_menu_state(MENU_MAIN)
            return
        self._pending_open_path = str(path)
        self._set_menu_state(MENU_OPEN_ICE_PROMPT)

    def _finish_open_file(self, *, use_ice: bool) -> None:
        liquid_path = self._pending_open_path
        if not liquid_path:
            self._set_menu_state(MENU_MAIN)
            return

        ice_path = None
        if use_ice:
            self.canvas.set_title("cloudyview OPEN FILE  choose ice .nc")
            ice_path = self._ask_netcdf_file(
                title="Open split ice-variable NetCDF",
                initialdir=str(Path(liquid_path).parent),
            )
            if not ice_path:
                self._pending_open_path = None
                self._set_menu_state(MENU_MAIN)
                return

        self.canvas.set_title(
            f"cloudyview LOADING  {Path(liquid_path).name} ..."
        )
        try:
            field = load_cloud_field(liquid_path, ice=ice_path or None)
            self._install_field(field)
        except Exception as e:
            self._pending_open_path = None
            self._set_menu_state(MENU_MAIN)
            message = f"cloudyview open failed: {e}"
            print(message)
            self._flash_title(message, seconds=6.0)
            return

        print(f"Loaded {self.renderer.field}")
        self._pending_open_path = None
        self._set_paused(False)
        self._flash_title(
            f"cloudyview loaded {Path(liquid_path).name}", seconds=3.0
        )

    @staticmethod
    def _fmt_num(value: float) -> str:
        return f"{float(value):.12g}"

    @staticmethod
    def _fmt_eta(seconds: float | None) -> str:
        if not seconds or seconds != seconds or seconds == float("inf"):
            return "--:--"
        seconds = int(max(0, round(seconds)))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _quote_command(parts) -> str:
        return " ".join(shlex.quote(str(part)) for part in parts)

    def _behold_reproduction_command(self, camera: Camera, quality: str) -> str:
        field = self.renderer.field
        source = field.source or "<in-memory>"
        parts = ["behold", source, quality, "--gpu"]
        if field.ice_source:
            parts.extend(["--ice", field.ice_source])
        parts.extend([
            "--camera-position",
            *(self._fmt_num(v) for v in camera.position),
            "--camera-azimuth",
            self._fmt_num(camera.azimuth),
            "--camera-elevation",
            self._fmt_num(camera.elevation),
            "--fov",
            self._fmt_num(camera.fov),
            "--sun-azimuth",
            self._fmt_num(self.sun_azimuth),
            "--sun-elevation",
            self._fmt_num(self.sun_elevation),
        ])
        return self._quote_command(parts)

    def _soar_reproduction_command(self, camera: Camera) -> str:
        field = self.renderer.field
        source = field.source or "<in-memory>"
        parts = ["python", "-m", "cloudyview.soar", source]
        if field.ice_source:
            parts.extend(["--ice", field.ice_source])
        parts.extend([
            "--camera-position",
            *(self._fmt_num(v) for v in camera.position),
            "--camera-azimuth",
            self._fmt_num(camera.azimuth),
            "--camera-elevation",
            self._fmt_num(camera.elevation),
            "--fov",
            self._fmt_num(camera.fov),
        ])
        return self._quote_command(parts)

    def _timestamped_png_path(self, prefix: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Path.cwd() / f"{prefix}_{stamp}.png"

    def _write_png_with_metadata(
        self, image: np.ndarray, path: Path, metadata: dict
    ) -> None:
        from PIL import Image

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Expected RGB image array, got shape {arr.shape}.")
        Image.fromarray(arr[:, :, :3]).save(path)
        embed_metadata(path, metadata)

    def _metadata(
        self,
        camera: Camera,
        *,
        renderer: str,
        quality: str | None = None,
        reproduction_command: str,
        render_options: dict | None = None,
    ) -> dict:
        return build_render_metadata(
            self.renderer.field,
            camera,
            sun_azimuth=self.sun_azimuth,
            sun_elevation=self.sun_elevation,
            renderer=renderer,
            quality=quality,
            reproduction_command=reproduction_command,
            timestamp=datetime.now(timezone.utc),
            render_options=render_options,
        )

    def _save_screenshot(self) -> None:
        camera = self.camera()
        w, h = self.canvas.get_physical_size()
        path = self._timestamped_png_path("cloudyview_soar")
        renderer = self.renderer
        accum_state = (
            getattr(renderer, "_accum_key", None),
            getattr(renderer, "_accum_count", 0),
            getattr(renderer, "_accum_index", 0),
        )
        had_bird = getattr(renderer, "_bird", None) is not None
        bird_state = None
        if self.bird_enabled and had_bird:
            bird = renderer._bird
            bird_attrs = (
                "position",
                "heading",
                "view_elevation",
                "bank",
                "pitch",
                "flap_phase",
                "flap_amp",
                "flap_angle",
                "_speed",
                "_vz",
                "_clock",
                "_prev_origin",
            )
            bird_state = {
                name: (
                    getattr(bird, name).copy()
                    if isinstance(getattr(bird, name), np.ndarray)
                    else getattr(bird, name)
                )
                for name in bird_attrs
            }
        try:
            image = renderer.render(
                camera,
                size=(w, h),
                bird=self.bird_enabled,
                hud=self.minimap_enabled,
                jitter=self.jitter,
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
                frame_index=self._frame_index,
            )
        finally:
            (
                renderer._accum_key,
                renderer._accum_count,
                renderer._accum_index,
            ) = accum_state
            if bird_state is not None:
                for name, value in bird_state.items():
                    setattr(renderer._bird, name, value)
            elif self.bird_enabled and not had_bird:
                renderer._bird = None

        metadata = self._metadata(
            camera,
            renderer="soar",
            reproduction_command=self._soar_reproduction_command(camera),
            render_options={
                "bird": bool(self.bird_enabled),
                "hud": bool(self.minimap_enabled),
                "jitter": bool(self.jitter),
                "size": [int(w), int(h)],
            },
        )
        self._write_png_with_metadata(image, path, metadata)
        print(f"Screenshot saved to {path}")
        self._flash_title(f"cloudyview screenshot saved {path}", seconds=4.0)

    def _run_behold_render(self, quality: str) -> None:
        camera = self.camera()
        path = self._timestamped_png_path(f"cloudyview_behold_{quality}")
        self._rendering = True
        self.canvas.set_title(
            f"cloudyview BEHOLD {quality}  0%  ETA --:--  "
            "cannot cancel once started"
        )

        def on_progress(progress: dict) -> None:
            eta = progress.get("eta")
            if eta is None and progress.get("taken_spp"):
                elapsed = progress.get("elapsed", 0.0)
                taken = progress["taken_spp"]
                total = progress.get("spp_total", taken)
                eta = elapsed * max(0, total - taken) / taken
            self.canvas.set_title(
                f"cloudyview BEHOLD {quality}  "
                f"{progress.get('percent', 0.0):5.1f}%  "
                f"ETA {self._fmt_eta(eta)}  cannot cancel once started"
            )

        try:
            from .. import behold as behold_render

            image = behold_render(
                self.renderer.field,
                camera=camera,
                quality=quality,
                gpu=True,
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
                progress_callback=on_progress,
            )
            metadata = self._metadata(
                camera,
                renderer="behold",
                quality=quality,
                reproduction_command=self._behold_reproduction_command(
                    camera, quality
                ),
            )
            self._write_png_with_metadata(image, path, metadata)
        except ImportError as e:
            self._rendering = False
            self._set_menu_state(MENU_MAIN)
            message = (
                "cloudyview behold requires Mitsuba; install the "
                "radiative-transfer extra, e.g. uv sync --extra "
                f"radiative-transfer ({e})"
            )
            print(message)
            self._flash_title(message, seconds=7.0)
            self.canvas.request_draw()
            return
        except Exception as e:
            self._rendering = False
            self._set_menu_state(MENU_MAIN)
            detail = str(e)
            if any(token in detail.lower() for token in ("mitsuba", "drjit")):
                message = (
                    "cloudyview behold requires Mitsuba; install the "
                    "radiative-transfer extra, e.g. uv sync --extra "
                    f"radiative-transfer ({e})"
                )
            else:
                message = f"cloudyview behold failed: {e}"
            print(message)
            self._flash_title(message, seconds=7.0)
            self.canvas.request_draw()
            return

        self._rendering = False
        print(f"Behold render saved to {path}")
        self._set_paused(False)
        self._flash_title(f"cloudyview behold saved {path}", seconds=5.0)

    def _on_event(self, event):
        if getattr(self, "_rendering", False):
            return

        etype = event["event_type"]
        if etype == "key_down":
            key = event["key"]
            transition = _menu_transition(
                self._paused, getattr(self, "_menu_state", MENU_MAIN), key
            )
            action = transition.action
            if action == ACTION_PAUSE:
                self._set_paused(True)
            elif action == ACTION_RESUME:
                self._set_paused(False)
            elif action == ACTION_QUIT:
                self.canvas.close()
            elif action == ACTION_TOGGLE_FULLSCREEN:
                self._toggle_fullscreen()
            elif action == ACTION_OPEN_FILE:
                self._start_open_file()
            elif action == ACTION_OPEN_ICE_YES:
                self._finish_open_file(use_ice=True)
            elif action == ACTION_OPEN_ICE_NO:
                self._finish_open_file(use_ice=False)
            elif action == ACTION_RENDER_MENU:
                self._set_menu_state(transition.next_state or MENU_RENDER_QUALITY)
            elif action == ACTION_RENDER_BEHOLD:
                self._run_behold_render(transition.quality)
            elif action == ACTION_MENU_BACK:
                self._pending_open_path = None
                self._set_menu_state(transition.next_state or MENU_MAIN)
            elif action == ACTION_SCREENSHOT:
                self._save_screenshot()
            elif self._paused:
                return
            elif key in ("j", "J"):
                self.jitter = not self.jitter
            elif key in ("b", "B"):
                self.bird_enabled = not self.bird_enabled
            elif key in ("m", "M"):
                self.minimap_enabled = not self.minimap_enabled
            elif key == "Tab":
                self._capture_mouse(not self._captured)
            else:
                self._keys.add(key.lower() if len(key) == 1 else key)
        elif etype == "key_up":
            key = event["key"]
            self._keys.discard(key.lower() if len(key) == 1 else key)
        elif etype == "pointer_down":
            if self._paused and getattr(self, "_menu_state", MENU_MAIN) == MENU_MAIN:
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
        if getattr(self, "_rendering", False):
            return

        now = perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if not self._paused:
            self._move(min(dt, 0.1))
        self.position = _clamp_position_above_ocean(self.position)

        w, h = self.canvas.get_physical_size()
        self.renderer.write_uniforms(
            self.camera(), (w, h), jitter=self.jitter,
            sun_azimuth=self.sun_azimuth,
            sun_elevation=self.sun_elevation,
            frame_index=self._frame_index)
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
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
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
            if not self._title_flash_active(now):
                if self._paused:
                    self.canvas.set_title(self._menu_title(fps))
                else:
                    cam = self.camera()
                    self.canvas.set_title(
                        f"cloudyview  {fps:5.1f} fps  "
                        f"pos=({cam.position[0]:+.2f},{cam.position[1]:+.2f},"
                        f"{cam.position[2]:+.2f}) az={cam.azimuth:.0f} "
                        f"el={cam.elevation:.0f} speed={self.speed:.0f}m/s "
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
