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
    F3          cycle the corner stats readout (subtle/expanded/hidden)
    F           toggle fullscreen/windowed
    F12         save a PNG screenshot with render metadata
    ESC         pause menu (releases the mouse)

Pause menu:
    ESC / R     resume and recapture the mouse
    O           open the in-window .nc browser
    G           render current view in behold (then 1=min, 2=low,
                3=medium, 4=high, 5=max/overnight; ESC backs out)
    S           performance settings (tier, render scale, temporal smoothing)
    P           toggle the periodic (horizontally tiled) domain — on by
                default; turn off for subvolume cutouts that are not
                physically periodic
    F           toggle fullscreen/windowed
    Q           quit from the top-level pause menu

Menus, file picking, loading progress, errors, and behold progress are drawn
inside the wgpu window with Dear ImGui. The window title remains a compact
flight readout — fps, camera state in cv.Camera terms, and the cumulonimbus
realism gate bitfield (e.g. cb:1010) — for transcription into
witness/behold/soar render calls.
"""

from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
import shlex
from time import perf_counter

import numpy as np

from ..camera import Camera
from ..cloudfield import CloudField, load as load_cloud_field
from ..render_metadata import build_render_metadata, embed_metadata
from .engine import (
    APP_LIGHT_MARCH_LOD_DEGREES,
    APP_VIEW_STEP_LOD_DEGREES,
    DEFAULT_AMBIENT_STRENGTH,
    DEFAULT_AMBIENT_OCCLUSION_STRENGTH,
    DEFAULT_BOUNCE_DEPTH_ATTENUATION,
    DEFAULT_DEEP_SHADOW_MS_SUPPRESSION,
    DEFAULT_EXPOSURE,
    DEFAULT_GRADIENT_SHADING_STRENGTH,
    DEFAULT_SUN_AZIMUTH,
    DEFAULT_SUN_ELEVATION,
    InteractiveRenderer,
    QUALITY_PRESETS,
    camera_world_origin,
    choose_volume_fp16,
    periodic_march_cap_m,
    request_device,
)
from .fullscreen import (
    choose_fullscreen_monitor,
    fullscreen_video_mode,
    safe_windowed_bounds,
    video_mode_fields,
)
from .jobs import BackgroundJob
from .menu import (
    ACTION_MENU_BACK,
    ACTION_OPEN_FILE,
    ACTION_OPEN_ICE_NO,
    ACTION_OPEN_ICE_YES,
    ACTION_PAUSE,
    ACTION_QUIT,
    ACTION_RENDER_BEHOLD,
    ACTION_RENDER_MENU,
    ACTION_RESUME,
    ACTION_SCREENSHOT,
    ACTION_TOGGLE_FULLSCREEN,
    ACTION_TOGGLE_PERIODIC,
    ACTION_SETTINGS_MENU,
    ACTION_SELECT_TIER,
    BEHOLD_QUALITIES_BY_KEY,
    MENU_ERROR,
    MENU_FILE_BROWSER_ICE,
    MENU_FILE_BROWSER_LIQUID,
    MENU_MAIN,
    MENU_OPEN_ICE_PROMPT,
    MENU_RENDER_QUALITY,
    MENU_SETTINGS,
    FileEntry,
    control_action_for_key as _control_action_for_key,
    list_netcdf_entries,
    menu_transition as _menu_transition,
)

DEFAULT_SPEED = 60.0        # m/s, comfortable for the 25 km dev domain
MOUSE_SENS = 0.12           # degrees per pixel
SPEED_WHEEL_FACTOR = 1.25   # per wheel notch
# Five dominant ocean wavelengths (FIF outer scale, ocean_fif.DEFAULT_OUTER_SCALE_M
# = 10 m): below this the normal-mapped water reads wrong (no displacement
# geometry) — Thomas 2026-07-10.
OCEAN_FLOOR_MARGIN_M = 5.0 * 10.0
BEHOLD_UNAVAILABLE_MESSAGE = "behold rendering requires the full dev install"

CONTROL_SUMMARY = (
    "Controls: W/S forward/back, A/D strafe, Space up, LShift/C down, mouse look "
    "(Tab releases, click recaptures), scroll speed, "
    "1 gradient, 2 MS floor, 3 ambient AO, 4 bounce attenuation, "
    "J jitter toggle, B bird toggle, M minimap toggle, F3 stats readout, "
    "R record flight track (re-render with soar --render-track), "
    "F fullscreen/window, F12 screenshot, ESC pause menu; "
    "paused: ESC/R resume, O open in-window file browser, G behold render, "
    "S performance settings, "
    "P periodic domain toggle, "
    "F fullscreen/window, Q quit from the top-level menu"
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


def behold_rendering_available() -> bool:
    """Return whether the optional offline renderer can be offered in-menu."""
    try:
        return find_spec("mitsuba") is not None and find_spec("drjit") is not None
    except (ImportError, ValueError):
        return False


def _clamp_position_above_ocean(
    position, margin: float = OCEAN_FLOOR_MARGIN_M
) -> np.ndarray:
    """Return a copy of a world-space camera position held above z=0 ocean."""
    clamped = np.asarray(position, dtype=np.float64).copy()
    clamped[2] = max(clamped[2], margin)
    return clamped


def _wrap_position_horizontal(position, bmin, bmax) -> np.ndarray:
    """Wrap a world-space camera position into the domain in x/y.

    Flight over a periodic domain is endless: crossing a lateral face lands
    the camera on the opposite face (modulo the domain extent). z is left
    alone — the ocean-floor clamp owns the vertical bound.
    """
    wrapped = np.asarray(position, dtype=np.float64).copy()
    for axis in (0, 1):
        extent = float(bmax[axis] - bmin[axis])
        wrapped[axis] = bmin[axis] + (wrapped[axis] - bmin[axis]) % extent
    return wrapped


def _slab_exit_t(origin, direction, lo, hi, axes) -> float:
    """Forward distance at which a ray leaves the slab(s) of ``axes``."""
    t_exit = np.inf
    for axis in axes:
        d = float(direction[axis])
        if abs(d) < 1e-12:
            continue
        t0 = (float(lo[axis]) - float(origin[axis])) / d
        t1 = (float(hi[axis]) - float(origin[axis])) / d
        t_exit = min(t_exit, max(t0, t1))
    return t_exit


def view_spans_domain_edge(
    origin,
    camera: Camera,
    bmin,
    bmax,
    *,
    aspect: float,
) -> bool:
    """True when a periodic view would march past the domain's x/y walls.

    Casts the center and four frustum-corner rays from the (wrapped,
    in-domain) camera origin and checks whether any leaves the horizontal
    AABB before its periodic march cap or the z slab — i.e. while wrapped
    volume is still visible along it. Behold's Mitsuba volume is finite, so
    such a view renders differently there (used for the render-menu notice).
    """
    origin = np.asarray(origin, dtype=np.float64)
    forward, right, up = camera.basis()
    tan_half = float(np.tan(np.deg2rad(camera.fov) * 0.5))
    directions = [np.asarray(forward, dtype=np.float64)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            d = forward + sx * aspect * tan_half * right + sy * tan_half * up
            directions.append(d / np.linalg.norm(d))
    for direction in directions:
        t_horizontal = _slab_exit_t(origin, direction, bmin, bmax, (0, 1))
        t_vertical = _slab_exit_t(origin, direction, bmin, bmax, (2,))
        cap = periodic_march_cap_m(origin[2], direction, bmin, bmax)
        if t_horizontal < min(t_vertical, cap):
            return True
    return False


class FlyThroughApp:
    def __init__(self, field: CloudField, *, size=(1280, 720),
                 extinction_multiplier: float = 1.0,
                 max_fps: float = 120.0,
                 camera: Camera | None = None,
                 periodic: bool = True,
                 tier: str = "auto",
                 volume_fp16: bool | None = None):
        # Import here so offscreen use never needs glfw / a display.
        from rendercanvas.glfw import RenderCanvas, loop

        self._loop = loop

        self._extinction_multiplier = float(extinction_multiplier)
        self.periodic = bool(periodic)
        # None = auto: choose_volume_fp16 decides per loaded field (so an
        # O-menu reload of a gigaLES gets fp16 even if the session started
        # on a small subvolume).
        self.volume_fp16 = None if volume_fp16 is None else bool(volume_fp16)
        self._requested_tier = str(tier).lower()
        if self._requested_tier not in (*QUALITY_PRESETS, "auto"):
            raise ValueError(
                f"tier must be one of auto, {', '.join(QUALITY_PRESETS)}; "
                f"got {tier!r}."
            )
        self._tier_source = (
            "auto" if self._requested_tier == "auto" else "user"
        )
        self._auto_benchmark_ms = None
        self.sun_azimuth = DEFAULT_SUN_AZIMUTH
        self.sun_elevation = DEFAULT_SUN_ELEVATION

        device = request_device()
        self.renderer = self._create_renderer(field, device=device)

        # Camera state: world meters + met angles. Start at the default
        # witness viewpoint.
        self._reset_camera_to_default(camera)
        if self._requested_tier == "auto":
            self._run_startup_tier_benchmark(size)

        # The window is created only after the heavy startup work (volume
        # upload, FIF ocean, shader compiles, tier benchmark). A visible
        # window that pumps no events for those seconds is what made GNOME
        # pop "not responding" over a perfectly healthy launch; now the
        # window appearing means the app is ready to fly.
        # "continuous" honors max_fps; uncapped ("fastest", vsync off)
        # burns a full GPU rendering ~4000 fps nobody can see.
        self.canvas = RenderCanvas(
            title="cloudyview", size=size, update_mode="continuous",
            max_fps=max_fps, vsync=True,
        )
        self._ensure_resizable()
        self.context = self.canvas.get_context("wgpu")
        self.format = self.context.get_preferred_format(device.adapter)
        self.context.configure(device=device, format=self.format)

        self.speed = DEFAULT_SPEED
        self.jitter = True
        self.cb_enabled = [True, True, True, True]
        self.distance_lod = True   # L toggles (A/B vs exact legacy marching)
        self._track_recording = False   # R toggles (see soar/track.py)
        self._track_samples: list[list[float]] = []
        self._track_t0 = 0.0
        self.bird_enabled = True
        self.minimap_enabled = True
        self._keys = set()
        self._closing = False
        self._last_pointer = None   # None -> ignore next move (capture jump guard)
        self._captured = False
        self._paused = False
        self._menu_state = MENU_MAIN
        self._pending_open_path = None
        self._file_browser_dir = (
            Path(field.source).expanduser().parent
            if field.source else Path.cwd()
        )
        self._last_file_dir = self._file_browser_dir
        self._file_browser_error = None
        self._loading_job = None
        self._behold_job = None
        self._rendering = False
        self._error_message = None
        self._imgui = None
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
        self._fps_value = None
        self._fps_frame_ms = None
        self._stats_mode = "subtle"   # subtle -> expanded -> hidden (F3)
        self._last_quality_camera_signature = None
        self._camera_moving_for_quality(self.camera())

        self.canvas.add_event_handler(self._on_event,
                                      "key_down", "key_up",
                                      "pointer_down", "pointer_up",
                                      "pointer_move", "wheel", "char")
        self.canvas.request_draw(self._draw)

    # ------------------------------------------------------------------

    def _create_renderer(self, field: CloudField, *, device=None, previous=None):
        """Create the field-resident renderer, reusing the app GPU device."""
        volume_fp16 = choose_volume_fp16(field.lwc.size, self.volume_fp16)
        if volume_fp16 and self.volume_fp16 is None:
            print(
                f"soar: {field.lwc.size / 1e6:.0f}M-voxel volume -> fp16 "
                "texture (auto; --fp32-volume forces full precision)"
            )
        kwargs = {
            "extinction_multiplier": self._extinction_multiplier,
            "periodic": self.periodic,
            "quality_tier": (
                previous.quality_tier
                if previous is not None
                else (
                    "high"
                    if self._requested_tier == "auto"
                    else self._requested_tier
                )
            ),
            "volume_fp16": volume_fp16,
            "device": device,
        }
        if previous is not None:
            kwargs.update({
                "ocean_enabled": previous.ocean_enabled,
                "ocean_z": previous.ocean_z,
                "ocean_reflectance": previous.ocean_reflectance,
                "fif_normals": previous.ocean_fif_normals,
                "motion_blend_alpha": previous.motion_blend_alpha,
            })
        renderer = InteractiveRenderer(field, **kwargs)
        if previous is not None and previous.quality_is_custom:
            renderer.set_render_scale(previous.flight_render_scale)
        return renderer

    def _reset_camera_to_default(self, camera: Camera | None = None) -> None:
        """Reset the app camera to the shared default or supplied ``cv.Camera``."""
        cam0 = camera or Camera()
        self.position = camera_world_origin(
            cam0, self.renderer.bmin, self.renderer.bmax)
        self.position = _clamp_position_above_ocean(self.position)
        self.azimuth = cam0.azimuth
        self.elevation = cam0.elevation
        self.fov = cam0.fov

    def _run_startup_tier_benchmark(self, size) -> None:
        """Measure one steady frame per moving preset and select for 60 fps.

        Runs before the window exists (see __init__), so the measurement
        target is the requested window size rather than the physical
        framebuffer; under HiDPI scaling the two can differ, which only
        shifts the tier cutoffs slightly.
        """
        size = tuple(int(v) for v in size)
        timings = {}
        timing_source = None
        camera = self.camera()
        chosen = "potato"
        target_ms = 1000.0 / 60.0
        # Ascend from the cheapest tier. Once one misses 60 fps, every more
        # expensive preset is intentionally skipped; this bounds startup on
        # the weak machines auto-selection exists to help.
        for name in reversed(tuple(QUALITY_PRESETS)):
            self.renderer.set_quality_tier(name, camera_moving=True)
            result = self.renderer.benchmark(
                camera,
                size=size,
                n_warmup=1,
                n_frames=1,
                azimuth_step=0.4,
            )
            timing_key = (
                "gpu_ms_mean"
                if result.get("timestamps_used")
                else "wall_ms_mean"
            )
            timing_source = "GPU timestamps" if result.get(
                "timestamps_used"
            ) else "wall clock"
            timings[name] = float(result[timing_key])
            if timings[name] > target_ms:
                break
            chosen = name
        self.renderer.set_quality_tier(chosen, camera_moving=False)
        self.renderer.reset_accumulation()
        self._auto_benchmark_ms = timings
        table = ", ".join(
            f"{QUALITY_PRESETS[name].label}: {timings[name]:.2f} ms"
            for name in reversed(tuple(QUALITY_PRESETS))
            if name in timings
        )
        print(
            f"soar auto-tier ({size[0]}x{size[1]}, {timing_source}): "
            f"{table}; chose {chosen}"
        )

    def _select_quality_tier(self, name: str) -> None:
        """Apply an explicit session-persistent menu choice."""
        self.renderer.set_quality_tier(
            name, camera_moving=getattr(self.renderer, "_camera_moving", False)
        )
        self._tier_source = "user"
        self.canvas.request_draw()

    def _set_render_scale(self, value: float) -> None:
        self.renderer.set_render_scale(value)
        self._tier_source = "user"
        self.canvas.request_draw()

    def _set_motion_blend_alpha(self, value: float) -> None:
        value = float(value)
        if not 0.3 <= value <= 0.9:
            raise ValueError(
                f"motion temporal smoothing must be in [0.3, 0.9]; got {value}."
            )
        self.renderer.motion_blend_alpha = value
        self._tier_source = "user"
        self.canvas.request_draw()

    def _camera_moving_for_quality(self, camera: Camera) -> bool:
        signature = (
            *tuple(float(v) for v in camera.position),
            float(camera.azimuth),
            float(camera.elevation),
            float(camera.fov),
        )
        previous = self._last_quality_camera_signature
        self._last_quality_camera_signature = signature
        return previous is not None and signature != previous

    def _install_field(self, field: CloudField) -> None:
        """Swap to a freshly loaded field and rebuild field-specific GPU state."""
        previous = self.renderer
        self.renderer = self._create_renderer(
            field, device=previous.device, previous=previous
        )
        self._reset_camera_to_default()
        self._frame_index = 0
        self.renderer.reset_accumulation()
        self._last_quality_camera_signature = None
        self._camera_moving_for_quality(self.camera())

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
        # Runtime diagnostics live in the toggleable corner readout, not the
        # native window chrome. Keep the optional argument for compatibility.
        return "cloudyview paused"

    def _open_ice_title(self) -> str:
        filename = (
            Path(self._pending_open_path).name if self._pending_open_path else ""
        )
        return f"cloudyview open file  {filename}"

    def _render_quality_title(self) -> str:
        return "cloudyview behold quality"

    def _menu_title(self, fps: float | None = None) -> str:
        state = getattr(self, "_menu_state", MENU_MAIN)
        if state == MENU_OPEN_ICE_PROMPT:
            return self._open_ice_title()
        if state in (MENU_FILE_BROWSER_LIQUID, MENU_FILE_BROWSER_ICE):
            return "cloudyview file browser"
        if state == MENU_RENDER_QUALITY:
            return self._render_quality_title()
        if state == MENU_SETTINGS:
            return "cloudyview settings"
        if state == MENU_ERROR:
            return "cloudyview error"
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
            # Distance LOD (2026-07-17 perf pass): app opts in; the library
            # default stays 0.0 (exact legacy) until the goldens move.
            # L toggles it live for A/B flying.
            "light_march_lod_degrees": (
                APP_LIGHT_MARCH_LOD_DEGREES if self.distance_lod else 0.0
            ),
            "view_step_lod_degrees": (
                APP_VIEW_STEP_LOD_DEGREES if self.distance_lod else 0.0
            ),
        }

    def _cycle_stats_mode(self) -> None:
        """F3 cycles the corner stats readout: subtle -> expanded -> hidden."""
        order = ("subtle", "expanded", "hidden")
        current = getattr(self, "_stats_mode", "subtle")
        index = order.index(current) if current in order else 0
        self._stats_mode = order[(index + 1) % len(order)]

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
            self._file_browser_error = None
            self._error_message = None
            self._capture_mouse(True)
            self._last_time = perf_counter()
        self.canvas.request_draw()

    def _current_monitor(self):
        import glfw

        window = self.canvas._window
        if window is None:
            raise RuntimeError("Cannot choose a fullscreen monitor: window closed.")
        monitor = choose_fullscreen_monitor(glfw, window)
        if monitor is None:
            raise RuntimeError("Cannot enter fullscreen: GLFW found no monitors.")
        return monitor

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
            self._windowed_bounds = safe_windowed_bounds(glfw, window)
            monitor = self._current_monitor()
            monitor, mode = fullscreen_video_mode(glfw, monitor)
            glfw.set_window_monitor(
                window, monitor, 0, 0,
                *video_mode_fields(mode),
            )
            self._fullscreen = True

        self.canvas._determine_size()
        if self._paused:
            self.canvas.set_title(self._menu_title())
        self.canvas.request_draw()

    def _close_after_frame(self) -> None:
        """Close on the event loop, never mid-frame (surface-destroy panic)."""
        loop = getattr(self, "_loop", None)
        if loop is not None:
            loop.call_soon(self.canvas.close)
        else:
            self.canvas.close()

    def _try_toggle_fullscreen(self) -> None:
        try:
            self._toggle_fullscreen()
        except Exception as e:
            self._show_error(f"Fullscreen failed: {e}")

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

    def _start_open_file(self) -> None:
        self._pending_open_path = None
        self._file_browser_error = None
        self._set_file_browser_dir(getattr(self, "_last_file_dir", Path.cwd()))
        self._set_menu_state(MENU_FILE_BROWSER_LIQUID)

    def _set_file_browser_dir(self, directory: str | Path) -> None:
        try:
            path = Path(directory).expanduser().resolve()
            if not path.is_dir():
                path = path.parent
            self._file_browser_dir = path
            self._last_file_dir = path
            self._file_browser_error = None
        except Exception as e:
            self._file_browser_error = str(e)

    def _select_browser_path(self, path: str | Path) -> None:
        selected = Path(path).expanduser().resolve()
        self._last_file_dir = selected.parent
        if self._menu_state == MENU_FILE_BROWSER_LIQUID:
            self._pending_open_path = str(selected)
            self._set_menu_state(MENU_OPEN_ICE_PROMPT)
        elif self._menu_state == MENU_FILE_BROWSER_ICE:
            self._start_loading_file(
                self._pending_open_path, str(selected)
            )

    def _finish_open_file(self, *, use_ice: bool) -> None:
        liquid_path = self._pending_open_path
        if not liquid_path:
            self._set_menu_state(MENU_MAIN)
            return

        if use_ice:
            self._file_browser_error = None
            self._set_file_browser_dir(Path(liquid_path).parent)
            self._set_menu_state(MENU_FILE_BROWSER_ICE)
            return

        self._start_loading_file(liquid_path, None)

    def _start_loading_file(
        self, liquid_path: str | Path | None, ice_path: str | Path | None
    ) -> None:
        if not liquid_path:
            self._show_error("Open file failed: no liquid NetCDF was selected.")
            return
        if self._loading_job is not None or self._behold_job is not None:
            self._show_error("Another operation is already running.")
            return

        liquid_path = str(liquid_path)
        ice_path = str(ice_path) if ice_path else None
        previous = self.renderer
        device = previous.device
        filename = Path(liquid_path).name

        def target(report):
            def stage(stage_name: str) -> None:
                report(stage_name)

            field = load_cloud_field(
                liquid_path, ice=ice_path, stage_callback=stage
            )
            report("building extinction")
            renderer = self._create_renderer(
                field, device=device, previous=previous
            )
            report("uploading texture")
            return {
                "field": field,
                "renderer": renderer,
                "liquid_path": liquid_path,
                "ice_path": ice_path,
            }

        self._pending_open_path = None
        self._file_browser_error = None
        self._set_menu_state(MENU_MAIN)
        self.canvas.set_title(f"cloudyview loading {filename}")
        self._loading_job = BackgroundJob(
            kind="loading",
            filename=filename,
            target=target,
            initial_stage="queued",
        )
        self._loading_job.start()
        self.canvas.request_draw()

    def _install_loaded_renderer(self, result: dict) -> None:
        self.renderer = result["renderer"]
        self._reset_camera_to_default()
        self._frame_index = 0
        self.renderer.reset_accumulation()
        self._last_quality_camera_signature = None
        self._camera_moving_for_quality(self.camera())
        print(f"Loaded {self.renderer.field}")
        self._set_paused(False)
        self._flash_title(
            f"cloudyview loaded {Path(result['liquid_path']).name}", seconds=3.0
        )

    def _show_error(self, message: str) -> None:
        self._error_message = str(message)
        self._loading_job = None
        self._behold_job = None
        self._rendering = False
        self._set_paused(True)
        self._set_menu_state(MENU_ERROR)
        print(f"cloudyview error: {self._error_message}")
        self.canvas.request_draw()

    def _active_job(self):
        for job in (self._loading_job, self._behold_job):
            if job is None:
                continue
            snapshot = job.pump()
            if not snapshot.done:
                return job
        return None

    def _active_job_snapshot(self):
        job = self._active_job()
        return None if job is None else job.snapshot()

    def _pump_jobs(self) -> None:
        if self._loading_job is not None:
            snapshot = self._loading_job.pump()
            if snapshot.done:
                job = self._loading_job
                self._loading_job = None
                if snapshot.error:
                    self._show_error(f"Open file failed: {snapshot.error}")
                else:
                    self._install_loaded_renderer(snapshot.result)
                job.join(0.0)

        if self._behold_job is not None:
            snapshot = self._behold_job.pump()
            if snapshot.done:
                job = self._behold_job
                self._behold_job = None
                self._rendering = False
                if snapshot.error:
                    self._show_error(snapshot.error)
                else:
                    print(f"Behold render saved to {snapshot.result}")
                    self._set_paused(False)
                    self._flash_title(
                        f"cloudyview behold saved {snapshot.result}", seconds=5.0
                    )
                job.join(0.0)

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
        parts.extend(["--tier", self.renderer.quality_tier])
        if self.renderer.volume_fp16:
            parts.append("--fp16-volume")
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

    def _toggle_track_recording(self) -> None:
        if getattr(self, "_track_recording", False):
            self._finish_track_recording()
            return
        self._track_samples = []
        self._track_t0 = perf_counter()
        self._track_recording = True
        self._flash_title("cloudyview recording track — R to stop",
                          seconds=3.0)

    def _finish_track_recording(self) -> None:
        from .track import save_track

        self._track_recording = False
        samples = self._track_samples
        self._track_samples = []
        if len(samples) < 2:
            self._flash_title("cloudyview track discarded (too short)",
                              seconds=3.0)
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / f"cloudyview_track_{stamp}.json"
        header = build_render_metadata(
            self.renderer.field,
            self.camera(),
            sun_azimuth=self.sun_azimuth,
            sun_elevation=self.sun_elevation,
            renderer="soar",
            quality=self.renderer.quality_tier,
            reproduction_command=f"soar --render-track {path.name}",
            render_options={
                **self._cb_strength_kwargs(),
                "periodic": self.renderer.periodic,
                "extinction_multiplier": self._extinction_multiplier,
                "volume_fp16": self.renderer.volume_fp16,
            },
        )
        save_track(path, header, samples)
        duration = samples[-1][0]
        self._flash_title(
            f"cloudyview track saved — {len(samples)} samples, "
            f"{duration:.0f}s",
            seconds=4.0,
        )
        print(
            f"Saved track {path} ({len(samples)} samples, {duration:.1f}s)\n"
            f"Render with: soar --render-track {path}"
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
                "tier": renderer.quality_tier,
                "render_scale": renderer.render_scale,
                "step_factor": renderer.step_factor,
                "max_light_steps": renderer.max_light_steps,
                "volume_fp16": renderer.volume_fp16,
            },
        )
        self._write_png_with_metadata(image, path, metadata)
        print(f"Screenshot saved to {path}")
        self._flash_title(f"cloudyview screenshot saved {path}", seconds=4.0)

    def _run_behold_render(self, quality: str) -> None:
        if self._loading_job is not None or self._behold_job is not None:
            self._show_error("Another operation is already running.")
            return
        camera = self.camera()
        path = self._timestamped_png_path(f"cloudyview_behold_{quality}")

        def target(report):
            def on_progress(progress: dict) -> None:
                eta = progress.get("eta")
                if eta is None and progress.get("taken_spp"):
                    elapsed = progress.get("elapsed", 0.0)
                    taken = progress["taken_spp"]
                    total = progress.get("spp_total", taken)
                    eta = elapsed * max(0, total - taken) / taken
                report(
                    "rendering",
                    percent=progress.get("percent", 0.0),
                    eta=eta,
                    note="cannot cancel once started",
                )

            try:
                from .. import behold as behold_render
            except ImportError as e:
                raise RuntimeError(
                    "cloudyview behold requires Mitsuba; install the "
                    "radiative-transfer extra, e.g. uv sync --extra "
                    f"radiative-transfer ({e})"
                ) from e

            try:
                image = behold_render(
                    self.renderer.field,
                    camera=camera,
                    quality=quality,
                    gpu=True,
                    sun_azimuth=self.sun_azimuth,
                    sun_elevation=self.sun_elevation,
                    progress_callback=on_progress,
                )
            except Exception as e:
                detail = str(e)
                if any(token in detail.lower() for token in ("mitsuba", "drjit")):
                    raise RuntimeError(
                        "cloudyview behold requires Mitsuba; install the "
                        "radiative-transfer extra, e.g. uv sync --extra "
                        f"radiative-transfer ({e})"
                    ) from e
                raise RuntimeError(f"cloudyview behold failed: {e}") from e

            metadata = self._metadata(
                camera,
                renderer="behold",
                quality=quality,
                reproduction_command=self._behold_reproduction_command(
                    camera, quality
                ),
            )
            self._write_png_with_metadata(image, path, metadata)
            return path

        self._rendering = True
        self._set_menu_state(MENU_MAIN)
        self.canvas.set_title(f"cloudyview behold {quality}")
        self._behold_job = BackgroundJob(
            kind=f"behold {quality}",
            filename=path.name,
            target=target,
            initial_stage="starting render",
            note="cannot cancel once started",
        )
        self._behold_job.start()
        self.canvas.request_draw()

    def _on_event(self, event):
        self._pump_jobs()
        active_job = self._active_job()
        if (
            getattr(self, "_paused", False) or active_job is not None
        ) and self._imgui is not None:
            self._imgui.handle_event(event)

        if active_job is not None:
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
                self._closing = True
                self._close_after_frame()
            elif action == ACTION_TOGGLE_FULLSCREEN:
                self._try_toggle_fullscreen()
            elif action == ACTION_OPEN_FILE:
                self._start_open_file()
            elif action == ACTION_OPEN_ICE_YES:
                self._finish_open_file(use_ice=True)
            elif action == ACTION_OPEN_ICE_NO:
                self._finish_open_file(use_ice=False)
            elif action == ACTION_RENDER_MENU:
                if behold_rendering_available():
                    self._set_menu_state(
                        transition.next_state or MENU_RENDER_QUALITY
                    )
                else:
                    self._flash_title(BEHOLD_UNAVAILABLE_MESSAGE, seconds=4.0)
            elif action == ACTION_SETTINGS_MENU:
                self._set_menu_state(transition.next_state or MENU_SETTINGS)
            elif action == ACTION_SELECT_TIER:
                self._select_quality_tier(transition.tier)
            elif action == ACTION_RENDER_BEHOLD:
                self._run_behold_render(transition.quality)
            elif action == ACTION_MENU_BACK:
                if transition.next_state == MENU_OPEN_ICE_PROMPT:
                    self._set_menu_state(MENU_OPEN_ICE_PROMPT)
                else:
                    self._pending_open_path = None
                    self._error_message = None
                    self._set_menu_state(transition.next_state or MENU_MAIN)
            elif action == ACTION_TOGGLE_PERIODIC:
                self._toggle_periodic()
            elif action == ACTION_SCREENSHOT:
                self._save_screenshot()
            elif key in ("r", "R"):
                # Only reached unpaused: while paused the menu table above
                # consumes R as resume (documented), so record start/stop
                # is a flying-only control.
                self._toggle_track_recording()
            elif self._paused:
                return
            elif key in ("j", "J"):
                self.jitter = not self.jitter
            elif key in ("b", "B"):
                self.bird_enabled = not self.bird_enabled
            elif key in ("m", "M"):
                self.minimap_enabled = not self.minimap_enabled
            elif key in ("l", "L"):
                self.distance_lod = not self.distance_lod
            elif key == "F3":
                self._cycle_stats_mode()
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
                return
            if not self._captured:
                self._capture_mouse(True)   # click back in to recapture
        elif etype == "char":
            return
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
        self.position = self._constrain_position(self.position)

    def _constrain_position(self, position) -> np.ndarray:
        """Ocean-floor clamp plus, when periodic, the horizontal x/y wrap."""
        constrained = _clamp_position_above_ocean(position)
        if self.periodic:
            constrained = _wrap_position_horizontal(
                constrained, self.renderer.bmin, self.renderer.bmax
            )
        return constrained

    def _toggle_periodic(self) -> None:
        """Flip horizontal domain tiling; the renderer rewrites its border."""
        self.periodic = not self.periodic
        self.renderer.set_periodic(self.periodic)
        self.position = self._constrain_position(self.position)
        self._flash_title(
            "cloudyview periodic domain "
            f"{'on' if self.periodic else 'off'}",
            seconds=2.5,
        )
        self.canvas.request_draw()

    def _view_spans_domain_edge(self) -> bool:
        w, h = self.canvas.get_physical_size()
        return view_spans_domain_edge(
            self.position,
            self.camera(),
            self.renderer.bmin,
            self.renderer.bmax,
            aspect=w / h,
        )

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

    def _ensure_imgui(self) -> None:
        if self._imgui is None:
            from .imgui_layer import SoarImguiLayer

            self._imgui = SoarImguiLayer(
                device=self.renderer.device,
                target_format=self.format,
                canvas=self.canvas,
            )

    @staticmethod
    def _imgui_flags(imgui, enum_name: str, *names: str) -> int:
        enum_cls = getattr(imgui, enum_name, None)
        if enum_cls is None:
            return 0
        value = 0
        for name in names:
            if hasattr(enum_cls, name):
                enum_value = getattr(enum_cls, name)
                value |= int(getattr(enum_value, "value", enum_value))
        return value

    @staticmethod
    def _imgui_cond_always(imgui) -> int:
        cond = getattr(imgui, "Cond_", None)
        if cond is not None and hasattr(cond, "always"):
            return getattr(cond, "always")
        return 0

    @property
    def _theme(self):
        """Theme attached to the lazily created ImGui layer."""
        self._ensure_imgui()
        return self._imgui.theme

    def _begin_imgui_window(
        self, imgui, name: str, width: float, height: float = 0.0
    ) -> None:
        """Begin a centered, fixed glass panel (height 0 = auto-fit)."""
        logical_w, logical_h = self.canvas.get_logical_size()
        cond = self._imgui_cond_always(imgui)
        flag_names = [
            "no_resize",
            "no_collapse",
            "no_saved_settings",
            "no_move",
            "no_title_bar",
            "no_scrollbar",
        ]
        if height <= 0.0:
            flag_names.append("always_auto_resize")
        flags = self._imgui_flags(
            imgui,
            "WindowFlags_",
            *flag_names,
        )
        imgui.set_next_window_pos(
            (logical_w * 0.5, logical_h * 0.46), cond, (0.5, 0.5)
        )
        imgui.set_next_window_size((width, height), cond)
        imgui.begin(f"##{name}", None, flags)

    @staticmethod
    def _end_imgui_window(imgui) -> None:
        imgui.end()

    def _draw_imgui(self, command_encoder, target_view) -> None:
        self._ensure_imgui()
        self._imgui.encode(
            command_encoder, target_view, self._draw_imgui_contents
        )

    def _draw_imgui_contents(self, imgui) -> None:
        snapshot = self._active_job_snapshot()
        if snapshot is not None:
            self._draw_job_overlay(imgui, snapshot)
            return

        if not getattr(self, "_paused", False):
            self._draw_stats_readout(imgui)
            return

        state = getattr(self, "_menu_state", MENU_MAIN)
        if state == MENU_RENDER_QUALITY:
            self._draw_quality_menu(imgui)
        elif state == MENU_SETTINGS:
            self._draw_settings_menu(imgui)
        elif state == MENU_OPEN_ICE_PROMPT:
            self._draw_ice_prompt(imgui)
        elif state in (MENU_FILE_BROWSER_LIQUID, MENU_FILE_BROWSER_ICE):
            self._draw_file_browser(imgui, state)
        elif state == MENU_ERROR:
            self._draw_error_dialog(imgui)
        else:
            self._draw_main_menu(imgui)

    def _field_display_name(self) -> str:
        renderer = getattr(self, "renderer", None)
        field = getattr(renderer, "field", None)
        source = getattr(field, "source", None)
        return Path(source).name if source else "in-memory field"

    @staticmethod
    def _truncate_middle(text: str, max_chars: int = 32) -> str:
        if len(text) <= max_chars:
            return text
        keep = max_chars - 1
        head = (keep + 1) // 2
        tail = keep - head
        return f"{text[:head]}…{text[-tail:]}"

    def _draw_main_menu(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "main_menu", 420.0)
        try:
            theme.header("cloudyview", "Paused")
            if theme.menu_button("Resume", "ESC"):
                self._set_paused(False)
            if theme.menu_button("Open file...", "O"):
                self._start_open_file()
            behold_available = behold_rendering_available()
            if theme.menu_button(
                "Render in behold...",
                "G" if behold_available else None,
                disabled=not behold_available,
            ):
                self._set_menu_state(MENU_RENDER_QUALITY)
            if not behold_available:
                theme.caption(BEHOLD_UNAVAILABLE_MESSAGE, wrapped=True)
            if theme.menu_button("Settings...", "S"):
                self._set_menu_state(MENU_SETTINGS)
            periodic_label = (
                "Periodic domain: on"
                if self.periodic
                else "Periodic domain: off"
            )
            if theme.menu_button(periodic_label, "P"):
                self._toggle_periodic()
            label = (
                "Exit fullscreen"
                if getattr(self, "_fullscreen", False)
                else "Enter fullscreen"
            )
            if theme.menu_button(label, "F"):
                self._try_toggle_fullscreen()
            if theme.menu_button("Quit", "Q"):
                self._closing = True
                self._close_after_frame()
            imgui.dummy((1.0, 8.0))
            theme.mono_text(
                self._truncate_middle(self._field_display_name(), 44),
                size=13.0,
            )
            theme.hint_row((
                ("WASD", "fly"),
                ("mouse", "look"),
                ("scroll", "speed"),
                ("F3", "stats"),
            ))
        finally:
            self._end_imgui_window(imgui)

    def _draw_quality_menu(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "quality_menu", 420.0)
        try:
            theme.header("render in behold", "Quality")
            if not behold_rendering_available():
                theme.caption(BEHOLD_UNAVAILABLE_MESSAGE, wrapped=True)
                imgui.dummy((1.0, 4.0))
                if theme.menu_button("Back", "ESC", height=38.0):
                    self._set_menu_state(MENU_MAIN)
                return
            if self.periodic and self._view_spans_domain_edge():
                theme.caption(
                    "view spans domain edge — behold will differ",
                    wrapped=True,
                )
                imgui.dummy((1.0, 4.0))
            for label, hint, quality, note in (
                ("Min", "1", "min", "fast preview"),
                ("Low", "2", "low", "draft"),
                ("Medium", "3", "medium", "balanced"),
                ("High", "4", "high", "~1 h"),
                ("Max", "5", "max", "overnight"),
            ):
                if theme.menu_button(label, hint, sublabel=note):
                    self._run_behold_render(quality)
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_settings_menu(self, imgui) -> None:
        from .theme import TEXT_FAINT

        theme = self._theme
        renderer = self.renderer
        self._begin_imgui_window(imgui, "settings_menu", 520.0)
        try:
            theme.header("performance", "Settings")
            for hint, name in (("1", "high"), ("2", "medium"),
                               ("3", "low"), ("4", "potato")):
                preset = QUALITY_PRESETS[name]
                selected = renderer.quality_tier == name
                if selected and self._tier_source == "auto":
                    label = f"{preset.label} (auto)"
                    right_text = None
                elif selected and renderer.quality_is_custom:
                    label = preset.label
                    right_text = "custom"
                elif selected:
                    label = preset.label
                    right_text = "selected"
                else:
                    label = preset.label
                    right_text = None
                if theme.menu_button(
                    label, hint, right_text=right_text
                ):
                    self._select_quality_tier(name)

            imgui.dummy((1.0, 8.0))
            theme.caption("Render scale")
            changed, scale = imgui.slider_float(
                "##render_scale",
                renderer.flight_render_scale,
                0.25,
                1.0,
                "%.2fx",
            )
            if changed:
                self._set_render_scale(scale)

            theme.caption("Motion temporal smoothing")
            changed, alpha = imgui.slider_float(
                "##motion_blend_alpha",
                renderer.motion_blend_alpha,
                0.3,
                0.9,
                "%.2f",
            )
            if changed:
                self._set_motion_blend_alpha(alpha)

            imgui.dummy((1.0, 6.0))
            fp_label = (
                "fp16 volume: on" if renderer.volume_fp16
                else "fp16 volume: off"
            )
            theme.menu_button(
                fp_label,
                None,
                disabled=True,
                text_color=TEXT_FAINT,
            )
            theme.caption(
                "Constructor-only to avoid retaining a second full volume; "
                "use --fp16-volume at load.",
                wrapped=True,
            )
            if renderer.quality_tier == "potato":
                theme.caption(
                    "Potato switches to exact High sampling when the camera "
                    "stops, then accumulates a smooth still.",
                    wrapped=True,
                )
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_ice_prompt(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "ice_prompt", 460.0)
        try:
            filename = (
                Path(self._pending_open_path).name
                if self._pending_open_path else "selected file"
            )
            theme.header("open file", "Ice phase?")
            theme.mono_text(filename, size=13.0)
            imgui.dummy((1.0, 6.0))
            if theme.menu_button("Yes, pick ice file", "Y"):
                self._finish_open_file(use_ice=True)
            if theme.menu_button("No ice", "N"):
                self._finish_open_file(use_ice=False)
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._pending_open_path = None
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_file_browser(self, imgui, state: str) -> None:
        theme = self._theme
        title = (
            "Open cloud file"
            if state == MENU_FILE_BROWSER_LIQUID
            else "Open ice file"
        )
        self._begin_imgui_window(imgui, "file_browser", 720.0, 560.0)
        try:
            current_dir = Path(getattr(self, "_file_browser_dir", Path.cwd()))
            theme.header("netcdf browser", title)
            if theme.menu_button("Up", None, width=110.0, height=34.0):
                self._set_file_browser_dir(current_dir.parent)
            imgui.same_line()
            if theme.menu_button("Back", "ESC", width=110.0, height=34.0):
                if state == MENU_FILE_BROWSER_ICE:
                    self._set_menu_state(MENU_OPEN_ICE_PROMPT)
                else:
                    self._pending_open_path = None
                    self._set_menu_state(MENU_MAIN)
                return
            imgui.same_line()
            theme.mono_text(str(current_dir), size=13.0)

            if self._file_browser_error:
                theme.caption(self._file_browser_error, wrapped=True)

            try:
                entries = list_netcdf_entries(current_dir)
                self._file_browser_error = None
            except Exception as e:
                entries = []
                self._file_browser_error = str(e)
                theme.caption(self._file_browser_error, wrapped=True)

            child_flags = self._imgui_flags(
                imgui, "ChildFlags_", "always_use_window_padding"
            )
            imgui.push_style_var(
                imgui.StyleVar_.window_padding, (12.0, 12.0)
            )
            imgui.begin_child("files", (0.0, 0.0), child_flags)
            try:
                if not entries:
                    theme.caption("No folders or .nc files here.")
                for entry in entries:
                    self._draw_file_entry(imgui, entry)
            finally:
                imgui.end_child()
                imgui.pop_style_var()
        finally:
            self._end_imgui_window(imgui)

    def _draw_file_entry(self, imgui, entry: FileEntry) -> None:
        from .theme import TEXT_MUTED

        theme = self._theme
        if entry.is_dir:
            clicked = theme.menu_button(
                f"{entry.name}/", None, height=36.0, text_color=TEXT_MUTED
            )
        else:
            clicked = theme.menu_button(
                entry.name, None, height=36.0, right_text=entry.display_size
            )
        if clicked:
            if entry.is_dir:
                self._set_file_browser_dir(entry.path)
            else:
                self._select_browser_path(entry.path)

    def _draw_error_dialog(self, imgui) -> None:
        from .theme import ERROR

        theme = self._theme
        self._begin_imgui_window(imgui, "error_dialog", 520.0)
        try:
            theme.header("error", "Something failed", kicker_color=ERROR)
            theme.body_text(
                self._error_message or "Unknown soar error.", wrapped=True
            )
            imgui.dummy((1.0, 6.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._error_message = None
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_job_overlay(self, imgui, snapshot) -> None:
        from .theme import TEXT_FAINT

        theme = self._theme
        loading = snapshot.kind == "loading"
        self._begin_imgui_window(imgui, "job_overlay", 520.0)
        try:
            if loading:
                theme.header("loading", self._truncate_middle(snapshot.filename))
            else:
                quality = snapshot.kind.split(" ", 1)[-1]
                theme.header("rendering in behold", quality.capitalize())
            status = f"{snapshot.stage} · {self._fmt_eta(snapshot.elapsed)}"
            theme.mono_text(status, size=13.0)
            imgui.dummy((1.0, 2.0))
            percent = snapshot.percent
            theme.progress_bar(None if percent is None else percent / 100.0)
            if percent is not None:
                theme.push_font(theme.font_mono, 13.0)
                theme.body_text(f"{float(percent):3.0f}%", TEXT_FAINT)
                if snapshot.eta is not None:
                    eta_text = f"ETA {self._fmt_eta(snapshot.eta)}"
                    text_w = imgui.calc_text_size(eta_text).x
                    imgui.same_line(
                        imgui.get_window_width()
                        - imgui.get_style().window_padding.x - text_w
                    )
                    theme.body_text(eta_text, TEXT_FAINT)
                theme.pop_font()
            if not loading:
                imgui.dummy((1.0, 2.0))
                theme.mono_text(snapshot.filename, TEXT_FAINT, size=12.0)
            if snapshot.note:
                theme.caption(snapshot.note, TEXT_FAINT)
        finally:
            self._end_imgui_window(imgui)

    def _draw_stats_readout(self, imgui) -> None:
        """Corner fps readout: subtle pill, or an expanded diagnostics card."""
        from .theme import PANEL_BG, TEXT_FAINT, TEXT_MUTED

        mode = getattr(self, "_stats_mode", "subtle")
        fps = getattr(self, "_fps_value", None)
        if mode == "hidden" or fps is None:
            return
        theme = self._theme
        frame_ms = getattr(self, "_fps_frame_ms", None) or (1000.0 / max(fps, 1e-6))

        logical_w, logical_h = self.canvas.get_logical_size()
        cond = self._imgui_cond_always(imgui)
        flags = self._imgui_flags(
            imgui,
            "WindowFlags_",
            "no_decoration",
            "no_move",
            "no_saved_settings",
            "always_auto_resize",
            "no_focus_on_appearing",
            "no_nav",
            "no_inputs",
        )
        imgui.set_next_window_pos(
            (logical_w - 14.0, logical_h - 12.0), cond, (1.0, 1.0)
        )
        # 0.38 washed out over bright cumulus (Thomas, 2026-07-17: "the fps
        # is hard to read when certain cloud colors are behind it").
        alpha = 0.62 if mode == "subtle" else 0.80
        imgui.push_style_color(
            imgui.Col_.window_bg, (*PANEL_BG[:3], alpha)
        )
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 10.0)
        imgui.push_style_var(
            imgui.StyleVar_.window_padding,
            (12.0, 7.0) if mode == "subtle" else (16.0, 12.0),
        )
        imgui.begin("##stats_readout", None, flags)
        try:
            if mode == "subtle":
                rec = (
                    "REC · " if getattr(self, "_track_recording", False)
                    else ""
                )
                theme.mono_text(
                    f"{rec}{fps:.0f} fps · {frame_ms:.1f} ms",
                    (*TEXT_MUTED[:3], 0.95), size=13.0,
                )
            else:
                cam = self.camera()
                rows = (
                    ("fps", f"{fps:5.1f} · {frame_ms:.1f} ms"),
                    ("pos", "({:+.2f}, {:+.2f}, {:+.2f})".format(
                        *cam.position)),
                    ("view", f"az {cam.azimuth:.0f}° · el {cam.elevation:.0f}°"
                             f" · fov {cam.fov:.0f}°"),
                    ("speed", f"{self.speed:.0f} m/s"),
                    ("tier", f"{self.renderer.quality_tier} · "
                             f"{self.renderer.render_scale:.2f}x"),
                    ("cb", self._cb_bits()),
                    ("flags", f"jitter {'on' if self.jitter else 'off'}"
                              f" · map {'on' if self.minimap_enabled else 'off'}"
                              f" · bird {'on' if self.bird_enabled else 'off'}"
                              + (" · REC"
                                 if getattr(self, "_track_recording", False)
                                 else "")),
                    ("frame", f"{self._frame_index}"),
                )
                theme.push_font(theme.font_mono, 13.0)
                label_w = imgui.calc_text_size("flags").x + 14.0
                for label, value in rows:
                    theme.body_text(label, TEXT_FAINT)
                    imgui.same_line(label_w + imgui.get_style().window_padding.x)
                    theme.body_text(value, TEXT_MUTED)
                theme.pop_font()
        finally:
            imgui.end()
            imgui.pop_style_var(2)
            imgui.pop_style_color()

    def _draw(self):
        # After close() the surface texture is destroyed; a queued draw
        # submitting against it panics wgpu-native at shutdown. Bail out.
        if self._closing or self.canvas.get_closed():
            return
        self._pump_jobs()
        active_snapshot = self._active_job_snapshot()

        now = perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if not self._paused and active_snapshot is None:
            self._move(min(dt, 0.1))
        self.position = self._constrain_position(self.position)

        w, h = self.canvas.get_physical_size()
        camera = self.camera()
        camera_moving = self._camera_moving_for_quality(camera)
        self.renderer.set_camera_moving(camera_moving)
        if self._behold_job is None:
            self.renderer.write_uniforms(
                camera, (w, h), jitter=self.jitter,
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
                frame_index=self._frame_index,
                **self._cb_strength_kwargs())
        if (getattr(self, "_track_recording", False) and not self._paused
                and self._behold_job is None):
            self._track_samples.append([
                perf_counter() - self._track_t0,
                *camera.position,
                camera.azimuth, camera.elevation, camera.fov,
            ])
        self._frame_index += 1

        texture = self.context.get_current_texture()
        view = texture.create_view()
        enc = self.renderer.device.create_command_encoder()
        if self._behold_job is not None:
            # Behold owns the GPU: blit the frozen last frame instead of
            # marching the volume every frame (near-zero cost). Falls back
            # to a normal pass when no accumulated frame exists (jitter off).
            if not self.renderer.encode_present_last(enc, view, self.format):
                self.renderer.encode_pass(enc, view, self.format)
        else:
            self.renderer.encode_pass(enc, view, self.format)
        if self.bird_enabled:
            bird = self.renderer.bird
            bird.update(
                0.0 if self._paused else dt,
                self.position, self.azimuth, self.elevation,
            )
            bird.write_uniforms(
                self.position, camera, (w, h),
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
                exposure=DEFAULT_EXPOSURE,
                ambient_strength=DEFAULT_AMBIENT_STRENGTH,
            )
            bird.encode_pass(enc, view, self.format, (w, h))
        if self.minimap_enabled:
            self.renderer.hud.write_uniforms(camera, (w, h))
            self.renderer.hud.encode_pass(enc, view, self.format)
        if self._paused or active_snapshot is not None:
            self._encode_pause_overlay(enc, view)
            self._draw_imgui(enc, view)
        elif (
            getattr(self, "_stats_mode", "subtle") != "hidden"
            and getattr(self, "_fps_value", None) is not None
        ):
            self._draw_imgui(enc, view)   # corner stats readout only
        self.renderer.device.queue.submit([enc.finish()])

        self._fps_acc.append(dt)
        if now - self._fps_last_title > 0.5 and self._fps_acc:
            mean_dt = sum(self._fps_acc) / len(self._fps_acc)
            fps = 1.0 / mean_dt
            self._fps_value = fps
            self._fps_frame_ms = mean_dt * 1000.0
            if not self._title_flash_active(now):
                if self._paused:
                    self.canvas.set_title(self._menu_title())
                else:
                    self.canvas.set_title("cloudyview")
            self._fps_acc = []
            self._fps_last_title = now

        self.canvas.request_draw()

    def run(self):
        self._loop.run()


def run_app(field: CloudField, *, startup_message: str | None = None, **kwargs):
    """Open the fly-through window for a loaded CloudField (blocks)."""
    app = FlyThroughApp(field, **kwargs)
    if startup_message:
        app._flash_title(startup_message, seconds=4.0)
    app.run()
