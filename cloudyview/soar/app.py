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
    F12         save a PNG (asks size, folder, and what to include; then
                shows the result)
    ESC         pause menu (releases the mouse; field of view lives there)

Pause menu:
    ESC / R     resume and recapture the mouse
    O           open the in-window .nc browser
    N           remove the loaded nested field
    G           behold render command for this view (1-5 pick the quality,
                C copies it; behold itself runs in a terminal)
    T           time of day: sunset/midday presets + zenith/azimuth sliders
    S           quality (tier, render scale, temporal smoothing)
    P           toggle the periodic (horizontally tiled) domain — on by
                default; turn off for subvolume cutouts that are not
                physically periodic
    F           toggle fullscreen/windowed
    Q           quit from the top-level pause menu

Menus, file picking, loading progress, errors, and video progress are drawn
inside the wgpu window with Dear ImGui. The window title remains a compact
flight readout — fps, camera state in cv.Camera terms, and the cumulonimbus
realism gate bitfield (e.g. cb:1010) — for transcription into
witness/behold/soar render calls.
"""

from datetime import datetime, timezone
from pathlib import Path
import shlex
from time import perf_counter
from typing import Tuple

import numpy as np

from .. import io
from ..camera import Camera
from ..cloudfield import CloudField, load as load_cloud_field
from ..render_metadata import build_render_metadata, embed_metadata
from .engine import (
    APP_LIGHT_MARCH_LOD_DEGREES,
    APP_VIEW_STEP_LOD_DEGREES,
    DEFAULT_TONE_MAP_GAMMA,
    TONE_MAP_GAMMA_AS_FLOWN,
    TONE_MAP_GAMMA_LIMITS,
    TONE_MAP_GAMMA_WITNESS,
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
    ACTION_REMOVE_NEST,
    ACTION_PAUSE,
    ACTION_QUIT,
    ACTION_COPY_BEHOLD_COMMAND,
    ACTION_SELECT_BEHOLD_FIELD,
    ACTION_SELECT_BEHOLD_QUALITY,
    ACTION_RENDER_MENU,
    ACTION_RESUME,
    ACTION_SCREENSHOT,
    ACTION_CLOSE_PREVIEW,
    ACTION_SCREENSHOT_CLOUDS_ONLY,
    ACTION_SCREENSHOT_WITH_OVERLAYS,
    ACTION_TOGGLE_FULLSCREEN,
    ACTION_TOGGLE_PERIODIC,
    ACTION_SELECT_SUN_PRESET,
    ACTION_QUALITY_MENU,
    ACTION_SUN_MENU,
    ACTION_SELECT_BOTH_GROUPS,
    ACTION_SELECT_GROUP,
    ACTION_SELECT_TIER,
    ACTION_SELECT_UNITS,
    ACTION_CONTROLS_MENU,
    ACTION_TRACK_SAVE,
    ACTION_TRACK_DISCARD,
    BEHOLD_QUALITIES_BY_KEY,
    MIN_SUN_ELEVATION_DEG,
    PAIR_KEY_BY_INDEX,
    SUN_PRESETS,
    MENU_CONTROLS,
    MENU_SCREENSHOT,
    MENU_SCREENSHOT_PREVIEW,
    MENU_TRACK_SAVE,
    MENU_ERROR,
    MENU_FILE_BROWSER_ICE,
    MENU_FILE_BROWSER_LIQUID,
    MENU_MAIN,
    MENU_OPEN_GROUP_PROMPT,
    MENU_OPEN_ICE_PROMPT,
    MENU_OPEN_UNITS_PROMPT,
    MENU_RENDER_QUALITY,
    MENU_QUALITY,
    MENU_SUN,
    FileEntry,
    control_action_for_key as _control_action_for_key,
    list_netcdf_entries,
    menu_transition as _menu_transition,
)

DEFAULT_SPEED = 60.0        # m/s, comfortable for the 25 km dev domain
MOUSE_SENS = 0.12           # degrees per pixel
SPEED_WHEEL_FACTOR = 1.25   # per wheel notch
# Minimum flight height above the z=0 ocean, and so the lowest the bird can
# get (it rides the camera — see bird.DISTANCE/DROP — and has no floor of its
# own). Originally five dominant ocean wavelengths (FIF outer scale,
# ocean_fif.DEFAULT_OUTER_SCALE_M = 10 m), below which the normal-mapped water
# reads wrong for want of displacement geometry — Thomas 2026-07-10. Halved to
# 2.5 wavelengths on 2026-07-31 (Thomas): flying lower is worth more than the
# water holding up at the very bottom of the range.
OCEAN_FLOOR_MARGIN_M = 2.5 * 10.0
# Behold quality presets, as offered by the command panel. These are
# behold's own CLI names — the panel hands over a command, it does not run
# anything.
BEHOLD_QUALITY_ROWS = (
    ("Min", "1", "min", "fast preview"),
    ("Low", "2", "low", "draft"),
    ("Medium", "3", "medium", "balanced"),
    ("High", "4", "high", "~1 h"),
    ("Max", "5", "max", "overnight"),
)

# Capture defaults shared by the screenshot (F12) and video (R) dialogs —
# they are the same decision twice, so they are the same settings twice.
CAPTURE_SIZE_PRESETS = (
    ("1280 x 720", (1280, 720)),
    ("1920 x 1080", (1920, 1080)),
    ("3840 x 2160", (3840, 2160)),
)
CAPTURE_SIZE_LIMITS = (64, 7680)
DEFAULT_VIDEO_FPS = 60.0
DEFAULT_VIDEO_ACCUMULATE = 24


def default_save_dir() -> Path:
    """Where captures land by default: ~/Downloads, else home."""
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.home()

CONTROL_SUMMARY = (
    "Controls: W/S forward/back, A/D strafe, Space up, LShift/C down, mouse look "
    "(Tab releases, click recaptures), scroll speed, "
    "B bird toggle, M minimap toggle, F3 stats readout, "
    "R record flight track (again to stop, then save; re-render with "
    "then pick size/folder and render it to mp4), "
    "F fullscreen/window, F12 screenshot, F1/? controls reference, "
    "ESC pause menu; "
    "paused: ESC/R resume, O open in-window file browser, "
    "N remove a loaded nested field, G behold render command, "
    "S quality (tiers, render scale, smoothing), T time of day, "
    "C controls reference, "
    "P periodic domain toggle, "
    "F fullscreen/window, Q quit from the top-level menu"
)

CB_DEFAULT_STRENGTHS = (
    DEFAULT_GRADIENT_SHADING_STRENGTH,
    DEFAULT_DEEP_SHADOW_MS_SUPPRESSION,
    DEFAULT_AMBIENT_OCCLUSION_STRENGTH,
    DEFAULT_BOUNCE_DEPTH_ATTENUATION,
)

# A saved still is the view you were looking at, held still — so it is
# rendered with the app's own distance LOD (APP_*_LOD_DEGREES), not the
# library's exact-legacy 0.0. Stills used to get 0.0 by accident, through
# render()'s defaults, and the difference is not subtle: the coarser light
# march the app flies with lets more light through the far field, giving
# the aerial haze that makes flight look right. Turning it off darkened
# and hardened every distant cloud, which is not what the app looks like.
#
# What a still does get is time: the live view converges by accumulating
# jittered frames while parked, and a one-frame still cannot. Rendering
# this many gets the same converged image instead of a grainy one.
STILL_ACCUMULATE_FRAMES = 64

def _present_format(preferred: str) -> str:
    """The swapchain format, with sRGB encoding refused.

    raymarch.wgsl's tone_map already gamma-encodes what it returns, so an
    ``*-srgb`` swapchain encodes a second time and the window stops
    matching the offscreen path, witness, and the browser build — which
    is exactly what happened, unnoticed, for as long as the app has had a
    window (found 2026-08-04, comparing a screenshot against the live
    view). Gamma belongs in one place: engine.DEFAULT_TONE_MAP_GAMMA.
    """
    if not preferred.endswith("-srgb"):
        return preferred
    return preferred[: -len("-srgb")]


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
            d = forward + sx * tan_half * right + sy * tan_half / aspect * up
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
                 nest: CloudField | None = None,
                 extinction_multiplier: float = 1.0,
                 max_fps: float = 120.0,
                 camera: Camera | None = None,
                 periodic: bool = True,
                 tier: str = "auto",
                 sun_azimuth: float | None = None,
                 sun_elevation: float | None = None,
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
        if sun_azimuth is not None or sun_elevation is not None:
            self._set_sun(
                azimuth=sun_azimuth,
                zenith=None if sun_elevation is None else 90.0 - sun_elevation,
            )

        device = request_device()
        # The nest belongs to the field it was launched with. Opening another
        # file from the O-menu replaces the scene, so it drops the nest
        # rather than silently re-placing it under unrelated data.
        self.renderer = self._create_renderer(field, nest=nest, device=device)

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
        self.format = _present_format(
            self.context.get_preferred_format(device.adapter)
        )
        self.context.configure(device=device, format=self.format)

        self.speed = DEFAULT_SPEED
        # Decided configuration (2026-07-17): the A/B toggle keys for
        # jitter (J), the realism gates (1-4), and the distance LOD (L)
        # are retired — these are permanently on. The attributes remain
        # for the metadata/uniform plumbing.
        self.jitter = True
        self.cb_enabled = [True, True, True, True]
        # How much the far field lifts into haze. See engine's
        # DEFAULT_TONE_MAP_GAMMA for why the default is neither witness's
        # 1.4 nor the 3.08 this app spent its life accidentally rendering.
        self.tone_map_gamma = DEFAULT_TONE_MAP_GAMMA
        self.distance_lod = True
        self._track_recording = False   # R toggles (see soar/track.py)
        self._track_samples: list[list[float]] = []
        self._track_t0 = 0.0
        self._track_pending = None      # stopped take awaiting save/discard
        self._speed_flash_until = 0.0   # transient m/s readout after scroll
        self.bird_enabled = True
        self.minimap_enabled = True
        self._keys = set()
        self._closing = False
        self._last_pointer = None   # None -> ignore next move (capture jump guard)
        self._captured = False
        self._paused = False
        self._menu_state = MENU_MAIN
        self._pending_open_path = None
        self._pending_ice_path = None
        self._pending_group = None        # chosen NetCDF group, None = root
        self._pending_group_choices = []  # groups awaiting the user's pick
        self._pending_units = None        # user-supplied condensate units
        self._pending_units_vars = []     # variables that carry no units
        self._pending_nest_group = None   # second group, loaded as the nest
        self._pending_nest_pairs = []     # (outer, inner)s offered by the picker
        # The browser opens at home, not next to whatever file the session was
        # launched with: data lives all over the disk and home is the one
        # place every tree is reachable from. It remembers the last directory
        # visited within the session (_last_file_dir).
        self._file_browser_dir = Path.home()
        self._last_file_dir = self._file_browser_dir
        self._file_browser_error = None
        self._loading_job = None
        self._video_render = None        # foreground track -> mp4 encode
        self._pending_screenshot = None   # True/False = overlays; taken in _draw
        # Capture settings, shared by the screenshot and video dialogs.
        self._save_dir = default_save_dir()
        self._save_dir_text = str(self._save_dir)
        self._capture_size = None        # None = follow the window
        self._video_fps = DEFAULT_VIDEO_FPS
        self._video_accumulate = DEFAULT_VIDEO_ACCUMULATE
        self._preview = None             # {"ref", "keep", "image", "path"}
        self._behold_quality = "high"    # which command the G panel shows
        self._behold_field_choice = "outer"  # "outer"/"nest": behold takes one
        self._clipboard_note = None
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

    def _create_renderer(self, field: CloudField, *, nest: CloudField | None = None,
                         device=None, previous=None):
        """Create the field-resident renderer, reusing the app GPU device."""
        n_voxels = field.lwc.size + (nest.lwc.size if nest is not None else 0)
        volume_fp16 = choose_volume_fp16(n_voxels, self.volume_fp16)
        if volume_fp16 and self.volume_fp16 is None:
            print(
                f"soar: {n_voxels / 1e6:.0f}M-voxel volume -> fp16 "
                "texture (auto; --fp32-volume forces full precision)"
            )
        kwargs = {
            "nest": nest,
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

    def _set_tone_map_gamma(self, value: float) -> None:
        """Set the tone-map gamma and drop the accumulated frames.

        Gamma is scene identity in the uniform block, so an accumulation
        built at the old value would blend two different looks together.
        """
        value = float(value)
        lo, hi = TONE_MAP_GAMMA_LIMITS
        if not lo <= value <= hi:
            raise ValueError(
                f"tone-map gamma must be in [{lo}, {hi}]; got {value}."
            )
        self.tone_map_gamma = value
        self.renderer.reset_accumulation()
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
        if state == MENU_OPEN_GROUP_PROMPT:
            return "cloudyview which NetCDF group?"
        if state == MENU_OPEN_UNITS_PROMPT:
            return "cloudyview which condensate units?"
        if state in (MENU_FILE_BROWSER_LIQUID, MENU_FILE_BROWSER_ICE):
            return "cloudyview file browser"
        if state == MENU_RENDER_QUALITY:
            return self._render_quality_title()
        if state == MENU_QUALITY:
            return "cloudyview quality"
        if state == MENU_SUN:
            return "cloudyview time of day"
        if state == MENU_CONTROLS:
            return "cloudyview controls"
        if state == MENU_TRACK_SAVE:
            return "cloudyview save flight track?"
        if state == MENU_SCREENSHOT:
            return "cloudyview screenshot: what to include?"
        if state == MENU_SCREENSHOT_PREVIEW:
            return "cloudyview screenshot saved"
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

    def _cb_strength_kwargs(self) -> dict:
        strengths = [
            value if enabled else 0.0
            for enabled, value in zip(self.cb_enabled, CB_DEFAULT_STRENGTHS)
        ]
        return {
            "tone_map_gamma": self.tone_map_gamma,
            "gradient_shading_strength": strengths[0],
            "deep_shadow_ms_suppression": strengths[1],
            "ambient_occlusion_strength": strengths[2],
            "bounce_depth_attenuation": strengths[3],
            # Distance LOD (decided 2026-07-17, aggressive angles): always
            # on in the app; the library default stays 0.0 (exact legacy).
            "light_march_lod_degrees": APP_LIGHT_MARCH_LOD_DEGREES,
            "view_step_lod_degrees": APP_VIEW_STEP_LOD_DEGREES,
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
            # A recording survives a pause, but paused time must not leak
            # into the track's clock (the offline video would render a
            # long hold between the two poses).
            if getattr(self, "_track_recording", False):
                self._track_pause_started = perf_counter()
            self._menu_state = MENU_MAIN
            self._capture_mouse(False)
            self.canvas.set_title(self._menu_title())
        else:
            self._menu_state = MENU_MAIN
            self._reset_pending_open()
            self._file_browser_error = None
            self._error_message = None
            self._capture_mouse(True)
            self._last_time = perf_counter()
            paused_for = getattr(self, "_track_pause_started", None)
            if paused_for is not None:
                if getattr(self, "_track_recording", False):
                    self._track_t0 += perf_counter() - paused_for
                self._track_pause_started = None
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

    def _relative_position_in(self, bmin, bmax) -> tuple:
        """Where the camera is as a fraction of the given box, ±1 at its edges.

        z is anchored to the physical surface rather than the box floor, so a
        field that starts aloft keeps its real altitude — the convention
        witness and behold both read (see camera.py).
        """
        return (
            2.0 * (self.position[0] - bmin[0]) / (bmax[0] - bmin[0]) - 1.0,
            2.0 * (self.position[1] - bmin[1]) / (bmax[1] - bmin[1]) - 1.0,
            2.0 * self.position[2] / bmax[2] - 1.0,
        )

    def camera(self) -> Camera:
        """Current viewpoint as a cv.Camera (relative-coordinate position)."""
        rel = self._relative_position_in(self.renderer.bmin, self.renderer.bmax)
        return Camera(position=rel, azimuth=self.azimuth,
                      elevation=self.elevation, fov=self.fov)

    def _start_open_file(self) -> None:
        self._reset_pending_open()
        self._file_browser_error = None
        self._set_file_browser_dir(getattr(self, "_last_file_dir", Path.home()))
        self._set_menu_state(MENU_FILE_BROWSER_LIQUID)

    # ------------------------------------------------------------------
    # Time of day
    # ------------------------------------------------------------------

    @property
    def sun_zenith(self) -> float:
        """Solar zenith angle in degrees — 0 overhead, 90 at the horizon."""
        return 90.0 - self.sun_elevation

    def _set_sun(self, azimuth: float | None = None,
                 zenith: float | None = None) -> None:
        """Move the sun. Elevation is floored just above the horizon.

        A periodic domain's light march exits only through the domain top,
        so write_uniforms refuses a sun at or below the horizon; clamping
        here keeps the slider usable all the way to its end instead of
        raising at the last degree.
        """
        if azimuth is not None:
            self.sun_azimuth = float(azimuth) % 360.0
        if zenith is not None:
            elevation = 90.0 - float(zenith)
            self.sun_elevation = max(MIN_SUN_ELEVATION_DEG, min(90.0, elevation))

    def _select_sun_preset(self, name: str | None) -> None:
        preset = SUN_PRESETS.get(name or "")
        if preset is None:
            return
        azimuth, elevation = preset
        self._set_sun(azimuth=azimuth, zenith=90.0 - elevation)
        self._flash_title(f"cloudyview sun: {name}", seconds=2.0)

    def _remove_nest(self) -> None:
        """Drop the nested level, keeping the outer field and the camera."""
        if not self.renderer.nested:
            return
        previous = self.renderer
        self.renderer = self._create_renderer(
            previous.field, nest=None, device=previous.device, previous=previous
        )
        self._frame_index = 0
        self.renderer.reset_accumulation()
        self._last_quality_camera_signature = None
        self._camera_moving_for_quality(self.camera())
        self._set_paused(False)
        self._flash_title("cloudyview nested field removed", seconds=3.0)

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

    def _reset_pending_open(self) -> None:
        self._pending_open_path = None
        self._pending_ice_path = None
        self._pending_group = None
        self._pending_group_choices = []
        self._pending_units = None
        self._pending_units_vars = []
        self._pending_nest_group = None
        self._pending_nest_pairs = []

    def _select_browser_path(self, path: str | Path) -> None:
        selected = Path(path).expanduser().resolve()
        self._last_file_dir = selected.parent
        if self._menu_state == MENU_FILE_BROWSER_LIQUID:
            self._reset_pending_open()
            self._pending_open_path = str(selected)
            self._start_group_selection()
        elif self._menu_state == MENU_FILE_BROWSER_ICE:
            self._start_loading_file(
                self._pending_open_path, str(selected)
            )

    def _start_group_selection(self) -> None:
        """Pick the NetCDF group holding the field, before anything is read.

        Files that keep each field in its own group (STEAM render nests)
        have an empty root, so the loader's variable search has nothing to
        find. One candidate group is taken automatically; several are the
        user's call.
        """
        try:
            groups = io.find_liquid_water_groups(self._pending_open_path)
        except Exception:
            # Unreadable file: leave it to the load, which reports the
            # failure with its own wording.
            groups = []

        # Several groups in one file are often a coarse domain and a
        # refinement of part of it — exactly a nested pair. Probe the
        # coordinates (no field data) so the picker can offer both at once
        # instead of making the user choose one and lose the other. Three
        # levels give several pairs and the renderer holds two: offer every
        # pair rather than deciding for the user which two levels they meant.
        self._pending_nest_pairs = []
        if len(groups) > 1:
            try:
                self._pending_nest_pairs = io.find_nestable_group_pairs(
                    self._pending_open_path, groups
                )
            except Exception:
                self._pending_nest_pairs = []

        if not groups or "" in groups:
            # Root group carries the field, or nothing anywhere does — in
            # which case the loader's own message names the variables it
            # looked for. Never guess a group in that case.
            self._pending_group = None
        elif len(groups) == 1:
            self._pending_group = groups[0]
            print(f"cloudyview: using NetCDF group '{groups[0]}'")
        else:
            self._pending_group_choices = groups
            self._set_menu_state(MENU_OPEN_GROUP_PROMPT)
            return

        self._set_menu_state(MENU_OPEN_ICE_PROMPT)

    def _select_group(self, index: int | None) -> None:
        choices = self._pending_group_choices
        if index is None or not (0 <= index < len(choices)):
            return
        self._pending_group = choices[index]
        self._pending_nest_group = None
        self._set_menu_state(MENU_OPEN_ICE_PROMPT)

    def _select_both_groups_nested(self, index: int = 0) -> None:
        """Load one coarse group and one refinement of it as one scene.

        `index` picks among the pairs the file offers — a three-level file
        has more than one, and the renderer holds two levels at a time.

        Skips the ice prompt: a second ice *file* makes no sense for a pair
        of groups that already live in this one.
        """
        pairs = self._pending_nest_pairs
        if not (0 <= index < len(pairs)):
            return
        self._pending_group, self._pending_nest_group = pairs[index]
        self._start_loading_file(self._pending_open_path, None)

    def _select_condensate_units(self, units: str | None) -> None:
        if units is None:
            return
        self._pending_units = units
        self._start_loading_file(self._pending_open_path, self._pending_ice_path)

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

    @staticmethod
    def _condensate_vars_missing_units(
        liquid_path: str, ice_path: str | None, group: str | None
    ) -> list:
        """Condensate variables with no 'units' attribute, liquid then ice.

        Probing costs one metadata open. Any failure here is left to the
        real load, which reports it with its own wording.
        """
        try:
            missing = io.condensate_vars_missing_units(liquid_path, group=group)
            if ice_path:
                missing = missing + io.condensate_vars_missing_units(ice_path)
        except Exception:
            return []
        return missing

    def _start_loading_file(
        self, liquid_path: str | Path | None, ice_path: str | Path | None
    ) -> None:
        if not liquid_path:
            self._show_error("Open file failed: no liquid NetCDF was selected.")
            return
        if self._loading_job is not None or self._video_render is not None:
            self._show_error("Another operation is already running.")
            return

        liquid_path = str(liquid_path)
        ice_path = str(ice_path) if ice_path else None
        group = self._pending_group
        units = self._pending_units
        nest_group = self._pending_nest_group

        if units is None:
            missing = self._condensate_vars_missing_units(
                liquid_path, ice_path, group
            )
            if nest_group is not None:
                # One answer covers both groups: they are the same file, and
                # asking twice for the same file's convention is noise.
                missing = missing + self._condensate_vars_missing_units(
                    liquid_path, None, nest_group
                )
            if missing:
                self._pending_open_path = liquid_path
                self._pending_ice_path = ice_path
                self._pending_units_vars = missing
                self._set_menu_state(MENU_OPEN_UNITS_PROMPT)
                return

        previous = self.renderer
        device = previous.device
        filename = Path(liquid_path).name

        def target(report):
            def stage(stage_name: str) -> None:
                report(stage_name)

            field = load_cloud_field(
                liquid_path,
                ice=ice_path,
                liquid_water_group=group,
                # A separate ice file is its own root; the group only
                # applies to ice living alongside the liquid variable.
                ice_water_group=None if ice_path else group,
                fallback_units=units,
                stage_callback=stage,
            )
            nest_field = None
            if nest_group is not None:
                report(f"loading nest group {nest_group}")
                nest_field = load_cloud_field(
                    liquid_path,
                    liquid_water_group=nest_group,
                    ice_water_group=nest_group,
                    fallback_units=units,
                )

            report("building extinction")
            if nest_group is not None:
                renderer = self._create_renderer(
                    field, nest=nest_field, device=device, previous=previous
                )
            else:
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

        self._reset_pending_open()
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
        # A choice of field belongs to the file it was made for.
        self._behold_field_choice = "outer"
        self._reset_camera_to_default()
        self._frame_index = 0
        self.renderer.reset_accumulation()
        self._last_quality_camera_signature = None
        self._camera_moving_for_quality(self.camera())
        print(f"Loaded {self.renderer.field}")
        if self.renderer.nested:
            coverage = self.renderer.nest_coverage_fraction
            print(
                f"Loaded nest {self.renderer.nest} "
                f"(covers {coverage * 100:.0f}% of the outer domain)"
            )
        self._set_paused(False)
        self._flash_title(
            f"cloudyview loaded {Path(result['liquid_path']).name}",
            seconds=3.0,
        )

    def _show_error(self, message: str) -> None:
        self._error_message = str(message)
        self._loading_job = None
        self._video_render = None
        self._set_paused(True)
        self._set_menu_state(MENU_ERROR)
        print(f"cloudyview error: {self._error_message}")
        self.canvas.request_draw()

    def _active_job(self):
        for job in (self._loading_job,):
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

    @staticmethod
    def _abs_source(path: str | None) -> str:
        """Absolute path for a command that will be run from anywhere.

        The panel's command is copied into some other terminal, in some
        other directory; the relative path soar itself was launched with
        would not resolve there.
        """
        if not path:
            return "<in-memory>"
        return str(Path(path).expanduser().resolve())

    def _behold_target_field(self) -> tuple:
        """(field, label, box) the behold command should name.

        Behold renders one field from one group. With a nest loaded there
        are two on screen, so the choice is the user's — `_behold_field_choice`
        holds it, and falls back to the outer field when there is no nest
        (or the nest went away under a stale selection).

        The box comes along because a relative position means "this far
        across THIS field": quoting the outer domain's fraction at a nest a
        fortieth of its width puts the camera kilometres from the view it
        was framed in.
        """
        nest = self.renderer.nest if self.renderer.nested else None
        if nest is not None and self._behold_field_choice == "nest":
            return nest, "nest", (self.renderer.nest_bmin, self.renderer.nest_bmax)
        return (self.renderer.field, "outer",
                (self.renderer.bmin, self.renderer.bmax))

    @staticmethod
    def _behold_group_arguments(field) -> list:
        """Group flags that point behold at the same arrays soar read.

        Nothing to say for a field that came from the root group. A
        separate ice file is its own root, so a group there can only apply
        to the liquid lookup; otherwise one group covers the whole dataset
        (coordinates included), which is how soar loaded it.
        """
        liquid_group = field.liquid_group
        ice_group = None if field.ice_source else field.ice_group
        if ice_group and ice_group != liquid_group:
            parts = (
                ["--liquid-water-group", liquid_group] if liquid_group else []
            )
            return parts + ["--ice-water-group", ice_group]
        if not liquid_group:
            return []
        if field.ice_source:
            return ["--liquid-water-group", liquid_group]
        return ["--group", liquid_group]

    def _behold_reproduction_command(self, camera: Camera, quality: str) -> str:
        field, _, (bmin, bmax) = self._behold_target_field()
        camera = Camera(
            position=self._relative_position_in(bmin, bmax),
            azimuth=camera.azimuth, elevation=camera.elevation,
            fov=camera.fov,
        )
        source = self._abs_source(field.source)
        parts = ["behold", source, quality, "--gpu"]
        if field.ice_source:
            parts.extend(["--ice", self._abs_source(field.ice_source)])
        parts.extend(self._behold_group_arguments(field))
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
        nest = self.renderer.nest
        if nest is not None and nest.source:
            parts.extend(["--nest", nest.source])
            if nest.ice_source:
                parts.extend(["--nest-ice", nest.ice_source])
        parts.extend(["--tier", self.renderer.quality_tier])
        if self.sun_azimuth != DEFAULT_SUN_AZIMUTH:
            parts.extend(["--sun-azimuth", self._fmt_num(self.sun_azimuth)])
        if self.sun_elevation != DEFAULT_SUN_ELEVATION:
            parts.extend(["--sun-elevation", self._fmt_num(self.sun_elevation)])
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

    # ------------------------------------------------------------------
    # Capture settings (shared by the F12 screenshot and the track video)
    # ------------------------------------------------------------------

    def capture_size(self) -> Tuple[int, int]:
        """The size captures render at — the window unless overridden."""
        if self._capture_size is not None:
            return self._capture_size
        w, h = self.canvas.get_physical_size()
        return (int(w), int(h))

    def _set_capture_size(self, size) -> None:
        """Clamp and store an explicit capture size, or None to follow the
        window. Sizes are clamped rather than rejected: a half-typed number
        in a text field must not throw."""
        if size is None:
            self._capture_size = None
            return
        lo, hi = CAPTURE_SIZE_LIMITS
        self._capture_size = tuple(
            int(min(max(int(v), lo), hi)) for v in size
        )

    def _set_save_dir(self, text: str) -> bool:
        """Accept a save directory if it exists; report whether it took."""
        self._save_dir_text = str(text)
        try:
            path = Path(text).expanduser()
        except Exception:
            return False
        if not path.is_dir():
            return False
        self._save_dir = path
        return True

    def _timestamped_path(self, prefix: str, suffix: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self._save_dir / f"{prefix}_{stamp}{suffix}"

    def _timestamped_png_path(self, prefix: str) -> Path:
        return self._timestamped_path(prefix, ".png")

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
            self._stop_track_recording()
            return
        self._track_samples = []
        self._track_t0 = perf_counter()
        self._track_recording = True
        self._flash_title("cloudyview recording flight track — R to stop",
                          seconds=3.0)

    def _stop_track_recording(self) -> None:
        """R while recording: hold the take and ask save / discard."""
        self._track_recording = False
        samples = self._track_samples
        self._track_samples = []
        if len(samples) < 2:
            self._flash_title("cloudyview track discarded (too short)",
                              seconds=3.0)
            return
        self._track_pending = samples
        self._set_paused(True)
        self._set_menu_state(MENU_TRACK_SAVE)

    def _track_save_pending(self) -> None:
        from .track import save_track

        samples = getattr(self, "_track_pending", None)
        self._track_pending = None
        if not samples:
            self._set_paused(False)
            return
        path = self._timestamped_path("cloudyview_track", ".json")
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
        print(
            f"Saved track {path} ({len(samples)} samples, {duration:.1f}s)"
        )
        self._start_video_render(path)

    # ------------------------------------------------------------------
    # Track video (foreground: the encode owns the GPU)
    # ------------------------------------------------------------------

    def _start_video_render(self, track_path) -> None:
        """Begin encoding the saved track, stepped from the draw loop.

        Deliberately not a background thread: the encode renders through the
        app's own resident volume, which is not shareable across threads,
        and the whole point is to give the GPU to the encode rather than
        split it with live marching.
        """
        from .track import TrackVideoRender

        out_path = track_path.with_suffix(".mp4")
        width, height = self.capture_size()
        try:
            self._video_render = TrackVideoRender(
                track_path, out_path,
                fps=self._video_fps,
                size=(width, height),
                accumulate_frames=self._video_accumulate,
                renderer=self.renderer,
            )
        except Exception as e:
            self._video_render = None
            self._show_error(f"Video render failed to start: {e}")
            return
        self._set_menu_state(MENU_MAIN)
        self._set_paused(True)
        self.canvas.set_title(f"cloudyview rendering {out_path.name}")
        print(
            f"Rendering {out_path} — {self._video_render.total} frames at "
            f"{width}x{height}, {self._video_fps:g} fps"
        )

    def _step_video_render(self, budget_seconds: float = 0.25) -> None:
        """Encode as many frames as fit in one draw's time budget.

        The budget is what keeps the progress bar and the window alive: a
        single 4K accumulated frame can take seconds, so this always steps
        at least one and stops as soon as it has overrun.
        """
        render = self._video_render
        if render is None:
            return
        deadline = perf_counter() + budget_seconds
        try:
            while True:
                finished = render.step()
                if finished or perf_counter() >= deadline:
                    break
        except Exception as e:
            self._video_render = None
            self._show_error(f"Video render failed: {e}")
            return
        if not render.done:
            return

        self._video_render = None
        try:
            out_path = render.close()
        except Exception as e:
            self._show_error(f"Video render failed: {e}")
            return
        print(f"Video saved to {out_path}")
        self._set_paused(False)
        self._flash_title(f"cloudyview video saved {out_path}", seconds=5.0)

    def _cancel_video_render(self) -> None:
        render = self._video_render
        self._video_render = None
        if render is not None:
            render.abort()
        self._set_paused(False)
        self._flash_title("cloudyview video render cancelled", seconds=3.0)

    def _draw_video_progress(self, imgui) -> None:
        from .theme import TEXT_FAINT

        render = self._video_render
        if render is None:
            return
        progress = render.progress()
        theme = self._theme
        self._begin_imgui_window(imgui, "video_progress", 520.0)
        try:
            theme.header("rendering video", render.out_path.name)
            theme.mono_text(
                f"frame {progress['frame']}/{progress['total']} · "
                f"{render.width}x{render.height} · "
                f"{self._fmt_eta(progress['elapsed'])} elapsed",
                size=13.0,
            )
            imgui.dummy((1.0, 2.0))
            theme.progress_bar(progress["percent"] / 100.0)
            theme.push_font(theme.font_mono, 13.0)
            theme.body_text(f"{progress['percent']:3.0f}%", TEXT_FAINT)
            if progress["eta"] is not None:
                eta_text = f"ETA {self._fmt_eta(progress['eta'])}"
                text_w = imgui.calc_text_size(eta_text).x
                imgui.same_line(
                    imgui.get_window_width()
                    - imgui.get_style().window_padding.x - text_w
                )
                theme.body_text(eta_text, TEXT_FAINT)
            theme.pop_font()
            theme.caption(
                f"{progress['fps']:.2f} frames/s rendered", TEXT_FAINT
            )
            imgui.dummy((1.0, 6.0))
            if theme.menu_button("Cancel", "ESC", height=38.0):
                self._cancel_video_render()
        finally:
            self._end_imgui_window(imgui)

    def _track_discard_pending(self) -> None:
        self._track_pending = None
        self._flash_title("cloudyview track discarded", seconds=2.5)
        self._set_paused(False)

    def _save_screenshot(self, *, overlays: bool = True) -> None:
        """Render and save a PNG. ``overlays`` includes bird + minimap.

        The choice is per shot (the F12 prompt), not the app's live B/M
        toggles: the frame you want to keep and the frame you want to fly
        with are rarely the same one.
        """
        camera = self.camera()
        w, h = self.capture_size()
        path = self._timestamped_png_path("cloudyview_soar")
        renderer = self.renderer
        want_bird = bool(overlays)
        want_hud = bool(overlays)
        accum_state = (
            getattr(renderer, "_accum_key", None),
            getattr(renderer, "_accum_count", 0),
            getattr(renderer, "_accum_index", 0),
        )
        had_bird = getattr(renderer, "_bird", None) is not None
        bird_state = None
        if want_bird and had_bird:
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
        # Exactly what the live view is showing — same LOD, same look gates
        # (see STILL_ACCUMULATE_FRAMES). Passing these was the whole fix:
        # render()'s own defaults are the library's, not the app's.
        look = self._cb_strength_kwargs()
        try:
            image = renderer.render(
                camera,
                size=(w, h),
                bird=want_bird,
                hud=want_hud,
                jitter=self.jitter,
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
                frame_index=self._frame_index,
                accumulate_frames=STILL_ACCUMULATE_FRAMES,
                **look,
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
            elif want_bird and not had_bird:
                renderer._bird = None

        metadata = self._metadata(
            camera,
            renderer="soar",
            reproduction_command=self._soar_reproduction_command(camera),
            render_options={
                "bird": want_bird,
                "hud": want_hud,
                "jitter": bool(self.jitter),
                "size": [int(w), int(h)],
                "tier": renderer.quality_tier,
                "render_scale": renderer.render_scale,
                "step_factor": renderer.step_factor,
                "max_light_steps": renderer.max_light_steps,
                "volume_fp16": renderer.volume_fp16,
                # Without these the recorded command does not reproduce the
                # recorded image: the LOD differs from the live view's.
                **look,
            },
        )
        self._write_png_with_metadata(image, path, metadata)
        print(f"Screenshot saved to {path}")
        self._show_preview(image, path)
        self._flash_title(f"cloudyview screenshot saved {path}", seconds=4.0)

    def _show_preview(self, image, path) -> None:
        """Park on the saved frame with a Close button.

        The shot is already on disk before this runs, so a failure to
        register the preview texture must not read as a failed capture —
        fall back to returning straight to flight.
        """
        self._release_preview()
        if self._imgui is None:
            self._close_preview()
            return
        try:
            ref, keep = self._imgui.register_image(image)
        except Exception as e:  # pragma: no cover - backend/driver specific
            # The frame is already on disk; a preview that will not register
            # must not read as a failed capture.
            print(f"cloudyview: preview unavailable ({e})")
            self._close_preview()
            return
        self._preview = {
            "ref": ref, "keep": keep, "image": image, "path": path,
        }
        self._set_paused(True)
        self._set_menu_state(MENU_SCREENSHOT_PREVIEW)

    def _on_event(self, event):
        self._pump_jobs()
        if self._video_render is not None:
            if self._imgui is not None:
                self._imgui.handle_event(event)
            if (event.get("event_type") == "key_down"
                    and event.get("key") == "Escape"):
                self._cancel_video_render()
            return
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
                # F1/? pauses straight into the controls reference.
                if transition.next_state not in (None, MENU_MAIN):
                    self._set_menu_state(transition.next_state)
            elif action == ACTION_RESUME:
                self._set_paused(False)
            elif action == ACTION_QUIT:
                self._closing = True
                self._close_after_frame()
            elif action == ACTION_TOGGLE_FULLSCREEN:
                self._try_toggle_fullscreen()
            elif action == ACTION_OPEN_FILE:
                self._start_open_file()
            elif action == ACTION_REMOVE_NEST:
                self._remove_nest()
            elif action == ACTION_OPEN_ICE_YES:
                self._finish_open_file(use_ice=True)
            elif action == ACTION_OPEN_ICE_NO:
                self._finish_open_file(use_ice=False)
            elif action == ACTION_SELECT_GROUP:
                self._select_group(transition.group_index)
            elif action == ACTION_SELECT_BOTH_GROUPS:
                self._select_both_groups_nested(transition.pair_index or 0)
            elif action == ACTION_SELECT_UNITS:
                self._select_condensate_units(transition.units)
            elif action == ACTION_RENDER_MENU:
                self._clipboard_note = None
                self._set_menu_state(
                    transition.next_state or MENU_RENDER_QUALITY
                )
            elif action == ACTION_QUALITY_MENU:
                self._set_menu_state(transition.next_state or MENU_QUALITY)
            elif action == ACTION_SUN_MENU:
                self._set_menu_state(transition.next_state or MENU_SUN)
            elif action == ACTION_SELECT_SUN_PRESET:
                self._select_sun_preset(transition.sun_preset)
            elif action == ACTION_SELECT_BEHOLD_QUALITY:
                self._behold_quality = transition.quality
                self._clipboard_note = None
            elif action == ACTION_SELECT_BEHOLD_FIELD:
                if self.renderer.nested:
                    self._select_behold_field(transition.behold_field)
            elif action == ACTION_COPY_BEHOLD_COMMAND:
                self._copy_behold_command()
            elif action == ACTION_SELECT_TIER:
                self._select_quality_tier(transition.tier)
            elif action == ACTION_MENU_BACK:
                if transition.next_state == MENU_OPEN_ICE_PROMPT:
                    self._set_menu_state(MENU_OPEN_ICE_PROMPT)
                else:
                    self._reset_pending_open()
                    self._error_message = None
                    self._set_menu_state(transition.next_state or MENU_MAIN)
            elif action == ACTION_TOGGLE_PERIODIC:
                self._toggle_periodic()
            elif action == ACTION_CONTROLS_MENU:
                self._set_menu_state(MENU_CONTROLS)
            elif action == ACTION_TRACK_SAVE:
                self._track_save_pending()
            elif action == ACTION_TRACK_DISCARD:
                self._track_discard_pending()
            elif action == ACTION_SCREENSHOT:
                self._set_paused(True)
                self._set_menu_state(transition.next_state or MENU_SCREENSHOT)
            elif action == ACTION_SCREENSHOT_WITH_OVERLAYS:
                self._save_screenshot(overlays=True)
            elif action == ACTION_SCREENSHOT_CLOUDS_ONLY:
                self._save_screenshot(overlays=False)
            elif action == ACTION_CLOSE_PREVIEW:
                self._close_preview()
            elif key in ("r", "R"):
                # Only reached unpaused: while paused the menu table above
                # consumes R as resume (documented), so record start/stop
                # is a flying-only control.
                self._toggle_track_recording()
            elif self._paused:
                return
            elif key in ("b", "B"):
                self.bird_enabled = not self.bird_enabled
            elif key in ("m", "M"):
                self.minimap_enabled = not self.minimap_enabled
            elif key == "F3":
                self._cycle_stats_mode()
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
            # Surface the new speed briefly in the corner pill (otherwise
            # it is only visible in the expanded F3 stats).
            self._speed_flash_until = perf_counter() + 1.5

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
        if self._video_render is not None:
            self._draw_video_progress(imgui)
            return

        snapshot = self._active_job_snapshot()
        if snapshot is not None:
            self._draw_job_overlay(imgui, snapshot)
            return

        if not getattr(self, "_paused", False):
            self._draw_stats_readout(imgui)
            return

        state = getattr(self, "_menu_state", MENU_MAIN)
        if state == MENU_RENDER_QUALITY:
            self._draw_behold_quality_menu(imgui)
        elif state == MENU_QUALITY:
            self._draw_quality_menu(imgui)
        elif state == MENU_SUN:
            self._draw_sun_menu(imgui)
        elif state == MENU_CONTROLS:
            self._draw_controls_menu(imgui)
        elif state == MENU_TRACK_SAVE:
            self._draw_track_save_menu(imgui)
        elif state == MENU_SCREENSHOT:
            self._draw_screenshot_menu(imgui)
        elif state == MENU_SCREENSHOT_PREVIEW:
            self._draw_screenshot_preview(imgui)
        elif state == MENU_OPEN_GROUP_PROMPT:
            self._draw_group_prompt(imgui)
        elif state == MENU_OPEN_UNITS_PROMPT:
            self._draw_units_prompt(imgui)
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

    def _nest_display_name(self) -> str:
        renderer = getattr(self, "renderer", None)
        nest = getattr(renderer, "nest", None)
        source = getattr(nest, "source", None)
        return Path(source).name if source else "in-memory nest"

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
            if self.renderer.nested:
                nest_name = self._truncate_middle(self._nest_display_name(), 22)
                if theme.menu_button(f"Remove nest ({nest_name})", "N"):
                    self._remove_nest()
            if theme.menu_button("Behold render command...", "G"):
                self._clipboard_note = None
                self._set_menu_state(MENU_RENDER_QUALITY)
            if theme.menu_button(
                "Time of day...", "T",
                right_text=f"{self.sun_zenith:.0f}deg zenith",
            ):
                self._set_menu_state(MENU_SUN)
            if theme.menu_button("Quality...", "S"):
                self._set_menu_state(MENU_QUALITY)
            if theme.menu_button("Controls...", "C"):
                self._set_menu_state(MENU_CONTROLS)
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
            # Field of view lives here rather than under Quality: it frames
            # the shot, it is not a performance dial, and it is reached often
            # enough that a submenu is one hop too many.
            imgui.dummy((1.0, 8.0))
            theme.caption("Field of view (vertical)")
            changed, fov = imgui.slider_float(
                "##fov", float(self.fov), 30.0, 110.0, "%.0f deg",
            )
            if changed:
                # Accumulation resets by itself: fov is scene identity via
                # the uniform key.
                self.fov = float(fov)

            imgui.dummy((1.0, 8.0))
            theme.mono_text(
                self._truncate_middle(self._field_display_name(), 44),
                size=13.0,
            )
            if self.renderer.nested:
                nest = self.renderer.nest
                refine = self.renderer.dt_view / self.renderer.dt_view_nest
                coverage = self.renderer.nest_coverage_fraction
                theme.mono_text(
                    "+ nest "
                    + self._truncate_middle(self._nest_display_name(), 30)
                    + f"  {refine:.0f}x finer",
                    size=13.0,
                )
                theme.caption(
                    f"{nest.shape[0]}x{nest.shape[1]}x{nest.shape[2]} voxels, "
                    f"covering {coverage * 100:.0f}% of the domain"
                    + (" — little of the outer field is visible"
                       if coverage > 0.75 else "")
                )
            theme.hint_row((
                ("WASD", "fly"),
                ("mouse", "look"),
                ("scroll", "speed"),
                ("F3", "stats"),
            ))
        finally:
            self._end_imgui_window(imgui)

    def _draw_behold_quality_menu(self, imgui) -> None:
        """Hand over a behold command for this view; do not run it.

        Behold is Mitsuba, is minutes-to-overnight, and wants the GPU to
        itself — none of which belongs inside a fly-through. The panel's job
        is to carry the current camera and sun out to a terminal exactly,
        which is also the one thing that works identically in the browser
        build, where there is no Mitsuba at all.
        """
        theme = self._theme
        camera = self.camera()
        quality = self._behold_quality
        command = self._behold_reproduction_command(camera, quality)
        self._begin_imgui_window(imgui, "behold_command", 640.0)
        try:
            theme.header("render in behold", "Command for this view")
            theme.caption(
                "Path-traced rendering runs in a terminal, not here. This "
                "command reproduces exactly what you are looking at — "
                "camera, sun, field and all.",
                wrapped=True,
            )
            if self.periodic and self._view_spans_domain_edge():
                theme.caption(
                    "view spans domain edge — behold does not tile, so its "
                    "frame will differ",
                    wrapped=True,
                )
            if self.renderer.nested:
                # Two fields on screen, and behold renders one. Ask rather
                # than pick: which one is wanted is not derivable from the
                # view — the camera sees both.
                _, chosen, _ = self._behold_target_field()
                imgui.dummy((1.0, 6.0))
                theme.caption(
                    "behold renders one field, not a nested pair — the other "
                    "one will be absent from its frame",
                    wrapped=True,
                )
                theme.caption("Which field")
                for label, key, name, field in (
                    ("Outer field", "O", "outer", self.renderer.field),
                    ("Nested field", "I", "nest", self.renderer.nest),
                ):
                    if theme.menu_button(
                        label, key,
                        sublabel=self._behold_field_summary(field),
                        height=32.0,
                        right_text="selected" if name == chosen else None,
                    ):
                        self._select_behold_field(name)
            imgui.dummy((1.0, 6.0))
            theme.caption("Quality")
            for label, hint, name, note in BEHOLD_QUALITY_ROWS:
                if theme.menu_button(
                    label, hint, sublabel=note, height=32.0,
                    right_text="selected" if name == quality else None,
                ):
                    self._behold_quality = name
            imgui.dummy((1.0, 8.0))
            theme.mono_text(command, size=12.0, wrapped=True)
            imgui.dummy((1.0, 6.0))
            if theme.menu_button("Copy command", "C"):
                self._copy_behold_command()
            if self._clipboard_note:
                theme.caption(self._clipboard_note)
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    @staticmethod
    def _behold_field_summary(field) -> str:
        """One line naming a field the way the picker needs: group, then file."""
        if field is None:
            return ""
        where = field.liquid_group or "root group"
        name = Path(field.source).name if field.source else "in-memory"
        return f"{where} — {name}"

    def _select_behold_field(self, name: str) -> None:
        self._behold_field_choice = name
        self._clipboard_note = None

    def _copy_behold_command(self) -> None:
        command = self._behold_reproduction_command(
            self.camera(), self._behold_quality
        )
        print(command)
        try:
            self._imgui.imgui.set_clipboard_text(command)
        except Exception as e:  # pragma: no cover - platform clipboard
            self._clipboard_note = f"clipboard unavailable ({e}) — printed to the terminal"
            return
        self._clipboard_note = "copied to the clipboard (also printed to the terminal)"

    _CONTROLS_REFERENCE = (
        ("Flying", (
            ("W / S", "forward / back along view"),
            ("A / D", "strafe left / right"),
            ("Space", "climb"),
            ("LShift / C", "descend"),
            ("mouse", "look (Tab releases, click recaptures)"),
            ("scroll", "flight speed"),
        )),
        ("Recording & captures", (
            ("R", "record a flight track; stop to render it to mp4"),
            ("F12", "screenshot — size, folder, overlays; then a preview"),
            ("G (paused)", "copy a behold command for this exact view"),
        )),
        ("Toggles", (
            ("B", "bird"),
            ("M", "minimap"),
            ("F3", "stats readout: subtle / expanded / hidden"),
            ("F", "fullscreen"),
        )),
        ("Menus", (
            ("ESC", "pause menu / back"),
            ("F1 or ?", "this reference"),
            ("O (paused)", "open a NetCDF cloud field"),
            ("N (paused)", "remove a loaded nested field"),
            ("T (paused)", "time of day: sun presets and zenith slider"),
            ("S (paused)", "quality: tier, render scale, smoothing"),
            ("P (paused)", "periodic domain on/off"),
        )),
    )

    def _draw_controls_menu(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "controls_menu", 460.0)
        try:
            theme.header("cloudyview", "Controls")
            from .theme import TEXT_FAINT, TEXT_MUTED

            key_w = 110.0
            for section, rows in self._CONTROLS_REFERENCE:
                imgui.dummy((1.0, 4.0))
                theme.caption(section)
                theme.push_font(theme.font_mono, 13.0)
                for key, what in rows:
                    theme.body_text(key, TEXT_MUTED)
                    imgui.same_line(
                        key_w + imgui.get_style().window_padding.x
                    )
                    theme.body_text(what, TEXT_FAINT)
                theme.pop_font()
            imgui.dummy((1.0, 8.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_track_save_menu(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "track_save_menu", 520.0)
        try:
            theme.header("flight track", "Render this recording?")
            samples = getattr(self, "_track_pending", None) or []
            duration = samples[-1][0] if samples else 0.0
            frames = int(round(duration * self._video_fps))
            theme.body_text(
                f"{duration:.0f} s of flight · {len(samples)} camera samples"
            )
            theme.caption(
                "Saving writes the .json track and then encodes it to mp4 "
                "right here — every frame fully converged, so no motion "
                "speckle. The window shows progress while it runs; the "
                "track file stays behind for re-rendering later with "
                "soar --render-track.",
                wrapped=True,
            )
            imgui.dummy((1.0, 6.0))
            self._draw_capture_settings(imgui, video=True)
            imgui.dummy((1.0, 4.0))
            theme.caption(
                f"~{frames} frames at {self._video_fps:g} fps, "
                f"{self._video_accumulate} passes each"
            )
            imgui.dummy((1.0, 8.0))
            usable = self._save_dir_is_usable()
            if theme.menu_button(
                "Save and render video", "S / Enter", disabled=not usable
            ):
                self._track_save_pending()
            if theme.menu_button("Discard", "D / ESC"):
                self._track_discard_pending()
        finally:
            self._end_imgui_window(imgui)

    def _draw_capture_settings(self, imgui, *, video: bool) -> None:
        """Size and destination controls, shared by both capture dialogs.

        A screenshot and a video are the same decision twice — how big, and
        where does it go — so they get the same controls rather than two
        drifting copies.
        """
        theme = self._theme
        win_w, win_h = self.canvas.get_physical_size()
        size = self.capture_size()

        theme.caption("Size")
        if theme.menu_button(
            f"Window ({int(win_w)} x {int(win_h)})", None, height=32.0,
            right_text="selected" if self._capture_size is None else None,
        ):
            self._set_capture_size(None)
        for label, preset in CAPTURE_SIZE_PRESETS:
            selected = self._capture_size == preset
            if theme.menu_button(
                label, None, height=32.0,
                right_text="selected" if selected else None,
            ):
                self._set_capture_size(preset)

        imgui.set_next_item_width(200.0)
        changed, values = imgui.input_int2("##capture_size", list(size))
        if changed:
            self._set_capture_size(values)
        imgui.same_line()
        theme.caption("custom w x h")

        imgui.dummy((1.0, 6.0))
        theme.caption("Save to")
        imgui.set_next_item_width(400.0)
        changed, text = imgui.input_text("##save_dir", self._save_dir_text)
        if changed and not self._set_save_dir(text):
            # Keep the typed text so it can be finished; the render buttons
            # below refuse to fire until it resolves to a real directory.
            pass
        if not Path(self._save_dir_text).expanduser().is_dir():
            theme.caption("Not a directory — captures will not be saved here.")

        if video:
            imgui.dummy((1.0, 6.0))
            theme.caption("Frame rate")
            imgui.set_next_item_width(200.0)
            changed, fps = imgui.slider_float(
                "##video_fps", float(self._video_fps), 12.0, 120.0, "%.0f fps"
            )
            if changed:
                self._video_fps = float(fps)
            theme.caption("Accumulation passes per frame")
            imgui.set_next_item_width(200.0)
            changed, passes = imgui.slider_int(
                "##video_accum", int(self._video_accumulate), 1, 64, "%d"
            )
            if changed:
                self._video_accumulate = int(passes)

    def _save_dir_is_usable(self) -> bool:
        return Path(self._save_dir_text).expanduser().is_dir()

    def _draw_screenshot_menu(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "screenshot_menu", 520.0)
        try:
            theme.header("screenshot", "What goes in the frame?")
            theme.caption(
                "Both save the same converged render and the same "
                "reproduction metadata; only the overlays differ.",
                wrapped=True,
            )
            imgui.dummy((1.0, 6.0))
            self._draw_capture_settings(imgui, video=False)

            imgui.dummy((1.0, 8.0))
            usable = self._save_dir_is_usable()
            if theme.menu_button(
                "With bird and location map", "W / Enter", disabled=not usable
            ):
                self._pending_screenshot = True
            if theme.menu_button("Clouds only", "C", disabled=not usable):
                self._pending_screenshot = False
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Cancel", "ESC", height=38.0):
                self._set_menu_state(MENU_MAIN)
                self._set_paused(False)
        finally:
            self._end_imgui_window(imgui)

    def _draw_screenshot_preview(self, imgui) -> None:
        preview = self._preview
        if preview is None:
            self._set_menu_state(MENU_MAIN)
            return
        theme = self._theme
        image = preview["image"]
        height, width = image.shape[:2]
        max_w, max_h = 760.0, 460.0
        scale = min(max_w / width, max_h / height, 1.0)
        draw_w, draw_h = width * scale, height * scale

        self._begin_imgui_window(imgui, "screenshot_preview", draw_w + 48.0)
        try:
            theme.header("screenshot", "Saved")
            imgui.image(preview["ref"], imgui.ImVec2(draw_w, draw_h))
            theme.mono_text(str(preview["path"]), size=12.0)
            theme.caption(f"{width} x {height} px")
            imgui.dummy((1.0, 6.0))
            if theme.menu_button("Close", "ESC / Enter", height=38.0):
                self._close_preview()
        finally:
            self._end_imgui_window(imgui)

    def _release_preview(self) -> None:
        """Drop the preview texture without touching menu/pause state."""
        preview = self._preview
        self._preview = None
        if preview is not None and self._imgui is not None:
            self._imgui.release_image(preview["ref"])

    def _close_preview(self) -> None:
        """Drop the preview and go back to flying."""
        self._release_preview()
        self._set_menu_state(MENU_MAIN)
        self._set_paused(False)

    _SUN_DIRECTION_LABELS = (
        (22.5, "N"), (67.5, "NE"), (112.5, "E"), (157.5, "SE"),
        (202.5, "S"), (247.5, "SW"), (292.5, "W"), (337.5, "NW"),
    )

    def _sun_compass_label(self) -> str:
        azimuth = self.sun_azimuth % 360.0
        for edge, label in self._SUN_DIRECTION_LABELS:
            if azimuth < edge:
                return label
        return "N"

    def _draw_sun_menu(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "sun_menu", 520.0)
        try:
            theme.header("time of day", "Sun")
            for index, name in enumerate(SUN_PRESETS):
                azimuth, elevation = SUN_PRESETS[name]
                selected = (
                    abs(self.sun_elevation - elevation) < 0.05
                    and abs((self.sun_azimuth - azimuth) % 360.0) < 0.05
                )
                if theme.menu_button(
                    name.capitalize(), str(index + 1),
                    right_text="selected" if selected else None,
                ):
                    self._select_sun_preset(name)

            imgui.dummy((1.0, 8.0))
            theme.caption("Solar zenith angle")
            changed, zenith = imgui.slider_float(
                "##sun_zenith",
                self.sun_zenith,
                0.0,
                90.0 - MIN_SUN_ELEVATION_DEG,
                "%.1f deg",
            )
            if changed:
                self._set_sun(zenith=zenith)

            # Azimuth matters as much as zenith for the look at low sun (the
            # warm horizon wedge is azimuth-dependent), and a zenith slider
            # with no way to move the sun round the sky is half a control.
            theme.caption("Solar azimuth")
            changed, azimuth = imgui.slider_float(
                "##sun_azimuth",
                self.sun_azimuth,
                0.0,
                360.0,
                "%.0f deg",
            )
            if changed:
                self._set_sun(azimuth=azimuth)

            theme.body_text(
                f"zenith {self.sun_zenith:.1f} deg  ·  "
                f"elevation {self.sun_elevation:.1f} deg  ·  "
                f"azimuth {self.sun_azimuth:.0f} deg "
                f"({self._sun_compass_label()})"
            )
            if self.sun_elevation <= MIN_SUN_ELEVATION_DEG + 1e-6:
                theme.caption(
                    "At the horizon. The sun cannot go below it while the "
                    "domain is periodic — the light march exits through the "
                    "domain top.",
                    wrapped=True,
                )
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_quality_menu(self, imgui) -> None:
        from .theme import TEXT_FAINT

        theme = self._theme
        renderer = self.renderer
        self._begin_imgui_window(imgui, "quality_menu", 520.0)
        try:
            theme.header("performance", "Quality")
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

            theme.caption("Tone-map gamma")
            changed, gamma = imgui.slider_float(
                "##tone_map_gamma",
                self.tone_map_gamma,
                *TONE_MAP_GAMMA_LIMITS,
                "%.2f",
            )
            if changed:
                self._set_tone_map_gamma(gamma)
            theme.caption(
                f"{TONE_MAP_GAMMA_WITNESS:.1f} is witness's own value — "
                "darker, harder far field. Higher lifts distance into haze; "
                f"{TONE_MAP_GAMMA_AS_FLOWN:.2f} is what the window used to "
                "render by encoding gamma twice.",
                wrapped=True,
            )

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

    def _pending_open_filename(self) -> str:
        return (
            Path(self._pending_open_path).name
            if self._pending_open_path else "selected file"
        )

    def _draw_group_prompt(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "group_prompt", 460.0)
        try:
            theme.header("open file", "Which group?")
            theme.mono_text(self._pending_open_filename(), size=13.0)
            theme.caption(
                "The root group holds no cloud field. These groups do:",
                wrapped=True,
            )
            imgui.dummy((1.0, 6.0))
            pairs = self._pending_nest_pairs
            if len(pairs) == 1:
                outer_group, inner_group = pairs[0]
                if theme.menu_button("Use both, nested", "B"):
                    self._select_both_groups_nested(0)
                theme.caption(
                    f"'{inner_group}' lies inside '{outer_group}' and is "
                    "finer — they can be rendered together, with the finer "
                    "one taking over where it covers.",
                    wrapped=True,
                )
                imgui.dummy((1.0, 6.0))
                theme.caption("Or just one:")
            elif pairs:
                # More than two nesting levels: every pair is a legitimate
                # scene and only the user knows which two they came for.
                theme.caption(
                    "These nest — two levels render together, the finer one "
                    "taking over where it covers:",
                    wrapped=True,
                )
                for index, (outer_group, inner_group) in enumerate(pairs):
                    hint = PAIR_KEY_BY_INDEX.get(index)
                    label = f"{outer_group}  +  {inner_group}"
                    if theme.menu_button(label, hint):
                        self._select_both_groups_nested(index)
                imgui.dummy((1.0, 6.0))
                theme.caption("Or just one:")
            for index, group in enumerate(self._pending_group_choices):
                hint = str(index + 1) if index < 9 else None
                if theme.menu_button(group, hint):
                    self._select_group(index)
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._reset_pending_open()
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_units_prompt(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "units_prompt", 460.0)
        try:
            theme.header("open file", "Which units?")
            theme.mono_text(self._pending_open_filename(), size=13.0)
            theme.caption(
                f"No units attribute on {', '.join(self._pending_units_vars)}. "
                "Mixing ratios are usually kg/kg; SAM-style output is g/kg.",
                wrapped=True,
            )
            imgui.dummy((1.0, 6.0))
            for label, hint, units in (
                ("g/kg", "G", "g/kg"),
                ("kg/kg", "K", "kg/kg"),
            ):
                if theme.menu_button(label, hint):
                    self._select_condensate_units(units)
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._reset_pending_open()
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_ice_prompt(self, imgui) -> None:
        theme = self._theme
        self._begin_imgui_window(imgui, "ice_prompt", 460.0)
        try:
            filename = self._pending_open_filename()
            theme.header("open file", "Ice phase?")
            theme.mono_text(filename, size=13.0)
            if self._pending_group:
                theme.caption(f"group: {self._pending_group}")
            imgui.dummy((1.0, 6.0))
            if theme.menu_button("Yes, pick ice file", "Y"):
                self._finish_open_file(use_ice=True)
            if theme.menu_button("No ice", "N"):
                self._finish_open_file(use_ice=False)
            imgui.dummy((1.0, 4.0))
            if theme.menu_button("Back", "ESC", height=38.0):
                self._reset_pending_open()
                self._set_menu_state(MENU_MAIN)
        finally:
            self._end_imgui_window(imgui)

    def _draw_file_browser(self, imgui, state: str) -> None:
        theme = self._theme
        title = (
            "Open ice file" if state == MENU_FILE_BROWSER_ICE
            else "Open cloud file"
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
                    self._reset_pending_open()
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
                recording = getattr(self, "_track_recording", False)
                if recording:
                    # Pulsing red record dot ahead of the readout.
                    pulse = 0.55 + 0.45 * np.sin(perf_counter() * 4.0)
                    draw = imgui.get_window_draw_list()
                    pos = imgui.get_cursor_screen_pos()
                    center = (pos.x + 6.0, pos.y + 8.0)
                    draw.add_circle_filled(
                        center, 5.0,
                        imgui.color_convert_float4_to_u32(
                            (0.92, 0.18, 0.15, pulse)
                        ),
                    )
                    imgui.dummy((16.0, 1.0))
                    imgui.same_line()
                speed_note = (
                    f" · {self.speed:.0f} m/s"
                    if perf_counter() < getattr(
                        self, "_speed_flash_until", 0.0
                    )
                    else ""
                )
                theme.mono_text(
                    f"{fps:.0f} fps · {frame_ms:.1f} ms{speed_note}",
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
                    ("flags", f"map {'on' if self.minimap_enabled else 'off'}"
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
        # The video encode gets this frame's time before anything else; the
        # window is only here to show progress while it runs.
        if self._video_render is not None:
            self._step_video_render()
        # A screenshot requested by clicking the prompt is taken here, not in
        # the imgui callback that raised it: that callback runs with a command
        # encoder open, and the screenshot render submits its own.
        if self._pending_screenshot is not None:
            overlays, self._pending_screenshot = self._pending_screenshot, None
            self._save_screenshot(overlays=overlays)
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
        if self._video_render is None:
            self.renderer.write_uniforms(
                camera, (w, h), jitter=self.jitter,
                sun_azimuth=self.sun_azimuth,
                sun_elevation=self.sun_elevation,
                frame_index=self._frame_index,
                **self._cb_strength_kwargs())
        if (getattr(self, "_track_recording", False) and not self._paused
                and self._video_render is None):
            self._track_samples.append([
                perf_counter() - self._track_t0,
                *camera.position,
                camera.azimuth, camera.elevation, camera.fov,
            ])
        self._frame_index += 1

        texture = self.context.get_current_texture()
        view = texture.create_view()
        enc = self.renderer.device.create_command_encoder()
        if self._video_render is not None:
            # The video encode owns the GPU: blit the frozen last frame
            # instead of marching the volume again for the window (near-zero
            # cost). Falls back to a normal pass when no accumulated frame
            # exists (jitter off).
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
