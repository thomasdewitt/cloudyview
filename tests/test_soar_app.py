"""Headless tests for the soar window app shell."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu", reason="requires the 'interactive' extra")

from cloudyview.soar.app import (
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
    ACTION_SELECT_TIER,
    ACTION_SETTINGS_MENU,
    ACTION_TOGGLE_FULLSCREEN,
    BEHOLD_QUALITIES_BY_KEY,
    MENU_FILE_BROWSER_ICE,
    MENU_FILE_BROWSER_LIQUID,
    FlyThroughApp,
    MENU_MAIN,
    MENU_OPEN_ICE_PROMPT,
    MENU_RENDER_QUALITY,
    MENU_SETTINGS,
    OCEAN_FLOOR_MARGIN_M,
    _clamp_position_above_ocean,
    _control_action_for_key,
    _menu_transition,
)
from cloudyview.soar.jobs import BackgroundJob
from cloudyview.soar.menu import format_file_size, list_netcdf_entries


class DummyCanvas:
    def __init__(self):
        self.closed = False
        self.titles = []
        self.draw_requests = 0

    def close(self):
        self.closed = True

    def set_title(self, title):
        self.titles.append(title)

    def request_draw(self, *args):
        self.draw_requests += 1

    def get_physical_size(self):
        return 1280, 720


def make_event_app():
    app = object.__new__(FlyThroughApp)
    app.canvas = DummyCanvas()
    app._paused = False
    app._menu_state = MENU_MAIN
    app._pending_open_path = None
    app._file_browser_dir = Path.cwd()
    app._last_file_dir = Path.cwd()
    app._file_browser_error = None
    app._loading_job = None
    app._behold_job = None
    app._rendering = False
    app._error_message = None
    app._imgui = None
    app._captured = True
    app._keys = {"w", "Shift"}
    app._last_pointer = (10.0, 20.0)
    app._fullscreen = False
    app._frame_index = 12
    app._last_time = 0.0
    app.jitter = True
    app.bird_enabled = True
    app.minimap_enabled = True
    app.speed = 60.0
    app.capture_calls = []
    app.fullscreen_calls = 0
    app.open_calls = 0
    app.render_calls = []
    app.screenshot_calls = 0
    app.tier_calls = []

    def capture_mouse(capture):
        app.capture_calls.append(capture)
        app._captured = capture
        app._last_pointer = None

    def toggle_fullscreen():
        app.fullscreen_calls += 1
        app._fullscreen = not app._fullscreen

    app._capture_mouse = capture_mouse
    app._toggle_fullscreen = toggle_fullscreen
    app._start_open_file = lambda: setattr(app, "open_calls", app.open_calls + 1)
    app._run_behold_render = lambda quality: app.render_calls.append(quality)
    app._save_screenshot = lambda: setattr(
        app, "screenshot_calls", app.screenshot_calls + 1
    )
    app._select_quality_tier = lambda tier: app.tier_calls.append(tier)
    return app


def test_control_action_for_key_active_and_paused():
    assert _control_action_for_key(False, "Escape") == ACTION_PAUSE
    assert _control_action_for_key(False, "f") == ACTION_TOGGLE_FULLSCREEN
    assert _control_action_for_key(False, "F") == ACTION_TOGGLE_FULLSCREEN
    assert _control_action_for_key(False, "F12") == ACTION_SCREENSHOT
    assert _control_action_for_key(False, "w") is None

    assert _control_action_for_key(True, "Escape") == ACTION_RESUME
    assert _control_action_for_key(True, "q") == ACTION_QUIT
    assert _control_action_for_key(True, "Q") == ACTION_QUIT
    assert _control_action_for_key(True, "r") == ACTION_RESUME
    assert _control_action_for_key(True, "R") == ACTION_RESUME
    assert _control_action_for_key(True, "F") == ACTION_TOGGLE_FULLSCREEN
    assert _control_action_for_key(True, "O") == ACTION_OPEN_FILE
    assert _control_action_for_key(True, "G") == ACTION_RENDER_MENU
    assert _control_action_for_key(True, "w") is None


def test_pause_submenu_state_transitions_are_explicit():
    transition = _menu_transition(True, MENU_MAIN, "O")
    assert transition.action == ACTION_OPEN_FILE
    assert transition.next_state == MENU_FILE_BROWSER_LIQUID

    transition = _menu_transition(True, MENU_MAIN, "G")
    assert transition.action == ACTION_RENDER_MENU
    assert transition.next_state == MENU_RENDER_QUALITY

    for key, quality in BEHOLD_QUALITIES_BY_KEY.items():
        transition = _menu_transition(True, MENU_RENDER_QUALITY, key)
        assert transition.action == ACTION_RENDER_BEHOLD
        assert transition.quality == quality
        assert transition.next_state == MENU_RENDER_QUALITY

    transition = _menu_transition(True, MENU_RENDER_QUALITY, "Escape")
    assert transition.action == ACTION_MENU_BACK
    assert transition.next_state == MENU_MAIN
    assert _menu_transition(True, MENU_RENDER_QUALITY, "q").action is None

    assert (
        _menu_transition(True, MENU_OPEN_ICE_PROMPT, "Y").action
        == ACTION_OPEN_ICE_YES
    )
    assert (
        _menu_transition(True, MENU_OPEN_ICE_PROMPT, "Y").next_state
        == MENU_FILE_BROWSER_ICE
    )
    assert (
        _menu_transition(True, MENU_OPEN_ICE_PROMPT, "N").action
        == ACTION_OPEN_ICE_NO
    )
    transition = _menu_transition(True, MENU_OPEN_ICE_PROMPT, "Escape")
    assert transition.action == ACTION_MENU_BACK
    assert transition.next_state == MENU_MAIN
    assert _menu_transition(True, MENU_FILE_BROWSER_ICE, "Escape").next_state == (
        MENU_OPEN_ICE_PROMPT
    )


def test_settings_submenu_transitions_are_explicit():
    transition = _menu_transition(True, MENU_MAIN, "S")
    assert transition.action == ACTION_SETTINGS_MENU
    assert transition.next_state == MENU_SETTINGS

    for key, tier in zip(("1", "2", "3", "4"),
                         ("high", "medium", "low", "potato")):
        transition = _menu_transition(True, MENU_SETTINGS, key)
        assert transition.action == ACTION_SELECT_TIER
        assert transition.next_state == MENU_SETTINGS
        assert transition.tier == tier

    transition = _menu_transition(True, MENU_SETTINGS, "Escape")
    assert transition.action == ACTION_MENU_BACK
    assert transition.next_state == MENU_MAIN


def test_settings_key_dispatches_tier_selection():
    app = make_event_app()
    app._paused = True
    app._menu_state = MENU_SETTINGS
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "3"})
    assert app.tier_calls == ["low"]


def test_startup_auto_benchmark_selects_highest_60_fps_tier():
    timings = {"potato": 4.0, "low": 8.0, "medium": 15.0, "high": 24.0}

    class FakeRenderer:
        def __init__(self):
            self.quality_tier = "high"
            self.calls = []
            self.reset_calls = 0

        def set_quality_tier(self, name, *, camera_moving):
            self.quality_tier = name
            self.calls.append((name, camera_moving))

        def benchmark(self, *args, **kwargs):
            return {
                "timestamps_used": True,
                "gpu_ms_mean": timings[self.quality_tier],
                "wall_ms_mean": timings[self.quality_tier] + 1.0,
            }

        def reset_accumulation(self):
            self.reset_calls += 1

    app = object.__new__(FlyThroughApp)
    app.canvas = DummyCanvas()
    app.renderer = FakeRenderer()
    app.camera = lambda: object()

    FlyThroughApp._run_startup_tier_benchmark(app)

    assert app.renderer.quality_tier == "medium"
    assert app._auto_benchmark_ms == timings
    assert app.renderer.calls[-1] == ("medium", False)
    assert app.renderer.reset_calls == 1


def test_clamp_position_above_ocean_uses_margin_and_returns_copy():
    original = np.array([10.0, 20.0, -50.0])
    clamped = _clamp_position_above_ocean(original)

    assert clamped.tolist() == [10.0, 20.0, OCEAN_FLOOR_MARGIN_M]
    assert original.tolist() == [10.0, 20.0, -50.0]

    # probe must sit above OCEAN_FLOOR_MARGIN_M (50 m since 2026-07-10)
    already_above = _clamp_position_above_ocean([1.0, 2.0, 80.0])
    assert already_above.tolist() == [1.0, 2.0, 80.0]


def test_escape_pauses_releases_capture_and_clears_movement_keys():
    app = make_event_app()

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "Escape"})

    assert app._paused is True
    assert app._captured is False
    assert app._keys == set()
    assert app.capture_calls == [False]
    assert app.canvas.closed is False
    assert "paused" in app.canvas.titles[-1]
    assert app.canvas.draw_requests == 1


def test_resume_key_recaptures_pointer_and_paused_click_does_not_resume():
    app = make_event_app()
    app._paused = True
    app._captured = False
    app._keys = {"w"}

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "R"})

    assert app._paused is False
    assert app._captured is True
    assert app._keys == set()
    assert app.capture_calls == [True]

    app._paused = True
    app._captured = False
    app.capture_calls.clear()

    FlyThroughApp._on_event(app, {"event_type": "pointer_down"})

    assert app._paused is True
    assert app._captured is False
    assert app.capture_calls == []

    app._paused = False
    FlyThroughApp._on_event(app, {"event_type": "pointer_down"})

    assert app._captured is True
    assert app.capture_calls == [True]


def test_paused_escape_resumes_and_q_quits_without_recapturing():
    app = make_event_app()
    app._paused = True
    app._captured = False

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "Escape"})

    assert app.canvas.closed is False
    assert app._paused is False
    assert app.capture_calls == [True]

    for key in ("q", "Q"):
        app = make_event_app()
        app._paused = True
        app._captured = False

        FlyThroughApp._on_event(app, {"event_type": "key_down", "key": key})

        assert app.canvas.closed is True
        assert app.capture_calls == []


def test_fullscreen_key_toggles_without_becoming_movement_input():
    app = make_event_app()
    app._keys = set()

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "F"})

    assert app.fullscreen_calls == 1
    assert app._keys == set()

    app._paused = True
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "F"})

    assert app.fullscreen_calls == 2
    assert app._paused is True
    assert app._keys == set()


def test_pause_open_and_render_keys_dispatch_to_submenus():
    app = make_event_app()
    app._paused = True
    app._captured = False
    app._keys = set()

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "O"})
    assert app.open_calls == 1

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "G"})
    assert app._menu_state == MENU_RENDER_QUALITY
    assert "behold quality" in app.canvas.titles[-1]

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "2"})
    assert app.render_calls == ["low"]


def test_f12_dispatches_screenshot_during_active_flight():
    app = make_event_app()
    app._paused = False

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "F12"})

    assert app.screenshot_calls == 1


def test_paused_mouse_move_and_wheel_do_not_change_camera_or_speed():
    app = make_event_app()
    app._paused = True
    app.azimuth = 20.0
    app.elevation = -10.0
    app.speed = 60.0

    FlyThroughApp._on_event(
        app, {"event_type": "pointer_move", "x": 30.0, "y": 40.0}
    )
    FlyThroughApp._on_event(app, {"event_type": "wheel", "dy": -100.0})

    assert app.azimuth == 20.0
    assert app.elevation == -10.0
    assert app.speed == 60.0


def test_move_clamps_shift_descent_to_ocean_margin():
    app = object.__new__(FlyThroughApp)
    app._paused = False
    app.periodic = False
    app._keys = {"Shift"}
    app.position = np.array([0.0, 0.0, OCEAN_FLOOR_MARGIN_M + 0.25])
    app.speed = 60.0
    app.azimuth = 0.0
    app.elevation = 0.0
    app.fov = 80.0

    FlyThroughApp._move(app, 1.0)

    assert app.position[2] == OCEAN_FLOOR_MARGIN_M


def test_install_field_rebuilds_renderer_on_existing_device_and_resets_camera():
    class FakeRenderer:
        def __init__(self, name):
            self.name = name
            self.device = object()
            self.bmin = np.array([0.0, 0.0, 0.0])
            self.bmax = np.array([100.0, 200.0, 1000.0])
            self.ocean_enabled = True
            self.ocean_z = 0.0
            self.ocean_reflectance = (0.1, 0.2, 0.3)
            self.ocean_fif_normals = ("nx", "ny", "nz", 1.0)
            self.reset_calls = 0

        def reset_accumulation(self):
            self.reset_calls += 1

    app = object.__new__(FlyThroughApp)
    old_renderer = FakeRenderer("old")
    new_renderer = FakeRenderer("new")
    app.renderer = old_renderer
    app._frame_index = 99
    calls = []

    def create_renderer(field, *, device=None, previous=None):
        calls.append((field, device, previous))
        return new_renderer

    app._create_renderer = create_renderer

    FlyThroughApp._install_field(app, field="new-field")

    assert calls == [("new-field", old_renderer.device, old_renderer)]
    assert app.renderer is new_renderer
    assert app._frame_index == 0
    assert app.azimuth == pytest.approx(0.0)
    assert app.elevation == pytest.approx(35.0)
    assert app.fov == pytest.approx(100.0)
    assert app.position[2] >= OCEAN_FLOOR_MARGIN_M
    assert new_renderer.reset_calls == 1


def test_move_clamps_forward_descent_when_pitched_down():
    app = object.__new__(FlyThroughApp)
    app._paused = False
    app.periodic = False
    app._keys = {"w"}
    app.position = np.array([0.0, 0.0, OCEAN_FLOOR_MARGIN_M + 0.25])
    app.speed = 60.0
    app.azimuth = 0.0
    app.elevation = -45.0
    app.fov = 80.0

    FlyThroughApp._move(app, 1.0)

    assert app.position[2] == OCEAN_FLOOR_MARGIN_M


def test_file_browser_filters_netcdf_files_and_formats_sizes(tmp_path):
    (tmp_path / "nested").mkdir()
    nc = tmp_path / "cloud.nc"
    nc.write_bytes(b"0" * 1536)
    (tmp_path / "notes.txt").write_text("ignore")

    entries = list_netcdf_entries(tmp_path)

    assert [entry.name for entry in entries] == ["nested", "cloud.nc"]
    assert entries[0].is_dir is True
    assert entries[1].display_size == "1.5 KB"
    assert format_file_size(4 * 1024**3) == "4.0 GB"


def test_select_browser_path_drives_liquid_and_ice_handoff(tmp_path):
    app = make_event_app()
    liquid = tmp_path / "liquid.nc"
    ice = tmp_path / "ice.nc"
    liquid.touch()
    ice.touch()
    calls = []

    app._start_loading_file = lambda liquid_path, ice_path: calls.append(
        (liquid_path, ice_path)
    )

    app._menu_state = MENU_FILE_BROWSER_LIQUID
    FlyThroughApp._select_browser_path(app, liquid)

    assert app._pending_open_path == str(liquid.resolve())
    assert app._menu_state == MENU_OPEN_ICE_PROMPT

    app._menu_state = MENU_FILE_BROWSER_ICE
    FlyThroughApp._select_browser_path(app, ice)

    assert calls == [(str(liquid.resolve()), str(ice.resolve()))]


def test_background_job_progress_and_result_handoff():
    def target(report):
        report("loading file")
        report("building extinction", percent=50.0)
        return "renderer"

    job = BackgroundJob(
        kind="loading",
        filename="cloud.nc",
        target=target,
        initial_stage="queued",
    )
    job.start()
    job.join(2.0)
    snapshot = job.snapshot()

    assert snapshot.done is True
    assert snapshot.error is None
    assert snapshot.result == "renderer"
    assert snapshot.stage == "building extinction"
    assert snapshot.percent == pytest.approx(50.0)


def test_async_load_job_installs_fake_renderer(tmp_path):
    class FakeRenderer:
        def __init__(self, field):
            self.field = field
            self.device = object()
            self.bmin = np.array([0.0, 0.0, 0.0])
            self.bmax = np.array([100.0, 200.0, 1000.0])
            self.reset_calls = 0

        def reset_accumulation(self):
            self.reset_calls += 1

    app = make_event_app()
    app._extinction_multiplier = 1.0
    app.renderer = FakeRenderer("old")
    app._frame_index = 5
    app._paused = True
    app._captured = False
    app.canvas.titles.clear()

    stages = []

    def fake_load(path, *, ice=None, stage_callback=None):
        if stage_callback is not None:
            stage_callback("loading file")
        stages.append((path, ice))
        return "new-field"

    def fake_create_renderer(field, *, device=None, previous=None):
        assert field == "new-field"
        assert previous.field == "old"
        return FakeRenderer(field)

    app._create_renderer = fake_create_renderer
    import cloudyview.soar.app as soar_app

    original_load = soar_app.load_cloud_field
    soar_app.load_cloud_field = fake_load
    try:
        FlyThroughApp._start_loading_file(app, tmp_path / "cloud.nc", None)
        app._loading_job.join(2.0)
        FlyThroughApp._pump_jobs(app)
    finally:
        soar_app.load_cloud_field = original_load

    assert stages == [(str(tmp_path / "cloud.nc"), None)]
    assert app.renderer.field == "new-field"
    assert app.renderer.reset_calls == 1
    assert app._frame_index == 0
    assert app._paused is False
    assert app.capture_calls == [True]
