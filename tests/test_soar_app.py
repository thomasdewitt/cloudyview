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
    ACTION_COPY_BEHOLD_COMMAND,
    ACTION_SELECT_BEHOLD_QUALITY,
    ACTION_RENDER_MENU,
    ACTION_RESUME,
    ACTION_SCREENSHOT,
    ACTION_SELECT_TIER,
    ACTION_QUALITY_MENU,
    ACTION_TOGGLE_FULLSCREEN,
    BEHOLD_QUALITIES_BY_KEY,
    MENU_FILE_BROWSER_ICE,
    MENU_FILE_BROWSER_LIQUID,
    FlyThroughApp,
    MENU_MAIN,
    MENU_OPEN_GROUP_PROMPT,
    MENU_OPEN_ICE_PROMPT,
    MENU_OPEN_UNITS_PROMPT,
    MENU_RENDER_QUALITY,
    MENU_SCREENSHOT,
    MENU_QUALITY,
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
    app._pending_ice_path = None
    app._pending_group = None
    app._pending_group_choices = []
    app._pending_units = None
    app._pending_units_vars = []
    app._pending_is_nest = False
    app._pending_nest_group = None
    app._pending_nest_pairs = []
    app._file_browser_dir = Path.cwd()
    app._last_file_dir = Path.cwd()
    app._file_browser_error = None
    app._loading_job = None
    app._video_render = None
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
    app._behold_quality = "high"
    app._clipboard_note = None
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
    app.screenshot_overlays = []

    def save_screenshot(*, overlays=True):
        app.screenshot_calls += 1
        app.screenshot_overlays.append(overlays)
        app._menu_state = MENU_MAIN
        app._paused = False

    app._save_screenshot = save_screenshot
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
        assert transition.action == ACTION_SELECT_BEHOLD_QUALITY
        assert transition.quality == quality
        assert transition.next_state == MENU_RENDER_QUALITY

    transition = _menu_transition(True, MENU_RENDER_QUALITY, "Escape")
    assert transition.action == ACTION_MENU_BACK
    assert transition.next_state == MENU_MAIN
    assert _menu_transition(True, MENU_RENDER_QUALITY, "q").action is None
    assert (
        _menu_transition(True, MENU_RENDER_QUALITY, "c").action
        == ACTION_COPY_BEHOLD_COMMAND
    )

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
    assert transition.action == ACTION_QUALITY_MENU
    assert transition.next_state == MENU_QUALITY

    for key, tier in zip(("1", "2", "3", "4"),
                         ("high", "medium", "low", "potato")):
        transition = _menu_transition(True, MENU_QUALITY, key)
        assert transition.action == ACTION_SELECT_TIER
        assert transition.next_state == MENU_QUALITY
        assert transition.tier == tier

    transition = _menu_transition(True, MENU_QUALITY, "Escape")
    assert transition.action == ACTION_MENU_BACK
    assert transition.next_state == MENU_MAIN


def test_settings_key_dispatches_tier_selection():
    app = make_event_app()
    app._paused = True
    app._menu_state = MENU_QUALITY
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
    app.renderer = FakeRenderer()
    app.camera = lambda: object()

    # Runs pre-window since 2026-07-17 (GNOME watchdog fix): the requested
    # window size is passed in; no canvas exists yet.
    FlyThroughApp._run_startup_tier_benchmark(app, (1280, 720))

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

    # The G panel hands over a command; the number keys only choose which
    # quality it names.
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "2"})
    assert app._behold_quality == "low"
    assert app._menu_state == MENU_RENDER_QUALITY


def test_n_removes_a_loaded_nest():
    """N is removal only — a nest arrives with the file it lives in."""
    from types import SimpleNamespace

    app = make_event_app()
    app._paused = True
    app.renderer = SimpleNamespace(nested=True)
    app.remove_calls = 0
    app._remove_nest = lambda: setattr(
        app, "remove_calls", app.remove_calls + 1
    )

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "N"})

    assert app.remove_calls == 1


def test_n_is_harmless_with_no_nest_loaded():
    from types import SimpleNamespace

    app = make_event_app()
    app._paused = True
    app.renderer = SimpleNamespace(nested=False, nest=None)
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "N"})
    assert app._menu_state == MENU_MAIN


def test_f12_opens_the_screenshot_prompt_during_active_flight():
    """F12 asks what belongs in the frame instead of shooting immediately."""
    app = make_event_app()
    app._paused = False

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "F12"})

    assert app.screenshot_calls == 0
    assert app._paused is True
    assert app._menu_state == MENU_SCREENSHOT


@pytest.mark.parametrize(
    "key, overlays",
    [("w", True), ("W", True), ("Enter", True), ("1", True),
     ("c", False), ("C", False), ("2", False)],
)
def test_screenshot_prompt_keys_choose_the_overlays(key, overlays):
    app = make_event_app()
    app._paused = False
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "F12"})
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": key})

    assert app.screenshot_calls == 1
    assert app.screenshot_overlays == [overlays]


def test_screenshot_prompt_escape_cancels_without_saving():
    app = make_event_app()
    app._paused = False
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "F12"})
    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "Escape"})

    assert app.screenshot_calls == 0
    assert app._menu_state == MENU_MAIN


def test_f3_cycles_corner_stats_without_becoming_movement_input():
    app = make_event_app()
    app._paused = False
    app._keys = set()

    assert getattr(app, "_stats_mode", "subtle") == "subtle"
    for expected in ("expanded", "hidden", "subtle"):
        FlyThroughApp._on_event(
            app, {"event_type": "key_down", "key": "F3"}
        )
        assert app._stats_mode == expected
        assert app._keys == set()


def test_runtime_diagnostics_are_not_duplicated_in_window_title():
    app = make_event_app()

    assert FlyThroughApp._paused_title(app, 120.0) == "cloudyview paused"


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


def test_file_browser_hides_dotfiles(tmp_path):
    """A home directory is mostly ~/.cache; none of it holds cloud fields."""
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".hidden.nc").write_bytes(b"0")
    (tmp_path / "cloud.nc").write_bytes(b"0")

    entries = list_netcdf_entries(tmp_path)

    assert [entry.name for entry in entries] == ["cloud.nc"]


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
            self.nest = None
            self.nested = False
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

    def fake_load(path, *, ice=None, liquid_water_group=None,
                  ice_water_group=None, fallback_units=None,
                  stage_callback=None):
        if stage_callback is not None:
            stage_callback("loading file")
        stages.append(
            (path, ice, liquid_water_group, ice_water_group, fallback_units)
        )
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

    assert stages == [(str(tmp_path / "cloud.nc"), None, None, None, None)]
    assert app.renderer.field == "new-field"
    assert app.renderer.reset_calls == 1
    assert app._frame_index == 0
    assert app._paused is False
    assert app.capture_calls == [True]


def _write_nested_render_file(path, groups=("render_a", "render_b"),
                              with_units=True):
    """A file shaped like a STEAM render nest: empty root, field per group."""
    import xarray as xr

    xr.Dataset().to_netcdf(path, mode="w")
    attrs = {"units": "kg/kg"} if with_units else {}
    for group in groups:
        xr.Dataset(
            data_vars={
                "qc": (
                    ("x", "y", "z"),
                    np.full((2, 3, 4), 0.001, dtype=np.float64),
                    dict(attrs),
                ),
            },
            coords={
                "x": np.array([0.0, 1000.0]),
                "y": np.array([0.0, 2000.0, 4000.0]),
                "z": np.array([100.0, 300.0, 800.0, 1600.0]),
            },
        ).to_netcdf(path, mode="a", group=group)


def test_single_candidate_group_is_chosen_without_asking(tmp_path):
    app = make_event_app()
    nested = tmp_path / "one_nest.nc"
    _write_nested_render_file(nested, groups=("render_a",))

    app._menu_state = MENU_FILE_BROWSER_LIQUID
    FlyThroughApp._select_browser_path(app, nested)

    assert app._pending_group == "render_a"
    assert app._menu_state == MENU_OPEN_ICE_PROMPT


def test_multiple_candidate_groups_ask_before_loading(tmp_path):
    app = make_event_app()
    nested = tmp_path / "nests.nc"
    _write_nested_render_file(nested)

    app._menu_state = MENU_FILE_BROWSER_LIQUID
    FlyThroughApp._select_browser_path(app, nested)

    assert app._menu_state == MENU_OPEN_GROUP_PROMPT
    assert app._pending_group_choices == ["render_a", "render_b"]
    assert app._pending_group is None

    # The number keys mirror the on-screen list order.
    transition = _menu_transition(True, MENU_OPEN_GROUP_PROMPT, "2")
    assert transition.group_index == 1
    FlyThroughApp._select_group(app, transition.group_index)

    assert app._pending_group == "render_b"
    assert app._menu_state == MENU_OPEN_ICE_PROMPT


def test_nested_pair_keys_run_from_b(tmp_path):
    """One pair keeps B; a three-level file's later pairs get C, D, ..."""
    from cloudyview.soar.menu import ACTION_SELECT_BOTH_GROUPS

    first = _menu_transition(True, MENU_OPEN_GROUP_PROMPT, "b")
    assert first.action == ACTION_SELECT_BOTH_GROUPS
    assert first.pair_index == 0

    third = _menu_transition(True, MENU_OPEN_GROUP_PROMPT, "d")
    assert third.action == ACTION_SELECT_BOTH_GROUPS
    assert third.pair_index == 2


def test_root_group_file_skips_the_group_prompt(tmp_path):
    import xarray as xr

    app = make_event_app()
    flat = tmp_path / "flat.nc"
    xr.Dataset(
        data_vars={
            "qc": (
                ("x", "y", "z"),
                np.zeros((2, 3, 4), dtype=np.float64),
                {"units": "g/kg"},
            )
        }
    ).to_netcdf(flat)

    app._menu_state = MENU_FILE_BROWSER_LIQUID
    FlyThroughApp._select_browser_path(app, flat)

    assert app._pending_group is None
    assert app._menu_state == MENU_OPEN_ICE_PROMPT


def test_missing_units_are_asked_for_before_the_load_starts(tmp_path):
    app = make_event_app()
    nested = tmp_path / "no_units.nc"
    _write_nested_render_file(nested, groups=("render_a",), with_units=False)
    app._pending_open_path = str(nested)
    app._pending_group = "render_a"
    loads = []
    app._create_renderer = lambda *a, **k: None

    FlyThroughApp._start_loading_file(app, nested, None)

    assert app._menu_state == MENU_OPEN_UNITS_PROMPT
    assert app._pending_units_vars == ["qc"]
    assert app._loading_job is None

    transition = _menu_transition(True, MENU_OPEN_UNITS_PROMPT, "k")
    assert transition.units == "kg/kg"
    app._start_loading_file = lambda liquid_path, ice_path: loads.append(
        (liquid_path, ice_path, app._pending_units)
    )
    FlyThroughApp._select_condensate_units(app, transition.units)

    assert loads == [(str(nested), None, "kg/kg")]


# ---------------------------------------------------------------------------
# Capture settings (shared by the screenshot and video dialogs)
# ---------------------------------------------------------------------------

def _capture_app():
    app = make_event_app()
    from cloudyview.soar.app import default_save_dir
    app._save_dir = default_save_dir()
    app._save_dir_text = str(app._save_dir)
    app._capture_size = None
    return app


def test_default_save_dir_prefers_downloads():
    from cloudyview.soar.app import default_save_dir

    chosen = default_save_dir()
    downloads = Path.home() / "Downloads"
    assert chosen == (downloads if downloads.is_dir() else Path.home())


def test_capture_size_follows_the_window_by_default():
    app = _capture_app()
    assert FlyThroughApp.capture_size(app) == app.canvas.get_physical_size()


def test_capture_size_override_and_reset():
    app = _capture_app()
    FlyThroughApp._set_capture_size(app, (1920, 1080))
    assert FlyThroughApp.capture_size(app) == (1920, 1080)
    FlyThroughApp._set_capture_size(app, None)
    assert FlyThroughApp.capture_size(app) == app.canvas.get_physical_size()


def test_capture_size_is_clamped_not_rejected():
    """A half-typed number in a text field must not throw."""
    from cloudyview.soar.app import CAPTURE_SIZE_LIMITS

    lo, hi = CAPTURE_SIZE_LIMITS
    app = _capture_app()
    FlyThroughApp._set_capture_size(app, (0, 999999))
    assert FlyThroughApp.capture_size(app) == (lo, hi)


def test_save_dir_accepts_a_real_directory(tmp_path):
    app = _capture_app()
    assert FlyThroughApp._set_save_dir(app, str(tmp_path)) is True
    assert app._save_dir == tmp_path


def test_save_dir_rejects_a_missing_directory_but_keeps_the_text(tmp_path):
    app = _capture_app()
    before = app._save_dir
    missing = str(tmp_path / "nope")
    assert FlyThroughApp._set_save_dir(app, missing) is False
    assert app._save_dir == before          # unchanged
    assert app._save_dir_text == missing    # still editable
    assert FlyThroughApp._save_dir_is_usable(app) is False


def test_timestamped_paths_land_in_the_save_dir(tmp_path):
    app = _capture_app()
    FlyThroughApp._set_save_dir(app, str(tmp_path))
    png = FlyThroughApp._timestamped_png_path(app, "shot")
    mp4 = FlyThroughApp._timestamped_path(app, "clip", ".mp4")
    assert png.parent == tmp_path and png.suffix == ".png"
    assert mp4.parent == tmp_path and mp4.suffix == ".mp4"
