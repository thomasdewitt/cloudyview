"""Headless tests for the soar window app shell."""

import numpy as np
import pytest

pytest.importorskip("wgpu", reason="requires the 'interactive' extra")

from cloudyview.soar.app import (
    ACTION_PAUSE,
    ACTION_QUIT,
    ACTION_RESUME,
    ACTION_TOGGLE_FULLSCREEN,
    FlyThroughApp,
    OCEAN_FLOOR_MARGIN_M,
    _clamp_position_above_ocean,
    _control_action_for_key,
)


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


def make_event_app():
    app = object.__new__(FlyThroughApp)
    app.canvas = DummyCanvas()
    app._paused = False
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

    def capture_mouse(capture):
        app.capture_calls.append(capture)
        app._captured = capture
        app._last_pointer = None

    def toggle_fullscreen():
        app.fullscreen_calls += 1
        app._fullscreen = not app._fullscreen

    app._capture_mouse = capture_mouse
    app._toggle_fullscreen = toggle_fullscreen
    return app


def test_control_action_for_key_active_and_paused():
    assert _control_action_for_key(False, "Escape") == ACTION_PAUSE
    assert _control_action_for_key(False, "f") == ACTION_TOGGLE_FULLSCREEN
    assert _control_action_for_key(False, "F") == ACTION_TOGGLE_FULLSCREEN
    assert _control_action_for_key(False, "w") is None

    assert _control_action_for_key(True, "Escape") == ACTION_QUIT
    assert _control_action_for_key(True, "q") == ACTION_QUIT
    assert _control_action_for_key(True, "Q") == ACTION_QUIT
    assert _control_action_for_key(True, "r") == ACTION_RESUME
    assert _control_action_for_key(True, "R") == ACTION_RESUME
    assert _control_action_for_key(True, "F") == ACTION_TOGGLE_FULLSCREEN
    assert _control_action_for_key(True, "w") is None


def test_clamp_position_above_ocean_uses_margin_and_returns_copy():
    original = np.array([10.0, 20.0, -50.0])
    clamped = _clamp_position_above_ocean(original)

    assert clamped.tolist() == [10.0, 20.0, OCEAN_FLOOR_MARGIN_M]
    assert original.tolist() == [10.0, 20.0, -50.0]

    already_above = _clamp_position_above_ocean([1.0, 2.0, 25.0])
    assert already_above.tolist() == [1.0, 2.0, 25.0]


def test_escape_pauses_releases_capture_and_clears_movement_keys():
    app = make_event_app()

    FlyThroughApp._on_event(app, {"event_type": "key_down", "key": "Escape"})

    assert app._paused is True
    assert app._captured is False
    assert app._keys == set()
    assert app.capture_calls == [False]
    assert app.canvas.closed is False
    assert "PAUSED" in app.canvas.titles[-1]
    assert app.canvas.draw_requests == 1


def test_resume_key_and_click_recapture_pointer():
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

    assert app._paused is False
    assert app._captured is True
    assert app.capture_calls == [True]


def test_paused_escape_or_q_quits_without_recapturing():
    for key in ("Escape", "q", "Q"):
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
    app._keys = {"Shift"}
    app.position = np.array([0.0, 0.0, OCEAN_FLOOR_MARGIN_M + 0.25])
    app.speed = 60.0
    app.azimuth = 0.0
    app.elevation = 0.0
    app.fov = 80.0

    FlyThroughApp._move(app, 1.0)

    assert app.position[2] == OCEAN_FLOOR_MARGIN_M


def test_move_clamps_forward_descent_when_pitched_down():
    app = object.__new__(FlyThroughApp)
    app._paused = False
    app._keys = {"w"}
    app.position = np.array([0.0, 0.0, OCEAN_FLOOR_MARGIN_M + 0.25])
    app.speed = 60.0
    app.azimuth = 0.0
    app.elevation = -45.0
    app.fov = 80.0

    FlyThroughApp._move(app, 1.0)

    assert app.position[2] == OCEAN_FLOOR_MARGIN_M
