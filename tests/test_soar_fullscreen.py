"""Pure tests for soar fullscreen monitor selection."""

from types import SimpleNamespace

from cloudyview.soar.fullscreen import (
    choose_fullscreen_monitor,
    fullscreen_video_mode,
    safe_windowed_bounds,
)


class FakeGlfw:
    def __init__(self):
        self.primary = "primary"
        self.monitors = ["left", None, "primary"]
        self.video_mode_calls = []
        self.raise_window_pos = False

    def get_window_monitor(self, _window):
        return None

    def get_primary_monitor(self):
        return self.primary

    def get_monitors(self):
        return list(self.monitors)

    def get_window_pos(self, _window):
        if self.raise_window_pos:
            raise RuntimeError("Wayland has no window positions")
        return (1800, 40)

    def get_window_size(self, _window):
        return (800, 600)

    def get_monitor_pos(self, monitor):
        return {"left": (0, 0), "primary": (1920, 0)}[monitor]

    def get_video_mode(self, monitor):
        assert monitor is not None
        self.video_mode_calls.append(monitor)
        return SimpleNamespace(width=1920, height=1080, refresh_rate=60)


def test_choose_fullscreen_monitor_falls_back_to_primary_when_pos_unavailable():
    glfw = FakeGlfw()
    glfw.raise_window_pos = True

    monitor = choose_fullscreen_monitor(glfw, window=object())

    assert monitor == "primary"
    assert glfw.video_mode_calls == []


def test_choose_fullscreen_monitor_ignores_null_monitor_candidates():
    glfw = FakeGlfw()

    monitor = choose_fullscreen_monitor(glfw, window=object())

    assert monitor == "primary"
    assert None not in glfw.video_mode_calls


def test_fullscreen_video_mode_never_accepts_null_monitor():
    glfw = FakeGlfw()

    monitor, mode = fullscreen_video_mode(glfw, "primary")

    assert monitor == "primary"
    assert mode.width == 1920


def test_safe_windowed_bounds_survives_wayland_position_failure():
    glfw = FakeGlfw()
    glfw.raise_window_pos = True

    assert safe_windowed_bounds(glfw, window=object()) == (100, 100, 800, 600)
