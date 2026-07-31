"""Tests for the soar time-of-day panel (ESC -> T).

The sun drives the whole spectral look package — beam colour, sky field,
low-sun warm wedge, ocean glint — so the panel is a look control, not a
convenience. Coverage:

- zenith <-> elevation conversion and the horizon floor;
- presets, and the periodic-domain constraint they must respect;
- menu wiring (key, state, preset keys);
- CLI flags and screenshot reproduction metadata.
"""

from pathlib import Path

import pytest

pytest.importorskip("wgpu", reason="requires the 'interactive' extra")

from cloudyview.soar.app import FlyThroughApp
from cloudyview.soar.menu import (
    ACTION_MENU_BACK,
    ACTION_SELECT_SUN_PRESET,
    ACTION_SUN_MENU,
    MENU_MAIN,
    MENU_SUN,
    MIN_SUN_ELEVATION_DEG,
    SUN_PRESETS,
    menu_transition,
)


def _sun_app(azimuth=20.0, elevation=55.0):
    app = object.__new__(FlyThroughApp)
    app.sun_azimuth = azimuth
    app.sun_elevation = elevation
    app._flash_title = lambda text, seconds=0.0: None
    return app


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------

def test_zenith_is_the_complement_of_elevation():
    app = _sun_app(elevation=55.0)
    assert FlyThroughApp.sun_zenith.fget(app) == pytest.approx(35.0)


def test_setting_zenith_moves_the_sun():
    app = _sun_app()
    FlyThroughApp._set_sun(app, zenith=60.0)
    assert app.sun_elevation == pytest.approx(30.0)
    assert FlyThroughApp.sun_zenith.fget(app) == pytest.approx(60.0)


def test_zenith_is_floored_just_above_the_horizon():
    """A periodic light march exits through the domain top; the sun must
    stay above the horizon or write_uniforms refuses the frame."""
    app = _sun_app()
    FlyThroughApp._set_sun(app, zenith=90.0)
    assert app.sun_elevation == pytest.approx(MIN_SUN_ELEVATION_DEG)

    FlyThroughApp._set_sun(app, zenith=200.0)
    assert app.sun_elevation == pytest.approx(MIN_SUN_ELEVATION_DEG)


def test_zenith_is_capped_overhead():
    app = _sun_app()
    FlyThroughApp._set_sun(app, zenith=-30.0)
    assert app.sun_elevation == pytest.approx(90.0)


def test_azimuth_wraps_into_a_single_turn():
    app = _sun_app()
    FlyThroughApp._set_sun(app, azimuth=380.0)
    assert app.sun_azimuth == pytest.approx(20.0)
    FlyThroughApp._set_sun(app, azimuth=-90.0)
    assert app.sun_azimuth == pytest.approx(270.0)


def test_setting_one_angle_leaves_the_other_alone():
    app = _sun_app(azimuth=123.0, elevation=45.0)
    FlyThroughApp._set_sun(app, zenith=10.0)
    assert app.sun_azimuth == pytest.approx(123.0)
    FlyThroughApp._set_sun(app, azimuth=200.0)
    assert app.sun_elevation == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def test_presets_include_midday_and_sunset():
    assert "midday" in SUN_PRESETS
    assert "sunset" in SUN_PRESETS
    midday = SUN_PRESETS["midday"][1]
    sunset = SUN_PRESETS["sunset"][1]
    assert midday > 45.0
    assert sunset < 5.0


@pytest.mark.parametrize("name", tuple(SUN_PRESETS))
def test_every_preset_keeps_the_sun_above_the_horizon(name):
    app = _sun_app()
    FlyThroughApp._select_sun_preset(app, name)
    assert app.sun_elevation >= MIN_SUN_ELEVATION_DEG
    assert app.sun_elevation <= 90.0
    assert 0.0 <= app.sun_azimuth < 360.0


def test_preset_applies_both_angles():
    app = _sun_app()
    FlyThroughApp._select_sun_preset(app, "sunset")
    azimuth, elevation = SUN_PRESETS["sunset"]
    assert app.sun_azimuth == pytest.approx(azimuth)
    assert app.sun_elevation == pytest.approx(elevation)


def test_unknown_preset_is_ignored():
    app = _sun_app(azimuth=20.0, elevation=55.0)
    FlyThroughApp._select_sun_preset(app, "twilight")
    FlyThroughApp._select_sun_preset(app, None)
    assert app.sun_azimuth == pytest.approx(20.0)
    assert app.sun_elevation == pytest.approx(55.0)


def test_compass_label_tracks_the_azimuth():
    for azimuth, label in ((0.0, "N"), (90.0, "E"), (180.0, "S"),
                           (270.0, "W"), (359.0, "N")):
        app = _sun_app(azimuth=azimuth)
        assert FlyThroughApp._sun_compass_label(app) == label


# ---------------------------------------------------------------------------
# Menu wiring
# ---------------------------------------------------------------------------

def test_t_opens_the_sun_menu_from_the_pause_menu():
    transition = menu_transition(True, MENU_MAIN, "T")
    assert transition.action == ACTION_SUN_MENU
    assert transition.next_state == MENU_SUN


def test_sun_menu_number_keys_pick_presets():
    for index, name in enumerate(SUN_PRESETS):
        transition = menu_transition(True, MENU_SUN, str(index + 1))
        assert transition.action == ACTION_SELECT_SUN_PRESET
        assert transition.sun_preset == name
        assert transition.next_state == MENU_SUN


def test_escape_leaves_the_sun_menu():
    transition = menu_transition(True, MENU_SUN, "Escape")
    assert transition.action == ACTION_MENU_BACK
    assert transition.next_state == MENU_MAIN


def test_t_does_nothing_while_flying():
    assert menu_transition(False, MENU_MAIN, "T").action is None


# ---------------------------------------------------------------------------
# Reproduction
# ---------------------------------------------------------------------------

def test_cli_exposes_sun_flags():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "cloudyview.soar", "--help"],
        capture_output=True, text=True, check=True,
        cwd=str(Path(__file__).parent.parent),
    ).stdout
    assert "--sun-azimuth" in out
    assert "--sun-elevation" in out


def test_reproduction_command_carries_a_moved_sun():
    from types import SimpleNamespace

    import cloudyview as cv

    app = object.__new__(FlyThroughApp)
    app.renderer = SimpleNamespace(
        field=SimpleNamespace(source="cloud.nc", ice_source=None),
        nest=None,
        quality_tier="high",
        volume_fp16=False,
    )
    app.sun_azimuth = 270.0
    app.sun_elevation = 0.5
    command = FlyThroughApp._soar_reproduction_command(app, cv.Camera())
    assert "--sun-azimuth 270" in command
    assert "--sun-elevation 0.5" in command


def test_reproduction_command_omits_an_untouched_sun():
    from types import SimpleNamespace

    import cloudyview as cv
    from cloudyview.soar.app import DEFAULT_SUN_AZIMUTH, DEFAULT_SUN_ELEVATION

    app = object.__new__(FlyThroughApp)
    app.renderer = SimpleNamespace(
        field=SimpleNamespace(source="cloud.nc", ice_source=None),
        nest=None,
        quality_tier="high",
        volume_fp16=False,
    )
    app.sun_azimuth = DEFAULT_SUN_AZIMUTH
    app.sun_elevation = DEFAULT_SUN_ELEVATION
    command = FlyThroughApp._soar_reproduction_command(app, cv.Camera())
    assert "--sun-" not in command
