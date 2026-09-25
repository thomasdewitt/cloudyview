"""`soar TRACK.json` reads what the browser wrote, and resamples it the same.

The resampler exists twice — web/soar/track.js for the in-tab video,
cloudyview/track.py for the desktop one — and a track is meant to render to
the same frames from either. So the Python port is pinned against the real
JavaScript, run under node on the same synthetic track: every column of every
frame agrees to floating-point noise, including the periodic folds and the
azimuth unwrap, for a v3 (10-column, with the scene clock), a v2 (9-column)
and a v1 (7-column) track.

The header reader is pinned on the schema soar writes today: every render
setting comes from the header, a v1 header (no city block) is a day scene,
odd capture sizes snap to even, and a missing key names itself.

The resample tests skip without node; the header tests need nothing.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview import track as tr

REPO = Path(__file__).resolve().parents[1]
TRACK_JS = REPO / "web" / "soar" / "track.js"

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not TRACK_JS.exists(),
    reason="needs node and web/soar/track.js")


def _synthetic_samples(version: int, seed: int = 7):
    """A hand-flown-looking track: irregular timing, a cloud-period crossing
    in x, a north crossing in azimuth, a tile crossing in sx."""
    rng = np.random.default_rng(seed)
    n = 60
    t = np.cumsum(rng.uniform(0.010, 0.030, n))
    t[0] = 0.0
    x = np.linspace(0.6, 1.5, n)                       # crosses +1 -> wraps
    x = (x + 1.0) % 2.0 - 1.0
    y = 0.2 * np.sin(np.linspace(0, 3, n))
    z = np.linspace(0.1, 0.4, n)
    az = (350.0 + np.linspace(0, 25, n)) % 360.0       # through north
    el = 10.0 * np.cos(np.linspace(0, 2, n))
    fov = np.full(n, 100.0)
    cols = [t, x, y, z, az, el, fov]
    if version >= 2:
        sx = (0.9 + np.linspace(0, 0.3, n)) % 1.0       # tile crossing
        sy = 0.4 + 0.05 * np.sin(np.linspace(0, 4, n))
        cols += [sx, sy]
    if version >= 3:
        # The scene clock: runs with the flight from wherever it stood, and
        # held for a stretch (a pause) — the column carries it either way.
        clock = 812.25 + t.copy()
        clock[30:36] = clock[30]
        clock[36:] -= clock[36] - clock[30]
        cols += [clock]
    samples = np.stack(cols, axis=1)
    # A duplicated timestamp, as a stalled frame writes one.
    samples[10, 0] = samples[9, 0]
    return samples


def _js_resample(samples, fps, periodic=True, linear_position=False):
    script = textwrap.dedent(f"""
        import {{ resampleTrack }} from "{TRACK_JS.as_uri()}";
        const frames = resampleTrack({json.dumps(samples.tolist())}, {fps},
                                     {{ periodic: {str(periodic).lower()},
                                        linearPosition:
                                          {str(linear_position).lower()} }});
        console.log(JSON.stringify(frames));
    """)
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@needs_node
@pytest.mark.parametrize("linear_position", [False, True])
@pytest.mark.parametrize("version", [3, 2, 1])
def test_python_resample_matches_track_js(version, linear_position):
    samples = _synthetic_samples(version)
    fps = 30.0
    py = tr.resample_track(samples, fps, periodic=True,
                           linear_position=linear_position)
    js = _js_resample(samples, fps, linear_position=linear_position)
    assert len(py) == len(js) == tr.frame_count_between(
        samples[0, 0], samples[-1, 0], fps)
    for p, j in zip(py, js):
        assert p.t == pytest.approx(j["t"], abs=1e-12)
        np.testing.assert_allclose(p.position, j["position"], atol=1e-12)
        assert p.azimuth == pytest.approx(j["azimuth"], abs=1e-9)
        assert p.elevation == pytest.approx(j["elevation"], abs=1e-12)
        assert p.fov == pytest.approx(j["fov"], abs=1e-12)
        if version >= 2:
            np.testing.assert_allclose(
                p.surface_position, j["surfacePosition"], atol=1e-12)
            assert 0.0 <= p.surface_position[0] < 1.0
        else:
            assert p.surface_position is None and j["surfacePosition"] is None
        # The scene clock: resampled like any column on a v3 track, and the
        # sea held at 0 on anything older.
        assert p.scene_time == pytest.approx(j["sceneTime"], abs=1e-9)
        if version < 3:
            assert p.scene_time == 0.0
        assert -1.0 <= p.position[0] < 1.0


def test_a_v3_city_track_with_knocks_loads_and_replays(tmp_path):
    """The two track changes of 2026-09-23 together: v3's scene-clock column
    and the city's top-level knocks. The knocks ride beside the samples, on
    the flight's clock, and this render reads no further than accepting
    them; the city replays with linear positions and the clock resamples."""
    samples = _synthetic_samples(3)
    path = tmp_path / "city.json"
    path.write_text(json.dumps({
        "schema": "cloudyview.track.v3",
        "header": {"schema": tr.HEADER_SCHEMA},
        "samples": samples.tolist(),
        "knocks": [[float(samples[20, 0]), 0.7, 1, 0.8, 0],
                   [float(samples[40, 0]), 0.3, -1, 0.5, 1]]}))
    _, loaded, _ = tr.load_track(path)
    assert loaded.shape == samples.shape
    frames = tr.resample_track(loaded, 30.0, periodic=True,
                               linear_position=True)
    clocks = [f.scene_time for f in frames]
    assert clocks[0] == pytest.approx(samples[0, 9])
    assert samples[0, 9] <= min(clocks) and max(clocks) <= samples[-1, 9]
    assert all(b >= a - 1e-9 for a, b in zip(clocks, clocks[1:]))


def test_resample_rejects_mixed_widths_and_short_tracks(tmp_path):
    path = tmp_path / "t.json"
    header = {"schema": tr.HEADER_SCHEMA}
    path.write_text(json.dumps({
        "schema": tr.TRACK_SCHEMAS[1], "header": header,
        "samples": [[0.0] * 9, [1.0] * 7]}))
    with pytest.raises(ValueError, match="9-column"):
        tr.load_track(path)
    # The schema decides how a row reads, so a row of the wrong width for
    # its schema is corrupt even when the width is some schema's.
    path.write_text(json.dumps({
        "schema": "cloudyview.track.v3", "header": header,
        "samples": [[0.0] * 9, [1.0] * 9]}))
    with pytest.raises(ValueError, match="10-column"):
        tr.load_track(path)
    path.write_text(json.dumps({
        "schema": tr.TRACK_SCHEMAS[1], "header": header,
        "samples": [[0.0] * 9]}))
    with pytest.raises(ValueError, match="not enough"):
        tr.load_track(path)
    with pytest.raises(ValueError, match="distinct"):
        tr.resample_track(np.zeros((3, 9)), 30.0)



# --- the surface period ----------------------------------------------------

def _write(tmp_path, schema, samples, **extra):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"schema": schema,
                                "header": {"schema": tr.HEADER_SCHEMA},
                                "samples": samples.tolist(), **extra}))
    return path


def test_a_v4_track_carries_its_surface_period(tmp_path):
    samples = _synthetic_samples(3)
    _, _, period = tr.load_track(_write(
        tmp_path, "cloudyview.track.v4", samples, surface_period_m=1638.4))
    assert period == 1638.4
    # Older tracks recorded none; the schema says what it was.
    _, _, period = tr.load_track(_write(tmp_path, "cloudyview.track.v3",
                                        samples))
    assert period is None
    for bad in ({}, {"surface_period_m": None}, {"surface_period_m": 0},
                {"surface_period_m": "1638.4"}):
        with pytest.raises(ValueError, match="surface period"):
            tr.load_track(_write(tmp_path, "cloudyview.track.v4", samples,
                                 **bad))


def test_a_track_at_the_installed_period_replays_as_recorded():
    samples = _synthetic_samples(3)
    out = tr.surface_samples(samples, 1638.4, "ocean", 102.4, 1638.4)
    assert out is samples
    # The city and the desert fold at their tile, which v3 recorded at too.
    assert tr.surface_samples(samples, None, "city", 2300.0, 2300.0) \
        is samples


def test_a_track_at_another_period_is_refused():
    """The 2026-09-23 case: the sea's period doubled under the recordings,
    and read at the new one a flight's water streams past at its speed."""
    samples = _synthetic_samples(3)
    with pytest.raises(RuntimeError, match="folds at 819.2 m.*1638.4 m"):
        tr.surface_samples(samples, 819.2, "ocean", 102.4, 1638.4)
    with pytest.raises(RuntimeError, match="city"):
        tr.surface_samples(samples, 2300.0, "city", 1150.0, 1150.0)


@pytest.mark.parametrize("version, unit", [(3, tr.V3_OCEAN_PERIOD_M),
                                           (2, 102.4)])
def test_an_older_sea_track_is_re_expressed_continuously(version, unit):
    """v3 sea tracks were fractions of 819.2 m, v2 of the 102.4 m tile: in
    metres the flight over the water is the same continuous path at the
    installed period — no jump where the old period wrapped."""
    period = 1638.4
    samples = _synthetic_samples(version)
    assert np.any(np.abs(np.diff(samples[:, 7])) > 0.5)    # it wraps
    out = tr.surface_samples(samples, None, "ocean", 102.4, period)
    order = np.argsort(samples[:, 0], kind="stable")
    for c in (7, 8):
        assert np.all((0.0 <= out[:, c]) & (out[:, c] < 1.0))
        was = tr.unwrap_periodic(samples[order, c], 1.0, 0.5) * unit
        now = tr.unwrap_periodic(out[:, c], 1.0, 0.5) * period
        np.testing.assert_allclose(now - now[0], was - was[0], atol=1e-9)
    np.testing.assert_array_equal(out[:, :7], samples[order, :7])


# --- the header ------------------------------------------------------------

def _header(tmp_path, *, surface="city", size=(1920, 961)):
    field = tmp_path / "demo.nc"
    field.write_bytes(b"")
    h = {
        "schema": tr.HEADER_SCHEMA,
        "source": {"path": str(field), "ice_path": None,
                   "liquid_var": "QC", "ice_var": None},
        "camera": {"position": [0, 0, 0.2], "azimuth": 181.2,
                   "elevation": 14.5, "fov": 100},
        "sun": {"azimuth": 181, "elevation": 20},
        "render": {
            "renderer": "soar-web", "size": list(size), "tier": "custom",
            "quality": "max", "periodic": True, "accumulate_frames": 32,
            "tone_map_gamma": 1.66, "tone_map_white_point": 15,
            "contrast": 1, "haze": -0.038, "haze_height_dependent": False,
            "exposure": 3.88, "lod_strength": 0.01,
        },
    }
    if surface == "city":
        h["city"] = {"position_m": [1, 2, 3],
                     "tile_offset_m": [75330.0, -17010.0],
                     "tile_extent_m": 92160.0,
                     "surface_offset_m": [79780.0, 47970.0]}
    if surface == "desert":
        h["desert"] = {"position_m": [1, 2, 3],
                       "tile_extent_m": 4096.0,
                       "surface_offset_m": [1780.5, 3970.25]}
    return h


def test_settings_come_from_the_header(tmp_path):
    s = tr.settings_from_header(_header(tmp_path))
    assert s.size == (1920, 960)              # odd height snapped, as H.264 needs
    assert s.fps == tr.DEFAULT_VIDEO_FPS
    assert s.quality == "max"                 # Custom names its preset here
    assert s.periodic is True
    assert (s.sun_azimuth, s.sun_elevation) == (181.0, 20.0)
    assert s.exposure == 3.88 and s.haze == -0.038 and s.lod_strength == 0.01
    assert s.haze_height_dependent is False
    assert s.liquid_var == "QC" and s.ice_var is None
    assert s.surface == "city"
    assert s.city_tile_offset_m == (75330.0, -17010.0)
    assert s.surface_tile_extent_m == 92160.0
    assert s.surface_offset_m == (79780.0, 47970.0)


def test_overrides_and_day_scene(tmp_path):
    s = tr.settings_from_header(_header(tmp_path, surface="ocean"),
                                size=(640, 361), fps=24, quality="low")
    assert s.surface == "ocean" and s.surface_offset_m is None
    assert s.size == (640, 360) and s.fps == 24.0 and s.quality == "low"


def test_a_desert_flight_names_its_ground(tmp_path):
    s = tr.settings_from_header(_header(tmp_path, surface="desert"))
    assert s.surface == "desert"
    assert s.surface_tile_extent_m == 4096.0
    assert s.surface_offset_m == (1780.5, 3970.25)
    assert s.city_tile_offset_m is None
    both = _header(tmp_path)
    both["desert"] = _header(tmp_path, surface="desert")["desert"]
    with pytest.raises(ValueError, match="one surface"):
        tr.settings_from_header(both)


def test_missing_field_and_missing_keys_are_named(tmp_path):
    h = _header(tmp_path)
    h["source"]["path"] = str(tmp_path / "elsewhere.nc")
    with pytest.raises(FileNotFoundError, match="--field"):
        tr.settings_from_header(h)
    h = _header(tmp_path)
    del h["render"]["lod_strength"]
    with pytest.raises(ValueError, match="'lod_strength'"):
        tr.settings_from_header(h)
    h = _header(tmp_path)
    with pytest.raises(ValueError, match="unknown quality"):
        tr.settings_from_header(h, quality="ultra")


def test_surface_offset_folds_like_the_browser():
    tile = 92160.0
    # fold(origin) first, surface position subtracted after; every result
    # lands in [0, tile) whatever the signs.
    off = tr.surface_offset_for((-1000.0, 200000.0), (5000.0, 100.0), tile)
    assert off == (tr.fold_tile(tr.fold_tile(-1000.0, tile) - 5000.0, tile),
                   tr.fold_tile(tr.fold_tile(200000.0, tile) - 100.0, tile))
    assert all(0.0 <= v < tile for v in off)
    # Zero drift is the literal 0.0 (the day goldens' argument).
    assert tr.surface_offset_for((1234.5, -67.0), (1234.5, tile - 67.0), tile) \
        == (0.0, 0.0)


def test_cli_help_needs_no_gpu():
    # The `soar` this environment installed, beside its interpreter, not the
    # first one on PATH: a stale user-site install's ~/.local/bin/soar (an
    # entry point from before soar was a browser app) shadowed the venv's
    # whenever the suite ran without `uv run` putting .venv/bin first.
    soar = Path(sys.executable).with_name("soar")
    assert soar.is_file(), f"no soar entry point beside {sys.executable}"
    out = subprocess.run([str(soar), "--help"], capture_output=True,
                         text=True)
    assert out.returncode == 0
    assert "--field" in out.stdout and "--quality" in out.stdout


def test_lens_water_comes_from_the_header_and_old_tracks_replay_dry(tmp_path):
    # A header from before the switch has no key: it replays dry, as it
    # always rendered.
    assert tr.settings_from_header(_header(tmp_path)).lens_water is False
    on = _header(tmp_path)
    on["render"]["lens_water"] = True
    assert tr.settings_from_header(on).lens_water is True
    off = _header(tmp_path)
    off["render"]["lens_water"] = False
    assert tr.settings_from_header(off).lens_water is False
