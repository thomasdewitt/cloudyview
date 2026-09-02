"""`soar TRACK.json` reads what the browser wrote, and resamples it the same.

The resampler exists twice — web/soar/track.js for the in-tab video,
cloudyview/track.py for the desktop one — and a track is meant to render to
the same frames from either. So the Python port is pinned against the real
JavaScript, run under node on the same synthetic track: every column of every
frame agrees to floating-point noise, including the periodic folds and the
azimuth unwrap, for a v2 (9-column) and a v1 (7-column) track.

The header reader is pinned on the schema soar writes today: every render
setting comes from the header, a v1 header (no city block) is a day scene,
odd capture sizes snap to even, and a missing key names itself.

The resample tests skip without node; the header tests need nothing.
"""

import json
import shutil
import subprocess
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


def _synthetic_samples(with_surface: bool, seed: int = 7):
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
    if with_surface:
        sx = (0.9 + np.linspace(0, 0.3, n)) % 1.0       # tile crossing
        sy = 0.4 + 0.05 * np.sin(np.linspace(0, 4, n))
        cols += [sx, sy]
    samples = np.stack(cols, axis=1)
    # A duplicated timestamp, as a stalled frame writes one.
    samples[10, 0] = samples[9, 0]
    return samples


def _js_resample(samples, fps, periodic=True):
    script = textwrap.dedent(f"""
        import {{ resampleTrack }} from "{TRACK_JS.as_uri()}";
        const frames = resampleTrack({json.dumps(samples.tolist())}, {fps},
                                     {{ periodic: {str(periodic).lower()} }});
        console.log(JSON.stringify(frames));
    """)
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@needs_node
@pytest.mark.parametrize("with_surface", [True, False])
def test_python_resample_matches_track_js(with_surface):
    samples = _synthetic_samples(with_surface)
    fps = 30.0
    py = tr.resample_track(samples, fps, periodic=True)
    js = _js_resample(samples, fps)
    assert len(py) == len(js) == tr.frame_count_between(
        samples[0, 0], samples[-1, 0], fps)
    for p, j in zip(py, js):
        assert p.t == pytest.approx(j["t"], abs=1e-12)
        np.testing.assert_allclose(p.position, j["position"], atol=1e-12)
        assert p.azimuth == pytest.approx(j["azimuth"], abs=1e-9)
        assert p.elevation == pytest.approx(j["elevation"], abs=1e-12)
        assert p.fov == pytest.approx(j["fov"], abs=1e-12)
        if with_surface:
            np.testing.assert_allclose(
                p.surface_position, j["surfacePosition"], atol=1e-12)
            assert 0.0 <= p.surface_position[0] < 1.0
        else:
            assert p.surface_position is None and j["surfacePosition"] is None
        assert -1.0 <= p.position[0] < 1.0


def test_resample_rejects_mixed_widths_and_short_tracks(tmp_path):
    path = tmp_path / "t.json"
    header = {"schema": tr.HEADER_SCHEMA}
    path.write_text(json.dumps({
        "schema": tr.TRACK_SCHEMAS[1], "header": header,
        "samples": [[0.0] * 9, [1.0] * 7]}))
    with pytest.raises(ValueError, match="7 columns"):
        tr.load_track(path)
    path.write_text(json.dumps({
        "schema": tr.TRACK_SCHEMAS[1], "header": header,
        "samples": [[0.0] * 9]}))
    with pytest.raises(ValueError, match="not enough"):
        tr.load_track(path)
    with pytest.raises(ValueError, match="distinct"):
        tr.resample_track(np.zeros((3, 9)), 30.0)


# --- the header ------------------------------------------------------------

def _header(tmp_path, *, city=True, size=(1920, 961)):
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
    if city:
        h["city"] = {"position_m": [1, 2, 3],
                     "tile_offset_m": [75330.0, -17010.0],
                     "tile_extent_m": 92160.0,
                     "surface_offset_m": [79780.0, 47970.0]}
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
    assert s.city is True
    assert s.city_tile_offset_m == (75330.0, -17010.0)
    assert s.city_tile_extent_m == 92160.0
    assert s.surface_offset_m == (79780.0, 47970.0)


def test_overrides_and_day_scene(tmp_path):
    s = tr.settings_from_header(_header(tmp_path, city=False),
                                size=(640, 361), fps=24, quality="low")
    assert s.city is False and s.surface_offset_m is None
    assert s.size == (640, 360) and s.fps == 24.0 and s.quality == "low"


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
    out = subprocess.run(["soar", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "--field" in out.stdout and "--quality" in out.stdout
