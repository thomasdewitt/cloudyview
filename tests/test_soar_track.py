"""Track recording/resampling unit tests (no GPU)."""

import numpy as np
import pytest

from cloudyview.soar.engine import AUTO_FP16_MIN_VOXELS, choose_volume_fp16
from cloudyview.soar.track import (
    TRACK_SCHEMA,
    load_track,
    resample_track,
    save_track,
)


def _samples(rows):
    return np.asarray(rows, dtype=np.float64)


def test_save_load_roundtrip(tmp_path):
    header = {"source": {"path": "x.nc"}, "sun": {"azimuth": 235.0}}
    rows = [[0.0, 0.1, -0.2, 0.3, 45.0, 5.0, 60.0],
            [0.5, 0.2, -0.1, 0.3, 50.0, 6.0, 60.0]]
    path = save_track(tmp_path / "t.json", header, rows)
    got_header, got = load_track(path)
    assert got_header == header
    np.testing.assert_allclose(got, rows)


def test_load_rejects_wrong_schema_and_short_tracks(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "nope", "header": {}, "samples": []}')
    with pytest.raises(ValueError, match=TRACK_SCHEMA):
        load_track(path)
    save_track(path, {}, [[0.0, 0, 0, 0, 0, 0, 60.0]])
    with pytest.raises(ValueError, match="at least 2"):
        load_track(path)


def test_resample_constant_camera_is_constant():
    rows = _samples([[t, 0.1, -0.4, 0.2, 120.0, -10.0, 60.0]
                     for t in np.linspace(0.0, 2.0, 25)])
    frames = resample_track(rows, 30.0)
    assert len(frames) == 61  # 2 s at 30 fps, inclusive endpoints
    for _, cam in frames:
        np.testing.assert_allclose(cam.position, (0.1, -0.4, 0.2), atol=1e-9)
        assert cam.azimuth == pytest.approx(120.0)
        assert cam.elevation == pytest.approx(-10.0)


def test_resample_output_cadence_is_exact():
    rows = _samples([[t, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0]
                     # deliberately irregular input timing
                     for t in np.cumsum([0.0, 0.03, 0.11, 0.02, 0.4, 0.05])])
    frames = resample_track(rows, 60.0)
    times = np.array([t for t, _ in frames])
    np.testing.assert_allclose(np.diff(times), 1.0 / 60.0, atol=1e-12)
    assert times[0] == pytest.approx(rows[0, 0])


def test_azimuth_interpolates_through_wrap():
    rows = _samples([
        [0.0, 0.0, 0.0, 0.0, 350.0, 0.0, 60.0],
        [1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 60.0],
    ])
    frames = resample_track(rows, 10.0)
    mid_az = frames[5][1].azimuth
    # Through north (0/360), not backwards through 180.
    assert mid_az < 20.0 or mid_az > 340.0


def test_periodic_position_interpolates_through_domain_wrap():
    rows = _samples([
        [0.0, 0.9, 0.0, 0.1, 90.0, 0.0, 60.0],
        [1.0, -0.9, 0.0, 0.1, 90.0, 0.0, 60.0],   # wrapped over +x edge
    ])
    frames = resample_track(rows, 10.0, periodic=True)
    xs = np.array([cam.position[0] for _, cam in frames])
    # Every interpolated x stays near the wrap edge; a naive lerp would
    # sweep straight through the domain center.
    assert np.all(np.abs(xs) > 0.75)

    frames_flat = resample_track(rows, 10.0, periodic=False)
    xs_flat = np.array([cam.position[0] for _, cam in frames_flat])
    assert np.any(np.abs(xs_flat) < 0.2)


def test_choose_volume_fp16_auto_and_explicit():
    assert not choose_volume_fp16(AUTO_FP16_MIN_VOXELS - 1, None)
    assert choose_volume_fp16(AUTO_FP16_MIN_VOXELS, None)
    # An explicit user choice always wins over the size heuristic.
    assert choose_volume_fp16(1, True)
    assert not choose_volume_fp16(AUTO_FP16_MIN_VOXELS * 4, False)


def test_unsorted_and_duplicate_times_are_cleaned():
    rows = _samples([
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 60.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0],
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 60.0],   # duplicate stamp
        [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 60.0],
    ])
    frames = resample_track(rows, 4.0, periodic=False)
    xs = [cam.position[0] for _, cam in frames]
    assert xs == sorted(xs)
    assert frames[0][1].position[0] == pytest.approx(0.0)
    assert frames[-1][1].position[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stepped video render (the app drives this from its draw loop)
# ---------------------------------------------------------------------------

def _video_fixture(tmp_path):
    """A tiny real scene + track, ready to encode."""
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    pytest.importorskip("wgpu", reason="requires the 'interactive' extra")

    import xarray as xr
    import cloudyview as cv
    from cloudyview.render_metadata import build_render_metadata
    from cloudyview.soar import InteractiveRenderer
    from cloudyview.soar.track import save_track

    n, nz, d = 16, 8, 100.0
    coords = {
        axis: ((np.arange(count) + 0.5) * d)
        for axis, count in (("x", n), ("y", n), ("z", nz))
    }
    path = tmp_path / "field.nc"
    xr.Dataset(
        {"QC": (("x", "y", "z"),
                np.full((n, n, nz), 0.2, np.float32), {"units": "g/kg"})},
        coords={k: (k, v, {"units": "m"}) for k, v in coords.items()},
    ).to_netcdf(path)

    field = cv.load(str(path))
    flat = np.zeros((8, 8), dtype=np.float32)
    renderer = InteractiveRenderer(
        field, periodic=True, ocean_enabled=False,
        fif_normals=(flat, flat, np.ones((8, 8), np.float32), 1.0),
    )
    header = build_render_metadata(
        field, cv.Camera(), sun_azimuth=235.0, sun_elevation=25.0,
        renderer="soar", quality="high", reproduction_command="test",
        render_options={"periodic": True},
    )
    samples = [
        [t / 8.0, 0.0, -0.9, -0.3, float(t * 5), 5.0, 70.0] for t in range(8)
    ]
    track = tmp_path / "clip.json"
    save_track(track, header, samples)
    return renderer, track


def test_stepped_video_render_writes_a_playable_file(tmp_path):
    from cloudyview.soar.track import TrackVideoRender

    renderer, track = _video_fixture(tmp_path)
    out = tmp_path / "clip.mp4"
    render = TrackVideoRender(
        track, out, fps=8.0, size=(160, 90), accumulate_frames=1,
        renderer=renderer,
    )
    assert render.total > 0
    assert render.done is False

    guard = 0
    while not render.step(2):
        guard += 1
        assert guard < 1000, "step() never reported completion"
    assert render.close() == out
    assert out.stat().st_size > 0


def test_stepped_video_render_reports_progress(tmp_path):
    from cloudyview.soar.track import TrackVideoRender

    renderer, track = _video_fixture(tmp_path)
    render = TrackVideoRender(
        track, tmp_path / "p.mp4", fps=8.0, size=(96, 54),
        accumulate_frames=1, renderer=renderer,
    )
    assert render.progress()["percent"] == 0.0
    render.step()
    mid = render.progress()
    assert 0.0 < mid["percent"] <= 100.0
    assert mid["frame"] == 1
    assert mid["eta"] is not None
    while not render.step(4):
        pass
    render.close()
    assert render.progress()["percent"] == pytest.approx(100.0)


def test_video_render_restores_the_quality_tier(tmp_path):
    """The app's renderer is borrowed, not consumed."""
    from cloudyview.soar.track import TrackVideoRender

    renderer, track = _video_fixture(tmp_path)
    renderer.set_quality_tier("low", camera_moving=False)
    render = TrackVideoRender(
        track, tmp_path / "t.mp4", fps=8.0, size=(96, 54),
        accumulate_frames=1, renderer=renderer,
    )
    assert renderer.quality_tier == "high"   # forced for the encode
    while not render.step(4):
        pass
    render.close()
    assert renderer.quality_tier == "low"


def test_aborted_video_render_leaves_no_file(tmp_path):
    from cloudyview.soar.track import TrackVideoRender

    renderer, track = _video_fixture(tmp_path)
    renderer.set_quality_tier("medium", camera_moving=False)
    out = tmp_path / "aborted.mp4"
    render = TrackVideoRender(
        track, out, fps=8.0, size=(96, 54), accumulate_frames=1,
        renderer=renderer,
    )
    render.step()
    render.abort()
    assert not out.exists()
    assert renderer.quality_tier == "medium"


def test_video_size_is_forced_even_for_yuv420p(tmp_path):
    from cloudyview.soar.track import TrackVideoRender

    renderer, track = _video_fixture(tmp_path)
    render = TrackVideoRender(
        track, tmp_path / "odd.mp4", fps=8.0, size=(161, 91),
        accumulate_frames=1, renderer=renderer,
    )
    assert (render.width, render.height) == (160, 90)
    render.abort()
