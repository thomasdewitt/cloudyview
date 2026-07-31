"""Tests for the wgpu/WGSL interactive renderer (cloudyview.soar).

Skip-marked when wgpu (the 'interactive' extra) or a GPU adapter with
float32-filterable is unavailable, so the suite stays green on CI boxes
without a GPU.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_FILE = Path(__file__).parent.parent / "data" / "TWPICE_subvolume_256x256_5km.nc"

pytestmark = pytest.mark.gpu

wgpu = pytest.importorskip("wgpu", reason="requires the 'interactive' extra")


def _adapter_ok():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:
        return False
    return "float32-filterable" in adapter.features


if not _adapter_ok():  # pragma: no cover
    pytest.skip("no wgpu adapter with float32-filterable available",
                allow_module_level=True)

HUD_SIZE = (480, 270)  # (w, h)


def test_shader_offsets_world_samples_to_padded_texel_centers():
    """WGSL world->texture mapping must match witness's ghost-zero grid."""
    from cloudyview.soar import engine

    shader = engine.SHADER_PATH.read_text()
    assert "textureDimensions(t, 0)" in shader
    assert "level_data_dims(vol)" in shader
    assert "data_g.z + 1.5" in shader
    assert "data_g.y + 1.5" in shader
    assert "data_g.x + 1.5" in shader
    assert ") / tex_dims" in shader


@pytest.fixture(scope="module")
def renderer():
    import cloudyview as cv
    from cloudyview.soar import InteractiveRenderer

    field = cv.load(str(DATA_FILE))
    return InteractiveRenderer(field)


def _luma01(img):
    return img.astype(np.float32).mean(axis=-1) / 255.0


def _cloud_edge_score(luma):
    return _horizontal_gradient(luma) + _vertical_gradient(luma)


def _horizontal_gradient(luma):
    gradient = np.zeros_like(luma)
    gradient[:, 1:] = np.abs(np.diff(luma, axis=1))
    return gradient


def _vertical_gradient(luma):
    gradient = np.zeros_like(luma)
    gradient[1:, :] = np.abs(np.diff(luma, axis=0))
    return gradient


def _above_horizon_mask(camera, size):
    w, h = size
    forward, right, up = camera.basis()
    tan_half_fov = np.tan(np.deg2rad(camera.fov) * 0.5)
    aspect = w / h
    yy, xx = np.mgrid[0:h, 0:w]
    ndc_x = (2.0 * (xx + 0.5) / w - 1.0) * aspect * tan_half_fov
    ndc_y = (1.0 - 2.0 * (yy + 0.5) / h) * tan_half_fov
    dirs = (
        forward
        + ndc_x[..., None] * right
        + ndc_y[..., None] * up
    )
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs[..., 2] > 0.0


def _cloud_edge_mask(reference, camera):
    luma = _luma01(reference)
    edge_score = _cloud_edge_score(luma)
    size = (reference.shape[1], reference.shape[0])
    above_horizon = _above_horizon_mask(camera, size)
    return above_horizon & (
        edge_score > np.percentile(edge_score[above_horizon], 90.0)
    )


def _cloud_edge_hf(img, edge):
    return float(_horizontal_gradient(_luma01(img))[edge].mean())


def _motion_flicker(frames, masks):
    values = []
    for prev, cur, mask in zip(frames[:-1], frames[1:], masks[1:]):
        diff = np.abs(cur.astype(np.float32) - prev.astype(np.float32))
        values.append(float(diff[mask].mean()))
    return float(np.mean(values))


def test_offscreen_render_structure(renderer):
    """Default view: finite, non-uniform, sky above / cloud+haze structure."""
    import cloudyview as cv

    img = renderer.render(cv.Camera(), size=(480, 270))
    assert img.shape == (270, 480, 3)
    assert img.dtype == np.uint8

    f = img.astype(np.float64) / 255.0
    assert np.all(np.isfinite(f))
    # Non-uniform: an all-sky or all-black frame is a broken pipeline.
    assert f.std() > 0.05

    # The default witness camera looks up at 35 deg: the very top rows are
    # sky or cloud, never pitch black; and somewhere in the frame there must
    # be a clearly blue (sky) region: b substantially above r.
    top = f[:20]
    assert top.mean() > 0.1
    blueness = f[..., 2] - f[..., 0]
    assert blueness.max() > 0.15, "no blue-sky region found"
    # And bright cloud: near-white pixels with low channel spread.
    bright = f.max(axis=-1) > 0.85
    assert bright.mean() > 0.005, "no bright cloud region found"


def test_camera_moves_change_image(renderer):
    """Different viewpoints must give different images (camera uniforms live)."""
    import cloudyview as cv

    a = renderer.render(cv.Camera(azimuth=0), size=(320, 180))
    b = renderer.render(cv.Camera(azimuth=180), size=(320, 180))
    assert np.mean(np.abs(a.astype(int) - b.astype(int))) > 2.0


def test_jitter_toggle(renderer):
    """Jitter on/off are both valid and measurably different (banding A/B)."""
    import cloudyview as cv

    cam = cv.Camera()
    on = renderer.render(cam, size=(480, 270), jitter=True)
    off = renderer.render(cam, size=(480, 270), jitter=False)

    # Same scene: large-scale content must agree...
    assert abs(on.astype(float).mean() - off.astype(float).mean()) < 10.0
    # ...but per-pixel the jitter must actually do something.
    assert np.mean(np.abs(on.astype(int) - off.astype(int))) > 0.1

    # Jitter should decorrelate neighboring pixels: high-frequency energy
    # (mean abs horizontal gradient) goes UP with jitter on.
    def hf(img):
        g = img.astype(float).mean(axis=-1)
        return np.abs(np.diff(g, axis=1)).mean()

    # Ocean normal LOD deliberately calms high-frequency water in the jittered
    # interactive path, so test the cloud/sky portion where ray-start jitter
    # is the mechanism under test.
    top = slice(0, int(on.shape[0] * 0.75))
    assert hf(on[top]) > hf(off[top])


def test_temporal_accumulation_anti_aliases_cloud_edges(renderer):
    """Static accumulation should smooth pixel-footprint cloud-edge aliasing."""
    import cloudyview as cv

    size = (640, 360)
    cam = cv.Camera()
    renderer.reset_accumulation()
    off = renderer.render(cam, size=size, jitter=False, frame_index=0)
    renderer.reset_accumulation()
    single = renderer.render(
        cam, size=size, jitter=True, frame_index=0, accumulate_frames=1
    )
    renderer.reset_accumulation()
    accum = renderer.render(
        cam, size=size, jitter=True, frame_index=0, accumulate_frames=32
    )

    # Focus on strong above-horizon luma gradients: the visual complaint is
    # salt-and-pepper cloud-edge noise, not smooth sky.
    edge = _cloud_edge_mask(off, cam)
    assert edge.sum() > 1000

    single_delta = np.abs(single.astype(np.int16) - off.astype(np.int16))[edge]
    accum_delta = np.abs(accum.astype(np.int16) - off.astype(np.int16))[edge]
    assert np.percentile(accum_delta, 95) < np.percentile(single_delta, 95)

    off_hf = _cloud_edge_hf(off, edge)
    single_hf = _cloud_edge_hf(single, edge)
    accum_hf = _cloud_edge_hf(accum, edge)
    assert accum_hf < off_hf
    assert accum_hf < single_hf


def test_motion_accumulation_lowers_flicker(renderer):
    """Small camera deltas should EMA-blend instead of hard-resetting."""
    import cloudyview as cv

    size = (192, 108)
    base = cv.Camera(position=(-0.15, -0.10, -0.95),
                     azimuth=5.0, elevation=24.0, fov=90.0)
    cams = [
        cv.Camera(
            position=(base.position[0] + 0.0008 * i,
                      base.position[1] + 0.0005 * i,
                      base.position[2]),
            azimuth=base.azimuth + 0.18 * i,
            elevation=base.elevation,
            fov=base.fov,
        )
        for i in range(9)
    ]

    renderer.reset_accumulation()
    baseline = [
        renderer.render(
            cam, size=size, jitter=True, frame_index=i,
            accumulate=True, motion_accumulation=False,
        )
        for i, cam in enumerate(cams)
    ]

    masks = []
    for cam, frame in zip(cams, baseline):
        above = _above_horizon_mask(cam, size)
        luma = _luma01(frame)
        cloud = above & (luma > np.percentile(luma[above], 55.0))
        assert cloud.sum() > 500
        masks.append(cloud)

    renderer.reset_accumulation()
    smoothed = [
        renderer.render(
            cam, size=size, jitter=True, frame_index=i,
            accumulate=True, motion_accumulation=True,
            motion_blend_alpha=0.45, motion_jitter_scale=0.65,
        )
        for i, cam in enumerate(cams)
    ]

    assert renderer._accum_motion
    assert (
        _motion_flicker(smoothed, masks)
        < 0.9 * _motion_flicker(baseline, masks)
    )


def test_motion_settings_do_not_change_static_accumulation(renderer):
    """No-motion convergence stays on the exact static running-average path."""
    import cloudyview as cv

    size = (192, 108)
    cam = cv.Camera()

    renderer.reset_accumulation()
    baseline = renderer.render(
        cam, size=size, jitter=True, frame_index=0,
        accumulate_frames=8, motion_accumulation=False,
    )
    renderer.reset_accumulation()
    with_motion_defaults = renderer.render(
        cam, size=size, jitter=True, frame_index=0,
        accumulate_frames=8, motion_accumulation=True,
        motion_blend_alpha=0.35, motion_jitter_scale=0.35,
    )

    assert np.array_equal(with_motion_defaults, baseline)


def test_temporal_accumulation_resets_on_camera_change(renderer):
    """A large camera jump must not blend with the previous static view."""
    import cloudyview as cv

    size = (320, 180)
    static_cam = cv.Camera()
    moved_cam = cv.Camera(azimuth=25.0)

    renderer.reset_accumulation()
    static = renderer.render(
        static_cam, size=size, jitter=True, frame_index=0, accumulate_frames=16
    )
    moved = renderer.render(
        moved_cam, size=size, jitter=True, frame_index=100, accumulate=True
    )
    assert renderer._accum_count == 1
    assert renderer._last_motion_reset

    renderer.reset_accumulation()
    fresh_moved = renderer.render(
        moved_cam, size=size, jitter=True, frame_index=100, accumulate=True
    )
    assert np.array_equal(moved, fresh_moved)
    assert np.mean(np.abs(moved.astype(np.int16) - static.astype(np.int16))) > 2.0


def test_hud_off_by_default_offscreen(renderer):
    """render() and render(hud=False) are the same HUD-free image."""
    import cloudyview as cv

    cam = cv.Camera()
    a = renderer.render(cam, HUD_SIZE)
    b = renderer.render(cam, HUD_SIZE, hud=False)
    assert np.array_equal(a, b)


def test_hud_differs_only_in_expected_corner_region(renderer):
    """hud=True changes only the top-right minimap rectangle."""
    import cloudyview as cv

    cam = cv.Camera()
    off = renderer.render(cam, HUD_SIZE, jitter=False).astype(int)
    on = renderer.render(cam, HUD_SIZE, jitter=False, hud=True).astype(int)

    diff = np.abs(on - off).sum(axis=-1)
    changed = diff > 4
    n = int(changed.sum())
    assert n > 500, "HUD invisible: too few pixels changed"

    x, y, w, h = renderer.hud.rect_for_size(HUD_SIZE)
    x0 = max(0, int(np.floor(x)) - 2)
    y0 = max(0, int(np.floor(y)) - 2)
    x1 = min(HUD_SIZE[0], int(np.ceil(x + w)) + 2)
    y1 = min(HUD_SIZE[1], int(np.ceil(y + h)) + 2)

    expected = np.zeros(changed.shape, dtype=bool)
    expected[y0:y1, x0:x1] = True
    assert not changed[~expected].any(), "HUD changed pixels outside corner"

    rect_area = max(1, (x1 - x0) * (y1 - y0))
    assert n > 0.25 * rect_area, "HUD changed too little of its rectangle"


def test_hud_marker_position_tracks_camera_position(renderer):
    """The per-frame HUD marker uniform follows relative camera x/y."""
    import cloudyview as cv

    center = cv.Camera(position=(0.0, 0.0, -0.95), azimuth=0.0,
                       elevation=0.0, fov=80.0)
    northeast = cv.Camera(position=(1.0, 1.0, -0.95), azimuth=0.0,
                          elevation=0.0, fov=80.0)

    renderer.hud.write_uniforms(center, HUD_SIZE)
    center_px = renderer.hud._last_state["marker_pixel"]
    renderer.hud.write_uniforms(northeast, HUD_SIZE)
    corner_px = renderer.hud._last_state["marker_pixel"]

    _x, _y, w, h = renderer.hud.rect_for_size(HUD_SIZE)
    assert corner_px[0] > center_px[0] + 0.45 * w
    assert corner_px[1] < center_px[1] - 0.45 * h


def test_hud_benchmark_runs(renderer):
    """benchmark(hud=True) exercises the overlay timestamp path."""
    import cloudyview as cv

    res = renderer.benchmark(cv.Camera(), size=(160, 90),
                             n_warmup=2, n_frames=5, hud=True)
    assert res["wall_ms_mean"] < 250.0


def test_volume_upload_once(renderer):
    """Rendering N frames must not re-upload the volume (resident texture)."""
    import cloudyview as cv

    # The texture is created in __init__; render() only writes the 128-byte
    # uniform buffer. Smoke-check by timing: 5 frames at a tiny size should
    # be far faster than one volume upload would allow if re-uploaded.
    res = renderer.benchmark(cv.Camera(), size=(160, 90),
                             n_warmup=2, n_frames=5)
    assert res["wall_ms_mean"] < 250.0
