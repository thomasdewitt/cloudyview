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
    assert "textureDimensions(vol, 0)" in shader
    assert "sigma_data_dims_xyz()" in shader
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

    assert hf(on) > hf(off)


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
