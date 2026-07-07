"""Tests for the wgpu/WGSL interactive renderer (cloudyview.soar).

Skip-marked when wgpu (the 'interactive' extra) or a GPU adapter with
float32-filterable is unavailable, so the suite stays green on CI boxes
without a GPU.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_FILE = Path(__file__).parent.parent / "data" / "TWPICE_subvolume_256x256_5km.nc"

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


def test_volume_upload_once(renderer):
    """Rendering N frames must not re-upload the volume (resident texture)."""
    import cloudyview as cv

    # The texture is created in __init__; render() only writes the 128-byte
    # uniform buffer. Smoke-check by timing: 5 frames at a tiny size should
    # be far faster than one volume upload would allow if re-uploaded.
    res = renderer.benchmark(cv.Camera(), size=(160, 90),
                             n_warmup=2, n_frames=5)
    assert res["wall_ms_mean"] < 250.0
