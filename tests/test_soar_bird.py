"""Tests for the soar flying subject (cloudyview.soar.bird).

Same skip policy as test_soar_render.py: skipped without wgpu (the
'interactive' extra) or a GPU adapter with float32-filterable.
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

SIZE = (480, 270)  # (w, h)


@pytest.fixture(scope="module")
def renderer():
    import cloudyview as cv
    from cloudyview.soar import InteractiveRenderer

    field = cv.load(str(DATA_FILE))
    return InteractiveRenderer(field)


def test_bird_off_by_default_offscreen(renderer):
    """render() and render(bird=False) are the same bird-free image."""
    import cloudyview as cv

    cam = cv.Camera()
    a = renderer.render(cam, SIZE)
    b = renderer.render(cam, SIZE, bird=False)
    assert np.array_equal(a, b)


def test_bird_differs_only_in_small_central_region(renderer):
    """bird=True changes a small patch ahead of/below the view center."""
    import cloudyview as cv

    cam = cv.Camera()
    off = renderer.render(cam, SIZE).astype(int)
    on = renderer.render(cam, SIZE, bird=True).astype(int)

    diff = np.abs(on - off).sum(axis=-1)
    changed = diff > 8
    n = int(changed.sum())
    h, w = changed.shape

    # The bird must actually be there...
    assert n > 15, "bird invisible: no pixels changed"
    # ...must be small (a subject, not a look change)...
    assert n < 0.02 * h * w, f"bird changed {n} px — far too large"
    # ...and confined to the central lower-middle region where it flies.
    ys, xs = np.nonzero(changed)
    assert ys.min() > 0.30 * h and ys.max() < 0.85 * h, \
        f"bird rows {ys.min()}-{ys.max()} outside expected band"
    assert xs.min() > 0.25 * w and xs.max() < 0.75 * w, \
        f"bird cols {xs.min()}-{xs.max()} outside expected band"


def test_bird_flap_animates(renderer):
    """Different wingbeat phases give different images (flap uniform live)."""
    import cloudyview as cv

    cam = cv.Camera()
    up = renderer.render(cam, SIZE, bird=True,
                         bird_pose={"flap_phase": np.pi / 2}).astype(int)
    down = renderer.render(cam, SIZE, bird=True,
                           bird_pose={"flap_phase": 3 * np.pi / 2}).astype(int)
    assert np.abs(up - down).sum() > 0, "flap phase has no effect"


def test_bird_banks(renderer):
    """A banked pose differs from level (roll uniform live)."""
    import cloudyview as cv

    cam = cv.Camera()
    level = renderer.render(cam, SIZE, bird=True,
                            bird_pose={"flap_phase": 0.5}).astype(int)
    banked = renderer.render(cam, SIZE, bird=True,
                             bird_pose={"flap_phase": 0.5,
                                        "bank": 40.0}).astype(int)
    assert np.abs(level - banked).sum() > 0, "bank has no effect"


def test_bird_benchmark_runs(renderer):
    """benchmark(bird=True) exercises the two-pass timestamp path."""
    import cloudyview as cv

    res = renderer.benchmark(cv.Camera(), size=(160, 90),
                             n_warmup=2, n_frames=5, bird=True)
    assert res["wall_ms_mean"] < 250.0
