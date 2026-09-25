"""Lens water's state (raymarch.wgsl fs_lens_state) and its dry contract.

The wetness a replay draws must not depend on how its frames split the
time: the browser's video export and `soar TRACK.json` both step the state
with the resampled track's fixed dt, and the law is linear in the wetness,
so a step of any length is exact at constant extinction. These pin the GPU
state pass against a numpy reference of that law, on a field of uniform
extinction (so sigma at the camera is known exactly), and pin the dry
contract every still depends on: no dt, no lens, byte for byte.
"""

import math

import numpy as np
import pytest

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

# raymarch.wgsl's lens-water constants.
L_WET, K_WET, K_DRY, DRY_FLOOR, PERIOD = 30.0, 0.6, 0.2, 0.002, 600.0
SIGMA = 0.0625           # m^-1: exact in half and single float


def reference_step(w, sigma, dt):
    """One step of dw/dt = K_WET f (1 - w) - K_DRY (1 - f) w, exactly."""
    f = 1.0 - math.exp(-sigma * L_WET)
    a = K_WET * f + K_DRY * (1.0 - f)
    w_star = K_WET * f / a
    w = w_star + (w - w_star) * math.exp(-a * dt)
    return 0.0 if (w < DRY_FLOOR and w_star < DRY_FLOOR) else w


@pytest.fixture(scope="module")
def scene():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar_host import SceneState, SoarRenderer, ViewState
    from cloudyview.witness import OCEAN_REFLECTANCE
    level = build_soar_level()
    r = SoarRenderer(periodic=True, surface="ocean")
    dx = float(min(level.dx))
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dx * 2.0, dt_light=dx * 2.0, periodic=True,
        ocean_reflectance=OCEAN_REFLECTANCE, ocean_enabled=True)
    mid = [0.5 * (a + b) for a, b in zip(level.bmin, level.bmax)]
    view = ViewState(camera_position=mid, azimuth=30.0, elevation=0.0,
                     fov=70.0, output_size=(64, 36), render_size=(64, 36),
                     sun_azimuth=310.0, sun_elevation=35.0)
    return r, state, view, level.sigma.shape


def _fill(r, shape, sigma):
    r.upload_volume(np.full(shape, sigma, dtype=np.float64))


def test_wetness_follows_the_exact_law_through_a_cloud_and_out(scene):
    r, state, view, shape = scene
    r.reset_lens()
    w = 0.0
    trace = []
    # In: 4 s of dense cloud at 20 fps; out: 6 s of clear air.
    for sigma, n in ((SIGMA, 80), (0.0, 120)):
        _fill(r, shape, sigma)
        for _ in range(n):
            r.render(state, view, frames=1, lens_dt=0.05)
            w = reference_step(w, sigma, 0.05)
            trace.append((float(r.lens_state()[0]), w))
    got, want = np.array(trace).T
    assert np.max(np.abs(got - want)) < 2e-4, (
        f"GPU wetness strays from the law by {np.max(np.abs(got - want)):.2e}")
    assert want[79] > 0.8 and want[-1] < 0.35      # wetted, and draining


def test_one_long_step_equals_many_short_ones(scene):
    r, state, view, shape = scene
    _fill(r, shape, SIGMA)
    r.reset_lens()
    r.render(state, view, frames=1, lens_dt=1.0)
    one = r.lens_state()
    r.reset_lens()
    for _ in range(10):
        r.render(state, view, frames=1, lens_dt=0.1)
    ten = r.lens_state()
    assert abs(one[0] - ten[0]) < 2e-5
    assert abs(one[1] - ten[1]) < 1e-5              # the clock too


def test_a_parked_lens_holds_still(scene):
    r, state, view, shape = scene
    _fill(r, shape, SIGMA)
    r.reset_lens()
    for _ in range(20):
        r.render(state, view, frames=1, lens_dt=0.1)
    before = r.lens_state()
    for _ in range(5):
        r.render(state, view, frames=1, lens_dt=0.0)
    assert np.array_equal(before, r.lens_state())


def test_a_dry_lens_draws_nothing(scene):
    r, state, view, shape = scene
    _fill(r, shape, 0.0)
    r.reset_lens()
    still = r.render(state, view, frames=2)
    r.reset_lens()
    lens = r.render(state, view, frames=2, lens_dt=0.5)   # clear air: dry
    assert r.lens_state()[0] == 0.0
    assert np.array_equal(still, lens)


def test_a_wet_lens_draws_drops(scene):
    r, state, view, shape = scene
    _fill(r, shape, SIGMA)
    r.reset_lens()
    for _ in range(40):
        img = r.render(state, view, frames=1, lens_dt=0.1)
    r.reset_lens()
    dry = r.render(state, view, frames=1)
    assert r.lens_state()[0] == 0.0
    changed = np.any(img != dry, axis=-1).mean()
    assert 0.02 < changed < 0.6, f"{changed:.3f} of the frame under water"
