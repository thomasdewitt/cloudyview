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
L_WET, K_WET, K_DRY, K_DRY_CLEAR, DRY_FLOOR = 30.0, 0.6, 0.2, 0.6, 0.002
SIGMA = 0.0625           # m^-1: exact in half and single float


def reference_step(w, sigma, dt):
    """One step of dw/dt = K_WET f (1 - w) - k_dry (1 - f) w, exactly, with
    k_dry = K_DRY in cloud and K_DRY_CLEAR in clear air: where the first
    law's level is under DRY_FLOOR."""
    f = 1.0 - math.exp(-sigma * L_WET)
    w_first = K_WET * f / (K_WET * f + K_DRY * (1.0 - f))
    k_dry = K_DRY if w_first >= DRY_FLOOR else K_DRY_CLEAR
    a = K_WET * f + k_dry * (1.0 - f)
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
    from cloudyview.soar.renderer import SoarRenderer
    from cloudyview.soar.uniforms import SceneState, ViewState
    from cloudyview.soar.uniforms import DEFAULT_OCEAN_REFLECTANCE
    level = build_soar_level()
    r = SoarRenderer(periodic=True, surface="ocean")
    dx = float(min(level.dx))
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dx * 2.0, dt_light=dx * 2.0, periodic=True,
        ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE, ocean_enabled=True)
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
    # Wetted toward w* = 0.94 (the in-cloud law, K_DRY 0.2), then drained
    # out of cloud at K_DRY_CLEAR: 0.83 exp(-0.6/s * 6 s) = 0.023.
    assert want[79] > 0.8 and want[-1] < 0.03


@pytest.mark.parametrize("sigma", [SIGMA, SIGMA / 8.0, SIGMA / 64.0])
def test_in_cloud_the_law_is_the_first_one(scene, sigma):
    # Only the drain in clear air got faster: in cloud, thin cloud included,
    # w* is the first law's K_WET f / (K_WET f + 0.2 (1 - f)).
    r, state, view, shape = scene
    _fill(r, shape, sigma)
    r.reset_lens()
    r.render(state, view, frames=1, lens_dt=0.5)
    f = 1.0 - math.exp(-sigma * L_WET)
    w_star_first = 0.6 * f / (0.6 * f + 0.2 * (1.0 - f))
    assert abs(float(r.lens_state()[1]) - w_star_first) < 1e-6


def test_out_of_cloud_the_lens_drains_at_three_times_the_rate(scene):
    r, state, view, shape = scene
    _fill(r, shape, SIGMA)
    r.reset_lens()
    r.render(state, view, frames=1, lens_dt=4.0)
    w0 = float(r.lens_state()[0])
    _fill(r, shape, 0.0)
    r.render(state, view, frames=1, lens_dt=2.0)
    w1 = float(r.lens_state()[0])
    assert abs(w1 / w0 - math.exp(-3.0 * 0.2 * 2.0)) < 1e-4


def test_a_faint_tail_of_extinction_is_clear_air(scene):
    # A field's clear air can carry a tiny positive extinction (the RCE
    # demo's tail): it drains at the clear rate, not the in-cloud one.
    r, state, view, shape = scene
    _fill(r, shape, SIGMA)
    r.reset_lens()
    r.render(state, view, frames=1, lens_dt=4.0)
    w0 = float(r.lens_state()[0])
    _fill(r, shape, 1e-6)
    r.render(state, view, frames=1, lens_dt=2.0)
    w1, w_star = (float(v) for v in r.lens_state()[:2])
    assert 0.0 < w_star < DRY_FLOOR
    ratio = (w1 - w_star) / (w0 - w_star)
    assert abs(ratio - math.exp(-K_DRY_CLEAR * 2.0)) < 1e-4


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
    assert abs(one[1] - ten[1]) < 1e-6                         # w*, the same
    assert np.array_equal(one[2:], np.zeros(2, np.float32))


def test_a_zero_step_holds_the_lens_still(scene):
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


def test_the_lens_never_touches_the_accumulation(scene):
    """A lens that changes every frame (the live view's, parked in cloud) is
    drawn after the accumulation and never writes it, so the picture under
    it converges exactly as it would with no lens."""
    r, state, view, shape = scene
    _fill(r, shape, SIGMA)

    def accumulated(**kw):
        r.reset_lens()
        r.render(state, view, frames=4, **kw)
        t = r._targets
        final = [a for a in t["accum"]]
        # render() leaves the finished average in one of the two ping-pong
        # targets; read both, the pair is what the next frame builds on.
        return [r._read_back(a, *t["size"]) for a in final]

    plain = accumulated()
    wet = accumulated(lens_dt=0.5)
    assert r.lens_state()[0] > 0.0
    assert all(np.array_equal(a, b) for a, b in zip(plain, wet))


def test_only_the_lens_state_pass_reads_row_24_w():
    """Row 24.w carries the lens dt, written for the state pass alone (and left
    over in the uniform block for the passes after it): nothing else may read
    it. No GPU."""
    import re
    from pathlib import Path
    from .conftest import ShaderModule
    src = ShaderModule("city").read_text()
    code = re.sub(r"//[^\n]*", "", src)
    start = code.index("fn fs_lens_state(")
    end = code.index("\n}\n", start)
    reads = [m.start() for m in re.finditer(r"surface_offset\.w\b", code)]
    assert reads, "the state pass no longer reads its dt from row 24.w"
    outside = [code[:i].count("\n") + 1 for i in reads if not start <= i < end]
    assert not outside, f"row 24.w read outside fs_lens_state at lines {outside}"
