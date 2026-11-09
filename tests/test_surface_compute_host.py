"""The Python host's surface_cs experiment (SoarRenderer surface_cs, the
browser's ?surfacecs): opt-in, loud, and the same picture to rounding.

Off, the renderer builds what it built before: every surface stage a render
pipeline, no storage-writable target. On, the
city's first stage is the compute pass cs_surface (raymarch.wgsl) at the
asked workgroup, and its targets are storage-writable; a device passed in
whose limits do not allow the workgroup is refused, and a workgroup nobody
defined raises. The compute stage renders the fragment stage's picture to
rounding: the two compile to different programs, so a 1-ulp difference in a
hit distance can tip a hashed or floored decision on a few pixels
(test_cs_and_frag_agree has the measured bounds).

GPU tests skip without a real adapter. Run them through tools/gpulock.
"""

from pathlib import Path

import numpy as np
import pytest

from cloudyview.soar_host import (check_surface_cs, parse_surface_cs,
                                  read_shader, surface_cs_limits,
                                  surface_cs_source)

try:
    from .conftest import build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import build_soar_level, soar_gpu_adapter

REPO = Path(__file__).resolve().parents[1]

# Aerial, street-level and grazing city views (test_parked_car_fall_through's
# street geometry and the night-city harness's heights).
VIEWS = {
    "aerial": (28081.0, 51500.0, 3300.0, 45.0, -28.0, 65.0),
    "street_eye": (28081.0, 51500.0, 2.2, 0.0, -2.0, 70.0),
    "lane_high": (28078.0, 51490.0, 7.0, 8.0, -12.0, 60.0),
    "horizon": (40000.0, 60000.0, 1600.0, 227.0, -4.0, 68.0),
}
SIZE = (480, 270)
# cs vs frag through render(), measured 2026-11-09 on these views (RTX 5080,
# wgpu-native 29): at most 0.003% of pixels apart by more than 1/255, mean
# |difference| at most 2e-6 (a single pixel flipping a hit reaches 0.22).
# The bounds are ~30-50x that: rounding, not a different picture.
MAX_FRACTION_OVER_1_255 = 1e-3
MAX_MEAN_ABS = 1e-4


def test_workgroups_and_their_spellings():
    assert check_surface_cs(None) is None
    assert check_surface_cs([32, 16]) == (32, 16)
    assert check_surface_cs((16, 16)) == (16, 16)
    for bad in ((64, 8), (32, 32), (8, 8)):
        with pytest.raises(ValueError, match="expected None or one of"):
            check_surface_cs(bad)
    assert parse_surface_cs("frag") is None
    assert parse_surface_cs("cs512") == (32, 16)
    assert parse_surface_cs("cs256") == (16, 16)
    with pytest.raises(ValueError):
        parse_surface_cs("cs1024")
    assert surface_cs_limits(None) == {}
    assert surface_cs_limits((32, 16)) == {
        "max_compute_invocations_per_workgroup": 512,
        "max_compute_workgroup_size_x": 32,
        "max_compute_workgroup_size_y": 16}


def test_the_workgroup_is_written_into_the_source():
    src = read_shader()
    out = surface_cs_source(src, (16, 16))
    assert "const SURFACE_CS_WX: u32 = 16u;" in out
    assert "const SURFACE_CS_WY: u32 = 16u;" in out
    assert "override SURFACE_CS" not in out
    with pytest.raises(RuntimeError, match="declares SURFACE_CS_WX 0 times"):
        surface_cs_source(out, (32, 16))


@pytest.fixture(scope="module")
def gpu():
    if soar_gpu_adapter() is None:
        pytest.skip("no real GPU adapter")
    import wgpu
    return wgpu


def _view(state, name):
    from cloudyview.soar_host import APP_LIGHT_MARCH_LOD_DEGREES, ViewState
    x, y, z, az, el, fov = VIEWS[name]
    off = state.city_tile_offset_m
    return ViewState(camera_position=[x + off[0], y + off[1], z],
                     azimuth=az, elevation=el, fov=fov,
                     output_size=SIZE, render_size=SIZE,
                     jitter=False, subpixel=False,
                     sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
                     light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES)


def _city(surface_cs, level):
    from cloudyview.soar_host import SceneState, SoarRenderer
    from cloudyview.witness import OCEAN_REFLECTANCE
    r = SoarRenderer(periodic=True, surface="city", surface_cs=surface_cs)
    r.upload_volume(level.sigma)
    meta = r.surface_meta
    dx = float(min(level.dx))
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dx * 2.0, dt_light=dx * 2.0, periodic=True,
        ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=float(meta["cell_m"]),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1, ocean_enabled=True,
        surface="city")
    return r, state


def _stage_kinds(wgpu, r, state):
    view = _view(state, "aerial")
    targets = r._ensure_targets(SIZE, view)
    stages = r._surface_stages(targets, view)
    kinds = ["compute" if isinstance(p, wgpu.GPUComputePipeline) else "render"
             for p, _g, _w in stages]
    storage = [bool(targets[k].usage & wgpu.TextureUsage.STORAGE_BINDING)
               for k in ("surface", "veil", "plain", "mid")]
    return kinds, storage


@pytest.fixture(scope="module")
def level():
    return build_soar_level()


def test_off_builds_what_it_did(gpu, level):
    r, state = _city(None, level)
    kinds, storage = _stage_kinds(gpu, r, state)
    assert kinds == ["render", "render", "render"]
    assert storage == [False, False, False, False]


@pytest.mark.parametrize("wg", [(32, 16), (16, 16)])
def test_on_makes_the_first_stage_compute(gpu, level, wg):
    r, state = _city(wg, level)
    kinds, storage = _stage_kinds(gpu, r, state)
    assert kinds == ["compute", "render", "render"]
    assert storage == [True, True, True, True]
    assert r.device.limits["max-compute-invocations-per-workgroup"] >= wg[0] * wg[1]


def test_a_device_that_cannot_run_the_workgroup_is_refused(gpu):
    from cloudyview.soar_host import SoarRenderer
    # A device held at WebGPU's floor (wgpu-native would otherwise hand
    # out the adapter's own limits).
    device = gpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync(
            required_limits={"max_compute_invocations_per_workgroup": 256})
    assert device.limits["max-compute-invocations-per-workgroup"] == 256
    with pytest.raises(ValueError, match="do not allow the workgroup"):
        SoarRenderer(device=device, periodic=True, surface="city",
                     surface_cs=(32, 16))


def test_cs_and_frag_agree(gpu, level):
    frag, state = _city(None, level)
    cs, _ = _city((32, 16), level)
    for name in VIEWS:
        view = _view(state, name)
        a = frag.render(state, view, frames=1)
        b = cs.render(state, view, frames=1)
        d = np.abs(a - b).max(axis=-1)
        assert np.mean(d > 1.0 / 255.0) < MAX_FRACTION_OVER_1_255, name
        assert float(np.mean(d)) < MAX_MEAN_ABS, name
