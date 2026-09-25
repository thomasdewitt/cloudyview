"""The desert's trace resolves low grazing views inside its budget.

At 192 steps against the tile's global slope bound, a ray skimming the
ground from a couple of metres up ran out of budget a few hundred metres
out: a rising one drew sky under the far plants (the strip flora found,
2026-09-25), a falling one landed on the mean plane, and most of the ground
of a low view was that plane with the texture sliding on it. The trace now
steps on the bounds pyramid (cloudyview/desert_bounds.py), and a ray that
still spends the budget is finished by a cell walk that lands it in-cell.

DESERT_TRACE_DEBUG paints those rays: magenta when the walk finished one,
cyan when it ran out as well. Here it is switched on for flora's 2.5 m
camera and a 30 m view over the same ground, full and simplified, with
the sky empty: no ray may run out of both (the walk's own fallback is
uncertified, so it must stay unreached), and walk-finished rays stay a
thin scatter. Needs a
real GPU; skips cleanly without one.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu")

from cloudyview.soar_host import (DESERT_LITE_TIERS, SceneState,         # noqa: E402
                                  SoarRenderer, ViewState,
                                  desert_scene_fields, surface_cell_m)
from cloudyview.witness import OCEAN_REFLECTANCE, QUALITY_PRESETS        # noqa: E402

try:
    from .conftest import soar_gpu_adapter
except ImportError:
    from conftest import soar_gpu_adapter

SHADER = (Path(__file__).resolve().parents[1] / "cloudyview" / "soar"
          / "raymarch.wgsl").read_text()
DEBUG_OFF = "const DESERT_TRACE_DEBUG: bool = false;"
# (position in the desert frame, azimuth, elevation, fov): flora's 2.5 m
# camera 30 m from the hero saguaro, and a view from 30 m up.
VIEWS = {"2.5 m": ((873.08, 735.86, 2.51), 325.0, 5.19, 45.0),
         "30 m": ((873.08, 735.86, 30.0), 325.0, -3.0, 45.0)}


def debug_counts(view_name, quality="high"):
    """(walk-finished, still out) pixels of a 1280 x 720 frame."""
    assert SHADER.count(DEBUG_OFF) == 1
    src = SHADER.replace(DEBUG_OFF, "const DESERT_TRACE_DEBUG: bool = true;")
    preset = QUALITY_PRESETS[quality]
    r = SoarRenderer(periodic=True, surface="desert", tone_map=False,
                     shader_source=src)
    r.upload_volume(np.zeros((8, 8, 4), np.float32))
    tile = r.surface_meta
    bmin, bmax = (0.0, 0.0, 1000.0), (4096.0, 4096.0, 2000.0)
    dx = (bmax[0] - bmin[0]) / 8
    state = SceneState(
        bmin=bmin, bmax=bmax,
        dt_view=dx * preset["step_factor"],
        dt_light=dx * preset["light_step_factor"],
        periodic=True, ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=surface_cell_m("desert", tile),
        ocean_tile_extent=float(tile["tile_extent_m"]),
        ocean_max_lod=int(tile["mips"]) - 1,
        surface="desert", **desert_scene_fields(tile))
    pos, az, el, fov = VIEWS[view_name]
    view = ViewState(
        camera_position=pos, azimuth=az, elevation=el, fov=fov,
        output_size=(1280, 720), render_size=(1280, 720),
        sun_azimuth=120.0, sun_elevation=50.0, exposure=1.0,
        light_cache=bool(preset["light_cache"]),
        sky_probe=bool(preset["sky_probe"]),
        desert_lite=quality in DESERT_LITE_TIERS)
    if preset["light_cache"]:
        r.bake_light_cache(state, view, divisor=2)
    img = r.render(state, view, frames=1)[..., :3]
    red, green, blue = img[..., 0], img[..., 1], img[..., 2]
    magenta = (red > 0.5) & (blue > 0.5) & (green < 0.2 * red)
    cyan = (green > 0.5) & (blue > 0.5) & (red < 0.2 * green)
    return int(magenta.sum()), int(cyan.sum())


# Walk-finished rays measured 2026-09-25 (of 921,600): 950 and 76 at High,
# 145 and 484 at Low (the simplified desert); the caps are about twice that.
@pytest.mark.parametrize("view_name, quality, finished_max", [
    ("2.5 m", "high", 2000), ("30 m", "high", 200),
    ("2.5 m", "low", 400), ("30 m", "low", 1000)])
def test_no_ray_runs_out_of_the_trace(view_name, quality, finished_max):
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count)")
    finished, out = debug_counts(view_name, quality)
    assert out == 0, (view_name, finished, out)
    assert finished <= finished_max, (view_name, finished)
