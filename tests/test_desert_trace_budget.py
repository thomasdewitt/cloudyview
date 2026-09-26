"""The desert's trace resolves low grazing views inside its budget.

At 192 steps against the tile's global slope bound, a ray skimming the
ground from a couple of metres up ran out of budget a few hundred metres
out: a rising one drew sky under the far plants (the strip flora found,
2026-09-25), a falling one landed on the mean plane, and most of the ground
of a low view was that plane with the texture sliding on it. The trace now
steps on the bounds pyramid (cloudyview/desert_bounds.py); a ray that
still spends the budget is finished by a cell walk, and marches again from
where the walk stopped.

DESERT_TRACE_DEBUG paints the rays the second march still leaves to the
walk magenta, and those the walk cannot finish either cyan. Here it is
switched on for flora's 2.5 m camera, a 30 m view over the same ground and
the wash below a bank, full and simplified, with the sky empty: none may
be either (a walk-landed ray can be ground in the air, and the walk's own
fallback is uncertified). Needs a real GPU; skips cleanly without one.
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
         "30 m": ((873.08, 735.86, 30.0), 325.0, -3.0, 45.0),
         # Down in a wash 1 m off its floor, looking up at a 2 m bank.
         "wash": ((780.60, 940.51, -5.0), 310.5, 3.0, 60.0)}


def render(view_name, quality="high", patches=(), source=SHADER):
    """A 1280 x 720 linear frame of one view, the shader patched."""
    for old, new in patches:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    preset = QUALITY_PRESETS[quality]
    r = SoarRenderer(periodic=True, surface="desert", tone_map=False,
                     shader_source=source)
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
    return r.render(state, view, frames=1)[..., :3]


def debug_counts(view_name, quality="high"):
    """(walk-finished, still out) pixels of a 1280 x 720 frame."""
    img = render(view_name, quality, [(DEBUG_OFF, "const DESERT_TRACE_DEBUG: "
                                                  "bool = true;")])
    red, green, blue = img[..., 0], img[..., 1], img[..., 2]
    magenta = (red > 0.5) & (blue > 0.5) & (green < 0.2 * red)
    cyan = (green > 0.5) & (blue > 0.5) & (red < 0.2 * green)
    return int(magenta.sum()), int(cyan.sum())


@pytest.mark.parametrize("view_name", list(VIEWS))
@pytest.mark.parametrize("quality", ["high", "low"])
def test_no_ray_is_left_to_the_walk(view_name, quality):
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count)")
    assert debug_counts(view_name, quality) == (0, 0), view_name


def black_pixels(source=SHADER, quality="high"):
    """Pixels of the wash view with no light at all: nothing in a lit
    desert under an open sky is black."""
    img = render("wash", quality, source=source)
    return int((img.sum(-1) < 1e-4).sum())


def test_no_black_line_along_a_lip():
    """Down in a wash looking up at a bank, rays skim its lip. The plain
    pass's budget ran out on some and its cell walk landed them in the lip's
    cell, in the air; the refine stage, re-tracing from just short of that,
    found the ray clears the lip, and on not finding the ground drew black:
    a black line along the silhouette (Thomas's flight, 2026-09-25; 241
    pixels here at 22c7948, 26 of them, sub-floor grazes, already at
    c933289). The trace now marches again from where the walk stopped, and
    the refine stage shades the plain pass's own hit."""
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count)")
    assert black_pixels() == 0
    assert black_pixels(quality="low") == 0
