"""The desert's ground darkens under a deeper overcast, at every tier.

desert_light measures the cloud straight above a ground point with its own
march, direction (0, 0, 1), and the diffuse light the ground gets falls with
that column's optical depth. In a bounded domain the march is clipped by
ray_box, and 1 / (0, 0, 1) has two infinite components; WGSL leaves an
operation that produces an infinity indeterminate, and the driver really did
disagree with itself: the same TWP-ICE ground point measured tau 157 overhead
under one build's Low tier and 0 under its High tier, and a refactor of
unrelated stone code flipped Low from the one to the other (2026-09-24).
Measuring nothing overhead leaves the ground lit by the whole clear sky under
any cloud at all, so no depth of overcast darkens it.

Here one bounded box holds a uniform slab of cloud over the camera, thick
enough that the slant path to the sun is black either way, so what is left
is the diffuse light through the column overhead. Eight times the column has
to take most of it away, in the full desert (High) and the simplified one
(Low). Needs a real GPU; skips cleanly without one.
"""

import numpy as np
import pytest

pytest.importorskip("wgpu")

from cloudyview.soar_host import (SceneState, SoarRenderer, ViewState,   # noqa: E402
                                  desert_scene_fields, surface_cell_m)
from cloudyview.witness import OCEAN_REFLECTANCE, QUALITY_PRESETS        # noqa: E402

try:
    from .conftest import soar_gpu_adapter
except ImportError:
    from conftest import soar_gpu_adapter

BMIN = (0.0, 0.0, 1000.0)
BMAX = (16000.0, 16000.0, 2000.0)
SHAPE = (16, 16, 8)
# Extinction of the slab (1/m): a 1 km column of optical depth 30, and 240.
SIGMA = {"shallow": 0.03, "deep": 0.24}
LUMA = np.array([0.2126, 0.7152, 0.0722])


def ground_luminance(quality, lite, sigma, device=None):
    """Mean linear luminance of a 64 x 36 view of the ground under the slab."""
    preset = QUALITY_PRESETS[quality]
    r = SoarRenderer(periodic=False, surface="desert", tone_map=False,
                     device=device)
    r.upload_volume(np.full(SHAPE, sigma, np.float32))
    tile = r.surface_meta
    dx = (BMAX[0] - BMIN[0]) / SHAPE[0]
    state = SceneState(
        bmin=BMIN, bmax=BMAX,
        dt_view=dx * preset["step_factor"],
        dt_light=dx * preset["light_step_factor"],
        periodic=False, ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=surface_cell_m("desert", tile),
        ocean_tile_extent=float(tile["tile_extent_m"]),
        ocean_max_lod=int(tile["mips"]) - 1,
        surface="desert", **desert_scene_fields(tile))
    # 60 m up, looking steeply down: every pixel is ground well inside the
    # box's footprint, and within the refine reach at High.
    view = ViewState(
        camera_position=(8000.0, 8000.0, 60.0), azimuth=40.0,
        elevation=-50.0, fov=30.0, output_size=(64, 36),
        render_size=(64, 36), sun_azimuth=235.0, sun_elevation=60.0,
        exposure=1.0, light_cache=bool(preset["light_cache"]),
        sky_probe=bool(preset["sky_probe"]), desert_lite=lite)
    if preset["light_cache"]:
        r.bake_light_cache(state, view, divisor=2)
    hdr = r.render(state, view, frames=2)
    return float((hdr[..., :3] * LUMA).sum(-1).mean()), r.device


@pytest.mark.parametrize("quality, lite", [("high", False), ("low", True)])
def test_a_deeper_overcast_darkens_the_ground(quality, lite):
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count)")
    shallow, device = ground_luminance(quality, lite, SIGMA["shallow"])
    deep, _ = ground_luminance(quality, lite, SIGMA["deep"], device)
    assert shallow > 0.0
    assert deep < 0.5 * shallow, (quality, shallow, deep)
