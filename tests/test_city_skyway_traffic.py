"""
The skyway's traffic is light on the deck, not bright squares.

Each car on an elevated freeway is drawn as the light it throws on the deck
(cc_skyway_streaks in city/components/skyway.wgsl). Its first form was a
7 m trail cut hard at both ends, with a 2.2 m halo, at radiance 16; the tone
map flattened it into a saturated rectangle half a deck wide, and flying low
along an artery Thomas found "spurious bright squares frequently". The pools
that replaced it are Gaussian along and across the lane, with the same light
per car.

This renders a view that reproduced the squares (low along an artery in
the tile's north-east, looking down the deck) twice: as shipped, and with
the traffic term removed from the deck's shading. The difference is the
traffic's own light. Its sharpest edge — the largest step between adjacent
pixels, as a fraction of its brightest pixel — is what a hard-cut patch
maximises and a soft pool keeps small.

Measured (2026-11-08, 32 frames, 768x432, exposure 1, the near deck): the
hard-cut trails 0.75; the pools 0.30 (what remains is the rails and the
paint occluding and the pools' own falloff at a grazing angle). The test refuses to pass vacuously: the traffic must still be a
visible share of the view (its brightest pixel at least MIN_PEAK in
display).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

SHADER = Path(__file__).resolve().parents[1] / "cloudyview/soar/raymarch.wgsl"
TRAFFIC = "        return e * j_mul + cc_skyway_streaks(s, lat, fam, glow, fp);\n"
NO_TRAFFIC = "        return e * j_mul;\n"
# Tile frame: x, y in metres from the tile origin, z in world metres; az,
# el, fov in degrees. 45 m up over the y = 51840 artery (block row 576),
# looking east along it.
CAMERA = (28000.0, 51840.0, 45.0, 90.0, -6.0, 70.0)
SIZE = (768, 432)
FRAMES = 32
# Exposure low enough that nothing on the deck clips: clipped paint would
# hide the traffic's light under it and read as an edge in the difference.
EXPOSURE = 1.0
# The near deck only: from the image centre down (within ~180 m). Nearer
# the vanishing point a pool is a pixel or two, and a large step between
# neighbours is the pool's honest size there.
NEAR_ROW = SIZE[1] // 2
MAX_EDGE = 0.45
MIN_PEAK = 0.15


def render_pair(source, frames=FRAMES):
    """The view shown with and without the traffic term, display [0, 1]."""
    import cloudyview as cv
    from cloudyview.soar_host import (APP_LIGHT_MARCH_LOD_DEGREES,
                                      SceneState, SoarRenderer, ViewState)
    from cloudyview.witness import OCEAN_REFLECTANCE

    assert source.count(TRAFFIC) == 1, (
        "the deck's traffic term moved; update TRAFFIC")
    level = build_soar_level()
    shown = []
    for src in (source, source.replace(TRAFFIC, NO_TRAFFIC)):
        r = SoarRenderer(periodic=True, surface="city", shader_source=src)
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
        off = state.city_tile_offset_m
        x, y, z, az, el, fov = CAMERA
        view = ViewState(
            camera_position=[x + off[0], y + off[1], z],
            azimuth=az, elevation=el, fov=fov,
            output_size=SIZE, render_size=SIZE,
            jitter=True, subpixel=True,
            sun_azimuth=310.0, sun_elevation=22.0, exposure=EXPOSURE,
            light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES)
        img = np.asarray(r.render(state, view, frames=frames))
        shown.append(cv.quantize_uint8(img).astype(np.float64)[..., :3]
                     / 255.0)
    return shown


def traffic_edges(shown):
    """(sharpest edge as a fraction of the peak, the peak), of the traffic's
    own light: the shipped view minus the view without it."""
    lum = (shown[0] - shown[1]).mean(axis=-1)[NEAR_ROW:]
    peak = float(lum.max())
    step = max(float(np.abs(np.diff(lum, axis=0)).max()),
               float(np.abs(np.diff(lum, axis=1)).max()))
    return step / peak, peak


@pytest.mark.gpu
def test_skyway_traffic_has_no_hard_edges():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    edge, peak = traffic_edges(render_pair(SHADER.read_text()))
    assert peak >= MIN_PEAK, (
        f"the traffic's brightest pixel is {peak:.3f}: the view no longer "
        "shows the artery's traffic, so the test would pass vacuously")
    assert edge <= MAX_EDGE, (
        f"the traffic's sharpest edge is {edge:.2f} of its peak: a hard-cut "
        "patch on the deck (the bright squares)")


if __name__ == "__main__":
    # Measure any revision: python tests/test_city_skyway_traffic.py FILE
    src = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else SHADER.read_text()
    print("edge %.3f  peak %.3f" % traffic_edges(render_pair(src)))
