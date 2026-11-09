"""
The facade zones' closed-form cover equals what the shell draws, bay by bay.

The flat facade takes the share of a pixel's footprint each zone holds
(solid side or core, plant, loft, shop) from city_zone_cover, a product of
interval overlaps; the shell asks city_zone_at for each bay it walks. If the
two disagree, a wall changes as the shell hands over to the flat facade, or
as a footprint grows. This probes both on the GPU, on the city's own
buildings: the plain pass, rewritten to take the building each pixel hits
and one of its faces, draws a footprint on that face from the pixel's ray
(centres anywhere on the face, half sizes from 0.3 m to 30 m, wide and
narrow), and returns the largest difference between the closed form and a
128 x 128 point count of city_zone_at over the same footprint. A point
count misplaces each zone edge by up to half a grid step, so the bound is
set by the grid, not by the cover; the half sizes stop at 30 m so that the
step (0.47 m at most) stays well under a bay or a storey, where a coarser
count would alias a one-storey plant band. Wider footprints run the same
closed form.

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/test_soar_city_zones.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

SHADER = Path(__file__).resolve().parents[1] / "cloudyview/soar/raymarch.wgsl"
ANCHOR = "    let t_lim = min(select(1e30, chit.t, chit.hit),"
GRID = 128
# Out: x the largest |cover - count| over the four shares, y whether the
# face has any zone, z the face's bays share of the footprint, w hit.
PROBE = """    if (chit.hit) {
        let pc = city_cell(chit.cell);
        let sd = i32(floor(fract(abs(dir.x) * 97.13 + abs(dir.y) * 31.7)
                           * 3.999));
        let bx = city_tier_box(pc, 0);
        let f = city_face_make(pc, bx.bmin, bx.bmax, sd, 0);
        let z = city_zones(pc, f);
        let any = z.blank || z.store || z.core.y > z.core.x
                  || z.plant.y > 0 || z.crown < (1 << 29) || z.loft.z > 0;
        let r = fract(vec4<f32>(13.7, 7.31, 17.1, 23.9)
                      * (abs(dir.x) + 2.0 * abs(dir.y) + 3.0 * abs(dir.z))
                      * 101.0);
        let c = vec2<f32>(r.x * f.len, f.z0 + r.y * (f.z1 - f.z0));
        // @domain: 100 > 0, exponent a hash draw in [0, 1)
        let hb = 0.3 * vec2<f32>(pow(100.0, r.z), pow(100.0, r.w));
        let zc = city_zone_cover(z, f, c, hb);
        var n = vec4<f32>(0.0);
        let g = f32(GRID_N);
        for (var j: i32 = 0; j < GRID_N; j = j + 1) {
            for (var i: i32 = 0; i < GRID_N; i = i + 1) {
                let q = c - hb + 2.0 * hb * (vec2<f32>(f32(i), f32(j)) + 0.5) / g;
                if (q.y < f.z0 || q.y >= f.z1) {
                    continue;
                }
                let b = city_bay(f, q.x, q.y);
                if (!b.in_bay) {
                    continue;
                }
                let k = city_zone_at(z, f, b.iu, b.iv);
                n = n + vec4<f32>(
                    select(0.0, 1.0, k == CITY_ZONE_BLANK || k == CITY_ZONE_CORE),
                    select(0.0, 1.0, k == CITY_ZONE_PLANT),
                    select(0.0, 1.0, k == CITY_ZONE_LOFT),
                    select(0.0, 1.0, k == CITY_ZONE_STORE));
            }
        }
        let d = abs(zc - n / (g * g));
        let strip = city_bay_strip(f);
        let bays = city_overlap(c.x - hb.x, c.x + hb.x, strip.x, strip.y)
                   * city_overlap(c.y - hb.y, c.y + hb.y, f.z0, f.z1)
                   / (4.0 * hb.x * hb.y);
        return vec4<f32>(max(max(d.x, d.y), max(d.z, d.w)),
                         select(0.0, 1.0, any), bays, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
""".replace("GRID_N", str(GRID))
# Over a dense district, looking across it, so most pixels hit a wall of a
# building that has zones. Tile frame (tests' TILE_OFFSET, as the shell
# continuity test).
TILE_OFFSET = (75330.0, -17010.0)
VIEWS = ((19500.0, 71000.0, 900.0, 45.0, -20.0, 70.0),
         (17000.0, 73000.0, 300.0, 200.0, -12.0, 70.0))
SIZE = (320, 180)
# A grid of GRID points per side misplaces a zone edge by up to half a
# step: two edges on each axis, four half steps on each, as shares.
# Measured (2026-11-09, 109850 probes): worst 0.0084, mean 0.00068; with a
# core not taken out of the plant floors it crosses 1.0, and with the lofts
# a storey too tall 0.33.
MAX_ERR = 4.0 / GRID
MIN_ZONED = 2000


def _plain(r, width, height):
    import wgpu
    tex = r._targets["plain"]
    row = math.ceil(width * 16 / 256) * 256
    buf = r.device.create_buffer(
        size=row * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc = r.device.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf, "offset": 0, "bytes_per_row": row,
         "rows_per_image": height}, (width, height, 1))
    r.device.queue.submit([enc.finish()])
    buf.map_sync(wgpu.MapMode.READ)
    a = np.frombuffer(bytearray(buf.read_mapped()), np.float32)
    buf.unmap()
    return a.reshape(height, row // 4)[:, :width * 4].reshape(height, width,
                                                              4)


@pytest.mark.gpu
def test_zone_cover_matches_the_bays():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar_host import SceneState, SoarRenderer, ViewState
    from cloudyview.witness import OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(ANCHOR) == 1, "the plain pass's anchor moved"
    level = build_soar_level()
    r = SoarRenderer(periodic=True, surface="city",
                     shader_source=source.replace(ANCHOR, PROBE + ANCHOR))
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
    errs, bays = [], []
    for x, y, z, az, el, fov in VIEWS:
        r.render(state, ViewState(
            camera_position=[x + TILE_OFFSET[0], y + TILE_OFFSET[1], z],
            azimuth=az, elevation=el, fov=fov, output_size=SIZE,
            render_size=SIZE, jitter=False, subpixel=False,
            sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
            view_step_lod_degrees=0.0), frames=1)
        p = _plain(r, *SIZE)
        zoned = (p[..., 3] > 0.0) & (p[..., 1] > 0.5)
        errs.append(p[..., 0][zoned])
        bays.append(p[..., 2][zoned])
    e = np.concatenate(errs)
    assert e.size >= MIN_ZONED, f"only {e.size} probes on zoned faces"
    worst = float(e.max())
    assert worst <= MAX_ERR, (
        f"city_zone_cover misses the bay-by-bay count by {worst:.4f} "
        f"(bound {MAX_ERR:.4f}; mean {float(e.mean()):.5f} over {e.size})")
