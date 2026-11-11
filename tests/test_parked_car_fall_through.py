"""
A parked car is not hidden by the bounding box of the one in front of it.

The street stage sphere-traces a cell's parked cars one bounding box at a
time (streetlife cc_streetlife_near_trace): the nearest box the ray enters,
then, if the ray misses that car's hull inside its box, the next box along
the ray, up to cc_streetlife_NEAR_TRIES boxes. Until 2026-11-09 it tried
one: a ray that entered a car's box and missed the hull returned a miss, and
from a few metres up the ray to a parked car's lower body passes through the
box of the car parked in front of it, just over that car's roof. Thomas saw
the body vanish behind an invisible box face, its roof left floating ("it
looked as though there was an invisible object in front of the car"; flight
of dev 1853300).

This renders the street stage's near trace (patched into the plain pass, as
test_city_props_mirror probes city_trace) over street views low along the
parking lanes, with the tree's try count and with an exhaustive one, and
requires the same hit on every pixel: NEAR_TRIES is enough on these views.
The negative control is the old single try, which must disagree with the
exhaustive trace on many pixels, or these views would not be holding
anything.

Skips without a real GPU adapter. Run it through tools/gpulock.
"""

from pathlib import Path
import re

import numpy as np
import pytest
from .conftest import ShaderModule

pytest.importorskip("wgpu")

try:
    from .conftest import build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import build_soar_level, soar_gpu_adapter

REPO = Path(__file__).resolve().parents[1]
SHADER = ShaderModule("city")

ANCHOR = ("    let chit = city_trace(city_o, dir, 0.0, CITY_TRACE_RANGE, "
          "CITY_TRACE_ALL);\n")
# The street stage's near trace, bounded by the plain pass's surface as the
# street stage bounds it: (kind, t), t -1 on a miss.
PROBE = (
    "    let nh = city_trace(city_o, dir, 0.0,\n"
    "                        select(CITY_TRACE_RANGE, chit.t, chit.hit),\n"
    "                        CITY_TRACE_NEAR);\n"
    "    return vec4<f32>(f32(nh.kind), select(-1.0, nh.t, nh.hit), 0.0, 0.0);\n"
)
TRIES = "const cc_streetlife_NEAR_TRIES: i32 = {};"

# Tile frame, as test_city_props_mirror's VIEWS: x, y in metres from the tile
# origin, z in world metres; az, el, fov in degrees. The minor street at
# centreline x = 28080 parks cars along its kerb; these look along it.
VIEWS = {
    # Eye level, north up the street (test_city_props_mirror's street_eye).
    "street_eye": (28081.0, 51500.0, 2.2, 0.0, -2.0, 70.0),
    # A storey up, looking down the parking lane: the rays to each car's
    # lower body cross the box of the car in front just over its roof.
    "lane_storey": (28081.0, 51500.0, 5.0, 0.0, -8.0, 70.0),
    # Two storeys up, a little across the lane.
    "lane_high": (28078.0, 51490.0, 7.0, 8.0, -12.0, 60.0),
}
SIZE = (480, 270)


def _tries():
    m = re.search(r"^const cc_streetlife_NEAR_TRIES: i32 = (\d+);",
                  SHADER.read_text(), re.M)
    assert m, "cc_streetlife_NEAR_TRIES is gone from raymarch.wgsl"
    return int(m.group(1))


def _near_hits(tries):
    """{view: (kind, t)} of the near trace with `tries` boxes per cell."""
    import wgpu
    from cloudyview.soar_host import (APP_LIGHT_MARCH_LOD_DEGREES,
                                      SceneState, SoarRenderer, ViewState)
    from cloudyview.witness import OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(ANCHOR) == 1, "surface_pass_city's trace moved"
    const = TRIES.format(_tries())
    assert source.count(const) == 1
    source = source.replace(const, TRIES.format(tries))
    source = source.replace(ANCHOR, ANCHOR + PROBE)
    level = build_soar_level()
    r = SoarRenderer(periodic=True, surface="city", shader_source=source)
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
    w, h = SIZE
    out = {}
    for name, (x, y, z, az, el, fov) in VIEWS.items():
        view = ViewState(
            camera_position=[x + off[0], y + off[1], z],
            azimuth=az, elevation=el, fov=fov,
            output_size=SIZE, render_size=SIZE,
            jitter=False, subpixel=False,
            sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
            light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES)
        r.render(state, view, frames=1)
        tex = r._targets["plain"]
        row = -(-w * 16 // 256) * 256
        buf = r.device.create_buffer(
            size=row * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        enc = r.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": buf, "offset": 0, "bytes_per_row": row,
             "rows_per_image": h}, (w, h, 1))
        r.device.queue.submit([enc.finish()])
        buf.map_sync(wgpu.MapMode.READ)
        a = np.frombuffer(bytearray(buf.read_mapped()), np.float32)
        buf.unmap()
        a = a.reshape(h, row // 4)[:, :w * 4].reshape(h * w, 4)
        out[name] = (np.rint(a[:, 0]).astype(int), a[:, 1].copy())
    return out


def _is_car(kind, t):
    return (t >= 0.0) & (kind >= 102) & (kind < 106)


@pytest.fixture(scope="module")
def near_hits():
    if soar_gpu_adapter() is None:
        pytest.skip("no real GPU adapter")
    return {n: _near_hits(n) for n in (_tries(), 64, 1)}


def test_a_car_behind_another_is_traced(near_hits):
    tree, full = near_hits[_tries()], near_hits[64]
    for name in VIEWS:
        (k0, t0), (k1, t1) = tree[name], full[name]
        cars = int(_is_car(k1, t1).sum())
        assert cars >= 500, f"{name}: only {cars} car pixels; the view moved"
        bad = (k0 != k1) | (np.abs(t0 - t1) > 1e-4 * np.maximum(t1, 1.0))
        assert not bad.any(), (
            f"{name}: {int(bad.sum())} pixels meet a different first car "
            f"with {_tries()} tries than with every box tried")


def test_one_try_loses_cars(near_hits):
    # The negative control: the old behaviour, one box per cell, must miss
    # cars the exhaustive trace finds on these views.
    one, full = near_hits[1], near_hits[64]
    lost = sum(int((_is_car(*full[n]) & ~_is_car(*one[n])).sum())
               for n in VIEWS)
    assert lost >= 200, f"one try lost only {lost} car pixels"
