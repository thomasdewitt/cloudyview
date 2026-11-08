"""
The flight collider's props are the renderer's props.

web/soar/cityprops.js mirrors, as solids, what the city's components stand
in the street and the sky — the freeways, the lamps, the parked cars, the
skybridges, the warehouse roofs' ridges — so that the flight stops where the
thing on screen is (collide.js, through citylots.js CityLots). This holds the
mirror to the shader the way the shader is actually used: the city's plain
pass (surface_pass_city) is patched to write, for every pixel, its ray and
what city_trace hit along it (distance and hit kind), the views are rendered
on the GPU, and the same rays are swept through the JS city with a point
body. Then:

  - where the GPU's first hit is a prop the mirror makes solid, the JS meets
    that kind of prop at the same distance, to a couple of centimetres plus
    the float32 spacing (the GPU traces in float32 at city coordinates of
    tens of kilometres);
  - where the GPU's first hit is anything else, the JS meets no prop in
    front of it — a solid where nothing is drawn is a wall of air.

The buildings are citylots.js's and their own mirror test's: the collider
grows them by their facade relief (which the plain pass does not draw), so
here a building may stand in front of a prop in the JS by up to that relief,
and those rays are left out of the first check and counted.

Measured (2026-11-08, eight views, ~20,000 rays): no disagreement. Its
negative controls, each a copy of cityprops.js with one prop moved: the
parked cars 4 cm further out from the kerb, 2,173 rays fail; the freeway
deck 3 cm higher, 3,717; the lamp masts 2 cm fatter, 1,030; the sawtooth
teeth 5 cm taller, 1,862. Its first run found a renderer bug, not a mirror
one: the freeway deck's fascia and its rails' outer faces were not drawn
from outside the corridor (skyway.wgsl, cc_skyway_route).

Skips without node or a real GPU adapter. Run it through tools/gpulock.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

REPO = Path(__file__).resolve().parents[1]
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"
SOAR = REPO / "web" / "soar"
TILE_DIR = REPO / "cloudyview" / "soar" / "city"

ANCHOR = ("    let chit = city_trace(city_o, dir, 0.0, CITY_TRACE_RANGE, "
          "CITY_TRACE_ALL);\n")
# Two probes, one render target's worth each: the ray and the hit distance
# (-1 on a miss), then the hit kind and the ray's city-frame origin.
PROBES = (
    "    return vec4<f32>(dir, select(-1.0, chit.t, chit.hit));\n",
    "    return vec4<f32>(f32(chit.kind), city_o);\n",
)

# Hit kinds (each component's kind_base + local, see their headers) and the
# source cityprops.js names the solid it mirrors them with.
KINDS = {"skyway": range(200, 216), "lamp": (100, 101),
         "car": range(102, 106), "bridge": range(500, 506),
         "roof": (700, 701)}

# What the renderer leaves out to save drawing it, which the solids do not:
# the parked cars inside half a metre a pixel are sphere-traced hulls inside
# their box (the street stage's), here all drawn as the two boxes they are
# from there on, and a pole's arm, dropped past 0.3 m a pixel, drawn at
# every footprint. Neither changes where anything stands.
ECONOMY = (
    ("const cc_streetlife_CAR_SDF_FP: f32 = 0.50;",
     "const cc_streetlife_CAR_SDF_FP: f32 = -1.0;"),
    ("    // luminaire carry the pole, which is what the eye reads anyway.\n"
     "    if (fp < 0.30) {",
     "    // luminaire carry the pole, which is what the eye reads anyway.\n"
     "    if (true) {"),
)

# Tile frame: x, y in metres from the tile origin, z in world metres; az, el,
# fov in degrees.
VIEWS = {
    # Low along the x-route at y = 51840 (block row 576), looking east.
    "deck_along": (28000.0, 51840.0, 45.0, 90.0, -6.0, 70.0),
    # Under that deck, between its two lines of columns, looking along it.
    "under_deck": (28100.0, 51842.0, 8.0, 85.0, 6.0, 80.0),
    # The interchange at (28800, 51840): the y-route's ramps and high span.
    "interchange": (28650.0, 51600.0, 40.0, 35.0, -10.0, 70.0),
    # Eye level down a minor street (centreline x = 28080, block column 312),
    # north: the lamp lines either side and the parked cars.
    "street_eye": (28081.0, 51500.0, 2.2, 0.0, -2.0, 70.0),
    # From a storey up, looking down across a kerb at a slant.
    "kerb_down": (28074.0, 51560.0, 7.0, 30.0, -25.0, 70.0),
    # The skybridge on the edge between blocks (334, 563) and (335, 563),
    # its deck at about 133 m, from the street side, level.
    "skybridge": (30150.0, 50600.0, 135.0, 0.0, 0.0, 60.0),
    # A sawtooth warehouse roof (block (343, 598), eaves at 13 m) and a
    # monitor roof (block (345, 606), 13.5 m), from above a corner.
    "sawtooth": (30860.0, 53800.0, 30.0, 30.0, -25.0, 70.0),
    "monitor": (31070.0, 54520.0, 28.0, 30.0, -20.0, 70.0),
}
SIZE = (480, 270)
MAX_PROBED = 6000
WINDOW = 4.0
TOL_ABS = 0.01
TOL_REL = 2e-7


def _probe_renders(view_names):
    """{view: (dirs, t, kind, origin)} from the patched plain pass."""
    import wgpu
    from cloudyview.soar_host import (APP_LIGHT_MARCH_LOD_DEGREES,
                                      SceneState, SoarRenderer, ViewState)
    from cloudyview.witness import OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(ANCHOR) == 1, "surface_pass_city's trace moved"
    for old, new in ECONOMY:
        assert source.count(old) == 1, f"moved: {old!r}"
        source = source.replace(old, new)
    level = build_soar_level()
    out = {name: [] for name in view_names}
    for probe in PROBES:
        r = SoarRenderer(periodic=True, surface="city",
                         shader_source=source.replace(ANCHOR, ANCHOR + probe))
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
        for name in view_names:
            x, y, z, az, el, fov = VIEWS[name]
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
            out[name].append(
                a.reshape(h, row // 4)[:, :w * 4].reshape(h * w, 4).copy())
    res = {}
    for name, (pa, pb) in out.items():
        dirs = pa[:, :3].astype(np.float64)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        res[name] = (dirs, pa[:, 3].astype(np.float64),
                     np.rint(pb[:, 0]).astype(int),
                     pb[0, 1:4].astype(np.float64))
    return res


# Where the renderer stops drawing a prop: the footprint (m a pixel) for the
# street's cars and poles (streetlife CAR_FAR_FP, POLE_FAR_FP), and for
# everything a block draws, CITY_PROP_RANGE metres. The freeways are drawn
# at every range.
def _shader_const(name):
    import re
    m = re.search(rf"^const {name}: f32 = ([-\d.eE+]+);", SHADER.read_text(),
                  re.M)
    assert m, f"{name} is gone from raymarch.wgsl"
    return float(m.group(1))


DRAWN_FP = {"car": _shader_const("cc_streetlife_CAR_FAR_FP"),
            "lamp": _shader_const("cc_streetlife_POLE_FAR_FP")}
PROP_RANGE = {s: _shader_const("CITY_PROP_RANGE")
              for s in ("car", "lamp", "bridge", "roof")}


def _drawn(source, t, pixel_angle):
    return (t * pixel_angle < DRAWN_FP.get(source, float("inf"))
            and t < PROP_RANGE.get(source, float("inf")))


def _pixel_angle(view):
    import math
    return 2.0 * math.tan(math.radians(0.5 * VIEWS[view][5])) / SIZE[0]


def _source_of(kind):
    for source, kinds in KINDS.items():
        if kind in kinds:
            return source
    return None


JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    import { CityTile, CityLots } from "%(citylots)s";
    import { CityProps } from "%(cityprops)s";
    const meta = JSON.parse(readFileSync("%(meta)s", "utf8"));
    const bytes = readFileSync("%(bin)s");
    const tile = CityTile.fromHalfBytes(
      new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
      meta.n, meta.cell_m);
    const city = new CityLots(tile, new CityProps(tile));
    const EPS = %(eps)r;
    const { origin, rays } = JSON.parse(readFileSync(0, "utf8"));
    const rows = rays.map(([dx, dy, dz, t0, t1]) => {
      const at = (t) => [origin[0] + dx * t, origin[1] + dy * t,
                         origin[2] + dz * t];
      return [EPS, -EPS].map((r) => {
        const hit = city.sweep(at(t0), at(t1), r);
        if (hit === null) return [-1.0, null, 1.0];
        const n = hit.normal;
        return [t0 + hit.t * (t1 - t0), hit.source,
                Math.abs(dx * n[0] + dy * n[1] + dz * n[2])];
      });
    });
    process.stdout.write(JSON.stringify(rows));
""") % {"citylots": (SOAR / "citylots.js").as_uri(),
        "cityprops": (SOAR / "cityprops.js").as_uri(),
        "meta": (TILE_DIR / "meta.json").as_posix(),
        "bin": (TILE_DIR / "fif_mip0.bin").as_posix(), "eps": 0.005}


def _js_hits(origin, rays):
    out = subprocess.run(["node", "--input-type=module", "--eval", JS],
                         input=json.dumps({"origin": list(origin),
                                           "rays": rays}),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _check_view(dirs, t_gpu, kind, origin, pixel_angle, rng):
    """(bad rows, prop rays compared, prop rays behind a building's relief)

    Each ray is swept over WINDOW metres before the GPU's hit to just past
    it, not from the eye: what the collider sweeps is a frame's motion, a few
    metres, and over a ray hundreds of metres long the renderer economizes in
    ways that change nothing a flight can meet — it raises only the freeway
    crossing nearest the stretch of deck the ray crosses, walks three bays of
    columns past the ray's entry under a deck and five parking slots along a
    kerb. Locally the two must agree.

    The tolerance is in the surface's own normal direction: a centimetre
    plus the float32 spacing at city coordinates of tens of kilometres
    (positions there are good to a few millimetres), divided by the cosine
    at which the ray meets the surface — at a grazing angle a millimetre
    across the face is centimetres along the ray. And each ray is swept
    twice, by a body of 5 mm and by one of -5 mm (the solids shrunk), which
    brackets float32's decision for a ray that passes within millimetres of
    an edge: a mast is 17 cm across, and a corner the GPU's ray clips the
    exact one can miss, or the other way round.

    A prop beyond the footprint the renderer stops drawing it at is not a
    phantom: cars end at DRAWN_FP["car"] m a pixel, poles at
    DRAWN_FP["lamp"], everything a block draws at CITY_PROP_RANGE."""
    src = np.array([_source_of(k) if t > 0 else None
                    for k, t in zip(kind, t_gpu)], dtype=object)
    prop = np.nonzero(src != None)[0]   # noqa: E711
    other = np.nonzero(src == None)[0]   # noqa: E711
    if len(prop) > MAX_PROBED // 2:
        prop = rng.choice(prop, MAX_PROBED // 2, replace=False)
    if len(other) > MAX_PROBED // 2:
        other = rng.choice(other, MAX_PROBED // 2, replace=False)
    idx = np.concatenate([prop, other])
    idx = idx[t_gpu[idx] > 0]
    rays = [[*dirs[i], max(float(t_gpu[i]) - WINDOW, 0.0),
             float(t_gpu[i]) + 1.0] for i in idx]
    js = _js_hits(origin, rays)
    bad, compared, shadowed = [], 0, 0
    for i, ((tg_, sg, cg), (ts, ss, cs)) in zip(idx, js):
        tg = t_gpu[i]

        def tol(cos):
            return (TOL_ABS + TOL_REL * tg) / max(cos, 0.05)
        if src[i] is not None:
            if "building" in (sg, ss) and min(
                    t for t, s in ((tg_, sg), (ts, ss)) if s == "building") < tg:
                shadowed += 1
                continue
            compared += 1
            # The GPU's hit lies between the grown sweep's (no later) and the
            # shrunk sweep's (no earlier; it may miss a face the ray only
            # grazes), on the same kind of prop.
            lo_ok = sg == src[i] and tg >= tg_ - tol(cg)
            hi_ok = ss is None or ts < 0 or tg <= ts + tol(cs)
            if not (lo_ok and hi_ok):
                bad.append((int(i), src[i], float(tg), sg, tg_, ss, ts))
                continue
        # Nothing the mirror makes solid stands robustly in front of what
        # the GPU drew (the shrunk sweep), within the range it is drawn at.
        if (ss not in (None, "building", "ground") and 0 <= ts < tg - tol(cs)
                and _drawn(ss, ts, pixel_angle)):
            bad.append((int(i), src[i] or int(kind[i]), float(tg), ss, ts))
    return bad, compared, shadowed


def test_every_mirrored_constant_is_the_shaders():
    """cityprops.js mirrors its constants by name, CITY_ (the core's) and
    cc_<component>_ (the composed components' in raymarch.wgsl)."""
    import re
    pat = r"^const\s+((?:CITY|cc)_\w+)\s*:\s*(?:f32|i32)\s*=\s*([-\d.eE+]+)\s*;"
    wgsl = {m.group(1): float(m.group(2))
            for m in re.finditer(pat, SHADER.read_text(), re.M)}
    js = {m.group(1): float(m.group(2)) for m in re.finditer(
        r"^export\s+const\s+((?:CITY|cc)_\w+)\s*=\s*([-\d.eE+]+)\s*;",
        (SOAR / "cityprops.js").read_text(), re.M)}
    assert len(js) > 60, "cityprops.js exports fewer constants than it mirrors"
    missing = sorted(set(js) - set(wgsl))
    assert not missing, f"cityprops.js mirrors constants the shader lacks: {missing}"
    differ = {k: (js[k], wgsl[k]) for k in js if js[k] != wgsl[k]}
    assert not differ, f"cityprops.js disagrees with raymarch.wgsl: {differ}"


@pytest.mark.gpu
@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_js_props_meet_the_rays_the_gpu_traces():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    renders = _probe_renders(list(VIEWS))
    rng = np.random.default_rng(20261108)
    report, seen = [], {}
    for name, (dirs, t, kind, origin) in renders.items():
        bad, compared, shadowed = _check_view(dirs, t, kind, origin,
                                              _pixel_angle(name), rng)
        seen[name] = (compared, shadowed, len(bad))
        report += [(name, *b) for b in bad[:6]]
    vacuous = [n for n, (c, _, _) in seen.items() if c < 200]
    assert not vacuous, (
        f"views that no longer look at a mirrored prop (fewer than 200 rays "
        f"hit one), and would pass vacuously: {vacuous}; {seen}")
    assert not report, (
        "the JS props and the GPU's disagree (view, pixel, GPU source, GPU t, "
        f"JS source, JS t): {report}; rays compared, shadowed, bad: {seen}")
