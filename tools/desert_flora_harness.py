#!/usr/bin/env python
"""Render soar's desert flora from a terminal, framed on a chosen plant.

The iteration loop for the DESERT FLORA section of raymarch.wgsl: the same
shader, uniform block and accumulation the browser runs (SoarRenderer with
surface="desert"), over a prebaked demo's cloud field, with the cameras
placed relative to a plant the shader itself put there.

Finding the plant is the one thing the full renderer cannot do, so a small
compute pass does it: the flora section is lifted out of raymarch.wgsl by
its markers, compiled beside stubs of the ground's API, and asked for every
big cell's item. What it reports is what the render draws, because it is the
same code — there is no Python mirror of the placement to drift out of step.

    tools/gpulock uv run python tools/desert_flora_harness.py --views all
    tools/gpulock uv run python tools/desert_flora_harness.py --time

Views are named by their distance from the hero saguaro (1m, 5m, 30m, 300m)
plus `backlit`, a low sun behind it. `--camera` takes a free view in the
desert frame.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SHADER = ROOT / "cloudyview" / "soar" / "raymarch.wgsl"
FLORA_BEGIN = "// === DESERT FLORA (desert-flora) ==="
FLORA_END = "// === END DESERT FLORA ==="
DEMOS = Path("/home/thomas/code-and-data/cloudyview/web/soar/demos")

# Stubs of the ground API, for the placement scan only. The tile itself is
# bound (the habitat reads its lightness), without the sub-tile octaves;
# the foot heights it reports are the tile's, and the harness does not use
# them.
SCAN_PRELUDE = """
const DESERT_FLORA_SLAB_H: f32 = 13.0;
struct ScanUniforms { cam_origin: vec4<f32>, params: vec4<f32>, };
var<private> u: ScanUniforms;
fn desert_camera_origin() -> vec3<f32> { return vec3<f32>(0.0); }
fn desert_height(xy: vec2<f32>) -> f32 { return 0.0; }
fn desert_normal(xy: vec2<f32>) -> vec3<f32> { return vec3<f32>(0.0, 0.0, 1.0); }
fn desert_height_range() -> vec2<f32> { return vec2<f32>(0.0, 0.0); }
@group(0) @binding(1) var scan_tile: texture_2d<f32>;
@group(0) @binding(2) var scan_samp: sampler;
fn desert_tile(xy: vec2<f32>, lod: f32) -> vec2<f32> {
    return textureSampleLevel(scan_tile, scan_samp, xy / 4096.0, lod).rg;
}
// The tile alone, without the octaves: the habitat reads lightness at
// decimetres, where the octaves are a small correction.
fn desert_surface(xy: vec2<f32>, fp: f32) -> vec2<f32> {
    return desert_tile(xy, 0.0);
}

struct DesertHit { hit: bool, t: f32, pos: vec3<f32>, normal: vec3<f32>,
                   kind: i32, mat: vec4<f32>, };
struct DesertLight { sun_dir: vec3<f32>, sun: vec3<f32>, sky: vec3<f32>,
                     ground: vec3<f32>, t_sun: f32, sky_open: f32, };
fn desert_lambert(albedo: vec3<f32>, n: vec3<f32>, L: DesertLight) -> vec3<f32> {
    return albedo;
}
"""

SCAN_MAIN = """
@group(0) @binding(0) var<storage, read_write> scan_out: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(DF_BIG_N);
    if (gid.x >= n || gid.y >= n) {
        return;
    }
    let ci = vec2<i32>(gid.xy);
    let org = vec2<f32>(ci) * DF_BIG_CELL;
    let it = df_big_item(ci, org, 0.1, df_big_foot(ci));
    let i = (gid.y * n + gid.x) * 2u;
    scan_out[i] = vec4<f32>(f32(it.kind), org + it.base.xy, it.a.x);
    scan_out[i + 1u] = vec4<f32>(it.a.y, it.b.x, it.b.w, it.a.z);
}

// The small cells of the tile's first 1024 m, where the harness looks.
@compute @workgroup_size(16, 16)
fn scan_small(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = 256u;
    if (gid.x >= n || gid.y >= n) {
        return;
    }
    let ci = vec2<i32>(gid.xy);
    let org = vec2<f32>(ci) * DF_SMALL_CELL;
    let it = df_small_item(ci, org, 0.1, df_small_foot(ci));
    let i = (gid.y * n + gid.x) * 2u;
    scan_out[i] = vec4<f32>(f32(it.kind), org + it.base.xy, it.a.x);
    scan_out[i + 1u] = vec4<f32>(it.a.y, it.b.x, it.b.y, it.a.z);
}
"""


def extract(src: str, begin: str, end: str) -> str:
    if src.count(begin) != 1 or src.count(end) != 1:
        raise RuntimeError(f"{SHADER.name} must hold exactly one "
                           f"'{begin}' ... '{end}' block.")
    return src[src.index(begin):src.index(end) + len(end)]


def extract_fn(src: str, name: str) -> str:
    m = re.search(rf"^fn {name}\(.*?^}}\n", src, re.S | re.M)
    if m is None:
        raise RuntimeError(f"fn {name} not found in {SHADER.name}")
    return m.group(0)


def scan_plants(entry="scan"):
    """Items as the shader places them, desert frame. `scan`: every big
    cell, rows (kind, x, y, height, radius, arms, age, ribs). `scan_small`:
    the small cells of the first 1024 m, rows (kind, x, y, height,
    half-width, variant, flowering, stems)."""
    import wgpu
    src = SHADER.read_text()
    # The ground's habitat accessors are lifted from the shader like the
    # flora section itself, over the stub tile read above.
    code = (extract_fn(src, "pcg2d") + SCAN_PRELUDE
            + extract_fn(src, "desert_wash") + extract_fn(src, "desert_wash_near")
            + extract_fn(src, "desert_material")
            + extract(src, FLORA_BEGIN, FLORA_END) + SCAN_MAIN)
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=code)
    pipe = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": entry})
    n = 256
    size = n * n * 2 * 16
    buf = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE
                               | wgpu.BufferUsage.COPY_SRC)
    rb = device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST
                              | wgpu.BufferUsage.MAP_READ)
    from cloudyview.soar_host import DESERT_DIR, decode_surface_level
    meta = json.loads((DESERT_DIR / "meta.json").read_text())
    side = int(meta["n"])
    tex = device.create_texture(
        size=(side, side, 1), format=wgpu.TextureFormat.rg16float,
        mip_level_count=len(meta["files"]),
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    for lvl, name in enumerate(meta["files"]):
        sd = max(side >> lvl, 1)
        raw = gzip.decompress((DESERT_DIR / name).read_bytes())
        device.queue.write_texture(
            {"texture": tex, "mip_level": lvl, "origin": (0, 0, 0)},
            decode_surface_level(raw, sd, meta),
            {"offset": 0, "bytes_per_row": sd * 4, "rows_per_image": sd},
            (sd, sd, 1))
    samp = device.create_sampler(address_mode_u="repeat",
                                 address_mode_v="repeat",
                                 mag_filter="linear", min_filter="linear",
                                 mipmap_filter="linear")
    bg = device.create_bind_group(layout=pipe.get_bind_group_layout(0),
                                  entries=[{"binding": 0,
                                            "resource": {"buffer": buf}},
                                           {"binding": 1,
                                            "resource": tex.create_view()},
                                           {"binding": 2, "resource": samp}])
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(n // 16, n // 16)
    cp.end()
    enc.copy_buffer_to_buffer(buf, 0, rb, 0, size)
    device.queue.submit([enc.finish()])
    rb.map_sync(wgpu.MapMode.READ)
    raw = np.frombuffer(bytearray(rb.read_mapped()), np.float32)
    rb.unmap()
    return raw.reshape(-1, 8)


def pick_hero(plants, near=(512.0, 512.0)):
    """The tallest saguaro with three or more arms nearest `near`."""
    sag = plants[(plants[:, 0] == 8) & (plants[:, 5] >= 3)]
    if len(sag) == 0:
        raise RuntimeError("no saguaro with three arms anywhere in the tile")
    d = np.hypot(sag[:, 1] - near[0], sag[:, 2] - near[1])
    close = sag[d < np.percentile(d, 5)]
    return close[np.argmax(close[:, 3])]


def load_demo(demo, cache_dir: Path):
    base = DEMOS / demo
    meta = json.loads((base / "meta.json").read_text())
    shape = tuple(meta["volume"]["shape_xyz"])
    cache = cache_dir / f"{demo}.npy"
    if cache.exists():
        sigma = np.load(cache)
    else:
        raw = gzip.decompress((base / meta["volume"]["file"]).read_bytes())
        sigma = np.frombuffer(raw, np.float16).reshape(shape)
        np.save(cache, sigma)
    return meta, sigma


def pick_shrub(small, hero):
    """The biggest creosote (not bursage) within 60 m of the hero."""
    c = small[(small[:, 0] == 9) & (small[:, 5] < 0.5)]
    d = np.hypot(c[:, 1] - hero[1], c[:, 2] - hero[2])
    c = c[d < 60.0]
    return c[np.argmax(c[:, 3])]


def nearest(plants, kind, hero):
    c = plants[plants[:, 0] == kind]
    return c[np.argmin(np.hypot(c[:, 1] - hero[1], c[:, 2] - hero[2]))]


class Ground:
    """The terrain height under a desert-frame point: the tile's mip 0,
    bilinear with the GPU's texel-centre convention, over the datum. The
    sub-tile octaves (decimetres) are left out: this only places cameras."""

    def __init__(self, datum=0.0):
        from cloudyview.soar_host import DESERT_DIR, decode_surface_level
        meta = json.loads((DESERT_DIR / "meta.json").read_text())
        n = int(meta["n"])
        raw = gzip.decompress((DESERT_DIR / meta["files"][0]).read_bytes())
        texels = np.frombuffer(decode_surface_level(raw, n, meta), np.float16)
        self.h = texels.reshape(n, n, 2)[:, :, 0].astype(np.float64)
        self.cell = float(meta["cell_m"])
        self.n = n
        self.datum = datum

    def __call__(self, x, y):
        u = x / self.cell - 0.5
        v = y / self.cell - 0.5
        i0, j0 = math.floor(u), math.floor(v)
        fu, fv = u - i0, v - j0
        n = self.n
        h = self.h

        def at(i, j):
            return h[j % n, i % n]
        return self.datum + (
            (1 - fu) * (1 - fv) * at(i0, j0) + fu * (1 - fv) * at(i0 + 1, j0)
            + (1 - fu) * fv * at(i0, j0 + 1) + fu * fv * at(i0 + 1, j0 + 1))


def aim(g, x, y, dist, z, look_z, fov, from_az, s_az, s_el):
    """A camera `dist` from the plant at (x, y) on bearing `from_az`, `z`
    above the ground under it, looking at `look_z` above the plant's foot."""
    a = math.radians(from_az)
    cx, cy = x + dist * math.sin(a), y + dist * math.cos(a)
    zc = g(cx, cy) + z
    el = math.degrees(math.atan2(g(x, y) + look_z - zc, dist))
    return (cx, cy, zc, (from_az + 180.0) % 360.0, el, fov, s_az, s_el)


def shrub_views(g, s, sun_az, sun_el):
    x, y, h, w = s[1], s[2], s[3], s[4]
    side = sun_az + 25.0
    return {
        "creo_1m": aim(g, x, y, 1.0 + w, 1.2, 0.6 * h, 70.0, side, sun_az, sun_el),
        "creo_5m": aim(g, x, y, 5.0, 1.6, 0.5 * h, 45.0, side, sun_az, sun_el),
        "creo_30m": aim(g, x, y, 30.0, 3.0, 0.5 * h, 25.0, side, sun_az, sun_el),
        "creo_top": aim(g, x, y, 12.0, 9.0, 0.0, 50.0, 180.0, sun_az, sun_el),
    }


def other_views(g, plants, hero, sun_az, sun_el):
    oc = nearest(plants, 10, hero)
    rk = nearest(plants, 14, hero)
    print(f"ocotillo at ({oc[1]:.1f}, {oc[2]:.1f}): H {oc[3]:.2f} m, "
          f"{int(oc[4])} canes; boulders at ({rk[1]:.1f}, {rk[2]:.1f})")
    side = sun_az + 25.0
    return {
        "oco_1m": aim(g, oc[1], oc[2], 1.3, 1.6, 1.9, 60.0, side, sun_az, sun_el),
        "oco_8m": aim(g, oc[1], oc[2], 8.0, 1.7, 0.5 * oc[3], 60.0, side,
                      sun_az, sun_el),
        "rock_4m": aim(g, rk[1], rk[2], 5.0, 1.8, 0.6, 60.0, side, sun_az, sun_el),
    }


def hero_views(g, h, sun_az, sun_el):
    """Cameras framed on the hero: (x, y, z, az, el, fov, sun_az, sun_el),
    desert frame. The close views stand on the sun's side so the plant is
    front-lit; `backlit` stands opposite a low sun."""
    x, y, height, r = h[1], h[2], h[3], h[4]
    side = sun_az + 25.0
    back = sun_az + 180.0
    return {
        "1m": aim(g, x, y, r + 0.55, 1.7, 1.55, 55.0, side, sun_az, sun_el),
        "trunk4m": aim(g, x, y, 4.0, 2.0, 2.0, 20.0, side, sun_az, sun_el),
        "5m": aim(g, x, y, 5.0, 1.8, 0.45 * height, 80.0, side, sun_az, sun_el),
        "30m": aim(g, x, y, 30.0, 2.5, 0.5 * height, 45.0, side, sun_az, sun_el),
        "300m": aim(g, x, y, 300.0, 60.0, 0.4 * height, 30.0, side,
                    sun_az, sun_el),
        "backlit": aim(g, x, y, 9.0, 1.6, 0.55 * height, 70.0, back + 8.0,
                       sun_az, 7.0),
        "backlit_close": aim(g, x, y, r + 0.9, 2.2, 2.4, 50.0, back + 12.0,
                             sun_az, 9.0),
        # Level with the arms, the low sun just off the trunk's edge.
        "backlit_low": aim(g, x, y, 7.0, 3.0, 4.2, 55.0, back + 4.0,
                           sun_az, sun_el),
    }


def main(argv=None) -> int:
    from cloudyview.soar_host import (
        APP_LIGHT_MARCH_LOD_DEGREES, APP_VIEW_STEP_LOD_DEGREES, SceneState,
        SoarRenderer, ViewState, desert_relief_m, surface_cell_m)
    from cloudyview.witness import OCEAN_REFLECTANCE, QUALITY_PRESETS
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--demo", default="desert-coarse")
    ap.add_argument("--views", default="all")
    ap.add_argument("--size", type=int, nargs=2, default=(768, 432),
                    help="render size (default: the low tier's 768x432)")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--quality", default="low")
    ap.add_argument("--exposure", type=float, default=1.2)
    ap.add_argument("--haze", type=float, default=0.55)
    ap.add_argument("--lod", type=float, default=0.5)
    # The demo's own sun stands behind a cloud over the hero; this one is
    # in the clear there.
    ap.add_argument("--sun", type=float, nargs=2, default=(120.0, 50.0),
                    metavar=("AZ", "EL"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--time", action="store_true",
                    help="also report ms/frame (median of single passes)")
    ap.add_argument("--camera", type=float, nargs=6, default=None,
                    metavar=("X", "Y", "Z", "AZ", "EL", "FOV"),
                    help="a free view in the desert frame")
    ap.add_argument("--hero-offset", type=float, nargs=3, default=None,
                    metavar=("DX", "DY", "Z"),
                    help="--camera relative to the hero's foot; aims at it")
    ap.add_argument("--patch", action="append", default=[],
                    metavar="OLD=>NEW",
                    help="substitute in the shader source before compiling "
                         "(each OLD must occur exactly once): forcing one "
                         "LOD representation for a calibration, say")
    ap.add_argument("--no-clouds", action="store_true",
                    help="an empty volume: the sun in clear sky, for judging "
                         "the plants' own optics (the backlit halo) without "
                         "the demo's cloud standing in front of a low sun")
    ap.add_argument("--npy", action="store_true",
                    help="also save the linear float image")
    args = ap.parse_args(argv)

    if args.patch:
        import cloudyview.soar_host as host
        original = host.read_shader

        def patched() -> str:
            src = original()
            for p in args.patch:
                old, new = p.split("=>")
                if src.count(old) != 1:
                    raise RuntimeError(f"--patch: {old!r} occurs "
                                       f"{src.count(old)} times")
                src = src.replace(old, new)
            return src
        host.read_shader = patched

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plants = scan_plants()
    kinds, counts = np.unique(plants[:, 0], return_counts=True)
    print("tile census (big cells):",
          ", ".join(f"kind {int(k)}: {c}" for k, c in zip(kinds, counts)))
    hero = pick_hero(plants)
    print(f"hero saguaro at ({hero[1]:.2f}, {hero[2]:.2f}): "
          f"H {hero[3]:.2f} m, R {hero[4]:.3f} m, {int(hero[5])} arms, "
          f"{int(hero[7])} ribs, age {hero[6]:.2f}")

    meta, sigma = load_demo(args.demo, out)
    bmin, bmax = meta["volume"]["bmin"], meta["volume"]["bmax"]
    sun_az, sun_el = args.sun
    preset = QUALITY_PRESETS[args.quality]
    r = SoarRenderer(periodic=bool(meta.get("periodic", True)),
                     surface="desert")
    if args.no_clouds:
        sigma = np.zeros_like(sigma)
    r.upload_volume(sigma)
    tile = r.surface_meta
    dx = (bmax[0] - bmin[0]) / sigma.shape[0]
    state = SceneState(
        bmin=bmin, bmax=bmax,
        dt_view=dx * preset["step_factor"],
        dt_light=dx * preset["light_step_factor"],
        periodic=bool(meta.get("periodic", True)),
        ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=surface_cell_m("desert", tile),
        ocean_tile_extent=float(tile["tile_extent_m"]),
        ocean_max_lod=int(tile["mips"]) - 1,
        surface="desert",
        desert_relief_m=desert_relief_m(tile),
    )

    small = scan_plants("scan_small")
    shrub = pick_shrub(small, hero)
    print(f"creosote at ({shrub[1]:.2f}, {shrub[2]:.2f}): H {shrub[3]:.2f} m, "
          f"W {shrub[4]:.2f} m, {int(shrub[7])} stems")
    g = Ground()
    bar = nearest(small, 12, hero)
    pp = nearest(small, 11, hero)
    print(f"barrel at ({bar[1]:.2f}, {bar[2]:.2f}): H {bar[3]:.2f} m")
    side = sun_az + 25.0
    small_views = {
        "barrel_1m": aim(g, bar[1], bar[2], 0.9 + bar[4], 1.1, 0.6 * bar[3],
                         55.0, side, sun_az, sun_el),
        "barrel_3m": aim(g, bar[1], bar[2], 3.0, 1.4, 0.5 * bar[3], 45.0, side,
                         sun_az, sun_el),
        "pear_1m": aim(g, pp[1], pp[2], 1.4, 1.0, 0.4, 60.0, side, sun_az, sun_el),
        "pear_4m": aim(g, pp[1], pp[2], 4.0, 1.5, 0.4, 45.0, side, sun_az, sun_el),
    }
    views = {**small_views, **hero_views(g, hero, sun_az, sun_el),
             **shrub_views(g, shrub, sun_az, sun_el),
             **other_views(g, plants, hero, sun_az, sun_el)}
    if args.views != "all":
        views = {k: views[k] for k in args.views.split(",")}
    if args.camera is not None:
        c = args.camera
        views = {"camera": (*c, sun_az, sun_el)}
    if args.hero_offset is not None:
        dxo, dyo, z = args.hero_offset
        x, y = hero[1] + dxo, hero[2] + dyo
        az = math.degrees(math.atan2(-dxo, -dyo)) % 360.0
        z = g(x, y) + z
        el = math.degrees(math.atan2(g(hero[1], hero[2]) + 0.5 * hero[3] - z,
                                     math.hypot(dxo, dyo)))
        views = {"hero_offset": (x, y, z, az, el, 60.0, sun_az, sun_el)}

    baked = None
    w, h = args.size
    for name, (x, y, z, az, el, fov, s_az, s_el) in views.items():
        view = ViewState(
            camera_position=[x, y, z], azimuth=az, elevation=el, fov=fov,
            output_size=(w, h), render_size=(w, h),
            sun_azimuth=s_az, sun_elevation=s_el,
            exposure=args.exposure, haze=args.haze,
            light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES * args.lod,
            view_step_lod_degrees=APP_VIEW_STEP_LOD_DEGREES * args.lod,
            light_cache=bool(preset["light_cache"]),
            sky_probe=bool(preset["sky_probe"]),
        )
        if preset["light_cache"] and baked != (s_az, s_el):
            r.bake_light_cache(state, view, divisor=2)
            baked = (s_az, s_el)
        t0 = time.perf_counter()
        img = r.render(state, view, frames=args.frames)
        el_s = time.perf_counter() - t0
        ms = ""
        if args.time:
            times = []
            for _ in range(9):
                t1 = time.perf_counter()
                r.render(state, view, frames=1)
                times.append(time.perf_counter() - t1)
            ms = f"  {1000 * float(np.median(times)):7.2f} ms/frame"
        png = out / f"{name}{args.tag}.png"
        rgb = np.clip(np.asarray(img), 0.0, 1.0)
        Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(png)
        if args.npy:
            np.save(png.with_suffix(".npy"), np.asarray(img, np.float32))
        print(f"  {name:<14} {el_s:6.2f} s{ms}  -> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
