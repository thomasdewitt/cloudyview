"""The city's graffiti (city/graffiti.wgsl) on a flat test wall,
and the measurements its far mean stands on.

The section is lifted from the shader with the three hashes it calls and
run in compute kernels of its own, so nothing here needs a city, a field or
a renderer:

  wall    paint a test wall: one band per run, every kind and level, at a
          chosen footprint; with --fps, the same wall at several footprints
          side by side with the finest box-filtered to each, so the rungs
          can be judged against what they stand for.
  means   measure the painted areas the far mean needs (CITY_GRAFFITI_A_*,
          _RHO_*) from the shader's own shapes, and print them.
  rungs   average FINE, MID and FAR over many runs at a set of heights;
          they must agree (tests/city/test_city_graffiti.py holds them to it).
  panel   chosen bands of the wall, each beside a reference photograph
          (temp/references/graffiti), for judging the look.

"""

import argparse
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


class _CityShader:
    """The city's shader module text (soar/shader.py assemble)."""
    name = "the city shader module"

    def read_text(self):
        from cloudyview.soar import shader
        return shader.assemble("city")


SHADER = _CityShader()
BEGIN = "// === CITY GRAFFITI (desert-flora) ==="
END = "// === END CITY GRAFFITI ==="

def extract(src, begin, end):
    if src.count(begin) != 1 or src.count(end) != 1:
        raise RuntimeError(f"{SHADER.name} must hold exactly one "
                           f"'{begin}' ... '{end}' block.")
    return src[src.index(begin):src.index(end) + len(end)]


def extract_fn(src, name):
    m = re.search(rf"^fn {name}\(.*?^}}\n", src, re.S | re.M)
    if m is None:
        raise RuntimeError(f"fn {name} not found in {SHADER.name}")
    return m.group(0)


def graffiti_code(src=None):
    """The graffiti section and the hashes it calls, lifted from the shader
    (or from `src`, a patched copy)."""
    src = SHADER.read_text() if src is None else src
    return (extract_fn(src, "pcg2d") + extract_fn(src, "city_u01")
            + extract_fn(src, "city_rand4") + extract(src, BEGIN, END) + "\n")


def run_kernel(code, entry, n):
    """Run compute entry `entry` (workgroup 64) over n threads, each
    writing four vec4s to `out`; return them, shape (n, 4, 4)."""
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("no wgpu adapter")
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=code)
    pipe = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": entry})
    size = n * 4 * 16
    buf = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE
                               | wgpu.BufferUsage.COPY_SRC)
    rb = device.create_buffer(size=size, usage=wgpu.BufferUsage.COPY_DST
                              | wgpu.BufferUsage.MAP_READ)
    bg = device.create_bind_group(layout=pipe.get_bind_group_layout(0),
                                  entries=[{"binding": 0,
                                            "resource": {"buffer": buf}}])
    enc = device.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pipe)
    cp.set_bind_group(0, bg)
    groups = (n + 63) // 64
    gx = min(groups, 65535)
    cp.dispatch_workgroups(gx, (groups + gx - 1) // gx, 1)
    cp.end()
    enc.copy_buffer_to_buffer(buf, 0, rb, 0, size)
    device.queue.submit([enc.finish()])
    rb.map_sync(wgpu.MapMode.READ)
    out = np.frombuffer(rb.read_mapped(), np.float32).copy()
    rb.unmap()
    return out.reshape(n, 4, 4)


PRELUDE = """
@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
fn tid(gid: vec3<u32>) -> u32 { return gid.x + gid.y * 65535u * 64u; }
fn put(t: u32, k: u32, x: vec4<f32>) { out[4u * t + k] = x; }
fn u01(s: vec2<u32>) -> f32 { return city_u01(pcg2d(s).x); }
"""


# --- wall ---------------------------------------------------------------------

WALL = """
const W: u32 = {w}u;
const H: u32 = {h}u;
const FP: f32 = {fp};
const SS: u32 = {ss}u;          // supersamples a side
const BAND: f32 = 3.0;          // metres of wall a band shows
const RUNS: u32 = {runs}u;      // bands
const BASE: vec3<f32> = vec3<f32>(0.30, 0.29, 0.27);

@compute @workgroup_size(64)
fn wall(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= W * H) {{ return; }}
    let x = t % W;
    let y = t / W;
    // Each band is a run of its own: kinds in turn, seeds from the band.
    let rows_per = H / RUNS;
    let b = y / rows_per;
    var run: CityGraffitiRun;
    run.seed = vec2<u32>(b * 7919u + {salt}u, b * 104729u + 17u);
    run.len = f32(W) * FP;
    run.kind = i32(b) % 4;
    let v_top = BAND;
    var acc = vec3<f32>(0.0);
    let sub = FP / f32(SS);
    for (var i: u32 = 0u; i < SS; i = i + 1u) {{
        for (var j: u32 = 0u; j < SS; j = j + 1u) {{
            let u = (f32(x) + (f32(i) + 0.5) / f32(SS)) * FP;
            let v = v_top - (f32(y % rows_per) + (f32(j) + 0.5) / f32(SS)) * FP;
            let pt = city_graffiti_paint(run, u, v, vec2<f32>(sub, sub));
            acc = acc + BASE * pt.keep + pt.add;
        }}
    }}
    let lv = city_graffiti_level(run);
    put(t, 0u, vec4<f32>(acc / f32(SS * SS), lv.x));
}}
"""


def paint_wall(fp, runs=12, length=60.0, ss=1, salt=1, src=None):
    rows_per = int(round(3.0 / fp))
    w = int(round(length / fp))
    h = rows_per * runs
    code = (graffiti_code(src) + PRELUDE
            + WALL.format(w=w, h=h, fp=repr(float(fp)), ss=ss, runs=runs,
                          salt=salt))
    o = run_kernel(code, "wall", w * h)
    return o[:, 0, :3].reshape(h, w, 3), o[:, 0, 3].reshape(h, w)


def to_png(alb, path):
    from PIL import Image
    # Display: albedo 0.5 at 0.78 of white, a plain gamma.
    img = np.clip(1.56 * alb, 0.0, 1.0) ** (1.0 / 2.2)
    Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(path)


def box_down(img, k):
    h, w = img.shape[0] // k * k, img.shape[1] // k * k
    return img[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


# --- means --------------------------------------------------------------------

MEANS = """
const K: u32 = 64u;            // points per element

@compute @workgroup_size(64)
fn means(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= {n}u) {{ return; }}
    var run: CityGraffitiRun;
    run.seed = pcg2d(vec2<u32>(t, 0x5bd1e995u));
    run.len = 1.0e4;
    run.kind = 0;
    // Letters (layer 0: pieces among them), buff (2), tags (5).
    let l = city_graffiti_letters_at(run, 0, 0);
    let m = city_graffiti_mark_at(run, 2, 0);
    let g = city_graffiti_mark_at(run, 5, 0);
    var a = vec4<f32>(0.0);     // fill, shine, line, drip
    var sp = 0.0;
    var cb = 0.0;
    var ct = 0.0;
    for (var k: u32 = 0u; k < K; k = k + 1u) {{
        let r = city_rand4(pcg2d(vec2<u32>(t * K + k, 0x68e31da4u)));
        let q = vec2<f32>(r.x * l.len, r.y);
        let pt = city_graffiti_letter_parts(q, 1e-4, l);
        a = a + vec4<f32>(pt.fill, pt.shine, pt.line, pt.drip);
        sp = sp + pt.splash;
        cb = cb + city_graffiti_mark_cover(2, vec2<f32>(r.z * m.len, r.w),
                                           1e-4, m);
        let r2 = city_rand4(pcg2d(vec2<u32>(t * K + k, 0x2f0f5d1bu)));
        ct = ct + city_graffiti_mark_cover(5, vec2<f32>(r2.x * g.len, r2.y),
                                           1e-4, g);
    }}
    // Areas in units of h^2: the box is len x 1.
    put(t, 0u, a * l.len / f32(K));
    put(t, 1u, vec4<f32>(sp * l.len / f32(K), select(0.0, 1.0, l.piece),
                         f32(l.n), 0.0));
    put(t, 2u, vec4<f32>(cb * m.len / f32(K), m.len,
                         ct * g.len / f32(K), g.len));
}}
"""


def measure_means(n=1 << 20, src=None):
    """The painted-area constants, from n elements of each family:
    letters per glyph (E[area] / E[n]), buff and tags per unit of box
    (E[area] / E[L]). Returns {name: (value, standard error)}."""
    o = run_kernel(graffiti_code(src) + PRELUDE + MEANS.format(n=n),
                   "means", n)
    glyphs = o[:, 1, 2]
    piece = o[:, 1, 1] > 0.5

    def ratio(num, den):
        # A ratio of means, and its standard error (delta method).
        mn, md = num.mean(), den.mean()
        r = mn / md
        se = np.sqrt(np.var(num - r * den) / len(num)) / md
        return float(r), float(se)
    return {
        "CITY_GRAFFITI_A_FILL": ratio(o[:, 0, 0], glyphs),
        "CITY_GRAFFITI_A_SHINE": ratio(o[:, 0, 1], glyphs),
        "CITY_GRAFFITI_A_LINE": ratio(o[:, 0, 2], glyphs),
        "CITY_GRAFFITI_A_DRIP": ratio(o[:, 0, 3], glyphs),
        "CITY_GRAFFITI_A_SPLASH": ratio(o[piece, 1, 0], glyphs[piece]),
        "CITY_GRAFFITI_RHO_BUFF": ratio(o[:, 2, 0], o[:, 2, 1]),
        "CITY_GRAFFITI_RHO_TAG": ratio(o[:, 2, 2], o[:, 2, 3]),
    }


# --- palettes -------------------------------------------------------------------

PALETTES = """
@compute @workgroup_size(64)
fn palettes(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= {n}u) {{ return; }}
    let x = (f32(t) + 0.5) / f32({n}u);
    put(t, 0u, vec4<f32>(city_graffiti_hue(x), 0.0));
    put(t, 1u, vec4<f32>(city_graffiti_fill(x), 0.0));
    put(t, 2u, vec4<f32>(city_graffiti_line(x), 0.0));
    put(t, 3u, vec4<f32>(city_graffiti_ink(x), 0.0));
}}
"""


def palette_means(n=1 << 20, src=None):
    """The four palettes' means over their uniform draw, at n midpoints:
    exact but for the n draws that straddle a threshold (error <= 1/n)."""
    o = run_kernel(graffiti_code(src) + PRELUDE + PALETTES.format(n=n),
                   "palettes", n)
    return {name: o[:, k, :3].astype(np.float64).mean(0)
            for k, name in enumerate(["HUE", "FILL", "LINE", "INK"])}


# --- the letters' height profile ---------------------------------------------

PROFILE = """
const NB: u32 = {nb}u;

@compute @workgroup_size(64)
fn profile(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= {n}u * NB) {{ return; }}
    var run: CityGraffitiRun;
    run.seed = pcg2d(vec2<u32>(t / NB, 0x2545f491u));
    run.len = 1.0e4;
    run.kind = 0;
    let l = city_graffiti_letters_at(run, 0, 0);
    let y = (f32(t % NB) + u01(vec2<u32>(t, 0x1b873593u))) / f32(NB);
    var a = 0.0;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {{
        let x = l.len * (f32(k) + u01(vec2<u32>(t, k))) / 32.0;
        let pt = city_graffiti_letter_parts(vec2<f32>(x, y), 1e-4, l);
        a = a + pt.fill + pt.line + pt.drip + pt.splash;
    }}
    // Painted length at this height, in units of h, and the glyphs.
    put(t, 0u, vec4<f32>(a * l.len / 32.0, f32(l.n), y, 0.0));
}}
"""


def letter_profile(n=1 << 16, nb=40, src=None):
    """The letters' painted length per glyph at each normalized height y in
    their box, as a density over y (integral 1): (bin centres, density,
    (alpha, beta)), the uniform band [alpha, beta] with the density's mean
    and variance — the profile MID and FAR give them."""
    code = graffiti_code(src) + PRELUDE + PROFILE.format(n=n, nb=nb)
    o = run_kernel(code, "profile", n * nb).reshape(n, nb, 4, 4)
    dens = o[:, :, 0, 0].sum(0) / o[:, :, 0, 1].sum(0)
    yc = (np.arange(nb) + 0.5) / nb
    dens = dens / dens.mean()
    m = np.mean(dens * yc)
    sd = np.sqrt(np.mean(dens * (yc - m) ** 2) - 1.0 / (12 * nb * nb))
    half = np.sqrt(3.0) * sd
    return yc, dens, (float(m - half), float(m + half))


# --- the filter's bias -------------------------------------------------------------

BIAS = """
@compute @workgroup_size(64)
fn bias(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= {n}u) {{ return; }}
    var run: CityGraffitiRun;
    run.seed = pcg2d(vec2<u32>(t, 0x5bd1e995u));
    run.len = 1.0e4;
    run.kind = 0;
    let l = city_graffiti_letters_at(run, 0, 0);
    let m = city_graffiti_mark_at(run, 2, 0);
    let g = city_graffiti_mark_at(run, 5, 0);
    let ff = {f};
    var a = 0.0;
    var b = 0.0;
    var c = 0.0;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {{
        let r = city_rand4(pcg2d(vec2<u32>(t * 64u + k, 0x68e31da4u)));
        // Each box widened by the filter's reach.
        let y = r.y * (1.0 + 2.0 * ff) - ff;
        let pt = city_graffiti_letter_parts(
            vec2<f32>(r.x * (l.len + 2.0 * ff) - ff, y), ff, l);
        a = a + pt.fill + pt.line + pt.drip + pt.splash;
        b = b + city_graffiti_mark_cover(
            2, vec2<f32>(r.z * (m.len + 2.0 * ff) - ff, y), ff, m);
        c = c + city_graffiti_mark_cover(
            5, vec2<f32>(r.w * (g.len + 2.0 * ff) - ff, y), ff, g);
    }}
    let s = (1.0 + 2.0 * ff) / 64.0;
    put(t, 0u, vec4<f32>(a * (l.len + 2.0 * ff) * s, b * (m.len + 2.0 * ff) * s,
                         c * (g.len + 2.0 * ff) * s, 0.0));
}}
"""


def filter_bias(fs, n=1 << 18, src=None):
    """Painted area of letters, buff and tags with their FINE edges filtered
    over a footprint f (in units of the element's height), over the area at
    f = 0, for each f in fs: rows (letters, buff, tags)."""
    code = graffiti_code(src) + PRELUDE

    def area(f):
        o = run_kernel(code + BIAS.format(n=n, f=repr(float(f))), "bias", n)
        return o[:, 0, :3].astype(np.float64).mean(0)
    a0 = area(1e-4)
    return np.array([area(f) / a0 for f in fs])


# --- rungs --------------------------------------------------------------------

# A layer's MID footprint: past the hand-over for its tallest element
# (CITY_GRAFFITI_MID1 of its height) and short of FAR (a quarter of its
# slot).
MID_FP = {0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.2, 6: 0.2, 7: 0.2}

RUNGS = """
const NV: u32 = {nv}u;
const KU: u32 = {ku}u;          // points along the run per height
const LEN: f32 = {length};

// Height bin i, at a point drawn within it: a fixed grid would sample a
// tag row on its centre line every time, where a scrawl is thinnest.
fn height(t: u32) -> f32 {{
    return 2.6 * (f32(t % NV) + u01(vec2<u32>(t, 0x9e3779b9u))) / f32(NV);
}}

@compute @workgroup_size(64)
fn rungs(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= {n}u * NV) {{ return; }}
    let v = height(t);
    var run: CityGraffitiRun;
    run.seed = pcg2d(vec2<u32>(t / NV, 0x1b56c4e9u));
    run.len = LEN;
    run.kind = 1;
    let base = city_graffiti_level(run).y;
    var fine: CityGraffitiCover;
    fine.a = 0.0; fine.ac = vec3<f32>(0.0); fine.at = 0.0;
    var mid = fine;
    for (var k: u32 = 0u; k < KU; k = k + 1u) {{
        let u = LEN * (f32(k) + u01(vec2<u32>(t, k))) / f32(KU);
        let cf = city_graffiti_layer(run, {layer}, u, v, vec2<f32>(1e-3),
                                     base, 2);
        let cm = city_graffiti_layer(run, {layer}, u, v,
                                     vec2<f32>({mid_fp}, 1e-3), base, 2);
        fine.a = fine.a + cf.a; fine.ac = fine.ac + cf.ac;
        fine.at = fine.at + cf.at;
        mid.a = mid.a + cm.a; mid.ac = mid.ac + cm.ac; mid.at = mid.at + cm.at;
    }}
    let far = city_graffiti_layer(run, {layer}, 0.5 * LEN, v,
                                  vec2<f32>(20.0, 1e-3), base, 2);
    let s = 1.0 / f32(KU);
    put(t, 0u, vec4<f32>(fine.ac * s, fine.a * s));
    put(t, 1u, vec4<f32>(mid.ac * s, mid.a * s));
    put(t, 2u, vec4<f32>(far.ac, far.a));
    put(t, 3u, vec4<f32>(fine.at * s, mid.at * s, far.at, v));
}}
"""


def measure_layer(layer, n=2048, nv=26, ku=2048, length=200.0, src=None):
    """One layer of heavy street-wall runs, FINE, MID and FAR, averaged over
    n runs at nv heights through the band: (v, fine, mid, far, se_mid_far,
    se_band), rows (ac r, g, b, a, at); se_mid_far the standard error of
    the paired per-run MID - FAR at each height, se_band that of the
    paired per-run FINE - FAR over the band."""
    code = graffiti_code(src) + PRELUDE + RUNGS.format(
        nv=nv, ku=ku, length=repr(float(length)), n=n, layer=layer,
        mid_fp=repr(MID_FP[layer]))
    o = run_kernel(code, "rungs", n * nv).reshape(n, nv, 4, 4)
    rows = [np.concatenate([o[:, :, k, :], o[:, :, 3, k:k + 1]], axis=2)
            for k in range(3)]
    fine, mid, far = rows
    v = o[:, :, 3, 3].mean(0)
    se_mid_far = (mid - far).std(0) / np.sqrt(n)
    se_band = (fine - far).mean(1).std(0) / np.sqrt(n)
    return (v, fine.mean(0), mid.mean(0), far.mean(0), se_mid_far, se_band)


PAINT = """
const NV: u32 = {nv}u;
const KU: u32 = {ku}u;
const LEN: f32 = {length};

@compute @workgroup_size(64)
fn paint(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let t = tid(gid);
    if (t >= {n}u * NV) {{ return; }}
    let v = 2.6 * (f32(t % NV) + u01(vec2<u32>(t, 0x9e3779b9u))) / f32(NV);
    var run: CityGraffitiRun;
    run.seed = pcg2d(vec2<u32>(t / NV, 0x3c6ef372u));
    run.len = LEN;
    run.kind = {kind};
    var fine = vec4<f32>(0.0);
    for (var k: u32 = 0u; k < KU; k = k + 1u) {{
        let u = LEN * (f32(k) + u01(vec2<u32>(t, k))) / f32(KU);
        let p = city_graffiti_paint(run, u, v, vec2<f32>(1e-3));
        fine = fine + vec4<f32>(p.add, p.keep);
    }}
    let pf = city_graffiti_paint(run, 0.5 * LEN, v, vec2<f32>(20.0, 1e-3));
    put(t, 0u, fine / f32(KU));
    put(t, 1u, vec4<f32>(pf.add, pf.keep));
    put(t, 2u, vec4<f32>(city_graffiti_level(run).x, v, 0.0, 0.0));
}}
"""


def measure_paint(kind=1, n=4096, nv=26, ku=1024, length=200.0, src=None):
    """The whole paint, FINE and FAR, of n runs of `kind` at nv heights:
    per level, (v, fine, far, se of fine), rows (add r, g, b, keep)."""
    code = graffiti_code(src) + PRELUDE + PAINT.format(
        nv=nv, ku=ku, length=repr(float(length)), n=n, kind=kind)
    o = run_kernel(code, "paint", n * nv).reshape(n, nv, 4, 4)
    level = o[:, 0, 2, 0]
    v = o[:, :, 2, 1].mean(0)
    res = {}
    for lv in (1, 2):
        s = o[level == lv]
        res[lv] = (v, s[:, :, 0].mean(0), s[:, :, 1].mean(0),
                   s[:, :, 0].std(0) / np.sqrt(max(len(s), 1)))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wall")
    w.add_argument("--fp", type=float, default=0.01)
    w.add_argument("--fps", type=float, nargs="*", default=None,
                   help="footprints for the rung comparison (multiples of "
                        "--fp): each rendered, and the --fp image box-"
                        "filtered to it")
    w.add_argument("--length", type=float, default=60.0)
    w.add_argument("--runs", type=int, default=12)
    w.add_argument("--salt", type=int, default=1)
    w.add_argument("--out", type=Path, required=True)
    sub.add_parser("means")
    sub.add_parser("palettes")
    sub.add_parser("bias")
    sub.add_parser("profile")
    pn = sub.add_parser("panel")
    pn.add_argument("--bands", type=int, nargs="+", required=True,
                    help="band indices of the wall (its runs, kind = index "
                         "mod 4)")
    pn.add_argument("--refs", nargs="+", required=True,
                    help="one reference image per band")
    pn.add_argument("--fp", type=float, default=0.01)
    pn.add_argument("--length", type=float, default=24.0)
    pn.add_argument("--out", type=Path, required=True)
    sub.add_parser("rungs")
    args = ap.parse_args()
    if args.cmd == "wall":
        args.out.mkdir(parents=True, exist_ok=True)
        alb, lv = paint_wall(args.fp, args.runs, args.length, salt=args.salt)
        to_png(alb, args.out / f"wall_{args.fp:g}.png")
        print("levels by band:", [int(lv[i * lv.shape[0] // args.runs, 0])
                                  for i in range(args.runs)])
        for f in args.fps or []:
            k = int(round(f / args.fp))
            ref = box_down(alb, k)
            got, _ = paint_wall(args.fp * k, args.runs, args.length,
                                salt=args.salt)
            got = got[:ref.shape[0], :ref.shape[1]]
            to_png(np.concatenate([ref, np.full((ref.shape[0], 2, 3), 1.0),
                                   got], axis=1),
                   args.out / f"rung_{args.fp * k:g}.png")
            print(f"fp {args.fp * k:g}: mean albedo box-filtered "
                  f"{ref.mean(axis=(0, 1)).round(4)} rendered "
                  f"{got.mean(axis=(0, 1)).round(4)}; mean |diff| "
                  f"{np.abs(ref - got).mean():.4f}")
    elif args.cmd == "panel":
        from PIL import Image
        runs = max(args.bands) + 1
        alb, lv = paint_wall(args.fp, runs, args.length)
        rows_per = alb.shape[0] // runs
        img = (np.clip(1.56 * alb, 0.0, 1.0) ** (1.0 / 2.2) * 255
               + 0.5).astype(np.uint8)
        tiles = []
        for b, ref in zip(args.bands, args.refs):
            band = Image.fromarray(img[b * rows_per:(b + 1) * rows_per])
            r = Image.open(ref).convert("RGB")
            r = r.resize((int(r.width * rows_per / r.height), rows_per))
            row = Image.new("RGB", (r.width + 8 + band.width, rows_per),
                            "white")
            row.paste(r, (0, 0))
            row.paste(band, (r.width + 8, 0))
            tiles.append(row)
            print(f"band {b}: kind {b % 4}, level {int(lv[b * rows_per, 0])}")
        w = max(t.width for t in tiles)
        out = Image.new("RGB", (w, sum(t.height + 8 for t in tiles)), "white")
        y = 0
        for t in tiles:
            out.paste(t, (0, y))
            y += t.height + 8
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.save(args.out)
    elif args.cmd == "profile":
        yc, dens, (a, b) = letter_profile()
        print(f"const CITY_GRAFFITI_LETTER_Y0: f32 = {a:.4f};")
        print(f"const CITY_GRAFFITI_LETTER_Y1: f32 = {b:.4f};")
        band = np.where((yc > a) & (yc < b), 1.0 / (b - a), 0.0)
        for y, d, f_ in zip(yc[::4], dens[::4], band[::4]):
            print(f"  y {y:.3f}  density {d:.3f}  band {f_:.3f}")
    elif args.cmd == "bias":
        fs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for f, r in zip(fs, filter_bias(fs)):
            print(f"f {f:4.2f} h: area / area(0): letters {r[0]:.4f} "
                  f"buff {r[1]:.4f} tags {r[2]:.4f}")
    elif args.cmd == "palettes":
        for k, m in palette_means().items():
            print(f"CITY_GRAFFITI_{k}_MEAN = {m.round(7)}")
    elif args.cmd == "means":
        for k, (val, se) in measure_means().items():
            print(f"const {k}: f32 = {val:.4f};   // +- {se:.4f}")
    else:
        for layer in range(8):
            v, fine, mid, far, se, se_band = measure_layer(layer)
            band = lambda x: x.mean(0)
            print(f"layer {layer}: band means (ac r g b, a, at) fine "
                  f"{band(fine).round(5)} mid {band(mid).round(5)} far "
                  f"{band(far).round(5)}; worst |mid - far| over heights "
                  f"{np.abs(mid - far).max():.5f}")
        for kind in (1, 0):
            for lv, (v, fine, far, se) in measure_paint(kind).items():
                print(f"kind {kind} level {lv}: band mean (add rgb, keep) "
                      f"fine {fine.mean(0).round(5)} far "
                      f"{far.mean(0).round(5)} (se {se.mean(0).round(5)})")


if __name__ == "__main__":
    main()
