"""Every saguaro and palo verde lies inside the bounds the walks trust.

A walk tests a plant only inside its item box (df_cell_candidate clips
both the march and the hit to it), finds a plant's cell by its foot's reach
(DF_SAG_REACH for a saguaro, the foot's own for a palo verde), and assumes
the plant stays in its cell; a palo verde's near form looks inside a lobe
only where the ray enters its grown ellipsoid (df_pv_lobe_bound), and at a
clump's sprigs only where it enters their sphere (DF_PV_SPRIG_BOUND). What
lies outside a bound is not drawn: an arm cut off at a box face, leaves
sliced along a smooth ellipse — the look the palo verde was rebuilt to lose.

The tile is periodic (every cell index wraps into it), so the plants of
its cells are every plant there is. This runs the shader's own functions
(the harness's placement-scan module) over all of them, each plant built
whatever its habitat, and measures how far any of it lies outside each
bound: the saguaro's columns (trunk and arms, sampled along their axes,
radius and spines included), and the palo verde's — a tree's and a bank
sapling's — every leaf against its grown lobe (a proof: see pv_check), each
clump's sprig sphere against the box, reach and cell, and every sprig
against its clump's sphere over a corner of the tile. It also holds the arms' tilt to leaning out, never back toward the
trunk (loose-ends' review, 2026-09-25). The negative controls run the same
over the shader with the old bounds, which must fail.

Skips without a wgpu adapter.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"

KERNELS = """
// Saguaros: every big cell whose foot is one. Row 0: (box excess, its
// floor aside; walk reach excess; cell excess; the least an arm's tilt
// moves its tip out against the plumb arm's); row 1: (1, arms, reach
// excess at every arm's greatest tilt, arms whose tip ends within the
// trunk's and its own radius of the trunk's axis).
@compute @workgroup_size(16, 16)
fn bounds_sag(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The scan's bindings are laid out from what an entry reads: read the
    // tile once, to nothing.
    if (desert_tile(vec2<f32>(0.0), 0.0).x > 1e30) {
        return;
    }
    let n = u32(DF_BIG_N);
    if (gid.x >= n || gid.y >= n) {
        return;
    }
    let i = (gid.y * n + gid.x) * 2u;
    scan_out[i] = vec4<f32>(-1e9, -1e9, -1e9, 1e9);
    scan_out[i + 1u] = vec4<f32>(0.0, 0.0, -1e9, 0.0);
    let ci = vec2<i32>(gid.xy);
    let foot = df_big_foot(ci);
    if (foot.kind != DF_KIND_SAGUARO) {
        return;
    }
    // As df_big_item builds it, the habitat aside.
    var it: DfItem;
    it.seed = df_cell_seed(ci, DF_BIG_N, 0x5a17u);
    it.base = vec3<f32>(foot.xy, 0.0);
    let r0 = df_rand4(it.seed);
    let rp = df_rand4(pcg2d(it.seed ^ vec2<u32>(0xb5297a4du, 0x68e31da4u)));
    it = df_saguaro_item(it, vec4<f32>(0.0, rp.z, rp.w, r0.y));
    var box_x = -1e9;
    var reach_x = -1e9;
    var reach_tilt_x = -1e9;
    var cell_x = -1e9;
    var gain = 1e9;
    var close = 0.0;
    let sh = df_sag_shape(it);
    let nc = df_cactus_ncols(it);
    for (var c: i32 = 0; c < nc; c = c + 1) {
        let col = df_cactus_col(it, c);
        var steep = col;
        if (c > 0) {
            steep.omega = 0.5 * DF_PI - DF_SAG_ARM_TILT;
        }
        let len = df_col_len(col);
        for (var k: i32 = 0; k <= 96; k = k + 1) {
            let s = len * f32(k) / 96.0;
            let rr = df_col_radius(col, s) + DF_SAG_SHELL;
            let pp = df_jpath_at(col, s);
            let ax = col.root + vec3<f32>(col.u * pp.p.x, pp.p.y);
            let p = it.base + ax;
            let lo = it.bmin - (p - vec3<f32>(rr));
            let hi = (p + vec3<f32>(rr)) - it.bmax;
            // The floor is below ground, where no ray goes: the buried
            // root's spines may pass it (checked apart, in Python).
            box_x = max(box_x, max(max(lo.x, lo.y),
                                   max(max(hi.x, hi.y), hi.z)));
            reach_x = max(reach_x, length(ax.xy) + rr - DF_SAG_REACH);
            cell_x = max(cell_x, max(max(rr - p.x, p.x + rr - DF_BIG_CELL),
                                     max(rr - p.y, p.y + rr - DF_BIG_CELL)));
            let ps = df_jpath_at(steep, df_col_len(steep) * f32(k) / 96.0);
            let axs = steep.root + vec3<f32>(steep.u * ps.p.x, ps.p.y);
            reach_tilt_x = max(reach_tilt_x,
                               length(axs.xy) + rr - DF_SAG_REACH);
        }
        if (c > 0) {
            var plumb = col;
            plumb.omega = 0.5 * DF_PI;
            gain = min(gain, df_jpath_at(col, len).p.x
                             - df_jpath_at(plumb, len).p.x);
            let tp = df_jpath_at(col, len).p;
            let tip = col.root + vec3<f32>(col.u * tp.x, tp.y);
            let tr = df_sag_axis_at(it, sh, tip.z + 0.4);
            let trunk = df_sag_trunk(it);
            let d = length(tip.xy - tr.xy)
                - df_col_radius(trunk, tip.z + 0.4) - df_col_radius(col, len);
            close = close + select(0.0, 1.0, d < 0.0);
        }
    }
    // The floor's depth below the foot rides in the arms' slot's sign-free
    // place: row 1's x holds 1 + that depth (> 1.3 means 0.3 m or more).
    scan_out[i] = vec4<f32>(box_x, reach_x, cell_x, gain);
    scan_out[i + 1u] = vec4<f32>(1.0 + (it.base.z - it.bmin.z), f32(nc - 1),
                                 reach_tilt_x, close);
}

// A palo verde against its bounds: (every sprig against its grown lobe,
// each clump's sprig sphere against the box, the foot's reach, the cell),
// each
// the most any lies outside; `sprigs` also every sprig against its clump's
// sphere.
struct PvX {
    lobe: f32,
    box_x: f32,
    reach: f32,
    cell: f32,
    sprig: f32,
}

fn pv_check(it: DfItem, reach: f32, cell: f32, sprigs: bool) -> PvX {
    var x: PvX;
    x.lobe = -1e9;
    x.box_x = -1e9;
    x.reach = -1e9;
    x.cell = -1e9;
    x.sprig = -1e9;
    let nl = df_pv_nlimbs(it);
    for (var n: i32 = 0; n < nl; n = n + 1) {
        let sg = df_pv_limb_n(it, n);
        for (var m: i32 = 0; m < DF_PV_LOBES_PER_LIMB; m = m + 1) {
            let e = df_pv_lobe(it, sg, n, m);
            let big = df_pv_lobe_bound(e);
            let lseed = it.seed
                + vec2<u32>(u32(n * DF_PV_LOBES_PER_LIMB + m) * 7919u, 17u);
            for (var j: i32 = 0; j < DF_PV_CLUMPS; j = j + 1) {
                let cl = df_pv_clump(e, lseed, j);
                let rb = DF_PV_SPRIG_BOUND * cl.w;
                let p = it.base + cl.xyz;
                let lo = it.bmin - (p - vec3<f32>(rb));
                let hi = (p + vec3<f32>(rb)) - it.bmax;
                x.box_x = max(x.box_x, max(max(max(lo.x, lo.y), lo.z),
                                           max(max(hi.x, hi.y), hi.z)));
                x.reach = max(x.reach, length(cl.xy) + rb - reach);
                x.cell = max(x.cell, max(max(rb - p.x, p.x + rb - cell),
                                         max(rb - p.y, p.y + rb - cell)));
                // Every sprig against the grown lobe: a sprig lies within hl
                // of its centre, and a ball of radius hl moves the lobe's
                // norm by at most hl over its smallest axis, so a centre
                // within the lobe scaled by 1 - hl / min(axis) keeps the
                // whole leaf inside. And, over `sprigs` cells, each against
                // its clump's sphere.
                let small_axis = min(min(big.x, big.y), big.z);
                var sd = lseed + vec2<u32>(u32(j) * 104729u, 0u);
                for (var k: i32 = 0; k < DF_PV_SPRIGS; k = k + 1) {
                    let sp = df_pv_sprig(cl.xyz, cl.w, sd);
                    sd = sp.seed;
                    x.lobe = max(x.lobe, length((sp.c - e.c) / big)
                                         - (1.0 - sp.hl / small_axis));
                    if (sprigs) {
                        x.sprig = max(x.sprig, length(sp.c - cl.xyz) + sp.hl
                                               - rb);
                    }
                }
            }
        }
    }
    return x;
}

// The big cells' palo verdes, as df_big_item builds them.
@compute @workgroup_size(16, 16)
fn bounds_pv(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The scan's bindings are laid out from what an entry reads: read the
    // tile once, to nothing.
    if (desert_tile(vec2<f32>(0.0), 0.0).x > 1e30) {
        return;
    }
    let n = u32(DF_BIG_N);
    if (gid.x >= n || gid.y >= n) {
        return;
    }
    let i = (gid.y * n + gid.x) * 2u;
    scan_out[i] = vec4<f32>(-1e9);
    scan_out[i + 1u] = vec4<f32>(0.0, -1e9, 0.0, 0.0);
    let ci = vec2<i32>(gid.xy);
    let foot = df_big_foot(ci);
    if (foot.kind != DF_KIND_PALOVERDE) {
        return;
    }
    var it: DfItem;
    it.seed = df_cell_seed(ci, DF_BIG_N, 0x5a17u);
    it.base = vec3<f32>(foot.xy, 0.0);
    let rs = df_rand4(pcg2d(it.seed ^ vec2<u32>(0x27d4eb2fu, 0x165667b1u)));
    it = df_paloverde_item(it, rs);
    it.base = vec3<f32>(foot.xy, 0.0);
    it.bmin = vec3<f32>(foot.xy - vec2<f32>(foot.reach), it.base.z - 0.45);
    it.bmax = vec3<f32>(foot.xy + vec2<f32>(foot.reach),
                        it.base.z + 1.02 * it.a.x);
    let x = pv_check(it, foot.reach, DF_BIG_CELL, all(gid.xy < vec2<u32>(48u)));
    scan_out[i] = vec4<f32>(x.lobe, x.box_x, x.reach, x.cell);
    scan_out[i + 1u] = vec4<f32>(1.0, x.sprig, 0.0, 0.0);
}

// Every small cell's slots whose draw can be a bank's sapling (a creosote's,
// not a bursage's), as df_small_item builds it.
@compute @workgroup_size(16, 16)
fn bounds_pv_small(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The scan's bindings are laid out from what an entry reads: read the
    // tile once, to nothing.
    if (desert_tile(vec2<f32>(0.0), 0.0).x > 1e30) {
        return;
    }
    let n = u32(DF_SMALL_N);
    if (gid.x >= n || gid.y >= n) {
        return;
    }
    let i = (gid.y * n + gid.x) * 2u;
    var acc = vec4<f32>(-1e9);
    var sprig = -1e9;
    var count = 0.0;
    let ci = vec2<i32>(gid.xy);
    for (var s: i32 = 0; s < DF_SMALL_SLOTS; s = s + 1) {
        let foot = df_small_foot(ci, s);
        var it: DfItem;
        it.seed = df_small_seed(ci, s);
        let dr = df_small_draws(it.seed);
        let r0 = df_small_r0(dr);
        if (foot.kind != DF_KIND_CREOSOTE || r0.x >= DF_P_CREOSOTE) {
            continue;
        }
        it = df_paloverde_bank_item(it, foot.reach, df_small_rs(dr, it.seed));
        it.base = vec3<f32>(foot.xy, 0.0);
        it.bmin = vec3<f32>(foot.xy - vec2<f32>(foot.reach), -0.65);
        it.bmax = vec3<f32>(foot.xy + vec2<f32>(foot.reach), 1.02 * it.a.x);
        let x = pv_check(it, foot.reach, DF_SMALL_CELL,
                         all(gid.xy < vec2<u32>(24u)));
        acc = max(acc, vec4<f32>(x.lobe, x.box_x, x.reach, x.cell));
        sprig = max(sprig, x.sprig);
        count = count + 1.0;
    }
    scan_out[i] = acc;
    scan_out[i + 1u] = vec4<f32>(count, sprig, 0.0, 0.0);
}
"""


def _harness():
    spec = importlib.util.spec_from_file_location(
        "desert_flora_harness", REPO / "tools" / "desert_flora_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run():
    wgpu = pytest.importorskip("wgpu")
    if wgpu.gpu.request_adapter_sync(power_preference="high-performance") is None:
        pytest.skip("no wgpu adapter")
    h = _harness()
    src = SHADER.read_text()

    def go(entry, n, patches=()):
        s = src
        for old, new in patches:
            assert s.count(old) == 1, old
            s = s.replace(old, new)
        return h.run_scan(h.scan_code(s) + KERNELS, entry, n)
    return go


def _sag(rows):
    a = rows.reshape(-1, 2, 4)
    return a[a[:, 1, 0] > 0]


def test_saguaros_lie_inside_their_bounds(run):
    s = _sag(run("bounds_sag", 256))
    assert len(s) > 5000
    box, reach, cell, gain = s[:, 0, 0], s[:, 0, 1], s[:, 0, 2], s[:, 0, 3]
    arms, reach_tilt, close = s[:, 1, 1], s[:, 1, 2], s[:, 1, 3]
    assert box.max() <= 1e-3, box.max()
    # The floor lies well below the ground, which stops every ray first.
    assert (s[:, 1, 0] - 1.0).min() >= 0.3, (s[:, 1, 0] - 1.0).min()
    assert reach.max() <= 1e-3, reach.max()
    # DF_SAG_REACH bounds any arm at its greatest tilt, not only the drawn.
    assert reach_tilt.max() <= 1e-3, reach_tilt.max()
    assert cell.max() <= 1e-3, cell.max()
    # An arm's tilt never brings its tip in toward the trunk.
    armed = arms > 0
    assert gain[armed].min() >= -1e-4, gain[armed].min()
    # And so arms whose tip ends against the trunk stay as rare as plumb
    # ones were (0.5%, loose-ends; 13% when the tilt could lean in).
    assert close.sum() / arms.sum() < 0.01, close.sum() / arms.sum()


def test_saguaro_negative_control(run):
    # The reach without the arms' lean: tall tilted arms leave the box.
    s = _sag(run("bounds_sag", 256, [(
        "    lo = lo + arm_lo;\n    hi = hi + arm_hi;\n", "")]))
    assert s[:, 0, 0].max() > 0.1, s[:, 0, 0].max()
    # A reach short of the tile's widest saguaro (2.20 m).
    s = _sag(run("bounds_sag", 256, [("const DF_SAG_REACH: f32 = 2.4;",
                                      "const DF_SAG_REACH: f32 = 2.0;")]))
    assert s[:, 1, 2].max() > 0.1, s[:, 1, 2].max()


def _pv(rows):
    a = rows.reshape(-1, 2, 4)
    return a[a[:, 1, 0] > 0]


@pytest.mark.parametrize("entry,n", [("bounds_pv", 256),
                                     ("bounds_pv_small", 512)])
def test_palo_verdes_lie_inside_their_bounds(run, entry, n):
    p = _pv(run(entry, n))
    assert len(p) > 1000
    lobe, box, reach, cell = (p[:, 0, k] for k in range(4))
    # A proof, leaf by leaf (see pv_check).
    assert lobe.max() <= 0.0, lobe.max()
    assert box.max() <= 1e-3, box.max()
    assert reach.max() <= 1e-3, reach.max()
    assert cell.max() <= 1e-3, cell.max()
    sprig = p[:, 1, 1]
    checked = sprig > -1e8
    assert checked.sum() > 20
    assert sprig[checked].max() <= 1e-4, sprig[checked].max()


def test_palo_verde_negative_control(run):
    # The clump's sphere at rc, and the lobe's entry at the lobe itself.
    p = _pv(run("bounds_pv", 256, [
        ("const DF_PV_SPRIG_BOUND: f32 = 1.35;",
         "const DF_PV_SPRIG_BOUND: f32 = 1.0;")]))
    sprig = p[:, 1, 1]
    assert sprig[sprig > -1e8].max() > 0.01
    p = _pv(run("bounds_pv", 256, [(
        "    return e.r + vec3<f32>(0.45 * min(e.r.x, e.r.y));",
        "    return e.r;")]))
    assert p[:, 0, 0].max() > 0.05, p[:, 0, 0].max()
