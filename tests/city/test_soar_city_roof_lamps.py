"""
The roof lamps' far mean is what the props draw, in expectation.

Every roof lamp (a tank's service lamp, a bulkhead's door lamp, a helipad's
corner lights) is drawn by rooftopworks' props, which stop at
CITY_PROP_RANGE. Past it the roof hook adds the lamps' light as a uniform
emission on the deck (cc_rooftopworks_lamps_far), built from constants that
stand for what the prop shader draws: a lamp's spots (source and spill)
integrated over the faces of the housing they land on, per face
orientation, and the tank lamps' count per square metre of deck. Those
constants are measured here, from the shader's own placement and its own
spot function, by probe variants of the plain pass that walk the city's
cells, and the shipped constants must match them:

1. MECH_RHO_AT: over the urban roofs with clutter, the tank lamps per
   square metre of summit deck (in the cell's own column, less the tier
   standing on it), at each clutter density the tiers use (MECH_RHO_D, the
   hosts' per-tier roof densities, plus 1 as tuned).
2. The table FAR_i_*, at each of its handoff footprints (FAR_i_MECH.w): per
   tank lamp the spots' integral over its housing's four sides together and
   over its top (MECH .xy); per lit bulkhead over its door face (less what
   its dark door leaf takes off), its back, its two sides together and its
   top (BULK), and with a lift room the room's faces joining the same four
   parts, less where the two boxes meet (LIFT); and the pad lights' share
   that lands on the pad (MECH .z, closed form, checked here in Python).
   Beside each, the same integrals of the light's square (the rows ending
   2): the second moment, from which the far mean shows the light as small
   bright faces, as it is up close, rather than spread over the deck.

The integrals are midpoint sums at the footprint where the props hand over,
because the far mean must continue what is drawn just inside the range, and
the housings are integrated as the far trace draws them there (boxes, past
SDF_FP). The spots widen with the footprint, area-preserving on an
unbounded plane but not on a housing's finite faces, so the constants move
with it (the stair's door face loses three quarters of its light between 1
and 12 m): hence a table, interpolated in the shader on the live LOD angle
times CITY_PROP_RANGE.

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/city/test_soar_city_roof_lamps.py -v -s
"""

import gc
import math
import re

import numpy as np
import pytest
from tests.conftest import ShaderModule

pytest.importorskip("wgpu")

from tests.conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

from tests.conftest import REPO as ROOT

SHADER = ShaderModule("city")
COMPONENT = ROOT / "cloudyview/soar/shader/city/components/rooftopworks.wgsl"
ANCHOR = "    let city_o = city_camera_origin();\n    let chit = city_trace("

# Module-scope helpers for the probes: one face's integral of a lamp's two
# spots, as the prop shader sums them (source + k x spill) at the footprint
# PROBE_FP, by a midpoint sum over the face rectangle o + s e1 + t e2; with
# PROBE_SQ, the integral of its square (the second moment, which sets how
# concentrated the light is).
HELPERS = """
const PROBE_FP: f32 = FP_VALUE;
const PROBE_SQ: bool = SQ_VALUE;
fn probe_lamp_rect(seat: vec3<f32>, o: vec3<f32>, e1: vec3<f32>,
                   e2: vec3<f32>, k: f32, sig_spill: f32) -> f32 {
    let n = 40;
    var sum = 0.0;
    for (var j: i32 = 0; j < n; j = j + 1) {
        for (var i: i32 = 0; i < n; i = i + 1) {
            let p = o + e1 * ((f32(i) + 0.5) / f32(n))
                      + e2 * ((f32(j) + 0.5) / f32(n));
            let v = cc_rooftopworks_spot(p, seat, cc_rooftopworks_LAMP_SIG,
                                         PROBE_FP)
                    + k * cc_rooftopworks_spot(p, seat, sig_spill, PROBE_FP);
            sum = sum + select(v, v * v, PROBE_SQ);
        }
    }
    return sum * length(e1) * length(e2) / f32(n * n);
}

// A box's face: axis 0 (x) or 1 (y) at side s (-1 or +1), or axis 2 its top.
fn probe_lamp_face(seat: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, axis: i32,
                   s: f32, k: f32, sig_spill: f32) -> f32 {
    let sz = hi - lo;
    if (axis == 0) {
        let x = select(lo.x, hi.x, s > 0.0);
        return probe_lamp_rect(seat, vec3<f32>(x, lo.y, lo.z),
                               vec3<f32>(0.0, sz.y, 0.0),
                               vec3<f32>(0.0, 0.0, sz.z), k, sig_spill);
    }
    if (axis == 1) {
        let y = select(lo.y, hi.y, s > 0.0);
        return probe_lamp_rect(seat, vec3<f32>(lo.x, y, lo.z),
                               vec3<f32>(sz.x, 0.0, 0.0),
                               vec3<f32>(0.0, 0.0, sz.z), k, sig_spill);
    }
    return probe_lamp_rect(seat, vec3<f32>(lo.x, lo.y, hi.z),
                           vec3<f32>(sz.x, 0.0, 0.0),
                           vec3<f32>(0.0, sz.y, 0.0), k, sig_spill);
}

fn probe_lamp_sides(seat: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>, k: f32,
                    sig_spill: f32) -> f32 {
    return probe_lamp_face(seat, lo, hi, 0, -1.0, k, sig_spill)
         + probe_lamp_face(seat, lo, hi, 0, 1.0, k, sig_spill)
         + probe_lamp_face(seat, lo, hi, 1, -1.0, k, sig_spill)
         + probe_lamp_face(seat, lo, hi, 1, 1.0, k, sig_spill);
}

// A bulkhead-like box's faces split by the door: (front, back, sides, top).
// `skip` is a horizontal normal whose face is left out (where a lift room
// meets the bulkhead), zero for none.
fn probe_lamp_parts(seat: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>,
                    door: vec2<f32>, skip: vec2<f32>) -> vec4<f32> {
    let k = 0.12;
    let sig = 1.6;
    var res = vec4<f32>(0.0);
    let normals = array<vec2<f32>, 4>(vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0),
                                      vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0));
    for (var q: i32 = 0; q < 4; q = q + 1) {
        let n = normals[q];
        if (dot(n, skip) > 0.5) {
            continue;
        }
        let axis = select(1, 0, abs(n.x) > 0.5);
        let a = probe_lamp_face(seat, lo, hi, axis, n.x + n.y, k, sig);
        let f = dot(n, door);
        if (f > 0.5) {
            res.x = res.x + a;
        } else if (f < -0.5) {
            res.y = res.y + a;
        } else {
            res.z = res.z + a;
        }
    }
    res.w = probe_lamp_face(seat, lo, hi, 2, 1.0, k, sig);
    return res;
}

// The bulkhead's door leaf, which returns BULK_DOOR_K of the light on it:
// what it takes off the door face's integral.
fn probe_lamp_door(seat: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>,
                   door: vec2<f32>) -> f32 {
    let t = vec3<f32>(-door.y, door.x, 0.0);
    let c = 0.5 * (lo + hi);
    let half = dot(abs(door), 0.5 * (hi - lo).xy);
    let o = vec3<f32>(c.xy + door * half, lo.z)
            - t * (0.5 * cc_rooftopworks_BULK_DOOR_W);
    let kd = cc_rooftopworks_BULK_DOOR_K;
    return select(1.0 - kd, 1.0 - kd * kd, PROBE_SQ)
           * probe_lamp_rect(seat, o, t * cc_rooftopworks_BULK_DOOR_W,
                             vec3<f32>(0.0, 0.0, cc_rooftopworks_BULK_DOOR_H),
                             0.12, 1.6);
}
"""

# The cell this pixel walks: the frame's coordinates at 4096 steps a side,
# about the camera's cell (distinct cells, the city's tile folding the
# rest), and the roof hook's own test for a roof it draws on.
CELL = """        let px = vec2<i32>(i32(clamp(0.5 + 0.7 * dir.x, 0.0, 1.0) * 4096.0),
                           i32(clamp(0.5 + 1.4 * dir.z, 0.0, 0.999) * 4096.0));
        let ci = vec2<i32>(floor(city_camera_origin().xy / u.tile_cell))
                 + px - vec2<i32>(2048);
        let cc = city_cell(ci);
        if (!cc.built || cc.arch == 4 || cc.land_use != CITY_USE_URBAN) {
            return vec4<f32>(0.0);
        }
        let d = cc_rooftopworks_deck(cc, 0);
        if (!d.ok) {
            return vec4<f32>(0.0);
        }
        let cr = city_crown(cc, ci);
        let bk = cc_rooftopworks_bulk(cc, ci, cr);
"""

# 1. (deck area, tank lamps, their sides' integral, their tops') on a roof
# with clutter whose deck the far mean is spread over.
MECH = """    {
""" + CELL + """        let a0 = cc_rooftopworks_cell_deck(d, ci);
        if (!cc_rooftopworks_mech_on(cc) || a0 <= 1.0) {
            return vec4<f32>(0.0);
        }
        var n = 0.0;
        var side = 0.0;
        var top = 0.0;
        let k = 0.10;
        let sig = cc_rooftopworks_LAMP_SPILL;
        for (var s: i32 = 0; s < 4; s = s + 1) {
            let m = cc_rooftopworks_mech(cc, ci, s, true, cr, bk);
            if (m.ok && m.lamp) {
                n = n + 1.0;
                // The housing's lit surfaces as the far trace draws them at
                // the handoff footprint (past SDF_FP every housing is boxes,
                // cc_rooftopworks_mech_hit): a square tank's body from the
                // deck to its rim band and the wider band round its top (the
                // band's top is the tank's); a round tank's box from the
                // deck, stand and all; a condenser row's box; a vent pair's
                // two stacks.
                var lo = m.lo;
                var hi = m.hi;
                if (m.kind == 0 && !m.round) {
                    hi.z = m.alo.z;
                }
                side = side + probe_lamp_sides(m.seat, lo, hi, k, sig);
                if (m.kind == 0 && !m.round) {
                    side = side + probe_lamp_sides(m.seat, m.alo, m.ahi, k,
                                                   sig);
                    top = top + probe_lamp_face(m.seat, m.alo, m.ahi, 2, 1.0,
                                                k, sig);
                } else {
                    top = top + probe_lamp_face(m.seat, lo, hi, 2, 1.0, k,
                                                sig);
                }
                if (m.has_acc && m.kind == 2) {
                    side = side + probe_lamp_sides(m.seat, m.alo, m.ahi, k, sig);
                    top = top + probe_lamp_face(m.seat, m.alo, m.ahi, 2, 1.0,
                                                k, sig);
                }
            }
        }
        return vec4<f32>(a0, n, side, top);
    }
"""

# 2. A lit bulkhead's (front, back, sides, top), without (LIFT false) or
# with (LIFT true) its lift room; zero for any other roof.
BULK = """    {
""" + CELL + """        if (!(bk.ok && bk.lamp) || bk.lift != LIFT) {
            return vec4<f32>(0.0);
        }
        let leaf = vec4<f32>(probe_lamp_door(bk.seat, bk.lo, bk.hi, bk.door),
                             0.0, 0.0, 0.0);
        if (!bk.lift) {
            return probe_lamp_parts(bk.seat, bk.lo, bk.hi, bk.door,
                                    vec2<f32>(0.0)) - leaf;
        }
        // The lift room stands beside the stair along the stair's long
        // axis, flush with its back and taller and wider than it: the
        // stair's face toward the room is inside, and of the room's face
        // back toward the stair only what stands clear of the stair shows.
        let off = (0.5 * (bk.lift_lo + bk.lift_hi) - 0.5 * (bk.lo + bk.hi)).xy;
        let ax = select(vec2<f32>(0.0, sign(off.y)), vec2<f32>(sign(off.x), 0.0),
                        abs(off.x) > abs(off.y));
        let hidden = probe_lamp_face(bk.seat, bk.lo, bk.hi,
                                     select(1, 0, abs(ax.x) > 0.5),
                                     ax.x + ax.y, 0.12, 1.6);
        return probe_lamp_parts(bk.seat, bk.lo, bk.hi, bk.door, ax)
             + probe_lamp_parts(bk.seat, bk.lift_lo, bk.lift_hi, bk.door,
                                vec2<f32>(0.0))
             - vec4<f32>(0.0, 0.0, hidden, 0.0) - leaf;
    }
"""

PROBES = {
    "mech": MECH,
    "bulk": BULK.replace("LIFT", "false"),
    "bulk_lift": BULK.replace("LIFT", "true"),
}
# A footprint for the probes' fixture to build at first (any node will do).
FP_HANDOFF = 2.0
# The table's nodes.
NODES = 8
# The pooled estimates' sampling error is well under this; the shipped
# constants are rounded to three figures.
TOL = 0.02
W, H = 128, 72


@pytest.fixture(scope="module")
def probes():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar.renderer import SoarRenderer
    from cloudyview.soar.uniforms import SceneState
    from cloudyview.soar.uniforms import DEFAULT_OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(ANCHOR) == 1, (
        "surface_pass_city's opening moved; update ANCHOR")
    level = build_soar_level()
    built = {}

    def get(key):
        if key not in built:
            # Each renderer holds a device; the table's nodes ask for dozens
            # of builds, so only the latest few are kept.
            while len(built) >= 6:
                built.pop(next(iter(built)))
                gc.collect()
            name, fp, sq = key
            r = SoarRenderer(periodic=True, surface="city",
                             shader_source=source.replace(
                                 ANCHOR, PROBES[name] + ANCHOR)
                             + HELPERS.replace("FP_VALUE", f"{fp:.3f}")
                             .replace("SQ_VALUE", "true" if sq else "false"))
            r.upload_volume(level.sigma)
            built[key] = r
        return built[key]

    meta = get(("mech", FP_HANDOFF, False)).surface_meta
    dx = float(min(level.dx))
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dx * 2.0, dt_light=dx * 2.0, periodic=True,
        ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE,
        ocean_fif_dx=float(meta["cell_m"]),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1, ocean_enabled=True,
        surface="city")
    return get, state


def _plain(r):
    import wgpu
    tex = r._targets["plain"]
    row = math.ceil(W * 16 / 256) * 256
    buf = r.device.create_buffer(
        size=row * H,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc = r.device.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf, "offset": 0, "bytes_per_row": row,
         "rows_per_image": H}, (W, H, 1))
    r.device.queue.submit([enc.finish()])
    buf.map_sync(wgpu.MapMode.READ)
    a = np.frombuffer(bytearray(buf.read_mapped()), np.float32)
    buf.unmap()
    return a.reshape(H, row // 4)[:, :W * 4].reshape(H, W, 4)


def _run(probes, name, fp=FP_HANDOFF, density=1.0, sq=False):
    from cloudyview.soar.uniforms import ViewState
    get, state = probes
    view = ViewState(
        camera_position=[80000.0, 55000.0, 300.0], azimuth=0.0,
        elevation=0.0, fov=90.0, output_size=(W, H), render_size=(W, H),
        jitter=False, subpixel=False, exposure=4.0,
        sun_azimuth=310.0, sun_elevation=22.0, view_step_lod_degrees=0.0,
        city_roof_density=density)
    r = get((name, fp, sq))
    r.render(state, view, frames=1)
    out = _plain(r).astype(np.float64)
    assert np.isfinite(out).all(), f"{name} produced non-finite values"
    return out.reshape(-1, 4)


def _shipped(name):
    """A constant of rooftopworks.wgsl, as floats: a scalar, or a vector's
    components."""
    text = COMPONENT.read_text()
    m = re.search(rf"const cc_rooftopworks_{name}: [^=]+=\s*([^;]+);", text)
    assert m, f"cc_rooftopworks_{name} not found"
    value = m.group(1).strip()
    if "(" in value:
        value = value[value.index("(") + 1:value.rindex(")")]
        return np.array([float(x) for x in value.split(",")])
    return np.float64(float(value))


def _mech(probes, fp=FP_HANDOFF, density=1.0, sq=False):
    out = _run(probes, "mech", fp, density, sq)
    roofs = out[out[:, 0] > 0.0]
    assert len(roofs) > 1000, "too few roofs with clutter sampled"
    n = roofs[:, 1].sum()
    assert n > 200, "too few tank lamps sampled"
    rho = n / roofs[:, 0].sum()
    s = np.array([roofs[:, 2].sum(), roofs[:, 3].sum()]) / n
    return rho, s


def _bulk(probes, lift, fp=FP_HANDOFF, sq=False):
    out = _run(probes, "bulk_lift" if lift else "bulk", fp, sq=sq)
    lamps = out[out[:, 0] > 0.0]
    assert len(lamps) > 100, "too few lit bulkheads sampled"
    return lamps.mean(axis=0)


def _node(i):
    """The shipped table's node i: its footprint and its rows."""
    mech = _shipped(f"FAR_{i}_MECH")
    return (float(mech[3]), mech, _shipped(f"FAR_{i}_BULK"),
            _shipped(f"FAR_{i}_LIFT"))


def _node2(i):
    """The node's second-moment rows."""
    return (_shipped(f"FAR_{i}_MECH2"), _shipped(f"FAR_{i}_BULK2"),
            _shipped(f"FAR_{i}_LIFT2"))


def _within(name, measured, shipped):
    measured = np.atleast_1d(measured)
    shipped = np.atleast_1d(shipped)
    fmt = {"float_kind": lambda x: f"{x:.5g}"}
    print(f"{name}: measured {np.array2string(measured, formatter=fmt)}"
          f" shipped {np.array2string(shipped, formatter=fmt)}")
    assert np.all(np.abs(shipped - measured) <= TOL * np.abs(measured)), name


@pytest.mark.gpu
@pytest.mark.parametrize("i", range(5))
def test_tank_lamp_density(probes, i):
    d = list(_shipped("MECH_RHO_D")) + [float(_shipped("MECH_RHO_D4"))]
    at = list(_shipped("MECH_RHO_AT")) + [float(_shipped("MECH_RHO_AT4"))]
    rho, _ = _mech(probes, density=d[i])
    _within(f"MECH_RHO at density {d[i]:g}", rho, at[i])


def test_density_nodes_are_the_tiers():
    from cloudyview.soar.tiers import QUALITY_PRESETS
    d = list(_shipped("MECH_RHO_D")) + [float(_shipped("MECH_RHO_D4"))]
    # Every tier's density is a node, so the far mean is measured, not
    # interpolated, at each.
    assert {p["city_roof_density"] for p in QUALITY_PRESETS.values()} <= set(d)


@pytest.mark.gpu
@pytest.mark.parametrize("i", range(NODES))
def test_far_table_node(probes, i):
    fp, mech, bulk, lift = _node(i)
    _, s = _mech(probes, fp)
    _within(f"FAR_{i}_MECH.xy (fp {fp:g})", s, mech[:2])
    _within(f"FAR_{i}_BULK (fp {fp:g})", _bulk(probes, False, fp), bulk)
    _within(f"FAR_{i}_LIFT (fp {fp:g})", _bulk(probes, True, fp), lift)
    mech2, bulk2, lift2 = _node2(i)
    _, s2 = _mech(probes, fp, sq=True)
    _within(f"FAR_{i}_MECH2.xy (fp {fp:g})", s2, mech2[:2])
    _within(f"FAR_{i}_BULK2 (fp {fp:g})", _bulk(probes, False, fp, sq=True),
            bulk2)
    _within(f"FAR_{i}_LIFT2 (fp {fp:g})", _bulk(probes, True, fp, sq=True),
            lift2)
    # The pad lights' share on the pad: each light 1 m in from its corner,
    # its spot widened to max(PAD_LAMP_SIG, 0.3 fp), so the square of the
    # normal integral to 1 m at that width.
    sig = float(_shipped("PAD_LAMP_SIG"))
    sw = max(sig, 0.3 * fp)
    keep = (0.5 * (1.0 + math.erf(1.0 / (sw * math.sqrt(2.0))))) ** 2
    assert abs(mech[2] - keep) <= 1e-3, (mech[2], keep)
