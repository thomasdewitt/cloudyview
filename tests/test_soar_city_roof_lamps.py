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

1. MECH_RHO and MECH_S: over the urban roofs with clutter, the tank lamps
   per square metre of summit deck (in the cell's own column, less the tier
   standing on it), and per lamp the spots' integral over its housing's
   four sides together and over its top.
2. BULK_S and BULK_LIFT_S: per lit bulkhead, the spots' integral over its
   door face, its back, its two sides together and its top; with a lift
   room, the room's faces join the same four parts, less where the two
   boxes meet.

The integrals are midpoint sums at fp = 0 (the spot unwidened): the spot is
area-preserving under widening, so the sum is the same at any footprint.

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/test_soar_city_roof_lamps.py -v -s
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / "cloudyview/soar/raymarch.wgsl"
COMPONENT = ROOT / "cloudyview/soar/city/components/rooftopworks.wgsl"
ANCHOR = "    let city_o = city_camera_origin();\n    let chit = city_trace("

# Module-scope helpers for the probes: one face's integral of a lamp's two
# spots, as the prop shader sums them (source + k x spill), by a midpoint
# sum over the face rectangle o + s e1 + t e2.
HELPERS = """
fn probe_lamp_rect(seat: vec3<f32>, o: vec3<f32>, e1: vec3<f32>,
                   e2: vec3<f32>, k: f32, sig_spill: f32) -> f32 {
    let n = 40;
    var sum = 0.0;
    for (var j: i32 = 0; j < n; j = j + 1) {
        for (var i: i32 = 0; i < n; i = i + 1) {
            let p = o + e1 * ((f32(i) + 0.5) / f32(n))
                      + e2 * ((f32(j) + 0.5) / f32(n));
            sum = sum + cc_rooftopworks_spot(p, seat, cc_rooftopworks_LAMP_SIG,
                                             0.0)
                      + k * cc_rooftopworks_spot(p, seat, sig_spill, 0.0);
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
"""

# The cell this pixel walks: the frame's coordinates at 4096 steps a side,
# about the camera's cell (distinct cells, the city's tile folding the
# rest), and the roof hook's own test for a roof it draws on.
CELL = """        let px = vec2<i32>(i32(clamp(0.5 + 0.7 * dir.x, 0.0, 1.0) * 4096.0),
                           i32(clamp(0.5 + 1.4 * dir.z, 0.0, 0.999) * 4096.0));
        let ci = vec2<i32>(floor(city_camera_origin().xy / u.ocean_params.x))
                 + px - vec2<i32>(2048);
        let cc = city_cell(ci);
        if (!cc.built || cc.arch == 4 || cc.land_use != CITY_USE_URBAN) {
            return vec4<f32>(0.0);
        }
        let d = cc_rooftopworks_deck(cc, 0);
        if (!d.ok) {
            return vec4<f32>(0.0);
        }
        let cr = cc_rooftopworks_crown(cc, ci);
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
                side = side + probe_lamp_sides(m.seat, m.lo, m.hi, k, sig);
                top = top + probe_lamp_face(m.seat, m.lo, m.hi, 2, 1.0, k, sig);
                // A vent pair's second stack is a housing of its own; a
                // tank's rim band is part of the tank's.
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
        if (!bk.lift) {
            return probe_lamp_parts(bk.seat, bk.lo, bk.hi, bk.door,
                                    vec2<f32>(0.0));
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
             - vec4<f32>(0.0, 0.0, hidden, 0.0);
    }
"""

PROBES = {
    "mech": MECH,
    "bulk": BULK.replace("LIFT", "false"),
    "bulk_lift": BULK.replace("LIFT", "true"),
}
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
    from cloudyview.soar_host import SceneState, SoarRenderer
    from cloudyview.witness import OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(ANCHOR) == 1, (
        "surface_pass_city's opening moved; update ANCHOR")
    level = build_soar_level()
    built = {}

    def get(name):
        if name not in built:
            r = SoarRenderer(periodic=True, surface="city",
                             shader_source=source.replace(
                                 ANCHOR, PROBES[name] + ANCHOR) + HELPERS)
            r.upload_volume(level.sigma)
            built[name] = r
        return built[name]

    meta = get("mech").surface_meta
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


def _run(probes, name):
    from cloudyview.soar_host import ViewState
    get, state = probes
    view = ViewState(
        camera_position=[80000.0, 55000.0, 300.0], azimuth=0.0,
        elevation=0.0, fov=90.0, output_size=(W, H), render_size=(W, H),
        jitter=False, subpixel=False, exposure=4.0,
        sun_azimuth=310.0, sun_elevation=22.0, view_step_lod_degrees=0.0)
    r = get(name)
    r.render(state, view, frames=1)
    out = _plain(r).astype(np.float64)
    assert np.isfinite(out).all(), f"{name} produced non-finite values"
    return out.reshape(-1, 4)


def _shipped(name):
    """A constant of rooftopworks.wgsl, as floats: a scalar, or a vector's
    components."""
    text = COMPONENT.read_text()
    m = re.search(rf"const cc_rooftopworks_{name}: [^=]+= ([^;]+);", text)
    assert m, f"cc_rooftopworks_{name} not found"
    value = m.group(1).strip()
    if "(" in value:
        value = value[value.index("(") + 1:value.rindex(")")]
        return np.array([float(x) for x in value.split(",")])
    return np.float64(float(value))


def _report(name, measured):
    print(f"{name}: measured "
          f"{np.array2string(np.asarray(measured), precision=5)}"
          f" shipped {np.array2string(_shipped(name), precision=5)}")


def _check(name, measured):
    shipped = _shipped(name)
    assert shipped.shape == np.shape(measured), name
    assert np.all(np.abs(shipped - measured) <= TOL * np.abs(measured)), name


@pytest.mark.gpu
def test_tank_lamps_far_constants(probes):
    out = _run(probes, "mech")
    roofs = out[out[:, 0] > 0.0]
    assert len(roofs) > 1000, "too few roofs with clutter sampled"
    n = roofs[:, 1].sum()
    assert n > 200, "too few tank lamps sampled"
    rho = n / roofs[:, 0].sum()
    s = np.array([roofs[:, 2].sum(), roofs[:, 3].sum()]) / n
    _report("MECH_RHO", rho)
    _report("MECH_S", s)
    _check("MECH_RHO", rho)
    _check("MECH_S", s)


@pytest.mark.gpu
@pytest.mark.parametrize("lift", (False, True))
def test_bulkhead_lamp_far_constants(probes, lift):
    out = _run(probes, "bulk_lift" if lift else "bulk")
    lamps = out[out[:, 0] > 0.0]
    assert len(lamps) > 100, "too few lit bulkheads sampled"
    name = "BULK_LIFT_S" if lift else "BULK_S"
    _report(name, lamps.mean(axis=0))
    _check(name, lamps.mean(axis=0))
