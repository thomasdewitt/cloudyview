// streetfurniture — what stands on the sidewalk: hydrants, parking meters,
// bollards, litter baskets, newspaper boxes, benches, the black bags heaped
// at the kerb, and the dumpsters and wheelie bins against the building line.
//
// The camera can fly down onto the pavement and stop a metre from any of
// this, so every object is sphere-traced from a real SDF, to wiper-blade
// detail and no further (Thomas): a hydrant has its bonnet, operating nut,
// two hose caps and a pumper cap on their chains; a dumpster has its sloped
// front, fork pockets, casters, hinge knuckles and two lids, propped or
// thrown open over the garbage heaped inside; a bag has its knot. No text,
// no logos, nothing that would be noise at ten metres.
//
// LAYOUT. A built cell's sidewalk runs round its plot (THE STREET SURFACE in
// raymarch.wgsl): CITY_SIDEWALK_W (3 m, 4.5 m on an avenue) from the building
// line to the kerb line, its top at CITY_KERB_H. Each of the four EDGES whose
// city_edge_street is true carries two LANES, each a row of slots on a
// lattice anchored at the cell's corner:
//   * the KERB lane, 0.35-1.1 m in from the kerb line — where cities put
//     their furniture, clear of both the walking path and the car doors.
//     3 m slots. Hydrants (about one per block face), a parking meter every
//     6 m on the kerbs that have parking beside them, litter baskets at the
//     corners, and elsewhere a district-weighted draw of bag piles,
//     newspaper-box clusters, baskets, bollards and (on avenues) benches.
//   * the WALL lane, against the building line. 4.5 m slots. Dumpsters,
//     mostly near the corners where the service doors are; wheelie bins;
//     bags slumped against the wall.
// The lamp poles (streetlife) stand on the core's lattice, CITY_LAMP_OFFSET
// in from the building line every CITY_LAMP_SPACING along the edge — inside
// the kerb lane on a minor street. A kerb slot that a pole falls in holds
// nothing, or a pile of bags heaped round the pole's foot, which is where
// a city leaves them.
// Corners stay clear: every prop keeps CORNER_CLEAR inside the plot's own
// extent along its edge, so the kerb's quarter circles and the crosswalk
// mouths in line with the cross street's sidewalk are never obstructed.
// The wall lane is where city-facades' door positions (city_shell_door_u)
// belong once they land: the test is a rejection in cc_streetfurniture_prop,
// dropping a wall slot whose along extent overlaps a door, beside the
// storefront test that is already there.
//
// Everything is deterministic in (tile-wrapped cell, side, lane, slot), so
// the shader re-derives a prop from its hit position and nothing but the
// lane rides in CityHit: kind 800 + 2 side + lane. Every prop lies strictly
// inside the cell column that draws it, because the DDA tests a cell only
// from the side it enters. Reflected rays off the wet street never see these
// props: the core traces them with props = false (see fs_main).
//
// COST. The wide scene pays gate 1 below (two multiply-adds against the
// tallest thing here) and nothing else. A ray near the ground pays one slab
// test per lane and a hash per slot it passes, walked from the near end and
// stopped at the second box entered. The SDF runs for at most two candidate
// boxes per cell, in ONE rolled sphere-trace loop (streetlife's decision,
// for streetlife's measured reason: the trace written per slot is inlined
// into the core's 512-iteration DDA once per slot, and the register pressure
// of those copies costs occupancy on every city pixel). The march budget and
// the candidate count are run-time values so the driver cannot unroll either.
// There is no box-proxy LOD for the eye: the SDF runs to the far cutoff with
// its own detail ladder, because only pixels whose rays enter a prop's box
// pay for it, and a dumpster at the footprint where a proxy would take over
// is six pixels — the proxy would save six pixels of work and cost a pop.

const cc_streetfurniture_FLOOR: f32 = CITY_KERB_H;  // the sidewalk top
const cc_streetfurniture_Z_GATE: f32 = 2.6;   // an open dumpster lid, + kerb
const cc_streetfurniture_KERB_TOP: f32 = 1.45;  // kerb lane: a meter head
const cc_streetfurniture_WALL_TOP: f32 = 2.30;  // wall lane: lids open

// --- layout -----------------------------------------------------------------
const cc_streetfurniture_KERB_PITCH: f32 = 3.0;   // divides the 90 m block
const cc_streetfurniture_WALL_PITCH: f32 = 4.5;   // so does this
const cc_streetfurniture_KERB_IN: f32 = 0.72;     // kerb lane centre, from kerb
const cc_streetfurniture_CORNER_CLEAR: f32 = 0.9;
const cc_streetfurniture_POLE_CLEAR: f32 = 0.30;  // around a lamp mast
// The slot walk's budget. Enough to cross a whole cell along a lane (30 kerb
// slots, 20 wall slots) — a ray skimming the sidewalk at eye level really
// does pass that many — and a run-time bound, so it stays one rolled loop.
const cc_streetfurniture_MAX_SLOTS: i32 = 32;

// District weighting, the cascade value that streetlife's parking follows:
// downtown sidewalks are cluttered, the outskirts nearly bare.
const cc_streetfurniture_DENS_LO: f32 = 0.22;
const cc_streetfurniture_DENS_HI: f32 = 1.45;
const cc_streetfurniture_DENS_START: f32 = 0.005;
const cc_streetfurniture_DENS_FULL: f32 = 0.080;
const cc_streetfurniture_KERB_OCC: f32 = 0.40;

// --- prop types ---------------------------------------------------------------
const cc_streetfurniture_T_HYDRANT: i32 = 0;
const cc_streetfurniture_T_METER: i32 = 1;
const cc_streetfurniture_T_BOLLARD: i32 = 2;
const cc_streetfurniture_T_BASKET: i32 = 3;   // wire-mesh litter basket
const cc_streetfurniture_T_CAN: i32 = 4;      // solid steel litter can
const cc_streetfurniture_T_NEWS: i32 = 5;     // newspaper boxes, 2-4
const cc_streetfurniture_T_BENCH: i32 = 6;
const cc_streetfurniture_T_BAGS: i32 = 7;
const cc_streetfurniture_T_DUMPSTER: i32 = 8;
const cc_streetfurniture_T_WHEELIE: i32 = 9;

// --- materials (the SDF's second channel) -------------------------------------
const cc_streetfurniture_M_BAG_BLACK: f32 = 1.0;
const cc_streetfurniture_M_BAG_CLEAR: f32 = 2.0;
const cc_streetfurniture_M_BAG_BLUE: f32 = 3.0;
const cc_streetfurniture_M_CARDBOARD: f32 = 4.0;
const cc_streetfurniture_M_BOTTLE: f32 = 5.0;
const cc_streetfurniture_M_CAN: f32 = 6.0;
const cc_streetfurniture_M_RUBBER: f32 = 7.0;
const cc_streetfurniture_M_METAL: f32 = 8.0;
const cc_streetfurniture_M_PAINT: f32 = 9.0;
const cc_streetfurniture_M_LID: f32 = 10.0;
const cc_streetfurniture_M_GLASS: f32 = 11.0;
const cc_streetfurniture_M_INSIDE: f32 = 12.0;
const cc_streetfurniture_M_ACCENT: f32 = 13.0;
const cc_streetfurniture_M_WOOD: f32 = 14.0;
const cc_streetfurniture_M_IRON: f32 = 15.0;
const cc_streetfurniture_M_NEWS0: f32 = 30.0;  // + box index, 0..3

// --- LOD ----------------------------------------------------------------------
// The detail ladder, in the sharpened footprint (cc_streetfurniture_fpx):
// below FINE_FP the features a few millimetres across — the basket's wire
// mesh, a bag's knot ears, a hydrant's cap chain, film creases; below MID_FP
// the centimetre ones — handles, hinges, casters, coin hoods, lumps. Past
// FAR the props are two pixels and gone, faded into the ground beneath
// first so nothing switches off. The kerb lane's things are small (a
// hydrant is 0.3 m across), the wall lane's dumpster is 1.8 m, so each lane
// has its own far edge.
const cc_streetfurniture_FINE_FP: f32 = 0.012;
const cc_streetfurniture_MID_FP: f32 = 0.06;
const cc_streetfurniture_KERB_FAR: vec2<f32> = vec2<f32>(0.18, 0.30);
const cc_streetfurniture_WALL_FAR: vec2<f32> = vec2<f32>(0.45, 0.75);

// The march. Bags carry sine lumps whose gradient runs to ~1.5, so the march
// steps this fraction of the reported distance.
const cc_streetfurniture_STEP: f32 = 0.66;
const cc_streetfurniture_ITERS: i32 = 72;
const cc_streetfurniture_ITERS_FAR: i32 = 28;
const cc_streetfurniture_ITER_FP: f32 = 0.05;

// --- light --------------------------------------------------------------------
// Everything here is lit by city_street_light — the sodium heads as point
// sources, in the ground's radiance units — times an albedo RELATIVE TO
// ASPHALT. A downtown sidewalk under a lamp is already near clipping, so a
// black bag at 0.10 is a dark shape on a bright floor, which is both the
// photometry and the reference photographs. What says "black plastic" is
// not its diffuse at all but the glints: tight, broken highlights of the
// lamp heads on the wrinkled film (london_kerb_trashbags_night.jpg), and on
// a wet night every painted lid and cap carries them too.
// The palettes below are written as plain reflectances, the numbers a paint
// chart would give; ALB_GAIN turns them into the relative albedo the
// lighting wants. It is set by the tone curve, not by physics: the core's
// pavement under a downtown lamp is lit to radiance ~0.7 and rendered near
// white, and at exposure 6 a black bag that is to read as black (display
// ~0.15) must come out near radiance 0.007. At a gain of 1 the first renders
// showed pale grey balloons and white parking meters; 0.35 still left dark
// paint mid-grey beside a clipped pavement.
const cc_streetfurniture_ALB_GAIN: f32 = 0.25;
const cc_streetfurniture_WATER_F0: f32 = 0.02;
const cc_streetfurniture_WATER_N: f32 = 70.0;
// The lit pavement is a light source for anything standing on it: the
// underside and the lower flanks of a prop are lit mostly from below.
const cc_streetfurniture_BOUNCE: f32 = 0.10;
const cc_streetfurniture_WALK_WET: f32 = 1.05;   // CITY_REL_WALK, half-wet
// The pavement as a reflected ray sees it: the mean of wet sidewalk (~1.6
// relative) and wet road (~0.6), and blurred by the paint's roughness, so
// lower than either peak.
const cc_streetfurniture_FLOOR_REFL: f32 = 0.30;

// ---------------------------------------------------------------------------
// SDF primitives
// ---------------------------------------------------------------------------

fn cc_streetfurniture_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec3<f32>(r);
    return length(max(q, vec3<f32>(0.0)))
         + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Capped cylinder, axis z, half-height h.
fn cc_streetfurniture_cylz(p: vec3<f32>, rad: f32, h: f32) -> f32 {
    let d = vec2<f32>(length(p.xy) - rad, abs(p.z) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn cc_streetfurniture_seg(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, rad: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t) - rad;
}

// Ellipsoid, the bounded approximation (exact on the axes, never an
// overestimate by more than the axis ratio).
fn cc_streetfurniture_ell(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-5);
}

fn cc_streetfurniture_torz(p: vec3<f32>, big: f32, tube: f32) -> f32 {
    return length(vec2<f32>(length(p.xy) - big, p.z)) - tube;
}

fn cc_streetfurniture_smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn cc_streetfurniture_rot(v: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Nearest of two (distance, material) pairs.
fn cc_streetfurniture_u(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return select(a, b, b.x < a.x);
}

// Four uniforms per item of a prop (a bag in a pile, a box in a row), from
// the prop's seed. One pcg2d, sixteen bits each: this runs inside the march.
fn cc_streetfurniture_item(seed: u32, i: i32) -> vec4<f32> {
    let h = pcg2d(vec2<u32>(seed, u32(i) * 0x9e3779b9u + 0x7f4a7c15u));
    return vec4<f32>(f32(h.x & 0xffffu), f32(h.x >> 16u),
                     f32(h.y & 0xffffu), f32(h.y >> 16u)) / 65535.0;
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

struct cc_streetfurniture_Edge {
    ok: bool,
    ax: i32,          // the axis the edge runs along: 0 = x, 1 = y
    out: f32,         // +-1: which way the street lies, across the edge
    bl: f32,          // the building line (plot edge), across coordinate
    w: f32,           // sidewalk width
    a_lo: f32,        // usable along range: the plot's extent less the
    a_hi: f32,        //   corner clearance, clipped to the cell column
    a_cell: f32,      // the slot lattice's origin: the cell's own corner
    avenue: bool,
    park: bool,       // a parking lane runs beside this kerb
    pole_kerb: bool,  // the lamp line falls inside the kerb lane
    line: i32,        // the centreline this edge faces (its lamp lattice)
}

fn cc_streetfurniture_edge(ci: vec2<i32>, cc: CityCell, side: i32)
        -> cc_streetfurniture_Edge {
    let cellm = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cellm;
    let cmax = cmin + vec2<f32>(cellm);
    var e: cc_streetfurniture_Edge;
    e.ax = select(0, 1, side < 2);
    e.out = select(1.0, -1.0, side == 0 || side == 2);
    let hi_side = side == 1 || side == 3;
    let xs = side < 2;
    e.bl = select(select(cc.plot_min.y, cc.plot_max.y, hi_side),
                  select(cc.plot_min.x, cc.plot_max.x, hi_side), xs);
    let c_lo = select(cmin.y, cmin.x, xs);
    let c_hi = select(cmax.y, cmax.x, xs);
    let p_lo = select(cc.plot_min.x, cc.plot_min.y, xs);
    let p_hi = select(cc.plot_max.x, cc.plot_max.y, xs);
    e.a_cell = select(cmin.x, cmin.y, xs);
    let a_top = select(cmax.x, cmax.y, xs);
    e.w = city_sidewalk_w(e.bl);
    let kb = i32(round(e.bl / cellm));
    e.line = kb;
    e.avenue = city_is_avenue(kb);
    // Our kerb sits on the -out side of the centreline; a minor street
    // parks on one side only (city_park_side), an avenue on both.
    e.park = e.avenue || city_park_side(kb) == -e.out;
    e.a_lo = max(p_lo + cc_streetfurniture_CORNER_CLEAR, e.a_cell);
    e.a_hi = min(p_hi - cc_streetfurniture_CORNER_CLEAR, a_top);
    let kerb = e.bl + e.out * e.w;
    // A merged superblock's plot edge can lie in a sibling's column: that
    // sidewalk is the sibling's to furnish.
    let inside = min(e.bl, kerb) >= c_lo && max(e.bl, kerb) <= c_hi;
    let lane_lo = e.w - cc_streetfurniture_KERB_IN - 0.40;
    let lane_hi = e.w - cc_streetfurniture_KERB_IN + 0.40;
    e.pole_kerb = CITY_LAMP_OFFSET > lane_lo - 0.2
               && CITY_LAMP_OFFSET < lane_hi + 0.2;
    e.ok = inside && e.a_hi > e.a_lo + 1.0 && city_edge_street(cc, side);
    return e;
}

struct cc_streetfurniture_Prop {
    ok: bool,
    typ: i32,
    n: i32,            // how many: bags, boxes, bollards, bins
    ctr: vec3<f32>,    // world position of the local origin, on the floor
    ax: i32,           // local x runs along world axis `ax`
    out: f32,          // local +y is the across axis times `out`
    hb: vec3<f32>,     // local half extents along, across; top z
    r: vec4<f32>,      // shape draws
    seed: u32,
}

fn cc_streetfurniture_none() -> cc_streetfurniture_Prop {
    var v: cc_streetfurniture_Prop;
    v.ok = false;
    v.typ = 0;
    v.n = 0;
    v.ctr = vec3<f32>(0.0);
    v.ax = 0;
    v.out = 1.0;
    v.hb = vec3<f32>(0.0);
    v.r = vec4<f32>(0.0);
    v.seed = 0u;
    return v;
}

// The draws: per slot, and per edge (lane = 2 names the edge itself).
// Tile-wrapped (city_tile_cell), and the slot index is the cell's own, so a
// replayed track meets the same bags on the same kerbs.
fn cc_streetfurniture_draw(ci: vec2<i32>, side: i32, lane: i32, k: i32)
        -> vec4<u32> {
    let cw = city_tile_cell(ci);
    let a = pcg2d(vec2<u32>(
        bitcast<u32>(cw.x) * 0x2c1b3c6du + u32(side) * 0x297a2d39u
            + u32(lane) * 0x68e31da4u + u32(k) * 0x1b56c4e9u + 0x3c6ef372u,
        bitcast<u32>(cw.y) * 0xcf1bbcddu + u32(side) * 0x9e3779b9u
            + u32(lane) * 0x7feb352du + u32(k) * 0x846ca68bu + 0xa54ff53au));
    let b = pcg2d(a);
    return vec4<u32>(a, b);
}

fn cc_streetfurniture_u4(h: vec4<u32>) -> vec4<f32> {
    return vec4<f32>(city_u01(h.x), city_u01(h.y), city_u01(h.z),
                     city_u01(h.w));
}

// Local half extents and top of each type, `n` of them where it counts.
// This is the bounding box the DDA pays for, so it is tight: every SDF below
// stays inside the box its type declares here, yaw jitter included.
fn cc_streetfurniture_extent(typ: i32, n: i32) -> vec3<f32> {
    let nf = f32(n);
    if (typ == cc_streetfurniture_T_HYDRANT) {
        return vec3<f32>(0.31, 0.31, 0.80);
    }
    if (typ == cc_streetfurniture_T_METER) {
        return vec3<f32>(select(0.13, 0.28, n > 1), 0.13, 1.42);
    }
    if (typ == cc_streetfurniture_T_BOLLARD) {
        return vec3<f32>(0.55 * (nf - 1.0) + 0.15, 0.15, 1.00);
    }
    if (typ == cc_streetfurniture_T_BASKET || typ == cc_streetfurniture_T_CAN) {
        return vec3<f32>(0.36, 0.36, 1.22);
    }
    if (typ == cc_streetfurniture_T_NEWS) {
        return vec3<f32>(0.28 * nf + 0.08, 0.31, 1.14);
    }
    if (typ == cc_streetfurniture_T_BENCH) {
        return vec3<f32>(1.00, 0.36, 0.90);
    }
    if (typ == cc_streetfurniture_T_BAGS) {
        let nb = (2 * n + 2) / 3;
        let rows = select(1, 2, nb >= 4);
        let cols = (nb + rows - 1) / rows;
        return vec3<f32>(0.25 * f32(cols) + 0.45, 0.22 * f32(rows) + 0.36,
                         select(0.74, 1.06, n > nb));
    }
    if (typ == cc_streetfurniture_T_DUMPSTER) {
        return vec3<f32>(1.08, 0.62, 2.30);
    }
    // Wheelie bins, 0.66 m apart, and a bag split beside them.
    return vec3<f32>(0.33 * nf + 0.48, 0.62, 1.36);
}

// Whatever stands in slot k of lane `lane` on edge `side`.
fn cc_streetfurniture_prop(ci: vec2<i32>, cc: CityCell,
                           e: cc_streetfurniture_Edge, side: i32, lane: i32,
                           k: i32) -> cc_streetfurniture_Prop {
    let kerb = lane == 0;
    let pitch = select(cc_streetfurniture_WALL_PITCH,
                       cc_streetfurniture_KERB_PITCH, kerb);
    let s0 = max(e.a_cell + f32(k) * pitch, e.a_lo);
    let s1 = min(e.a_cell + f32(k + 1) * pitch, e.a_hi);
    if (s1 - s0 < 0.5) {
        return cc_streetfurniture_none();
    }
    let hsd = cc_streetfurniture_draw(ci, side, lane, k);
    let r = cc_streetfurniture_u4(hsd);
    let er = cc_streetfurniture_u4(cc_streetfurniture_draw(ci, side, 2, 0));
    let dens = mix(cc_streetfurniture_DENS_LO, cc_streetfurniture_DENS_HI,
                   smoothstep(cc_streetfurniture_DENS_START,
                              cc_streetfurniture_DENS_FULL, cc.density));
    let k_lo = i32(floor((e.a_lo - e.a_cell) / pitch));
    let k_hi = i32(floor((e.a_hi - e.a_cell) / pitch - 1e-4));
    let mid = 0.5 * (s0 + s1);
    var typ = -1;
    var n = 1;
    var along = mid;
    var u_c = e.w - cc_streetfurniture_KERB_IN;   // distance from building line
    if (kerb) {
        // The nearest lamp mast, and whether it stands in this slot.
        let pole_a = city_lamp_station(e.line, mid);
        let pole_in = e.pole_kerb && pole_a > s0 - 0.45 && pole_a < s1 + 0.45;
        let from_lo = k - k_lo;
        let from_hi = k_hi - k;
        // One hydrant per block face, most of them, a slot or two in from
        // a corner.
        let k_hyd = select(k_lo + 1 + i32(er.y * 2.99),
                           k_hi - 1 - i32(fract(er.y * 5.3) * 2.99),
                           fract(er.y * 17.1) > 0.5);
        if (pole_in) {
            if (r.x < 0.40 * dens) {
                typ = cc_streetfurniture_T_BAGS;
                n = 3 + i32(r.z * 5.99);
                along = pole_a + select(-0.25, 0.25, r.w > 0.5);
            }
        } else if (k == k_hyd && er.y < 0.85) {
            typ = cc_streetfurniture_T_HYDRANT;
            along = mid + (r.y - 0.5) * 1.2;
        } else if ((from_lo == 0 && er.w < 0.55)
                   || (from_hi == 0 && fract(er.w * 13.7) < 0.55)) {
            // The corner basket: the one piece of furniture every city puts
            // at the corner, where people are waiting to cross.
            typ = select(cc_streetfurniture_T_CAN, cc_streetfurniture_T_BASKET,
                         r.y < 0.68);
            along = select(s1 - 0.45, s0 + 0.45, from_lo == 0);
        } else if (e.park && er.x < 0.55 && (from_lo & 1) == 0) {
            // A meter every 6 m along a parking kerb, at the kerb's own
            // setback: they are for the driver, not the walker.
            typ = cc_streetfurniture_T_METER;
            n = select(1, 2, r.y > 0.72);
            u_c = e.w - 0.45;
        } else if (r.x < cc_streetfurniture_KERB_OCC * dens) {
            let d = r.y;
            if (d < 0.32) {
                typ = cc_streetfurniture_T_BAGS;
                n = 3 + i32(r.z * 9.99);
                u_c = e.w - 0.62;
            } else if (d < 0.50) {
                typ = cc_streetfurniture_T_NEWS;
                n = 2 + i32(r.z * 2.99);
                u_c = e.w - 0.68;
            } else if (d < 0.63) {
                typ = select(cc_streetfurniture_T_CAN,
                             cc_streetfurniture_T_BASKET, r.z < 0.6);
            } else if (d < 0.76) {
                typ = cc_streetfurniture_T_BOLLARD;
                n = select(2, 3, r.z > 0.55);
            } else if (d < 0.90 && e.avenue) {
                typ = cc_streetfurniture_T_BENCH;
                u_c = e.w - 0.85;
            } else {
                typ = cc_streetfurniture_T_NEWS;
                n = 2 + i32(r.z * 2.99);
                u_c = e.w - 0.68;
            }
            along = mid + (r.w - 0.5) * 0.8;
        }
    } else {
        // Dumpsters stand where the service doors are: near the corners,
        // and rarely in front of a shop window. (city-facades' door
        // positions will sharpen both: see the header.)
        let corner = k - k_lo <= 1 || k_hi - k <= 1;
        let store = dot(city_storefront_glow(cc), vec3<f32>(1.0)) > 0.0;
        let p_dump = select(0.05, 0.30, corner) * select(1.0, 0.5, store)
                   * dens;
        let p_wheel = 0.07 * dens;
        let p_bags = 0.09 * dens;
        if (r.x < p_dump) {
            typ = cc_streetfurniture_T_DUMPSTER;
            u_c = 0.60;
        } else if (r.x < p_dump + p_wheel) {
            typ = cc_streetfurniture_T_WHEELIE;
            n = select(1, 2, r.z > 0.45);
            u_c = 0.44;
        } else if (r.x < p_dump + p_wheel + p_bags) {
            typ = cc_streetfurniture_T_BAGS;
            n = 4 + i32(r.z * 8.99);
            u_c = 0.52;
        }
        along = mid + (r.w - 0.5) * 1.2;
    }
    if (typ < 0) {
        return cc_streetfurniture_none();
    }
    let hb = cc_streetfurniture_extent(typ, n);
    // Inside the slot (the shader finds the slot again by floor), inside the
    // usable edge, and clear of the lamp mast where the lane crosses it.
    if (2.0 * hb.x + 0.04 > s1 - s0) {
        return cc_streetfurniture_none();
    }
    along = clamp(along, s0 + hb.x + 0.02, s1 - hb.x - 0.02);
    if (kerb && e.pole_kerb && typ != cc_streetfurniture_T_BAGS) {
        let pole_a = city_lamp_station(e.line, along);
        if (abs(along - pole_a) < hb.x + cc_streetfurniture_POLE_CLEAR) {
            return cc_streetfurniture_none();
        }
    }
    // The building line is solid wall: nothing may stand into it.
    u_c = max(u_c, hb.y + 0.03);
    var v: cc_streetfurniture_Prop;
    v.ok = true;
    v.typ = typ;
    v.n = n;
    let across = e.bl + e.out * u_c;
    v.ctr = vec3<f32>(select(across, along, e.ax == 0),
                      select(along, across, e.ax == 0),
                      cc_streetfurniture_FLOOR);
    v.ax = e.ax;
    v.out = e.out;
    v.hb = hb;
    v.r = r;
    v.seed = hsd.w ^ (hsd.x * 0x9e3779b9u);
    return v;
}

fn cc_streetfurniture_to_local(v: cc_streetfurniture_Prop, w: vec3<f32>)
        -> vec3<f32> {
    let rel = w - v.ctr;
    return select(vec3<f32>(rel.y, v.out * rel.x, rel.z),
                  vec3<f32>(rel.x, v.out * rel.y, rel.z), v.ax == 0);
}

// Direction local -> world (the frame is a signed axis permutation).
fn cc_streetfurniture_to_world(v: cc_streetfurniture_Prop, d: vec3<f32>)
        -> vec3<f32> {
    return select(vec3<f32>(v.out * d.y, d.x, d.z),
                  vec3<f32>(d.x, v.out * d.y, d.z), v.ax == 0);
}

// The world bounding box. A struct and not a two-element array: a CITY-only
// function-local array has moved DAY pixels in this kernel before.
struct cc_streetfurniture_Box {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

fn cc_streetfurniture_box(v: cc_streetfurniture_Prop) -> cc_streetfurniture_Box {
    let hx = select(vec2<f32>(v.hb.y, v.hb.x), v.hb.xy, v.ax == 0);
    return cc_streetfurniture_Box(vec3<f32>(v.ctr.xy - hx, v.ctr.z - 0.03),
                                  vec3<f32>(v.ctr.xy + hx, v.ctr.z + v.hb.z));
}

// ---------------------------------------------------------------------------
// Garbage: the bag, and what is heaped in an open container
// ---------------------------------------------------------------------------

// One bag, q relative to its centre. A bag is a slumped ellipsoid, wider low
// where its load settles, pressed out into lumps by what is inside it, and
// gathered at the top into a neck and a knot with two tied ears — the one
// silhouette feature that says "bag" rather than "boulder". The film's
// creases are what break the lamp's reflection into the scattered sparks the
// London reference shows; smooth, the glint would be one clean highlight and
// the bag would read as a bowling ball.
fn cc_streetfurniture_bag(q: vec3<f32>, rad: vec3<f32>, ph: f32, lod: i32)
        -> f32 {
    let s = clamp(q.z / rad.z, -1.0, 1.0);
    let rr = vec3<f32>(rad.xy * (1.10 - 0.22 * s), rad.z);
    // Half ellipsoid, half rounded box: the film drapes over whatever is
    // inside, and what is inside has corners. A pure ellipsoid rendered as a
    // glossy gourd — a balloon, not a bag.
    var d = mix(cc_streetfurniture_ell(q, rr),
                cc_streetfurniture_rbox(q, 0.86 * rr, 0.14), 0.30);
    if (lod <= 1) {
        d = d + 0.034 * sin(q.x * 8.0 + ph) * sin(q.y * 9.0 + 1.7 * ph)
                      * sin(q.z * 7.0 + 2.3 * ph);
    }
    if (lod == 0) {
        // Creases: V-shaped folds (the abs), which is what breaks the lamp's
        // reflection into separate sparks along the fold lines.
        d = d + 0.009 * abs(sin(q.x * 27.0 + 3.0 * ph) * sin(q.y * 23.0 + ph)
                            + sin(q.z * 25.0 - ph));
    }
    // The gathered neck, short, leaning the way the bag was dropped.
    let lean = vec2<f32>(cos(ph), sin(ph)) * 0.04;
    let knot = vec3<f32>(lean, rad.z + 0.03);
    d = cc_streetfurniture_smin(
        d, cc_streetfurniture_seg(q, vec3<f32>(0.0, 0.0, 0.70 * rad.z), knot,
                                  0.030), 0.07);
    if (lod <= 1) {
        // The tied ears: two flat lobes either side of the knot.
        let kq = q - knot - vec3<f32>(0.0, 0.0, 0.014);
        let ea = vec2<f32>(cos(ph * 1.3), sin(ph * 1.3));
        let side = dot(kq.xy, ea);
        let big = select(0.75, 1.0, side > 0.0);   // one ear longer
        let ax = vec3<f32>(abs(side) - 0.040 * big,
                           dot(kq.xy, vec2<f32>(-ea.y, ea.x)),
                           kq.z - 0.012 - 0.35 * abs(side));
        d = min(d, cc_streetfurniture_ell(
            ax, vec3<f32>(0.048 * big, 0.009, 0.020)));
    }
    return d;
}

fn cc_streetfurniture_bag_mat(h: f32) -> f32 {
    if (h < 0.80) {
        return cc_streetfurniture_M_BAG_BLACK;
    }
    return select(cc_streetfurniture_M_BAG_BLUE, cc_streetfurniture_M_BAG_CLEAR,
                  h < 0.92);
}

// A heap of n bags: a bottom layer in one or two rows along the kerb, the
// rest thrown on top across the joints. Every bag is its own draw, so no
// two heaps agree.
fn cc_streetfurniture_pile(p: vec3<f32>, v: cc_streetfurniture_Prop, lod: i32)
        -> vec2<f32> {
    let n = v.n;
    let nb = (2 * n + 2) / 3;
    let rows = select(1, 2, nb >= 4);
    let cols = (nb + rows - 1) / rows;
    let nt = n - nb;
    let q0 = vec3<f32>(cc_streetfurniture_rot(p.xy, (v.r.w - 0.5) * 0.12), p.z);
    var best = vec2<f32>(1e9, cc_streetfurniture_M_BAG_BLACK);
    for (var i: i32 = 0; i < n; i = i + 1) {
        let h = cc_streetfurniture_item(v.seed, i);
        var c: vec3<f32>;
        var rad: vec3<f32>;
        if (i < nb) {
            let col = i / rows;
            let row = i % rows;
            rad = vec3<f32>(0.22 + 0.07 * h.z, 0.20 + 0.06 * h.w,
                            0.19 + 0.07 * h.x);
            c = vec3<f32>((f32(col) - 0.5 * f32(cols - 1)) * 0.44
                              + (h.x - 0.5) * 0.10 + f32(row) * 0.11,
                          (f32(row) - 0.5 * f32(rows - 1)) * 0.40
                              + (h.y - 0.5) * 0.10,
                          0.70 * rad.z);
        } else {
            let j = i - nb;
            rad = vec3<f32>(0.20 + 0.06 * h.z, 0.18 + 0.05 * h.w,
                            0.18 + 0.06 * h.x);
            c = vec3<f32>((f32(j) - 0.5 * f32(nt - 1)) * 0.46
                              + (h.x - 0.5) * 0.14,
                          (h.y - 0.5) * 0.22,
                          0.36 + 0.50 * rad.z);
        }
        let q = q0 - c;
        // A bag cannot be nearer than its bounding sphere: skip the field
        // for the ones that cannot win.
        if (length(q) - 1.12 * max(rad.x, max(rad.y, rad.z)) - 0.11
            > best.x) {
            continue;
        }
        let d = cc_streetfurniture_bag(q, rad, h.w * 6.2831853 + f32(i), lod);
        let m = cc_streetfurniture_bag_mat(fract(h.z * 7.13 + h.y));
        // Bags pressed together share a crease, not a seam.
        let dd = cc_streetfurniture_smin(best.x, d, 0.06);
        best = vec2<f32>(dd, select(best.y, m, d < best.x));
    }
    return best;
}

// What fills an open container, relative to the rim's centre (z = 0 at the
// rim), clipped to the interior — a rounded rectangle of half extents `hi`
// and corner radius `cr` — and allowed to bulge a little over the rim: a
// couple of bags, a flap of soggy cardboard standing up out of it, a bottle
// and a can on top. `nb` bags spread along the long side.
fn cc_streetfurniture_heap(q: vec3<f32>, hi: vec2<f32>, cr: f32, nb: i32,
                           seed: u32, lod: i32) -> vec2<f32> {
    var best = vec2<f32>(1e9, cc_streetfurniture_M_BAG_BLACK);
    for (var i: i32 = 0; i < nb; i = i + 1) {
        let h = cc_streetfurniture_item(seed, i + 17);
        let fx = (f32(i) + 0.5) / f32(nb) * 2.0 - 1.0;
        let rad = vec3<f32>(min(0.17 + 0.06 * h.z, 0.9 * hi.x),
                            min(0.16 + 0.05 * h.w, 0.9 * hi.y),
                            0.15 + 0.06 * h.x);
        let c = vec3<f32>(fx * max(hi.x - rad.x, 0.0) + (h.y - 0.5) * 0.08,
                          (h.x - 0.5) * max(hi.y - rad.y, 0.0),
                          -0.10 + 0.14 * h.w);
        let qq = q - c;
        if (length(qq) - 1.12 * max(rad.x, rad.z) - 0.11 > best.x) {
            continue;
        }
        let d = cc_streetfurniture_bag(qq, rad, h.z * 6.2831853 + f32(i), lod);
        let dd = cc_streetfurniture_smin(best.x, d, 0.03);
        best = vec2<f32>(dd, select(best.y,
                                    cc_streetfurniture_bag_mat(h.y), d < best.x));
    }
    let g = cc_streetfurniture_item(seed, 3);
    // Cardboard: a flat flap, tilted up out of the heap.
    let cq = q - vec3<f32>((g.x - 0.5) * hi.x, (g.y - 0.5) * 0.4 * hi.y, 0.10);
    let cy = cc_streetfurniture_rot(cq.xy, g.z * 3.14159);
    let ct = cc_streetfurniture_rot(vec2<f32>(cy.y, cq.z), 0.9 + 0.5 * g.w);
    best = cc_streetfurniture_u(best, vec2<f32>(
        cc_streetfurniture_rbox(vec3<f32>(cy.x, ct.x, ct.y),
                                vec3<f32>(0.15, 0.004, 0.12), 0.003),
        cc_streetfurniture_M_CARDBOARD));
    if (lod <= 1) {
        // A bottle lying on the heap, neck up, and a crushed can.
        let bq = q - vec3<f32>((g.w - 0.5) * hi.x, (0.5 - g.x) * 0.5 * hi.y,
                               0.08 + 0.05 * g.z);
        let ba = vec3<f32>(cos(g.y * 6.28), sin(g.y * 6.28), 0.45);
        let bottle = cc_streetfurniture_smin(
            cc_streetfurniture_seg(bq, vec3<f32>(0.0), ba * 0.15, 0.032),
            cc_streetfurniture_seg(bq, ba * 0.15, ba * 0.25, 0.011), 0.025);
        best = cc_streetfurniture_u(best, vec2<f32>(
            bottle, cc_streetfurniture_M_BOTTLE));
        let kq = q - vec3<f32>((0.5 - g.z) * hi.x, (g.w - 0.5) * 0.5 * hi.y,
                               0.12);
        best = cc_streetfurniture_u(best, vec2<f32>(
            cc_streetfurniture_seg(kq, vec3<f32>(-0.05, 0.02, 0.0),
                                   vec3<f32>(0.05, -0.02, 0.02), 0.030),
            cc_streetfurniture_M_CAN));
    }
    // Clip to the container: inside its walls below the rim, a hand's width
    // of overflow above it.
    let grow = 0.05 * smoothstep(0.0, 0.08, q.z);
    let qa = abs(q.xy) - (hi - vec2<f32>(cr)) - vec2<f32>(grow);
    let wall = length(max(qa, vec2<f32>(0.0))) + min(max(qa.x, qa.y), 0.0)
             - cr + 0.012;
    best.x = max(best.x, wall);
    return best;
}

// ---------------------------------------------------------------------------
// The props
// ---------------------------------------------------------------------------
//
// Local frame: x along the kerb, y toward the street, z up from the sidewalk
// top. Each returns (distance, material).

// A fire hydrant: flange, barrel, bonnet with its operating nut, two hose
// outlets with caps and a pumper outlet facing the street, and on the close
// read the chains that keep the caps from walking off.
fn cc_streetfurniture_hydrant(p: vec3<f32>, v: cc_streetfurniture_Prop,
                              lod: i32) -> vec2<f32> {
    let q = vec3<f32>(cc_streetfurniture_rot(p.xy, (v.r.z - 0.5) * 0.8), p.z);
    var d = cc_streetfurniture_cylz(q - vec3<f32>(0.0, 0.0, 0.025), 0.150,
                                    0.035);
    d = min(d, cc_streetfurniture_cylz(q - vec3<f32>(0.0, 0.0, 0.26), 0.112,
                                       0.30));
    d = min(d, cc_streetfurniture_cylz(q - vec3<f32>(0.0, 0.0, 0.555), 0.132,
                                       0.020));
    // Bonnet: a squat dome.
    d = min(d, cc_streetfurniture_ell(q - vec3<f32>(0.0, 0.0, 0.585),
                                      vec3<f32>(0.118, 0.118, 0.095)));
    var res = vec2<f32>(d, cc_streetfurniture_M_PAINT);
    // Operating nut: pentagonal in life, a short cylinder at this scale.
    res = cc_streetfurniture_u(res, vec2<f32>(
        cc_streetfurniture_cylz(q - vec3<f32>(0.0, 0.0, 0.70), 0.030, 0.028),
        cc_streetfurniture_M_ACCENT));
    // Side outlets, mirrored on |x|.
    let sq = vec3<f32>(abs(q.x), q.y, q.z);
    let noz = cc_streetfurniture_cylz(
        vec3<f32>(sq.z - 0.43, q.y, sq.x - 0.16), 0.050, 0.055);
    res = cc_streetfurniture_u(res, vec2<f32>(noz, cc_streetfurniture_M_PAINT));
    var caps = cc_streetfurniture_cylz(
        vec3<f32>(sq.z - 0.43, q.y, sq.x - 0.225), 0.062, 0.022);
    // Pumper outlet, bigger, facing the street.
    res = cc_streetfurniture_u(res, vec2<f32>(
        cc_streetfurniture_cylz(vec3<f32>(q.x, q.z - 0.38, q.y - 0.16), 0.070,
                                0.055),
        cc_streetfurniture_M_PAINT));
    caps = min(caps, cc_streetfurniture_cylz(
        vec3<f32>(q.x, q.z - 0.38, q.y - 0.235), 0.086, 0.026));
    if (lod <= 1) {
        // The cap nuts.
        caps = min(caps, cc_streetfurniture_rbox(
            vec3<f32>(sq.x - 0.255, q.y, q.z - 0.43),
            vec3<f32>(0.012, 0.022, 0.022), 0.003));
        caps = min(caps, cc_streetfurniture_rbox(
            vec3<f32>(q.x, q.y - 0.268, q.z - 0.38),
            vec3<f32>(0.026, 0.012, 0.026), 0.003));
    }
    res = cc_streetfurniture_u(res, vec2<f32>(caps, cc_streetfurniture_M_ACCENT));
    if (lod == 0) {
        // Each side cap hangs a chain back to a lug on the barrel.
        let ch = min(
            cc_streetfurniture_seg(sq, vec3<f32>(0.225, 0.045, 0.40),
                                   vec3<f32>(0.175, 0.085, 0.315), 0.005),
            cc_streetfurniture_seg(sq, vec3<f32>(0.175, 0.085, 0.315),
                                   vec3<f32>(0.105, 0.060, 0.37), 0.005));
        res = cc_streetfurniture_u(res, vec2<f32>(ch, cc_streetfurniture_M_METAL));
    }
    return res;
}

// A parking meter: post, collar, and a head with a domed display window on
// each face and a coin-slot hood under it. Two in five kerbs have twin heads
// on a yoke.
fn cc_streetfurniture_meter(p: vec3<f32>, v: cc_streetfurniture_Prop,
                            lod: i32) -> vec2<f32> {
    var res = vec2<f32>(
        min(cc_streetfurniture_cylz(p - vec3<f32>(0.0, 0.0, 0.52), 0.034, 0.56),
            cc_streetfurniture_cylz(p - vec3<f32>(0.0, 0.0, 0.03), 0.058,
                                    0.045)),
        cc_streetfurniture_M_PAINT);
    var hq = p;
    if (v.n > 1) {
        res = cc_streetfurniture_u(res, vec2<f32>(
            cc_streetfurniture_rbox(p - vec3<f32>(0.0, 0.0, 1.07),
                                    vec3<f32>(0.16, 0.025, 0.025), 0.01),
            cc_streetfurniture_M_METAL));
        hq.x = abs(p.x) - 0.15;
    }
    res = cc_streetfurniture_u(res, vec2<f32>(
        min(cc_streetfurniture_rbox(hq - vec3<f32>(0.0, 0.0, 1.22),
                                    vec3<f32>(0.10, 0.075, 0.14), 0.03),
            cc_streetfurniture_ell(hq - vec3<f32>(0.0, 0.0, 1.35),
                                   vec3<f32>(0.10, 0.075, 0.055))),
        cc_streetfurniture_M_PAINT));
    let fq = vec3<f32>(hq.x, abs(hq.y), hq.z);
    res = cc_streetfurniture_u(res, vec2<f32>(
        cc_streetfurniture_ell(fq - vec3<f32>(0.0, 0.066, 1.27),
                               vec3<f32>(0.068, 0.030, 0.058)),
        cc_streetfurniture_M_GLASS));
    if (lod <= 1) {
        res = cc_streetfurniture_u(res, vec2<f32>(
            cc_streetfurniture_rbox(fq - vec3<f32>(0.0, 0.080, 1.15),
                                    vec3<f32>(0.032, 0.012, 0.018), 0.004),
            cc_streetfurniture_M_METAL));
    }
    return res;
}

// Steel bollards, a row of two or three, domed tops and a cap ring.
fn cc_streetfurniture_bollard(p: vec3<f32>, v: cc_streetfurniture_Prop)
        -> vec2<f32> {
    let half = 0.5 * f32(v.n - 1);
    let i = clamp(round(p.x / 1.1 + half), 0.0, f32(v.n - 1));
    let q = vec3<f32>(p.x - (i - half) * 1.1, p.y, p.z);
    var d = cc_streetfurniture_cylz(q - vec3<f32>(0.0, 0.0, 0.40), 0.105,
                                    0.43);
    d = min(d, cc_streetfurniture_ell(q - vec3<f32>(0.0, 0.0, 0.83),
                                      vec3<f32>(0.105, 0.105, 0.07)));
    let ring = cc_streetfurniture_cylz(q - vec3<f32>(0.0, 0.0, 0.76), 0.118,
                                       0.018);
    return cc_streetfurniture_u(vec2<f32>(d, cc_streetfurniture_M_PAINT),
                                vec2<f32>(ring, cc_streetfurniture_M_ACCENT));
}

// A litter basket. Wire mesh (the green New York one) or a solid steel can
// with ribs and a liner bag folded over the rim; both open-topped, both
// full, the garbage standing proud of the rim. The mesh is real geometry at
// the finest step — the rays pass through its holes and meet the garbage
// inside, which is what makes a wire basket look like one — and a solid
// shell, shaded as its own mesh mean, past it.
fn cc_streetfurniture_basket(p: vec3<f32>, v: cc_streetfurniture_Prop,
                             lod: i32) -> vec2<f32> {
    let wire = v.typ == cc_streetfurniture_T_BASKET;
    let top = 0.84;
    let rz = mix(0.262, 0.298, clamp(p.z / top, 0.0, 1.0));
    let rho = length(p.xy);
    let radial = rho - rz;
    var shell = max(abs(radial) - select(0.011, 0.007, wire),
                    abs(p.z - 0.5 * top) - 0.5 * top);
    if (wire && lod == 0) {
        let ang = atan2(p.y, p.x) * (32.0 / 6.2831853);
        let cu = (fract(ang) - 0.5) * (6.2831853 * 0.28 / 32.0);
        let cz = (fract((p.z - 0.075) / 0.055) - 0.5) * 0.055;
        let hole = max(max(abs(cu), abs(cz)) - 0.019,
                       abs(p.z - 0.43) - 0.33);
        shell = max(shell, -hole);
    }
    // Interior faces are a different material: in shadow down the barrel.
    var res = vec2<f32>(shell, select(cc_streetfurniture_M_PAINT,
                                      cc_streetfurniture_M_INSIDE, radial < 0.0));
    var ring = cc_streetfurniture_torz(p - vec3<f32>(0.0, 0.0, top), 0.298,
                                       0.015);
    ring = min(ring, cc_streetfurniture_cylz(p - vec3<f32>(0.0, 0.0, 0.01),
                                             0.264, 0.022));
    if (wire) {
        ring = min(ring, cc_streetfurniture_torz(
            p - vec3<f32>(0.0, 0.0, 0.29), 0.275, 0.009));
        ring = min(ring, cc_streetfurniture_torz(
            p - vec3<f32>(0.0, 0.0, 0.57), 0.287, 0.009));
    } else {
        ring = min(ring, cc_streetfurniture_torz(
            p - vec3<f32>(0.0, 0.0, 0.22), 0.272, 0.013));
        ring = min(ring, cc_streetfurniture_torz(
            p - vec3<f32>(0.0, 0.0, 0.52), 0.284, 0.013));
    }
    res = cc_streetfurniture_u(res, vec2<f32>(ring, cc_streetfurniture_M_PAINT));
    if (!wire) {
        // The liner, folded down over the rim.
        let lq = vec2<f32>(rho - 0.305, p.z - (top - 0.045));
        let liner = length(max(abs(lq) - vec2<f32>(0.004, 0.045),
                               vec2<f32>(0.0))) - 0.004;
        res = cc_streetfurniture_u(res, vec2<f32>(
            liner, cc_streetfurniture_M_BAG_BLACK));
    }
    return cc_streetfurniture_u(res, cc_streetfurniture_heap(
        p - vec3<f32>(0.0, 0.0, top), vec2<f32>(0.285), 0.285, 2, v.seed, lod));
}

// Newspaper boxes, two to four in a row, each its own colour and a few
// degrees off square: body on four short legs, a chamfered top carrying the
// coin mechanism, a recessed window on the sidewalk face with the paper
// behind it, a pull handle, and the coin slot's hood.
fn cc_streetfurniture_news(p: vec3<f32>, v: cc_streetfurniture_Prop, lod: i32)
        -> vec2<f32> {
    var best = vec2<f32>(1e9, cc_streetfurniture_M_NEWS0);
    let half = 0.5 * f32(v.n - 1);
    for (var i: i32 = 0; i < v.n; i = i + 1) {
        let h = cc_streetfurniture_item(v.seed, i);
        let c = vec2<f32>((f32(i) - half) * 0.56 + (h.x - 0.5) * 0.04,
                          (h.z - 0.5) * 0.05);
        let b = vec3<f32>(cc_streetfurniture_rot(p.xy - c, (h.y - 0.5) * 0.14),
                          p.z);
        if (cc_streetfurniture_rbox(b - vec3<f32>(0.0, 0.0, 0.54),
                                    vec3<f32>(0.26, 0.24, 0.58), 0.0)
            > best.x) {
            continue;
        }
        var body = cc_streetfurniture_rbox(b - vec3<f32>(0.0, 0.0, 0.60),
                                           vec3<f32>(0.245, 0.21, 0.43), 0.022);
        body = max(body, dot(vec2<f32>(b.y + 0.21, b.z - 0.95),
                             vec2<f32>(-0.6, 0.8)));
        // The window, recessed 12 mm into the sidewalk face.
        let win = cc_streetfurniture_rbox(b - vec3<f32>(0.0, -0.225, 0.715),
                                          vec3<f32>(0.19, 0.027, 0.16), 0.012);
        body = max(body, -win);
        let mat = select(cc_streetfurniture_M_NEWS0 + f32(i),
                         cc_streetfurniture_M_GLASS,
                         win < 0.004 && b.y > -0.212);
        var r = vec2<f32>(body, mat);
        r = cc_streetfurniture_u(r, vec2<f32>(
            cc_streetfurniture_rbox(b - vec3<f32>(0.0, 0.03, 1.04),
                                    vec3<f32>(0.11, 0.10, 0.045), 0.015),
            cc_streetfurniture_M_NEWS0 + f32(i)));
        var legs = cc_streetfurniture_rbox(
            vec3<f32>(abs(b.x) - 0.20, abs(b.y) - 0.165, b.z - 0.07),
            vec3<f32>(0.022, 0.022, 0.10), 0.005);
        if (lod <= 1) {
            legs = min(legs, cc_streetfurniture_rbox(
                b - vec3<f32>(0.0, -0.228, 0.50),
                vec3<f32>(0.09, 0.012, 0.012), 0.006));
            legs = min(legs, cc_streetfurniture_rbox(
                b - vec3<f32>(0.05, -0.085, 1.085),
                vec3<f32>(0.03, 0.014, 0.018), 0.004));
        }
        r = cc_streetfurniture_u(r, vec2<f32>(legs, cc_streetfurniture_M_IRON));
        best = cc_streetfurniture_u(best, r);
    }
    return best;
}

// A bench (avenues): cast-iron ends with arms, four seat slats and three
// back slats raked 12 degrees, the gaps real.
fn cc_streetfurniture_bench(p: vec3<f32>, v: cc_streetfurniture_Prop)
        -> vec2<f32> {
    let fs = select(-1.0, 1.0, v.r.z > 0.5);   // which way it faces
    let y = p.y * fs;                          // back at +y
    let e = vec3<f32>(abs(p.x) - 0.82, y, p.z);
    var iron = cc_streetfurniture_rbox(e - vec3<f32>(0.0, -0.20, 0.21),
                                       vec3<f32>(0.024, 0.028, 0.23), 0.008);
    iron = min(iron, cc_streetfurniture_seg(e, vec3<f32>(0.0, 0.17, -0.02),
                                            vec3<f32>(0.0, 0.29, 0.86), 0.026));
    iron = min(iron, cc_streetfurniture_rbox(e - vec3<f32>(0.0, 0.0, 0.42),
                                             vec3<f32>(0.024, 0.23, 0.024),
                                             0.006));
    iron = min(iron, cc_streetfurniture_seg(e, vec3<f32>(0.0, -0.24, 0.64),
                                            vec3<f32>(0.0, 0.22, 0.67), 0.022));
    iron = min(iron, cc_streetfurniture_seg(e, vec3<f32>(0.0, -0.21, 0.43),
                                            vec3<f32>(0.0, -0.24, 0.64), 0.020));
    let si = clamp(round((y + 0.20) / 0.11), 0.0, 3.0);
    var wood = cc_streetfurniture_rbox(
        vec3<f32>(p.x, y + 0.20 - si * 0.11, p.z - 0.455),
        vec3<f32>(0.95, 0.046, 0.017), 0.007);
    let bz = p.z - 0.50;
    let by = (y - 0.215 - 0.2126 * bz) * 0.978;
    let bi = clamp(round((bz - 0.10) / 0.12), 0.0, 2.0);
    wood = min(wood, cc_streetfurniture_rbox(
        vec3<f32>(p.x, by, bz - 0.10 - bi * 0.12),
        vec3<f32>(0.95, 0.016, 0.044), 0.007));
    return cc_streetfurniture_u(vec2<f32>(iron, cc_streetfurniture_M_IRON),
                                vec2<f32>(wood, cc_streetfurniture_M_WOOD));
}

// One dumpster lid in its hinge frame: the point relative to the hinge line,
// turned back through the lid's opening angle about x.
fn cc_streetfurniture_lid_frame(p: vec3<f32>, cx: f32, ang: f32) -> vec3<f32> {
    let q = p - vec3<f32>(cx, -0.50, 1.20);
    let r = cc_streetfurniture_rot(q.yz, -ang);
    return vec3<f32>(q.x, r.x, r.y);
}

// The lid's opening angle, per half. Fully open lids lean against the wall
// (the hinge is 6 cm off it, so past ~92 degrees they would pass into the
// building); propped ones rest on the garbage; closed ones are closed. Never
// both closed — a shut dumpster is a green box.
fn cc_streetfurniture_lid_angle(v: cc_streetfurniture_Prop, side: i32) -> f32 {
    let d = fract(select(v.r.z * 3.7, v.r.z * 7.9 + v.r.y, side == 1));
    let open = 1.40 + 0.18 * fract(d * 11.3);
    let prop = 0.30 + 0.30 * fract(d * 5.1);
    let other_closed = fract(select(v.r.z * 7.9 + v.r.y, v.r.z * 3.7,
                                    side == 1)) >= 0.80;
    if (d < 0.45) {
        return open;
    }
    if (d < 0.80 || other_closed) {
        return prop;
    }
    return 0.0;
}

// A front-load dumpster, 1.8 x 1.05 x 1.2 m: the sloped lower front, the top
// lip, the fork pockets on the ends (hollow — you see into them), four
// casters, two HDPE lids on hinge knuckles along the back edge, and the
// garbage heaped inside up under the propped lids.
fn cc_streetfurniture_dumpster(p: vec3<f32>, v: cc_streetfurniture_Prop,
                               lod: i32) -> vec2<f32> {
    let q = vec3<f32>(cc_streetfurniture_rot(p.xy, (v.r.y - 0.5) * 0.07), p.z);
    let bot = 0.16;
    let top = 1.18;
    var body = cc_streetfurniture_rbox(q - vec3<f32>(0.0, 0.0, 0.5 * (bot + top)),
                                       vec3<f32>(0.90, 0.52, 0.5 * (top - bot)),
                                       0.025);
    // The slope: from the base 0.32 m in from the front up to z ~0.82.
    let slope = dot(vec2<f32>(q.y - 0.20, q.z - bot), vec2<f32>(0.90, -0.436));
    body = max(body, slope);
    let inner = max(cc_streetfurniture_rbox(q - vec3<f32>(0.0, 0.0, 1.20),
                                            vec3<f32>(0.875, 0.495, 1.0), 0.01),
                    slope + 0.025);
    body = max(body, -inner);
    let lip = max(cc_streetfurniture_rbox(q - vec3<f32>(0.0, 0.0, top - 0.02),
                                          vec3<f32>(0.915, 0.535, 0.028), 0.01),
                  -cc_streetfurniture_rbox(q - vec3<f32>(0.0, 0.0, top),
                                           vec3<f32>(0.875, 0.495, 0.2), 0.0));
    var res = vec2<f32>(min(body, lip),
                        select(cc_streetfurniture_M_PAINT,
                               cc_streetfurniture_M_INSIDE, inner < 0.0));
    // Fork pockets: a hollow steel sleeve along each end.
    let fq = vec3<f32>(abs(q.x) - 0.957, q.y + 0.02, q.z - 0.74);
    let pocket = max(cc_streetfurniture_rbox(fq, vec3<f32>(0.055, 0.40, 0.075),
                                             0.008),
                     -cc_streetfurniture_rbox(fq, vec3<f32>(0.040, 0.45, 0.058),
                                              0.0));
    res = cc_streetfurniture_u(res, vec2<f32>(pocket, cc_streetfurniture_M_PAINT));
    // Casters: a swivel plate, a fork, a wheel.
    if (lod <= 1) {
        let cy = select(-0.38, 0.13, q.y > -0.13);
        let wq = vec3<f32>(abs(q.x) - 0.76, q.y - cy, q.z);
        let plate = min(
            cc_streetfurniture_rbox(wq - vec3<f32>(0.0, 0.0, 0.145),
                                    vec3<f32>(0.045, 0.045, 0.010), 0.004),
            cc_streetfurniture_rbox(vec3<f32>(abs(wq.x) - 0.026, wq.y,
                                              wq.z - 0.10),
                                    vec3<f32>(0.005, 0.03, 0.045), 0.003));
        res = cc_streetfurniture_u(res, vec2<f32>(plate,
                                                  cc_streetfurniture_M_METAL));
        let wheel = max(length(wq.yz - vec2<f32>(0.0, 0.065)) - 0.065,
                        abs(wq.x) - 0.018);
        res = cc_streetfurniture_u(res, vec2<f32>(wheel,
                                                  cc_streetfurniture_M_RUBBER));
    } else {
        res = cc_streetfurniture_u(res, vec2<f32>(
            cc_streetfurniture_rbox(vec3<f32>(abs(q.x) - 0.76, q.y + 0.12,
                                              q.z - 0.08),
                                    vec3<f32>(0.04, 0.30, 0.08), 0.0),
            cc_streetfurniture_M_RUBBER));
    }
    // The lids.
    let la = cc_streetfurniture_lid_angle(v, 0);
    let lb = cc_streetfurniture_lid_angle(v, 1);
    let hx = select(-0.455, 0.455, q.x > 0.0);
    let lq = cc_streetfurniture_lid_frame(q, hx, select(lb, la, q.x > 0.0));
    var lid = cc_streetfurniture_rbox(lq - vec3<f32>(0.0, 0.535, 0.02),
                                      vec3<f32>(0.44, 0.535, 0.018), 0.012);
    lid = min(lid, cc_streetfurniture_rbox(lq - vec3<f32>(0.0, 1.058, -0.015),
                                           vec3<f32>(0.44, 0.014, 0.045), 0.008));
    res = cc_streetfurniture_u(res, vec2<f32>(lid, cc_streetfurniture_M_LID));
    if (lod <= 1) {
        // Hinge knuckles, two per lid.
        let kx = abs(abs(q.x) - 0.455) - 0.29;
        let kn = cc_streetfurniture_cylz(
            vec3<f32>(q.y + 0.50, q.z - 1.20, kx), 0.022, 0.045);
        res = cc_streetfurniture_u(res, vec2<f32>(kn, cc_streetfurniture_M_METAL));
    }
    // Garbage heaped in the box, kept under the lid on each half.
    var g = cc_streetfurniture_heap(q - vec3<f32>(0.0, 0.0, top),
                                    vec2<f32>(0.875, 0.495), 0.02, 4, v.seed,
                                    lod);
    g.x = max(g.x, lq.z);
    return cc_streetfurniture_u(res, g);
}

// Wheelie bins (240 l), one or two side by side, handles to the street and
// lids thrown back over them, hanging down the handle side — how bins are
// left on collection night — with bags standing out of the open tops and,
// in half of them, one more split open on the pavement beside.
fn cc_streetfurniture_wheelie(p: vec3<f32>, v: cc_streetfurniture_Prop,
                              lod: i32) -> vec2<f32> {
    let half = 0.5 * f32(v.n - 1);
    let i = clamp(round(p.x / 0.66 + half), 0.0, f32(v.n - 1));
    // The body sits 0.24 m wallward of the local origin, so the box that
    // holds it and the lid hanging out over the handle is symmetric.
    let q = vec3<f32>(p.x - (i - half) * 0.66 + 0.10 * f32(v.n - 1)
                          * select(0.0, 1.0, v.r.w > 0.5),
                      p.y + 0.24, p.z);
    // Tapered body, wider at the top.
    let t = clamp(q.z / 0.97, 0.0, 1.0);
    let hx = vec2<f32>(0.245 + 0.045 * t, 0.30 + 0.06 * t);
    var body = 0.96 * cc_streetfurniture_rbox(
        vec3<f32>(q.x * 0.29 / hx.x, q.y * 0.36 / hx.y, q.z - 0.51),
        vec3<f32>(0.29, 0.36, 0.46), 0.04);
    let inner = cc_streetfurniture_rbox(
        vec3<f32>(q.x * 0.29 / hx.x, q.y * 0.36 / hx.y, q.z - 1.10),
        vec3<f32>(0.275, 0.345, 0.55), 0.03);
    body = max(body, -inner);
    var res = vec2<f32>(body, select(cc_streetfurniture_M_PAINT,
                                     cc_streetfurniture_M_INSIDE, inner < 0.0));
    let lip = max(cc_streetfurniture_rbox(q - vec3<f32>(0.0, 0.0, 0.955),
                                          vec3<f32>(0.30, 0.375, 0.022), 0.01),
                  -cc_streetfurniture_rbox(q - vec3<f32>(0.0, 0.0, 1.0),
                                           vec3<f32>(0.27, 0.34, 0.2), 0.0));
    res = cc_streetfurniture_u(res, vec2<f32>(lip, cc_streetfurniture_M_PAINT));
    // Lid: hinged at the handle, thrown back over it through ~240 degrees
    // (the closed lid points wallward from the hinge; this turns its free
    // edge up, over, and down the street side, 27-33 degrees off vertical).
    let hq = q - vec3<f32>(0.0, 0.385, 0.985);
    let ang = 2.03 + 0.12 * v.r.z;
    let lr = cc_streetfurniture_rot(hq.yz, -ang);
    let lq = vec3<f32>(hq.x, lr.x, lr.y);
    var lid = cc_streetfurniture_rbox(lq - vec3<f32>(0.0, -0.39, 0.016),
                                      vec3<f32>(0.305, 0.39, 0.016), 0.01);
    lid = min(lid, cc_streetfurniture_rbox(lq - vec3<f32>(0.0, -0.765, -0.02),
                                           vec3<f32>(0.305, 0.014, 0.04), 0.008));
    res = cc_streetfurniture_u(res, vec2<f32>(lid, cc_streetfurniture_M_PAINT));
    // Handle bar and wheels.
    var hw = cc_streetfurniture_seg(q, vec3<f32>(-0.20, 0.425, 0.93),
                                    vec3<f32>(0.20, 0.425, 0.93), 0.018);
    let wq = vec3<f32>(abs(q.x) - 0.25, q.y - 0.30, q.z - 0.10);
    res = cc_streetfurniture_u(res, vec2<f32>(hw, cc_streetfurniture_M_PAINT));
    res = cc_streetfurniture_u(res, vec2<f32>(
        max(length(wq.yz) - 0.10, abs(wq.x) - 0.03),
        cc_streetfurniture_M_RUBBER));
    if (lod <= 1) {
        res = cc_streetfurniture_u(res, vec2<f32>(
            cc_streetfurniture_seg(q, vec3<f32>(-0.28, 0.30, 0.10),
                                   vec3<f32>(0.28, 0.30, 0.10), 0.012),
            cc_streetfurniture_M_METAL));
    }
    res = cc_streetfurniture_u(res, cc_streetfurniture_heap(
        q - vec3<f32>(0.0, 0.0, 0.97), vec2<f32>(0.27, 0.34), 0.04, 2,
        v.seed + u32(i), lod));
    if (v.r.y > 0.5) {
        // A bag on the pavement beside the end bin.
        let bq = p - vec3<f32>(0.33 * f32(v.n) + 0.12, -0.12, 0.23);
        let bh = cc_streetfurniture_item(v.seed, 9);
        res = cc_streetfurniture_u(res, vec2<f32>(
            cc_streetfurniture_bag(bq, vec3<f32>(0.24, 0.21, 0.26),
                                   bh.x * 6.28, lod),
            cc_streetfurniture_bag_mat(bh.y)));
    }
    return res;
}

// THE field: one function for every type, so the march, the four gradient
// taps and the shader's material lookup all call the same code once.
fn cc_streetfurniture_sdf(p: vec3<f32>, v: cc_streetfurniture_Prop, lod: i32)
        -> vec2<f32> {
    let t = v.typ;
    if (t == cc_streetfurniture_T_BAGS) {
        return cc_streetfurniture_pile(p, v, lod);
    }
    if (t == cc_streetfurniture_T_HYDRANT) {
        return cc_streetfurniture_hydrant(p, v, lod);
    }
    if (t == cc_streetfurniture_T_METER) {
        return cc_streetfurniture_meter(p, v, lod);
    }
    if (t == cc_streetfurniture_T_BOLLARD) {
        return cc_streetfurniture_bollard(p, v);
    }
    if (t == cc_streetfurniture_T_BASKET || t == cc_streetfurniture_T_CAN) {
        return cc_streetfurniture_basket(p, v, lod);
    }
    if (t == cc_streetfurniture_T_NEWS) {
        return cc_streetfurniture_news(p, v, lod);
    }
    if (t == cc_streetfurniture_T_BENCH) {
        return cc_streetfurniture_bench(p, v);
    }
    if (t == cc_streetfurniture_T_DUMPSTER) {
        return cc_streetfurniture_dumpster(p, v, lod);
    }
    return cc_streetfurniture_wheelie(p, v, lod);
}

fn cc_streetfurniture_lod(fpx: f32) -> i32 {
    return select(select(2, 1, fpx < cc_streetfurniture_MID_FP), 0,
                  fpx < cc_streetfurniture_FINE_FP);
}

// ---------------------------------------------------------------------------
// The trace
// ---------------------------------------------------------------------------

fn cc_streetfurniture_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 0, ci);
}

// The footprint the props are resolved at: the core's (which carries the
// app's LOD slider) sharpened toward the true pixel, never below ~1.5 of
// them. streetlife's cc_streetlife_fp_px, for streetlife's reason — a hinge
// is one feature, not a lattice that could moire.
fn cc_streetfurniture_fpx(t: f32) -> f32 {
    let pa = 2.0 * u.cam_origin.w / max(u.params.x, 1.0);
    return max(0.35 * max(pa, u.periodic.z), pa) * max(t, 0.0);
}

// A candidate: a box the ray entered, named by where it lives, so holding
// one costs five registers across the slot walk and not a whole Prop.
struct cc_streetfurniture_Cand {
    side: i32,
    lane: i32,
    k: i32,
    ta: f32,
    tb: f32,
}

fn cc_streetfurniture_trace(o: vec3<f32>, dir: vec3<f32>, ta: f32, tb: f32,
                            v: cc_streetfurniture_Prop, kind: i32,
                            ci: vec2<i32>) -> CityHit {
    let fpx = cc_streetfurniture_fpx(ta);
    let lod = cc_streetfurniture_lod(fpx);
    let eps = max(0.0012, 0.30 * fpx);
    let iters = select(cc_streetfurniture_ITERS, cc_streetfurniture_ITERS_FAR,
                       fpx > cc_streetfurniture_ITER_FP);
    // The ray into the prop's frame once: a signed permutation, so t stays
    // world distance.
    let ro = cc_streetfurniture_to_local(v, o);
    let rd = select(vec3<f32>(dir.y, v.out * dir.x, dir.z),
                    vec3<f32>(dir.x, v.out * dir.y, dir.z), v.ax == 0);
    var t = ta;
    var got = false;
    for (var i: i32 = 0; i < iters; i = i + 1) {
        let d = cc_streetfurniture_sdf(ro + t * rd, v, lod).x;
        if (d < eps) {
            got = true;
            break;
        }
        t = t + d * cc_streetfurniture_STEP;
        if (t > tb) {
            break;
        }
    }
    if (!got || t > tb) {
        return cc_streetfurniture_miss(ci);
    }
    // Normal: the four tetrahedron taps as a LOOP (aircars' measured
    // lesson — four inline copies of the field cost occupancy everywhere).
    let q = ro + t * rd;
    let hs = max(eps, 0.0015);
    var g = vec3<f32>(0.0);
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let s = 2.0 * vec3<f32>(f32((0x9u >> k) & 1u), f32((0xau >> k) & 1u),
                                f32((0xcu >> k) & 1u)) - 1.0;
        g = g + s * cc_streetfurniture_sdf(q + s * hs, v, lod).x;
    }
    return CityHit(true, t, o + t * dir,
                   cc_streetfurniture_to_world(v, normalize(g)), kind, ci);
}

fn cc_streetfurniture_props_trace(o: vec3<f32>, dir: vec3<f32>,
                                  inv_dir: vec3<f32>, t0: f32, t1: f32,
                                  ci: vec2<i32>, cc: CityCell) -> CityHit {
    // Gate 1, no hash and no memory: the segment against the tallest thing
    // here. Every pixel looking at a facade, a roof, a cloud or the sky
    // leaves on this line.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_streetfurniture_Z_GATE || max(za, zb) < -0.5
        || !city_has_walk(cc)) {
        return cc_streetfurniture_miss(ci);
    }
    // Gate 2: past the wall lane's far edge nothing here survives.
    let fpx = cc_streetfurniture_fpx(t0);
    if (fpx > cc_streetfurniture_WALL_FAR.y) {
        return cc_streetfurniture_miss(ci);
    }
    var t_end = t1;
    var c0: cc_streetfurniture_Cand;
    var c1: cc_streetfurniture_Cand;
    var nc = 0;
    for (var side: i32 = 0; side < 4; side = side + 1) {
        let e = cc_streetfurniture_edge(ci, cc, side);
        if (!e.ok) {
            continue;
        }
        for (var lane: i32 = 0; lane < 2; lane = lane + 1) {
            let kerb = lane == 0;
            if (kerb && fpx > cc_streetfurniture_KERB_FAR.y) {
                continue;
            }
            // The lane's slab: its across band, the usable along range, and
            // the height of the tallest thing it can hold.
            let u_lo = select(0.0, e.w - 1.45, kerb);
            let u_hi = select(1.30, e.w - 0.02, kerb);
            let c_a = e.bl + e.out * u_lo;
            let c_b = e.bl + e.out * u_hi;
            let ztop = cc_streetfurniture_FLOOR + select(
                cc_streetfurniture_WALL_TOP, cc_streetfurniture_KERB_TOP, kerb);
            let lo2 = vec2<f32>(e.a_lo, min(c_a, c_b));
            let hi2 = vec2<f32>(e.a_hi, max(c_a, c_b));
            let bmin = vec3<f32>(select(lo2.yx, lo2, e.ax == 0), 0.0);
            let bmax = vec3<f32>(select(hi2.yx, hi2, e.ax == 0), ztop);
            let sb = city_box_hit(o, inv_dir, bmin, bmax);
            let la = max(sb.x, t0);
            let lb = min(sb.y, t_end);
            if (sb.x > sb.y || lb <= la) {
                continue;
            }
            let pitch = select(cc_streetfurniture_WALL_PITCH,
                               cc_streetfurniture_KERB_PITCH, kerb);
            let pa = o + la * dir;
            let pb = o + lb * dir;
            let sa = select(pa.y, pa.x, e.ax == 0);
            let sc = select(pb.y, pb.x, e.ax == 0);
            let k0 = i32(floor((min(sa, sc) - e.a_cell) / pitch));
            let k1 = i32(floor((max(sa, sc) - e.a_cell) / pitch));
            let fwd = sc >= sa;
            let count = min(k1 - k0 + 1, cc_streetfurniture_MAX_SLOTS);
            var found = 0;
            // Near end first: the first box entered is the nearest one.
            for (var i: i32 = 0; i < count; i = i + 1) {
                let k = select(k1 - i, k0 + i, fwd);
                let v = cc_streetfurniture_prop(ci, cc, e, side, lane, k);
                if (!v.ok) {
                    continue;
                }
                let bx = cc_streetfurniture_box(v);
                let hb = city_box_hit(o, inv_dir, bx.lo, bx.hi);
                let ta = max(hb.x, la);
                let tb = min(hb.y, t_end);
                if (hb.x > hb.y || tb <= ta) {
                    continue;
                }
                let c = cc_streetfurniture_Cand(side, lane, k, ta, tb);
                // Keep the two nearest boxes of the cell. Boxes are disjoint,
                // so their intervals on the ray are too, and entry order is
                // depth order.
                if (nc == 0) {
                    c0 = c;
                } else if (ta < c0.ta) {
                    c1 = c0;
                    c0 = c;
                } else if (nc == 1 || ta < c1.ta) {
                    c1 = c;
                }
                nc = min(nc + 1, 2);
                if (nc == 2) {
                    t_end = min(t_end, c1.tb);
                }
                found = found + 1;
                if (found == 2) {
                    break;
                }
            }
        }
    }
    // The sphere trace, ONE call site: the nearest candidate first, and the
    // second only if the ray grazed past the first — without it, a ray
    // through the empty corner of a hydrant's box would lose the bench
    // behind it, and a ghost of the hydrant's box would cut the bench.
    for (var j: i32 = 0; j < nc; j = j + 1) {
        var c = c0;
        if (j == 1) {
            c = c1;
        }
        let e = cc_streetfurniture_edge(ci, cc, c.side);
        let v = cc_streetfurniture_prop(ci, cc, e, c.side, c.lane, c.k);
        let h = cc_streetfurniture_trace(o, dir, c.ta, c.tb, v,
                                         800 + 2 * c.side + c.lane, ci);
        if (h.hit) {
            return h;
        }
    }
    return cc_streetfurniture_miss(ci);
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

struct cc_streetfurniture_Mat {
    alb: vec3<f32>,   // diffuse, relative to dry asphalt
    f0: f32,          // specular at normal incidence
    shine: f32,       // lobe exponent
    metal: bool,      // specular tinted by alb
}

fn cc_streetfurniture_mk(alb: vec3<f32>, f0: f32, shine: f32)
        -> cc_streetfurniture_Mat {
    return cc_streetfurniture_Mat(alb, f0, shine, false);
}

// Paint by type, from the prop's own draw. The palettes are what each thing
// is actually painted in a North American city, and the sodium light sorts
// them for us: red hydrants and yellow boxes stay colours, green bins and
// blue dumpsters go nearly black.
fn cc_streetfurniture_paint(v: cc_streetfurniture_Prop) -> vec3<f32> {
    let d = fract(v.r.w * 7.31 + v.r.z * 0.37);
    let t = v.typ;
    if (t == cc_streetfurniture_T_HYDRANT) {
        if (d < 0.35) { return vec3<f32>(0.55, 0.07, 0.05); }
        if (d < 0.55) { return vec3<f32>(0.85, 0.62, 0.10); }
        if (d < 0.85) { return vec3<f32>(0.42, 0.44, 0.46); }
        return vec3<f32>(0.08, 0.28, 0.12);
    }
    if (t == cc_streetfurniture_T_METER) {
        if (d < 0.45) { return vec3<f32>(0.07, 0.08, 0.10); }
        return vec3<f32>(0.035, 0.035, 0.04);
    }
    if (t == cc_streetfurniture_T_BOLLARD) {
        if (d < 0.45) { return vec3<f32>(0.05, 0.05, 0.055); }
        if (d < 0.75) { return vec3<f32>(0.85, 0.62, 0.10); }
        return vec3<f32>(0.30, 0.31, 0.32);
    }
    if (t == cc_streetfurniture_T_BASKET || t == cc_streetfurniture_T_CAN) {
        if (d < 0.60) { return vec3<f32>(0.07, 0.20, 0.10); }
        if (d < 0.85) { return vec3<f32>(0.05, 0.05, 0.055); }
        return vec3<f32>(0.14, 0.17, 0.22);
    }
    if (t == cc_streetfurniture_T_DUMPSTER) {
        if (d < 0.35) { return vec3<f32>(0.08, 0.20, 0.10); }
        if (d < 0.55) { return vec3<f32>(0.06, 0.12, 0.30); }
        if (d < 0.72) { return vec3<f32>(0.28, 0.12, 0.06); }
        if (d < 0.88) { return vec3<f32>(0.20, 0.21, 0.22); }
        return vec3<f32>(0.05, 0.05, 0.055);
    }
    // Wheelie bins.
    if (d < 0.35) { return vec3<f32>(0.05, 0.22, 0.08); }
    if (d < 0.60) { return vec3<f32>(0.20, 0.21, 0.22); }
    if (d < 0.78) { return vec3<f32>(0.05, 0.05, 0.055); }
    if (d < 0.90) { return vec3<f32>(0.05, 0.12, 0.40); }
    return vec3<f32>(0.20, 0.10, 0.05);
}

fn cc_streetfurniture_news_paint(d: f32) -> vec3<f32> {
    if (d < 0.22) { return vec3<f32>(0.90, 0.70, 0.08); }
    if (d < 0.40) { return vec3<f32>(0.08, 0.22, 0.65); }
    if (d < 0.55) { return vec3<f32>(0.65, 0.06, 0.05); }
    if (d < 0.68) { return vec3<f32>(0.08, 0.40, 0.14); }
    if (d < 0.82) { return vec3<f32>(0.85, 0.85, 0.83); }
    if (d < 0.92) { return vec3<f32>(0.05, 0.05, 0.055); }
    return vec3<f32>(0.90, 0.40, 0.06);
}

fn cc_streetfurniture_material(m: f32, v: cc_streetfurniture_Prop)
        -> cc_streetfurniture_Mat {
    let mi = i32(round(m));
    if (mi >= 30) {
        let h = cc_streetfurniture_item(v.seed, mi - 30);
        return cc_streetfurniture_mk(cc_streetfurniture_news_paint(h.w), 0.04,
                                     110.0);
    }
    switch (mi) {
        case 1: {
            // Black polyethylene: almost no diffuse, and the tightest lobe on
            // the street.
            return cc_streetfurniture_mk(vec3<f32>(0.028, 0.029, 0.032), 0.05,
                                         420.0);
        }
        case 2: {
            // Clear and white bags, the contents dimly showing through.
            return cc_streetfurniture_mk(vec3<f32>(0.52, 0.60, 0.64), 0.045,
                                         260.0);
        }
        case 3: {
            return cc_streetfurniture_mk(vec3<f32>(0.04, 0.09, 0.30), 0.05,
                                         320.0);
        }
        case 4: {
            // Cardboard, rained on: darker than dry and completely matte.
            return cc_streetfurniture_mk(vec3<f32>(0.40, 0.26, 0.14), 0.02,
                                         8.0);
        }
        case 5: {
            // Beer-bottle brown; the green of a wine bottle, at sodium, is
            // the same near-black, and bright green read as a leaf.
            return cc_streetfurniture_mk(vec3<f32>(0.16, 0.08, 0.03), 0.05,
                                         700.0);
        }
        case 6: {
            return cc_streetfurniture_Mat(vec3<f32>(0.75, 0.75, 0.80), 0.55,
                                          90.0, true);
        }
        case 7: {
            return cc_streetfurniture_mk(vec3<f32>(0.02), 0.03, 18.0);
        }
        case 8: {
            return cc_streetfurniture_Mat(vec3<f32>(0.18, 0.18, 0.19), 0.30,
                                          55.0, true);
        }
        case 10: {
            // HDPE lids: black, textured, a broad dull sheen.
            return cc_streetfurniture_mk(vec3<f32>(0.035, 0.037, 0.040), 0.04,
                                         35.0);
        }
        case 11: {
            return cc_streetfurniture_mk(vec3<f32>(0.03), 0.045, 900.0);
        }
        case 12: {
            // Inside a container: the same steel, in its own shadow.
            return cc_streetfurniture_mk(0.22 * cc_streetfurniture_paint(v),
                                         0.02, 20.0);
        }
        case 13: {
            // Hydrant caps and nut: a second colour against the body.
            let d = fract(v.r.y * 5.77);
            let c = select(select(vec3<f32>(0.40, 0.42, 0.44),
                                  vec3<f32>(0.55, 0.07, 0.05), d < 0.6),
                           vec3<f32>(0.05, 0.05, 0.055), d < 0.25);
            let body = cc_streetfurniture_paint(v);
            return cc_streetfurniture_mk(
                select(c, body, v.typ == cc_streetfurniture_T_BOLLARD), 0.04,
                110.0);
        }
        case 14: {
            return cc_streetfurniture_mk(vec3<f32>(0.34, 0.20, 0.10), 0.03,
                                         30.0);
        }
        case 15: {
            return cc_streetfurniture_mk(vec3<f32>(0.035, 0.038, 0.037), 0.04,
                                         60.0);
        }
        default: {
            return cc_streetfurniture_mk(cc_streetfurniture_paint(v), 0.04,
                                         110.0);
        }
    }
}

// The lamp heads' glints: the eight heads city_street_light sums, as point
// sources seen in a lobe of exponent `shine` about the reflected ray — and
// only on the lamp lines city_lamp_lines lights, so a glint never shows a
// lamp that the pools and city_street_light say is not there. The
// lobe is normalised ((n + 2) / 2 pi), so tightening it concentrates the same
// energy into a smaller, brighter spark rather than dimming it — which is
// what black plastic does: the highlight is small and it clips.
fn cc_streetfurniture_glints(p: vec3<f32>, n: vec3<f32>, rf: vec3<f32>,
                             shine: f32) -> f32 {
    let cellm = u.ocean_params.x;
    let h2 = CITY_LAMP_HEAD_Z * CITY_LAMP_HEAD_Z;
    let norm = (shine + 2.0) * 0.15915494;
    let lit = city_lamp_lines(p.xy);
    var e = 0.0;
    for (var dirn: i32 = 0; dirn < 2; dirn = dirn + 1) {
        let along_y = dirn == 0;
        if (!select(lit.y, lit.x, along_y)) {
            continue;
        }
        let c = select(p.y, p.x, along_y);
        let a = select(p.x, p.y, along_y);
        let k = i32(round(c / cellm));
        let amp = select(1.0, 1.6, city_is_avenue(k));
        let inset = city_lamp_inset(k);
        let lat = city_lamp_lattice(k);
        let j0 = floor((a - lat.y) / lat.x);
        for (var sd: i32 = 0; sd < 2; sd = sd + 1) {
            let sg = select(-1.0, 1.0, sd == 1);
            let hc = f32(k) * cellm + sg * (inset - CITY_LAMP_ARM);
            for (var jj: i32 = 0; jj < 2; jj = jj + 1) {
                let ha = lat.y + (j0 + f32(jj)) * lat.x;
                let head = select(vec3<f32>(ha, hc, CITY_LAMP_HEAD_Z),
                                  vec3<f32>(hc, ha, CITY_LAMP_HEAD_Z), along_y);
                let lv = head - p;
                let r2 = max(dot(lv, lv), 0.25);
                let l = lv * inverseSqrt(r2);
                let lobe = pow(max(dot(rf, l), 0.0), shine) * norm;
                e = e + amp * h2 / r2 * max(l.z, 0.0) * lobe
                      * step(0.0, dot(n, l));
            }
        }
    }
    return city_lamp_scale(p.xy) * e;
}

fn cc_streetfurniture_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let local = (h.kind - 800) % 8;
    let side = local / 2;
    let lane = local % 2;
    let e = cc_streetfurniture_edge(h.cell, cc, side);
    let pitch = select(cc_streetfurniture_WALL_PITCH,
                       cc_streetfurniture_KERB_PITCH, lane == 0);
    let along = select(h.pos.y, h.pos.x, e.ax == 0);
    let k = i32(floor((along - e.a_cell) / pitch));
    let v = cc_streetfurniture_prop(h.cell, cc, e, side, lane, k);
    if (!v.ok) {
        // The slot the hit came from must re-derive; if it does not, the
        // placement and the trace disagree, and that should be seen.
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    let fpx = max(0.35 * fp, 2.0 * u.cam_origin.w / max(u.params.x, 1.0)
                             * max(h.t, 0.0));
    let pl = cc_streetfurniture_to_local(v, h.pos);
    let n = h.normal;
    let mid = cc_streetfurniture_sdf(pl, v, cc_streetfurniture_lod(fpx)).y;
    let mat = cc_streetfurniture_material(mid, v);
    let rf = reflect(dir, n);
    let cosv = clamp(-dot(dir, n), 0.0, 1.0);
    let schlick = pow(1.0 - cosv, 5.0);

    // Diffuse: the sodium heads, the lit pavement from below, the store
    // window behind, and the night fill.
    let lamp = city_street_light(h.pos, n);
    let floor_l = city_street_light(vec3<f32>(h.pos.xy, 0.0),
                                    vec3<f32>(0.0, 0.0, 1.0))
                * cc_streetfurniture_WALK_WET;
    var light = lamp + floor_l * (cc_streetfurniture_BOUNCE
                                  * (0.5 - 0.5 * n.z));
    let store = city_storefront_glow(cc);
    let bq = max(max(cc.b1min.xy - h.pos.xy, h.pos.xy - cc.b1max.xy),
                 vec2<f32>(0.0));
    let bdist = length(bq);
    let to_bldg = -bq / max(bdist, 1e-3);
    let store_w = exp(-bdist / 1.8)
                * (1.0 - smoothstep(CITY_STORE_H - 0.5, CITY_STORE_H + 1.0,
                                    h.pos.z));
    light = light + store * (0.05 * store_w
                             * (0.3 + 0.7 * max(dot(n.xy, to_bldg), 0.0)));
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(h.pos.xy, 3.0))
             + CITY_MOONLIGHT * max(u.sun_dir.z, 0.0);
    light = light + fill * (1.5 + n.z);
    var alb = mat.alb;
    if (v.typ == cc_streetfurniture_T_DUMPSTER
        && round(mid) == cc_streetfurniture_M_PAINT) {
        // A dumpster's skin is stiffened by pressed ribs, vertical on the
        // front and ends, and it is grimiest where the truck's forks and the
        // pavement's splash reach. Bands, filtered to their mean past a pixel.
        let nl = select(vec3<f32>(n.y, v.out * n.x, n.z),
                        vec3<f32>(n.x, v.out * n.y, n.z), v.ax == 0);
        let run = select(pl.y, pl.x, abs(nl.y) > abs(nl.x));
        let rib = city_stripes(run + 2.0, 0.0, 0.05, 0.45, fpx);
        let grime = 1.0 - 0.45 * (1.0 - smoothstep(0.15, 0.75, pl.z));
        alb = alb * (1.0 - 0.45 * rib * step(abs(nl.z), 0.5)) * grime;
    }
    let diffuse = (cc_streetfurniture_ALB_GAIN * alb) * light;

    // Wet: a water film over everything, glossier than any paint, and the
    // material's own lobe under it.
    let f_mat = mat.f0 + (1.0 - mat.f0) * schlick;
    let f_wat = cc_streetfurniture_WATER_F0
              + (1.0 - cc_streetfurniture_WATER_F0) * schlick;
    let tint = select(vec3<f32>(1.0), mat.alb / max(mat.alb.g, 1e-3),
                      mat.metal);
    var spec = CITY_LAMP_COLOR * tint
             * (f_mat * cc_streetfurniture_glints(h.pos, n, rf, mat.shine));
    spec = spec + CITY_LAMP_COLOR
         * (f_wat * cc_streetfurniture_glints(h.pos, n, rf,
                                              cc_streetfurniture_WATER_N));
    // What the reflected ray sees when it is not a lamp: the facades and the
    // skyglow at and above the horizon, the storefront if it faces one, and
    // the lit pavement only when it really looks DOWN at it. The first pass
    // blended the pavement in from the horizon up, which gave every flank a
    // floor of radiance 0.03 — display 0.3 — and no black bag could be
    // black.
    // The pavement a downward reflection sees is where that ray lands, not
    // the pool at the prop's own foot: a bag at the kerb reflects the road.
    let land = h.pos.xy + rf.xy * (h.pos.z / max(-rf.z, 0.05));
    let floor_r = city_street_light(vec3<f32>(land, 0.0),
                                    vec3<f32>(0.0, 0.0, 1.0))
                * cc_streetfurniture_FLOOR_REFL;
    let env = mix(floor_r,
                  fill + CITY_PALETTE_MEAN * (0.015 + 0.06 * cc.lit_frac),
                  smoothstep(-0.45, -0.05, rf.z))
            + store * (0.5 * store_w * max(dot(rf.xy, to_bldg), 0.0));
    // The film's own reflection of the scene is left to the lamp glints: rain
    // on a vertical face is a few beads, not a mirror, and carrying it here
    // too put every dark flank at mid-grey (a 20:1 pavement-to-paint ratio
    // is only 0.87 to 0.45 after this tone curve, so a few percent of
    // pavement is all a dark prop has room for).
    spec = spec + env * f_mat * tint;
    // Inside a container the rim hides the lamp heads: no glints there.
    var col = select(diffuse + spec, diffuse,
                     round(mid) == cc_streetfurniture_M_INSIDE);

    // Past the lane's far edge the prop has dissolved into the pavement
    // under it — faded into the ground's own radiance, not switched off.
    let far = select(cc_streetfurniture_WALL_FAR, cc_streetfurniture_KERB_FAR,
                     lane == 0);
    let fade = smoothstep(far.x, far.y, fpx);
    if (fade > 0.0) {
        col = mix(col, city_ground_radiance(vec3<f32>(h.pos.xy, 0.0), h.cell, cc,
                                            dir, fp),
                  fade);
    }
    return col;
}
