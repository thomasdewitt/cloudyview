// --- Facade shell ------------------------------------------------------------
//
// A facade is not a painted box face. Up close it is a SHELL: windows set
// into the wall with the glass pulled back behind a reveal, sills and ledges
// standing proud of it, frames and mullions, drainpipes on brackets, air
// conditioners hung under the sills — and behind every lit pane, a room.
//
// The shell lives in the frame of one vertical box face (CityFace):
//   u  along the wall, metres from the face's own start corner,
//   v  height (world z),
//   w  outward from the wall plane; w < 0 is inside the building.
// The face is divided into BAYS: an integer number of window cells between
// two corner margins, the building's pitch stretched to fit. A world-anchored
// lattice (what the flat facade used to be) puts a window across a corner,
// which a painted window survives and a recessed one does not — it opens a
// hole in the building. Floors are CITY_FLOOR_H and the tiers start on floor
// lines, so v needs no such care.
//
// Every shell element is a closed-form box (or, for pipes, circle) problem
// in face-local coordinates, clipped to the bay cell it belongs to, and the
// cells a ray crosses inside the shell layer are walked with a short 2D DDA.
// So the union of the per-cell pieces is exact and no feature needs to know
// about its neighbours.
//
// LOD, and why there is no seam. The shell is traced only where a pixel can
// resolve it. What stands proud of the wall morphs to zero depth over the
// plain footprint CITY_SHELL_FP_MORPH -> CITY_SHELL_FP_FLAT (city_shell_scale),
// and the shading blends into the flat facade's, read where the ray crosses
// the wall plane, over CITY_SHELL_FP_BLEND -> FLAT on the same footprint.
// The recess
// keeps its true depth at every footprint, and the flat facade sees its
// glass through the opening (city_recess_off), so at the hand-off the
// resolved shell IS the flat facade: same point, same glass, same
// function. The flat facade's octave ladder carries the far field from
// there. tests/test_soar_city_shell_continuity.py holds this.
//
// Light. At night a wall is seen by what lights it: the lit pane in its own
// reveal (the jambs and the sill catch the room's light — a rectangle-source
// form factor, closed form), the street lamps below (sodium from the lamp
// heads, falling off with height, so sill and ledge UNDERSIDES catch it and
// read as lines of light), and the city's own skyglow. The skyglow fill is
// held to the level the flat city was tuned at (CITY_SHELL_FILL_K times a
// real albedo gives the old CITY_FACADE_ALBEDO), so the far city keeps its
// look while materials read as brick or concrete where a lamp reaches them.

const CITY_KIND_SHELL: i32 = 1000;  // 1000 + part + 16 * (side + 4 * tier)
// The deepest any facade relief may stand off its wall (m); a style deeper
// than this is clamped to it. The flight collider (web/soar/citylots.js
// shellDepth) grows each building by its own style's depth, mirrored from
// city_shell_style, so the camera is never inside a feature — change a
// relief depth there and tests/test_city_lots_mirror.py says so.
const CITY_SHELL_OUT_MAX: f32 = 1.5;
const CITY_SHELL_FP_MORPH: f32 = 0.4;
const CITY_SHELL_FP_BLEND: f32 = 0.6;
const CITY_SHELL_FP_FLAT: f32 = 1.2;
const CITY_SHELL_CORNER: f32 = 0.9;   // plain wall kept at each corner (m)
const CITY_SHELL_MAX_CELLS: i32 = 8;
const CITY_SHELL_FRAME: f32 = 0.06;   // window frame, outside the pane (m)
const CITY_SHELL_ROOM_D: f32 = 4.2;   // room depth behind the glass (m)
// Real albedo times this is the old flat facade albedo under the fill.
const CITY_SHELL_FILL_K: f32 = 0.15;
const CITY_SHELL_EPS: f32 = 1e-4;

// The tolerance at distance t. The same crossing reached two ways (a
// division here, a multiply by the reciprocal in city_box_hit) differs by an
// ulp, and past 1024 m one float32 ulp (1.19e-7 t) is already more than
// CITY_SHELL_EPS; WGSL allows division 2.5 ulp besides, so ~4e-7 t in all.
// 1e-6 t covers that and stays far below a pixel at any field of view.
fn city_shell_eps(t: f32) -> f32 {
    return max(CITY_SHELL_EPS, 1e-6 * t);
}

const CITY_PART_WALL: i32 = 0;
const CITY_PART_REVEAL: i32 = 1;
const CITY_PART_GLASS: i32 = 2;
const CITY_PART_FRAME: i32 = 3;
const CITY_PART_SILL: i32 = 4;
const CITY_PART_TRIM: i32 = 5;   // ledges, cornices, fins, mullion caps
const CITY_PART_AC: i32 = 6;
const CITY_PART_PIPE: i32 = 7;
const CITY_PART_STEEL: i32 = 8;  // brackets, fire escapes, balcony bars
const CITY_PART_SLAB: i32 = 9;   // balcony slabs
const CITY_PART_PANEL: i32 = 10; // balcony glass panels
const CITY_PART_CANOPY: i32 = 11; // corrugated rain canopies
const CITY_PART_CORE: i32 = 12;   // a core strip's proud front (wall)

struct CityBox {
    bmin: vec3<f32>,
    bmax: vec3<f32>,
}

// Distance along a ray from a point INSIDE a box to each of the box's far
// planes: the plane the ray is heading for on each axis, or never (1e30) on
// an axis the ray does not move along. The reciprocal stand-in 1e30 alone is
// not enough — it multiplies a negative offset into a huge negative time.
fn city_exit_t(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>,
               bmax: vec3<f32>) -> vec3<f32> {
    let far = select(bmin, bmax, d > vec3<f32>(0.0));
    let moves = abs(d) > vec3<f32>(1e-9);
    return select(vec3<f32>(1e30), (far - o) / select(vec3<f32>(1.0), d, moves),
                  moves);
}

fn city_tier_box(cc: CityCell, k: i32) -> CityBox {
    if (k == 1) {
        return CityBox(cc.b2min, cc.b2max);
    }
    if (k == 2) {
        return CityBox(cc.b3min, cc.b3max);
    }
    return CityBox(cc.b1min, cc.b1max);
}

// --- the crown ----------------------------------------------------------------
//
// A penthouse: a 2-3 storey box inset on the summit deck of ~20% of tall
// flat-topped towers. It is massing, at every range, so a far tower's
// silhouette never changes as the camera closes on it (as a prop it
// vanished past CITY_PROP_RANGE, a ~0.26 degree pop on a far tower); its
// walls are core facades on the tower's own lattice, tier index
// CITY_CROWN_TIER, and its storeys ride the same rows (the roof is
// storey-quantized and the crown is 2 or 3 storeys).
//
// It is per CELL, not per lot: on a merged superblock a checkerboard of the
// member cells may carry one, each clipped to its own cell's column and
// inset at least 3 m inside it, so the DDA cell that draws a crown is the
// cell that contains it, relief and all (CITY_SHELL_OUT_MAX is 1.5 m).
// A helipad (rooftopworks' prop) claims the same real estate first, so the
// draw decides both; kind 2 is the pad, which the core does not trace.
struct CityCrown {
    kind: i32,   // 0 none, 1 penthouse, 2 helipad
    lo: vec2<f32>,
    hi: vec2<f32>,
    z0: f32,
    z1: f32,
}

fn city_crown_none() -> CityCrown {
    return CityCrown(0, vec2<f32>(1e30), vec2<f32>(-1e30), 0.0, 0.0);
}

fn city_crown_overlap(alo: vec2<f32>, ahi: vec2<f32>, blo: vec2<f32>,
                      bhi: vec2<f32>) -> bool {
    return alo.x < bhi.x && ahi.x > blo.x && alo.y < bhi.y && ahi.y > blo.y;
}

fn city_crown(cc: CityCell, ci: vec2<i32>) -> CityCrown {
    // Gate 0, no hash: unbuilt lots, spires and warehouses carry no crown,
    // and neither does anything too short for a pad.
    if (!cc.built || cc.arch == 4 || cc.land_use == CITY_USE_INDUSTRIAL
        || cc.height <= CITY_CROWN_PAD_MIN_H) {
        return city_crown_none();
    }
    // The summit deck (rooftopworks' deck 0, which never has a hole): the
    // tapered shaft's top cross-section, a growth tower's main roof, or a
    // setback stack's top tier.
    var dlo = cc.b1min.xy;
    var dhi = cc.b1max.xy;
    var dz = cc.b1max.z;
    if (cc.arch == 3) {
        let c = 0.5 * (cc.fmin.xy + cc.fmax.xy);
        let hf = 0.5 * (cc.fmax.xy - cc.fmin.xy) * cc.fscale;
        dlo = c - hf;
        dhi = c + hf;
        dz = cc.fmax.z;
    } else if (cc.arch != 2 && cc.tiers == 2) {
        dlo = cc.b2min.xy; dhi = cc.b2max.xy; dz = cc.b2max.z;
    } else if (cc.arch != 2 && cc.tiers >= 3) {
        dlo = cc.b3min.xy; dhi = cc.b3max.xy; dz = cc.b3max.z;
    }
    let cell = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cell;
    let rlo = max(dlo, cmin);
    let rhi = min(dhi, cmin + vec2<f32>(cell, cell));
    let size = rhi - rlo;
    if (min(size.x, size.y) < CITY_CROWN_MIN_DECK) {
        return city_crown_none();
    }
    // Eligibility before entropy: a building too short for either a
    // penthouse or a pad never draws a hash.
    let foot0 = max(dhi.x - dlo.x, dhi.y - dlo.y);
    let pad_size = cc.merged || foot0 > CITY_CROWN_PAD_BIG;
    if (cc.height <= CITY_CROWN_MIN_H && !pad_size) {
        return city_crown_none();
    }
    // Building-level draws (shared by every cell of a merged group) decide
    // WHETHER; a cell-level draw decides the shape, so the two penthouses on
    // a superblock are not identical twins.
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x1b56c4e9u, 0x0d2f1a37u));
    // Tile-wrapped, like every cell-index draw (city_tile_cell); the
    // quadrant test below is wrap-invariant on its own (n is even).
    let cw = city_tile_cell(ci);
    let lh = city_rand4(vec2<u32>(
        bitcast<u32>(cw.x) * 0x9e3779b9u + 0x2545f491u,
        bitcast<u32>(cw.y) * 0x85ebca6bu + 0x27d4eb2fu));
    // Helipad first: it wants the same real estate and it is the rarer
    // thing. A building gets one pad, in the member cell under a point of
    // its top deck that the building-level draw picks.
    let pad_at = mix(dlo, dhi, vec2<f32>(0.25) + 0.5 * gh.zw);
    let pad_here = all(ci == vec2<i32>(floor(pad_at / cell)));
    if (pad_size && pad_here && gh.x < CITY_CROWN_PAD_FRAC) {
        let half = clamp(0.30 * min(size.x, size.y), 4.5, 11.0);
        let c = mix(rlo + half + 1.0, rhi - half - 1.0,
                    vec2<f32>(0.30 + 0.40 * lh.x, 0.30 + 0.40 * lh.y));
        let plo = c - half;
        let phi = c + half;
        // Not on top of the mast.
        if (!cc.has_mast
            || !city_crown_overlap(plo, phi, cc.mast_min.xy - 1.0,
                                   cc.mast_max.xy + 1.0)) {
            return CityCrown(2, plo, phi, dz, dz + CITY_CROWN_PAD_T);
        }
    }
    // Penthouse. On a multi-block roof only a checkerboard of its cells may
    // carry one, so neighbouring penthouses never share a party wall.
    let pent_here = !cc.merged || ((ci.x + ci.y) & 1) == 0;
    if (cc.height > CITY_CROWN_MIN_H && pent_here
        && gh.y < CITY_CROWN_FRAC) {
        let inset = max(CITY_CROWN_INSET * min(size.x, size.y), 3.0);
        let plo = rlo + inset;
        let phi = rhi - inset;
        if (min(phi.x - plo.x, phi.y - plo.y) > 6.0) {
            return CityCrown(1, plo, phi, dz,
                             dz + select(CITY_CROWN_H_LO, CITY_CROWN_H_HI,
                                         lh.z > 0.45));
        }
    }
    return city_crown_none();
}

// Could this cell carry a penthouse? No draw: the gate every crown use
// passes before it hashes.
fn city_crown_possible(cc: CityCell) -> bool {
    return cc.built && cc.height > CITY_CROWN_MIN_H && cc.arch != 4
           && cc.land_use != CITY_USE_INDUSTRIAL;
}

// The penthouse's box, or an empty one (bmin > bmax) where there is none.
// The draw: call it once per cell and hand the box on.
fn city_crown_box(cc: CityCell) -> CityBox {
    let cr = city_crown(cc, cc.cell);
    if (cr.kind != 1) {
        return CityBox(vec3<f32>(1e30), vec3<f32>(-1e30));
    }
    return CityBox(vec3<f32>(cr.lo, cr.z0), vec3<f32>(cr.hi, cr.z1));
}

// Where a penthouse can stand, without the draw: the summit deck's
// rectangle (which holds every crown's footprint) under its headroom.
fn city_crown_bound(cc: CityCell) -> CityBox {
    var lo = cc.b1min.xy;
    var hi = cc.b1max.xy;
    if (cc.arch == 3) {
        let c = 0.5 * (cc.fmin.xy + cc.fmax.xy);
        let hf = 0.5 * (cc.fmax.xy - cc.fmin.xy) * cc.fscale;
        lo = c - hf;
        hi = c + hf;
    } else if (cc.arch != 2 && cc.tiers == 2) {
        lo = cc.b2min.xy; hi = cc.b2max.xy;
    } else if (cc.arch != 2 && cc.tiers >= 3) {
        lo = cc.b3min.xy; hi = cc.b3max.xy;
    }
    return CityBox(vec3<f32>(lo, cc.height),
                   vec3<f32>(hi, cc.height + CITY_CROWN_H_HI));
}

// The massing box k of a cell whose crown box (city_crown_box, or empty) is
// `cb`: the tiers 0 .. tiers-1, then the crown as CITY_CROWN_TIER. Walk
// k < CITY_CROWN_TIER + 1 and skip what city_mass_has says is not there.
fn city_mass_box(cc: CityCell, cb: CityBox, k: i32) -> CityBox {
    if (k == CITY_CROWN_TIER) {
        return cb;
    }
    return city_tier_box(cc, k);
}

fn city_mass_has(cc: CityCell, cb: CityBox, k: i32) -> bool {
    return k < cc.tiers || (k == CITY_CROWN_TIER && cb.bmin.x <= cb.bmax.x);
}

// One vertical face and its bay lattice. `aligned` false is a frustum's
// sloped skin, which keeps the world-anchored lattice and no shell: its
// width changes with height, so no integer bay count fits it.
struct CityFace {
    aligned: bool,
    side: i32,        // 0 +x, 1 -x, 2 +y, 3 -y
    tier: i32,
    n: vec2<f32>,     // outward normal
    t: vec2<f32>,     // tangent, (-n.y, n.x) — the core's facade tangent
    ua: f32,          // u_local = dot(p.xy, t) - ua
    len: f32,
    wp: f32,          // w_local = dot(p.xy, n) - wp
    z0: f32,
    z1: f32,
    margin: f32,
    pitch: f32,
    nbays: i32,
    salt: u32,        // decorrelates the four faces' window draws
    // The lot column in face-local terms: nothing of the shell may leave it,
    // or the DDA would test it from a cell that does not own it.
    col_u0: f32,
    col_u1: f32,
    col_w: f32,
}

fn city_side_normal(side: i32) -> vec2<f32> {
    if (side == 0) { return vec2<f32>(1.0, 0.0); }
    if (side == 1) { return vec2<f32>(-1.0, 0.0); }
    if (side == 2) { return vec2<f32>(0.0, 1.0); }
    return vec2<f32>(0.0, -1.0);
}

fn city_face_make(cc: CityCell, bmin: vec3<f32>, bmax: vec3<f32>,
                  side: i32, tier: i32) -> CityFace {
    var f: CityFace;
    f.aligned = true;
    f.side = side;
    f.tier = tier;
    f.n = city_side_normal(side);
    f.t = vec2<f32>(-f.n.y, f.n.x);
    let a = dot(bmin.xy, f.t);
    let b = dot(bmax.xy, f.t);
    f.ua = min(a, b);
    f.len = abs(b - a);
    f.wp = max(dot(bmin.xy, f.n), dot(bmax.xy, f.n));
    f.z0 = bmin.z;
    f.z1 = bmax.z;
    f.margin = min(CITY_SHELL_CORNER, 0.25 * f.len);
    let usable = f.len - 2.0 * f.margin;
    // The crown is the glass room, on its own bays (CITY_PENT_PITCH).
    let pitch = select(cc.win_pitch, CITY_PENT_PITCH, tier == CITY_CROWN_TIER);
    let nb = floor(usable / pitch + 0.5);
    f.nbays = select(0, i32(nb), usable > 0.5 * pitch);
    f.pitch = usable / max(nb, 1.0);
    f.salt = (u32(side) + 4u * u32(tier) + 1u) * 0x9e3779b9u;
    // The lot column (the lot's cell rectangle) in face-local terms.
    let ca_ = dot(cc.lot_min, f.t);
    let cb_ = dot(cc.lot_max, f.t);
    f.col_u0 = min(ca_, cb_) - f.ua;
    f.col_u1 = max(ca_, cb_) - f.ua;
    f.col_w = max(dot(cc.lot_min, f.n), dot(cc.lot_max, f.n)) - f.wp;
    return f;
}

// The frustum skin: the core's old world-anchored lattice, unbounded.
fn city_face_skin(cc: CityCell, nrm: vec3<f32>) -> CityFace {
    var f: CityFace;
    f.aligned = false;
    f.side = 0;
    f.tier = 0;
    // @domain: a skin or unclaimed facade normal (kind 1: |n.z| <= 0.7), so |xy| >= 0.7
    f.n = normalize(nrm.xy + vec2<f32>(1e-9, 0.0));
    f.t = vec2<f32>(-f.n.y, f.n.x);
    f.ua = 0.0;
    f.len = 1e9;
    f.wp = 0.0;
    f.z0 = 0.0;
    f.z1 = CITY_MAX_H;
    f.margin = 0.0;
    f.pitch = cc.win_pitch;
    f.nbays = 1 << 30;
    f.salt = 0u;
    f.col_u0 = -1e9;
    f.col_u1 = 1e9;
    f.col_w = 0.0;
    return f;
}

// Which face a flat facade hit lies on: the box whose face plane holds the
// point. A frustum skin (sloped normal) and anything no box claims keep the
// world lattice — the only non-box facades are frustum skins, and a box face
// the point is not on would hand it someone else's windows.
fn city_face_at(cc: CityCell, pos: vec3<f32>, nrm: vec3<f32>) -> CityFace {
    var side = -1;
    if (abs(nrm.z) < 1e-3) {
        if (nrm.x > 0.999) { side = 0; }
        else if (nrm.x < -0.999) { side = 1; }
        else if (nrm.y > 0.999) { side = 2; }
        else if (nrm.y < -0.999) { side = 3; }
    }
    if (side < 0) {
        return city_face_skin(cc, nrm);
    }
    for (var k: i32 = 0; k < cc.tiers; k = k + 1) {
        let bx = city_tier_box(cc, k);
        let f = city_face_make(cc, bx.bmin, bx.bmax, side, k);
        let w = dot(pos.xy, f.n) - f.wp;
        let uu = dot(pos.xy, f.t) - f.ua;
        if (abs(w) < 0.02 && uu > -0.02 && uu < f.len + 0.02
            && pos.z > f.z0 - 0.02 && pos.z < f.z1 + 0.02) {
            return f;
        }
    }
    return city_face_skin(cc, nrm);
}

// city_face_at, crowns included: the core's own facade shading. Only a
// point above the roof can be on a crown's wall, so only those pay for the
// draw. (Components ask city_face_at, tiers only: a crown wall comes back
// as a skin, not aligned, and their base-wall and flat-wall gates leave it.)
fn city_face_at_mass(cc: CityCell, pos: vec3<f32>, nrm: vec3<f32>)
        -> CityFace {
    if (pos.z > cc.height - 0.02 && city_crown_possible(cc)
        && abs(nrm.z) < 1e-3) {
        let cb = city_crown_box(cc);
        if (cb.bmin.x <= cb.bmax.x) {
            var side = 3;
            if (nrm.x > 0.999) { side = 0; }
            else if (nrm.x < -0.999) { side = 1; }
            else if (nrm.y > 0.999) { side = 2; }
            let f = city_face_make(cc, cb.bmin, cb.bmax, side,
                                   CITY_CROWN_TIER);
            // On the crown's own plane, as city_face_at asks of a tier: the
            // top 2 cm of the summit's walls are not the crown's.
            if (abs(dot(pos.xy, f.n) - f.wp) < 0.02) {
                return f;
            }
        }
    }
    return city_face_at(cc, pos, nrm);
}

fn city_face_local(f: CityFace, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(p.xy, f.t) - f.ua, p.z, dot(p.xy, f.n) - f.wp);
}

fn city_face_dir(f: CityFace, d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(d.xy, f.t), d.z, dot(d.xy, f.n));
}

fn city_face_world(f: CityFace, l: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(f.t * l.x + f.n * l.z, l.y);
}

// A face-local POINT back in the city frame (city_face_world rotates
// directions only).
fn city_face_point(f: CityFace, l: vec3<f32>) -> vec3<f32> {
    return city_face_world(f, l) + vec3<f32>(f.t * f.ua + f.n * f.wp, 0.0);
}

// Whether a component's panel covers the wall at face-local (u, v): the
// shell then shows the panel's flat wall, no opening and no relief.
fn city_face_covered(cc: CityCell, f: CityFace, uu: f32, v: f32,
                     fp: f32) -> bool {
    var hw: CityHit;
    hw.hit = true;
    hw.t = 0.0;
    hw.pos = city_face_point(f, vec3<f32>(uu, v, 0.0));
    hw.normal = vec3<f32>(f.n, 0.0);
    hw.kind = 1;
    hw.cell = vec2<i32>(0);
    return cc_facade_cover(cc, hw, uu, v, fp) > 0.5;
}

// The bay cell holding face-local (u, v). iu runs 0..nbays-1 over the bays,
// with -1 and nbays for the two corner strips; ca is the cell's u start.
struct CityBay {
    iu: i32,
    iv: i32,
    bu: f32,
    bv: f32,
    ca: f32,
    in_bay: bool,
    zone: i32,        // CITY_ZONE_*: what the facade zoning makes of it
}

fn city_bay(f: CityFace, uu: f32, v: f32) -> CityBay {
    var b: CityBay;
    let x = uu - f.margin;
    var iu = i32(floor(x / f.pitch));
    if (f.aligned) {
        iu = clamp(iu, -1, f.nbays);
        if (x < 0.0) { iu = -1; }
    }
    b.iu = iu;
    b.in_bay = !f.aligned || (iu >= 0 && iu < f.nbays);
    b.ca = f.margin + f32(iu) * f.pitch;
    if (f.aligned && iu < 0) { b.ca = -1e9; }
    if (f.aligned && iu >= f.nbays) { b.ca = f.len - f.margin; }
    b.bu = uu - b.ca;
    b.iv = i32(floor(v / CITY_FLOOR_H));
    b.bv = v - f32(b.iv) * CITY_FLOOR_H;
    b.zone = CITY_ZONE_WIN;
    return b;
}

// city_bay with the face's zoning applied: a solid or plant bay has no
// opening (in_bay false, so nothing that hangs on an opening stands there).
fn city_bay_zoned(z: CityZones, f: CityFace, uu: f32, v: f32) -> CityBay {
    return city_bay_zone(z, f, city_bay(f, uu, v));
}

fn city_bay_zone(z: CityZones, f: CityFace, b_in: CityBay) -> CityBay {
    var b = b_in;
    b.zone = city_zone_at(z, f, b.iu, b.iv);
    b.in_bay = b.in_bay && city_zone_open(b.zone);
    return b;
}

// One window's state: whether it burns, and at what radiance. The style
// decides which UNIT switches on: single panes (grid, punched),
// floor-segments (ribbon), multi-storey column runs (strips), or whole
// floors (curtain wall) — so towers separate at range by texture, not just
// by occupancy.
struct CityWin {
    lit: bool,
    e: vec3<f32>,     // lit pane radiance before glyphs; zero when dark
    wh: vec4<f32>,
}

// The threshold a window's lit draw on storey iv is held to: its unit is lit
// when the draw falls under it. The storey's own draw decides the rate: a
// few storeys dark, the rest at an occupancy of their own around the
// building's, which is what gives a far tower its floors without making
// them stripes. Whole floors switched on or off (as curtain walls once were)
// read as zebra bands at range, and real towers show mostly scattered rooms
// with some floor-wide structure.

// The style's rate against the building's lit_frac: a ribbon's paired bays
// and a strip's column runs switch in larger units, and an office floor
// plate burns more of its glass.
fn city_style_rate(cc: CityCell) -> f32 {
    return select(select(select(1.0, 1.32, cc.win_style == 3),
                         1.10, cc.win_style == 2),
                  1.15, cc.win_style == 1);
}

// The share of its rate a building's storeys burn at on average, dark ones
// included: 0.90, as when a tenth of the storeys went dark and the rest
// burned at the full rate.
const CITY_LIT_SHARE: f32 = 0.90;
// A storey's occupancy against the building's, from its draw r in [0, 1):
// mean CITY_STOREY_OCC_MEAN (CITY_LIT_SHARE over the storeys left lit),
// spread 0.6-1.4x of that, a little wider on an office tower's floors.
const CITY_STOREY_OCC_MEAN: f32 = CITY_LIT_SHARE / (1.0 - CITY_DARK_FLOOR_FRAC);
fn city_storey_occupancy(cc: CityCell, r: f32) -> f32 {
    let spread = select(0.40, 0.50, cc.win_style == 3);
    return CITY_STOREY_OCC_MEAN * (1.0 - spread + 2.0 * spread * r);
}
// Tenants take floors in runs: a building's storeys fall into runs of 3-6
// whose occupancy is mostly shared (CITY_TENANT_SHARE of the draw is the
// run's), so a tower reads as stacks of busier and quieter floors rather
// than as independent ones. The draw stays uniform in mean, so the storey
// expectation (city_storey_lit_mean) is unchanged.
const CITY_TENANT_SHARE: f32 = 0.65;
fn city_storey_rate(cc: CityCell, iv: i32) -> f32 {
    return city_storey_rate_at(cc, iv, cc.lit_frac);
}

// city_storey_rate at a lit_frac of the caller's: a loft's or a shop's
// (city_zone_lit_frac), drawn from the building's storey
// draws.
fn city_storey_rate_at(cc: CityCell, iv: i32, lit_frac: f32) -> f32 {
    let fh = city_rand4(vec2<u32>(
        cc.seed.x ^ 0x2545f491u,
        bitcast<u32>(iv) * 0x9e3779b9u
    ));
    // A lit_frac past 1 is a row of shops (city_zone_lit_frac), which never
    // goes dark with the floor.
    if (fh.x < CITY_DARK_FLOOR_FRAC && lit_frac <= 1.0) {
        return 0.0;
    }
    let tb = city_rand4(cc.seed ^ vec2<u32>(0x7137449fu, 0xb5c0fbcfu));
    let len = 3 + i32(floor(tb.x * 3.999));
    let run = (iv + i32(floor(tb.y * f32(len)))) / len;
    let rr = city_rand4(vec2<u32>(cc.seed.y ^ 0xe9b5dba5u,
                                  bitcast<u32>(run) * 0x85ebca6bu)).x;
    return lit_frac * city_style_rate(cc)
           * city_storey_occupancy(cc, mix(fh.y, rr, CITY_TENANT_SHARE));
}

// A storey's lit probability averaged over the storey draws (the dark
// storeys and the occupancy spread cancel by construction; the clamp at 1
// is ignored), for a footprint too tall to walk storey by storey.
fn city_storey_lit_mean(cc: CityCell) -> f32 {
    return min(cc.lit_frac * city_style_rate(cc) * CITY_LIT_SHARE, 1.0);
}

// A window of storey iv drawn at that storey's rate (city_storey_rate, or
// city_storey_rate_at for a loft's or a shop's).
fn city_window_rated(cc: CityCell, f: CityFace, iu: i32, iv: i32,
                     rate: f32) -> CityWin {
    var w: CityWin;
    w.wh = city_rand4(vec2<u32>(
        cc.seed.x ^ bitcast<u32>(iu) ^ f.salt,
        cc.seed.y ^ bitcast<u32>(iv)
    ));
    // The unit the style switches on: one pane (grid, punched, curtain),
    // a pair of bays (ribbon), or a four-storey column run (strips).
    var unit = w.wh.x;
    if (cc.win_style == 1) {
        unit = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu >> 1) * 0x9e3779b9u) ^ f.salt,
            cc.seed.y ^ (bitcast<u32>(iv) * 0x51ed270bu))).x;
    } else if (cc.win_style == 2) {
        unit = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu) * 0x9e3779b9u) ^ f.salt,
            cc.seed.y ^ (bitcast<u32>(iv >> 2) * 0x51ed270bu))).x;
    }
    w.lit = unit < rate;
    w.e = vec3<f32>(0.0);
    if (w.lit) {
        // Curtain-wall floors run dimmer per pane than a lamp-lit room
        // — office fluorescents against far more glass — and ribbon
        // bands a touch below grid; without this the glass towers
        // white out whole canyons.
        let style_gain = select(
            select(1.0, 0.80, cc.win_style == 1),
            0.55, cc.win_style == 3);
        // @domain: a hash draw, in [0, 1)
        let bright = (0.25 + 5.0 * pow(w.wh.z, 7.0)) * style_gain;
        // A monochrome house draws every window near its one palette
        // coordinate; a mixed house draws freely.
        var cdraw = w.wh.y;
        if (cc.win_mono >= 0.0) {
            cdraw = clamp(cc.win_mono + (w.wh.y - 0.5) * 0.08, 0.0, 1.0);
        }
        w.e = city_room_color(cdraw, cc.palette_bias)
              * (CITY_ROOM_RADIANCE * bright);
    }
    return w;
}

// What the building is built of, and how deep its facade is. Keyed to the
// window style the core already draws (curtain walls on towers, punched
// openings on low-rise), then varied per building so two brick walk-ups do
// not share a sill.
struct CityShellStyle {
    reveal: f32,        // glass setback behind the wall plane
    sill_out: f32,      // sill protrusion (0 = none)
    ledge_out: f32,     // horizontal band protrusion (0 = none)
    ledge_v0: f32,      // band extent within the storey (m above the floor)
    ledge_v1: f32,
    ledge_every: i32,   // storeys between bands
    cornice_out: f32,   // cornice at the top of each tier (0 = none)
    cap_out: f32,       // fins / mullion caps at bay edges (0 = none)
    cap_hw: f32,
    mull: i32,          // 0 none, 1 centre mullion, 2 meeting rail
    albedo: vec3<f32>,  // wall
    trim: vec3<f32>,    // sills, ledges, cornices, fins
    frame: vec3<f32>,   // window frames and mullions
    hood_out: f32,      // cap over each window opening (0 = none)
    canopy_out: f32,    // rain canopy over each window (0 = none)
    ac_frac: f32,       // share of bays with a wall unit under the sill
    pipes: i32,         // drainpipes per face
    brick: bool,        // coursed brick, visible at arm's length
    escape: bool,       // the building carries a fire escape
    balcony: bool,      // ... or balconies (never both)
    depth: f32,         // deepest protrusion: the shell layer's thickness
}

fn city_shell_style(cc: CityCell) -> CityShellStyle {
    var st: CityShellStyle;
    let r = city_rand4(cc.seed ^ vec2<u32>(0x6a09e667u, 0xbb67ae85u));
    let q = city_rand4(cc.seed ^ vec2<u32>(0x510e527fu, 0x9b05688cu));
    let tall = cc.height > 150.0;
    st.reveal = 0.18;
    st.sill_out = 0.0;
    st.ledge_out = 0.0;
    st.ledge_v0 = 0.0;
    st.ledge_v1 = 0.0;
    st.ledge_every = 1;
    st.cornice_out = 0.0;
    st.cap_out = 0.0;
    st.cap_hw = 0.0;
    st.mull = 0;
    st.albedo = vec3<f32>(0.22, 0.21, 0.20);        // concrete
    st.trim = vec3<f32>(0.30, 0.29, 0.27);
    st.frame = vec3<f32>(0.05, 0.05, 0.055);        // dark anodized
    st.ac_frac = 0.0;
    st.pipes = 0;
    st.brick = false;
    st.hood_out = 0.0;
    st.canopy_out = 0.0;
    let pane_w = (cc.pane_hi.x - cc.pane_lo.x) * cc.win_pitch;
    if (cc.win_style == 4) {
        // Punched masonry: the walk-up. Deep reveals, stone sills, a
        // meeting rail across every sash, and a cornice to finish the wall.
        st.reveal = 0.26 + 0.12 * r.x;
        st.sill_out = 0.06;
        st.mull = 2;
        if (r.y < 0.30) {
            st.albedo = vec3<f32>(0.20, 0.085, 0.055);   // red brick
        } else if (r.y < 0.55) {
            st.albedo = vec3<f32>(0.13, 0.075, 0.055);   // brown brick
        } else if (r.y < 0.75) {
            st.albedo = vec3<f32>(0.30, 0.23, 0.15);     // buff brick
        } else {
            st.albedo = vec3<f32>(0.34, 0.32, 0.29);     // stucco
        }
        st.brick = r.y < 0.75;
        // Most walk-ups cap each window with a projecting hood or lintel.
        st.hood_out = select(0.0, 0.07 + 0.05 * q.y, r.z > 0.3);
        st.trim = vec3<f32>(0.42, 0.39, 0.34);           // limestone
        st.frame = select(vec3<f32>(0.45, 0.44, 0.42),
                          vec3<f32>(0.06, 0.05, 0.05), r.z < 0.5);
        if (r.w < 0.40) {
            st.ledge_out = 0.05;                          // band course
            st.ledge_v0 = 0.0;
            st.ledge_v1 = 0.20;
            st.ledge_every = 1 + i32(floor(q.x * 3.0));
        }
        st.cornice_out = select(0.0, 0.30 + 0.20 * q.y, cc.height < 80.0);
        st.ac_frac = select(0.0, 0.10 + 0.20 * q.z, !tall);
        st.pipes = i32(floor(q.w * 2.99));
    } else if (cc.win_style == 0) {
        // Concrete grid: the slab edge expressed as a ledge at every floor,
        // the glass set back behind it, units on the spandrels.
        st.reveal = 0.16 + 0.10 * r.x;
        st.sill_out = select(0.0, 0.04, r.y < 0.5);
        st.mull = select(0, 1, pane_w > 1.6);
        if (r.y > 0.7) {
            st.albedo = vec3<f32>(0.40, 0.40, 0.38);      // white tile
        } else if (r.y > 0.5) {
            st.albedo = vec3<f32>(0.26, 0.24, 0.21);      // painted render
        }
        if (r.w < 0.60) {
            st.ledge_out = 0.10 + 0.08 * q.x;
            st.ledge_v0 = 0.0;
            st.ledge_v1 = 0.28;
        }
        st.ac_frac = select(0.0, 0.08 + 0.25 * q.z, !tall);
        st.pipes = select(0, i32(floor(q.w * 2.99)), !tall);
        st.canopy_out = select(0.0, 0.45 + 0.25 * r.z,
                               !tall && cc.height < 90.0 && r.z > 0.6);
    } else if (cc.win_style == 1) {
        // Ribbon windows: continuous bands, and on some buildings a
        // sunshade slab over every band.
        st.reveal = 0.10;
        st.cap_out = 0.05;
        st.cap_hw = 0.03;
        if (r.w < 0.45) {
            st.ledge_out = 0.35 + 0.20 * q.x;
            st.ledge_v0 = cc.pane_hi.y * CITY_FLOOR_H + 0.02;
            st.ledge_v1 = st.ledge_v0 + 0.10;
        }
        st.albedo = select(vec3<f32>(0.24, 0.24, 0.25),
                           vec3<f32>(0.07, 0.07, 0.08), r.y < 0.4);
        st.trim = st.albedo * 1.2;
        st.ac_frac = select(0.0, 0.06, !tall);
        st.pipes = select(0, 1, !tall && q.w < 0.4);
    } else if (cc.win_style == 2) {
        // Vertical strips between deep fins.
        st.reveal = 0.12;
        st.cap_out = 0.25 + 0.20 * r.w;
        st.cap_hw = 0.08 + 0.06 * q.x;
        st.albedo = select(vec3<f32>(0.30, 0.28, 0.25),
                           vec3<f32>(0.20, 0.20, 0.21), r.y < 0.5);
        st.trim = st.albedo;
    } else {
        // Curtain wall: glass nearly flush, aluminium caps on every
        // mullion, a transom cap at each floor line.
        st.reveal = 0.03;
        st.cap_out = 0.08 + 0.06 * r.w;
        st.cap_hw = 0.03;
        st.ledge_out = 0.04;
        st.ledge_v0 = 0.0;
        st.ledge_v1 = 0.06;
        st.albedo = vec3<f32>(0.035, 0.038, 0.045);       // spandrel glass
        st.trim = vec3<f32>(0.10, 0.10, 0.11);
    }
    st.depth = max(max(max(st.sill_out, st.ledge_out),
                       max(st.cornice_out, st.cap_out)),
                   max(st.hood_out, st.canopy_out));
    if (st.ac_frac > 0.0) {
        st.depth = max(st.depth, 0.58);
    }
    if (st.pipes > 0) {
        st.depth = max(st.depth, 0.24);
    }
    // Industrial walls are the industry component's: nothing of the shell's
    // stands on them, near or far (the refine skips them; the flat twin's
    // covers must too).
    if (cc.land_use != CITY_USE_URBAN) {
        st.ac_frac = 0.0;
    }
    st.escape = city_escape_building(cc);
    st.balcony = city_balcony_building(cc);
    if (st.escape) {
        st.depth = max(st.depth, CITY_ESCAPE_OUT);
    }
    if (st.balcony) {
        st.depth = max(st.depth, CITY_BALCONY_OUT);
    }
    // A core strip may stand on any face of a tower that has them
    // (city_zones); the layer thickens only there, since every centimetre
    // of it is cells a grazing ray walks.
    if (city_core_building(cc)) {
        st.depth = max(st.depth, CITY_CORE_OUT);
    }
    st.depth = min(st.depth + 0.02, CITY_SHELL_OUT_MAX);
    return st;
}

// The pane rectangle in bay-local metres (bu, bv).
fn city_pane_lo(cc: CityCell, f: CityFace) -> vec2<f32> {
    return cc.pane_lo * vec2<f32>(f.pitch, CITY_FLOOR_H);
}

fn city_pane_hi(cc: CityCell, f: CityFace) -> vec2<f32> {
    return cc.pane_hi * vec2<f32>(f.pitch, CITY_FLOOR_H);
}

// A bay's air conditioner: whether it has one, and its box in bay-local
// (bu, bv) and w. Two kinds by the building's draw: hung on the spandrel
// under the sill, off-centre by the bay's own draw; or set IN the window,
// on the sill in the lower sash, standing out past the wall under a little
// sheet-metal drip hood (the Hong Kong and New York walk-up unit).
struct CityAC {
    on: bool,
    in_window: bool,
    lo: vec2<f32>,
    hi: vec2<f32>,
    w0: f32,
    out: f32,
}

fn city_shell_ac(cc: CityCell, f: CityFace, st: CityShellStyle,
                 b: CityBay) -> CityAC {
    var a: CityAC;
    a.on = false;
    a.in_window = false;
    a.lo = vec2<f32>(0.0);
    a.hi = vec2<f32>(0.0);
    a.w0 = 0.0;
    a.out = 0.55;
    let plo = city_bay_pane_lo(cc, f, b);
    let phi = city_bay_pane_hi(cc, f, b);
    if (st.ac_frac <= 0.0 || !b.in_bay || b.iv < 1
        || b.zone != CITY_ZONE_WIN) {
        return a;
    }
    let kind = city_rand4(cc.seed ^ vec2<u32>(0x3243f6a8u, 0x885a308du));
    a.in_window = kind.x < 0.45 && (phi.x - plo.x) > 0.8;
    if (!a.in_window && plo.y < 0.85) {
        return a;
    }
    let h = city_rand4(vec2<u32>(
        cc.seed.x ^ (bitcast<u32>(b.iu) * 0x632be5abu) ^ f.salt,
        cc.seed.y ^ (bitcast<u32>(b.iv) * 0x2e1b2138u)));
    if (h.x >= st.ac_frac) {
        return a;
    }
    a.on = true;
    if (a.in_window) {
        // In the lower sash: as wide as the unit, not the window.
        let wid = min(0.62 + 0.10 * h.y, phi.x - plo.x - 0.1);
        let cu = 0.5 * (plo.x + phi.x)
                 + (h.w - 0.5) * max(phi.x - plo.x - wid - 0.1, 0.0);
        a.lo = vec2<f32>(cu - 0.5 * wid, plo.y);
        a.hi = vec2<f32>(cu + 0.5 * wid, plo.y + 0.40 + 0.06 * h.z);
        a.w0 = -st.reveal;
        a.out = 0.30 + 0.12 * h.y;
        return a;
    }
    let wid = 0.70 + 0.16 * h.y;
    let hgt = 0.42 + 0.10 * h.z;
    let cu = 0.5 * f.pitch + (h.w - 0.5) * max(f.pitch - wid - 0.2, 0.0);
    let top = plo.y - CITY_SHELL_FRAME - 0.14;
    a.lo = vec2<f32>(cu - 0.5 * wid, top - hgt);
    a.hi = vec2<f32>(cu + 0.5 * wid, top);
    a.out = 0.45 + 0.10 * h.y;
    return a;
}

// A face's drainpipes: up to two, each at a u position on the face. Corner
// strips first (where a downpipe runs on a real building), then a bay edge
// on fin-less styles.
fn city_shell_pipe_u(cc: CityCell, f: CityFace, st: CityShellStyle,
                     k: i32) -> f32 {
    let h = city_rand4(vec2<u32>(cc.seed.x ^ f.salt ^ 0x94d049bbu,
                                 cc.seed.y ^ (u32(k) * 0x7f4a7c15u)));
    if (k >= st.pipes || h.x < 0.35 || f.len < 6.0) {
        return -1e9;
    }
    if (k == 0 || st.cap_out > 0.0 || f.nbays < 2) {
        return select(0.45 * f.margin, f.len - 0.45 * f.margin, h.y < 0.5);
    }
    let edge = 1 + i32(floor(h.z * f32(f.nbays - 1)));
    return f.margin + f32(edge) * f.pitch;
}

fn city_shell_pipe_r(cc: CityCell, f: CityFace, k: i32) -> f32 {
    let h = pcg2d(vec2<u32>(cc.seed.x ^ f.salt, u32(k) + 77u));
    return 0.050 + 0.025 * city_u01(h.x);
}

// What a face's features draw once, rather than in every bay cell the walk
// crosses: its two pipes' places (u < -1e8: none) and radii, and its
// balcony rhythm.
// --- facade zones --------------------------------------------------------------
//
// The scale between the window and the building. A real wall is not one
// lattice from corner to corner: a tower's narrow end is a solid core with
// the lifts behind it, every twenty-odd floors a plant floor runs dark
// behind louvres, the top storeys are plant under the crown, the corner flats
// at the top have the big glass, and on some buildings one whole side is
// solid. Seen at night from the air (Midtown from the Empire State, Commons)
// these are what separate one tower from the next once the windows blur:
// dark bands across lit shafts, a dark stripe up an end, a bright corner.
//
// Each zone is a rectangle of whole bays by whole storeys on one face, so
// the shell finds it per bay (city_zone_at) and the flat facade covers the
// pixel's footprint with it in closed form (city_zone_cover): the share of
// the footprint each zone takes is a product of interval overlaps, and the
// windows the zones leave keep the lattice's statistics. Plant floors are
// a property of the building's storeys, the same on every face; cores, solid
// sides and their placement are the face's own draw.
const CITY_ZONE_WIN: i32 = 0;    // the ordinary lattice
const CITY_ZONE_BLANK: i32 = 1;  // a solid side: wall, no openings
const CITY_ZONE_CORE: i32 = 2;   // a solid core strip standing proud
const CITY_ZONE_PLANT: i32 = 3;  // a plant floor behind louvres
const CITY_ZONE_LOFT: i32 = 4;   // a corner flat with the big glass
const CITY_ZONE_STORE: i32 = 5;  // a shop window on a commercial ground floor
const CITY_ZONE_PENT: i32 = 6;   // the crown's glass room
// How far a core strip stands out of the wall.
const CITY_CORE_OUT: f32 = 0.45;
// Louvres: dark painted blades, every LOUVRE_PITCH metres.
const CITY_LOUVRE_ALBEDO: vec3<f32> = vec3<f32>(0.07, 0.072, 0.075);
const CITY_LOUVRE_PITCH: f32 = 0.15;
// A loft's glass, as fractions of its bay cell: nearly floor to ceiling.
const CITY_LOFT_PANE_LO: vec2<f32> = vec2<f32>(0.06, 0.08);
const CITY_LOFT_PANE_HI: vec2<f32> = vec2<f32>(0.94, 0.95);
// A shop window: the bay glazed nearly wall to wall, 0.2-3.3 m up.
const CITY_STORE_PANE_LO: vec2<f32> = vec2<f32>(0.06, 0.055);
const CITY_STORE_PANE_HI: vec2<f32> = vec2<f32>(0.94, 0.92);
const CITY_STORE_LIT_FRAC: f32 = 4.0;
// The crown's glass room (Thomas, 2026-11-09: "Bring back glass room"): a
// penthouse glazed on every side, its own 2.1 m bays, the glass between
// slim mullions floor to ceiling, 58% of the bays lit, each at its own level
// over 1.8 in the house's colour (warm where the house has none). It was
// rooftopworks' prop until the crown became core massing; now it is a zone
// of the crown's faces, so the windows machinery draws it exactly at every
// range.
const CITY_PENT_PITCH: f32 = 2.10;
const CITY_PENT_PANE_LO: vec2<f32> = vec2<f32>(0.065, 0.03);
const CITY_PENT_PANE_HI: vec2<f32> = vec2<f32>(0.935, 0.97);
const CITY_PENT_LIT: f32 = 0.58;
const CITY_PENT_RAD: f32 = 1.80;
const CITY_PENT_COL: vec3<f32> = vec3<f32>(1.00, 0.72, 0.42);
// The odd shop whose interior is cyan neon, drawn shop by shop (Thomas:
// "if some rare shops had an interior cyan color like the original, that
// would be kinda cool"): the palette's own cyan (band 4).
const CITY_SHOP_CYAN_FRAC: f32 = 0.05;

struct CityZones {
    blank: bool,        // the whole face is solid
    core: vec2<i32>,    // bays [x, y) are a core strip, from storey 1 up
    plant: vec4<i32>,   // plant storeys iv in [x, w] with (iv - x) % y < z
    crown: i32,         // storeys from here up are plant (1 << 30: none)
    loft: vec3<i32>,    // corner lofts on storeys [x, y), z bays each end
    store: bool,        // storey 0 is shop windows
    pent: bool,         // the crown's glass room: every bay of the face
}

// Whether a building carries core strips at all: towers above 40 m, not
// walk-ups, half of them. Which faces carry one is the face's own draw.
fn city_core_building(cc: CityCell) -> bool {
    let h = city_rand4(cc.seed ^ vec2<u32>(0x9b05688cu, 0x1f83d9abu));
    return cc.land_use == CITY_USE_URBAN && cc.height > 40.0
        && cc.win_style != 4 && h.x < 0.5;
}

fn city_zones_none() -> CityZones {
    return CityZones(false, vec2<i32>(0), vec4<i32>(0, 0, 0, -1), 1 << 30,
                     vec3<i32>(0), false, false);
}

fn city_zones(cc: CityCell, f: CityFace) -> CityZones {
    var z = city_zones_none();
    // The crown is the glass room, whole: no blank side, no loft, no core
    // strip (and its base is the roof, so never a shopfront).
    if (f.tier == CITY_CROWN_TIER) {
        z.pent = f.aligned && cc.land_use == CITY_USE_URBAN;
        return z;
    }
    if (!f.aligned || cc.land_use != CITY_USE_URBAN) {
        return z;
    }
    // A commercial ground floor is a row of shops, whatever is above it,
    // and on every face, as the street's glow (city_storefront_glow, the
    // building's) has it.
    z.store = f.z0 < 0.5 && dot(city_storefront_glow(cc), vec3<f32>(1.0)) > 0.0;
    if (f.nbays < 3) {
        return z;
    }
    let zb = city_rand4(cc.seed ^ vec2<u32>(0x3956c25bu, 0x59f111f1u));
    let zc = city_rand4(cc.seed ^ vec2<u32>(0x5be0cd19u, 0x1f83d9abu));
    let zf = city_rand4(cc.seed ^ vec2<u32>(f.salt, 0x6a09e667u));
    let top = i32(floor(cc.height * CITY_PER_STOREY + 0.5));
    let tall = cc.height > 60.0;
    // One side of the building may be solid: a quarter of the lower ones
    // and a few towers (never a curtain wall, which is glass all round).
    if (cc.win_style != 3 && f.side == i32(floor(zb.z * 3.999))
        && zf.x < select(0.45, 0.15, tall)) {
        z.blank = true;
        return z;
    }
    // The crown: a plant storey or two under the roof of half the towers.
    if (tall && zc.w < 0.5) {
        z.crown = top - 1 - select(0, 1, zc.z < 0.5);
    }
    // Plant floors through a tall tower, every 16-29 storeys.
    if (cc.height > 100.0 && zb.x < 0.70) {
        let p = 16 + i32(floor(zb.y * 13.999));
        z.plant = vec4<i32>(6 + i32(floor(fract(zb.y * 7.31) * f32(p))), p,
                            select(1, 2, zb.w < 0.35),
                            min(top, z.crown) - 4);
    }
    // Corner lofts on the top tier, under the crown.
    if (cc.win_style != 3 && cc.height >= 15.0 && zc.x < 0.35
        && f.z1 >= cc.height - 0.5) {
        let hi = min(top, z.crown);
        let n = 2 + i32(floor(zc.y * 1.999));
        z.loft = vec3<i32>(hi - n, hi,
                           select(1, 2, zc.z < 0.6 && f.nbays >= 6));
        z.plant.w = min(z.plant.w, hi - n - 2);
    }
    if (z.plant.w < z.plant.x) {
        z.plant.y = 0;
    }
    // A core strip up a tower's face: one or two bays, centred or near an
    // end (clear of the corner lofts).
    if (zf.y < 0.6 && f.nbays >= 5 && city_core_building(cc)) {
        let w = 1 + select(0, 1, zf.z < 0.6 && f.nbays >= 7);
        var c0 = (f.nbays - w) / 2;
        if (zf.w < 0.5 && f.nbays >= 8) {
            c0 = select(2, f.nbays - 2 - w, zf.w < 0.25);
        }
        z.core = vec2<i32>(c0, c0 + w);
    }
    return z;
}

// The zones in three words, for the flat facade's window paths, which
// consult them window by window and storey by storey: thirteen values
// held through those loops are registers for the whole surface pass.
// Bays fit 8 bits (a 360 m superblock face has ~105), storeys 12
// (CITY_MAX_STOREYS is 1500; 4095 is "no crown").
fn city_zones_pack(z: CityZones) -> vec3<u32> {
    let a = select(0u, 1u, z.blank) | select(0u, 2u, z.store)
          | (u32(z.core.x) << 2u) | (u32(z.core.y) << 10u)
          | (u32(z.loft.z) << 18u) | (u32(z.plant.z) << 20u)
          | (u32(z.plant.y) << 22u) | select(0u, 1u << 27u, z.pent);
    let b = u32(z.plant.x) | (u32(max(z.plant.w, 0)) << 12u);
    let c = u32(min(z.crown, 4095)) | (u32(max(z.loft.x, 0)) << 12u)
          | (u32(z.loft.y - z.loft.x) << 24u);
    return vec3<u32>(a, b, c);
}

fn city_zones_unpack(p: vec3<u32>) -> CityZones {
    var z: CityZones;
    z.blank = (p.x & 1u) != 0u;
    z.store = (p.x & 2u) != 0u;
    z.pent = (p.x & (1u << 27u)) != 0u;
    z.core = vec2<i32>(i32((p.x >> 2u) & 255u), i32((p.x >> 10u) & 255u));
    z.loft = vec3<i32>(i32((p.z >> 12u) & 4095u), 0, i32((p.x >> 18u) & 3u));
    z.loft.y = z.loft.x + i32((p.z >> 24u) & 15u);
    z.plant = vec4<i32>(i32(p.y & 4095u), i32((p.x >> 22u) & 31u),
                        i32((p.x >> 20u) & 3u), i32((p.y >> 12u) & 4095u));
    let crown = i32(p.z & 4095u);
    z.crown = select(crown, 1 << 30, crown == 4095);
    return z;
}

fn city_zone_at(z: CityZones, f: CityFace, iu: i32, iv: i32) -> i32 {
    if (iu < 0 || iu >= f.nbays) {
        return CITY_ZONE_WIN;
    }
    if (z.pent) {
        return CITY_ZONE_PENT;
    }
    if (z.store && iv == 0) {
        return CITY_ZONE_STORE;
    }
    if (z.blank) {
        return CITY_ZONE_BLANK;
    }
    if (iu >= z.core.x && iu < z.core.y && iv >= 1) {
        return CITY_ZONE_CORE;
    }
    if (city_storey_plant(z, iv)) {
        return CITY_ZONE_PLANT;
    }
    if (iv >= z.loft.x && iv < z.loft.y
        && (iu < z.loft.z || iu >= f.nbays - z.loft.z)) {
        return CITY_ZONE_LOFT;
    }
    return CITY_ZONE_WIN;
}

// Whether storey iv is plant (behind a core, the core's). The period test
// is in float (exact for storey counts, and an integer modulo is the most
// expensive thing the window paths would do per storey).
fn city_storey_plant(z: CityZones, iv: i32) -> bool {
    let q = f32(iv - z.plant.x);
    let p = f32(max(z.plant.y, 1));
    return iv >= z.crown
        || (z.plant.y > 0 && iv >= z.plant.x && iv <= z.plant.w
            && q - p * floor((q + 0.5) / p) < f32(z.plant.z));
}

// Whether a bay of this zone has an opening.
fn city_zone_open(zone: i32) -> bool {
    return zone == CITY_ZONE_WIN || zone == CITY_ZONE_LOFT
        || zone == CITY_ZONE_STORE || zone == CITY_ZONE_PENT;
}

// The pane rectangle of this bay, in bay-local metres: the building's, or a
// loft's big glass.
// (By the zone, not through a copy of the building: one per pane lookup
// cost the street view 0.05 ms of its 0.84 at Low.)
fn city_bay_pane_lo(cc: CityCell, f: CityFace, b: CityBay) -> vec2<f32> {
    var fr = select(select(cc.pane_lo, CITY_LOFT_PANE_LO,
                           b.zone == CITY_ZONE_LOFT),
                    CITY_STORE_PANE_LO, b.zone == CITY_ZONE_STORE);
    fr = select(fr, CITY_PENT_PANE_LO, b.zone == CITY_ZONE_PENT);
    return fr * vec2<f32>(f.pitch, CITY_FLOOR_H);
}
fn city_bay_pane_hi(cc: CityCell, f: CityFace, b: CityBay) -> vec2<f32> {
    var fr = select(select(cc.pane_hi, CITY_LOFT_PANE_HI,
                           b.zone == CITY_ZONE_LOFT),
                    CITY_STORE_PANE_HI, b.zone == CITY_ZONE_STORE);
    fr = select(fr, CITY_PENT_PANE_HI, b.zone == CITY_ZONE_PENT);
    return fr * vec2<f32>(f.pitch, CITY_FLOOR_H);
}

// A shop window as it is lit: the house's light, or in one shop in twenty
// cyan neon at the brightness the window drew. The draw is the window's
// own wh.x: a shop is lit whatever its unit draws, so it decides nothing
// else there.
fn city_shop_window(cc: CityCell, w_in: CityWin) -> CityWin {
    var w = w_in;
    if (w.lit && w.wh.x < CITY_SHOP_CYAN_FRAC) {
        let style_gain = select(select(1.0, 0.80, cc.win_style == 1),
                                0.55, cc.win_style == 3);
        // @domain: a hash draw, in [0, 1)
        let bright = 0.25 + 5.0 * pow(w.wh.z, 7.0);
        w.e = city_room_band_colour(4)
              * (CITY_ROOM_RADIANCE * bright * style_gain);
    }
    return w;
}

// A bay of the glass room as it is lit: its own draw against the room's
// rate (no dark floors, no tenants: it is one room), at its own level over
// CITY_PENT_RAD in the house's colour, or warm. The draw is the lattice's
// window hash, so the shell and the flat facade agree bay by bay.
fn city_pent_window(cc: CityCell, f: CityFace, iu: i32, iv: i32) -> CityWin {
    var w: CityWin;
    w.wh = city_rand4(vec2<u32>(
        cc.seed.x ^ bitcast<u32>(iu) ^ f.salt,
        cc.seed.y ^ bitcast<u32>(iv)
    ));
    w.lit = w.wh.x < CITY_PENT_LIT;
    w.e = vec3<f32>(0.0);
    if (w.lit) {
        w.e = city_pent_colour(cc) * (CITY_PENT_RAD * (0.45 + 1.10 * w.wh.z));
    }
    return w;
}

fn city_pent_colour(cc: CityCell) -> vec3<f32> {
    if (cc.win_mono >= 0.0) {
        return city_room_color(cc.win_mono, cc.palette_bias);
    }
    return CITY_PENT_COL;
}

// The lit_frac a bay's windows are drawn at (city_storey_rate_at), shell
// and flat facade alike: a shop's puts every storey's rate past 1 for any
// occupancy draw, and city_storey_rate never darkens it with the floor (a
// row of shops is lit, in the house's colours, so the flat facade's lit
// pane is the lattice's but for the odd cyan shop); a loft's is lit twice
// as often as the floors below, from at least 0.7, but never so often its
// rate could pass 1 (on ribbon and strip houses that cap, 0.66-0.69, is
// below 0.7 and every loft sits at it); the corner flats are the ones
// lived in.
const CITY_LOFT_LIT_GAIN: f32 = 2.0;
const CITY_LOFT_LIT_MIN: f32 = 0.70;
fn city_zone_lit_frac(cc: CityCell, zone: i32) -> f32 {
    if (zone == CITY_ZONE_STORE) {
        return CITY_STORE_LIT_FRAC;
    }
    if (zone == CITY_ZONE_LOFT) {
        return city_loft_lit_frac(cc);
    }
    return cc.lit_frac;
}

fn city_loft_lit_frac(cc: CityCell) -> f32 {
    // Held where the storey rate cannot pass 1 for any occupancy draw
    // (city_storey_occupancy at its top), so the walk's E[min(rate, 1)] and
    // the far mean's min(E[rate], 1) are the same number.
    let cap = 1.0 / (city_style_rate(cc) * city_storey_occupancy(cc, 1.0));
    return min(clamp(cc.lit_frac * CITY_LOFT_LIT_GAIN, CITY_LOFT_LIT_MIN,
                     1.0), cap);
}

// Plant storeys below storey coordinate s (storeys, not metres).
fn city_plant_below(z: CityZones, s: f32) -> f32 {
    let q = clamp(s, f32(z.plant.x), f32(z.plant.w + 1)) - f32(z.plant.x);
    let p = f32(z.plant.y);
    let k = floor(q / p);
    return k * f32(z.plant.z) + min(q - k * p, f32(z.plant.z));
}

// The shares of the footprint [c - hb, c + hb] (face-local u and z) that
// the zones take: solid (x: a solid side or a core), plant (y), loft (z)
// and shop (w), each within the face's bays (the corner strips are wall
// whatever the zoning, and the lattice's mean already holds them so). The
// zones are disjoint, so the four add, to at most the bays' share.
fn city_zone_cover(z: CityZones, f: CityFace, c: vec2<f32>, hb: vec2<f32>)
        -> vec4<f32> {
    let u0 = c.x - hb.x;
    let u1 = c.x + hb.x;
    let v0 = max(c.y - hb.y, f.z0);
    let v1 = min(c.y + hb.y, f.z1);
    let du = 2.0 * hb.x;
    let dv = 2.0 * hb.y;
    let strip = city_bay_strip(f);
    let bu = city_overlap(u0, u1, strip.x, strip.y) / du;
    // v shares are lengths within the face over the whole footprint height.
    let vface = max(v1 - v0, 0.0) / dv;
    var store = 0.0;
    if (z.store) {
        store = bu * city_overlap(v0, v1, 0.0, CITY_FLOOR_H) / dv;
    }
    if (z.blank) {
        return vec4<f32>(bu * vface - store, 0.0, 0.0, store);
    }
    let cu = city_overlap(u0, u1, f.margin + f32(z.core.x) * f.pitch,
                          f.margin + f32(z.core.y) * f.pitch) / du;
    let core = cu * city_overlap(v0, v1, CITY_FLOOR_H, 1e9) / dv;
    var pv = city_overlap(v0, v1, f32(z.crown) * CITY_FLOOR_H, 1e9) / dv;
    if (z.plant.y > 0 && v1 > v0) {
        pv = pv + (city_plant_below(z, v1 * CITY_PER_STOREY)
                   - city_plant_below(z, v0 * CITY_PER_STOREY))
                  * CITY_FLOOR_H / dv;
    }
    // Plant storeys all lie above storey 1, where a core takes its bays.
    let plant = pv * (bu - cu);
    let lu = city_overlap(u0, u1, f.margin,
                          f.margin + f32(z.loft.z) * f.pitch)
           + city_overlap(u0, u1,
                          f.margin + f32(f.nbays - z.loft.z) * f.pitch,
                          f.margin + f32(f.nbays) * f.pitch);
    let loft = lu / du * city_overlap(v0, v1, f32(z.loft.x) * CITY_FLOOR_H,
                                      f32(z.loft.y) * CITY_FLOOR_H) / dv;
    return vec4<f32>(core, plant, loft, store);
}

// The face's bays' share of the footprint: what the lattice's own mean
// holds besides its corner strips' wall.
fn city_zone_bays(f: CityFace, c: vec2<f32>, hb: vec2<f32>) -> f32 {
    let strip = city_bay_strip(f);
    return city_overlap(c.x - hb.x, c.x + hb.x, strip.x, strip.y)
           * city_overlap(c.y - hb.y, c.y + hb.y, f.z0, f.z1)
           / (4.0 * hb.x * hb.y);
}

struct CityFaceDraw {
    pipe_u: vec2<f32>,
    pipe_r: vec2<f32>,
    bal: CityBalconyRhythm,
    zones: CityZones,
}

fn city_face_draw(cc: CityCell, f: CityFace, st: CityShellStyle)
        -> CityFaceDraw {
    var d: CityFaceDraw;
    d.pipe_u = vec2<f32>(-1e9);
    d.pipe_r = vec2<f32>(0.05);
    if (st.pipes > 0) {
        d.pipe_u = vec2<f32>(city_shell_pipe_u(cc, f, st, 0),
                             city_shell_pipe_u(cc, f, st, 1));
        d.pipe_r = vec2<f32>(city_shell_pipe_r(cc, f, 0),
                             city_shell_pipe_r(cc, f, 1));
    }
    d.bal = CityBalconyRhythm(1, -1, false, 3);
    if (st.balcony) {
        d.bal = city_balcony_rhythm(cc, f);
    }
    d.zones = city_zones(cc, f);
    return d;
}

// --- the shell trace ---------------------------------------------------------

struct CityShellHit {
    t: f32,
    n: vec3<f32>,     // face-local normal
    part: i32,
}

fn city_shell_miss() -> CityShellHit {
    return CityShellHit(1e30, vec3<f32>(0.0), -1);
}

// Nearest entry into a face-local box, if any, as a candidate hit.
fn city_shell_try(best: CityShellHit, ol: vec3<f32>, dl: vec3<f32>,
                  il: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>,
                  part: i32) -> CityShellHit {
    let h = city_box_hit(ol, il, bmin, bmax);
    if (h.x <= h.y && h.x > 0.0 && h.x < best.t) {
        return CityShellHit(h.x, city_box_normal(ol + h.x * dl, bmin, bmax),
                            part);
    }
    return best;
}

// The things that stand proud of the wall, enumerated: feature k of the bay
// cell b, as a face-local box (shape 1) or vertical cylinder (shape 2), or
// nothing (shape 0). u extents may run past the cell; the caller clips them
// to it. `s` scales the depth of what stands proud of the wall; a unit set
// into the window keeps its back at the glass.
const CITY_SHELL_FEATURES: i32 = 19;

struct CityShellFeature {
    shape: i32,
    part: i32,
    lo: vec3<f32>,
    hi: vec3<f32>,
}

fn city_shell_feature(cc: CityCell, f: CityFace, st: CityShellStyle,
                      fd: CityFaceDraw, s: f32, b: CityBay, cb: f32, va: f32,
                      vb: f32, k: i32) -> CityShellFeature {
    var fe = CityShellFeature(0, 0, vec3<f32>(0.0), vec3<f32>(0.0));
    let ca = b.ca;
    let vfl = f32(b.iv) * CITY_FLOOR_H;
    let olo = city_bay_pane_lo(cc, f, b) - vec2<f32>(CITY_SHELL_FRAME);
    let ohi = city_bay_pane_hi(cc, f, b) + vec2<f32>(CITY_SHELL_FRAME);
    if (k == 0) {
        // The sill: the reveal's floor carried out past the wall.
        if (b.in_bay && st.sill_out > 0.0) {
            fe = CityShellFeature(1, CITY_PART_SILL,
                vec3<f32>(ca + olo.x - 0.06, vfl + olo.y - 0.07, 0.0),
                vec3<f32>(ca + ohi.x + 0.06, vfl + olo.y, st.sill_out * s));
        }
    } else if (k == 15) {
        // The hood over the opening: a cap wider than the window, standing
        // a little proud, as walk-ups carry them.
        if (b.in_bay && st.hood_out > 0.0) {
            fe = CityShellFeature(1, CITY_PART_TRIM,
                vec3<f32>(ca + olo.x - 0.10, vfl + ohi.y, 0.0),
                vec3<f32>(ca + ohi.x + 0.10, vfl + ohi.y + 0.16,
                          st.hood_out * s));
        }
    } else if (k == 1 || k == 2) {
        // A horizontal band (k 1) or the tier's cornice (k 2). A band runs
        // round the corner: the corner cells carry it past the face's end by
        // its own depth, where the neighbouring face's band meets it.
        let every = max(st.ledge_every, 1);
        let band = k == 1 && st.ledge_out > 0.0
                   && max(b.iv, 0) % every == 0;
        let corn = k == 2 && st.cornice_out > 0.0 && vb >= f.z1 - 1e-3;
        if (band || corn) {
            let out = select(st.ledge_out, st.cornice_out, corn) * s;
            let v0 = select(max(vfl + st.ledge_v0, f.z0), f.z1 - 0.55, corn);
            let v1 = select(min(vfl + st.ledge_v1, f.z1), f.z1, corn);
            let lo_u = select(ca, -out, b.iu < 0);
            let hi_u = select(cb, f.len + out, b.iu >= f.nbays);
            fe = CityShellFeature(1, CITY_PART_TRIM,
                vec3<f32>(lo_u, v0, 0.0), vec3<f32>(hi_u, v1, out));
        }
    } else if (k == 3 || k == 4) {
        // Fins or mullion caps on both edges of the bay; each cell owns the
        // half inside it.
        if (st.cap_out > 0.0 && b.in_bay) {
            let u0 = select(cb - st.cap_hw, ca, k == 3);
            fe = CityShellFeature(1, CITY_PART_TRIM,
                vec3<f32>(u0, va, 0.0),
                vec3<f32>(u0 + st.cap_hw, vb, st.cap_out * s));
        }
    } else if (k == 5 || k == 16) {
        let ac = city_shell_ac(cc, f, st, b);
        if (ac.on && k == 5) {
            fe = CityShellFeature(1, CITY_PART_AC,
                vec3<f32>(ca + ac.lo.x, vfl + ac.lo.y, ac.w0),
                vec3<f32>(ca + ac.hi.x, vfl + ac.hi.y, ac.out * s));
        } else if (ac.on && ac.in_window) {
            // The drip hood: a sheet a little wider and deeper than the unit.
            fe = CityShellFeature(1, CITY_PART_STEEL,
                vec3<f32>(ca + ac.lo.x - 0.05, vfl + ac.hi.y + 0.03, 0.0),
                vec3<f32>(ca + ac.hi.x + 0.05, vfl + ac.hi.y + 0.05,
                          (ac.out + 0.08) * s));
        }
    } else if (k == 17) {
        // A rain canopy over the window: corrugated sheet on two brackets,
        // on the buildings whose draw says so.
        if (b.in_bay && st.canopy_out > 0.0) {
            fe = CityShellFeature(1, CITY_PART_CANOPY,
                vec3<f32>(ca + olo.x - 0.20, vfl + ohi.y + 0.22, 0.0),
                vec3<f32>(ca + ohi.x + 0.20, vfl + ohi.y + 0.26,
                          st.canopy_out * s));
        }
    } else if (k < 10) {
        // Drainpipes (k 6, 8) and their brackets (k 7, 9), strapping the
        // pipe to the wall every 2.4 m.
        let j = (k - 6) >> 1;
        if (j < st.pipes) {
            let up = select(fd.pipe_u.x, fd.pipe_u.y, j == 1);
            let pr = select(fd.pipe_r.x, fd.pipe_r.y, j == 1);
            let wc = (0.04 + pr) * s;
            let rw = pr * max(s, 1e-3);
            if ((k & 1) == 0) {
                fe = CityShellFeature(2, CITY_PART_PIPE,
                    vec3<f32>(up, max(va, f.z0), wc),
                    vec3<f32>(pr, min(vb, f.z1 - 0.3), rw));
            } else {
                let vbk = floor(max(va, f.z0) / 2.4) * 2.4 + 1.2;
                if (vbk > va && vbk < vb && up > -1e8) {
                    fe = CityShellFeature(1, CITY_PART_STEEL,
                        vec3<f32>(up - pr - 0.02, vbk, 0.0),
                        vec3<f32>(up + pr + 0.02, vbk + 0.05,
                                  wc + rw + 0.01 * s));
                }
            }
        }
    } else if (k == 18) {
        // A core strip's front, standing proud of the wall over its bays:
        // each cell owns its part, so the strip is one slab.
        if (b.zone == CITY_ZONE_CORE) {
            fe = CityShellFeature(1, CITY_PART_CORE,
                vec3<f32>(ca, max(va, f.z0), 0.0),
                vec3<f32>(cb, vb, CITY_CORE_OUT * s));
        }
    } else {
        // A balcony (k 10-14): slab, front railing, its top rail and the two
        // side panels, on the bays and storeys city_balcony gives.
        let bal = city_balcony_at(fd.bal, f, st, b);
        if (bal.x > 0.0) {
            let out = CITY_BALCONY_OUT * s;
            let u0 = ca + 0.08;
            let u1 = cb - 0.08;
            if (k == 10) {
                fe = CityShellFeature(1, CITY_PART_SLAB,
                    vec3<f32>(u0, vfl - 0.02, 0.0),
                    vec3<f32>(u1, vfl + 0.16, out));
            } else if (k == 11) {
                // Glass (a solid panel) or bars (an alpha-tested plane).
                fe = CityShellFeature(select(3, 1, bal.y > 0.5),
                    select(CITY_PART_STEEL, CITY_PART_PANEL, bal.y > 0.5),
                    vec3<f32>(u0, vfl + 0.16, out - 0.04 * s),
                    vec3<f32>(u1, vfl + 1.02, out - 0.01 * s));
            } else if (k == 12) {
                fe = CityShellFeature(1, CITY_PART_STEEL,
                    vec3<f32>(u0, vfl + 1.02, out - 0.06 * s),
                    vec3<f32>(u1, vfl + 1.07, out));
            } else {
                let ue = select(u1 - 0.04, u0, k == 13);
                fe = CityShellFeature(1, CITY_PART_STEEL,
                    vec3<f32>(ue, vfl + 0.16, 0.0),
                    vec3<f32>(ue + 0.04, vfl + 1.07, out));
            }
        }
    }
    return fe;
}

// --- balconies ---------------------------------------------------------------
//
// Residential mid-rise, a third of it: stacked balconies in columns on every
// second or third bay, from the first floor to the one under the roof, glass
// railed or barred by the building's draw. Bay features, so the cell walk
// finds them; their flat twin (city_balcony_cover) paints the slab line and
// the railing and dissolves them into their mean coverage.
const CITY_BALCONY_OUT: f32 = 1.10;
const CITY_BALCONY_BAR: f32 = 0.11;

// Whether a building has balconies at all (never with a fire escape).
fn city_balcony_building(cc: CityCell) -> bool {
    let h = city_rand4(cc.seed ^ vec2<u32>(0x27220a95u, 0x165667b1u));
    return (cc.win_style == 0 || cc.win_style == 4 || cc.win_style == 1)
        && cc.land_use == CITY_USE_URBAN
        && cc.height >= 12.0 && cc.height <= 120.0 && h.x < 0.33
        && !city_escape_building(cc);
}

// A face's balcony columns: every `period`-th bay from `phase`, whether
// they are glass railed, and where on the face they stand: a balcony stack
// is a region of the facade, not all of it. `zone` 0 keeps them to the two
// bays at each corner (every bay there), 1 to the middle half of the face,
// 2 the whole face, 3 none on this face.
struct CityBalconyRhythm {
    period: i32,
    phase: i32,
    glass: bool,
    zone: i32,
}

fn city_balcony_rhythm(cc: CityCell, f: CityFace) -> CityBalconyRhythm {
    let h = city_rand4(cc.seed ^ vec2<u32>(0x9e3779b1u, 0x7f4a7c15u)
                       ^ vec2<u32>(f.salt));
    let period = 2 + i32(floor(h.x * 1.999));
    return CityBalconyRhythm(period, i32(floor(h.y * f32(period) * 0.999)),
                             h.z < 0.5, i32(floor(h.w * 3.999)));
}

// This bay's balcony: x > 0 when it has one, y > 0.5 for a glass railing.
fn city_balcony_at(rh: CityBalconyRhythm, f: CityFace, st: CityShellStyle,
                   b: CityBay) -> vec2<f32> {
    if (!st.balcony || !b.in_bay || !f.aligned || b.iv < 1
        || f32(b.iv + 1) * CITY_FLOOR_H > f.z1 - 1.0 || rh.zone == 3
        || f.tier == CITY_CROWN_TIER) {
        return vec2<f32>(0.0);
    }
    let corner = b.iu < 2 || b.iu >= f.nbays - 2;
    let middle = 4 * b.iu >= f.nbays && 4 * b.iu < 3 * f.nbays;
    let on = select(select(b.iu % rh.period == rh.phase && middle,
                           b.iu % rh.period == rh.phase, rh.zone == 2),
                    corner, rh.zone == 0);
    if (!on) {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(1.0, select(0.0, 1.0, rh.glass));
}

fn city_balcony(cc: CityCell, f: CityFace, st: CityShellStyle,
                b: CityBay) -> vec2<f32> {
    return city_balcony_at(city_balcony_rhythm(cc, f), f, st, b);
}

// The balcony's coverage of the wall seen face-on at bay-local (bu, bv),
// for the flat twin, and the albedo it paints there.
fn city_balcony_cover(cc: CityCell, f: CityFace, st: CityShellStyle,
                      b: CityBay, fp_eff: f32) -> f32 {
    let bal = city_balcony(cc, f, st, b);
    if (bal.x <= 0.0 || b.bu < 0.08 || b.bu > f.pitch - 0.08) {
        return 0.0;
    }
    let slab = b.bv < 0.16 || b.bv > CITY_FLOOR_H - 0.02;
    let rail = b.bv > 0.16 && b.bv < 1.07;
    let bars = rail && (bal.y > 0.5 || b.bv > 1.02
        || fract(b.bu / CITY_BALCONY_BAR) < 0.2);
    let c = select(0.0, 1.0, slab || bars);
    let mean = (0.18 + select(0.91 * 0.2 + 0.05, 0.91, bal.y > 0.5))
               / CITY_FLOOR_H;
    return mix(c, mean, smoothstep(0.10, 0.45, fp_eff));
}

// Everything in one bay cell: the wall plane with its opening, the recess
// behind it, and whatever stands proud of the wall. `s` scales the depth of
// what stands proud (see the LOD note at the top); the recess keeps its
// own. u in [ca, cb], v in [va, vb].
fn city_shell_cell(ol: vec3<f32>, dl: vec3<f32>, il: vec3<f32>,
                   cc: CityCell, f: CityFace, st: CityShellStyle,
                   fd: CityFaceDraw, s: f32, fp_cell: f32, b_walk: CityBay,
                   cb: f32, va: f32, vb: f32, ta: f32, tb: f32)
        -> CityShellHit {
    var best = city_shell_miss();
    let b = city_bay_zone(fd.zones, f, b_walk);
    let ca = b.ca;
    let plo = city_bay_pane_lo(cc, f, b);
    let phi = city_bay_pane_hi(cc, f, b);
    let olo = plo - vec2<f32>(CITY_SHELL_FRAME);   // the opening
    let ohi = phi + vec2<f32>(CITY_SHELL_FRAME);
    let vfl = f32(b.iv) * CITY_FLOOR_H;

    // A panel on the wall (a screen, a sign box) is all there is where it
    // hangs: judged where the ray is nearest the wall in this cell.
    var tq = tb;
    if (dl.z < -1e-6) {
        tq = clamp(-ol.z / dl.z, ta, tb);
    }
    let pq = ol + tq * dl;
    let covered = city_face_covered(cc, f, pq.x, pq.y, fp_cell);

    // The wall plane, and through the opening the recess behind it.
    if (dl.z < -1e-6) {
        let tw = -ol.z / dl.z;
        let ew = city_shell_eps(tw);
        if (tw > 0.0 && tw >= ta - ew && tw <= tb + ew) {
            let pw = ol + tw * dl;
            let on_face = pw.x >= 0.0 && pw.x <= f.len
                       && pw.y >= f.z0 && pw.y <= f.z1;
            let bu = pw.x - ca;
            let bv = pw.y - vfl;
            let opening = b.in_bay && !covered && bu > olo.x && bu < ohi.x
                       && bv > olo.y && bv < ohi.y;
            if (on_face && !opening) {
                best = CityShellHit(tw, vec3<f32>(0.0, 0.0, 1.0),
                                    CITY_PART_WALL);
            } else if (on_face) {
                // The recess keeps its true depth at every footprint: the
                // flat facade sees its glass through the opening
                // (city_recess_off, city_windows_seen), so the two agree
                // wherever they meet. Only relief standing proud of the wall
                // flattens with s.
                let d = st.reveal;
                let rmin = vec3<f32>(ca + olo.x, vfl + olo.y, -d);
                let rmax = vec3<f32>(ca + ohi.x, vfl + ohi.y, 0.0);
                // Exit of the recess from inside: the nearest far plane.
                let tx = city_exit_t(ol, dl, rmin, rmax);
                var te = min(min(tx.x, tx.y), tx.z);
                var nl = vec3<f32>(0.0, 0.0, 1.0);
                var part = CITY_PART_GLASS;
                if (tx.x <= tx.y && tx.x <= tx.z) {
                    nl = vec3<f32>(-sign(dl.x), 0.0, 0.0);
                    part = CITY_PART_REVEAL;
                } else if (tx.y <= tx.z) {
                    nl = vec3<f32>(0.0, -sign(dl.y), 0.0);
                    part = select(CITY_PART_REVEAL, CITY_PART_SILL, dl.y < 0.0);
                } else {
                    let pg = ol + te * dl;
                    let gu = pg.x - ca;
                    let gv = pg.y - vfl;
                    let in_pane = gu > plo.x && gu < phi.x
                               && gv > plo.y && gv < phi.y;
                    part = select(CITY_PART_FRAME, CITY_PART_GLASS, in_pane);
                }
                var rec = CityShellHit(te, nl, part);
                // Mullion or meeting rail, standing proud of the glass.
                let md = min(0.07 * s, d);
                if (st.mull == 1) {
                    let um = ca + 0.5 * (plo.x + phi.x);
                    rec = city_shell_try(rec, ol, dl, il,
                        vec3<f32>(um - 0.03, vfl + plo.y, -d),
                        vec3<f32>(um + 0.03, vfl + phi.y, -d + md),
                        CITY_PART_FRAME);
                } else if (st.mull == 2) {
                    let vm = vfl + 0.5 * (plo.y + phi.y);
                    rec = city_shell_try(rec, ol, dl, il,
                        vec3<f32>(ca + plo.x, vm - 0.03, -d),
                        vec3<f32>(ca + phi.x, vm + 0.03, -d + md),
                        CITY_PART_FRAME);
                }
                best = rec;
            }
        }
    }
    if (s <= 0.0 || covered) {
        return best;
    }

    // Proud of the wall: one feature at a time, rolled in a loop so only the
    // feature under test is live (the shell shares its kernel with the cloud
    // march, and every register it holds is one the march loses). Each is
    // clipped to the cell, so the DDA's union of cells is the whole feature
    // exactly once.
    let clo = max(max(b.ca, -st.depth), f.col_u0);
    let chi = min(min(cb, f.len + st.depth), f.col_u1);
    for (var k: i32 = 0; k < CITY_SHELL_FEATURES; k = k + 1) {
        let fe = city_shell_feature(cc, f, st, fd, s, b, cb, va, vb, k);
        if (fe.shape == 1) {
            best = city_shell_try(best, ol, dl, il,
                                  vec3<f32>(max(fe.lo.x, clo), fe.lo.yz),
                                  vec3<f32>(min(fe.hi.x, chi), fe.hi.y,
                                            min(fe.hi.z, f.col_w)),
                                  fe.part);
        } else if (fe.shape == 3) {
            // A thin plane that is solid only on its vertical bars.
            let blo = vec3<f32>(max(fe.lo.x, clo), fe.lo.yz);
            let bhi = vec3<f32>(min(fe.hi.x, chi), fe.hi.y,
                                min(fe.hi.z, f.col_w));
            let hb = city_box_hit(ol, il, blo, bhi);
            if (hb.x <= hb.y && hb.x > 0.0 && hb.x < best.t) {
                let pb = ol + hb.x * dl;
                if (fract((pb.x - b.ca) / CITY_BALCONY_BAR) < 0.2) {
                    best = CityShellHit(hb.x, city_box_normal(pb, blo, bhi),
                                        fe.part);
                }
            }
        } else if (fe.shape == 2) {
            // A vertical elliptic cylinder in (u, w): axis at (lo.x, lo.z),
            // radius hi.x along the wall and hi.z out from it (the shell's
            // flattening squashes a pipe's depth, never its width), over v in
            // [lo.y, hi.y], clipped to the cell in u. Its top is capped; its
            // foot stands on the ground or the tier below.
            let du = ol.x - fe.lo.x;
            let dw = (ol.z - fe.lo.z) * (fe.hi.x / fe.hi.z);
            let ew = dl.z * (fe.hi.x / fe.hi.z);
            let qa = dl.x * dl.x + ew * ew;
            let qb = du * dl.x + dw * ew;
            let qc = du * du + dw * dw - fe.hi.x * fe.hi.x;
            let disc = qb * qb - qa * qc;
            if (disc > 0.0 && qa > 1e-12) {
                // @domain: the discriminant is tested positive (or non-negative) just above
                let sq = sqrt(disc);
                let tp = (-qb - sq) / qa;
                let pp = ol + tp * dl;
                if (tp > 0.0 && tp < best.t && pp.y > fe.lo.y
                    && pp.y < fe.hi.y && pp.x >= clo && pp.x <= chi) {
                    let rw = fe.hi.x / fe.hi.z;
                    // @domain: the hit lies on the rail's elliptic cylinder, never on its axis
                    best = CityShellHit(tp, normalize(vec3<f32>(
                        pp.x - fe.lo.x, 0.0, (pp.z - fe.lo.z) * rw * rw)),
                        fe.part);
                } else if (pp.y >= fe.hi.y && dl.y < 0.0) {
                    // Came in over the top: the cap, if the ray crosses it
                    // inside the section.
                    let tc = (fe.hi.y - ol.y) / dl.y;
                    let pc = ol + tc * dl;
                    let cu = pc.x - fe.lo.x;
                    let cw = (pc.z - fe.lo.z) * (fe.hi.x / fe.hi.z);
                    if (tc > 0.0 && tc < best.t
                        && cu * cu + cw * cw < fe.hi.x * fe.hi.x
                        && pc.x >= clo && pc.x <= chi) {
                        best = CityShellHit(tc, vec3<f32>(0.0, 1.0, 0.0),
                                            fe.part);
                    }
                }
            }
        }
    }
    return best;
}

// --- the fire escape ------------------------------------------------------------
//
// The walk-up's iron: a two-bay column of platforms, one per storey, joined
// by switchback stairs, railed along the front and at both ends, from the
// first floor to the last. It spans cells, so it is one structure per face
// rather than a bay feature: tested once, beside the cell walk, nearest hit
// wins. Its flat twin (city_facade_flat) paints the same lattice on the wall
// and dissolves it into its mean coverage, so where the shell flattens into
// the flat facade the escape does too.
const CITY_ESCAPE_OUT: f32 = 1.25;     // platform depth off the wall (m)
const CITY_ESCAPE_BAR: f32 = 0.12;     // baluster pitch
const CITY_ESCAPE_COVER: f32 = 0.26;   // mean lattice coverage seen face-on
const CITY_ESCAPE_ALBEDO: vec3<f32> = vec3<f32>(0.06, 0.055, 0.05); // black paint

struct CityEscape {
    on: bool,
    u0: f32,
    u1: f32,
    v0: f32,     // first platform
    v1: f32,     // last platform
    col_w: f32,  // the lot column's reach off the wall
}

// Whether a building carries an escape at all: brick or concrete walk-ups
// of five to eighteen storeys, a quarter of them.
fn city_escape_building(cc: CityCell) -> bool {
    let h = city_rand4(cc.seed ^ vec2<u32>(0x1b873593u, 0xcc9e2d51u));
    return (cc.win_style == 4 || cc.win_style == 0)
        && cc.land_use == CITY_USE_URBAN
        && cc.height >= 18.0 && cc.height <= 66.0 && h.x < 0.25;
}

fn city_escape(cc: CityCell, f: CityFace, st: CityShellStyle) -> CityEscape {
    var e = CityEscape(false, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (!st.escape || !f.aligned || f.tier != 0 || f.nbays < 4) {
        return e;
    }
    // Not on a solid side, nor across a core strip.
    let z = city_zones(cc, f);
    if (z.blank || z.core.y > z.core.x) {
        return e;
    }
    let h = city_rand4(cc.seed ^ vec2<u32>(0x85ebca77u, 0xc2b2ae3du));
    if (i32(floor(h.x * 3.999)) != f.side) {
        return e;
    }
    let k0 = 1 + i32(floor(h.y * f32(f.nbays - 3)));
    e.on = true;
    e.u0 = f.margin + f32(k0) * f.pitch - 0.15;
    e.u1 = f.margin + f32(k0 + 2) * f.pitch + 0.15;
    e.v0 = f.z0 + CITY_FLOOR_H;
    e.v1 = f.z0 + (floor((f.z1 - f.z0) / CITY_FLOOR_H) - 1.0) * CITY_FLOOR_H;
    e.col_w = f.col_w;
    return e;
}

// The stair of storey k: a steel plate rising from platform k to k+1,
// switching direction each storey, against the wall.
fn city_escape_stair(ol: vec3<f32>, dl: vec3<f32>, e: CityEscape, k: f32,
                     s: f32, best: CityShellHit) -> CityShellHit {
    let run = min(e.u1 - e.u0 - 0.9, 2.4);
    let up = (k - 2.0 * floor(0.5 * k)) < 0.5;
    let ua = select(e.u1 - 0.3, e.u0 + 0.3, up);
    let ub = select(ua - run, ua + run, up);
    let va = e.v0 + k * CITY_FLOOR_H;
    let d2 = vec2<f32>(ub - ua, CITY_FLOOR_H);
    let len = length(d2);
    let ea = d2 / len;
    let eb = vec2<f32>(-ea.y, ea.x);
    // The ray in the plate's own frame: along the run, across the plate,
    // out from the wall.
    let rel = ol.xy - vec2<f32>(ua, va);
    let o2 = vec3<f32>(dot(rel, ea), dot(rel, eb), ol.z);
    let d3 = vec3<f32>(dot(dl.xy, ea), dot(dl.xy, eb), dl.z);
    let i3 = vec3<f32>(
        select(1e30, 1.0 / d3.x, abs(d3.x) > 1e-9),
        select(1e30, 1.0 / d3.y, abs(d3.y) > 1e-9),
        select(1e30, 1.0 / d3.z, abs(d3.z) > 1e-9));
    let bmin = vec3<f32>(0.0, -0.07, 0.12 * s);
    let bmax = vec3<f32>(len, 0.0, 0.64 * s);
    let hh = city_box_hit(o2, i3, bmin, bmax);
    if (hh.x <= hh.y && hh.x > 0.0 && hh.x < best.t) {
        let nl = city_box_normal(o2 + hh.x * d3, bmin, bmax);
        let nuv = ea * nl.x + eb * nl.y;
        return CityShellHit(hh.x, vec3<f32>(nuv, nl.z), CITY_PART_STEEL);
    }
    return best;
}

fn city_shell_escape(ol: vec3<f32>, dl: vec3<f32>, il: vec3<f32>,
                     e: CityEscape, s: f32) -> CityShellHit {
    var best = city_shell_miss();
    let out = min(CITY_ESCAPE_OUT * s, e.col_w);
    let hb = city_box_hit(ol, il, vec3<f32>(e.u0, e.v0 - 0.08, 0.0),
                          vec3<f32>(e.u1, e.v1 + 1.05, out));
    if (hb.x > hb.y || hb.y <= 0.0) {
        return best;
    }
    // The storeys the ray passes through inside the structure, walked from
    // the one it enters in the direction it travels (one storey behind it
    // too, for the rails and stair that reach up out of it), so the nearest
    // are always among those tested.
    let va = ol.y + max(hb.x, 0.0) * dl.y;
    let k_top = (e.v1 - e.v0) / CITY_FLOOR_H;
    let k_in = clamp(floor((va - e.v0) / CITY_FLOOR_H), 0.0, k_top);
    let step = select(-1.0, 1.0, dl.y >= 0.0);
    for (var i: i32 = 0; i < 6; i = i + 1) {
        let k = k_in + step * (f32(i) - 1.0);
        if (k < 0.0 || k > k_top + 0.5) {
            continue;
        }
        let vk = e.v0 + k * CITY_FLOOR_H;
        // The platform grating, and the rails along its front and ends.
        best = city_shell_try(best, ol, dl, il,
            vec3<f32>(e.u0, vk - 0.06, 0.05 * s),
            vec3<f32>(e.u1, vk, out), CITY_PART_STEEL);
        for (var r: i32 = 0; r < 2; r = r + 1) {
            let vr = vk + select(0.50, 0.97, r == 1);
            best = city_shell_try(best, ol, dl, il,
                vec3<f32>(e.u0, vr, out - 0.05 * s),
                vec3<f32>(e.u1, vr + 0.04, out), CITY_PART_STEEL);
            best = city_shell_try(best, ol, dl, il,
                vec3<f32>(e.u0, vr, 0.0),
                vec3<f32>(e.u0 + 0.04, vr + 0.04, out), CITY_PART_STEEL);
            best = city_shell_try(best, ol, dl, il,
                vec3<f32>(e.u1 - 0.04, vr, 0.0),
                vec3<f32>(e.u1, vr + 0.04, out), CITY_PART_STEEL);
        }
        // Balusters: a thin plane that is solid only on the bars.
        let bmin = vec3<f32>(e.u0, vk, out - 0.04 * s);
        let bmax = vec3<f32>(e.u1, vk + 0.97, out - 0.01 * s);
        let hbar = city_box_hit(ol, il, bmin, bmax);
        if (hbar.x <= hbar.y && hbar.x > 0.0 && hbar.x < best.t) {
            let pb = ol + hbar.x * dl;
            if (fract((pb.x - e.u0) / CITY_ESCAPE_BAR) < 0.18) {
                best = CityShellHit(hbar.x, city_box_normal(pb, bmin, bmax),
                                    CITY_PART_STEEL);
            }
        }
        if (vk < e.v1 - 0.5) {
            best = city_escape_stair(ol, dl, e, k, s, best);
        }
    }
    return best;
}

// The lattice's coverage of the wall seen face-on at (u, v), for the flat
// twin: platforms, rails, bars and stair, dissolving into their mean.
fn city_escape_cover(e: CityEscape, uu: f32, v: f32, fp_eff: f32) -> f32 {
    if (!e.on || uu < e.u0 || uu > e.u1 || v < e.v0 - 0.08
        || v > e.v1 + 1.05) {
        return 0.0;
    }
    let k = floor((v - e.v0 + 0.08) / CITY_FLOOR_H);
    let dv = v - (e.v0 + k * CITY_FLOOR_H);
    let rails = (dv > -0.06 && dv < 0.0) || abs(dv - 0.52) < 0.02
             || abs(dv - 0.99) < 0.02;
    let bars = dv > 0.0 && dv < 0.97
             && fract((uu - e.u0) / CITY_ESCAPE_BAR) < 0.18;
    let run = min(e.u1 - e.u0 - 0.9, 2.4);
    let up = (k - 2.0 * floor(0.5 * k)) < 0.5;
    let ua = select(e.u1 - 0.3, e.u0 + 0.3, up);
    let x = (uu - ua) * select(-1.0, 1.0, up);
    let stair = k * CITY_FLOOR_H + e.v0 < e.v1 - 0.5 && x > 0.0 && x < run
             && abs(dv - x * CITY_FLOOR_H / run) < 0.09;
    let c = select(0.0, 1.0, rails || bars || stair);
    return mix(c, CITY_ESCAPE_COVER, smoothstep(0.10, 0.45, fp_eff));
}

// One face's shell layer, [t_lo, t_hi] of the ray inside it: walk its bay
// cells in order and return the first hit.
fn city_shell_face(o: vec3<f32>, dir: vec3<f32>, cc: CityCell, f: CityFace,
                   st: CityShellStyle, s: f32, pixel_angle: f32,
                   t_lo: f32, t_hi: f32) -> CityShellHit {
    let ol = city_face_local(f, o);
    let dl = city_face_dir(f, dir);
    let il = vec3<f32>(
        select(1e30, 1.0 / dl.x, abs(dl.x) > 1e-9),
        select(1e30, 1.0 / dl.y, abs(dl.y) > 1e-9),
        select(1e30, 1.0 / dl.z, abs(dl.z) > 1e-9));
    // The escape, if this face carries one: its hit bounds the walk.
    let esc = city_escape(cc, f, st);
    var fe_hit = city_shell_miss();
    if (esc.on && s > 0.0) {
        fe_hit = city_shell_escape(ol, dl, il, esc, s);
    }
    let fd = city_face_draw(cc, f, st);
    let p0 = ol + (t_lo + city_shell_eps(t_lo)) * dl;
    var b = city_bay(f, p0.x, p0.y);
    var t = t_lo;
    for (var i: i32 = 0; i < CITY_SHELL_MAX_CELLS; i = i + 1) {
        if (t >= fe_hit.t) {
            return fe_hit;
        }
        var cb = b.ca + f.pitch;
        if (b.iu < 0) { cb = f.margin; }
        if (b.iu >= f.nbays) { cb = 1e9; }
        let va = f32(b.iv) * CITY_FLOOR_H;
        let vb = va + CITY_FLOOR_H;
        let fu = select(b.ca, cb, dl.x > 0.0);
        let fv = select(va, vb, dl.y > 0.0);
        let tu = select((fu - ol.x) / select(1.0, dl.x, abs(dl.x) > 1e-9),
                        1e30, abs(fu) > 1e8 || abs(dl.x) <= 1e-9);
        let tv = select((fv - ol.y) / select(1.0, dl.y, abs(dl.y) > 1e-9),
                        1e30, abs(dl.y) <= 1e-9);
        let t_exit = min(min(tu, tv), t_hi);
        let h = city_shell_cell(ol, dl, il, cc, f, st, fd, s,
                                pixel_angle * t, b, cb, max(va, f.z0),
                                min(vb, f.z1), t, t_exit);
        if (h.part >= 0) {
            if (fe_hit.t < h.t) {
                return fe_hit;
            }
            return h;
        }
        if (t_exit >= t_hi) {
            return fe_hit;
        }
        t = t_exit;
        if (tu < tv) {
            b.iu = b.iu + select(-1, 1, dl.x > 0.0);
            b.ca = f.margin + f32(b.iu) * f.pitch;
            if (b.iu < 0) { b.ca = -1e9; }
            if (b.iu >= f.nbays) { b.ca = f.len - f.margin; }
            b.in_bay = b.iu >= 0 && b.iu < f.nbays;
        } else {
            b.iv = b.iv + select(-1, 1, dl.y > 0.0);
        }
    }
    // Out of cells before the layer ended. city_shell_scale sizes the relief
    // so a walk of the scaled layer fits the cap, so this is a numerical
    // edge; even so, the cell holding the wall crossing is traced in full
    // (its opening, recess and relief), never painted over as plain wall.
    if (dl.z < -1e-6) {
        let tw = -ol.z / dl.z;
        let pw = ol + tw * dl;
        if (tw > 0.0 && tw < fe_hit.t && pw.x >= 0.0 && pw.x <= f.len
            && pw.y >= f.z0 && pw.y <= f.z1) {
            let bw = city_bay(f, pw.x, pw.y);
            var cbw = bw.ca + f.pitch;
            if (bw.iu < 0) { cbw = f.margin; }
            if (bw.iu >= f.nbays) { cbw = 1e9; }
            let vaw = f32(bw.iv) * CITY_FLOOR_H;
            let hw = city_shell_cell(ol, dl, il, cc, f, st, fd, s,
                                     pixel_angle * tw, bw, cbw,
                                     max(vaw, f.z0),
                                     min(vaw + CITY_FLOOR_H, f.z1),
                                     tw - city_shell_eps(tw),
                                     tw + city_shell_eps(tw));
            if (hw.part >= 0 && hw.t < fe_hit.t) {
                return hw;
            }
        }
    }
    return fe_hit;
}

// The shell depth scale for one face along one ray: 1 resolved, 0 flat.
// Relief is seen in profile as often as face-on (an air conditioner ahead,
// flying along the wall), so its DEPTH dissolves by the plain footprint at
// t_ref — where the ray meets the wall plane, or for a ray that never will,
// where it enters the layer. The in-plane pattern keeps the core's
// foreshortened fp_eff (city_shell_shade). And the cell walk must end: a ray
// skimming the wall crosses depth * (|du|/pitch + |dv|/floor) / |dw| cells
// of a layer `depth` thick, so relief flattens as that count nears the cap
// (only the far reach of a wall you are skimming ever sees this). The
// block walk's own cap ends the reach too (city_shell_cap), so the relief
// also flattens over the last quarter before it.
fn city_shell_scale(pixel_angle: f32, t_ref: f32, dl: vec3<f32>,
                    depth: f32, pitch: f32, t_cap: f32) -> f32 {
    let s_fp = min(1.0 - smoothstep(CITY_SHELL_FP_MORPH, CITY_SHELL_FP_FLAT,
                                    pixel_angle * max(t_ref, 0.0)),
                   1.0 - city_shell_cap_fade(t_ref, t_cap));
    let cells = depth * (abs(dl.x) / pitch + abs(dl.y) / CITY_FLOOR_H)
                / max(abs(dl.z), 1e-6);
    let s_walk = clamp(f32(CITY_SHELL_MAX_CELLS - 2) / max(cells, 1e-6),
                       0.0, 1.0);
    return min(s_fp, s_walk);
}

struct CityBoxHit {
    t: f32,
    normal: vec3<f32>,
    kind: i32,
}

// One building box, as the DDA sees it. Beyond the shell's reach (or with
// `shell` false) this is exactly the old slab test: kind 1 facade or kind 2
// roof on the box's entry face. Within it the box is inflated by the
// building's shell depth and each facing wall's layer is traced.
fn city_building_box(o: vec3<f32>, dir: vec3<f32>, inv3: vec3<f32>,
                     t_cap: f32,
                     bmin: vec3<f32>, bmax: vec3<f32>, cc: CityCell,
                     tier: i32, st: CityShellStyle,
                     pixel_angle: f32) -> CityBoxHit {
    var res = CityBoxHit(1e30, vec3<f32>(0.0), -1);
    let hc = city_box_hit(o, inv3, bmin, bmax);
    let core = hc.x <= hc.y && hc.x > 0.0;
    var core_n = vec3<f32>(0.0);
    if (core) {
        core_n = city_box_normal(o + hc.x * dir, bmin, bmax);
        res = CityBoxHit(hc.x, core_n, select(1, 2, core_n.z > 0.5));
    }
    let imin = vec3<f32>(max(bmin.xy - vec2<f32>(st.depth), cc.lot_min),
                         bmin.z);
    let imax = vec3<f32>(min(bmax.xy + vec2<f32>(st.depth), cc.lot_max),
                         bmax.z);
    let hi = city_box_hit(o, inv3, imin, imax);
    if (hi.x > hi.y || hi.y <= 0.0
        || pixel_angle * max(hi.x, 0.0) >= CITY_SHELL_FP_FLAT
        || hi.x >= t_cap) {
        return res;
    }
    var best = city_shell_miss();
    var best_side = 0;
    // The side whose wall the slab hit is on (-1: a roof, or no slab hit).
    // Only that wall's own shell may put the hit behind the slab's (the
    // glass behind its reveal); every other side's must stand in front of
    // it. Without that, a ray whose facing walk came up empty found the far
    // wall's sills from inside the building.
    var core_side = -1;
    if (core && abs(core_n.z) < 0.5) {
        core_side = select(select(3, 2, core_n.y > 0.0),
                           select(1, 0, core_n.x > 0.0),
                           abs(core_n.x) > abs(core_n.y));
    }
    for (var side: i32 = 0; side < 4; side = side + 1) {
        // The layer: the box grown by the shell depth, its inner bound on
        // this face's axis pinned to the wall plane. Every face whose layer
        // the ray passes through is walked — a ray running parallel to a
        // wall, or away from it, still meets the side of what stands on it.
        var lmin = imin;
        var lmax = imax;
        if (side == 0) { lmin.x = bmax.x; }
        if (side == 1) { lmax.x = bmin.x; }
        if (side == 2) { lmin.y = bmax.y; }
        if (side == 3) { lmax.y = bmin.y; }
        let hl = city_box_hit(o, inv3, lmin, lmax);
        // A side other than the slab hit's can only matter in front of it.
        var t_accept = best.t;
        if (core && side != core_side) {
            t_accept = min(t_accept, res.t);
        }
        if (hl.x > hl.y || hl.y <= 0.0 || max(hl.x, 0.0) >= t_accept) {
            continue;
        }
        let f = city_face_make(cc, bmin, bmax, side, tier);
        let dl = city_face_dir(f, dir);
        var t_ref = max(hl.x, 0.0);
        if (dl.z < -1e-3) {
            t_ref = (f.wp - dot(o.xy, f.n)) / dl.z;
        }
        let s = city_shell_scale(pixel_angle, t_ref, dl, st.depth, f.pitch,
                                 t_cap);
        if (s <= 0.0) {
            continue;   // flat: the core slab hit already is this wall
        }
        // Walk only the layer the SCALED relief occupies: the full-depth
        // layer's outer part is empty once s < 1, and walking it would spend
        // the cell budget city_shell_scale sized for the relief.
        let dd = st.depth * s;
        var smin = vec3<f32>(max(bmin.xy - vec2<f32>(dd), cc.lot_min),
                             bmin.z);
        var smax = vec3<f32>(min(bmax.xy + vec2<f32>(dd), cc.lot_max),
                             bmax.z);
        if (side == 0) { smin.x = bmax.x; }
        if (side == 1) { smax.x = bmin.x; }
        if (side == 2) { smin.y = bmax.y; }
        if (side == 3) { smax.y = bmin.y; }
        let hs = city_box_hit(o, inv3, smin, smax);
        if (hs.x > hs.y || hs.y <= 0.0) {
            continue;
        }
        let sh = city_shell_face(o, dir, cc, f, st, s, pixel_angle,
                                 max(hs.x, 0.0), hs.y);
        if (sh.t < t_accept) {
            best = sh;
            best_side = side;
        }
    }
    if (best.part >= 0) {
        let f = city_face_make(cc, bmin, bmax, best_side, tier);
        return CityBoxHit(best.t, city_face_world(f, best.n),
                          CITY_KIND_SHELL + best.part
                          + 16 * (best_side + 4 * tier));
    }
    return res;
}
