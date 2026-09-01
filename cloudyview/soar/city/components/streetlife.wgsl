// streetlife — the city at eye level: the poles the light already comes
// from, the cars parked under them, and the bins behind those.
//
// The core already lights the asphalt: city_street_pools puts a sodium pool
// every CITY_LAMP_SPACING (26 m) along two lines set CITY_LAMP_OFFSET (2.5 m)
// in from each block edge, three times brighter on avenues. Those pools had
// no lamps over them. This component puts the lamps there — on exactly that
// lattice, derived from the same arithmetic, so the light and its source
// cannot drift apart. Everything else here is what stands in that light.
//
// LAYOUT. A cell owns four KERBS, one per plot edge. Each kerb carries, in
// its own across-coordinate:
//   * the lamp line, at the block edge +- CITY_LAMP_OFFSET — the core's
//     lattice, not a new one. A mast at every lattice point inside the cell,
//     with the luminaire arm reaching AWAY from the plot, out over the
//     roadway, which is where a real one hangs.
//   * the parking line, CAR_PARK_OFF in from the plot edge (the kerb proper).
//     Slots on a 7 m lattice in world space, so a run of cars lines up
//     across cell boundaries; ~35% occupied, scaled by the cascade the way
//     the streetlights and the air traffic are, and ~10% of the rest hold a
//     dumpster shoved against the wall instead.
// Every prop lives strictly inside the cell that draws it, because the DDA
// tests a cell only from the side it enters (the rule aircars states). For
// merged 2x2 superblocks the plot edge can fall in a sibling's column; that
// kerb simply has no parking, and the sibling whose column does contain it
// draws those cars.
//
// THE CARS ARE THE POINT. A parked car is the one object in this city the
// camera can stand next to, and boxes with circles on them would say so
// immediately. Inside its bounding box, at fp < CAR_SDF_FP, a car is
// sphere-traced from a real SDF: a tapering lifting-body hull, smooth-min'd
// to a cabin bubble set back from the nose, wheels in cut arches (or a hover
// plenum and intake scallops, 38% of them), mirror stalks, wiper blades
// across the base of the windshield, and lamp housings and a nose intake cut
// as recesses. Detail to wiper-blade level and no further (Thomas,
// 2026-08-20): door seams, shut lines and a rocker crease are SHADING bands
// in the hull's own frame, which is where panel lines belong — no badges, no
// text, nothing that would be noise at 10 m.
//
// Beyond CAR_SDF_FP the hull falls back to two axis-aligned boxes cut to the
// same silhouette (hull + greenhouse); the 5-degree yaw jitter goes with it,
// which at that footprint is a sub-pixel corner.
//
// LIGHT. One lamp, straight overhead, is the entire lighting situation on a
// night street, so the shading is built around it: the incident estimate is
// the core's own asphalt formula (pool x district scale x CITY_LAMP_RADIANCE
// x CITY_LAMP_COLOR) and the direction is the vector to the nearest lamp
// HEAD, recovered from the same lattice. Curvature reads through the
// specular lobe sliding along the hull shoulder — Lambert alone on a dark
// paint at night is nearly flat, and the glint is what says "this surface is
// round". 30% of cars carry an underglow strip; the hue draw is aircars'
// 60/25/15 cyan/magenta/amber, so ground and air read as one traffic system.
//
// COST. In order, each gate cheaper than the one it protects:
//   1. the segment's z range against 12 m — no hash, two multiply-adds, and
//      it rejects the whole hook for every pixel looking at a facade, a roof,
//      a cloud or the sky, which is most of them;
//   2. one slab test per kerb (4), spanning lamp line to parking line;
//   3. inside a live kerb, a second z test that drops the ground props
//      (everything under 1.7 m) for a segment that only passes through the
//      lamp heads;
//   4. the along-range of the segment picks at most 2 pole slots and 4
//      parking slots, walked from the near end with an early break, each one
//      hash-gated before any box test;
//   5. the SDF runs only for a ray that has already entered a car's bounding
//      box at fp < 0.5 m/px.
// Worst case for a ray running the length of a kerb at eye level is 1 slab +
// 2 poles x 2 boxes + 4 slots x 1 box = 9 box tests for that kerb; the three
// other kerbs of the same cell are crossed, not followed, and cost 1-3 each.
// t1 is narrowed by every hit found so far, so later kerbs see shorter
// segments than earlier ones.

// --- the lamp lattice (the core's, restated) --------------------------------
const cc_streetlife_Z_GATE: f32 = 12.0;   // whole-hook z reject
const cc_streetlife_POLE_H: f32 = 9.0;    // mast top
const cc_streetlife_POLE_TOP: f32 = 9.15; // slab ceiling
const cc_streetlife_MAST_R: f32 = 0.085;
const cc_streetlife_ARM_Z: f32 = 8.72;    // the arm's own centre height
const cc_streetlife_ARM_HZ: f32 = 0.065;
const cc_streetlife_ARM_REACH: f32 = 1.35;
const cc_streetlife_HEAD_Z: f32 = 8.56;   // luminaire centre
const cc_streetlife_HEAD_HZ: f32 = 0.115;
const cc_streetlife_HEAD_HA: f32 = 0.17;  // half-extent along the kerb
const cc_streetlife_HEAD_HC: f32 = 0.40;  // half-extent along the arm
// A sodium luminaire seen from underneath is the brightest thing on the
// street by a wide margin — an order over the pool it throws (0.7) and twice
// a lit window (3.5). The housing above it is opaque and near-black, which
// is what stops a row of lamps reading as floating lozenges.
const cc_streetlife_HEAD_RAD: f32 = 6.0;
const cc_streetlife_HEAD_COLOR: vec3<f32> = vec3<f32>(1.0, 0.52, 0.18);
// The housing is a box, so the lens has to be found in the shader: the lower
// lip of each side face is the glass, the rest of the side and the whole top
// are painted aluminium. At a uniform 0.34 of RAD every side face clipped to
// white along with the underside, and a luminaire whose cowl is as bright as
// its lamp is the floating lozenge this was supposed to avoid.
const cc_streetlife_HEAD_SIDE: f32 = 0.62;  // lens lip, fraction of RAD
const cc_streetlife_HEAD_COWL: f32 = 0.030; // painted housing, same units
const cc_streetlife_HEAD_LIP: f32 = 0.072;  // how far the lens runs up (m)
// Galvanised steel, seen at night, lit by a lamp standing on its own axis.
// The Lambert term is deliberately tiny and the grazing edge does the work:
// at MAST_ALB 0.10 the first pass rendered a flat gold bar that read as a
// wooden telegraph pole, because a diffuse fraction of a clipped sodium road
// is a clipped sodium pole. A dark face between two bright edges is both the
// correct photometry for a cylinder under an axial source and the only thing
// that says "round" at this radius (see cc_streetlife_pole_shade).
const cc_streetlife_MAST_FILL: f32 = 0.06;
const cc_streetlife_MAST_ALB: f32 = 0.026;
const cc_streetlife_MAST_EDGE: f32 = 0.105;

// --- parking ----------------------------------------------------------------
// The parking line is set from the LAMP line, not from the plot edge, and
// that is a correction the first renders forced. Parking at the kerb is where
// cars belong on a minor street — the kerb is 3.5 m outboard of the lamps and
// well inside the pool — but an avenue's plot edge is 16 m out (the avenue
// gets CITY_AVENUE_EXTRA on both sides while its lamp lines stay 2.5 m off
// the block edge, so the lamps are effectively a median). Cars parked at an
// avenue kerb sit 12 m from the nearest lamp, where the pool has fallen by
// e^-3.5, and rendered as invisible black shapes on black tarmac. So: a lane
// PARK_LANE outboard of the lamps, pulled in to the kerb whenever the kerb is
// closer than that. Light and source agree by construction, and the kerb slab
// gets narrow enough to be a cheap reject as a side effect.
const cc_streetlife_PARK_LANE: f32 = 3.2;
const cc_streetlife_PARK_KERB: f32 = 0.25;  // clearance from the plot edge
const cc_streetlife_SLOT: f32 = 7.0;
const cc_streetlife_OCC: f32 = 0.35;
const cc_streetlife_BIN_CUT: f32 = 0.945;  // draws above this are dumpsters
// Dumpsters belong at block corners — the alley mouth, the service door —
// far more than they belong in a parking bay, so most of them are placed
// there instead (cc_streetlife_corner_prop) and the kerb keeps only the
// occasional one. A quarter of the corners of a built plot carry one.
const cc_streetlife_CORNER_BIN: f32 = 0.25;
const cc_streetlife_DENS_LO: f32 = 0.55;
const cc_streetlife_DENS_HI: f32 = 1.35;
const cc_streetlife_DENS_START: f32 = 0.005;
const cc_streetlife_DENS_FULL: f32 = 0.070;
const cc_streetlife_YAW: f32 = 0.0873;     // +- 5 degrees
const cc_streetlife_ALONG_JIT: f32 = 0.60;
const cc_streetlife_LAT_JIT: f32 = 0.16;

// Bounding box, in the car's own (along, across) frame. Wide enough for the
// hull yawed 5 degrees and for the mirror stalks; nothing but the reject test
// ever sees it, because the far silhouette is the two proxy boxes below.
const cc_streetlife_BB_A: f32 = 2.60;
const cc_streetlife_BB_C: f32 = 1.36;
const cc_streetlife_BB_Z: f32 = 1.74;

// --- the hull ---------------------------------------------------------------
const cc_streetlife_HL: f32 = 2.32;    // hull half-length
const cc_streetlife_HW: f32 = 0.98;    // hull half-width, at its widest
const cc_streetlife_SILL: f32 = 0.26;  // hull underside
const cc_streetlife_BELT: f32 = 0.94;  // shoulder line
const cc_streetlife_ROOF: f32 = 1.46;
const cc_streetlife_HOVER_CUT: f32 = 0.65;  // draws above this hover
const cc_streetlife_HOVER_LIFT: f32 = 0.14;
const cc_streetlife_AXLE_X: f32 = 1.46;
const cc_streetlife_AXLE_Z: f32 = 0.35;
const cc_streetlife_TRACK: f32 = 0.82;   // wheel centreplane, |y|
const cc_streetlife_TYRE_R: f32 = 0.34;
const cc_streetlife_TYRE_HW: f32 = 0.135; // half the tread width
const cc_streetlife_ARCH_R: f32 = 0.43;
const cc_streetlife_ARCH_HW: f32 = 0.31;  // the arch cuts only the flank skin
const cc_streetlife_ARCH_Z: f32 = 0.29;
const cc_streetlife_RIM_R: f32 = 0.255;

// The SDF is approximate — the plan-form taper and the falling deck make the
// hull's half-extents functions of x, so |grad| runs above 1 near the nose.
// The march steps this fraction of the reported distance, which is what keeps
// it from stepping through the skin.
const cc_streetlife_STEP: f32 = 0.72;
// The march budget, and the reason it is not a compile-time constant. A
// literal bound here is unrolled by the driver, and because cell_props is
// inlined into a 512-iteration DDA loop the unrolled body costs occupancy on
// every city pixel in the frame — including the ones nowhere near a car. The
// measurement: 32 versus 12 iterations moved the `aerial` view, which never
// admits a single car to the SDF at all, from 0.52 s to 0.29 s. Selecting
// between two counts at run time keeps the loop rolled, and doubles as
// honest LOD: a car ten pixels across does not need a 34-step trace.
const cc_streetlife_ITERS: i32 = 34;
const cc_streetlife_ITERS_FAR: i32 = 16;
const cc_streetlife_ITER_FP: f32 = 0.09;

// --- LOD --------------------------------------------------------------------
// Below this footprint a car is sphere-traced; above it, two boxes. 0.5 m/px
// puts a car at ten pixels, which is where a curved shoulder stops being a
// thing you can see and starts being a thing you can only infer.
const cc_streetlife_CAR_SDF_FP: f32 = 0.50;
const cc_streetlife_FINE_FP: f32 = 0.10;   // wipers, and sharp seams
// Where cars and poles stop being traced at all, and where their emission
// has already ramped to zero so nothing pops. Set by cost, not by the eye:
// at fp 2.6 a car is under two pixels long and the sodium road behind it is
// what that pixel was going to be anyway, while every cell inside
// CITY_PROP_RANGE pays for the test. A pole is thinner but taller, and its
// luminaire is the brightest thing on the street, so it runs further.
const cc_streetlife_CAR_FAR_FP: f32 = 2.6;
const cc_streetlife_CAR_FAR_FADE: f32 = 1.6;
const cc_streetlife_POLE_FAR_FP: f32 = 4.0;
const cc_streetlife_POLE_FAR_FADE: f32 = 2.4;
// Seams, shut lines and lamp dots hand over to their own means here; past
// LOD_FULL a car's paint is one colour and its lamp is that lamp's mean over
// the face it sits on, which is what a long lens does to a parked car.
const cc_streetlife_DETAIL_LOD: vec2<f32> = vec2<f32>(0.06, 0.26);
const cc_streetlife_LAMP_LOD: vec2<f32> = vec2<f32>(0.10, 0.45);
// The population edge at CITY_PROP_RANGE, approached rather than stepped.
const cc_streetlife_FADE_START: f32 = 0.92;

// --- car light --------------------------------------------------------------
const cc_streetlife_GLOW_FRAC: f32 = 0.32; // cars carrying an underglow
const cc_streetlife_GLOW_RAD: f32 = 1.1;
const cc_streetlife_LAMP_RAD: f32 = 1.9;
// cc_streetlife_pool returns the core's own asphalt RADIANCE — what city_shade
// emits for the road, with no albedo applied at all. Everything here is a
// fraction of THAT, not of an irradiance, and the fractions are small on
// purpose. A downtown avenue's pool runs near radiance 10, and at exposure 6
// under a Reinhard with white point 15 anything past ~2.5 is white: the road
// under these cars is already clipped, so the whole readable range for a
// painted panel is radiance 0.02 to 0.4. The first pass shaded cars at a
// physical-looking reflectance of the pool and produced pale ceramic
// bathtubs. Dark cars against a hot sodium road is both the correct
// photometry and the shot.
const cc_streetlife_PAINT_GAIN: f32 = 0.10;
const cc_streetlife_GLOSS: f32 = 1.2;
const cc_streetlife_SHEEN: f32 = 0.025;
const cc_streetlife_GLASS_GLOSS: f32 = 4.0;
const cc_streetlife_GLASS_ROAD: f32 = 0.055;
const cc_streetlife_ROAD_BOUNCE: f32 = 0.055;
const cc_streetlife_TYRE_ALB: f32 = 0.0045;
const cc_streetlife_RIM_ALB: f32 = 0.055;
const cc_streetlife_BIN_ALB: f32 = 0.10;

// ---------------------------------------------------------------------------
// SDF primitives
// ---------------------------------------------------------------------------

fn cc_streetlife_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec3<f32>(r);
    return length(max(q, vec3<f32>(0.0)))
         + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Capped cylinder whose axis is y (the car's across direction: wheels, arches
// and intake scallops all share it).
fn cc_streetlife_cyl_y(p: vec3<f32>, rad: f32, h: f32) -> f32 {
    let d = vec2<f32>(length(vec2<f32>(p.x, p.z)) - rad, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn cc_streetlife_seg(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, rad: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t) - rad;
}

fn cc_streetlife_smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn cc_streetlife_smax(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

fn cc_streetlife_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

fn cc_streetlife_nearer(a: CityHit, b: CityHit) -> CityHit {
    if (b.hit && (!a.hit || b.t < a.t)) {
        return b;
    }
    return a;
}

// ---------------------------------------------------------------------------
// The kerb: where a cell's four edges put their lamp line and parking line
// ---------------------------------------------------------------------------

struct cc_streetlife_Side {
    ax: i32,          // 0 = the kerb runs along x, 1 = along y
    lamp_c: f32,      // lamp line, across coordinate
    park_c: f32,      // parking line, across coordinate
    plot_sign: f32,   // which way the plot lies from the kerb
    a_min: f32, a_max: f32,   // the cell's extent along the kerb
    c_min: f32, c_max: f32,   // the cell's extent across it
    park_ok: bool,
}

fn cc_streetlife_side(ci: vec2<i32>, cc: CityCell, side: i32)
        -> cc_streetlife_Side {
    let cellm = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cellm;
    let cmax = cmin + vec2<f32>(cellm);
    var s: cc_streetlife_Side;
    var plot_c: f32;
    if (side == 0) {          // the plot's -x edge; kerb runs along y
        s.ax = 1;
        s.lamp_c = cmin.x + city_lamp_inset(ci.x);
        plot_c = cc.plot_min.x;
        s.plot_sign = 1.0;
    } else if (side == 1) {   // +x edge
        s.ax = 1;
        s.lamp_c = cmax.x - city_lamp_inset(ci.x + 1);
        plot_c = cc.plot_max.x;
        s.plot_sign = -1.0;
    } else if (side == 2) {   // -y edge; kerb runs along x
        s.ax = 0;
        s.lamp_c = cmin.y + city_lamp_inset(ci.y);
        plot_c = cc.plot_min.y;
        s.plot_sign = 1.0;
    } else {                  // +y edge
        s.ax = 0;
        s.lamp_c = cmax.y - city_lamp_inset(ci.y + 1);
        plot_c = cc.plot_max.y;
        s.plot_sign = -1.0;
    }
    // Outboard of the lamps by PARK_LANE, or hard against the kerb if the
    // kerb is nearer than that (see the constant's note).
    let gap = s.plot_sign * (plot_c - s.lamp_c);
    let off = min(cc_streetlife_PARK_LANE,
                  gap - cc_streetlife_BB_C - cc_streetlife_PARK_KERB);
    s.park_c = s.lamp_c + s.plot_sign * off;
    s.a_min = select(cmin.y, cmin.x, s.ax == 0);
    s.a_max = select(cmax.y, cmax.x, s.ax == 0);
    s.c_min = select(cmin.x, cmin.y, s.ax == 0);
    s.c_max = select(cmax.x, cmax.y, s.ax == 0);
    // A merged superblock's plot edge can sit in a sibling's column; that
    // kerb keeps its lamps (they are on the cell's own lattice) and loses its
    // parking to whichever member owns the ground.
    s.park_ok = off > 1.6
             && s.park_c > s.c_min + cc_streetlife_BB_C
             && s.park_c < s.c_max - cc_streetlife_BB_C;
    return s;
}

// The nearest lamp HEAD to a street point, on the core's own lattice: the
// four candidates city_street_pools sums over, and the one that wins. Used
// as the light direction for everything this component shades.
fn cc_streetlife_nearest_lamp(p: vec2<f32>) -> vec3<f32> {
    let cellm = u.ocean_params.x;
    let bx = round(p.x / cellm) * cellm;
    let by = round(p.y / cellm) * cellm;
    let inx = city_lamp_inset(i32(round(p.x / cellm)));
    let iny = city_lamp_inset(i32(round(p.y / cellm)));
    let lx = round(p.x / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
    let ly = round(p.y / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
    var best = vec2<f32>(bx - inx, ly);
    var bd = 1e30;
    for (var k: i32 = 0; k < 4; k = k + 1) {
        var c: vec2<f32>;
        if (k == 0) {
            c = vec2<f32>(bx - inx, ly);
        } else if (k == 1) {
            c = vec2<f32>(bx + inx, ly);
        } else if (k == 2) {
            c = vec2<f32>(lx, by - iny);
        } else {
            c = vec2<f32>(lx, by + iny);
        }
        let d = dot(c - p, c - p);
        if (d < bd) {
            bd = d;
            best = c;
        }
    }
    return vec3<f32>(best, cc_streetlife_HEAD_Z);
}

// The core's own asphalt radiance at a street point, reused verbatim as the
// incident estimate for anything standing on it. Sharing the formula is the
// point: a car in a pool is exactly as lit as the tarmac it is parked on.
fn cc_streetlife_pool(p: vec2<f32>) -> vec3<f32> {
    let district = city_glow_sample(p, 2.0);
    let scale = 0.20 + 2.2 * smoothstep(0.02, 0.45, district);
    return CITY_LAMP_COLOR
         * (CITY_LAMP_RADIANCE * scale * city_street_pools(p));
}

fn cc_streetlife_fill(p: vec3<f32>) -> vec3<f32> {
    return CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

struct cc_streetlife_Prop {
    ok: bool,
    bin: bool,
    ctr: vec2<f32>,   // ground point under the prop's centre
    fwd: vec2<f32>,   // the prop's own forward, yaw included
    rgt: vec2<f32>,
    r: vec4<f32>,
}

fn cc_streetlife_no_prop() -> cc_streetlife_Prop {
    return cc_streetlife_Prop(false, false, vec2<f32>(0.0),
                              vec2<f32>(1.0, 0.0), vec2<f32>(0.0, -1.0),
                              vec4<f32>(0.0));
}

// Tile-wrapped (city_tile_cell): kerb furniture belongs to its tile
// coordinate, so live and replay place the same bins on the same corners.
fn cc_streetlife_slot_draw(ci: vec2<i32>, side: i32, j: i32) -> vec4<f32> {
    let cw = city_tile_cell(ci);
    return city_rand4(vec2<u32>(
        bitcast<u32>(cw.x) * 0x9e3779b9u + u32(side) * 0x2545f491u
            + bitcast<u32>(j) * 0x85ebca6bu + 0x51ed270bu,
        bitcast<u32>(cw.y) * 0xc2b2ae35u + u32(side) * 0x27d4eb2fu
            + bitcast<u32>(j) * 0x165667b1u + 0x9e3779b9u));
}

// Whatever occupies slot `j` of kerb `side` — deterministic in (cell, side,
// slot) alone, so the shader re-derives a car from its hit position without
// anything being smuggled through CityHit.
fn cc_streetlife_prop(ci: vec2<i32>, cc: CityCell, s: cc_streetlife_Side,
                      side: i32, j: i32) -> cc_streetlife_Prop {
    if (!s.park_ok) {
        return cc_streetlife_no_prop();
    }
    let r = cc_streetlife_slot_draw(ci, side, j);
    let dens = mix(cc_streetlife_DENS_LO, cc_streetlife_DENS_HI,
                   smoothstep(cc_streetlife_DENS_START,
                              cc_streetlife_DENS_FULL, cc.density));
    let occ = cc_streetlife_OCC * dens;
    let bin = r.x > cc_streetlife_BIN_CUT;
    if (r.x >= occ && !bin) {
        return cc_streetlife_no_prop();
    }
    // Slot centre on the world lattice, so a run of cars lines up across the
    // cell boundary rather than restarting inside every block.
    let along = (f32(j) + 0.5) * cc_streetlife_SLOT
              + (r.y - 0.5) * 2.0 * cc_streetlife_ALONG_JIT;
    if (along - cc_streetlife_BB_A < s.a_min
        || along + cc_streetlife_BB_A > s.a_max) {
        return cc_streetlife_no_prop();
    }
    var across = s.park_c + (r.z - 0.5) * 2.0 * cc_streetlife_LAT_JIT;
    if (bin) {
        // Bins get shoved against the wall, not left at the kerb.
        across = s.park_c + s.plot_sign * 0.85;
    }
    var base = vec2<f32>(1.0, 0.0);
    if (s.ax == 1) {
        base = vec2<f32>(0.0, 1.0);
    }
    let perp = vec2<f32>(-base.y, base.x);
    var fwd = base;
    if (!bin) {
        let yaw = (r.w - 0.5) * 2.0 * cc_streetlife_YAW;
        fwd = base * cos(yaw) + perp * sin(yaw);
        if (fract(r.y * 17.31) > 0.5) {
            fwd = -fwd;
        }
    }
    var p: cc_streetlife_Prop;
    p.ok = true;
    p.bin = bin;
    p.ctr = base * along + perp * across;
    p.fwd = fwd;
    p.rgt = vec2<f32>(fwd.y, -fwd.x);
    p.r = r;
    return p;
}

// ---------------------------------------------------------------------------
// The car SDF
// ---------------------------------------------------------------------------
//
// Local frame: +x forward, +y left, z up from the road surface. Everything
// symmetric about the centreline is evaluated once on abs(y), and the wheels
// and lamp housings once on abs(x) too — four wheels for the price of one
// cylinder.

// Two draws of shape per car: an overall scale, and where the cabin sits
// fore and aft. Between them a row of parked cars stops being one model
// repeated — a short car with the cabin back is a coupe, a long one with it
// forward is a saloon, and the eye reads the difference before it reads any
// panel line.
fn cc_streetlife_car_shape(r: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(0.92 + 0.13 * fract(r.z * 7.71),
                     -0.30 + (fract(r.w * 11.37) - 0.5) * 0.34);
}

// Which cars hover. This must NOT be read off r.x, and that it was is the one
// outright bug the salvaged draft carried: r.x is the OCCUPANCY draw, and a
// slot holds a car only where r.x < occ — at most 0.47 even downtown — so a
// hover test of `r.x > 0.62` was unreachable in every cell of the city. Nobody
// would ever have seen it fail; the plenum, the skirt and the intake scallops
// simply never ran. An independent draw off the other three components gives
// the ~35% the file always claimed.
fn cc_streetlife_is_hover(r: vec4<f32>) -> bool {
    return fract(r.y * 43.17 + r.z * 7.31 + r.w * 2.53)
           > cc_streetlife_HOVER_CUT;
}

fn cc_streetlife_car_sdf(p: vec3<f32>, r: vec4<f32>, fine: bool) -> f32 {
    let hover = cc_streetlife_is_hover(r);
    let lift = select(0.0, cc_streetlife_HOVER_LIFT, hover);
    let sh = cc_streetlife_car_shape(r);
    let q = vec3<f32>(p.x, p.y, p.z - lift) / sh.x;
    let cab_c = sh.y;
    let off = cab_c + 0.30;

    // Plan-form: the hull narrows toward both ends, hard at the very tips.
    let xn = clamp(q.x / cc_streetlife_HL, -1.0, 1.0);
    let xn2 = xn * xn;
    let xn4 = xn2 * xn2;
    let hw = cc_streetlife_HW * (1.0 - 0.34 * xn4);
    // Profile: the deck falls away toward the nose and, less, toward the
    // tail. A flat deck is what makes a box read as a box.
    let drop = select(0.08, 0.16, xn > 0.0);
    let deck = cc_streetlife_BELT - drop * xn2;
    let hc = 0.5 * (deck + cc_streetlife_SILL);
    let hh = 0.5 * (deck - cc_streetlife_SILL);
    var d = cc_streetlife_rbox(vec3<f32>(q.x, q.y, q.z - hc),
                               vec3<f32>(cc_streetlife_HL, hw, hh), 0.22);

    // The greenhouse: inset from the shoulder, set back from the nose, and
    // blended only just enough to give it a fillet. A soft blend here is what
    // turned the first pass into a loaf — the shoulder line has to survive.
    // Rounded hard, this reads as a bubble stuck on the deck rather than as
    // a cabin: the roof has to be flat enough to be a roof and the side glass
    // near enough to vertical to be glass.
    let cx = q.x - cab_c;
    var cab = cc_streetlife_rbox(
        vec3<f32>(cx, q.y, q.z - 1.20),
        vec3<f32>(0.92, 0.74, 0.26), 0.10);
    // TUMBLEHOME. The side glass leans in toward the roof, so the greenhouse
    // is a tapered turret rather than a box, and its widest point is the belt
    // line where it meets the shoulder. Without this the cabin read as a loaf
    // of bread set on the deck — full-width, vertical-sided, visibly a second
    // box. One slanted half-space does the whole job; the 0.958 is 1/|grad|,
    // which keeps the march from stepping through the lean.
    cab = cc_streetlife_smax(
        cab, 0.958 * (abs(q.y) - 0.74 + 0.30 * (q.z - 1.02)), 0.07);
    // Windshield and backlight rake: two half-spaces that take the front and
    // rear off the greenhouse, so the cabin is a cabin and not a second box.
    // Cut into the cabin alone — applied to the whole hull the front plane
    // would take the bonnet with it. The planes are placed to MEET THE BELT,
    // not to clip a corner: the first pass's intercepts put the start of the
    // windshield at cx 1.18, outside the cabin box entirely, so the rake took
    // only the top corner and everything below it stayed the box's own
    // vertical wall — which is exactly what the renders showed. Now the glass
    // starts within 5 mm of the shoulder and the roof comes out 1.14 m long
    // by 1.22 wide, which is a car; the first pass's would have been 0.66 by
    // 1.22, which is a bubble canopy.
    cab = cc_streetlife_smax(cab, 0.824 * cx + 0.567 * q.z - 1.339, 0.10);
    cab = cc_streetlife_smax(cab, -0.745 * cx + 0.667 * q.z - 1.361, 0.10);
    d = cc_streetlife_smin(d, cab, 0.11);

    // WHEELS AND ARCHES, both mirrored on abs(y) so there are four of them.
    //
    // The first pass mirrored only on abs(x) and gave the cylinders a
    // half-width of 0.86 — wider than the hull's own 0.98 half-width. That is
    // not four wheels, it is two drums spanning the full track, and the arch
    // that cut them free was 1.10 wide, which bored a tunnel clean through
    // the body. From the side the two errors cancelled and it read correctly;
    // head-on the drum showed under the nose as a hard dark bar with square
    // ends, wider than the car, and it is visible in every frontal frame the
    // draft ever produced. Splitting them fixes the frontal read and makes
    // the arch what an arch actually is: a cut in the outer skin of a flank.
    let qa = vec3<f32>(abs(q.x) - cc_streetlife_AXLE_X,
                       abs(q.y) - cc_streetlife_TRACK,
                       q.z - cc_streetlife_ARCH_Z);
    if (hover) {
        // A plenum instead of wheels, and the arches become intake scallops
        // cut into the flanks — the same silhouette cue read the other way.
        let skirt = cc_streetlife_rbox(
            vec3<f32>(q.x, q.y, q.z - 0.14),
            vec3<f32>(1.94, 0.80, 0.09), 0.08);
        d = cc_streetlife_smin(d, skirt, 0.13);
        let scallop = cc_streetlife_cyl_y(
            vec3<f32>(qa.x, qa.y - 0.10, qa.z), 0.30,
            cc_streetlife_ARCH_HW + 0.10);
        d = cc_streetlife_smax(d, -scallop, 0.05);
    } else {
        let arch = cc_streetlife_cyl_y(
            vec3<f32>(qa.x, qa.y - 0.13, qa.z),
            cc_streetlife_ARCH_R, cc_streetlife_ARCH_HW);
        d = cc_streetlife_smax(d, -arch, 0.035);
        let tyre = cc_streetlife_cyl_y(
            vec3<f32>(qa.x, qa.y, q.z - cc_streetlife_AXLE_Z),
            cc_streetlife_TYRE_R, cc_streetlife_TYRE_HW);
        d = min(d, tyre);
    }

    // Door mirrors, at the base of the A-pillar. A bare capsule reaching from
    // the cabin flank to the hull's widest point — which is what the first
    // pass had — renders as a dark bar floating clear of the car, because a
    // uniform 10 cm cylinder 30 cm long is a stick and nothing about it says
    // mirror. A mirror is a SHORT arm carrying a HOUSING, and the housing is
    // what the eye finds: a flat-backed pod, wider than it is tall.
    let qm = vec3<f32>(q.x - 0.74 - off, abs(q.y), q.z - 1.01);
    d = min(d, cc_streetlife_seg(qm, vec3<f32>(0.0, 0.60, 0.0),
                                 vec3<f32>(0.02, 0.76, 0.03), 0.032));
    d = min(d, cc_streetlife_rbox(qm - vec3<f32>(0.0, 0.855, 0.035),
                                  vec3<f32>(0.075, 0.095, 0.055), 0.038));

    // Lamp housings, cut as recesses at nose and tail, and an intake slot
    // low in the nose. The recess is a LETTERBOX — wider than tall — because
    // the lens the shader paints inside it is an ellipse of the same aspect,
    // and a round lens in a round hole is the headlight shape that made the
    // first pass read as a face with eyes.
    let ql = vec3<f32>(abs(q.x) - 2.15, abs(q.y) - 0.50, q.z - 0.71);
    d = cc_streetlife_smax(
        d, -cc_streetlife_rbox(ql, vec3<f32>(0.13, 0.27, 0.082), 0.04), 0.022);
    let qi = vec3<f32>(q.x - 2.06, q.y, q.z - 0.40);
    d = cc_streetlife_smax(
        d, -cc_streetlife_rbox(qi, vec3<f32>(0.16, 0.44, 0.055), 0.03), 0.03);

    if (fine) {
        // Wiper blades across the base of the windshield. At the footprint
        // that admits them a blade is two or three pixels of hard line on a
        // dark curved reflection, which is exactly what says "windscreen".
        let qw = vec3<f32>(q.x - off, abs(q.y), q.z);
        d = min(d, cc_streetlife_seg(qw, vec3<f32>(0.66, 0.07, 0.925),
                                     vec3<f32>(1.00, 0.52, 0.908), 0.021));
    }
    return d * sh.x;
}

// World point -> the car's own frame.
fn cc_streetlife_to_local(w: vec3<f32>, ctr: vec2<f32>, fwd: vec2<f32>,
                          rgt: vec2<f32>) -> vec3<f32> {
    let rel = w.xy - ctr;
    return vec3<f32>(dot(rel, fwd), dot(rel, rgt), w.z);
}

fn cc_streetlife_car_normal(pl: vec3<f32>, r: vec4<f32>, fine: bool, hh: f32)
        -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * hh;
    let n = vec3<f32>(1.0, -1.0, -1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.x, e.y, e.y), r, fine)
          + vec3<f32>(-1.0, -1.0, 1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.y, e.y, e.x), r, fine)
          + vec3<f32>(-1.0, 1.0, -1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.y, e.x, e.y), r, fine)
          + vec3<f32>(1.0, 1.0, 1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.x, e.x, e.x), r, fine);
    return normalize(n);
}

// Sphere-trace one car inside a bounding box the ray has already entered.
// A miss inside the box is a real answer, not a failure: rays graze past a
// curved hull, and that is what makes the hull read as curved.
fn cc_streetlife_trace_car(o: vec3<f32>, dir: vec3<f32>, ta: f32, tb: f32,
                           v: cc_streetlife_Prop, ci: vec2<i32>, side: i32,
                           fp: f32) -> CityHit {
    let fine = fp < cc_streetlife_FINE_FP;
    let eps = max(0.0025, 0.30 * fp);
    let iters = select(cc_streetlife_ITERS, cc_streetlife_ITERS_FAR,
                       fp > cc_streetlife_ITER_FP);
    var t = ta + 0.001;
    var got = false;
    for (var i: i32 = 0; i < iters; i = i + 1) {
        let pl = cc_streetlife_to_local(o + t * dir, v.ctr, v.fwd, v.rgt);
        let d = cc_streetlife_car_sdf(pl, v.r, fine);
        if (d < eps) {
            got = true;
            break;
        }
        t = t + d * cc_streetlife_STEP;
        if (t > tb) {
            break;
        }
    }
    if (!got || t > tb) {
        return cc_streetlife_miss(ci);
    }
    let pos = o + t * dir;
    let pl = cc_streetlife_to_local(pos, v.ctr, v.fwd, v.rgt);
    let nl = cc_streetlife_car_normal(pl, v.r, fine, max(0.004, 0.4 * fp));
    let nw = vec3<f32>(v.fwd * nl.x + v.rgt * nl.y, nl.z);
    return CityHit(true, t, pos, nw, 102 + side, ci);
}

// The far read: hull and greenhouse as two axis-aligned boxes cut to the
// SDF's own silhouette. The yaw goes with the SDF, which at this footprint
// is a sub-pixel corner.
fn cc_streetlife_trace_car_box(o: vec3<f32>, inv_dir: vec3<f32>, dir: vec3<f32>,
                               t0: f32, t1: f32, v: cc_streetlife_Prop,
                               ci: vec2<i32>, side: i32) -> CityHit {
    let hover = cc_streetlife_is_hover(v.r);
    let lift = select(0.0, cc_streetlife_HOVER_LIFT, hover);
    let ea = abs(v.fwd) * 2.24 + abs(v.rgt) * 0.90;
    let eb = abs(v.fwd) * 1.06 + abs(v.rgt) * 0.76;
    let amin = vec3<f32>(v.ctr - ea, select(0.02, lift + 0.10, hover));
    let amax = vec3<f32>(v.ctr + ea, lift + 1.00);
    let bmin = vec3<f32>(v.ctr - eb + v.fwd * -0.24, lift + 0.94);
    let bmax = vec3<f32>(v.ctr + eb + v.fwd * -0.24,
                         lift + cc_streetlife_ROOF);
    var best = 1e30;
    var bmn = amin;
    var bmx = amax;
    let ha = city_box_hit(o, inv_dir, amin, amax);
    if (ha.x <= ha.y && ha.y > t0 && ha.x <= t1) {
        best = max(ha.x, t0);
    }
    let hb = city_box_hit(o, inv_dir, bmin, bmax);
    if (hb.x <= hb.y && hb.y > t0 && hb.x <= t1 && max(hb.x, t0) < best) {
        best = max(hb.x, t0);
        bmn = bmin;
        bmx = bmax;
    }
    if (best >= 1e30) {
        return cc_streetlife_miss(ci);
    }
    let pos = o + best * dir;
    return CityHit(true, best, pos, city_box_normal(pos, bmn, bmx),
                   102 + side, ci);
}

// A dumpster at one of the plot's four corners, shoved against the wall a
// couple of metres in from the corner itself. `k` selects the corner: bit 0
// the x wall, bit 1 the y end.
//
// Corners are a per-CELL question, not a per-kerb one, and that is why this
// does not live in cc_streetlife_kerb with the other ground props. The kerb's
// bounding slab runs from the lamp line to the parking line, and on an avenue
// the plot edge is thirteen metres outboard of that — a bin against the wall
// would sit entirely outside the slab the kerb tests, so a kerb-side test
// would silently never fire on exactly the streets that are widest and most
// visible.
//
// Placed strictly inside the drawing cell, per the DDA rule: a merged
// superblock's corner can fall in a sibling's column, and there the sibling
// that owns the ground draws it.
fn cc_streetlife_corner_prop(ci: vec2<i32>, cc: CityCell, k: i32)
        -> cc_streetlife_Prop {
    if (!cc.built) {
        return cc_streetlife_no_prop();
    }
    let cw = city_tile_cell(ci);
    let r = city_rand4(vec2<u32>(
        bitcast<u32>(cw.x) * 0x27d4eb2fu + u32(k) * 0x9e3779b9u + 0x2f1e3a7bu,
        bitcast<u32>(cw.y) * 0x165667b1u + u32(k) * 0xc2b2ae35u + 0x7feb352du));
    if (r.x > cc_streetlife_CORNER_BIN) {
        return cc_streetlife_no_prop();
    }
    let xlo = (k & 1) == 0;
    let ylo = (k & 2) == 0;
    let wall_x = select(cc.plot_max.x, cc.plot_min.x, xlo);
    let out_x = select(1.0, -1.0, xlo);        // away from the plot
    let corner_y = select(cc.plot_max.y, cc.plot_min.y, ylo);
    let in_y = select(-1.0, 1.0, ylo);         // along the wall, into the plot
    let px = wall_x + out_x * (0.92 + 0.28 * r.z);
    let py = corner_y + in_y * (1.30 + 1.10 * r.y);
    let cellm = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cellm;
    let cmax = cmin + vec2<f32>(cellm);
    if (px - 0.62 < cmin.x || px + 0.62 > cmax.x
        || py - 0.92 < cmin.y || py + 0.92 > cmax.y) {
        return cc_streetlife_no_prop();
    }
    var p: cc_streetlife_Prop;
    p.ok = true;
    p.bin = true;
    p.ctr = vec2<f32>(px, py);
    p.fwd = vec2<f32>(0.0, 1.0);               // long side along the wall
    p.rgt = vec2<f32>(1.0, 0.0);
    p.r = r;
    return p;
}

fn cc_streetlife_trace_bin(o: vec3<f32>, inv_dir: vec3<f32>, dir: vec3<f32>,
                           t0: f32, t1: f32, v: cc_streetlife_Prop,
                           ci: vec2<i32>, side: i32) -> CityHit {
    let e = abs(v.fwd) * 0.80 + abs(v.rgt) * 0.50;
    let bmin = vec3<f32>(v.ctr - e, 0.0);
    let bmax = vec3<f32>(v.ctr + e, 1.20);
    let hb = city_box_hit(o, inv_dir, bmin, bmax);
    if (hb.x > hb.y || hb.y <= t0 || hb.x > t1) {
        return cc_streetlife_miss(ci);
    }
    let t = max(hb.x, t0);
    let pos = o + t * dir;
    return CityHit(true, t, pos, city_box_normal(pos, bmin, bmax),
                   106 + side, ci);
}

// ---------------------------------------------------------------------------
// One kerb
// ---------------------------------------------------------------------------

fn cc_streetlife_pole(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t0: f32, t1: f32, s: cc_streetlife_Side, a: f32,
                      ci: vec2<i32>, fp: f32) -> CityHit {
    // The arm hangs AWAY from the plot, out over the roadway.
    let arm = -s.plot_sign;
    let head_c = s.lamp_c + arm * cc_streetlife_ARM_REACH;
    var mmin: vec3<f32>;
    var mmax: vec3<f32>;
    var hmin: vec3<f32>;
    var hmax: vec3<f32>;
    if (s.ax == 0) {
        mmin = vec3<f32>(a - cc_streetlife_MAST_R,
                         s.lamp_c - cc_streetlife_MAST_R, 0.0);
        mmax = vec3<f32>(a + cc_streetlife_MAST_R,
                         s.lamp_c + cc_streetlife_MAST_R,
                         cc_streetlife_POLE_H);
        hmin = vec3<f32>(a - cc_streetlife_HEAD_HA,
                         head_c - cc_streetlife_HEAD_HC,
                         cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ);
        hmax = vec3<f32>(a + cc_streetlife_HEAD_HA,
                         head_c + cc_streetlife_HEAD_HC,
                         cc_streetlife_HEAD_Z + cc_streetlife_HEAD_HZ);
    } else {
        mmin = vec3<f32>(s.lamp_c - cc_streetlife_MAST_R,
                         a - cc_streetlife_MAST_R, 0.0);
        mmax = vec3<f32>(s.lamp_c + cc_streetlife_MAST_R,
                         a + cc_streetlife_MAST_R, cc_streetlife_POLE_H);
        hmin = vec3<f32>(head_c - cc_streetlife_HEAD_HC,
                         a - cc_streetlife_HEAD_HA,
                         cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ);
        hmax = vec3<f32>(head_c + cc_streetlife_HEAD_HC,
                         a + cc_streetlife_HEAD_HA,
                         cc_streetlife_HEAD_Z + cc_streetlife_HEAD_HZ);
    }
    var best = 1e30;
    var bmn = mmin;
    var bmx = mmax;
    var kind = 100;
    let hm = city_box_hit(o, inv_dir, mmin, mmax);
    if (hm.x <= hm.y && hm.y > t0 && hm.x <= t1) {
        best = max(hm.x, t0);
    }
    let hh = city_box_hit(o, inv_dir, hmin, hmax);
    if (hh.x <= hh.y && hh.y > t0 && hh.x <= t1 && max(hh.x, t0) < best) {
        best = max(hh.x, t0);
        bmn = hmin;
        bmx = hmax;
        kind = 101;
    }
    // The arm is a 0.13 m bar: it only earns a box test while it is a
    // resolvable line rather than an aliasing one. Past that the mast and the
    // luminaire carry the pole, which is what the eye reads anyway.
    if (fp < 0.30) {
        var amin: vec3<f32>;
        var amax: vec3<f32>;
        let c0 = min(s.lamp_c, head_c);
        let c1 = max(s.lamp_c, head_c);
        if (s.ax == 0) {
            amin = vec3<f32>(a - 0.055, c0,
                             cc_streetlife_ARM_Z - cc_streetlife_ARM_HZ);
            amax = vec3<f32>(a + 0.055, c1,
                             cc_streetlife_ARM_Z + cc_streetlife_ARM_HZ);
        } else {
            amin = vec3<f32>(c0, a - 0.055,
                             cc_streetlife_ARM_Z - cc_streetlife_ARM_HZ);
            amax = vec3<f32>(c1, a + 0.055,
                             cc_streetlife_ARM_Z + cc_streetlife_ARM_HZ);
        }
        let ha = city_box_hit(o, inv_dir, amin, amax);
        if (ha.x <= ha.y && ha.y > t0 && ha.x <= t1 && max(ha.x, t0) < best) {
            best = max(ha.x, t0);
            bmn = amin;
            bmx = amax;
            kind = 100;
        }
    }
    if (best >= 1e30) {
        return cc_streetlife_miss(ci);
    }
    let pos = o + best * dir;
    return CityHit(true, best, pos, city_box_normal(pos, bmn, bmx), kind, ci);
}

// What one kerb found. The sphere trace is deliberately NOT run here: a near
// car is returned as a CANDIDATE and resolved once per cell, after all four
// kerbs have reported. The reason is measured rather than stylistic — see the
// note in cc_streetlife_props_trace.
struct cc_streetlife_Kerb {
    hit: CityHit,      // already resolved: poles, bins, far cars
    car_ok: bool,      // a near car whose bounding box the ray entered
    car: cc_streetlife_Prop,
    side: i32,
    ta: f32,
    tb: f32,           // the box interval to sphere-trace inside
}

fn cc_streetlife_kerb(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell,
                      side: i32, fp: f32) -> cc_streetlife_Kerb {
    var out: cc_streetlife_Kerb;
    out.hit = cc_streetlife_miss(ci);
    out.car_ok = false;
    out.car = cc_streetlife_no_prop();
    out.side = side;
    out.ta = 0.0;
    out.tb = 0.0;

    let s = cc_streetlife_side(ci, cc, side);
    // One slab over the whole kerb: lamp line to parking line, ground to the
    // top of a mast, clipped to the cell's own column. This is the test the
    // wide scene pays, and the only one most cells ever reach.
    let arm_c = s.lamp_c - s.plot_sign * (cc_streetlife_ARM_REACH + 0.5);
    var lo = min(min(s.lamp_c, arm_c), s.park_c - cc_streetlife_BB_C);
    var hi = max(max(s.lamp_c, arm_c), s.park_c + cc_streetlife_BB_C);
    if (!s.park_ok) {
        lo = min(s.lamp_c, arm_c);
        hi = max(s.lamp_c, arm_c);
    }
    lo = max(lo - 0.2, s.c_min);
    hi = min(hi + 0.2, s.c_max);
    if (hi <= lo) {
        return out;
    }
    var bmin: vec3<f32>;
    var bmax: vec3<f32>;
    if (s.ax == 0) {
        bmin = vec3<f32>(s.a_min, lo, 0.0);
        bmax = vec3<f32>(s.a_max, hi, cc_streetlife_POLE_TOP);
    } else {
        bmin = vec3<f32>(lo, s.a_min, 0.0);
        bmax = vec3<f32>(hi, s.a_max, cc_streetlife_POLE_TOP);
    }
    let sb = city_box_hit(o, inv_dir, bmin, bmax);
    let ta = max(sb.x, t0);
    let tb = min(sb.y, t1);
    if (sb.x > sb.y || tb <= ta) {
        return out;
    }

    let pa = o + ta * dir;
    let pb = o + tb * dir;
    let sa = select(pa.y, pa.x, s.ax == 0);
    let sc = select(pb.y, pb.x, s.ax == 0);
    let s_lo = min(sa, sc);
    let s_hi = max(sa, sc);
    let z_lo = min(pa.z, pb.z);
    let fwd_first = select(dir.y, dir.x, s.ax == 0) >= 0.0;

    var res = cc_streetlife_miss(ci);
    var t_end = tb;

    // Poles, on the core's 26 m lattice, restricted to those standing wholly
    // inside this cell.
    if (fp < cc_streetlife_POLE_FAR_FP) {
        let sp = CITY_LAMP_SPACING;
        var j0 = i32(ceil((s_lo - 0.45) / sp));
        var j1 = i32(floor((s_hi + 0.45) / sp));
        j0 = max(j0, i32(ceil((s.a_min + 0.45) / sp)));
        j1 = min(j1, i32(floor((s.a_max - 0.45) / sp)));
        if (j1 >= j0) {
            let jstart = select(j1, j0, fwd_first);
            let jstep = select(-1, 1, fwd_first);
            // Walked from the near end, so the first pole the ray actually
            // strikes is the nearest and the loop is done.
            for (var k: i32 = 0; k < 3; k = k + 1) {
                let j = jstart + k * jstep;
                if (j < j0 || j > j1) {
                    break;
                }
                let hp = cc_streetlife_pole(o, dir, inv_dir, ta, t_end, s,
                                            f32(j) * sp, ci, fp);
                if (hp.hit) {
                    res = cc_streetlife_nearer(res, hp);
                    t_end = min(t_end, res.t);
                    break;
                }
            }
        }
    }

    // Ground props: everything below is under 1.8 m, so a segment that only
    // clips the lamp heads stops here.
    out.hit = res;
    if (!s.park_ok || z_lo > cc_streetlife_BB_Z + 0.1
        || fp > cc_streetlife_CAR_FAR_FP) {
        return out;
    }
    let g0 = i32(floor((s_lo - cc_streetlife_BB_A) / cc_streetlife_SLOT));
    let g1 = i32(floor((s_hi + cc_streetlife_BB_A) / cc_streetlife_SLOT));
    let gstart = select(g1, g0, fwd_first);
    let gstep = select(-1, 1, fwd_first);
    // Same rule as the poles: near end first, stop at the first prop the ray
    // actually strikes. Five iterations is what an empty run costs, and at
    // ~40% occupancy the loop usually ends on the first or the second.
    let near_sdf = fp < cc_streetlife_CAR_SDF_FP;
    for (var k: i32 = 0; k < 5; k = k + 1) {
        let j = gstart + k * gstep;
        if (j < g0 || j > g1) {
            break;
        }
        let v = cc_streetlife_prop(ci, cc, s, side, j);
        if (!v.ok) {
            continue;
        }
        if (v.bin) {
            let hb = cc_streetlife_trace_bin(o, inv_dir, dir, ta, t_end, v,
                                             ci, side);
            if (hb.hit) {
                res = cc_streetlife_nearer(res, hb);
                t_end = min(t_end, res.t);
                break;
            }
            continue;
        }
        // The bounding box: the only cost the wide scene pays for a car.
        let e = abs(v.fwd) * cc_streetlife_BB_A
              + abs(v.rgt) * cc_streetlife_BB_C;
        let cmin = vec3<f32>(v.ctr - e, 0.0);
        let cmax = vec3<f32>(v.ctr + e, cc_streetlife_BB_Z);
        let hc = city_box_hit(o, inv_dir, cmin, cmax);
        if (hc.x > hc.y || hc.y <= ta || hc.x > t_end) {
            continue;
        }
        if (near_sdf) {
            out.car_ok = true;
            out.car = v;
            out.ta = max(hc.x, ta);
            out.tb = min(hc.y, t_end);
            t_end = min(t_end, out.ta);
            break;
        }
        let hit = cc_streetlife_trace_car_box(o, inv_dir, dir, ta, t_end, v,
                                              ci, side);
        if (hit.hit) {
            res = cc_streetlife_nearer(res, hit);
            t_end = min(t_end, res.t);
            break;
        }
    }
    out.hit = res;
    return out;
}

fn cc_streetlife_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                             t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    // WHERE THIS COMPONENT'S COST ACTUALLY IS, measured rather than guessed,
    // so the next reader does not repeat the experiments (RTX 5080, shared,
    // interleaved on/off, 64 accumulated frames at 960x540):
    //
    //     view      off      on     delta
    //     base     0.34 s  0.40 s   +18%
    //     aerial   0.22 s  0.30 s   +36%
    //     horizon  0.31 s  0.40 s   +29%
    //
    // All of it is in this hook, none in the shade hook: unregistering
    // `shade` and leaving `cell_props` reproduced the enabled timings to the
    // centisecond. And none of it is work — on all three views EVERY call
    // leaves at gate 1 below, because the nearest ground within
    // CITY_PROP_RANGE is still a kilometre under the ray. Three attempts to
    // shrink the inlined body moved nothing at all: hoisting the fp cutoff
    // above the four kerb slabs, rolling the four-kerb loop behind a bound
    // the driver cannot fold (the draft's own trick for the march budget),
    // and generating the normal's four tetrahedron taps in a loop instead of
    // spelling them out. Two of the three were reverted for being clutter
    // that bought nothing; the fp cutoff stayed because it is exact.
    //
    // What is left is the gate itself, two fused multiply-adds and two
    // compares, run once per DDA cell within CITY_PROP_RANGE — on the aerial
    // view roughly 26 million times a frame. 0.08 s over 64 frames is about
    // three flops per evaluation, which is the whole of it. This is the floor
    // for ANY cell_props hook in a 512-iteration DDA, not a streetlife
    // problem, and it is not reducible from inside a component.
    //
    // Gate 1, no hash and no memory: does this segment come within reach of
    // the ground at all? Every pixel looking at a facade, a roof, a cloud or
    // the sky leaves here.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_streetlife_Z_GATE || max(za, zb) < -0.5) {
        return cc_streetlife_miss(ci);
    }
    let fp = max(2.0 * u.cam_origin.w / max(u.params.x, 1.0), u.periodic.z)
             * max(t0, 0.0);
    // Gate 2: past the pole cutoff nothing this component draws survives, and
    // every prop's emission has already faded to zero before reaching it
    // (POLE_FAR_FADE 2.4 -> 4.0, CAR_FAR_FADE 1.6 -> 2.6). Hoisting the test
    // above the four kerb slabs makes it exact rather than merely cheap: it
    // is the same answer those per-kerb fp tests would have reached, arrived
    // at before any of the four Side structs is built.
    if (fp > cc_streetlife_POLE_FAR_FP) {
        return cc_streetlife_miss(ci);
    }
    var res = cc_streetlife_miss(ci);
    var t_end = t1;
    var cand: cc_streetlife_Kerb;
    var have = false;
    var cand_t = 1e30;
    for (var side: i32 = 0; side < 4; side = side + 1) {
        let k = cc_streetlife_kerb(o, dir, inv_dir, t0, t_end, ci, cc, side,
                                   fp);
        if (k.hit.hit) {
            res = cc_streetlife_nearer(res, k.hit);
            t_end = min(t_end, res.t);
        }
        if (k.car_ok && k.ta < cand_t) {
            cand = k;
            have = true;
            cand_t = k.ta;
            t_end = min(t_end, k.ta);
        }
    }
    // Corner dumpsters. Four hash draws behind a second z gate — a dumpster
    // is 1.2 m tall, so a segment that only passes through the lamp heads
    // leaves here — and a box test only for the quarter of corners that draw.
    // Placed after the kerb loop so t_end is already as short as the kerbs
    // could make it, and before the sphere trace so a bin standing in front
    // of a car correctly stops the car being resolved at all.
    if (cc.built && min(za, zb) < 1.5 && fp < cc_streetlife_CAR_FAR_FP) {
        for (var k: i32 = 0; k < 4; k = k + 1) {
            let b = cc_streetlife_corner_prop(ci, cc, k);
            if (!b.ok) {
                continue;
            }
            let hb = cc_streetlife_trace_bin(o, inv_dir, dir, t0, t_end, b,
                                             ci, k);
            if (hb.hit) {
                res = cc_streetlife_nearer(res, hb);
                t_end = min(t_end, res.t);
            }
        }
    }
    // ONE sphere trace per cell, on the nearest car bounding box any kerb
    // accepted — never one per slot. This is the most consequential
    // structural decision in the file and it was forced by measurement.
    // cell_props is inlined into the core's DDA, whose loop runs up to 512
    // times, and the four-kerb by five-slot loops are small enough that the
    // driver unrolls them: written at the slot, the trace appeared twenty
    // times in that loop body, and the register pressure alone cost 1.0 s of
    // a 1.6 s frame set on the `base` view, where no car is anywhere near
    // the footprint that admits an SDF. Hoisted here it appears once.
    // The price is one artifact: a ray that enters a car's box and then
    // grazes past the hull returns a miss rather than falling through to the
    // car behind it. Parked cars sit 2.4 m apart, so that is a sliver on a
    // silhouette edge, and it buys back the whole rest of the scene.
    if (have && cand_t <= t_end + 1e-4) {
        let hit = cc_streetlife_trace_car(o, dir, cand.ta, cand.tb, cand.car,
                                          ci, cand.side, fp);
        if (hit.hit && (!res.hit || hit.t < res.t)) {
            res = hit;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

// Underglow palette. Deliberately the same draw as aircars' — 60% cyan, 25%
// magenta, 15% amber — so a street of parked cars and the lane of flying ones
// above it are visibly the same fleet.
fn cc_streetlife_glow_color(d: f32) -> vec3<f32> {
    if (d < 0.60) {
        return vec3<f32>(0.16, 0.90, 1.00);
    }
    if (d < 0.85) {
        return vec3<f32>(1.00, 0.20, 0.70);
    }
    return vec3<f32>(1.00, 0.60, 0.16);
}

// Night car paint. Weighted dark on purpose: the road under these cars is a
// clipped sodium wash, so a car is a hole in it with a lit edge, and a
// palette of mid-greys renders a street of pale ceramic bathtubs. One car in
// seven is light enough to be the bright one in the row.
fn cc_streetlife_paint(d: f32) -> vec3<f32> {
    if (d < 0.30) {
        return vec3<f32>(0.055, 0.060, 0.070);  // graphite
    }
    if (d < 0.44) {
        return vec3<f32>(0.320, 0.335, 0.350);  // silver
    }
    if (d < 0.58) {
        return vec3<f32>(0.230, 0.055, 0.055);  // oxblood
    }
    if (d < 0.72) {
        return vec3<f32>(0.045, 0.090, 0.185);  // midnight blue
    }
    if (d < 0.82) {
        return vec3<f32>(0.300, 0.260, 0.170);  // sand
    }
    if (d < 0.93) {
        return vec3<f32>(0.040, 0.155, 0.130);  // deep teal
    }
    return vec3<f32>(0.235, 0.085, 0.150);      // faded plum
}

// A band of width `w` around `x0`, antialiased against the footprint and
// blended to its own mean coverage once the line is sub-pixel — a seam that
// simply vanished would take the panel's mean brightness with it.
fn cc_streetlife_seam(x: f32, x0: f32, w: f32, pitch: f32, fp: f32) -> f32 {
    let e = 0.5 * w + 0.6 * fp;
    let sharp = 1.0 - smoothstep(0.5 * w, e + 1e-4, abs(x - x0));
    let mean = w / max(pitch, 1e-3);
    return mix(sharp, mean,
               smoothstep(cc_streetlife_DETAIL_LOD.x,
                          cc_streetlife_DETAIL_LOD.y, fp));
}

// The footprint the CAR's own detail is resolved at, which is not the one the
// core hands the shade hook.
//
// The core passes `fp = pixel_angle * t`, where pixel_angle is floored by the
// app's view-step LOD slider — tan(0.3 deg) by default, four and a half times
// the actual pixel at 960 px and 60 degrees. That floor is right for what it
// was built for: it stops sub-pixel window LATTICES from moireing as the
// camera moves. A car's wiper blade, rim spoke or lamp lens is not a lattice,
// it is one feature, and blurring it across four pixels throws away detail
// the accumulation would otherwise resolve — the whole reason the LOD floor
// dropped to a quarter pixel in the first place.
//
// So: sharpen toward the true pixel, but never below about one and a half of
// them, which is what keeps a moving 1-spp frame from crawling. With the
// default slider this lands at ~1.5 px; with a fine slider the true-pixel
// term takes over and holds the floor at 1 px.
fn cc_streetlife_fp_px(fp: f32, t: f32) -> f32 {
    return max(0.35 * fp,
               2.0 * u.cam_origin.w / max(u.params.x, 1.0) * max(t, 0.0));
}

// One lamp LENS on a face, resolved while it is bigger than a pixel and
// handed to the face's mean when it is not (aircars' treatment, same
// reasoning). The lens is an ellipse, not a disc, and it is nested inside the
// letterbox recess the SDF cut for it: a circular lens of radius 0.12
// overflowed a housing only 0.16 tall, so it rendered as a white ball with a
// dark eyebrow, and a row of parked cars looked back at the camera. Real
// vehicle lamps are wide and shallow; the aspect alone does most of the work.
const cc_streetlife_LENS_A: f32 = 0.160;   // half-width, along the face
const cc_streetlife_LENS_B: f32 = 0.049;   // half-height
fn cc_streetlife_dot(a: f32, b: f32, sa: f32, sb: f32, span: f32, fp: f32)
        -> f32 {
    let e = vec2<f32>((a - sa) / cc_streetlife_LENS_A,
                      (b - sb) / cc_streetlife_LENS_B);
    let d = length(e);
    // The edge softens in the ellipse's OWN metric, with the footprint
    // normalised by the geometric mean of the two semi-axes and then clamped.
    // Both of the obvious alternatives failed on this shape: dividing fp by
    // the semi-MINOR axis alone inflates it eighteenfold and blew the lens
    // into a lobe covering the whole nose, while normalising by the implicit
    // gradient, (d-1)/|grad d|, looks exact but asymptotes to a constant —
    // LENS_B, 0.056 — far from an eccentric ellipse, so once the footprint
    // crossed that constant the test stopped bounding anything at all and the
    // lens grew vertically without limit. A clamped width in the normalised
    // metric cannot do either: the lens is never more than 1 + w across.
    let w = clamp(fp / sqrt(cc_streetlife_LENS_A * cc_streetlife_LENS_B),
                  0.06, 0.42);
    let sharp = 1.0 - smoothstep(1.0 - w, 1.0 + w, d);
    let mean = 3.14159265 * cc_streetlife_LENS_A * cc_streetlife_LENS_B
             / max(span, 1e-3);
    return mix(sharp, mean,
               smoothstep(cc_streetlife_LAMP_LOD.x,
                          cc_streetlife_LAMP_LOD.y, fp));
}

// How far into the glazing a point on the greenhouse lies, in body units.
//
// The greenhouse is bounded by five planes — windshield, backlight, roof
// rail, belt line, tumblehome flank — and the window surround is the distance
// to the nearest of them. The catch is that the border you are STANDING on is
// at distance zero by definition, so a naive minimum reports "no glass"
// everywhere. Each border is therefore pushed out of the running in
// proportion to how closely the surface normal agrees with the border's own,
// and only when it agrees closely: the ramp starts at dot 0.82, so a
// windshield is excused from its own plane but not from the roof rail it runs
// up to, even though the two are only 55 degrees apart.
//
// Five dot products buys A-pillars, C-pillars, a roof rail and a belt line
// that are all the same band, on every face, with no branch and no per-face
// special case — which is what the first pass tried to get from one `ay`
// bound and one normal test, and got windowless cars instead.
fn cc_streetlife_glass_inset(cx: f32, q: vec3<f32>, nl: vec3<f32>) -> f32 {
    let ay = abs(q.y);
    let sy = select(-1.0, 1.0, q.y >= 0.0);
    let nf = vec3<f32>(0.824, 0.0, 0.567);          // windshield
    let nb = vec3<f32>(-0.745, 0.0, 0.667);         // backlight
    let ns = vec3<f32>(0.0, sy * 0.958, 0.287);     // flank, leaning in
    var m = 1.339 - (nf.x * cx + nf.z * q.z)
          + 12.0 * max(dot(nl, nf) - 0.82, 0.0);
    m = min(m, 1.361 - (nb.x * cx + nb.z * q.z)
               + 12.0 * max(dot(nl, nb) - 0.82, 0.0));
    m = min(m, 1.425 - q.z + 12.0 * max(nl.z - 0.82, 0.0));
    m = min(m, q.z - 1.045 + 12.0 * max(-nl.z - 0.82, 0.0));
    m = min(m, 0.958 * (0.74 - 0.30 * (q.z - 1.02) - ay)
               + 12.0 * max(dot(nl, ns) - 0.82, 0.0));
    return m;
}

fn cc_streetlife_pole_shade(h: CityHit, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let fill = cc_streetlife_fill(h.pos);
    // Two edges to approach, never to step over: the population edge at
    // CITY_PROP_RANGE, and the footprint at which poles stop being traced.
    // The second was declared (POLE_FAR_FADE) and then never applied, so a
    // luminaire at radiance 6 — the brightest thing on the street — simply
    // switched off the instant fp crossed POLE_FAR_FP. That is the one thing
    // the SPEC says outright must not happen: sub-pixel detail dissolves into
    // its own mean, it does not vanish.
    let fade = (1.0 - smoothstep(cc_streetlife_FADE_START * CITY_PROP_RANGE,
                                 CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_streetlife_POLE_FAR_FADE,
                                 cc_streetlife_POLE_FAR_FP, fp));
    if (h.kind == 101) {
        // The luminaire. Its underside is the lamp; its sides are the lens
        // edge; its top is a painted aluminium housing, and dark. A glowing
        // box would read as a floating lozenge, not as a light fitting.
        let district = city_glow_sample(h.pos.xy, 2.0);
        let out = 0.45 + 1.6 * smoothstep(0.02, 0.45, district);
        if (h.normal.z < -0.5) {
            return cc_streetlife_HEAD_COLOR
                   * (cc_streetlife_HEAD_RAD * out * fade);
        }
        if (h.normal.z > 0.5) {
            return 0.10 * fill + vec3<f32>(0.004, 0.003, 0.002);
        }
        // A side face is mostly painted cowl, with the lens showing as a lip
        // along its bottom edge. Uniformly bright, the sides clipped to white
        // with the lamp and the whole fitting became one glowing lozenge —
        // the exact failure the head geometry exists to avoid. Antialiased
        // against fp so the lip fades into the face's own mean rather than
        // strobing once it is thinner than a pixel.
        let zl = cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ;
        let lip = 1.0 - smoothstep(zl + cc_streetlife_HEAD_LIP,
                                   zl + cc_streetlife_HEAD_LIP + 0.5 * fp
                                       + 0.012,
                                   h.pos.z);
        let mean = cc_streetlife_HEAD_LIP
                 / (2.0 * cc_streetlife_HEAD_HZ);
        let k = mix(lip, mean, smoothstep(0.010, 0.075, fp));
        return cc_streetlife_HEAD_COLOR
               * (cc_streetlife_HEAD_RAD * out * fade
                  * mix(cc_streetlife_HEAD_COWL,
                        cc_streetlife_HEAD_SIDE, k))
             + 0.08 * fill;
    }
    // Mast and arm: galvanised steel, lit almost entirely by its own lamp,
    // and more of it the closer to the head — the falloff up the pole is the
    // single cue that says the light is at the top.
    let up = clamp(h.pos.z / cc_streetlife_POLE_H, 0.0, 1.0);
    let near_lamp = 0.12 + 0.88 * up * up;
    let pool = cc_streetlife_pool(h.pos.xy);
    let lamp = cc_streetlife_nearest_lamp(h.pos.xy);
    let l = normalize(lamp - h.pos);
    // The mast is a box because a box is what the DDA wants, but 17 cm of
    // steel is round, and from two metres away a flat-shaded rectangle says
    // so loudly. The silhouette stays square — sub-pixel at any distance you
    // would notice — while the NORMAL is remapped across the width of the
    // face to the cylinder the box circumscribes.
    //
    // The remap has to be driven by WHERE ON THE FACE the hit is, and the
    // draft drove it by the direction to the lamp instead. On a mast the lamp
    // is directly overhead, so that vector is nearly zero in xy, the remap
    // collapsed to the identity, and the note beside it recording that "the
    // remapped normal on its own changed nothing" was reporting a bug rather
    // than a fact about the geometry. Every face then carried ONE normal and
    // one grazing value, which is why a pole came out as a flat gold bar with
    // a dark side rather than as a cylinder with two bright edges.
    //
    // The axis is recoverable exactly: cc_streetlife_nearest_lamp returns the
    // lattice point, which is where the mast stands. The offset of the hit
    // from it, resolved along the face, is the cylinder angle.
    var n = h.normal;
    let rel = h.pos.xy - lamp.xy;
    if (abs(h.normal.z) < 0.5
        && dot(rel, rel) < cc_streetlife_MAST_R * cc_streetlife_MAST_R * 2.9) {
        let tang = vec2<f32>(-h.normal.y, h.normal.x);
        let uu = clamp(dot(rel, tang) / cc_streetlife_MAST_R, -1.0, 1.0);
        n = normalize(vec3<f32>(h.normal.xy * sqrt(max(1.0 - uu * uu, 0.0))
                                + tang * uu, 0.0));
    }
    let lam = 0.30 + 0.70 * max(dot(n, l), 0.0);
    // A vertical pole lit by a lamp on its own axis has almost no shading
    // variation around its circumference — that is the geometry, not a bug,
    // and it is why Lambert alone cannot draw a pole. What makes one look
    // round at night is the grazing edge: the two sides of the cylinder catch
    // the street at glancing incidence and the middle of the face returns
    // almost nothing. So the cylinder is read out through a Fresnel edge —
    // which needs a normal that actually turns across the face, hence the
    // remap above.
    let edge = pow(1.0 - clamp(abs(dot(dir, n)), 0.0, 1.0), 3.0);
    return (cc_streetlife_MAST_FILL * fill
            + cc_streetlife_MAST_ALB * pool * (near_lamp * lam)
            + pool * (cc_streetlife_MAST_EDGE * edge * near_lamp)) * fade;
}

fn cc_streetlife_bin_shade(h: CityHit, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let side = h.kind - 106;
    let fill = cc_streetlife_fill(h.pos);
    let pool = cc_streetlife_pool(h.pos.xy);
    let lamp = cc_streetlife_nearest_lamp(h.pos.xy);
    let l = normalize(lamp - h.pos);
    let lam = 0.22 + 0.78 * max(dot(h.normal, l), 0.0);
    // Steel, painted once and repainted never: a dull olive that the sodium
    // pulls most of the colour out of anyway.
    let body = vec3<f32>(0.20, 0.24, 0.18);
    // The lid, and one rib per side. Both are dark lines, not geometry.
    var k = 1.0;
    if (h.normal.z < 0.5) {
        k = k * (1.0 - 0.55 * cc_streetlife_seam(h.pos.z, 0.98, 0.05, 1.2, fp));
        let along = select(h.pos.y, h.pos.x, side >= 2);
        k = k * (1.0 - 0.35 * cc_streetlife_seam(fract(along * 1.6), 0.5,
                                                 0.07, 1.0, fp));
    }
    return cc_streetlife_BIN_ALB * body * pool * lam * k
         + 0.5 * cc_streetlife_BIN_ALB * body * fill;
}

fn cc_streetlife_car_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let side = h.kind - 102;
    let s = cc_streetlife_side(h.cell, cc, side);
    // Recover the slot from the hit itself: the car's centre is within 3.1 m
    // of its slot centre and the slots are 7 m apart, so the floor is exact.
    let along = select(h.pos.y, h.pos.x, s.ax == 0);
    let j = i32(floor(along / cc_streetlife_SLOT));
    let v = cc_streetlife_prop(h.cell, cc, s, side, j);
    let r = v.r;
    let hover = cc_streetlife_is_hover(r);
    let lift = select(0.0, cc_streetlife_HOVER_LIFT, hover);

    let sh = cc_streetlife_car_shape(r);
    let cab_c = sh.y;
    let off = cab_c + 0.30;
    // The same body frame the SDF works in, scale and all, so every band
    // below lands on the geometry it names.
    let q = (cc_streetlife_to_local(h.pos, v.ctr, v.fwd, v.rgt)
             - vec3<f32>(0.0, 0.0, lift)) / sh.x;
    let nl = vec3<f32>(dot(h.normal.xy, v.fwd), dot(h.normal.xy, v.rgt),
                       h.normal.z);
    let ay = abs(q.y);
    // Detail resolves at fpd; the distance FADES below still key off the
    // core's fp, because those follow the app's LOD slider by design and a
    // car must not outlive the population edge just because it is sharp.
    let fpd = cc_streetlife_fp_px(fp, h.t);

    let fill = cc_streetlife_fill(h.pos);
    let pool = cc_streetlife_pool(h.pos.xy);
    let lamp = cc_streetlife_nearest_lamp(h.pos.xy);
    let l = normalize(lamp - h.pos);
    let fade = (1.0 - smoothstep(cc_streetlife_FADE_START * CITY_PROP_RANGE,
                                 CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_streetlife_CAR_FAR_FADE,
                                 cc_streetlife_CAR_FAR_FP, fp));

    // The one specular lobe is what carries curvature. A dark paint under a
    // single overhead source is nearly flat in Lambert; the glint sliding
    // along the shoulder is the whole read.
    let rf = reflect(dir, h.normal);
    let spec_c = max(dot(rf, l), 0.0);
    let lam = max(dot(h.normal, l), 0.0);

    // Tyres first: matte rubber, and the only part that wants none of the
    // clearcoat treatment. The wheel only reads at all because of the rim
    // face inside it — a black disc on a black car under a dim lamp is
    // nothing, and the first pass had cars that appeared to float.
    let qw = vec3<f32>(abs(q.x) - cc_streetlife_AXLE_X,
                       ay - cc_streetlife_TRACK,
                       q.z - cc_streetlife_AXLE_Z);
    let wheel = cc_streetlife_cyl_y(qw, cc_streetlife_TYRE_R,
                                    cc_streetlife_TYRE_HW);
    if (!hover && wheel < 0.035) {
        let rad = length(vec2<f32>(qw.x, qw.z));
        // The rim: the only part of a wheel that is legible at night, and the
        // first pass gave it a quarter of the tyre's radius, so it never
        // showed and the wheels read as flat pale discs. The face is most of
        // the wheel, as it is on a real one; the tyre is the band around it.
        if (qw.y > 0.085 && rad < cc_streetlife_RIM_R) {
            // Brushed metal, five spokes, a hub, and a rolled lip at the rim
            // edge that catches the road — that lip is what turns a disc into
            // something with depth.
            let ang = atan2(qw.z, qw.x) * 0.795774715;   // turns
            let spoke = cc_streetlife_seam(fract(ang * 5.0), 0.5, 0.30, 1.0,
                                           fpd * 3.0);
            let hub = 1.0 - smoothstep(0.048, 0.070, rad);
            let lip = smoothstep(cc_streetlife_RIM_R - 0.035,
                                 cc_streetlife_RIM_R - 0.008, rad);
            let face = max(max(1.0 - 0.80 * spoke, hub), 0.85 * lip);
            return vec3<f32>(0.94, 0.97, 1.00) * cc_streetlife_RIM_ALB
                   * pool * (0.26 + 0.74 * lam) * face
                 + 0.6 * fill;
        }
        // Rubber. Weathered tyre reflectance is about 0.02 in daylight and
        // less than that here, and it has to come out DARKER than graphite
        // paint or the car floats on four pale coins — which is precisely
        // what the first pass rendered, because 0.012 of a clipped sodium
        // road still beats 0.10 of a 0.055 paint.
        let tread = 1.0 - 0.42 * cc_streetlife_seam(
            fract(atan2(qw.z, qw.x) * 4.6), 0.5, 0.15, 1.0, fpd * 4.0);
        return vec3<f32>(cc_streetlife_TYRE_ALB) * pool * (0.18 + 0.82 * lam)
               * tread
             + 0.16 * fill
             + pool * (0.008 * pow(spec_c, 12.0));
    }

    // GLASS. This test decides whether a car has windows at all, and the
    // salvaged draft's version answered no everywhere in the city — the two
    // renders that motivated the rewrite showed a solid loaf of paint where
    // the cabin should be. Two independent reasons, both worth stating
    // because both are the kind of test that looks obviously right:
    //   * `ay < 0.71` against a cabin whose own half-width is 0.70, inflated
    //     outward by the shoulder's smooth-min: the side glass sat a few
    //     millimetres OUTSIDE its own window, so the flanks were never glass.
    //   * `nl.z < 0.72` on a raked windshield, whose normal is (0.82, 0,
    //     0.57) by construction: the more like a windscreen the windscreen
    //     got, the more certainly it was classified as bodywork.
    // So the region is now the greenhouse's own five-plane interior, inset by
    // a surround (cc_streetlife_glass_inset), with the roof panel taken back
    // out — a car may have a raked screen at every angle, but not a glass
    // roof.
    let cx = q.x - cab_c;
    let roof_face = q.z > 1.33 && nl.z > 0.80;
    let gin = cc_streetlife_glass_inset(cx, q, nl);
    // A B-pillar between the two side lights, while it is wide enough to be a
    // pillar rather than an aliasing line; past 6 cm/px the greenhouse is
    // uniform glass, which is that band's honest mean.
    let b_pillar = select(0.0,
                          1.0 - smoothstep(0.034, 0.062, abs(cx + 0.02)),
                          fpd < 0.06);
    let is_glass = q.z > 1.05 && gin > 0.055 + 0.35 * fpd
                && !roof_face && b_pillar < 0.5;
    if (is_glass) {
        let fres = pow(1.0 - clamp(abs(dot(dir, h.normal)), 0.0, 1.0), 4.0);
        // What a parked car's glass carries at night, in order of how much of
        // it there is: the road it is standing on (bright, and reflected by
        // every window that leans at all), the skyglow, and a wash of the
        // building opposite. Without the road term a dark car's greenhouse
        // is the same value as its paint and the whole cabin stops existing.
        let env = mix(cc_streetlife_GLASS_ROAD * pool,
                      3.0 * fill
                      + CITY_PALETTE_MEAN * (0.05 + 0.22 * cc.lit_frac),
                      clamp(rf.z * 1.8 + 0.45, 0.0, 1.0));
        let glint = pow(spec_c, 220.0) * cc_streetlife_GLASS_GLOSS;
        // Wiper blades cross the glass as hard dark lines; the windshield
        // also carries the demist banding at its base.
        let wip = cc_streetlife_seg(vec3<f32>(q.x - off, ay, q.z),
                                    vec3<f32>(0.66, 0.07, 0.925),
                                    vec3<f32>(1.00, 0.52, 0.908), 0.021);
        let wmask = select(1.0, 0.30, wip < 0.014);
        return ((1.0 + 2.2 * fres) * env + pool * glint) * wmask * fade;
    }

    // Paint.
    var col = cc_streetlife_paint(fract(r.z * 5.17 + r.w * 0.31));
    // Panel lines, in the hull's own frame. A shut line at the door, one at
    // the rear quarter, a rocker crease under the sill, and a hood seam on
    // the deck: functional lines only, nothing that would be noise at 10 m.
    var seam = 1.0;
    if (abs(nl.z) < 0.75) {
        seam = seam * (1.0 - 0.75 * cc_streetlife_seam(cx, 0.86, 0.05,
                                                       1.6, fpd));
        seam = seam * (1.0 - 0.75 * cc_streetlife_seam(cx, -0.78, 0.05,
                                                       1.6, fpd));
        seam = seam * (1.0 - 0.40 * cc_streetlife_seam(q.z, 0.46, 0.06,
                                                       1.0, fpd));
        // Door handles: one recess per door, on the belt line.
        let dh = (1.0 - smoothstep(0.10, 0.17, abs(cx + 0.18)))
               * (1.0 - smoothstep(0.02, 0.045, abs(q.z - 0.80)));
        seam = seam * (1.0 - 0.55 * select(0.0, dh, fpd < 0.05));
    } else {
        seam = seam * (1.0 - 0.55 * cc_streetlife_seam(cx, 1.28, 0.045,
                                                       2.0, fpd));
        seam = seam * (1.0 - 0.55 * cc_streetlife_seam(cx, -0.96, 0.045,
                                                       2.0, fpd));
    }
    var e = cc_streetlife_PAINT_GAIN * col * pool * (0.22 + 0.78 * lam) * seam
          + 1.4 * col * fill;
    // Clearcoat: a tight lobe on the sodium, plus a broad sheen that picks up
    // the whole lit street. Both scale with the pool, so a car parked between
    // lamps stays a silhouette.
    e = e + pool * (cc_streetlife_GLOSS * pow(spec_c, 60.0)
                    + cc_streetlife_SHEEN * pow(spec_c, 6.0));
    // Rim: the skyglow catching the top of a curved shoulder.
    let graze = pow(1.0 - clamp(abs(dot(dir, h.normal)), 0.0, 1.0), 3.0);
    e = e + fill * (2.5 * graze * clamp(nl.z + 0.4, 0.0, 1.0));
    // Road bounce. The single most useful term on the whole car: a vertical
    // flank at night reflects the sodium wash it is parked on, and it does
    // so hardest at grazing incidence and lowest on the body. That is what
    // draws the bright outline around the wheel arch, along the rocker and
    // over the shoulder crease — the specular lobe cannot, because the lamp
    // is overhead and a flank never reflects it toward the camera.
    e = e + pool * (cc_streetlife_ROAD_BOUNCE * graze
                    * clamp(0.55 - 0.75 * nl.z, 0.0, 1.0));

    // Head and tail lamps, in the housings the SDF cut for them.
    let facing = nl.x;
    if (abs(facing) > 0.35) {
        let fwd_face = facing > 0.0;
        let dot_e = cc_streetlife_dot(ay, q.z, 0.50, 0.71, 1.2, fpd);
        // A parked car's lamps are standing lights, not driving beams: white
        // forward, red aft, and both a good deal under the sodium luminaire
        // overhead (radiance 6) rather than clipped alongside it. Tails run
        // brighter than heads because a red lens at this exposure loses two
        // of its three channels.
        let col_l = select(vec3<f32>(1.00, 0.06, 0.03),
                           vec3<f32>(1.00, 0.93, 0.82), fwd_face);
        let rad_l = cc_streetlife_LAMP_RAD * select(1.15, 0.72, fwd_face);
        e = e + col_l * (rad_l * dot_e * fade);
        // A tail light bar on the cars that carry one — the cheapest possible
        // cyberpunk signature, and it survives to a distance the dots do not.
        if (!fwd_face && r.z > 0.55) {
            let bar = (1.0 - smoothstep(0.020, 0.055 + 0.5 * fpd,
                                        abs(q.z - 0.74)))
                    * (1.0 - smoothstep(0.55, 0.72, ay));
            e = e + vec3<f32>(1.00, 0.06, 0.03)
                    * (1.3 * mix(bar, 0.10, smoothstep(0.05, 0.30, fpd))
                       * fade);
        }
    }

    // Underglow: a strip under the sill, or the whole plenum on a hover car.
    let glow_on = r.w < cc_streetlife_GLOW_FRAC;
    if (glow_on) {
        let gc = cc_streetlife_glow_color(fract(r.y * 3.77 + r.z * 0.13));
        let hi_z = select(0.34, 0.30, hover);
        let strip = (1.0 - smoothstep(hi_z - 0.10, hi_z + 0.06, q.z))
                  * clamp(1.0 - abs(nl.z), 0.12, 1.0);
        let amp = select(1.0, 1.7, hover);
        e = e + gc * (cc_streetlife_GLOW_RAD * amp * strip * fade);
    }
    return e;
}

fn cc_streetlife_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    if (h.kind <= 101) {
        return cc_streetlife_pole_shade(h, dir, fp);
    }
    if (h.kind <= 105) {
        return cc_streetlife_car_shade(h, cc, dir, fp);
    }
    return cc_streetlife_bin_shade(h, dir, fp);
}
