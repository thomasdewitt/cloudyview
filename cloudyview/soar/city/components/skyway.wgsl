// skyway — the elevated freeway network.
//
// One system the block grid does not own. Everything else in the city is a
// property of a cell; a freeway is a LINE that runs through thousands of
// them, so it is traced once per ray (`extra_trace`) rather than per visited
// cell, and its whole geometry is closed form: a handful of slab tests
// against a ribbon whose top surface is a piecewise-linear function of the
// distance along it. No marcher, no DDA, no per-cell state.
//
// LATTICE. Routes ride the avenue lattice, sparsely. Blocks are
// `u.ocean_params.x` metres (90 on the shipped tile) and avenues fall every
// `CITY_AVENUE_PERIOD` = 8 blocks, so the finest legal route spacing is 720 m
// — far too dense. The route period is 24 blocks (2160 m), three avenues
// apart, which is what makes the network read as sparse drama rather than as
// a second lattice competing with the streets:
//
//   * an x-running deck on every avenue line with block index y = 0 (mod 24)
//   * a y-running deck on every avenue line with block index x = 8 (mod 24)
//
// The brief asked for the two directions to be offset by 12 blocks. Twelve is
// not a multiple of the avenue period, so a y-route offset by 12 would run
// down the middle of a BLOCK — through the buildings. Eight is the nearest
// offset that keeps both families on avenues (8 and 16 are equidistant from
// 12 and are reflections of each other), so the crossings sit off the
// diagonal without any route ever leaving an avenue. The whole thing is
// defined in world coordinates, so it is endless and identical from every
// camera and on every frame.
//
// Why the avenue matters: `city_cell` insets each plot by CITY_STREET_HALF
// (6 m) plus CITY_AVENUE_EXTRA (10 m) on an avenue boundary, so an avenue is
// a 32 m corridor of guaranteed empty air centred on the line. A 16 m deck on
// the centreline therefore clears the facades by 8 m on each side BY
// CONSTRUCTION, at any height — the flat-roof rule in SPEC's "Respect the
// architecture" is satisfied without a single building test. (Superblocks
// cannot break this either: `merged` groups are anchored on even indices and
// swallow only their odd internal boundary, and avenues are at multiples of
// 8.)
//
// PROFILE. The x-family runs dead flat at 26 m. Where the two families cross,
// the y-family climbs to 34 m and passes over — a real interchange decision,
// and the thing that stops the network being a flat plaid. The rise is
// modelled as the deck's top plane being piecewise linear in the along
// coordinate: flat base, a 200 m linear ramp, 124 m flat at the top, a ramp
// back down. Each piece is an exact slab test in the SHEARED coordinate
// h = z - z_top(a), so the ramps are true sloped slabs, not stair steps, and
// consecutive pieces share their end planes exactly — there is nowhere for a
// gap to open.
//
// Under the raised span the pylons would spear the deck they are crossing, so
// pylons inside CROSS_CLEAR of a crossing are simply absent: the high span
// carries itself across a 110 m gap, which is what a real overpass does.
//
// STRUCTURE. Three things, and no more, because the calibration is wiper
// blades and not tire brands: paired columns on 55 m bays, a pier CAP spanning
// each pair (a column running into the underside of a slab is the detail whose
// absence reads as unbuilt — the crosshead is what turns a row of sticks into
// a colonnade), and an expansion JOINT in the running surface over every bay,
// which is the only thing that says a viaduct is a chain of spans rather than
// a poured ribbon. Bearings, drainage, parapet posts and sign gantries are all
// below that line and are not here.
//
// CANDIDATE ROUTES. The expensive mistake here would be looping over route
// lines. Instead: the whole component lives in z ∈ [0, 35.1], so clip the ray
// to that slab first (this alone rejects every sky ray and most cloud rays
// for free), read off the interval of the lateral coordinate the ray spans
// inside it, and convert that interval directly into a range of route
// indices. For any ray that is not both near-horizontal AND inside the slab
// that range holds 0, 1 or 2 entries. Rays that are — a camera parked on the
// deck looking down it — get the nearest three, ordered by t, which is
// exactly the ordering that matters for an opaque first hit.
//
// LIGHT. The artery read from altitude is NOT the lamp heads: a 0.36 m rail
// at 4 m/px covers a tenth of a pixel and delivers a tenth of its radiance.
// It is what those strings put on the 16 m of DECK — the wash, plus the
// traffic under it — which is what actually makes a lit road photograph as a
// continuous bright ribbon from the air. The string dashes (radiance 5,
// warm white, every 9 m along both rail tops, collapsing into a continuous
// line of the same mean energy once a dash is sub-pixel) are the close read;
// the deck's own mean radiance, about 0.47, is the far one — roughly half
// wash and half frozen traffic.
//
// A note on what "bright" can mean here. At exposure 6 the tone map is
// nearly flat above radiance 1: 0.47 renders at 0.85 display and 1.7 (a
// downtown sodium pool at its peak) at 0.97. No honest deck radiance makes an
// artery outshine a lit avenue, and chasing one only blows out the close
// views. What separates the network from the lattice at a kilometre is
// therefore CONTINUITY and COLOUR — an unbroken near-white ribbon cutting
// diagonally across chains of orange dots, at its own altitude, three avenues
// from the next one. That is also how a real freeway reads from a plane.
//
// COST. Per ray: one z-slab clip (which returns immediately for anything
// looking at the sky), then per family one lateral-interval-to-index
// derivation and at most three corridor rejects, each two divides. A corridor
// the ray actually enters costs, worst case, five envelope slabs (only one of
// which can be non-empty for a ray that is not running along the deck), three
// detail slabs for the piece that hit, and — over at most three bays — one
// cap box and two column boxes each. The common city pixel, looking at a
// facade 2 km from any route, pays the z clip and six corridor rejects and
// nothing else.
//
// Measured on a 5080 at 200 accumulated passes, 960x540, registry entry on vs
// off, three interleaved repetitions per camera (interleaved because the box
// is SHARED and a block of A followed by a block of B measures the drift
// between them; the numbers below are the minimum of three and are contended
// regardless, so read them as an upper bound):
//
//     sky only   0.860 on / 0.850 off        artery aerial  0.470 / 0.460
//     on deck    0.350 on / 0.350 off        under deck     0.380 / 0.370
//
// That is +0.01 s on 200 passes where it is non-zero at all — 50 microseconds
// a pass, and exactly one tick of the harness's 10 ms print granularity. The
// honest reading is that the component's cost is at or below what this rig can
// measure, not that it is precisely 50 us.
//
// One caveat on the last two rows, since it would be easy to quote them as an
// occlusion win: with skyway disabled those cameras are not looking at the
// same scene at all — they are parked in mid-air over an empty avenue — so
// they are not an A/B of anything. An earlier revision of this header claimed
// a canyon view got twice as FAST with the network, from that same confound.
// It does not reproduce, and the claim is withdrawn.

// --- lattice ---------------------------------------------------------------
// In blocks, so the network scales with the tile's own cell size.
const cc_skyway_ROUTE_BLOCKS: f32 = 24.0;   // 3 avenues between routes
const cc_skyway_OFFSET_BLOCKS: f32 = 8.0;   // y-family phase (one avenue)

// --- deck ------------------------------------------------------------------
const cc_skyway_HW: f32 = 8.0;        // half width: a 16 m deck in a 32 m avenue
const cc_skyway_THICK: f32 = 1.4;
const cc_skyway_DECK_Z: f32 = 26.0;   // top of the base deck
const cc_skyway_HIGH_Z: f32 = 34.0;   // top of a raised crossing
const cc_skyway_RAMP: f32 = 200.0;    // linear approach length
const cc_skyway_FLAT_HALF: f32 = 62.0; // half the flat run over a crossing
const cc_skyway_RAIL_H: f32 = 1.1;
const cc_skyway_RAIL_W: f32 = 0.36;
const cc_skyway_PYL_SP: f32 = 55.0;   // pylon bays
const cc_skyway_PYL_HW: f32 = 0.6;    // 1.2 m square
const cc_skyway_PYL_OFF: f32 = 5.2;   // paired, in from the deck edges
// Pier cap: the crosshead the two columns of a bay carry, and what the deck
// actually sits on. A column meeting a slab edge-on is the thing that reads as
// unbuilt — every real viaduct puts a transverse beam in between, and at
// wiper-blades calibration that beam IS the feature (no bearings, no pintles,
// no plaque). It spans the pair and oversails each column by CAP_OS, so the
// colonnade under a deck reads as a row of T's rather than a row of sticks.
const cc_skyway_CAP_H: f32 = 1.05;    // depth of the crosshead
const cc_skyway_CAP_HL: f32 = 1.15;   // half-length along the deck
const cc_skyway_CAP_OS: f32 = 0.55;   // oversail beyond the outer column face
// No pylon within this of a crossing centre: the span it would land in is the
// one the other deck occupies.
const cc_skyway_CROSS_CLEAR: f32 = 22.0;
// The along-range used to choose WHICH crossing a corridor is near. Bounded
// so a near-horizontal ray running kilometres down a deck picks the
// interchange in front of the camera rather than one 15 km away.
const cc_skyway_MC_SPAN: f32 = 1400.0;
// Stands in for "endless" on the along axis. World coordinates are ~1e5 and
// CITY_TRACE_RANGE is 3e4, so this is out of reach without costing precision.
const cc_skyway_FAR: f32 = 1.0e6;
const cc_skyway_Z_TOP: f32 = 35.1;    // HIGH_Z + RAIL_H: the component's slab

// --- light -----------------------------------------------------------------
// SPEC rule 5 yardsticks: lit window 3.5, storefront 2.2, sodium pool 0.7.
const cc_skyway_STR_COLOR: vec3<f32> = vec3<f32>(1.00, 0.85, 0.60);
// The brief's number. It is also, at this exposure, an unfalsifiable one: the
// lit face of a rail lands at radiance 3 either way and the tone map has
// nothing left above 2, so 5 and the 6.5 this file used to carry are the same
// pixel. The rail is 0.36 m of a 17 m cross-section, so the far-field mean
// does not notice either. Kept at the contract value because deviating from it
// buys nothing.
const cc_skyway_STR_RAD: f32 = 5.0;
const cc_skyway_STR_P: f32 = 9.0;     // light-string period along the rail
const cc_skyway_STR_L: f32 = 3.3;     // lit length of each dash
// Where a dash stops being resolvable and becomes a continuous line of the
// same mean energy. A 3.3 m dash is one pixel at ~3.3 m/px, so the hand-off
// straddles the brief's 4 m.
const cc_skyway_STR_FP_LO: f32 = 2.4;
const cc_skyway_STR_FP_HI: f32 = 4.8;
// How much of the string a rail face carries. The strip fixture sits on the
// rail top; the road side sees more of it than the outside does. It is a
// BAND, not the whole face — a 1.1 m barrier lit end to end reads as a broken
// white wall, whereas the same light in the top 0.35 m reads as what it is: a
// dark concrete barrier with a lamp strip running along it.
const cc_skyway_RAIL_FACE_IN: f32 = 0.60;
const cc_skyway_RAIL_FACE_OUT: f32 = 0.34;
const cc_skyway_RAIL_BAND: f32 = 0.35;  // lit depth below the rail top (m)
// The wash the strings throw across the deck. This is half the artery — but
// it is a GRADIENT, not a bar, and the difference is everything. At exposure
// 6 with a white point of 15 the tone map has already spent most of its range
// by radiance 1.5 (0.1 renders at 0.55 display, 1.0 at 0.93, 3.5 at white),
// so a deck lit evenly at 1.3 arrives as a featureless white ribbon with its
// traffic invisible inside it. E-folded over 1.9 m instead, the same deck
// runs 0.95 display under a lamp at the rail down to about 0.2 along the
// centreline: bright margins, dark lanes, and headroom for headlights.
//
// A string of lamps 9 m apart also does not light a road evenly — it scallops
// it, and that rhythm running away toward the vanishing point is most of what
// says "lit road at night" rather than "glowing strip". SCALLOP is the floor
// between lamps, and WASH_AMP is pre-divided by the pattern's own mean (0.857)
// so adding the rhythm does not move the deck's far-field brightness.
const cc_skyway_WASH_AMP: f32 = 1.30;
const cc_skyway_WASH_E: f32 = 1.90;   // e-folding in from the rail (m)
const cc_skyway_SCALLOP: f32 = 0.35;  // between-lamp floor
const cc_skyway_SCALLOP_S: f32 = 2.80; // pool sigma along the deck (m)
const cc_skyway_ASPHALT: f32 = 0.05;  // deck albedo against the skyglow
const cc_skyway_CONCRETE: f32 = 0.055;
// Uplight: the sodium pools on the street 25 m below, thrown back onto the
// underside and the pylons. `city_street_pools` is the same function the
// ground uses, sampled at the point directly below — so the scallops along an
// underside land exactly over the lamps that cause them.
//
// The coefficient is small because what it multiplies is not: `pools` sums
// four lamp terms and carries a x3 factor on an avenue, so it peaks near 6
// right where a deck sits, and `street_scale` multiplies that by up to 2.4
// downtown. The 0.3 the brief suggested puts the soffit at radiance 4 — a
// glowing cream ceiling, brighter than the road on top of it. At 0.010 the
// same geometry peaks near 0.14, which is a dark concrete soffit with warm
// scallops picked out along the lamp line: what the brief actually asked for.
const cc_skyway_UPLIGHT: f32 = 0.010;
const cc_skyway_PYL_UP_E: f32 = 16.0; // uplight e-folds up a pylon (m)
// Transverse girders under the deck. Zero-mean modulation, so it is texture
// and not brightness, and it hands over to flat once it is sub-pixel.
const cc_skyway_RIB_P: f32 = 3.2;
const cc_skyway_RIB: f32 = 0.34;
const cc_skyway_RIB_FP: f32 = 1.20;

// --- lane paint ------------------------------------------------------------
const cc_skyway_PAINT_W: f32 = 0.18;
const cc_skyway_PAINT_AMP: f32 = 0.55;
const cc_skyway_LANE_DIV: f32 = 4.0;  // four 4 m lanes across the 16 m deck
const cc_skyway_EDGE_LINE: f32 = 7.35;
const cc_skyway_CENTRE_LINE: f32 = 0.45;
const cc_skyway_DASH_P: f32 = 12.0;
const cc_skyway_DASH_DUTY: f32 = 0.42;
const cc_skyway_PAINT_FP_LO: f32 = 0.30;
const cc_skyway_PAINT_FP_HI: f32 = 1.10;

// --- expansion joints -------------------------------------------------------
// A viaduct is not a ribbon, it is a chain of spans, and the joint over each
// pier is where that fact becomes visible. One per pylon bay (55 m), the full
// width of the deck, in the running surface and carried down the fascia — the
// same s for both, so the line on the side is the end of the line on top.
//
// Two-tone, because a joint is two things: the GAP (a slot with nothing in it
// — the darkest thing on a lit deck, and the part the eye actually reads) and
// the COMB plates either side of it (bare steel, which under a warm lamp
// strip is a shade brighter than asphalt, not darker). One without the other
// reads as a crack or as a stripe; together they read as hardware.
//
// Derived from s alone, so it does not care whether the bay under it has a
// pylon: segments still butt over a suppressed pier, and the chain stays
// unbroken across a crossing.
// Sized as what a 55 m span actually needs. A span that long moves 40-60 mm
// between summer and winter, which is past what a single sealed gap will take,
// so the joint is a MODULAR one: a wide steel frame with the movement split
// across several seals. That is a genuine 0.7 m of deck hardware — the first
// draft here used 0.38 m, which is a joint for a 20 m span and, at this
// renderer's footprints, one that is never once resolved.
const cc_skyway_JNT_GAP: f32 = 0.11;    // half-width of the sealed slot (m)
const cc_skyway_JNT_PLATE: f32 = 0.35;  // half-width of the whole frame (m)
const cc_skyway_JNT_DARK: f32 = 0.80;   // how much of the wash the slot eats
const cc_skyway_JNT_STEEL: f32 = 0.22;  // plate lift, as a fraction of wash
// Where the 0.70 m frame stops being resolvable, in the ALONG footprint
// (fp_eff, not fp — see the note in the shader). Under the app's default LOD
// slider that puts the last legible joint about 25 m ahead and the pattern
// fully collapsed by 75 m, which is roughly one span and a half: you see the
// joint you are crossing and the one after it, and beyond that the deck is
// honestly smooth rather than dishonestly striped.
const cc_skyway_JNT_FP_LO: f32 = 0.65;
const cc_skyway_JNT_FP_HI: f32 = 2.00;

// --- traffic streaks -------------------------------------------------------
// Four lanes: two per direction, at |lat| = 2 and 6. One hash draw per
// (route, lane, 70 m cell), so with the occupancy below a lane carries a
// trail every ~85 m — inside the brief's 80-200, and dense enough that the
// population's MEAN is a real share of what the deck delivers at a kilometre
// rather than a rounding error on the wash.
const cc_skyway_STK_CELL: f32 = 70.0;
const cc_skyway_STK_L: f32 = 7.0;
const cc_skyway_STK_W: f32 = 0.65;    // lateral sigma (m)
const cc_skyway_STK_P: f32 = 0.85;    // slot occupancy at full district glow
const cc_skyway_STK_TAIL: f32 = 2.2;  // e-folds along the frozen trail
const cc_skyway_STK_HALO: f32 = 0.16;   // glow around the trail core
const cc_skyway_STK_HALO_W: f32 = 2.20; // its sigma (m)
const cc_skyway_STK_FP_LO: f32 = 1.6;
const cc_skyway_STK_FP_HI: f32 = 5.0;
// Headlights blow out; tail lamps must NOT. The tone map desaturates anything
// it clips, so a red at radiance 6 arrives pink-white — the colour that says
// "receding traffic" only survives if the green and blue channels stay off
// the shoulder. Hence a deeply saturated red at a third of the white's
// radiance, which lands at (1.0, 0.64, 0.42) in display: unmistakably red,
// still the brightest thing in its lane.
const cc_skyway_HEAD_COL: vec3<f32> = vec3<f32>(1.00, 0.95, 0.86);
const cc_skyway_HEAD_RAD: f32 = 16.0;
const cc_skyway_TAIL_COL: vec3<f32> = vec3<f32>(1.00, 0.055, 0.018);
const cc_skyway_TAIL_RAD: f32 = 4.5;

// --- hit kinds --------------------------------------------------------------
// kind = 200 + 8*family + part. SPEC allows local in [0, 15]; a stride of 8
// with five parts spends 200..212 of it and leaves the arithmetic to a shift.
// The family bit is what lets the shader rebuild the local frame from nothing
// but the kind and the world position — no per-hit payload to carry.
const cc_skyway_KIND_BASE: i32 = 200;
const cc_skyway_FAM_STRIDE: i32 = 8;
const cc_skyway_P_DECK: i32 = 0;   // running surface
const cc_skyway_P_UNDER: i32 = 1;  // soffit and fascia
const cc_skyway_P_PYL: i32 = 2;    // column
const cc_skyway_P_RAIL: i32 = 3;   // edge barrier
const cc_skyway_P_CAP: i32 = 4;    // pier cap

// The local frame of a route: x = along the deck, y = lateral, z = up. The
// two families differ only by which world axis is "along", so one x/y swap is
// the whole transform — and it is its own inverse, which is why the same call
// maps a normal back to world.
fn cc_skyway_swap(v: vec3<f32>, fam: i32) -> vec3<f32> {
    return select(vec3<f32>(v.y, v.x, v.z), v, fam == 0);
}

// A ray interval being whittled down by slabs. `axis` and `sgn` remember
// which slab last raised the near end, which is the face the ray enters
// through and therefore the one whose normal it wears.
struct cc_skyway_Span {
    t0: f32,
    t1: f32,
    axis: i32,
    sgn: f32,
}

// One slab: keep the part of the interval where a + b*t lies in [lo, hi].
// Works for a sheared coordinate as readily as for a world axis, which is the
// whole trick behind the sloped ramps.
fn cc_skyway_clip(sp: cc_skyway_Span, a: f32, b: f32, lo: f32, hi: f32,
                  axis: i32) -> cc_skyway_Span {
    var r = sp;
    if (abs(b) < 1.0e-9) {
        if (a < lo || a > hi) {
            r.t1 = r.t0 - 1.0;   // parallel and outside: empty forever
        }
        return r;
    }
    let ta = (lo - a) / b;
    let tb = (hi - a) / b;
    let tn = min(ta, tb);
    if (tn > r.t0) {
        r.t0 = tn;
        r.axis = axis;
        r.sgn = select(1.0, -1.0, b > 0.0);
    }
    r.t1 = min(r.t1, max(ta, tb));
    return r;
}

// The entering face's outward normal, in world space. Axes 0 and 1 are the
// along and lateral slabs; axis 2 is the sheared one, whose gradient is
// (-m, 0, 1) in the local frame — that tilt is what makes a ramp shade as a
// ramp rather than as a flat deck at the wrong height.
fn cc_skyway_normal(axis: i32, sgn: f32, m: f32, fam: i32) -> vec3<f32> {
    var nl = vec3<f32>(0.0, 0.0, 1.0);
    if (axis == 0) {
        nl = vec3<f32>(sgn, 0.0, 0.0);
    } else if (axis == 1) {
        nl = vec3<f32>(0.0, sgn, 0.0);
    } else {
        nl = normalize(vec3<f32>(-m, 0.0, 1.0)) * sgn;
    }
    return cc_skyway_swap(nl, fam);
}

struct cc_skyway_Best {
    t: f32,
    nrm: vec3<f32>,
    kind: i32,
}

// The deck's top surface as a function of the along coordinate. The x-family
// never rises; the y-family rises over its crossing with the x-family. Kept
// LINEAR on purpose — it has to agree exactly with the sloped slabs below, or
// the pylons would not meet the underside they hold up.
fn cc_skyway_ztop(a: f32, ac: f32, raised: bool) -> f32 {
    if (!raised) {
        return cc_skyway_DECK_Z;
    }
    let d = abs(a - ac);
    if (d <= cc_skyway_FLAT_HALF) {
        return cc_skyway_HIGH_Z;
    }
    if (d >= cc_skyway_FLAT_HALF + cc_skyway_RAMP) {
        return cc_skyway_DECK_Z;
    }
    return cc_skyway_HIGH_Z
        + (cc_skyway_DECK_Z - cc_skyway_HIGH_Z)
          * (d - cc_skyway_FLAT_HALF) / cc_skyway_RAMP;
}

// One piece of ribbon: the deck slab over [a0, a1] with top plane
// z = zb + m*(a - ab), and the two rails riding on it. An envelope test
// covering deck and rails together gates the three detail tests, so a piece
// the ray never reaches costs three clips and stops.
fn cc_skyway_ribbon(lo3: vec3<f32>, ld3: vec3<f32>, ct0: f32, ct1: f32,
                    a0: f32, a1: f32, ab: f32, zb: f32, m: f32,
                    fam: i32, best: cc_skyway_Best) -> cc_skyway_Best {
    var r = best;
    let h0 = lo3.z - zb - m * (lo3.x - ab);
    let hd = ld3.z - m * ld3.x;
    let seed = cc_skyway_Span(ct0, ct1, -1, 1.0);

    var env = cc_skyway_clip(seed, lo3.x, ld3.x, a0, a1, 0);
    env = cc_skyway_clip(env, lo3.y, ld3.y, -cc_skyway_HW, cc_skyway_HW, 1);
    env = cc_skyway_clip(env, h0, hd, -cc_skyway_THICK, cc_skyway_RAIL_H, 2);
    if (env.t0 > env.t1 || env.t0 >= r.t) {
        return r;
    }

    // The deck itself: everything from the running surface down to the
    // fascia's lower edge.
    var d = cc_skyway_clip(seed, lo3.x, ld3.x, a0, a1, 0);
    d = cc_skyway_clip(d, lo3.y, ld3.y, -cc_skyway_HW, cc_skyway_HW, 1);
    d = cc_skyway_clip(d, h0, hd, -cc_skyway_THICK, 0.0, 2);
    if (d.t0 <= d.t1 && d.axis >= 0 && d.t0 < r.t) {
        let top = d.axis == 2 && d.sgn > 0.0;
        r = cc_skyway_Best(d.t0,
                           cc_skyway_normal(d.axis, d.sgn, m, fam),
                           cc_skyway_KIND_BASE + cc_skyway_FAM_STRIDE * fam
                           + select(1, 0, top));
    }

    // Edge rails, one either side, standing on the deck top.
    for (var side: i32 = 0; side < 2; side = side + 1) {
        let l0 = select(-cc_skyway_HW,
                        cc_skyway_HW - cc_skyway_RAIL_W, side == 1);
        let l1 = select(-cc_skyway_HW + cc_skyway_RAIL_W,
                        cc_skyway_HW, side == 1);
        var q = cc_skyway_clip(seed, lo3.x, ld3.x, a0, a1, 0);
        q = cc_skyway_clip(q, lo3.y, ld3.y, l0, l1, 1);
        q = cc_skyway_clip(q, h0, hd, 0.0, cc_skyway_RAIL_H, 2);
        if (q.t0 <= q.t1 && q.axis >= 0 && q.t0 < r.t) {
            r = cc_skyway_Best(q.t0,
                               cc_skyway_normal(q.axis, q.sgn, m, fam),
                               cc_skyway_KIND_BASE + cc_skyway_FAM_STRIDE * fam
                               + cc_skyway_P_RAIL);
        }
    }
    return r;
}

// Paired pylons, ground to underside. The along position is quantized, so the
// nearest few are found by rounding the ray's own along coordinate rather
// than by searching: take the bay at the entry to the under-deck z-band and
// walk at most three of them in the direction of travel. That is the whole
// colonnade a ray can see before the near pylons occlude the far ones.
fn cc_skyway_pylons(lo3: vec3<f32>, ld3: vec3<f32>, ct0: f32, ct1: f32,
                    ac: f32, raised: bool, fam: i32,
                    best: cc_skyway_Best) -> cc_skyway_Best {
    var r = best;
    var pt0 = ct0;
    var pt1 = ct1;
    let ztop_max = cc_skyway_HIGH_Z - cc_skyway_THICK;
    if (abs(ld3.z) > 1.0e-9) {
        let ta = (0.0 - lo3.z) / ld3.z;
        let tb = (ztop_max - lo3.z) / ld3.z;
        pt0 = max(pt0, min(ta, tb));
        pt1 = min(pt1, max(ta, tb));
    } else if (lo3.z < 0.0 || lo3.z > ztop_max) {
        return r;
    }
    if (pt0 > pt1 || pt0 >= r.t) {
        return r;
    }

    let sa = lo3.x + pt0 * ld3.x;
    let sb = lo3.x + pt1 * ld3.x;
    let na = i32(round(sa / cc_skyway_PYL_SP));
    let nb = i32(round(sb / cc_skyway_PYL_SP));
    let dn = select(-1, 1, nb >= na);
    let seed = cc_skyway_Span(ct0, ct1, -1, 1.0);
    for (var q: i32 = 0; q < 3; q = q + 1) {
        let n = na + q * dn;
        if ((dn > 0 && n > nb) || (dn < 0 && n < nb)) {
            break;
        }
        let s_n = f32(n) * cc_skyway_PYL_SP;
        // The bay a crossing occupies carries no pylon: the high span crosses
        // it clean.
        if (raised && abs(s_n - ac) < cc_skyway_CROSS_CLEAR) {
            continue;
        }
        // The underside at this bay. On a ramp it is sloped, but the cap is
        // only 2.3 m long, so evaluating the profile at the bay centre puts
        // the cap's flat top within 5 cm of it — and that discrepancy is
        // BURIED, because the cap is narrower than the deck on both axes and
        // the deck is the nearer surface wherever the two overlap.
        let soffit = cc_skyway_ztop(s_n, ac, raised) - cc_skyway_THICK;
        let cap_bot = soffit - cc_skyway_CAP_H;

        // The crosshead, spanning the pair.
        let cap_hw = cc_skyway_PYL_OFF + cc_skyway_PYL_HW + cc_skyway_CAP_OS;
        var cp = cc_skyway_clip(seed, lo3.x, ld3.x,
                                s_n - cc_skyway_CAP_HL,
                                s_n + cc_skyway_CAP_HL, 0);
        cp = cc_skyway_clip(cp, lo3.y, ld3.y, -cap_hw, cap_hw, 1);
        cp = cc_skyway_clip(cp, lo3.z, ld3.z, cap_bot, soffit, 2);
        if (cp.t0 <= cp.t1 && cp.axis >= 0 && cp.t0 < r.t) {
            r = cc_skyway_Best(cp.t0,
                               cc_skyway_normal(cp.axis, cp.sgn, 0.0, fam),
                               cc_skyway_KIND_BASE
                               + cc_skyway_FAM_STRIDE * fam
                               + cc_skyway_P_CAP);
        }

        // The columns, ground to the underside of the crosshead.
        for (var side: i32 = 0; side < 2; side = side + 1) {
            let lc = select(-cc_skyway_PYL_OFF, cc_skyway_PYL_OFF, side == 1);
            var p = cc_skyway_clip(seed, lo3.x, ld3.x,
                                   s_n - cc_skyway_PYL_HW,
                                   s_n + cc_skyway_PYL_HW, 0);
            p = cc_skyway_clip(p, lo3.y, ld3.y,
                               lc - cc_skyway_PYL_HW,
                               lc + cc_skyway_PYL_HW, 1);
            p = cc_skyway_clip(p, lo3.z, ld3.z, 0.0, cap_bot, 2);
            if (p.t0 <= p.t1 && p.axis >= 0 && p.t0 < r.t) {
                r = cc_skyway_Best(p.t0,
                                   cc_skyway_normal(p.axis, p.sgn, 0.0, fam),
                                   cc_skyway_KIND_BASE
                                   + cc_skyway_FAM_STRIDE * fam
                                   + cc_skyway_P_PYL);
            }
        }
    }
    return r;
}

// One route line: the corridor reject, then the ribbon pieces and the pylons.
fn cc_skyway_route(o: vec3<f32>, dir: vec3<f32>, fam: i32, lat_c: f32,
                   tz0: f32, tz1: f32, period: f32,
                   best: cc_skyway_Best) -> cc_skyway_Best {
    var r = best;
    let lw = cc_skyway_swap(o, fam);
    let ld3 = cc_skyway_swap(dir, fam);
    // Lateral coordinates measured from the route's own centreline, so every
    // constant below is a half-width and the two families share the code.
    let lo3 = vec3<f32>(lw.x, lw.y - lat_c, lw.z);

    var ct0 = tz0;
    var ct1 = tz1;
    if (abs(ld3.y) > 1.0e-9) {
        let ta = (-cc_skyway_HW - lo3.y) / ld3.y;
        let tb = (cc_skyway_HW - lo3.y) / ld3.y;
        ct0 = max(ct0, min(ta, tb));
        ct1 = min(ct1, max(ta, tb));
    } else if (abs(lo3.y) > cc_skyway_HW) {
        return r;
    }
    if (ct0 > ct1 || ct0 >= r.t) {
        return r;
    }

    let raised = fam == 1;
    if (!raised) {
        // Flat from horizon to horizon.
        return cc_skyway_pylons(
            lo3, ld3, ct0, ct1, 0.0, false, fam,
            cc_skyway_ribbon(lo3, ld3, ct0, ct1,
                             -cc_skyway_FAR, cc_skyway_FAR,
                             0.0, cc_skyway_DECK_Z, 0.0, fam, r));
    }

    // Which crossing this corridor is near. For anything but a ray running
    // along the deck the corridor is metres long and this is exact; for one
    // that is, MC_SPAN picks the interchange ahead of the camera. Farther
    // crossings on the same line stay flat — 2 km down a deck, behind fog and
    // towers, and the pieces still tile the line with no gap.
    let tm = ct0 + 0.5 * min(ct1 - ct0, cc_skyway_MC_SPAN);
    let ac = round((lo3.x + tm * ld3.x) / period) * period;
    let e = cc_skyway_FLAT_HALF;
    let rp = cc_skyway_RAMP;
    let slope = (cc_skyway_HIGH_Z - cc_skyway_DECK_Z) / rp;

    // Five pieces sharing four end planes exactly: base, ramp up, high flat,
    // ramp down, base.
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, -cc_skyway_FAR, ac - e - rp,
                         0.0, cc_skyway_DECK_Z, 0.0, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac - e - rp, ac - e,
                         ac - e - rp, cc_skyway_DECK_Z, slope, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac - e, ac + e,
                         0.0, cc_skyway_HIGH_Z, 0.0, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac + e, ac + e + rp,
                         ac + e, cc_skyway_HIGH_Z, -slope, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac + e + rp, cc_skyway_FAR,
                         0.0, cc_skyway_DECK_Z, 0.0, fam, r);
    return cc_skyway_pylons(lo3, ld3, ct0, ct1, ac, true, fam, r);
}

// `inv_dir` is the hook's, and deliberately unused: the core builds it with
// 1e30 standing in for a zero component, which is fine for an axis-aligned box
// but not for the sheared slab a ramp needs. Every divide below is guarded at
// the point of use instead.
fn cc_skyway_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
        -> CityHit {
    var miss: CityHit;
    miss.hit = false;
    miss.t = 1e30;
    miss.pos = vec3<f32>(0.0);
    miss.normal = vec3<f32>(0.0, 0.0, 1.0);
    miss.kind = 0;
    miss.cell = vec2<i32>(0);

    // Gate 1, and the one that pays for the component: everything skyway owns
    // lives between the ground and 35.1 m. A ray that never enters that slab
    // — every sky ray, every ray already above the deck and climbing — is
    // done in four operations.
    var tz0 = 1.0e-3;
    var tz1 = CITY_TRACE_RANGE;
    if (abs(dir.z) > 1.0e-9) {
        let ta = (0.0 - o.z) / dir.z;
        let tb = (cc_skyway_Z_TOP - o.z) / dir.z;
        tz0 = max(tz0, min(ta, tb));
        tz1 = min(tz1, max(ta, tb));
    } else if (o.z < 0.0 || o.z > cc_skyway_Z_TOP) {
        return miss;
    }
    if (tz0 >= tz1) {
        return miss;
    }

    let cell = u.ocean_params.x;
    let period = cc_skyway_ROUTE_BLOCKS * cell;
    var best = cc_skyway_Best(1e30, vec3<f32>(0.0, 0.0, 1.0), -1);

    for (var fam: i32 = 0; fam < 2; fam = fam + 1) {
        let ld3 = cc_skyway_swap(dir, fam);
        let lw = cc_skyway_swap(o, fam);
        let phase = select(0.0, cc_skyway_OFFSET_BLOCKS * cell, fam == 1);
        // The lateral interval the ray spans while inside the z slab, widened
        // by the deck half-width, converted straight into route indices. This
        // is the analytic step that replaces a loop over lines.
        let wa = lw.y + tz0 * ld3.y;
        let wb = lw.y + tz1 * ld3.y;
        let w_lo = min(wa, wb) - cc_skyway_HW;
        let w_hi = max(wa, wb) + cc_skyway_HW;
        let m_lo = i32(ceil((w_lo - phase) / period));
        let m_hi = i32(floor((w_hi - phase) / period));
        let n_cand = min(m_hi - m_lo + 1, 3);
        // Nearest first: with an opaque network that ordering is what makes
        // the three-line cap harmless.
        let step = select(-1, 1, ld3.y >= 0.0);
        let m_start = select(m_hi, m_lo, ld3.y >= 0.0);
        for (var k: i32 = 0; k < n_cand; k = k + 1) {
            let lat_c = phase + f32(m_start + k * step) * period;
            best = cc_skyway_route(o, dir, fam, lat_c, tz0, tz1, period, best);
        }
    }

    if (best.kind < 0) {
        return miss;
    }
    var res: CityHit;
    res.hit = true;
    res.t = best.t;
    res.pos = o + best.t * dir;
    res.normal = best.nrm;
    res.kind = best.kind;
    res.cell = vec2<i32>(floor(res.pos.xy / cell));
    return res;
}

// Frozen traffic. One draw per (route family, lane, 70 m cell) places a single
// trail; which side of the centreline the lane is on decides which way it is
// going and therefore whether you are looking at its headlights or its tail
// lamps. Only the two lanes on the shaded point's own side are evaluated —
// the far pair is four metres of a 0.65 m sigma away and contributes e^-19.
// Below the LOD gate the trail is resolved; above it, the population's mean
// over the whole deck, both colours, which is the honest asymptote.
fn cc_skyway_streaks(s: f32, lat: f32, fam: i32, glow: f32, fp: f32)
        -> vec3<f32> {
    let sd = select(0u, 1u, lat >= 0.0);
    let side = select(-1.0, 1.0, lat >= 0.0);
    let n = floor(s / cc_skyway_STK_CELL);
    let ni = bitcast<u32>(i32(n));
    let p = cc_skyway_STK_P
        * mix(0.35, 1.25, smoothstep(0.02, 0.35, glow));
    let col = select(cc_skyway_TAIL_COL, cc_skyway_HEAD_COL, sd == 1u);
    let rad = select(cc_skyway_TAIL_RAD, cc_skyway_HEAD_RAD, sd == 1u);

    var f = 0.0;
    for (var slot: i32 = 0; slot < 2; slot = slot + 1) {
        let key = (u32(fam) * 3u + u32(slot)) * 2u + sd;
        let r = city_rand4(vec2<u32>(
            ni * 0x9e3779b9u + key * 0x51ed270bu + 0x165667b1u,
            (ni * 0x85ebca6bu) ^ (key * 0xc2b2ae35u + 0x27d4eb2fu)));
        if (r.x >= p) {
            continue;
        }
        let s_c = (n + 0.15 + 0.70 * r.y) * cc_skyway_STK_CELL;
        let lane = side * (2.0 + 4.0 * f32(slot));
        // Travel is +along on the lat>0 side, -along on the other; the trail
        // is frozen BEHIND the head, so it points against travel.
        let x = -((s - s_c) * side) / cc_skyway_STK_L;
        if (x < 0.0 || x > 1.0) {
            continue;
        }
        let dl = lat - lane;
        // Core plus halo: a lamp seen on a road surface is a hard bright
        // patch inside a soft glow, and without the second term the trails
        // read as painted dashes rather than as light.
        let core = exp(-dl * dl
                       / (2.0 * cc_skyway_STK_W * cc_skyway_STK_W));
        let halo = cc_skyway_STK_HALO
            * exp(-dl * dl / (2.0 * cc_skyway_STK_HALO_W
                              * cc_skyway_STK_HALO_W));
        f = f + (0.55 + 0.90 * r.w) * exp(-cc_skyway_STK_TAIL * x)
                * (core + halo);
    }

    // The population's mean over the deck: occupancy x the trail's integral
    // along s x its lateral integral, over the cell's area, summed over the
    // two lanes of each colour. Written out rather than fitted, so changing
    // STK_L or STK_W keeps the two ends of the LOD agreeing by construction.
    let along_int = cc_skyway_STK_L * (1.0 - exp(-cc_skyway_STK_TAIL))
                    / cc_skyway_STK_TAIL;
    let lat_int = 2.5066 * (cc_skyway_STK_W
                            + cc_skyway_STK_HALO * cc_skyway_STK_HALO_W);
    let unit = 2.0 * p * along_int * lat_int
               / (cc_skyway_STK_CELL * 2.0 * cc_skyway_HW);
    let mean = (cc_skyway_HEAD_COL * cc_skyway_HEAD_RAD
                + cc_skyway_TAIL_COL * cc_skyway_TAIL_RAD) * unit;

    let k = smoothstep(cc_skyway_STK_FP_LO, cc_skyway_STK_FP_HI, fp);
    return mix(col * (rad * f), mean, k);
}

fn cc_skyway_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let code = h.kind - cc_skyway_KIND_BASE;
    let fam = code / cc_skyway_FAM_STRIDE;
    let part = code % cc_skyway_FAM_STRIDE;
    let cell = u.ocean_params.x;
    let period = cc_skyway_ROUTE_BLOCKS * cell;
    let phase = select(0.0, cc_skyway_OFFSET_BLOCKS * cell, fam == 1);

    let lp = cc_skyway_swap(h.pos, fam);
    let lat = lp.y - (phase + round((lp.y - phase) / period) * period);
    let s = lp.x;

    let glow = city_glow_sample(h.pos.xy, 3.0);
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * glow);

    // A footprint for detail that runs ACROSS the deck rather than along it.
    //
    // `fp` is isotropic — the pixel angle times the range — but a deck is a
    // surface nobody looks at squarely, and at grazing incidence a pixel is
    // long in the along direction and unchanged in the lateral one. The two
    // families of detail here therefore do not share a footprint, and giving
    // them one is the mistake to avoid in both directions: charge the lane
    // lines the grazing factor and the paint dissolves five times too early
    // (their WIDTH is lateral and foreshortens not at all); charge the
    // expansion joints the plain `fp` and a transverse band that is a fifth
    // of a pixel still believes itself resolved, and crawls.
    //
    // So: lateral-extent detail (lane-line widths, the streaks' lateral
    // profile) keeps `fp`; along-extent detail that is thin — the joints —
    // gets this. Clamped at 5x like `city_shade`'s window grid, because at
    // true grazing the exact factor stops meaning anything.
    //
    // Worth knowing when sizing any of these windows: `fp` is NOT the naive
    // pixel footprint. `pixel_angle` is `max(2*tan(fov/2)/width, u.periodic.z)`
    // and the LOD floor wins by 3.6x at 960 px with the app's default slider,
    // so every window below is exercised at 3.6x the range you would guess.
    let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);
    // The deck's top surface here, re-derived from the same profile the
    // tracer used — so heights measured against it (the rail's lamp band, the
    // fascia's depth) hold on the ramps as well as on the flats.
    let ztop = select(cc_skyway_DECK_Z,
                      cc_skyway_ztop(s, round(s / period) * period, true),
                      fam == 1);
    let duty = cc_skyway_STR_L / cc_skyway_STR_P;
    // The dash collapses into a continuous line of the same mean energy —
    // which is what a light string a kilometre off actually is.
    let str_lod = smoothstep(cc_skyway_STR_FP_LO, cc_skyway_STR_FP_HI, fp);
    let dash = select(0.0, 1.0, fract(s / cc_skyway_STR_P) < duty);
    let strength = mix(dash, duty, str_lod);

    // The expansion joint over the nearest pier, as a signed profile: -1 in
    // the slot, +1 on the comb plates, 0 on open deck. One round and two
    // compares, and it is shared by the running surface and the fascia so the
    // two agree on where a segment ends.
    let jd = abs(s - round(s / cc_skyway_PYL_SP) * cc_skyway_PYL_SP);
    let j_raw = select(select(0.0, 1.0, jd < cc_skyway_JNT_PLATE),
                       -1.0, jd < cc_skyway_JNT_GAP);
    // Its area mean over a bay, for the footprint where the assembly stops
    // being resolvable. Written from the same three constants the profile is,
    // so widening the slot cannot drift the two ends of the LOD apart.
    let j_mean = (2.0 * (cc_skyway_JNT_PLATE - cc_skyway_JNT_GAP)
                  - 2.0 * cc_skyway_JNT_GAP) / cc_skyway_PYL_SP;
    let joint = mix(j_raw, j_mean,
                    smoothstep(cc_skyway_JNT_FP_LO, cc_skyway_JNT_FP_HI, fp_eff));
    // Slot eats light, plate adds a little. Both are multipliers on whatever
    // is lighting the surface, so a joint in the dark stays dark.
    let j_mul = 1.0 + cc_skyway_JNT_STEEL * max(joint, 0.0)
                    + cc_skyway_JNT_DARK * min(joint, 0.0);

    if (part == cc_skyway_P_RAIL) {
        var face = 1.0;
        if (h.normal.z < 0.5) {
            let nl = cc_skyway_swap(h.normal, fam);
            let side = select(cc_skyway_RAIL_FACE_OUT,
                              cc_skyway_RAIL_FACE_IN, nl.y * lat < 0.0);
            // Only the top band of the barrier carries the fixture.
            let below = ztop + cc_skyway_RAIL_H - h.pos.z;
            face = side * (1.0 - smoothstep(cc_skyway_RAIL_BAND * 0.5,
                                            cc_skyway_RAIL_BAND, below));
        }
        return cc_skyway_CONCRETE * fill
               + cc_skyway_STR_COLOR * (cc_skyway_STR_RAD * strength * face);
    }

    if (part == cc_skyway_P_DECK) {
        // The wash: the strings' light across the deck, and the reason the
        // network reads as an artery from altitude. Exponential across, and
        // scalloped along under the individual lamps.
        let din = max(cc_skyway_HW - abs(lat), 0.0);
        let q = fract(s / cc_skyway_STR_P - 0.5 * duty);
        let dd = min(q, 1.0 - q) * cc_skyway_STR_P;
        let pool = exp(-dd * dd
                       / (2.0 * cc_skyway_SCALLOP_S * cc_skyway_SCALLOP_S));
        // Between-lamp floor plus pool, handed to the pattern's own mean at
        // the same footprint where the dashes themselves dissolve.
        let pool_mean = 2.5066 * cc_skyway_SCALLOP_S / cc_skyway_STR_P;
        let rhythm = cc_skyway_SCALLOP
            + (1.0 - cc_skyway_SCALLOP) * mix(pool, pool_mean, str_lod);
        let wash = exp(-din / cc_skyway_WASH_E) * rhythm;
        var e = cc_skyway_ASPHALT * fill
                + cc_skyway_STR_COLOR * (cc_skyway_WASH_AMP * wash);

        // Lane paint: a double centre line, dashed lane dividers, solid edge
        // lines. Sub-pixel almost everywhere, so it hands over to its own
        // area mean rather than shimmering.
        let aw = cc_skyway_PAINT_W;
        let al = abs(lat);
        var paint = 0.0;
        paint = paint + select(0.0, 1.0,
                               abs(al - cc_skyway_CENTRE_LINE) < aw);
        paint = paint + select(0.0, 1.0,
                               abs(al - cc_skyway_EDGE_LINE) < aw);
        paint = paint + select(0.0, 1.0,
                               abs(al - cc_skyway_LANE_DIV) < aw
                               && fract(s / cc_skyway_DASH_P)
                                  < cc_skyway_DASH_DUTY);
        let paint_mean = (4.0 * aw + 4.0 * aw
                          + 4.0 * aw * cc_skyway_DASH_DUTY)
                         / (2.0 * cc_skyway_HW);
        let paint_l = mix(paint, paint_mean,
                          smoothstep(cc_skyway_PAINT_FP_LO,
                                     cc_skyway_PAINT_FP_HI, fp));
        e = e + cc_skyway_STR_COLOR
                * (cc_skyway_PAINT_AMP * paint_l * (0.20 + wash));

        // The joint crosses the paint and the wash — a comb plate is bare
        // steel, so the lane line stops at it — but NOT the traffic, which is
        // light thrown onto the deck rather than a property of the deck.
        return e * j_mul + cc_skyway_streaks(s, lat, fam, glow, fp);
    }

    // Underside, fascia, pylons: near-black structure carrying the sodium
    // thrown up at it from the street directly below. The pools are the same
    // ones the ground draws, so an underside's scallops sit exactly over the
    // lamps that make them.
    let pools = city_street_pools(h.pos.xy);
    let district = city_glow_sample(h.pos.xy, 2.0);
    let street_scale = 0.20 + 2.2 * smoothstep(0.02, 0.45, district);
    var up = cc_skyway_UPLIGHT * pools * street_scale;
    if (part == cc_skyway_P_PYL) {
        // Column: brightest at its foot, standing in the pool itself.
        up = up * exp(-max(h.pos.z, 0.0) / cc_skyway_PYL_UP_E);
    } else if (part == cc_skyway_P_CAP) {
        // Crosshead. It sits as high as the soffit but is not the soffit:
        // its own soffit faces the street squarely and catches the same
        // uplight, while its ends and sides are sheer. The 1.35 is the one
        // place this component brightens anything — a projecting beam catches
        // light on three faces where the flat soffit above it catches one,
        // and without it the caps read as holes rather than as hardware.
        up = up * select(0.45, 1.35, h.normal.z < -0.5)
                * exp(-max(h.pos.z, 0.0) / (2.0 * cc_skyway_PYL_UP_E));
    } else if (h.normal.z > -0.5) {
        // Fascia, not soffit: it faces sideways, so it catches much less —
        // and it carries the end of the deck's expansion joint, which is the
        // only thing on this component that says where one span stops and the
        // next starts when you are looking at it from the side.
        up = up * 0.35 * j_mul;
    } else {
        // Soffit: the girders the deck is carried on, as shading rather than
        // as geometry — the read is entirely in the banding. The joint runs
        // across them, and being a real gap it is darker here than the ribs.
        let rib = 1.0 + cc_skyway_RIB
            * select(-1.0, 1.0, fract(s / cc_skyway_RIB_P) < 0.5);
        up = up * mix(rib, 1.0, smoothstep(0.0, cc_skyway_RIB_FP, fp)) * j_mul;
    }
    return cc_skyway_CONCRETE * fill + CITY_LAMP_COLOR * up;
}
