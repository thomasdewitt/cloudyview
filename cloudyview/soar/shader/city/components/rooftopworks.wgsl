// rooftopworks — the machinery on the roofs.
//
// A skyline is not a row of clean rectangles. What separates a real one from
// an extruded-footprint one is the last two metres: water tanks, condenser
// banks, vent stacks, a lit penthouse, a parapet with a strip of light along
// it. From a kilometre out none of it is resolved and all of it matters —
// the roofline goes from a ruled edge to a serrated one, and the eye reads
// "buildings people work in" from the serration alone.
//
// RESPECT THE ARCHITECTURE (SPEC, the rule above all others here). Roof
// furniture needs a flat top to stand on:
//   * arch 0 (setback stack) — the full set, on EVERY tier top. A wedding
//     cake has three roofs, not one, and the lower ones are where the tanks
//     go. The tier standing on a deck is carried as a hole in it, so nothing
//     is ever placed inside the mass above.
//   * arch 2 (growth) — the main roof AND the cantilevered bud tops, which
//     are flat shelves hanging over the street and read beautifully with a
//     tank on them.
//   * arch 3 (tapered shaft) — the crown cap ONLY: the footprint scaled by
//     cc.fscale about the frustum's centre. The podium ledge is a metre of
//     sloped shoulder and gets nothing.
//   * arch 4 (spire crown) — NOTHING. Not one box, and the component returns
//     before it draws a hash.
//
// WHAT IS HERE, in the order it matters to the shot:
//
//   1. MECHANICAL CLUSTERS. 2-4 dark boxes per eligible roof: square water
//      tanks (3 x 3 x 4 m, with the rim band that makes them read as tanks
//      and not as crates), condenser rows (three 2 x 2 x 1 m units in a
//      line, sometimes doubled), vent stacks (1 x 1 x 5 m, in pairs). Dark:
//      their whole job at distance is silhouette. ~30% carry one warm
//      service lamp — a bulkhead light on the housing, radiance 2 — which is
//      what puts a few sparks up on the roofs above the core's windows.
//   2. PARAPET. A real low wall around the top deck of ~55% of buildings
//      (the core only fakes one by brightening the roof albedo near the
//      edge), and on ~12% of roofs overall a strip of light along its
//      coping: cool white, or the building's own house colour where it has
//      one. The wall is a hollow box — two slab tests, not four — and it is
//      deliberately NOT clipped to the DDA cell, because it belongs to the
//      building rather than to the cell: every member cell of a merged
//      superblock derives the same ring and catches whatever part of it
//      falls in its own ray segment.
//   3. PENTHOUSE CROWNS are not here: since 2026-11 the crown is core
//      massing (raymarch.wgsl city_crown), so a far tower's silhouette no
//      longer changes at CITY_PROP_RANGE. The same draw places the helipad
//      below, and every piece here keeps clear of what it puts on the roof.
//   4. HELIPADS. On the biggest flat roofs only (a merged superblock, or a
//      footprint over 60 m): a pad slab 0.3 m proud with four warm corner
//      lights at radiance 3. Rare on purpose — a helipad on every tower is
//      a video-game city.
//   5. EXTERNAL PIPEWORK. 1-3 risers, 0.4 m across, running down a facade
//      corner of 30-150 m buildings — the mid-rise band where services are
//      bolted on the outside rather than buried in a core. Dark with a faint
//      rim, gated to close range: it is a texture on the wall, not a
//      silhouette, and past ~230 m there is nothing there to resolve.
//
// DETAIL CALIBRATION (SPEC, global): wiper blades, not tire brands. A tank
// gets its rim band and its legs; a condenser row gets its unit gaps and
// grille bands. Nothing gets a logo, a warning label, or a hatch handle.
//
// COST. Every cell pays: two compares (built, arch 4) and one z-interval
// test against [lowest deck, roof + HEADROOM]. A cell whose ray segment misses
// that band — most of them, since most city pixels are looking at a wall or
// at the road — costs exactly that. Inside the band each piece has its own
// tighter z window and its own eligibility test BEFORE it draws a hash: the
// parapet's 1.25 m band, the clutter's 5 m one, and a crown draw
// (city_crown) that never hashes on a building too short to carry one.
// Worst case is 4 slots x 2 boxes + 2 parapet + 1 pad = 11 slab tests, and that only on a large roof
// filling the segment; the typical accepted cell runs 3-4. Pipework adds one
// box behind a footprint gate that keeps it inside ~230 m.
//
// Measured on the harness (RTX 5080, CONTENDED — other agents and the pilot
// share the card, so these are upper bounds), 960x540 x 48 passes, A/B by
// toggling `enabled` and interleaving the runs: a downward view over a
// district of roofs costs 0.16 s off / 0.18 s on, and the 900 m `base` view
// 0.28 s off / 0.33 s on — call it 12-18% of the city frame where roofs fill
// the screen. A street-level view, where the roof band is behind the walls,
// measures 0.13 s either way. The parapet is the widest-reaching piece: two
// slab tests on nearly every roof cell a downward ray visits, and that IS
// where the money goes. It buys a wall that is actually there.

// --- kinds (kind_base 600, local 0..15) -------------------------------------
// 600 + slot  mechanical body        604 + slot  its accessory piece
const cc_rooftopworks_K_MECH: i32 = 600;
const cc_rooftopworks_K_ACC: i32 = 604;
const cc_rooftopworks_K_PARAPET: i32 = 608;
// 609 was the penthouse (core massing since 2026-11: city_crown).
const cc_rooftopworks_K_PAD: i32 = 611;
const cc_rooftopworks_K_PIPE: i32 = 612;
const cc_rooftopworks_K_UNIT: i32 = 610;
const cc_rooftopworks_K_BULK: i32 = 613;
const cc_rooftopworks_K_RUN: i32 = 614;

// --- mechanical clusters ----------------------------------------------------
const cc_rooftopworks_MECH_FRAC: f32 = 0.86;   // buildings with any clutter
// Slot occupancy. Slot 0 is unconditional on a deck that fits it; the rest
// thin out, so the modal roof carries two or three objects and a big one
// four. Confetti is the failure mode being avoided here: a roof with a
// uniform sprinkle of boxes reads as noise, a roof with a tank, a condenser
// row and a gap reads as a roof.
const cc_rooftopworks_SLOT_P1: f32 = 0.88;
const cc_rooftopworks_SLOT_P2: f32 = 0.62;
const cc_rooftopworks_SLOT_P3: f32 = 0.34;
const cc_rooftopworks_TANK_CUT: f32 = 0.38;    // type draw: tank below this
const cc_rooftopworks_AC_CUT: f32 = 0.76;      // condenser row below this
const cc_rooftopworks_TANK_HW: f32 = 1.50;     // 3 x 3 m
const cc_rooftopworks_TANK_H: f32 = 4.00;
const cc_rooftopworks_TANK_RIM_HW: f32 = 1.78;
const cc_rooftopworks_TANK_RIM_T: f32 = 0.38;
const cc_rooftopworks_AC_HL: f32 = 3.60;       // three 2 m units, 2.6 pitch
const cc_rooftopworks_AC_HW: f32 = 1.00;
const cc_rooftopworks_AC_H: f32 = 1.00;
const cc_rooftopworks_AC_ROW: f32 = 3.10;      // second row offset
const cc_rooftopworks_AC_P2: f32 = 0.45;       // draw above this: doubled row
const cc_rooftopworks_VENT_HW: f32 = 0.50;
const cc_rooftopworks_VENT_H: f32 = 5.00;
const cc_rooftopworks_VENT2_H: f32 = 3.40;
const cc_rooftopworks_VENT_GAP: f32 = 1.70;    // the two stacks' centre spacing
const cc_rooftopworks_EDGE_M: f32 = 1.10;      // keep off the parapet line
// Close-range refinement (SPEC's SDF-in-a-box). A tank is the one object here
// the camera can end up standing next to, and a bare box is exactly what a
// bare box looks like. Inside its envelope, and only inside it, the tank is
// sphere-traced: a rounded tub on four legs with a chamfered rim band. The
// rounding is the point — rays graze past the corners and the silhouette
// stops being a rectangle. 42 m at the harness lens, so the wide scene pays
// one box test and nothing else.
const cc_rooftopworks_SDF_FP: f32 = 0.22;
const cc_rooftopworks_SDF_ITERS: i32 = 24;
const cc_rooftopworks_TANK_LEG: f32 = 0.55;    // stand-off from the deck
const cc_rooftopworks_TANK_R: f32 = 0.20;      // corner rounding
const cc_rooftopworks_LAMP_FRAC: f32 = 0.30;
const cc_rooftopworks_LAMP_RAD: f32 = 2.0;
const cc_rooftopworks_LAMP_SIG: f32 = 0.30;    // source radius (m)
const cc_rooftopworks_LAMP_SPILL: f32 = 1.70;  // what the housing catches (m)
const cc_rooftopworks_LAMP_COL: vec3<f32> = vec3<f32>(1.00, 0.58, 0.24);
const cc_rooftopworks_MECH_ALBEDO: f32 = 0.140;

// --- parapet ----------------------------------------------------------------
const cc_rooftopworks_PAR_FRAC: f32 = 0.55;    // roofs with a real wall
const cc_rooftopworks_PAR_LIT: f32 = 0.21;     // ...of which lit (~12% of roofs)
const cc_rooftopworks_PAR_H: f32 = 1.25;
const cc_rooftopworks_PAR_W: f32 = 0.38;
const cc_rooftopworks_PAR_TRIM: f32 = 0.28;    // lit band under the coping
const cc_rooftopworks_TRIM_RAD: f32 = 1.20;
const cc_rooftopworks_TRIM_COOL: vec3<f32> = vec3<f32>(0.72, 0.86, 1.00);
const cc_rooftopworks_TRIM_PITCH: f32 = 6.50;  // fixture centres (m)
const cc_rooftopworks_TRIM_DUTY: f32 = 0.82;   // lit fraction of the run

// --- helipad ----------------------------------------------------------------
// Which roofs carry a pad, and where, is the crown draw's (city_crown,
// CITY_CROWN_PAD_*); this is how it looks.
const cc_rooftopworks_PAD_LAMP_RAD: f32 = 3.00;
const cc_rooftopworks_PAD_LAMP_SIG: f32 = 0.26;
const cc_rooftopworks_PAD_LAMP_COL: vec3<f32> = vec3<f32>(1.00, 0.66, 0.30);
const cc_rooftopworks_PAD_MARK: f32 = 0.34;    // paint albedo (a ring + bar)

// --- pipework ---------------------------------------------------------------
const cc_rooftopworks_PIPE_FRAC: f32 = 0.45;
const cc_rooftopworks_PIPE_H_LO: f32 = 30.0;
const cc_rooftopworks_PIPE_H_HI: f32 = 150.0;
const cc_rooftopworks_PIPE_D: f32 = 0.40;      // one riser
const cc_rooftopworks_PIPE_GAP: f32 = 0.18;
const cc_rooftopworks_PIPE_OUT: f32 = 0.42;    // stand-off from the wall
const cc_rooftopworks_PIPE_CORNER: f32 = 1.20; // in from the building corner
// Footprint gate: a 0.4 m riser at 1.5 m/px is a quarter pixel of dark line
// on a lit wall and contributes nothing but cost. The emission and the
// contrast ramp to the wall's own value over the run-up, so it dissolves
// rather than pops.
const cc_rooftopworks_PIPE_FP: f32 = 1.20;
const cc_rooftopworks_PIPE_FADE: f32 = 0.75;

// --- bulkheads, runs, round tanks, skylights ---------------------------------
// Thomas, 2026-09: "more roof detail could be added. pipes etc." A roof is
// not a lid with three boxes on it. The stair comes up through a bulkhead
// with a lamp over its door; pipe racks and ducts run across the deck
// between the plant; walk-ups carry the round wooden tank on its steel legs
// (New York's, seen from the Empire State); a low roof often has a field of
// skylights, some lit from the room under them.
//
// The bulkhead: one per cell on the summit deck (the crown's penthouse is
// the bulkhead where there is one), against the deck's edge as stairs are.
const cc_rooftopworks_BULK_HX: f32 = 1.6;      // 3.2 x 4.4 m
const cc_rooftopworks_BULK_HY: f32 = 2.2;
const cc_rooftopworks_BULK_H: f32 = 3.0;
const cc_rooftopworks_BULK_LAMP_FRAC: f32 = 0.90;
const cc_rooftopworks_BULK_DOOR_W: f32 = 0.95;
const cc_rooftopworks_BULK_DOOR_H: f32 = 2.10;
const cc_rooftopworks_BULK_DOOR_K: f32 = 0.30;   // the leaf against the wall
const cc_rooftopworks_BULK_ALBEDO: f32 = 0.20;
// Taller buildings carry the lift's machine room beside the stair: a
// bigger box against the bulkhead.
const cc_rooftopworks_LIFT_MIN_H: f32 = 40.0;
const cc_rooftopworks_LIFT_HX: f32 = 2.6;
const cc_rooftopworks_LIFT_H: f32 = 4.2;
// Its vent panel: width, height band above its foot (m; a whole number of
// pitches, so the pattern's mean over it is exactly 1), slat pitch, the
// slats' share of the pitch and their factor on the wall and its light.
const cc_rooftopworks_LOUVRE_W: f32 = 1.8;
const cc_rooftopworks_LOUVRE_Z: vec2<f32> = vec2<f32>(1.5, 3.3);
const cc_rooftopworks_LOUVRE_PITCH: f32 = 0.15;
const cc_rooftopworks_LOUVRE_DARK: f32 = 0.45;
const cc_rooftopworks_LOUVRE_K: f32 = 0.30;
// Runs: up to four per cell, along x or y: a pipe rack on sleepers (three
// pipes) or a sheet-metal duct.
const cc_rooftopworks_RUN_N: i32 = 4;
const cc_rooftopworks_RUN_DUCT: f32 = 0.35;    // draw below this: a duct
// Packaged units: up to four rows per cell of one to four units each, the
// commonest thing on a low roof. A row is one box, its units drawn.
const cc_rooftopworks_UNIT_N: i32 = 4;
const cc_rooftopworks_UNIT_HXY: vec2<f32> = vec2<f32>(1.1, 0.7);
const cc_rooftopworks_UNIT_PITCH: f32 = 2.4;    // along the row
const cc_rooftopworks_UNIT_H: f32 = 1.3;
const cc_rooftopworks_RUN_LEN_LO: f32 = 6.0;
const cc_rooftopworks_RUN_LEN_HI: f32 = 24.0;
const cc_rooftopworks_RACK_HW: f32 = 0.40;
const cc_rooftopworks_RACK_Z0: f32 = 0.20;     // sleepers under the pipes
const cc_rooftopworks_RACK_Z1: f32 = 0.55;
const cc_rooftopworks_DUCT_HW: f32 = 0.45;
const cc_rooftopworks_DUCT_H: f32 = 0.90;
// Round tanks: on walk-ups always, and on most of the older low-rise.
const cc_rooftopworks_ROUND_H: f32 = 70.0;
const cc_rooftopworks_ROUND_FRAC: f32 = 0.60;
// The round tank stands a storey up on its steel stand: posts and tie
// beams under a stave barrel, the shallow cone of its roof overhanging the
// staves by an eave.
const cc_rooftopworks_ROUND_TANK_H: f32 = 5.20;
const cc_rooftopworks_ROUND_STAND: f32 = 1.80;  // stand height (barrel floor)
const cc_rooftopworks_CONE_H: f32 = 0.70;
const cc_rooftopworks_EAVE: f32 = 0.15;         // the cone's overhang
const cc_rooftopworks_POST_HW: f32 = 0.07;      // a stand post's half width
const cc_rooftopworks_TIE_H: f32 = 0.10;        // a tie beam's depth
const cc_rooftopworks_WOOD: vec3<f32> = vec3<f32>(1.25, 0.95, 0.70);
// Skylights: a field of them in the middle of a low roof, on a lattice.
const cc_rooftopworks_SKY_FRAC: f32 = 0.40;
const cc_rooftopworks_SKY_MAX_H: f32 = 120.0;
const cc_rooftopworks_SKY_PITCH: vec2<f32> = vec2<f32>(4.0, 6.0);
const cc_rooftopworks_SKY_PANE: vec2<f32> = vec2<f32>(1.4, 2.8);
const cc_rooftopworks_SKY_LIT: f32 = 0.30;
const cc_rooftopworks_SKY_RAD: f32 = 1.0;

// Headroom above the main roof that everything here fits under: the round
// tank on its stand is the tallest thing (5.2 m), a vent stack next (5 m).
// (12 m while the 10.8 m penthouse was a prop here.)
const cc_rooftopworks_HEADROOM: f32 = 6.0;

fn cc_rooftopworks_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// Nearest-hit accumulator over one box. Every piece of geometry in this file
// except the parapet ring goes through here, which is also where the segment
// discipline lives: a hit outside [t0, t1] belongs to another cell's visit
// and must not be returned, because the core's DDA stops the moment a cell
// reports anything.
fn cc_rooftopworks_box(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                       inv_dir: vec3<f32>, t0: f32, t1: f32,
                       bmin: vec3<f32>, bmax: vec3<f32>, kind: i32,
                       ci: vec2<i32>) -> CityHit {
    let s = city_box_hit(o, inv_dir, bmin, bmax);
    if (s.x <= s.y && s.x > max(t0, 1e-3) && s.x <= t1 && s.x < best.t) {
        let p = o + s.x * dir;
        return CityHit(true, s.x, p, city_box_normal(p, bmin, bmax), kind,
                       ci);
    }
    return best;
}

fn cc_rooftopworks_overlap(alo: vec2<f32>, ahi: vec2<f32>,
                           blo: vec2<f32>, bhi: vec2<f32>) -> bool {
    return alo.x < bhi.x && ahi.x > blo.x && alo.y < bhi.y && ahi.y > blo.y;
}

// The DDA cell's own column in world xy. Per-cell-hashed props are placed
// inside it so that the cell that draws a thing is the cell that contains it
// — which is what the DDA requires, and what gives a merged superblock four
// independently furnished quadrants instead of one roof drawn four times.
fn cc_rooftopworks_column(ci: vec2<i32>) -> vec4<f32> {
    let cell = u.tile_cell;
    let cmin = vec2<f32>(ci) * cell;
    return vec4<f32>(cmin, cmin + vec2<f32>(cell, cell));
}

// --- decks ------------------------------------------------------------------
// A flat top you may stand something on: its rect, its height, and the hole
// punched in it by whatever tier stands on it.
struct cc_rooftopworks_Deck {
    ok: bool,
    lo: vec2<f32>,
    hi: vec2<f32>,
    z: f32,
    hlo: vec2<f32>,
    hhi: vec2<f32>,
}

fn cc_rooftopworks_no_deck() -> cc_rooftopworks_Deck {
    return cc_rooftopworks_Deck(false, vec2<f32>(0.0), vec2<f32>(0.0), 0.0,
                                vec2<f32>(1e30), vec2<f32>(-1e30));
}

fn cc_rooftopworks_mk_deck(lo: vec2<f32>, hi: vec2<f32>, z: f32,
                           hlo: vec2<f32>, hhi: vec2<f32>)
        -> cc_rooftopworks_Deck {
    return cc_rooftopworks_Deck(true, lo, hi, z, hlo, hhi);
}

// Deck `i` of this building, index 0 being the topmost. This function IS the
// architecture rule; everything else in the file only asks it for a rect.
fn cc_rooftopworks_deck(cc: CityCell, i: i32) -> cc_rooftopworks_Deck {
    let nohole_lo = vec2<f32>(1e30);
    let nohole_hi = vec2<f32>(-1e30);
    if (cc.arch == 4) {
        return cc_rooftopworks_no_deck();   // a spire has no roof. Ever.
    }
    if (cc.arch == 3) {
        // The crown cap of the tapered shaft: the frustum's cross-section at
        // its top, which is the base rect scaled by fscale about its centre.
        if (i != 0) {
            return cc_rooftopworks_no_deck();
        }
        let c = 0.5 * (cc.fmin.xy + cc.fmax.xy);
        let hf = 0.5 * (cc.fmax.xy - cc.fmin.xy) * cc.fscale;
        return cc_rooftopworks_mk_deck(c - hf, c + hf, cc.fmax.z,
                                       nohole_lo, nohole_hi);
    }
    if (cc.arch == 2) {
        // Growth: the main roof, plus each cantilevered bud's top. The buds
        // overlap the main mass by a metre at their root and punch no hole
        // in it (they hang below the roof), so no holes here either.
        if (i == 0) {
            return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy,
                                           cc.b1max.z, nohole_lo, nohole_hi);
        }
        if (i == 1 && cc.tiers >= 2) {
            return cc_rooftopworks_mk_deck(cc.b2min.xy, cc.b2max.xy,
                                           cc.b2max.z, nohole_lo, nohole_hi);
        }
        if (i == 2 && cc.tiers >= 3) {
            return cc_rooftopworks_mk_deck(cc.b3min.xy, cc.b3max.xy,
                                           cc.b3max.z, nohole_lo, nohole_hi);
        }
        return cc_rooftopworks_no_deck();
    }
    // Setback stack. Deck 0 is the summit; the ones below it are annular,
    // and carry the tier above as their hole.
    if (cc.tiers == 1) {
        if (i == 0) {
            return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy,
                                           cc.b1max.z, nohole_lo, nohole_hi);
        }
        return cc_rooftopworks_no_deck();
    }
    if (cc.tiers == 2) {
        if (i == 0) {
            return cc_rooftopworks_mk_deck(cc.b2min.xy, cc.b2max.xy,
                                           cc.b2max.z, nohole_lo, nohole_hi);
        }
        if (i == 1) {
            return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy,
                                           cc.b1max.z, cc.b2min.xy,
                                           cc.b2max.xy);
        }
        return cc_rooftopworks_no_deck();
    }
    if (i == 0) {
        return cc_rooftopworks_mk_deck(cc.b3min.xy, cc.b3max.xy, cc.b3max.z,
                                       nohole_lo, nohole_hi);
    }
    if (i == 1) {
        return cc_rooftopworks_mk_deck(cc.b2min.xy, cc.b2max.xy, cc.b2max.z,
                                       cc.b3min.xy, cc.b3max.xy);
    }
    if (i == 2) {
        return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy, cc.b1max.z,
                                       cc.b2min.xy, cc.b2max.xy);
    }
    return cc_rooftopworks_no_deck();
}

// The lowest deck this building owns: the bottom of the z band the whole
// roof section gates on. For a setback stack that is the first tier's top,
// for a growth tower the lower bud, otherwise the roof itself.
fn cc_rooftopworks_deck_low(cc: CityCell) -> f32 {
    var z = cc.height;
    if (cc.arch != 3 && cc.tiers >= 2) {
        z = min(z, min(cc.b1max.z, cc.b2max.z));
        if (cc.tiers >= 3) {
            z = min(z, cc.b3max.z);
        }
    }
    return z;
}

// --- mechanical clusters ----------------------------------------------------
struct cc_rooftopworks_Mech {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    has_acc: bool,
    alo: vec3<f32>,
    ahi: vec3<f32>,
    kind: i32,         // 0 tank, 1 condenser row, 2 vent stacks
    lamp: bool,
    seat: vec3<f32>,   // the service lamp's own position
    round: bool,       // a tank: the round wooden one
}

fn cc_rooftopworks_no_mech() -> cc_rooftopworks_Mech {
    return cc_rooftopworks_Mech(false, vec3<f32>(0.0), vec3<f32>(0.0), false,
                                vec3<f32>(0.0), vec3<f32>(0.0), 0, false,
                                vec3<f32>(0.0), false);
}

// Does this building carry rooftop clutter at all? One building-level draw,
// hoisted out of the slot loop so a cell pays it once rather than four times.
fn cc_rooftopworks_mech_on(cc: CityCell) -> bool {
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x3c6ef372u, 0x165667b1u));
    return gh.w < cc_rooftopworks_MECH_FRAC;
}

// The clutter's density at this tier (u.roof_density, the hosts'
// CITY_ROOF_DENSITY_BY_TIER; 1 as the roofs were tuned): a factor on the
// clutter families' slot probabilities, each held to 1. The stair, its lift
// room, the crown and the skylights are not clutter and do not scale.
fn cc_rooftopworks_density() -> f32 {
    return u.roof_density;
}

fn cc_rooftopworks_slot_draw(ci: vec2<i32>, s: i32) -> vec4<f32> {
    let cw = city_tile_cell(ci);
    let l = u32(s);
    return city_rand4(vec2<u32>(
        bitcast<u32>(cw.x) * 0x85ebca6bu + l * 0x9e3779b9u + 0x7feb352du,
        bitcast<u32>(cw.y) * 0xc2b2ae35u + l * 0x51ed270bu + 0x846ca68bu));
}

// Slot `s` of cell `ci`: which deck it stands on, where, and what it is.
// Deterministic in (cc, ci, s), so the shader re-derives the object from the
// hit kind alone — nothing is smuggled through CityHit.
fn cc_rooftopworks_mech(cc: CityCell, ci: vec2<i32>, s: i32,
                        mech_on: bool, cr: CityCrown,
                        bk: cc_rooftopworks_Bulk) -> cc_rooftopworks_Mech {
    if (!mech_on) {
        return cc_rooftopworks_no_mech();
    }
    let r = cc_rooftopworks_slot_draw(ci, s);
    var p_slot = 1.0;
    if (s == 1) { p_slot = cc_rooftopworks_SLOT_P1; }
    if (s == 2) { p_slot = cc_rooftopworks_SLOT_P2; }
    if (s == 3) { p_slot = cc_rooftopworks_SLOT_P3; }
    if (r.x >= min(p_slot * cc_rooftopworks_density(), 1.0)) {
        return cc_rooftopworks_no_mech();
    }
    // Slots 0 and 1 take the summit; 2 and 3 look for a lower tier top or a
    // bud, and fall back to the summit when the building has none.
    var di = 0;
    if (s == 2) { di = 1; }
    if (s == 3) { di = 2; }
    var d = cc_rooftopworks_deck(cc, di);
    if (!d.ok) {
        d = cc_rooftopworks_deck(cc, 0);
    }
    if (!d.ok) {
        return cc_rooftopworks_no_mech();
    }

    // Type and footprint.
    var hx = vec2<f32>(cc_rooftopworks_TANK_HW);
    let round = cc_rooftopworks_round_tanks(cc);
    var hgt = select(cc_rooftopworks_TANK_H, cc_rooftopworks_ROUND_TANK_H,
                     round);
    var ty = 0;
    let axis_y = r.w > 0.5;
    if (r.y >= cc_rooftopworks_TANK_CUT && r.y < cc_rooftopworks_AC_CUT) {
        ty = 1;
        let along = cc_rooftopworks_AC_HL;
        let across = select(cc_rooftopworks_AC_HW,
                            cc_rooftopworks_AC_HW + 0.5 * cc_rooftopworks_AC_ROW,
                            r.z > cc_rooftopworks_AC_P2);
        hx = select(vec2<f32>(along, across), vec2<f32>(across, along),
                    axis_y);
        hgt = cc_rooftopworks_AC_H;
    } else if (r.y >= cc_rooftopworks_AC_CUT) {
        ty = 2;
        let along = cc_rooftopworks_VENT_HW + 0.5 * cc_rooftopworks_VENT_GAP;
        hx = select(vec2<f32>(along, cc_rooftopworks_VENT_HW),
                    vec2<f32>(cc_rooftopworks_VENT_HW, along), axis_y);
        hgt = cc_rooftopworks_VENT_H;
    }

    // Where it may stand: the deck, inset off the parapet line, clipped to
    // this cell's own column.
    let col = cc_rooftopworks_column(ci);
    let m = hx + cc_rooftopworks_EDGE_M + cc_rooftopworks_PAR_W;
    let plo = max(d.lo + m, col.xy + m - cc_rooftopworks_EDGE_M);
    let phi = min(d.hi - m, col.zw - m + cc_rooftopworks_EDGE_M);
    if (plo.x > phi.x || plo.y > phi.y) {
        return cc_rooftopworks_no_mech();   // deck too small for this object
    }
    let c = mix(plo, phi, vec2<f32>(fract(r.z * 13.7), fract(r.w * 7.31)));
    let lo2 = c - hx;
    let hi2 = c + hx;
    // Three exclusions, all cheap rect tests: the tier standing on this deck,
    // the mast, and whatever the crown put here.
    if (cc_rooftopworks_overlap(lo2, hi2, d.hlo - 1.0, d.hhi + 1.0)) {
        return cc_rooftopworks_no_mech();
    }
    if (cc.has_mast && abs(d.z - cc.height) < 0.5
        && cc_rooftopworks_overlap(lo2, hi2, cc.mast_min.xy - 0.8,
                                   cc.mast_max.xy + 0.8)) {
        return cc_rooftopworks_no_mech();
    }
    if (cr.kind != 0 && abs(d.z - cc.height) < 0.5
        && cc_rooftopworks_overlap(lo2, hi2, cr.lo - 1.5, cr.hi + 1.5)) {
        return cc_rooftopworks_no_mech();
    }
    if (bk.ok && abs(d.z - bk.lo.z) < 0.5
        && (cc_rooftopworks_overlap(lo2, hi2, bk.lo.xy - 1.2, bk.hi.xy + 1.2)
            || (bk.lift
                && cc_rooftopworks_overlap(lo2, hi2, bk.lift_lo.xy - 1.2,
                                           bk.lift_hi.xy + 1.2)))) {
        return cc_rooftopworks_no_mech();
    }
    if (!cc_rooftopworks_sky_clear(bk, d.z, lo2, hi2)) {
        return cc_rooftopworks_no_mech();
    }

    var res: cc_rooftopworks_Mech;
    res.ok = true;
    res.lo = vec3<f32>(lo2, d.z);
    res.hi = vec3<f32>(hi2, d.z + hgt);
    res.kind = ty;
    res.has_acc = false;
    res.alo = vec3<f32>(0.0);
    res.ahi = vec3<f32>(0.0);
    if (ty == 0) {
        // The rim band: a tank without one is a crate.
        res.has_acc = true;
        let rw = vec2<f32>(cc_rooftopworks_TANK_RIM_HW);
        res.alo = vec3<f32>(c - rw,
                            d.z + hgt - cc_rooftopworks_TANK_RIM_T);
        res.ahi = vec3<f32>(c + rw, d.z + hgt + 0.10);
    } else if (ty == 2) {
        // A second, shorter stack beside the first, the pair filling the
        // footprint reserved for it (its half-length VENT_HW + VENT_GAP / 2).
        res.has_acc = true;
        let off = select(vec2<f32>(0.5 * cc_rooftopworks_VENT_GAP, 0.0),
                         vec2<f32>(0.0, 0.5 * cc_rooftopworks_VENT_GAP),
                         axis_y);
        res.alo = vec3<f32>(c + off - cc_rooftopworks_VENT_HW, d.z);
        res.ahi = vec3<f32>(c + off + cc_rooftopworks_VENT_HW,
                            d.z + cc_rooftopworks_VENT2_H);
        res.lo = vec3<f32>(c - off - cc_rooftopworks_VENT_HW, d.z);
        res.hi = vec3<f32>(c - off + cc_rooftopworks_VENT_HW, d.z + hgt);
    }
    // One warm bulkhead light on the housing, on the face the draw picks.
    res.lamp = fract(r.x * 31.7) < cc_rooftopworks_LAMP_FRAC;
    let face = fract(r.y * 17.3);
    var n = vec2<f32>(1.0, 0.0);
    if (face > 0.75) { n = vec2<f32>(-1.0, 0.0); }
    else if (face > 0.50) { n = vec2<f32>(0.0, 1.0); }
    else if (face > 0.25) { n = vec2<f32>(0.0, -1.0); }
    res.seat = vec3<f32>(c + n * (hx + 0.04), d.z + 0.72 * hgt);
    if (ty == 2) {
        // On the tall stack's own face, whichever: never in the gap between
        // the stacks or above the short one.
        let tall = 0.5 * (res.lo.xy + res.hi.xy);
        res.seat = vec3<f32>(tall + n * (cc_rooftopworks_VENT_HW + 0.04),
                             res.seat.z);
    }
    res.round = ty == 0 && round;
    if (res.round) {
        // A round tank has hoops, not a rim band: the cap is its own cone,
        // and its envelope is its own box.
        res.has_acc = false;
        res.alo = res.lo;
        res.ahi = res.hi;
    }
    return res;
}

// Whether this building's tanks are the round wooden kind.
fn cc_rooftopworks_round_tanks(cc: CityCell) -> bool {
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x2f8b3c1du, 0x6d4a9e27u));
    return cc.win_style == 4
        || (cc.height < cc_rooftopworks_ROUND_H
            && gh.x < cc_rooftopworks_ROUND_FRAC);
}

// --- the bulkhead -------------------------------------------------------------
struct cc_rooftopworks_Bulk {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    door: vec2<f32>,   // the door face's outward normal
    lamp: bool,
    seat: vec3<f32>,   // the lamp over the door
    lift: bool,        // the lift's machine room beside it
    lift_lo: vec3<f32>,
    lift_hi: vec3<f32>,
    // The summit's skylight field the clutter keeps off (cc_rooftopworks_sky
    // and _sky_clear): carried here because every prop that needs it is
    // handed the stair, so the field is drawn once per cell, not per slot.
    sky_ok: bool,
    sky_lo: vec2<f32>,
    sky_hi: vec2<f32>,
    sky_z: f32,
}

fn cc_rooftopworks_no_bulk() -> cc_rooftopworks_Bulk {
    return cc_rooftopworks_Bulk(false, vec3<f32>(0.0), vec3<f32>(0.0),
                                vec2<f32>(0.0), false, vec3<f32>(0.0), false,
                                vec3<f32>(0.0), vec3<f32>(0.0), false,
                                vec2<f32>(0.0), vec2<f32>(0.0), 0.0);
}

// The stair (and its lift room), and with it the skylight field it leaves.
fn cc_rooftopworks_bulk(cc: CityCell, ci: vec2<i32>,
                        cr: CityCrown) -> cc_rooftopworks_Bulk {
    var b = cc_rooftopworks_bulk_place(cc, ci, cr);
    let d0 = cc_rooftopworks_deck(cc, 0);
    let sk = cc_rooftopworks_sky(cc, d0, cr, b);
    b.sky_ok = sk.ok;
    b.sky_lo = sk.flo;
    b.sky_hi = sk.fhi;
    b.sky_z = d0.z;
    return b;
}

fn cc_rooftopworks_bulk_place(cc: CityCell, ci: vec2<i32>,
                              cr: CityCrown) -> cc_rooftopworks_Bulk {
    if (cr.kind == 1) {
        return cc_rooftopworks_no_bulk();   // the penthouse is the bulkhead
    }
    let d = cc_rooftopworks_deck(cc, 0);
    if (!d.ok) {
        return cc_rooftopworks_no_bulk();
    }
    let r = cc_rooftopworks_slot_draw(ci, 7);
    let along_y = r.w > 0.5;
    let hx = select(vec2<f32>(cc_rooftopworks_BULK_HY, cc_rooftopworks_BULK_HX),
                    vec2<f32>(cc_rooftopworks_BULK_HX, cc_rooftopworks_BULK_HY),
                    along_y);
    let col = cc_rooftopworks_column(ci);
    let m = hx + cc_rooftopworks_EDGE_M + cc_rooftopworks_PAR_W;
    let plo = max(d.lo + m, col.xy + m - cc_rooftopworks_EDGE_M);
    let phi = min(d.hi - m, col.zw - m + cc_rooftopworks_EDGE_M);
    if (plo.x > phi.x || plo.y > phi.y) {
        return cc_rooftopworks_no_bulk();
    }
    // Against one edge of the free deck, anywhere along it.
    let side = r.x > 0.5;
    let t = fract(r.y * 7.31);
    var c = select(vec2<f32>(mix(plo.x, phi.x, t),
                             select(plo.y, phi.y, side)),
                   vec2<f32>(select(plo.x, phi.x, side),
                             mix(plo.y, phi.y, t)), along_y);
    let lo2 = c - hx;
    let hi2 = c + hx;
    if (cc_rooftopworks_overlap(lo2, hi2, d.hlo - 1.0, d.hhi + 1.0)
        || (cc.has_mast
            && cc_rooftopworks_overlap(lo2, hi2, cc.mast_min.xy - 0.8,
                                       cc.mast_max.xy + 0.8))
        || (cr.kind != 0
            && cc_rooftopworks_overlap(lo2, hi2, cr.lo - 1.5, cr.hi + 1.5))) {
        return cc_rooftopworks_no_bulk();
    }
    var b: cc_rooftopworks_Bulk;
    b.ok = true;
    b.lo = vec3<f32>(lo2, d.z);
    b.hi = vec3<f32>(hi2, d.z + cc_rooftopworks_BULK_H);
    // The door looks into the deck, on the bulkhead's long side.
    // Toward the middle of the free deck in this cell's own column: the
    // building's middle on a lone roof, but on a merged one the stair must
    // not face a column boundary (its pool would land in the next cell,
    // whose hook does not draw it).
    let into = 0.5 * (plo + phi) - c;
    b.door = select(vec2<f32>(0.0, sign(into.y + 1e-6)),
                    vec2<f32>(sign(into.x + 1e-6), 0.0), along_y);
    b.lamp = fract(r.z * 13.1) < cc_rooftopworks_BULK_LAMP_FRAC;
    b.seat = vec3<f32>(c + b.door * (dot(abs(b.door), hx) + 0.10),
                       d.z + cc_rooftopworks_BULK_DOOR_H + 0.30);
    // The machine room: beside the stair along its long axis, on the side
    // with deck to spare, flush with the stair's back (the deck edge it
    // stands against) and so reaching further into the deck than the stair
    // does, since it is the wider of the two; wholly inside the free deck,
    // clear of what the stair keeps clear of, or not at all.
    b.lift = false;
    b.lift_lo = b.lo;
    b.lift_hi = b.hi;
    if (cc.height > cc_rooftopworks_LIFT_MIN_H) {
        let ax = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), along_y);
        let toward = select(-1.0, 1.0, dot(into, ax) > 0.0);
        let lh = vec2<f32>(cc_rooftopworks_LIFT_HX);
        let lc = c + ax * toward * (dot(ax, hx) + cc_rooftopworks_LIFT_HX)
                 + b.door * (cc_rooftopworks_LIFT_HX - dot(abs(b.door), hx));
        let llo = lc - lh;
        let lhi = lc + lh;
        // Flush with the stair's back, the room meets the free deck's edge
        // exactly in real arithmetic: a millimetre's slack for the float.
        if (all(llo >= plo - hx - 1e-3) && all(lhi <= phi + hx + 1e-3)
            && !cc_rooftopworks_overlap(llo, lhi, d.hlo - 1.0, d.hhi + 1.0)
            && !(cc.has_mast
                 && cc_rooftopworks_overlap(llo, lhi, cc.mast_min.xy - 0.8,
                                            cc.mast_max.xy + 0.8))
            && !(cr.kind != 0
                 && cc_rooftopworks_overlap(llo, lhi, cr.lo - 1.5,
                                            cr.hi + 1.5))) {
            b.lift = true;
            b.lift_lo = vec3<f32>(lc - lh, d.z);
            b.lift_hi = vec3<f32>(lc + lh, d.z + cc_rooftopworks_LIFT_H);
        }
    }
    return b;
}

// --- packaged units -------------------------------------------------------------
fn cc_rooftopworks_unit(cc: CityCell, ci: vec2<i32>, k: i32,
                        cr: CityCrown, bk: cc_rooftopworks_Bulk)
        -> cc_rooftopworks_Run {
    var res = cc_rooftopworks_Run(false, vec3<f32>(0.0), vec3<f32>(0.0),
                                  false, false);
    let r = cc_rooftopworks_slot_draw(ci, 12 + k);
    if (r.x >= min((0.90 - 0.18 * f32(k)) * cc_rooftopworks_density(),
                   1.0)) {
        return res;
    }
    let d = cc_rooftopworks_deck(cc, 0);
    if (!d.ok) {
        return res;
    }
    let along_y = r.w > 0.5;
    let n_units = 1.0 + floor(fract(r.x * 23.1) * 3.999);
    let row = vec2<f32>(cc_rooftopworks_UNIT_HXY.x
                        + 0.5 * (n_units - 1.0) * cc_rooftopworks_UNIT_PITCH,
                        cc_rooftopworks_UNIT_HXY.y);
    let hx = select(row, row.yx, along_y);
    let col = cc_rooftopworks_column(ci);
    let m = hx + cc_rooftopworks_EDGE_M + cc_rooftopworks_PAR_W;
    let plo = max(d.lo + m, col.xy + m - cc_rooftopworks_EDGE_M);
    let phi = min(d.hi - m, col.zw - m + cc_rooftopworks_EDGE_M);
    if (plo.x > phi.x || plo.y > phi.y) {
        return res;
    }
    let c = mix(plo, phi, vec2<f32>(fract(r.y * 9.7), fract(r.z * 6.13)));
    let lo2 = c - hx;
    let hi2 = c + hx;
    if (cc_rooftopworks_overlap(lo2, hi2, d.hlo - 1.0, d.hhi + 1.0)
        || (cc.has_mast
            && cc_rooftopworks_overlap(lo2, hi2, cc.mast_min.xy - 0.8,
                                       cc.mast_max.xy + 0.8))
        || (cr.kind != 0
            && cc_rooftopworks_overlap(lo2, hi2, cr.lo - 1.0, cr.hi + 1.0))
        || (bk.ok
            && (cc_rooftopworks_overlap(lo2, hi2, bk.lo.xy - 0.5,
                                        bk.hi.xy + 0.5)
                || cc_rooftopworks_overlap(lo2, hi2, bk.lift_lo.xy - 0.5,
                                           bk.lift_hi.xy + 0.5)))) {
        return res;
    }
    if (!cc_rooftopworks_sky_clear(bk, d.z, lo2, hi2)) {
        return res;
    }
    res.ok = true;
    res.along_y = along_y;
    res.lo = vec3<f32>(lo2, d.z);
    res.hi = vec3<f32>(hi2, d.z + cc_rooftopworks_UNIT_H);
    return res;
}

// --- runs ---------------------------------------------------------------------
struct cc_rooftopworks_Run {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    duct: bool,
    along_y: bool,
}

fn cc_rooftopworks_run(cc: CityCell, ci: vec2<i32>, k: i32,
                       cr: CityCrown, bk: cc_rooftopworks_Bulk)
        -> cc_rooftopworks_Run {
    var res = cc_rooftopworks_Run(false, vec3<f32>(0.0), vec3<f32>(0.0),
                                  false, false);
    let r = cc_rooftopworks_slot_draw(ci, 8 + k);
    if (r.x >= min((0.85 - 0.18 * f32(k)) * cc_rooftopworks_density(),
                   1.0)) {
        return res;
    }
    let d = cc_rooftopworks_deck(cc, 0);
    if (!d.ok) {
        return res;
    }
    let duct = fract(r.x * 17.3) < cc_rooftopworks_RUN_DUCT;
    let along_y = r.w > 0.5;
    let half_len = 0.5 * mix(cc_rooftopworks_RUN_LEN_LO,
                             cc_rooftopworks_RUN_LEN_HI, r.y);
    let across = select(cc_rooftopworks_RACK_HW, cc_rooftopworks_DUCT_HW,
                        duct);
    let hx = select(vec2<f32>(half_len, across), vec2<f32>(across, half_len),
                    along_y);
    let col = cc_rooftopworks_column(ci);
    let m = hx + cc_rooftopworks_EDGE_M + cc_rooftopworks_PAR_W;
    let plo = max(d.lo + m, col.xy + m - cc_rooftopworks_EDGE_M);
    let phi = min(d.hi - m, col.zw - m + cc_rooftopworks_EDGE_M);
    if (plo.x > phi.x || plo.y > phi.y) {
        return res;
    }
    let c = mix(plo, phi, vec2<f32>(fract(r.z * 11.3), fract(r.w * 5.17)));
    let lo2 = c - hx;
    let hi2 = c + hx;
    if (cc_rooftopworks_overlap(lo2, hi2, d.hlo - 1.0, d.hhi + 1.0)
        || (cc.has_mast
            && cc_rooftopworks_overlap(lo2, hi2, cc.mast_min.xy - 0.8,
                                       cc.mast_max.xy + 0.8))
        || (cr.kind != 0
            && cc_rooftopworks_overlap(lo2, hi2, cr.lo - 1.0, cr.hi + 1.0))
        || (bk.ok
            && (cc_rooftopworks_overlap(lo2, hi2, bk.lo.xy - 0.3,
                                        bk.hi.xy + 0.3)
                || cc_rooftopworks_overlap(lo2, hi2, bk.lift_lo.xy - 0.3,
                                           bk.lift_hi.xy + 0.3)))) {
        return res;
    }
    if (!cc_rooftopworks_sky_clear(bk, d.z, lo2, hi2)) {
        return res;
    }
    res.ok = true;
    res.duct = duct;
    res.along_y = along_y;
    res.lo = vec3<f32>(lo2, d.z + select(cc_rooftopworks_RACK_Z0, 0.0, duct));
    res.hi = vec3<f32>(hi2, d.z + select(cc_rooftopworks_RACK_Z1,
                                         cc_rooftopworks_DUCT_H, duct));
    return res;
}

// --- skylights (the roof hook) --------------------------------------------------
// A field of them over the middle half of a low roof, flush enough to be
// drawn on the roof rather than built: lit panes glow with the room's light
// over the roof's own radiance.
//
// The field is shown the way the flat facade shows its windows (see THE
// FLAT FACADE'S WINDOWS in the core): the pixel's footprint on the deck is
// averaged as it would be shown, pane by pane while it holds few lattice
// cells (cc_rooftopworks_sky_exact), and in expectation past that
// (cc_rooftopworks_sky_mean), blending across the last cell of the exact
// sum's room; city_windows_resolve hands the average back as a radiance.
// A footprint wholly on one pane, or wholly off the lit ones, comes back as
// that surface exactly, so up close the panes are drawn sharp.
struct cc_rooftopworks_Sky {
    ok: bool,
    flo: vec2<f32>,
    fhi: vec2<f32>,
    pitch: vec2<f32>,
    pane: vec2<f32>,
}

// The field is the summit deck's middle half, on a roof with no crown and
// whose stair (and lift room) stand clear of it; the clutter keeps off it in
// turn (cc_rooftopworks_sky_clear). So nothing stands on a pane: a pane
// hidden by a prop up close would otherwise show in the far field's mean.
fn cc_rooftopworks_sky(cc: CityCell, d: cc_rooftopworks_Deck,
                       cr: CityCrown, bk: cc_rooftopworks_Bulk)
        -> cc_rooftopworks_Sky {
    var sk: cc_rooftopworks_Sky;
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x58f2a6c3u, 0x13d7b9e1u));
    let size = d.hi - d.lo;
    sk.flo = d.lo + 0.25 * size;
    sk.fhi = d.hi - 0.25 * size;
    sk.ok = d.ok && gh.x < cc_rooftopworks_SKY_FRAC
            && cc.height <= cc_rooftopworks_SKY_MAX_H && cr.kind == 0
            && !(bk.ok && cc_rooftopworks_overlap(bk.lo.xy, bk.hi.xy, sk.flo,
                                                   sk.fhi))
            && !(bk.ok && bk.lift
                 && cc_rooftopworks_overlap(bk.lift_lo.xy, bk.lift_hi.xy,
                                            sk.flo, sk.fhi));
    sk.pitch = select(cc_rooftopworks_SKY_PITCH,
                      cc_rooftopworks_SKY_PITCH.yx, gh.y > 0.5);
    sk.pane = select(cc_rooftopworks_SKY_PANE, cc_rooftopworks_SKY_PANE.yx,
                     gh.y > 0.5);
    return sk;
}

// Whether a prop's footprint [lo, hi] on the deck at height z keeps clear of
// the summit's skylight field (true where there is none).
fn cc_rooftopworks_sky_clear(bk: cc_rooftopworks_Bulk, z: f32,
                             lo: vec2<f32>, hi: vec2<f32>) -> bool {
    return !bk.sky_ok || abs(z - bk.sky_z) > 0.5
        || !cc_rooftopworks_overlap(lo, hi, bk.sky_lo - 0.3, bk.sky_hi + 0.3);
}

// A lattice cell's pane: its brightness over the room's light, 0 if dark.
fn cc_rooftopworks_sky_lit(cc: CityCell, q: vec2<f32>) -> f32 {
    let wh = city_rand4(vec2<u32>(
        cc.seed.x ^ (bitcast<u32>(i32(q.x)) * 0x9e3779b9u) ^ 0x41c64e6du,
        cc.seed.y ^ (bitcast<u32>(i32(q.y)) * 0x85ebca6bu)));
    return select(0.0, 0.6 + 0.8 * wh.z, wh.x < cc_rooftopworks_SKY_LIT);
}

fn cc_rooftopworks_area(lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let s = max(hi - lo, vec2<f32>(0.0));
    return s.x * s.y;
}

// The lattice cells the exact sum walks at most, on each axis: whole while
// the footprint spans under one pitch, handed to the expectation between
// half a pitch and one.
const cc_rooftopworks_SKY_EXACT_MAX: i32 = 3;

// The footprint box [blo, bhi] summed pane by pane: the lit panes, each
// clipped to the box and the field, at its own brightness; the rest of the
// box is roof. `pane_rad` is a pane's radiance over the roof at brightness
// 1. Also returns, in `whole`, the lit brightness when one lit pane covers
// the box (else -1).
struct cc_rooftopworks_SkySum {
    sum: CityWinSum,
    lit_area: f32,
    whole: f32,
}

fn cc_rooftopworks_sky_exact(cc: CityCell, sk: cc_rooftopworks_Sky,
                             blo: vec2<f32>, bhi: vec2<f32>,
                             base: vec3<f32>, pane_rad: vec3<f32>,
                             fog: vec4<f32>, shown: bool)
        -> cc_rooftopworks_SkySum {
    var res = cc_rooftopworks_SkySum(city_win_zero(), 0.0, -1.0);
    let box_a = cc_rooftopworks_area(blo, bhi);
    let c0 = floor((blo - sk.flo) / sk.pitch);
    let c1 = floor((bhi - sk.flo) / sk.pitch);
    let inset = 0.5 * (sk.pitch - sk.pane);
    for (var j: i32 = 0; j < cc_rooftopworks_SKY_EXACT_MAX; j = j + 1) {
        for (var i: i32 = 0; i < cc_rooftopworks_SKY_EXACT_MAX; i = i + 1) {
            let q = c0 + vec2<f32>(f32(i), f32(j));
            if (q.x > c1.x || q.y > c1.y) {
                continue;
            }
            let plo = sk.flo + q * sk.pitch + inset;
            let olo = max(max(plo, blo), sk.flo);
            let ohi = min(min(plo + sk.pane, bhi), sk.fhi);
            let a = cc_rooftopworks_area(olo, ohi);
            if (a <= 0.0) {
                continue;
            }
            let lit = cc_rooftopworks_sky_lit(cc, q);
            if (lit > 0.0) {
                res.sum = city_win_add(res.sum, base + pane_rad * lit, a,
                                       fog, shown);
                res.lit_area = res.lit_area + a;
                // Clipping returns one of its operands, so a box wholly on
                // this pane clips to itself exactly.
                if (all(olo == blo) && all(ohi == bhi)) {
                    res.whole = lit;
                }
            }
        }
    }
    res.sum = city_win_add(res.sum, base, box_a - res.lit_area, fog, shown);
    return res;
}

// How much of [0, x] (each axis, from the field's corner) the lattice's
// panes cover: whole pitches, then the part of the last one past its inset.
fn cc_rooftopworks_sky_cover(x: vec2<f32>, pitch: vec2<f32>,
                             inset: vec2<f32>, pane: vec2<f32>) -> vec2<f32> {
    let q = floor(x / pitch);
    return q * pane + clamp(x - q * pitch - inset, vec2<f32>(0.0), pane);
}

// The same box in expectation: of the field it overlaps, the panes' area lit
// at the rate, the brightness uniform on [0.6, 1.4] taken by three-point
// Gauss-Legendre (the display curve is smooth over it); the rest is roof.
fn cc_rooftopworks_sky_mean(sk: cc_rooftopworks_Sky, blo: vec2<f32>,
                            bhi: vec2<f32>, base: vec3<f32>,
                            pane_rad: vec3<f32>, fog: vec4<f32>,
                            shown: bool) -> cc_rooftopworks_SkySum {
    let box_a = cc_rooftopworks_area(blo, bhi);
    // The panes' area in the box, exactly: the lattice is separable, so it
    // is the product of each axis' pane coverage over the box's overlap
    // with the field, counted from the field's corner where the lattice
    // starts.
    let lo = max(blo, sk.flo) - sk.flo;
    let hi = max(min(bhi, sk.fhi) - sk.flo, lo);
    let inset = 0.5 * (sk.pitch - sk.pane);
    let cover = cc_rooftopworks_sky_cover(hi, sk.pitch, inset, sk.pane)
                - cc_rooftopworks_sky_cover(lo, sk.pitch, inset, sk.pane);
    let a_lit = cc_rooftopworks_SKY_LIT * cover.x * cover.y;
    let g = 0.8 * 0.3872983346;   // 0.8 * sqrt(3/5) / 2, on [0.6, 1.4]
    var acc = city_win_add(city_win_zero(), base + pane_rad, a_lit * (4.0 / 9.0),
                           fog, shown);
    acc = city_win_add(acc, base + pane_rad * (1.0 - g), a_lit * (5.0 / 18.0),
                       fog, shown);
    acc = city_win_add(acc, base + pane_rad * (1.0 + g), a_lit * (5.0 / 18.0),
                       fog, shown);
    acc = city_win_add(acc, base, box_a - a_lit, fog, shown);
    return cc_rooftopworks_SkySum(acc, a_lit, -1.0);
}

// --- the lamps, past CITY_PROP_RANGE (the roof hook) ---------------------------
// Every roof lamp is drawn by props: the tank's service lamp, the
// bulkhead's door lamp, the helipad's corner lights. Past CITY_PROP_RANGE
// none is drawn, and the hook adds their light instead, as a uniform
// emission on the summit deck of the cell that holds them: what the
// lamps' spots, as the prop shader draws them, send toward the eye, spread
// over the deck as it is seen. Exact in expectation:
// - a lamp's spots (the source and its spill) integrate, over the faces of
//   the housing they are drawn on, to a seen area per unit radiance: a
//   face's share is seen in proportion to max(0, n.v), v toward the eye,
//   and the top's in proportion to |v.z|;
// - a tank lamp's face is one of the four cardinal ones, equally likely,
//   so each side face is seen, in expectation, as (|v.x| + |v.y|) / 4, and
//   the sides enter as their sum; how many tank lamps a roof carries is
//   taken per square metre of its summit deck (cc_rooftopworks_mech_rho, at
//   the tier's clutter density), pooled over the
//   city's roofs with clutter (cc_rooftopworks_mech_on);
// - the bulkhead's lamp and the pad's lights are the cell's own: the
//   bulkhead's door face seen as max(0, door.v), its back as
//   max(0, -door.v), its two sides together as |t.v| per side, t the door
//   face's tangent, and a lift room beside it in the same four parts;
//   the pad's lights sit on its top, so their deck share is their area,
//   less what a widened spot spills over the pad's edges (each light 1 m in
//   from its corner: the square of the normal integral to 1 m at the
//   handoff's spot width, tabled with the rest).
// The constants were measured by a probe that integrates the prop
// shader's own spots over the props' own faces, at the footprint where the
// props hand over (tests/test_soar_city_roof_lamps.py, which pins them).
// Tank lamps per m^2 of deck, on roofs with clutter, at the clutter
// densities the tiers use (CITY_ROOF_DENSITY_BY_TIER: 0.5, 0.7, 1, 1.2,
// 1.4); a slot's probability is held to 1, so it is not proportional. Linear
// between them (cc_rooftopworks_mech_rho).
const cc_rooftopworks_MECH_RHO_D: vec4<f32> = vec4<f32>(0.5, 0.7, 1.0, 1.2);
const cc_rooftopworks_MECH_RHO_AT: vec4<f32> =
    vec4<f32>(1.110e-4, 1.580e-4, 2.230e-4, 2.467e-4);
const cc_rooftopworks_MECH_RHO_D4: f32 = 1.4;
const cc_rooftopworks_MECH_RHO_AT4: f32 = 2.635e-4;
// The seen areas, m^2 per unit lamp radiance, depend on the footprint where
// the props hand over, because the far mean continues what is drawn just
// inside the range, and there a spot is widened to the pixel (area-
// preserving on an unbounded plane, but a housing's faces are finite, and a
// widened spill spills off them). That footprint is the LOD angle times
// CITY_PROP_RANGE: about 1 m at high tier and 1440p, 11.5 m at minimal. So
// they are tabled at eight handoff footprints and interpolated
// (cc_rooftopworks_far_at): per node, a tank lamp's sides together and top,
// the pad lights' share that lands on the pad, and the node's footprint (m);
// the stair's front, back, two sides together and top, without and with its
// lift room.
// Beside each (the rows ending 2), the same integrals of the light's square:
// the second moment, which says how concentrated it is, so the far mean can
// show it as the small bright faces it is up close (cc_rooftopworks_lamps_far)
// rather than spread thin over the deck, which the display would show
// brighter.
const cc_rooftopworks_FAR_0_MECH: vec4<f32> =
    vec4<f32>(1.8828, 0.48987, 0.99914, 1.0);
const cc_rooftopworks_FAR_0_MECH2: vec4<f32> =
    vec4<f32>(0.45331, 0.051481, 0.0, 0.0);
const cc_rooftopworks_FAR_0_BULK: vec4<f32> =
    vec4<f32>(1.2809, 0.1108, 0.39435, 0.70767);
const cc_rooftopworks_FAR_0_BULK2: vec4<f32> =
    vec4<f32>(0.39779, 0.0010687, 0.011022, 0.051587);
const cc_rooftopworks_FAR_0_LIFT: vec4<f32> =
    vec4<f32>(1.346, 0.12641, 0.51962, 0.78258);
const cc_rooftopworks_FAR_0_LIFT2: vec4<f32> =
    vec4<f32>(0.39867, 0.001108, 0.015437, 0.052411);
const cc_rooftopworks_FAR_1_MECH: vec4<f32> =
    vec4<f32>(1.8816, 0.53155, 0.9067, 2.0);
const cc_rooftopworks_FAR_1_MECH2: vec4<f32> =
    vec4<f32>(0.24228, 0.049523, 0.0, 0.0);
const cc_rooftopworks_FAR_1_BULK: vec4<f32> =
    vec4<f32>(1.2143, 0.1108, 0.39484, 0.82765);
const cc_rooftopworks_FAR_1_BULK2: vec4<f32> =
    vec4<f32>(0.19514, 0.0010687, 0.011062, 0.083146);
const cc_rooftopworks_FAR_1_LIFT: vec4<f32> =
    vec4<f32>(1.2793, 0.12641, 0.5203, 0.90258);
const cc_rooftopworks_FAR_1_LIFT2: vec4<f32> =
    vec4<f32>(0.19584, 0.001108, 0.015493, 0.08398);
const cc_rooftopworks_FAR_2_MECH: vec4<f32> =
    vec4<f32>(1.8573, 0.56596, 0.75124, 3.0);
const cc_rooftopworks_FAR_2_MECH2: vec4<f32> =
    vec4<f32>(0.19006, 0.048338, 0.0, 0.0);
const cc_rooftopworks_FAR_2_BULK: vec4<f32> =
    vec4<f32>(1.1689, 0.1113, 0.41349, 0.88255);
const cc_rooftopworks_FAR_2_BULK2: vec4<f32> =
    vec4<f32>(0.14688, 0.0010801, 0.012471, 0.087037);
const cc_rooftopworks_FAR_2_LIFT: vec4<f32> =
    vec4<f32>(1.2343, 0.12691, 0.54691, 0.95802);
const cc_rooftopworks_FAR_2_LIFT2: vec4<f32> =
    vec4<f32>(0.14758, 0.0011195, 0.017499, 0.087884);
const cc_rooftopworks_FAR_3_MECH: vec4<f32> =
    vec4<f32>(1.8224, 0.57827, 0.63628, 4.0);
const cc_rooftopworks_FAR_3_MECH2: vec4<f32> =
    vec4<f32>(0.16178, 0.045837, 0.0, 0.0);
const cc_rooftopworks_FAR_3_BULK: vec4<f32> =
    vec4<f32>(1.1209, 0.11885, 0.45931, 0.89553);
const cc_rooftopworks_FAR_3_BULK2: vec4<f32> =
    vec4<f32>(0.12318, 0.0012434, 0.015597, 0.082121);
const cc_rooftopworks_FAR_3_LIFT: vec4<f32> =
    vec4<f32>(1.1907, 0.13485, 0.60983, 0.9762);
const cc_rooftopworks_FAR_3_LIFT2: vec4<f32> =
    vec4<f32>(0.12401, 0.0012853, 0.021946, 0.083114);
const cc_rooftopworks_FAR_4_MECH: vec4<f32> =
    vec4<f32>(1.6966, 0.55076, 0.50516, 6.0);
const cc_rooftopworks_FAR_4_MECH2: vec4<f32> =
    vec4<f32>(0.11675, 0.035838, 0.0, 0.0);
const cc_rooftopworks_FAR_4_BULK: vec4<f32> =
    vec4<f32>(0.92252, 0.19527, 0.56513, 0.8159);
const cc_rooftopworks_FAR_4_BULK2: vec4<f32> =
    vec4<f32>(0.075852, 0.0031687, 0.020655, 0.05806);
const cc_rooftopworks_FAR_4_LIFT: vec4<f32> =
    vec4<f32>(1.0416, 0.23389, 0.72852, 0.95365);
const cc_rooftopworks_FAR_4_LIFT2: vec4<f32> =
    vec4<f32>(0.077739, 0.0033668, 0.028411, 0.060259);
const cc_rooftopworks_FAR_5_MECH: vec4<f32> =
    vec4<f32>(1.3405, 0.44154, 0.43763, 8.0);
const cc_rooftopworks_FAR_5_MECH2: vec4<f32> =
    vec4<f32>(0.057849, 0.018695, 0.0, 0.0);
const cc_rooftopworks_FAR_5_BULK: vec4<f32> =
    vec4<f32>(0.62375, 0.2736, 0.57417, 0.61767);
const cc_rooftopworks_FAR_5_BULK2: vec4<f32> =
    vec4<f32>(0.032674, 0.0058555, 0.01866, 0.029292);
const cc_rooftopworks_FAR_5_LIFT: vec4<f32> =
    vec4<f32>(0.82436, 0.38007, 0.71573, 0.85359);
const cc_rooftopworks_FAR_5_LIFT2: vec4<f32> =
    vec4<f32>(0.036163, 0.0068382, 0.024315, 0.033319);
const cc_rooftopworks_FAR_6_MECH: vec4<f32> =
    vec4<f32>(1.0415, 0.3474, 0.3976, 10.0);
const cc_rooftopworks_FAR_6_MECH2: vec4<f32> =
    vec4<f32>(0.031627, 0.010357, 0.0, 0.0);
const cc_rooftopworks_FAR_6_BULK: vec4<f32> =
    vec4<f32>(0.43768, 0.26892, 0.49106, 0.46062);
const cc_rooftopworks_FAR_6_BULK2: vec4<f32> =
    vec4<f32>(0.015807, 0.0055516, 0.01304, 0.015599);
const cc_rooftopworks_FAR_6_LIFT: vec4<f32> =
    vec4<f32>(0.67386, 0.42639, 0.62575, 0.74226);
const cc_rooftopworks_FAR_6_LIFT2: vec4<f32> =
    vec4<f32>(0.019526, 0.0072047, 0.0164, 0.019938);
const cc_rooftopworks_FAR_7_MECH: vec4<f32> =
    vec4<f32>(0.81481, 0.27506, 0.37138, 12.0);
const cc_rooftopworks_FAR_7_MECH2: vec4<f32> =
    vec4<f32>(0.018442, 0.0060927, 0.0, 0.0);
const cc_rooftopworks_FAR_7_BULK: vec4<f32> =
    vec4<f32>(0.32014, 0.23621, 0.40109, 0.34915);
const cc_rooftopworks_FAR_7_BULK2: vec4<f32> =
    vec4<f32>(0.0084041, 0.0042525, 0.008543, 0.0088102);
const cc_rooftopworks_FAR_7_LIFT: vec4<f32> =
    vec4<f32>(0.55916, 0.41658, 0.54297, 0.63689);
const cc_rooftopworks_FAR_7_LIFT2: vec4<f32> =
    vec4<f32>(0.011671, 0.0061128, 0.010653, 0.012667);

// A wall's projected area over the deck's grows without bound toward
// grazing; past this ratio the deck stops standing in for the walls.
const cc_rooftopworks_FAR_LAMP_CAP: f32 = 3.0;

// The summit deck inside the cell's own column, less the tier standing on
// it: the area a cell's lamps are spread over.
fn cc_rooftopworks_cell_deck(d: cc_rooftopworks_Deck, ci: vec2<i32>) -> f32 {
    let col = cc_rooftopworks_column(ci);
    let lo = max(d.lo, col.xy);
    let hi = min(d.hi, col.zw);
    let hole = cc_rooftopworks_area(max(d.hlo, lo), min(d.hhi, hi));
    return cc_rooftopworks_area(lo, hi) - hole;
}

struct cc_rooftopworks_Far {
    mech: vec4<f32>,
    mech2: vec4<f32>,
    bulk: vec4<f32>,
    bulk2: vec4<f32>,
    lift: vec4<f32>,
    lift2: vec4<f32>,
}

fn cc_rooftopworks_mech_rho() -> f32 {
    let d = clamp(cc_rooftopworks_density(), cc_rooftopworks_MECH_RHO_D.x,
                  cc_rooftopworks_MECH_RHO_D4);
    let x = cc_rooftopworks_MECH_RHO_D;
    let y = cc_rooftopworks_MECH_RHO_AT;
    if (d < x.y) {
        return mix(y.x, y.y, (d - x.x) / (x.y - x.x));
    }
    if (d < x.z) {
        return mix(y.y, y.z, (d - x.y) / (x.z - x.y));
    }
    if (d < x.w) {
        return mix(y.z, y.w, (d - x.z) / (x.w - x.z));
    }
    return mix(y.w, cc_rooftopworks_MECH_RHO_AT4,
               (d - x.w) / (cc_rooftopworks_MECH_RHO_D4 - x.w));
}

fn cc_rooftopworks_far_node(i: i32) -> cc_rooftopworks_Far {
    switch (i) {
        case 0: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_0_MECH,
                cc_rooftopworks_FAR_0_MECH2, cc_rooftopworks_FAR_0_BULK,
                cc_rooftopworks_FAR_0_BULK2, cc_rooftopworks_FAR_0_LIFT,
                cc_rooftopworks_FAR_0_LIFT2);
        }
        case 1: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_1_MECH,
                cc_rooftopworks_FAR_1_MECH2, cc_rooftopworks_FAR_1_BULK,
                cc_rooftopworks_FAR_1_BULK2, cc_rooftopworks_FAR_1_LIFT,
                cc_rooftopworks_FAR_1_LIFT2);
        }
        case 2: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_2_MECH,
                cc_rooftopworks_FAR_2_MECH2, cc_rooftopworks_FAR_2_BULK,
                cc_rooftopworks_FAR_2_BULK2, cc_rooftopworks_FAR_2_LIFT,
                cc_rooftopworks_FAR_2_LIFT2);
        }
        case 3: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_3_MECH,
                cc_rooftopworks_FAR_3_MECH2, cc_rooftopworks_FAR_3_BULK,
                cc_rooftopworks_FAR_3_BULK2, cc_rooftopworks_FAR_3_LIFT,
                cc_rooftopworks_FAR_3_LIFT2);
        }
        case 4: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_4_MECH,
                cc_rooftopworks_FAR_4_MECH2, cc_rooftopworks_FAR_4_BULK,
                cc_rooftopworks_FAR_4_BULK2, cc_rooftopworks_FAR_4_LIFT,
                cc_rooftopworks_FAR_4_LIFT2);
        }
        case 5: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_5_MECH,
                cc_rooftopworks_FAR_5_MECH2, cc_rooftopworks_FAR_5_BULK,
                cc_rooftopworks_FAR_5_BULK2, cc_rooftopworks_FAR_5_LIFT,
                cc_rooftopworks_FAR_5_LIFT2);
        }
        case 6: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_6_MECH,
                cc_rooftopworks_FAR_6_MECH2, cc_rooftopworks_FAR_6_BULK,
                cc_rooftopworks_FAR_6_BULK2, cc_rooftopworks_FAR_6_LIFT,
                cc_rooftopworks_FAR_6_LIFT2);
        }
        default: {
            return cc_rooftopworks_Far(cc_rooftopworks_FAR_7_MECH,
                cc_rooftopworks_FAR_7_MECH2, cc_rooftopworks_FAR_7_BULK,
                cc_rooftopworks_FAR_7_BULK2, cc_rooftopworks_FAR_7_LIFT,
                cc_rooftopworks_FAR_7_LIFT2);
        }
    }
}

// The table at the handoff footprint fp_h (m): linear between its nodes,
// held at its ends. The nodes are 1 m apart to 4 m, then 2 m apart to 12 m,
// so the segment is found in closed form.
fn cc_rooftopworks_far_at(fp_h: f32) -> cc_rooftopworks_Far {
    let x = clamp(fp_h, 1.0, 12.0);
    let i = select(min(3 + i32(floor((x - 4.0) * 0.5)), 6),
                   min(i32(floor(x)) - 1, 2), x < 4.0);
    let a = cc_rooftopworks_far_node(i);
    let b = cc_rooftopworks_far_node(i + 1);
    let w = clamp((x - a.mech.w) / (b.mech.w - a.mech.w), 0.0, 1.0);
    return cc_rooftopworks_Far(mix(a.mech, b.mech, w),
                               mix(a.mech2, b.mech2, w),
                               mix(a.bulk, b.bulk, w), mix(a.bulk2, b.bulk2, w),
                               mix(a.lift, b.lift, w), mix(a.lift2, b.lift2, w));
}

// The lamps' far light, per square metre of deck as seen: for the tank
// lamps and for the stair's, the share of the deck its light is shown on and
// the level it is shown at (over LAMP_RAD in LAMP_COL). From each kind's
// seen flux F and seen second moment S (the table's rows and their squares,
// each face class at its projection): the light is shown as an element of
// area F^2 / S at level S / F, which keeps its flux (their product is F) and
// its concentration (a spot on a small face, not a wash over the deck), so
// the deck's shown mean is what the faces' would be. The pad's lights sit on
// its top, flat and small; their light stays a plain emission.
struct cc_rooftopworks_FarLamps {
    a_m: f32,
    l_m: f32,
    a_b: f32,
    l_b: f32,
    pad: vec3<f32>,
}

fn cc_rooftopworks_lamps_far(cc: CityCell, ci: vec2<i32>, fp_h: f32,
                             d: cc_rooftopworks_Deck,
                             cr: CityCrown,
                             bk: cc_rooftopworks_Bulk, dir: vec3<f32>)
        -> cc_rooftopworks_FarLamps {
    var res = cc_rooftopworks_FarLamps(0.0, 0.0, 0.0, 0.0, vec3<f32>(0.0));
    let a0 = cc_rooftopworks_cell_deck(d, ci);
    if (a0 <= 1.0) {
        return res;
    }
    let v = -dir;
    let vz = max(abs(v.z), 1e-3);
    let cap = cc_rooftopworks_FAR_LAMP_CAP;
    let far = cc_rooftopworks_far_at(fp_h);
    if (cc_rooftopworks_mech_on(cc)) {
        // Each face's projection capped, then their expectation.
        let side = 0.25 * (min(abs(v.x) / vz, cap) + min(abs(v.y) / vz, cap));
        let rho = cc_rooftopworks_mech_rho();
        let fm = rho * (far.mech.x * side + far.mech.y);
        let sm = rho * (far.mech2.x * side + far.mech2.y);
        if (fm > 0.0 && sm > 0.0) {
            res.a_m = fm * fm / sm;
            res.l_m = sm / fm;
        }
    }
    if (bk.ok && bk.lamp) {
        let b = select(far.bulk, far.lift, bk.lift);
        let b2 = select(far.bulk2, far.lift2, bk.lift);
        let f = dot(bk.door, v.xy);
        let t = abs(dot(vec2<f32>(-bk.door.y, bk.door.x), v.xy));
        let pr = vec4<f32>(min(max(f, 0.0) / vz, cap),
                           min(max(-f, 0.0) / vz, cap),
                           min(0.5 * t / vz, cap), 1.0);
        let fb = dot(b, pr) / a0;
        let sb = dot(b2, pr) / a0;
        if (fb > 0.0 && sb > 0.0) {
            res.a_b = fb * fb / sb;
            res.l_b = sb / fb;
        }
    }
    // The two kinds' shares can not outgrow the deck: past it, the light is
    // spread over the whole of it at the level that keeps its flux.
    let a = res.a_m + res.a_b;
    if (a > 1.0) {
        res.l_m = res.l_m * a;
        res.l_b = res.l_b * a;
        res.a_m = res.a_m / a;
        res.a_b = res.a_b / a;
    }
    if (cr.kind == 2) {
        let sig = cc_rooftopworks_PAD_LAMP_SIG;
        res.pad = cc_rooftopworks_PAD_LAMP_COL
                  * (cc_rooftopworks_PAD_LAMP_RAD * 4.0 * 6.283185307 * sig
                     * sig * far.mech.z / a0);
    }
    return res;
}

// Whether the stair or its lift room stands over a deck point p, or between
// it and the eye along a ray `dir` arriving at it.
fn cc_rooftopworks_bulk_hides(bk: cc_rooftopworks_Bulk, p: vec3<f32>,
                              dir: vec3<f32>) -> bool {
    if (!bk.ok) {
        return false;
    }
    let o = p + vec3<f32>(0.0, 0.0, 0.01);
    let back = -dir;
    let inv = vec3<f32>(select(1e30, 1.0 / back.x, abs(back.x) > 1e-9),
                        select(1e30, 1.0 / back.y, abs(back.y) > 1e-9),
                        select(1e30, 1.0 / back.z, abs(back.z) > 1e-9));
    let s = city_box_hit(o, inv, bk.lo, bk.hi);
    if (s.x <= s.y && s.y > 0.0) {
        return true;
    }
    if (bk.lift) {
        let l = city_box_hit(o, inv, bk.lift_lo, bk.lift_hi);
        return l.x <= l.y && l.y > 0.0;
    }
    return false;
}

// The deck under the skylight field as it is shown: the field averaged over
// the pixel's footprint (exact pane by pane, the expectation past a pitch),
// handed back as the radiance that shows as that average; `base` where the
// footprint holds no lit pane.
fn cc_rooftopworks_sky_deck(cc: CityCell, sk: cc_rooftopworks_Sky,
                            h: CityHit, dir: vec3<f32>, fp_filter: f32,
                            base: vec3<f32>, fog: vec4<f32>, shown: bool)
        -> vec3<f32> {
    // The pixel's footprint on the deck (z out of it, the ray into it).
    let hb = max(city_wall_footprint(dir, fp_filter), vec2<f32>(1e-3));
    let blo = h.pos.xy - hb;
    let bhi = h.pos.xy + hb;
    if (any(bhi <= sk.flo) || any(blo >= sk.fhi)) {
        return base;
    }
    var col = CITY_PALETTE_MEAN;
    if (cc.win_mono >= 0.0) {
        col = city_room_color(cc.win_mono, cc.palette_bias);
    }
    let pane_rad = col * cc_rooftopworks_SKY_RAD;
    let span = max(2.0 * hb.x / sk.pitch.x, 2.0 * hb.y / sk.pitch.y);
    let w_mean = smoothstep(0.5, 1.0, span);
    var ex = cc_rooftopworks_SkySum(city_win_zero(), 0.0, -1.0);
    if (w_mean < 1.0) {
        ex = cc_rooftopworks_sky_exact(cc, sk, blo, bhi, base, pane_rad, fog,
                                       shown);
        if (w_mean <= 0.0) {
            if (ex.lit_area <= 0.0) {
                return base;   // roof alone: as the core drew it
            }
            if (ex.whole > 0.0) {
                return base + pane_rad * ex.whole;   // one pane, exactly
            }
        }
    }
    var mn = cc_rooftopworks_SkySum(city_win_zero(), 0.0, -1.0);
    if (w_mean > 0.0) {
        mn = cc_rooftopworks_sky_mean(sk, blo, bhi, base, pane_rad, fog,
                                      shown);
    }
    let box_a = cc_rooftopworks_area(blo, bhi);
    let mean = CityWinSum(mix(ex.sum.shown, mn.sum.shown, w_mean) / box_a,
                          mix(ex.sum.light, mn.sum.light, w_mean) / box_a);
    return city_windows_resolve(mean, fog, shown);
}

fn cc_rooftopworks_roof(cc: CityCell, h: CityHit, fp: f32, fp_filter: f32,
                        dir: vec3<f32>, direct: bool, base: vec3<f32>)
        -> vec3<f32> {
    if (!cc.built || cc.arch == 4 || cc.land_use != CITY_USE_URBAN) {
        return vec3<f32>(0.0);
    }
    let d = cc_rooftopworks_deck(cc, 0);
    if (!d.ok || abs(h.pos.z - d.z) > 0.2) {
        return vec3<f32>(0.0);
    }
    let cr = city_crown(cc, h.cell);
    let bk = cc_rooftopworks_bulk(cc, h.cell, cr);
    // Props are drawn only on the view ray, inside the range; a reflection
    // traces none (CITY_TRACE_PLAIN), so it sees the far means at any range.
    var far = cc_rooftopworks_FarLamps(0.0, 0.0, 0.0, 0.0, vec3<f32>(0.0));
    var hidden = false;
    if (!(direct && city_props_drawn(h.cell, h.pos, h.t))) {
        // The footprint where the props hand over, which the far table is
        // keyed on.
        let fp_h = lod_pixel_angle() * CITY_PROP_RANGE;
        far = cc_rooftopworks_lamps_far(cc, h.cell, fp_h, d, cr, bk, dir);
        // What the stair and its lift room would hide, were they drawn: a
        // deck point under them, or behind them along the ray (the one
        // prop computed here anyway). Up close that deck is never hit, so
        // the door pool and the skylights on it must not show past the
        // range either; the light on the stair's faces is in the far
        // lamps above.
        hidden = cc_rooftopworks_bulk_hides(bk, h.pos, dir);
    }
    let lamps = far.a_m + far.a_b > 0.0;
    let shown = direct && TONE_MAP;
    var fog = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (lamps || (bk.sky_ok && !hidden)) {
        fog = city_fog_ab(h.pos - h.t * dir, dir, h.t, h.pos);
    }
    // The deck itself, as it is shown: the skylights resolved over it.
    var deck = base;
    if (bk.sky_ok && !hidden) {
        deck = cc_rooftopworks_sky_deck(cc, cc_rooftopworks_sky(cc, d, cr, bk),
                                        h, dir, fp_filter, base, fog, shown);
    }
    // The door lamp's pool on the deck in front of the bulkhead: what a
    // roof is read by from the air at night. And the pad's lights, far off.
    // Both are added to the skylights' resolved radiance rather than to each
    // pane before the resolve: second order, and only where the pool reaches
    // the field (the pool is smooth; the panes carry the contrast).
    deck = deck + far.pad;
    if (bk.ok && bk.lamp && !hidden) {
        let pool = cc_rooftopworks_spot(h.pos, bk.seat - vec3<f32>(0.0, 0.0,
                                        bk.seat.z - d.z) + vec3<f32>(bk.door
                                        * 1.4, 0.0), 2.2, fp);
        deck = deck + cc_rooftopworks_LAMP_COL
                      * (0.35 * cc_rooftopworks_LAMP_RAD * CITY_ROOF_ALBEDO
                         * 4.0 * pool);
    }
    if (!lamps) {
        return deck - base;
    }
    // Past the range, the lamps' light shown as the small bright faces it
    // is on up close, beside the deck (cc_rooftopworks_lamps_far).
    let lamp = cc_rooftopworks_LAMP_COL * cc_rooftopworks_LAMP_RAD;
    var acc = city_win_add(city_win_zero(), deck, 1.0 - far.a_m - far.a_b,
                           fog, shown);
    acc = city_win_add(acc, base + lamp * far.l_m, far.a_m, fog, shown);
    acc = city_win_add(acc, base + lamp * far.l_b, far.a_b, fog, shown);
    return city_windows_resolve(acc, fog, shown) - base;
}

// --- the tank, close up: SDF in a box ---------------------------------------
fn cc_rooftopworks_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0)))
         + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// The tank in its own frame: origin at the centre of its footprint, z from
// the deck it stands on. Tub, rim band, four legs — the functional features
// and nothing below them (SPEC: wiper blades, not tire brands).
fn cc_rooftopworks_tank_sdf(p: vec3<f32>) -> f32 {
    let hw = cc_rooftopworks_TANK_HW;
    let hh = cc_rooftopworks_TANK_H;
    let leg = cc_rooftopworks_TANK_LEG;
    let r = cc_rooftopworks_TANK_R;
    let bz1 = hh - cc_rooftopworks_TANK_RIM_T;
    let body = cc_rooftopworks_rbox(
        p - vec3<f32>(0.0, 0.0, 0.5 * (leg + bz1)),
        vec3<f32>(hw - r, hw - r, max(0.5 * (bz1 - leg) - r, 0.05)), r);
    let rim = cc_rooftopworks_rbox(
        p - vec3<f32>(0.0, 0.0, hh - 0.5 * cc_rooftopworks_TANK_RIM_T),
        vec3<f32>(cc_rooftopworks_TANK_RIM_HW - 0.10,
                  cc_rooftopworks_TANK_RIM_HW - 0.10,
                  0.5 * cc_rooftopworks_TANK_RIM_T - 0.04), 0.07);
    // Four legs, folded into one quadrant by the absolute value.
    let legs = cc_rooftopworks_rbox(
        vec3<f32>(abs(p.xy) - vec2<f32>(hw - 0.32), p.z - 0.5 * leg),
        vec3<f32>(0.13, 0.13, 0.5 * leg), 0.04);
    return min(min(body, rim), legs);
}

fn cc_rooftopworks_tank_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.2e-3, -1.2e-3);
    // @domain: the SDF's gradient at a surface point: nonzero off the medial axis
    return normalize(
        e.xyy * cc_rooftopworks_tank_sdf(p + e.xyy)
      + e.yyx * cc_rooftopworks_tank_sdf(p + e.yyx)
      + e.yxy * cc_rooftopworks_tank_sdf(p + e.yxy)
      + e.xxx * cc_rooftopworks_tank_sdf(p + e.xxx));
}

// The round wooden tank, close up, in closed form rather than marched: its
// staved barrel (a vertical cylinder, with its floor), the eave (the
// annulus under the cap's overhang) and its conical roof, each a quadratic
// or a plane in t. Its stand is boxes (the hit's other branch). Returns
// (t, normal), t < 0 for a miss. A sphere-traced barrel beside the square
// tank's cost the far views 3% without ever running.
fn cc_rooftopworks_round_hit(o: vec3<f32>, dir: vec3<f32>, t_in: f32,
                             t_out: f32, base: vec3<f32>) -> vec4<f32> {
    let r = cc_rooftopworks_TANK_HW - 0.08;
    let re = r + cc_rooftopworks_EAVE;
    let z0 = cc_rooftopworks_ROUND_STAND;
    let zt = cc_rooftopworks_ROUND_TANK_H;
    let z1 = zt - cc_rooftopworks_CONE_H;
    let p = o - base;
    var best = vec4<f32>(-1.0, 0.0, 0.0, 1.0);
    // The barrel: |p.xy + t d.xy| = r, entering from outside.
    let a = dot(dir.xy, dir.xy);
    let b = dot(p.xy, dir.xy);
    let c = dot(p.xy, p.xy) - r * r;
    let disc = b * b - a * c;
    if (a > 1e-8 && disc >= 0.0) {
        // @domain: disc >= 0 by the branch
        let t = (-b - sqrt(disc)) / a;
        let z = p.z + t * dir.z;
        if (t >= t_in && t <= t_out && z >= z0 && z <= z1) {
            best = vec4<f32>(t, (p.xy + t * dir.xy) / r, 0.0);
        }
    }
    // The barrel's floor (from below, between the stand's posts) and the
    // eave's underside: horizontal planes, a disc and an annulus.
    if (abs(dir.z) > 1e-6) {
        let tf = (z0 - p.z) / dir.z;
        let qf = p.xy + tf * dir.xy;
        if (dir.z > 0.0 && tf >= t_in && tf <= t_out && dot(qf, qf) <= r * r
            && (best.x < 0.0 || tf < best.x)) {
            best = vec4<f32>(tf, 0.0, 0.0, -1.0);
        }
        let te = (z1 - p.z) / dir.z;
        let qe = p.xy + te * dir.xy;
        let qq = dot(qe, qe);
        if (dir.z > 0.0 && te >= t_in && te <= t_out && qq >= r * r
            && qq <= re * re && (best.x < 0.0 || te < best.x)) {
            best = vec4<f32>(te, 0.0, 0.0, -1.0);
        }
    }
    // The roof: |p.xy + t d.xy| = k (zt - z) for z in [z1, zt], its rim at
    // the eave's radius.
    let k = re / cc_rooftopworks_CONE_H;
    let w0 = zt - p.z;
    let qa = a - k * k * dir.z * dir.z;
    let qb = b + k * k * w0 * dir.z;
    let qc = dot(p.xy, p.xy) - k * k * w0 * w0;
    let qd = qb * qb - qa * qc;
    if (abs(qa) > 1e-8 && qd >= 0.0) {
        // @domain: qd >= 0 by the branch
        let sq = sqrt(qd);
        let ta = (-qb - sq) / qa;
        let tb = (-qb + sq) / qa;
        for (var i: i32 = 0; i < 2; i = i + 1) {
            let t = select(min(ta, tb), max(ta, tb), i == 1);
            let z = p.z + t * dir.z;
            if (t >= t_in && t <= t_out && z >= z1 && z <= zt
                && (best.x < 0.0 || t < best.x)) {
                let q = p + t * dir;
                // @domain: the cone's gradient on its surface, nonzero below the apex
                let n = normalize(vec3<f32>(q.xy, k * k * (zt - q.z)));
                best = vec4<f32>(t, n);
                break;
            }
        }
    }
    return best;
}

// Sphere-trace the tank between the entry and exit of its envelope box.
// Returns the hit t, or -1 when the ray threads past the rounded hull —
// which is what makes the hull read as rounded.
fn cc_rooftopworks_tank_trace(o: vec3<f32>, dir: vec3<f32>, t_in: f32,
                              t_out: f32, base: vec3<f32>) -> f32 {
    var t = max(t_in, 0.0) + 1.0e-3;
    for (var i: i32 = 0; i < cc_rooftopworks_SDF_ITERS; i = i + 1) {
        if (t > t_out) {
            return -1.0;
        }
        let d = cc_rooftopworks_tank_sdf(o + t * dir - base);
        if (d < 2.0e-3) {
            return t;
        }
        t = t + max(d, 3.0e-3);
    }
    return -1.0;
}

// One mechanical object against the segment: two slab tests far out, the
// sphere-traced hull for a tank close in.
fn cc_rooftopworks_mech_hit(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                            inv_dir: vec3<f32>, t0: f32, t1: f32,
                            m: cc_rooftopworks_Mech, s: i32, ci: vec2<i32>,
                            fp: f32) -> CityHit {
    if (m.kind == 0 && fp < cc_rooftopworks_SDF_FP) {
        let elo = min(m.lo, m.alo);
        let ehi = max(m.hi, m.ahi);
        let sb = city_box_hit(o, inv_dir, elo, ehi);
        if (sb.x > sb.y || sb.y <= max(t0, 1e-3) || sb.x > t1) {
            return best;
        }
        let base = vec3<f32>(0.5 * (m.lo.xy + m.hi.xy), m.lo.z);
        if (m.round) {
            var r = best;
            let rh = cc_rooftopworks_round_hit(o, dir, max(sb.x, t0),
                                               min(sb.y, t1), base);
            if (rh.x > max(t0, 1e-3) && rh.x < r.t) {
                let p = o + rh.x * dir;
                let is_cap = p.z > m.hi.z - cc_rooftopworks_CONE_H - 0.01;
                r = CityHit(true, rh.x, p, rh.yzw,
                            select(cc_rooftopworks_K_MECH,
                                   cc_rooftopworks_K_ACC, is_cap) + s, ci);
            }
            // The stand: four posts under the barrel's rim and two tie beams
            // across them at mid-height (the bracing, as a beam can be at
            // this size). Kind ACC, the steel.
            let pr = 0.70 * cc_rooftopworks_TANK_HW;
            let ph = vec2<f32>(cc_rooftopworks_POST_HW);
            let zs = m.lo.z + cc_rooftopworks_ROUND_STAND;
            for (var k: i32 = 0; k < 4; k = k + 1) {
                let q = base.xy + pr * vec2<f32>(select(-1.0, 1.0, (k & 1) == 1),
                                                 select(-1.0, 1.0, k >= 2));
                r = cc_rooftopworks_box(r, o, dir, inv_dir, t0, t1,
                                        vec3<f32>(q - ph, m.lo.z),
                                        vec3<f32>(q + ph, zs),
                                        cc_rooftopworks_K_ACC + s, ci);
            }
            let zt = m.lo.z + 0.5 * cc_rooftopworks_ROUND_STAND;
            r = cc_rooftopworks_box(r, o, dir, inv_dir, t0, t1,
                vec3<f32>(base.xy - vec2<f32>(pr, ph.y), zt),
                vec3<f32>(base.xy + vec2<f32>(pr, ph.y),
                          zt + cc_rooftopworks_TIE_H),
                cc_rooftopworks_K_ACC + s, ci);
            return cc_rooftopworks_box(r, o, dir, inv_dir, t0, t1,
                vec3<f32>(base.xy - vec2<f32>(ph.x, pr), zt),
                vec3<f32>(base.xy + vec2<f32>(ph.x, pr),
                          zt + cc_rooftopworks_TIE_H),
                cc_rooftopworks_K_ACC + s, ci);
        }
        let t = cc_rooftopworks_tank_trace(o, dir, max(sb.x, t0),
                                           min(sb.y, t1), base);
        if (t < 0.0 || t > t1 || t <= max(t0, 1e-3) || t >= best.t) {
            return best;
        }
        let p = o + t * dir;
        // The rim band keeps its own kind, so the shader still gives it the
        // galvanised albedo it had when it was a separate box.
        let is_rim = p.z > m.hi.z - cc_rooftopworks_TANK_RIM_T - 0.02;
        return CityHit(true, t, p,
                       cc_rooftopworks_tank_normal(p - base),
                       select(cc_rooftopworks_K_MECH, cc_rooftopworks_K_ACC,
                              is_rim) + s, ci);
    }
    var r = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1, m.lo, m.hi,
                                cc_rooftopworks_K_MECH + s, ci);
    if (m.has_acc) {
        r = cc_rooftopworks_box(r, o, dir, inv_dir, t0, t1, m.alo, m.ahi,
                                cc_rooftopworks_K_ACC + s, ci);
    }
    return r;
}

// --- the parapet ring -------------------------------------------------------
// A hollow box: outer minus inner, two slab tests. Both rects are the
// BUILDING's, not the cell's — see the header note — and correctness comes
// from accepting only a hit inside this visit's ray segment.
fn cc_rooftopworks_has_parapet(cc: CityCell) -> vec2<f32> {
    // .x: does the wall exist. .y: is its coping lit.
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x9e3779b9u, 0x68e31da4u));
    return vec2<f32>(gh.x, gh.y);
}

fn cc_rooftopworks_parapet(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                           inv_dir: vec3<f32>, t0: f32, t1: f32,
                           d: cc_rooftopworks_Deck, ci: vec2<i32>)
        -> CityHit {
    let omin = vec3<f32>(d.lo, d.z);
    let omax = vec3<f32>(d.hi, d.z + cc_rooftopworks_PAR_H);
    let so = city_box_hit(o, inv_dir, omin, omax);
    if (so.x > so.y || so.y <= max(t0, 1e-3) || so.x > t1) {
        return best;
    }
    let w = cc_rooftopworks_PAR_W;
    let imin = vec3<f32>(d.lo + w, d.z - 1.0);
    let imax = vec3<f32>(d.hi - w, d.z + cc_rooftopworks_PAR_H + 1.0);
    let si = city_box_hit(o, inv_dir, imin, imax);
    let inner = si.x <= si.y;
    // The ring along the ray is [so.x, so.y] with (si.x, si.y) removed. Its
    // two possible front surfaces are the outer entry and the inner exit.
    let lo_t = max(t0, 1e-3);
    var bt = 1e30;
    var on_inner = false;
    if (!(inner && si.x < so.x && si.y > so.x) && so.x > lo_t && so.x <= t1) {
        bt = so.x;
    }
    if (inner && si.y > so.x && si.y < so.y && si.y > lo_t && si.y <= t1
        && si.y < bt) {
        bt = si.y;
        on_inner = true;
    }
    if (bt >= 1e30 || bt >= best.t) {
        return best;
    }
    let p = o + bt * dir;
    let n = select(city_box_normal(p, omin, omax),
                   -city_box_normal(p, imin, imax), on_inner);
    return CityHit(true, bt, p, n, cc_rooftopworks_K_PARAPET, ci);
}

// --- external pipework ------------------------------------------------------
struct cc_rooftopworks_Pipes {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    n: i32,
    axis_y: bool,   // the bundle runs along the y wall
}

fn cc_rooftopworks_pipes(cc: CityCell) -> cc_rooftopworks_Pipes {
    var res: cc_rooftopworks_Pipes;
    res.ok = false;
    res.lo = vec3<f32>(0.0);
    res.hi = vec3<f32>(0.0);
    res.n = 1;
    res.axis_y = false;
    // Mid-rise only, and only on the two archetypes whose base mass is a
    // plain vertical box: a riser cannot follow a sloped frustum wall.
    if (cc.arch != 0 && cc.arch != 2) {
        return res;
    }
    if (cc.height < cc_rooftopworks_PIPE_H_LO
        || cc.height > cc_rooftopworks_PIPE_H_HI) {
        return res;
    }
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x27d4eb2fu, 0xb5297a4du));
    if (gh.x >= cc_rooftopworks_PIPE_FRAC) {
        return res;
    }
    let n = 1 + i32(gh.y * 2.999);
    let span = f32(n) * cc_rooftopworks_PIPE_D
             + f32(n - 1) * cc_rooftopworks_PIPE_GAP;
    let b = cc.b1min.xy;
    let bt = cc.b1max.xy;
    let top = cc.b1max.z - 0.8;
    if (top < 8.0) {
        return res;
    }
    let ay = gh.z > 0.5;
    let neg_x = gh.w < 0.5;
    let neg_y = fract(gh.w * 13.0) < 0.5;
    if (ay) {
        // Riding a wall whose normal is +-x, running up beside a corner.
        let xw = select(bt.x, b.x, neg_x);
        let x0 = select(xw, xw - cc_rooftopworks_PIPE_OUT, neg_x);
        let y0 = select(bt.y - cc_rooftopworks_PIPE_CORNER - span,
                        b.y + cc_rooftopworks_PIPE_CORNER, neg_y);
        res.lo = vec3<f32>(x0, y0, 2.0);
        res.hi = vec3<f32>(x0 + cc_rooftopworks_PIPE_OUT, y0 + span, top);
    } else {
        let yw = select(bt.y, b.y, neg_y);
        let y0 = select(yw, yw - cc_rooftopworks_PIPE_OUT, neg_y);
        let x0 = select(bt.x - cc_rooftopworks_PIPE_CORNER - span,
                        b.x + cc_rooftopworks_PIPE_CORNER, neg_x);
        res.lo = vec3<f32>(x0, y0, 2.0);
        res.hi = vec3<f32>(x0 + span, y0 + cc_rooftopworks_PIPE_OUT, top);
    }
    res.ok = true;
    res.n = n;
    res.axis_y = ay;
    return res;
}

// --- the trace hook ---------------------------------------------------------
fn cc_rooftopworks_props_trace(o: vec3<f32>, dir: vec3<f32>,
                               inv_dir: vec3<f32>, t0: f32, t1: f32,
                               ci: vec2<i32>, cc: CityCell) -> CityHit {
    // Gate 0, no hash, no arithmetic: unbuilt lots and spire crowns are not
    // ours, and neither are warehouse roofs (the industry component's
    // sawtooth and monitor ridges). The architecture rule is enforced here
    // first and in the deck function second, so neither can be got around.
    if (!cc.built || cc.arch == 4 || cc.land_use == CITY_USE_INDUSTRIAL) {
        return cc_rooftopworks_miss(ci);
    }
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    let z_lo = min(za, zb);
    let z_hi = max(za, zb);
    let fp = lod_pixel_angle() * max(t0, 0.0);
    var best = cc_rooftopworks_miss(ci);

    // Gate 1: the roof band, from the lowest deck this building owns to the
    // headroom above its summit. Most city rays never enter it.
    if (z_hi >= cc_rooftopworks_deck_low(cc) - 1.0
        && z_lo <= cc.height + cc_rooftopworks_HEADROOM) {
        let d0 = cc_rooftopworks_deck(cc, 0);
        let cr = city_crown(cc, ci);
        if (d0.ok) {
            // The parapet lives in a 1.25 m band on one deck: a tighter z
            // test than the section gate, and it comes before its hash.
            if (z_hi >= d0.z - 0.2
                && z_lo <= d0.z + cc_rooftopworks_PAR_H + 0.2) {
                let par = cc_rooftopworks_has_parapet(cc);
                if (par.x < cc_rooftopworks_PAR_FRAC) {
                    best = cc_rooftopworks_parapet(best, o, dir, inv_dir,
                                                   t0, t1, d0, ci);
                }
            }
            // The helipad. (A penthouse, kind 1, is the core's massing.)
            if (cr.kind == 2) {
                best = cc_rooftopworks_box(
                    best, o, dir, inv_dir, t0, t1,
                    vec3<f32>(cr.lo, cr.z0), vec3<f32>(cr.hi, cr.z1),
                    cc_rooftopworks_K_PAD, ci);
            }
        }
        // Nothing mechanical stands more than 5 m off a deck, so the slot
        // loop gets its own band inside the section gate.
        let bk = cc_rooftopworks_bulk(cc, ci, cr);
        if (bk.ok && z_lo <= bk.hi.z + 0.2) {
            best = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1, bk.lo,
                                       bk.hi, cc_rooftopworks_K_BULK, ci);
        }
        if (d0.ok && z_lo <= d0.z + cc_rooftopworks_UNIT_H + 0.2) {
            for (var k: i32 = 0; k < cc_rooftopworks_RUN_N; k = k + 1) {
                let rn = cc_rooftopworks_run(cc, ci, k, cr, bk);
                if (rn.ok) {
                    best = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1,
                                               rn.lo, rn.hi,
                                               cc_rooftopworks_K_RUN, ci);
                }
            }
            for (var k: i32 = 0; k < cc_rooftopworks_UNIT_N; k = k + 1) {
                let un = cc_rooftopworks_unit(cc, ci, k, cr, bk);
                if (un.ok) {
                    best = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1,
                                               un.lo, un.hi,
                                               cc_rooftopworks_K_UNIT, ci);
                }
            }
        }
        if (bk.ok && bk.lift && z_lo <= bk.lift_hi.z + 0.2) {
            best = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1,
                                       bk.lift_lo, bk.lift_hi,
                                       cc_rooftopworks_K_BULK, ci);
        }
        let mech_on = z_lo <= cc.height + cc_rooftopworks_VENT_H + 0.2
                      && cc_rooftopworks_mech_on(cc);
        for (var s: i32 = 0; s < 4; s = s + 1) {
            let m = cc_rooftopworks_mech(cc, ci, s, mech_on, cr, bk);
            if (m.ok) {
                best = cc_rooftopworks_mech_hit(best, o, dir, inv_dir, t0, t1,
                                                m, s, ci, fp);
            }
        }
    }

    // Gate 2: pipework, close range only.
    if (fp < cc_rooftopworks_PIPE_FP && z_lo <= cc.b1max.z && z_hi >= 1.5) {
        let pp = cc_rooftopworks_pipes(cc);
        if (pp.ok) {
            best = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1,
                                       pp.lo, pp.hi,
                                       cc_rooftopworks_K_PIPE, ci);
        }
    }
    return best;
}

// --- shading ----------------------------------------------------------------
// A point source seen on a surface, with the energy-preserving widening the
// signage uses: as the footprint grows the spot spreads to a pixel and dims
// by the same factor, so a lamp never strobes and never vanishes.
fn cc_rooftopworks_spot(p: vec3<f32>, seat: vec3<f32>, sig: f32, fp: f32)
        -> f32 {
    let s = max(sig, 0.30 * fp);
    let k = (sig * sig) / (s * s);
    let d = p - seat;
    return k * exp(-dot(d, d) / (2.0 * s * s));
}

fn cc_rooftopworks_fill(p: vec3<f32>) -> vec3<f32> {
    return CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
}

// Dark painted steel against the night: a little skyglow, a little moon, and
// a rim that brightens toward the horizontal because the light on a roof
// comes from the street below and from the city all around it.
fn cc_rooftopworks_dark(h: CityHit, albedo: f32) -> vec3<f32> {
    let fill = cc_rooftopworks_fill(h.pos);
    let side = 1.0 - abs(h.normal.z);
    return albedo * fill * (1.0 + 1.6 * side)
         + albedo * CITY_MOONLIGHT * max(dot(h.normal, u.sun_dir), 0.0);
}

// The house colour, where the building has one; cool white otherwise.
fn cc_rooftopworks_trim_color(cc: CityCell) -> vec3<f32> {
    if (cc.win_mono >= 0.0) {
        return city_room_color(cc.win_mono, cc.palette_bias);
    }
    return cc_rooftopworks_TRIM_COOL;
}

// --- the coping trim, as shown -------------------------------------------------
// Bright detail on a dark wall, averaged under the pixel's footprint as the
// flat facade averages its windows (see the skylights): what each element
// SHOWS, weighted by its share of the footprint, handed back by
// city_windows_resolve as the radiance that shows as that average.
// Component shading is not told whether a hit is seen directly or in a
// reflection, so it is averaged as shown; a parapet seen in a reflection
// gets its shown mean, not its plain one.

// How much of [u0, u1] the trim's runs light: fixtures on a TRIM_PITCH
// centre, lit over the first TRIM_DUTY of it.
fn cc_rooftopworks_trim_run_int(u: f32) -> f32 {
    let p = cc_rooftopworks_TRIM_PITCH;
    let q = floor(u / p);
    return (q * cc_rooftopworks_TRIM_DUTY
            + min((u / p) - q, cc_rooftopworks_TRIM_DUTY)) * p;
}

fn cc_rooftopworks_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    // --- parapet -----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PARAPET) {
        let d0 = cc_rooftopworks_deck(cc, 0);
        var e = cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO * 1.3);
        let par = cc_rooftopworks_has_parapet(cc);
        if (par.y < cc_rooftopworks_PAR_LIT) {
            // A strip of light under the coping, in runs: fixtures on a
            // 6.5 m centre with a dark joint between them read as
            // hardware, where an unbroken glowing rectangle round every
            // roof reads as wireframe. The run coordinate is the perimeter
            // direction, taken from which edge of the deck the hit is
            // nearest. The band and the runs are separable, so the share
            // of the footprint they light is closed form at any size, and
            // the footprint is shown as that mixture of lit and dark (see
            // above): a line at two kilometres, not a sparkle and not a
            // wash.
            let dxy = min(h.pos.xy - d0.lo, d0.hi - h.pos.xy);
            let along_y = dxy.x < dxy.y;
            let uc = select(h.pos.x, h.pos.y, along_y);
            let top = h.normal.z > 0.5;
            let face = select(1.0, 0.45, top);
            let d_u = select(dir.x, dir.y, along_y);
            let d_w = select(dir.y, dir.x, along_y);
            // The footprint in the face's own frame: along the run, then
            // up the wall (across the coping, for its top), then out.
            var dl = vec3<f32>(d_u, dir.z, dot(dir.xy, h.normal.xy));
            if (top) {
                dl = vec3<f32>(d_u, d_w, dir.z);
            }
            let hb = max(city_wall_footprint(dl, fp), vec2<f32>(1e-3));
            let run = (cc_rooftopworks_trim_run_int(uc + hb.x)
                       - cc_rooftopworks_trim_run_int(uc - hb.x))
                      / (2.0 * hb.x);
            var band = 1.0;
            if (!top) {
                let up = h.pos.z - d0.z;
                let v0 = clamp(up - hb.y, 0.0, cc_rooftopworks_PAR_H);
                let v1 = clamp(up + hb.y, 0.0, cc_rooftopworks_PAR_H);
                let b0 = cc_rooftopworks_PAR_H - cc_rooftopworks_PAR_TRIM;
                if (v1 - v0 < 1e-4) {
                    band = select(0.0, 1.0, up > b0);
                } else {
                    band = max(v1 - max(v0, b0), 0.0) / (v1 - v0);
                }
            }
            let share = clamp(run * band, 0.0, 1.0);
            let lit = e + cc_rooftopworks_trim_color(cc)
                          * (cc_rooftopworks_TRIM_RAD * face);
            if (share >= 1.0) {
                return lit;
            }
            if (share > 0.0) {
                let fog = city_fog_ab(h.pos - h.t * dir, dir, h.t, h.pos);
                var acc = city_win_add(city_win_zero(), lit, share, fog,
                                       TONE_MAP);
                acc = city_win_add(acc, e, 1.0 - share, fog, TONE_MAP);
                return city_windows_resolve(acc, fog, TONE_MAP);
            }
        }
        return e;
    }

    // --- helipad -----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PAD) {
        let cr = city_crown(cc, h.cell);
        var e = cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO * 0.8);
        if (h.normal.z > 0.5) {
            // Paint: a ring and a bar. Albedo only — at night it is barely
            // there, which is right; the pad is read by its corner lights.
            let c = 0.5 * (cr.lo + cr.hi);
            let half = 0.5 * (cr.hi - cr.lo);
            let q = (h.pos.xy - c) / max(half, vec2<f32>(1e-3));
            let rr = length(q);
            let ring = select(0.0, 1.0, abs(rr - 0.62) < 0.06);
            let bar = select(0.0, 1.0,
                             abs(q.x) < 0.10 && abs(q.y) < 0.34);
            let paint = max(ring, bar) * (1.0 - smoothstep(0.4, 1.4, fp))
                      + 0.22 * smoothstep(0.4, 1.4, fp);
            e = e + vec3<f32>(cc_rooftopworks_PAD_MARK * paint)
                    * cc_rooftopworks_fill(h.pos) * 3.0;
            // Four corner lights, 1 m in from the corners.
            let s = half - 1.0;
            var lamps = 0.0;
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(s.x, s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(-s.x, s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(s.x, -s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(-s.x, -s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            e = e + cc_rooftopworks_PAD_LAMP_COL
                    * (cc_rooftopworks_PAD_LAMP_RAD * lamps);
        }
        return e;
    }

    // --- the bulkhead --------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_BULK) {
        let bk = cc_rooftopworks_bulk(cc, h.cell,
                                      city_crown(cc, h.cell));
        var e = cc_rooftopworks_dark(h, cc_rooftopworks_BULK_ALBEDO);
        if (bk.lamp) {
            let src = cc_rooftopworks_spot(h.pos, bk.seat,
                                           cc_rooftopworks_LAMP_SIG, fp);
            let spill = cc_rooftopworks_spot(h.pos, bk.seat, 1.6, fp);
            e = e + cc_rooftopworks_LAMP_COL
                    * (cc_rooftopworks_LAMP_RAD * (src + 0.12 * spill));
        }
        // The door: a dark steel leaf in the middle of its face, under the
        // lamp, so it darkens the light the face returns as well as the
        // face (the far lamp constants are measured with it). Box-filtered
        // over the pixel's footprint on the face: the leaf's share of it,
        // which is the leaf itself up close and its area's mean far off.
        let t = vec2<f32>(-bk.door.y, bk.door.x);
        let c = 0.5 * (bk.lo.xy + bk.hi.xy);
        let along = dot(h.pos.xy - c, t);
        let fw = 0.5 * abs(dot(bk.hi.xy - bk.lo.xy, t));
        // On the stair's own door face: the lift room beside it faces the
        // same way and is drawn under the same kind.
        let on_face = dot(h.normal.xy, bk.door) > 0.5 && abs(along) <= fw + 1e-3
            && abs(dot(h.pos.xy - c, bk.door)
                   - 0.5 * abs(dot(bk.hi.xy - bk.lo.xy, bk.door))) < 0.05;
        if (on_face) {
            let up = h.pos.z - bk.lo.z;
            let dl = vec3<f32>(dot(dir.xy, t), dir.z, dot(dir.xy, bk.door));
            let hb = max(city_wall_footprint(dl, fp), vec2<f32>(1e-4));
            // The footprint clipped to the face it lies on: what falls off
            // the face (onto the deck at its foot, round its corners) is not
            // this face's to darken, and the leaf's share is of what is left.
            let u0 = max(along - hb.x, -fw);
            let u1 = max(min(along + hb.x, fw), u0 + 1e-4);
            let v0 = max(up - hb.y, 0.0);
            let v1 = max(min(up + hb.y, bk.hi.z - bk.lo.z), v0 + 1e-4);
            let w = 0.5 * cc_rooftopworks_BULK_DOOR_W;
            let su = max(min(u1, w) - max(u0, -w), 0.0);
            let sv = max(min(v1, cc_rooftopworks_BULK_DOOR_H) - v0, 0.0);
            let share = su * sv / ((u1 - u0) * (v1 - v0));
            e = e * (1.0 - share * (1.0 - cc_rooftopworks_BULK_DOOR_K));
        }
        // The machine room's walls: a louvred vent panel in the middle of
        // each, where the hoist's motor room breathes. The slats are what
        // the door lamp's light picks out, so the pattern multiplies the
        // light as well as the wall; it averages to exactly 1 over its pitch
        // (dark slats, brighter gaps between them), so the panel's mean,
        // and the far lamp constants measured without it, are unchanged.
        // Drawn while a slat is wider than a pixel, its mean (1) past it.
        let lc = 0.5 * (bk.lift_lo.xy + bk.lift_hi.xy);
        if (bk.lift && abs(h.normal.z) < 0.5
            && all(abs(h.pos.xy - lc) < vec2<f32>(cc_rooftopworks_LIFT_HX
                                                  + 0.01))) {
            // @domain: a wall's normal: |xy| = 1
            let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
            let along = abs(dot(h.pos.xy - lc, vec2<f32>(-nh.y, nh.x)));
            let up = h.pos.z - bk.lift_lo.z;
            if (along < 0.5 * cc_rooftopworks_LOUVRE_W
                && up > cc_rooftopworks_LOUVRE_Z.x
                && up < cc_rooftopworks_LOUVRE_Z.y) {
                let dk = cc_rooftopworks_LOUVRE_DARK;
                let gap = (1.0 - dk * cc_rooftopworks_LOUVRE_K) / (1.0 - dk);
                let slat = select(gap, cc_rooftopworks_LOUVRE_K,
                                  fract(up / cc_rooftopworks_LOUVRE_PITCH)
                                  < dk);
                e = e * mix(slat, 1.0, smoothstep(0.06, 0.25, fp));
            }
        }
        return e;
    }

    // --- packaged units -----------------------------------------------------
    if (h.kind == cc_rooftopworks_K_UNIT) {
        // Grey casing, a fan grille on the top, a dark louvre band on the
        // sides; a few pixels out it is the casing's mean.
        var e = cc_rooftopworks_dark(h, 0.22);
        let k = 1.0 - smoothstep(0.15, 0.6, fp);
        let crs = city_crown(cc, h.cell);
        let bk = cc_rooftopworks_bulk(cc, h.cell, crs);
        var un = cc_rooftopworks_unit(cc, h.cell, 0, crs, bk);
        for (var j: i32 = 1; j < cc_rooftopworks_UNIT_N; j = j + 1) {
            if (un.ok && all(h.pos >= un.lo - 0.02)
                && all(h.pos <= un.hi + 0.02)) {
                break;
            }
            un = cc_rooftopworks_unit(cc, h.cell, j, crs, bk);
        }
        let along = select(h.pos.x - un.lo.x, h.pos.y - un.lo.y, un.along_y);
        let fu = fract(along / cc_rooftopworks_UNIT_PITCH);
        let seam = select(1.0, 0.3, fu > 0.92);
        if (h.normal.z > 0.5) {
            // A fan grille in the middle of each unit's top.
            let cu = (fu - 0.46) * cc_rooftopworks_UNIT_PITCH;
            let grille = select(1.0, 0.45, abs(cu) < 0.55);
            e = e * mix(1.0, grille * seam, k);
        } else {
            let band = select(1.0, 0.5, fract(h.pos.z * 6.0) < 0.5);
            e = e * mix(1.0, band * seam, k * 0.6);
        }
        return e;
    }

    // --- runs --------------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_RUN) {
        let crs = city_crown(cc, h.cell);
        let bk = cc_rooftopworks_bulk(cc, h.cell, crs);
        // Which run: the one whose box holds the hit.
        var rn = cc_rooftopworks_run(cc, h.cell, 0, crs, bk);
        for (var k: i32 = 1; k < cc_rooftopworks_RUN_N; k = k + 1) {
            if (rn.ok && all(h.pos >= rn.lo - 0.02)
                && all(h.pos <= rn.hi + 0.02)) {
                break;
            }
            rn = cc_rooftopworks_run(cc, h.cell, k, crs, bk);
        }
        let k = 1.0 - smoothstep(0.15, 0.6, fp);
        if (rn.duct) {
            // Galvanised sheet with a joint every 1.2 m.
            let along = select(h.pos.x, h.pos.y, rn.along_y);
            let seam = select(1.0, 0.55, fract(along / 1.2) < 0.05);
            return cc_rooftopworks_dark(h, 0.24) * mix(1.0, seam, k);
        }
        // Three pipes across the rack, each read as a cylinder by its
        // shading; the sleepers under them every 2 m.
        let across = select(h.pos.y - rn.lo.y, h.pos.x - rn.lo.x, rn.along_y)
                     / (2.0 * cc_rooftopworks_RACK_HW);
        let fu = fract(across * 3.0);
        let round = sqrt(max(1.0 - (2.0 * fu - 1.0) * (2.0 * fu - 1.0), 0.0));
        let gap = select(1.0, 0.25, fu < 0.08 || fu > 0.92);
        return cc_rooftopworks_dark(h, 0.16)
               * mix(1.0, (0.55 + 0.6 * round) * gap, k);
    }

    // --- pipework ----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PIPE) {
        let pp = cc_rooftopworks_pipes(cc);
        // Across the bundle: each riser reads as a cylinder by shading, a
        // bright sliver where the wall light grazes it and dark at the seam.
        let across = select(h.pos.x - pp.lo.x, h.pos.y - pp.lo.y, pp.axis_y);
        let pitch = cc_rooftopworks_PIPE_D + cc_rooftopworks_PIPE_GAP;
        let fu = clamp(fract(across / pitch) / (cc_rooftopworks_PIPE_D / pitch),
                       0.0, 1.0);
        let round = sqrt(max(1.0 - (2.0 * fu - 1.0) * (2.0 * fu - 1.0), 0.0));
        let face = 1.0 - abs(h.normal.z);
        // The rim: the lit facade behind a riser wraps its edges.
        let k = 1.0 - smoothstep(cc_rooftopworks_PIPE_FADE,
                                 cc_rooftopworks_PIPE_FP, fp);
        let rim = mix(0.35, 1.0, round) * face * k;
        return cc_rooftopworks_dark(h, CITY_FACADE_ALBEDO * 1.4)
             + CITY_PALETTE_MEAN * (0.10 * rim);
    }

    // --- mechanical clusters ------------------------------------------------
    let is_acc = h.kind >= cc_rooftopworks_K_ACC;
    let s = h.kind - select(cc_rooftopworks_K_MECH, cc_rooftopworks_K_ACC,
                            is_acc);
    let crs = city_crown(cc, h.cell);
    let m = cc_rooftopworks_mech(cc, h.cell, s,
                                 cc_rooftopworks_mech_on(cc), crs,
                                 cc_rooftopworks_bulk(cc, h.cell, crs));
    var alb = cc_rooftopworks_MECH_ALBEDO;
    // A round tank's stand is dark steel; the barrel and its roof are wood.
    let stand = m.round
                && h.pos.z - m.lo.z < cc_rooftopworks_ROUND_STAND - 0.01;
    if (m.kind == 0 && is_acc && !m.round) {
        alb = alb * 1.35;  // the rim band is galvanised: it catches more
    }
    var e = cc_rooftopworks_dark(h, alb);
    if (m.round && !stand) {
        e = e * cc_rooftopworks_WOOD;
    }
    if (m.lamp) {
        // The fixture itself, plus the pool it throws on its own housing.
        let src = cc_rooftopworks_spot(h.pos, m.seat,
                                       cc_rooftopworks_LAMP_SIG, fp);
        let spill = cc_rooftopworks_spot(h.pos, m.seat,
                                         cc_rooftopworks_LAMP_SPILL, fp);
        e = e + cc_rooftopworks_LAMP_COL
                * (cc_rooftopworks_LAMP_RAD * (src + 0.10 * spill));
    }
    // Condenser rows: the unit gaps and the grille bands, drawn rather than
    // built. Three units on a 2.6 m pitch; the seams are dark, the grille a
    // shade lighter than the casing.
    if (m.kind == 1 && !is_acc) {
        let along = select(h.pos.y - m.lo.y, h.pos.x - m.lo.x,
                           (m.hi.x - m.lo.x) > (m.hi.y - m.lo.y));
        let fu = fract(along / 2.60);
        let seam = select(1.0, 0.35, fu < 0.12 || fu > 0.88);
        let grille = select(0.0, 1.0, h.normal.z > 0.5 && fu > 0.30
                            && fu < 0.70);
        // Both are modulations of the casing's own albedo, never additions:
        // a fan grille is lighter metal, not a light. (It was an additive
        // skyglow term once, and every condenser on the roof came out cyan.)
        let k = 1.0 - smoothstep(0.5, 1.6, fp);
        e = e * (mix(1.0, seam, k) * (1.0 + 0.8 * grille * k));
    }
    return e;
}
