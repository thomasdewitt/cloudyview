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
//      what puts a few sparks up on the roofs above the window ladder.
//   2. PARAPET. A real low wall around the top deck of ~55% of buildings
//      (the core only fakes one by brightening the roof albedo near the
//      edge), and on ~12% of roofs overall a strip of light along its
//      coping: cool white, or the building's own house colour where it has
//      one. The wall is a hollow box — two slab tests, not four — and it is
//      deliberately NOT clipped to the DDA cell, because it belongs to the
//      building rather than to the cell: every member cell of a merged
//      superblock derives the same ring and catches whatever part of it
//      falls in its own ray segment.
//   3. PENTHOUSE CROWNS. ~20% of tall flat-topped buildings get a 2-3 storey
//      inset glass box on the roof, lit warm at 1.8. On a merged superblock
//      two of the four quadrant cells may carry one, so a big roof reads as
//      a plant room and a penthouse rather than as one centred lump.
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
// test against [lowest deck, roof + 12 m]. A cell whose ray segment misses
// that band — most of them, since most city pixels are looking at a wall or
// at the road — costs exactly that. Inside the band each piece has its own
// tighter z window and its own eligibility test BEFORE it draws a hash: the
// parapet's 1.25 m band, the clutter's 5 m one, and a crown that never
// hashes on a building too short to carry one. Worst case is 4 slots x 2
// boxes + 2 parapet + 1 crown = 11 slab tests, and that only on a large roof
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
const cc_rooftopworks_K_PENT: i32 = 609;
const cc_rooftopworks_K_PAD: i32 = 611;
const cc_rooftopworks_K_PIPE: i32 = 612;

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
const cc_rooftopworks_VENT_GAP: f32 = 1.70;
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
// The trim is a 0.28 m band on a 1.25 m wall: past this footprint the band is
// thinner than a pixel and hands over to its mean over the whole wall face,
// which is the same energy spread out. Nothing switches off (SPEC rule 3).
const cc_rooftopworks_TRIM_LOD_START: f32 = 1.60;
const cc_rooftopworks_TRIM_LOD_FULL: f32 = 4.00;
const cc_rooftopworks_TRIM_MEAN_COMP: f32 = 0.20;  // see PENT_MEAN_COMP

// --- penthouse --------------------------------------------------------------
const cc_rooftopworks_PENT_FRAC: f32 = 0.20;
const cc_rooftopworks_PENT_MIN_H: f32 = 105.0; // "tall": tower, not mid-rise
const cc_rooftopworks_PENT_MIN_DECK: f32 = 15.0;
const cc_rooftopworks_PENT_INSET: f32 = 0.28;
const cc_rooftopworks_PENT_RAD: f32 = 1.80;
const cc_rooftopworks_PENT_COL: vec3<f32> = vec3<f32>(1.00, 0.72, 0.42);
const cc_rooftopworks_PENT_PITCH: f32 = 2.10;  // mullion pitch (m)
const cc_rooftopworks_PENT_MULL: f32 = 0.13;   // dark fraction of the pitch
const cc_rooftopworks_PENT_SILL: f32 = 0.80;   // dark spandrel at its foot
const cc_rooftopworks_PENT_LIT: f32 = 0.58;    // bays with the light on
// Tone-map compensation on the far-field means, the same correction the
// core applies to its own window octaves (CITY_MEAN_COMP_BLOCK/_FLAT): a
// mean taken in linear radiance ahead of a Reinhard curve renders CREAM,
// because a lit bay at 1.8 is already near the knee and its average is not.
// Solved rather than guessed: match tone(mean * E) to lit_frac * tone(B * E)
// at E = 6, which puts the coefficient near a quarter.
const cc_rooftopworks_PENT_MEAN_COMP: f32 = 0.26;

// --- helipad ----------------------------------------------------------------
const cc_rooftopworks_PAD_FRAC: f32 = 0.13;
const cc_rooftopworks_PAD_BIG: f32 = 60.0;     // footprint that qualifies
const cc_rooftopworks_PAD_MIN_H: f32 = 55.0;   // and a building worth landing on
const cc_rooftopworks_PAD_T: f32 = 0.30;
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

// Headroom above the main roof that everything here fits under: a 3-storey
// penthouse is the tallest thing (10.8 m).
const cc_rooftopworks_HEADROOM: f32 = 12.0;

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
    let cell = u.ocean_params.x;
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

// --- the crown: a penthouse, or a helipad, or neither -----------------------
struct cc_rooftopworks_Crown {
    kind: i32,   // 0 none, 1 penthouse, 2 helipad
    lo: vec2<f32>,
    hi: vec2<f32>,
    z0: f32,
    z1: f32,
}

fn cc_rooftopworks_crown(cc: CityCell, ci: vec2<i32>)
        -> cc_rooftopworks_Crown {
    var res: cc_rooftopworks_Crown;
    res.kind = 0;
    res.lo = vec2<f32>(1e30);
    res.hi = vec2<f32>(-1e30);
    res.z0 = 0.0;
    res.z1 = 0.0;
    let d = cc_rooftopworks_deck(cc, 0);
    if (!d.ok) {
        return res;
    }
    let col = cc_rooftopworks_column(ci);
    let rlo = max(d.lo, col.xy);
    let rhi = min(d.hi, col.zw);
    let size = rhi - rlo;
    if (min(size.x, size.y) < cc_rooftopworks_PENT_MIN_DECK) {
        return res;
    }
    // Eligibility before entropy: a building too short for either a
    // penthouse or a pad never draws a hash. This is the gate that keeps the
    // low-rise majority of the tile out of the crown's cost entirely.
    let foot0 = max(d.hi.x - d.lo.x, d.hi.y - d.lo.y);
    let pad_size = cc.merged || foot0 > cc_rooftopworks_PAD_BIG;
    if (cc.height <= cc_rooftopworks_PENT_MIN_H
        && !(pad_size && cc.height > cc_rooftopworks_PAD_MIN_H)) {
        return res;
    }
    // Building-level draws (shared by every cell of a merged group) decide
    // WHETHER; a cell-level draw decides the shape, so the two penthouses on
    // a superblock are not identical twins.
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x1b56c4e9u, 0x0d2f1a37u));
    let lh = city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x9e3779b9u + 0x2545f491u,
        bitcast<u32>(ci.y) * 0x85ebca6bu + 0x27d4eb2fu));
    // Which quadrant of a superblock this cell is.
    let q = (ci.x & 1) + 2 * (ci.y & 1);

    // Helipad first: it wants the same real estate and it is the rarer thing.
    let pad_q = i32(gh.z * 4.0);
    let pad_here = !cc.merged || q == pad_q;
    if (pad_size && pad_here
        && cc.height > cc_rooftopworks_PAD_MIN_H
        && gh.x < cc_rooftopworks_PAD_FRAC) {
        let half = clamp(0.30 * min(size.x, size.y), 4.5, 11.0);
        let c = mix(rlo + half + 1.0, rhi - half - 1.0,
                    vec2<f32>(0.30 + 0.40 * lh.x, 0.30 + 0.40 * lh.y));
        let plo = c - half;
        let phi = c + half;
        // Not on top of the mast, and not hanging over the hole.
        let clear_mast = !cc.has_mast
            || !cc_rooftopworks_overlap(plo, phi, cc.mast_min.xy - 1.0,
                                        cc.mast_max.xy + 1.0);
        if (clear_mast && !cc_rooftopworks_overlap(plo, phi, d.hlo, d.hhi)) {
            res.kind = 2;
            res.lo = plo;
            res.hi = phi;
            res.z0 = d.z;
            res.z1 = d.z + cc_rooftopworks_PAD_T;
            return res;
        }
    }
    // Penthouse. On a superblock only the two diagonal quadrants may carry
    // one, so a big roof gets at most two and they sit apart.
    let pent_here = !cc.merged || ((ci.x + ci.y) & 1) == 0;
    if (cc.height > cc_rooftopworks_PENT_MIN_H && pent_here
        && gh.y < cc_rooftopworks_PENT_FRAC) {
        let inset = max(cc_rooftopworks_PENT_INSET * min(size.x, size.y), 3.0);
        let plo = rlo + inset;
        let phi = rhi - inset;
        if (min(phi.x - plo.x, phi.y - plo.y) > 6.0
            && !cc_rooftopworks_overlap(plo, phi, d.hlo, d.hhi)) {
            res.kind = 1;
            res.lo = plo;
            res.hi = phi;
            res.z0 = d.z;
            res.z1 = d.z + select(7.2, 10.8, lh.z > 0.45);
            return res;
        }
    }
    return res;
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
}

fn cc_rooftopworks_no_mech() -> cc_rooftopworks_Mech {
    return cc_rooftopworks_Mech(false, vec3<f32>(0.0), vec3<f32>(0.0), false,
                                vec3<f32>(0.0), vec3<f32>(0.0), 0, false,
                                vec3<f32>(0.0));
}

// Does this building carry rooftop clutter at all? One building-level draw,
// hoisted out of the slot loop so a cell pays it once rather than four times.
fn cc_rooftopworks_mech_on(cc: CityCell) -> bool {
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x3c6ef372u, 0x165667b1u));
    return gh.w < cc_rooftopworks_MECH_FRAC;
}

fn cc_rooftopworks_slot_draw(ci: vec2<i32>, s: i32) -> vec4<f32> {
    let l = u32(s);
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x85ebca6bu + l * 0x9e3779b9u + 0x7feb352du,
        bitcast<u32>(ci.y) * 0xc2b2ae35u + l * 0x51ed270bu + 0x846ca68bu));
}

// Slot `s` of cell `ci`: which deck it stands on, where, and what it is.
// Deterministic in (cc, ci, s), so the shader re-derives the object from the
// hit kind alone — nothing is smuggled through CityHit.
fn cc_rooftopworks_mech(cc: CityCell, ci: vec2<i32>, s: i32,
                        mech_on: bool, cr: cc_rooftopworks_Crown)
        -> cc_rooftopworks_Mech {
    if (!mech_on) {
        return cc_rooftopworks_no_mech();
    }
    let r = cc_rooftopworks_slot_draw(ci, s);
    var p_slot = 1.0;
    if (s == 1) { p_slot = cc_rooftopworks_SLOT_P1; }
    if (s == 2) { p_slot = cc_rooftopworks_SLOT_P2; }
    if (s == 3) { p_slot = cc_rooftopworks_SLOT_P3; }
    if (r.x >= p_slot) {
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
    var hgt = cc_rooftopworks_TANK_H;
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
        // A second, shorter stack beside the first.
        res.has_acc = true;
        let off = select(vec2<f32>(cc_rooftopworks_VENT_GAP, 0.0),
                         vec2<f32>(0.0, cc_rooftopworks_VENT_GAP), axis_y);
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
    return res;
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
    return normalize(
        e.xyy * cc_rooftopworks_tank_sdf(p + e.xyy)
      + e.yyx * cc_rooftopworks_tank_sdf(p + e.yyx)
      + e.yxy * cc_rooftopworks_tank_sdf(p + e.yxy)
      + e.xxx * cc_rooftopworks_tank_sdf(p + e.xxx));
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
    // ours. The architecture rule is enforced here first and in the deck
    // function second, so neither can be got around.
    if (!cc.built || cc.arch == 4) {
        return cc_rooftopworks_miss(ci);
    }
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    let z_lo = min(za, zb);
    let z_hi = max(za, zb);
    let fp = max(2.0 * u.cam_origin.w / max(u.params.x, 1.0), u.periodic.z)
             * max(t0, 0.0);
    var best = cc_rooftopworks_miss(ci);

    // Gate 1: the roof band, from the lowest deck this building owns to the
    // headroom above its summit. Most city rays never enter it.
    if (z_hi >= cc_rooftopworks_deck_low(cc) - 1.0
        && z_lo <= cc.height + cc_rooftopworks_HEADROOM) {
        let d0 = cc_rooftopworks_deck(cc, 0);
        let cr = cc_rooftopworks_crown(cc, ci);
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
            if (cr.kind != 0) {
                best = cc_rooftopworks_box(
                    best, o, dir, inv_dir, t0, t1,
                    vec3<f32>(cr.lo, cr.z0), vec3<f32>(cr.hi, cr.z1),
                    select(cc_rooftopworks_K_PAD, cc_rooftopworks_K_PENT,
                           cr.kind == 1),
                    ci);
            }
        }
        // Nothing mechanical stands more than 5 m off a deck, so the slot
        // loop gets its own band inside the section gate — the penthouse
        // headroom above it is the crown's business, not the clutter's.
        let mech_on = z_lo <= cc.height + cc_rooftopworks_VENT_H + 0.2
                      && cc_rooftopworks_mech_on(cc);
        for (var s: i32 = 0; s < 4; s = s + 1) {
            let m = cc_rooftopworks_mech(cc, ci, s, mech_on, cr);
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
         + albedo * CITY_MOONLIGHT * max(dot(h.normal, u.sun_dir.xyz), 0.0);
}

// The house colour, where the building has one; cool white otherwise.
fn cc_rooftopworks_trim_color(cc: CityCell) -> vec3<f32> {
    if (cc.win_mono >= 0.0) {
        return city_window_color(cc.win_mono, cc.palette_bias);
    }
    return cc_rooftopworks_TRIM_COOL;
}

fn cc_rooftopworks_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    // --- parapet -----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PARAPET) {
        let d0 = cc_rooftopworks_deck(cc, 0);
        var e = cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO * 1.3);
        let par = cc_rooftopworks_has_parapet(cc);
        if (par.y < cc_rooftopworks_PAR_LIT) {
            // A strip of light under the coping. Sharp while the band is
            // wider than a pixel; past that it is the same energy spread
            // over the whole wall face, which is what a real coping light
            // looks like from two kilometres — a line, not a sparkle.
            let up = h.pos.z - d0.z;
            let band = select(
                0.0, 1.0,
                up > cc_rooftopworks_PAR_H - cc_rooftopworks_PAR_TRIM);
            let mean = (cc_rooftopworks_PAR_TRIM / cc_rooftopworks_PAR_H)
                       * cc_rooftopworks_TRIM_MEAN_COMP;
            let k = smoothstep(cc_rooftopworks_TRIM_LOD_START,
                               cc_rooftopworks_TRIM_LOD_FULL, fp);
            let face = select(1.0, 0.45, h.normal.z > 0.5);
            // Runs, not a continuous tube. An unbroken glowing rectangle
            // around every roof reads as wireframe; fixtures on a 6.5 m
            // centre with a dark joint between them read as hardware. The
            // run coordinate is the perimeter direction, taken from which
            // edge of the deck the hit is nearest.
            let dxy = min(h.pos.xy - d0.lo, d0.hi - h.pos.xy);
            let along_y = dxy.x < dxy.y;
            let uc = select(h.pos.x, h.pos.y, along_y);
            let seg = select(1.0, 0.0,
                             fract(uc / cc_rooftopworks_TRIM_PITCH)
                             > cc_rooftopworks_TRIM_DUTY);
            let segk = mix(seg, cc_rooftopworks_TRIM_DUTY,
                           smoothstep(0.9, 2.4, fp));
            e = e + cc_rooftopworks_trim_color(cc)
                    * (cc_rooftopworks_TRIM_RAD * face * segk
                       * mix(band, mean, k));
        }
        return e;
    }

    // --- penthouse ---------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PENT) {
        let cr = cc_rooftopworks_crown(cc, h.cell);
        if (h.normal.z > 0.5) {
            return cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO);
        }
        // Glass on three-and-a-bit sides, with mullions and a dark spandrel
        // at its foot. The colour follows the house where there is one.
        let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
        let tangent = vec2<f32>(-nh.y, nh.x);
        let uc = dot(h.pos.xy, tangent);
        let up = h.pos.z - cr.z0;
        let mull = select(
            1.0, 0.10,
            fract(uc / cc_rooftopworks_PENT_PITCH) < cc_rooftopworks_PENT_MULL);
        let sill = smoothstep(0.0, cc_rooftopworks_PENT_SILL, up);
        // Panes, not a slab. Lit uniformly at 1.8 the crown came out as the
        // brightest thing on the skyline — brighter in the mean than the
        // facade below it, which is backwards. It is a room: most of the
        // glass is lit, some is not, and each bay has its own level.
        let iu = i32(floor(uc / cc_rooftopworks_PENT_PITCH));
        let iv = i32(floor(up / CITY_FLOOR_H));
        let wh = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu) * 0x9e3779b9u) ^ 0x0d2f1a37u,
            cc.seed.y ^ (bitcast<u32>(iv) * 0x85ebca6bu) ^ 0x1b56c4e9u));
        let bay = select(0.0, 0.45 + 1.10 * wh.z,
                         wh.x < cc_rooftopworks_PENT_LIT);
        // Mullions are 13% of the pitch and a bay is 2.1 m: both are
        // sub-pixel past a couple of metres, where the wall settles to its
        // own mean rather than flickering.
        let k = smoothstep(0.7, 2.2, fp);
        let m = mix(mull * bay,
                    (1.0 - 0.9 * cc_rooftopworks_PENT_MULL)
                    * cc_rooftopworks_PENT_LIT
                    * cc_rooftopworks_PENT_MEAN_COMP, k);
        var col = cc_rooftopworks_PENT_COL;
        if (cc.win_mono >= 0.0) {
            col = city_window_color(cc.win_mono, cc.palette_bias);
        }
        return cc_rooftopworks_dark(h, CITY_FACADE_ALBEDO)
             + col * (cc_rooftopworks_PENT_RAD * m * sill);
    }

    // --- helipad -----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PAD) {
        let cr = cc_rooftopworks_crown(cc, h.cell);
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
    let m = cc_rooftopworks_mech(cc, h.cell, s,
                                 cc_rooftopworks_mech_on(cc),
                                 cc_rooftopworks_crown(cc, h.cell));
    var alb = cc_rooftopworks_MECH_ALBEDO;
    if (m.kind == 0 && is_acc) {
        alb = alb * 1.35;  // the rim band is galvanised: it catches more
    }
    var e = cc_rooftopworks_dark(h, alb);
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
