// aircars — flying cars, frozen mid-flow.
//
// The sky layer of a two-layer traffic system: a sibling component parks the
// cars on the ground, these hold the air above the avenues. There is no time
// here (SPEC rule 4), so this is a long-exposure photograph of living
// traffic — every craft is where its hash says it is, on every frame and from
// every camera, and the streams read as streams because of where they are,
// not because they move.
//
// LAYOUT. Traffic follows the avenue lattice (`city_is_avenue`, every 8
// blocks) but flies off to one side of it and well above. Each avenue carries
// two counter-flowing lanes, one over each verge, ~9 m in from the avenue
// centerline; the two lanes belong to the two cells that flank the avenue —
// the cell whose MIN edge is the avenue owns the lane on its own side, the
// cell whose MAX edge is the avenue owns the other. That ownership rule is
// what keeps every craft geometrically inside the cell that draws it, which
// the DDA requires: a box straddling a cell boundary is tested only from one
// side and disappears from the other. It also makes the flow directions fall
// out right-hand — keep to the right of the avenue you are flying up, and the
// +x verge of a north-south avenue runs north.
//
// Altitude is a two-deck system, ~55 m and ~95 m, +-8 m per craft, chosen by
// the craft's own hash — so a lane is a loose braid rather than a wire, and
// the low deck passes under the high one at every crossing. Over ordinary
// blocks a few percent of cells carry a free flyer at 40-130 m, off-lattice
// and on any heading, which is what stops the air from being a pure grid.
//
// Occupancy rides the cascade the way the streetlights do: a downtown avenue
// is a stream, an outskirts lane is three craft in a kilometre. That is
// `cc.density`, already loaded by the core — free.
//
// LIGHT. The Cloudpunk cue is the UNDERGLOW: an emissive belly panel, cyan /
// magenta / amber, which is the only part of a craft big enough to survive
// past a few hundred metres. From the street you read bellies overhead; from
// 400 m the avenues are strings of colored points above the sodium lattice;
// further out the layer is drifting specks. White nose and red tail lamps
// live in the hull's own shading rather than in geometry of their own, and
// dissolve into their face's mean as the footprint outgrows them.
//
// COST. Cheap gates first, in order: the segment's z-extent against the
// layer's [35, 140] m slab (no hash, no arithmetic beyond two multiply-adds);
// the pixel footprint; then the two avenue tests, which are integer. A cell
// with no avenue edge therefore spends exactly one hash draw — the free-flyer
// coin — and returns. At most three craft touch a cell (one x-lane, one
// y-lane, one free flyer; the two x-edge tests are mutually exclusive at
// avenue period 8), and each craft is two boxes, the second of which drops
// out once it is sub-pixel: <= 6 box tests and 6 pcg2d draws worst case,
// 2 pcg2d in the common one.

// --- craft geometry (a compact wedge, 3.8 x 1.9 x 1.1 m) --------------------
const cc_aircars_HALF_L: f32 = 1.90;
const cc_aircars_HALF_W: f32 = 0.95;
const cc_aircars_HULL_H: f32 = 0.70;
const cc_aircars_CANOPY_H: f32 = 0.40;
const cc_aircars_CANOPY_HALF_L: f32 = 1.10;
const cc_aircars_CANOPY_HALF_W: f32 = 0.60;
const cc_aircars_CANOPY_AFT: f32 = 0.45;  // canopy sits aft: the nose tapers

// --- lanes ------------------------------------------------------------------
const cc_aircars_LANE_OFF: f32 = 9.0;    // lane centerline in from the avenue
const cc_aircars_LANE_JIT: f32 = 4.0;    // +- lateral wander
const cc_aircars_ALONG_JIT: f32 = 34.0;  // +- along-lane wander in its slot
const cc_aircars_Z_LOW: f32 = 55.0;
const cc_aircars_Z_HIGH: f32 = 95.0;
const cc_aircars_Z_JIT: f32 = 8.0;
const cc_aircars_HIGH_CUT: f32 = 0.55;   // hash above this -> the high deck
const cc_aircars_LANE_P: f32 = 0.72;     // slot occupancy at full density
const cc_aircars_DENS_LO: f32 = 0.50;    // occupancy scale, outskirts
const cc_aircars_DENS_HI: f32 = 1.35;    // occupancy scale, downtown
// The window the occupancy ramp spans, in block density. Set against the
// tile's own distribution, which is lognormal and brutally skewed — median
// block 0.045, p75 0.10, p90 0.22 — so a ramp calibrated by eye on the
// megatower district leaves the MODAL city with one craft every 300 m of
// lane, which reads as scattered dust rather than as traffic. Anchored at
// the median instead: a typical avenue carries one craft per ~80 m counting
// both directions, downtown roughly one per 50, the outskirts one per 300.
const cc_aircars_DENS_START: f32 = 0.005;
const cc_aircars_DENS_FULL: f32 = 0.070;
// Free flyers: off-lattice singles over ordinary blocks.
const cc_aircars_FREE_FRAC: f32 = 0.07;
const cc_aircars_FREE_Z_LO: f32 = 40.0;
const cc_aircars_FREE_Z_HI: f32 = 130.0;
const cc_aircars_FREE_MARGIN: f32 = 12.0;
// Segment z-gate: the whole air layer lives in here, hull bottom to canopy.
const cc_aircars_Z_MIN: f32 = 35.0;
const cc_aircars_Z_MAX: f32 = 140.0;

// --- light ------------------------------------------------------------------
// Yardsticks from SPEC rule 5: lit window 3.5, storefront 2.2, lamp pool 0.7.
// The underglow sits between a lamp pool and a storefront — a lit panel, not
// a source you look into — and the nav lamps are point-bright.
const cc_aircars_GLOW_RAD: f32 = 2.5;
// The panel is inset in the belly, not the whole belly: seen from underneath
// at street range a craft has to read as a dark object carrying a light, not
// as a floating rectangle of light. The border keeps a quarter of the
// radiance (the panel spilling onto its own frame), which is also what keeps
// the far-field mean near the full value where the inset is sub-pixel.
const cc_aircars_PANEL_A: f32 = 1.62;    // panel half-length (of 1.90)
const cc_aircars_PANEL_W: f32 = 0.79;    // panel half-width (of 0.95)
const cc_aircars_PANEL_RIM: f32 = 0.25;  // what the border keeps
const cc_aircars_LAMP_RAD: f32 = 6.0;
const cc_aircars_LAMP_R: f32 = 0.12;     // nav-lamp dot radius (m)
const cc_aircars_CANOPY_RAD: f32 = 0.40;
const cc_aircars_HULL_FILL: f32 = 0.05;  // hull albedo against the skyglow
// The panel does not end at the belly seam: it wraps the lower flanks as a
// skirt and dies out toward the shoulder. This is what carries the layer at
// distance — from anywhere but directly underneath, the skirt is the only
// piece of a craft with the craft's own color on it, and from 400 m up a
// lane the strings of light you read are skirts, not bellies. Its mean over
// the 0.7 m flank is what a sub-pixel craft delivers, so it is set against
// the window yardstick (3.5) rather than against the belly's own radiance.
const cc_aircars_SKIRT: f32 = 0.86;
const cc_aircars_SKIRT_LO: f32 = 0.10;   // fully lit below this height (m)
const cc_aircars_SKIRT_HI: f32 = 0.62;   // gone by this one
// Dorsal running light: the same color, dimmer, on the hull's back. Without
// it a craft seen from above is a black chip on a bright street, and the
// whole aerial read of the layer goes with it.
const cc_aircars_DORSAL: f32 = 0.76;

// --- LOD --------------------------------------------------------------------
// Footprint (m/px) where each piece of detail hands over to its own mean. The
// canopy is a 2.2 m silhouette bump on a 3.8 m body: it stops paying for
// itself first. The nav lamps are 0.24 m dots: they fade into the mean
// radiance of the face they sit on, so a distant craft keeps its light and
// loses only the resolution — which is what a long lens does to traffic.
const cc_aircars_CANOPY_FP: f32 = 0.80;
const cc_aircars_LAMP_LOD_START: f32 = 0.35;
const cc_aircars_LAMP_LOD_FULL: f32 = 1.20;
// Where the craft stops being traced at all: 6 m/px puts a whole craft at
// two thirds of a pixel. The emission ramps to zero over the run-up to it
// (cc_aircars_FAR_FADE) so the geometry disappears at the moment it stops
// contributing anything, rather than popping out — the tracer's cutoff and
// the shader's mean meet at the same place. At the harness's 960 px / 65 deg
// this threshold sits past CITY_PROP_RANGE and never fires; it exists for
// low-resolution or very wide-angle frames, where it does.
const cc_aircars_FAR_FP: f32 = 6.0;
const cc_aircars_FAR_FADE: f32 = 3.6;    // emission gone by FAR_FP from here
// The core stops calling cell_props at CITY_PROP_RANGE. A hard population
// edge there would be a ring in any long sightline, so the emission tapers
// over the last quarter of the range instead: the far-field mean of a layer
// this sparse IS near zero, and this is that mean approached rather than
// stepped into.
// Kept short deliberately: a craft is about one pixel across out here and the
// population is sparse, so there is no continuous structure for an edge to
// show up in — all the taper has to do is take the last few craft down, and a
// long taper instead eats the whole mid-field of any shallow aerial view,
// which is exactly where the layer is supposed to read.
const cc_aircars_FADE_START: f32 = 0.92;
const cc_aircars_FADE_FULL: f32 = 1.00;

// Per-craft placement draw. One city_rand4 (two pcg2d) carries everything the
// geometry needs; appearance draws its own hash in the shader, where the cost
// is one per hit pixel rather than one per visited cell.
fn cc_aircars_draw(ci: vec2<i32>, lane: i32) -> vec4<f32> {
    let l = u32(lane);
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x9e3779b9u + l * 0x2545f491u + 0x165667b1u,
        bitcast<u32>(ci.y) * 0x85ebca6bu + l * 0xc2b2ae35u + 0x27d4eb2fu));
}

fn cc_aircars_look(ci: vec2<i32>, lane: i32) -> vec4<f32> {
    let l = u32(lane);
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0xc2b2ae35u + l * 0x9e3779b9u + 0x51ed270bu,
        bitcast<u32>(ci.y) * 0x27d4eb2fu + l * 0x85ebca6bu + 0xdeadbeefu));
}

// Underglow palette: 60% cyan, 25% magenta, 15% amber. Cyan carries the layer
// because it is the one hue the sodium-and-tungsten ground does not already
// own — the traffic reads as a separate system, not as more windows.
fn cc_aircars_glow_color(d: f32) -> vec3<f32> {
    if (d < 0.60) {
        return vec3<f32>(0.16, 0.90, 1.00);
    }
    if (d < 0.85) {
        return vec3<f32>(1.00, 0.20, 0.70);
    }
    return vec3<f32>(1.00, 0.60, 0.16);
}

struct cc_aircars_Craft {
    ok: bool,
    base: vec3<f32>,  // hull bottom, craft centered in xy
    axis: i32,        // 0 = travels along x, 1 = travels along y
    fwd: f32,         // +1 or -1 along that axis
}

fn cc_aircars_no_craft() -> cc_aircars_Craft {
    return cc_aircars_Craft(false, vec3<f32>(0.0), 0, 1.0);
}

fn cc_aircars_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// Where craft `lane` of cell `ci` is, if it exists at all.
//   lane 0 — the x-avenue lane (runs north/south)
//   lane 1 — the y-avenue lane (runs east/west)
//   lane 2 — the free flyer
// Deterministic in (ci, cc, lane) alone, so the shader re-derives a craft from
// its hit kind and the core's own CityCell — nothing has to be smuggled
// through CityHit.
fn cc_aircars_craft(ci: vec2<i32>, cc: CityCell, lane: i32)
        -> cc_aircars_Craft {
    let cell = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cell;

    if (lane == 2) {
        // Free flyers only over the built city; open lots keep empty sky.
        if (!cc.built) {
            return cc_aircars_no_craft();
        }
        let r = cc_aircars_draw(ci, lane);
        if (r.x >= cc_aircars_FREE_FRAC) {
            return cc_aircars_no_craft();
        }
        // r.x is uniform on [0, FREE_FRAC): rescaled it is a free draw, so a
        // heading costs no extra hash.
        let q = r.x / cc_aircars_FREE_FRAC;
        let span = cell - 2.0 * cc_aircars_FREE_MARGIN;
        return cc_aircars_Craft(
            true,
            vec3<f32>(cmin.x + cc_aircars_FREE_MARGIN + r.y * span,
                      cmin.y + cc_aircars_FREE_MARGIN + r.z * span,
                      mix(cc_aircars_FREE_Z_LO, cc_aircars_FREE_Z_HI, r.w)),
            select(0, 1, q > 0.5),
            select(-1.0, 1.0, fract(q * 2.0) > 0.5));
    }

    // Avenue lanes. Exactly one of the two edge tests can fire — the avenue
    // period is 8, so consecutive indices are never both avenues — which is
    // why one lane slot per axis per cell is the whole story.
    let k = select(ci.y, ci.x, lane == 0);
    let lo_edge = city_is_avenue(k);
    if (!lo_edge && !city_is_avenue(k + 1)) {
        return cc_aircars_no_craft();
    }
    let r = cc_aircars_draw(ci, lane);
    // Traffic clusters where the city does.
    let p = cc_aircars_LANE_P * mix(cc_aircars_DENS_LO, cc_aircars_DENS_HI,
                                    smoothstep(cc_aircars_DENS_START,
                                               cc_aircars_DENS_FULL,
                                               cc.density));
    if (r.x >= p) {
        return cc_aircars_no_craft();
    }

    // Lateral seat: LANE_OFF in from whichever edge is the avenue, jittered.
    let jit = (r.z - 0.5) * 2.0 * cc_aircars_LANE_JIT;
    let lat = select(cell - cc_aircars_LANE_OFF + jit,
                     cc_aircars_LANE_OFF + jit, lo_edge);
    let along = 0.5 * cell + (r.y - 0.5) * 2.0 * cc_aircars_ALONG_JIT;
    let z = select(cc_aircars_Z_LOW, cc_aircars_Z_HIGH,
                   r.w > cc_aircars_HIGH_CUT)
            + (fract(r.w * 37.0) - 0.5) * 2.0 * cc_aircars_Z_JIT;

    if (lane == 0) {
        // Runs along y. On the avenue's +x verge (this cell's min edge is the
        // avenue) right-hand traffic heads north.
        return cc_aircars_Craft(
            true, vec3<f32>(cmin.x + lat, cmin.y + along, z),
            1, select(-1.0, 1.0, lo_edge));
    }
    // Runs along x; the same rule rotated.
    return cc_aircars_Craft(
        true, vec3<f32>(cmin.x + along, cmin.y + lat, z),
        0, select(1.0, -1.0, lo_edge));
}

// Half-extents in xy for a body of the given along/across half-sizes.
fn cc_aircars_extent(axis: i32, hl: f32, hw: f32) -> vec2<f32> {
    return select(vec2<f32>(hw, hl), vec2<f32>(hl, hw), axis == 0);
}

// Nearest hit of one craft against [t0, t1]. The kind encodes both the lane
// (so the shader can find the craft again) and the part that was hit:
// 300 + 4 * lane + part, part 0 = hull side/top, 1 = the underglow panel on
// the belly, 2 = the canopy.
fn cc_aircars_hit_craft(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                        t0: f32, t1: f32, v: cc_aircars_Craft, lane: i32,
                        ci: vec2<i32>, fp: f32) -> CityHit {
    let hx = cc_aircars_extent(v.axis, cc_aircars_HALF_L, cc_aircars_HALF_W);
    let hmin = vec3<f32>(v.base.xy - hx, v.base.z);
    let hmax = vec3<f32>(v.base.xy + hx, v.base.z + cc_aircars_HULL_H);
    let sh = city_box_hit(o, inv_dir, hmin, hmax);
    let t_hull = select(1e30, sh.x,
                        sh.x <= sh.y && sh.x > t0 && sh.x <= t1);

    let aft = select(vec2<f32>(0.0, -v.fwd * cc_aircars_CANOPY_AFT),
                     vec2<f32>(-v.fwd * cc_aircars_CANOPY_AFT, 0.0),
                     v.axis == 0);
    let cx = cc_aircars_extent(v.axis, cc_aircars_CANOPY_HALF_L,
                               cc_aircars_CANOPY_HALF_W);
    let kmin = vec3<f32>(v.base.xy + aft - cx, v.base.z + cc_aircars_HULL_H);
    let kmax = vec3<f32>(v.base.xy + aft + cx,
                         v.base.z + cc_aircars_HULL_H + cc_aircars_CANOPY_H);
    // The composer's namespace check is line-based, so even this local wears
    // the prefix.
    var cc_aircars_t_canopy = 1e30;
    if (fp < cc_aircars_CANOPY_FP) {
        let sk = city_box_hit(o, inv_dir, kmin, kmax);
        if (sk.x <= sk.y && sk.x > t0 && sk.x <= t1) {
            cc_aircars_t_canopy = sk.x;
        }
    }

    let t = min(t_hull, cc_aircars_t_canopy);
    if (t >= 1e30) {
        return cc_aircars_miss(ci);
    }
    let is_canopy = cc_aircars_t_canopy < t_hull;
    let bmin = select(hmin, kmin, is_canopy);
    let bmax = select(hmax, kmax, is_canopy);
    let pos = o + t * dir;
    let nrm = city_box_normal(pos, bmin, bmax);
    // The belly of the hull is the underglow panel.
    let part = select(select(0, 1, nrm.z < -0.5), 2, is_canopy);
    return CityHit(true, t, pos, nrm, 300 + 4 * lane + part, ci);
}

fn cc_aircars_nearer(a: CityHit, b: CityHit) -> CityHit {
    if (b.hit && (!a.hit || b.t < a.t)) {
        return b;
    }
    return a;
}

// One lane's contribution: place, then test. Both halves early-out cheaply,
// so an absent lane costs an integer compare and an absent craft one hash.
fn cc_aircars_lane_hit(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell,
                       lane: i32, fp: f32) -> CityHit {
    let v = cc_aircars_craft(ci, cc, lane);
    if (!v.ok) {
        return cc_aircars_miss(ci);
    }
    return cc_aircars_hit_craft(o, dir, inv_dir, t0, t1, v, lane, ci, fp);
}

fn cc_aircars_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                          t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    // Gate 1, no hash: does this segment pass through the air layer at all?
    // Most city pixels are looking at a facade or at the road.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_aircars_Z_MAX || max(za, zb) < cc_aircars_Z_MIN) {
        return cc_aircars_miss(ci);
    }
    // Gate 2: a craft smaller than this is not worth a box test.
    let fp = (2.0 * u.cam_origin.w / max(u.params.x, 1.0)) * max(t0, 0.0);
    if (fp > cc_aircars_FAR_FP) {
        return cc_aircars_miss(ci);
    }
    let h0 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 0, fp);
    let h1 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 1, fp);
    let h2 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 2, fp);
    return cc_aircars_nearer(cc_aircars_nearer(h0, h1), h2);
}

// One nav lamp. `a` and `b` are the hit's coordinates in the face's own plane
// relative to the craft, `seat_*` the lamp's seat in the same frame, and
// `span_*` the face's size. Resolved while the dot is bigger than a pixel,
// handed to the face's mean when it is not: dot area over face area is the
// honest sub-pixel value, so the craft keeps its light and loses only the
// resolution.
fn cc_aircars_lamp(a: f32, b: f32, seat_a: f32, seat_b: f32,
                   span_a: f32, span_b: f32, col: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let d = length(vec2<f32>(a - seat_a, b - seat_b));
    let sharp = select(0.0, 1.0, d < cc_aircars_LAMP_R);
    let area = 3.14159265 * cc_aircars_LAMP_R * cc_aircars_LAMP_R;
    let mean = area / max(span_a * span_b, 1e-3);
    let k = smoothstep(cc_aircars_LAMP_LOD_START, cc_aircars_LAMP_LOD_FULL,
                       fp);
    return col * (cc_aircars_LAMP_RAD * mix(sharp, mean, k));
}

// White forward, red aft, on whichever face is looking at you. A craft
// crossing your view still declares which way it is going, because the flank
// carries the same pair near its ends.
fn cc_aircars_navlights(h: CityHit, v: cc_aircars_Craft, fp: f32)
        -> vec3<f32> {
    let rel = h.pos - v.base;
    let a = select(rel.y * v.fwd, rel.x * v.fwd, v.axis == 0);
    let across = select(rel.x, rel.y, v.axis == 0);
    let b = rel.z;
    let seat_b = 0.60 * cc_aircars_HULL_H;
    let white = vec3<f32>(1.00, 0.96, 0.88);
    let red = vec3<f32>(1.00, 0.13, 0.06);
    let n_along = select(h.normal.y * v.fwd, h.normal.x * v.fwd, v.axis == 0);
    let n_across = select(h.normal.x, h.normal.y, v.axis == 0);
    let face_w = 2.0 * cc_aircars_HALF_W;
    let face_l = 2.0 * cc_aircars_HALF_L;

    if (n_along > 0.5) {
        return cc_aircars_lamp(across, b, -0.52, seat_b, face_w,
                               cc_aircars_HULL_H, white, fp)
             + cc_aircars_lamp(across, b, 0.52, seat_b, face_w,
                               cc_aircars_HULL_H, white, fp);
    }
    if (n_along < -0.5) {
        return cc_aircars_lamp(across, b, -0.52, seat_b, face_w,
                               cc_aircars_HULL_H, red, fp)
             + cc_aircars_lamp(across, b, 0.52, seat_b, face_w,
                               cc_aircars_HULL_H, red, fp);
    }
    if (abs(n_across) > 0.5) {
        return cc_aircars_lamp(a, b, 1.55, seat_b, face_l,
                               cc_aircars_HULL_H, white, fp)
             + cc_aircars_lamp(a, b, -1.55, seat_b, face_l,
                               cc_aircars_HULL_H, red, fp);
    }
    return vec3<f32>(0.0);
}

fn cc_aircars_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let code = h.kind - 300;
    let lane = code / 4;
    let part = code % 4;
    let v = cc_aircars_craft(h.cell, cc, lane);
    let a4 = cc_aircars_look(h.cell, lane);
    let glow = cc_aircars_glow_color(a4.x);
    let bright = 0.70 + 0.70 * a4.y;

    // The two edges of the layer, each approached rather than stepped: the
    // population edge at CITY_PROP_RANGE, and the footprint at which the
    // tracer stops testing craft at all.
    let fade = (1.0 - smoothstep(cc_aircars_FADE_START * CITY_PROP_RANGE,
                                 cc_aircars_FADE_FULL * CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_aircars_FAR_FADE, cc_aircars_FAR_FP, fp));

    // The night fill the core gives every city surface. A hull is dark metal
    // over a dark city: essentially a silhouette with a lit edge.
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(h.pos.xy, 3.0));
    let body = cc_aircars_HULL_FILL * fill;

    if (part == 1) {   // the underglow panel: the whole point of the layer
        let rel = h.pos - v.base;
        let a = select(rel.y, rel.x, v.axis == 0);
        let across = select(rel.x, rel.y, v.axis == 0);
        let panel =
            (1.0 - smoothstep(cc_aircars_PANEL_A, cc_aircars_HALF_L, abs(a)))
          * (1.0 - smoothstep(cc_aircars_PANEL_W, cc_aircars_HALF_W,
                              abs(across)));
        return body + glow * (cc_aircars_GLOW_RAD * bright * fade
                              * mix(cc_aircars_PANEL_RIM, 1.0, panel));
    }
    if (part == 2) {   // canopy: a warm interior seen through the glass
        // Its BACK is dorsal like the hull's, and that matters more than it
        // sounds: the canopy covers the middle of the top, so from directly
        // above it is most of what a craft shows. Lit warm, the whole aerial
        // read of the layer went with it. This also makes the LOD hand-off
        // exact — when the canopy box drops out the hull top behind it is
        // already the same radiance.
        if (h.normal.z > 0.5) {
            return body + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL
                                  * bright * fade);
        }
        return body + vec3<f32>(1.00, 0.74, 0.44)
                      * (cc_aircars_CANOPY_RAD * (0.6 + 0.8 * a4.z) * fade);
    }
    // Hull. The back carries a dorsal running light; the flanks carry the
    // skirt of the belly panel, e-folding upward from the seam. Nav lamps are
    // drawn on rather than built.
    if (h.normal.z > 0.5) {
        return body + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL
                              * bright * fade);
    }
    let up = max(h.pos.z - v.base.z, 0.0);
    let wrap = 1.0 - smoothstep(cc_aircars_SKIRT_LO, cc_aircars_SKIRT_HI, up);
    let skirt = glow * (cc_aircars_GLOW_RAD * cc_aircars_SKIRT * bright
                        * wrap);
    return body + (skirt + cc_aircars_navlights(h, v, fp)) * fade;
}
