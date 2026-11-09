// industry — the industrial belt: warehouse roofs, the floodlights, and what
// stands in their light.
//
// Where the cascade thins, the city stops being towers (Thomas: the regions
// with no buildings "would be sort of industrial regions with things like
// forklifts/large warehouses/empty lots with construction refuse"). The core
// already lays that belt out: INDUSTRIAL lots carry one low warehouse box
// pushed against one long side of the plot, the other side being its truck
// court; YARD lots front a street and carry no building; OPEN land is scrub.
// It also already decides where the belt's few lamps are (city_flood) and
// paints their pools on the ground. This component owns everything above
// that ground except the warehouse WALLS (city-facades: cladding, doors,
// docks) and the streets (city-street):
//
//   1. ROOFS. Every warehouse draws one of three: SAWTOOTH (40%) — teeth 8-12
//      m apart and 2.5-3.5 m tall, each a sloped sheet with a vertical glazed
//      face, all facing one way; MONITOR (30%) — a raised clerestory strip
//      down the long axis, glazed along both sides; FLAT (30%) — the core's
//      own roof, carrying rows of translucent skylights. About a third of the
//      buildings are working a night shift, and on those the glazing glows
//      from the high-bays inside: the belt's signature from the air, stripes
//      of cool or amber light ruled across dark roofs.
//   2. FLOOD FIXTURES. The lamp at every city_flood head — a wall-pack on a
//      bracket over the court, or a head on a 10 m pole at a yard corner —
//      with a lens bright enough (~30) to read as the source of its pool,
//      and, through the facade hook, the scallop of light each wall-pack
//      throws down the court wall below it.
//   3. PROPS on courts and yards: semi-trailers backed onto the docks,
//      stacked shipping containers, pallet stacks, rebar bundles, a skip
//      with refuse in it, a forklift, debris mounds, and a chain-link fence
//      (posts and rails) around most yards.
//
// ONE PRIMITIVE CARRIES THE PROPS: a GRID of slots. A container block, a row
// of trailers at a dock, a pallet stack field, a line of rebar bundles, a
// row of skylights, and a single skip or forklift (a 1x1 grid) are all the
// same thing — one bounding box, a 2D DDA over the slots the ray crosses,
// and a per-slot hash that says whether the slot is filled and how high.
// One routine, one call site, so the per-object detail (a trailer's
// undercarriage, a forklift's mast and guard, a skip's cavity) is inlined
// exactly once however many grids a lot has.
//
// LOTS SPAN CELLS. A warehouse lot is 2x1 or 2x2 blocks and a yard may be
// several, but cell_props is called per DDA cell. Everything here is laid
// out per LOT — a pure function of the CityCell every member cell decodes
// identically — and every member cell traces all of it, accepting only hits
// inside its own ray segment [t0, t1]. The segment IS the cell's column, so
// a container straddling two cells is found by whichever cell the ray is
// in when it meets the steel, and nothing need be placed inside a cell.
//
// LIGHT. The belt is dark on purpose, a foil for the neon core. Props are
// lit by the lot's floods — the pool's own Gaussian at the prop, times the
// surface's facing toward the fixture over the ground's — by the street
// lamps where a yard fronts one, and by the core's skyglow fill and moon.
// A yard with no lit fixture stays nearly black. No shadows: the pools are
// Gaussians painted on the ground, and a shadow they do not have would be
// a lie in the other direction.
//
// DETAIL CALIBRATION (SPEC): wiper blades, not tire brands. A container has
// its corrugation, corner posts, rails and door-end locking bars; a trailer
// its bogie, landing legs and rub rail; a forklift its mast, forks and
// overhead guard. Nothing has a logo.
//
// LOD keys on the pixel footprint fp (m/px), never on distance: shading
// bands dissolve into their means as they go sub-pixel; the fence, pallets,
// rebar and forklift stop being traced past FINE_FP (dark, sub-pixel, and
// the ground under them is the same near-black); containers, trailers and
// the roofs are traced to CITY_PROP_RANGE; and the roof hook carries the
// glazing's exact, view-dependent mean beyond that, so no roof ever goes
// out. See cc_industry_roof for that derivation.

// --- kinds (kind_base 700, local 0..15) -------------------------------------
const cc_industry_K_ROOF: i32 = 700;     // sheet roofing, gable ends
const cc_industry_K_GLAZE: i32 = 701;    // sawtooth and clerestory glazing
const cc_industry_K_SKY: i32 = 702;      // flat-roof skylights
const cc_industry_K_HEAD: i32 = 703;     // flood lamp head
const cc_industry_K_STEEL: i32 = 704;    // pole, wall bracket
const cc_industry_K_BOX: i32 = 705;      // shipping container
const cc_industry_K_TRAILER: i32 = 706;
const cc_industry_K_PALLET: i32 = 707;
const cc_industry_K_REBAR: i32 = 708;
const cc_industry_K_SKIP: i32 = 709;
const cc_industry_K_REFUSE: i32 = 710;   // the fill inside a skip
const cc_industry_K_PILE: i32 = 711;     // debris mound
const cc_industry_K_FORK: i32 = 712;     // forklift
const cc_industry_K_FENCE: i32 = 713;
const cc_industry_K_DOCK: i32 = 714;     // dock shelter, bumpers, leveller

// --- roofs ------------------------------------------------------------------
const cc_industry_SAW_FRAC: f32 = 0.40;
const cc_industry_MONITOR_FRAC: f32 = 0.30;   // after the sawtooth draw
const cc_industry_ROOF_TOP: f32 = 3.6;        // tallest roof feature above h
// A working night shift: the glazing is lit from the high-bays inside. Dim
// against a lit window (3.5) — this is light seen through a dirty wired
// pane from across a factory floor — but a whole roof of it.
const cc_industry_NIGHT_SHIFT: f32 = 0.38;
const cc_industry_GLAZE_RAD: f32 = 0.08;
// The high-bays behind the glass: a luminaire every LAMP_P along the ridge,
// each a soft hot spot on the pane over a floor glow. Uniform glass at this
// level read as grey sheet material, not as light.
const cc_industry_LAMP_P: f32 = 6.0;
const cc_industry_LAMP_SIG: f32 = 1.1;
const cc_industry_LAMP_FLOOR: f32 = 0.15;
// Its mean along the ridge, the Gaussian cut at +-P/2 by the next lamp's
// cell: FLOOR + (1 - FLOOR) sqrt(2 pi) SIG / P erf(P / (2 sqrt(2) SIG)).
const cc_industry_LAMP_MEAN: f32 = 0.5381;
const cc_industry_BAY: f32 = 12.0;            // lighting zone along a ridge
const cc_industry_BAY_LIT: f32 = 0.65;
const cc_industry_MULL: f32 = 1.8;            // mullion pitch (m)
const cc_industry_MULL_W: f32 = 0.14;
const cc_industry_TRANSOM_W: f32 = 0.10;      // the bar across mid-pane
const cc_industry_SILL: f32 = 0.35;           // flashing at a pane's foot
const cc_industry_HEAD_BAND: f32 = 0.22;      // and its head
// Glazing's mean emission over a face, as a fraction of GLAZE_RAD: lit bays
// x mean bay level x the glass fraction of the framed face. Kept as one
// function (cc_industry_glaze_mean) because the shader's far field and the
// roof hook must agree on it exactly.
const cc_industry_BAY_MEAN: f32 = 0.9;        // E[0.4 + u] for u ~ U(0,1)
// Tone-map compensation on the far-field mean, the correction the core
// applies to its own window octaves: a mean taken in linear radiance ahead
// of the Reinhard curve renders brighter than the pattern it replaces.
const cc_industry_GLAZE_MEAN_COMP: f32 = 0.55;
const cc_industry_GLAZE_LOD_START: f32 = 1.5;
const cc_industry_GLAZE_LOD_FULL: f32 = 5.0;
// Sheet roofing: standing seams every 0.6 m, sub-pixel past a few cm/px.
const cc_industry_SEAM: f32 = 0.6;
const cc_industry_ROOF_ALB: f32 = 0.10;
const cc_industry_CLAD_ALB: f32 = 0.045;
// Skylights: translucent panels in rows down a flat roof, glowing when the
// shift is on. Past SKY_FP a row of them is sub-pixel relief, and the grid
// is traced as one flat panel carrying their mean.
const cc_industry_SKY_W: f32 = 1.4;
const cc_industry_SKY_L: f32 = 3.0;
const cc_industry_SKY_PITCH: f32 = 6.0;
const cc_industry_SKY_ROW: f32 = 10.0;
const cc_industry_SKY_H: f32 = 0.32;
const cc_industry_SKY_RAD: f32 = 0.35;
const cc_industry_SKY_FP: f32 = 0.45;

// --- flood fixtures ---------------------------------------------------------
// The lens radiance: a sodium or LED flood seen in its beam is a point of
// glare like a beacon (40), not a glowing panel.
const cc_industry_LENS_RAD: f32 = 30.0;
const cc_industry_HEAD_HW_WALL: vec3<f32> = vec3<f32>(0.22, 0.32, 0.14);
const cc_industry_HEAD_HW_POLE: vec3<f32> = vec3<f32>(0.26, 0.40, 0.16);
const cc_industry_POLE_HW: f32 = 0.11;
const cc_industry_POLE_REACH: f32 = 0.35;     // head overhang past the pole
// Energy-preserving widening: past a few metres per pixel a 0.6 m head is a
// fraction of a pixel, so its box grows to a pixel and its emission drops by
// the same area ratio — a far lamp is a steady dim point, never a strobe.
const cc_industry_HEAD_WIDEN: f32 = 0.45;
const cc_industry_HEAD_WIDEN_MAX: f32 = 1.0;
// Wall wash: the scallop a wall-pack throws down the wall under it. Narrow
// at the fixture, widening with depth; rises over the first half metre (the
// housing shades the wall right behind it) and fades over WASH_FADE.
const cc_industry_WASH_GAIN: f32 = 0.6;
const cc_industry_WASH_W0: f32 = 0.7;
const cc_industry_WASH_W1: f32 = 0.45;
const cc_industry_WASH_RISE: f32 = 0.5;
const cc_industry_WASH_FADE: f32 = 5.0;

// --- props ------------------------------------------------------------------
const cc_industry_PROP_TOP: f32 = 7.9;        // three containers high
// A yard's tallest thing: a pole-top flood at its widest.
const cc_industry_YARD_TOP: f32 = CITY_FLOOD_POLE_H + cc_industry_HEAD_WIDEN_MAX
                                  + 0.1;
const cc_industry_EDGE: f32 = 1.5;            // props stay this far inside
const cc_industry_TIER: f32 = 2.59;           // ISO container height
const cc_industry_BOX_L: f32 = 12.19;         // 40 ft
const cc_industry_BOX_W: f32 = 2.44;
const cc_industry_TRL_L: f32 = 13.6;
const cc_industry_TRL_W: f32 = 2.5;
const cc_industry_TRL_H: f32 = 4.0;
const cc_industry_TRL_DECK: f32 = 1.22;       // underside of the box body
const cc_industry_DOCK_PITCH: f32 = 4.2;      // dock door centres (m)
const cc_industry_DOCK_GAP: f32 = 0.25;       // trailer rear off the wall
// The dock doors: sectional doors DOOR_W x DOOR_H standing on the dock
// floor (DOCK_H above the court), one per dock slot; a black shelter frame
// around each on some buildings, bumpers and a leveller lip on all. Relief
// stays inside SHELL_OUT of the wall. (Not mirrored by the flight collider,
// which meets a warehouse at its wall: web/soar/cityprops.js.)
const cc_industry_DOCK_H: f32 = 1.2;
const cc_industry_DOOR_W: f32 = 3.0;
const cc_industry_DOOR_H: f32 = 3.0;
const cc_industry_SHELL_OUT: f32 = 0.60;
const cc_industry_PALLET_H: f32 = 0.144;
const cc_industry_SKIP_L: f32 = 5.8;
const cc_industry_SKIP_W: f32 = 2.3;
const cc_industry_SKIP_H: f32 = 1.7;
const cc_industry_SKIP_WALL: f32 = 0.10;
const cc_industry_FORK_H: f32 = 2.3;
const cc_industry_REBAR_H: f32 = 0.42;
const cc_industry_FENCE_H: f32 = 2.4;
const cc_industry_FENCE_IN: f32 = 0.8;        // fence line inside the plot
const cc_industry_FENCE_POST: f32 = 3.0;
const cc_industry_FENCE_FRAC: f32 = 0.75;
const cc_industry_GATE_W: f32 = 9.0;
// Past this footprint the small, dark props (fence, pallets, rebar, the
// forklift) are under a pixel and near-black on near-black ground; they stop
// being traced. Containers, trailers and the skip keep going: they are big,
// and they catch the floods.
const cc_industry_FINE_FP: f32 = 0.60;
// Detail inside a trailer's or a forklift's envelope, below this footprint.
const cc_industry_PART_FP: f32 = 0.25;
// A surface's response to the flood pools is measured against the ground
// the pools are painted on: a prop of this albedo facing the lamp the way
// the ground does returns the pool's own radiance.
const cc_industry_ALB_REF: f32 = 0.35;
// Where lamp light reflected off a prop starts to roll off: exposure 6 puts
// 0.3 at 1.8, below the Reinhard curve's shoulder.
const cc_industry_LAMP_KNEE: f32 = 0.3;
// Weathering's means over a face (tools: a brute-force average of the
// patterns in cc_industry_weather and the container roof, 4e5 samples).
const cc_industry_WEATHER_MEAN: vec3<f32> = vec3<f32>(0.962, 0.960, 0.959);
const cc_industry_ROOF_MEAN: vec3<f32> = vec3<f32>(0.769, 0.745, 0.728);

// --- the lot's frame ----------------------------------------------------------
// Everything is laid out along the lot's long axis `a` and across it `c`, and
// swizzled to world xy at the end, so one layout serves both orientations.
struct cc_industry_Lot {
    long_x: bool,
    pa: vec2<f32>,    // plot extent along the long axis
    pc: vec2<f32>,    // ... across it
    ba: vec2<f32>,    // warehouse extent along (industrial lots)
    bc: vec2<f32>,    // ... across
    court_hi: bool,   // the truck court is on the +c side of the building
}

fn cc_industry_lot(cc: CityCell) -> cc_industry_Lot {
    var l: cc_industry_Lot;
    let bs = cc.b1max.xy - cc.b1min.xy;
    let ps = cc.plot_max - cc.plot_min;
    // The warehouse's own long axis, by city_flood's rule, so the court the
    // props fill is the court the lamps light.
    l.long_x = select(ps.x >= ps.y, bs.x >= bs.y,
                      cc.land_use == CITY_USE_INDUSTRIAL);
    l.pa = select(vec2<f32>(cc.plot_min.y, cc.plot_max.y),
                  vec2<f32>(cc.plot_min.x, cc.plot_max.x), l.long_x);
    l.pc = select(vec2<f32>(cc.plot_min.x, cc.plot_max.x),
                  vec2<f32>(cc.plot_min.y, cc.plot_max.y), l.long_x);
    l.ba = select(vec2<f32>(cc.b1min.y, cc.b1max.y),
                  vec2<f32>(cc.b1min.x, cc.b1max.x), l.long_x);
    l.bc = select(vec2<f32>(cc.b1min.x, cc.b1max.x),
                  vec2<f32>(cc.b1min.y, cc.b1max.y), l.long_x);
    l.court_hi = (l.pc.y - l.bc.y) >= (l.bc.x - l.pc.x);
    return l;
}

// (a, c) -> world xy. A swizzle, so it preserves min/max order.
fn cc_industry_xy(l: cc_industry_Lot, ac: vec2<f32>) -> vec2<f32> {
    return select(ac.yx, ac, l.long_x);
}

// The dock doors along a warehouse's court wall: (centre of door 0 along
// the lot's long axis, count). Trailers, the forklift, the dock hardware and
// the doors painted on the wall all take their slots from here.
fn cc_industry_docks(l: cc_industry_Lot) -> vec2<f32> {
    let len = l.ba.y - l.ba.x - 8.0;
    let n = floor(len / cc_industry_DOCK_PITCH);
    return vec2<f32>(l.ba.x + 4.0 + 0.5 * (len - n * cc_industry_DOCK_PITCH)
                     + 0.5 * cc_industry_DOCK_PITCH, n);
}

// The truck court's depth, dock wall to the plot edge less the prop margin,
// and whether trailers fit in it.
fn cc_industry_court_depth(l: cc_industry_Lot) -> f32 {
    return select(l.bc.x - l.pc.x, l.pc.y - l.bc.y, l.court_hi)
           - cc_industry_EDGE;
}

fn cc_industry_has_trailers(l: cc_industry_Lot) -> bool {
    return cc_industry_court_depth(l)
           >= cc_industry_TRL_L + 2.0 * cc_industry_DOCK_GAP + 2.0;
}

// A lot-level draw: the same in every member cell, salted per use.
fn cc_industry_draw(cc: CityCell, salt: u32, i: i32, j: i32) -> vec4<f32> {
    return city_rand4(vec2<u32>(
        cc.seed.x ^ (salt * 0x9e3779b9u) ^ (bitcast<u32>(i) * 0x85ebca6bu),
        cc.seed.y ^ (salt * 0x51ed270bu) ^ (bitcast<u32>(j) * 0xc2b2ae35u)));
}

fn cc_industry_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// Nearest-hit accumulator over one box, with the segment discipline: a hit
// outside [t0, t1] belongs to another cell's visit.
fn cc_industry_box(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
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

// The pixel's angular size: the footprint at distance t is px * t (m/px),
// the core's own pixel_angle.
fn cc_industry_px() -> f32 {
    return lod_pixel_angle();
}

// ============================================================================
// ROOFS
// ============================================================================

struct cc_industry_Roof {
    kind: i32,          // 0 flat (+ skylights), 1 sawtooth, 2 monitor
    step_x: bool,       // sawtooth: the teeth step along world x
    face: f32,          // +1: glazing faces +step axis; -1: faces -step
    s0: f32,            // step-axis coordinate where tooth 0 starts (face +1)
    pitch: f32,
    n: i32,             // teeth
    rise: f32,
    mlo: vec3<f32>,     // monitor strip box
    mhi: vec3<f32>,
    lit: bool,
    tint: vec3<f32>,    // the high-bays' colour
}

fn cc_industry_roof_of(cc: CityCell) -> cc_industry_Roof {
    var r: cc_industry_Roof;
    let d = cc_industry_draw(cc, 11u, 0, 0);
    let e = cc_industry_draw(cc, 12u, 0, 0);
    let l = cc_industry_lot(cc);
    r.kind = 0;
    if (d.x < cc_industry_SAW_FRAC) {
        r.kind = 1;
    } else if (d.x < cc_industry_SAW_FRAC + cc_industry_MONITOR_FRAC) {
        r.kind = 2;
    }
    // Most sawtooth sheds step along their long axis — the classic many-
    // toothed profile down a long factory — some across it.
    r.step_x = select(!l.long_x, l.long_x, d.y < 0.70);
    let a0 = select(cc.b1min.y, cc.b1min.x, r.step_x);
    let a1 = select(cc.b1max.y, cc.b1max.x, r.step_x);
    r.n = max(i32(round((a1 - a0) / (8.0 + 4.0 * d.z))), 1);
    r.pitch = (a1 - a0) / f32(r.n);
    r.rise = 2.5 + d.w;
    r.face = select(-1.0, 1.0, e.x < 0.5);
    r.s0 = select(a1, a0, r.face > 0.0);
    // The clerestory: a quarter to a third of the span, centred on the ridge.
    let w = (l.bc.y - l.bc.x) * (0.24 + 0.10 * e.y);
    let cm = 0.5 * (l.bc.x + l.bc.y);
    let lo = cc_industry_xy(l, vec2<f32>(l.ba.x + 3.0, cm - 0.5 * w));
    let hi = cc_industry_xy(l, vec2<f32>(l.ba.y - 3.0, cm + 0.5 * w));
    r.mlo = vec3<f32>(lo, cc.height);
    r.mhi = vec3<f32>(hi, cc.height + 2.6 + 0.9 * e.z);
    r.lit = e.w < cc_industry_NIGHT_SHIFT;
    let tc = fract(e.w * 17.3);
    r.tint = select(select(vec3<f32>(1.00, 0.52, 0.18),     // sodium high-bay
                           vec3<f32>(1.00, 0.84, 0.58), tc < 0.70),  // halide
                    vec3<f32>(0.72, 0.85, 1.00), tc < 0.30);          // LED
    return r;
}

// The skylight grid on a flat roof: rows down the long axis.
struct cc_industry_Grid {
    ok: bool,
    kind: i32,
    lo: vec3<f32>,      // min corner; z is the base the objects stand on
    cell: vec2<f32>,    // slot pitch in world x, y
    n: vec2<i32>,       // slots in x, y
    obj: vec2<f32>,     // object footprint in its slot (centred), world x, y
    zmax: f32,          // tallest object above lo.z
    occ: f32,           // fraction of slots filled
    tiers: i32,         // containers: most stacked; docks: 1 with shelters
    front: i32,         // object's forward axis: 0 +x, 1 -x, 2 +y, 3 -y
    salt: u32,
}

fn cc_industry_no_grid() -> cc_industry_Grid {
    var g: cc_industry_Grid;
    g.ok = false;
    g.kind = 0;
    g.lo = vec3<f32>(0.0);
    g.cell = vec2<f32>(1.0);
    g.n = vec2<i32>(0);
    g.obj = vec2<f32>(0.0);
    g.zmax = 0.0;
    g.occ = 0.0;
    g.tiers = 1;
    g.front = 0;
    g.salt = 0u;
    return g;
}

// A grid from lot-frame (a, c) quantities. `front_ac` names the object's
// forward axis in the lot frame (0 +a, 1 -a, 2 +c, 3 -c).
fn cc_industry_mk_grid(l: cc_industry_Lot, kind: i32, lo_ac: vec2<f32>,
                       cell_ac: vec2<f32>, n_ac: vec2<i32>, obj_ac: vec2<f32>,
                       z0: f32, zmax: f32, occ: f32, tiers: i32,
                       front_ac: i32, salt: u32) -> cc_industry_Grid {
    var g: cc_industry_Grid;
    g.ok = n_ac.x > 0 && n_ac.y > 0;
    g.kind = kind;
    g.lo = vec3<f32>(cc_industry_xy(l, lo_ac), z0);
    g.cell = cc_industry_xy(l, cell_ac);
    g.n = select(n_ac.yx, n_ac, l.long_x);
    g.obj = cc_industry_xy(l, obj_ac);
    g.zmax = zmax;
    g.occ = occ;
    g.tiers = tiers;
    // a -> x and c -> y when the lot runs along x; a -> y and c -> x if not.
    var f = front_ac;
    if (!l.long_x) {
        f = select(select(select(1, 0, front_ac == 2), 2, front_ac == 0),
                   3, front_ac == 1);
    }
    g.front = f;
    g.salt = salt;
    return g;
}

fn cc_industry_sky_grid(cc: CityCell) -> cc_industry_Grid {
    let l = cc_industry_lot(cc);
    let span_a = l.ba.y - l.ba.x - 8.0;
    let span_c = l.bc.y - l.bc.x - 8.0;
    let n_a = i32(floor(span_a / cc_industry_SKY_PITCH));
    let n_c = i32(floor(span_c / cc_industry_SKY_ROW));
    let lo = vec2<f32>(
        l.ba.x + 4.0 + 0.5 * (span_a - f32(n_a) * cc_industry_SKY_PITCH),
        l.bc.x + 4.0 + 0.5 * (span_c - f32(n_c) * cc_industry_SKY_ROW));
    return cc_industry_mk_grid(
        l, cc_industry_K_SKY, lo,
        vec2<f32>(cc_industry_SKY_PITCH, cc_industry_SKY_ROW),
        vec2<i32>(n_a, n_c),
        vec2<f32>(cc_industry_SKY_L, cc_industry_SKY_W),
        cc.height, cc_industry_SKY_H, 1.0, 1, 0, 13u);
}

// Sawtooth: a heightfield along the step axis s, z = h + rise * frac(s / p),
// with a vertical glazed face at every tooth boundary. Walked tooth by tooth
// like a 1D DDA: within a tooth the gap g(t) between the ray and the roof
// plane is linear in t, so the hit is where it changes sign; crossing a
// boundary against the glazing's facing with the ray below the tooth top is
// a hit on the glass. Exact, and a handful of teeth per cell at most.
fn cc_industry_saw_trace(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                         inv_dir: vec3<f32>, t0: f32, t1: f32, cc: CityCell,
                         r: cc_industry_Roof, ci: vec2<i32>) -> CityHit {
    let h = cc.height;
    let bmin = vec3<f32>(cc.b1min.xy, h);
    let bmax = vec3<f32>(cc.b1max.xy, h + r.rise);
    let sb = city_box_hit(o, inv_dir, bmin, bmax);
    let ta = max(max(sb.x, t0), 1e-3);
    let tb = min(min(sb.y, t1), best.t);
    if (sb.x > sb.y || ta >= tb) {
        return best;
    }
    // s increases along the glazing's facing, from 0 at tooth 0's low edge.
    let o_a = select(o.y, o.x, r.step_x);
    let d_a = select(dir.y, dir.x, r.step_x);
    let s_o = r.face * (o_a - r.s0);
    let d_s = r.face * d_a;
    let p = r.pitch;
    let slope = r.rise / p;
    var k = clamp(i32(floor((s_o + d_s * ta) / p)), 0, r.n - 1);
    var t = ta;
    // The walk's bound is a runtime count (a literal one gets unrolled into
    // the core's DDA loop, see streetlife's ITERS note).
    for (var i: i32 = 0; i <= r.n; i = i + 1) {
        var t_next = 1e30;
        if (d_s > 1e-7) {
            t_next = (f32(k + 1) * p - s_o) / d_s;
        } else if (d_s < -1e-7) {
            t_next = (f32(k) * p - s_o) / d_s;
        }
        let t_out = min(t_next, tb);
        let base = f32(k) * p;
        let g_in = (o.z + dir.z * t - h) - slope * (s_o + d_s * t - base);
        let g_out = (o.z + dir.z * t_out - h)
                    - slope * (s_o + d_s * t_out - base);
        if (g_in <= 0.0) {
            // Under the roof at a tooth's entry: the face just crossed.
            // Inside the walk that is always a glazed face (the ray came
            // down-slope, against the facing); at the walk's start it is the
            // box face the ray entered by, if it entered by one rather than
            // by the cell's column boundary.
            if (i > 0 || sb.x > max(t0, 1e-3)) {
                let q = o + t * dir;
                var n = city_box_normal(q, bmin, bmax);
                var kind = cc_industry_K_ROOF;
                if (i > 0 || abs(select(n.y, n.x, r.step_x)) > 0.5) {
                    n = select(vec3<f32>(0.0, r.face, 0.0),
                               vec3<f32>(r.face, 0.0, 0.0), r.step_x);
                    kind = cc_industry_K_GLAZE;
                }
                return CityHit(true, t, q, n, kind, ci);
            }
            return best;
        }
        if (g_out <= 0.0) {
            let th = t + (t_out - t) * g_in / (g_in - g_out);
            let ns = normalize(vec2<f32>(-slope * r.face, 1.0));
            let n = select(vec3<f32>(0.0, ns.x, ns.y),
                           vec3<f32>(ns.x, 0.0, ns.y), r.step_x);
            return CityHit(true, th, o + th * dir, n, cc_industry_K_ROOF, ci);
        }
        if (t_out >= tb) {
            break;
        }
        t = t_out;
        k = k + select(-1, 1, d_s > 0.0);
        if (k < 0 || k >= r.n) {
            break;
        }
    }
    return best;
}

// ============================================================================
// FLOOD FIXTURES
// ============================================================================

// The head box of fixture f: a wall-pack sits on f.head, 0.8 m off the wall;
// a pole head overhangs its pole toward the aim. Half extents are (along the
// aim, across it, vertical), widened toward a pixel as the footprint grows —
// the footprint AT THE HEAD, from the camera's distance to it, which is what
// both the trace and the shader compute, so the box a ray met and the
// dilution it is shaded with always agree. The widening stops at
// HEAD_WIDEN_MAX, which keeps every head inside its own lot and inside the
// trace's fixed z gates; past that (fp ~2.2 m/px) a lamp is a fixed-size
// spark, as a real one is.
fn cc_industry_head_c(cc: CityCell, f: CityFlood) -> vec3<f32> {
    if (cc.land_use == CITY_USE_YARD) {
        return f.head + vec3<f32>(f.aim * cc_industry_POLE_REACH, 0.0);
    }
    return f.head;
}

fn cc_industry_head_real(cc: CityCell) -> vec3<f32> {
    return select(cc_industry_HEAD_HW_WALL, cc_industry_HEAD_HW_POLE,
                  cc.land_use == CITY_USE_YARD);
}

fn cc_industry_head_hw(cc: CityCell, fp: f32) -> vec3<f32> {
    return max(cc_industry_head_real(cc),
               vec3<f32>(min(cc_industry_HEAD_WIDEN * fp,
                             cc_industry_HEAD_WIDEN_MAX)));
}

// World half extents of a head whose aim is `aim` (along x or y).
fn cc_industry_head_box_hw(hw: vec3<f32>, aim: vec2<f32>) -> vec3<f32> {
    let ax = abs(aim.x) >= abs(aim.y);
    return vec3<f32>(select(hw.y, hw.x, ax), select(hw.x, hw.y, ax), hw.z);
}

fn cc_industry_fixtures(best_in: CityHit, o: vec3<f32>, dir: vec3<f32>,
                        inv_dir: vec3<f32>, t0: f32, t1: f32, cc: CityCell,
                        ci: vec2<i32>, px: f32) -> CityHit {
    var best = best_in;
    let yard = cc.land_use == CITY_USE_YARD;
    // A runtime count (always CITY_FLOOD_MAX here: open land never gets this
    // far) keeps the loop rolled.
    let nk = select(CITY_FLOOD_MAX, 0, cc.land_use == CITY_USE_OPEN);
    for (var k: i32 = 0; k < nk; k = k + 1) {
        let f = city_flood(cc, k);
        if (!f.on) {
            continue;
        }
        let c = cc_industry_head_c(cc, f);
        let fp = px * distance(o, c);
        let bh = cc_industry_head_box_hw(cc_industry_head_hw(cc, fp), f.aim);
        best = cc_industry_box(best, o, dir, inv_dir, t0, t1, c - bh, c + bh,
                               cc_industry_K_HEAD, ci);
        // The hardware: dark, 0.2 m, gone under a pixel past FINE_FP.
        if (fp < cc_industry_FINE_FP * 2.0) {
            var lo = vec3<f32>(0.0);
            var hi = vec3<f32>(0.0);
            if (yard) {
                let ph = vec2<f32>(cc_industry_POLE_HW);
                lo = vec3<f32>(f.head.xy - ph, 0.0);
                hi = vec3<f32>(f.head.xy + ph, f.head.z - 0.1);
            } else {
                // The bracket: from the wall out to the head's back.
                let wall = f.head.xy - f.aim * 0.8;
                let back = f.head.xy - f.aim * 0.2;
                let lat = vec2<f32>(abs(f.aim.y), abs(f.aim.x)) * 0.04;
                lo = vec3<f32>(min(wall, back) - lat, f.head.z + 0.05);
                hi = vec3<f32>(max(wall, back) + lat, f.head.z + 0.13);
            }
            best = cc_industry_box(best, o, dir, inv_dir, t0, t1, lo, hi,
                                   cc_industry_K_STEEL, ci);
        }
    }
    return best;
}

// ============================================================================
// PROP LAYOUT
// ============================================================================
//
// Grid i of the lot (i < cc_industry_N_GRIDS; -1 is a flat roof's skylights).
// Everything is sized from the plot and the building, so a narrow court gets
// pallets, a deep one trailers and a row of boxes along its far fence. No
// two grids of a lot overlap: the shader finds a hit's slot by position.
const cc_industry_N_GRIDS: i32 = 7;

fn cc_industry_grid(cc: CityCell, i: i32) -> cc_industry_Grid {
    if (i < 0) {
        return cc_industry_sky_grid(cc);
    }
    let l = cc_industry_lot(cc);
    let E = cc_industry_EDGE;
    let box_cell_l = cc_industry_BOX_L + 0.5;
    let box_cell_w = cc_industry_BOX_W + 0.26;
    let box_obj = vec2<f32>(cc_industry_BOX_L, cc_industry_BOX_W);
    let pal_cell = vec2<f32>(1.6, 1.4);
    let pal_obj = vec2<f32>(1.2, 1.0);
    let fork_cell = vec2<f32>(3.6, 1.4);
    let fork_obj = vec2<f32>(3.3, 1.2);
    let skip_cell = vec2<f32>(cc_industry_SKIP_L + 0.4,
                              cc_industry_SKIP_W + 0.4);
    let skip_obj = vec2<f32>(cc_industry_SKIP_L, cc_industry_SKIP_W);
    let pal_h = cc_industry_PALLET_H * 13.0;

    if (cc.land_use == CITY_USE_INDUSTRIAL) {
        // The court: from the dock wall out to the plot edge.
        let sgn = select(-1.0, 1.0, l.court_hi);
        let wall = select(l.bc.x, l.bc.y, l.court_hi);
        let far = select(l.pc.x, l.pc.y, l.court_hi);
        let depth = cc_industry_court_depth(l);
        let docks = cc_industry_docks(l);
        let n_dock = i32(docks.y);
        let dock_a0 = docks.x - 0.5 * cc_industry_DOCK_PITCH;
        let trl_c = cc_industry_TRL_L + 2.0 * cc_industry_DOCK_GAP;
        let has_docks = cc_industry_has_trailers(l);
        let d = cc_industry_draw(cc, 21u, i, 0);
        if (i == 0 && has_docks) {
            // Trailers backed onto the dock doors, rear to the wall, legs
            // down at the nose.
            let c_lo = select(wall - trl_c, wall, l.court_hi);
            return cc_industry_mk_grid(
                l, cc_industry_K_TRAILER, vec2<f32>(dock_a0, c_lo),
                vec2<f32>(cc_industry_DOCK_PITCH, trl_c), vec2<i32>(n_dock, 1),
                vec2<f32>(cc_industry_TRL_W, cc_industry_TRL_L),
                0.0, cc_industry_TRL_H, 0.45, 1,
                select(3, 2, l.court_hi), 20u);
        }
        if (i == 1) {
            if (depth >= trl_c + 2.0 + 12.0) {
                // A deep court: a line of boxes along its far fence, end to
                // end, one or two high — along the building's own run, so
                // it never reaches into an end yard's grid.
                let n = i32(floor((l.ba.y - l.ba.x) / box_cell_l
                                  * (0.5 + 0.35 * d.y)));
                let c_lo = select(far + E, far - E - box_cell_w, l.court_hi);
                return cc_industry_mk_grid(
                    l, cc_industry_K_BOX, vec2<f32>(l.ba.x, c_lo),
                    vec2<f32>(box_cell_l, box_cell_w), vec2<i32>(n, 1),
                    box_obj, 0.0, 2.0 * cc_industry_TIER, 0.80, 2, 0, 22u);
            }
            if (!has_docks && depth >= 5.0) {
                // A shallow court: pallet stacks along the far edge.
                let n = min(i32(floor((l.pa.y - l.pa.x) * 0.3 / pal_cell.x)),
                            12);
                let c_lo = select(far + E, far - E - pal_cell.y, l.court_hi);
                return cc_industry_mk_grid(
                    l, cc_industry_K_PALLET,
                    vec2<f32>(l.ba.x + 6.0 + 20.0 * d.z, c_lo), pal_cell,
                    vec2<i32>(n, 1), pal_obj, 0.0, pal_h, 0.70, 1, 0, 23u);
            }
            return cc_industry_no_grid();
        }
        if (i == 2 || i == 3) {
            // The end yards, where the building leaves the plot's ends open.
            let lo_end = i == 2;
            let a_lo = select(l.ba.y + E, l.pa.x + E, lo_end);
            let a_hi = select(l.pa.y - E, l.ba.x - E, lo_end);
            let w = a_hi - a_lo;
            let span_c = l.pc.y - l.pc.x - 2.0 * E;
            if (w < 4.0) {
                return cc_industry_no_grid();
            }
            if (d.x < 0.40 && span_c >= box_cell_l) {
                // Containers parked across the end yard, side by side.
                let n_c = min(i32(floor(span_c / box_cell_l)), 3);
                let n_a = min(i32(floor(w / box_cell_w)), 8);
                return cc_industry_mk_grid(
                    l, cc_industry_K_BOX,
                    vec2<f32>(select(a_lo, a_hi - f32(n_a) * box_cell_w,
                                     lo_end),
                              l.pc.x + E),
                    vec2<f32>(box_cell_w, box_cell_l), vec2<i32>(n_a, n_c),
                    box_obj.yx, 0.0, 3.0 * cc_industry_TIER, 0.70, 3, 0,
                    24u + u32(i));
            }
            if (d.x < 0.75) {
                let n_a = min(i32(floor(w / pal_cell.x)), 6);
                let n_c = min(i32(floor(span_c * 0.4 / pal_cell.y)), 9);
                return cc_industry_mk_grid(
                    l, cc_industry_K_PALLET,
                    vec2<f32>(a_lo, select(l.pc.x + E,
                                           l.pc.y - E - f32(n_c) * pal_cell.y,
                                           d.y < 0.5)),
                    pal_cell, vec2<i32>(n_a, n_c), pal_obj, 0.0, pal_h, 0.65,
                    1, 0, 24u + u32(i));
            }
            if (w >= 7.0) {
                // Rebar bundles on dunnage, laid across the end yard.
                let bar = min(w - 0.6, 12.0);
                let n_c = 3 + i32(d.y * 4.0);
                return cc_industry_mk_grid(
                    l, cc_industry_K_REBAR,
                    vec2<f32>(a_lo, l.pc.x + E + span_c * 0.5 * d.z),
                    vec2<f32>(bar + 0.6, 0.7), vec2<i32>(1, n_c),
                    vec2<f32>(bar, 0.45), 0.0, cc_industry_REBAR_H, 0.9, 1, 0,
                    24u + u32(i));
            }
            return cc_industry_no_grid();
        }
        if (i == 4) {
            // A skip against the court's far edge, at the end of the
            // building's run away from the row of boxes (which stops short
            // of it: that row covers at most 85% of a run of 130 m or more).
            let fits = depth >= trl_c + 2.0 + skip_cell.y + 2.0
                       || (!has_docks && depth >= skip_cell.y + 4.0);
            if (!fits || d.x > 0.55) {
                return cc_industry_no_grid();
            }
            let c_lo = select(far + E, far - E - skip_cell.y, l.court_hi);
            return cc_industry_mk_grid(
                l, cc_industry_K_SKIP,
                vec2<f32>(l.ba.y - 2.0 - skip_cell.x - 6.0 * d.y, c_lo),
                skip_cell, vec2<i32>(1, 1), skip_obj, 0.0,
                cc_industry_SKIP_H, 1.0, 1, 0, 25u);
        }
        if (i == 6) {
            // The dock hardware: a slot per door, flush on the wall.
            let c_lo = select(wall - cc_industry_SHELL_OUT, wall, l.court_hi);
            return cc_industry_mk_grid(
                l, cc_industry_K_DOCK,
                vec2<f32>(dock_a0, c_lo),
                vec2<f32>(cc_industry_DOCK_PITCH, cc_industry_SHELL_OUT),
                vec2<i32>(n_dock, 1),
                vec2<f32>(cc_industry_DOCK_PITCH - 0.2, cc_industry_SHELL_OUT),
                0.0, 4.95, 1.0,
                select(0, 1, cc_industry_wall_of(cc).shelter),
                select(3, 2, l.court_hi), 27u);
        }
        if (i == 5 && has_docks && d.x < 0.6) {
            // A forklift working an open door: in a dock slot with no trailer
            // in it, forks toward the wall.
            let j = i32(d.y * f32(n_dock));
            // The dock grid's own slot draw (its slot index is (j, 0) in a lot
            // along x, (0, j) in one along y).
            let slot = cc_industry_draw(cc, 20u, select(0, j, l.long_x),
                                        select(j, 0, l.long_x));
            if (slot.x < 0.45) {
                return cc_industry_no_grid();
            }
            let a = docks.x + f32(j) * cc_industry_DOCK_PITCH;
            let c = wall + sgn * (2.6 + 3.0 * d.z);
            return cc_industry_mk_grid(
                l, cc_industry_K_FORK,
                vec2<f32>(a - 0.5 * fork_cell.y, c - 0.5 * fork_cell.x),
                fork_cell.yx, vec2<i32>(1, 1), fork_obj.yx, 0.0,
                cc_industry_FORK_H, 1.0, 1, select(2, 3, l.court_hi), 26u);
        }
        return cc_industry_no_grid();
    }

    // --- a yard ---------------------------------------------------------------
    let ua = l.pa + vec2<f32>(E, -E);
    let uc = l.pc + vec2<f32>(E, -E);
    let la = ua.y - ua.x;
    let lc = uc.y - uc.x;
    let ty = cc_industry_draw(cc, 30u, 0, 0);
    let d = cc_industry_draw(cc, 31u, i, 0);
    if (ty.x < 0.35) {
        // Container storage: two blocks, boxes along the long axis, stacked
        // up to three, a driving aisle between them.
        if (i < 2) {
            let n_a = i32(floor(la / box_cell_l));
            let n_c = i32(floor((lc - 12.0) * 0.5 / box_cell_w));
            let a0 = ua.x + 0.5 * (la - f32(n_a) * box_cell_l);
            let c0 = select(uc.x, uc.y - f32(n_c) * box_cell_w, i == 1);
            return cc_industry_mk_grid(
                l, cc_industry_K_BOX, vec2<f32>(a0, c0),
                vec2<f32>(box_cell_l, box_cell_w), vec2<i32>(n_a, n_c),
                box_obj, 0.0, 3.0 * cc_industry_TIER, 0.88, 3, 0,
                40u + u32(i));
        }
        if (i == 2) {
            return cc_industry_mk_grid(
                l, cc_industry_K_FORK,
                vec2<f32>(ua.x + la * (0.2 + 0.6 * d.y),
                          0.5 * (uc.x + uc.y) - 0.5 * fork_cell.y),
                fork_cell, vec2<i32>(1, 1), fork_obj, 0.0,
                cc_industry_FORK_H, 1.0, 1, select(0, 1, d.z < 0.5), 42u);
        }
        return cc_industry_no_grid();
    }
    if (ty.x < 0.55) {
        // Trailer storage: two facing rows, noses to the aisle.
        let cell = vec2<f32>(3.7, cc_industry_TRL_L + 0.6);
        if (i < 2 && lc >= 2.0 * cell.y + 8.0) {
            let n_a = i32(floor(la / cell.x));
            let a0 = ua.x + 0.5 * (la - f32(n_a) * cell.x);
            let c0 = select(uc.x, uc.y - cell.y, i == 1);
            return cc_industry_mk_grid(
                l, cc_industry_K_TRAILER, vec2<f32>(a0, c0), cell,
                vec2<i32>(n_a, 1),
                vec2<f32>(cc_industry_TRL_W, cc_industry_TRL_L), 0.0,
                cc_industry_TRL_H, 0.60, 1, select(2, 3, i == 1),
                50u + u32(i));
        }
        return cc_industry_no_grid();
    }
    // Construction refuse, or a vacant lot. Along the long axis: the debris
    // mounds (cc_industry_pile) own the first ~40%; pallets, the skip and
    // the forklift the middle; rebar and a site box or two the far end.
    let refuse = ty.x < 0.87;
    let zb = ua.x + 0.42 * la;
    let zc = ua.x + 0.72 * la;
    if (i == 0 && refuse) {
        let n_a = min(i32(floor(0.26 * la / pal_cell.x)), 7);
        let n_c = min(i32(floor(0.45 * lc / pal_cell.y)), 9);
        return cc_industry_mk_grid(
            l, cc_industry_K_PALLET, vec2<f32>(zb, uc.x), pal_cell,
            vec2<i32>(n_a, n_c), pal_obj, 0.0, pal_h, 0.6, 1, 0, 60u);
    }
    if (i == 1 && (refuse || d.x < 0.4)) {
        return cc_industry_mk_grid(
            l, cc_industry_K_SKIP, vec2<f32>(zb, uc.y - skip_cell.y),
            skip_cell, vec2<i32>(1, 1), skip_obj, 0.0, cc_industry_SKIP_H,
            1.0, 1, 0, 61u);
    }
    if (i == 2 && refuse && d.x < 0.7) {
        return cc_industry_mk_grid(
            l, cc_industry_K_FORK,
            vec2<f32>(zb + 0.5, uc.x + 0.55 * lc), fork_cell,
            vec2<i32>(1, 1), fork_obj, 0.0, cc_industry_FORK_H, 1.0, 1,
            select(0, 1, d.y < 0.5), 62u);
    }
    if (i == 3 && refuse) {
        let bar = min(ua.y - zc - 0.6, 12.0);
        let n_c = 3 + i32(d.y * 5.0);
        return cc_industry_mk_grid(
            l, cc_industry_K_REBAR, vec2<f32>(zc, uc.x), vec2<f32>(bar + 0.6, 0.7),
            vec2<i32>(1, n_c), vec2<f32>(bar, 0.45), 0.0,
            cc_industry_REBAR_H, 0.9, 1, 0, 63u);
    }
    if (i == 4 && refuse && ua.y - zc >= box_cell_l) {
        return cc_industry_mk_grid(
            l, cc_industry_K_BOX, vec2<f32>(zc, uc.y - 2.0 * box_cell_w),
            vec2<f32>(box_cell_l, box_cell_w), vec2<i32>(1, 2), box_obj, 0.0,
            cc_industry_TIER, 0.75, 1, 0, 64u);
    }
    return cc_industry_no_grid();
}

// Per-slot height: 0 for an empty slot.
fn cc_industry_slot_h(g: cc_industry_Grid, d: vec4<f32>) -> f32 {
    if (d.x >= g.occ) {
        return 0.0;
    }
    if (g.kind == cc_industry_K_BOX) {
        // @domain: d is cc_industry_draw: a hash draw, in [0, 1)
        let t = min(1.0 + floor(pow(d.y, 0.8) * f32(g.tiers)), f32(g.tiers));
        return t * cc_industry_TIER;
    }
    if (g.kind == cc_industry_K_PALLET) {
        return cc_industry_PALLET_H * (2.0 + floor(d.y * 11.0));
    }
    return g.zmax;
}

fn cc_industry_slot_draw(cc: CityCell, g: cc_industry_Grid, ij: vec2<i32>)
        -> vec4<f32> {
    return cc_industry_draw(cc, g.salt, ij.x, ij.y);
}

// --- local frames for things with a front -----------------------------------
// (u forward, v to the left, z up) about a centre on the ground.
fn cc_industry_fwd(front: i32) -> vec2<f32> {
    return select(select(select(vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0),
                                front == 2),
                         vec2<f32>(-1.0, 0.0), front == 1),
                  vec2<f32>(1.0, 0.0), front == 0);
}

struct cc_industry_Box {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

fn cc_industry_local_box(c: vec3<f32>, front: i32, lo: vec3<f32>,
                         hi: vec3<f32>) -> cc_industry_Box {
    let f = cc_industry_fwd(front);
    let lft = vec2<f32>(-f.y, f.x);
    let p0 = c.xy + f * lo.x + lft * lo.y;
    let p1 = c.xy + f * hi.x + lft * hi.y;
    return cc_industry_Box(vec3<f32>(min(p0, p1), c.z + lo.z),
                           vec3<f32>(max(p0, p1), c.z + hi.z));
}

// A forklift, as a union of boxes about its footprint centre: body,
// counterweight, overhead guard (roof and four posts), mast uprights and
// carriage, forks, wheels, seat. Part 0-4 is the far-field set.
fn cc_industry_fork_part(i: i32) -> cc_industry_Box {
    var lo = vec3<f32>(0.0);
    var hi = vec3<f32>(0.0);
    switch i {
        case 0: { lo = vec3<f32>(-1.35, -0.55, 0.30); hi = vec3<f32>(0.25, 0.55, 1.10); }
        case 1: { lo = vec3<f32>(-1.65, -0.58, 0.22); hi = vec3<f32>(-1.30, 0.58, 1.22); }
        case 2: { lo = vec3<f32>(-1.25, -0.56, 2.05); hi = vec3<f32>(0.20, 0.56, 2.12); }
        case 3: { lo = vec3<f32>(0.32, 0.26, 0.08); hi = vec3<f32>(0.44, 0.40, 2.30); }
        case 4: { lo = vec3<f32>(0.32, -0.40, 0.08); hi = vec3<f32>(0.44, -0.26, 2.30); }
        case 5: { lo = vec3<f32>(0.44, -0.46, 0.08); hi = vec3<f32>(0.50, 0.46, 1.05); }
        case 6: { lo = vec3<f32>(0.50, 0.22, 0.05); hi = vec3<f32>(1.65, 0.34, 0.10); }
        case 7: { lo = vec3<f32>(0.50, -0.34, 0.05); hi = vec3<f32>(1.65, -0.22, 0.10); }
        case 8: { lo = vec3<f32>(-0.25, -0.60, 0.0); hi = vec3<f32>(0.25, 0.60, 0.55); }
        case 9: { lo = vec3<f32>(-1.30, -0.56, 0.0); hi = vec3<f32>(-0.85, 0.56, 0.48); }
        case 10: { lo = vec3<f32>(0.12, 0.46, 1.10); hi = vec3<f32>(0.20, 0.54, 2.05); }
        case 11: { lo = vec3<f32>(0.12, -0.54, 1.10); hi = vec3<f32>(0.20, -0.46, 2.05); }
        case 12: { lo = vec3<f32>(-1.25, 0.46, 1.20); hi = vec3<f32>(-1.17, 0.54, 2.05); }
        case 13: { lo = vec3<f32>(-1.25, -0.54, 1.20); hi = vec3<f32>(-1.17, -0.46, 2.05); }
        default: { lo = vec3<f32>(-0.95, -0.25, 1.10); hi = vec3<f32>(-0.50, 0.25, 1.50); }
    }
    return cc_industry_Box(lo, hi);
}

// A semi-trailer, rear (doors) at -u: the box body on its deck; under the
// rear the tandem bogie's two tyre banks with the axles between them; under
// the nose the two landing legs; the chassis rails and the under-ride bar.
// Under a trailer you see through to the yard beyond — legs and tyres, not
// a skirt — and that gap is most of what says "trailer" rather than "box".
// Parts 0-2 are the far-field set.
fn cc_industry_trl_part(i: i32) -> cc_industry_Box {
    let hl = 0.5 * cc_industry_TRL_L;
    let hw = 0.5 * cc_industry_TRL_W;
    let dk = cc_industry_TRL_DECK;
    var lo = vec3<f32>(0.0);
    var hi = vec3<f32>(0.0);
    switch i {
        case 0: { lo = vec3<f32>(-hl, -hw, dk); hi = vec3<f32>(hl, hw, cc_industry_TRL_H); }
        case 1: { lo = vec3<f32>(-hl + 0.6, 0.62, 0.0); hi = vec3<f32>(-hl + 4.6, hw - 0.03, 1.05); }
        case 2: { lo = vec3<f32>(-hl + 0.6, -hw + 0.03, 0.0); hi = vec3<f32>(-hl + 4.6, -0.62, 1.05); }
        case 3: { lo = vec3<f32>(hl - 3.40, 0.80, 0.0); hi = vec3<f32>(hl - 3.22, 0.98, dk); }
        case 4: { lo = vec3<f32>(hl - 3.40, -0.98, 0.0); hi = vec3<f32>(hl - 3.22, -0.80, dk); }
        case 5: { lo = vec3<f32>(-hl + 0.4, -0.55, 1.0); hi = vec3<f32>(hl - 0.3, 0.55, dk); }
        case 6: { lo = vec3<f32>(-hl + 0.9, -0.62, 0.38); hi = vec3<f32>(-hl + 4.3, 0.62, 0.62); }
        default: { lo = vec3<f32>(-hl + 0.05, -hw + 0.1, 0.50); hi = vec3<f32>(-hl + 0.20, hw - 0.1, 0.68); }
    }
    return cc_industry_Box(lo, hi);
}

// A dock's hardware, about its slot centre (SHELL_OUT/2 off the wall), u out
// from the wall: the shelter's two side pads and head pad, black fabric over
// a frame, standing proud of the wall around the door; the two rubber
// bumpers either side of the door at trailer-bed height; the leveller's
// steel lip at the dock floor, which a trailer backs under. Parts 0-2 are
// the shelter.
fn cc_industry_dock_part(i: i32) -> cc_industry_Box {
    let w = -0.5 * cc_industry_SHELL_OUT;      // the wall, in u
    let hd = 0.5 * cc_industry_DOOR_W;
    let top = cc_industry_DOCK_H + cc_industry_DOOR_H;
    var lo = vec3<f32>(0.0);
    var hi = vec3<f32>(0.0);
    switch i {
        case 0: { lo = vec3<f32>(w, hd, 1.0); hi = vec3<f32>(w + 0.60, hd + 0.50, top + 0.75); }
        case 1: { lo = vec3<f32>(w, -hd - 0.50, 1.0); hi = vec3<f32>(w + 0.60, -hd, top + 0.75); }
        case 2: { lo = vec3<f32>(w, -hd - 0.50, top + 0.10); hi = vec3<f32>(w + 0.60, hd + 0.50, top + 0.75); }
        case 3: { lo = vec3<f32>(w, 1.05, 0.75); hi = vec3<f32>(w + 0.18, 1.30, 1.20); }
        case 4: { lo = vec3<f32>(w, -1.30, 0.75); hi = vec3<f32>(w + 0.18, -1.05, 1.20); }
        default: { lo = vec3<f32>(w, -1.0, cc_industry_DOCK_H - 0.08); hi = vec3<f32>(w + 0.30, 1.0, cc_industry_DOCK_H); }
    }
    return cc_industry_Box(lo, hi);
}

// One object in its slot, against the segment. Boxes for the simple kinds;
// part unions for trailers and forklifts close up; a hollow for the skip.
fn cc_industry_obj(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                   inv_dir: vec3<f32>, t0: f32, t1: f32, g: cc_industry_Grid,
                   ctr: vec2<f32>, hgt: f32, ci: vec2<i32>, fp: f32)
        -> CityHit {
    let omin = vec3<f32>(ctr - 0.5 * g.obj, g.lo.z);
    let omax = vec3<f32>(ctr + 0.5 * g.obj, g.lo.z + hgt);
    let sb = city_box_hit(o, inv_dir, omin, omax);
    if (sb.x > sb.y || sb.y <= max(t0, 1e-3) || sb.x > min(t1, best.t)) {
        return best;
    }
    if (g.kind == cc_industry_K_TRAILER || g.kind == cc_industry_K_FORK
        || g.kind == cc_industry_K_DOCK) {
        let is_fork = g.kind == cc_industry_K_FORK;
        let is_dock = g.kind == cc_industry_K_DOCK;
        let near = fp < cc_industry_PART_FP;
        var n = select(select(3, 8, near), select(5, 15, near), is_fork);
        var i0 = 0;
        if (is_dock) {
            // The shelter (parts 0-2) where the building has them; the
            // bumpers and leveller lip close in only.
            i0 = select(3, 0, g.tiers > 0);
            n = select(3, 6, near);
        }
        var r = best;
        let c = vec3<f32>(ctr, g.lo.z);
        for (var i: i32 = i0; i < n; i = i + 1) {
            var pb: cc_industry_Box;
            if (is_fork) {
                pb = cc_industry_fork_part(i);
            } else if (is_dock) {
                pb = cc_industry_dock_part(i);
            } else {
                pb = cc_industry_trl_part(i);
            }
            let wb = cc_industry_local_box(c, g.front, pb.lo, pb.hi);
            r = cc_industry_box(r, o, dir, inv_dir, t0, t1, wb.lo, wb.hi,
                                g.kind, ci);
        }
        return r;
    }
    if (g.kind == cc_industry_K_SKIP) {
        // A roll-off skip: walls around a cavity filled to a draw-set level.
        // Entering through the open top inside the walls, the ray meets the
        // cavity's far side — a wall's inner face or the refuse.
        let q = o + sb.x * dir;
        let wl = vec2<f32>(cc_industry_SKIP_WALL);
        let imin = vec3<f32>(omin.xy + wl, g.lo.z + 0.75 + 0.55 * fract(
            ctr.x * 0.37 + ctr.y * 0.61));
        let imax = vec3<f32>(omax.xy - wl, omax.z + 1.0);
        let into = q.z > omax.z - 1e-3 && all(q.xy > imin.xy)
                   && all(q.xy < imax.xy);
        if (!into) {
            return cc_industry_box(best, o, dir, inv_dir, t0, t1, omin, omax,
                                   cc_industry_K_SKIP, ci);
        }
        let si = city_box_hit(o, inv_dir, imin, imax);
        let te = si.y;
        if (te <= max(t0, 1e-3) || te > t1 || te >= best.t) {
            return best;
        }
        let p = o + te * dir;
        let nb = city_box_normal(p, imin, imax);
        let refuse = nb.z < -0.5;
        return CityHit(true, te, p, select(-nb, vec3<f32>(0.0, 0.0, 1.0),
                                           refuse),
                       select(cc_industry_K_SKIP, cc_industry_K_REFUSE,
                              refuse), ci);
    }
    if (sb.x > max(t0, 1e-3) && sb.x <= t1 && sb.x < best.t) {
        let p = o + sb.x * dir;
        return CityHit(true, sb.x, p, city_box_normal(p, omin, omax), g.kind,
                       ci);
    }
    return best;
}

// A grid against the segment: its bounding box, then a 2D DDA over the slots
// the ray crosses in order. Objects stay inside their slots, so the first
// slot that reports a hit holds the nearest one.
//
// LOD is decided here, at the footprint where the ray enters the grid — not
// at the segment's start, which for the camera's own cell is the camera: a
// skylight grid seen from 5 km straight down would otherwise be traced
// skylight by skylight and shaded as their mean, and lose the coverage
// factor twice. `fp0` (the footprint at t0, a lower bound) rejects the
// small, dark props before their box is even tested.
fn cc_industry_grid_hit(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                        inv_dir: vec3<f32>, t0: f32, t1: f32, cc: CityCell,
                        g: cc_industry_Grid, ci: vec2<i32>, px: f32, fp0: f32)
        -> CityHit {
    let small = g.kind == cc_industry_K_PALLET || g.kind == cc_industry_K_REBAR
                || g.kind == cc_industry_K_FORK || g.kind == cc_industry_K_DOCK;
    if (small && fp0 >= cc_industry_FINE_FP) {
        return best;
    }
    let span = vec2<f32>(g.n) * g.cell;
    let gmin = g.lo;
    let gmax = vec3<f32>(g.lo.xy + span, g.lo.z + g.zmax);
    let sb = city_box_hit(o, inv_dir, gmin, gmax);
    let ta = max(max(sb.x, t0), 1e-3);
    let tb = min(min(sb.y, t1), best.t);
    if (sb.x > sb.y || ta > tb) {
        return best;
    }
    let fp = px * ta;
    if (g.kind == cc_industry_K_SKY && fp >= cc_industry_SKY_FP) {
        // Sub-pixel relief: the whole grid as one thin panel, shaded with
        // the skylights' pattern dissolving into its mean
        // (cc_industry_sky_shade).
        return cc_industry_box(best, o, dir, inv_dir, t0, t1, gmin,
                               vec3<f32>(gmax.xy, gmin.z + 0.02),
                               cc_industry_K_SKY, ci);
    }
    let p = o + ta * dir - g.lo;
    var ij = clamp(vec2<i32>(floor(p.xy / g.cell)), vec2<i32>(0), g.n - 1);
    let stp = vec2<i32>(select(-1, 1, dir.x > 0.0), select(-1, 1, dir.y > 0.0));
    let nxt = vec2<f32>(ij + max(stp, vec2<i32>(0))) * g.cell;
    var tmax = vec2<f32>(
        select(1e30, ta + (nxt.x - p.x) * inv_dir.x, abs(dir.x) > 1e-9),
        select(1e30, ta + (nxt.y - p.y) * inv_dir.y, abs(dir.y) > 1e-9));
    let tdel = g.cell * abs(inv_dir.xy);
    let steps = g.n.x + g.n.y;
    for (var s: i32 = 0; s < steps; s = s + 1) {
        let d = cc_industry_slot_draw(cc, g, ij);
        let hgt = cc_industry_slot_h(g, d);
        if (hgt > 0.0) {
            let ctr = g.lo.xy + (vec2<f32>(ij) + 0.5) * g.cell;
            let r = cc_industry_obj(best, o, dir, inv_dir, t0, t1, g, ctr,
                                    hgt, ci, fp);
            if (r.hit && r.t < best.t) {
                return r;
            }
        }
        if (min(tmax.x, tmax.y) > tb) {
            break;
        }
        if (tmax.x < tmax.y) {
            ij.x = ij.x + stp.x;
            tmax.x = tmax.x + tdel.x;
        } else {
            ij.y = ij.y + stp.y;
            tmax.y = tmax.y + tdel.y;
        }
        if (any(ij < vec2<i32>(0)) || any(ij >= g.n)) {
            break;
        }
    }
    return best;
}

// --- debris mounds (yards) ----------------------------------------------------
// A mound is three to five overlapping half-ellipsoids — a main heap and the
// lobes slumped off it — so the silhouette is lumpy rather than a dome.
struct cc_industry_Pile {
    ok: bool,
    c: vec2<f32>,
    r: f32,
    h: f32,
    seed: vec4<f32>,
}

const cc_industry_N_PILES: i32 = 4;

fn cc_industry_pile(cc: CityCell, i: i32) -> cc_industry_Pile {
    var m: cc_industry_Pile;
    m.ok = false;
    m.c = vec2<f32>(0.0);
    m.r = 1.0;
    m.h = 0.0;
    m.seed = vec4<f32>(0.0);
    if (cc.land_use != CITY_USE_YARD) {
        return m;
    }
    let ty = cc_industry_draw(cc, 30u, 0, 0);
    if (ty.x < 0.55) {
        return m;   // container or trailer storage: kept clear
    }
    let refuse = ty.x < 0.87;
    let n = select(1 + i32(ty.y * 2.99), 2 + i32(ty.y * 2.99), refuse);
    if (i >= n) {
        return m;
    }
    let l = cc_industry_lot(cc);
    let d = cc_industry_draw(cc, 70u, i, 0);
    m.r = 2.0 + 2.5 * d.x;
    m.h = 0.8 + 1.4 * d.y;
    let marg = 1.6 * m.r + cc_industry_EDGE;   // the lobes reach 1.55 r
    let a_hi = select(l.pa.y, l.pa.x + 0.40 * (l.pa.y - l.pa.x), refuse);
    let a = mix(l.pa.x + marg, max(a_hi - marg, l.pa.x + marg), d.z);
    let c = mix(l.pc.x + marg, l.pc.y - marg, d.w);
    m.c = cc_industry_xy(l, vec2<f32>(a, c));
    m.seed = cc_industry_draw(cc, 71u, i, 0);
    m.ok = true;
    return m;
}

// Lobe j of a mound: (centre xy, horizontal radius, height).
fn cc_industry_lobe(m: cc_industry_Pile, j: i32) -> vec4<f32> {
    // The main heap is kept small enough that the lobes break its outline
    // rather than hiding inside it.
    if (j == 0) {
        return vec4<f32>(m.c, 0.8 * m.r, m.h);
    }
    let fj = f32(j);
    let ang = 6.2831853 * m.seed.x + 1.7 * fj;
    let off = vec2<f32>(cos(ang), sin(ang)) * m.r
              * (0.50 + 0.40 * fract(m.seed.y + 0.31 * fj));
    let k = 0.40 + 0.25 * fract(m.seed.z + 0.53 * fj);
    return vec4<f32>(m.c + off, m.r * k,
                     m.h * (0.55 + 0.30 * fract(m.seed.w + 0.71 * fj)));
}

fn cc_industry_piles_hit(best_in: CityHit, o: vec3<f32>, dir: vec3<f32>,
                         t0: f32, t1: f32, cc: CityCell, ci: vec2<i32>)
        -> CityHit {
    var best = best_in;
    let np = select(0, cc_industry_N_PILES, cc.land_use == CITY_USE_YARD);
    for (var i: i32 = 0; i < np; i = i + 1) {
        let m = cc_industry_pile(cc, i);
        if (!m.ok) {
            continue;
        }
        let nl = 3 + i32(m.seed.w * 2.999);
        for (var j: i32 = 0; j < nl; j = j + 1) {
            let e = cc_industry_lobe(m, j);
            let rad = vec3<f32>(e.z, e.z, e.w);
            let qo = (o - vec3<f32>(e.xy, 0.0)) / rad;
            let qd = dir / rad;
            let a = dot(qd, qd);
            let b = dot(qo, qd);
            let c = dot(qo, qo) - 1.0;
            let disc = b * b - a * c;
            if (disc <= 0.0) {
                continue;
            }
            // @domain: the discriminant is tested positive (or non-negative) just above
            let t = (-b - sqrt(disc)) / a;
            if (t > max(t0, 1e-3) && t <= t1 && t < best.t) {
                let p = o + t * dir;
                if (p.z >= 0.0) {
                    // @domain: the hit lies on the dome, never at its centre
                    let n = normalize((p - vec3<f32>(e.xy, 0.0))
                                      / (rad * rad));
                    best = CityHit(true, t, p, n, cc_industry_K_PILE, ci);
                }
            }
        }
    }
    return best;
}

// --- the fence (yards) --------------------------------------------------------
// Chain-link on galvanised posts, around 75% of yards, with a gate gap in
// one side. The mesh itself is not drawn: it has no opaque surface to shade,
// and seen through at night it is invisible against the dark yard. Posts
// every 3 m and a top rail are what reads, and only close in.
//
// Written for the register file, not for generality: a first version found
// its gate by probing the neighbouring lots' land use in a loop, broke the
// rail in two around it and walked the posts over the ray's run, and that
// body alone cost 12% on URBAN vistas that never reach it (inlined into the
// core's DDA, its registers are paid by every city pixel; measured by
// ablation, 2026-09-23). The gate is now a drawn side, the rail one box
// whose hit is refused inside the gap, and the posts the two nearest the
// ray's run through the fence line — a ray is inside that 8 cm slab for a
// post pitch or less unless it runs down the fence, where the nearest post
// hides the rest.
fn cc_industry_fence_on(cc: CityCell) -> vec4<f32> {
    return cc_industry_draw(cc, 80u, 0, 0);
}

// Edge e (0 -x, 1 +x, 2 -y, 3 +y) of the fence line: its axis-aligned line
// as (fixed coordinate, run start, run end) and whether it runs along x.
fn cc_industry_fence_edge(cc: CityCell, e: i32) -> vec4<f32> {
    let lo = cc.plot_min + vec2<f32>(cc_industry_FENCE_IN);
    let hi = cc.plot_max - vec2<f32>(cc_industry_FENCE_IN);
    if (e == 0) { return vec4<f32>(lo.x, lo.y, hi.y, 0.0); }
    if (e == 1) { return vec4<f32>(hi.x, lo.y, hi.y, 0.0); }
    if (e == 2) { return vec4<f32>(lo.y, lo.x, hi.x, 1.0); }
    return vec4<f32>(hi.y, lo.x, hi.x, 1.0);
}

fn cc_industry_fence_hit(best_in: CityHit, o: vec3<f32>, dir: vec3<f32>,
                         inv_dir: vec3<f32>, t0: f32, t1: f32, cc: CityCell,
                         ci: vec2<i32>) -> CityHit {
    var best = best_in;
    let fo = cc_industry_fence_on(cc);
    let gate_e = i32(fo.y * 3.999);
    let pw = 0.04;       // post half-width
    let rw = 0.025;      // rail half-width
    let ne = select(0, 4, fo.x < cc_industry_FENCE_FRAC);
    for (var e: i32 = 0; e < ne; e = e + 1) {
        let ln = cc_industry_fence_edge(cc, e);
        let along_x = ln.w > 0.5;
        // The fence line's slab: a cheap reject for the whole edge.
        let lo = select(vec2<f32>(ln.x - pw, ln.y - pw),
                        vec2<f32>(ln.y - pw, ln.x - pw), along_x);
        let hi = select(vec2<f32>(ln.x + pw, ln.z + pw),
                        vec2<f32>(ln.z + pw, ln.x + pw), along_x);
        let sb = city_box_hit(o, inv_dir, vec3<f32>(lo, 0.0),
                              vec3<f32>(hi, cc_industry_FENCE_H + 0.06));
        let ta = max(max(sb.x, t0), 1e-3);
        let tb = min(min(sb.y, t1), best.t);
        if (sb.x > sb.y || ta > tb) {
            continue;
        }
        let g0 = select(1e30, mix(ln.y + 2.0, ln.z - 2.0 - cc_industry_GATE_W,
                                  fo.z), e == gate_e);
        var cand = best;
        // The top rail.
        let rlo = select(vec2<f32>(ln.x - rw, ln.y), vec2<f32>(ln.y, ln.x - rw),
                         along_x);
        let rhi = select(vec2<f32>(ln.x + rw, ln.z), vec2<f32>(ln.z, ln.x + rw),
                         along_x);
        cand = cc_industry_box(cand, o, dir, inv_dir, t0, t1,
                               vec3<f32>(rlo, cc_industry_FENCE_H - 0.05),
                               vec3<f32>(rhi, cc_industry_FENCE_H),
                               cc_industry_K_FENCE, ci);
        // The two posts nearest the ray's entry and exit of the slab.
        let sa = select((o + ta * dir).y, (o + ta * dir).x, along_x) - ln.y;
        let se = select((o + tb * dir).y, (o + tb * dir).x, along_x) - ln.y;
        let kmax = floor((ln.z - ln.y) / cc_industry_FENCE_POST);
        for (var j: i32 = 0; j < 2; j = j + 1) {
            let k = clamp(round(select(se, sa, j == 0)
                                / cc_industry_FENCE_POST), 0.0, kmax);
            let c = select(vec2<f32>(ln.x, ln.y + k * cc_industry_FENCE_POST),
                           vec2<f32>(ln.y + k * cc_industry_FENCE_POST, ln.x),
                           along_x);
            cand = cc_industry_box(
                cand, o, dir, inv_dir, t0, t1,
                vec3<f32>(c - vec2<f32>(pw), 0.0),
                vec3<f32>(c + vec2<f32>(pw), cc_industry_FENCE_H + 0.06),
                cc_industry_K_FENCE, ci);
        }
        // Nothing stands in the gate.
        if (cand.hit && cand.t < best.t) {
            let s = select(cand.pos.y, cand.pos.x, along_x);
            if (s < g0 || s > g0 + cc_industry_GATE_W) {
                best = cand;
            }
        }
    }
    return best;
}

// ============================================================================
// THE TRACE HOOK
// ============================================================================
fn cc_industry_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                           t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var best = cc_industry_miss(ci);
    // Gate 0, no arithmetic: most of the tile is urban, and open scrub
    // carries nothing of ours.
    let ind = cc.land_use == CITY_USE_INDUSTRIAL;
    if (!ind && cc.land_use != CITY_USE_YARD) {
        return best;
    }
    // Gate 1: the segment's z range against the tallest thing we own here.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    let z_lo = min(za, zb);
    let z_hi = max(za, zb);
    let top = select(cc_industry_YARD_TOP, cc.height + cc_industry_ROOF_TOP,
                     ind);
    if (z_lo > top) {
        return best;
    }
    // LOD decisions take the footprint at the thing they decide about (see
    // cc_industry_grid_hit, cc_industry_fixtures); fp at the segment's start
    // is only a lower bound, good for rejecting whole families of small
    // props at once.
    let px = cc_industry_px();
    let fp0 = px * max(t0, 0.0);

    // Roofs: the sawtooth and the clerestory here; a flat roof's skylights
    // are a grid, and go through the grid loop below as grid -1.
    var roof_kind = 0;
    if (ind && z_hi >= cc.height - 0.1) {
        let r = cc_industry_roof_of(cc);
        roof_kind = r.kind;
        if (r.kind == 1) {
            best = cc_industry_saw_trace(best, o, dir, inv_dir, t0, t1, cc, r,
                                         ci);
        } else if (r.kind == 2) {
            let sb = city_box_hit(o, inv_dir, r.mlo, r.mhi);
            if (sb.x <= sb.y && sb.x > max(t0, 1e-3) && sb.x <= t1) {
                let p = o + sb.x * dir;
                let n = city_box_normal(p, r.mlo, r.mhi);
                // The long faces are the clerestory glazing.
                let long_x = (r.mhi.x - r.mlo.x) >= (r.mhi.y - r.mlo.y);
                let glazed = abs(select(n.x, n.y, long_x)) > 0.5;
                best = CityHit(true, sb.x, p, n,
                               select(cc_industry_K_ROOF, cc_industry_K_GLAZE,
                                      glazed), ci);
            }
        }
    }

    // Flood fixtures: a wall-pack band on a warehouse, the poles in a yard.
    let fx_lo = select(0.0, CITY_FLOOD_WALL_Z - cc_industry_HEAD_WIDEN_MAX
                            - 0.1, ind);
    let fx_hi = select(cc_industry_YARD_TOP,
                       CITY_FLOOD_WALL_Z + cc_industry_HEAD_WIDEN_MAX + 0.2,
                       ind);
    if (z_hi >= fx_lo && z_lo <= fx_hi) {
        best = cc_industry_fixtures(best, o, dir, inv_dir, t0, t1, cc, ci, px);
    }

    // The grids: -1 is a flat roof's skylights, 0.. the ground props. One
    // loop, one call site, so the per-object detail is inlined once.
    let sky = ind && roof_kind == 0 && z_hi >= cc.height - 0.1
              && z_lo <= cc.height + cc_industry_SKY_H;
    let ground = z_lo <= cc_industry_PROP_TOP;
    let g_first = select(0, -1, sky);
    let g_end = select(0, select(5, cc_industry_N_GRIDS, ind), ground);
    for (var i: i32 = g_first; i < g_end; i = i + 1) {
        let g = cc_industry_grid(cc, i);
        if (g.ok) {
            best = cc_industry_grid_hit(best, o, dir, inv_dir, t0, t1, cc, g,
                                        ci, px, fp0);
        }
    }
    if (ground && !ind) {
        if (z_lo <= 2.4) {
            best = cc_industry_piles_hit(best, o, dir, t0, t1, cc, ci);
        }
        if (fp0 < cc_industry_FINE_FP && z_lo <= cc_industry_FENCE_H + 0.1) {
            best = cc_industry_fence_hit(best, o, dir, inv_dir, t0, t1, cc,
                                         ci);
        }
    }
    return best;
}

// ============================================================================
// LIGHT
// ============================================================================

fn cc_industry_fill(p: vec3<f32>) -> vec3<f32> {
    return CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
}

// What lights a prop surface, as the radiance a surface of ALB_REF would
// return: each of the lot's pools at the prop's own ground position — the
// beam's footprint there — times how squarely the surface faces the
// fixture, relative to the ground under it, and never more than the ground
// gets: a surface facing away catches none, one squarely facing catches
// what the ground does. The pools' peaks are calibrated for ground a head's
// height below the lamp; a container top two metres under it, lit with the
// inverse square, would take several times that, and first renders showed
// exactly that as flat white tops (the lead, 2026-09-23). Then the street
// lamps, for the yards that front one, falling mostly on up-facing
// surfaces.
fn cc_industry_light(cc: CityCell, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var e = vec3<f32>(0.0);
    let nk = select(0, CITY_FLOOD_MAX,
                    cc.land_use == CITY_USE_INDUSTRIAL
                    || cc.land_use == CITY_USE_YARD);
    for (var k: i32 = 0; k < nk; k = k + 1) {
        let f = city_flood(cc, k);
        if (f.on) {
            // @domain: a lamp head in the air minus a point on a surface: never zero
            let l = normalize(f.head - p);
            let facing = min(max(dot(n, l), 0.0) / max(l.z, 0.2), 1.0);
            e = e + f.color * (city_flood_pool(f, p.xy) * facing);
        }
    }
    let district = city_glow_sample(p.xy, 2.0);
    let raw = CITY_LAMP_RADIANCE * (0.20 + 1.3 * smoothstep(0.02, 0.45, district))
              * city_street_pools(p.xy);
    e = e + CITY_LAMP_COLOR * (raw / (1.0 + raw * 0.30))
            * (0.35 + 0.65 * max(n.z, 0.0));
    return e;
}

// The lamp-lit part passes a soft knee before it is returned: a bright
// painted box under a bright pool still keeps its shading instead of
// meeting the tone map's shoulder as a flat white face.
fn cc_industry_lit(cc: CityCell, p: vec3<f32>, n: vec3<f32>, alb: vec3<f32>)
        -> vec3<f32> {
    let moon = u.sun_dir.xyz;
    let lamp = (alb / cc_industry_ALB_REF) * cc_industry_light(cc, p, n);
    return lamp / (vec3<f32>(1.0) + lamp / cc_industry_LAMP_KNEE)
           + alb * (cc_industry_fill(p)
                    + CITY_MOONLIGHT * max(dot(n, moon), 0.0));
}

// A shading band that dissolves: `pattern` while the band is wider than a
// pixel, its own mean `mean` once it is not.
fn cc_industry_band(pattern: f32, mean: f32, width: f32, fp: f32) -> f32 {
    return mix(pattern, mean, smoothstep(0.5 * width, 2.0 * width, fp));
}

// ============================================================================
// ROOF SHADING, AND THE ROOF HOOK
// ============================================================================

// Glazing's mean emission over a whole glazed face of height `hgt`: the lit
// bays at their mean level, the glass between mullions, and the glass band
// between the sill and head flashing. The shader's far field and the roof
// hook both take it from here, so they cannot disagree.
fn cc_industry_glaze_mean(r: cc_industry_Roof, hgt: f32) -> vec3<f32> {
    if (!r.lit) {
        return vec3<f32>(0.0);
    }
    return r.tint * (cc_industry_GLAZE_RAD * cc_industry_BAY_LIT
                     * cc_industry_BAY_MEAN * cc_industry_LAMP_MEAN
                     * cc_industry_glass_frac(hgt));
}

// The fraction of a glazed face that is glass: the band between the sill and
// head flashing, less a transom across it and the mullions down it.
fn cc_industry_glass_frac(hgt: f32) -> f32 {
    let pane = max(hgt - cc_industry_SILL - cc_industry_HEAD_BAND, 0.0);
    return (pane - cc_industry_TRANSOM_W) / hgt
           * (1.0 - cc_industry_MULL_W / cc_industry_MULL);
}

fn cc_industry_glaze_shade(h: CityHit, cc: CityCell, fp: f32) -> vec3<f32> {
    let r = cc_industry_roof_of(cc);
    // @domain: a wall's normal: |xy| = 1
    let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
    let uc = dot(h.pos.xy, vec2<f32>(-nh.y, nh.x));
    let v = h.pos.z - cc.height;
    let hgt = select(r.mhi.z - cc.height, r.rise, r.kind == 1);
    // Unlit glass: a dark pane holding a little of the city's glow.
    let base = CITY_SKYGLOW * 0.8 + 0.03 * cc_industry_fill(h.pos);
    if (!r.lit) {
        return base;
    }
    // Which face (a sawtooth's tooth, or a clerestory side) and which
    // lighting bay along it: the high-bays are switched in zones.
    let face_id = i32(round(dot(h.pos.xy, nh) * 0.5));
    let bay = i32(floor(uc / cc_industry_BAY));
    let dl = cc_industry_draw(cc, 14u, face_id, bay);
    let bay_on = select(0.0, 0.4 + dl.y, dl.x < cc_industry_BAY_LIT);
    // The frame, as it was built: flashing bands, a transom, mullions. Each
    // goes to its own mean as it goes sub-pixel, so what is left at range is
    // exactly cc_industry_glass_frac.
    let mid = 0.5 * (cc_industry_SILL + hgt - cc_industry_HEAD_BAND);
    let band = select(0.0, 1.0,
                      v > cc_industry_SILL && v < hgt - cc_industry_HEAD_BAND
                      && abs(v - mid) > 0.5 * cc_industry_TRANSOM_W);
    let band_k = cc_industry_band(
        band, cc_industry_glass_frac(hgt)
              / (1.0 - cc_industry_MULL_W / cc_industry_MULL),
        cc_industry_TRANSOM_W, fp);
    let mull = select(1.0, 0.0,
                      fract(uc / cc_industry_MULL)
                      < cc_industry_MULL_W / cc_industry_MULL);
    let mull_k = cc_industry_band(mull,
                                  1.0 - cc_industry_MULL_W / cc_industry_MULL,
                                  cc_industry_MULL_W, fp);
    let dq = (fract(uc / cc_industry_LAMP_P) - 0.5) * cc_industry_LAMP_P;
    let spot = cc_industry_LAMP_FLOOR + (1.0 - cc_industry_LAMP_FLOOR)
        * exp(-0.5 * dq * dq / (cc_industry_LAMP_SIG * cc_industry_LAMP_SIG));
    let spot_k = cc_industry_band(spot, cc_industry_LAMP_MEAN,
                                  cc_industry_LAMP_SIG, fp);
    let near = r.tint * (cc_industry_GLAZE_RAD * bay_on * band_k * mull_k
                         * spot_k);
    let far = cc_industry_glaze_mean(r, hgt) * cc_industry_GLAZE_MEAN_COMP;
    let k = smoothstep(cc_industry_GLAZE_LOD_START, cc_industry_GLAZE_LOD_FULL,
                       fp);
    return base + mix(near, far, k);
}

fn cc_industry_roof_shade(h: CityHit, cc: CityCell, fp: f32) -> vec3<f32> {
    let moon = u.sun_dir.xyz;
    let light = cc_industry_fill(h.pos)
                + CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    if (abs(h.normal.z) < 0.3) {
        // A gable end or a clerestory's end wall: the cladding carried up.
        return cc_industry_CLAD_ALB * light;
    }
    // Sheet roofing: standing seams running down the slope, and on a
    // sawtooth the dark gutter in every valley.
    let r = cc_industry_roof_of(cc);
    let across = select(h.pos.x, h.pos.y, r.kind == 1 && r.step_x);
    let seam = select(1.0, 1.6, fract(across / cc_industry_SEAM) < 0.08);
    var m = cc_industry_band(seam, 1.048, 0.05, fp);
    if (r.kind == 1) {
        let v = h.pos.z - cc.height;
        m = m * cc_industry_band(select(1.0, 0.35, v < 0.12), 0.97,
                                 0.12 * r.pitch / r.rise, fp);
    }
    return cc_industry_ROOF_ALB * m * light;
}

fn cc_industry_sky_mean(cc: CityCell, g: cc_industry_Grid,
                        r: cc_industry_Roof) -> vec3<f32> {
    let cover = (g.obj.x * g.obj.y) / (g.cell.x * g.cell.y);
    return select(vec3<f32>(0.0), r.tint * (cc_industry_SKY_RAD * cover),
                  r.lit);
}

fn cc_industry_sky_shade(h: CityHit, cc: CityCell, fp: f32) -> vec3<f32> {
    let g = cc_industry_sky_grid(cc);
    let r = cc_industry_roof_of(cc);
    let rel = h.pos.xy - g.lo.xy;
    let ij = vec2<i32>(floor(rel / g.cell));
    let loc = rel - (vec2<f32>(ij) + 0.5) * g.cell;
    let inside = all(abs(loc) < 0.5 * g.obj);
    let light = cc_industry_fill(h.pos)
                + CITY_MOONLIGHT * max(dot(h.normal, u.sun_dir.xyz), 0.0);
    // The pattern: a glowing panel (its level drawn per skylight) inside a
    // skylight's footprint, dark curb on its sides, roof between.
    let dl = cc_industry_draw(cc, 15u, ij.x, ij.y);
    var pat = cc_industry_ROOF_ALB * light;
    if (inside) {
        pat = select(0.04 * light, 0.25 * light, h.normal.z > 0.5);
        if (r.lit && h.normal.z > 0.5) {
            pat = pat + r.tint * (cc_industry_SKY_RAD * (0.6 + 0.8 * dl.y));
        }
    }
    let cover = (g.obj.x * g.obj.y) / (g.cell.x * g.cell.y);
    let mean = cc_industry_sky_mean(cc, g, r) * cc_industry_GLAZE_MEAN_COMP
               + light * (cc_industry_ROOF_ALB * (1.0 - cover) + 0.25 * cover);
    return mix(pat, mean, smoothstep(1.0, 4.0, fp));
}

// The roof hook: a warehouse roof hit by the CORE — which, wherever this
// component's roof geometry is drawn, is never the case on a sawtooth (the
// teeth cover the deck) and is the roof around a clerestory or between
// skylights. So the hook adds nothing where the geometry was drawn, and
// beyond CITY_PROP_RANGE it adds the geometry's exact mean.
//
// "Drawn" is the core's own test: props run for a cell whose ray segment
// starts inside CITY_PROP_RANGE, and that start is where the ray enters the
// hit cell's column, or enters the city slab from above if that is later
// (city_trace's t_cur) — recomputed here from the camera, since the hook is
// not handed the ray.
//
// The sawtooth mean is view-dependent, and exactly so. In the tooth's own
// (s, z) plane a ray descending with slope m = |dz| / |ds| against the
// glazing's facing sees, of each face of height H, the part not hidden by
// the next tooth's rising sheet: min(H, m p). Per tooth pitch p of roof the
// rays that land there span p along s, and those striking glass span that
// visible height over m. So the glass covers a fraction min(H / (m p), 1)
// of the roof as seen along the ray — zero from the facing's back, the whole
// roof at grazing — and the mean emission is that fraction of the face mean.
// A clerestory is the same argument with one long face in view and no
// occluder: its projected strip, length x height x |d_c| / |dz|, over the
// roof's area.
fn cc_industry_roof(cc: CityCell, h: CityHit, fp: f32) -> vec3<f32> {
    if (cc.land_use != CITY_USE_INDUSTRIAL) {
        return vec3<f32>(0.0);
    }
    let r = cc_industry_roof_of(cc);
    if (!r.lit) {
        return vec3<f32>(0.0);
    }
    let o = city_camera_origin();
    let dir = (h.pos - o) / max(h.t, 1e-3);
    let cell = u.ocean_params.x;
    let inv = vec3<f32>(select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-9),
                        select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-9),
                        select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-9));
    let cmin = vec3<f32>(vec2<f32>(h.cell) * cell, -1e6);
    let col = city_box_hit(o, inv, cmin, cmin + vec3<f32>(cell, cell, 2e6));
    var t_slab = 0.0;
    if (o.z > CITY_SLAB_TOP) {
        t_slab = (CITY_SLAB_TOP - o.z) * inv.z;
    }
    if (max(col.x, t_slab) < CITY_PROP_RANGE) {
        return vec3<f32>(0.0);
    }
    let dz = max(abs(dir.z), 1e-3);
    if (r.kind == 1) {
        let d_s = r.face * select(dir.y, dir.x, r.step_x);
        if (d_s >= 0.0) {
            return vec3<f32>(0.0);
        }
        let frac = min(r.rise * (-d_s) / (dz * r.pitch), 1.0);
        return cc_industry_glaze_mean(r, r.rise)
               * (frac * cc_industry_GLAZE_MEAN_COMP);
    }
    if (r.kind == 2) {
        let long_x = (r.mhi.x - r.mlo.x) >= (r.mhi.y - r.mlo.y);
        let len = select(r.mhi.y - r.mlo.y, r.mhi.x - r.mlo.x, long_x);
        let d_c = abs(select(dir.x, dir.y, long_x));
        let hm = r.mhi.z - cc.height;
        let area = (cc.b1max.x - cc.b1min.x) * (cc.b1max.y - cc.b1min.y);
        // Unlike a sawtooth's, a lone clerestory face has no next tooth to
        // hide it, and its projection can outgrow the roof it is averaged
        // over; the cap only stops a grazing ray from turning a sliver of
        // roof into a searchlight.
        let frac = min(len * hm * d_c / (dz * area), 3.0);
        return cc_industry_glaze_mean(r, hm)
               * (frac * cc_industry_GLAZE_MEAN_COMP);
    }
    let g = cc_industry_sky_grid(cc);
    if (!g.ok) {
        return vec3<f32>(0.0);
    }
    let garea = f32(g.n.x * g.n.y) * g.cell.x * g.cell.y;
    let area = (cc.b1max.x - cc.b1min.x) * (cc.b1max.y - cc.b1min.y);
    return cc_industry_sky_mean(cc, g, r)
           * (cc_industry_GLAZE_MEAN_COMP * garea / area);
}

// ============================================================================
// FIXTURE AND WALL-WASH SHADING
// ============================================================================

// The fixture nearest a hit point: the one whose head it is on.
fn cc_industry_nearest_flood(cc: CityCell, p: vec3<f32>) -> CityFlood {
    var best = city_flood(cc, 0);
    var bd = 1e30;
    for (var k: i32 = 0; k < CITY_FLOOD_MAX; k = k + 1) {
        let f = city_flood(cc, k);
        if (f.on) {
            let d = distance(cc_industry_head_c(cc, f), p);
            if (d < bd) {
                bd = d;
                best = f;
            }
        }
    }
    return best;
}

fn cc_industry_head_shade(h: CityHit, cc: CityCell) -> vec3<f32> {
    let f = cc_industry_nearest_flood(cc, h.pos);
    let c = cc_industry_head_c(cc, f);
    // The footprint at the head, exactly as the trace sized the box.
    let fp = cc_industry_px() * distance(city_camera_origin(), c);
    let real = cc_industry_head_real(cc);
    let hw = cc_industry_head_hw(cc, fp);
    // The lens: the underside, and the lower half of the face toward the
    // aim — a flood is tilted down into its pool. Housing everywhere else.
    let fwd = dot(h.normal.xy, f.aim) > 0.5;
    let lens = h.normal.z < -0.5 || (fwd && h.pos.z < c.z);
    let e_lens = f.color * (cc_industry_LENS_RAD
                            / max(max(f.color.r, f.color.g), f.color.b));
    let housing = 0.06 * cc_industry_fill(h.pos) + 0.02 * f.color;
    let near = select(housing, e_lens, lens);
    // Widened toward a pixel: the whole box glows at the lens's mean over
    // the housing (about half of what is seen), diluted by the area it grew
    // — the sum of its three face areas, so the dilution holds from any
    // side.
    let a_real = real.x * real.y + real.y * real.z + real.x * real.z;
    let a_wide = hw.x * hw.y + hw.y * hw.z + hw.x * hw.z;
    let far = e_lens * (0.5 * a_real / a_wide);
    return mix(near, far, smoothstep(0.6 * real.x / cc_industry_HEAD_WIDEN,
                                     real.x / cc_industry_HEAD_WIDEN, fp));
}

// ============================================================================
// THE WAREHOUSE WALLS (the facade hook)
// ============================================================================
//
// city-facades handed the warehouse walls to this component (2026-09-23):
// the core draws no windows in their ground band and adds no reflectance of
// its own on them, so what a warehouse wall looks like is decided here, at
// every range — the facade hook runs wherever a wall is hit. A wall is:
//
//   * CLADDING: corrugated metal sheet (ribs every 0.2 m, as shading bands,
//     over a concrete plinth the bottom metre) on two thirds of buildings,
//     tilt-up concrete panels 7-9 m wide with their joints on the rest. Real
//     albedo, lit by the skyglow fill, the moon, the court's own wall packs
//     (the scallop each throws down the wall under it), the lamp over each
//     personnel door, and the street lamps where a street runs past.
//   * DOCK DOORS on the court wall, one per dock slot (cc_industry_docks),
//     so they line up with the trailers: sectional doors on the dock floor
//     with a row of vision panels, most dark, a few with light leaking
//     through, and some open onto a lit interior — mostly where a trailer
//     is backed in, on a building working a night shift. The shelters,
//     bumpers and leveller are relief geometry (cc_industry_dock_part); the
//     wall under the shelter is painted in its colour, so at range, where
//     the geometry is no longer traced, the frame is still there.
//   * PERSONNEL DOORS, one or two per other wall, each with a small lamp
//     over it and the lamp's own scallop of light.
//   * A painted company band near the top of some buildings, and grime in
//     the bottom metre.
//
// Every sub-pixel pattern (ribs, panel joints, door seams, vision panels)
// dissolves to its own mean; a dock door past a few metres per pixel is its
// cell's mean emission; a lamp is widened toward a pixel at constant energy.
//
// FACE FRAME. `s` below is metres along the wall from the building box's
// min corner (x or y), computed here from b1. When city-facades' city_face_at
// lands, the hook's `uc` becomes metres from the face's start corner and
// replaces cc_industry_wall_s — the only place the frame is taken. Their
// coming `facade_cover` hook is where the dock and personnel doors should
// also clear the windows and relief under them (today the core's door-band
// cut does that for the windows).

struct cc_industry_Wall {
    concrete: bool,
    alb: vec3<f32>,
    panel_w: f32,        // tilt-up panel width
    band: vec2<f32>,     // painted band's z range (empty: none)
    band_col: vec3<f32>,
    door_col: vec3<f32>,
    shelter: bool,
    lamp_col: vec3<f32>, // the personnel doors' lamps
}

fn cc_industry_wall_of(cc: CityCell) -> cc_industry_Wall {
    var w: cc_industry_Wall;
    let d = cc_industry_draw(cc, 91u, 0, 0);
    let e = cc_industry_draw(cc, 92u, 0, 0);
    w.concrete = d.x < 0.35;
    // Cladding colours, dimmed as they read under a night sky: galvanised
    // grey, beige, the blue-grey and green of profiled sheet, off-white.
    var c = vec3<f32>(0.34, 0.35, 0.36);
    if (d.y > 0.30) { c = vec3<f32>(0.36, 0.33, 0.27); }
    if (d.y > 0.50) { c = vec3<f32>(0.22, 0.27, 0.33); }
    if (d.y > 0.70) { c = vec3<f32>(0.14, 0.21, 0.16); }
    if (d.y > 0.85) { c = vec3<f32>(0.45, 0.45, 0.44); }
    w.alb = select(c, vec3<f32>(0.30, 0.29, 0.27), w.concrete);
    w.panel_w = 7.0 + 2.0 * d.z;
    let top = cc.height - (0.6 + 1.2 * e.x);
    w.band = select(vec2<f32>(1e3, -1e3), vec2<f32>(top - 1.0 - e.y, top),
                    d.w < 0.45);
    var b = vec3<f32>(0.30, 0.05, 0.04);                         // red
    if (e.z > 0.30) { b = vec3<f32>(0.05, 0.12, 0.30); }         // blue
    if (e.z > 0.55) { b = vec3<f32>(0.06, 0.20, 0.09); }         // green
    if (e.z > 0.75) { b = vec3<f32>(0.42, 0.18, 0.03); }         // orange
    if (e.z > 0.90) { b = vec3<f32>(0.45, 0.35, 0.04); }         // yellow
    w.band_col = b;
    var dc = vec3<f32>(0.42, 0.42, 0.41);                        // white
    if (e.w > 0.40) { dc = vec3<f32>(0.24, 0.24, 0.25); }        // grey
    if (e.w > 0.70) { dc = vec3<f32>(0.07, 0.13, 0.28); }        // blue
    if (e.w > 0.90) { dc = vec3<f32>(0.42, 0.30, 0.05); }        // yellow
    w.door_col = dc;
    w.shelter = fract(d.w * 7.13) < 0.60;
    w.lamp_col = select(vec3<f32>(0.80, 0.90, 1.00), vec3<f32>(1.0, 0.60, 0.28),
                        fract(e.x * 5.31) < 0.55);
    return w;
}

// (metres along the wall from the box's min corner, the wall's length) —
// see FACE FRAME above.
fn cc_industry_wall_s(cc: CityCell, p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let ext = cc.b1max.xy - cc.b1min.xy;
    if (abs(n.x) > abs(n.y)) {
        return vec2<f32>(p.y - cc.b1min.y, ext.y);
    }
    return vec2<f32>(p.x - cc.b1min.x, ext.x);
}

// A lamp's scallop of light on the wall below it: narrow at the lamp,
// widening with the drop d; rising over `rise` (the housing shades the wall
// right behind it) and fading over `fade`.
fn cc_industry_wash(d: f32, du: f32, w0: f32, w1: f32, rise: f32,
                    fade: f32) -> f32 {
    if (d <= 0.0) {
        return 0.0;
    }
    let w = w0 + w1 * d;
    return (1.0 - exp(-d / rise)) * exp(-d / fade)
           * exp(-0.5 * du * du / (w * w));
}

// The integral of e^{-k d} (w0 + w1 d) over d in [0, z].
fn cc_industry_wash_int(k: f32, z: f32, w0: f32, w1: f32) -> f32 {
    let e = exp(-k * z);
    return w0 * (1.0 - e) / k + w1 * (1.0 - e * (1.0 + k * z)) / (k * k);
}

// cc_industry_wash's integral over the wall, from a lamp at height z.
fn cc_industry_wash_flux(z: f32, w0: f32, w1: f32, rise: f32, fade: f32)
        -> f32 {
    let kf = 1.0 / fade;
    return 2.5066283 * (cc_industry_wash_int(kf, z, w0, w1)
                        - cc_industry_wash_int(kf + 1.0 / rise, z, w0, w1));
}

// Street light on a wall, in ground-pool units: the core's pools at the
// wall's foot, and past the pools' own footprint LOD their mean along the
// wall — a periodic trapezoid rule over one lamp spacing, which for these
// Gaussians (sigma 3.2 m, 4 samples 6.5 m apart) is good to 2%. Lamp lines
// crossing the wall's line do not vary along it, so the rule holds for
// them trivially.
fn cc_industry_street_wall(p: vec2<f32>, tang: vec2<f32>, fp: f32) -> f32 {
    let near = city_street_pools(p);
    let k = smoothstep(CITY_STREET_LOD_START, CITY_STREET_LOD_FULL, fp);
    let nq = select(0, 4, k > 0.0);
    var m = 0.0;
    for (var q: i32 = 0; q < nq; q = q + 1) {
        m = m + city_street_pools(
            p + tang * (f32(q) * 0.25 * CITY_LAMP_SPACING));
    }
    return mix(near, 0.25 * m, k);
}

// A personnel door on wall `face` (0 -x, 1 +x, 2 -y, 3 +y): (position along
// the wall, lamp on (1) or not (0), present (1) or not (0)).
fn cc_industry_pdoor(cc: CityCell, face: i32, k: i32, len: f32) -> vec3<f32> {
    let dk = cc_industry_draw(cc, 93u, face, k);
    let present = select(len > 12.0, len > 40.0 && dk.x < 0.5, k == 1);
    // Two doors keep to their own halves of the wall.
    let lo = select(3.0, 0.5 * len + 1.0, k == 1);
    let hi = select(select(len - 3.0, 0.5 * len - 1.0, len > 40.0),
                    len - 3.0, k == 1);
    return vec3<f32>(mix(lo, hi, dk.y), select(0.0, 1.0, dk.z < 0.8),
                     select(0.0, 1.0, present));
}

const cc_industry_PDOOR_W: f32 = 1.0;
const cc_industry_PDOOR_H: f32 = 2.1;
const cc_industry_PLAMP_Z: f32 = 2.45;
const cc_industry_PLAMP_R: f32 = 0.10;       // lens radius
const cc_industry_PLAMP_RAD: f32 = 12.0;
const cc_industry_PLAMP_GAIN: f32 = 0.35;    // its wash, in pool units
const cc_industry_OPEN_RAD: f32 = 1.0;       // an open dock door's interior
const cc_industry_LEAK_RAD: f32 = 0.8;       // a vision panel, lit inside
const cc_industry_SEAM_RAD: f32 = 0.25;      // light through a door seam

// A dock door's state: .x open, .y leaking light, .z its level draw.
fn cc_industry_door_state(cc: CityCell, l: cc_industry_Lot, j: i32,
                          lit: bool) -> vec3<f32> {
    let dd = cc_industry_draw(cc, 90u, j, 0);
    // The trailer backed onto this door, by the dock grid's own draw.
    let ts = cc_industry_draw(cc, 20u, select(0, j, l.long_x),
                              select(j, 0, l.long_x));
    let trailer = cc_industry_has_trailers(l) && ts.x < 0.45;
    let open = (lit && trailer && dd.x < 0.55) || dd.x < 0.04;
    let leak = !open && dd.y < select(0.08, 0.35, lit);
    return vec3<f32>(select(0.0, 1.0, open), select(0.0, 1.0, leak), dd.z);
}

// Vision panels in a sectional door: four, 0.4 m square, in the fourth
// section.
fn cc_industry_vision(dv: f32, dz: f32) -> bool {
    let a = abs(dv);
    return dz > 1.90 && dz < 2.30
           && (abs(a - 0.35) < 0.20 || abs(a - 1.05) < 0.20);
}
// Their share of a door's face, and the seams' (5 of 0.03 m).
const cc_industry_VISION_FRAC: f32 = 0.0711;   // 4 x 0.4 x 0.4 / 9
const cc_industry_SEAM_FRAC: f32 = 0.05;       // 5 x 0.03 / 3

// GRAFFITI (city_graffiti_paint). Each wall of a warehouse is a run, u from
// its left end as you stand facing it, v from the ground: the court wall
// (CITY_GRAFFITI_WALL_COURT) and the others (WALL_STREET), the personnel
// doors and the dock wall under the dock doors painted with the wall they
// are in. A closed dock door is a run of its own (CITY_GRAFFITI_DOOR), v
// from the dock floor its painter stands on; an open one is rolled up, and
// its paint with it.
fn cc_industry_wall_run(cc: CityCell, face: i32, len: f32, court: bool)
        -> CityGraffitiRun {
    var run: CityGraffitiRun;
    run.seed = pcg2d(cc.seed ^ vec2<u32>(0x5bd1e995u,
                                         u32(face) * 0x9e3779b9u + 0x27d4eb2fu));
    run.len = len;
    run.kind = select(CITY_GRAFFITI_WALL_STREET, CITY_GRAFFITI_WALL_COURT,
                      court);
    return run;
}

fn cc_industry_door_run(cc: CityCell, j: i32) -> CityGraffitiRun {
    var run: CityGraffitiRun;
    run.seed = pcg2d(cc.seed ^ vec2<u32>(u32(j) * 0x85ebca6bu + 0x68e31da4u,
                                         0xb5297a4du));
    run.len = cc_industry_DOOR_W;
    run.kind = CITY_GRAFFITI_DOOR;
    return run;
}

fn cc_industry_paint(alb: vec3<f32>, g: CityGraffitiPaint) -> vec3<f32> {
    return alb * g.keep + g.add;
}

// The frame a wall's paint is taken in: (u from its left end, its length).
fn cc_industry_wall_u(cc: CityCell, p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let ws = cc_industry_wall_s(cc, p, n);
    let along = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
                       abs(n.x) > abs(n.y));
    return vec2<f32>(select(ws.y - ws.x, ws.x,
                            dot(vec2<f32>(-n.y, n.x), along) > 0.0), ws.y);
}

// A dock door on the court wall at p: (door index, offset from its centre
// along the wall's u, in its face (1) or not (0)) — the doors' own test in
// cc_industry_facade, and their state's, so the paint picks the run the
// door section will show.
fn cc_industry_door_at(cc: CityCell, l: cc_industry_Lot, p: vec3<f32>,
                       tang: vec2<f32>) -> vec3<f32> {
    let docks = cc_industry_docks(l);
    let a = select(p.y, p.x, l.long_x);
    let jf = round((a - docks.x) / cc_industry_DOCK_PITCH);
    let j = i32(jf);
    let dv = a - (docks.x + jf * cc_industry_DOCK_PITCH);
    let du = dot(tang, select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0),
                              l.long_x)) * dv;
    let in_face = j >= 0 && j < i32(docks.y)
        && abs(dv) < 0.5 * cc_industry_DOOR_W && p.z > cc_industry_DOCK_H
        && p.z < cc_industry_DOCK_H + cc_industry_DOOR_H;
    return vec3<f32>(jf, du, select(0.0, 1.0, in_face));
}

// The pixel's paint: a closed dock door's own where the pixel is on one,
// the wall's everywhere else. Taken first thing in cc_industry_facade, one
// paint a pixel, while little else is live: taken after the wall's look
// and the doors', it raised the city's first stage from 128 to 168
// registers (tools/soar_regcount.py), which cost every city pixel.
fn cc_industry_wall_graffiti(cc: CityCell, h: CityHit, dir: vec3<f32>,
                             fp: f32, court: bool, face: i32)
        -> CityGraffitiPaint {
    let n = h.normal;
    let tang = vec2<f32>(-n.y, n.x);
    let fuv = city_graffiti_footprint(fp, n, dir, vec3<f32>(tang, 0.0),
                                      vec3<f32>(0.0, 0.0, 1.0));
    let wu = cc_industry_wall_u(cc, h.pos, n);
    var run = cc_industry_wall_run(cc, face, wu.y, court);
    var u = wu.x;
    var v = h.pos.z;
    if (court) {
        let l = cc_industry_lot(cc);
        let dd = cc_industry_door_at(cc, l, h.pos, tang);
        if (dd.z > 0.5) {
            let j = i32(dd.x);
            let st = cc_industry_door_state(cc, l, j,
                                            cc_industry_roof_of(cc).lit);
            if (st.x < 0.5) {
                run = cc_industry_door_run(cc, j);
                u = dd.y + 0.5 * cc_industry_DOOR_W;
                v = h.pos.z - cc_industry_DOCK_H;
            }
        }
    }
    return city_graffiti_paint(run, u, v, fuv);
}

fn cc_industry_facade(cc: CityCell, h: CityHit, dir: vec3<f32>, uc: f32,
                      vc: f32, fp: f32) -> vec3<f32> {
    if (cc.land_use != CITY_USE_INDUSTRIAL) {
        return vec3<f32>(0.0);
    }
    let n = h.normal;
    let l = cc_industry_lot(cc);
    let out = cc_industry_xy(l, vec2<f32>(0.0, select(-1.0, 1.0, l.court_hi)));
    let court = dot(n.xy, out) > 0.9;
    let face = select(select(2, 3, n.y > 0.0), select(0, 1, n.x > 0.0),
                      abs(n.x) > abs(n.y));
    // The paint first (cc_industry_wall_graffiti): applied below, over the
    // wall's grime or a closed door's face.
    let pnt = cc_industry_wall_graffiti(cc, h, dir, fp, court, face);
    let w = cc_industry_wall_of(cc);
    let tang = vec2<f32>(-n.y, n.x);
    let ws = cc_industry_wall_s(cc, h.pos, n);
    let sw = ws.x;

    // --- cladding ------------------------------------------------------------
    var alb = w.alb;
    if (w.concrete) {
        let pi = floor(sw / w.panel_w);
        let tone = 0.90 + 0.20 * hash12(vec2<f32>(pi, f32(face) * 17.0
                                                   + f32(cc.seed.x & 1023u)));
        let fj = fract(sw / w.panel_w) * w.panel_w;
        let joint = select(1.0, 0.45, fj < 0.025 || fj > w.panel_w - 0.025);
        alb = alb * (tone * cc_industry_band(
            joint, 1.0 - 0.05 / w.panel_w * 0.55, 0.05, fp));
    } else {
        let rib = 0.78 + 0.44 * (0.5 + 0.5 * cos(6.2831853 * sw / 0.2));
        alb = alb * cc_industry_band(rib, 1.0, 0.1, fp);
        // The sheets' side laps, one per 1 m cover width: the rhythm that
        // still reads once the ribs have gone to their mean.
        let lap = select(1.0, 0.55, fract(sw) < 0.04);
        alb = alb * cc_industry_band(lap, 1.0 - 0.04 * 0.45, 0.04, fp);
        if (vc < 1.0) {
            alb = vec3<f32>(0.26, 0.25, 0.23);          // concrete plinth
        }
    }
    if (vc > w.band.x && vc < w.band.y) {
        alb = w.band_col * (alb / max(w.alb, vec3<f32>(1e-3)));
    }
    alb = alb * (0.6 + 0.4 * smoothstep(0.0, 1.0, vc));   // grime
    // The eave flashing: a 0.3 m band of trim along the top of the wall.
    if (vc > cc.height - 0.3) {
        alb = alb * 0.55;
    }

    // --- doors ---------------------------------------------------------------
    var emit = vec3<f32>(0.0);
    var light = vec3<f32>(0.0);
    let r = cc_industry_roof_of(cc);
    if (court) {
        let docks = cc_industry_docks(l);
        let a = select(h.pos.y, h.pos.x, l.long_x);
        let jf = round((a - docks.x) / cc_industry_DOCK_PITCH);
        let j = i32(jf);
        let dv = a - (docks.x + jf * cc_industry_DOCK_PITCH);
        let top = cc_industry_DOCK_H + cc_industry_DOOR_H;
        let hd = 0.5 * cc_industry_DOOR_W;
        if (j >= 0 && j < i32(docks.y)) {
            let st = cc_industry_door_state(cc, l, j, r.lit);
            let in_door = abs(dv) < hd && vc > cc_industry_DOCK_H && vc < top;
            let dz = vc - cc_industry_DOCK_H;
            // The door's own look, and its emission.
            var d_alb = w.door_col;
            var d_emit = vec3<f32>(0.0);
            let seam = fract(dz / 0.6) * 0.6 < 0.03;
            let vis = cc_industry_vision(dv, dz);
            let level = 0.55 + 0.45 * st.z;
            if (st.x > 0.5) {
                // Open: the lit interior, the dock floor's lip at its foot.
                d_alb = vec3<f32>(0.02);
                d_emit = r.tint * (cc_industry_OPEN_RAD * level
                                   * smoothstep(0.0, 0.3, dz));
            } else {
                d_alb = d_alb * cc_industry_band(
                    select(1.0, 0.45, seam), 1.0 - 0.55 * cc_industry_SEAM_FRAC,
                    0.03, fp);
                if (vis) {
                    d_alb = vec3<f32>(0.04);
                }
                if (st.y > 0.5) {
                    let pat = select(0.0, cc_industry_LEAK_RAD, vis)
                        + cc_industry_band(select(0.0, cc_industry_SEAM_RAD,
                                                  seam),
                                           cc_industry_SEAM_RAD
                                           * cc_industry_SEAM_FRAC, 0.03, fp);
                    d_emit = r.tint * (level * pat);
                }
            }
            // The door's mean over its face, for the far field below.
            let d_mean = select(
                r.tint * (level * st.y * (cc_industry_LEAK_RAD
                                          * cc_industry_VISION_FRAC
                                          + cc_industry_SEAM_RAD
                                          * cc_industry_SEAM_FRAC)),
                r.tint * (cc_industry_OPEN_RAD * level * 0.95), st.x > 0.5);
            // The frame around it: the shelter's fabric where there is one
            // (under its geometry close in, standing in for it far out), a
            // steel jamb where not; the dock wall under the door.
            let rim = select(0.15, 0.50, w.shelter);
            let in_rim = abs(dv) < hd + rim && vc > 1.0
                         && vc < top + select(0.15, 0.75, w.shelter);
            var p_alb = alb;
            if (in_rim) {
                p_alb = select(vec3<f32>(0.10), vec3<f32>(0.035), w.shelter);
            }
            if (abs(dv) < hd + rim && vc <= cc_industry_DOCK_H) {
                p_alb = vec3<f32>(0.12);
            }
            var p_emit = vec3<f32>(0.0);
            // The paint: a closed door's own on its face (an open door is
            // rolled up, and its paint with it), the wall's on everything
            // else, the shelter and the dock wall included.
            let closed = st.x < 0.5;
            if (in_door) {
                p_alb = select(d_alb, cc_industry_paint(d_alb, pnt), closed);
                p_emit = d_emit;
            } else {
                p_alb = cc_industry_paint(p_alb, pnt);
            }
            // Past a few metres per pixel a door is its dock cell's mean.
            let k = smoothstep(1.0, 3.0, fp);
            if (vc > cc_industry_DOCK_H && vc < top) {
                let fd = cc_industry_DOOR_W / cc_industry_DOCK_PITCH;
                let fr = 2.0 * rim / cc_industry_DOCK_PITCH;
                let rim_alb = select(vec3<f32>(0.10), vec3<f32>(0.035),
                                     w.shelter);
                // The paint's means over the cell: the door's over its
                // share, the wall's over the rest.
                let fv = city_graffiti_footprint(
                    fp, n, dir, vec3<f32>(tang, 0.0),
                    vec3<f32>(0.0, 0.0, 1.0)).y;
                var door_far = w.door_col;
                if (closed) {
                    door_far = cc_industry_paint(door_far, city_graffiti_mean(
                        cc_industry_door_run(cc, j), dz, fv));
                }
                let pm = city_graffiti_mean(
                    cc_industry_wall_run(cc, face, ws.y, court), vc, fv);
                let far_alb = door_far * fd
                    + (rim_alb * fr + alb * (1.0 - fd - fr)) * pm.keep
                    + pm.add * (1.0 - fd);
                alb = mix(p_alb, far_alb, k);
                emit = mix(p_emit, d_mean * fd, k);
            } else {
                alb = p_alb;
            }
        } else {
            alb = cc_industry_paint(alb, pnt);
        }
        // The court's wall packs, each washing the wall under it.
        let k_far = smoothstep(CITY_STREET_LOD_START, CITY_STREET_LOD_FULL, fp);
        var near = vec3<f32>(0.0);
        var flux = vec3<f32>(0.0);
        for (var q: i32 = 0; q < CITY_FLOOD_MAX; q = q + 1) {
            let f = city_flood(cc, q);
            if (!f.on) {
                continue;
            }
            let gain = f.color * cc_industry_WASH_GAIN;
            near = near + gain * cc_industry_wash(
                f.head.z - vc, dot(h.pos.xy - f.head.xy, tang),
                cc_industry_WASH_W0, cc_industry_WASH_W1,
                cc_industry_WASH_RISE, cc_industry_WASH_FADE);
            flux = flux + gain * cc_industry_wash_flux(
                f.head.z, cc_industry_WASH_W0, cc_industry_WASH_W1,
                cc_industry_WASH_RISE, cc_industry_WASH_FADE);
        }
        light = mix(near, flux / (ws.y * cc.height), k_far);
    } else {
        // Personnel doors, each with its lamp and the lamp's own wash.
        let k_far = smoothstep(CITY_STREET_LOD_START, CITY_STREET_LOD_FULL, fp);
        var near = vec3<f32>(0.0);
        var flux = vec3<f32>(0.0);
        let np = select(0, 2, ws.y > 12.0);
        for (var q: i32 = 0; q < np; q = q + 1) {
            let pd = cc_industry_pdoor(cc, face, q, ws.y);
            if (pd.z < 0.5) {
                continue;
            }
            let ds = sw - pd.x;
            let hw = 0.5 * cc_industry_PDOOR_W;
            if (abs(ds) < hw && vc < cc_industry_PDOOR_H) {
                alb = vec3<f32>(0.10) * select(1.0, 0.6,
                                               abs(ds) > hw - 0.06);
            } else if (abs(ds) < hw + 0.08 && vc < cc_industry_PDOOR_H + 0.08) {
                alb = vec3<f32>(0.06);                     // the frame
            }
            if (pd.y > 0.5) {
                let col = w.lamp_col;
                // The lens, widened toward a pixel at constant energy.
                let sg = max(cc_industry_PLAMP_R, 0.35 * fp);
                let kk = (cc_industry_PLAMP_R * cc_industry_PLAMP_R)
                         / (sg * sg);
                let dz = vc - cc_industry_PLAMP_Z;
                emit = emit + col * (cc_industry_PLAMP_RAD * kk
                    * exp(-0.5 * (ds * ds + dz * dz) / (sg * sg)));
                let gain = col * cc_industry_PLAMP_GAIN;
                near = near + gain * cc_industry_wash(
                    cc_industry_PLAMP_Z - vc, ds, 0.3, 0.5, 0.15, 1.6);
                flux = flux + gain * cc_industry_wash_flux(
                    cc_industry_PLAMP_Z, 0.3, 0.5, 0.15, 1.6);
            }
        }
        light = mix(near, flux / (ws.y * cc.height), k_far);
        alb = cc_industry_paint(alb, pnt);
    }

    // --- light ---------------------------------------------------------------
    let district = city_glow_sample(h.pos.xy, 2.0);
    let raw = CITY_LAMP_RADIANCE * (0.20 + 1.3 * smoothstep(0.02, 0.45, district))
              * cc_industry_street_wall(h.pos.xy, tang, fp);
    // Street lamps stand 9 m tall and throw down: their light on a wall
    // fades out above their heads rather than running up to the eaves.
    light = light + CITY_LAMP_COLOR * (0.35 * raw / (1.0 + raw * 0.30)
                                       * smoothstep(11.0, 5.0, vc));
    let lamp = (alb / cc_industry_ALB_REF) * light;
    let refl = lamp / (vec3<f32>(1.0) + lamp / cc_industry_LAMP_KNEE)
               + alb * (cc_industry_fill(h.pos)
                        + CITY_MOONLIGHT * max(dot(n, u.sun_dir.xyz), 0.0));
    return refl + emit;
}

// Dock hardware: black fabric shelters, black rubber bumpers, a steel lip.
fn cc_industry_dock_shade(h: CityHit, cc: CityCell, s: cc_industry_Slot)
        -> vec3<f32> {
    let fwd = cc_industry_fwd(s.front);
    let v = abs(fwd.x * (h.pos.y - s.ctr.y) - fwd.y * (h.pos.x - s.ctr.x));
    var alb = vec3<f32>(0.035);
    if (h.pos.z <= cc_industry_DOCK_H + 0.001 && v < 1.0) {
        alb = vec3<f32>(0.14);
    }
    return cc_industry_lit(cc, h.pos, h.normal, alb);
}

// ============================================================================
// PROP SHADING
// ============================================================================

// The object a hit landed on: its grid's slot, re-derived from the lot so
// nothing is smuggled through CityHit. A grid of the right kind must hold a
// filled slot whose object contains the point — grids may share ground (a
// forklift works an empty dock slot), objects never do. The dispatcher
// finds it once and hands it to whichever shader needs it.
struct cc_industry_Slot {
    ok: bool,
    ctr: vec2<f32>,
    d: vec4<f32>,
    obj: vec2<f32>,
    front: i32,
}

fn cc_industry_find(cc: CityCell, p: vec3<f32>, kind: i32)
        -> cc_industry_Slot {
    var s: cc_industry_Slot;
    s.ok = false;
    s.ctr = p.xy;
    s.d = vec4<f32>(0.0);
    s.obj = vec2<f32>(1.0);
    s.front = 0;
    let ng = select(0, cc_industry_N_GRIDS,
                    cc.land_use == CITY_USE_INDUSTRIAL
                    || cc.land_use == CITY_USE_YARD);
    for (var i: i32 = 0; i < ng; i = i + 1) {
        let g = cc_industry_grid(cc, i);
        if (!g.ok || g.kind != kind) {
            continue;
        }
        let rel = p.xy - g.lo.xy;
        let ij = clamp(vec2<i32>(floor(rel / g.cell)), vec2<i32>(0), g.n - 1);
        let ctr = g.lo.xy + (vec2<f32>(ij) + 0.5) * g.cell;
        let d = cc_industry_slot_draw(cc, g, ij);
        let hgt = cc_industry_slot_h(g, d);
        if (hgt > 0.0 && all(abs(p.xy - ctr) < 0.5 * g.obj + 0.05)
            && p.z < g.lo.z + hgt + 0.05) {
            s.ok = true;
            s.ctr = ctr;
            s.d = d;
            s.obj = g.obj;
            s.front = g.front;
            return s;
        }
    }
    return s;
}

fn cc_industry_box_color(x: f32) -> vec3<f32> {
    // Weathered container liveries, dimmed: rust red and the shipping
    // lines' blues most, then grey, green, white, orange, brown.
    if (x < 0.24) { return vec3<f32>(0.30, 0.08, 0.05); }
    if (x < 0.44) { return vec3<f32>(0.07, 0.13, 0.28); }
    if (x < 0.58) { return vec3<f32>(0.20, 0.20, 0.20); }
    if (x < 0.70) { return vec3<f32>(0.08, 0.20, 0.11); }
    if (x < 0.82) { return vec3<f32>(0.45, 0.45, 0.42); }
    if (x < 0.92) { return vec3<f32>(0.40, 0.16, 0.04); }
    return vec3<f32>(0.20, 0.11, 0.06);
}

// Weathering on a painted steel side, as an albedo multiplier: rust
// streaks bleeding down from the roof rail (a third of the 0.9 m cells, of
// hashed width and length, redder than the paint), grime darkening the
// bottom 0.9 m where the yard's dust sits, and a paint level that varies
// panel to panel (2.4 m). `u` runs along the face, `zt` is height in the
// box, `hgt` its height, `seed` the box's own draw. Dissolves to
// WEATHER_MEAN, the pattern's own average, as the streaks go sub-pixel.
fn cc_industry_weather(u: f32, zt: f32, hgt: f32, seed: f32, fp: f32)
        -> vec3<f32> {
    let r = hash22(vec2<f32>(floor(u / 0.9), seed * 97.0));
    let x = fract(u / 0.9);
    let w = 0.15 + 0.35 * r.y;
    let len = 0.4 + 1.8 * r.x;
    let depth = hgt - zt;
    var streak = 0.0;
    if (r.x < 0.35 && abs(x - 0.5) < 0.5 * w && depth < len) {
        streak = (1.0 - depth / len) * (0.5 + 0.5 * r.x);
    }
    let rust = vec3<f32>(1.0) - streak * 0.6
               * (vec3<f32>(1.0) - vec3<f32>(1.0, 0.62, 0.45));
    let dirt = 0.78 + 0.22 * smoothstep(0.0, 0.9, zt);
    let panel = 0.88 + 0.24 * hash12(vec2<f32>(floor(u / 2.4), seed * 53.0));
    return mix(rust * (dirt * panel), cc_industry_WEATHER_MEAN,
               smoothstep(0.08, 0.4, fp));
}

// A container roof: shallow transverse corrugation every 0.5 m, two dents
// as local shading hollows at spots the box's draw picks (periodic swells
// read as stripes, not dents), and a fifth of its 1.2 m panels carrying a
// dark wet rust patch; the whole deck dirtier than the sides. Dissolves to
// ROOF_MEAN.
fn cc_industry_box_roof(lu: f32, lv: f32, seed: f32, fp: f32) -> vec3<f32> {
    let rib = 0.85 + 0.30 * (0.5 + 0.5 * cos(6.2831853 * lu / 0.5));
    let rib_k = cc_industry_band(rib, 1.0, 0.25, fp);
    let span = vec2<f32>(cc_industry_BOX_L - 2.0, cc_industry_BOX_W - 1.0);
    let c1 = (fract(seed * vec2<f32>(13.7, 29.1)) - 0.5) * span;
    let c2 = (fract(seed * vec2<f32>(41.3, 57.9)) - 0.5) * span;
    let q = vec2<f32>(lu, lv);
    let d1 = dot(q - c1, q - c1) / (2.0 * 0.8 * 0.8);
    let d2 = dot(q - c2, q - c2) / (2.0 * 0.6 * 0.6);
    let dent = (1.0 - 0.25 * exp(-d1)) * (1.0 - 0.18 * exp(-d2));
    let pr = hash12(vec2<f32>(floor(lu / 1.2), seed * 71.0));
    let rust = select(vec3<f32>(1.0), vec3<f32>(0.70, 0.55, 0.45), pr < 0.2);
    let pat = rust * (rib_k * dent * 0.85);
    return mix(pat, cc_industry_ROOF_MEAN, smoothstep(0.3, 1.2, fp));
}

// Graffiti on a container side (city_graffiti_paint): each face of a box
// standing on the yard is a run of its own, u from the face's left end as
// you stand facing it, v from the yard; a box up the stack is out of reach.
fn cc_industry_box_graffiti(h: CityHit, cc: CityCell, s: cc_industry_Slot,
                            dir: vec3<f32>, fp: f32, zt: f32)
        -> CityGraffitiPaint {
    let n = h.normal.xy;
    let tang = vec2<f32>(-n.y, n.x);
    let half = 0.5 * dot(abs(tang), s.obj);
    let face = select(select(2u, 3u, n.y > 0.0), select(0u, 1u, n.x > 0.0),
                      abs(n.x) > abs(n.y));
    var run: CityGraffitiRun;
    run.seed = pcg2d(cc.seed ^ vec2<u32>(bitcast<u32>(s.ctr.x),
                                         bitcast<u32>(s.ctr.y)
                                         + face * 0x9e3779b9u));
    run.len = 2.0 * half;
    run.kind = CITY_GRAFFITI_CONTAINER;
    let fuv = city_graffiti_footprint(fp, h.normal, dir, vec3<f32>(tang, 0.0),
                                      vec3<f32>(0.0, 0.0, 1.0));
    return city_graffiti_paint(run, dot(h.pos.xy - s.ctr, tang) + half, zt,
                               fuv);
}

fn cc_industry_container_shade(h: CityHit, cc: CityCell,
        s: cc_industry_Slot, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let tier = floor(max(h.pos.z - 1e-3, 0.0) / cc_industry_TIER);
    let zt = h.pos.z - tier * cc_industry_TIER;
    // The paint first, while little else is live (see
    // cc_industry_wall_graffiti): a side of a box on the yard.
    var g = CityGraffitiPaint(1.0, vec3<f32>(0.0));
    if (tier < 0.5 && abs(h.normal.z) < 0.5) {
        g = cc_industry_box_graffiti(h, cc, s, dir, fp, zt);
    }
    let hc = fract(s.d.w * 7.31 + tier * 0.618);
    let col = cc_industry_box_color(hc);
    let long_x = s.obj.x > s.obj.y;
    let lu = select(h.pos.y - s.ctr.y, h.pos.x - s.ctr.x, long_x);
    let lv = select(h.pos.x - s.ctr.x, h.pos.y - s.ctr.y, long_x);
    let n_long = select(h.normal.y, h.normal.x, long_x);
    var m = 1.0;
    if (abs(h.normal.z) < 0.5) {
        if (abs(n_long) < 0.5) {
            // A long side: trapezoidal corrugation, and the corner posts.
            let rib = 0.78 + 0.44 * (0.5 + 0.5 * cos(6.2831853 * lu / 0.28));
            m = cc_industry_band(rib, 1.0, 0.14, fp);
            m = m * cc_industry_band(
                select(1.0, 0.70,
                       abs(lu) > 0.5 * cc_industry_BOX_L - 0.16),
                0.99, 0.16, fp);
        } else if ((n_long > 0.0) == (hc < 0.5)) {
            // The door end: centre seam and four locking bars.
            var p = select(1.0, 0.25, abs(lv) < 0.02);
            let b = min(abs(abs(lv) - 0.30), abs(abs(lv) - 0.90));
            p = select(p, 1.45, b < 0.03);
            p = p * select(1.0, 0.75, zt > cc_industry_TIER - 0.25);
            m = cc_industry_band(p, 0.98, 0.03, fp);
        } else {
            let rib = 0.80 + 0.40 * (0.5 + 0.5 * cos(6.2831853 * lv / 0.28));
            m = cc_industry_band(rib, 1.0, 0.14, fp);
        }
        // Top and bottom rails, and the dark seam between stacked boxes.
        var rail = select(1.0, 0.65,
                          zt < 0.12 || zt > cc_industry_TIER - 0.10);
        rail = select(rail, 0.20, zt < 0.03 && tier > 0.0);
        m = m * cc_industry_band(rail, 0.93, 0.10, fp);
        let wu = select(lu, lv, abs(n_long) > 0.5);
        // The paint goes on over the weathered side, and the corrugation
        // and rails shade through it.
        let base = col * cc_industry_weather(wu, zt, cc_industry_TIER, hc, fp);
        return cc_industry_lit(cc, h.pos, h.normal,
                               cc_industry_paint(base, g) * m);
    }
    // The top of the stack, or a box seen from above.
    return cc_industry_lit(cc, h.pos, h.normal,
                           col * cc_industry_box_roof(lu, lv, hc, fp));
}

fn cc_industry_trailer_shade(h: CityHit, cc: CityCell,
        s: cc_industry_Slot, fp: f32)
        -> vec3<f32> {
    if (h.pos.z < cc_industry_TRL_DECK - 0.01) {
        // Running gear: tyres, legs, rails — near-black.
        return cc_industry_lit(cc, h.pos, h.normal, vec3<f32>(0.03));
    }
    var col = vec3<f32>(0.38, 0.38, 0.36);             // white box van
    if (s.d.z > 0.55) { col = vec3<f32>(0.24, 0.24, 0.25); }
    if (s.d.z > 0.75) { col = vec3<f32>(0.07, 0.12, 0.26); }
    if (s.d.z > 0.85) { col = vec3<f32>(0.28, 0.06, 0.05); }
    if (s.d.z > 0.95) { col = vec3<f32>(0.08, 0.18, 0.10); }
    let fwd = cc_industry_fwd(s.front);
    let u = dot(h.pos.xy - s.ctr, fwd);
    let z = h.pos.z;
    var m = 1.0;
    if (abs(h.normal.z) < 0.5) {
        // Rub rail along the foot of the body, the roof rail, and the
        // vertical side posts; the rear doors' seam and bars at -u.
        var p = select(1.0, 0.55,
                       z < cc_industry_TRL_DECK + 0.18
                       || z > cc_industry_TRL_H - 0.08);
        if (abs(dot(h.normal.xy, fwd)) < 0.5) {
            p = p * select(1.0, 0.85, fract(u / 1.2) < 0.05);
        } else if (dot(h.normal.xy, fwd) < 0.0) {
            let lv = fwd.x * (h.pos.y - s.ctr.y) - fwd.y * (h.pos.x - s.ctr.x);
            p = p * select(1.0, 0.3, abs(lv) < 0.02);
            p = p * select(1.0, 1.35, abs(abs(lv) - 0.62) < 0.03);
        }
        m = cc_industry_band(p, 0.92, 0.06, fp);
        let wu = u + dot(h.pos.xy - s.ctr, vec2<f32>(-fwd.y, fwd.x));
        return cc_industry_lit(
            cc, h.pos, h.normal,
            col * m * cc_industry_weather(
                wu, z - cc_industry_TRL_DECK,
                cc_industry_TRL_H - cc_industry_TRL_DECK, s.d.w, fp));
    }
    // The roof: bows every 0.6 m under a grey film of road grime, heavier
    // toward the nose where the tractor's exhaust lands.
    let bow = cc_industry_band(select(1.0, 0.8, fract(u / 0.6) < 0.08),
                               0.984, 0.05, fp);
    let grime = 0.72 + 0.12 * smoothstep(-6.8, 6.8, -u);
    return cc_industry_lit(cc, h.pos, h.normal, col * (bow * grime));
}

fn cc_industry_pallet_shade(h: CityHit, cc: CityCell,
        s: cc_industry_Slot, fp: f32) -> vec3<f32> {
    let wood = vec3<f32>(0.34, 0.27, 0.18) * (0.7 + 0.5 * s.d.z);
    var m = 1.0;
    if (abs(h.normal.z) < 0.5) {
        // Each pallet: a deck board over three blocks; between the blocks
        // the dark fork openings.
        let fz = fract(h.pos.z / cc_industry_PALLET_H);
        let lat = select(h.pos.x - s.ctr.x, h.pos.y - s.ctr.y,
                         abs(h.normal.x) > 0.5);
        let half_w = 0.5 * select(s.obj.x, s.obj.y,
                                  abs(h.normal.x) > 0.5);
        let q = abs(lat) / half_w;
        let block = q > 0.84 || q < 0.12;
        let p = select(1.0, 0.12, fz < 0.72 && !block);
        m = cc_industry_band(p, 0.55, 0.08, fp);
    }
    return cc_industry_lit(cc, h.pos, h.normal, wood * m);
}

fn cc_industry_rebar_shade(h: CityHit, cc: CityCell,
        s: cc_industry_Slot, fp: f32) -> vec3<f32> {
    let rust = vec3<f32>(0.20, 0.09, 0.05) * (0.8 + 0.4 * s.d.z);
    // Bars run along the bundle's long axis; across it, the bar pitch.
    let long_x = s.obj.x > s.obj.y;
    let lat = select(h.pos.x, h.pos.y, long_x);
    let along_n = abs(select(h.normal.y, h.normal.x, long_x)) > 0.5;
    var p = 0.75 + 0.5 * (0.5 + 0.5 * cos(6.2831853 * lat / 0.032));
    if (along_n) {
        // A bundle's end: the bar ends, bright dots on shadow.
        let e = fract(vec2<f32>(lat, h.pos.z) / 0.032) - 0.5;
        p = select(0.35, 1.3, dot(e, e) < 0.12);
    }
    let m = cc_industry_band(p, 0.95, 0.03, fp);
    return cc_industry_lit(cc, h.pos, h.normal, rust * m);
}

fn cc_industry_skip_shade(h: CityHit, cc: CityCell,
        s: cc_industry_Slot, fp: f32) -> vec3<f32> {
    var col = vec3<f32>(0.07, 0.16, 0.09);
    if (s.d.w > 0.45) { col = vec3<f32>(0.38, 0.26, 0.03); }
    if (s.d.w > 0.75) { col = vec3<f32>(0.25, 0.07, 0.04); }
    // Vertical stiffening ribs every 0.9 m, and a rolled top rim.
    let along = select(h.pos.y, h.pos.x, abs(h.normal.y) > 0.5);
    var p = select(1.0, 0.65, fract(along / 0.9) < 0.1);
    p = p * select(1.0, 1.25, h.pos.z > cc_industry_SKIP_H - 0.12);
    let m = cc_industry_band(p, 0.97, 0.09, fp);
    return cc_industry_lit(cc, h.pos, h.normal, col * m);
}

// Refuse and rubble share one texture: irregular fragments — the cells of a
// Voronoi lattice, so chunks are polygons, not tiles — each its own material
// (soil, broken concrete, brick, a dark void, or the heap's own tint: timber
// and plaster in a skip) on its own tilted facet, with a dark crack where two
// fragments meet. The lattice is taken in a plane sheared by height so a
// heap's flanks are not stretched. A first cut on a square lattice read as a
// quilted dome; one spanning albedo 0.04 to 0.26 read as a checkerboard.
fn cc_industry_rubble(h: CityHit, cc: CityCell, fp: f32, cell: f32,
                      tint: vec3<f32>) -> vec3<f32> {
    let q = (h.pos.xy + vec2<f32>(0.6, -0.4) * h.pos.z) / cell;
    let qi = floor(q);
    var d1 = 1e9;
    var d2 = 1e9;
    var id = qi;
    // The 3x3 neighbourhood, behind a bound the compiler cannot fold (a
    // literal one is unrolled into the monolithic kernel's register file).
    let nn = select(9, 0, fp > 1e6);
    for (var k: i32 = 0; k < nn; k = k + 1) {
        let c = qi + vec2<f32>(f32(k % 3 - 1), f32(k / 3 - 1));
        let d = distance(q, c + hash22(c));
        if (d < d1) {
            d2 = d1;
            d1 = d;
            id = c;
        } else if (d < d2) {
            d2 = d;
        }
    }
    let r = hash22(id + vec2<f32>(0.37, 0.71));
    var alb_frag = vec3<f32>(0.10, 0.08, 0.06);                 // soil
    if (r.x < 0.62) { alb_frag = vec3<f32>(0.15, 0.14, 0.12); } // concrete
    if (r.x < 0.40) { alb_frag = vec3<f32>(0.16, 0.08, 0.05); } // brick
    if (r.x < 0.25) { alb_frag = tint; }
    if (r.x < 0.10) { alb_frag = vec3<f32>(0.05); }             // voids
    let crack = smoothstep(0.02, 0.10, d2 - d1);
    // The mean of the fragments above (and of the 0.8-1.2 level spread),
    // less the cracks, which cover about a tenth of the surface.
    let alb_mean = (0.15 * tint + vec3<f32>(0.100, 0.078, 0.062)) * 0.93;
    let k = smoothstep(0.5 * cell, 2.0 * cell, fp);
    let tilt = (r - 0.5) * 0.9 * (1.0 - k);
    // @domain: a roof's normal (z = 1) plus a horizontal tilt
    let n = normalize(h.normal + vec3<f32>(tilt, 0.0));
    let frag = alb_frag * ((0.8 + 0.4 * r.y) * (0.35 + 0.65 * crack));
    return cc_industry_lit(cc, h.pos, n, mix(frag, alb_mean, k));
}

fn cc_industry_fork_shade(h: CityHit, cc: CityCell,
        s: cc_industry_Slot, fp: f32) -> vec3<f32> {
    let fwd = cc_industry_fwd(s.front);
    let u = dot(h.pos.xy - s.ctr, fwd);
    let v = abs(fwd.x * (h.pos.y - s.ctr.y) - fwd.y * (h.pos.x - s.ctr.x));
    let z = h.pos.z;
    var paint = vec3<f32>(0.55, 0.38, 0.04);          // yellow
    if (s.d.z > 0.60) { paint = vec3<f32>(0.50, 0.19, 0.03); }   // orange
    if (s.d.z > 0.85) { paint = vec3<f32>(0.36, 0.06, 0.04); }   // red
    var alb = paint;
    let steel = vec3<f32>(0.06);
    if (z > 2.0 || u > 0.30 || (z > 1.1 && v > 0.45)) {
        alb = steel;                      // guard, posts, mast, forks
    }
    if (z < 0.56 && (abs(u) < 0.26 || (u < -0.84 && u > -1.31))) {
        alb = vec3<f32>(0.02);            // tyres
    }
    if (z > 1.1 && z < 1.55 && u < -0.49 && u > -0.96 && v < 0.26) {
        alb = vec3<f32>(0.025);           // seat
    }
    return cc_industry_lit(cc, h.pos, h.normal, alb);
}

// ============================================================================
// THE SHADE HOOK
// ============================================================================
fn cc_industry_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    // Roofs and lamps first; everything else stands in a grid slot, found
    // once here. (The refuse is the skip's.)
    switch h.kind {
        case 700: { return cc_industry_roof_shade(h, cc, fp); }
        case 701: { return cc_industry_glaze_shade(h, cc, fp); }
        case 702: { return cc_industry_sky_shade(h, cc, fp); }
        case 703: { return cc_industry_head_shade(h, cc); }
        case 704: {
            return cc_industry_lit(cc, h.pos, h.normal, vec3<f32>(0.10));
        }
        case 711: {
            return cc_industry_rubble(h, cc, fp, 0.22,
                                      vec3<f32>(0.13, 0.12, 0.11));
        }
        case 713: {
            // The fence: galvanised posts and rail.
            return cc_industry_lit(cc, h.pos, h.normal, vec3<f32>(0.22));
        }
        default: {}
    }
    let s = cc_industry_find(cc, h.pos,
                             select(h.kind, cc_industry_K_SKIP,
                                    h.kind == cc_industry_K_REFUSE));
    switch h.kind {
        case 705: { return cc_industry_container_shade(h, cc, s, dir, fp); }
        case 706: { return cc_industry_trailer_shade(h, cc, s, fp); }
        case 707: { return cc_industry_pallet_shade(h, cc, s, fp); }
        case 708: { return cc_industry_rebar_shade(h, cc, s, fp); }
        case 709: { return cc_industry_skip_shade(h, cc, s, fp); }
        case 710: {
            return cc_industry_rubble(h, cc, fp, 0.28,
                                      vec3<f32>(0.22, 0.17, 0.10));
        }
        case 714: { return cc_industry_dock_shade(h, cc, s); }
        default: { return cc_industry_fork_shade(h, cc, s, fp); }
    }
}
