// The night city: a procedurally generated city of skyscrapers under the
// cloud field. The sun becomes a crescent moon, and the buildings are the
// light.
//
// The surface tile (binding 3) is the city's density tile, one texel per
// city block, as an rgba16float mip chain. r is a multifractal cascade
// normalized by its 99.5th percentile, the building-height modulator; g is
// the same field rank-transformed to uniform [0, 1], which building presence
// thresholds; b and a hold the lots and streets. The coarse mips of r are the
// ground-glow field: mip k is the mean lit density over 2^k blocks, which is
// what a cloud base or a long sightline integrates over. The surface rows of
// the uniform block keep their geometric meaning (surface z; tile cell size,
// extent, enable, top mip); the host packs night colours into the spectral
// rows, and u.sun_dir is the moon.
//
// The geometry is exact: a 2D DDA over the block grid, with slab tests
// against each building's stacked boxes and its mast. The lighting is where
// the approximations live: procedural window grids with an analytic far-field
// mean (blended in as the pixel footprint outgrows a window), streetlight
// pools with the same treatment, and an exponential ground fog whose
// in-scatter is the glow mip at the hit point.

const CITY: bool = true;

// Grid. The block pitch and tile extent arrive in u.ocean_params (x, y) so
// the tile stays self-describing; everything else about the city is a
// compile-time choice.
const CITY_GROUND_Z: f32 = 0.0;
// The height law is deliberately unclipped below this: the cascade's
// lognormal tail IS the skyline — the p99.5 building lands near 390 m and
// the rare 5x-8x excursions are the megatowers that reach the cloud deck.
// The ceiling only exists so the DDA has a z gate, and only the single
// wildest draw of the shipped tile touches it.
const CITY_MAX_H: f32 = 5400.0;     // tallest roof (before mast)
const CITY_SLAB_TOP: f32 = 5600.0;  // roof + tallest mast: the DDA's z gate
const CITY_FLOOR_H: f32 = 3.6;      // one storey; window rows ride this
const CITY_PER_STOREY: f32 = 0.2777777777777778;  // 1 / CITY_FLOOR_H
const CITY_MAX_STOREYS: f32 = 1500.0;             // CITY_MAX_H / CITY_FLOOR_H
const CITY_STREET_HALF: f32 = 6.0;  // half-width of a minor street (m)
const CITY_AVENUE_EXTRA: f32 = 10.0; // extra half-width on avenue boundaries
const CITY_AVENUE_PERIOD: i32 = 8;  // avenue every this many blocks
const CITY_TRACE_MAX_CELLS: i32 = 512;
const CITY_TRACE_RANGE: f32 = 30000.0;
// Building presence and mass. rank below EMPTY -> unbuilt lot; the sprawl
// ramp suppresses height just above the threshold so the outskirts are
// low-rise rather than a cliff of towers at the empty-lot boundary.
const CITY_EMPTY_RANK: f32 = 0.22;
const CITY_SPRAWL_RANK_FULL: f32 = 0.60;
const CITY_SPRAWL_MIN_FRAC: f32 = 0.15;
const CITY_H_BASE: f32 = 14.0;
const CITY_H_SCALE: f32 = 390.0;
const CITY_H_EXP: f32 = 1.2;
// Massing bands (see the footprint block in city_cell). Below LOWRISE a
// block is a row of frontages; from TOWER up a building is a tower whose
// side is K sqrt(h) metres up to the KNEE and holds there (see
// SLENDER_MAX for the floor on the very tallest).
const CITY_LOWRISE_H: f32 = 22.0;
const CITY_TOWER_H: f32 = 60.0;
const CITY_TOWER_SIDE_K: f32 = 4.2;
const CITY_TOWER_KNEE_H: f32 = 600.0;
// Above SPIRE_H every tower is a spire crown or a tapered shaft — a tall
// prism reads from across the city as a slab. Above SUPERTALL_H the taper
// is sharper, the spire longer, and a tapered shaft ends in a needle.
const CITY_SPIRE_H: f32 = 600.0;
const CITY_SUPERTALL_H: f32 = 1200.0;
// The slenderness floor: a tower is never narrower than its height over
// this, on its broad face (the narrow face is up to 0.72 of it). It binds
// only on the very tallest; everything below takes its slenderness from its
// taper and spire, not from its width.
const CITY_SLENDER_MAX: f32 = 60.0;
// Land use, from the baked lot words (cloudyview/city_lots.py). A lot word
// is an integer of at most 12 bits stored offset by -2048, where float16
// holds every integer exactly.
const CITY_USE_URBAN: i32 = 0;
const CITY_USE_INDUSTRIAL: i32 = 1;
const CITY_USE_YARD: i32 = 2;
const CITY_USE_OPEN: i32 = 3;
const CITY_WORD_OFFSET: f32 = 2048.0;
// A warehouse's occupancy: a few office windows at one end, the rest dark.
const CITY_INDUSTRY_LIT_FRAC: f32 = 0.03;
// A warehouse's ground band: dock doors, personnel doors and their lamps,
// which the industry component draws; no windows below it.
const CITY_INDUSTRY_DOOR_BAND: f32 = 5.0;
// Windows.
const CITY_WIN_PITCH_U: f32 = 3.4;  // horizontal window pitch (m)
const CITY_WIN_U_LO: f32 = 0.14;    // pane bounds within the pitch cell
const CITY_WIN_U_HI: f32 = 0.86;
const CITY_WIN_V_LO: f32 = 0.28;
const CITY_WIN_V_HI: f32 = 0.90;
const CITY_DARK_FLOOR_FRAC: f32 = 0.04; // whole storeys gone dark
// A lit room's radiance through its pane, before the per-window brightness
// draw (0.25 + 5 w^7, mean CITY_WIN_BRIGHT_MEAN): THE knob for how bright
// the city's windows are, near and far alike (the far facade averages the
// same windows). Held where a typical pane does not clip to white at the
// night exposure, so rooms keep their colour.
const CITY_ROOM_RADIANCE: f32 = 2.4;
// The city's window light in one warm colour, for what glows with it (the
// lamps' and the rooms' spill on props, reflections, rooftops).
const CITY_PALETTE_MEAN: vec3<f32> = vec3<f32>(1.0, 0.55, 0.25);
const CITY_WIN_BRIGHT_MEAN: f32 = 0.875; // E[0.25 + 5 w^7]

// Streets.
const CITY_LAMP_SPACING: f32 = 26.0;
const CITY_LAMP_OFFSET: f32 = 2.5;   // lamp line inset from the block edge
const CITY_LAMP_SIGMA: f32 = 3.2;    // light-pool radius on the asphalt
const CITY_LAMP_RADIANCE: f32 = 0.55;
const CITY_LAMP_COLOR: vec3<f32> = vec3<f32>(1.0, 0.42, 0.12); // sodium
const CITY_STREET_LOD_START: f32 = 8.0;
const CITY_STREET_LOD_FULL: f32 = 26.0;
// The shop row's glass as the street sees it: a strip from the ground to
// the top of the shop glass (CITY_STORE_PANE_HI) at city_storefront_glow.
const CITY_STORE_H: f32 = CITY_STORE_PANE_HI.y * CITY_FLOOR_H;
// Roof furniture.
const CITY_MAST_MIN_H: f32 = 150.0;  // buildings shorter than this: no mast
const CITY_MAST_CROSS: f32 = 1.4;
// The crown (city_crown): a penthouse on ~20% of tall flat-topped towers,
// core massing at every range, or the helipad that pre-empts it (the pad
// itself is rooftopworks' prop). The draw is the one rooftopworks made when
// the penthouse was its prop, unchanged, so the crowns stand where they did.
const CITY_CROWN_TIER: i32 = 3;          // the crown's face/shell box index
const CITY_CROWN_MIN_H: f32 = 105.0;     // "tall": tower, not mid-rise
const CITY_CROWN_FRAC: f32 = 0.20;
const CITY_CROWN_MIN_DECK: f32 = 15.0;
const CITY_CROWN_INSET: f32 = 0.28;
const CITY_CROWN_H_LO: f32 = 7.2;        // two storeys
const CITY_CROWN_H_HI: f32 = 10.8;       // three
const CITY_CROWN_PAD_FRAC: f32 = 0.13;
const CITY_CROWN_PAD_BIG: f32 = 60.0;    // footprint that qualifies
const CITY_CROWN_PAD_MIN_H: f32 = 55.0;  // and a building worth landing on
const CITY_CROWN_PAD_T: f32 = 0.30;
const CITY_BEACON_LEN: f32 = 2.5;
const CITY_BEACON_COLOR: vec3<f32> = vec3<f32>(40.0, 2.0, 1.6);
// Night materials and fill light.
const CITY_MOONLIGHT: vec3<f32> = vec3<f32>(0.020, 0.024, 0.032);
const CITY_SKYGLOW: vec3<f32> = vec3<f32>(0.012, 0.010, 0.016);
const CITY_FACADE_ALBEDO: f32 = 0.030;
const CITY_ROOF_ALBEDO: f32 = 0.085;
const CITY_ASPHALT_ALBEDO: f32 = 0.035;
const CITY_YARD_ALBEDO: f32 = 0.060;   // gravel, packed dirt, old concrete
const CITY_SCRUB_ALBEDO: f32 = 0.050;  // dry scrub and bare soil
// The city-lit cloud deck and sky as open fringe ground sees them (see the
// ground branch of city_shade): a floor plus the wide glow, warm.
const CITY_OPEN_SKY: vec3<f32> = vec3<f32>(0.060, 0.042, 0.030);
const CITY_OPEN_SKY_BASE: f32 = 0.25;
// Ground fog: exponential profile, uniform-in-z beyond its scale height
// none — this is the night twin of the aerial-perspective machinery, kept
// separate because its in-scatter source is the city itself, not the sky.
const CITY_FOG_BETA: f32 = 6.0e-5;   // m^-1 at ground level
const CITY_FOG_H: f32 = 320.0;       // scale height (m)
const CITY_FOG_GLOW: vec3<f32> = vec3<f32>(1.0, 0.42, 0.15);
const CITY_FOG_GLOW_AMP: f32 = 0.10;
const CITY_FOG_BASE: vec3<f32> = vec3<f32>(0.003, 0.004, 0.008);
// Night sky.
const CITY_NIGHT_HORIZON: vec3<f32> = vec3<f32>(0.006, 0.007, 0.012);
const CITY_NIGHT_ZENITH: vec3<f32> = vec3<f32>(0.0008, 0.0011, 0.0024);
const CITY_SKYGLOW_DOME: vec3<f32> = vec3<f32>(0.90, 0.50, 0.24);
const CITY_SKYGLOW_DOME_AMP: f32 = 0.050;
const CITY_SKYGLOW_DOME_DIST: f32 = 3000.0;
const CITY_SKYGLOW_DOME_SCALE: f32 = 0.09; // e-folding in dir.z
const CITY_STAR_GRID: f32 = 160.0;
const CITY_STAR_FRAC: f32 = 0.10;
const CITY_STAR_RADIUS: f32 = 0.16;  // in star-grid cells
const CITY_STAR_AMP: f32 = 0.05;
// Moon: drawn ~4x the true angular radius — at flight FOV the true disc is
// an unreadable dot, and the crescent is the shot. The crescent's own sun
// sits mostly behind the disc (phase angle ~120 deg) and a little
// below-right, which is what makes the lit limb a waxing crescent rather
// than a half. Earthshine kept near-invisible: an exposed night render
// lifts it fast, and a glowing dark side reads as a full moon in fog.
const CITY_MOON_SIN_R: f32 = 0.018;
const CITY_MOON_DISC: vec3<f32> = vec3<f32>(4.0, 4.1, 4.35);
const CITY_MOON_EARTHSHINE: f32 = 0.005;
const CITY_MOON_TERMINATOR_SOFT: f32 = 0.14;
const CITY_MOON_HALO_G: f32 = 0.78;
const CITY_MOON_HALO_AMP: f32 = 0.008;
const CITY_MOON_BLOOM_W: f32 = 0.0008;

// --- City component hooks --------------------------------------------------
//
// The city grows by COMPONENTS: small WGSL files under
// cloudyview/soar/city/components/, each owning one kind of thing (a street
// prop, a facade layer, an elevated highway), registered in registry.json
// and spliced into this file by tools/compose_city.py between the GENERATED
// markers below. The core calls ten hooks; the composer generates their
// dispatchers over whatever is registered, and when nothing is, the
// defaults below make every hook a no-op — except the shade dispatcher,
// which returns magenta, because an unclaimed hit kind is a bug and should
// look like one.
//
//   cc_extra_trace       geometry independent of the block grid (highways):
//                        traced once per ray, nearest hit wins against the
//                        DDA's.
//   cc_cell_props_trace  per-cell prop geometry (streetlights, cars, bins),
//                        called only within CITY_PROP_RANGE of the camera —
//                        the props' whole cost budget lives inside that
//                        radius.
//   cc_near_props_trace  per-cell prop geometry seen only close (street
//                        furniture, the parked cars' hulls), below
//                        CITY_FURNITURE_TOP, traced only by the street stage
//                        (surface_street_city) out to city_near_reach.
//   cc_component_shade   radiance at a component hit (kind >= 100; each
//                        component owns kind_base .. kind_base+15).
//   cc_facade_detail     additive facade emission (balconies, signage,
//                        fire escapes) on top of the core's windows.
//   cc_facade_cover      how much of the wall at a point a component's
//                        panel covers (0..1): covered wall shows no
//                        windows and, up close, no shell relief — the panel
//                        is what is there.
//   cc_window_glyph      transmission multiplier inside a lit pane
//                        (curtains, figures, androids) — 1 leaves the
//                        window exactly as it was. Two uvs: the glass, and
//                        where the ray crosses the occupant's plane inside
//                        the room (the same point on a flat facade).
//   cc_window_glyph_mean the glyph's mean over all lit windows, for a
//                        facade seen too small to sum window by window.
//   cc_media             participating media in front of the city (manhole
//                        steam): in-scattered radiance and transmittance
//                        over the view ray up to the city hit, applied as
//                        rgb + a * city. (0, 0, 0, 1) is the identity.
//   cc_roof_detail       additive roof emission (a warehouse's lit
//                        sawtooth glazing, a roof's skylights) at EVERY
//                        range — the far-field mean of roof geometry that
//                        props draw only near. Handed the core's own roof
//                        radiance, so detail can be averaged as it is shown.
//
// Component hits reuse CityHit with kind >= 100. See components/SPEC.md for
// the contract agents write against.

const CITY_PROP_RANGE: f32 = 2200.0;

// Whether city_trace ran cell_props for the camera's view ray to a point
// in cell ci: props run for a cell whose ray segment starts inside
// CITY_PROP_RANGE, and that start is where the ray enters the cell's
// column, or enters the city slab from above if that is later
// (city_trace's t_cur). Recomputed from the camera, for the roof hooks,
// which add a geometry's far mean exactly where the geometry is not drawn.
// For a hit on the view ray only (t its distance from the camera): a
// reflection's trace (CITY_TRACE_PLAIN) runs no props, which callers test
// first.
fn city_props_drawn(ci: vec2<i32>, p: vec3<f32>, t: f32) -> bool {
    let o = city_camera_origin();
    let dir = (p - o) / max(t, 1e-3);
    let cell = u.tile_cell;
    let inv = vec3<f32>(select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-9),
                        select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-9),
                        select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-9));
    let cmin = vec3<f32>(vec2<f32>(ci) * cell, -1e6);
    let col = city_box_hit(o, inv, cmin, cmin + vec3<f32>(cell, cell, 2e6));
    var t_slab = 0.0;
    if (o.z > CITY_SLAB_TOP) {
        t_slab = (CITY_SLAB_TOP - o.z) * inv.z;
    }
    return max(col.x, t_slab) < CITY_PROP_RANGE;
}

fn city_rand4(s: vec2<u32>) -> vec4<f32> {
    let a = pcg2d(s);
    let b = pcg2d(a);
    return vec4<f32>(city_u01(a.x), city_u01(a.y),
                     city_u01(b.x), city_u01(b.y));
}

// Four independent uniforms for this pixel and accumulation frame, one draw
// per `stream`. White, not stratified: all ten dimensions of the strat
// lattice (strat1/strat2) already drive the camera and cloud jitters, and
// sharing one of them — a different stream only rotates it — would lock a
// city draw to, say, the subpixel offset for every frame of a pixel. Adding
// dimensions would change every existing alpha, and the day scene with them.
// (The wet reflection's draws have their own generator instead, an R4 of its
// own constant: city_refl_rand.)
fn city_frame_rand(pixel: vec2<u32>, frame: u32, stream: u32) -> vec4<f32> {
    return city_rand4(vec2<u32>(pixel.x ^ (frame * 0x9e3779b9u),
                                pixel.y + frame * 0x85ebca6bu
                                + stream * 0xc2b2ae35u));
}
const CITY_RAND_REFLECT: u32 = 1u;
const CITY_RAND_MEDIA: u32 = 2u;

// The city repeats at the tile, exactly. One texel is one block, so the
// block count per tile IS the glow texture's mip-0 dimension (1024 at the
// shipped bake, with the tile extent defined as n * cell — no rounding),
// and every draw seeded by a cell index folds the index into [0, n) first.
// That is what makes the city a pure function of the tile frame: a tile
// coordinate names a building, so a replayed track — which serializes only
// the tile phase — renders the very buildings that were flown past, and the
// whole-tile ambiguity in row 24's folded offset is invisible.
fn city_tile_cell(ci: vec2<i32>) -> vec2<i32> {
    let dims = textureDimensions(ocean_normals, 0);
    let n = vec2<i32>(i32(dims.x), i32(dims.y));
    return ((ci % n) + n) % n;
}

// World xy into the canonical city frame: the offset removed, then folded
// into the primary tile. Everything tile-periodic is invariant under the
// fold; the one pattern whose period does not divide the tile (skyway's
// 24-block route lattice — 1024 is not a multiple of 24) is pinned by it to
// one canonical alignment, so live flight and replay agree there too.
fn city_fold_xy(xy: vec2<f32>) -> vec2<f32> {
    let t = u.tile_extent;
    let q = (xy - u.surface_offset) / t;
    return t * (q - floor(q));
}

// The camera in the city frame — the ONE place a ray enters it. city_trace,
// the sky-dome glow probe and any component that needs the view origin all
// take it from here, so the frame cannot be entered two different ways.
fn city_camera_origin() -> vec3<f32> {
    return vec3<f32>(city_fold_xy(u.cam_origin.xy), u.cam_origin.z);
}

fn city_is_avenue(k: i32) -> bool {
    return ((k % CITY_AVENUE_PERIOD) + CITY_AVENUE_PERIOD)
           % CITY_AVENUE_PERIOD == 0;
}

// The lamp line's inset from the CELL BOUNDARY on line k. Lamps stand
// CITY_LAMP_OFFSET off the KERB — 3.5 m from the boundary of a 12 m minor
// street, 13.5 m on a 32 m avenue. The old fixed 2.5 m inset measured from
// the boundary, which on an avenue is the road's CENTERLINE: both lamp
// lines hugged the median and floodlit the middle of every downtown
// avenue into clipping (found via streetlife's report, proven by
// ablation + a numpy rebuild of the pool field, 2026-08-20). streetlife's
// poles and parking share this function, so light, hardware and cars
// move together.
fn city_lamp_inset(k: i32) -> f32 {
    return CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(k))
        - CITY_LAMP_OFFSET;
}

// A seed for everything drawn per street: centreline k folded at the tile
// first, like every cell-index seed (city_tile_cell), so a street is the
// same street on every tile and in every replay.
fn city_line_seed(k: i32) -> u32 {
    return bitcast<u32>(city_tile_cell(vec2<i32>(k, 0)).x);
}

// The lamp stations along the street on centreline k: (spacing, phase), so
// station j stands at phase + j * spacing. Avenues keep the civic 26 m
// cadence at phase 0; a minor street draws its own spacing (22-31 m) and
// phase, so the pools and poles of neighbouring streets stop lining up into
// one grid over the whole city (a blind judge's complaint: the lamp lattice
// read as wallpaper). The tile extent is not a multiple of every spacing, so
// a street's stations restart once per tile, 92 km apart.
fn city_lamp_lattice(k: i32) -> vec2<f32> {
    if (city_is_avenue(k)) {
        return vec2<f32>(CITY_LAMP_SPACING, 0.0);
    }
    let h = pcg2d(vec2<u32>(city_line_seed(k) ^ 0x3c6ef372u, 0xa54ff53au));
    let sp = 22.0 + 3.0 * f32(h.x & 3u);
    return vec2<f32>(sp, sp * city_u01(h.y));
}

// The station on centreline k nearest the along coordinate a.
fn city_lamp_station(k: i32, a: f32) -> f32 {
    let l = city_lamp_lattice(k);
    return l.y + round((a - l.y) / l.x) * l.x;
}

// The glow field: mean lit density over 2^lod blocks, read with the repeat
// sampler so the tile wraps. Never below mip 1: the tile's r is the raw
// density at mip 0 (heights) and the land-use-weighted glow from mip 1 up
// (tools/export_web_assets.py), so the clamp is here, not a convention. This is the one number the fog, the sky dome
// and the cloud uplight all share. `xy` is in the CITY FRAME (world minus
// the row-24 offset — see city_camera_origin), whose origin IS the tile's
// origin, so the uv is simply the coordinate over the extent: which
// district sits where is decided entirely by the offset the host wrote.
fn city_glow_sample(xy: vec2<f32>, lod: f32) -> f32 {
    let uv = xy / u.tile_extent;
    let l = clamp(lod, 1.0, u.tile_max_lod);
    return textureSampleLevel(ocean_normals, ocean_samp, uv, l).r;
}

// Everything the march and the shader need to know about one block.
struct CityCell {
    built: bool,
    density: f32,
    rank: f32,
    height: f32,      // main roof z
    top_z: f32,       // roof + mast: the cell's cheap z reject
    tiers: i32,
    b1min: vec3<f32>, b1max: vec3<f32>,
    b2min: vec3<f32>, b2max: vec3<f32>,
    b3min: vec3<f32>, b3max: vec3<f32>,
    has_mast: bool,
    mast_min: vec3<f32>, mast_max: vec3<f32>,
    seed: vec2<u32>,
    lit_frac: f32,
    palette_bias: f32,
    store_draw: f32,
    plot_min: vec2<f32>, plot_max: vec2<f32>,
    // The architecture (see the archetype block in city_cell): 0 slab /
    // setback stack, 2 growth (cantilevered buds), 3 tapered shaft,
    // 4 spire crown. Archetypes 3 and 4 carry a frustum: a rectangular
    // cross-section scaling linearly from 1 at fmin.z to fscale at fmax.z.
    arch: i32,
    has_frustum: bool,
    fmin: vec3<f32>, fmax: vec3<f32>,
    fscale: f32,
    // One building can own a lot of several blocks — up to 4x4 under a tall
    // tower among lower neighbours, 2x2 under a mid-rise superblock, 2x1 or
    // 2x2 under a warehouse (cloudyview/city_lots.py); every member cell
    // then reports the same plot, seed and boxes.
    merged: bool,
    // The window style (see the style block in city_cell): 0 grid,
    // 1 ribbon bands, 2 vertical strips, 3 curtain wall, 4 punched.
    // Pitch and pane bounds are per building so no two lattices agree.
    win_style: i32,
    win_pitch: f32,
    pane_lo: vec2<f32>, pane_hi: vec2<f32>,
    // Monochrome house palette: below 0 the building draws window colors
    // freely; at or above 0 it is THE cyan (or amber, or magenta) tower —
    // every window draws near this one palette coordinate.
    win_mono: f32,
    // What the lot is (CITY_USE_*): URBAN and INDUSTRIAL are built (a
    // warehouse is a building); a YARD fronts a street but carries no
    // building (vacant lot, container and trailer storage, refuse); OPEN
    // land is scrub with no street at all.
    land_use: i32,
    // Which of this CELL's edges carry a street (bits 0-3: -x, +x, -y, +y)
    // and which of its corner squares are paved (bits 4-7: (-x,-y) (+x,-y)
    // (-x,+y) (+x,+y)). Per cell, not per lot: the edges inside a lot carry
    // none. Read it through city_edge_street / city_ground_paved.
    streets: u32,
    // The whole lot's rectangle of cells (city frame, metres): the column
    // every prim of the lot's building stays inside.
    lot_min: vec2<f32>, lot_max: vec2<f32>,
    // The DDA cell this was decoded for. Everything above is the lot's;
    // the crown (city_crown) is the one per-CELL thing in the massing.
    cell: vec2<i32>,
}

// A packed lot word from its float16 channel (see CITY_WORD_OFFSET).
fn city_word(v: f32) -> u32 {
    return u32(v + CITY_WORD_OFFSET);
}

// Does this cell's `side` edge carry a street? 0 -x, 1 +x, 2 -y, 3 +y.
fn city_edge_street(cc: CityCell, side: i32) -> bool {
    return ((cc.streets >> u32(side)) & 1u) != 0u;
}

fn city_cell(ci: vec2<i32>) -> CityCell {
    var c: CityCell;
    c.cell = ci;
    let cell = u.tile_cell;

    // The lot. Which blocks belong together, what the land is used for and
    // where the streets run were decided offline (cloudyview/city_lots.py)
    // and packed into b and a at mip 0, because every one of those answers
    // is non-local: a supertall claims the blocks around its peak, and a
    // street exists only if it joins the network. Every member cell of a lot
    // decodes the same rectangle and reads the same OWNER texel, so the DDA
    // assembles one seamless building from whichever member it visits, and
    // the owner — the lot's peak — keeps exactly the height it had as a
    // block of its own. Texels AND seeds read the tile-wrapped index
    // (city_tile_cell): the city frame's origin is the tile's, cell i maps
    // to texel i, and the wrap before pcg2d is what makes a building a
    // property of its tile coordinate rather than of how many domains the
    // camera has crossed. The lot arithmetic runs on the UNWRAPPED index, so
    // a lot that straddles the tile seam is still one rectangle.
    let w = city_tile_cell(ci);
    let texel = textureLoad(ocean_normals, w, 0);
    let lot = city_word(texel.b);
    let st = city_word(texel.a);
    let back = vec2<i32>(i32(lot & 3u), i32((lot >> 2u) & 3u));
    let span = vec2<i32>(i32((lot >> 4u) & 3u), i32((lot >> 6u) & 3u))
               + vec2<i32>(1);
    let own = vec2<i32>(i32((lot >> 8u) & 3u), i32((lot >> 10u) & 3u));
    let base = ci - back;
    c.merged = span.x > 1 || span.y > 1;
    c.streets = st & 0xffu;
    c.land_use = i32((st >> 8u) & 3u);
    let wo = city_tile_cell(base + own);
    var t_own = texel;
    if (any(wo != w)) {
        t_own = textureLoad(ocean_normals, wo, 0);
    }
    c.density = t_own.r;
    c.rank = t_own.g;
    c.seed = pcg2d(vec2<u32>(bitcast<u32>(wo.x), bitcast<u32>(wo.y)));

    // The plot: the lot's rectangle less half a street on every outer edge,
    // avenue boundaries wider. The inset holds on an edge that carries no
    // street too — there the strip is bare lot ground rather than asphalt —
    // so a plot edge is a kerb line wherever a street runs, and an avenue
    // corridor is empty air whatever runs on it (skyway's decks rely on
    // that). The avenue test can take the unwrapped index: its period (8)
    // divides the blocks per tile (1024), so it is already the same function
    // of the tile coordinate.
    let cmin = vec2<f32>(base) * cell;
    let cmax = vec2<f32>(base + span) * cell;
    c.lot_min = cmin;
    c.lot_max = cmax;
    let hx0 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.x));
    let hx1 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.x + span.x));
    let hy0 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.y));
    let hy1 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.y + span.y));
    c.plot_min = cmin + vec2<f32>(hx0, hy0);
    c.plot_max = cmax - vec2<f32>(hx1, hy1);
    let r4 = city_rand4(c.seed);
    let r4b = city_rand4(c.seed ^ vec2<u32>(0x9e3779b9u, 0x85ebca6bu));
    let r4c = city_rand4(c.seed ^ vec2<u32>(0xdeadbeefu, 0x41c64e6du));

    c.built = c.land_use == CITY_USE_URBAN
           || c.land_use == CITY_USE_INDUSTRIAL;
    // Occupancy follows the cascade: downtown towers burn, outskirts
    // buildings are mostly asleep. This is what makes the density clusters
    // read at night — the same field that raised the buildings lights them.
    c.lit_frac = min(
        (0.14 + 0.55 * r4c.x * r4c.x)
            * mix(0.55, 1.5, smoothstep(0.05, 0.50, c.density)),
        0.40);
    c.palette_bias = r4c.y;
    c.store_draw = r4c.z;
    c.tiers = 1;
    c.has_mast = false;
    c.height = 0.0;
    c.top_z = 0.0;
    c.b1min = vec3<f32>(0.0); c.b1max = vec3<f32>(0.0);
    c.b2min = vec3<f32>(0.0); c.b2max = vec3<f32>(0.0);
    c.b3min = vec3<f32>(0.0); c.b3max = vec3<f32>(0.0);
    c.mast_min = vec3<f32>(0.0); c.mast_max = vec3<f32>(0.0);
    c.arch = 0;
    c.has_frustum = false;
    c.fmin = vec3<f32>(0.0); c.fmax = vec3<f32>(0.0);
    c.fscale = 1.0;
    if (!c.built) {
        return c;
    }

    let plot_size = c.plot_max - c.plot_min;
    let pc = 0.5 * (c.plot_min + c.plot_max);
    let r4d = city_rand4(c.seed ^ vec2<u32>(0x27d4eb2fu, 0x165667b1u));

    if (c.land_use == CITY_USE_INDUSTRIAL) {
        // A warehouse: one low box across most of the lot, pushed to one
        // long side so the other is the truck court its loading docks face.
        // Clear heights run 9-15 m; the roof is flat to the core (the
        // sawtooth and monitor ridges are the industry component's roof
        // geometry). No mast, no setbacks, almost no windows.
        let h = floor((9.0 + 6.0 * r4.x) * 2.0) * 0.5;
        c.height = h;
        c.top_z = h;
        let long_x = plot_size.x > plot_size.y
            || (plot_size.x == plot_size.y && r4.y < 0.5);
        let along = select(plot_size.y, plot_size.x, long_x);
        let across = select(plot_size.x, plot_size.y, long_x);
        let fa = along * (0.78 + 0.17 * r4.z);
        let fc = across * (0.58 + 0.22 * r4.w);
        let slack_a = (along - fa) * r4b.x;
        // The court side: the building hugs one long edge.
        let slack_c = select(across - fc - 2.0, 2.0, r4b.y < 0.5);
        var lo = c.plot_min;
        var hi = c.plot_min;
        if (long_x) {
            lo = c.plot_min + vec2<f32>(slack_a, slack_c);
            hi = lo + vec2<f32>(fa, fc);
        } else {
            lo = c.plot_min + vec2<f32>(slack_c, slack_a);
            hi = lo + vec2<f32>(fc, fa);
        }
        c.b1min = vec3<f32>(lo, CITY_GROUND_Z);
        c.b1max = vec3<f32>(hi, h);
        c.win_style = 4;
        c.win_pitch = 3.2 * (0.85 + 0.40 * r4b.z);
        c.pane_lo = vec2<f32>(0.30, 0.45);
        c.pane_hi = vec2<f32>(0.70, 0.80);
        c.win_mono = -1.0;
        c.palette_bias = min(c.palette_bias * 1.4 + 0.1, 1.0);
        c.lit_frac = CITY_INDUSTRY_LIT_FRAC * (0.3 + 1.4 * r4c.x);
        return c;
    }

    // Height: base + the cascade raised to a power, jittered, suppressed
    // toward the outskirts, and quantized to storeys so roofs are flat
    // shelves rather than a continuum. Every level below is a whole number
    // of storeys, carried as the count ns and multiplied, never divided:
    // WGSL rounds a product exactly but a quotient only to 2.5 ulp (NVIDIA
    // divides by 3.6 as a multiply by its reciprocal), so floor(h * 0.75 /
    // 3.6) lands a storey high on one GPU and not on another whenever
    // 0.75 ns is whole — and the city must be the same city on every GPU,
    // and in the JavaScript collision mirror (web/soar/citylots.js).
    let sprawl = mix(
        CITY_SPRAWL_MIN_FRAC, 1.0,
        smoothstep(CITY_EMPTY_RANK, CITY_SPRAWL_RANK_FULL, c.rank)
    );
    // @domain: density is the city tile's channel, in [0, 1]
    let h_raw = (CITY_H_BASE + CITY_H_SCALE * pow(c.density, CITY_H_EXP))
                * (0.70 + 0.60 * r4.x) * sprawl;
    let ns = clamp(floor(h_raw * CITY_PER_STOREY), 2.0, CITY_MAX_STOREYS);
    let h = ns * CITY_FLOOR_H;
    c.height = h;

    // Footprint grows with height, the way a real city's does. Low-rise is
    // a row of narrow frontages that fills its block; mid-rise fills the
    // block with one mass; a tower's side grows as the square root of its
    // height — 35 m at 70 m, 55 m at 180, 85 m at 400, 103 m at 600, the
    // run real towers follow from floor-plate economics and sway — and then
    // holds, continuous and never shrinking with height: what carries a
    // supertall is its taper, setbacks and spire, not bulk (a 3 km tower
    // 150 m wide read from across the city as a slab, where the skyline
    // Thomas loved was a cluster of needles; a step down in width at some
    // height read as wrong the other way). A slenderness floor (SLENDER_MAX)
    // widens only the very tallest. Where the tower leaves most of its plot
    // open it stands on a podium that fills the plot — on the multi-block
    // lot its peak claimed, when it stood among lower neighbours: the street
    // wall every real downtown tower keeps.
    let r4f = city_rand4(c.seed ^ vec2<u32>(0x6a09e667u, 0xbb67ae85u));
    var fw = plot_size * vec2<f32>(0.80 + 0.17 * r4.y, 0.80 + 0.17 * r4.z);
    var podium = false;
    var zp = 0.0;
    if (h >= CITY_TOWER_H) {
        // @domain: h >= CITY_TOWER_H > 0 on this branch
        let side = max(CITY_TOWER_SIDE_K * sqrt(min(h, CITY_TOWER_KNEE_H))
                           * (0.80 + 0.40 * r4.y),
                       h / CITY_SLENDER_MAX);
        let aspect = 0.72 + 0.28 * r4.z;
        let sides = select(vec2<f32>(side * aspect, side),
                           vec2<f32>(side, side * aspect), r4f.y < 0.5);
        fw = min(sides, plot_size * 0.94);
        // A podium where the tower leaves most of the plot open: three to
        // eight storeys (up to twelve under a supertall), nearly plot-wide,
        // and never more than a third of the tower.
        if (min(fw.x / plot_size.x, fw.y / plot_size.y) < 0.75) {
            podium = true;
            let storeys = floor(mix(3.0, select(8.0, 12.0, c.merged), r4f.z));
            zp = min(storeys * CITY_FLOOR_H,
                     floor(ns * 0.33333334) * CITY_FLOOR_H);
        }
    }
    let slack = plot_size - fw;
    let bmin2 = c.plot_min + slack * vec2<f32>(r4.w, r4b.x);
    let bmax2 = bmin2 + fw;
    let bc = 0.5 * (bmin2 + bmax2);
    let pod_fw = plot_size * (0.92 + 0.06 * r4f.w);
    let pod_min = pc - 0.5 * pod_fw;
    let pod_max = pc + 0.5 * pod_fw;

    // The architecture. Low buildings stay slabs; above 60 m the draw picks
    // among four archetypes, spires growing commoner with height, and above
    // SPIRE_H every tower is one of the two frustum forms — a 3 km
    // rectangular extrusion is a wall, a 3 km tapering shaft is a spire
    // (Thomas: not perfect rectangular prisms; architectures that grow, or
    // have a spire). Spire crowns are the majority up there: the needles
    // are what the skyline is.
    if (h > 60.0) {
        let spire_p = select(0.13, 0.30, h > 300.0);
        if (r4d.x < 0.20) {
            c.arch = 2;                   // growth: cantilevered buds
        } else if (r4d.x < 0.42) {
            c.arch = 3;                   // tapered shaft on a podium
        } else if (r4d.x < 0.42 + spire_p) {
            c.arch = 4;                   // spire crown
        }
    }
    if (h > CITY_SPIRE_H) {
        c.arch = select(3, 4, r4d.y < 0.60);
    }

    if (h < CITY_LOWRISE_H) {
        // Low-rise: a row of frontages, not one box. The block is filled at
        // the lower height and one run of it — a third to two thirds of the
        // long side, flush to one end — rises one to three storeys higher,
        // with sometimes a third step on that; from the street the block
        // reads as neighbours of different heights sharing party walls.
        c.arch = 0;
        let long_x = fw.x >= fw.y;
        let n_up = 1.0 + floor(3.0 * r4d.z);
        let z1 = max(h - n_up * CITY_FLOOR_H, CITY_FLOOR_H);
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, z1);
        if (z1 < h) {
            c.tiers = 2;
            let run = 0.33 + 0.34 * r4d.w;
            let at_hi = r4b.w < 0.5;
            var lo = bmin2;
            var hi = bmax2;
            if (long_x) {
                let len = fw.x * run;
                lo.x = select(bmin2.x, bmax2.x - len, at_hi);
                hi.x = lo.x + len;
            } else {
                let len = fw.y * run;
                lo.y = select(bmin2.y, bmax2.y - len, at_hi);
                hi.y = lo.y + len;
            }
            c.b2min = vec3<f32>(lo, z1);
            c.b2max = vec3<f32>(hi, h);
            if (h - z1 >= 2.0 * CITY_FLOOR_H && r4f.x < 0.40) {
                // A last storey or two over the far half of the run.
                c.tiers = 3;
                let z2 = h - CITY_FLOOR_H;
                c.b2max.z = z2;
                var lo3 = lo;
                var hi3 = hi;
                if (long_x) {
                    let len3 = 0.5 * (hi.x - lo.x);
                    lo3.x = select(lo.x, hi.x - len3, at_hi);
                    hi3.x = lo3.x + len3;
                } else {
                    let len3 = 0.5 * (hi.y - lo.y);
                    lo3.y = select(lo.y, hi.y - len3, at_hi);
                    hi3.y = lo3.y + len3;
                }
                c.b3min = vec3<f32>(lo3, z2);
                c.b3max = vec3<f32>(hi3, h);
            }
        }
    } else if (c.arch == 3) {
        // Podium + a shaft whose cross-section closes toward the crown. The
        // podium is the plot-wide one where the tower left room for it,
        // otherwise a base course of the tower's own footprint.
        zp = select(max(floor(ns * 0.12) * CITY_FLOOR_H,
                        CITY_FLOOR_H), zp, podium);
        c.b1min = vec3<f32>(select(bmin2, pod_min, podium), CITY_GROUND_Z);
        c.b1max = vec3<f32>(select(bmax2, pod_max, podium), zp);
        let fw_shaft = fw * select(0.90, 1.0, podium);
        c.has_frustum = true;
        c.fmin = vec3<f32>(bc - 0.5 * fw_shaft, zp);
        c.fmax = vec3<f32>(bc + 0.5 * fw_shaft, h);
        // A supertall's shaft closes a little further than a tower's and
        // ends in a needle. Closed much
        // further (to a fifth) the shafts read as cones, not the needles of
        // the skyline Thomas loved.
        c.fscale = select(0.55 + 0.25 * r4d.z, 0.40 + 0.20 * r4d.z,
                          h > CITY_SUPERTALL_H);
    } else if (c.arch == 4) {
        // A straight shaft to seven tenths, then a steep spire — on the
        // podium, when there is one, as a stacked tier. Supertalls keep a
        // long straight shaft too (a spire over half the height reads as a
        // cone) but end in a finer point: the needles of the skyline.
        let zs = floor(ns * select(0.70, 0.62 + 0.10 * r4d.w,
                                   h > CITY_SUPERTALL_H)) * CITY_FLOOR_H;
        if (podium) {
            c.tiers = 2;
            c.b1min = vec3<f32>(pod_min, CITY_GROUND_Z);
            c.b1max = vec3<f32>(pod_max, zp);
            c.b2min = vec3<f32>(bmin2, zp);
            c.b2max = vec3<f32>(bmax2, zs);
        } else {
            c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
            c.b1max = vec3<f32>(bmax2, zs);
        }
        c.has_frustum = true;
        c.fmin = vec3<f32>(bmin2, zs);
        c.fmax = vec3<f32>(bmax2, h);
        c.fscale = select(0.12 + 0.10 * r4d.z, 0.06 + 0.08 * r4d.z,
                          h > CITY_SUPERTALL_H);
    } else if (c.arch == 2) {
        // Growth: a straight tower with one or two boxes budding from its
        // upper flanks, cantilevered over the street. Buds stay inside the
        // lot so the DDA's per-cell testing stays exact.
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, h);
        c.tiers = 2;
        let side = i32(floor(r4d.z * 4.0));   // which flank the bud rides
        let bz0 = floor(ns * (0.42 + 0.20 * r4d.w))
                  * CITY_FLOOR_H;
        let bz1 = min(bz0 + floor((0.10 + 0.10 * r4b.y) * ns)
                            * CITY_FLOOR_H + CITY_FLOOR_H,
                      h - CITY_FLOOR_H);
        let reach = 0.22 * min(fw.x, fw.y);
        var bud_min = vec2<f32>(0.0);
        var bud_max = vec2<f32>(0.0);
        if (side == 0) {
            bud_min = vec2<f32>(bmax2.x - 1.0, bmin2.y + 0.2 * fw.y);
            bud_max = vec2<f32>(bmax2.x + reach, bmax2.y - 0.2 * fw.y);
        } else if (side == 1) {
            bud_min = vec2<f32>(bmin2.x - reach, bmin2.y + 0.2 * fw.y);
            bud_max = vec2<f32>(bmin2.x + 1.0, bmax2.y - 0.2 * fw.y);
        } else if (side == 2) {
            bud_min = vec2<f32>(bmin2.x + 0.2 * fw.x, bmax2.y - 1.0);
            bud_max = vec2<f32>(bmax2.x - 0.2 * fw.x, bmax2.y + reach);
        } else {
            bud_min = vec2<f32>(bmin2.x + 0.2 * fw.x, bmin2.y - reach);
            bud_max = vec2<f32>(bmax2.x - 0.2 * fw.x, bmin2.y + 1.0);
        }
        // Clamped to the lot minus a hair, so no prim leaves it — and to
        // the plot on avenue edges, whose corridor is skyway's empty air.
        let bud_lo = cmin + vec2<f32>(
            select(0.5, hx0, city_is_avenue(base.x)),
            select(0.5, hy0, city_is_avenue(base.y)));
        let bud_hi = cmax - vec2<f32>(
            select(0.5, hx1, city_is_avenue(base.x + span.x)),
            select(0.5, hy1, city_is_avenue(base.y + span.y)));
        bud_min = max(bud_min, bud_lo);
        bud_max = min(bud_max, bud_hi);
        c.b2min = vec3<f32>(bud_min, bz0);
        c.b2max = vec3<f32>(bud_max, bz1);
        if (h > 200.0 && r4b.y > 0.55) {
            c.tiers = 3;
            let cz0 = floor(ns * (0.68 + 0.12 * r4b.w))
                      * CITY_FLOOR_H;
            let cz1 = min(cz0 + 8.0 * CITY_FLOOR_H, h - CITY_FLOOR_H);
            // The second bud rides the opposite flank: mirror about bc.
            let m0 = 2.0 * bc - bud_max;
            let m1 = 2.0 * bc - bud_min;
            c.b3min = vec3<f32>(max(min(m0, m1), bud_lo), cz0);
            c.b3max = vec3<f32>(min(max(m0, m1), bud_hi), cz1);
        }
    } else if (podium) {
        // Podium, tower, crown: the setback stack with the street wall at
        // the bottom and the tower's own crown setback stepping in over
        // its top third or so.
        c.tiers = 3;
        let zc = floor(ns * (0.62 + 0.12 * r4b.w))
                 * CITY_FLOOR_H;
        c.b1min = vec3<f32>(pod_min, CITY_GROUND_Z);
        c.b1max = vec3<f32>(pod_max, zp);
        c.b2min = vec3<f32>(bmin2, zp);
        c.b2max = vec3<f32>(bmax2, zc);
        let fw3 = fw * (0.55 + 0.15 * r4b.z);
        c.b3min = vec3<f32>(bc - 0.5 * fw3, zc);
        c.b3max = vec3<f32>(bc + 0.5 * fw3, h);
    } else {
        // Setback tiers: the wedding cake (the founding archetype).
        var z1 = h;
        if (h > 90.0 && r4b.y > 0.45) {
            c.tiers = 2;
            z1 = floor(ns * 0.60) * CITY_FLOOR_H;
            if (h > 180.0 && r4b.y > 0.80) {
                c.tiers = 3;
                z1 = floor(ns * 0.50) * CITY_FLOOR_H;
            }
        }
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, z1);
        if (c.tiers >= 2) {
            var z2 = h;
            var f2 = 0.68;
            if (c.tiers == 3) {
                z2 = floor(ns * 0.78) * CITY_FLOOR_H;
                f2 = 0.72;
            }
            let fw2 = fw * f2;
            c.b2min = vec3<f32>(bc - 0.5 * fw2, z1);
            c.b2max = vec3<f32>(bc + 0.5 * fw2, z2);
            if (c.tiers == 3) {
                let fw3 = fw * 0.45;
                c.b3min = vec3<f32>(bc - 0.5 * fw3, z2);
                c.b3max = vec3<f32>(bc + 0.5 * fw3, h);
            }
        }
    }

    // The window style: the lattice itself varies building to building, or
    // mid-distance towers all wear the same texture and blur together
    // (Thomas, in flight). Style follows height and architecture loosely —
    // curtain walls on towers, punched openings on low-rise — and pitch is
    // jittered so even two buildings sharing a style disagree in period.
    let r4e = city_rand4(c.seed ^ vec2<u32>(0xb5297a4du, 0x68e31da4u));
    var style = 0;
    if (h > 350.0 || c.arch == 3) {
        if (r4e.x < 0.45) { style = 3; }
        else if (r4e.x < 0.70) { style = 0; }
        else if (r4e.x < 0.90) { style = 1; }
        else { style = 2; }
    } else if (h < 40.0) {
        if (r4e.x < 0.50) { style = 4; }
        else if (r4e.x < 0.80) { style = 0; }
        else { style = 1; }
    } else {
        if (r4e.x < 0.30) { style = 0; }
        else if (r4e.x < 0.55) { style = 1; }
        else if (r4e.x < 0.75) { style = 2; }
        else if (r4e.x < 0.90) { style = 3; }
        else { style = 4; }
    }
    c.win_style = style;
    if (style == 1) {          // ribbon: low continuous bands, thin mullions
        c.win_pitch = 3.0 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.04, 0.34);
        c.pane_hi = vec2<f32>(0.96, 0.86);
    } else if (style == 2) {   // vertical strips: narrow, full-height
        c.win_pitch = 2.2 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.32, 0.10);
        c.pane_hi = vec2<f32>(0.58, 0.94);
    } else if (style == 3) {   // curtain wall: all glass, hairline mullions
        c.win_pitch = 2.8 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.05, 0.08);
        c.pane_hi = vec2<f32>(0.95, 0.94);
        c.palette_bias = c.palette_bias * 0.5;   // cooler house spectrum
    } else if (style == 4) {   // punched: small openings in a lot of wall
        c.win_pitch = 2.6 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.30, 0.38);
        c.pane_hi = vec2<f32>(0.70, 0.80);
        c.palette_bias = min(c.palette_bias * 1.4 + 0.1, 1.0); // warmer
    } else {                   // grid: the founding lattice
        c.win_pitch = 3.4 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.14, 0.28);
        c.pane_hi = vec2<f32>(0.86, 0.90);
    }
    // A fifth of buildings keep a house color: every window draws near one
    // palette coordinate, so a tower can be THE cyan tower — the color
    // balance is a property of the building, not only of the window
    // (Thomas, 2026-08-20).
    let mono_draw = city_rand4(
        c.seed ^ vec2<u32>(0x3c6ef372u, 0xa54ff53au));
    c.win_mono = select(-1.0, mono_draw.y, mono_draw.x < 0.20);
    // Widen the occupancy spread too: some towers near-dark, some ablaze,
    // so neighbours separate at range by brightness as well as texture.
    c.lit_frac = clamp(
        c.lit_frac * (0.45 + 1.35 * r4e.z * r4e.z), 0.02, 0.50);

    // Roof furniture belongs on roofs: a spire crown has no flat top to
    // stand a mast on, and a tapered shaft only offers its shrunken crown
    // cap (Thomas, 2026-08-20: bridges/antennas only on the building types
    // that can carry them — components read cc.arch for the same rule).
    // A supertall's tapered shaft always ends in a needle: a mast scaled to
    // the tower (6-11% of its height, as thick as a needle that long must
    // be), centred on the crown, and kept under the DDA's z gate.
    let needle = c.arch == 3 && h > CITY_SUPERTALL_H;
    if (h > CITY_MAST_MIN_H && (r4b.z < 0.65 || needle) && c.arch != 4) {
        c.has_mast = true;
        var mast_h = 12.0 + 45.0 * r4b.w;
        var cross = CITY_MAST_CROSS;
        var moff = vec2<f32>(0.0);
        if (needle) {
            mast_h = min(h * (0.06 + 0.05 * r4b.w), CITY_SLAB_TOP - h - 1.0);
            cross = max(CITY_MAST_CROSS, 0.03 * mast_h);
        } else {
            let crown = select(1.0, c.fscale * 0.85, c.arch == 3);
            moff = (vec2<f32>(r4c.w, r4.x) - 0.5) * fw * (0.4 * crown);
        }
        let mc = bc + moff;
        c.mast_min = vec3<f32>(mc - vec2<f32>(0.5 * cross), h);
        c.mast_max = vec3<f32>(mc + vec2<f32>(0.5 * cross), h + mast_h);
        c.top_z = h + mast_h;
    } else {
        c.top_z = h;
    }
    // The crown's headroom, as a bound: it costs no draw (city_crown).
    if (h > CITY_CROWN_MIN_H && c.arch != 4) {
        c.top_z = max(c.top_z, h + CITY_CROWN_H_HI);
    }
    return c;
}

// Slab test against an arbitrary box (ray_box is hardwired to the volume).
fn city_box_hit(o: vec3<f32>, inv_dir: vec3<f32>,
                bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let t0 = (bmin - o) * inv_dir;
    let t1 = (bmax - o) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2<f32>(max(max(tmin.x, tmin.y), tmin.z),
                     min(min(tmax.x, tmax.y), tmax.z));
}

fn city_box_normal(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>)
        -> vec3<f32> {
    let c = 0.5 * (bmin + bmax);
    let h = max(0.5 * (bmax - bmin), vec3<f32>(1e-4));
    let q = (p - c) / h;
    let aq = abs(q);
    if (aq.z >= aq.x && aq.z >= aq.y) {
        return vec3<f32>(0.0, 0.0, sign(q.z));
    }
    if (aq.x >= aq.y) {
        return vec3<f32>(sign(q.x), 0.0, 0.0);
    }
    return vec3<f32>(0.0, sign(q.y), 0.0);
}

// Exact intersection with a rectangular frustum: cross-section centred on
// (fmin.xy+fmax.xy)/2, scale 1 at fmin.z shrinking (or growing) linearly to
// fscale at fmax.z. Each tilted face is linear in the ray parameter, so the
// test is the slab method with four sloped slabs — closed form, no
// iteration. The normal of the entering face comes back with the interval.
struct CityFrustumHit {
    t_near: f32,
    t_far: f32,
    normal: vec3<f32>,
}

fn city_frustum_hit(o: vec3<f32>, dir: vec3<f32>,
                    fmin: vec3<f32>, fmax: vec3<f32>, fscale: f32)
        -> CityFrustumHit {
    var res: CityFrustumHit;
    res.t_near = 1.0;
    res.t_far = 0.0;   // empty interval = miss unless everything below runs
    let c2 = 0.5 * (fmin.xy + fmax.xy);
    let h2 = max(0.5 * (fmax.xy - fmin.xy), vec2<f32>(1e-4));
    let zspan = max(fmax.z - fmin.z, 1e-4);
    let sp = (fscale - 1.0) / zspan;          // ds/dz
    // s along the ray: s(t) = s_a + s_b t
    let s_a = 1.0 + sp * (o.z - fmin.z);
    let s_b = sp * dir.z;

    var t_lo = -1e30;
    var t_hi = 1e30;
    var n_lo = vec3<f32>(0.0, 0.0, -1.0);

    // z slab.
    if (abs(dir.z) > 1e-9) {
        let ta = (fmin.z - o.z) / dir.z;
        let tb = (fmax.z - o.z) / dir.z;
        let zn = select(vec3<f32>(0.0, 0.0, 1.0),
                        vec3<f32>(0.0, 0.0, -1.0), dir.z > 0.0);
        if (min(ta, tb) > t_lo) {
            t_lo = min(ta, tb);
            n_lo = zn;
        }
        t_hi = min(t_hi, max(ta, tb));
    } else if (o.z < fmin.z || o.z > fmax.z) {
        return res;
    }

    // Four sloped faces, each "A + B t <= 0". B > 0 caps t from above;
    // B < 0 raises the floor (and owns the entry normal); B ~ 0 is a
    // parallel ray, inside or out by the sign of A.
    for (var f: i32 = 0; f < 4; f = f + 1) {
        var a: f32;
        var b: f32;
        var nf: vec3<f32>;
        if (f == 0) {
            a = (o.x - c2.x) - h2.x * s_a;
            b = dir.x - h2.x * s_b;
            nf = normalize(vec3<f32>(1.0, 0.0, -h2.x * sp));
        } else if (f == 1) {
            a = -(o.x - c2.x) - h2.x * s_a;
            b = -dir.x - h2.x * s_b;
            nf = normalize(vec3<f32>(-1.0, 0.0, -h2.x * sp));
        } else if (f == 2) {
            a = (o.y - c2.y) - h2.y * s_a;
            b = dir.y - h2.y * s_b;
            nf = normalize(vec3<f32>(0.0, 1.0, -h2.y * sp));
        } else {
            a = -(o.y - c2.y) - h2.y * s_a;
            b = -dir.y - h2.y * s_b;
            nf = normalize(vec3<f32>(0.0, -1.0, -h2.y * sp));
        }
        if (abs(b) < 1e-9) {
            if (a > 0.0) {
                return res;
            }
        } else {
            let tb = -a / b;
            if (b > 0.0) {
                t_hi = min(t_hi, tb);
            } else if (tb > t_lo) {
                t_lo = tb;
                n_lo = nf;
            }
        }
    }
    res.t_near = t_lo;
    res.t_far = t_hi;
    res.normal = n_lo;
    return res;
}

struct CityHit {
    hit: bool,
    t: f32,
    pos: vec3<f32>,
    normal: vec3<f32>,
    kind: i32,   // 0 ground, 1 facade, 2 roof, 3 mast, 4 beacon
    cell: vec2<i32>,
}

// 2D DDA over the block grid. Exact: every building the ray passes is slab-
// tested; the only rejections are cells whose whole z-extent the segment
// clears. Opaque geometry, so the first hit wins. `t_max` bounds the walk:
// the view ray passes CITY_TRACE_RANGE, a wet street's reflected ray its own
// short range (a rising ray would otherwise walk cells until it cleared
// CITY_SLAB_TOP, kilometres up). `mode` picks what is traced; every call
// site passes a constant, so each call inlines only its own mode's code:
//   CITY_TRACE_ALL    buildings, ground, extra_trace and cell_props hooks —
//                     the view ray (surface_pass_city);
//   CITY_TRACE_PLAIN  buildings and ground only — the wet street's reflected
//                     ray (a second inlined copy of every component's hooks
//                     is what a reflection costs where none is traced);
//   CITY_TRACE_NEAR   the near_props hooks only (street furniture), for the
//                     street stage (surface_street_city), which already
//                     knows the nearest opaque surface and asks only what
//                     stands in front of it.
const CITY_TRACE_ALL: i32 = 0;
const CITY_TRACE_PLAIN: i32 = 1;
const CITY_TRACE_NEAR: i32 = 2;
fn city_trace(o: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32,
              mode: i32) -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let cell = u.tile_cell;

    var t0 = t_min;
    if (o.z + t0 * dir.z > CITY_SLAB_TOP) {
        if (dir.z >= -1e-7) {
            return res;
        }
        t0 = max(t0, (CITY_SLAB_TOP - o.z) / dir.z);
        if (t0 > t_max) {
            return res;
        }
    }
    let p0 = o + t0 * dir;

    var ci = vec2<i32>(floor(p0.xy / cell));
    let stp = vec2<i32>(select(-1, 1, dir.x > 0.0),
                        select(-1, 1, dir.y > 0.0));
    let inv2 = vec2<f32>(
        select(1e30, 1.0 / dir.x, abs(dir.x) > 1e-9),
        select(1e30, 1.0 / dir.y, abs(dir.y) > 1e-9)
    );
    let cmin0 = vec2<f32>(ci) * cell;
    var tmax = vec2<f32>(
        t0 + (select(cmin0.x, cmin0.x + cell, dir.x > 0.0) - p0.x) * inv2.x,
        t0 + (select(cmin0.y, cmin0.y + cell, dir.y > 0.0) - p0.y) * inv2.y
    );
    // A zero direction component never crosses that axis.
    tmax = vec2<f32>(
        select(1e30, tmax.x, abs(dir.x) > 1e-9),
        select(1e30, tmax.y, abs(dir.y) > 1e-9)
    );
    let tdelta = vec2<f32>(cell * abs(inv2.x), cell * abs(inv2.y));

    let inv3 = vec3<f32>(
        inv2.x, inv2.y,
        select(1e30, 1.0 / dir.z, abs(dir.z) > 1e-9)
    );
    var ground_t = 1e30;
    if (dir.z < -1e-9) {
        ground_t = (CITY_GROUND_Z - o.z) / dir.z;
    }

    // Grid-independent component geometry (elevated highways and their
    // kin): traced once, and the nearest of it and whatever the DDA finds
    // wins at every exit below.
    var extra: CityHit;
    extra.hit = false;
    extra.t = 1e30;
    if (mode == CITY_TRACE_ALL) {
        extra = cc_extra_trace(o, dir, inv3);
    }

    var t_cur = t0;
    for (var i: i32 = 0; i < CITY_TRACE_MAX_CELLS; i = i + 1) {
        let t_exit = min(min(tmax.x, tmax.y), t_max);
        let z_a = o.z + t_cur * dir.z;
        let z_b = o.z + t_exit * dir.z;
        if (min(z_a, z_b) > CITY_SLAB_TOP && dir.z >= 0.0) {
            return extra;   // climbed out of the slab for good
        }

        let cc = city_cell(ci);
        var best_t = 1e30;
        var best_min = vec3<f32>(0.0);
        var best_max = vec3<f32>(0.0);
        var best_kind = -1;
        var best_fnormal = vec3<f32>(0.0);
        if (mode != CITY_TRACE_NEAR && cc.built
            && min(z_a, z_b) < cc.top_z + 1.0) {
            let h1 = city_box_hit(o, inv3, cc.b1min, cc.b1max);
            if (h1.x <= h1.y && h1.x > 0.0 && h1.x < best_t) {
                best_t = h1.x; best_min = cc.b1min; best_max = cc.b1max;
                best_kind = 1;
            }
            if (cc.tiers >= 2) {
                let h2 = city_box_hit(o, inv3, cc.b2min, cc.b2max);
                if (h2.x <= h2.y && h2.x > 0.0 && h2.x < best_t) {
                    best_t = h2.x; best_min = cc.b2min; best_max = cc.b2max;
                    best_kind = 1;
                }
            }
            if (cc.tiers >= 3) {
                let h3 = city_box_hit(o, inv3, cc.b3min, cc.b3max);
                if (h3.x <= h3.y && h3.x > 0.0 && h3.x < best_t) {
                    best_t = h3.x; best_min = cc.b3min; best_max = cc.b3max;
                    best_kind = 1;
                }
            }
            if (cc.has_frustum) {
                let hf = city_frustum_hit(o, dir, cc.fmin, cc.fmax,
                                          cc.fscale);
                if (hf.t_near <= hf.t_far && hf.t_near > 0.0
                    && hf.t_near < best_t) {
                    best_t = hf.t_near;
                    best_fnormal = hf.normal;
                    best_kind = 5;
                }
            }
            // The crown, behind a gate that costs no draw: only a cell whose
            // ray segment crosses the crown's z band hashes for it. Without
            // the gate every tall cell a ray passes hashes — on a street
            // view, looking up past towers, that was +7% of the stage.
            if (city_crown_possible(cc)
                && max(z_a, z_b) >= cc.height
                && min(z_a, z_b) <= cc.height + CITY_CROWN_H_HI) {
                let cr = city_crown(cc, ci);
                if (cr.kind == 1) {
                    let pmin = vec3<f32>(cr.lo, cr.z0);
                    let pmax = vec3<f32>(cr.hi, cr.z1);
                    let hp = city_box_hit(o, inv3, pmin, pmax);
                    if (hp.x <= hp.y && hp.x > 0.0 && hp.x < best_t) {
                        best_t = hp.x; best_min = pmin; best_max = pmax;
                        best_kind = 1;
                    }
                }
            }
            if (cc.has_mast) {
                let hm = city_box_hit(o, inv3, cc.mast_min, cc.mast_max);
                if (hm.x <= hm.y && hm.x > 0.0 && hm.x < best_t) {
                    best_t = hm.x; best_min = cc.mast_min;
                    best_max = cc.mast_max;
                    best_kind = 3;
                }
            }
        }
        if (mode != CITY_TRACE_NEAR
            && ground_t > t_cur - 1e-4 && ground_t <= t_exit + 1e-4
            && ground_t < best_t) {
            best_t = ground_t;
            best_kind = 0;
        }
        if (best_kind >= 0 && best_t <= t_exit + 1e-4) {
            res.hit = true;
            res.t = best_t;
            res.pos = o + best_t * dir;
            res.cell = ci;
            if (best_kind == 0) {
                res.normal = vec3<f32>(0.0, 0.0, 1.0);
                res.kind = 0;
            } else if (best_kind == 5) {
                // A frustum face carries its own normal; the crown cap is
                // roof, the sloped skin is facade.
                res.normal = best_fnormal;
                res.kind = select(1, 2, res.normal.z > 0.7);
            } else {
                res.normal = city_box_normal(res.pos, best_min, best_max);
                if (best_kind == 3) {
                    res.kind = select(
                        3, 4,
                        res.pos.z > best_max.z - CITY_BEACON_LEN);
                } else {
                    res.kind = select(1, 2, res.normal.z > 0.5);
                }
            }
        }
        // Near-field props: the whole prop budget lives inside
        // CITY_PROP_RANGE, so the far city never pays for a garbage can.
        if (mode == CITY_TRACE_NEAR) {
            let nh = cc_near_props_trace(o, dir, inv3, t_cur, t_exit, ci, cc);
            if (nh.hit && nh.t < res.t && nh.t <= t_exit + 1e-3) {
                res = nh;
            }
        }
        if (mode == CITY_TRACE_ALL && t_cur < CITY_PROP_RANGE) {
            let ph = cc_cell_props_trace(o, dir, inv3, t_cur, t_exit, ci, cc);
            if (ph.hit && ph.t < res.t && ph.t <= t_exit + 1e-3) {
                res = ph;
            }
        }
        if (res.hit) {
            if (extra.hit && extra.t < res.t) {
                return extra;
            }
            return res;
        }

        if (t_exit >= t_max) {
            return extra;
        }
        if (tmax.x < tmax.y) {
            ci.x = ci.x + stp.x;
            t_cur = tmax.x;
            tmax.x = tmax.x + tdelta.x;
        } else {
            ci.y = ci.y + stp.y;
            t_cur = tmax.y;
            tmax.y = tmax.y + tdelta.y;
        }
    }
    return extra;
}

// One window's color: a palette draw shifted by the building's own mood.
fn city_window_color(draw: f32, bias: f32) -> vec3<f32> {
    let warm_cut = 0.55 + 0.30 * (bias - 0.5);
    if (draw < warm_cut) {
        return vec3<f32>(1.0, 0.62, 0.30);        // tungsten / sodium spill
    }
    if (draw < warm_cut + 0.30) {
        return vec3<f32>(0.75, 0.85, 1.00);       // fluorescent / LED
    }
    if (draw < warm_cut + 0.38) {
        return vec3<f32>(0.15, 0.95, 0.85);       // cyan neon
    }
    return vec3<f32>(1.00, 0.25, 0.55);           // magenta neon
}

// A lit room's colour: what the light inside a window is. Incandescent and
// sodium interiors dominate a city's night; warm LED, cool fluorescent and
// the odd neon are the few. The building's palette_bias leans the warm share
// (a mixed house draws freely; city_window_rated keeps a monochrome house
// near its one draw). Its mean is about (0.97, 0.59, 0.32). Shops draw from
// it too; signs and other lit dressing draw from city_window_color instead.
fn city_room_color(draw: f32, bias: f32) -> vec3<f32> {
    for (var k: i32 = 0; k < 5; k = k + 1) {
        if (draw < city_room_band_edge(k, bias)) {
            return city_room_band_colour(k);
        }
    }
    return city_room_band_colour(5);
}

// The palette's bands: band k is drawn below its upper edge and at or
// above the one before; the last runs to 1.
fn city_room_band_edge(k: i32, bias: f32) -> f32 {
    let cut = 0.60 + 0.20 * (bias - 0.5);
    switch k {
        case 0: { return cut; }
        case 1: { return cut + 0.22; }
        case 2: { return cut + 0.32; }
        case 3: { return cut + 0.36; }
        case 4: { return cut + 0.38; }
        default: { return 1e9; }
    }
}

fn city_room_band_colour(k: i32) -> vec3<f32> {
    switch k {
        case 0: { return vec3<f32>(1.0, 0.58, 0.26); }    // tungsten
        case 1: { return vec3<f32>(1.0, 0.45, 0.14); }    // amber / sodium
        case 2: { return vec3<f32>(0.95, 0.82, 0.62); }   // warm LED
        case 3: { return vec3<f32>(0.75, 0.85, 1.00); }   // fluorescent
        case 4: { return vec3<f32>(0.15, 0.95, 0.85); }   // cyan neon
        default: { return vec3<f32>(1.00, 0.25, 0.55); } // magenta neon
    }
}

// Streetlight pools on the asphalt around block edges. Evaluated for the
// nearest x- and y- boundary lines; each street has a lamp line either side
// — where the street exists. The nearest boundary lines are always the
// cell's own edges, so the cell's one street word says which of the two
// lines carries lamps; a line through open scrub is dark.
fn city_street_pools(p: vec2<f32>) -> f32 {
    let cell = u.tile_cell;
    let ci = vec2<i32>(floor(p / cell));
    let st = city_word(textureLoad(ocean_normals, city_tile_cell(ci), 0).a);
    var pool = 0.0;
    // Nearest x-boundary (a street running along y).
    let bx = round(p.x / cell) * cell;
    let bxi = i32(round(p.x / cell));
    if (((st >> select(0u, 1u, bxi > ci.x)) & 1u) != 0u) {
        let ax = select(1.0, 1.6, city_is_avenue(bxi));
        let inx = city_lamp_inset(bxi);
        for (var s: i32 = 0; s < 2; s = s + 1) {
            let lx = bx + select(-inx, inx, s == 1);
            let ly = city_lamp_station(bxi, p.y);
            let d2 = (p.x - lx) * (p.x - lx) + (p.y - ly) * (p.y - ly);
            pool = pool
                + ax * exp(-d2 / (2.0 * CITY_LAMP_SIGMA * CITY_LAMP_SIGMA));
        }
    }
    // Nearest y-boundary (a street running along x).
    let by = round(p.y / cell) * cell;
    let byi = i32(round(p.y / cell));
    if (((st >> select(2u, 3u, byi > ci.y)) & 1u) != 0u) {
        let ay = select(1.0, 1.6, city_is_avenue(byi));
        let iny = city_lamp_inset(byi);
        for (var s: i32 = 0; s < 2; s = s + 1) {
            let ly = by + select(-iny, iny, s == 1);
            let lx = city_lamp_station(byi, p.x);
            let d2 = (p.x - lx) * (p.x - lx) + (p.y - ly) * (p.y - ly);
            pool = pool
                + ay * exp(-d2 / (2.0 * CITY_LAMP_SIGMA * CITY_LAMP_SIGMA));
        }
    }
    return pool;
}
