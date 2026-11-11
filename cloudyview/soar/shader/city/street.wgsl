// THE STREET SURFACE
// ===========================================================================
//
// What a street is, in cross-section, measured outward from the BUILDING
// LINE (the plot edge) toward the centreline (the cell boundary):
//
//   plot | sidewalk | kerb | gutter | parking lane | travel lanes | centreline
//          3.0 m      0.15   0.45 m    2.4 m
//   (4.5 m of sidewalk on an avenue)
//
// The sidewalk is a rounded rectangle around the plot, raised CITY_KERB_H,
// and its edge is the KERB LINE. One signed distance to that rectangle
// (city_kerb_sd) names every zone of the section — negative on the
// sidewalk, zero at the kerb face, then the gutter, the parking lane and the
// lanes at increasing distance — and it turns the corner with the kerb,
// which is the thing a street most obviously does and the old pool-lit
// plane never did. The core's lamp lattice (city_lamp_inset: 2.5 m in from
// the building line) now stands on the sidewalk, at its kerb side, which is
// where a real pole is.
//
// Geometry and shading split the way the rest of the city does. The raised
// slab is traced by streetlife (a cell prop, within reach of the camera and
// while its 15 cm step is resolvable), and it reports its TOP as ground,
// kind 0 — so the slab top and the z = 0 plane the DDA falls back to beyond
// it are shaded by this one function and cannot disagree. Everything here is
// a function of xy and the cell alone.
//
// Every pattern dissolves into its own mean (SPEC rule 3): painted lines,
// flag joints and grate bars are box-filtered analytically over the
// footprint (city_stripes, exact at every fp), noise textures hand over to
// their mean of 1, and puddles to their coverage. The one term that cannot
// be filtered analytically — the lamp pools under the knee, multiplied by
// the zone albedos they fall on — has its far-field mean MEASURED
// (CITY_GROUND_FAR_*, see tools/street_means.py).

const CITY_SIDEWALK_W: f32 = 3.0;      // building line to kerb, minor street
const CITY_SIDEWALK_W_AVE: f32 = 4.5;  // and on an avenue
const CITY_KERB_H: f32 = 0.15;         // the sidewalk above the carriageway
const CITY_KERB_R: f32 = 3.0;          // kerb radius where two streets meet
const CITY_KERB_TOP_W: f32 = 0.30;     // the kerbstone's own top face
const CITY_GUTTER_W: f32 = 0.45;       // concrete gutter pan off the kerb
const CITY_PARK_W: f32 = 2.4;          // parking lane, kerb outward
const CITY_FLAG: f32 = 1.5;            // sidewalk flag pitch
const CITY_JOINT_W: f32 = 0.018;       // the joint between two flags

// Diffuse response of each surface RELATIVE TO BARE DRY ASPHALT, which is
// what the lamp pools were calibrated on (the pool is the asphalt's
// radiance). Concrete under sodium is the bright band of every night street
// photograph; paint brighter still, and worn.
const CITY_REL_WALK: f32 = 1.5;
const CITY_REL_KERB: f32 = 2.6;
const CITY_REL_GUTTER: f32 = 1.7;
const CITY_REL_LOT: f32 = 1.3;          // unpaved yards and empty lots
const CITY_REL_PAINT: f32 = 4.2;
const CITY_PAINT_YELLOW: vec3<f32> = vec3<f32>(1.0, 0.78, 0.22);
const CITY_REL_IRON: f32 = 0.70;        // manhole covers, grate bars
const CITY_REL_SLOT: f32 = 0.08;        // the dark between grate bars

// After rain. The film is the fraction of a surface under water thin enough
// to take its texture's shape (a rough, glossy reflector); puddles are
// standing water (a mirror). Water fills the pores of asphalt and concrete
// and takes most of their diffuse light with it — wet asphalt is the darkest
// thing on a street, which is exactly what lets the reflections read.
const CITY_WET_ROAD: f32 = 0.90;
const CITY_WET_WALK: f32 = 0.85;
const CITY_WET_DARKEN: f32 = 0.97;
// Of a wet surface, the share whose film is smooth enough to be glossy: damp
// asphalt is a patchwork of film and dull aggregate, not a sheet of water.
const CITY_FILM_GLOSS: f32 = 0.4;
const CITY_PUDDLE_DARKEN: f32 = 0.85;
// Puddle mask: two octaves of value noise plus a bias per zone (gutters
// collect, the road crown sheds), thresholded. Past CITY_PUDDLE_LOD the mask
// is its own expected coverage at that bias, a logistic fitted to the
// noise's measured distribution (tools/street_means.py).
const CITY_PUDDLE_THR: f32 = 0.67;
const CITY_PUDDLE_EDGE: f32 = 0.015;
const CITY_PUDDLE_LOD: vec2<f32> = vec2<f32>(0.30, 1.20);
const CITY_PUDDLE_FIT: vec2<f32> = vec2<f32>(0.6700, 0.0950);

// Manholes: CITY_MH_PITCH apart along every centreline (a divisor of the
// block, so the lattice is the same on every block), set in a travel lane,
// present at a hashed share of the sites. The steam component reads the same
// sites through city_manhole, so a plume always rises from a cover.
const CITY_MH_PITCH: f32 = 30.0;
const CITY_MH_R: f32 = 0.34;
const CITY_MH_PRESENT: f32 = 0.55;
// Share of PRESENT sites that steam, by district (city_manhole_steam). A
// block owns about 3.3 present covers (two street edges x ~1.65 each), so
// 0.26 downtown is a steaming cover every block or so and 0.015 in the
// outskirts is one in twenty blocks. The ramp is the lamp scale's
// (city_lamp_scale): the same districts that burn brightest steam most.
const CITY_MH_STEAM_FRAC_LO: f32 = 0.015;
const CITY_MH_STEAM_FRAC_HI: f32 = 0.26;
// A steaming cover is hot: it and the asphalt round it stay dry on a wet
// street, a pale patch the plume rises out of, so the steam is seen to come
// from a cover and not from a random bit of street: the dark wet lid under
// it does not read, and from some poses the plume's own foot hides the lid
// itself. Dry out to
// CITY_MH_R + DRY_IN, fading to wet by CITY_MH_R + DRY_OUT.
const CITY_MH_DRY_IN: f32 = 0.35;
const CITY_MH_DRY_OUT: f32 = 0.45;
// A wet cover's gloss: tyre-polished iron under a film, sheened all over
// rather than mottled like the asphalt round it, so at night a lid reads as
// a lid — it catches the lamps (city_lamp_glint) and the lit windows (the
// film lobe), where dark iron alone is a dark disc on dark asphalt and the
// plume above it seems to rise from nowhere.
const CITY_MH_GLOSS: f32 = 0.9;
// Storm-drain grates: one in each gutter, this far from the cross street's
// centreline, beyond its crosswalk and stop line.
const CITY_GRATE_AT: f32 = 5.2;       // past the cross street's building line
const CITY_GRATE_HL: f32 = 0.45;      // half-length along the kerb

// The reflected ray (city_wet_lobe). Water's F0 is 0.02; a puddle is
// a near-mirror whose ripples still blur it a little, a damp film a rough
// one — rough enough at grazing incidence to stretch every light source into
// the long vertical streak that IS the wet-street look.
const CITY_REFL_F0: f32 = 0.02;
const CITY_REFL_ALPHA_PUDDLE: f32 = 0.02;
const CITY_REFL_ALPHA_DAMP: f32 = 0.08;
// How far a reflected ray looks for the city. Far enough to cross any
// canyon and reach the towers beyond it; a rising ray that has found nothing
// by then sees sky.
const CITY_REFL_RANGE: f32 = 600.0;
// Lobes that carry less than this are traced only by Russian roulette: a
// lobe of weight w < RR is traced with probability w / RR and counted RR / w
// times over, binary noise on the dim film (a steep view, a damp patch). At
// 0.05 rather than 0.15 a view down a wet avenue traces more of them: its
// one-sample spread 8 % lower (3.54 vs 3.90 levels on the street, with the
// stratified draws), its first 8 frames 8 % nearer the converged mean, for
// +1.6 % of the frame time. Unbiased in
// linear light; the accumulation averages tone-mapped samples, so fewer
// rare RR / w samples brighten a converged wet reflection a little (+0.6
// levels mean on reflective pixels), toward the tone map of the true mean.
const CITY_REFL_RR: f32 = 0.05;
// Footprint over which the traced reflection hands over to its mean. Past
// it a wet pixel reflects the environment estimate, and pays nothing.
const CITY_REFL_LOD: vec2<f32> = vec2<f32>(0.30, 0.80);

// Storefront light spilling onto the pavement in front of it.
const CITY_STORE_SPILL: f32 = 0.12;
const CITY_STORE_SPILL_L: f32 = 2.5;

// The lamp heads, as streetlife hangs them: CITY_LAMP_ARM out over the road
// from the lamp line, at CITY_LAMP_HEAD_Z.
const CITY_LAMP_HEAD_Z: f32 = 8.56;
const CITY_LAMP_ARM: f32 = 1.35;
// The ground knee (a downtown avenue's pools run to radiance ~10 and would
// clip to featureless white; the knee passes the dim outskirts untouched).
const CITY_POOL_KNEE: f32 = 0.30;
// The measured far-field mean of knee(s * pools) * albedo over the ground,
// as A s / (1 + B s) in the district's lamp scale s. Street is everything
// outside the plots; plot is the plot interiors, where only the pools' tails
// reach (tools/street_means.py).
const CITY_GROUND_FAR_STREET: vec2<f32> = vec2<f32>(0.0695, 0.2216);
const CITY_GROUND_FAR_PLOT: vec2<f32> = vec2<f32>(0.0084, 0.1047);

// Building line to kerb on the street a plot edge faces. The plot edge alone
// says which street that is: it sits CITY_STREET_HALF (plus the avenue
// extra) from the centreline, and the centreline is the nearest boundary.
fn city_sidewalk_w(plot_edge: f32) -> f32 {
    let cellm = u.tile_cell;
    let gap = abs(plot_edge - round(plot_edge / cellm) * cellm);
    return select(CITY_SIDEWALK_W, CITY_SIDEWALK_W_AVE,
                  gap > CITY_STREET_HALF + 1.0);
}

// Does this plot get sidewalks at all? Built URBAN plots do. Yards,
// industrial frontage and open land keep a bare kerb at their own edge.
fn city_has_walk(cc: CityCell) -> bool {
    return cc.built && cc.land_use == CITY_USE_URBAN;
}

// Which lamp lines light p: the street along y through the nearest
// x-boundary (x) and the one along x (y) — each only where that edge of p's
// cell carries a street, read from the cell's street word exactly as
// city_street_pools reads it, so pools, glints and lamp light agree.
fn city_lamp_lines(p: vec2<f32>) -> vec2<bool> {
    let cell = u.tile_cell;
    let ci = vec2<i32>(floor(p / cell));
    let st = city_word(textureLoad(ocean_normals, city_tile_cell(ci), 0).a);
    let bxi = i32(round(p.x / cell));
    let byi = i32(round(p.y / cell));
    return vec2<bool>(
        ((st >> select(0u, 1u, bxi > ci.x)) & 1u) != 0u,
        ((st >> select(2u, 3u, byi > ci.y)) & 1u) != 0u);
}

// The kerb line's rectangle before its corners are rounded (xy min, zw max):
// the plot pushed out by a sidewalk on each edge that fronts a street.
fn city_kerb_rect(cc: CityCell) -> vec4<f32> {
    if (!city_has_walk(cc)) {
        return vec4<f32>(cc.plot_min, cc.plot_max);
    }
    let w0 = select(0.0, city_sidewalk_w(cc.plot_min.x),
                    city_edge_street(cc, 0));
    let w1 = select(0.0, city_sidewalk_w(cc.plot_max.x),
                    city_edge_street(cc, 1));
    let w2 = select(0.0, city_sidewalk_w(cc.plot_min.y),
                    city_edge_street(cc, 2));
    let w3 = select(0.0, city_sidewalk_w(cc.plot_max.y),
                    city_edge_street(cc, 3));
    return vec4<f32>(cc.plot_min - vec2<f32>(w0, w2),
                     cc.plot_max + vec2<f32>(w1, w3));
}

// Signed distance to the kerb line: negative on the sidewalk (and over the
// plot it surrounds), zero at the kerb face, positive in the gutter and the
// road. The corner between two street edges is a CITY_KERB_R quarter circle
// about the plot corner; a corner where one edge has no street stays
// square, because nobody rounds a kerb into a fence line.
fn city_kerb_sd(p: vec2<f32>, cc: CityCell, kr: vec4<f32>) -> f32 {
    let c = 0.5 * (kr.xy + kr.zw);
    let hs = 0.5 * (kr.zw - kr.xy);
    let round_it = city_has_walk(cc)
        && city_edge_street(cc, select(0, 1, p.x > c.x))
        && city_edge_street(cc, select(2, 3, p.y > c.y));
    let r = select(0.0, min(CITY_KERB_R, min(hs.x, hs.y)), round_it);
    let q = abs(p - c) - hs + vec2<f32>(r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Where a point sits in the street network: its signed offset from the
// centreline of the street running along y (a line of constant x, the
// nearest cell boundary) and of the one running along x, those streets'
// half-widths (centreline to building line), and whether they are avenues.
struct CityRoad {
    sx: f32, sy: f32,      // signed offsets from the two centrelines
    hx: f32, hy: f32,      // half-widths
    kx: i32, ky: i32,      // the centrelines' boundary indices
    ave_x: bool, ave_y: bool,
}

fn city_road(p: vec2<f32>) -> CityRoad {
    let cellm = u.tile_cell;
    var r: CityRoad;
    r.kx = i32(round(p.x / cellm));
    r.ky = i32(round(p.y / cellm));
    r.sx = p.x - f32(r.kx) * cellm;
    r.sy = p.y - f32(r.ky) * cellm;
    r.ave_x = city_is_avenue(r.kx);
    r.ave_y = city_is_avenue(r.ky);
    r.hx = CITY_STREET_HALF + select(0.0, CITY_AVENUE_EXTRA, r.ave_x);
    r.hy = CITY_STREET_HALF + select(0.0, CITY_AVENUE_EXTRA, r.ave_y);
    return r;
}

// Which side of a minor street keeps its parking (a minor carriageway is
// 6 m: one parking lane and one travel lane). Avenues park both sides.
// +1 or -1, the sign of the offset the parking lane sits at.
fn city_park_side(k: i32) -> f32 {
    let h = pcg2d(vec2<u32>(city_line_seed(k) ^ 0x6c8e9cf5u, 0x2c1b3c6du));
    return select(-1.0, 1.0, (h.x & 1u) == 1u);
}

// --- analytic filters --------------------------------------------------------

// Coverage of [a, a + w) repeated every `period`, box-filtered over a
// footprint fp centred on x. Exact at every footprint: resolved, a stripe is
// a hard edge antialiased over one pixel; unresolved, it is its duty cycle
// w / period; and nothing in between moves the mean. Callers pass LOCAL
// coordinates (metres from a nearby origin), never the city frame's 1e5 m.
fn city_stripe_cdf(x: f32, a: f32, w: f32, period: f32) -> f32 {
    let q = x - a;
    let k = floor(q / period);
    return k * w + clamp(q - k * period, 0.0, w);
}

fn city_stripes(x: f32, a: f32, w: f32, period: f32, fp: f32) -> f32 {
    let h = 0.5 * max(fp, 1e-4);
    return (city_stripe_cdf(x + h, a, w, period)
            - city_stripe_cdf(x - h, a, w, period)) / (2.0 * h);
}

// One band [a, a + w), the same filter.
fn city_band(x: f32, a: f32, w: f32, fp: f32) -> f32 {
    let h = 0.5 * max(fp, 1e-4);
    return max(min(x + h, a + w) - max(x - h, a), 0.0) / (2.0 * h);
}

// The box-filtered step up through x = a: 0 below, 1 above.
fn city_step(x: f32, a: f32, fp: f32) -> f32 {
    let h = 0.5 * max(fp, 1e-4);
    return clamp((x - a + h) / (2.0 * h), 0.0, 1.0);
}

// Value noise of p * freq + off, mean 0.5 by symmetry — which is what lets a
// texture built as 1 + k (n - 0.5) dissolve into exactly 1. Periodic in p
// with period 1000 m: p is folded first and the lattice wraps with it (every
// frequency used here times 1000 m is a whole number of cells), so the hash
// only ever sees small integers. Fed raw city-frame coordinates (up to
// 92 km, times 4 for the aggregate) an f32 float hash collapses — the review
// measured the aggregate repeating every ~24 m in the far tile.
fn city_vnoise(p: vec2<f32>, freq: f32, off: f32) -> f32 {
    let q = p - 1000.0 * floor(p / 1000.0);
    let n = max(round(1000.0 * freq), 1.0);
    let x = q * freq + vec2<f32>(off);
    let i = floor(x);
    let f = x - i;
    let w = f * f * (3.0 - 2.0 * f);
    let i0 = i - n * floor(i / n);
    let i1 = select(i0 + 1.0, vec2<f32>(0.0), i0 + 1.0 >= vec2<f32>(n));
    let a = city_u01(pcg2d(vec2<u32>(u32(i0.x), u32(i0.y))).x);
    let b = city_u01(pcg2d(vec2<u32>(u32(i1.x), u32(i0.y))).x);
    let c = city_u01(pcg2d(vec2<u32>(u32(i0.x), u32(i1.y))).x);
    let d = city_u01(pcg2d(vec2<u32>(u32(i1.x), u32(i1.y))).x);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// A mean-1 texture, 1 + amp (n - 0.5) over city_vnoise, dissolving into its
// mean 1 as the footprint crosses the band `lod` — and not evaluated at all
// once it has: past the band the value IS the mean, so skipping the noise
// changes nothing but the cost.
fn city_detail(p: vec2<f32>, freq: f32, off: f32, amp: f32, fp: f32,
               lod: vec2<f32>) -> f32 {
    let k = smoothstep(lod.x, lod.y, fp);
    if (k >= 1.0) {
        return 1.0;
    }
    return mix(1.0 + amp * (city_vnoise(p, freq, off) - 0.5), 1.0, k);
}

// --- manholes -----------------------------------------------------------------

// Site j on centreline k of the streets running along y (`along_y`) or
// along x. Returns (centre.xy, present 0/1, steam draw in [0,1)). A site sits
// in a travel lane; sites that would fall in an intersection's box, its
// crosswalks or its stop lines are not present.
fn city_manhole(k: i32, j: i32, along_y: bool) -> vec4<f32> {
    let cellm = u.tile_cell;
    // The along index folds at the tile too: CITY_MH_PITCH divides it.
    let jn = i32(round(u.tile_extent / CITY_MH_PITCH));
    let r = city_rand4(vec2<u32>(
        city_line_seed(k) * 0x9e3779b9u + select(0x1b873593u, 0xcc9e2d51u,
                                                 along_y),
        bitcast<u32>(((j % jn) + jn) % jn) * 0x85ebca6bu + 0x27d4eb2fu));
    let a = (f32(j) + 0.5) * CITY_MH_PITCH + (r.y - 0.5) * 6.0;
    // The cross street nearest this site, and how far into its box it is.
    let kc = i32(round(a / cellm));
    let hc = CITY_STREET_HALF
           + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(kc));
    let clear = abs(a - f32(kc) * cellm) > hc + 2.5;
    // Lateral: a travel lane. On a minor street that is the half without
    // the parking; on an avenue any of the three lanes each way.
    let ave = city_is_avenue(k);
    var lat: f32;
    if (ave) {
        let lane = floor(r.z * 6.0);            // 0..5
        let side = select(-1.0, 1.0, lane >= 3.0);
        lat = side * (1.6 + 3.1 * (lane - select(0.0, 3.0, lane >= 3.0)));
    } else {
        lat = -city_park_side(k) * (1.1 + 0.8 * r.z);
    }
    let line = f32(k) * cellm + lat;
    let c = select(vec2<f32>(a, line), vec2<f32>(line, a), along_y);
    let present = r.x < CITY_MH_PRESENT && clear;
    return vec4<f32>(c, select(0.0, 1.0, present), r.w);
}

// Whether a site steams, for the ground and the manhole-steam component
// alike, so a dry lid and its plume can never disagree: the site's steam
// draw over its district's share, v in [0, 1) uniform over the steaming
// sites (the component sizes its plume by v) and >= 1 for every other, an
// absent site included. The hash rejects all but the candidates before the
// texture read.
fn city_manhole_steam(m: vec4<f32>) -> f32 {
    if (m.z < 0.5 || m.w >= CITY_MH_STEAM_FRAC_HI) {
        return 2.0;
    }
    let district = city_glow_sample(m.xy, 2.0);
    let frac = mix(CITY_MH_STEAM_FRAC_LO, CITY_MH_STEAM_FRAC_HI,
                   smoothstep(0.02, 0.45, district));
    return m.w / frac;
}

// A disc of radius r at offset lq (|lq| = d) from p, filtered over fp:
// resolved, the disc with its edge antialiased over a pixel; unresolved,
// the equal-area square box-filtered in both axes, whose integral IS the
// disc's area at every footprint — a 1-D ramp in d would claim half the
// pixel at fp = 10 m for a cover that is 0.4% of it.
fn city_disc_cover(lq: vec2<f32>, d: f32, r: f32, fp: f32) -> f32 {
    let h = 0.5 * max(fp, 1e-4);
    let sq = r * 1.7724539;          // sqrt(pi) r
    return mix(clamp((r - d + h) / (2.0 * h), 0.0, 1.0),
               city_band(lq.x, -0.5 * sq, sq, fp)
               * city_band(lq.y, -0.5 * sq, sq, fp),
               smoothstep(0.05, 0.25, fp));
}

// A disc that fades from 1 inside r_in to 0 at r_out (a smoothstep in d),
// filtered over fp: resolved, the profile itself; unresolved, the
// equal-area disc, radius sqrt(r_in^2 + r_in w + 0.3 w^2) for the fade's
// width w (the smoothstep's exact area), so its mean holds at every range.
fn city_soft_disc_cover(lq: vec2<f32>, d: f32, r_in: f32, r_out: f32,
                        fp: f32) -> f32 {
    let w = r_out - r_in;
    // @domain: a sum of squares and r_in w, all >= 0
    let r_eq = sqrt(r_in * r_in + r_in * w + 0.3 * w * w);
    return mix(1.0 - smoothstep(r_in, r_out, d),
               city_disc_cover(lq, d, r_eq, fp),
               smoothstep(0.05, 0.25, fp));
}

// Cover coverage at p for the two streets p may stand on, filtered: a
// resolved disc with its rim, or its area share once sub-pixel. The
// result: (cover coverage, rim/pattern darkening already folded in as a
// relative albedo to multiply, the share kept dry by a steaming cover's
// heat: the lid and the asphalt round it, fading out between DRY_IN and
// DRY_OUT, filtered so the far field's mean is the near field's).
fn city_manhole_cover(p: vec2<f32>, rd: CityRoad, fp: f32) -> vec3<f32> {
    var cov = 0.0;
    var tone = 1.0;
    var dried = 0.0;
    for (var s: i32 = 0; s < 2; s = s + 1) {
        let along_y = s == 0;
        let k = select(rd.ky, rd.kx, along_y);
        let on = select(abs(rd.sy) < rd.hy, abs(rd.sx) < rd.hx, along_y);
        if (!on) {
            continue;
        }
        let a = select(p.x, p.y, along_y);
        let j = i32(floor(a / CITY_MH_PITCH));
        for (var dj: i32 = 0; dj < 2; dj = dj + 1) {
            let jj = j + dj - select(1, 0, fract(a / CITY_MH_PITCH) > 0.5);
            let m = city_manhole(k, jj, along_y);
            if (m.z < 0.5) {
                continue;
            }
            let d = length(p - m.xy);
            let lq = p - m.xy;
            // The dry disc only for a pixel it reaches, and the steam test
            // (a texture read) only for a cover whose dry disc that is.
            let warm = city_soft_disc_cover(lq, d, CITY_MH_R + CITY_MH_DRY_IN,
                                            CITY_MH_R + CITY_MH_DRY_OUT, fp);
            if (warm > 0.0 && city_manhole_steam(m) < 1.0) {
                dried = max(dried, warm);
            }
            let disc = city_disc_cover(lq, d, CITY_MH_R, fp);
            if (disc <= 0.0) {
                continue;
            }
            // The cast pattern: a raised grid, and the rim ring. Both are
            // stripes, so both dissolve to their duty cycle.
            // Union of the two bar sets as 1 - (1-a)(1-b): exact in the
            // mean (independent axes), where max() dissolved to 1/3 of a
            // resolved 5/9 (the review).
            let grid = 1.0 - (1.0 - city_stripes(lq.x, 0.0, 0.025, 0.075, fp))
                           * (1.0 - city_stripes(lq.y, 0.0, 0.025, 0.075,
                                                 fp));
            let rim = city_band(d, CITY_MH_R - 0.035, 0.035, fp);
            cov = max(cov, disc);
            tone = mix(1.0, 0.72 + 0.55 * grid, 1.0 - rim) * (1.0 + 0.6 * rim);
        }
    }
    return vec3<f32>(cov, tone, dried);
}

// --- the ground material --------------------------------------------------------

struct CityGround {
    alb: vec3<f32>,   // diffuse response, bare dry asphalt = 1
    wet: f32,         // share of the surface under a thin film, 0..1
    puddle: f32,      // standing water, a mirror, 0..1
    gloss: f32,       // glossy film weight: CITY_FILM_GLOSS x wet x mottle
    walk: f32,        // how much of the footprint is raised (sidewalk/plaza)
}

fn city_ground(p: vec2<f32>, ci: vec2<i32>, cc: CityCell, fp: f32)
        -> CityGround {
    let kr = city_kerb_rect(cc);
    let d = city_kerb_sd(p, cc, kr);
    let rd = city_road(p);
    let walk_ok = city_has_walk(cc);
    // Local coordinates for every filtered pattern: metres from the kerb
    // rectangle's corner, so nothing is filtered at 1e5 m magnitudes.
    let lp = p - kr.xy;

    // Zone shares across the section, each a box-filtered step in d.
    let w_road = city_step(d, CITY_GUTTER_W, fp);
    let w_gutter = city_step(d, 0.0, fp) - w_road;
    let w_kerb = select(0.0, city_step(d, -CITY_KERB_TOP_W, fp)
                             - city_step(d, 0.0, fp), walk_ok);
    let w_in = 1.0 - w_road - w_gutter - w_kerb;

    // Sidewalk and plaza: flags on a lattice anchored at the kerb corner,
    // one joint per CITY_FLAG, a tone per flag. Past a flag per pixel the
    // tone is its mean, 1.
    let joint = 1.0 - (1.0 - city_stripes(lp.x, -0.5 * CITY_JOINT_W,
                                          CITY_JOINT_W, CITY_FLAG, fp))
                    * (1.0 - city_stripes(lp.y, -0.5 * CITY_JOINT_W,
                                          CITY_JOINT_W, CITY_FLAG, fp));
    let fid = floor(lp / CITY_FLAG);
    let fh = pcg2d(vec2<u32>(bitcast<u32>(i32(fid.x)) ^ cc.seed.x,
                             bitcast<u32>(i32(fid.y)) ^ cc.seed.y));
    let ftone = mix(1.0 + 0.30 * (city_u01(fh.x) - 0.5), 1.0,
                    smoothstep(0.4 * CITY_FLAG, 1.5 * CITY_FLAG, fp));
    // Stains: the dark blots every city sidewalk carries, mean 1.
    let stain = city_detail(p, 0.9, 3.7, 0.40, fp, vec2<f32>(0.5, 2.0));
    let rel_walk = CITY_REL_WALK * ftone * stain * (1.0 - 0.6 * joint);
    let rel_in = select(CITY_REL_LOT * stain, rel_walk, walk_ok);

    // Asphalt: aggregate at a quarter metre, patching at six. Both mean 1.
    let agg = city_detail(p, 4.0, 0.0, 0.30, fp, vec2<f32>(0.08, 0.30));
    let patching = city_detail(p, 0.17, 31.7, 0.45, fp,
                               vec2<f32>(2.0, 8.0));
    var road = vec3<f32>(agg * patching);

    // Markings, painted only on the carriageway. `ax`/`ay` are the offsets
    // along each street, measured from the cross street's centreline, whose
    // period (the block) every dash lattice divides.
    let on_x = abs(rd.sx) < rd.hx;     // on the street running along y
    let on_y = abs(rd.sy) < rd.hy;
    let ex = abs(rd.sx);
    let ey = abs(rd.sy);
    let swx = select(CITY_SIDEWALK_W, CITY_SIDEWALK_W_AVE, rd.ave_x);
    let swy = select(CITY_SIDEWALK_W, CITY_SIDEWALK_W_AVE, rd.ave_y);
    var white = 0.0;
    var yellow = 0.0;
    if (on_x && ey > rd.hy) {
        if (rd.ave_x) {
            yellow = yellow + city_band(ex, 0.10, 0.12, fp);
            white = white + (city_band(ex, 3.05, 0.12, fp)
                             + city_band(ex, 6.15, 0.12, fp))
                          * city_stripes(ey, 0.0, 3.0, 9.0, fp);
            white = white + city_band(ex, rd.hx - swx - CITY_PARK_W - 0.12,
                                      0.12, fp);
        }
        // Stop line, across the approach.
        white = white + city_band(ey, rd.hy + 1.2, 0.40, fp);
    }
    if (on_y && ex > rd.hx) {
        if (rd.ave_y) {
            yellow = yellow + city_band(ey, 0.10, 0.12, fp);
            white = white + (city_band(ey, 3.05, 0.12, fp)
                             + city_band(ey, 6.15, 0.12, fp))
                          * city_stripes(ex, 0.0, 3.0, 9.0, fp);
            white = white + city_band(ey, rd.hy - swy - CITY_PARK_W - 0.12,
                                      0.12, fp);
        }
        white = white + city_band(ex, rd.hx + 1.2, 0.40, fp);
    }
    // Crosswalks: in line with the cross street's sidewalks, bars running
    // with the traffic, 0.5 m on 1.0 m across the carriageway.
    if (on_x) {
        white = white + city_band(ey, rd.hy - swy + 0.3, swy - 0.1, fp)
                      * city_stripes(rd.sx, 0.25, 0.5, 1.0, fp);
    }
    if (on_y) {
        white = white + city_band(ex, rd.hx - swx + 0.3, swx - 0.1, fp)
                      * city_stripes(rd.sy, 0.25, 0.5, 1.0, fp);
    }
    // Paint wears through where the tyres run; mean 0.8.
    let wear = 0.8 * city_detail(p, 1.3, 9.1, 0.5, fp, vec2<f32>(0.3, 1.2));
    white = min(white, 1.0) * wear;
    yellow = min(yellow, 1.0) * wear;
    road = mix(road, vec3<f32>(CITY_REL_PAINT), white);
    road = mix(road, CITY_REL_PAINT * CITY_PAINT_YELLOW, yellow);
    let mh = city_manhole_cover(p, rd, fp);
    road = mix(road, vec3<f32>(CITY_REL_IRON * mh.y), mh.x);

    // Storm grates in the gutter: bars across the pan, box-filtered.
    var gut = CITY_REL_GUTTER;
    if (w_gutter > 0.0) {
        let along = select(ey, ex, !on_x || (on_y && ex > rd.hx));
        let hcross = select(rd.hy, rd.hx, !on_x || (on_y && ex > rd.hx));
        let gp = city_band(along, hcross + CITY_GRATE_AT - CITY_GRATE_HL,
                           2.0 * CITY_GRATE_HL, fp);
        let bars = city_stripes(along, 0.0, 0.03, 0.06, fp);
        gut = mix(gut, mix(CITY_REL_SLOT, CITY_REL_IRON, bars), gp);
    }

    let dry = w_road * road + vec3<f32>(w_gutter * gut + w_kerb * CITY_REL_KERB
                                        + w_in * rel_in);

    // Wetness. The film by zone; the puddles by noise, biased into the
    // gutter and off the crown of the road; neither on the share a steaming
    // cover's heat keeps dry (mh.z).
    let film = (w_road * CITY_WET_ROAD + w_gutter
                + (w_kerb + w_in) * CITY_WET_WALK) * (1.0 - mh.z);
    let pud_k = smoothstep(CITY_PUDDLE_LOD.x, CITY_PUDDLE_LOD.y, fp);
    let region = city_vnoise(p, 0.025, 5.3) - 0.5;
    let crown = 1.0 - smoothstep(0.0, 2.5, min(select(1e3, ex, on_x),
                                               select(1e3, ey, on_y)));
    // Where water actually stands: along the foot of the kerb (the gutter
    // is the wettest line on any wet street, and it ends AT the kerb — the
    // raised sidewalk sheds), and in the tyre ruts worn into every lane,
    // 0.85 m either side of its centre. Noise only decides where along
    // those the water breaks.
    // Both are narrow bands and both feed the puddle threshold, so both are
    // footprint-filtered or they alias into sparkle at a few m/px: the
    // gutter widens with fp at constant area, and the
    // ruts hand over to their duty cycle inside the travel lanes (two
    // ~0.4 m ruts per 3.1 m lane).
    let gutter = select(0.0, (1.0 - smoothstep(0.10, 0.70 + fp, d))
                             * (0.4 / (0.4 + 0.5 * fp)), d > 0.0);
    var lane_c = 1e3;
    if (on_x && ey > rd.hy) {
        lane_c = select(rd.sx + city_park_side(rd.kx) * 1.2,
                        ex - 1.6 - round((ex - 1.6) / 3.1) * 3.1, rd.ave_x);
    }
    if (on_y && ex > rd.hx) {
        lane_c = select(rd.sy + city_park_side(rd.ky) * 1.2,
                        ey - 1.6 - round((ey - 1.6) / 3.1) * 3.1, rd.ave_y);
    }
    let rut = mix(1.0 - smoothstep(0.10, 0.30, abs(abs(lane_c) - 0.85)),
                  select(0.0, 0.26, lane_c < 100.0),
                  smoothstep(0.3, 1.2, fp));
    let bias = 0.38 * gutter + 0.14 * rut * w_road
             - 0.12 * crown - 0.35 * (w_kerb + w_in) + 0.12 * region;
    let pud_mean = 1.0 / (1.0 + exp(-(bias + 0.5 - CITY_PUDDLE_FIT.x)
                                    / CITY_PUDDLE_FIT.y));
    // The noise only while the puddles are resolved: past the band the
    // coverage IS its expectation.
    var pud_sharp = pud_mean;
    if (pud_k < 1.0) {
        let n = 0.62 * city_vnoise(p, 0.38, 0.0)
              + 0.38 * city_vnoise(p, 1.1, 37.1);
        pud_sharp = smoothstep(CITY_PUDDLE_THR - CITY_PUDDLE_EDGE,
                               CITY_PUDDLE_THR + CITY_PUDDLE_EDGE, n + bias);
    }
    let puddle = mix(pud_sharp, pud_mean, pud_k)
               * (1.0 - max(mh.x, mh.z));

    // The film is not a sheet: it pools in the texture and drains off the
    // high spots, so its gloss mottles at half a metre and the joints hold
    // water. Mean 1, so the far field keeps the film's average sheen.
    let mottle = city_detail(p, 1.7, 11.3, 1.4, fp, vec2<f32>(0.3, 1.2));
    var g: CityGround;
    g.alb = dry * (1.0 - CITY_WET_DARKEN * film * (1.0 - puddle)
                   - CITY_PUDDLE_DARKEN * puddle);
    g.wet = film;
    g.puddle = puddle;
    // A wet lid sheens as polished iron; a hot one is dry, and matte.
    g.gloss = mix(CITY_FILM_GLOSS * film * max(mottle, 0.0), CITY_MH_GLOSS,
                  mh.x) * (1.0 - mh.z);
    g.walk = select(0.0, w_in + w_kerb, walk_ok);
    // Outside the plot on an edge that carries no street there is no
    // street at all: lot ground, and no kerb to collect water against.
    let inside_plot = p.x > cc.plot_min.x && p.x < cc.plot_max.x
                   && p.y > cc.plot_min.y && p.y < cc.plot_max.y;
    if (!inside_plot && !city_ground_paved(cc, ci, p)) {
        g.alb = vec3<f32>(CITY_REL_LOT * stain
                          * (1.0 - CITY_WET_DARKEN * CITY_WET_WALK));
        g.wet = CITY_WET_WALK;
        g.puddle = 0.0;
        g.gloss = CITY_FILM_GLOSS * CITY_WET_WALK * max(mottle, 0.0);
        g.walk = 0.0;
    }
    return g;
}

// Storefront light on the pavement in front of it: the glow of the cell's
// store band falling off over a couple of metres from the building face.
// It is what the store throws DOWN; the store itself reaches the puddles
// through the reflected ray.
fn city_storefront_spill(p: vec2<f32>, cc: CityCell) -> vec3<f32> {
    let glow = city_storefront_glow(cc);
    let q = max(max(cc.b1min.xy - p, p - cc.b1max.xy), vec2<f32>(0.0));
    let dist = length(q);
    return glow * (CITY_STORE_SPILL * exp(-dist / CITY_STORE_SPILL_L)
                   * step(1e-3, dist));
}

// The lamp scale of the district: how hard the lattice burns here.
fn city_lamp_scale(p: vec2<f32>) -> f32 {
    let district = city_glow_sample(p, 2.0);
    return CITY_LAMP_RADIANCE * (0.20 + 1.3 * smoothstep(0.02, 0.45, district));
}

// Radiance of the ground (kind 0) before fog: lamp pools and storefront
// spill on the material above, plus the night fill. Near, the pools are
// resolved and each lands on the albedo under it; far, the whole product is
// its measured area mean, split street / plot interior as the old ground
// was, so a district's floor keeps its brightness as it dissolves.
fn city_ground_radiance(p: vec3<f32>, ci: vec2<i32>, cc: CityCell,
                        dir: vec3<f32>, fp: f32) -> vec3<f32> {
    let paved = city_ground_paved(cc, ci, p.xy);
    if (city_ground_is_lot(paved, cc)) {
        return city_lot_radiance(p, cc, fp);
    }
    return city_ground_lit(p, cc, dir, fp, city_ground(p.xy, ci, cc, fp),
                           paved);
}

// Open ground on the fringe (city-layout's yard / scrub / industrial lot):
// no street, no sidewalk, and no wet reflection traced off it.
fn city_ground_is_lot(paved: bool, cc: CityCell) -> bool {
    return !paved && cc.land_use != CITY_USE_URBAN;
}

// It sees the whole sky where a canyon floor sees a slot of it, and on a
// city night that sky is the cloud deck lit from below by the whole city —
// the wide glow mip is that light's source — so yards and scrub read,
// faintly, as moonlit and city-lit dirt rather than a void. Their mottle has
// mean one and dissolves as the footprint outgrows a patch.
fn city_lot_radiance(p: vec3<f32>, cc: CityCell, fp: f32) -> vec3<f32> {
    let moon = u.sun_dir;
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
    let alb = select(CITY_YARD_ALBEDO, CITY_SCRUB_ALBEDO,
                     cc.land_use == CITY_USE_OPEN)
            * city_ground_mottle(p.xy, fp);
    let sky = fill + CITY_OPEN_SKY
            * (CITY_OPEN_SKY_BASE + city_glow_sample(p.xy, 8.0));
    return city_flood_ground(cc, p.xy, fp)
         + alb * (sky + CITY_MOONLIGHT * max(moon.z, 0.0));
}

// The street and sidewalk under their lights, from a material already
// evaluated (the view ray's pass evaluates it once and hands the same one to
// the wet lobe). The far field's lamp term is the measured mean for the
// street (tools/street_means.py) only where the ground IS street: inside the
// plot, and on a streetless edge's strip (whose own lamps city-layout's gate
// has switched off), it is the mean of the neighbouring pools' tails.
fn city_ground_lit(p: vec3<f32>, cc: CityCell, dir: vec3<f32>, fp: f32,
                   g: CityGround, paved: bool) -> vec3<f32> {
    let moon = u.sun_dir;
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
    let s = city_lamp_scale(p.xy);
    let raw = s * city_street_pools(p.xy);
    let near = CITY_LAMP_COLOR * (raw / (1.0 + CITY_POOL_KNEE * raw)) * g.alb;
    let fit = select(CITY_GROUND_FAR_PLOT, CITY_GROUND_FAR_STREET, paved);
    let far = CITY_LAMP_COLOR * (fit.x * s / (1.0 + fit.y * s));
    let lamp = mix(near, far, smoothstep(CITY_STREET_LOD_START,
                                         CITY_STREET_LOD_FULL, fp));
    return lamp + city_flood_ground(cc, p.xy, fp)
         + city_storefront_spill(p.xy, cc) * g.alb
         + city_lamp_glint(p, dir, g, fp)
         + CITY_ASPHALT_ALBEDO * g.alb
           * (fill + CITY_MOONLIGHT * max(moon.z, 0.0));
}

// Sodium light arriving at a point that is not the road — a bin, a facade
// sill, a plume of steam. The lamp heads on the core's lattice
// (city_lamp_inset, CITY_LAMP_SPACING) are point sources hung CITY_LAMP_ARM
// out over the road at CITY_LAMP_HEAD_Z, with a downward cosine lobe, scaled
// so that the ground straight under a head receives the peak of one pool.
// (The pools themselves centre on the pole's foot, CITY_LAMP_ARM inboard of
// the head — well inside their CITY_LAMP_SIGMA, so the two agree to within
// what the eye can place.)
// The result is radiance-scale, in the ground's units: multiply by a
// surface's albedo relative to asphalt, as city_ground_radiance does. The
// eight heads nearest p (the lamp line either side of the nearest boundary,
// in each direction, at the two nearest lattice stations) carry it.
//
// `view` selects the receiver. For a surface, pass its normal as `n` and
// g = 2 (no phase): each head is weighted by max(n.l, 0). For a medium,
// pass the view ray's direction as `n` and g in (-1, 1): each head is
// weighted by the Henyey-Greenstein phase between the light and the eye,
// times 4 pi so that isotropic (g = 0) matches a surface facing the lamp —
// strong forward g is what makes steam blaze where a lamp stands behind it.
fn city_street_light_g(p: vec3<f32>, n: vec3<f32>, g: f32) -> vec3<f32> {
    let cellm = u.tile_cell;
    let s = city_lamp_scale(p.xy);
    let h2 = CITY_LAMP_HEAD_Z * CITY_LAMP_HEAD_Z;
    let medium = g < 1.5;
    let lit = city_lamp_lines(p.xy);
    var e = 0.0;
    for (var dirn: i32 = 0; dirn < 2; dirn = dirn + 1) {
        let along_y = dirn == 0;            // the street runs along y
        if (!select(lit.y, lit.x, along_y)) {
            continue;
        }
        let c = select(p.y, p.x, along_y);  // across coordinate
        let a = select(p.x, p.y, along_y);  // along coordinate
        let k = i32(round(c / cellm));
        let amp = select(1.0, 1.6, city_is_avenue(k));
        let inset = city_lamp_inset(k);
        let lat = city_lamp_lattice(k);
        let j0 = floor((a - lat.y) / lat.x);
        for (var sd: i32 = 0; sd < 2; sd = sd + 1) {
            let sg = select(-1.0, 1.0, sd == 1);
            // The head hangs toward the centreline from its lamp line.
            let hc = f32(k) * cellm + sg * (inset - CITY_LAMP_ARM);
            for (var jj: i32 = 0; jj < 2; jj = jj + 1) {
                let ha = lat.y + (j0 + f32(jj)) * lat.x;
                let head = select(vec3<f32>(ha, hc, CITY_LAMP_HEAD_Z),
                                  vec3<f32>(hc, ha, CITY_LAMP_HEAD_Z),
                                  along_y);
                let v = head - p;
                let r2 = max(dot(v, v), 0.25);
                let l = v * inverseSqrt(r2);
                var w = max(dot(n, l), 0.0);
                if (medium) {
                    // The light propagates along -l and leaves toward the
                    // eye along -n, so the scattering cosine is
                    // dot(-l, -n) = dot(l, n): 1 for a lamp straight
                    // behind the medium.
                    w = 12.566371 * hg_phase(dot(l, n), g);
                }
                e = e + amp * h2 / r2 * max(l.z, 0.0) * w;
            }
        }
    }
    let raw = s * e;
    return CITY_LAMP_COLOR * (raw / (1.0 + CITY_POOL_KNEE * raw));
}

// The surface form: sodium light on a face with normal n.
fn city_street_light(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return city_street_light_g(p, n, 2.0);
}

// The lamps in the wet ground: each head a point source, reflected by the
// puddle lobe and the film lobe analytically (GGX), because a lamp is far
// too small for one reflected ray a frame to find — and it is the brightest
// thing a wet street reflects. At grazing incidence the GGX lobe stretches
// toward the eye, which is the long streak under every lamp in the
// photographs; the traced ray carries everything else (neon, windows, sky)
// and never sees a lamp head (it skips the prop hooks, see fs_main), so
// nothing is counted twice. Irradiance in the pools' units: the pool is asphalt's radiance,
// rho E / pi, so E / pi is the pool over CITY_LAMP_SPEC_RHO.
const CITY_LAMP_SPEC_RHO: f32 = 0.20;
const CITY_LAMP_SPEC_MAX: f32 = 40.0;
fn city_ggx_d(nh: f32, a2: f32) -> f32 {
    let k = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * k * k);
}
fn city_lamp_glint(p: vec3<f32>, dir: vec3<f32>, g: CityGround, fp: f32)
        -> vec3<f32> {
    let film = g.gloss * (1.0 - g.puddle);
    if (film + g.puddle <= 0.0) {
        return vec3<f32>(0.0);
    }
    let cellm = u.tile_cell;
    let s = city_lamp_scale(p.xy);
    let h2 = CITY_LAMP_HEAD_Z * CITY_LAMP_HEAD_Z;
    let v = -dir;
    let nv = max(v.z, 1e-3);
    let ap = CITY_REFL_ALPHA_PUDDLE * CITY_REFL_ALPHA_PUDDLE;
    let ad = CITY_REFL_ALPHA_DAMP * CITY_REFL_ALPHA_DAMP;
    // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
    let g1v_p = 2.0 * nv / (nv + sqrt(ap + (1.0 - ap) * nv * nv));
    // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
    let g1v_d = 2.0 * nv / (nv + sqrt(ad + (1.0 - ad) * nv * nv));
    let lit = city_lamp_lines(p.xy);
    // The heads a mirror shows are the ones near where the MIRROR ray (the
    // view ray reflected about the ground) reaches head height — at grazing
    // incidence that is far down the street, not beside the point. Summing
    // the two stations beside the point loses every lamp past ~35 m from an
    // eye-level camera. The rough lobe
    // centres on the same direction, so the same stations serve both: the
    // three nearest the mirror point, because with only the two bracketing
    // it the set switches as the point crosses a station and drops a lamp
    // the film lobe still sees — a hard band across every wet road.
    let t_head = min((CITY_LAMP_HEAD_Z - p.z) / max(-dir.z, 0.02), 300.0);
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
        let a_m = a + select(dir.x, dir.y, along_y) * t_head;
        let j0 = round((a_m - lat.y) / lat.x) - 1.0;
        for (var sd: i32 = 0; sd < 2; sd = sd + 1) {
            let hc = f32(k) * cellm
                   + select(-1.0, 1.0, sd == 1) * (inset - CITY_LAMP_ARM);
            for (var jj: i32 = 0; jj < 3; jj = jj + 1) {
                let ha = lat.y + (j0 + f32(jj)) * lat.x;
                let head = select(vec3<f32>(ha, hc, CITY_LAMP_HEAD_Z),
                                  vec3<f32>(hc, ha, CITY_LAMP_HEAD_Z),
                                  along_y);
                let lv = head - p;
                let r2 = max(dot(lv, lv), 0.25);
                let l = lv * inverseSqrt(r2);
                let nl = max(l.z, 1e-3);
                // The pool's own irradiance law (downward lobe x cosine).
                let e_pool = amp * h2 / r2 * nl * nl;
                // @domain: v (to the eye) and l (to a lamp head) both point up from the ground
                let hv = normalize(v + l);
                let nh = max(hv.z, 0.0);
                let fres = CITY_REFL_F0 + (1.0 - CITY_REFL_F0)
                         * pow(1.0 - clamp(dot(v, hv), 0.0, 1.0), 5.0);
                // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
                let g1l_p = 2.0 * nl / (nl + sqrt(ap + (1.0 - ap) * nl * nl));
                // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
                let g1l_d = 2.0 * nl / (nl + sqrt(ad + (1.0 - ad) * nl * nl));
                // Each lobe widened by the pixel's footprint as the lamp
                // sees it (fp / r radians, half of it in half-vector
                // angle): D is normalised, so a distant puddle's mirror
                // image of a lamp spreads into its mean rather than
                // sparkling on the one pixel that happens to catch it.
                let wide = 0.25 * fp * fp / r2;
                let lobe = g.puddle * city_ggx_d(nh, ap + wide) * g1v_p * g1l_p
                         + film * city_ggx_d(nh, ad + wide) * g1v_d * g1l_d;
                e = e + e_pool * fres * lobe / (4.0 * nv * nl);
            }
        }
    }
    let raw = s * e * (3.14159265 / CITY_LAMP_SPEC_RHO);
    return CITY_LAMP_COLOR * min(raw, CITY_LAMP_SPEC_MAX);
}

// --- the reflected ray ----------------------------------------------------------

// What a wet ground point reflects. One microfacet sample per frame: a
// puddle's near-mirror lobe with probability equal to the puddle share, a
// damp film's rough lobe otherwise, weighted by the film's glossy coverage —
// so the accumulation converges to the mixture, and the rough lobe's
// samples, stretched at grazing incidence, smear every sign and window into
// the streaks of a wet street (the lamps have their own analytic term,
// city_lamp_glint). This returns the sampled direction and its weight; the
// call site in fs_main follows it with city_trace — without the component
// prop hooks (see the note there) — and city_shade, fogged along its own
// path. Past
// CITY_REFL_LOD the sample is the environment estimate instead, and costs
// no trace; a ray that finds no city within CITY_REFL_RANGE sees the sky.
//
// rnd: three independent uniforms for this pixel and frame (city_frame_rand).
struct CityWetLobe {
    ok: bool,          // anything to add at all
    r: vec3<f32>,      // the reflected direction
    w: f32,            // estimator weight: lobe share x F G (v.m)/((n.v)(n.m))
    blend: f32,        // 0 traced .. 1 environment estimate
    alpha: f32,        // the lobe's roughness, which widens the footprint
    env: vec3<f32>,    // the environment estimate, where blend > 0
}

fn city_wet_lobe(h: CityHit, g: CityGround, dir: vec3<f32>,
                 pixel_angle: f32, rnd: vec3<f32>) -> CityWetLobe {
    var out: CityWetLobe;
    out.ok = false;
    out.r = vec3<f32>(0.0, 0.0, 1.0);
    out.w = 0.0;
    out.blend = 1.0;
    out.alpha = 0.0;
    out.env = vec3<f32>(0.0);
    let fp = pixel_angle * h.t;
    let puddle_lobe = rnd.z < g.puddle;
    let weight = select(g.gloss * (1.0 - g.puddle), g.puddle, puddle_lobe)
               / max(select(1.0 - g.puddle, g.puddle, puddle_lobe), 1e-4);
    if (weight <= 0.0) {
        return out;
    }
    let alpha = select(CITY_REFL_ALPHA_DAMP, CITY_REFL_ALPHA_PUDDLE,
                       puddle_lobe);
    // GGX, sampled by its distribution of normals.
    let a2 = alpha * alpha;
    // @domain: rnd.x in [0, 1), a2 in (0, 1]: numerator > 0, denominator >= a2 > 0
    let ct = sqrt((1.0 - rnd.x) / (1.0 + (a2 - 1.0) * rnd.x));
    let st = sqrt(max(1.0 - ct * ct, 0.0));
    let phi = 6.2831853 * rnd.y;
    let m = vec3<f32>(st * cos(phi), st * sin(phi), ct);
    let vm = min(-dot(dir, m), 1.0);
    if (vm <= 0.0) {
        return out;
    }
    let r = dir + 2.0 * vm * m;
    if (r.z <= 1e-3) {
        return out;   // into the ground: masked
    }
    // Estimator: F G2 (v.m) / ((n.v)(n.m)), Smith's G1 for each direction.
    let nv = max(-dir.z, 1e-4);
    let nl = r.z;
    // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
    let g1v = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
    // @domain: a squared GGX roughness in (0, 1]: both terms >= 0
    let g1l = 2.0 * nl / (nl + sqrt(a2 + (1.0 - a2) * nl * nl));
    // @domain: vm <= 1 (the min above) and vm > 0 (returned above otherwise)
    let fres = CITY_REFL_F0 + (1.0 - CITY_REFL_F0) * pow(1.0 - vm, 5.0);
    out.ok = true;
    out.r = r;
    // Capped at 1: at n.v near 0.01 single samples reach 5.5 (puddle) and
    // 1.75 (film), and one such sample in a 1-spp flight frame is a
    // firefly. The cap biases grazing reflections slightly dark, knowingly.
    out.w = weight * min(fres * g1v * g1l * vm / (nv * ct), 1.0);
    out.alpha = alpha;
    out.blend = smoothstep(CITY_REFL_LOD.x, CITY_REFL_LOD.y, fp);
    if (out.blend > 0.0) {
        out.env = city_reflect_env(h.pos.xy, r);
    }
    return out;
}

// A reflected ray that leaves the city sees the night sky — and, over a
// lit city, the cloud deck the city lights from below, which is half of
// what a wet street shows. A short probe of the optical depth along the ray
// through the cloud slab, and the base's own radiance in front of the sky
// it hides: the uplight the march delivers to a base (CITY_UPLIGHT_*,
// integrated over the skin, 1 - e^-tau), with the glow taken under the
// point where the probe first becomes opaque. `world_o` is the reflection
// point in WORLD coordinates: the clouds live there, not in the city frame.
const CITY_REFL_CLOUD_STEPS: i32 = 10;
const CITY_REFL_CLOUD_SPAN: f32 = 4000.0;
fn city_reflect_sky(world_o: vec3<f32>, r: vec3<f32>, jit: f32) -> vec3<f32> {
    let sky = night_sky_radiance(r, u.sun_dir);
    let ta = max((u.bmin.z - world_o.z) / r.z, 0.0);
    let tb = min((u.bmax.z - world_o.z) / r.z, ta + CITY_REFL_CLOUD_SPAN);
    if (tb <= ta) {
        return sky;
    }
    let dt = (tb - ta) / f32(CITY_REFL_CLOUD_STEPS);
    var tau = 0.0;
    var base = world_o + tb * r;
    for (var k: i32 = 0; k < CITY_REFL_CLOUD_STEPS; k = k + 1) {
        let q = world_o + (ta + (f32(k) + jit) * dt) * r;
        let was_thin = tau < 1.0;
        tau = tau + sample_sigma(q) * dt;
        if (was_thin && tau >= 1.0) {
            base = q;
        }
    }
    // @domain: a positive max over u.tile_cell, the surface tile's cell size (> 0)
    let lod = clamp(log2(max(base.z, u.tile_cell) / u.tile_cell),
                    0.0, u.tile_max_lod);
    let g_raw = max(city_glow_sample(city_fold_xy(base.xy), lod)
                    - CITY_UPLIGHT_GLOW_BIAS, 0.0);
    let g_up = g_raw / (CITY_UPLIGHT_GLOW_HALF + g_raw);
    let lit = CITY_UPLIGHT_COLOR * (CITY_UPLIGHT_STRENGTH * g_up)
            + CITY_REFL_CLOUD_FLOOR;
    return mix(sky, lit, 1.0 - exp(-tau));
}
// What an unlit cloud base still returns at night (moon and skyglow).
const CITY_REFL_CLOUD_FLOOR: vec3<f32> = vec3<f32>(0.010, 0.011, 0.015);

// The mean of what a reflected ray sees, for pixels too far to trace one:
// the night sky above the local skyline, and below it the district's own
// facades at their far-field glow. How much of the hemisphere the skyline
// covers comes from the district density, measured against traced
// reflections at the hand-off.
const CITY_REFL_ENV_FACADE: f32 = 0.35;
const CITY_REFL_ENV_ELEV: f32 = 0.35;
fn city_reflect_env(p: vec2<f32>, r: vec3<f32>) -> vec3<f32> {
    let district = city_glow_sample(p, 2.0);
    let skyline = clamp(1.6 * district, 0.0, 1.0)
                * exp(-max(r.z, 0.0) / CITY_REFL_ENV_ELEV);
    let facade = CITY_PALETTE_MEAN * (CITY_REFL_ENV_FACADE
                                      * (0.3 + 1.4 * district));
    // Fogged over half the traced range, as the traced reflections it
    // stands in for are over theirs (the review's M2: unfogged, the far wet
    // street stepped at the hand-off under a lit sky).
    let o = vec3<f32>(p, 0.0);
    let l = 0.5 * CITY_REFL_RANGE;
    return city_fog(mix(night_sky_radiance(r, u.sun_dir), facade, skyline),
                    o, r, l, o + l * r);
}

// Is the ground at p (inside cell ci) street asphalt? The strips along the
// cell's edges, half a street wide, are asphalt where that edge carries a
// street; the corner squares where any street meets there. Everything else
// is lot ground — including a strip along an edge with no street, and
// every boundary inside a multi-block lot.
fn city_ground_paved(cc: CityCell, ci: vec2<i32>, p: vec2<f32>) -> bool {
    let cell = u.tile_cell;
    let q = p - vec2<f32>(ci) * cell;
    let hw0 = vec2<f32>(
        CITY_STREET_HALF + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(ci.x)),
        CITY_STREET_HALF + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(ci.y)));
    let hw1 = vec2<f32>(
        CITY_STREET_HALF
            + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(ci.x + 1)),
        CITY_STREET_HALF
            + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(ci.y + 1)));
    let lo = q < hw0;
    let hi = q > vec2<f32>(cell) - hw1;
    var bit = 99u;
    if ((lo.x || hi.x) && (lo.y || hi.y)) {
        bit = 4u + select(0u, 1u, hi.x) + select(0u, 2u, hi.y);
    } else if (lo.x) {
        bit = 0u;
    } else if (hi.x) {
        bit = 1u;
    } else if (lo.y) {
        bit = 2u;
    } else if (hi.y) {
        bit = 3u;
    }
    return bit < 8u && ((cc.streets >> bit) & 1u) != 0u;
}
