// manholesteam — steam rising from the manholes, and the striped stacks
// Con Ed stands over the worst of them.
//
// WHERE. The core's manhole lattice (city_manhole: a site every
// CITY_MH_PITCH along every centreline, present at a hashed share) is the
// only place steam comes from; the ground shader draws a cover at exactly
// those sites, so a plume always rises from a cover. A present site steams
// when its steam draw is under a district fraction (cc_manholesteam_site),
// which puts a steaming cover every block or two downtown and almost none in
// the outskirts. The most vigorous STACK_SHARE of the steaming sites wear a
// stack: an orange-and-white banded tube 3 m tall standing on the cover,
// whose plume starts at its mouth instead of the road.
//
// THE PLUME is a bent-over buoyant plume, written in the plume's own
// coordinates (cc_manholesteam_density). It widens linearly with height (the
// entrainment law of a turbulent plume, r = r0 + SPREAD z), thins as it
// widens, leans downwind along one city-wide WIND, and dissolves over its
// top third. Its lumps are value noise measured in LOCAL RADII — across the
// plume by d / r, up it by the integral of dz / r — so an eddy is always
// the same fraction of the plume that carries it: small and tight at the
// cover, large and soft aloft. The lumps rise on the scene clock
// (scene_time; the live view draws them after its accumulation, so they go
// on rising over a parked view: fs_steam_live) at a fixed rate in zeta:
// in metres that is slow at the cover and faster aloft, where each eddy is
// larger — the plume billows up and opens out rather than scrolling.
//
// THE LIGHT is all it has. Steam is white (single-scattering albedo ~1), so
// its colour is entirely what lights it: the sodium lamp heads through
// city_street_light_g with a forward-peaked Henyey-Greenstein phase (a plume
// with a lamp behind it blazes, the look of every reference photograph),
// the storefront bands on both sides of the street (an infinite emitting
// strip, its solid angle exact, its phase approximated about the strip's
// nearest point), and the two hemispheres — the lit street below, the
// night sky above. Two density taps toward the light give the billows lit
// tops and grey undersides, and a knee on the source keeps a plume inside
// a lamp's reach from clipping to one flat sheet.
//
// By the city's own calibration the lamps light a parcel of steam about
// twenty times harder than a storefront does (a pool of 0.7 on asphalt of
// albedo 0.035 is an irradiance of ~60; a 2.2 storefront filling 1.6 sr is
// ~3.5), so downtown steam is sodium amber, and a storefront's colour shows
// only in plumes standing away from the lamp lines.
//
// THE COST. The medium is marched only where a ray enters a plume's box,
// and finding those boxes is the part every city pixel pays for, so it is
// built as a cascade of rejects (cc_manholesteam_media):
//   1. the ray segment is clipped to the slab z in [0, ZTOP] (the tallest
//      plume top), to the range cap and to the footprint cap. A ray that
//      never gets that low leaves with no work at all;
//   2. per street orientation, only the centrelines whose BAND (every site's
//      lateral offset plus the widest plume's reach) the segment enters are
//      visited, in ray order, and on each only the lattice sites whose
//      along-street extent overlaps the part of the segment inside the band.
//      A ray running along a street stays inside one band for its whole
//      length and walks that street's sites front to back — the eye-level
//      case, designed for; a ray crossing streets meets each band for a few
//      tens of metres and one or two sites;
//   3. one hash per site (present, steam draw) before any texture read;
//   4. one slab test per steaming site's box, and a march of 8-16 run-time
//      steps over the entered interval, jittered by the pixel's stratified
//      uniform so the accumulation integrates what a few steps sample.
// The loops are all rolled with run-time bounds and the density and the
// lighting are single functions called once, in one loop (the shadow taps
// share the sample's call site): the city and the clouds share one kernel,
// and occupancy is every pixel's cost.
//
// Measured (RTX 5080, 768x432, 256 frames at LOD 0.01, min of two
// back-to-back runs, component unregistered vs registered, ms/frame):
//     eye level, two plumes and a stack in view   2.11 -> 2.30   (+9%)
//     aerial                                       4.34 -> 4.38
//     base                                         5.31 -> 5.35
//     street                                       0.98 -> 0.98
//     horizon                                      5.86 -> 5.94
// The wide views pay ~1% — the gates plus the register footprint — and the
// eye-level view pays for its enumeration and its marches.

// --- where steam is ----------------------------------------------------------
// Which present sites steam is the core's (city_manhole_steam, with its
// district shares CITY_MH_STEAM_FRAC_*), which the ground asks too: a
// steaming cover is a dry, pale lid on a wet street.
// The most vigorous share of steaming sites carry a stack.
const cc_manholesteam_STACK_SHARE: f32 = 0.22;

// --- the plume ---------------------------------------------------------------
// One wind for the whole city, blowing toward the north-north-east: a
// diagonal to the grid, so no street sees its plumes lean straight along
// or straight across it.
const cc_manholesteam_WIND: vec2<f32> = vec2<f32>(0.4472136, 0.8944272);
// The source radius of a plume from a bare cover: the radius its body
// grows from (r = R0_COVER + SPREAD z, which the thinning and the lumps
// follow), wider than the iron (CITY_MH_R) because the steam billows out
// within a metre. A stack's leaves its own mouth.
const cc_manholesteam_R0_COVER: f32 = 0.9;
// The first metre of a cover's plume. A plume dense from its 0.9 m base
// buries its cover and seems to rise from a random bit of street. Steam
// leaves through the lid's vent holes and rim gap,
// so the plume's edge starts at the lid (CITY_MH_R) and relaxes onto the
// body's radius over NECK_L; and it leaves as hot vapour, near transparent,
// condensing into the visible plume as it mixes with the cold air — its
// density ramps as 1 - exp(-z / COND_L). The lid shows through the foot of
// its plume from an ordinary viewing angle. Stacks keep their mouths.
const cc_manholesteam_NECK_L: f32 = 0.35;
const cc_manholesteam_COND_L: f32 = 0.10;
const cc_manholesteam_H_MIN: f32 = 6.0;    // plume height above its source
const cc_manholesteam_H_MAX: f32 = 12.0;
// How fast the lumps climb, in zeta per second of scene clock: a lump at
// the cover (radius r0 ~0.3 m) rises ~0.35 m/s, one a metre across ~1.2 m/s.
const cc_manholesteam_RISE_RATE: f32 = 1.2;
// The city's scene clock wraps at its tile meta's clock_period_s, 2600 s,
// and the steam repeats exactly at that period: the lumps' lattice
// coordinate up the plume is 1.3 zeta, which climbs 1.3 * RISE_RATE * 2600
// = 4056 cells per period, so the lattice wraps in z at exactly that many
// cells (ZWRAP) and every octave steps up by exactly 2 in z (a whole
// multiple of it) — the wrap lands every octave on the lattice point it
// left, and no plume pops. Change RISE_RATE, the 1.3, or the period
// together, keeping 1.3 * RISE_RATE * CLOCK_PERIOD_S an integer.
const cc_manholesteam_CLOCK_PERIOD_S: f32 = 2600.0;
const cc_manholesteam_ZWRAP: f32 = 4056.0;
// Radius growth per metre of rise. 0.12 is a pure plume in still air; a
// plume bent over by a crosswind entrains faster, and steam also sheds
// visible mass as it mixes, which the thinning law below carries.
const cc_manholesteam_SPREAD: f32 = 0.24;
// Density falls as (R0_COVER / r)^THIN: 2 would conserve the steam mass of a
// plume of constant speed, but condensate evaporates as it mixes AND the
// plume decelerates, and the two roughly cancel in what the eye sees — a
// plume that stays readable to its top. 1.4 is set against the references.
const cc_manholesteam_THIN: f32 = 1.4;
// Downwind drift of the top, as a fraction of the plume height; the
// centreline bends as the square of the rise (it leaves the source
// vertical, which is what a jet does, and bends over as it slows).
const cc_manholesteam_LEAN_LO: f32 = 0.35;
const cc_manholesteam_LEAN_HI: f32 = 0.80;
const cc_manholesteam_LEAN_STACK: f32 = 0.95; // a stack's plume lies over
// Extinction at a cover's source core (m^-1): a metre of it takes four
// fifths of the light. A stack's mouth is denser still (see the density).
const cc_manholesteam_SIGMA: f32 = 1.6;
// The billows. n, the noise centred on zero, lies in [-0.5, 0.5]; the
// plume's edge sits at EDGE + BILLOW n radii from the axis, with a soft
// shoulder SOFT radii wide, and the density inside is scaled by
// 1 + LUMP n. CUT = EDGE + BILLOW / 2 is the farthest a billow can reach,
// so the profile is zero there by construction — a bound, never a visible
// edge. (The first pass cut a noise-displaced edge at a fixed radius, and
// every plume showed the straight silhouette of that cone.)
const cc_manholesteam_EDGE: f32 = 1.0;
const cc_manholesteam_BILLOW: f32 = 2.2;
const cc_manholesteam_SOFT: f32 = 0.3;
const cc_manholesteam_LUMP: f32 = 4.0;
const cc_manholesteam_CUT: f32 = 2.1;
// Light that has crossed the plume's own steam keeps this share even when
// the single-scattering estimate says it is gone: multiple scattering is
// what makes a dense cloud grey inside rather than black.
const cc_manholesteam_MS_FLOOR: f32 = 0.30;
// Self-shadowing: two density taps toward the light, at these distances in
// local radii, each weighted by SHADOW_SEG radii of path. This is what
// gives the billows lit tops and grey undersides.
const cc_manholesteam_SHADOW_TAP: vec2<f32> = vec2<f32>(0.4, 1.2);
const cc_manholesteam_SHADOW_SEG: f32 = 0.8;

// --- the stack ---------------------------------------------------------------
const cc_manholesteam_STACK_R: f32 = 0.45;
const cc_manholesteam_STACK_H: f32 = 3.0;
// Reflective sheeting, banded: an orange band of ORANGE_W every BAND_P,
// white between, the lowest band starting at the foot. The albedos are
// DIFFUSE albedos, about half what the white looks like in a headlight:
// retroreflective sheeting sends most of what it returns back toward its
// source. At the paint's full 0.6 the lamp light (in asphalt-relative
// units, divided by 0.035) clipped both bands to one cream colour.
const cc_manholesteam_BAND_P: f32 = 0.42;
const cc_manholesteam_ORANGE_W: f32 = 0.24;
const cc_manholesteam_ORANGE: vec3<f32> = vec3<f32>(0.32, 0.05, 0.006);
const cc_manholesteam_WHITE: vec3<f32> = vec3<f32>(0.30, 0.29, 0.27);
// Wet sheeting: the mirror lobe of the lamp heads, weighted by a Fresnel
// share. g sets the lobe's width.
const cc_manholesteam_SHEEN: f32 = 0.06;
const cc_manholesteam_SHEEN_G: f32 = 0.85;
const cc_manholesteam_INNER: f32 = 0.02;   // the tube's sooty inside

// --- enumeration bounds (derived from the above; restated as numbers because
// WGSL consts cannot call functions) ---------------------------------------
// Tallest plume top: a stack's mouth plus the tallest plume.
const cc_manholesteam_ZTOP: f32 = 15.0;
// Farthest a plume box reaches from its site in xy: the most-bent plume's
// drift, LEAN_STACK x H_MAX = 11.4, plus CUT x (the larger source radius,
// R0_COVER, + SPREAD x H_MAX) = 7.9.
const cc_manholesteam_REACH: f32 = 19.4;
// A site's lateral offset from its centreline (city_manhole): at most
// 1.6 + 3.1 x 2 on an avenue, 1.1 + 0.8 on a minor street; and its along
// jitter, +-3 m about the lattice point.
const cc_manholesteam_LAT_AVE: f32 = 7.8;
const cc_manholesteam_LAT_MINOR: f32 = 1.9;
const cc_manholesteam_ALONG_JIT: f32 = 3.0;

// --- LOD ---------------------------------------------------------------------
// Steam is a near thing. A plume is marched out to RANGE and fades to zero
// density over its last third; independently it fades with footprint, gone
// by FP_END (a plume then spans a few pixels, where it was only ever a faint
// smudge on the road). Whichever comes first ends the march.
const cc_manholesteam_RANGE: f32 = 360.0;
const cc_manholesteam_RANGE_FADE: f32 = 240.0;
const cc_manholesteam_FP_START: f32 = 0.6;
const cc_manholesteam_FP_END: f32 = 1.5;
// Noise octaves by footprint. The base lattice is r / 1.6, so octave k is
// r / (1.6 x 2^k) — at the widest plume (r ~ 3.8 m) 0.3 m for the fourth,
// 0.6 m for the third. Each is dropped once it is under a couple of pixels
// everywhere in the plume, which leaves its mean (the noise is centred).
const cc_manholesteam_OCT4_FP: f32 = 0.05;
const cc_manholesteam_OCT3_FP: f32 = 0.15;
// The march: STEP metres per step, 8 to 16 steps.
const cc_manholesteam_STEP: f32 = 0.7;
const cc_manholesteam_STEPS_MIN: i32 = 8;
const cc_manholesteam_STEPS_MAX: i32 = 16;

// --- lighting ----------------------------------------------------------------
// city_street_light_g is in the GROUND's units: radiance of a surface of
// asphalt's albedo facing the lamp, with its 4 pi phase normalisation making
// an isotropic medium equal to that surface. A white medium scatters
// 1 / (4 pi) of the irradiance per steradian where asphalt returns
// CITY_ASPHALT_ALBEDO / pi, so the conversion is 1 / (4 x 0.035).
const cc_manholesteam_LAMP_REL: f32 = 7.14;
const cc_manholesteam_G: f32 = 0.6;
// The storefront strip is broad, so its phase is taken about its nearest
// point with a gentler g.
const cc_manholesteam_STORE_G: f32 = 0.35;
// The source knee (the ground's CITY_POOL_KNEE, for the medium): the light
// a steam parcel sees within a few metres of a lamp head runs past 20 in
// these units, and uncompressed every plume downtown is one clipped sheet.
// Hue-preserving, and the dim fringes pass through untouched.
const cc_manholesteam_SRC_CAP: f32 = 6.0;

// ---------------------------------------------------------------------------
// A steaming site
// ---------------------------------------------------------------------------

struct cc_manholesteam_Site {
    on: bool,            // a present cover that steams
    stack: bool,         // ... under a stack
    c: vec2<f32>,        // the cover's centre
    z0: f32,             // the plume's source height (road or stack mouth)
    r0: f32,             // its radius there
    h: f32,              // plume height above z0
    lean: f32,           // top drift / h
    vigor: f32,          // density scale
    seed: vec2<f32>,     // noise offset
}

fn cc_manholesteam_site(k: i32, j: i32, along_y: bool) -> cc_manholesteam_Site {
    var s: cc_manholesteam_Site;
    s.on = false;
    s.stack = false;
    let m = city_manhole(k, j, along_y);
    let v = city_manhole_steam(m);   // uniform in [0, 1) over steaming sites
    if (v >= 1.0) {
        return s;
    }
    // Seeded like the site itself (city_manhole): the street and the along
    // index folded at the tile, so a plume keeps its shape across the seam.
    let jn = i32(round(u.tile_extent / CITY_MH_PITCH));
    let r = city_rand4(vec2<u32>(
        city_line_seed(k) * 0x2545f491u + select(0x6a09e667u, 0xbb67ae85u,
                                                 along_y),
        bitcast<u32>(((j % jn) + jn) % jn) * 0x9e3779b9u + 0x3c6ef372u));
    s.on = true;
    s.stack = v < cc_manholesteam_STACK_SHARE;
    s.c = m.xy;
    s.z0 = select(0.0, cc_manholesteam_STACK_H, s.stack);
    s.r0 = select(cc_manholesteam_R0_COVER, 0.85 * cc_manholesteam_STACK_R,
                  s.stack);
    s.h = mix(cc_manholesteam_H_MIN, cc_manholesteam_H_MAX, r.x);
    s.lean = select(mix(cc_manholesteam_LEAN_LO, cc_manholesteam_LEAN_HI, r.y),
                    cc_manholesteam_LEAN_STACK, s.stack);
    // The strongest leaks are the ones that got a stack; the weakest
    // barely clear the cover.
    s.vigor = 1.0 - 0.6 * v;
    s.seed = 97.0 * r.zw;
    return s;
}

// The plume's bounding box: the source disc and the top disc, each widened
// to the profile cut at the top radius.
fn cc_manholesteam_box_min(s: cc_manholesteam_Site) -> vec3<f32> {
    let rc = cc_manholesteam_CUT * (s.r0 + cc_manholesteam_SPREAD * s.h);
    let drift = cc_manholesteam_WIND * (s.lean * s.h);
    return vec3<f32>(s.c + min(drift, vec2<f32>(0.0)) - rc, s.z0);
}

fn cc_manholesteam_box_max(s: cc_manholesteam_Site) -> vec3<f32> {
    let rc = cc_manholesteam_CUT * (s.r0 + cc_manholesteam_SPREAD * s.h);
    let drift = cc_manholesteam_WIND * (s.lean * s.h);
    return vec3<f32>(s.c + max(drift, vec2<f32>(0.0)) + rc, s.z0 + s.h);
}

// ---------------------------------------------------------------------------
// The density
// ---------------------------------------------------------------------------

// 3-D value noise, mean 0.5, on the integer lattice.
fn cc_manholesteam_vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let w = f * f * (3.0 - 2.0 * f);
    // Two z slices of the 2-D lattice, told apart by a large shift; z wraps
    // at ZWRAP cells, which is what makes the steam periodic in the clock
    // (and keeps the hash's arguments small however long the flight).
    // i.z and ZWRAP are integers held exactly in f32, so the floor-div wrap
    // is exact but for one case: at an exact multiple a reciprocal a ulp
    // off can floor one short and give z0 = ZWRAP (or one long, -ZWRAP) —
    // a seam across the plume. The two selects put that back in [0, ZWRAP).
    // Not the euclidean ((a % n) + n) % n: that is docs/soar-bugs.md bug
    // 22's trigger (NVIDIA's compiler segfaults on it under Firefox).
    var z0 = i.z - cc_manholesteam_ZWRAP * floor(i.z / cc_manholesteam_ZWRAP);
    z0 = select(z0, z0 - cc_manholesteam_ZWRAP, z0 >= cc_manholesteam_ZWRAP);
    z0 = select(z0, z0 + cc_manholesteam_ZWRAP, z0 < 0.0);
    let z1 = select(z0 + 1.0, 0.0, z0 + 1.0 >= cc_manholesteam_ZWRAP);
    let s0 = i.xy + vec2<f32>(37.0, 17.0) * z0;
    let s1 = i.xy + vec2<f32>(37.0, 17.0) * z1;
    let lo = mix(mix(hash12(s0), hash12(s0 + vec2<f32>(1.0, 0.0)), w.x),
                 mix(hash12(s0 + vec2<f32>(0.0, 1.0)),
                     hash12(s0 + vec2<f32>(1.0, 1.0)), w.x), w.y);
    let hi = mix(mix(hash12(s1), hash12(s1 + vec2<f32>(1.0, 0.0)), w.x),
                 mix(hash12(s1 + vec2<f32>(0.0, 1.0)),
                     hash12(s1 + vec2<f32>(1.0, 1.0)), w.x), w.y);
    return mix(lo, hi, w.z);
}

// The plume at p, in its own coordinates: rise z above the source, radius
// r(z), and the offset q from the bent centreline. Returns the density
// relative to a full-vigour source core.
fn cc_manholesteam_density(p: vec3<f32>, s: cc_manholesteam_Site,
                           octaves: i32) -> f32 {
    let z = p.z - s.z0;
    if (z < 0.0 || z > s.h) {
        return 0.0;
    }
    let rise = z / s.h;
    let r = s.r0 + cc_manholesteam_SPREAD * z;
    let q = p.xy - s.c - cc_manholesteam_WIND * (s.lean * s.h * rise * rise);
    // A cover's plume: its edge from the lid's radius, its density from
    // nothing, over its first decimetres (NECK_L, COND_L).
    var r_edge = r;
    var cond = 1.0;
    if (!s.stack) {
        r_edge = r - (s.r0 - CITY_MH_R) * exp(-z / cc_manholesteam_NECK_L);
        cond = 1.0 - exp(-z / cc_manholesteam_COND_L);
    }
    // Distance from the axis in local radii. Past CUT nothing can reach
    // (r_edge <= r, so the box, sized by r, still bounds it).
    let d = length(q) / r_edge;
    if (d >= cc_manholesteam_CUT) {
        return 0.0;
    }
    // Lumps in local radii: across by q / r, up by zeta = integral of
    // dz / r = ln(r / r0) / SPREAD.
    // @domain: r >= r0 > 0: the plume's radius only grows from its source
    let zeta = log(r / s.r0) / cc_manholesteam_SPREAD
             - cc_manholesteam_RISE_RATE * scene_time();
    var x = vec3<f32>(1.6 * q / r + s.seed, 1.3 * zeta);
    var n = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    for (var o: i32 = 0; o < octaves; o = o + 1) {
        n = n + amp * cc_manholesteam_vnoise3(x);
        norm = norm + amp;
        // Exactly 2 in z, so each octave's lattice is periodic with the
        // first's (CLOCK_PERIOD_S); 2.03 across, off the lattice's grain.
        x = vec3<f32>(2.03, 2.03, 2.0) * x + vec3<f32>(19.1, 7.3, 3.7);
        amp = 0.5 * amp;
    }
    n = n / norm - 0.5;
    // The noise moves the boundary (a billow pushes it out, a gap pulls
    // it in) about EDGE radii, and pulls it in to nothing over the top half
    // of the rise, so the top is billowed too; inside, it modulates the
    // density. The steam thins as the plume widens.
    let boundary = (cc_manholesteam_EDGE + cc_manholesteam_BILLOW * n)
                 * (1.0 - smoothstep(0.6, 1.0, rise));
    // The thinning is referred to a cover's source radius for every
    // plume: a stack carries the same leak through a narrower mouth, so it
    // leaves denser — the opaque burst at the top of every stack — and
    // thins to the same plume aloft.
    let cover = smoothstep(0.0, cc_manholesteam_SOFT, boundary - d);
    return s.vigor * cond * cover * max(1.0 + cc_manholesteam_LUMP * n, 0.0)
           // @domain: r >= r0 > 0: the plume's radius only grows from its source
           * pow(cc_manholesteam_R0_COVER / r, cc_manholesteam_THIN);
}

// ---------------------------------------------------------------------------
// The light on it
// ---------------------------------------------------------------------------

// What stands either side of a site's street: the across coordinate of each
// building's storefront face and that storefront's glow. Found once per
// entered plume, not per step.
struct cc_manholesteam_Street {
    along_y: bool,
    face_lo: f32, face_hi: f32,
    glow_lo: vec3<f32>, glow_hi: vec3<f32>,
    fill: vec3<f32>,
}

fn cc_manholesteam_street(s: cc_manholesteam_Site, k: i32, along_y: bool)
        -> cc_manholesteam_Street {
    let cellm = u.tile_cell;
    var st: cc_manholesteam_Street;
    st.along_y = along_y;
    let a = i32(floor(select(s.c.x, s.c.y, along_y) / cellm));
    let ci_lo = select(vec2<i32>(a, k - 1), vec2<i32>(k - 1, a), along_y);
    let ci_hi = select(vec2<i32>(a, k), vec2<i32>(k, a), along_y);
    let c_lo = city_cell(ci_lo);
    let c_hi = city_cell(ci_hi);
    st.face_lo = select(c_lo.b1max.y, c_lo.b1max.x, along_y);
    st.face_hi = select(c_hi.b1min.y, c_hi.b1min.x, along_y);
    st.glow_lo = city_storefront_glow(c_lo);
    st.glow_hi = city_storefront_glow(c_hi);
    // The two hemispheres: the lit street below (its measured area mean,
    // CITY_GROUND_FAR_STREET, at this district's lamp scale) and the night
    // sky above. Each fills half the sphere an isotropic parcel sees. The
    // street half is what lights a plume from underneath, and what keeps
    // it from reading darker than the road it veils when seen from above.
    let ls = city_lamp_scale(s.c);
    let street = CITY_LAMP_COLOR * (CITY_GROUND_FAR_STREET.x * ls
                                    / (1.0 + CITY_GROUND_FAR_STREET.y * ls));
    let sky = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(s.c, 3.0));
    st.fill = 0.5 * (street + sky);
    return st;
}

// In-scattered radiance per unit scattering, at p, toward the eye along
// -dir. A storefront band is an emitting strip of height CITY_STORE_H and
// (for these purposes) infinite length at distance d: it subtends 2 theta
// steradians, theta its vertical angle, so an isotropic medium sees
// glow x theta / (2 pi) of it.
fn cc_manholesteam_light(p: vec3<f32>, dir: vec3<f32>,
                         st: cc_manholesteam_Street) -> vec3<f32> {
    var l = cc_manholesteam_LAMP_REL
          * city_street_light_g(p, dir, cc_manholesteam_G) + st.fill;
    let x = select(p.y, p.x, st.along_y);
    for (var side: i32 = 0; side < 2; side = side + 1) {
        let face = select(st.face_lo, st.face_hi, side == 1);
        let glow = select(st.glow_lo, st.glow_hi, side == 1);
        let d = max(abs(face - x), 0.3);
        let theta = atan((CITY_STORE_H - p.z) / d) + atan(p.z / d);
        // Toward the band across the same clamped distance theta uses, so
        // a sample on the facade plane at mid-band height has a direction.
        let dx = select(-d, d, face >= x);
        let toward = select(vec2<f32>(0.0, dx), vec2<f32>(dx, 0.0),
                            st.along_y);
        // @domain: |toward| = d >= 0.3
        let v = normalize(vec3<f32>(toward, 0.5 * CITY_STORE_H - p.z));
        l = l + glow * (theta / 6.2831853)
              * (12.566371 * hg_phase(dot(v, dir), cc_manholesteam_STORE_G));
    }
    let peak = max(l.r, max(l.g, l.b));
    return l / (1.0 + peak / cc_manholesteam_SRC_CAP);
}

// ---------------------------------------------------------------------------
// The media hook
// ---------------------------------------------------------------------------

// March one plume over [ta, tb]: (in-scattered radiance, transmittance).
fn cc_manholesteam_march(o: vec3<f32>, dir: vec3<f32>, ta: f32, tb: f32,
                         s: cc_manholesteam_Site, st: cc_manholesteam_Street,
                         fade: f32, octaves: i32, rnd: f32) -> vec4<f32> {
    let n = clamp(i32(ceil((tb - ta) / cc_manholesteam_STEP)),
                  cc_manholesteam_STEPS_MIN, cc_manholesteam_STEPS_MAX);
    let dt = (tb - ta) / f32(n);
    var rad = vec3<f32>(0.0);
    var trans = 1.0;
    for (var i: i32 = 0; i < n; i = i + 1) {
        let p = o + (ta + (f32(i) + rnd) * dt) * dir;
        // The sample, then two shadow taps toward the light, SHADOW_TAP
        // local radii out, each standing for SHADOW_SEG radii of path. The
        // light comes from above below the lamp heads and from below above
        // them (the heads, then the lit street), so the taps go vertically
        // toward whichever it is, and shrink to nothing through the heads'
        // own height, where the light arrives sideways — continuous, so the
        // plume has no seam there. The lateral offset of the head is what
        // this ignores. Rolled, with one call site for the density.
        let r = s.r0 + cc_manholesteam_SPREAD * max(p.z - s.z0, 0.0);
        let up = clamp(0.5 * (CITY_LAMP_HEAD_Z - p.z), -1.0, 1.0);
        var rho = 0.0;
        var tau_l = 0.0;
        for (var tap: i32 = 0; tap < 3; tap = tap + 1) {
            let off = select(0.0, select(cc_manholesteam_SHADOW_TAP.x,
                                         cc_manholesteam_SHADOW_TAP.y,
                                         tap == 2), tap > 0);
            let d = fade * cc_manholesteam_density(
                p + vec3<f32>(0.0, 0.0, up * off * r), s,
                select(octaves, 2, tap > 0));
            if (tap == 0) {
                rho = d;
                if (rho <= 1e-4) {
                    break;
                }
            } else {
                tau_l = tau_l + cc_manholesteam_SIGMA * d
                              * cc_manholesteam_SHADOW_SEG * r;
            }
        }
        if (rho <= 1e-4) {
            continue;
        }
        // Albedo 1: what the step removes from the beam it scatters in.
        // Multiple scattering keeps a floor under the shadowed light.
        let a = 1.0 - exp(-cc_manholesteam_SIGMA * rho * dt);
        let shadow = mix(exp(-tau_l), 1.0, cc_manholesteam_MS_FLOOR);
        rad = rad + trans * a * shadow * cc_manholesteam_light(p, dir, st);
        trans = trans * (1.0 - a);
        // Opaque enough that nothing behind can show: stop marching.
        if (trans < 0.01) {
            break;
        }
    }
    return vec4<f32>(rad, trans);
}

fn cc_manholesteam_media(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                         t_max: f32, pixel_angle: f32, rnd: f32)
        -> vec4<f32> {
    var acc = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    // 1. The segment that can meet a plume at all: under ZTOP, within the
    //    range, short of the footprint where every plume has faded out.
    var t_lo = 0.0;
    var t_hi = min(min(t_max, cc_manholesteam_RANGE),
                   cc_manholesteam_FP_END / pixel_angle);
    if (abs(dir.z) > 1e-6) {
        let tz = (cc_manholesteam_ZTOP - o.z) / dir.z;
        if (dir.z > 0.0) {
            t_hi = min(t_hi, tz);
        } else {
            t_lo = max(t_lo, tz);
        }
    } else if (o.z > cc_manholesteam_ZTOP) {
        return acc;
    }
    if (t_lo >= t_hi) {
        return acc;
    }
    let cellm = u.tile_cell;
    // Plumes are composited as found. Within one street they are found in
    // ray order; between streets, a plume wholly in front of everything so
    // far goes in front, and anything else behind — exact unless two
    // plumes overlap in depth, and then only the order of their colours is
    // approximate, never their transmittance.
    var t_front = 1e30;
    var t_back = 0.0;
    for (var axis: i32 = 0; axis < 2; axis = axis + 1) {
        let along_y = axis == 0;   // the streets x = k cellm, sites along y
        let oc = select(o.y, o.x, along_y);
        let dc = select(dir.y, dir.x, along_y);
        let oa = select(o.x, o.y, along_y);
        let da = select(dir.x, dir.y, along_y);
        // 2. Centrelines whose band the segment enters, in ray order.
        let band_max = cc_manholesteam_LAT_AVE + cc_manholesteam_REACH;
        let c0 = oc + t_lo * dc;
        let c1 = oc + t_hi * dc;
        let k_lo = i32(ceil((min(c0, c1) - band_max) / cellm));
        let k_hi = i32(floor((max(c0, c1) + band_max) / cellm));
        let k_step = select(-1, 1, dc >= 0.0);
        let k_first = select(k_hi, k_lo, dc >= 0.0);
        for (var ik: i32 = 0; ik <= k_hi - k_lo; ik = ik + 1) {
            let k = k_first + k_step * ik;
            let line = f32(k) * cellm;
            let band = cc_manholesteam_REACH + select(
                cc_manholesteam_LAT_MINOR, cc_manholesteam_LAT_AVE,
                city_is_avenue(k));
            var ta = t_lo;
            var tb = t_hi;
            if (abs(dc) > 1e-6) {
                let e0 = (line - band - oc) / dc;
                let e1 = (line + band - oc) / dc;
                ta = max(ta, min(e0, e1));
                tb = min(tb, max(e0, e1));
            } else if (abs(oc - line) > band) {
                continue;
            }
            if (ta >= tb) {
                continue;
            }
            // The lattice sites whose along extent overlaps the segment's
            // along range inside the band, in ray order.
            let a0 = oa + ta * da;
            let a1 = oa + tb * da;
            let slack = cc_manholesteam_REACH + cc_manholesteam_ALONG_JIT;
            let j_lo = i32(ceil((min(a0, a1) - slack) / CITY_MH_PITCH - 0.5));
            let j_hi = i32(floor((max(a0, a1) + slack) / CITY_MH_PITCH - 0.5));
            let j_step = select(-1, 1, da >= 0.0);
            let j_first = select(j_hi, j_lo, da >= 0.0);
            for (var ij: i32 = 0; ij <= j_hi - j_lo; ij = ij + 1) {
                let s = cc_manholesteam_site(k, j_first + j_step * ij,
                                             along_y);
                if (!s.on) {
                    continue;
                }
                // 4. The box, then the march.
                let bt = city_box_hit(o, inv_dir, cc_manholesteam_box_min(s),
                                      cc_manholesteam_box_max(s));
                let pa = max(bt.x, 0.0);
                let pb = min(bt.y, t_max);
                if (pa >= pb) {
                    continue;
                }
                let fp = pixel_angle * pa;
                let fade = (1.0 - smoothstep(cc_manholesteam_RANGE_FADE,
                                             cc_manholesteam_RANGE, pa))
                         * (1.0 - smoothstep(cc_manholesteam_FP_START,
                                             cc_manholesteam_FP_END, fp));
                if (fade <= 0.0) {
                    continue;
                }
                let octaves = 2 + select(0, 1, fp < cc_manholesteam_OCT3_FP)
                              + select(0, 1, fp < cc_manholesteam_OCT4_FP);
                let m = cc_manholesteam_march(
                    o, dir, pa, pb, s, cc_manholesteam_street(s, k, along_y),
                    fade, octaves, rnd);
                if (pb <= t_front && t_back > 0.0) {
                    acc = vec4<f32>(m.rgb + m.a * acc.rgb, acc.a * m.a);
                } else {
                    acc = vec4<f32>(acc.rgb + acc.a * m.rgb, acc.a * m.a);
                }
                t_front = min(t_front, pa);
                t_back = max(t_back, pb);
            }
        }
    }
    return acc;
}

// ---------------------------------------------------------------------------
// The stacks: geometry and shading
// ---------------------------------------------------------------------------

fn cc_manholesteam_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// The stacks standing in cell ci. A stack's site is on one of the cell's
// four boundary centrelines, offset into the cell by its lateral (always
// more than a stack radius, so no stack straddles two cells); each edge is
// a strip box, and only a ray that enters a strip looks up the few sites
// along the part of it the ray crosses.
fn cc_manholesteam_props_trace(o: vec3<f32>, dir: vec3<f32>,
                               inv_dir: vec3<f32>, t0: f32, t1: f32,
                               ci: vec2<i32>, cc: CityCell) -> CityHit {
    var res = cc_manholesteam_miss(ci);
    // Gate 1: the segment reaches down to the stacks' height.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_manholesteam_STACK_H || max(za, zb) < 0.0) {
        return res;
    }
    // Gate 2: past FP_END the plume above a stack has faded out, and the
    // stack is under half a pixel wide.
    let fp = lod_pixel_angle() * max(t0, 0.0);
    if (fp > cc_manholesteam_FP_END) {
        return res;
    }
    let cellm = u.tile_cell;
    let lo = vec2<f32>(ci) * cellm;
    let hi = lo + vec2<f32>(cellm);
    var t_end = t1;
    for (var e: i32 = 0; e < 4; e = e + 1) {
        let along_y = e < 2;
        let k = select(ci.y, ci.x, along_y) + (e & 1);
        let line = f32(k) * cellm;
        let lat = select(cc_manholesteam_LAT_MINOR, cc_manholesteam_LAT_AVE,
                         city_is_avenue(k)) + cc_manholesteam_STACK_R;
        // The strip: the edge's centreline +- lat, clipped to the cell.
        let c_lo = max(line - lat, select(lo.y, lo.x, along_y));
        let c_hi = min(line + lat, select(hi.y, hi.x, along_y));
        let a_lo = select(lo.x, lo.y, along_y);
        let a_hi = select(hi.x, hi.y, along_y);
        let bmin = select(vec3<f32>(a_lo, c_lo, 0.0), vec3<f32>(c_lo, a_lo, 0.0),
                          along_y);
        let bmax = select(vec3<f32>(a_hi, c_hi, cc_manholesteam_STACK_H),
                          vec3<f32>(c_hi, a_hi, cc_manholesteam_STACK_H),
                          along_y);
        let bt = city_box_hit(o, inv_dir, bmin, bmax);
        let ta = max(bt.x, t0);
        let tb = min(bt.y, t_end);
        if (ta >= tb) {
            continue;
        }
        let oa = select(o.x, o.y, along_y);
        let da = select(dir.x, dir.y, along_y);
        let s0 = oa + ta * da;
        let s1 = oa + tb * da;
        let slack = cc_manholesteam_ALONG_JIT + cc_manholesteam_STACK_R;
        let j_lo = i32(ceil((min(s0, s1) - slack) / CITY_MH_PITCH - 0.5));
        let j_hi = i32(floor((max(s0, s1) + slack) / CITY_MH_PITCH - 0.5));
        for (var j: i32 = j_lo; j <= j_hi; j = j + 1) {
            let s = cc_manholesteam_site(k, j, along_y);
            if (!s.stack || any(s.c < lo) || any(s.c >= hi)) {
                continue;
            }
            // A vertical open tube: the outside wall, or — for a ray that
            // comes in over the rim — the inside wall or the floor.
            let oc = o.xy - s.c;
            let qa = max(dot(dir.xy, dir.xy), 1e-12);
            let qb = dot(oc, dir.xy);
            let qc = dot(oc, oc)
                   - cc_manholesteam_STACK_R * cc_manholesteam_STACK_R;
            let disc = qb * qb - qa * qc;
            if (disc <= 0.0) {
                continue;
            }
            // @domain: the discriminant is tested positive (or non-negative) just above
            let sq = sqrt(disc);
            let tn = (-qb - sq) / qa;
            let tf = (-qb + sq) / qa;
            let zn = o.z + tn * dir.z;
            var th = -1.0;
            var inner = false;
            if (zn >= 0.0 && zn <= cc_manholesteam_STACK_H) {
                th = tn;
            } else if (zn > cc_manholesteam_STACK_H) {
                // In over the rim — but only if the far wall is met below
                // it: a ray that crosses the whole circle above 3 m passes
                // clean over the mouth.
                let zf = o.z + tf * dir.z;
                if (zf > cc_manholesteam_STACK_H) {
                    continue;
                }
                inner = true;
                th = select(tf, -o.z / dir.z, zf < 0.0);
            }
            if (th < t0 || th >= t_end) {
                continue;
            }
            let p = o + th * dir;
            var nrm = vec3<f32>((p.xy - s.c) / cc_manholesteam_STACK_R, 0.0);
            if (inner) {
                nrm = select(-nrm, vec3<f32>(0.0, 0.0, 1.0), p.z < 1e-3);
            }
            res = CityHit(true, th, p, nrm, select(900, 901, inner), ci);
            t_end = th;
        }
    }
    return res;
}

fn cc_manholesteam_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let p = h.pos;
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
    if (h.kind == 901) {
        return cc_manholesteam_INNER
               * (city_street_light(p, vec3<f32>(0.0, 0.0, 1.0)) + fill);
    }
    // The bands, box-filtered in height (they dissolve to their duty
    // cycle once they are finer than a pixel).
    let orange = city_stripes(p.z, 0.0, cc_manholesteam_ORANGE_W,
                              cc_manholesteam_BAND_P, fp);
    let alb = mix(cc_manholesteam_WHITE, cc_manholesteam_ORANGE, orange);
    let diffuse = alb * (city_street_light(p, h.normal) / CITY_ASPHALT_ALBEDO
                         + fill);
    // Wet: the lamps' mirror lobe about the reflected view ray.
    let r = reflect(dir, h.normal);
    let sheen = cc_manholesteam_SHEEN
              * city_street_light_g(p, r, cc_manholesteam_SHEEN_G);
    return diffuse + sheen;
}
