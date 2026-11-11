// --- The industrial belt's floodlights -----------------------------------
//
// The fringe is dark on purpose — a foil for the neon core — and what light
// it has comes from a few floods: wall packs over a warehouse's loading
// court, and a row of poles along one side of a yard. A lot's fixtures are a
// pure function of its cell (the building box says which long wall faces
// the court), so the ground pools here, the wall wash and the fixture
// geometry in the industry component all agree on where each lamp is.
//
// A pool is shaped by its fixture. A wall pack throws a half-ellipse
// forward from the wall foot, long along the wall and short out from it; a
// pole flood throws a lopsided ellipse down its aim, centred well ahead of
// the pole, long ahead and short behind — a symmetric one reached back past
// the pole and out of the lot, where the next lot's ground stops painting it
// and its tail ended on a hard line across the street (a blind judge's
// first catch). Both are Gaussians in the fixture's own (aim, across) frame
// with a sigma ahead, a sigma behind (zero: cut, the wall pack) and one
// across,
// and every aim is along a grid axis, so the far field keeps an exact mean:
// each pool's integral over the ground the near field actually shows — the
// lot's rectangle (a pool's tail past the lot edge is painted by nobody),
// cut at the wall for a wall pack, which also keeps it off the warehouse
// footprint — is a separable product of erf differences, and the mean is
// their sum over the visible ground's area (the lot less the footprint).
// Fixtures are spaced wider than their pools, so a court reads as a run of
// lit patches with dark between them, not as a lit floor.
//
// Colour follows the lot, the way a site is fitted out in one go: sodium,
// metal halide, cool LED or warm LED, with the odd fixture replaced by
// another family. Each family carries its own radiance (one shared peak
// renders every LED pool as a clipped white disc), set so the brightest
// family at the brightest draw stays under the tone map's knee at the
// reference exposure, and then 30% lower again: at the knee, a close view
// under a cool LED pole still reads as white ground.

const CITY_FLOOD_MAX: i32 = 4;
const CITY_FLOOD_SODIUM: vec3<f32> = vec3<f32>(0.42, 0.19, 0.052);
const CITY_FLOOD_HALIDE: vec3<f32> = vec3<f32>(0.29, 0.27, 0.21);
const CITY_FLOOD_LED: vec3<f32> = vec3<f32>(0.22, 0.26, 0.31);
const CITY_FLOOD_LED_WARM: vec3<f32> = vec3<f32>(0.32, 0.25, 0.16);
const CITY_FLOOD_WALL_OUT: f32 = 7.0;    // pool sigma out from the wall (m)
const CITY_FLOOD_WALL_ALONG: f32 = 7.0;  // ... and along it
const CITY_FLOOD_WALL_SPACING: f32 = 30.0;
const CITY_FLOOD_WALL_Z: f32 = 7.5;      // wall-pack mounting height
const CITY_FLOOD_POLE_THROW: f32 = 9.0;  // pool sigma down the aim
const CITY_FLOOD_POLE_BACK: f32 = 3.5;   // ... back toward the pole
const CITY_FLOOD_POLE_SIDE: f32 = 5.5;   // ... and across it
const CITY_FLOOD_POLE_REACH: f32 = 8.0;  // pool centre ahead of the pole
const CITY_FLOOD_POLE_SLEW: f32 = 3.0;   // ... and at most this far aside
const CITY_FLOOD_POLE_SPACING: f32 = 38.0;
const CITY_FLOOD_POLE_INSET: f32 = 1.1;  // pole line inside the plot edge
const CITY_FLOOD_POLE_H: f32 = 10.0;

struct CityFlood {
    on: bool,
    head: vec3<f32>,   // the fixture (city frame, m)
    aim: vec2<f32>,    // horizontal direction it throws its light (unit)
    pool: vec2<f32>,   // centre of its pool on the ground
    // Pool sigmas: ahead along the aim, behind (0: cut, as a wall pack's is
    // at the wall), and across.
    sigma: vec3<f32>,
    color: vec3<f32>,  // radiance at the pool centre
}

fn city_flood_family(i: i32) -> vec3<f32> {
    if (i == 0) {
        return CITY_FLOOD_SODIUM;
    }
    if (i == 1) {
        return CITY_FLOOD_HALIDE;
    }
    if (i == 2) {
        return CITY_FLOOD_LED;
    }
    return CITY_FLOOD_LED_WARM;
}

// Fixture k (0 .. CITY_FLOOD_MAX-1) of the lot this cell belongs to.
fn city_flood(cc: CityCell, k: i32) -> CityFlood {
    var f: CityFlood;
    f.on = false;
    f.head = vec3<f32>(0.0);
    f.aim = vec2<f32>(1.0, 0.0);
    f.pool = vec2<f32>(0.0);
    f.sigma = vec3<f32>(1.0);
    f.color = vec3<f32>(0.0);
    let fh = city_rand4(cc.seed ^ vec2<u32>(0x1f83d9abu, 0x5be0cd19u));
    let kh = city_rand4(cc.seed ^ vec2<u32>(
        0x9b05688cu * u32(k + 1), 0x510e527fu));
    // The lot's lamp family, and now and then a replaced fixture.
    var fam = select(select(select(3, 2, fh.x < 0.85), 1, fh.x < 0.60), 0,
                     fh.x < 0.40);
    if (kh.z < 0.15) {
        fam = (fam + 1 + i32(fract(kh.z * 23.7) * 2.999)) % 4;
    }
    // Per-fixture spread in output and throw, so no two pools are twins.
    let size = 0.80 + 0.40 * kh.w;
    if (cc.land_use == CITY_USE_INDUSTRIAL) {
        // Along the court wall: one slot per ~30 m, most of them lit.
        let long_x = (cc.b1max.x - cc.b1min.x) >= (cc.b1max.y - cc.b1min.y);
        var a0 = cc.b1min.y;
        var a1 = cc.b1max.y;
        var gap_lo = cc.b1min.x - cc.plot_min.x;
        var gap_hi = cc.plot_max.x - cc.b1max.x;
        if (long_x) {
            a0 = cc.b1min.x;
            a1 = cc.b1max.x;
            gap_lo = cc.b1min.y - cc.plot_min.y;
            gap_hi = cc.plot_max.y - cc.b1max.y;
        }
        let n = clamp(i32(floor((a1 - a0) / CITY_FLOOD_WALL_SPACING)), 1,
                      CITY_FLOOD_MAX);
        if (k >= n || kh.x > 0.75) {
            return f;
        }
        let court_hi = gap_hi >= gap_lo;
        let along = mix(a0, a1, (f32(k) + 0.5) / f32(n));
        let z = min(cc.height - 1.2, CITY_FLOOD_WALL_Z);
        var wall_foot = vec2<f32>(0.0);
        if (long_x) {
            let wall = select(cc.b1min.y, cc.b1max.y, court_hi);
            f.aim = vec2<f32>(0.0, select(-1.0, 1.0, court_hi));
            wall_foot = vec2<f32>(along, wall);
        } else {
            let wall = select(cc.b1min.x, cc.b1max.x, court_hi);
            f.aim = vec2<f32>(select(-1.0, 1.0, court_hi), 0.0);
            wall_foot = vec2<f32>(wall, along);
        }
        f.head = vec3<f32>(wall_foot + 0.8 * f.aim, z);
        f.pool = wall_foot;
        f.sigma = vec3<f32>(CITY_FLOOD_WALL_OUT, 0.0, CITY_FLOOD_WALL_ALONG)
                  * size;
    } else if (cc.land_use == CITY_USE_YARD) {
        // A row of poles along one long side of the yard, throwing across
        // it; a third of yards are dark, and the odd lamp is dead.
        let ps = cc.plot_max - cc.plot_min;
        let long_x = ps.x >= ps.y;
        let len = select(ps.y, ps.x, long_x);
        let n = clamp(i32(round(len / CITY_FLOOD_POLE_SPACING)), 1,
                      CITY_FLOOD_MAX);
        if (k >= n || fh.y < 0.33 || kh.x > 0.85) {
            return f;
        }
        let side_hi = fh.z < 0.5;
        let s = (f32(k) + 0.5) / f32(n);
        let lo = cc.plot_min + vec2<f32>(CITY_FLOOD_POLE_INSET);
        let hi = cc.plot_max - vec2<f32>(CITY_FLOOD_POLE_INSET);
        var at = vec2<f32>(0.0);
        var inward = vec2<f32>(0.0);
        if (long_x) {
            at = vec2<f32>(mix(lo.x, hi.x, s), select(lo.y, hi.y, side_hi));
            inward = vec2<f32>(0.0, select(1.0, -1.0, side_hi));
        } else {
            at = vec2<f32>(select(lo.x, hi.x, side_hi), mix(lo.y, hi.y, s));
            inward = vec2<f32>(select(1.0, -1.0, side_hi), 0.0);
        }
        // Heads are aimed by hand: the pool lands a few metres to one side
        // of square. The throw stays on the grid axis (a slewed pool, not a
        // rotated one) so its integral over the lot stays separable.
        let slew = CITY_FLOOD_POLE_SLEW * (2.0 * fract(kh.y * 7.31) - 1.0);
        f.aim = inward;
        f.head = vec3<f32>(at, CITY_FLOOD_POLE_H);
        f.pool = at + inward * (CITY_FLOOD_POLE_REACH * size)
                 + vec2<f32>(inward.y, inward.x) * slew;
        f.sigma = vec3<f32>(CITY_FLOOD_POLE_THROW, CITY_FLOOD_POLE_BACK,
                            CITY_FLOOD_POLE_SIDE) * size;
    } else {
        return f;
    }
    f.on = true;
    f.color = city_flood_family(fam) * (0.70 + 0.60 * kh.y);
    return f;
}

// Fixture f's pool at ground point p, in [0, 1] (times f.color for the
// radiance): the lopsided Gaussian in the fixture's own frame, cut behind
// the centre where its back sigma is zero.
fn city_flood_pool(f: CityFlood, p: vec2<f32>) -> f32 {
    let d = p - f.pool;
    let a = dot(d, f.aim);
    let b = f.aim.x * d.y - f.aim.y * d.x;
    let sa = select(f.sigma.y, f.sigma.x, a >= 0.0);
    if (sa <= 0.0) {
        return 0.0;
    }
    let qa = a / sa;
    let qb = b / f.sigma.z;
    return exp(-0.5 * (qa * qa + qb * qb));
}

// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). tests/test_city_floods.py
// pins these coefficients and the integral below against brute force.
fn city_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
               + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - poly * exp(-a * a));
}

// The integral of exp(-x^2 / 2 sigma^2) over [x0, x1].
fn city_gauss_int(x0: f32, x1: f32, sigma: f32) -> f32 {
    let k = 0.70710678 / sigma;
    return sigma * 1.2533141 * (city_erf(x1 * k) - city_erf(x0 * k));
}

// The integral of city_flood_pool over the rectangle [lo, hi] (m^2). The aim
// is along x or y, so the pool is separable in world axes: along the aim it
// is two half-Gaussians, the one behind absent for a wall pack; across it
// the Gaussian is whole.
fn city_flood_visible(f: CityFlood, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    let rlo = lo - f.pool;
    let rhi = hi - f.pool;
    let along_x = abs(f.aim.x) > 0.5;
    let s = select(f.aim.y, f.aim.x, along_x);
    let u0 = select(rlo.y, rlo.x, along_x);
    let u1 = select(rhi.y, rhi.x, along_x);
    // In the aim's own sense: a = s u.
    let a0 = select(-u1, u0, s > 0.0);
    let a1 = select(-u0, u1, s > 0.0);
    var along = city_gauss_int(max(a0, 0.0), max(a1, 0.0), f.sigma.x);
    if (f.sigma.y > 0.0) {
        along = along + city_gauss_int(min(a0, 0.0), min(a1, 0.0), f.sigma.y);
    }
    let b0 = select(rlo.x, rlo.y, along_x);
    let b1 = select(rhi.x, rhi.y, along_x);
    return along * city_gauss_int(b0, b1, f.sigma.z);
}

// Flood light on the ground at p: the lot's pools, dissolving into their
// exact mean over the lot as the pixel footprint outgrows a pool.
fn city_flood_ground(cc: CityCell, p: vec2<f32>, fp: f32) -> vec3<f32> {
    if (cc.land_use != CITY_USE_INDUSTRIAL && cc.land_use != CITY_USE_YARD) {
        return vec3<f32>(0.0);
    }
    var near = vec3<f32>(0.0);
    var flux = vec3<f32>(0.0);
    for (var k: i32 = 0; k < CITY_FLOOD_MAX; k = k + 1) {
        let f = city_flood(cc, k);
        if (f.on) {
            near = near + f.color * city_flood_pool(f, p);
            flux = flux + f.color * city_flood_visible(f, cc.lot_min,
                                                       cc.lot_max);
        }
    }
    // The ground the near field shows: the lot, less the warehouse on it
    // (b1 is zero-sized on a yard).
    let foot = (cc.b1max.x - cc.b1min.x) * (cc.b1max.y - cc.b1min.y);
    let area = (cc.lot_max.x - cc.lot_min.x) * (cc.lot_max.y - cc.lot_min.y)
               - foot;
    let blend = smoothstep(CITY_STREET_LOD_START, CITY_STREET_LOD_FULL, fp);
    return mix(near, flux / area, blend);
}

// One lattice corner of the ground mottle's octave o: a uniform draw at the
// corner, wrapped at the tile's `per` lattice cells.
fn city_mottle_corner(j: vec2<i32>, per: i32, o: i32) -> f32 {
    let jw = ((j % per) + per) % per;
    return city_u01(pcg2d(vec2<u32>(
        bitcast<u32>(jw.x) ^ 0x2c1b3c6du,
        bitcast<u32>(jw.y) ^ (u32(o) * 0x297a2d39u))).x);
}

// Patchy value noise over lot ground, mean one: scrub clumps at 14.4 m and
// grit at 3.6 m. Both lattices divide the tile (92160 m = 6400 x 14.4 =
// 25600 x 3.6), and the corners wrap at it,
// so the pattern is as tile-periodic as everything else in the city. Each
// octave fades to its mean (one) as the footprint approaches its scale.
fn city_ground_mottle(p: vec2<f32>, fp: f32) -> f32 {
    var m = 1.0;
    for (var o: i32 = 0; o < 2; o = o + 1) {
        let scale = select(3.6, 14.4, o == 0);
        let amp = select(0.35, 0.65, o == 0)
                  * (1.0 - smoothstep(0.25 * scale, scale, fp));
        if (amp > 0.0) {
            let per = i32(round(u.tile_extent / scale));
            let q = p / scale;
            let i = vec2<i32>(floor(q));
            let f = q - floor(q);
            let w = f * f * (3.0 - 2.0 * f);
            let c00 = city_mottle_corner(i, per, o);
            let c10 = city_mottle_corner(i + vec2<i32>(1, 0), per, o);
            let c01 = city_mottle_corner(i + vec2<i32>(0, 1), per, o);
            let c11 = city_mottle_corner(i + vec2<i32>(1, 1), per, o);
            let v = mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
            m = m * (1.0 + amp * (2.0 * v - 1.0));
        }
    }
    return m;
}
