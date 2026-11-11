// The sea: an opaque plane at z = u.surface.x whose surface is a static wave
// field. ocean_shade reflects the rendered sky over the unresolved slope
// distribution, adds a GGX sun glint drawn as glitter, lights a Lambertian
// water body, and hazes the water with distance.

// Added to the mip level a pixel's footprint selects.
const OCEAN_MIP_BIAS: f32 = -0.5;
// The sun glint: its strength, and its GGX base roughness, the RMS slope of
// the ripples smaller than a tile texel. With the tile's own slope variance
// the sea's mean square slope is Cox and Munk's at a wind of about 7 m/s.
const OCEAN_GLINT_STRENGTH: f32 = 0.85;
const OCEAN_GLINT_ROUGHNESS: f32 = 0.170;
// The fraction of the unresolved slope variance that the sky reflection
// integrates over. The glint takes all of it, as lobe width.
const OCEAN_SLOPE_RESOLVED_FRACTION: f32 = 0.5;
// The floor on the cloud-shadow dimming of the sky reflection.
const OCEAN_SKY_SHADOW_FLOOR: f32 = 0.75;

// ---------------------------------------------------------------------------
// The wave field: a static FIF (fractionally integrated flux) tile, 1024^2
// texels of 0.1 m (102.4 m, repeated), stored as unit normals with
// box-filtered mips at binding 3. A pixel reads the mip its footprint
// covers. What that mip averaged away is handed back as slope variance: to
// the glint's microfacet lobe, and in part to the sky reflection's integral
// over slope.

// The slope variance each mip level has removed from the tile's normals:
// none at level 0, and by the top level the tile's whole variance (an RMS
// slope of 0.098). The table belongs to the tile and changes with it. It is
// a switch and not a const array because a module-scope array indexed at run
// time costs a private variable at fs_main's entry, which enlarges every
// scene's fragment program.
fn fif_slope_variance(k: i32) -> f32 {
    switch k {
        case 0: { return 0.000000; }
        case 1: { return 0.000523; }
        case 2: { return 0.001609; }
        case 3: { return 0.003032; }
        case 4: { return 0.004580; }
        case 5: { return 0.006378; }
        case 6: { return 0.008571; }
        case 7: { return 0.009524; }
        case 8: { return 0.009662; }
        case 9: { return 0.009699; }
        default: { return 0.009703; }
    }
}

// The table, linearly interpolated at a fractional mip level.
fn fif_slope_variance_removed(lod: f32) -> f32 {
    let l0 = clamp(floor(lod), 0.0, 10.0);
    let l1 = min(l0 + 1.0, 10.0);
    let f = clamp(lod - l0, 0.0, 1.0);
    return mix(fif_slope_variance(i32(l0)), fif_slope_variance(i32(l1)), f);
}

// The tile's normal at one mip level. The tile is read at
// (world - surface offset) / extent; the half-texel shift puts texel i's
// centre at i cells.
fn ocean_normal_mip_sample(world_xy: vec2<f32>, level: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(ocean_normals, i32(level)));
    let coord = (world_xy - u.surface_offset) / u.tile_extent
        + 0.5 / dims;
    return textureSampleLevel(ocean_normals, ocean_samp, coord, level).rgb;
}

// The footprint's mean normal at a fractional mip: the two levels blended.
fn ocean_wave_normal_mipped(world_xy: vec2<f32>, lod: f32) -> vec3<f32> {
    let level0 = floor(lod);
    let level1 = min(level0 + 1.0, u.tile_max_lod);
    let f = lod - level0;
    var n = ocean_normal_mip_sample(world_xy, level0);
    if (level1 > level0 && f > 0.0) {
        n = mix(n, ocean_normal_mip_sample(world_xy, level1), f);
    }
    // @domain: every texel of every mip has z >= 0.923 (the tile's bytes), and filtering and the mip mix are convex
    return normalize(n);
}

// Glitter. The glint lobe is the mean of what the unresolved facets in a
// footprint reflect. But the sun is a disc 0.53 deg across, so of the facets
// under a pixel only a handful, or none, are tilted to send it to the
// camera: a pixel is one brilliant facet or dark water. So the footprint's
// glint is drawn as K / lambda times the lobe, K ~ Poisson(lambda), with
// lambda the expected number of facets that reflect the disc:
// (footprint / facet area) times the probability that one does,
// D(h) (n.h) / (4 v.h) times the sun's solid angle. E[K / lambda] = 1, so
// the mean is untouched, and so is the far field, where lambda is large and
// the draw is 1 to within its own noise. The draw is seeded by a cell of the
// sea at the pixel's own scale, fixed in the tile's frame, so a parked view
// keeps its sparkles: they are the surface, not noise for the accumulation
// to average away.
//
// A capillary facet is ~1 cm (the base roughness stands for everything below
// the tile's 0.1 m texel). Cells are that times a power of two, so a whole
// number of them spans the tile and the pattern repeats with it.
const SEA_GLITTER_FACET_M: f32 = 0.0125;          // 102.4 m / 2^13
const SEA_SUN_SOLID_ANGLE: f32 = 6.8e-5;          // pi (0.266 deg)^2

// Two uniforms in [0, 1) from a seed: 24 bits each, so they stay below 1 in
// float32 (a full u32 rounds to 1.0 at the top).
fn sea_uniforms(seed: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(seed >> vec2<u32>(8u)) * (1.0 / 16777216.0);
}

// How bright the K facets that catch the sun are, in units of the mean
// facet. A facet's contribution goes as its area and as how squarely it
// holds the disc, and both vary from facet to facet: most glints are faint
// and a few are blinding. So each facet is a lognormal draw of mean exactly
// 1 (log standard deviation SEA_GLITTER_SPREAD) and the K are summed, which
// keeps E[sum / lambda] = 1. Past a few facets the sum is replaced by its
// mean, K.
const SEA_GLITTER_SPREAD: f32 = 1.3;
// Where glitter is drawn.
//
// Only in the core of the sun's lobe, around its reflection. Out in the
// lobe's tails a lone facet catching the sun is possible but rare, and drawn
// as such it strews the whole sea with equally bright dots. So the draw is
// blended toward its own mean, the smooth lobe, by where the half vector
// sits in the lobe: GGX D(h) over its peak, which is 1 on the sun's
// reflection (CORE_LO to CORE_HI).
//
// And only where a footprint expects a few glints (FEW to MANY). With lambda
// well under one a cell is almost always empty, and beside smooth sheen an
// empty cell on a sunlit wave face reads as a hole in the water. From a few
// glints up an empty cell is rare (e^-4 at lambda = 4) and the draw reads as
// texture. This weight takes the expected count at the footprint's own size
// and not the cell's: the cell is rounded to a power of two, so its lambda
// jumps fourfold where the rounding steps, as wide as the ramp itself, and
// a ramp on it would draw a contour at a fixed footprint.
//
// The count over lambda and each facet's brightness both have mean 1, so any
// blend keeps the lobe's mean exactly: the energy is the smooth specular's.
const SEA_GLITTER_CORE_LO: f32 = 0.3;
const SEA_GLITTER_CORE_HI: f32 = 0.7;
const SEA_GLITTER_FEW: f32 = 1.0;
const SEA_GLITTER_MANY: f32 = 4.0;

// The summed brightness of k facets: k lognormal draws of mean 1, from
// Box-Muller normals. Zero facets give 0, and four or more give k.
fn sea_glitter_brightness(k: f32, seed: vec2<u32>) -> f32 {
    if (k >= 4.0 || k <= 0.0) {
        return k;
    }
    let s2 = SEA_GLITTER_SPREAD * SEA_GLITTER_SPREAD;
    var total = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (f32(i) >= k) {
            break;
        }
        let r = sea_uniforms(pcg2d(seed + vec2<u32>(0x9e3779b9u * (i + 1u),
                                                   0x85ebca6bu)));
        // @domain: a uniform < 1, so the log is <= 0
        let z = sqrt(-2.0 * log(max(r.x, 1e-7)))
            * cos(6.283185307179586 * r.y);
        total = total + exp(SEA_GLITTER_SPREAD * z - 0.5 * s2);
    }
    return total;
}

// K ~ Poisson(lambda) from two uniforms: inversion while lambda is small,
// the normal approximation (rounded, clamped at 0) past it.
fn sea_poisson(lambda: f32, r: vec2<f32>) -> f32 {
    if (lambda > 16.0) {
        // @domain: a uniform < 1, so the log is <= 0
        let z = sqrt(-2.0 * log(max(r.x, 1e-7)))
            * cos(6.283185307179586 * r.y);
        // @domain: lambda > 16 on this branch
        return max(round(lambda + sqrt(lambda) * z), 0.0);
    }
    let e = exp(-lambda);
    var pmf = e;
    var cdf = e;
    var k = 0.0;
    for (var i = 0; i < 48; i = i + 1) {
        if (r.x <= cdf) {
            break;
        }
        k = k + 1.0;
        pmf = pmf * lambda / k;
        let next = cdf + pmf;
        // In float32 the cdf can stall just below 1; there is no more mass
        // past that.
        if (next <= cdf) {
            break;
        }
        cdf = next;
    }
    return k;
}

// The sky the water reflects: the sky the camera sees, without the solar
// disc (the glint is the sun's reflection). A sunset sea therefore takes the
// sunset's colour.
fn ocean_reflected_sky(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    return sky_radiance(vec3<f32>(dir.xy, max(dir.z, 0.0)), sun,
                        u.sky_horizon, u.sky_bloom, vec3<f32>(0.0));
}

// What the sea reflects of the sky, and lets through to the water below,
// integrated over the sub-pixel slope distribution about the footprint's
// mean slope: a Gaussian of standard deviation `sigma` per axis (the slope
// of a filtered, roughly Gaussian wave field is Gaussian). At grazing
// incidence the reflected radiance is strongly convex in slope, because
// Fresnel goes as (1 - n.v)^5 and the sky's gradient is steep at the
// horizon, so the footprint's shading is not the shading at its mean slope.
//
// The integral is a fixed quadrature and not a random draw, so it has no
// noise for the accumulation to remove: four taps at +-sqrt(2) sigma along
// each slope axis, weight 1/4 each, which match the Gaussian's moments
// through the third. What no moment-matching rule captures is the
// integrand's creases: a few degrees below the horizon a sizeable part of
// the footprint's slopes face away from the viewer (n.v clamps to 0, Fresnel
// to 1) or reflect below the horizon (folded back up, below).
struct SeaReflection {
    // The Fresnel-weighted reflected sky, averaged over the taps.
    sky: vec3<f32>,
    // (1 - Fresnel) * max(n.l, 0) averaged over the taps: the share of the
    // direct sun the water body below receives, per unit beam.
    transmitted_n_dot_l: f32,
}

fn ocean_sky_reflection(dir: vec3<f32>, sun: vec3<f32>, slope: vec2<f32>,
                        sigma: f32) -> SeaReflection {
    var r = SeaReflection(vec3<f32>(0.0), 0.0);
    let d = 1.41421356237 * sigma;
    for (var k = 0; k < 4; k++) {
        // The taps without an array: +d, -d along x, then along y.
        let along = select(d, -d, (k & 1) == 1);
        let offset = select(vec2<f32>(along, 0.0), vec2<f32>(0.0, along),
                            k >= 2);
        let n = normalize(vec3<f32>(-(slope + offset), 1.0));
        let vdotn = dot(dir, n);
        var refl = dir - 2.0 * vdotn * n;
        if (refl.z < 0.0) {
            refl.z = -refl.z;
        }
        let sky = ocean_reflected_sky(refl, sun);
        // Water Fresnel for the resolved sky reflection.
        let n_dot_v = clamp(-vdotn, 0.0, 1.0);
        let one_minus = 1.0 - n_dot_v;
        let om2 = one_minus * one_minus;
        let fresnel = 0.02 + 0.98 * om2 * om2 * one_minus;
        r.sky = r.sky + 0.25 * fresnel * sky;
        r.transmitted_n_dot_l = r.transmitted_n_dot_l
            + 0.25 * (1.0 - fresnel) * max(0.0, dot(sun, n));
    }
    return r;
}

// The sea at `hit`: the sky reflection of ocean_sky_reflection, the sun's
// glint, and the Lambertian water body, then haze toward the horizon colour.
// `sun_shadow` is the direction drawn on the solar disc and serves only the
// cloud-shadow march; every shading term keeps the disc centre.
fn ocean_shade(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
               sun_shadow: vec3<f32>, t_hit: f32,
               jit: f32) -> vec3<f32> {
    let haze_beta0 = u.ocean_haze_beta;

    // One pixel's angular span projected onto the water. The grazing factor
    // stretches it along the view and takes the horizon to coarser mips.
    let grazing = max(abs(dir.z), 0.03);
    let pixel_angular_span = true_pixel_angle();
    let across = t_hit * pixel_angular_span;
    let along = across / grazing;
    let texel_span = max(along / u.tile_cell, 1.0);
    var lod = log2(texel_span) + OCEAN_MIP_BIAS;
    lod = clamp(lod, 0.0, u.tile_max_lod);

    let h_len = length(dir.xy);
    var h = vec2<f32>(1.0, 0.0);
    if (h_len > 1e-8) {
        h = dir.xy / h_len;
    }
    // The footprint's mean slope from the tile's mip, and the slope variance
    // that mip averaged away.
    let n_mip = ocean_wave_normal_mipped(hit.xy, lod);
    let sea_slope = -n_mip.xy / max(n_mip.z, 1e-4);
    let sea_variance = fif_slope_variance_removed(lod);
    // The glitter's cell: half the footprint, rounded up to a power of two
    // of facets and folded at the tile. It is smaller than the pixel, so the
    // jittered samples of one pixel see several cells and a sparkle lands as
    // an antialiased point, not as a cell-shaped tile.
    // @domain: along and across are footprints: t_hit times a positive pixel span
    let side = max(0.5 * sqrt(along * across), SEA_GLITTER_FACET_M);
    // @domain: side >= SEA_GLITTER_FACET_M > 0 (the max above)
    let level = ceil(log2(side / SEA_GLITTER_FACET_M));
    let cell = SEA_GLITTER_FACET_M * exp2(level);
    let p = hit.xy - u.surface_offset;
    let q = p - u.tile_extent * floor(p / u.tile_extent);
    let glitter_seed = pcg2d(vec2<u32>(floor(q / cell))
                             + vec2<u32>(u32(level) * 40503u, 0u));
    let glitter_facets = exp2(2.0 * level);
    // The same count before the cell is rounded to a power of two. The
    // draw's weight reads this one, so it varies smoothly with the
    // footprint.
    let glitter_count = (side / SEA_GLITTER_FACET_M)
        * (side / SEA_GLITTER_FACET_M);

    // A pixel covers many waves: a slope distribution whose variance the mip
    // removed. The sky reflection integrates OCEAN_SLOPE_RESOLVED_FRACTION
    // of that variance (per axis, hence the 0.5: the variances are totals
    // over both axes). The sun's glint below takes all of it, as lobe width.
    let reflection = ocean_sky_reflection(
        dir, sun, sea_slope,
        // @domain: a fraction of a variance: >= 0
        sqrt(0.5 * OCEAN_SLOPE_RESOLVED_FRACTION * sea_variance));

    let tau_ocean = light_march_tau(hit, sun_shadow, t_hit * u.light_lod_tan,
                                    jit);
    let t_sun_ocean = exp(-tau_ocean);

    // Sun glint: a GGX lobe tinted by the beam colour, about the footprint's
    // mean normal, with all of the unresolved variance in the lobe's width.
    // It is not evaluated at the sky reflection's slope taps: the glitter
    // draw below already is the statistics of the unresolved facets.
    let n_g = normalize(vec3<f32>(-sea_slope, 1.0));
    let n_g_dot_l = max(0.0, dot(sun, n_g));
    let n_g_dot_v = clamp(-dot(dir, n_g), 0.0, 1.0);
    var glint = vec3<f32>(0.0);
    var sun_fresnel = 0.02;
    if (n_g_dot_l > 0.0 && n_g_dot_v > 1e-8) {
        var hv = sun - dir;
        let hv_len = length(hv);
        if (hv_len > 1e-8) {
            hv = hv / hv_len;
            let n_dot_h = dot(n_g, hv);
            if (n_dot_h > 0.0) {
                let v_dot_h = clamp(dot(-dir, hv), 0.0, 1.0);
                let one_minus_vh = 1.0 - v_dot_h;
                let vh2 = one_minus_vh * one_minus_vh;
                sun_fresnel = 0.02 + 0.98 * vh2 * vh2 * one_minus_vh;

                // The base roughness is what lies below the tile's 0.1 m
                // texels, ripple that no tile holds, plus the variance the
                // footprint's mip filtered away. Variances add; roughnesses
                // do not.
                let alpha = clamp(
                    // @domain: a square plus a variance: >= 0
                    sqrt(OCEAN_GLINT_ROUGHNESS * OCEAN_GLINT_ROUGHNESS
                         + sea_variance),
                    0.02, 0.75
                );
                let alpha_squared = alpha * alpha;
                let denom = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
                let d_ggx = alpha_squared
                    / (3.14159265358979 * denom * denom);
                let g_smith = ggx_smith_g1(n_g_dot_v, alpha_squared)
                    * ggx_smith_g1(n_g_dot_l, alpha_squared);
                let spec = OCEAN_GLINT_STRENGTH * t_sun_ocean * d_ggx * g_smith
                    * sun_fresnel / (4.0 * n_g_dot_v);
                // Glitter (see SEA_GLITTER_FACET_M): the lobe drawn as the
                // facets that catch the disc, in the lobe's core only.
                let lambda = glitter_facets * d_ggx * n_dot_h
                    * SEA_SUN_SOLID_ANGLE / (4.0 * max(v_dot_h, 1e-4));
                // D(h) / D(n.h = 1) = alpha^4 / denom^2, in (0, 1].
                let lobe = alpha_squared * alpha_squared / (denom * denom);
                let core = smoothstep(SEA_GLITTER_CORE_LO,
                                      SEA_GLITTER_CORE_HI, lobe)
                    * smoothstep(SEA_GLITTER_FEW, SEA_GLITTER_MANY,
                                 lambda * glitter_count
                                     / glitter_facets);
                var sparkle = 1.0;
                if (core > 0.0) {
                    let k = sea_poisson(lambda, sea_uniforms(glitter_seed));
                    let draw = sea_glitter_brightness(k, glitter_seed)
                        / lambda;
                    sparkle = 1.0 + core * (draw - 1.0);
                }
                glint = spec * sparkle * u.cloud_sun;
            }
        }
    }

    // The water body takes the complement of the Fresnel share given to the
    // glint, so the specular sun path does not create energy.
    let diffuse_partition = 1.0 - min(OCEAN_GLINT_STRENGTH, 1.0) * sun_fresnel;
    let diff_irr = t_sun_ocean * reflection.transmitted_n_dot_l
        * 0.3183098861837907 * diffuse_partition;
    let diffuse = diff_irr * SUN_COLOR * u.surface.yzw;

    // Cloud shadow, per term. The sun terms already carry t_sun_ocean. The
    // sky reflection dims only to OCEAN_SKY_SHADOW_FLOOR: under a cloud the water
    // still reflects the mostly unblocked sky dome and keeps its wave
    // texture.
    let sky_shadow = OCEAN_SKY_SHADOW_FLOOR
        + (1.0 - OCEAN_SKY_SHADOW_FLOOR) * t_sun_ocean;
    var ol = reflection.sky * sky_shadow + diffuse + glint;

    // Haze along the sightline to the water, toward the horizon sky colour
    // the cloud haze fades into (solar disc excluded), so the sea meets the
    // sky without a seam.
    let haze_tau = haze_beta0 * t_hit;
    if (haze_tau > 0.0) {
        let haze = 1.0 - exp(-haze_tau);
        let haze_col = sky_radiance(
            vec3<f32>(h, 0.0), sun,
            u.sky_horizon, u.sky_bloom, vec3<f32>(0.0)
        );
        ol = mix(ol, haze_col, haze);
    }

    return ol;
}
