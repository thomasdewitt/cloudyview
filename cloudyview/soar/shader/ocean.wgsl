const OCEAN_SHADOW_FLOOR: f32 = 0.35; // witness legacy ocean shadow floor

// Ocean reflection sky ported from witness._reflection_sky
// (witness.py lines 445-470).
fn ocean_reflection_sky(dir: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    t = 1.0 - one_minus * one_minus * one_minus;

    let zenith = vec3<f32>(0.0044, 0.035, 0.1156);
    let horizon = vec3<f32>(0.10, 0.18, 0.38);
    var col = horizon + (zenith - horizon) * t;

    let cos_rs = dot(dir, sun);
    if (cos_rs > 0.0) {
        let glint_w = 0.02;
        let a = glint_w / ((1.0 - cos_rs) + glint_w);
        col = col + a * vec3<f32>(1.2, 1.0, 0.6);
    }
    return col;
}

// ---------------------------------------------------------------------------
// The sea: a static FIF wave field (cloudyview/ocean_fif.py), 1024^2 texels of
// 0.1 m (102.4 m, tiled), stored as unit normals with box-filtered mips and
// bound at 3. A pixel reads the mip its footprint covers; what that mip
// averaged away is handed back as slope variance — to the microfacet lobe,
// and in part to the sky reflection's integral over slope.

// The slope variance each mip level removed from the tile's normals (level 0
// removes none; by the top, the tile's whole 0.098 RMS slope). A switch, not
// a const array: a runtime-indexed array costs a private variable at
// fs_main's entry and moves every scene's registers.
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

fn fif_slope_variance_removed(lod: f32) -> f32 {
    let l0 = clamp(floor(lod), 0.0, 10.0);
    let l1 = min(l0 + 1.0, 10.0);
    let f = clamp(lod - l0, 0.0, 1.0);
    return mix(fif_slope_variance(i32(l0)), fif_slope_variance(i32(l1)), f);
}

fn ocean_normal_mip_sample(world_xy: vec2<f32>, level: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(ocean_normals, i32(level)));
    let coord = (world_xy - u.surface_offset.xy) / u.ocean_params.y
        + 0.5 / dims;
    return textureSampleLevel(ocean_normals, ocean_samp, coord, level).rgb;
}

// The footprint's mean normal at a fractional mip: the two levels blended.
fn ocean_wave_normal_mipped(world_xy: vec2<f32>, lod: f32) -> vec3<f32> {
    let level0 = floor(lod);
    let level1 = min(level0 + 1.0, u.ocean_params.w);
    let f = lod - level0;
    var n = ocean_normal_mip_sample(world_xy, level0);
    if (level1 > level0 && f > 0.0) {
        n = mix(n, ocean_normal_mip_sample(world_xy, level1), f);
    }
    // @domain: every texel of every mip has z >= 0.923 (the tile's bytes), and filtering and the mip mix are convex
    return normalize(n);
}

// Glitter. The glint lobe is the MEAN of what the unresolved facets in a
// footprint reflect, and it is the right mean — but the sun is a disc 0.27
// degrees across, so of the facets under a pixel only a handful, or none,
// tilt to send it to the camera: a pixel is either one brilliant facet or
// dark water, which is what glitter is. So the footprint's glint is drawn as
// K / lambda times the lobe, K ~ Poisson(lambda), lambda the expected number
// of facets that reflect the disc: (footprint / facet area) times the
// probability one does, D(h) (n.h) / (4 v.h) times the sun's solid angle.
// E[K / lambda] = 1, so the mean — and the far field, where lambda is
// large and the draw is 1 to within its own noise — is untouched. The draw
// is anchored to a cell of the sea at the pixel's own scale, in the tile's
// frame, so a parked view keeps its sparkles (accumulation does not average
// them away: they are the surface, not noise).
//
// A capillary facet is ~1 cm (the base roughness is everything below the
// tile's 0.1 m). Cells are that times a power of two, so a whole number of
// them spans the tile and the pattern folds with it.
const SEA_GLITTER_FACET_M: f32 = 0.0125;          // 102.4 m / 2^13
const SEA_SUN_SOLID_ANGLE: f32 = 6.8e-5;          // pi (0.266 deg)^2

// Two uniforms in [0, 1) from a seed: 24 bits each, so they stay below 1 in
// float32 (a full u32 rounds to 1.0 at the top).
fn sea_uniforms(seed: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(seed >> vec2<u32>(8u)) * (1.0 / 16777216.0);
}

// How bright the K facets that catch the sun are, in units of the mean
// facet. A facet's contribution goes as its area and as how squarely it
// holds the disc, and neither is the same from facet to facet: most glints
// are faint and a few are blinding. So each facet is a lognormal draw of
// mean exactly 1 (sigma SEA_GLITTER_SPREAD) and the K are summed, which
// keeps E[sum / lambda] = 1 — the far field untouched — while a lone
// sparkle is no longer always the same clipped dot. Past a few facets the
// sum is its mean, K: the pixel holds enough of them to average.
const SEA_GLITTER_SPREAD: f32 = 1.3;
// Glitter lives in the core of the sun's lobe — around its reflection —
// and nowhere else. Out in the lobe's tails a lone facet catching the sun is
// possible but rare, and drawn as such it strewed the whole sea with
// equally bright dots that read as snow (Thomas, 2026-09-25: "Exactly.
// Snow."). So the draw is blended toward its own mean, the smooth lobe, by
// where the half vector sits in the lobe: GGX D(h) over its peak, 1 on the
// sun's reflection, falling as the facets that would reflect it get rarer.
// And only where a glitter cell expects a few glints: with lambda well
// under one, a drawn cell is almost always empty, and next to smooth sheen
// an empty cell on a sunlit wave face reads as a hole in the water (a dark
// blob round one dot) rather than as glitter; from a few glints up an empty
// cell is rare (e^-4 at lambda 4) and the draw reads as texture. The weight
// reads the count at the footprint's own size, not the cell's: the cell is
// rounded to a power of two, so lambda jumps fourfold where it steps — as
// wide as this ramp — and a ramp on it would draw a contour at a fixed
// footprint, sheen on one side and texture on the other. The draw's mean is 1 (E[K / lambda] = 1, and each facet's
// brightness has mean 1), so any blend keeps the lobe's mean exactly — the
// energy stays the smooth Cox-Munk specular's, with the sparkles only where
// it is bright and the footprint holds enough of them.
const SEA_GLITTER_CORE_LO: f32 = 0.3;
const SEA_GLITTER_CORE_HI: f32 = 0.7;
const SEA_GLITTER_FEW: f32 = 1.0;
const SEA_GLITTER_MANY: f32 = 4.0;

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
        // float32's cdf can stall just below 1; past that, no more mass.
        if (next <= cdf) {
            break;
        }
        cdf = next;
    }
    return k;
}

// The legacy shader's LOD (realism 0): no bias, the one true pixel angle.
fn ocean_normal_lod(t_hit: f32, dir: vec3<f32>) -> f32 {
    let pixel_span = max(t_hit, 0.0) * true_pixel_angle();
    let ocean_span = pixel_span / max(abs(dir.z), 0.03);
    let texel_span = max(ocean_span / u.ocean_params.x, 1.0);
    return clamp(log2(texel_span), 0.0, u.ocean_params.w);
}

// The legacy shader's normal: the tile at one mip.
fn ocean_wave_normal(world_xy: vec2<f32>, lod: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(ocean_normals, 0));
    let coord = (world_xy - u.surface_offset.xy) / u.ocean_params.y
        + 0.5 / dims;
    let n = textureSampleLevel(ocean_normals, ocean_samp, coord, lod).rgb;
    // @domain: every texel of every mip has z >= 0.923 (the tile's bytes), and filtering is convex
    return normalize(n);
}

// Ocean shade ported from witness._ocean_shade (witness.py lines 473-541).
// `sun_shadow` is the penumbra-deflected direction (iter_007) and is used
// only for the cloud-shadow march; every shading term keeps the disc centre.
fn ocean_shade(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
               sun_shadow: vec3<f32>, t_hit: f32,
               jit: f32) -> vec3<f32> {
    var normal_lod = 0.0;
    if (u.cam_up.w > 0.5) {
        normal_lod = ocean_normal_lod(t_hit, dir);
    }
    let n = ocean_wave_normal(hit.xy, normal_lod);

    let vdotn = dot(dir, n);
    var refl = dir - 2.0 * vdotn * n;
    if (refl.z < 0.0) {
        refl.z = -refl.z;
    }
    let sky = ocean_reflection_sky(refl, sun);

    let cos_i = clamp(-vdotn, 0.0, 1.0);
    let one_minus = 1.0 - cos_i;
    let om2 = one_minus * one_minus;
    let fresnel = 0.02 + 0.98 * om2 * om2 * one_minus;

    let tau_ocean = light_march_tau(hit, sun_shadow, t_hit * u.periodic.y,
                                    jit);
    let t_sun_ocean = exp(-tau_ocean);
    let cos_sun_n = max(0.0, dot(sun, n));
    let diff_irr = t_sun_ocean * cos_sun_n * 0.3183098861837907;
    let diffuse = diff_irr * SUN_COLOR * u.ocean.yzw;

    let lit = fresnel * sky + (1.0 - fresnel) * diffuse;
    let t_eff = OCEAN_SHADOW_FLOOR + (1.0 - OCEAN_SHADOW_FLOOR) * t_sun_ocean;
    return lit * t_eff;
}

// ---------------------------------------------------------------------------
// Ocean realism path (witness iter_004/009/011): footprint-filtered normal
// mips, energy-partitioned spectral GGX sun glint, per-term cloud shadowing,
// and sky-field haze along the ocean sightline.
// ---------------------------------------------------------------------------

// The sky the water reflects: the rendered sky (solar disc excluded — the GGX
// glint is the sun's reflection), faded by the master gate toward the legacy
// fixed gradient and its fixed glint lobe (witness._reflection_sky_realism).
// At ocean_realism = 1 the water mirrors the same sky the camera sees, so a
// sunset sea takes the sunset's colour; the legacy gradient was the sky of a
// 55-degree sun and left the water blue under every other.
fn reflection_sky_realism(dir: vec3<f32>, sun: vec3<f32>,
                          legacy_glint_weight: f32) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    t = 1.0 - one_minus * one_minus * one_minus;
    let legacy = SKY_BASE_HORIZON + (SKY_ZENITH - SKY_BASE_HORIZON) * t;
    let rendered = sky_radiance(vec3<f32>(dir.xy, max(dir.z, 0.0)), sun,
                                u.sky_horizon.xyz, u.sky_bloom.xyz,
                                vec3<f32>(0.0), u.cloud_sun.w);
    var col = mix(rendered, legacy, legacy_glint_weight);
    let cos_rs = dot(dir, sun);
    if (cos_rs > 0.0 && legacy_glint_weight > 0.0) {
        let glint_w = 0.02;
        let a = glint_w / ((1.0 - cos_rs) + glint_w);
        col = col + legacy_glint_weight * a * vec3<f32>(1.2, 1.0, 0.6);
    }
    return col;
}

// What the sea reflects of the sky, and lets through to the water below,
// over the sub-pixel slope distribution about the footprint's mean slope:
// Gaussian, per-axis standard deviation `sigma` (the slope of a filtered
// Gaussian-ish wave field is Gaussian). At grazing incidence the reflected
// radiance is strongly convex in slope — Fresnel goes as (1 - n.v)^5 and the
// sky's gradient is steep at the horizon — so the footprint's shading is not
// the shading at its mean slope.
//
// Integrated, not drawn. Until 2026-11-09 this was one random slope per
// pixel per frame (lighting-loop iter_008), exact once accumulated, and the
// sea's speckle in flight and on the first parked frames: at grazing angles
// a slope spread of ~0.07 outruns the ~0.03 grazing angle, so one draw
// reflected the horizon and the next the sky well above it. Four taps at
// +-sqrt(2) sigma along each slope axis, weight 1/4 each, match the
// Gaussian's moments through the third, with no noise to accumulate away:
// a flying frame and a parked one shade the same footprint the same way.
// What no moment-matching rule integrates is the integrand's creases: at a
// few degrees below the horizon a third or more of the footprint's slopes
// face away from the viewer (n.v clamps to 0, Fresnel to 1) or reflect
// below the horizon (folded back up, below). Against the accumulated draw
// the converged sea moves by up to ~3.5 display levels (exposure 4) in the
// band below the horizon, mostly brighter; a degree-5 rule (seven taps)
// closes none of it, which is the creases' signature.
struct SeaReflection {
    // The Fresnel-weighted reflected sky, averaged over the taps.
    sky: vec3<f32>,
    // (1 - Fresnel) * max(n.l, 0) averaged over the taps: the share of the
    // direct sun the water body below receives, per unit beam.
    transmitted_n_dot_l: f32,
}

fn ocean_sky_reflection(dir: vec3<f32>, sun: vec3<f32>, slope: vec2<f32>,
                        sigma: f32,
                        legacy_glint_weight: f32) -> SeaReflection {
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
        let sky = reflection_sky_realism(refl, sun, legacy_glint_weight);
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

// Footprint-filtered ocean with microfacet sun glint and path haze
// (witness._ocean_shade_realism).
fn ocean_shade_realism(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
                       sun_shadow: vec3<f32>, t_hit: f32,
                       jit: f32) -> vec3<f32> {
    let ocean_realism = u.ocean_realism_a.x;
    let ocean_mip_bias = u.ocean_realism_a.y;
    let glint_strength = u.ocean_realism_a.z;
    let glint_roughness = u.ocean_realism_a.w;
    let slope_resolved_fraction = clamp(u.ocean_realism_b.x, 0.0, 1.0);
    let haze_beta0 = u.ocean_realism_b.y;
    let sky_shadow_floor = u.ocean_realism_b.z;

    // Project one pixel's angular span onto the water; the grazing-angle
    // factor stretches it along the view and drives the horizon toward
    // coarser mip levels.
    let grazing = max(abs(dir.z), 0.03);
    let pixel_angular_span = true_pixel_angle();
    let across = t_hit * pixel_angular_span;
    let along = across / grazing;
    let texel_span = max(along / u.ocean_params.x, 1.0);
    var lod = log2(texel_span) + ocean_mip_bias;
    lod = clamp(lod, 0.0, u.ocean_params.w);
    // Scaling LOD with the master gate keeps intermediate gate values a
    // continuous tuning range (witness does the same).
    lod = lod * ocean_realism;

    let h_len = length(dir.xy);
    var h = vec2<f32>(1.0, 0.0);
    if (h_len > 1e-8) {
        h = dir.xy / h_len;
    }
    // The footprint's mean slope from the tile's mip, and the slope
    // variance the mip averaged away.
    let n_mip = ocean_wave_normal_mipped(hit.xy, lod);
    let sea_slope = -n_mip.xy / max(n_mip.z, 1e-4);
    let sea_variance = fif_slope_variance_removed(lod);
    // The glitter's cell: half the footprint, rounded to a power of two of
    // facets and folded at the tile. Smaller than the pixel, so the
    // jittered samples of one pixel see several cells and a sparkle lands
    // as a point, antialiased, not as a cell-shaped tile.
    let fp_along = along * ocean_realism;
    let fp_across = across * ocean_realism;
    // @domain: along and across are footprints: t_hit times a positive pixel span
    let side = max(0.5 * sqrt(fp_along * fp_across), SEA_GLITTER_FACET_M);
    // @domain: side >= SEA_GLITTER_FACET_M > 0 (the max above)
    let level = ceil(log2(side / SEA_GLITTER_FACET_M));
    let cell = SEA_GLITTER_FACET_M * exp2(level);
    let p = hit.xy - u.surface_offset.xy;
    let q = p - u.ocean_params.y * floor(p / u.ocean_params.y);
    let glitter_seed = pcg2d(vec2<u32>(floor(q / cell))
                             + vec2<u32>(u32(level) * 40503u, 0u));
    let glitter_facets = exp2(2.0 * level);
    // The same count before the cell is rounded to a power of two: what the
    // draw's weight reads, so it varies with the footprint smoothly.
    let glitter_count = (side / SEA_GLITTER_FACET_M)
        * (side / SEA_GLITTER_FACET_M);

    // The footprint mean slope is not the footprint's shading. Every judge
    // view's water is sub-pixel: at v7's near field one pixel covers tens
    // of metres of a 0.1 m wave field, so the honest statement is not "this
    // facet is flat" but "this pixel contains a slope distribution whose
    // variance the filters removed". The sky reflection integrates
    // `slope_resolved_fraction` of that variance (per axis, hence the 0.5 —
    // the variances are totals over both axes; ocean_sky_reflection); the
    // sun's glint below carries all of it as microfacet-lobe width.
    let legacy_glint_weight = 1.0 - ocean_realism;
    let reflection = ocean_sky_reflection(
        dir, sun, sea_slope,
        // @domain: a fraction of a variance: >= 0
        sqrt(0.5 * slope_resolved_fraction * sea_variance),
        legacy_glint_weight);

    let tau_ocean = light_march_tau(hit, sun_shadow, t_hit * u.periodic.y,
                                    jit);
    let t_sun_ocean = exp(-tau_ocean);

    // Direct-sun GGX glint, tinted by the spectral beam color, about the
    // footprint's RESOLVED mean normal with all the unresolved variance in
    // the lobe — not about the sky reflection's slope taps. The glitter
    // draw below is already the unresolved facets' own statistics;
    // evaluating it at a random slope as well (the sky's draw, before
    // 2026-11-09) smeared the sparkles evenly over the sea, where real
    // glitter clusters on the wave faces tilted to the sun.
    let n_g = normalize(vec3<f32>(-sea_slope, 1.0));
    let n_g_dot_l = max(0.0, dot(sun, n_g));
    let n_g_dot_v = clamp(-dot(dir, n_g), 0.0, 1.0);
    let glint_weight = ocean_realism * glint_strength;
    var glint = vec3<f32>(0.0);
    var sun_fresnel = 0.02;
    if (glint_weight > 0.0 && n_g_dot_l > 0.0 && n_g_dot_v > 1e-8) {
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

                // Base roughness is what lives BELOW the tile's 0.1 m
                // texels — ripple no tile holds — plus the variance the
                // footprint's mip filtered away. Variances add; roughnesses
                // do not.
                let alpha = clamp(
                    // @domain: a square plus a variance: >= 0
                    sqrt(glint_roughness * glint_roughness + sea_variance),
                    0.02, 0.75
                );
                let alpha_squared = alpha * alpha;
                let denom = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
                let d_ggx = alpha_squared
                    / (3.14159265358979 * denom * denom);
                let g_smith = ggx_smith_g1(n_g_dot_v, alpha_squared)
                    * ggx_smith_g1(n_g_dot_l, alpha_squared);
                let spec = glint_weight * t_sun_ocean * d_ggx * g_smith
                    * sun_fresnel / (4.0 * n_g_dot_v);
                // Glitter (see SEA_GLITTER_*): the lobe drawn as the facets
                // that actually catch the disc, in the lobe's core only.
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
                glint = spec * sparkle * u.cloud_sun.xyz;
            }
        }
    }

    // Direct subsurface light uses the complement of the incident Fresnel
    // allocation so the specular sun path does not create energy.
    let energy_weight = min(glint_weight, 1.0);
    let diffuse_partition = 1.0 - energy_weight * sun_fresnel;
    let diff_irr = t_sun_ocean * reflection.transmitted_n_dot_l
        * 0.3183098861837907 * diffuse_partition;
    let diffuse = diff_irr * SUN_COLOR * u.ocean.yzw;

    // Per-term cloud shadowing (witness iter_011): the sun terms already
    // carry t_sun_ocean; only the sky-reflection term dims, and only to
    // sky_shadow_floor — under a cloud the water still reflects the mostly
    // unblocked sky dome and keeps its wave texture.
    let sky_shadow = sky_shadow_floor + (1.0 - sky_shadow_floor) * t_sun_ocean;
    var ol = reflection.sky * sky_shadow + diffuse + glint;

    // Ocean-only aerial perspective toward the same angular sky field the
    // cloud haze uses (solar disc excluded), killing the horizon seam.
    let haze_tau = ocean_realism * haze_beta0 * t_hit;
    if (haze_tau > 0.0) {
        let haze = 1.0 - exp(-haze_tau);
        let haze_col = sky_radiance(
            vec3<f32>(h, 0.0), sun,
            u.sky_horizon.xyz, u.sky_bloom.xyz, vec3<f32>(0.0),
            u.cloud_sun.w
        );
        ol = mix(ol, haze_col, haze);
    }

    return ol;
}

// Master-gate dispatch: exact-zero realism keeps the untouched legacy ocean
// arithmetic (witness._ocean_shade_dispatch).
fn ocean_shade_dispatch(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
                        sun_shadow: vec3<f32>, t_hit: f32,
                        jit: f32) -> vec3<f32> {
    if (u.ocean_realism_a.x == 0.0) {
        return ocean_shade(hit, dir, sun, sun_shadow, t_hit, jit);
    }
    return ocean_shade_realism(hit, dir, sun, sun_shadow, t_hit, jit);
}
