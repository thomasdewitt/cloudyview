// Low-sun sky color field (witness iter_007). Direction-cosine-space
// constants derived from the witness tuning block:
//   sin(LOW_SUN_SKY_WARM_ELEVATION_DEG = 32),
//   cos(LOW_SUN_SKY_HORIZON_AZIMUTH_DEG = 105),
//   cos(LOW_SUN_SKY_UPPER_AZIMUTH_DEG = 45).
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.0044, 0.035, 0.1156);
const SKY_BASE_HORIZON: vec3<f32> = vec3<f32>(0.10, 0.18, 0.38);
// The haze knob's anchor. Every haze-dependent expression below is written
// as a deviation from its tuned constant, so haze == HAZE_ANCHOR recovers
// that constant EXACTLY rather than to within a ULP — the default look is
// meant to be a fixed point of the whole parameterization.
//
// It very nearly is. What survives is one hardware artifact: the horizon
// wedge divides by a sigma that used to be a literal and is now a value,
// and this GPU's runtime divide is a reciprocal-and-refine that lands ~1
// ULP off the compile-time fold. Measured on the TWPICE still, that moves
// 47 of 172800 samples by at most 0.002 of an 8-bit level and changes zero
// 8-bit codes. Writing the divide as t * (1/sigma) does not help; only
// freezing sigma back into a literal would, which is the feature.
// Twin of look.DEFAULT_HAZE.
const HAZE_ANCHOR: f32 = 0.35;

// Angular scale height (in dir.z) of the horizon-whitening wedge; see
// sky_radiance. Solved against the 2026-08-11 reference photo so mid-sky
// (z ~ 0.6) lands within ~1/255 of the photo on all three channels.
const SKY_HAZE_ELEVATION_SIGMA: f32 = 0.33;
// How far up the sky the whitening reaches as haze grows. Thicker aerosol
// is deeper as well as denser, so the wedge climbs; the bounds stop it
// before it either collapses to a hairline at the skyline (0.15) or washes
// the zenith out, which no photograph of a hazy day does (0.65).
const SKY_HAZE_SIGMA_PER_HAZE: f32 = 0.45;
const SKY_HAZE_SIGMA_MIN: f32 = 0.15;
const SKY_HAZE_SIGMA_MAX: f32 = 0.65;
// Reach alone is not enough: a clear day is not a hazy day with a shorter
// wedge, it is a *weaker* one, so the amplitude moves too. Piecewise linear
// through (0, 0.60), (HAZE_ANCHOR, 1.0), (1, 1.585) — the kink is the
// anchor, and the steeper upper leg is what lets haze 1 read as murk.
const SKY_HAZE_WEDGE_GAIN_AT_ZERO: f32 = 0.60;
const SKY_HAZE_WEDGE_GAIN_SLOPE: f32 = 0.9;

// Broad circumsolar aerosol lobe (sky_radiance). Peak-normalized HG:
// g = 0.68 is ~17 deg half-width; the amplitude is the radiance added at
// the peak. The tint is the aerosol's own mild-Angstrom spectrum; the
// spectral bloom ratio warms it at low sun.
const CIRCUMSOLAR_G: f32 = 0.68;
const CIRCUMSOLAR_AMPLITUDE: f32 = 0.045;
// The lobe is the aerosol seen head-on, so it scales with the same knob.
// A floor of 0.015 remains at haze 0: even Rayleigh air has a faint
// forward brightening around the sun. The 1.2 power is milder than the
// aerial extinction's 1.5 because the lobe saturates — past a certain
// loading the sky near the sun is simply white.
const CIRCUMSOLAR_AMPLITUDE_FLOOR: f32 = 0.015;
const CIRCUMSOLAR_HAZE_EXPONENT: f32 = 1.2;
const CIRCUMSOLAR_TINT: vec3<f32> = vec3<f32>(0.75, 0.92, 1.00);
const LEGACY_BLOOM_CONST: vec3<f32> = vec3<f32>(0.8, 0.6, 0.3);
const LOW_SUN_SKY_MAX_WARM_DZ: f32 = 0.5299192642332049;
const LOW_SUN_SKY_HORIZON_AZIMUTH_COS: f32 = -0.25881904510252074;
const LOW_SUN_SKY_UPPER_AZIMUTH_COS: f32 = 0.7071067811865476;
const LOW_SUN_SKY_NEUTRAL_RADIANCE: vec3<f32> = vec3<f32>(0.27, 0.30, 0.32);
const SUNSET_HORIZON_RADIANCE_R: f32 = 0.42; // SUNSET_HORIZON_RADIANCE[0]

// --- the haze knob's three sky terms ---------------------------------------
//
// All three read u.flags.z rather than taking a parameter: haze is a property
// of the air, not of the ray, and threading it through sky_radiance's five
// existing arguments would only invite a call site that forgets it.

fn sky_haze_elevation_sigma(haze: f32) -> f32 {
    return clamp(
        SKY_HAZE_ELEVATION_SIGMA + SKY_HAZE_SIGMA_PER_HAZE * (haze - HAZE_ANCHOR),
        SKY_HAZE_SIGMA_MIN, SKY_HAZE_SIGMA_MAX);
}

fn sky_haze_wedge_gain(haze: f32) -> f32 {
    if (haze < HAZE_ANCHOR) {
        return SKY_HAZE_WEDGE_GAIN_AT_ZERO
            + (1.0 - SKY_HAZE_WEDGE_GAIN_AT_ZERO) * haze / HAZE_ANCHOR;
    }
    // At the anchor this branch is 1.0 exactly, and a multiply by 1.0 is
    // exact — which is how the default wedge survives untouched.
    return 1.0 + SKY_HAZE_WEDGE_GAIN_SLOPE * (haze - HAZE_ANCHOR);
}

fn circumsolar_amplitude(haze: f32) -> f32 {
    // Deviation form, not the algebraically equal floor + span * pow(h/h0, k):
    // pow(1, k) is exactly 1, so the correction vanishes at the anchor,
    // whereas summing floor and span would land a ULP or two off 0.045.
    let span = CIRCUMSOLAR_AMPLITUDE - CIRCUMSOLAR_AMPLITUDE_FLOOR;
    return CIRCUMSOLAR_AMPLITUDE
        // @domain: haze >= 0: the only caller, sky_radiance, reads max(u.flags.z, 0.0) (docs/soar-bugs.md 21)
        + span * (pow(haze / HAZE_ANCHOR, CIRCUMSOLAR_HAZE_EXPONENT) - 1.0);
}

// Procedural sky ported from witness._sky_radiance, including the spectral
// horizon and low-sun elevation x azimuth warm wedge (iter_002 + iter_007).
// hor/bloom/disc are the per-frame spectral colors from the uniforms; the
// aerial/ocean haze targets pass disc = 0 to exclude the solar disc.
fn sky_radiance(dir: vec3<f32>, sun: vec3<f32>,
                hor: vec3<f32>, bloom: vec3<f32>, disc: vec3<f32>,
                low_sun_sky_field_strength: f32) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    // Horizon whitening = legacy cubic shape times a Gaussian elevation
    // cutoff. The cubic alone still mixed ~6% horizon white at 35-40
    // degrees, which read as a steel-grey wash across the whole upper sky;
    // reference photos hold zenith-deep saturation down to ~35 degrees.
    // The Gaussian kills the whitening by mid-elevation while leaving the
    // horizon band (z < ~0.15) within ~1/255 of the approved legacy sky.
    // Its width is the whitening's angular scale height, which is exactly
    // what an aerosol amount sets — so the haze knob drives both it and the
    // wedge's strength. The ceiling matters at the top of the slider: a gain
    // above 1 would otherwise push the skyline past pure horizon colour and
    // start subtracting zenith blue back out.
    // Clamped at zero: the aerosol coordinate runs NEGATIVE past a 70 km
    // e-folding (the slider's clear end reaches 200 km), and these sky
    // cosmetics are written against a non-negative haze — circumsolar_
    // amplitude takes pow(haze/anchor, 1.4), which is NaN for haze < 0, and
    // one NaN in sky radiance is a black screen. Extinction handles the
    // negative range itself (aerialBetaPerKm is sign-preserving); the sky's
    // wedge and lobe just hold their aerosol-free look from zero down.
    let haze = max(u.flags.z, 0.0);
    let zc = t / sky_haze_elevation_sigma(haze);
    let w_horizon = min(
        one_minus * one_minus * one_minus * exp(-zc * zc)
            * sky_haze_wedge_gain(haze),
        1.0);
    t = 1.0 - w_horizon;

    let base_sky = SKY_BASE_HORIZON + (SKY_ZENITH - SKY_BASE_HORIZON) * t;
    var col = base_sky;

    // Strength-0 spectral lighting and the calibrated 55-degree sun both
    // reproduce the base horizon exactly, so the low-sun angular wedge is
    // skipped. NOTE (2026-08-11): this bypass is no longer a bit-exact
    // legacy contract — the Gaussian horizon cutoff above and the
    // circumsolar lobe below run unconditionally; they are the base look
    // now, pinned by the re-baked goldens rather than by strength-0.
    if (any(hor != SKY_BASE_HORIZON)) {
        let view_h_len = length(dir.xy);
        let sun_h_len = length(sun.xy);
        var cos_sun_az = -1.0;
        if (view_h_len > 1e-12 && sun_h_len > 1e-12) {
            cos_sun_az = clamp(
                dot(dir.xy, sun.xy) / (view_h_len * sun_h_len), -1.0, 1.0
            );
        }

        // Exact iter_006 azimuth-only field (the strength-0 bypass target).
        var legacy_az_weight = 0.5 + 0.5 * cos_sun_az;
        legacy_az_weight = legacy_az_weight * legacy_az_weight
                           * (3.0 - 2.0 * legacy_az_weight);
        let legacy_hor = SKY_BASE_HORIZON
                         + legacy_az_weight * (hor - SKY_BASE_HORIZON);
        let legacy_sky = legacy_hor + (SKY_ZENITH - legacy_hor) * t;

        // Warmth confined vertically; azimuthal support widens toward the
        // horizon where the aerosol slant path is longest.
        let elevation_progress = smoothstep(
            0.0, LOW_SUN_SKY_MAX_WARM_DZ, max(0.0, dir.z)
        );
        let azimuth_cutoff = LOW_SUN_SKY_HORIZON_AZIMUTH_COS
            + elevation_progress * (LOW_SUN_SKY_UPPER_AZIMUTH_COS
                                    - LOW_SUN_SKY_HORIZON_AZIMUTH_COS);
        let azimuth_weight = smoothstep(azimuth_cutoff, 1.0, cos_sun_az);
        let warm_weight = (1.0 - elevation_progress) * azimuth_weight;

        // Recover the horizon's spectral mix so the neutral bridge also
        // vanishes exactly at SPECTRAL_LIGHTING_STRENGTH = 0.
        let sunset_red_span = SUNSET_HORIZON_RADIANCE_R - SKY_BASE_HORIZON.r;
        let horizon_mix = clamp(
            (hor.r - SKY_BASE_HORIZON.r) / sunset_red_span, 0.0, 1.0
        );
        let neutral_hor = SKY_BASE_HORIZON + horizon_mix
            * (LOW_SUN_SKY_NEUTRAL_RADIANCE - SKY_BASE_HORIZON);
        let neutral_sky = neutral_hor + (SKY_ZENITH - neutral_hor) * t;
        let warm_sky = hor + (SKY_ZENITH - hor) * t;

        // Quadratic Bezier in linear radiance: blue -> neutral -> warm,
        // avoiding the purple midpoint of a straight blue/orange lerp.
        let cool_weight = 1.0 - warm_weight;
        let wedge_sky = cool_weight * cool_weight * base_sky
            + 2.0 * cool_weight * warm_weight * neutral_sky
            + warm_weight * warm_weight * warm_sky;

        col = legacy_sky + low_sun_sky_field_strength * (wedge_sky - legacy_sky);
    }

    let cos_sun = dot(dir, sun);

    // Broad aerosol forward lobe: the faint brightening-and-desaturation
    // the sky shows for tens of degrees around the sun, which the tight
    // Lorentzian bloom (half-width ~3.6 deg) cannot supply. Peak-normalized
    // Henyey-Greenstein, no angular cutoff and no elevation gate — the
    // sunward hue change is a permanent feature of a hazy sky, at noon as
    // at dusk. The tint is cool (mild Angstrom spectrum) scaled by the
    // spectral bloom's warm ratio, so at noon it desaturates toward white
    // and at low sun it inherits the beam's warmth.
    let g_a = CIRCUMSOLAR_G;
    let d_hg = 1.0 + g_a * g_a - 2.0 * g_a * cos_sun;
    let one_minus_g = 1.0 - g_a;
    let lobe = (one_minus_g * one_minus_g * one_minus_g)
        // @domain: d = 1 + g^2 - 2 g cos >= (1 - |g|)^2 >= 0 for |cos| <= 1 (Henyey-Greenstein)
        / max(pow(d_hg, 1.5), 1e-6);
    let warm_ratio = bloom / LEGACY_BLOOM_CONST;
    col = col + circumsolar_amplitude(haze) * lobe
        * CIRCUMSOLAR_TINT * warm_ratio;

    if (cos_sun > 0.0) {
        let sun_half_width = 0.002;
        let a = sun_half_width / ((1.0 - cos_sun) + sun_half_width);
        col = col + a * bloom;
    }
    if (cos_sun > 0.9998) {
        col = col + disc;
    }
    return col;
}
