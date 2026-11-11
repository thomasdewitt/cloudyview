// The daytime sky: radiance as a function of direction. A zenith-to-horizon
// gradient, a horizon whitening that the haze amount controls, a warm field
// around a low sun, a broad circumsolar aerosol lobe, a tight bloom and the
// solar disc. It is a look model, tuned by eye against photographs, in
// linear RGB.

// Zenith and horizon colours of the base gradient. SKY_BASE_HORIZON is also
// the horizon colour the host packs at high sun, and sky_radiance compares
// u.sky_horizon against it.
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.0044, 0.035, 0.1156);
const SKY_BASE_HORIZON: vec3<f32> = vec3<f32>(0.10, 0.18, 0.38);
// The haze amount at which every haze-dependent expression below equals its
// tuned constant. Each is written as a deviation from that constant, so that
// haze == HAZE_ANCHOR recovers the constant exactly and not to within
// rounding. It is the hosts' default haze, and they hold the same value.
const HAZE_ANCHOR: f32 = 0.35;

// Scale height, in dir.z, of the Gaussian that confines the horizon
// whitening to low elevation (see sky_radiance), at the anchor haze.
const SKY_HAZE_ELEVATION_SIGMA: f32 = 0.33;
// How far up the sky the whitening reaches as haze grows: thicker aerosol is
// deeper as well as denser. The bounds keep the wedge from collapsing to a
// line at the skyline or from washing out the zenith.
const SKY_HAZE_SIGMA_PER_HAZE: f32 = 0.45;
const SKY_HAZE_SIGMA_MIN: f32 = 0.15;
const SKY_HAZE_SIGMA_MAX: f32 = 0.65;
// The whitening's amplitude moves with haze too: a clear day's wedge is
// weaker, not only shorter. Piecewise linear through (0, 0.60),
// (HAZE_ANCHOR, 1.0), (1, 1.585), with the kink at the anchor.
const SKY_HAZE_WEDGE_GAIN_AT_ZERO: f32 = 0.60;
const SKY_HAZE_WEDGE_GAIN_SLOPE: f32 = 0.9;

// Broad circumsolar aerosol lobe (sky_radiance): a peak-normalized HG lobe,
// (1 - g)^3 / (1 + g^2 - 2 g mu)^(3/2). g = 0.68 gives a half-width of ~17
// deg; the amplitude is the radiance added at the peak, at the anchor haze.
const CIRCUMSOLAR_G: f32 = 0.68;
const CIRCUMSOLAR_AMPLITUDE: f32 = 0.045;
// The lobe is the aerosol seen head-on, so its amplitude scales with haze:
// FLOOR at haze 0 (even Rayleigh air brightens a little toward the sun),
// rising to AMPLITUDE at the anchor as (haze / anchor)^EXPONENT.
const CIRCUMSOLAR_AMPLITUDE_FLOOR: f32 = 0.015;
const CIRCUMSOLAR_HAZE_EXPONENT: f32 = 1.2;
// The lobe's colour: the aerosol's own, slightly cool, times u.sky_bloom
// over the bloom colour at high sun (LEGACY_BLOOM_CONST), so the lobe warms
// with the beam at low sun.
const CIRCUMSOLAR_TINT: vec3<f32> = vec3<f32>(0.75, 0.92, 1.00);
const LEGACY_BLOOM_CONST: vec3<f32> = vec3<f32>(0.8, 0.6, 0.3);
// The low-sun warm field, in direction cosines.
// MAX_WARM_DZ = sin(32 deg): the elevation above which no warmth is added.
// HORIZON_AZIMUTH_COS = cos(105 deg), UPPER_AZIMUTH_COS = cos(45 deg): the
// azimuth from the sun at which the warmth ends, at the horizon and at the
// top of the field.
const LOW_SUN_SKY_MAX_WARM_DZ: f32 = 0.5299192642332049;
const LOW_SUN_SKY_HORIZON_AZIMUTH_COS: f32 = -0.25881904510252074;
const LOW_SUN_SKY_UPPER_AZIMUTH_COS: f32 = 0.7071067811865476;
// Horizon colour of the neutral stage between the blue sky and the warm one.
const LOW_SUN_SKY_NEUTRAL_RADIANCE: vec3<f32> = vec3<f32>(0.27, 0.30, 0.32);
// Red channel of the host's sunset horizon colour: the far end of the range
// over which u.sky_horizon.r moves away from SKY_BASE_HORIZON.r.
const SUNSET_HORIZON_RADIANCE_R: f32 = 0.42;

// --- the haze amount's three sky terms --------------------------------------
//
// Each takes the haze amount, which the caller has clamped at 0.

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
    // Exactly 1.0 at the anchor.
    return 1.0 + SKY_HAZE_WEDGE_GAIN_SLOPE * (haze - HAZE_ANCHOR);
}

fn circumsolar_amplitude(haze: f32) -> f32 {
    // Written as a deviation from the anchor value, not as the algebraically
    // equal floor + span * pow(h / h0, k): pow(1, k) is exactly 1, so the
    // correction vanishes at the anchor.
    let span = CIRCUMSOLAR_AMPLITUDE - CIRCUMSOLAR_AMPLITUDE_FLOOR;
    return CIRCUMSOLAR_AMPLITUDE
        // @domain: haze >= 0: the only caller, sky_radiance, reads max(u.flags.z, 0.0) (docs/soar-bugs.md 21)
        + span * (pow(haze / HAZE_ANCHOR, CIRCUMSOLAR_HAZE_EXPONENT) - 1.0);
}

// Sky radiance toward `dir`. hor, bloom and disc are the per-frame colours
// from the uniforms. A caller that wants the sky as a haze colour or as a
// reflection passes disc = 0 to leave the solar disc out.
fn sky_radiance(dir: vec3<f32>, sun: vec3<f32>,
                hor: vec3<f32>, bloom: vec3<f32>, disc: vec3<f32>,
                low_sun_sky_field_strength: f32) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    // Horizon whitening. The weight of the horizon colour is
    //   (1 - z)^3 * exp(-(z / sigma)^2) * gain,  z = dir.z.
    // The cubic alone still mixes several percent of horizon white at 35-40
    // deg elevation, a grey wash over the upper sky; the Gaussian confines
    // it to low elevation. Its width is the angular scale height of the
    // whitening, which is what an aerosol amount sets, so haze drives both
    // sigma and gain. The min() matters at high haze: a gain above 1 would
    // push the skyline past the pure horizon colour.
    //
    // Haze is clamped at zero. It is negative at the clear end of its range,
    // and circumsolar_amplitude raises haze / anchor to a power, which is
    // NaN for a negative base. From zero down the sky holds its aerosol-free
    // look; the host's extinction handles the negative range itself.
    let haze = max(u.flags.z, 0.0);
    let zc = t / sky_haze_elevation_sigma(haze);
    let w_horizon = min(
        one_minus * one_minus * one_minus * exp(-zc * zc)
            * sky_haze_wedge_gain(haze),
        1.0);
    t = 1.0 - w_horizon;

    let base_sky = SKY_BASE_HORIZON + (SKY_ZENITH - SKY_BASE_HORIZON) * t;
    var col = base_sky;

    // The warm field is skipped while the horizon colour is the base one
    // (a high sun), where it would reproduce the base sky.
    if (any(hor != SKY_BASE_HORIZON)) {
        let view_h_len = length(dir.xy);
        let sun_h_len = length(sun.xy);
        var cos_sun_az = -1.0;
        if (view_h_len > 1e-12 && sun_h_len > 1e-12) {
            cos_sun_az = clamp(
                dot(dir.xy, sun.xy) / (view_h_len * sun_h_len), -1.0, 1.0
            );
        }

        // The azimuth-only field, which low_sun_sky_field_strength = 0
        // selects: the horizon colour goes from base to `hor` by a
        // smoothstep of the azimuth from the sun.
        var legacy_az_weight = 0.5 + 0.5 * cos_sun_az;
        legacy_az_weight = legacy_az_weight * legacy_az_weight
                           * (3.0 - 2.0 * legacy_az_weight);
        let legacy_hor = SKY_BASE_HORIZON
                         + legacy_az_weight * (hor - SKY_BASE_HORIZON);
        let legacy_sky = legacy_hor + (SKY_ZENITH - legacy_hor) * t;

        // The warm field is confined in elevation, and its azimuthal extent
        // widens toward the horizon, where the slant path through the
        // aerosol is longest.
        let elevation_progress = smoothstep(
            0.0, LOW_SUN_SKY_MAX_WARM_DZ, max(0.0, dir.z)
        );
        let azimuth_cutoff = LOW_SUN_SKY_HORIZON_AZIMUTH_COS
            + elevation_progress * (LOW_SUN_SKY_UPPER_AZIMUTH_COS
                                    - LOW_SUN_SKY_HORIZON_AZIMUTH_COS);
        let azimuth_weight = smoothstep(azimuth_cutoff, 1.0, cos_sun_az);
        let warm_weight = (1.0 - elevation_progress) * azimuth_weight;

        // How far the horizon colour has moved from base toward sunset, read
        // off its red channel, so the neutral stage scales with it and
        // vanishes when `hor` is the base colour.
        let sunset_red_span = SUNSET_HORIZON_RADIANCE_R - SKY_BASE_HORIZON.r;
        let horizon_mix = clamp(
            (hor.r - SKY_BASE_HORIZON.r) / sunset_red_span, 0.0, 1.0
        );
        let neutral_hor = SKY_BASE_HORIZON + horizon_mix
            * (LOW_SUN_SKY_NEUTRAL_RADIANCE - SKY_BASE_HORIZON);
        let neutral_sky = neutral_hor + (SKY_ZENITH - neutral_hor) * t;
        let warm_sky = hor + (SKY_ZENITH - hor) * t;

        // Quadratic Bezier in linear radiance, blue -> neutral -> warm,
        // which avoids the purple midpoint of a straight blue-to-orange mix.
        let cool_weight = 1.0 - warm_weight;
        let wedge_sky = cool_weight * cool_weight * base_sky
            + 2.0 * cool_weight * warm_weight * neutral_sky
            + warm_weight * warm_weight * warm_sky;

        col = legacy_sky + low_sun_sky_field_strength * (wedge_sky - legacy_sky);
    }

    let cos_sun = dot(dir, sun);

    // Broad aerosol forward lobe: the faint brightening and desaturation of
    // the sky for tens of degrees around the sun, which the tight bloom
    // below cannot supply. It has no angular cutoff and no elevation gate.
    // Its tint is cool, times the bloom's warm ratio, so at noon it
    // desaturates toward white and at low sun it takes the beam's warmth.
    let g_a = CIRCUMSOLAR_G;
    let d_hg = 1.0 + g_a * g_a - 2.0 * g_a * cos_sun;
    let one_minus_g = 1.0 - g_a;
    let lobe = (one_minus_g * one_minus_g * one_minus_g)
        // @domain: d = 1 + g^2 - 2 g cos >= (1 - |g|)^2 >= 0 for |cos| <= 1 (Henyey-Greenstein)
        / max(pow(d_hg, 1.5), 1e-6);
    let warm_ratio = bloom / LEGACY_BLOOM_CONST;
    col = col + circumsolar_amplitude(haze) * lobe
        * CIRCUMSOLAR_TINT * warm_ratio;

    // Tight bloom: a Lorentzian in 1 - cos, at half its peak where
    // 1 - cos = 0.002 (3.6 deg from the sun).
    if (cos_sun > 0.0) {
        let sun_half_width = 0.002;
        let a = sun_half_width / ((1.0 - cos_sun) + sun_half_width);
        col = col + a * bloom;
    }
    // The solar disc: cos > 0.9998 is a radius of 1.15 deg.
    if (cos_sun > 0.9998) {
        col = col + disc;
    }
    return col;
}
