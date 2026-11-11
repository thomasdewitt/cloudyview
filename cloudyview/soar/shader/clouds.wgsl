// Cloud lighting: the constants of the lighting model, the marches that feed
// it (toward the sun, straight up, ahead along the sightline) and their
// helpers. fs_main (march.wgsl) puts the terms together.
//
// The model is single scattering along the view ray with a marched sun
// transmittance, plus a stack of approximations for multiply scattered
// light. Per view sample the source terms are:
//   - the octaves: the beam scattered once (octave 0) and, as successively
//     more transmitted and more isotropic copies of it, many times;
//   - the diffraction spike, added to octave 0 near the sun;
//   - the diffused beam: the diffusion tail of the sun through thick cloud;
//   - the ambient (skylight) fill and two shadow-skylight fills;
//   - the surface bounce (by day) or the city's uplight (by night).
// Most of the constants are tuned by eye against photographs. Where a value
// is derived, the derivation is given.

// The strengths of the model's terms.
const CLOUD_G: f32 = 0.76;                 // HG asymmetry of the cloud phase function
const AMBIENT_STRENGTH: f32 = 0.15;        // the ambient (skylight) fill
const GRADIENT_SHADING_STRENGTH: f32 = 1.50;
const GRADIENT_SHADING_COARSE_WEIGHT: f32 = 0.65;   // weight of the coarse stencil
const CONE_STENCIL_TAN: f32 = 0.03492076949174773;  // tan(2 deg): the coarse stencil's angle
const DEEP_SHADOW_MS_SUPPRESSION: f32 = 0.90;
const AMBIENT_OCCLUSION_FLOOR: f32 = 0.24; // the ambient fill a fully buried sample keeps
const BOUNCE_DEPTH_ATTENUATION: f32 = 0.80; // per unit optical depth into cloud

// Diffraction spike on the once-scattered beam.
//
// One HG lobe at g ~ 0.76 has the shape of a cloud's side and back
// scattering, but its forward peak is far too weak: (1 + g) / (1 - g)^2 = 31
// times isotropic, where a 10 um droplet puts about half its scattered
// energy into a diffraction lobe a couple of degrees wide (lambda / d ~ 3
// deg). Without that peak a thin veil in front of the sun is no brighter
// than a sunlit thick cloud.
//
// The spike is a second, narrow HG lobe (g = 0.95: half-width ~2 deg, peak
// 780 times isotropic) with weight DIFFRACTION_WEIGHT, the diffraction share
// of the scattered energy. It is added to the once-scattered term only and
// takes octave 0's transmittance exp(-tau_sun): a second scattering
// convolves the lobe with the whole phase function, and that light belongs
// to the octaves. It is added on top of the main lobe, not split from it
// (the main lobe is not scaled by 1 - w), so side and back scattering are
// what they are without it and the phase function is not normalized.
//
// The lobe is windowed to zero between 15 and 30 deg from the sun. An HG
// tail is not compact (at 90 deg the weighted HG(0.95) is still ~9% of
// HG(0.76)), while diffraction has no such tail, so the window is closer to
// the physics than the HG it trims.
const DIFFRACTION_G: f32 = 0.95;
const DIFFRACTION_WEIGHT: f32 = 0.5;
const DIFFRACTION_WINDOW_COS_START: f32 = 0.8660254037844387; // cos 30 deg
const DIFFRACTION_WINDOW_COS_FULL: f32 = 0.9659258262890683;  // cos 15 deg

// Powder: the darkening of a cloud's sunlit skin seen with the sun behind
// the viewer. A thin edge has not built up the multiple scattering that
// sends light back, so it reads darker than the interior. The factor is
//   1 - w exp(-POWDER_COEFF tau_depth),
// tau_depth the optical depth since the ray entered the cloud. It depends on
// that depth and not on the step size, so two levels of different resolution
// composite without a seam. Toward the sun the same edge is lit in
// transmission, so w fades from 1 to 1 - POWDER_FWD_FADE over the forward
// cone. The cone starts where the HG lobe rises above isotropic, i.e. where
// a photon crossing a thin edge is more likely to continue toward the camera
// than a random walk would send it. Solving HG(mu, g) = 1 / 4 pi:
//   mu_cross = (1 + g^2 - (1 - g^2)^(2/3)) / (2 g),
// 0.667 at g = 0.76 and 0.762 at g = 0.85.
const POWDER_COEFF: f32 = 1.5;
const POWDER_FWD_FADE: f32 = 0.85;       // fraction of powder removed at mu = 1

// The ambient fill ramps linearly with height in the box, from this fraction
// at the bottom to 1 at the top.
const AMBIENT_HEIGHT_FLOOR: f32 = 0.3;

// Surface bounce: sunlight reflected off the surface up into the cloud. Its
// strength is BOUNCE_STRENGTH at z = 0 and falls linearly to zero at the
// box top.
const BOUNCE_STRENGTH: f32 = 0.05;
const BOUNCE_TINT: vec3<f32> = vec3<f32>(1.0, 0.97, 0.92);

// Under a column thick enough to shadow the ground the bounce is scaled by
// the column's two-stream transmittance, but not below this floor: the
// ground a base sees is a hemisphere, and the sunlit part of it beyond the
// cloud's own shadow still lights the base.
const BOUNCE_LATERAL_FLOOR: f32 = 0.25;

// Gradient shading. The density gradient stands in for a surface normal n,
// and the beam terms are scaled by 1 + strength * gate * (n . sun).
// RADIUS_VOXELS is the fine stencil's radius (sigma_gradient).
const GRADIENT_SHADING_RADIUS_VOXELS: f32 = 1.0;

// The gate is a product of two smoothsteps. One is in tau_depth: a ray that
// has just entered a cloud has no resolved skin yet, and a wisp never gets
// one. The other is in the gradient's confidence (sigma_gradient).
const GRADIENT_SHADING_TAU_START: f32 = 0.25;
const GRADIENT_SHADING_TAU_FULL: f32 = 1.60;
const GRADIENT_SHADING_CONF_START: f32 = 0.06;
const GRADIENT_SHADING_CONF_FULL: f32 = 0.28;

// A face turned away from the sun (n . sun < 0) darkens by this much less
// than a face turned toward it brightens.
const GRADIENT_SHADING_SHADOW_SIDE_SCALE: f32 = 0.55;

// The deep-shadow gate: a smoothstep in ln(tau_sun) from tau_sun = 15 to
// 1000. It marks the samples the beam no longer reaches, whose light is all
// diffuse, and switches on the terms that treat them: suppression of the
// isotropic octaves, ambient occlusion, the forward pre-march, the buried
// fill tint. It runs in log tau because a linear smoothstep over 15..1000
// would spend nearly all of its travel above tau = 200; the darkest bases
// belong to the thickest columns, and a storm core reaches several hundred.
const DEEP_SHADOW_LOG_TAU_START: f32 = 2.70805020110221;  // ln(15)
const DEEP_SHADOW_LOG_TAU_FULL: f32 = 6.907755278982137;  // ln(1000)

// Shallow or storm. A fair-weather cumulus base and a storm base can both
// saturate tau_sun, but the first sits under a few hundred metres of cloud
// and the second under kilometres, and they do not look alike. The sky probe
// tells them apart: shallow_open = smoothstep(START, FULL, t_sky) is 1 for a
// shadowed sample with open sky close overhead and 0 for a buried one. It
// fades the deep-shadow treatment out of open shadow: less suppression and
// occlusion, a bluer fill, more skylight.
const SHALLOW_OPEN_TSKY_START: f32 = 0.05;
const SHALLOW_OPEN_TSKY_FULL: f32 = 0.30;

// Fraction of the octave suppression and the ambient occlusion that an open
// shallow shadow keeps. A buried sample keeps all of it.
const SHALLOW_SUPPRESSION_KEEP: f32 = 0.25;

// High-sun shadow skylight: skylight reaching shadowed cloud, as a
// coefficient on the ambient tint, for a buried (STORM) and an open
// (SHALLOW) sample. It is weighted by 1 - the light-transfer split strength,
// so it is the high-sun counterpart of LIGHT_TRANSFER_SHADOW_SKYLIGHT.
const HIGH_SUN_SHADOW_SKYLIGHT_STORM: f32 = 0.03;
const HIGH_SUN_SHADOW_SKYLIGHT_SHALLOW: f32 = 0.24;

// That fill comes in as 1 - exp(-tau_sun / ONSET): on moderate shadow, well
// before the deep gate. The beam is spent by tau_sun ~ 5 and the octaves by
// ~ 20, so without it a fair-weather base at tau_sun 15-40 is dark. The
// onset is an exponential and not a smoothstep window because a gate that
// opens after the beam has died makes brightness non-monotonic in tau; this
// one rises while the beam and the octaves are still decaying, so their sum
// only decreases.
const SHADOW_SKYLIGHT_TAU_ONSET: f32 = 9.0;

// Diffused beam: the diffusion tail of the sun through the cloud mass,
// emitted isotropically. The octaves decay geometrically in tau_sun and are
// spent by ~20, but light diffusing through a scattering slab decays only as
// T_d = 1 / (1 + 0.75 (1 - g) tau_sun) (diffuse_transmittance), which is why
// a thick fair-weather base is translucent and often brighter than the sky
// behind it. The term is
//   STRENGTH * T_d(tau_sun) * (1 - exp(-tau_sun / TAU_ONSET))
//       * ISO_PHASE * DIFFUSE_BEAM_TINT * u.cloud_sun.
// The onset keeps it out of the range the octaves already cover. The "out"
// path is the tau_sun the sun march has measured anyway; the return is this
// closed form.
const DIFFUSE_BEAM_STRENGTH: f32 = 0.75;
const DIFFUSE_BEAM_TAU_ONSET: f32 = 7.0;

// The fraction of the diffused beam a buried storm interior keeps (deep gate
// open, sky closed overhead). Tuned by eye against a photograph of an
// anvil's underside.
const DIFFUSE_BEAM_STORM_KEEP: f32 = 0.80;

// Weight of the gradient shading on the diffuse fills (ambient, skylight,
// diffused beam, uplight). At 0 the fills ignore orientation and shadowed
// cloud is flat; at 1 they shade as hard as the beam does. Diffuse light is
// directional enough to shade billows, only more softly.
const FILL_GRADIENT_WEIGHT: f32 = 0.6;

// Colour of the diffused beam. Many scatterings put a long absorption path
// in liquid water on this light, and water absorbs red more than blue
// (single-scattering albedo ~0.998 at 680 nm, ~0.9999 at 460 nm), so deeply
// diffused daylight is bluish.
const DIFFUSE_BEAM_TINT: vec3<f32> = vec3<f32>(0.85, 0.97, 1.08);

// The ambient height ramp's floor for open shallow cloud, in place of
// AMBIENT_HEIGHT_FLOOR: an open base sees the sky sideways and the bright
// surface below.
const AMBIENT_HEIGHT_FLOOR_SHALLOW: f32 = 0.6;

// The deep-shadow suppression never takes an octave below this fraction.
const DEEP_SHADOW_MS_FLOOR: f32 = 0.28;

// The multiple-scattering octaves. Octave k stands for light scattered about
// k more times than octave 0. Its weight and its optical depth to the sun
// are both scaled by a^k, a = MS_ATTEN (more scatterings carry less energy
// and find longer ways round), and its phase function is blended toward
// isotropic by b_k = min(1, k MS_BLEND_RATE):
//   sum_k a^k exp(-a^k tau_sun) ((1 - b_k) HG(mu) + b_k / 4 pi).
// Octave 0 is single scattering.
const MS_OCTAVES: i32 = 6;
const MS_ATTEN: f32 = 0.4;
const MS_BLEND_RATE: f32 = 0.35;

// Step caps of the view march and the sun march in a bounded domain.
const MAX_VIEW_STEPS: i32 = 2048;
const MAX_LIGHT_STEPS: i32 = 512;

// The view march ends when the transmittance left falls below this.
const TRANSMITTANCE_CUTOFF: f32 = 0.002;

// The sun march ends at this optical depth. It runs that far because the
// log-space deep gate and the two-stream terms still discriminate up there.
const LIGHT_TAU_CUTOFF: f32 = 1000.0;

// Past tau = LIGHT_DEEP_QUAD_TAU_START the sun march's step grows by GROWTH
// per step, up to MAX_BOOST times its base length, which is what makes
// measuring to tau = 1000 affordable. Beyond the threshold tau is consumed
// only by the two-stream terms and the log-space gate, which are
// logarithmically sensitive. The exponential consumers are spent by then:
// the beam by tau ~ 8 and octave 2 by ~ 30, and octaves 3-5 have weights of
// 0.064 and less.
const LIGHT_DEEP_QUAD_TAU_START: f32 = 32.0;
const LIGHT_DEEP_QUAD_GROWTH: f32 = 1.2;
const LIGHT_DEEP_QUAD_MAX_BOOST: f32 = 64.0;

// A view step of optical depth below this is clear air: nothing is shaded,
// and tau_depth, the optical depth since the ray entered cloud, resets.
const EMPTY_DTAU_CUTOFF: f32 = 1e-5;

// The view step shortens in dense cloud so that no step exceeds this optical
// depth (step_dt_for_sigma).
const TAU_STEP_MAX: f32 = 0.5;
const ISO_PHASE: f32 = 0.07957747154594767; // 1 / (4*pi)

// Sky probe: how much cloud is immediately overhead.
//
// Once the sun march saturates, tau_sun says nothing more about a sample.
// What still distinguishes deep samples is the skylight that reaches them:
// sky radiance diffusely transmitted through the cloud above. A short march
// straight up measures tau_up, and the two-stream diffuse transmittance of a
// conservatively scattering slab, in similarity-scaled optical depth, turns
// it into a visibility:
//
//     T_sky = 1 / (1 + 0.75 * (1 - g) * tau_up)
//
// The (1 - g) scaling is what makes this usable. For g ~ 0.8 it still
// discriminates at tau_up ~ 40, where exp(-tau_up) is zero by tau_up ~ 5, so
// a base under 300 m of cloud and a base under 3 km are different pixels.
//
// The probe covers SKY_PROBE_SPAN and not the whole column. Skylight also
// enters through the side of a wall, which a vertical ray does not see, so a
// whole-column integral over-darkens vertical faces.
const SKY_PROBE_SPAN: f32 = 1500.0;   // m of headroom sampled above p
const SKY_PROBE_STEPS: i32 = 15;      // 100 m quadrature at full span
// The fraction of the shadow skylight that a sample keeps at T_sky = 0, so
// that saturated shadow does not go black.
const SKY_PROBE_FILL_FLOOR: f32 = 0.40;

// Diffusion depth of the isotropic octaves.
//
// Octave k is lit by exp(-MS_ATTEN^k tau_sun). For k = 4 and 5 that is
// exp(-0.026 tau_sun) and exp(-0.010 tau_sun): nearly constant over any
// tau_sun a cloud has. Left alone, the high octaves are a pedestal with the
// same value on a turret's top and in the crevice beside it, though the two
// differ by tens of optical depths to the sun.
//
// Those octaves stand for light that diffused in from the lit part of the
// cloud, so they are scaled by the two-stream transmittance along the sun
// path, in proportion to how isotropic the octave is:
//
//     MS_TAIL_FLOOR + (1 - MS_TAIL_FLOOR) * T_d(max(tau_sun - KNEE, 0)),
//     T_d(tau) = 1 / (1 + 0.75 * (1 - g) * tau)
//
// T_d still discriminates at tau_sun = 40, where exp(-tau_sun) has long
// collapsed and exp(-MS_ATTEN^k tau_sun) has not begun to. The floor is the
// fully buried limit: a deep core keeps a tail and does not go black.
const MS_TAIL_FLOOR: f32 = 0.18;
// Below this sun optical depth the factor is exactly 1. A sample the sun
// reaches nearly unattenuated is not at the bottom of a diffusion well, and
// scaling it would only dim thin cloud.
const MS_TAIL_TAU_KNEE: f32 = 1.0;

// Incidence of the beam on the local skin.
//
// The factor above gives the isotropic octaves a depth but no direction. On
// directly sunlit skin tau_sun is below the knee, so every lit sample gets
// them at full strength whichever way its surface faces. Their source is the
// irradiance the beam delivers locally, and for a collimated beam on a
// tilted skin element that goes as the incidence cosine mu0 = n . sun. So
// they are also scaled by
//
//     TAIL_MU_FLOOR + (1 - TAIL_MU_FLOOR) * clamp(mu0 / mu_ref, 0, 1),
//     mu_ref = max(sun.z, TAIL_MU_REF_MIN)
//
// mu_ref is what a horizontal face receives, so level cloud top is unchanged
// and only skin tilted away from the sun gives anything up. The factor
// applies where a skin is resolved (surface_gate) and fades out with the
// deep-shadow gate: where tau_sun has saturated there is no local beam left
// for the orientation to matter to.
const TAIL_MU_FLOOR: f32 = 0.30;
// Guard for a sun near the horizon, where sun.z -> 0 and the ratio would
// saturate at 1 for every orientation. That is the right limit (a grazing
// beam lights nothing preferentially) but a numerically noisy way to reach
// it.
const TAIL_MU_REF_MIN: f32 = 0.25;
// The sun march's quadrature phase is randomized only where the distance LOD
// has coarsened its step: not at all while the LOD floor is at or below the
// march's own step, fully once the floor is this many times that step. See
// light_march_tau.
const LIGHT_JITTER_LOD_FULL: f32 = 2.0;

// Solar penumbra.
//
// The sun subtends 0.53 deg, so a shadow's edge is a penumbra whose width
// grows linearly with the distance from caster to receiver, about 9 m per
// km. Each shadow ray is deflected to a point drawn uniformly on the disc:
// one draw per pixel per frame, advanced along the view march by an additive
// recurrence so that one view ray's shadow rays cover the disc too.
// Averaging the radiance over frames gives <exp(-tau)> over the disc, which
// is the penumbra. Averaging tau first would not, since exp(-<tau>) is not
// <exp(-tau)>.
//
// Only the shadow ray is deflected. The phase function, the gradient
// shading, the sea's glint and the sky keep the disc centre: a quarter of a
// degree changes none of them visibly, while occlusion is a discontinuous
// function of direction. With jitter off the cone has zero width and the sun
// is a point.
const SUN_ANGULAR_RADIUS: f32 = 0.0046542; // tan(0.2666 deg)
// A factor on the disc's radius. 1 is the true sun; more blurs every shadow
// in proportion to its throw.
const SUN_CONE_WIDEN: f32 = 1.0;

// The forward-scattering skirt.
//
// A real cloud shadow's edge is two nested penumbrae: a sharp one from the
// disc and a wide, faint foot. The foot is sunlight that clipped the
// caster's edge and was deflected a couple of degrees by diffraction (a
// 10 um droplet's forward lobe is about lambda / d ~ 3 deg wide), continuing
// almost undiminished into the geometric shadow. A shadow march counts every
// scattering as extinction and cannot produce that light, so it is put into
// the source: AUREOLE_FRACTION of the draws come from a cone AUREOLE_WIDEN
// times the disc's radius. The rest still land on the disc, so the sharp
// edge stays where it is.
const AUREOLE_FRACTION: f32 = 0.15;
const AUREOLE_WIDEN: f32 = 8.0;   // 8 x 0.267 deg = 2.1 deg
// Numerical guard: the deflected direction stays above the horizon, because
// the periodic light march leaves through the domain top and divides by
// sun.z. The hosts require a sun above the horizon in a periodic domain.
const SUN_CONE_MIN_Z: f32 = 0.02;

// Forward pre-march: how much cloud lies beyond the sample.
//
// Everything else the march knows about a sample lies behind it or to one
// side: the transmittance already crossed, the depth since entry, the path
// to the sun, the column overhead. Compositing ends within a few optical
// depths, so the image of a cloud is a thin skin, and nothing tells that
// skin whether it belongs to a wisp or to an anvil. The diffuse light that
// reaches it differs: a wisp's skin is within a few mean free paths of open
// air on both sides, an anvil's is open only toward the camera.
//
// One coarse march along the view ray, before compositing, gives tau_total.
// At a sample tau_ahead = tau_total - tau_view, and the diffuse fills are
// scaled by
//
//     AHEAD_FLOOR + (1 - AHEAD_FLOOR) * T_d(tau_ahead),
//
// weighted by AHEAD_WEIGHT, by the deep-shadow gate, and by the sine of the
// angle between the sightline and the vertical (a sightline straight up
// repeats what the sky probe measured). Because the direction is the
// sightline, the result is registered with the silhouette: a brighter fringe
// along every edge and gap, and a body that darkens toward a floor.
const AHEAD_STEP_SCALE: f32 = 4.0;   // pre-march dt = this x the view dt
const AHEAD_MAX_STEPS: i32 = 192;
const AHEAD_TAU_CAP: f32 = 60.0;     // the pre-march stops at this depth
const AHEAD_FLOOR: f32 = 0.30;
const AHEAD_WEIGHT: f32 = 1.0;

// Periodic-domain march caps. A tiled domain has no horizontal exit, so the
// view march ends where the clear-air transmittance from the camera (the
// aerial perspective's, rows 15-17) falls to 2%: a cloud beyond that is
// invisible through the haze. A ray that climbs out of the haze never
// reaches that optical depth, so there is also a ceiling on horizontal
// range. The ceiling is a distance and not a number of domain widths, so
// that the visible range does not depend on the domain's size. 400 km is
// beyond the geometric horizon from any flight altitude, and the angular
// view-step LOD covers it in O(log) steps.
const PERIODIC_AIR_TAU_CUTOFF: f32 = 3.912023005428146; // -ln(0.02)
const PERIODIC_MAX_RANGE_M: f32 = 4.0e5;
// Step cap of the periodic view march, which can cover many domain widths.
const MAX_VIEW_STEPS_PERIODIC: i32 = 4096;

// Light-transfer split: a low-sun adjustment whose strength
// (u.light_split) the host fades with sun elevation. The beam terms are
// scaled by 1 + strength * DIRECT_BOOST * exp(-tau_sun), and a skylight fill
// of strength * SHADOW_SKYLIGHT is added in deep shadow.
const LIGHT_TRANSFER_DIRECT_BOOST: f32 = 0.25;
const LIGHT_TRANSFER_SHADOW_SKYLIGHT: f32 = 0.26;

// An orthonormal pair perpendicular to the sun, in which the draw on the
// solar disc is made. It depends only on the sun, so it is built once per
// fragment. Any helper not parallel to the sun gives a valid frame, and the
// azimuthal origin is arbitrary because the azimuth drawn is uniform.
fn sun_tangent_frame(sun: vec3<f32>) -> mat2x3<f32> {
    var helper = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(sun.z) > 0.9) {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    // @domain: helper is swapped for x when |sun.z| > 0.9: never parallel to sun
    let t1 = normalize(cross(helper, sun));
    return mat2x3<f32>(t1, cross(sun, t1));
}

// One draw from the solar cone about `sun` (see SUN_ANGULAR_RADIUS). `r` is
// a pair of uniforms in [0, 1); `tan_max` is the tangent of the cone's
// half-angle, and 0 returns `sun` exactly. The draw is uniform over the disc
// (radius ~ sqrt(r.x)), the right measure for a source of uniform radiance.
// Limb darkening is not modelled.
fn sun_cone_dir(sun: vec3<f32>, frame: mat2x3<f32>, r: vec2<f32>,
                tan_max: f32) -> vec3<f32> {
    if (tan_max <= 0.0) {
        return sun;
    }
    let t1 = frame[0];
    let t2 = frame[1];
    // Two components, stratified out of the one radial uniform: the first
    // AUREOLE_FRACTION of the interval draws from the wide skirt, the rest
    // from the disc. Each is remapped onto [0, 1), so each stays uniform
    // over its own disc.
    var scale = 1.0;
    var ru = r.x;
    if (ru < AUREOLE_FRACTION) {
        scale = AUREOLE_WIDEN;
        ru = ru / max(AUREOLE_FRACTION, 1e-6);
    } else {
        ru = (ru - AUREOLE_FRACTION) / max(1.0 - AUREOLE_FRACTION, 1e-6);
    }
    // @domain: ru is the uniform r.x, rescaled into [0, 1) on either branch
    let radius = tan_max * scale * sqrt(ru);
    let phi = 6.283185307179586 * r.y;
    var d = sun + radius * (cos(phi) * t1 + sin(phi) * t2);
    d.z = max(d.z, SUN_CONE_MIN_Z);
    // @domain: d.z >= SUN_CONE_MIN_Z > 0
    return normalize(d);
}

// The view step in cloud: at most dt_max, and short enough that the step's
// optical depth does not exceed TAU_STEP_MAX. It is continuous in sigma. A
// threshold on sigma would make the step length, and the quadrature bias
// with it, jump across an iso-surface of the cloud, and that contour would
// show in the image as a crease.
fn step_dt_for_sigma(sigma: f32, dt_max: f32) -> f32 {
    return min(dt_max, TAU_STEP_MAX / max(sigma, 1e-12));
}

// Optical depth from p toward the sun, marched to the domain's exit or to
// LIGHT_TAU_CUTOFF.
//
// `dt_floor` (m) coarsens the quadrature for distant samples without
// shortening its range: callers pass view_distance * u.light_lod_tan. At 0 the
// step is the level's own.
//
// `jit` in [0, 1) shifts the whole quadrature grid by a fraction of a step.
// With the LOD floor above about a voxel the march undersamples the field,
// and neighbouring pixels would undersample it on the same grid: over a flat
// sea every shadow ray starts at the same height, so a neighbourhood tests
// the cloud at the same few heights and the voxel structure of those slices
// prints onto the water. One random phase per pixel per frame makes the same
// step count an unbiased estimator of the integral (the shifted grids tile
// the ray), so the pattern becomes zero-mean noise that accumulation
// removes. The grid is shifted whole, not stratified step by step:
// consecutive taps stay exactly dt apart, which is a randomized midpoint
// rule and less noisy than independent strata, whose taps fall anywhere
// from 0 to 2 dt apart.
//
// The phase is applied only where the LOD floor has coarsened the step
// (LIGHT_JITTER_LOD_FULL). A march at the level's own step is a
// left-endpoint rule whose first tap is the sample point itself, which
// biases tau_sun high by about 0.5 * dt * sigma.
fn light_march_tau(p: vec3<f32>, sun: vec3<f32>, dt_floor: f32,
                   jit: f32) -> f32 {
    var t_exit: f32;
    var t_start: f32;
    if (PERIODIC_DOMAIN) {
        // Tiled domain: no horizontal exit. The sun is above the horizon
        // (the hosts require it when periodic), so the march always leaves
        // through the domain top.
        if (p.z >= u.bmax.z) {
            return 0.0;
        }
        t_exit = (u.bmax.z - p.z) / sun.z;
        t_start = max((u.bmin.z - p.z) / sun.z, 0.0);
    } else {
        let hit = ray_box(p, safe_inv_dir(sun));
        if (hit.x > hit.y || hit.y <= 0.0) {
            return 0.0;
        }
        t_exit = hit.y;
        t_start = max(hit.x, 0.0);
    }
    var tau = 0.0;
    // The phase is sized by the outer level's step, the only one known
    // here. A shadow ray that starts inside the nest is offset by more than
    // one of its own steps, which still decorrelates it and still
    // integrates the same range.
    let dt_fine = u.dt_light;
    let dt_nominal = max(dt_fine, dt_floor);
    let lod_fade = clamp(
        (dt_floor / max(dt_fine, 1e-6) - 1.0) / (LIGHT_JITTER_LOD_FULL - 1.0),
        0.0, 1.0
    );
    var t = t_start + clamp(jit, 0.0, 1.0) * dt_nominal * lod_fade;
    // The step's growth factor past LIGHT_DEEP_QUAD_TAU_START.
    var dt_boost = 1.0;
    for (var i: i32 = 0; i < MAX_LIGHT_STEPS; i = i + 1) {
        if (t >= t_exit) {
            break;
        }
        // The step follows the level the shadow ray is in, so a nest is
        // integrated at its own resolution and the coarse field outside it
        // is not. The step cap is shared: a ray crossing a deep nest can
        // exhaust MAX_LIGHT_STEPS, though tau saturation normally ends it
        // far sooner.
        let s = sample_level_at(p + t * sun);
        var dt = max(s.dt_light, dt_floor) * dt_boost;
        if (t + dt > t_exit) {
            dt = t_exit - t;
        }
        tau = tau + s.sigma * dt;
        if (tau > LIGHT_TAU_CUTOFF) {
            break;
        }
        if (tau > LIGHT_DEEP_QUAD_TAU_START) {
            dt_boost = min(dt_boost * LIGHT_DEEP_QUAD_GROWTH,
                           LIGHT_DEEP_QUAD_MAX_BOOST);
        }
        t = t + dt;
    }
    return tau;
}

// Two-stream (Eddington) diffuse transmittance of a conservatively
// scattering slab of optical depth tau, in similarity-scaled optical depth
// (1 - g) tau:
//   T = 1 / (1 + 0.75 (1 - g) tau)
// Unlike exp(-tau) it still discriminates at tau of several tens, so "under
// 300 m of cloud" and "under 3 km" stay different values. Every term for
// buried cloud is built on it.
fn diffuse_transmittance(tau: f32, g: f32) -> f32 {
    return 1.0 / (1.0 + 0.75 * (1.0 - g) * tau);
}

// Colour of the diffuse fill that a buried sample keeps.
//
// The fills are tinted with u.ambient_tint, the clear-sky colour. That is
// right for light that arrived straight from the sky, which is the part of
// each fill proportional to t_sky. The part that survives at t_sky = 0 (the
// floors ambient_occlusion_floor and SKY_PROBE_FILL_FLOOR) did not come from
// the sky overhead: it entered through the cloud's sunlit top and flanks and
// diffused in. Droplets scatter almost neutrally across the visible, so
// that light keeps the spectrum it entered with.
//
// The floor part therefore takes the ambient's chromaticity pulled
// DEEP_FILL_SUN_FRACTION of the way toward the beam's, at the ambient's
// luminance: a change of hue only. The fraction is tuned by eye against
// photographs of storm bases, which stay cool.
const DEEP_FILL_SUN_FRACTION: f32 = 0.15;
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722); // Rec. 709

fn deep_fill_tint(ambient: vec3<f32>) -> vec3<f32> {
    let sun_luma = max(dot(u.cloud_sun, LUMA_WEIGHTS), 1e-6);
    let sun_chroma = u.cloud_sun / sun_luma;
    // Both chromaticities have unit luma, so their mix does too and the
    // result has exactly the ambient's luminance.
    let amb_luma = max(dot(ambient, LUMA_WEIGHTS), 1e-6);
    let amb_chroma = ambient / amb_luma;
    let chroma = mix(amb_chroma, sun_chroma, DEEP_FILL_SUN_FRACTION);
    return amb_luma * chroma;
}

// Optical depth of the cloud above p (see SKY_PROBE_SPAN): a midpoint rule
// over the span. The sky visibility is diffuse_transmittance of it. It
// depends on the field alone, so it is baked into the light cache
// (fs_bake_light) and the march reads it from there.
fn sky_probe_tau(p: vec3<f32>) -> f32 {
    let headroom = u.bmax.z - p.z;
    if (headroom <= 0.0) {
        return 0.0;
    }
    let span = min(SKY_PROBE_SPAN, headroom);
    let dt = span / f32(SKY_PROBE_STEPS);
    var tau = 0.0;
    for (var i: i32 = 0; i < SKY_PROBE_STEPS; i = i + 1) {
        let t = (f32(i) + 0.5) * dt;
        tau = tau + sample_sigma(p + vec3<f32>(0.0, 0.0, t)) * dt;
    }
    return tau;
}

// Optical depth along the view ray from t0 to t1 (see AHEAD_STEP_SCALE): one
// coarse pass per pixel before compositing, so that the main march can
// subtract what it has crossed and know what is still ahead.
//
// It is coarse because its consumer is a two-stream transmittance whose
// useful range spans two decades of tau, on which a 20 percent quadrature
// error does not show. It takes the view march's distance LOD, and the tau
// cap ends it once the answer can no longer matter.
fn premarch_tau_ahead(origin: vec3<f32>, dir: vec3<f32>,
                      t0: f32, t1: f32, jit: f32) -> f32 {
    var tau = 0.0;
    let dt_base = u.dt_view * AHEAD_STEP_SCALE;
    var t = t0 + clamp(jit, 0.0, 1.0) * dt_base;
    for (var i: i32 = 0; i < AHEAD_MAX_STEPS; i = i + 1) {
        if (t >= t1 || tau > AHEAD_TAU_CAP) {
            break;
        }
        let s = sample_level_at(origin + t * dir);
        var dt = max(s.dt_view * AHEAD_STEP_SCALE, t * u.view_lod_tan);
        if (t + dt > t1) {
            dt = t1 - t;
        }
        tau = tau + s.sigma * dt;
        t = t + dt;
    }
    return tau;
}

// The periodic view march's maximum distance. The haze optical depth from
// the camera along the ray, in the exponential atmosphere the aerial
// perspective uses, is
//   tau(t) = beta0 * (H / mu) * (exp(-z0/H) - exp(-z1/H)),  z1 = z0 + mu * t
// (mu = dir.z), inverted here for tau = PERIODIC_AIR_TAU_CUTOFF / strength,
// i.e. a clear-air transmittance of 2%. PERIODIC_MAX_RANGE_M of horizontal
// travel caps the result. With aerial perspective off (strength == 0) only
// that range ceiling applies.
fn periodic_march_cap(cam_z: f32, dir: vec3<f32>) -> f32 {
    var cap = 3.4e38;
    let h_len = length(dir.xy);
    if (h_len > 1e-8) {
        cap = PERIODIC_MAX_RANGE_M / h_len;
    }
    let aer_beta0 = u.haze_beta0;
    let aer_h = u.haze_scale_height;
    let z0 = max(cam_z, 0.0);
    let mu = dir.z;
    let tau_cap = PERIODIC_AIR_TAU_CUTOFF;
    if (aer_h <= 0.0) {
        // Uniform haze: tau = beta0 * t, so every ray ends at the same
        // distance whatever its direction.
        return min(cap, tau_cap / aer_beta0);
    }
    let e0 = exp(-z0 / aer_h);
    if (mu > 1e-6 || mu < -1e-6) {
        let a = e0 - tau_cap * mu / (aer_beta0 * aer_h);
        if (a > 0.0) {
            // @domain: a > 0 on this branch
            let t_sol = (-aer_h * log(a) - z0) / mu;
            if (t_sol > 0.0) {
                cap = min(cap, t_sol);
            }
        }
        // a <= 0: an upward ray leaves the haze before it reaches the
        // cutoff optical depth, and only the range ceiling caps it.
    } else {
        cap = min(cap, tau_cap / (aer_beta0 * e0));
    }
    return cap;
}
