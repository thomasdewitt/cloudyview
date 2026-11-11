// ---------------------------------------------------------------------------
// Constants (witness.py values where the concept carries over)
// ---------------------------------------------------------------------------

const SUN_COLOR: vec3<f32> = vec3<f32>(20.2, 21.0, 22.4); // look.py SUN_COLOR
// Droplet diffraction spike on the once-scattered beam (2026-08-14).
//
// A single HG lobe at the shipped g = 0.76 is the tuned SHAPE of the side-
// and back-scattered light, but as a forward peak it is off by orders of
// magnitude: a 10 um droplet puts roughly half its scattered energy into a
// diffraction lobe a couple of degrees wide (the same lambda/d ~ 3 deg the
// aureole draws from), whose peak is ~10^3 x isotropic where HG(0.76)
// manages ~30x. That missing factor is why a thin veil in front of the sun
// rendered as the same order of brightness as a sunlit thick base, when a
// camera pointed sunward meters the veil orders of magnitude brighter and
// exposes the base down to dark grey.
//
// The spike is a second, narrow HG lobe ADDED to the once-scattered term
// only. Deliberately additive rather than an energy split: re-weighting the
// tuned lobe by (1 - w) would dim every side- and back-scatter view by
// nearly 2x, and the tuned look is the baseline this change is measured
// against. The lobe's own solid angle is so small (half-width ~2 deg) that
// the added energy integrates to a few percent of the total.
//
// It rides octave 0's transmittance exp(-tau_sun): the diffraction peak is
// a property of light scattered ONCE — a second scattering convolves the
// lobe with the full phase function and hands the result to the MS ladder,
// which already owns it. g = 0.95 puts the half-width at ~2 deg, matching
// the droplet lobe the aureole models; the weight is the Mie diffraction
// share of ~half the scattered energy.
//
// The lobe is windowed to zero by 30 deg from the sun. An HG tail is not
// compact — at 90 deg HG(0.95) still runs ~9% of the tuned lobe, so the
// unwindowed sum would brighten every side- and back-scatter view by a
// near-constant tithe, which is exactly the look change this term must not
// make. Diffraction has no such tail (it is an aperture transform, not a
// resonance), so the window is more physical than the HG it trims, and the
// smoothstep keeps the sky around the sun free of a visible rim.
const DIFFRACTION_G: f32 = 0.95;
const DIFFRACTION_WEIGHT: f32 = 0.5;
const DIFFRACTION_WINDOW_COS_START: f32 = 0.8660254037844387; // cos 30 deg
const DIFFRACTION_WINDOW_COS_FULL: f32 = 0.9659258262890683;  // cos 15 deg
const POWDER_COEFF: f32 = 1.5;       // witness.py:63
// Powder is a *backscatter* phenomenon (lighting-loop iter_014). The boundary
// darkening it models is the deficit of low-order paths that have to reverse
// direction near a surface: with the sun behind the observer a thin edge has
// not had room to build the multiple scattering that would send light back,
// so it reads dark. Looking *toward* the sun the same edge is lit by the
// forward peak in transmission — the paths are filled by construction — and
// the darkening should not be there at all. The term below fades powder out
// over the genuinely forward cone only; side- and back-scatter keep it whole,
// which also makes any view whose whole frame sits below the cone start
// bit-identical to iter_012.
//
// "Genuinely forward" is not a taste threshold, it is a property of the phase
// function and is derived rather than tuned: the fade begins exactly where the
// HG lobe rises above isotropic, i.e. where a photon crossing a thin edge is
// more likely to continue toward the camera than a random walk would send it.
// Solving HG(mu, g) = 1/4pi for mu gives
//     mu_cross = (1 + g^2 - (1 - g^2)^(2/3)) / (2 g),
// which is 0.667 at the shipped g = 0.76 (and 0.762 at g = 0.85). Below it the
// forward lobe is *weaker* than isotropic and there is nothing to fill the
// backscatter paths with, so powder stays whole.
const POWDER_FWD_FADE: f32 = 0.85;       // fraction of powder removed at mu = 1
// Ambient tint now arrives per-frame in u.ambient_tint (spectral fill);
// witness legacy value is (0.19, 0.225, 0.30) since iter_010.
const AMBIENT_HEIGHT_FLOOR: f32 = 0.3; // witness.py:85
const BOUNCE_STRENGTH: f32 = 0.05;   // witness.py:95
const BOUNCE_TINT: vec3<f32> = vec3<f32>(1.0, 0.97, 0.92); // witness.py:96-98
// Sunlit ground beyond the edge of this cloud's own shadow still bounces up
// into its base (lighting-loop iter_005; see the use site).
const BOUNCE_LATERAL_FLOOR: f32 = 0.25;
const GRADIENT_SHADING_RADIUS_VOXELS: f32 = 1.0; // witness.py tuning block
const GRADIENT_SHADING_COARSE_MIN_VOXELS: f32 = 4.0;
const GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION: f32 = 0.125;
const GRADIENT_SHADING_TAU_START: f32 = 0.25;
const GRADIENT_SHADING_TAU_FULL: f32 = 1.60;
const GRADIENT_SHADING_CONF_START: f32 = 0.06;
const GRADIENT_SHADING_CONF_FULL: f32 = 0.28;
const GRADIENT_SHADING_SHADOW_SIDE_SCALE: f32 = 0.55;
// The deep-shadow gate spans 15 -> 1000 in LOG tau (2026-08-14, Thomas: the
// darkest bottoms belong to the genuinely thickest columns, and tau 100 is
// not that — a storm core runs to several hundred). Log space rather than
// linear because a linear smoothstep over 15..1000 spends nearly all its
// travel above tau 200 and the transition would collapse back into a cliff
// at the deep end; per-e-fold is how the rest of the transition curve was
// shaped, so the gate matches it. The march that feeds this saturates at
// LIGHT_TAU_CUTOFF, whose deep-quadrature escalation (see light_march_tau)
// is what makes measuring tau 1000 affordable.
const DEEP_SHADOW_TAU_START: f32 = 15.0;
const DEEP_SHADOW_TAU_FULL: f32 = 1000.0;
const DEEP_SHADOW_LOG_TAU_START: f32 = 2.70805020110221;  // ln(15)
const DEEP_SHADOW_LOG_TAU_FULL: f32 = 6.907755278982137;  // ln(1000)
// Shallow/storm discrimination (2026-08-11 tuning). tau_sun saturates at
// LIGHT_TAU_CUTOFF for a fair-weather cumulus base and a storm base alike,
// so the deep-shadow machinery — tuned on storm references — darkened both.
// The sky probe already measures the difference: a shallow base sits under
// a few hundred meters of cloud (t_sky well above zero), a buried storm
// interior under kilometres (t_sky ~ 0). shallow_open fades the storm
// treatment out for optically open shadow: less MS suppression, less
// occlusion, skylight-blue fill, and a high-sun shadow skylight the
// light-transfer split (gated to low sun) never provided.
const SHALLOW_OPEN_TSKY_START: f32 = 0.05;
const SHALLOW_OPEN_TSKY_FULL: f32 = 0.30;
// Fraction of MS suppression / ambient occlusion an open shallow shadow
// keeps (a buried storm keeps 1.0 = the tuned look, exactly).
const SHALLOW_SUPPRESSION_KEEP: f32 = 0.25;
// High-sun skylight reaching saturated shadow, by openness. The low-sun
// path (light-transfer split) keeps its exact 0.26 as split -> 1.
const HIGH_SUN_SHADOW_SKYLIGHT_STORM: f32 = 0.03;
const HIGH_SUN_SHADOW_SKYLIGHT_SHALLOW: f32 = 0.24;
// The high-sun skylight fill engages on *moderate* shadow, not the deep
// gate: direct light is spent by tau_sun ~ 5 and the MS octaves by ~ 20,
// so a fair-weather base at tau_sun 15-38 was pitch dark yet entirely
// below DEEP_SHADOW_TAU_START. The gate is exponential-onset, not a
// smoothstep window: a late-arriving gate made brightness NON-MONOTONE
// in tau (white -> grey -> white -> grey moving into a base, verified on
// the linear-tau wedge harness). 1 - exp(-tau/t0) rises while the beam
// and MS ladder are still alive, so the summed profile only decays.
const SHADOW_SKYLIGHT_TAU_ONSET: f32 = 9.0;
// Diffused-beam glow: the Eddington diffusion tail of the sun through the
// cloud mass, emitted isotropically. The MS octave ladder decays
// geometrically in tau_sun and is spent by ~20; real diffusion decays as
// 1/(1 + 0.18 tau), and the gap is exactly the tau_sun 15-50 band where
// fair-weather bases went dark — a real base is diffusely TRANSLUCENT and
// often brighter than the sky behind it. This is the out-and-back
// multiple-scattering approximation at zero added cost: the "out" march is
// the tau_sun the light march already measured, the return is analytic.
// Gated in as the MS ladder dies (no double counting below tau ~8); a
// buried storm keeps only DIFFUSE_BEAM_STORM_KEEP of it.
const DIFFUSE_BEAM_STRENGTH: f32 = 0.75;
const DIFFUSE_BEAM_TAU_ONSET: f32 = 7.0;
// A buried storm interior keeps this much of the glow. 0.25 was tuned to
// AI storm references and produced a floor DARKER than the real anvil
// photo (IMG_7053: darkest cores ~115/255; the 0.25 floor rendered 68)
// and an 8x step across the deep gate. 0.55 lands the floor on the photo
// and halves the step.
const DIFFUSE_BEAM_STORM_KEEP: f32 = 0.80;
// Spectral tint of the diffused beam: hundreds of scatterings put an
// effective absorption path of tens of meters of liquid water on this
// light, and water absorbs red far more than blue (single-scatter albedo
// ~0.998 at 680 nm vs ~0.9999 at 460 nm) — deep-diffused daylight is
// measurably blue. Reference bases sit at linear B/R ~ 2.4.
// Weight of the measured surface gradient on the diffuse fills (ambient,
// skylight, diffuse beam). 0 = the fills stay orientation-blind (flat
// shadow); 1 = fills shade as hard as the beam does.
const FILL_GRADIENT_WEIGHT: f32 = 0.6;

const DIFFUSE_BEAM_TINT: vec3<f32> = vec3<f32>(0.85, 0.97, 1.08);
// Ambient height ramp floor for open shallow cloud (storms keep
// AMBIENT_HEIGHT_FLOOR = 0.3): an open base sees the sky sideways and the
// bright surface below, so its fill should not fall to a third.
const AMBIENT_HEIGHT_FLOOR_SHALLOW: f32 = 0.6;
const DEEP_SHADOW_MS_FLOOR: f32 = 0.28;
const MS_OCTAVES: i32 = 6;           // witness.py:76
const MS_ATTEN: f32 = 0.4;           // witness.py:77
const MS_BLEND_RATE: f32 = 0.35;     // witness.py:78
const MAX_VIEW_STEPS: i32 = 2048;   // witness.py:102
const MAX_LIGHT_STEPS: i32 = 512;   // witness.py:72
const TRANSMITTANCE_CUTOFF: f32 = 0.002; // witness early-exit
const LIGHT_TAU_CUTOFF: f32 = 1000.0; // was 80; see DEEP_SHADOW_TAU_FULL
// Deep-quadrature escalation for the sun march. Fine steps to tau 1000
// would cost hundreds of extra taps; past the escalation threshold nothing
// consumes tau but the two-stream diffuse terms and the log-space deep
// gate, both logarithmically sensitive, so the step grows geometrically
// and the whole 32 -> 1000 tail costs ~20 extra taps. The threshold sits
// beyond every exponential consumer that matters: the beam is dead by
// tau ~ 8 and MS octave 2 by ~ 30; octaves 3-5 still breathe there, but
// their weights (0.064 down) put the quadrature error of a growing step
// below a display level.
const LIGHT_DEEP_QUAD_TAU_START: f32 = 32.0;
const LIGHT_DEEP_QUAD_GROWTH: f32 = 1.2;
const LIGHT_DEEP_QUAD_MAX_BOOST: f32 = 64.0;
const EMPTY_DTAU_CUTOFF: f32 = 1e-5; // witness.py:701
const TAU_STEP_MAX: f32 = 0.5;       // witness.py:689
// Tone-map gamma is per-frame (u.periodic.w), not a const — see tone_map.
// witness's own value is 1.4; the app ships a higher default (engine
// DEFAULT_TONE_MAP_GAMMA) because 1.4 alone is not the look soar has
// actually been flown with.
const ISO_PHASE: f32 = 0.07957747154594767; // 1 / (4*pi)

// Sky-visibility probe (lighting-loop iter_001).
//
// Once the sun march saturates (LIGHT_TAU_CUTOFF), tau_sun carries no
// information: every deep sample sees exp(-tau_sun) = 0, the MS octaves die,
// and the whole interior collapses onto one ambient value scaled by the
// constant deep-shadow floor. That constant is the flat grey of a thick
// cloud base.
//
// The skylight actually reaching such a sample is not constant — it is sky
// radiance diffusely transmitted through whatever cloud lies between the
// sample and the sky. Measure that with a short march straight up and feed
// the result through the Eddington / two-stream diffuse transmittance of a
// conservative scattering slab in similarity-scaled optical depth:
//
//     T_sky = 1 / (1 + 0.75 * (1 - g) * tau_up)
//
// The (1-g) scaling is what makes this usable: with g = 0.85 the factor
// still discriminates out to tau_up ~ 40 instead of collapsing to zero by
// tau_up ~ 5 the way exp(-tau_up) would, so a base under 300 m of cloud and
// a base under 3 km of cloud are different pixels.
//
// The probe is bounded (SKY_PROBE_SPAN) rather than run to the domain top:
// skylight entering the side of a wall is not measured by a vertical ray, so
// a whole-column integral over-darkens vertical faces. A bounded span reads
// as "how much cloud is immediately overhead", which is the cue that puts
// dark cores under the deep parts and luminous fringes near thin spots and
// gaps.
const SKY_PROBE_SPAN: f32 = 1500.0;   // m of headroom sampled above p
const SKY_PROBE_STEPS: i32 = 15;      // 100 m quadrature at full span
// The deepest limit of the skylight-split fill: at T_sky -> 0 the shadow
// skylight keeps this fraction, so saturated shadow still does not go black.
const SKY_PROBE_FILL_FLOOR: f32 = 0.40;

// Isotropic-tail diffusion depth (lighting-loop iter_002).
//
// The MS cascade's octave k illuminates with exp(-MS_ATTEN^k * tau_sun). With
// MS_ATTEN = 0.4 that is exp(-0.010 * tau_sun) by octave 5 and exp(-0.026 *
// tau_sun) by octave 4: constants for any tau_sun a cloud can produce. Those
// octaves are therefore applied orientation-blind and depth-blind, and they
// are not small - ablation on v6 (drop octaves 1-5) takes the sunlit cloud
// from mean 197 to 159, so the tail is roughly a fifth of a lit core's
// brightness, delivered as a flat pedestal. That pedestal is what turns
// sunlit cauliflower into paste: the turret tops and the crevices between
// them differ by tens of optical depths to the sun, octaves 0-1 have already
// died in both, and everything that remains is the same number.
//
// The tail is diffuse light that arrived by many scatterings from the
// illuminated part of the cloud, so what should set its magnitude is the
// diffusion depth to that illuminated region - which is the same two-stream
// slab transmittance iter_001 introduced, evaluated on the sun path:
//
//     T_d = 1 / (1 + 0.75 * (1 - g) * tau_sun)
//
// Unlike exp(-MS_ATTEN^k * tau_sun) this still discriminates at tau_sun = 40,
// and unlike exp(-tau_sun) it does not collapse by tau_sun = 5, so it fills
// exactly the range the cascade leaves flat. A lit face (tau_sun ~ 0) is
// unchanged; a crevice shadowed by the next turret darkens smoothly.
//
// The floor is the fully buried limit, so a deep core keeps a tail rather
// than going black - and it bounds how much this can compound with the
// existing deep-shadow MS suppression.
const MS_TAIL_FLOOR: f32 = 0.18;
// Below this sun optical depth the tail is left exactly alone. A sample that
// the sun still reaches nearly unattenuated is not sitting at the bottom of a
// diffusion well, and dimming it would only cost brightness without buying
// form; the knee is what turns this into a contrast change instead of an
// exposure change, and it is why the thin-cloud regression views are
// essentially untouched.
const MS_TAIL_TAU_KNEE: f32 = 1.0;

// Incidence cosine of the beam on the local skin (lighting-loop iter_006).
//
// iter_002 gave the isotropic tail a *depth* (how far the sample sits from the
// illuminated region, measured along the sun path). It still has no
// *direction*. On directly sunlit skin — which is essentially all of a cloud
// field seen from above — tau_sun is below the knee everywhere the camera can
// see, so the tail is delivered at full strength to every lit sample
// regardless of which way that sample's surface faces. Ablation on the aerial
// views (drop octaves 1-5) takes v8's cloud tops from L 205 to 151: a quarter
// of a cloud top's brightness is a pedestal with no orientation dependence at
// all. That is why tops read uniform from altitude while the same cloud's side
// view does not — every other term that could shade them is inert up there
// (the whole deep-shadow machinery of iterations 001/004/005 is worth 0.11
// levels on v8, gradient shading 2.0, ambient 1.4).
//
// The tail is diffuse light that entered this neighbourhood of the cloud and
// random-walked to the sample, so its source strength is the irradiance
// available locally, and for a collimated beam on a tilted skin element that
// is the incidence cosine mu0 = n.s. A turret crown turned toward the beam
// collects the full irradiance; the wall of the crevice beside it, tilted
// away, collects a fraction of it and gets a proportionally weaker tail even
// though the beam still reaches both and tau_sun cannot tell them apart.
//
// Normalized by the value a *horizontal* face would see (mu_ref = sun.z), so
// this is a contrast change and not an exposure change: level cloud top keeps
// exactly its iter_005 value and only surfaces tilted away from the sun give
// anything up. Same discipline as iter_002's knee, and it is most of what
// keeps the thin regression views still — a wisp never reaches the tau_depth
// the surface gate asks for, so it has no resolved skin to tilt.
//
// It rides (1 - deep_shadow_gate), the reverse of every other term this loop
// has added: once tau_sun saturates there is no local beam irradiance left to
// be proportional to, the tail is pure diffusion from elsewhere, and the
// orientation of this particular skin element stops meaning anything. That
// hand-off is also what keeps v1's and v2's hard-won shadowed masses out of
// this iteration's way, by construction rather than by tuning.
const TAIL_MU_FLOOR: f32 = 0.30;
// Guard for a sun near the horizon: without it mu_ref -> 0 and the ratio
// saturates at 1 for every orientation. That is the correct limit (a grazing
// beam lights nothing preferentially) but a numerically noisy way to reach it.
const TAIL_MU_REF_MIN: f32 = 0.25;
// Sun-march quadrature jitter (lighting-loop iter_003). The random phase is
// off while the distance-LOD floor is at or below the light march's own step
// and full once the floor has coarsened it by this factor, so a march the
// fine step already resolves stays exactly as it was. Two is the smallest
// value that reaches full randomization before the coherent pattern becomes
// visible: the artifact needs dt to exceed roughly a voxel of structure, and
// dt_light is two voxels.
const LIGHT_JITTER_LOD_FULL: f32 = 2.0;

// Solar penumbra (lighting-loop iter_007).
//
// Every shadow in this renderer is cast by a point source: one shadow ray per
// sample, toward one direction. The sun is not a point — it subtends 0.53
// degrees, so a shadow edge is not an edge but a penumbra whose width grows
// linearly with the distance from caster to receiver. A cloud 4 km above the
// water throws a ~40 m penumbra onto it; a cloud shadow falling on another
// cloud 2 km behind carries ~20 m; and every terminator in the references is
// visibly a gradient, not a step.
//
// The implementation is the cheap one that the accumulation buffer makes
// exact: deflect the shadow ray's direction into the solar cone, one draw per
// pixel per frame (and advanced along the view march by an R2 additive
// recurrence, so a single view ray's many shadow rays sample the disc
// low-discrepancy rather than all landing on the same point of it). Averaging
// the *radiance* over frames gives <exp(-tau)> over the disc, which is the
// correct penumbra; averaging tau first would not be. Nothing else in the
// shader sees the deflected direction — the phase function, the gradient
// shading incidence, the ocean glint and the sky all keep the disc centre,
// because a quarter-degree rotation is invisible in all of them and only the
// *occlusion* is a discontinuous function of direction.
//
// It rides the same jitter_on * jitter_scale switch as iter_003's quadrature
// phase: asking for a deterministic march asks for a point sun, and gets the
// exact previous image.
const SUN_ANGULAR_RADIUS: f32 = 0.0046542; // tan(0.2666 deg), the true disc
// Widening factor, an art knob, and the honest measurement is that it should
// be 1. Sweeping it (see the journal) shows the true disc doing exactly the
// work the geometry predicts and no more: at 9-degree sun the shadow throw is
// 6.4 times the caster height, so a 2 km cloud casts a 120 m penumbra onto the
// water — wider than a voxel, and it visibly dissolves v7's hard sawtooth
// shadow edge. At 55-degree sun the same cloud casts 23 m, a fifth of a voxel
// and narrower than the trilinear ramp the rendered cloud edge already has, so
// v5/v8 do not move and *should* not. Forcing them to move by widening is a
// distance-weighted blur of the shadow field, and it visibly eats the small
// cloud shadows that carry the LES structure.
const SUN_CONE_WIDEN: f32 = 1.0;

// The second component of the source: the forward-scattering skirt.
//
// A shadow's edge in a real cloud photograph is not one penumbra but two
// nested ones — a sharp core from the disc, and a wide, low-contrast foot. The
// foot is not geometry, it is diffraction: a 10 um droplet's forward lobe is
// about lambda/d ~ 3 degrees wide, so sunlight clipping the caster's edge is
// deflected by a couple of degrees and continues almost undiminished into what
// the geometric shadow says is dark. The same happens in the haze along the
// path. A single-scattering shadow march has no way to represent that light —
// it counts every scattering event as an extinction — which is exactly why our
// shadow edges read as cutouts.
//
// The cheap equivalence is to put that flux into the *source*: a fraction of
// the shadowing illumination is drawn from a much wider cone. Unlike raising
// SUN_CONE_WIDEN this does not blur the shadow, because 85 percent of the
// draws still land on the disc — it adds a faint skirt outside the core edge
// and leaves the core where it was. The two-component draw is stratified out
// of the same radial uniform, so it costs one compare.
const AUREOLE_FRACTION: f32 = 0.15;
const AUREOLE_WIDEN: f32 = 8.0;   // ~2.1 degrees, the droplet forward lobe
// Numerical guard only: the deflected ray must stay above the horizon, since
// the periodic light march exits through the domain top and divides by sun.z.
// engine.write_uniforms validates sun elevation > 0; with the shipped cone
// (1.6 deg half-angle) this clamp is inactive above ~2 deg elevation, and the
// judge set's lowest sun is 9 deg.
const SUN_CONE_MIN_Z: f32 = 0.02;

// Forward pre-march: what is ahead of the sample (lighting-loop iter_004).
//
// Every quantity the march has ever had is *backward*-looking. tau_depth is
// depth since the last cloud entry (and resets at every gap), transmittance
// is what the ray has already crossed, the sun march looks toward the sun,
// and iter_001's probe looks up. Nothing knows what lies beyond the sample
// along the sightline — and because compositing kills everything past
// tau_view ~ 6, the image of a cloud is formed entirely by a thin visible
// skin. That skin therefore has no idea whether it is the skin of a tau-2
// wisp or of a tau-300 anvil, and it is lit identically in both cases.
//
// Physically the difference is large and it is a *diffuse-illumination*
// difference. The fills iter_001 measures (the AO'd ambient and the
// shadow-skylight floor) are diffuse radiance arriving at the sample from the
// hemisphere around it. iter_001 samples one direction of that hemisphere,
// straight up. The other direction that matters for a visible sample is the
// one the camera cannot see behind it: a skin sample on a wisp is within a
// couple of mean free paths of open air in *both* directions, while the same
// skin on a monolith is open only backward, toward the camera, and buried
// forward. One coarse pre-march along the view ray measures exactly that.
//
// This is not iter_002's rejected hemispherical probe. That failed because it
// spent a third of its cosine weight on near-horizontal directions chosen
// blind, which in a horizontally extensive layer all report T ~ 0 and flatten
// everything. Here the extra direction is not chosen: it is the sightline,
// the one direction along which the renderer is already integrating, so the
// structure it reports is registered with the silhouette the viewer is
// looking at. That is why it produces a luminous fringe hugging every edge
// and gap rather than a uniform veil.
//
// Weight and floor bound how far it may go: at AHEAD_FLOOR the buried limit
// still keeps that fraction of the vertical probe's answer, so a deep body
// darkens toward a floor rather than toward black.
const AHEAD_STEP_SCALE: f32 = 4.0;   // pre-march dt = this x the view dt
const AHEAD_MAX_STEPS: i32 = 192;
const AHEAD_TAU_CAP: f32 = 60.0;     // T_d(60) = 0.036; past here it is floor
const AHEAD_FLOOR: f32 = 0.30;
const AHEAD_WEIGHT: f32 = 1.0;

// Periodic-domain march caps. A tiled domain has no horizontal exit, so the
// view march ends where camera->sample clear-air transmittance (the aerial
// perspective exponential atmosphere, rows 16-17) drops to ~2percent — beyond
// that a cloud sample is invisible through the haze — with an absolute
// range ceiling so rays above the haze (where the transmittance cap
// diverges) still terminate. The ceiling is a distance, not a wrap count:
// a wrap count made the tiling end visibly early on small domains, and the
// eye judges range in meters of haze and horizon, not in domain widths.
// 400 km is past the Earth-curvature horizon from any sane camera altitude,
// and the angular view-step LOD reaches it in O(log) steps.
const PERIODIC_AIR_TAU_CUTOFF: f32 = 3.912023005428146; // -ln(0.02)
const PERIODIC_MAX_RANGE_M: f32 = 4.0e5;
// Extra step headroom for the (much longer) periodic view march; the
// non-periodic path keeps MAX_VIEW_STEPS exactly.
const MAX_VIEW_STEPS_PERIODIC: i32 = 4096;

// Light-transfer split (witness iter_006). The elevation fade lives on the
// CPU (engine._effective_light_transfer_split); these are the fixed knobs.
const LIGHT_TRANSFER_DIRECT_BOOST: f32 = 0.25;
const LIGHT_TRANSFER_SHADOW_SKYLIGHT: f32 = 0.26;

// One draw from the solar cone about `sun` (see the SUN_* block). `r` is a
// pair of uniforms in [0,1); `tan_max` is the cone's tangent half-angle, and 0
// returns `sun` exactly. Uniform on the disc (sqrt(r1) radius), which is the
// right measure for a source of uniform radiance — limb darkening and the
// aureole's radial profile are both out of scope here.
// The tangent frame the draw is made in. It depends only on the sun, which is
// constant over the whole frame, so it is built once per fragment (fs_main)
// rather than per draw — and there is one draw per view step, plus the
// ocean's. Any helper not parallel to the sun gives a valid frame; the
// azimuthal origin is arbitrary because r.y is uniform.
fn sun_tangent_frame(sun: vec3<f32>) -> mat2x3<f32> {
    var helper = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(sun.z) > 0.9) {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    // @domain: helper is swapped for x when |sun.z| > 0.9: never parallel to sun
    let t1 = normalize(cross(helper, sun));
    return mat2x3<f32>(t1, cross(sun, t1));
}

fn sun_cone_dir(sun: vec3<f32>, frame: mat2x3<f32>, r: vec2<f32>,
                tan_max: f32) -> vec3<f32> {
    if (tan_max <= 0.0) {
        return sun;
    }
    let t1 = frame[0];
    let t2 = frame[1];
    // Two-component source, stratified out of the same radial uniform: the
    // first AUREOLE_FRACTION of the interval draws from the wide forward-
    // scattering skirt, the rest from the solar disc. Both are remapped back
    // onto [0,1) so each component keeps its own uniform-on-the-disc measure.
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

// Continuous in sigma on purpose: a threshold (the old DENSE_SIGMA_CUTOFF
// branch) makes the step length jump discontinuously across an iso-surface
// of the cloud, and the quadrature bias jumps with it — which printed the
// iso-contour onto the image as a sharp crease. Thin decks whose sigma sits
// near the threshold (stratified cirrus) showed it worst.
fn step_dt_for_sigma(sigma: f32, dt_max: f32) -> f32 {
    return min(dt_max, TAU_STEP_MAX / max(sigma, 1e-12));
}

// Sun optical depth from p toward the sun: witness adaptive-step shadow march
// to the box exit with tau saturation (witness.py:299-367).
// `dt_floor` coarsens (never truncates) the quadrature for distant samples:
// callers pass view_distance * u.periodic.y, so the full integration range
// is kept while far cloud copies stop paying near-field step counts.
// dt_floor = 0 -> bit-exact legacy.
//
// `jit` in [0, 1) offsets the quadrature grid (lighting-loop iter_003). The
// LOD floor is an *angular* step: it gives a distant sample the step count a
// footprint-filtered march would need, but the samples themselves are still
// point taps of the field, so above ~1 voxel of dt the march is undersampled
// rather than filtered. Undersampling on a grid that neighbouring pixels
// share is what produced the stair-steps: a flat ocean puts every pixel's
// shadow ray at the same t_start, dt_floor varies only slowly across the
// image, so a whole neighbourhood tests the cloud at the same few heights
// and the raw voxel structure of those slices prints onto the water.
// Offsetting the whole quadrature grid by one uniform random phase per pixel
// per frame makes the same step count an unbiased estimator of the same
// integral (the shifted grids tile the ray), so the coherent pattern becomes
// zero-mean noise that accumulation averages away.
//
// A randomized *lattice* rather than per-step stratification, deliberately:
// stratifying each step independently was tried and is visibly noisier
// (v7 ocean high-frequency std 2.36 vs 1.68 at 64 frames, 9.5 vs 6.6 at one
// frame). Independent strata put consecutive taps anywhere from 0 to 2 dt
// apart; the shifted regular grid keeps them exactly dt apart, which is a
// randomized midpoint rule and converges like h^2 on a field this smooth
// instead of h^1.5.
//
// The phase fades in only where the LOD floor actually coarsens the march
// past its own step (LIGHT_JITTER_LOD_FULL), and that gate is not cosmetic.
// The unjittered march is a left-endpoint rule whose first tap is the sample
// point itself, so it charges a full step of the local density and biases
// tau_sun high by about 0.5 * dt * sigma. Randomizing the phase removes that
// bias as well as the aliasing — measured as a 3-6 level brightening of every
// cloud in every view, near field included, which is a global lighting change
// and not this iteration's business. Gated, a march the fine step already
// resolves keeps its exact previous value (bias and all) and only the
// LOD-coarsened rays — the ones with the artifact — are randomized.
fn light_march_tau(p: vec3<f32>, sun: vec3<f32>, dt_floor: f32,
                   jit: f32) -> f32 {
    var t_exit: f32;
    var t_start: f32;
    if (PERIODIC_DOMAIN) {
        // Tiled domain: no horizontal exit; the sun is above the horizon
        // (engine.write_uniforms validates sun_elevation > 0 when periodic),
        // so the light march always leaves through the domain top.
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
    // The phase is sized by the step this ray will actually take. Only the
    // outer level's dt_light is known here; a shadow ray that starts inside
    // the nest is offset by more than one of its own steps, which still
    // decorrelates it and still integrates the same range.
    let dt_fine = u.bmax.w;
    let dt_nominal = max(dt_fine, dt_floor);
    let lod_fade = clamp(
        (dt_floor / max(dt_fine, 1e-6) - 1.0) / (LIGHT_JITTER_LOD_FULL - 1.0),
        0.0, 1.0
    );
    var t = t_start + clamp(jit, 0.0, 1.0) * dt_nominal * lod_fade;
    // Deep-quadrature escalation state (see LIGHT_DEEP_QUAD_*): grows the
    // step geometrically once tau passes the threshold, so the march can
    // afford to measure all the way to LIGHT_TAU_CUTOFF = 1000.
    var dt_boost = 1.0;
    for (var i: i32 = 0; i < MAX_LIGHT_STEPS; i = i + 1) {
        if (t >= t_exit) {
            break;
        }
        // Step size follows the level the shadow ray is currently in, so a
        // fine nest is integrated at its own resolution and the coarse field
        // outside it is not (witness._light_march). The step cap is shared:
        // a shadow ray crossing a deep nest can exhaust MAX_LIGHT_STEPS, but
        // tau saturation (LIGHT_TAU_CUTOFF) normally ends it far sooner.
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

// Eddington / two-stream diffuse transmittance of a conservative scattering
// slab, in similarity-scaled optical depth. This is the one function that
// replaces the constant deep-shadow floors: unlike exp(-tau) it keeps
// discriminating out to tau ~ 40, so "buried under 300 m of cloud" and
// "buried under 3 km" stay different pixels instead of both collapsing onto
// the same saturated grey.
fn diffuse_transmittance(tau: f32, g: f32) -> f32 {
    return 1.0 / (1.0 + 0.75 * (1.0 - g) * tau);
}

// Spectrum of the buried residual of the diffuse fills (lighting-loop
// iter_005).
//
// Both diffuse fills are tinted with u.ambient_tint, which is the *clear-sky*
// spectrum — strongly blue (legacy 0.19/0.225/0.30, B/R = 1.58). That is the
// right colour for light which arrived at the sample straight from the sky,
// and it is what the t_sky-proportional part of each fill is. It is the wrong
// colour for the part that survives when t_sky -> 0.
//
// Both fills carry a floor (ambient_occlusion_floor, SKY_PROBE_FILL_FLOOR)
// precisely because one vertical measurement cannot drive a deeply buried
// sample black: light still gets in, but not from the sky overhead. It gets
// in through the *sunlit* top and flanks and diffuses. Liquid-water droplets
// scatter essentially neutrally across the visible, so many scatterings do
// not blue that light further — it keeps the spectrum it entered with, which
// is the direct beam's (u.cloud_sun, warm), softened toward white by the
// sideways skylight mixed into it on the way.
//
// So the floor part of each fill gets this spectrum instead, at the same
// luminance as the ambient tint it replaces. The change is a pure hue
// rotation of the deepest shadow: level, contrast and structure are
// untouched, and any sample the sun march still resolves (gate 0) keeps
// u.ambient_tint exactly. This is what stops accumulated deep shadow from
// converging on slate blue, which the references never do — real dark bases
// hold a neutral-to-faintly-warm grey (ai07's storm base B/R 1.10, its
// nearby dark cumulus 0.78) even where they are very dark.
const DEEP_FILL_SUN_FRACTION: f32 = 0.15;
const LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn deep_fill_tint(ambient: vec3<f32>) -> vec3<f32> {
    let sun_luma = max(dot(u.cloud_sun.xyz, LUMA_WEIGHTS), 1e-6);
    let sun_chroma = u.cloud_sun.xyz / sun_luma;
    // Luma-preserving by construction: every endpoint has unit luma, so the
    // mix does too and the returned tint carries exactly the ambient tint's
    // luminance. Only the chromaticity moves.
    //
    // The base chroma is the AMBIENT's blue, not neutral (2026-08-11): the
    // neutral anchor came from AI-generated storm references, but the real
    // anvil-underside photo (IMG_7053) holds display B-R = +15..+18 in its
    // darkest cores — through the tone curve's ~0.30 log-slope that needs
    // linear B/R ~ 1.5, i.e. buried shadow KEEPS the skylight's coolness,
    // with only a small warm pull from sun that diffused in.
    let amb_luma = max(dot(ambient, LUMA_WEIGHTS), 1e-6);
    let amb_chroma = ambient / amb_luma;
    let chroma = mix(amb_chroma, sun_chroma, DEEP_FILL_SUN_FRACTION);
    return amb_luma * chroma;
}

// Diffuse sky transmittance above p (see the SKY_PROBE_* block). `jit` is a
// per-pixel offset in [-0.5, 0.5] that decorrelates the quadrature shells
// between neighbouring pixels; temporal accumulation averages it out.
fn sky_probe_transmittance(p: vec3<f32>, g: f32, jit: f32) -> f32 {
    let headroom = u.bmax.z - p.z;
    if (headroom <= 0.0) {
        return 1.0;
    }
    let span = min(SKY_PROBE_SPAN, headroom);
    let dt = span / f32(SKY_PROBE_STEPS);
    var tau = 0.0;
    for (var i: i32 = 0; i < SKY_PROBE_STEPS; i = i + 1) {
        let t = (f32(i) + 0.5 + jit) * dt;
        tau = tau + sample_sigma(p + vec3<f32>(0.0, 0.0, t)) * dt;
    }
    return diffuse_transmittance(tau, g);
}

// Total optical depth along the view ray (see the AHEAD_* block). One coarse
// pass, run once per pixel before compositing starts, so the main march can
// subtract what it has already crossed and know what is still ahead of it.
//
// Coarse on purpose: the consumer is a two-stream transmittance whose useful
// dynamic range spans two decades of tau, so a 20 percent quadrature error is
// invisible, and the same distance LOD the view march uses keeps far tiles
// cheap. The tau cap ends it as soon as the answer can no longer matter, which
// in the thick views it is meant for happens within a few hundred metres.
fn premarch_tau_ahead(origin: vec3<f32>, dir: vec3<f32>,
                      t0: f32, t1: f32, jit: f32) -> f32 {
    var tau = 0.0;
    let dt_base = u.bmin.w * AHEAD_STEP_SCALE;
    var t = t0 + clamp(jit, 0.0, 1.0) * dt_base;
    for (var i: i32 = 0; i < AHEAD_MAX_STEPS; i = i + 1) {
        if (t >= t1 || tau > AHEAD_TAU_CAP) {
            break;
        }
        let s = sample_level_at(origin + t * dir);
        var dt = max(s.dt_view * AHEAD_STEP_SCALE, t * u.periodic.z);
        if (t + dt > t1) {
            dt = t1 - t;
        }
        tau = tau + s.sigma * dt;
        t = t + dt;
    }
    return tau;
}

// Max view-march distance in a periodic domain. Closed-form inversion of the
// exponential-atmosphere transmittance the in-march aerial perspective uses:
//   tau(t) = beta0 * (H / mu) * (exp(-z0/H) - exp(-z1/H)),  z1 = z0 + mu * t
// solved for tau = PERIODIC_AIR_TAU_CUTOFF / strength (i.e. air_t ~ 2%),
// plus the PERIODIC_MAX_RANGE_M horizontal-travel ceiling. With the aerial
// machinery disabled (strength == 0) only the range ceiling applies — far
// repeats then stay crisp but the march still terminates.
fn periodic_march_cap(cam_z: f32, dir: vec3<f32>) -> f32 {
    var cap = 3.4e38;
    let h_len = length(dir.xy);
    if (h_len > 1e-8) {
        cap = PERIODIC_MAX_RANGE_M / h_len;
    }
    let aerial_strength = u.sky_horizon.w;
    if (aerial_strength > 0.0) {
        let aer_beta0 = u.sky_bloom.w;
        let aer_h = u.sky_disc.w;
        let z0 = max(cam_z, 0.0);
        let mu = dir.z;
        let tau_cap = PERIODIC_AIR_TAU_CUTOFF / aerial_strength;
        if (aer_h <= 0.0) {
            // Uniform haze: tau = beta0 * t, so every ray caps at the same
            // distance whatever its direction — including the upward ones the
            // exponential profile lets escape.
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
            // a <= 0: an upward ray leaves the haze before reaching the
            // cutoff optical depth — the wrap ceiling is the only cap.
        } else {
            cap = min(cap, tau_cap / (aer_beta0 * e0));
        }
    }
    return cap;
}
