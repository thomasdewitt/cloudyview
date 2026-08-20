// CloudyView interactive volume raymarcher (WGSL).
//
// Ray-box entry, hardware-trilinear sampling of resident 3D extinction
// textures, witness procedural sky, per-pixel jittered ray starts, and the
// witness cloud-scattering model (adaptive view/light marches, dt-invariant
// powder, MS octaves, ambient ramp, surface bounce, analytic boundary
// taper, and the witness FIF ocean), ported function by function against the
// numba reference in witness.py (docs/architecture.md).
//
// This is the single raymarch module. The host specializes it by rewriting
// three const declarations before compiling (engine._shader_for):
// PERIODIC_DOMAIN, NESTED, and MAX_LIGHT_STEPS. Every branch on the first
// two folds away at compile time, so an unused feature costs nothing.
//
// Periodic domain: SAM LES fields are doubly periodic in x/y, so the volume
// tiles horizontally — density sampling wraps (wrap_to_domain), the view
// march never exits sideways (z slab + periodic_march_cap), and the light
// march exits only through the domain top.
//
// Nested levels: an optional second, finer field ("the nest") sits inside
// the outer AABB and wins wherever it covers, exactly like
// witness._active_level. The nest tiles WITH the parent — the whole scene is
// one periodic tile — so a wrapped world point is tested against the nest
// box after wrapping, and every domain copy carries a copy of the nest.
//
// Conventions shared with the CPU renderer (cloudyview/witness.py):
// - World space is absolute meters, +x east, +y north, +z up.
// - The volume AABB is cell-edge aligned (half-cell padding around centers).
// - Vertical FOV; pixel row 0 is the top of the image.
// - Tone map: Reinhard + gamma 1.4 (witness.tone_map), applied in-shader.
//
// Texture layout note: the density texture is uploaded with
// width=nz, height=ny, depth=nx so the (nx,ny,nz) C-order numpy array is
// uploaded with zero host-side reshuffling. Normalized sample coords are
// therefore (gz, gy, gx) — see sample_sigma().

struct Uniforms {
    // xyz = camera origin (m), w = tan(fov_horizontal / 2)
    cam_origin: vec4<f32>,
    // xyz = forward, w = aspect (width / height)
    cam_forward: vec4<f32>,
    // xyz = right, w = exposure (Reinhard pre-scale)
    cam_right: vec4<f32>,
    // xyz = up, w = jitter enable (0.0 or 1.0)
    cam_up: vec4<f32>,
    // xyz = unit direction toward the sun, w = frame index (jitter decorrelation)
    sun_dir: vec4<f32>,
    // xyz = volume AABB min (m), w = view-ray step dt (m)
    bmin: vec4<f32>,
    // xyz = volume AABB max (m), w = light-march step dt (m)
    bmax: vec4<f32>,
    // x = image width (px), y = image height (px), z = HG asymmetry g, w = ambient strength
    params: vec4<f32>,
    // x = ocean z (m), yzw = ocean reflectance (witness.py:104-106).
    // Under CITY: x = ground z, yz = city tile world offset (m, a whole
    // number of blocks — see city_glow_sample), w unused.
    ocean: vec4<f32>,
    // x = FIF dx (m), y = FIF tile extent (m), z = ocean enabled, w = max normal LOD
    ocean_params: vec4<f32>,
    // x = subpixel camera-ray jitter enable (0.0 or 1.0),
    // y = jitter amplitude scale (x and y are sampling flags, excluded from
    //     the host's scene-identity key),
    // z = haze in [0, HAZE_MAX] — the user's one aerosol knob, which also
    //     sets row 16.w and row 19.y on the host side;
    // w = tone-map white point (exposed radiance that displays as 1.0)
    flags: vec4<f32>,
    // x = gradient shading, y = deep-shadow MS suppression,
    // z = directional ambient occlusion, w = bounce depth attenuation
    cb_realism: vec4<f32>,
    // x = gradient coarse weight, y = gradient coarse radius (m),
    // z = directional ambient occlusion floor,
    // w = cone stencil tan(theta) (0 = legacy fixed coarse radius)
    cb_params: vec4<f32>,
    // Rows 13-17: witness realism package (per-frame CPU spectral precompute,
    // see witness._spectral_lighting_colors / engine.write_uniforms).
    // xyz = spectral direct-beam color for cloud MS octaves,
    // w = low-sun sky field strength (0 = iter_006 azimuth-only field)
    cloud_sun: vec4<f32>,
    // xyz = spectral ambient tint, w = effective light-transfer split
    // strength (already elevation-faded on the CPU; 0 = legacy shader)
    ambient_tint: vec4<f32>,
    // xyz = spectral sky horizon color, w = aerial perspective strength
    // (0 = exact legacy: no haze extinction, no in-scatter)
    sky_horizon: vec4<f32>,
    // xyz = circumsolar bloom color, w = aerial beta0 (m^-1, sea level)
    sky_bloom: vec4<f32>,
    // xyz = solar disc color, w = aerial haze scale height (m), where
    //     w <= 0 means the haze does NOT thin with height: the sea-level
    //     extinction applies at every altitude. That is unphysical and it is
    //     the point — an exponential atmosphere lets an upward ray leave the
    //     haze without ever reaching the cutoff optical depth, so it marches
    //     to the range ceiling, while a uniform one caps every ray at the
    //     same distance. It is the cheapest range lever there is.
    sky_disc: vec4<f32>,
    // Rows 18-19: ocean realism (witness iter_004/009/011).
    // x = master gate (0 = exact legacy ocean shader), y = mip bias,
    // z = GGX glint strength, w = GGX base roughness
    ocean_realism_a: vec4<f32>,
    // x = ocean sub-pixel slope draw fraction, y = ocean haze extinction
    // (m^-1), z = sky-reflection cloud-shadow floor,
    // w = display contrast about mid-grey (1.0 = identity)
    ocean_realism_b: vec4<f32>,
    // Row 20: periodic domain + distance LOD (2026-07-17 perf pass).
    // x = periodic enable in host scene identity (0.0 or 1.0); the shader
    //     itself branches on the compile-time PERIODIC_DOMAIN const.
    // y = light-march LOD tan(theta): sun-march dt floor grows as
    //     view_distance * y so distant cloud copies get coarser (never
    //     truncated) tau quadrature. 0 = fixed dt.
    // z = view-step LOD tan(theta): view-march dt floor grows as t * z —
    //     the degrees-not-meters step. 0 = fixed dt.
    periodic: vec4<f32>,
    // Rows 21-22: the optional nest (NESTED). Same packing as rows 5-6 but
    // for the finer level: xyz = nest AABB min/max (m), w = the nest's own
    // view-ray / light-march step dt (m). Zero-filled when NESTED is false.
    nest_bmin: vec4<f32>,
    nest_bmax: vec4<f32>,
    // Row 23: the baked sun-tau cache (2026-08-19 prototype, behind a
    // toggle while its worth is decided from A/B stills and benches).
    // x = read tau_sun for CLOUD samples from the light_tau texture instead
    //     of marching to the sun per sample (0.0 or 1.0). The ocean's two
    //     shadow marches and the sky probe stay live either way. The cached
    //     value is the central sun direction at zero quadrature jitter, so
    //     the solar-disc penumbra sampling (iter_007) does not apply to it —
    //     trilinear filtering of the cache is the only softening left, which
    //     is exactly what the A/B is judging.
    // y = bake slice index (the fs_bake_light pass only; fs_main never
    //     reads it): the field-x plane of the cache being rendered.
    // z = skip the vertical sky probe (0.0 or 1.0): every consumer of
    //     t_sky then sees a fully open sky. A cost/look toggle while the
    //     per-tier fate of these marches is decided.
    // w = unused (zero).
    light_cache: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var vol: texture_3d<f32>;
@group(0) @binding(2) var vol_samp: sampler;
@group(0) @binding(3) var ocean_normals: texture_2d<f32>;
@group(0) @binding(4) var ocean_samp: sampler;
// The finer nested level. Always bound (a 1x1x1 zero texture when there is
// no nest) so one bind-group layout serves both specializations.
@group(0) @binding(5) var nest_vol: texture_3d<f32>;
// The same volume texture read the periodic way: repeat on the two lateral
// texture axes (w indexes field x, v indexes field y), clamp on the vertical.
// A doubly periodic domain wraps in hardware rather than through an uploaded
// ghost ring — see sample_level.
@group(0) @binding(6) var vol_wrap_samp: sampler;
// The baked sun-tau cache over the outer AABB — see Uniforms.light_cache and
// sample_light_tau. Always bound (a 1x1x1 zero texture while the cache is
// off or stale) so one bind-group layout serves both paths.
@group(0) @binding(7) var light_tau: texture_3d<f32>;

// Compile-time, rather than dynamically read in hot loops: the host rewrites
// these two declarations per specialization (engine._shader_for), so the
// density, light, and view branches fold away completely.
const PERIODIC_DOMAIN: bool = true;
const NESTED: bool = false;
// Off returns linear HDR radiance instead of a display-referred image, for
// callers that want the physical quantity — render_nested(return_linear=True)
// in Python. The browser always leaves this on: it is presenting to a canvas,
// and this is the only place the encode may happen.
const TONE_MAP: bool = true;

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

const OCEAN_SHADOW_FLOOR: f32 = 0.35; // witness legacy ocean shadow floor

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

// Light-transfer split (witness iter_006). The elevation fade lives on the
// CPU (engine._effective_light_transfer_split); these are the fixed knobs.
const LIGHT_TRANSFER_DIRECT_BOOST: f32 = 0.25;
const LIGHT_TRANSFER_SHADOW_SKYLIGHT: f32 = 0.26;

// EMPTY-SPACE SKIPPING: measured three times, lost three times. Do not read
// the sparseness of cloud fields as an unclaimed opportunity — it is a claim
// that has been tested to destruction, and this note replaces the TODO that
// used to invite it.
//
//   2026-07-17  view-march occupancy skip. 1.2-2.5x SLOWER. Fine-dt empty
//               samples are cheap; the skip machinery is not.
//   2026-08-11  8^3 majorant/occupancy grid on the light march. Proven
//               max|diff| == 0.0 and measured 0.5-0.9x. Machinery cost 47 ms;
//               skipping 36% of bricks recovered 0.8 ms.
//   2026-08-13  full sparse bricks: page table, atlas with a 1-voxel apron,
//               brick-DDA traversal that consults only the page table in
//               vacuum, and a skip that lands on the march's own sample
//               lattice so the picture is unchanged. Proven correct against a
//               dense render (the difference is a Monte Carlo realization and
//               falls as 1/sqrt(frames)). Measured 3-6x SLOWER on an RTX 5080
//               across both a 33%-occupied field and an 8%-occupied one, and
//               slower again on Apple silicon — which was the one hypothesis
//               left standing after the first two, since both of those were
//               5080 texture-latency measurements wearing the costume of a
//               measurement about an algorithm.
//
// The mechanism, consistent across all three: this march is texture-LATENCY
// bound, and every scheme that knows where the empty space is must ask
// something to find out. In occupied space that ask lands on the critical
// path of a loop that tolerates no added control flow (a bit-identical
// micro-opt — reusing the view sample as the light march's first tap — also
// regressed). The skip cannot outrun the question.
//
// Two things worth knowing before a fourth attempt. The one lever left
// untried is hoisting the page fetch to one per brick CROSSING instead of one
// per tap, which needs the march restructured around brick crossings rather
// than around samples. And bricking has a memory break-even that is lower
// than it looks: an 8^3 brick with its apron stores 1000 texels to hold 512,
// so it only shrinks a field below 51% brick occupancy (16^3: 70%, 32^3:
// 83%). The whole attempt is in git — see the commit that reverted it for
// the entry point, and benchmarking/soar_frame_results.md for the numbers.
//
// What DID pay, and shipped, is cropping the empty sky off a field at load:
// same insight about sparseness, applied to storage rather than traversal,
// where nothing has to be asked per sample. See web/soar/zcrop.js.

// The host may bind either r32float (reference/default) or filterable r16float
// density. Both expose texture_3d<f32> here; fp16 only changes storage/bandwidth.
// Both levels always share one format.

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Slab-method ray/AABB intersection. Returns (t_near, t_far); miss when
// t_near > t_far.
fn ray_box(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    let t0 = (u.bmin.xyz - origin) * inv_dir;
    let t1 = (u.bmax.xyz - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
}

// One refinement level's sample, plus the march step sizes that level asks
// for. witness carries the same two things out of _sample_sigma_nested (the
// value and the level index that sets dt_max).
struct LevelSample {
    sigma: f32,
    dt_view: f32,
    dt_light: f32,
    in_nest: bool,
}

// Data dims in FIELD axis order (nx, ny, nz). The texture is stored
// transposed — (w=nz, h=ny, d=nx) — and holds nothing but the field, so this
// is a swizzle and no longer a swizzle-minus-a-border.
fn level_data_dims(t: texture_3d<f32>) -> vec3<f32> {
    let tex_dims = vec3<f32>(textureDimensions(t, 0));
    return vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
}

// Voxel size (m) of whichever level is active at the sample point.
fn level_voxel_size(in_nest: bool) -> vec3<f32> {
    if (NESTED && in_nest) {
        return (u.nest_bmax.xyz - u.nest_bmin.xyz) / level_data_dims(nest_vol);
    }
    return (u.bmax.xyz - u.bmin.xyz) / level_data_dims(vol);
}

// Fold a world point into the outer tile. The scene — nest included — is one
// periodic tile, so this runs before any level test: every domain copy then
// carries a copy of the nest at the same place within the tile.
fn wrap_to_domain(p: vec3<f32>) -> vec3<f32> {
    if (!PERIODIC_DOMAIN) {
        return p;
    }
    let extent = u.bmax.xy - u.bmin.xy;
    let g = fract((p.xy - u.bmin.xy) / extent);
    return vec3<f32>(u.bmin.xy + g * extent, p.z);
}

// Containment test for the already-wrapped point (witness._active_level).
fn in_nest_box(q: vec3<f32>) -> bool {
    if (!NESTED) {
        return false;
    }
    return all(q >= u.nest_bmin.xyz) && all(q <= u.nest_bmax.xyz);
}

// The linear taper a zero ghost texel used to supply, computed instead.
//
// Sampling a zero-bordered volume at data coordinate g gives, along one axis,
// v_0*(g+1) for g in [-1, 0] and v_{N-1}*(N-g) for g in [N-1, N] — a ramp from
// the outermost data value down to zero across the outer half-cell either
// side. Clamp-to-edge on an UNPADDED texture returns exactly those same
// outermost values over that range and holds them beyond it, so multiplying by
// this window reproduces the padded result. Trilinear filtering is separable
// and the border was zero on every axis, so the three windows multiply, which
// makes the identity hold at edges, at corners, and everywhere between.
//
// That is what buys the two texels back: a 2048-cell axis no longer asks a
// browser clamped to the WebGPU spec floor for 2050.
fn edge_taper(g: f32, n: f32) -> f32 {
    return clamp(min(g + 1.0, n - g), 0.0, 1.0);
}

// Extinction (m^-1) from one level via hardware trilinear filtering.
// Coordinate swizzle: texture is (w=nz, h=ny, d=nx), see file header. Witness
// uses gx=(p-bmin)/dx with data value i at gx=i, so texel i IS data value i
// and the normalized coord is (gx+0.5)/N — every pre-existing world/data
// sample preserved, with no border to skip past.
//
// `periodic_level` picks how the domain edge behaves. A doubly periodic outer
// level wraps: the repeat sampler fetches the far face as the trilinear
// partner across the seam, which is what the uploaded ghost ring used to do,
// exactly and for free, corners included. Everything else — every nest, and a
// non-periodic outer level — tapers into zero at its own boundary, which is
// how a nest blends out into the coarse field around it. Either way the
// vertical is never periodic and always tapers.
fn sample_level(t: texture_3d<f32>, q: vec3<f32>,
                bmin: vec3<f32>, bmax: vec3<f32>, periodic_level: bool) -> f32 {
    let tex_dims = vec3<f32>(textureDimensions(t, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    let data_g = ((q - bmin) / (bmax - bmin)) * dims;
    let tex_coord = vec3<f32>(
        data_g.z + 0.5,
        data_g.y + 0.5,
        data_g.x + 0.5
    ) / tex_dims;
    let taper_z = edge_taper(data_g.z, dims.z);
    if (periodic_level) {
        return textureSampleLevel(t, vol_wrap_samp, tex_coord, 0.0).r * taper_z;
    }
    let taper = edge_taper(data_g.x, dims.x) * edge_taper(data_g.y, dims.y)
              * taper_z;
    return textureSampleLevel(t, vol_samp, tex_coord, 0.0).r * taper;
}

// Sun optical depth from the baked cache — the trilinear read that replaces
// light_march_tau for cloud samples when u.light_cache.x is on.
//
// Same coordinate convention as sample_level over the same outer AABB (the
// cache's texel i sits AT data coordinate i, see fs_bake_light), same
// (w=nz, h=ny, d=nx) transposition, same lateral wrap in a periodic domain —
// but NO edge taper: tau is not extinction, and ramping it to zero at the
// domain edge would punch sunlight into the boundary column. Clamp-to-edge
// on the vertical is right at both ends: above the top slice tau is ~0 and
// below the bottom the nearest baked value is the honest answer.
fn sample_light_tau(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    let tex_dims = vec3<f32>(textureDimensions(light_tau, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    let data_g = ((q - u.bmin.xyz) / (u.bmax.xyz - u.bmin.xyz)) * dims;
    let tex_coord = vec3<f32>(
        data_g.z + 0.5,
        data_g.y + 0.5,
        data_g.x + 0.5
    ) / tex_dims;
    if (PERIODIC_DOMAIN) {
        return textureSampleLevel(light_tau, vol_wrap_samp, tex_coord, 0.0).r;
    }
    return textureSampleLevel(light_tau, vol_samp, tex_coord, 0.0).r;
}

// Sample a *chosen* level at an already-wrapped point.
fn sample_sigma_pinned(q: vec3<f32>, in_nest: bool) -> f32 {
    if (NESTED && in_nest) {
        return sample_level(nest_vol, q, u.nest_bmin.xyz, u.nest_bmax.xyz,
                            false);
    }
    return sample_level(vol, q, u.bmin.xyz, u.bmax.xyz, PERIODIC_DOMAIN);
}

// Finest level covering p wins (witness._sample_sigma_nested).
fn sample_sigma(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    return sample_sigma_pinned(q, in_nest_box(q));
}

// Same dispatch, carrying the active level's step sizes back to the caller.
fn sample_level_at(p: vec3<f32>) -> LevelSample {
    let q = wrap_to_domain(p);
    let nested_here = in_nest_box(q);
    var s: LevelSample;
    s.in_nest = nested_here;
    s.sigma = sample_sigma_pinned(q, nested_here);
    if (NESTED && nested_here) {
        s.dt_view = u.nest_bmin.w;
        s.dt_light = u.nest_bmax.w;
    } else {
        s.dt_view = u.bmin.w;
        s.dt_light = u.bmax.w;
    }
    return s;
}

// One gradient tap. `pin` keeps the tap on the caller's level instead of
// re-dispatching — used for the one-voxel fine stencil, where crossing into
// the coarse field mid-stencil would measure the resolution change rather
// than the cloud. The coarse stencil deliberately does NOT pin: its radius
// routinely leaves the nest, and reading the parent field there is both
// cheaper and more honest than reading the nest's ghost-zero taper (witness
// pins at every radius, which puts a spurious edge on the nest boundary).
fn sigma_tap(p: vec3<f32>, pin: bool, pin_nest: bool) -> f32 {
    if (NESTED && pin) {
        return sample_sigma_pinned(wrap_to_domain(p), pin_nest);
    }
    return sample_sigma(p);
}

fn sigma_gradient_at_radius(p: vec3<f32>, h: vec3<f32>,
                            pin: bool, pin_nest: bool) -> vec3<f32> {
    let sxp = sigma_tap(p + vec3<f32>(h.x, 0.0, 0.0), pin, pin_nest);
    let sxm = sigma_tap(p - vec3<f32>(h.x, 0.0, 0.0), pin, pin_nest);
    let syp = sigma_tap(p + vec3<f32>(0.0, h.y, 0.0), pin, pin_nest);
    let sym = sigma_tap(p - vec3<f32>(0.0, h.y, 0.0), pin, pin_nest);
    let szp = sigma_tap(p + vec3<f32>(0.0, 0.0, h.z), pin, pin_nest);
    let szm = sigma_tap(p - vec3<f32>(0.0, 0.0, h.z), pin, pin_nest);
    return vec3<f32>(
        (sxp - sxm) / (2.0 * h.x),
        (syp - sym) / (2.0 * h.y),
        (szp - szm) / (2.0 * h.z)
    );
}

fn sigma_gradient(p: vec3<f32>, sigma: f32,
                  coarse_weight_in: f32,
                  coarse_radius_m: f32,
                  sample_distance_m: f32,
                  cone_stencil_tan_theta: f32,
                  in_nest: bool) -> vec4<f32> {
    // One voxel-size lookup for both stencils: the coarse branch below asked
    // level_voxel_size(in_nest) for the same answer a second time.
    let voxel = level_voxel_size(in_nest);
    let fine_h = voxel * GRADIENT_SHADING_RADIUS_VOXELS;
    let fine_grad = sigma_gradient_at_radius(p, fine_h, true, in_nest);
    let fine_len = length(fine_grad);
    let fine_conf = (
        fine_len * min(min(fine_h.x, fine_h.y), fine_h.z)
    ) / (sigma + 1e-4);

    let coarse_weight = clamp(coarse_weight_in, 0.0, 1.0);
    if (coarse_weight <= 0.0) {
        return vec4<f32>(fine_grad, fine_conf);
    }

    let extent = u.bmax.xyz - u.bmin.xyz;

    // Cone stencil (witness iter_001): a fixed angular radius follows
    // apparent cloud scale — distant samples use a broader world-space
    // normal while nearby samples converge on the fine stencil. An exact
    // zero explicitly selects the legacy fixed-radius path, which keeps
    // its larger minimum-voxel floor and the domain-fraction ceiling.
    var cone_radius_m: f32;
    var coarse_min_voxels: f32;
    if (cone_stencil_tan_theta > 0.0) {
        cone_radius_m = sample_distance_m * cone_stencil_tan_theta;
        coarse_min_voxels = GRADIENT_SHADING_RADIUS_VOXELS;
    } else {
        cone_radius_m = coarse_radius_m;
        coarse_min_voxels = GRADIENT_SHADING_COARSE_MIN_VOXELS;
    }
    var coarse_h = max(vec3<f32>(cone_radius_m), voxel * coarse_min_voxels);
    if (cone_stencil_tan_theta == 0.0) {
        coarse_h = min(
            coarse_h,
            extent * GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION
        );
    }
    coarse_h = max(coarse_h, fine_h);
    let coarse_grad = sigma_gradient_at_radius(p, coarse_h, false, in_nest);
    let coarse_len = length(coarse_grad);
    let coarse_conf = (
        coarse_len * min(min(coarse_h.x, coarse_h.y), coarse_h.z)
    ) / (sigma + 1e-4);

    if (coarse_weight >= 1.0) {
        return vec4<f32>(coarse_grad, coarse_conf);
    }

    let fine_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, fine_conf
    );
    let coarse_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, coarse_conf
    );
    let fine_w = (1.0 - coarse_weight) * fine_gate;
    let coarse_w = coarse_weight * coarse_gate;

    var blended = vec3<f32>(0.0);
    if (fine_len > 1e-12) {
        blended = blended + fine_w * fine_grad / fine_len;
    }
    if (coarse_len > 1e-12) {
        blended = blended + coarse_w * coarse_grad / coarse_len;
    }
    return vec4<f32>(blended, max(fine_conf, coarse_conf));
}

// Henyey-Greenstein phase function.
fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (12.566370614 * pow(max(denom, 1e-6), 1.5));
}

// Per-pixel hash for jittered ray starts (Dave Hoskins style hash12).
// Blue-noise would be better; a white-noise hash is enough to break the
// coherent step-shell banding this exists to kill.
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        hash12(p + vec2<f32>(17.17, 41.93)),
        hash12(p + vec2<f32>(71.31, 11.57))
    );
}

// ---------------------------------------------------------------------------
// Stratified draws over the accumulation (lighting-loop iter_011).
//
// Seven quantities in this shader are estimated by Monte Carlo with one draw
// per pixel per frame: the sub-pixel camera offset, the view march's entry
// phase, the sun march's quadrature phase (iter_003), the sky probe's and the
// forward pre-march's quadrature phases (iter_001, iter_004), the solar-disc
// and aureole direction (iter_007), and the ocean's sub-pixel slope
// (iter_008). Every one of them drew white noise in the frame index, so an
// accumulated still converged at the white-noise rate and kept 1/sqrt(N) of a
// single frame's error. That residue is the speckle grain the blind judge saw.
//
// White noise is the wrong sequence for this. A pixel's N frames are N samples
// of one fixed integral, and a low-discrepancy point set covers its domain
// evenly instead of clumping. What the draws form here is a rank-1 lattice —
// the frame index times a generating vector, wrapped — shifted by a per-pixel
// random offset (a Cranley-Patterson rotation). The generating vector is
// Roberts' generalised golden ratio in ten dimensions, alpha_j = phi_10^-j
// with phi_10 the real root of x^11 = x + 1, one component per stream.
//
// The single measurement that fixed this shape, and it cost two rejected
// implementations to find: **the streams must share one point set.** The
// second version of this iteration gave every stream its own Owen-scrambled
// van der Corput sequence — the textbook 1D construction, and by direct GPU
// audit a perfect one: each pixel got exactly one sample in each 1/64
// interval, per-pixel mean 0.5 to 2.4e-4, no neighbour correlation. It bought
// **nothing** (64-frame noise within 10 percent of white noise on every view).
// Independent per-stream sequences are Latin-hypercube sampling: they
// stratify each margin and leave the joint distribution random, so they can
// only remove the variance that is additive across streams. Almost none of
// this renderer's is — a pixel's radiance depends on the entry phase and the
// sun-march phase and the sub-pixel offset *jointly*. One lattice indexed by
// the frame, with a different component per stream, is low-discrepancy in the
// joint space, and it cuts the 64-frame noise by 2 to 4x on every view.
//
// A Halton sequence (a different prime base per stream) is jointly
// low-discrepancy too and measured within a few percent of the lattice; it is
// not used because its later streams have to draw base 17, 19, 23, ... which
// is visibly the worse end of the sequence at 64 samples, and because it needs
// a digit loop per stream where the lattice needs a multiply-add.
//
// Three properties this construction keeps, all of which matter here:
//
//   * Unbiased. The rotation is uniform, so each individual frame is still a
//     uniform draw and the converged image is exactly the one iterations
//     001-009 produced. This buys convergence, not a different look — measured
//     as a 0.01-level mean shift per view.
//   * A single frame is unchanged in character. The rotation is per pixel and
//     the lattice offset is common to the frame, so within one frame the draws
//     are as decorrelated between neighbouring pixels as they ever were. At
//     frame 0 the lattice term is zero and every stream returns its rotation
//     alone, i.e. exactly the per-pixel white noise it used to draw, so soar's
//     first frame after a camera move is statistically identical.
//   * Progressive. Every prefix of an additive recurrence is well distributed,
//     so this does not depend on knowing N in advance — which matters because
//     soar accumulates for as long as the camera is parked.
//
// The rotations come from an *integer* hash of the pixel and the stream, not
// from the float hash12 taps these streams used to draw. hash12 does not carry
// enough distinct values to be one rotation per pixel: measured on GPU over a
// 256x256 block it produces about 11.5k of them, i.e. ~13.5 bits, so on a
// 960x540 frame roughly forty pixels share each rotation exactly — and pixels
// that share a rotation draw the *identical* 64-frame sequence, which is
// precisely the coherent-shell correlation the entry jitter exists to break.
// It is a latent weakness of the white-noise version too; it only becomes
// load-bearing once the sequence a rotation selects is a fixed lattice rather
// than fresh noise every frame.
//
// pcg2d is the integer hash iter_009 put in the present pass's dither, and for
// the same reason: its negative result there — float hashes are measurably
// biased and coarse on integer input — is this one restated.
fn pcg2d(v_in: vec2<u32>) -> vec2<u32> {
    var v = v_in * 1664525u + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return v;
}

fn strat_seed(pixel: vec2<u32>, stream: u32) -> vec2<u32> {
    return pcg2d(pixel + vec2<u32>(stream * 0x9e3779b9u,
                                   stream * 0x85ebca6bu));
}

// Stream indices: each selects its own per-pixel rotation, so two streams that
// happen to share a lattice component still land on unrelated offsets.
const STRAT_STREAM_SUBPIXEL: u32 = 1u;
const STRAT_STREAM_ENTRY: u32 = 2u;
const STRAT_STREAM_SKY_PROBE: u32 = 3u;
const STRAT_STREAM_SUN_MARCH: u32 = 4u;
const STRAT_STREAM_SUN_CONE: u32 = 5u;
const STRAT_STREAM_OCEAN_SLOPE: u32 = 6u;
const STRAT_STREAM_PREMARCH: u32 = 7u;
const STRAT_STREAM_NEST_SEAM: u32 = 8u;

const STRAT_INV_2P32: f32 = 2.3283064365386963e-10;

// The generating vector: alpha_j = phi_10^-j, phi_10 the real root of
// x^11 = x + 1 (Roberts 2018). Components are assigned to streams in the order
// they are consumed; the ordering is not load-bearing, since every component
// of an R_d vector is an equally good 1D generator and the joint quality is a
// property of the vector as a whole.
const STRAT_ALPHA_1: f32 = 0.9360691110777584;
const STRAT_ALPHA_2: f32 = 0.8762253807139048;
const STRAT_ALPHA_3: f32 = 0.8202075132286352;
const STRAT_ALPHA_4: f32 = 0.7677709178072273;
const STRAT_ALPHA_5: f32 = 0.7186866405431660;
const STRAT_ALPHA_6: f32 = 0.6727403647567018;
const STRAT_ALPHA_7: f32 = 0.6297314752239328;
const STRAT_ALPHA_8: f32 = 0.5894721822305522;
const STRAT_ALPHA_9: f32 = 0.5517867016256194;
const STRAT_ALPHA_10: f32 = 0.5165104872952219;

fn strat1(index: u32, pixel: vec2<u32>, stream: u32, alpha: f32) -> f32 {
    let seed = strat_seed(pixel, stream);
    return fract(f32(index) * alpha + f32(seed.x) * STRAT_INV_2P32);
}

fn strat2(index: u32, pixel: vec2<u32>, stream: u32,
          alpha: vec2<f32>) -> vec2<f32> {
    let seed = strat_seed(pixel, stream);
    return fract(f32(index) * alpha
                 + vec2<f32>(seed) * STRAT_INV_2P32);
}

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
    let radius = tan_max * scale * sqrt(ru);
    let phi = 6.283185307179586 * r.y;
    var d = sun + radius * (cos(phi) * t1 + sin(phi) * t2);
    d.z = max(d.z, SUN_CONE_MIN_Z);
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
        let inv_dir = 1.0 / sun;
        let hit = ray_box(p, inv_dir);
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

// FIF normal sampling follows witness._ocean_wave_normal_fif
// (witness.py lines 417-442): periodic wrap, bilinear interpolation, then
// renormalization. The host uploads the generated nx/ny/nz arrays as RGB.
fn ocean_normal_lod(t_hit: f32, dir: vec3<f32>) -> f32 {
    let pixel_span = 2.0 * max(t_hit, 0.0) * u.cam_origin.w / u.params.y;
    let ocean_span = pixel_span / max(abs(dir.z), 0.03);
    let texel_span = max(ocean_span / u.ocean_params.x, 1.0);
    return clamp(log2(texel_span), 0.0, u.ocean_params.w);
}

fn ocean_wave_normal(world_xy: vec2<f32>, lod: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(ocean_normals, 0));
    let coord = world_xy / u.ocean_params.y + 0.5 / dims;
    let n = textureSampleLevel(ocean_normals, ocean_samp, coord, lod).rgb;
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

// Legacy reflected sky with its fixed glint lobe faded by the master gate
// (witness._reflection_sky_realism). At ocean_realism = 1 the legacy lobe is
// fully off and the GGX glint replaces it.
fn reflection_sky_realism(dir: vec3<f32>, sun: vec3<f32>,
                          legacy_glint_weight: f32) -> vec3<f32> {
    var t = max(0.0, dir.z);
    let one_minus = 1.0 - t;
    t = 1.0 - one_minus * one_minus * one_minus;
    var col = SKY_BASE_HORIZON + (SKY_ZENITH - SKY_BASE_HORIZON) * t;
    let cos_rs = dot(dir, sun);
    if (cos_rs > 0.0 && legacy_glint_weight > 0.0) {
        let glint_w = 0.02;
        let a = glint_w / ((1.0 - cos_rs) + glint_w);
        col = col + legacy_glint_weight * a * vec3<f32>(1.2, 1.0, 0.6);
    }
    return col;
}

// Smith masking for a GGX distribution parameterized by RMS slope.
fn ggx_smith_g1(n_dot_x: f32, alpha_squared: f32) -> f32 {
    let root = sqrt(alpha_squared + (1.0 - alpha_squared) * n_dot_x * n_dot_x);
    return (2.0 * n_dot_x) / (n_dot_x + root);
}

// Slope variance the FIF normal mip chain removes, level by level, measured
// directly on the shipped tile (tools/export_web_assets.py, seed 20260717).
// Entry k is mean(sx^2 + sy^2) at level 0 minus the same quantity at level k,
// where s = n.xy / n.z. Level 0 is the resolved surface and removes nothing;
// by level 6 the 512^2 tile is down to 8x8 texels and essentially the whole
// 0.0908 RMS slope of this sea has been filtered out of the normal.
//
// This is what the microfacet lobe has to put back. It is a *variance*: it
// adds to alpha^2, not to alpha (lighting-loop iter_008), and its total is a
// property of the tile rather than a free parameter — 0.00825 in slope
// variance is alpha 0.0908 of widening, no matter how many levels the mip
// chain has.
const FIF_SLOPE_VARIANCE_REMOVED = array<f32, 10>(
    0.000000, 0.000449, 0.001407, 0.002788, 0.004635,
    0.007078, 0.008090, 0.008211, 0.008242, 0.008249
);

// Interpolated the same way the normal itself is: linearly between the two
// bracketing levels, so roughness and normal always describe the same surface.
fn fif_slope_variance_removed(lod: f32) -> f32 {
    let l0 = clamp(floor(lod), 0.0, 9.0);
    let l1 = min(l0 + 1.0, 9.0);
    let f = clamp(lod - l0, 0.0, 1.0);
    return mix(FIF_SLOPE_VARIANCE_REMOVED[i32(l0)],
               FIF_SLOPE_VARIANCE_REMOVED[i32(l1)], f);
}

// The removed variance is split between two estimators of the same thing.
// `draw_fraction` of it is sampled stochastically (per-axis, hence the 0.5 —
// the table is the total over both axes); the remainder stays analytic, as
// extra width on the microfacet lobe. The endpoints are both exact: at 1 the
// surface is fully sampled, at 0 the sun lobe is exactly as wide as the sea
// it stands for and only the Fresnel/sky nonlinearity goes unresolved. In
// between it is a pure noise-vs-nonlinearity dial with no bias either way.
fn slope_jitter_sigma(lod: f32, draw_fraction: f32) -> f32 {
    return sqrt(0.5 * clamp(draw_fraction, 0.0, 1.0)
                * fif_slope_variance_removed(lod));
}

// Slope variance left for the lobe to carry.
fn slope_lobe_variance(lod: f32, draw_fraction: f32) -> f32 {
    return (1.0 - clamp(draw_fraction, 0.0, 1.0))
           * fif_slope_variance_removed(lod);
}

// One draw from the sub-pixel slope distribution about a filtered normal.
// Gaussian by Box-Muller, which is the right shape here: the slope of a
// filtered Gaussian-ish wave field is Gaussian, and a Gaussian slope density
// is what makes a sun glitter path an elongated ellipse rather than a disc.
fn ocean_slope_sample(n: vec3<f32>, sigma: f32,
                      seed: vec2<f32>) -> vec3<f32> {
    if (sigma <= 0.0) {
        return n;
    }
    let nz = max(n.z, 1e-4);
    let radius = sigma * sqrt(-2.0 * log(max(seed.x, 1e-6)));
    let phi = 6.283185307179586 * seed.y;
    let slope = n.xy / nz + radius * vec2<f32>(cos(phi), sin(phi));
    return normalize(vec3<f32>(slope, 1.0));
}

// One packed FIF normal mip level, sampled the witness way: the half-texel
// offset is per-LEVEL (witness bilinear grid puts texel value i at gx = i),
// so trilinear-in-LOD is done as two explicit level samples + mix +
// renormalize (witness._ocean_wave_normal_mipped) rather than a single
// hardware trilinear fetch whose half-texel offset would only fit level 0.
fn ocean_normal_mip_sample(world_xy: vec2<f32>, level: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(ocean_normals, i32(level)));
    let coord = world_xy / u.ocean_params.y + 0.5 / dims;
    return textureSampleLevel(ocean_normals, ocean_samp, coord, level).rgb;
}

fn ocean_wave_normal_mipped(world_xy: vec2<f32>, lod: f32) -> vec3<f32> {
    let level0 = floor(lod);
    let level1 = min(level0 + 1.0, u.ocean_params.w);
    let f = lod - level0;
    var n = ocean_normal_mip_sample(world_xy, level0);
    if (level1 > level0 && f > 0.0) {
        n = mix(n, ocean_normal_mip_sample(world_xy, level1), f);
    }
    return normalize(n);
}

// Footprint-filtered ocean with microfacet sun glint and path haze
// (witness._ocean_shade_realism).
fn ocean_shade_realism(hit: vec3<f32>, dir: vec3<f32>, sun: vec3<f32>,
                       sun_shadow: vec3<f32>, t_hit: f32,
                       jit: f32, slope_seed: vec2<f32>,
                       slope_jitter: f32) -> vec3<f32> {
    let ocean_realism = u.ocean_realism_a.x;
    let ocean_mip_bias = u.ocean_realism_a.y;
    let glint_strength = u.ocean_realism_a.z;
    let glint_roughness = u.ocean_realism_a.w;
    let slope_draw_fraction = u.ocean_realism_b.x;
    let haze_beta0 = u.ocean_realism_b.y;
    let sky_shadow_floor = u.ocean_realism_b.z;

    // Project one pixel's angular span onto the water; the grazing-angle
    // factor drives the horizon toward coarser mip levels.
    let grazing = max(abs(dir.z), 0.03);
    let pixel_angular_span = 2.0 * u.cam_origin.w / u.params.y;
    let ocean_span = t_hit * pixel_angular_span / grazing;
    let texel_span = max(ocean_span / u.ocean_params.x, 1.0);
    var lod = log2(texel_span) + ocean_mip_bias;
    lod = clamp(lod, 0.0, u.ocean_params.w);
    // Scaling LOD with the master gate keeps intermediate gate values a
    // continuous tuning range (witness does the same).
    lod = lod * ocean_realism;

    // The mip gives the footprint's MEAN normal. Every judge view's water is
    // sub-pixel: at v7's near field one pixel covers ~30 m of a 0.2 m wave
    // field, so the honest statement is not "this facet is flat" but "this
    // pixel contains a slope distribution whose variance the filter removed".
    // Evaluating the shading at the mean normal is wrong for exactly the
    // reason the sea looks the way it does: Fresnel goes as (1 - n.v)^5 and
    // the sky gradient is steep near the horizon, so at grazing incidence the
    // radiance is strongly convex in slope and its mean is not the value at
    // the mean. Draw the missing slope instead of averaging it away, the same
    // move iter_003 made for the sun-march quadrature and iter_007 for the
    // solar disc: one unbiased sample per pixel per frame, accumulated.
    let n = ocean_slope_sample(ocean_wave_normal_mipped(hit.xy, lod),
                               slope_jitter * slope_jitter_sigma(
                                   lod, slope_draw_fraction),
                               slope_seed);

    // Reflect the view direction around the filtered surface normal.
    let vdotn = dot(dir, n);
    var refl = dir - 2.0 * vdotn * n;
    if (refl.z < 0.0) {
        refl.z = -refl.z;
    }
    let legacy_glint_weight = 1.0 - ocean_realism;
    let sky = reflection_sky_realism(refl, sun, legacy_glint_weight);

    // Water Fresnel for the resolved sky reflection.
    let n_dot_v = clamp(-vdotn, 0.0, 1.0);
    let one_minus = 1.0 - n_dot_v;
    let om2 = one_minus * one_minus;
    let view_fresnel = 0.02 + 0.98 * om2 * om2 * one_minus;

    let tau_ocean = light_march_tau(hit, sun_shadow, t_hit * u.periodic.y,
                                    jit);
    let t_sun_ocean = exp(-tau_ocean);
    let n_dot_l = max(0.0, dot(sun, n));

    // Direct-sun GGX glint, tinted by the spectral beam color. Mip filtering
    // removes unresolved slope variance; alpha widening per LOD folds it back
    // into a broader stable highlight instead of point-sample sparkles.
    let glint_weight = ocean_realism * glint_strength;
    var glint = vec3<f32>(0.0);
    var sun_fresnel = 0.02;
    if (glint_weight > 0.0 && n_dot_l > 0.0 && n_dot_v > 1e-8) {
        var h = sun - dir;
        let h_len = length(h);
        if (h_len > 1e-8) {
            h = h / h_len;
            let n_dot_h = dot(n, h);
            if (n_dot_h > 0.0) {
                let v_dot_h = clamp(dot(-dir, h), 0.0, 1.0);
                let one_minus_vh = 1.0 - v_dot_h;
                let vh2 = one_minus_vh * one_minus_vh;
                sun_fresnel = 0.02 + 0.98 * vh2 * vh2 * one_minus_vh;

                // Base roughness is what lives BELOW the tile's own 0.2 m
                // sampling — capillary ripple the FIF field never had — plus
                // whatever share of the filtered-away variance the slope draw
                // did not take. Variances add; roughnesses do not, which is
                // what the old `roughness + per_lod * lod` ramp got wrong (it
                // reached alpha 0.51 at the horizon for a sea whose entire
                // RMS slope is 0.091).
                let alpha = clamp(
                    sqrt(glint_roughness * glint_roughness
                         + slope_lobe_variance(lod, slope_draw_fraction)),
                    0.02, 0.75
                );
                let alpha_squared = alpha * alpha;
                let denom = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
                let d_ggx = alpha_squared
                    / (3.14159265358979 * denom * denom);
                let g_smith = ggx_smith_g1(n_dot_v, alpha_squared)
                    * ggx_smith_g1(n_dot_l, alpha_squared);
                let spec = glint_weight * t_sun_ocean * d_ggx * g_smith
                    * sun_fresnel / (4.0 * n_dot_v);
                glint = spec * u.cloud_sun.xyz;
            }
        }
    }

    // Direct subsurface light uses the complement of the incident Fresnel
    // allocation so the specular sun path does not create energy.
    let energy_weight = min(glint_weight, 1.0);
    let diffuse_partition = 1.0 - energy_weight * sun_fresnel;
    let diff_irr = t_sun_ocean * n_dot_l * 0.3183098861837907
        * diffuse_partition;
    let diffuse = diff_irr * SUN_COLOR * u.ocean.yzw;

    // Per-term cloud shadowing (witness iter_011): the sun terms already
    // carry t_sun_ocean; only the sky-reflection term dims, and only to
    // sky_shadow_floor — under a cloud the water still reflects the mostly
    // unblocked sky dome and keeps its wave texture.
    let sky_shadow = sky_shadow_floor + (1.0 - sky_shadow_floor) * t_sun_ocean;
    var ol = view_fresnel * sky * sky_shadow
        + (1.0 - view_fresnel) * diffuse
        + glint;

    // Ocean-only aerial perspective toward the same angular sky field the
    // cloud haze uses (solar disc excluded), killing the horizon seam.
    let haze_tau = ocean_realism * haze_beta0 * t_hit;
    if (haze_tau > 0.0) {
        let haze = 1.0 - exp(-haze_tau);
        let h_len = length(dir.xy);
        var hdir = dir.xy;
        if (h_len > 1e-8) {
            hdir = dir.xy / h_len;
        }
        let haze_col = sky_radiance(
            vec3<f32>(hdir, 0.0), sun,
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
                        jit: f32, slope_seed: vec2<f32>,
                        slope_jitter: f32) -> vec3<f32> {
    if (u.ocean_realism_a.x == 0.0) {
        return ocean_shade(hit, dir, sun, sun_shadow, t_hit, jit);
    }
    return ocean_shade_realism(hit, dir, sun, sun_shadow, t_hit, jit,
                               slope_seed, slope_jitter);
}

// Highlight shoulder: how far bright regions blend from per-channel
// Reinhard toward a luminance-normalized curve. Per-channel compression
// desaturates highlights toward grey; the luminance-preserving branch
// keeps the beam's chroma, so lit faces read vibrant rather than chalky.
// Gated on luminance so sky and shadow keep the per-channel curve exactly.
const TONE_MAP_SHOULDER: f32 = 0.35;

// Reinhard + gamma, matching radiative_transfer.tone_map (lines 675-680)
// with witness's default exposure=4.0 (witness.py lines 955-959, 1072-1073).
// Gamma arrives per-frame (u.periodic.w) rather than as the GAMMA const:
// it is the one knob that decides how much the far field lifts, and it is
// the only place the encode may happen — the swapchain must be a plain
// unorm format, never *-srgb, or this runs a second time.
// Extended Reinhard: plain e/(1+e) crushes lit cloud faces into a grey
// ceiling (a face at exposed radiance ~4.7 displays at 227/255 and cannot
// move). The white point (u.flags.w, user slider) is the exposed radiance
// that reaches 1.0, so sunlit faces read vibrant white; below ~10% of it
// the curve is within 0.3% of plain Reinhard, so sky and shadow are
// untouched. Contrast (u.ocean_realism_b.w) pivots the encoded value on
// mid-grey; at 1.0 the multiply-add is the exact identity.

fn tone_map(hdr: vec3<f32>, exposure: f32, gamma: f32) -> vec3<f32> {
    let exposed = hdr * exposure;
    let wp = u.flags.w;
    let w2 = wp * wp;
    let per_channel = exposed * (1.0 + exposed / w2) / (1.0 + exposed);
    let y = dot(exposed, vec3<f32>(0.2126, 0.7152, 0.0722));
    let chroma_preserving = exposed * (1.0 + y / w2) / (1.0 + y);
    let k = TONE_MAP_SHOULDER * smoothstep(1.0, 3.0, y);
    let mapped = mix(per_channel, chroma_preserving, k);
    let encoded = pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)),
                      vec3<f32>(1.0 / gamma));
    let c = u.ocean_realism_b.w;
    return clamp(vec3<f32>(0.5) + (encoded - vec3<f32>(0.5)) * c,
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// The night city (CITY specialization, 2026-08-20)
// ---------------------------------------------------------------------------
//
// A procedurally generated city of skyscrapers under the cloud field, at
// night: the sun becomes a crescent moon, and the buildings are the light.
//
// Reuses the ocean tile binding: under CITY the rgba16float mip chain at
// binding 3 is the CITY density tile (cloudyview/soar/city), one texel per
// city block. r = the FIF cascade (H=0, C1=0.1, alpha=2) normalized by its
// 99.5th percentile — the building-height modulator; g = the same field
// rank-transformed to uniform [0,1] — the thresholdable channel (building
// presence). The coarse mips of r ARE the ground-glow field: mip k is the
// mean lit density over 2^k blocks, which is what a cloud base or a long
// sightline integrates over. Rows 8-9 keep their geometric meaning (surface
// z; tile texel size / extent / enable / max lod); the five spectral rows
// arrive packed with night values by the host, and u.sun_dir is the moon.
//
// Geometry is exact — a 2D DDA over the block grid with slab tests against
// up to three stacked boxes per building plus a rooftop mast — because
// geometry is sacred. The lighting is where the hacks live, as usual:
// procedural window grids with an analytic far-field average (the facade's
// mean emission, blended in as the pixel footprint outgrows a window — the
// degrees-not-meters principle applied to architecture), streetlight pools
// with the same treatment, and an exponential ground-fog whose in-scatter
// is the glow mip at the hit point.

const CITY: bool = false;

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
// Windows.
const CITY_WIN_PITCH_U: f32 = 3.4;  // horizontal window pitch (m)
const CITY_WIN_U_LO: f32 = 0.14;    // pane bounds within the pitch cell
const CITY_WIN_U_HI: f32 = 0.86;
const CITY_WIN_V_LO: f32 = 0.28;
const CITY_WIN_V_HI: f32 = 0.90;
const CITY_PANE_FRAC: f32 = 0.446;  // (U_HI-U_LO)*(V_HI-V_LO), for the mean
const CITY_DARK_FLOOR_FRAC: f32 = 0.10; // whole storeys gone dark
const CITY_WIN_RADIANCE: f32 = 3.5;
// Mean of the palette draw below, for the far-field facade average.
const CITY_PALETTE_MEAN: vec3<f32> = vec3<f32>(1.0, 0.55, 0.25);
const CITY_WIN_BRIGHT_MEAN: f32 = 0.875; // E[0.25 + 5 w^7]
// Tone-map compensation per far-field octave. Each LOD average is taken in
// linear radiance, but the display transform is compressive: a few blazing
// windows tone-map to little more than their dim neighbours, so an
// uncompensated mean reads far brighter than the resolved facade it
// replaces — mid-distance towers rendered cream-white. The right number is
// the ratio <T(E)>/T(<E>) of the population inside the footprint, which
// grows toward 1 as the octave's own variance hands the clipping down to
// it: the block octave keeps bright members (heavy compensation), the flat
// asymptote keeps none (mild).
const CITY_MEAN_COMP_BLOCK: f32 = 0.20;
const CITY_MEAN_COMP_FLAT: f32 = 0.10;
// Where blocks dissolve into the flat asymptote (m of pixel footprint;
// the window->block hand-off is CITY_WIN_LOD_*).
const CITY_BLOCK_LOD_START: f32 = 9.0;
const CITY_BLOCK_LOD_FULL: f32 = 30.0;
// Facade LOD: pixel footprint (m) where the window grid starts and finishes
// dissolving into the facade average.
const CITY_WIN_LOD_START: f32 = 1.6;
const CITY_WIN_LOD_FULL: f32 = 6.5;
// Streets.
const CITY_LAMP_SPACING: f32 = 26.0;
const CITY_LAMP_OFFSET: f32 = 2.5;   // lamp line inset from the block edge
const CITY_LAMP_SIGMA: f32 = 4.5;    // light-pool radius on the asphalt
const CITY_LAMP_RADIANCE: f32 = 0.7;
const CITY_LAMP_COLOR: vec3<f32> = vec3<f32>(1.0, 0.42, 0.12); // sodium
// Mean lamp-pool coverage of the street band: 2*pi*sigma^2 over one lamp's
// street area (spacing * ~2*(half+offset)), times two lamp lines.
const CITY_STREET_MEAN_POOL: f32 = 0.08;
const CITY_STREET_LOD_START: f32 = 8.0;
const CITY_STREET_LOD_FULL: f32 = 26.0;
// Storefront strip: the bottom of a lit building's facade.
const CITY_STORE_H: f32 = 4.4;
const CITY_STORE_RADIANCE: f32 = 2.2;
// Roof furniture.
const CITY_MAST_MIN_H: f32 = 150.0;  // buildings shorter than this: no mast
const CITY_MAST_CROSS: f32 = 1.4;
const CITY_BEACON_LEN: f32 = 2.5;
const CITY_BEACON_COLOR: vec3<f32> = vec3<f32>(40.0, 2.0, 1.6);
// Night materials and fill light.
const CITY_MOONLIGHT: vec3<f32> = vec3<f32>(0.020, 0.024, 0.032);
const CITY_SKYGLOW: vec3<f32> = vec3<f32>(0.012, 0.010, 0.016);
const CITY_FACADE_ALBEDO: f32 = 0.030;
const CITY_ROOF_ALBEDO: f32 = 0.085;
const CITY_ASPHALT_ALBEDO: f32 = 0.035;
// Ground fog: exponential profile, uniform-in-z beyond its scale height
// none — this is the night twin of the aerial-perspective machinery, kept
// separate because its in-scatter source is the city itself, not the sky.
const CITY_FOG_BETA: f32 = 6.0e-5;   // m^-1 at ground level
const CITY_FOG_H: f32 = 320.0;       // scale height (m)
const CITY_FOG_GLOW: vec3<f32> = vec3<f32>(1.0, 0.42, 0.15);
const CITY_FOG_GLOW_AMP: f32 = 0.10;
const CITY_FOG_BASE: vec3<f32> = vec3<f32>(0.003, 0.004, 0.008);
// Cloud uplight: the city as the cloud field's second light source. The
// glow mip under the sample (footprint ~ its height, so the mip level is
// log2(z / cell)) times the two-stream transmittance of the cloud below.
const CITY_UPLIGHT_STRENGTH: f32 = 0.5;
// Only glow above the citywide background lifts a cloud base: the bias is
// what keeps sprawl bases moonlit blue-grey while the district overhead
// burns amber. Without it even a 0.02-radiance warm pedestal reads, because
// the night clouds it lands on sit near 0.05.
const CITY_UPLIGHT_GLOW_BIAS: f32 = 0.15;
const CITY_UPLIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.38, 0.10);
// The glow feeding the uplight saturates: g/(HALF+g). Over the megatower
// district the raw mip runs to ~3x the p99.5 level, and any linear scaling
// bright enough to warm an ordinary district's cloud base bleaches the
// district's whole sky cream — the saturation is what lets both exist.
const CITY_UPLIGHT_GLOW_HALF: f32 = 0.6;
const CITY_UP_PROBE_SPAN: f32 = 1400.0;
const CITY_UP_PROBE_STEPS: i32 = 12;
const CITY_UPLIGHT_FLOOR: f32 = 0.02;
// e-folding of the amber in probe tau: 1/0.25 = 4 optical depths of skin.
const CITY_UPLIGHT_TAU_SCALE: f32 = 0.25;
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
// markers below. The core calls five hooks; the composer generates their
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
//   cc_component_shade   radiance at a component hit (kind >= 100; each
//                        component owns kind_base .. kind_base+15).
//   cc_facade_detail     additive facade emission (balconies, signage,
//                        fire escapes) on top of the window ladder.
//   cc_window_glyph      transmission multiplier inside a lit pane
//                        (curtains, figures, androids) — 1 leaves the
//                        window exactly as it was.
//
// Component hits reuse CityHit with kind >= 100. See components/SPEC.md for
// the contract agents write against.

const CITY_PROP_RANGE: f32 = 2200.0;

fn city_u01(x: u32) -> f32 {
    return f32(x) * 2.3283064365386963e-10; // 1 / 2^32
}

fn city_rand4(s: vec2<u32>) -> vec4<f32> {
    let a = pcg2d(s);
    let b = pcg2d(a);
    return vec4<f32>(city_u01(a.x), city_u01(a.y),
                     city_u01(b.x), city_u01(b.y));
}

fn city_is_avenue(k: i32) -> bool {
    return ((k % CITY_AVENUE_PERIOD) + CITY_AVENUE_PERIOD)
           % CITY_AVENUE_PERIOD == 0;
}

// The glow field: mean lit density over 2^lod blocks, read with the repeat
// sampler so the tile wraps. This is the one number the fog, the sky dome
// and the cloud uplight all share.
// Where the tile sits in world space rides in u.ocean.yz (meters, a whole
// number of cells): the cascade's wildest excursion — a 5 km twin spire
// ringed by 3.7-3.9 km towers — is placed under each field's central
// cameras rather than left wherever the seed put it. The megatower district
// is the shot, so it gets the stage, whatever the domain.
fn city_glow_sample(xy: vec2<f32>, lod: f32) -> f32 {
    let uv = (xy - u.ocean.yz) / u.ocean_params.y;
    let l = clamp(lod, 0.0, u.ocean_params.w);
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
    // One building can own a 2x2 block group in a dense district; every
    // member cell then reports the same merged plot and seed.
    merged: bool,
}

fn city_cell(ci: vec2<i32>) -> CityCell {
    var c: CityCell;
    let cell = u.ocean_params.x;
    let dims = textureDimensions(ocean_normals, 0);
    let n = vec2<i32>(i32(dims.x), i32(dims.y));
    let off_cells = vec2<i32>(floor(u.ocean.yz / cell + vec2<f32>(0.5)));

    // Superblocks: in a dense district a 2x2 group of blocks is sometimes
    // one building — the internal streets are swallowed and all four cells
    // report the same plot, seed and cascade values, so the DDA assembles
    // one seamless structure from whichever member it happens to visit.
    // The group decision reads the ANCHOR cell's texel, so members agree.
    let anchor = ci & vec2<i32>(-2);
    let wa = (((anchor - off_cells) % n) + n) % n;
    let ta = textureLoad(ocean_normals, wa, 0);
    let gseed = pcg2d(vec2<u32>(
        bitcast<u32>(anchor.x) ^ 0x51ed270bu, bitcast<u32>(anchor.y)));
    let gh = city_rand4(gseed);
    c.merged = ta.g > 0.50 && gh.x < 0.10;

    var base = ci;
    if (c.merged) {
        base = anchor;
        c.density = ta.r;
        c.rank = ta.g;
        c.seed = gseed;
    } else {
        let w = (((ci - off_cells) % n) + n) % n;
        let texel = textureLoad(ocean_normals, w, 0);
        c.density = texel.r;
        c.rank = texel.g;
        c.seed = pcg2d(vec2<u32>(bitcast<u32>(ci.x), bitcast<u32>(ci.y)));
    }

    // The plot: one cell's, or the whole group's. Street half-widths on the
    // outer edges; avenue boundaries get extra. The avenue test uses the
    // unwrapped index so the pattern is a property of world space, not of
    // the tile.
    let span = select(1, 2, c.merged);
    let cmin = vec2<f32>(base) * cell;
    let cmax = vec2<f32>(base + vec2<i32>(span)) * cell;
    let hx0 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.x));
    let hx1 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.x + span));
    let hy0 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.y));
    let hy1 = CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(base.y + span));
    c.plot_min = cmin + vec2<f32>(hx0, hy0);
    c.plot_max = cmax - vec2<f32>(hx1, hy1);
    let r4 = city_rand4(c.seed);
    let r4b = city_rand4(c.seed ^ vec2<u32>(0x9e3779b9u, 0x85ebca6bu));
    let r4c = city_rand4(c.seed ^ vec2<u32>(0xdeadbeefu, 0x41c64e6du));

    c.built = c.rank > CITY_EMPTY_RANK;
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

    // Height: base + the cascade raised to a power, jittered, suppressed
    // toward the outskirts, and quantized to storeys so roofs are flat
    // shelves rather than a continuum.
    let sprawl = mix(
        CITY_SPRAWL_MIN_FRAC, 1.0,
        smoothstep(CITY_EMPTY_RANK, CITY_SPRAWL_RANK_FULL, c.rank)
    );
    var h = (CITY_H_BASE + CITY_H_SCALE * pow(c.density, CITY_H_EXP))
            * (0.70 + 0.60 * r4.x) * sprawl;
    h = clamp(floor(h / CITY_FLOOR_H) * CITY_FLOOR_H,
              2.0 * CITY_FLOOR_H, CITY_MAX_H);
    c.height = h;

    // Footprint inside the plot, jittered in size and position. Supertalls
    // slim down — a 2 km tower on a full-plot footprint is a wall, and the
    // structural taper is what makes the megatowers read as spires.
    let plot_size = c.plot_max - c.plot_min;
    let slender = mix(1.0, 0.55, smoothstep(300.0, 2000.0, h));
    let fw = plot_size * slender
        * vec2<f32>(0.55 + 0.40 * r4.y, 0.55 + 0.40 * r4.z);
    let slack = plot_size - fw;
    let bmin2 = c.plot_min + slack * vec2<f32>(r4.w, r4b.x);
    let bmax2 = bmin2 + fw;
    let bc = 0.5 * (bmin2 + bmax2);

    // The architecture. Low buildings stay slabs; above 60 m the draw picks
    // among four archetypes, and the megatowers lean hard toward the two
    // frustum forms — a 3 km rectangular extrusion is a wall, a 3 km
    // tapering shaft is a spire (Thomas: not perfect rectangular prisms;
    // architectures that grow, or have a spire).
    let r4d = city_rand4(c.seed ^ vec2<u32>(0x27d4eb2fu, 0x165667b1u));
    if (h > 60.0) {
        if (r4d.x < 0.20) {
            c.arch = 2;                   // growth: cantilevered buds
        } else if (r4d.x < 0.42) {
            c.arch = 3;                   // tapered shaft on a podium
        } else if (r4d.x < 0.55) {
            c.arch = 4;                   // spire crown
        }
    }
    if (h > 600.0 && r4d.y < 0.65) {
        c.arch = select(3, 4, r4d.y < 0.30);
    }

    if (c.arch == 3) {
        // Podium + a shaft whose cross-section closes toward the crown.
        let zp = max(floor(h * 0.12 / CITY_FLOOR_H) * CITY_FLOOR_H,
                     CITY_FLOOR_H);
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, zp);
        let fw_shaft = fw * 0.90;
        c.has_frustum = true;
        c.fmin = vec3<f32>(bc - 0.5 * fw_shaft, zp);
        c.fmax = vec3<f32>(bc + 0.5 * fw_shaft, h);
        c.fscale = 0.55 + 0.25 * r4d.z;
    } else if (c.arch == 4) {
        // A straight shaft to three quarters, then a steep spire.
        let zs = floor(h * 0.75 / CITY_FLOOR_H) * CITY_FLOOR_H;
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, zs);
        c.has_frustum = true;
        c.fmin = vec3<f32>(bmin2, zs);
        c.fmax = vec3<f32>(bmax2, h);
        c.fscale = 0.12 + 0.10 * r4d.z;
    } else if (c.arch == 2) {
        // Growth: a straight tower with one or two boxes budding from its
        // upper flanks, cantilevered over the street. Buds stay inside the
        // cell (or merged group) so the DDA's per-cell testing stays exact.
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, h);
        c.tiers = 2;
        let side = i32(floor(r4d.z * 4.0));   // which flank the bud rides
        let bz0 = floor(h * (0.42 + 0.20 * r4d.w) / CITY_FLOOR_H)
                  * CITY_FLOOR_H;
        let bz1 = min(bz0 + floor((0.10 + 0.10 * r4b.y) * h / CITY_FLOOR_H)
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
        // Clamped to the cell column minus a hair, so no prim leaves it.
        bud_min = max(bud_min, cmin + vec2<f32>(0.5));
        bud_max = min(bud_max, cmax - vec2<f32>(0.5));
        c.b2min = vec3<f32>(bud_min, bz0);
        c.b2max = vec3<f32>(bud_max, bz1);
        if (h > 200.0 && r4b.y > 0.55) {
            c.tiers = 3;
            let cz0 = floor(h * (0.68 + 0.12 * r4b.w) / CITY_FLOOR_H)
                      * CITY_FLOOR_H;
            let cz1 = min(cz0 + 8.0 * CITY_FLOOR_H, h - CITY_FLOOR_H);
            // The second bud rides the opposite flank: mirror about bc.
            let m0 = 2.0 * bc - bud_max;
            let m1 = 2.0 * bc - bud_min;
            c.b3min = vec3<f32>(
                max(min(m0, m1), cmin + vec2<f32>(0.5)), cz0);
            c.b3max = vec3<f32>(
                min(max(m0, m1), cmax - vec2<f32>(0.5)), cz1);
        }
    } else {
        // Setback tiers: the wedding cake (the founding archetype).
        var z1 = h;
        if (h > 90.0 && r4b.y > 0.45) {
            c.tiers = 2;
            z1 = floor(h * 0.60 / CITY_FLOOR_H) * CITY_FLOOR_H;
            if (h > 180.0 && r4b.y > 0.80) {
                c.tiers = 3;
                z1 = floor(h * 0.50 / CITY_FLOOR_H) * CITY_FLOOR_H;
            }
        }
        c.b1min = vec3<f32>(bmin2, CITY_GROUND_Z);
        c.b1max = vec3<f32>(bmax2, z1);
        if (c.tiers >= 2) {
            var z2 = h;
            var f2 = 0.68;
            if (c.tiers == 3) {
                z2 = floor(h * 0.78 / CITY_FLOOR_H) * CITY_FLOOR_H;
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

    if (h > CITY_MAST_MIN_H && r4b.z < 0.65) {
        c.has_mast = true;
        let mast_h = 12.0 + 45.0 * r4b.w;
        let moff = (vec2<f32>(r4c.w, r4.x) - 0.5) * fw * 0.4;
        let mc = bc + moff;
        c.mast_min = vec3<f32>(mc - vec2<f32>(0.5 * CITY_MAST_CROSS), h);
        c.mast_max = vec3<f32>(mc + vec2<f32>(0.5 * CITY_MAST_CROSS),
                               h + mast_h);
        c.top_z = h + mast_h;
    } else {
        c.top_z = h;
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
// clears. Opaque geometry, so the first hit wins.
fn city_trace(o: vec3<f32>, dir: vec3<f32>) -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let cell = u.ocean_params.x;

    var t0 = 0.0;
    if (o.z > CITY_SLAB_TOP) {
        if (dir.z >= -1e-7) {
            return res;
        }
        t0 = (CITY_SLAB_TOP - o.z) / dir.z;
        if (t0 > CITY_TRACE_RANGE) {
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
    let extra = cc_extra_trace(o, dir, inv3);

    var t_cur = t0;
    for (var i: i32 = 0; i < CITY_TRACE_MAX_CELLS; i = i + 1) {
        let t_exit = min(min(tmax.x, tmax.y), CITY_TRACE_RANGE);
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
        if (cc.built && min(z_a, z_b) < cc.top_z + 1.0) {
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
            if (cc.has_mast) {
                let hm = city_box_hit(o, inv3, cc.mast_min, cc.mast_max);
                if (hm.x <= hm.y && hm.x > 0.0 && hm.x < best_t) {
                    best_t = hm.x; best_min = cc.mast_min;
                    best_max = cc.mast_max;
                    best_kind = 3;
                }
            }
        }
        if (ground_t > t_cur - 1e-4 && ground_t <= t_exit + 1e-4
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
        if (t_cur < CITY_PROP_RANGE) {
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

        if (t_exit >= CITY_TRACE_RANGE) {
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

// Streetlight pools on the asphalt around block edges. Evaluated for the
// nearest x- and y- boundary lines; each street has a lamp line either side.
fn city_street_pools(p: vec2<f32>) -> f32 {
    let cell = u.ocean_params.x;
    var pool = 0.0;
    // Nearest x-boundary (a street running along y).
    let bx = round(p.x / cell) * cell;
    let bxi = i32(round(p.x / cell));
    let ax = select(1.0, 3.0, city_is_avenue(bxi));
    for (var s: i32 = 0; s < 2; s = s + 1) {
        let lx = bx + select(-CITY_LAMP_OFFSET, CITY_LAMP_OFFSET, s == 1);
        let ly = round(p.y / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
        let d2 = (p.x - lx) * (p.x - lx) + (p.y - ly) * (p.y - ly);
        pool = pool + ax * exp(-d2 / (2.0 * CITY_LAMP_SIGMA * CITY_LAMP_SIGMA));
    }
    // Nearest y-boundary (a street running along x).
    let by = round(p.y / cell) * cell;
    let byi = i32(round(p.y / cell));
    let ay = select(1.0, 3.0, city_is_avenue(byi));
    for (var s: i32 = 0; s < 2; s = s + 1) {
        let ly = by + select(-CITY_LAMP_OFFSET, CITY_LAMP_OFFSET, s == 1);
        let lx = round(p.x / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
        let d2 = (p.x - lx) * (p.x - lx) + (p.y - ly) * (p.y - ly);
        pool = pool + ay * exp(-d2 / (2.0 * CITY_LAMP_SIGMA * CITY_LAMP_SIGMA));
    }
    return pool;
}

// Surface radiance at a city hit, before fog. fp is the pixel footprint at
// the hit (m/px): the analytic LOD that dissolves windows and lamp pools
// into their means once they are sub-pixel, so the far city is smooth
// glow rather than shimmer.
fn city_shade(h: CityHit, dir: vec3<f32>, fp: f32) -> vec3<f32> {
    let moon = u.sun_dir.xyz;
    let cc = city_cell(h.cell);
    let local_glow = city_glow_sample(h.pos.xy, 3.0);
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * local_glow);

    if (h.kind >= 100) {   // a component's hit; its shader owns the look
        return cc_component_shade(h, cc, dir, fp);
    }
    if (h.kind == 4) {   // beacon
        return CITY_BEACON_COLOR;
    }
    if (h.kind == 3) {   // mast steel
        return 0.05 * fill
               + 0.04 * CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    }
    if (h.kind == 2) {   // roof
        var alb = CITY_ROOF_ALBEDO;
        // Parapet ring: the roof edge catches the street glow from below.
        // The ring belongs to whichever tier this roof is — pick the box
        // whose top the hit sits on.
        var rmin = cc.b1min.xy;
        var rmax = cc.b1max.xy;
        if (cc.tiers >= 2 && h.pos.z > cc.b2min.z + 0.5) {
            rmin = cc.b2min.xy;
            rmax = cc.b2max.xy;
        }
        if (cc.tiers >= 3 && h.pos.z > cc.b3min.z + 0.5) {
            rmin = cc.b3min.xy;
            rmax = cc.b3max.xy;
        }
        let din = min(
            min(h.pos.x - rmin.x, rmax.x - h.pos.x),
            min(h.pos.y - rmin.y, rmax.y - h.pos.y)
        );
        if (din < 1.2) {
            alb = alb * 2.2;
        }
        return alb * (fill + CITY_MOONLIGHT * max(moon.z, 0.0));
    }
    if (h.kind == 1) {   // facade
        // Normalized horizontal tangent: a frustum face's normal carries a
        // z tilt, and an unnormalized tangent would stretch the windows.
        let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
        let tangent = vec2<f32>(-nh.y, nh.x);
        let uc = dot(h.pos.xy, tangent);
        let vc = h.pos.z;

        // Per-window pattern.
        let iu = i32(floor(uc / CITY_WIN_PITCH_U));
        let iv = i32(floor(vc / CITY_FLOOR_H));
        let wh = city_rand4(vec2<u32>(
            cc.seed.x ^ bitcast<u32>(iu),
            cc.seed.y ^ bitcast<u32>(iv)
        ));
        let fh = city_rand4(vec2<u32>(
            cc.seed.x ^ 0x2545f491u,
            bitcast<u32>(iv) * 0x9e3779b9u
        ));
        let floor_dark = fh.x < CITY_DARK_FLOOR_FRAC;
        let fu = fract(uc / CITY_WIN_PITCH_U);
        let fv = fract(vc / CITY_FLOOR_H);
        let pane = fu > CITY_WIN_U_LO && fu < CITY_WIN_U_HI
                && fv > CITY_WIN_V_LO && fv < CITY_WIN_V_HI;
        let lit = pane && !floor_dark && wh.x < cc.lit_frac;
        var e_win = vec3<f32>(0.0);
        if (lit) {
            let bright = 0.25 + 5.0 * pow(wh.z, 7.0);
            e_win = city_window_color(wh.y, cc.palette_bias)
                    * (CITY_WIN_RADIANCE * bright);
            // Life inside: whatever the glyph components put between the
            // light and the glass (curtains, figures, androids).
            let pane_uv = vec2<f32>(
                (fu - CITY_WIN_U_LO) / (CITY_WIN_U_HI - CITY_WIN_U_LO),
                (fv - CITY_WIN_V_LO) / (CITY_WIN_V_HI - CITY_WIN_V_LO));
            e_win = e_win * cc_window_glyph(cc, wh, pane_uv, fp);
        }

        // Far field: an octave ladder rather than one mean. A single flat
        // mean fails twice — averaged in linear radiance ahead of a
        // compressive tone map it renders cream, and dimmed enough not to,
        // it renders as a lightless brown slab whose windows visibly
        // vanish at one distance. So the windows dissolve the way they
        // were built, scale by scale: individual panes into 4x4-window
        // BLOCKS (hashed, a few bright, most dim — the speckle character
        // survives past the point where any one pane does, and a rare
        // block keeps a neon accent), and blocks into the flat asymptote,
        // which is the palette mean at the occupancy the statistics say.
        // Each coarser octave carries its own tone-map compensation, so
        // the display-space brightness holds across both hand-offs.
        let e_mean_base = CITY_PALETTE_MEAN
            * (cc.lit_frac * CITY_PANE_FRAC * (1.0 - CITY_DARK_FLOOR_FRAC)
               * CITY_WIN_RADIANCE * CITY_WIN_BRIGHT_MEAN);

        // Octave 1: 4x4-window blocks.
        let ibu = iu >> 2;
        let ibv = iv >> 2;
        let bh1 = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(ibu) * 0x85ebca6bu),
            cc.seed.y ^ (bitcast<u32>(ibv) * 0xc2b2ae35u)
        ));
        var block_color = CITY_PALETTE_MEAN;
        if (bh1.y < 0.03) {
            block_color = city_window_color(
                0.90 + 0.09 * bh1.z, cc.palette_bias);
        }
        let block_var = 0.10 + 1.8 * pow(bh1.x, 5.0);
        let e_block = block_color
            * (cc.lit_frac * CITY_PANE_FRAC * (1.0 - CITY_DARK_FLOOR_FRAC)
               * CITY_WIN_RADIANCE * CITY_WIN_BRIGHT_MEAN
               * CITY_MEAN_COMP_BLOCK * block_var);

        // Octave 2: the flat asymptote, softly banded per 16 floors.
        let bh2 = city_rand4(vec2<u32>(
            cc.seed.x ^ 0x51ed270bu,
            bitcast<u32>(iv >> 4) * 0x9e3779b9u
        ));
        let band = 0.70 + 0.60 * bh2.x;
        let e_flat = e_mean_base * (CITY_MEAN_COMP_FLAT * band);

        let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);
        let b1 = smoothstep(CITY_WIN_LOD_START, CITY_WIN_LOD_FULL, fp_eff);
        let b2 = smoothstep(CITY_BLOCK_LOD_START, CITY_BLOCK_LOD_FULL,
                            fp_eff);
        var e = mix(e_win, mix(e_block, e_flat, b2), b1);

        // Street-level storefront strip: bright, saturated, and only on
        // buildings whose draw says the ground floor is commercial.
        if (vc < CITY_STORE_H && cc.store_draw < 0.65) {
            let sc = city_window_color(
                fract(cc.store_draw * 7.31), 1.0 - cc.palette_bias);
            e = e + sc * CITY_STORE_RADIANCE
                    * smoothstep(CITY_STORE_H, CITY_STORE_H * 0.4, vc);
        }

        // Whatever the facade-layer components add: balconies, signage,
        // fire escapes.
        e = e + cc_facade_detail(cc, h, uc, vc, fp);

        return e + CITY_FACADE_ALBEDO * fill
               + CITY_FACADE_ALBEDO * CITY_MOONLIGHT
                 * max(dot(h.normal, moon), 0.0);
    }

    // Ground. Streets carry sodium lamp pools; unbuilt lots stay dark.
    // Lamp output follows the district: a downtown artery blazes, an
    // outskirts street is a sparse dim line — the multifractal again, so
    // the lattice is not wallpaper.
    let inside_plot = h.pos.x > cc.plot_min.x && h.pos.x < cc.plot_max.x
                   && h.pos.y > cc.plot_min.y && h.pos.y < cc.plot_max.y;
    var street = 0.0;
    if (!inside_plot || !cc.built) {
        let district = city_glow_sample(h.pos.xy, 2.0);
        let street_scale = 0.20 + 2.2 * smoothstep(0.02, 0.45, district);
        let pool_blend = smoothstep(
            CITY_STREET_LOD_START, CITY_STREET_LOD_FULL, fp);
        let pools = mix(
            city_street_pools(h.pos.xy), CITY_STREET_MEAN_POOL, pool_blend);
        street = select(0.0,
                        CITY_LAMP_RADIANCE * street_scale * pools,
                        !inside_plot);
    }
    return CITY_LAMP_COLOR * street
           + CITY_ASPHALT_ALBEDO * (fill + CITY_MOONLIGHT * max(moon.z, 0.0));
}

// Night ground-fog: extinction toward the hit plus in-scattered city glow.
// The glow source is the mip under the midpoint of the path, coarsened with
// distance — the fog over downtown is orange, the fog over the outskirts is
// barely there.
fn city_fog(radiance: vec3<f32>, o: vec3<f32>, dir: vec3<f32>,
            t_hit: f32, hit_pos: vec3<f32>) -> vec3<f32> {
    let z0 = max(o.z, 0.0);
    let z1 = max(hit_pos.z, 0.0);
    let mu = dir.z;
    var tau: f32;
    if (abs(mu) > 1e-6) {
        tau = CITY_FOG_BETA * (CITY_FOG_H / mu)
              * (exp(-z0 / CITY_FOG_H) - exp(-z1 / CITY_FOG_H));
    } else {
        tau = CITY_FOG_BETA * t_hit * exp(-z0 / CITY_FOG_H);
    }
    let ftrans = exp(-max(tau, 0.0));
    let mid = mix(o.xy, hit_pos.xy, 0.6);
    let lod = 5.0 + log2(1.0 + t_hit / 2000.0);
    // Clamped: over the megatower district the raw mip runs to ~3x the
    // p99.5 level, and unclamped it bleaches the whole frame sepia.
    let glow = min(city_glow_sample(mid, lod), 1.2);
    let in_scatter = (1.0 - ftrans)
        * (CITY_FOG_GLOW * (CITY_FOG_GLOW_AMP * glow) + CITY_FOG_BASE);
    return radiance * ftrans + in_scatter;
}

// Downward twin of sky_probe_transmittance: the optical depth between a
// sample and the city below it. The caller applies its own decay — NOT the
// two-stream diffuse transmittance, whose heavy tail (T(20) ~ 0.3) let the
// city glow soak whole cloud masses ochre; the amber belongs to the base
// skin, so the uplight uses an exponential in this tau instead.
fn city_uplight_probe_tau(p: vec3<f32>, jit: f32) -> f32 {
    let span = min(CITY_UP_PROBE_SPAN, max(p.z - u.bmin.z, 0.0));
    if (span <= 0.0) {
        return 0.0;
    }
    let dz = span / f32(CITY_UP_PROBE_STEPS);
    var tau = 0.0;
    for (var k: i32 = 0; k < CITY_UP_PROBE_STEPS; k = k + 1) {
        let z_off = (f32(k) + 0.5 + jit) * dz;
        let q = vec3<f32>(p.x, p.y, p.z - z_off);
        tau = tau + sample_sigma(q) * dz;
    }
    return tau;
}

// The night sky: gradient, stars, the city's own glow dome at the skyline,
// and a crescent moon where the sun used to be.
fn night_sky_radiance(dir: vec3<f32>, moon: vec3<f32>) -> vec3<f32> {
    let zc = clamp(dir.z, 0.0, 1.0);
    var col = mix(CITY_NIGHT_HORIZON, CITY_NIGHT_ZENITH, pow(zc, 0.55));

    // Skyline glow dome: sample the city ahead of the sightline.
    let probe_xy = u.cam_origin.xy + dir.xy * CITY_SKYGLOW_DOME_DIST;
    let dome = city_glow_sample(probe_xy, 7.0);
    col = col + CITY_SKYGLOW_DOME * (CITY_SKYGLOW_DOME_AMP * dome)
                * exp(-max(dir.z, 0.0) / CITY_SKYGLOW_DOME_SCALE);

    // Stars: hashed cells on the dominant-axis cube face, faded into the
    // horizon haze and small enough that subpixel jitter antialiases them.
    let ad = abs(dir);
    var uv: vec2<f32>;
    var face: i32;
    if (ad.z >= ad.x && ad.z >= ad.y) {
        uv = dir.xy / ad.z;
        face = 0;
    } else if (ad.x >= ad.y) {
        uv = dir.yz / ad.x;
        face = 1;
    } else {
        uv = dir.xz / ad.y;
        face = 2;
    }
    let sc = uv * CITY_STAR_GRID;
    let cid = vec2<i32>(floor(sc));
    let sh = city_rand4(pcg2d(vec2<u32>(
        bitcast<u32>(cid.x * 3 + face),
        bitcast<u32>(cid.y)
    )));
    if (sh.x < CITY_STAR_FRAC) {
        let spos = vec2<f32>(cid) + vec2<f32>(0.15) + 0.70 * sh.yz;
        let d = length(sc - spos);
        if (d < CITY_STAR_RADIUS) {
            let b = pow(sh.w, 4.0) * (1.0 - d / CITY_STAR_RADIUS);
            col = col + vec3<f32>(0.9, 0.95, 1.0)
                * (CITY_STAR_AMP * b * smoothstep(0.05, 0.25, dir.z));
        }
    }

    // Moon: halo, tight bloom, then the crescent disc itself.
    let cm = dot(dir, moon);
    let g_h = CITY_MOON_HALO_G;
    let dh = 1.0 + g_h * g_h - 2.0 * g_h * cm;
    let one_minus_gh = 1.0 - g_h;
    let halo = (one_minus_gh * one_minus_gh * one_minus_gh)
        / max(pow(dh, 1.5), 1e-6);
    col = col + CITY_MOON_HALO_AMP * halo * vec3<f32>(0.75, 0.80, 0.95);
    if (cm > 0.0) {
        let a = CITY_MOON_BLOOM_W / ((1.0 - cm) + CITY_MOON_BLOOM_W);
        col = col + a * u.sky_bloom.xyz;
    }
    let off = dir - moon * cm;
    let sin_r = length(off);
    if (cm > 0.0 && sin_r < CITY_MOON_SIN_R) {
        let r = sin_r / CITY_MOON_SIN_R;
        let t1 = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), moon));
        let t2 = cross(moon, t1);
        let a1 = dot(off, t1) / CITY_MOON_SIN_R;
        let a2 = dot(off, t2) / CITY_MOON_SIN_R;
        let c = sqrt(max(1.0 - r * r, 0.0));
        let n = a1 * t1 + a2 * t2 - c * moon;
        let l = normalize(0.82 * t1 - 0.20 * t2 + 0.55 * moon);
        let lit = smoothstep(
            0.0, CITY_MOON_TERMINATOR_SOFT, dot(n, l));
        let edge = smoothstep(1.0, 0.97, r);
        col = col + CITY_MOON_DISC * edge
              * (lit + CITY_MOON_EARTHSHINE);
    }
    return col;
}

// >>> GENERATED CITY COMPONENTS — written by tools/compose_city.py from
// >>> cloudyview/soar/city/components/; edit the component files and re-run
// >>> the composer, never this block.
fn cc_extra_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    return res;
}

fn cc_cell_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    return res;
}

fn cc_component_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    // An unclaimed component kind is a bug, and this is its color.
    return vec3<f32>(1.0, 0.0, 1.0);
}

fn cc_facade_detail(cc: CityCell, h: CityHit, uc: f32, vc: f32, fp: f32)
        -> vec3<f32> {
    var e = vec3<f32>(0.0);
    return e;
}

fn cc_window_glyph(cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>, fp: f32)
        -> vec3<f32> {
    var t = vec3<f32>(1.0);
    return t;
}
// <<< GENERATED CITY COMPONENTS

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle.
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// One field-x plane of the sun-tau cache. The host renders this into a 2D
// r16float target of size (cache_nz, cache_ny) for slice u.light_cache.y,
// then copies the target into that depth slice of the light_tau texture —
// 3D textures are not renderable, and r16float is not a storage format, but
// it IS renderable, so the bake is a render pass rather than a compute pass.
// Rendering slice by slice is also what lets the browser spread a bake over
// frames instead of stalling one.
//
// Each texel evaluates the exact live march — light_march_tau through
// sample_level_at, nest included — at the cache texel's own world position,
// with dt_floor = 0 (full quadrature: the bake is paid once, so it takes the
// fine step everywhere) and zero jitter (a static cache cannot average a
// randomized phase away, so it takes the unbiased-enough left-endpoint rule
// the near field always used). Texel i sits AT data coordinate i, matching
// sample_level's convention, so sample_light_tau reads back exactly what was
// baked at texel centers.
@fragment
fn fs_bake_light(@builtin(position) frag_pos: vec4<f32>)
        -> @location(0) vec4<f32> {
    let tex_dims = vec3<f32>(textureDimensions(light_tau, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    // frag_pos.xy = (iz + 0.5, iy + 0.5) over the (nz, ny) target; the slice
    // index is the field-x plane.
    let data_g = vec3<f32>(
        u.light_cache.y,
        frag_pos.y - 0.5,
        frag_pos.x - 0.5
    );
    let p = u.bmin.xyz
        + (data_g / dims) * (u.bmax.xyz - u.bmin.xyz);
    let tau = light_march_tau(p, u.sun_dir.xyz, 0.0, 0.0);
    return vec4<f32>(tau, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let img_w = u.params.x;
    let img_h = u.params.y;
    let g_hg = u.params.z;
    let ambient_strength = u.params.w;
    let tan_half_fov = u.cam_origin.w;
    let aspect = u.cam_forward.w;
    let exposure = u.cam_right.w;
    let jitter_on = u.cam_up.w;
    let subpixel_on = u.flags.x;
    let jitter_scale = clamp(u.flags.y, 0.0, 1.0);
    let sun = u.sun_dir.xyz;
    let gradient_shading_strength = u.cb_realism.x;
    let deep_shadow_ms_suppression = u.cb_realism.y;
    let ambient_occlusion_strength = u.cb_realism.z;
    let bounce_depth_attenuation = u.cb_realism.w;
    let gradient_coarse_weight = u.cb_params.x;
    let gradient_coarse_radius_m = u.cb_params.y;
    let ambient_occlusion_floor = u.cb_params.z;

    // This frame's index in the accumulation, the sequence coordinate every
    // stratified stream below is indexed by, and this pixel, which seeds their
    // scrambles (see the strat1/strat2 block).
    let frame_index = u32(max(u.sun_dir.w, 0.0));
    let strat_pixel = vec2<u32>(frag_pos.xy);

    // Pixel -> camera ray. Framebuffer y=0 is the image top, matching the
    // witness convention ndc_y = 1 - 2*(py+0.5)/h.
    var sample_pos = frag_pos.xy;
    if (subpixel_on > 0.5) {
        let subpixel_offset = strat2(frame_index, strat_pixel,
                                     STRAT_STREAM_SUBPIXEL,
                                     vec2<f32>(STRAT_ALPHA_1, STRAT_ALPHA_2));
        sample_pos = sample_pos
                     + (subpixel_offset - vec2<f32>(0.5)) * jitter_scale;
    }
    let ndc_x = (2.0 * sample_pos.x / img_w - 1.0) * tan_half_fov;
    let ndc_y = (1.0 - 2.0 * sample_pos.y / img_h) * tan_half_fov / aspect;
    let dir = normalize(u.cam_forward.xyz
                        + ndc_x * u.cam_right.xyz
                        + ndc_y * u.cam_up.xyz);

    let inv_dir = 1.0 / dir;
    let periodic_on = PERIODIC_DOMAIN;
    var t_near: f32;
    var t_far: f32;
    if (periodic_on) {
        // Tiled domain: only the z slab bounds the march (rays never exit
        // horizontally); the horizontal exit is replaced by the clear-air
        // transmittance / wrap-ceiling cap.
        let tz0 = (u.bmin.z - u.cam_origin.z) * inv_dir.z;
        let tz1 = (u.bmax.z - u.cam_origin.z) * inv_dir.z;
        t_near = max(min(tz0, tz1), 0.0);
        t_far = min(
            max(tz0, tz1),
            periodic_march_cap(u.cam_origin.z, dir)
        );
    } else {
        let hit = ray_box(u.cam_origin.xyz, inv_dir);
        t_near = max(hit.x, 0.0);
        t_far = hit.y;
    }

    // Under CITY the ocean plane is compiled out entirely: the surface below
    // is the traced city, which fills the same role (an opaque floor at
    // t_city) through the same guards.
    let ocean_on = !CITY && u.ocean_params.z > 0.5;
    var t_ocean = 1e30;
    if (ocean_on && dir.z < -1e-8) {
        let t_ocean_candidate = (u.ocean.x - u.cam_origin.z) / dir.z;
        if (t_ocean_candidate > 0.0) {
            t_ocean = t_ocean_candidate;
        }
    }

    // The city trace runs once, up front: opaque geometry independent of the
    // volume march, composited when the march reaches it exactly as the
    // ocean plane is. The radiance is shaded and fogged here so the march
    // loop only ever multiplies it by transmittance.
    var t_city = 1e30;
    var city_rgb = vec3<f32>(0.0);
    if (CITY) {
        let pixel_angle = 2.0 * tan_half_fov / img_w;
        let chit = city_trace(u.cam_origin.xyz, dir);
        if (chit.hit) {
            t_city = chit.t;
            city_rgb = city_fog(
                city_shade(chit, dir, pixel_angle * chit.t),
                u.cam_origin.xyz, dir, chit.t, chit.pos);
        }
    }
    let t_surface = select(t_ocean, t_city, CITY);

    let cos_scatter = dot(dir, sun);
    let phase = hg_phase(cos_scatter, g_hg);
    // Once-scattered diffraction spike (see DIFFRACTION_G). Constant along
    // the ray, like `phase`; the window makes it exactly zero beyond 30 deg
    // from the sun, so any view not containing the near-sun sky is
    // bit-identical.
    let phase_diffraction = DIFFRACTION_WEIGHT
        * hg_phase(cos_scatter, DIFFRACTION_G)
        * smoothstep(DIFFRACTION_WINDOW_COS_START,
                     DIFFRACTION_WINDOW_COS_FULL, cos_scatter);

    // Scattering-geometry weight on the powder term (see POWDER_FWD_*). One
    // value per pixel: cos_scatter is constant along a view ray. 1 keeps the
    // full boundary darkening, 1 - POWDER_FWD_FADE is what survives at exact
    // forward scattering.
    let g2_hg = g_hg * g_hg;
    // clamped short of 1 so the smoothstep below can never be degenerate; the
    // select handles the isotropic case, where no forward cone exists at all
    // and powder must stay whole in every direction.
    let powder_fwd_cos_start = clamp(
        (1.0 + g2_hg - pow(max(1.0 - g2_hg, 1e-6), 2.0 / 3.0))
        / (2.0 * max(g_hg, 1e-4)),
        -1.0, 0.999
    );
    let powder_weight = 1.0 - POWDER_FWD_FADE * select(
        0.0,
        smoothstep(powder_fwd_cos_start, 1.0, cos_scatter),
        g_hg > 1e-4
    );

    // Aerial perspective (witness iter_008): this sightline's horizon sky
    // color (solar disc excluded) — the same asymptotic target the ocean
    // haze uses, so cloud and water converge to one haze color.
    let aerial_strength = u.sky_horizon.w;
    var aer = vec3<f32>(0.0);
    if (aerial_strength > 0.0) {
        if (CITY) {
            // At night the horizon asymptote is the host-packed night haze
            // color directly; the daytime sky model has nothing to add.
            aer = u.sky_horizon.xyz;
        } else {
            let ah_len = length(dir.xy);
            var ah = dir.xy;
            if (ah_len > 1e-8) {
                ah = dir.xy / ah_len;
            }
            aer = sky_radiance(
                vec3<f32>(ah, 0.0), sun,
                u.sky_horizon.xyz, u.sky_bloom.xyz, vec3<f32>(0.0),
                u.cloud_sun.w
            );
        }
    }

    // Jittered first step: decorrelates the sampling shells between
    // neighboring pixels, killing the coherent ring/banding artifact.
    // Stratified over the accumulation (iter_011), so the shells a pixel tests
    // across frames tile its entry interval instead of clumping.
    // The entry step scale is the outer level's: the nest is required to lie
    // strictly inside the outer AABB, so a ray always enters through it.
    let jitter = strat1(frame_index, strat_pixel, STRAT_STREAM_ENTRY,
                        STRAT_ALPHA_3);
    // Independent stream for the sky probe's quadrature offset.
    let probe_jitter = strat1(frame_index, strat_pixel,
                              STRAT_STREAM_SKY_PROBE,
                              STRAT_ALPHA_10) - 0.5;
    // Independent stream for the sun march's quadrature offset (iter_003).
    // Under the same jitter_on switch as the view march: turning jitter off
    // asks for a deterministic march, and this is one, artifact and all.
    let shadow_jitter = jitter_on * jitter_scale
                        * strat1(frame_index, strat_pixel,
                                 STRAT_STREAM_SUN_MARCH,
                                 STRAT_ALPHA_4);
    // Independent stream for the solar-cone draw (iter_007). Two uniforms:
    // squared radius and azimuth on the disc.
    let sun_cone_seed = strat2(frame_index, strat_pixel,
                               STRAT_STREAM_SUN_CONE,
                               vec2<f32>(STRAT_ALPHA_5, STRAT_ALPHA_6));
    let penumbra_tan = SUN_ANGULAR_RADIUS * SUN_CONE_WIDEN
                       * jitter_on * jitter_scale;
    // Built once here, not once per draw: same crosses, same normalize, same
    // order, so every deflected direction is the number it always was.
    let sun_frame = sun_tangent_frame(sun);
    // Independent stream for the ocean's sub-pixel slope draw (iter_008).
    let ocean_slope_seed = strat2(frame_index, strat_pixel,
                                  STRAT_STREAM_OCEAN_SLOPE,
                                  vec2<f32>(STRAT_ALPHA_7, STRAT_ALPHA_8));
    // Independent stream for the forward pre-march's quadrature offset.
    let ahead_jitter = jitter_on * jitter_scale
                       * strat1(frame_index, strat_pixel,
                                STRAT_STREAM_PREMARCH,
                                STRAT_ALPHA_9);
    let entry_dt = u.bmin.w;
    var t = t_near + jitter_on * jitter * jitter_scale * entry_dt;

    var transmittance = 1.0;
    var col = vec3<f32>(0.0);
    var tau_depth = 0.0;
    // Whether the march below shaded the water, tracked rather than inferred
    // (docs/soar-bugs.md 6). The far-water fallback used to decide the same
    // thing from `t_ocean > t_far`, which is only meaningful when the ray hit
    // the box at all: ray_box returns t_near > t_far on a MISS with t_far
    // still a positive finite number — the nearest exit plane of a box the
    // ray never entered. Such a ray skips the march (the guard below wants
    // t_near < t_far) and then failed the fallback too whenever that
    // leftover t_far happened to exceed t_ocean, so the ocean was hit and
    // nothing shaded it. Because t_far varies smoothly with direction that
    // came out as a contiguous region of sky where the water should be,
    // reachable with periodicity off and the camera just outside the domain
    // looking down.
    var ocean_consumed = false;
    // Optical depth this ray has already crossed, so the pre-march total can
    // be turned into "what is still ahead of this sample". Unlike tau_depth
    // this never resets: it is the ray's own coordinate along its chord.
    var tau_view = 0.0;
    // Whether the previous view sample sat inside the nest, so the march can
    // notice the fine->coarse crossing (docs/soar-bugs.md 4).
    var was_in_nest = false;

    // The periodic march can legitimately cover several domain widths, so
    // it gets more step headroom; the non-periodic bound is untouched.
    let max_view_steps = select(MAX_VIEW_STEPS, MAX_VIEW_STEPS_PERIODIC,
                                periodic_on);

    // How much new hemisphere the sightline probe covers that iter_001's
    // vertical probe did not: the sine of the angle between the two.
    let ahead_novelty = sqrt(max(1.0 - dir.z * dir.z, 0.0));

    // Spectrum of the buried residual of the diffuse fills — per frame, not
    // per sample (it depends only on the two lighting spectra).
    let deep_tint = deep_fill_tint(u.ambient_tint.xyz);

    // One coarse pass over the whole sightline before anything is composited.
    var tau_total = 0.0;
    if (t_near >= 0.0 && t_near < t_far) {
        tau_total = premarch_tau_ahead(
            u.cam_origin.xyz, dir, t_near, min(t_far, t_surface), ahead_jitter
        );
    }

    if (t_near >= 0.0 && t_near < t_far) {
        for (var i: i32 = 0; i < max_view_steps; i = i + 1) {
            // witness.py:621-646 tests ocean before the t_far break so an
            // ocean plane coincident with the box floor is still shaded.
            if (ocean_on && t >= t_ocean) {
                let ocean_hit = u.cam_origin.xyz + t_ocean * dir;
                col = col + transmittance
                            * ocean_shade_dispatch(
                                ocean_hit, dir, sun,
                                sun_cone_dir(sun, sun_frame,
                                             sun_cone_seed, penumbra_tan),
                                t_ocean, shadow_jitter,
                                ocean_slope_seed,
                                jitter_on * jitter_scale);
                transmittance = 0.0;
                ocean_consumed = true;
                break;
            }
            if (CITY && t >= t_city) {
                col = col + transmittance * city_rgb;
                transmittance = 0.0;
                ocean_consumed = true;
                break;
            }

            if (t >= t_far || transmittance < TRANSMITTANCE_CUTOFF) {
                break;
            }
            let p = u.cam_origin.xyz + t * dir;
            // The active level sets the step scale, so the march refines on
            // entering the nest and coarsens on leaving it. The dt-invariant
            // powder term below is what lets the two levels composite
            // without a brightness seam (witness.py header).
            let level = sample_level_at(p);
            // Crossing OUT of the nest, the step length jumps fine -> coarse,
            // but every ray crosses at the same geometric plane and arrives
            // with at most a fine step's worth of phase — so sample phase
            // past the seam covered barely half the coarse period, and the
            // coherent remainder survived accumulation as stripes parallel
            // to the seam (docs/soar-bugs.md 4; the A/B pinned the view
            // march, not the light march). The cure is the file's usual one:
            // an independent per-pixel-per-frame uniform phase, spanning the
            // FULL effective coarse step — adding a full-period uniform
            // modulo the period is uniform whatever the crossing
            // distribution was, where re-using the entry draw or spanning
            // only the step difference both leave a coherent residual
            // (measured; the first attempt did the former and the stripes
            // survived it). The reverse crossing needs nothing: a coarse
            // period of phase folded into a fine one already covers it.
            if (NESTED && was_in_nest && !level.in_nest) {
                was_in_nest = false;
                let seam_jitter = strat1(frame_index, strat_pixel,
                                         STRAT_STREAM_NEST_SEAM,
                                         STRAT_ALPHA_10);
                let skip = jitter_on * seam_jitter * jitter_scale
                           * max(u.bmin.w, t * u.periodic.z);
                if (skip > 0.0) {
                    t = t + skip;
                    continue;
                }
            }
            was_in_nest = level.in_nest;
            let sigma = level.sigma;

            // Distance LOD: the step floor grows with distance so far
            // wrapped copies march in angular, not metric, resolution
            // (0 = fixed dt). Dense-sigma refinement still applies.
            var dt = step_dt_for_sigma(
                sigma, max(level.dt_view, t * u.periodic.z)
            );
            if (t + dt > t_far) {
                dt = t_far - t;
            }
            if (ocean_on && t + dt > t_ocean) {
                dt = max(0.0001, t_ocean - t);
            }
            if (CITY && t + dt > t_city) {
                dt = max(0.0001, t_city - t);
            }

            let d_tau = sigma * dt;
            if (d_tau < EMPTY_DTAU_CUTOFF) {
                tau_depth = 0.0;
                t = t + dt;
                continue;
            }

            tau_depth = tau_depth + d_tau;
            tau_view = tau_view + d_tau;

            // Aerial perspective: clear-air transmittance camera->sample via
            // the closed-form exponential atmosphere (witness._render_image).
            var air_t = 1.0;
            if (aerial_strength > 0.0) {
                let aer_beta0 = u.sky_bloom.w;
                let aer_h = u.sky_disc.w;
                let aer_z0 = max(u.cam_origin.z, 0.0);
                let aer_z1 = max(p.z, 0.0);
                let aer_mu = dir.z;
                var tau_air: f32;
                if (aer_h <= 0.0) {
                    tau_air = aer_beta0 * t;          // uniform: no z profile
                } else if (aer_mu > 1e-6 || aer_mu < -1e-6) {
                    tau_air = aer_beta0 * (aer_h / aer_mu)
                        * (exp(-aer_z0 / aer_h) - exp(-aer_z1 / aer_h));
                } else {
                    tau_air = aer_beta0 * t * exp(-aer_z0 / aer_h);
                }
                air_t = exp(-aerial_strength * tau_air);
            }

            // A view ray takes many steps, so its shadow rays advance the
            // offset by the golden ratio instead of reusing one value: the
            // sequence is low-discrepancy along the march, which averages the
            // quadrature noise down within a single frame rather than leaving
            // it all for temporal accumulation.
            let step_shadow_jitter = fract(
                shadow_jitter + f32(i) * 0.6180339887498949
            );
            // Draw this sample's point on the solar disc (iter_007). The R2
            // additive recurrence advances both coordinates along the view
            // march for the same reason the golden ratio advances the
            // quadrature phase above: the many shadow rays that composite into
            // one pixel then cover the disc low-discrepancy within a single
            // frame, so soar's first frame shows a soft edge with light dither
            // rather than a hard edge that only melts once accumulation
            // settles.
            let sun_shadow = sun_cone_dir(
                sun, sun_frame,
                fract(sun_cone_seed + f32(i) * vec2<f32>(
                    0.7548776662466927, 0.5698402909980532
                )),
                penumbra_tan
            );
            // The cache replaces the march, the disc jitter and the
            // distance-LOD coarsening in one move: it is the central-sun,
            // zero-jitter tau at the cache's own resolution, trilinearly
            // filtered. See Uniforms.light_cache.
            var tau_sun: f32;
            if (u.light_cache.x > 0.5) {
                tau_sun = sample_light_tau(p);
            } else {
                tau_sun = light_march_tau(p, sun_shadow, t * u.periodic.y,
                                          step_shadow_jitter);
            }
            let light_transfer_split_strength = u.ambient_tint.w;
            // Unconditional: the diffuse beam and the high-sun skylight
            // consume this gate too, so computing it only when the storm
            // machinery is enabled would hand a buried storm the full
            // shallow treatment when suppression/AO/split are all zero
            // (codex review 2026-08-11, finding 3).
            // Log-space: per-e-fold travel between tau 15 and 1000 (see the
            // DEEP_SHADOW_TAU constants). The max() guard only matters for
            // tau_sun < 1, which is far below the gate's onset anyway.
            let deep_shadow_gate = smoothstep(
                DEEP_SHADOW_LOG_TAU_START, DEEP_SHADOW_LOG_TAU_FULL,
                log(max(tau_sun, 1e-3))
            );

            // Moderate-shadow gate: where the beam and MS octaves are spent
            // but the deep-shadow machinery has not engaged. This is where
            // the skylight fill lives. Exponential onset — see the
            // SHADOW_SKYLIGHT_TAU_ONSET comment for why not a smoothstep.
            let shadow_gate = 1.0 - exp(-tau_sun / SHADOW_SKYLIGHT_TAU_ONSET);

            // Sky visibility, measured (hoisted from the diffuse block below
            // so the MS floor can use it too). Run for any meaningfully
            // shadowed sample — elsewhere tau_sun still carries the shading
            // and this costs nothing. One probe serves every consumer.
            var t_sky = 1.0;
            var shallow_open = 0.0;
            if (shadow_gate > 0.0 || deep_shadow_gate > 0.0) {
                if (u.light_cache.z > 0.5) {
                    // Sky probe disabled (a cost/look toggle, J in the
                    // browser): every consumer sees a fully open sky, which
                    // is the neutral the floors already assume. Buried
                    // samples brighten — that is the look being priced.
                    shallow_open = 1.0;
                } else {
                    t_sky = sky_probe_transmittance(p, g_hg, probe_jitter);
                    shallow_open = smoothstep(
                        SHALLOW_OPEN_TSKY_START, SHALLOW_OPEN_TSKY_FULL, t_sky
                    );
                }
            }
            // A buried storm sample keeps the full tuned suppression; an
            // optically open shallow shadow keeps only a quarter of it.
            let storm_weight = mix(
                SHALLOW_SUPPRESSION_KEEP, 1.0, 1.0 - shallow_open
            );

            // Diffusion depth for the isotropic tail (see MS_TAIL_FLOOR).
            // The knee is what keeps this a *contrast* change rather than a
            // dimming: below it the factor is exactly 1, so thin cloud and
            // the directly lit shoulder of a turret are bit-identical to
            // iter_001 and only the genuinely self-shadowed parts move.
            let ms_tail_factor = MS_TAIL_FLOOR + (1.0 - MS_TAIL_FLOOR)
                * diffuse_transmittance(
                    max(tau_sun - MS_TAIL_TAU_KNEE, 0.0), g_hg
                );

            // Local surface frame. Hoisted above the MS loop (iter_006) so the
            // tail can be weighted by the beam's incidence on the skin; the
            // gradient *shading* below consumes exactly these values in exactly
            // the same way it did, so that term is untouched.
            var grad = vec3<f32>(0.0);
            var grad_len = 0.0;
            var surface_gate = 0.0;
            if (gradient_shading_strength > 0.0) {
                let grad_conf_v = sigma_gradient(
                    p, sigma, gradient_coarse_weight, gradient_coarse_radius_m,
                    t, u.cb_params.w, level.in_nest
                );
                grad = grad_conf_v.xyz;
                grad_len = length(grad);
                surface_gate = smoothstep(
                    GRADIENT_SHADING_TAU_START,
                    GRADIENT_SHADING_TAU_FULL,
                    tau_depth
                ) * smoothstep(
                    GRADIENT_SHADING_CONF_START,
                    GRADIENT_SHADING_CONF_FULL,
                    grad_conf_v.w
                );
            }

            // Beam incidence on the skin, relative to a horizontal face (see
            // the TAIL_MU_* block). Exactly 1 where the field is level, so it
            // costs no brightness on flat cloud top and only the flanks of
            // turrets and the walls of the crevices between them give the tail
            // up.
            var tail_mu_factor = 1.0;
            if (grad_len > 1e-12) {
                let mu0 = max(-dot(grad, sun) / grad_len, 0.0);
                let mu_ratio = clamp(
                    mu0 / max(sun.z, TAIL_MU_REF_MIN), 0.0, 1.0
                );
                tail_mu_factor = mix(
                    1.0,
                    TAIL_MU_FLOOR + (1.0 - TAIL_MU_FLOOR) * mu_ratio,
                    surface_gate * (1.0 - deep_shadow_gate)
                );
            }

            var ms = vec3<f32>(0.0);
            var ms_atten = 1.0;
            for (var octave: i32 = 0; octave < MS_OCTAVES; octave = octave + 1) {
                let t_sun_ms = exp(-tau_sun * ms_atten);
                let blend = min(1.0, f32(octave) * MS_BLEND_RATE);
                let oct_phase = phase * (1.0 - blend) + ISO_PHASE * blend;
                var contrib = ms_atten * t_sun_ms * oct_phase;
                let iso_gate = smoothstep(0.35, 1.0, blend);
                // Octaves 0-1 still carry the beam and keep their exact
                // baseline value; the tail is put on the diffusion depth in
                // proportion to how isotropic it has already become, so the
                // handover is continuous and nothing is double-counted.
                contrib = contrib
                    * mix(1.0, ms_tail_factor * tail_mu_factor, iso_gate);
                if (deep_shadow_ms_suppression > 0.0) {
                    let ms_floor = max(
                        DEEP_SHADOW_MS_FLOOR,
                        1.0 - deep_shadow_ms_suppression
                              * storm_weight
                              * deep_shadow_gate
                              * iso_gate
                    );
                    contrib = contrib * ms_floor;
                }
                ms = ms + contrib * u.cloud_sun.xyz;
                ms_atten = ms_atten * MS_ATTEN;
            }

            // The diffraction spike joins the ladder as part of the
            // once-scattered term: octave 0's transmittance, no tail or
            // suppression factors (octave 0's iso_gate is 0), and it shares
            // the direct-boost and gradient multipliers below exactly as
            // octave 0 does. The guard is per-pixel-coherent (the window is
            // constant along the ray), so far-from-sun pixels skip the exp.
            if (phase_diffraction > 0.0) {
                ms = ms + phase_diffraction * exp(-tau_sun) * u.cloud_sun.xyz;
            }

            // Light-transfer split, warm side: modest boost of the unoccluded
            // direct/MS source at low sun (witness iter_006).
            if (light_transfer_split_strength > 0.0) {
                let direct_factor = 1.0 + light_transfer_split_strength
                    * LIGHT_TRANSFER_DIRECT_BOOST
                    * exp(-tau_sun);
                ms = ms * direct_factor;
            }

            // fill_shape carries the surface's billow orientation into the
            // diffuse fills below (2026-08-11): once tau_sun saturates, ms
            // is spent and every remaining light source was orientation-
            // blind, so shaded cloud went flat. Real anvil undersides
            // (IMG_7053) keep soft billow definition deep into shadow. The
            // fills take the same measured gradient at reduced weight —
            // diffuse light is directional enough to shade billows, just
            // more softly than the beam.
            var fill_shape = 1.0;
            if (gradient_shading_strength > 0.0) {
                if (grad_len > 1e-12) {
                    var n_dot_sun = -dot(grad, sun) / grad_len;
                    if (n_dot_sun < 0.0) {
                        n_dot_sun = n_dot_sun * GRADIENT_SHADING_SHADOW_SIDE_SCALE;
                    }
                    let gradient_factor = max(
                        0.20,
                        1.0 + gradient_shading_strength
                              * surface_gate
                              * n_dot_sun
                    );
                    ms = ms * gradient_factor;
                    fill_shape = mix(1.0, gradient_factor,
                                     FILL_GRADIENT_WEIGHT);
                }
            }

            // Powder is a function of cumulative optical depth since the current
            // cloud entry, not the current step size (witness.py:729-732).
            // powder_weight is the scattering-geometry dependence powder has
            // always been missing; at weight 1 this is exactly the old
            // expression. Note no extra depth gate is needed: the deficit it
            // scales, exp(-POWDER_COEFF * tau_depth), is already dead by
            // tau_depth ~ 2, so the change is confined to the thin skin and
            // the edges, which is where the physics lives.
            let powder = 1.0 - powder_weight * exp(-POWDER_COEFF * tau_depth);
            let scatter_weight = d_tau * powder * transmittance * air_t;
            col = col + scatter_weight * ms;

            // (t_sky and shallow_open were measured above the MS loop.)

            // Second probe direction, free from the pre-march: how much cloud
            // lies beyond this sample along the sightline. A skin sample on a
            // wisp is open both ways and keeps its full diffuse fill; the
            // same skin on a deep mass is open only backward, toward the
            // camera, and is a poor place for diffuse light to reach. This is
            // the whole of the entry-face thick/thin distinction, and because
            // the direction is the sightline the structure it reports is
            // registered with the silhouette: a luminous fringe around every
            // edge and gap, with the body behind it falling away.
            //
            // It multiplies the diffuse terms rather than feeding t_sky,
            // deliberately. Both consumers of t_sky wrap it in a floor
            // (ambient_occlusion_floor, SKY_PROBE_FILL_FLOOR) that exists so
            // one vertical measurement cannot drive a sample to black; the
            // forward measurement is independent evidence about a different
            // part of the hemisphere and should not be spent inside another
            // term's safety floor. It carries its own.
            //
            // Its weight is the sine of the angle between the two probe
            // directions (ahead_novelty, hoisted out of the loop). A sightline
            // pointing straight up is re-measuring what iter_001 already
            // measured and must not be counted twice; an oblique or horizontal
            // one is genuinely new information about the hemisphere. That is
            // the whole of the two-direction quadrature: coincident samples
            // collapse to one, orthogonal samples both count.
            var ahead_factor = 1.0;
            if (deep_shadow_gate > 0.0) {
                let tau_ahead = max(tau_total - tau_view, 0.0);
                let t_ahead = AHEAD_FLOOR + (1.0 - AHEAD_FLOOR)
                    * diffuse_transmittance(tau_ahead, g_hg);
                ahead_factor = mix(
                    1.0, t_ahead,
                    AHEAD_WEIGHT * ahead_novelty * deep_shadow_gate
                );
            }

            // Ambient: height-based on the outer box (witness._render_image).
            // The ramp floor is by openness: a storm interior keeps the tuned
            // 0.3, an open shallow base — which sees sky sideways and bright
            // surface below — keeps 0.6 of the full fill.
            let h = clamp((p.z - u.bmin.z) / (u.bmax.z - u.bmin.z), 0.0, 1.0);
            let height_floor = mix(
                AMBIENT_HEIGHT_FLOOR, AMBIENT_HEIGHT_FLOOR_SHALLOW,
                shallow_open * deep_shadow_gate
            );
            let height_ramp = height_floor + (1.0 - height_floor) * h;
            let amb = ambient_strength * height_ramp * ahead_factor
                      * fill_shape;
            // The constant deep-shadow floor becomes the T_sky -> 0 limit
            // of a measured factor: fully buried samples land on exactly
            // ambient_occlusion_floor as before, and everything less
            // buried lifts continuously toward unoccluded.
            //
            //   amb_factor = (1 - s) + s*floor + s*(1 - floor)*t_sky
            //
            // written out rather than as a mix() because its three pieces do
            // not share a spectrum. The first two are the fill a buried
            // sample keeps; the third is the part measured to arrive from the
            // sky. Summed, the weights are exactly the old amb_factor, so
            // this is a spectral split at unchanged luminance.
            let ao_s = select(
                0.0,
                clamp(ambient_occlusion_strength, 0.0, 1.0)
                    * storm_weight * deep_shadow_gate,
                ambient_occlusion_strength > 0.0
            );
            // Open shallow shadow is filled by skylight and keeps the
            // ambient's blue; only genuinely buried fill decays to the
            // neutral deep tint (same luminance, storm-tuned chroma).
            let fill_tint = mix(deep_tint, u.ambient_tint.xyz, shallow_open);
            let amb_w_deep = ao_s * ambient_occlusion_floor;
            let amb_w_sky = (1.0 - ao_s)
                + ao_s * (1.0 - ambient_occlusion_floor) * t_sky;
            col = col + transmittance * d_tau * amb * air_t
                        * (amb_w_sky * u.ambient_tint.xyz
                           + amb_w_deep * fill_tint);

            // Light-transfer split, cool side: a skylight floor restored only
            // in saturated sun shadow; lit faces keep their contrast. The
            // tuned low-sun path, unchanged.
            if (light_transfer_split_strength > 0.0
                && deep_shadow_gate > 0.0) {
                // Same measured visibility: this fill is skylight too, and it
                // is the larger of the two diffuse terms, so leaving it flat
                // would wash the structure back out.
                // Same spectral split as the ambient above: the floor part is
                // sunlight that diffused in, the t_sky part is sky.
                let fill_w_deep = deep_shadow_gate * SKY_PROBE_FILL_FLOOR;
                let fill_w_sky = (1.0 - deep_shadow_gate)
                    + deep_shadow_gate * (1.0 - SKY_PROBE_FILL_FLOOR) * t_sky;
                let sky_fill = light_transfer_split_strength
                    * LIGHT_TRANSFER_SHADOW_SKYLIGHT
                    * height_ramp
                    * deep_shadow_gate * ahead_factor * fill_shape;
                col = col + transmittance * d_tau * sky_fill * air_t
                            * (fill_w_sky * u.ambient_tint.xyz
                               + fill_w_deep * fill_tint);
            }

            // High-sun shadow skylight (2026-08-11): where the split above
            // is gated off, moderately shadowed samples used to get no
            // skylight at all — the beam and the MS octaves are spent by
            // tau_sun ~ 20, so a fair-weather base went from white to dark
            // grey across a few steps. Openness decides the coefficient
            // (a buried storm interior keeps almost nothing, which is what
            // keeps storm bases dark), and the measured sky visibility
            // carves the spatial structure so bases brighten toward their
            // edges and thin spots.
            let high_sun_fill = (1.0 - light_transfer_split_strength)
                * mix(HIGH_SUN_SHADOW_SKYLIGHT_STORM,
                      HIGH_SUN_SHADOW_SKYLIGHT_SHALLOW,
                      shallow_open)
                * shadow_gate * height_ramp * ahead_factor * fill_shape;
            if (high_sun_fill > 0.0) {
                let vis = SKY_PROBE_FILL_FLOOR
                    + (1.0 - SKY_PROBE_FILL_FLOOR) * t_sky;
                col = col + transmittance * d_tau * high_sun_fill * air_t
                            * vis * u.ambient_tint.xyz;
            }

            // Diffused-beam glow (see the constants block): isotropic
            // re-emission of the diffusion tail of the direct beam. The
            // dominant base-lighting term for moderately thick fair-weather
            // cloud; fades to a quarter for buried storm interiors.
            let diffuse_beam = DIFFUSE_BEAM_STRENGTH
                * diffuse_transmittance(tau_sun, g_hg)
                * (1.0 - exp(-tau_sun / DIFFUSE_BEAM_TAU_ONSET))
                * mix(1.0, DIFFUSE_BEAM_STORM_KEEP,
                      deep_shadow_gate * (1.0 - shallow_open))
                * fill_shape;
            if (diffuse_beam > 0.0) {
                col = col + transmittance * d_tau * diffuse_beam * air_t
                            * ISO_PHASE * DIFFUSE_BEAM_TINT * u.cloud_sun.xyz;
            }

            // City uplight: the second light source of the night. The glow
            // mip under the sample at a footprint of its own height (the
            // solid-angle average of the city it actually sees), times the
            // two-stream transmittance of the cloud below, delivered like
            // the ambient fill. Replaces the daytime surface bounce, which
            // is gated off under CITY just below.
            if (CITY) {
                let up_lod = clamp(
                    log2(max(p.z, u.ocean_params.x) / u.ocean_params.x),
                    0.0, u.ocean_params.w);
                let g_raw = max(
                    city_glow_sample(p.xy, up_lod) - CITY_UPLIGHT_GLOW_BIAS,
                    0.0);
                let g_up = g_raw / (CITY_UPLIGHT_GLOW_HALF + g_raw);
                let tau_dn = city_uplight_probe_tau(p, probe_jitter);
                let t_dn = exp(-tau_dn * CITY_UPLIGHT_TAU_SCALE);
                let up = CITY_UPLIGHT_STRENGTH * g_up
                    * (CITY_UPLIGHT_FLOOR + (1.0 - CITY_UPLIGHT_FLOOR) * t_dn);
                col = col + transmittance * d_tau * up * air_t
                            * fill_shape * CITY_UPLIGHT_COLOR;
            }

            // Surface bounce is anchored at physical z=0, not the AABB floor.
            if (BOUNCE_STRENGTH > 0.0 && !CITY) {
                let bounce_frac = clamp(1.0 - p.z / u.bmax.z, 0.0, 1.0);
                var bounce = BOUNCE_STRENGTH * bounce_frac;
                if (bounce_depth_attenuation > 0.0) {
                    bounce = bounce * exp(-bounce_depth_attenuation * tau_depth);
                }
                // The surface underneath is in this cloud's own shadow. The
                // bounce pedestal only exists in proportion to the sunlight
                // that actually reaches the ground, and under a saturated
                // column that is nearly none — which is why a real storm
                // base is dark and this one was not. tau_depth alone could
                // never see it: every base sample has just entered the
                // cloud, so the existing depth attenuation is the same
                // number everywhere across a base and the term composited
                // as a flat pedestal over the whole frame. Faded in by the
                // saturation gate, so thin cloud keeps its bounce exactly.
                //
                // A lateral floor (iter_001 flagged this as the one thing it
                // would tune): the ground a base sample sees is a hemisphere,
                // not a point, and a cloud shadow is finite. Beyond its edge
                // is sunlit surface, and that light reaches the base without
                // ever passing through the saturated column overhead — so the
                // sunward attenuation must not run to zero. T(80) = 0.10 with
                // no floor; the floor puts it back to a small but nonzero
                // pedestal, and because BOUNCE_TINT is warm it restores
                // warmth exactly at the cloud base, where the accumulated
                // blue collected worst.
                bounce = bounce * mix(
                    1.0,
                    max(diffuse_transmittance(tau_sun, g_hg),
                        BOUNCE_LATERAL_FLOOR),
                    deep_shadow_gate
                );
                col = col + transmittance * d_tau * bounce * air_t
                            * BOUNCE_TINT;
            }

            // This step's attenuation, computed once. The aerial in-scatter
            // below and the transmittance update are the same exp of the same
            // operand; the compiler cannot always see that across the branch,
            // and it is one transcendental per view step either way.
            let step_atten = exp(-d_tau);

            // Aerial in-scatter: sky light scattered into the path replaces
            // exactly the radiance this sample occludes.
            if (aerial_strength > 0.0 && air_t < 1.0) {
                let aer_in = transmittance
                    * (1.0 - step_atten)
                    * (1.0 - air_t);
                col = col + aer_in * aer;
            }

            transmittance = transmittance * step_atten;
            t = t + dt;
        }
    }

    // Ocean for rays that exit or MISS the outer box without becoming opaque.
    // The condition is what it always meant to say — the water was hit, the
    // ray still carries light, and the march did not already shade it — with
    // no reference to a t_far that a missing ray never had (see
    // ocean_consumed above).
    //
    // The 50-outer-width clamp below is a second, much larger-scale version
    // of the same shape and is deliberately left alone: past it the water is
    // unshaded and the sky shows through, but at that range the ocean's own
    // aerial haze (ocean_realism_b.y * t_hit) has already carried it onto the
    // horizon sky colour, so the seam is invisible at the shipped settings
    // and only appears with the legacy no-haze ocean. Removing it is a look
    // change — water drawn to the edge of the float — not a bug fix.
    if (ocean_on
        && !ocean_consumed
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_ocean < 1e29) {
        let ocean_hit = u.cam_origin.xyz + t_ocean * dir;
        let outer_size = u.bmax.xyz - u.bmin.xyz;
        let center = 0.5 * (u.bmin.xy + u.bmax.xy);
        if (abs(ocean_hit.x - center.x) < outer_size.x * 50.0
            && abs(ocean_hit.y - center.y) < outer_size.y * 50.0) {
            col = col + transmittance
                        * ocean_shade_dispatch(
                            ocean_hit, dir, sun,
                            sun_cone_dir(sun, sun_frame, sun_cone_seed,
                                         penumbra_tan),
                            t_ocean, shadow_jitter,
                            ocean_slope_seed, jitter_on * jitter_scale);
            transmittance = 0.0;
        }
    }

    // City for rays that exit or miss the volume without becoming opaque —
    // the night twin of the far-water fallback above, minus the range clamp
    // (the city fog has already carried a distant hit onto the haze).
    if (CITY
        && !ocean_consumed
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_city < 1e29) {
        col = col + transmittance * city_rgb;
        transmittance = 0.0;
    }

    if (transmittance > TRANSMITTANCE_CUTOFF) {
        var sky: vec3<f32>;
        if (CITY) {
            sky = night_sky_radiance(dir, sun);
        } else {
            sky = sky_radiance(
                dir, sun,
                u.sky_horizon.xyz, u.sky_bloom.xyz, u.sky_disc.xyz,
                u.cloud_sun.w
            );
        }
        col = col + transmittance * sky;
    }
    if (!TONE_MAP) {
        return vec4<f32>(col, 1.0);
    }
    return vec4<f32>(tone_map(col, exposure, u.periodic.w), 1.0);
}
