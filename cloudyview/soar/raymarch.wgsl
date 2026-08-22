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
    // w = ice-detection mode (0.0 or 1.0), a false-color view:
    //     cloud source terms are tinted by the per-voxel ice fraction
    //     (binding 8) and the sky is remapped to an alien green. At 0.0
    //     every multiply below is by exactly 1.0 and the frame is
    //     bit-identical to a build without the mode.
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
// Ice-detection mode: per-voxel ice extinction fraction
// sigma_ice / (sigma_liq + sigma_ice) over the OUTER level's grid, same
// texel layout as `vol`. Always bound — a 1x1x1 zero stand-in when the field
// carries no ice variable — so one bind-group layout serves both. Read only
// when u.light_cache.w is set.
@group(0) @binding(8) var ice_frac: texture_3d<f32>;

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

// Ice-detection mode (false-color phase) ------------------------------------
//
// Ice extinction fraction at a world point, from the outer level's grid —
// the outer AABB covers any nest, so one texture serves both levels. Same
// coordinate convention as sample_light_tau: no edge taper (a fraction is
// not extinction; ramping it to zero at the boundary would paint the domain
// edge liquid-red), lateral wrap in a periodic domain, clamp vertically.
fn ice_fraction_at(p: vec3<f32>) -> f32 {
    let q = wrap_to_domain(p);
    let tex_dims = vec3<f32>(textureDimensions(ice_frac, 0));
    let dims = vec3<f32>(tex_dims.z, tex_dims.y, tex_dims.x);
    let data_g = ((q - u.bmin.xyz) / (u.bmax.xyz - u.bmin.xyz)) * dims;
    let tex_coord = vec3<f32>(
        data_g.z + 0.5,
        data_g.y + 0.5,
        data_g.x + 0.5
    ) / tex_dims;
    if (PERIODIC_DOMAIN) {
        return textureSampleLevel(ice_frac, vol_wrap_samp, tex_coord, 0.0).r;
    }
    return textureSampleLevel(ice_frac, vol_samp, tex_coord, 0.0).r;
}

// The false-color ramp: liquid burns crimson, ice glows cyan, and the mixed
// phase between them crosses through electric violet. NOT luminance-
// normalized: dividing by the hue's luminance blows the red channel far past
// the tone map's shoulder and deep red comes out pink. Max-component ~1
// instead, so red reads darker than cyan — deep red to vibrant cyan, which
// is the intended reading.
fn ice_tint(f_in: f32) -> vec3<f32> {
    let f = clamp(f_in, 0.0, 1.0);
    // Authored for the display: the tone map's gamma lifts weak channels,
    // so the linear palette is deeper than the hue it lands on.
    let c_liquid = vec3<f32>(1.00, 0.010, 0.030);
    let c_mixed  = vec3<f32>(0.60, 0.015, 1.00);
    let c_ice    = vec3<f32>(0.010, 0.85, 1.00);
    return mix(mix(c_liquid, c_mixed, smoothstep(0.0, 0.5, f)),
               c_ice, smoothstep(0.5, 1.0, f));
}

const ICE_LUM: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// The sea under the alien sky: its shaded luminance on a dark teal, so the
// glint and wave structure survive the recolor.
const ICE_OCEAN_TINT: vec3<f32> = vec3<f32>(0.04, 0.30, 0.24);

// A cloud source term recolored for the false-color view: its luminance —
// which carries all the shading, shadowing and silhouette structure — on the
// ramp's hue. With the mode off the term passes through untouched, so every
// call site is bit-identical to the pre-mode shader.
fn ice_recolor(x: vec3<f32>, tint: vec3<f32>, on: bool) -> vec3<f32> {
    if (on) {
        return dot(x, ICE_LUM) * tint;
    }
    return x;
}

// The alien sky: the physical sky's own luminance — gradients, circumsolar
// bloom, disc, haze behavior all intact — recolored onto a green palette,
// acid chartreuse at the horizon falling to deep emerald-teal overhead.
// Luminance-preserving, so exposure, tone map and aerial haze all keep
// working; `disc` is passed through so the aerial-perspective caller can
// exclude the solar disc exactly as it does from the physical sky.
fn alien_sky(dir: vec3<f32>, sun: vec3<f32>, disc: vec3<f32>) -> vec3<f32> {
    let phys = sky_radiance(dir, sun, u.sky_horizon.xyz, u.sky_bloom.xyz,
                            disc, u.cloud_sun.w);
    let lum = dot(phys, ICE_LUM);
    let up = clamp(dir.z, 0.0, 1.0);
    // Dimmer than the physical sky it replaces (the ramp luminances are
    // ~0.7 and ~0.3, not 1), so the red/cyan clouds carry the frame.
    let horizon = vec3<f32>(0.35, 0.90, 0.10);
    let zenith  = vec3<f32>(0.01, 0.38, 0.26);
    return lum * mix(horizon, zenith, pow(up, 0.45));
}
// ---------------------------------------------------------------------------

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
const CITY_MEAN_COMP_BLOCK: f32 = 0.13;
const CITY_MEAN_COMP_FLAT: f32 = 0.07;
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
const CITY_LAMP_SIGMA: f32 = 3.2;    // light-pool radius on the asphalt
const CITY_LAMP_RADIANCE: f32 = 0.55;
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

// The lamp line's inset from the CELL BOUNDARY on line k. Lamps stand
// CITY_LAMP_OFFSET off the KERB — 3.5 m from the boundary of a 12 m minor
// street, 13.5 m on a 32 m avenue. The old fixed 2.5 m inset measured from
// the boundary, which on an avenue is the road's CENTERLINE: both lamp
// lines hugged the median and floodlit the middle of every downtown
// avenue into clipping (found via streetlife's report, proven by
// ablation + a numpy rebuild of the pool field, 2026-08-20). streetlife's
// poles and parking share this function, so light, hardware and cars
// move together.
fn city_lamp_inset(k: i32) -> f32 {
    return CITY_STREET_HALF
        + select(0.0, CITY_AVENUE_EXTRA, city_is_avenue(k))
        - CITY_LAMP_OFFSET;
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
    // The window style (see the style block in city_cell): 0 grid,
    // 1 ribbon bands, 2 vertical strips, 3 curtain wall, 4 punched.
    // Pitch and pane bounds are per building so no two lattices agree.
    win_style: i32,
    win_pitch: f32,
    pane_lo: vec2<f32>, pane_hi: vec2<f32>,
    pane_frac: f32,
    // Monochrome house palette: below 0 the building draws window colors
    // freely; at or above 0 it is THE cyan (or amber, or magenta) tower —
    // every window draws near this one palette coordinate.
    win_mono: f32,
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

    // The window style: the lattice itself varies building to building, or
    // mid-distance towers all wear the same texture and blur together
    // (Thomas, in flight). Style follows height and architecture loosely —
    // curtain walls on towers, punched openings on low-rise — and pitch is
    // jittered so even two buildings sharing a style disagree in period.
    let r4e = city_rand4(c.seed ^ vec2<u32>(0xb5297a4du, 0x68e31da4u));
    var style = 0;
    if (h > 350.0 || c.arch == 3) {
        if (r4e.x < 0.45) { style = 3; }
        else if (r4e.x < 0.70) { style = 0; }
        else if (r4e.x < 0.90) { style = 1; }
        else { style = 2; }
    } else if (h < 40.0) {
        if (r4e.x < 0.50) { style = 4; }
        else if (r4e.x < 0.80) { style = 0; }
        else { style = 1; }
    } else {
        if (r4e.x < 0.30) { style = 0; }
        else if (r4e.x < 0.55) { style = 1; }
        else if (r4e.x < 0.75) { style = 2; }
        else if (r4e.x < 0.90) { style = 3; }
        else { style = 4; }
    }
    c.win_style = style;
    if (style == 1) {          // ribbon: low continuous bands, thin mullions
        c.win_pitch = 3.0 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.04, 0.34);
        c.pane_hi = vec2<f32>(0.96, 0.86);
    } else if (style == 2) {   // vertical strips: narrow, full-height
        c.win_pitch = 2.2 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.32, 0.10);
        c.pane_hi = vec2<f32>(0.58, 0.94);
    } else if (style == 3) {   // curtain wall: all glass, hairline mullions
        c.win_pitch = 2.8 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.05, 0.08);
        c.pane_hi = vec2<f32>(0.95, 0.94);
        c.palette_bias = c.palette_bias * 0.5;   // cooler house spectrum
    } else if (style == 4) {   // punched: small openings in a lot of wall
        c.win_pitch = 2.6 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.30, 0.38);
        c.pane_hi = vec2<f32>(0.70, 0.80);
        c.palette_bias = min(c.palette_bias * 1.4 + 0.1, 1.0); // warmer
    } else {                   // grid: the founding lattice
        c.win_pitch = 3.4 * (0.85 + 0.40 * r4e.y);
        c.pane_lo = vec2<f32>(0.14, 0.28);
        c.pane_hi = vec2<f32>(0.86, 0.90);
    }
    let pane_span = c.pane_hi - c.pane_lo;
    c.pane_frac = pane_span.x * pane_span.y;
    // A fifth of buildings keep a house color: every window draws near one
    // palette coordinate, so a tower can be THE cyan tower — the color
    // balance is a property of the building, not only of the window
    // (Thomas, 2026-08-20).
    let mono_draw = city_rand4(
        c.seed ^ vec2<u32>(0x3c6ef372u, 0xa54ff53au));
    c.win_mono = select(-1.0, mono_draw.y, mono_draw.x < 0.20);
    // Widen the occupancy spread too: some towers near-dark, some ablaze,
    // so neighbours separate at range by brightness as well as texture.
    c.lit_frac = clamp(
        c.lit_frac * (0.45 + 1.35 * r4e.z * r4e.z), 0.02, 0.50);

    // Roof furniture belongs on roofs: a spire crown has no flat top to
    // stand a mast on, and a tapered shaft only offers its shrunken crown
    // cap (Thomas, 2026-08-20: bridges/antennas only on the building types
    // that can carry them — components read cc.arch for the same rule).
    if (h > CITY_MAST_MIN_H && r4b.z < 0.65 && c.arch != 4) {
        c.has_mast = true;
        let mast_h = 12.0 + 45.0 * r4b.w;
        let crown = select(1.0, c.fscale * 0.85, c.arch == 3);
        let moff = (vec2<f32>(r4c.w, r4.x) - 0.5) * fw * (0.4 * crown);
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
    let ax = select(1.0, 1.6, city_is_avenue(bxi));
    let inx = city_lamp_inset(bxi);
    for (var s: i32 = 0; s < 2; s = s + 1) {
        let lx = bx + select(-inx, inx, s == 1);
        let ly = round(p.y / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
        let d2 = (p.x - lx) * (p.x - lx) + (p.y - ly) * (p.y - ly);
        pool = pool + ax * exp(-d2 / (2.0 * CITY_LAMP_SIGMA * CITY_LAMP_SIGMA));
    }
    // Nearest y-boundary (a street running along x).
    let by = round(p.y / cell) * cell;
    let byi = i32(round(p.y / cell));
    let ay = select(1.0, 1.6, city_is_avenue(byi));
    let iny = city_lamp_inset(byi);
    for (var s: i32 = 0; s < 2; s = s + 1) {
        let ly = by + select(-iny, iny, s == 1);
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

        // Per-window pattern, on this building's own lattice. The style
        // decides which UNIT switches on: single panes (grid, punched),
        // floor-segments (ribbon), multi-storey column runs (strips), or
        // whole floors (curtain wall) — so towers separate at range by
        // texture, not just by occupancy.
        let iu = i32(floor(uc / cc.win_pitch));
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
        let fu = fract(uc / cc.win_pitch);
        let fv = fract(vc / CITY_FLOOR_H);
        let pane = fu > cc.pane_lo.x && fu < cc.pane_hi.x
                && fv > cc.pane_lo.y && fv < cc.pane_hi.y;
        var is_on = wh.x < cc.lit_frac;
        if (cc.win_style == 1) {
            let sh = city_rand4(vec2<u32>(
                cc.seed.x ^ (bitcast<u32>(iu >> 1) * 0x9e3779b9u),
                cc.seed.y ^ (bitcast<u32>(iv) * 0x51ed270bu)));
            is_on = sh.x < cc.lit_frac * 1.15;
        } else if (cc.win_style == 2) {
            let sh = city_rand4(vec2<u32>(
                cc.seed.x ^ (bitcast<u32>(iu) * 0x9e3779b9u),
                cc.seed.y ^ (bitcast<u32>(iv >> 2) * 0x51ed270bu)));
            is_on = sh.x < cc.lit_frac * 1.10;
        } else if (cc.win_style == 3) {
            let fl = city_rand4(vec2<u32>(
                cc.seed.x ^ 0x7feb352du,
                bitcast<u32>(iv) * 0x846ca68bu));
            is_on = fl.x < cc.lit_frac * 1.5 && wh.x < 0.88;
        }
        let lit = pane && !floor_dark && is_on;
        var e_win = vec3<f32>(0.0);
        if (lit) {
            // Curtain-wall floors run dimmer per pane than a lamp-lit room
            // — office fluorescents against far more glass — and ribbon
            // bands a touch below grid; without this the glass towers
            // white out whole canyons.
            let style_gain = select(
                select(1.0, 0.80, cc.win_style == 1),
                0.55, cc.win_style == 3);
            let bright = (0.25 + 5.0 * pow(wh.z, 7.0)) * style_gain;
            // A monochrome house draws every window near its one palette
            // coordinate; a mixed house draws freely.
            var cdraw = wh.y;
            if (cc.win_mono >= 0.0) {
                cdraw = clamp(cc.win_mono + (wh.y - 0.5) * 0.08, 0.0, 1.0);
            }
            e_win = city_window_color(cdraw, cc.palette_bias)
                    * (CITY_WIN_RADIANCE * bright);
            // Life inside: whatever the glyph components put between the
            // light and the glass (curtains, figures, androids).
            let pane_uv = (vec2<f32>(fu, fv) - cc.pane_lo)
                          / max(cc.pane_hi - cc.pane_lo, vec2<f32>(1e-4));
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
            * (cc.lit_frac * cc.pane_frac * (1.0 - CITY_DARK_FLOOR_FRAC)
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
            * (cc.lit_frac * cc.pane_frac * (1.0 - CITY_DARK_FLOOR_FRAC)
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
        let street_scale = 0.20 + 1.3 * smoothstep(0.02, 0.45, district);
        let pool_blend = smoothstep(
            CITY_STREET_LOD_START, CITY_STREET_LOD_FULL, fp);
        let pools = mix(
            city_street_pools(h.pos.xy), CITY_STREET_MEAN_POOL, pool_blend);
        // Soft knee: a downtown avenue's pools ran to radiance ~10 and
        // clipped to featureless white (streetlife's report). The knee
        // passes the dim outskirts untouched and compresses only the top.
        let raw_street = CITY_LAMP_RADIANCE * street_scale * pools;
        street = select(0.0,
                        raw_street / (1.0 + raw_street * 0.30),
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
// --- component: windowlife (windowlife.wgsl) ---
// windowlife — what stands between the light and the glass.
//
// A lit pane is 2.45 m wide by 2.23 m tall (CITY_WIN_PITCH_U and
// CITY_FLOOR_H times the pane fractions). Most rooms show nothing but their
// light. A quarter have something drawn across them. One in seven holds a
// person, dark against the room behind them — the thing that makes a wall
// read as inhabited rather than as a lamp array. One in a hundred holds
// something whose eyes are lit from inside.
//
// Everything here is a transmission MULTIPLIER on the pane's own emission,
// so it can only take light away — except the android's eyes, which return
// a multiplier above 1 and so read as emission through the same interface.
//
// LOD is the whole discipline here. These glyphs are sub-window detail:
// they must be gone before the core's window->block dissolve even starts
// (CITY_WIN_LOD_START, fp_eff 1.6 m/px). Each case blends to its OWN mean
// transmission over a footprint window sized to that case's FINEST feature,
// not to the pane — a blind's 0.4 m stripe and an android's 5 cm eye alias
// long before a curtain's edge does, so they die sooner. Past its window a
// case returns a constant, which is what lets the core's own dissolve stay
// smooth. Nothing here ever just vanishes.
//
// The means are measured, not guessed: 4e6 Monte Carlo draws of (window
// hash, pane_uv) against a numpy port of this file, so the hand-off from
// pattern to mean is invisible and the far-field facade statistics are
// exactly what they were before this component existed.

const cc_windowlife_PANE_W: f32 = 2.448;   // CITY_WIN_PITCH_U * (U_HI - U_LO)
const cc_windowlife_PANE_H: f32 = 2.232;   // CITY_FLOOR_H * (V_HI - V_LO)
const cc_windowlife_ASPECT: f32 = 1.0968;  // W / H; makes circles round

// Case shares. Nothing takes the rest (60%).
const cc_windowlife_T_DRAPE: f32 = 0.60;    // .60-.85  drapes    25%
const cc_windowlife_T_FIGURE: f32 = 0.85;   // .85-.99  figures   14%
const cc_windowlife_T_ANDROID: f32 = 0.99;  // .99-1.0  androids   1%

// Transmissions through each thing, and the ensemble mean each dissolves
// into. Curtain cloth is warm (it is lit from behind by a warm room and
// eats blue); a body is near-neutral and much darker; aluminium blinds are
// neutral. See the header on how the means were obtained.
const cc_windowlife_CURTAIN_T: vec3<f32> = vec3<f32>(0.32, 0.26, 0.19);
const cc_windowlife_CURTAIN_MEAN: vec3<f32> = vec3<f32>(0.686, 0.660, 0.631);
const cc_windowlife_BLIND_T: f32 = 0.42;
const cc_windowlife_BLIND_MEAN: vec3<f32> = vec3<f32>(0.733);
const cc_windowlife_BODY_T: vec3<f32> = vec3<f32>(0.10, 0.09, 0.09);
const cc_windowlife_FIGURE_MEAN: vec3<f32> = vec3<f32>(0.822, 0.820, 0.820);
// The eyes: a multiplier well above 1 in G and B and near zero in R, so the
// dot lands cyan whatever colour the room behind it is.
const cc_windowlife_EYE_GAIN: vec3<f32> = vec3<f32>(0.05, 3.0, 2.8);
const cc_windowlife_EYE_SEP: f32 = 0.028;   // half-separation, pane widths
const cc_windowlife_EYE_SIGMA: f32 = 0.0060; // ~0.02 pane: a point, not a lamp

// Footprint windows (m/px) over which each case blends into its mean. Set
// by the case's finest feature, all of them closed before fp_eff 1.6.
const cc_windowlife_LOD_CURTAIN: vec2<f32> = vec2<f32>(0.40, 1.50);
const cc_windowlife_LOD_BLIND: vec2<f32> = vec2<f32>(0.18, 0.60);
const cc_windowlife_LOD_FIGURE: vec2<f32> = vec2<f32>(0.30, 1.20);
const cc_windowlife_LOD_EYE: vec2<f32> = vec2<f32>(0.05, 0.18);

struct cc_windowlife_Pose {
    head: vec2<f32>,   // head centre in pane_uv
    head_r: f32,       // head radius in u units (u is the round one)
    torso_x: f32,      // torso centre in u
    torso_w: f32,      // torso half-width in u
    shoulder: f32,     // v of the shoulder line
    tilt: f32,         // shoulder line slope; one side rides higher
}

// wh's components already carry the core's meanings — x decides lit, y the
// palette, z the brightness — so a draw taken straight off them would be
// conditioned on the window being lit and correlated with its colour. Mix
// all four back through the city's own hash instead.
fn cc_windowlife_hash(wh: vec4<f32>) -> vec4<f32> {
    let a = u32(wh.x * 16777216.0) * 0x9e3779b9u
          ^ u32(wh.w * 16777216.0) * 0x85ebca6bu;
    let b = u32(wh.y * 16777216.0) * 0xc2b2ae35u
          ^ u32(wh.z * 16777216.0) * 0x27d4eb2fu;
    return city_rand4(vec2<u32>(a ^ 0x7f4a7c15u, b ^ 0x1b56c4e9u));
}

// Half a pixel in pane-width units: every soft edge below opens by this, so
// an edge stays about a pixel wide instead of crawling as the camera moves.
fn cc_windowlife_aa_u(fp: f32) -> f32 {
    return min(0.6 * fp / cc_windowlife_PANE_W, 0.30);
}

fn cc_windowlife_aa_v(fp: f32) -> f32 {
    return min(0.6 * fp / cc_windowlife_PANE_H, 0.30);
}

// Curtains drawn part-way, or blinds. Both are whole-pane furniture, so
// both hold to a coarser footprint than a figure does — but the blind's
// stripe is finer than the curtain's edge, and it dies first.
fn cc_windowlife_drape(h: vec4<f32>, uv: vec2<f32>, fp: f32) -> vec3<f32> {
    if (h.y < 0.55) {
        // Two masses closing in from the sides, 20-70% of the pane between
        // them, split unevenly — one side is nearly always drawn further.
        let lod = smoothstep(cc_windowlife_LOD_CURTAIN.x,
                             cc_windowlife_LOD_CURTAIN.y, fp);
        if (lod >= 1.0) {
            return cc_windowlife_CURTAIN_MEAN;
        }
        let cov = 0.20 + 0.50 * h.z;
        let split = 0.15 + 0.70 * h.w;
        let soft = 0.022 + cc_windowlife_aa_u(fp);
        // Cloth hangs: the drawn edge leans a few centimetres off vertical,
        // and that lean is most of what separates a curtain from a gradient.
        let lean = (0.055 * (h.z - 0.5)) * (uv.y - 0.5);
        let l = cov * split + lean;
        let r = 1.0 - cov * (1.0 - split) + lean;
        let m = clamp(max(1.0 - smoothstep(l - soft, l + soft, uv.x),
                          smoothstep(r - soft, r + soft, uv.x)),
                      0.0, 1.0);
        // The hem leaks a little more light than the rod end. Two touches
        // of cloth, both monotone in position and so free of any repeat
        // frequency to alias: the vertical lean of the light, and a denser
        // gather in the first few centimetres behind the drawn edge, which
        // is what a curtain has and a gradient does not.
        let din = max(max(l - uv.x, uv.x - r), 0.0);
        let bunch = 1.0 - 0.25 * exp(-din * 18.0);
        let cloth = cc_windowlife_CURTAIN_T
                  * ((0.89 + 0.22 * (1.0 - uv.y)) * bunch);
        return mix(mix(vec3<f32>(1.0), cloth, m),
                   cc_windowlife_CURTAIN_MEAN, lod);
    }
    // Blinds: 4-6 slats. The duty cycle is symmetric about its edges so
    // widening them for the footprint blurs without darkening.
    let lod = smoothstep(cc_windowlife_LOD_BLIND.x,
                         cc_windowlife_LOD_BLIND.y, fp);
    if (lod >= 1.0) {
        return cc_windowlife_BLIND_MEAN;
    }
    let nb = floor(4.0 + 2.99 * h.z);
    let s = fract(uv.y * nb + h.w);
    let e = 0.07 + cc_windowlife_aa_v(fp) * nb;
    let band = smoothstep(0.20 - e, 0.20 + e, s)
             * (1.0 - smoothstep(0.66 - e, 0.66 + e, s));
    return mix(mix(vec3<f32>(1.0), vec3<f32>(cc_windowlife_BLIND_T), band),
               cc_windowlife_BLIND_MEAN, lod);
}

// Three poses. Not three people — three ways a body stands against a lit
// room, which is all that survives 30 m of night air anyway. The shoulder
// line is always tilted and the head always carried a little off the torso
// centre: a square-on symmetric figure reads as a pictogram, not a person,
// and that is the failure mode this case has to avoid.
fn cc_windowlife_pose(h: vec4<f32>) -> cc_windowlife_Pose {
    let g = city_rand4(vec2<u32>(u32(h.y * 16777216.0) ^ 0x68e31da4u,
                                 u32(h.z * 16777216.0) ^ 0xb5297a4du));
    let side = select(-1.0, 1.0, g.x < 0.5);
    let scale = 0.86 + 0.30 * g.z;
    let tilt = side * (0.05 + 0.13 * g.w);
    if (h.w < 0.45) {
        // Standing at one edge of the pane, looking out; often half out of
        // frame, which is how you actually catch someone at a window.
        let hx = 0.5 + side * (0.22 + 0.17 * g.y);
        let hy = 0.66 + 0.10 * g.w;
        return cc_windowlife_Pose(vec2<f32>(hx, hy), 0.085 * scale,
                                  hx - side * (0.02 + 0.03 * g.y),
                                  0.190 * scale, hy - 0.080 * scale, tilt);
    }
    if (h.w < 0.80) {
        // Sitting: low in the frame and wide, the shoulders nearer the sill
        // than the head is to the lintel.
        let hx = 0.5 + (g.y - 0.5) * 0.58;
        let hy = 0.40 + 0.12 * g.w;
        return cc_windowlife_Pose(vec2<f32>(hx, hy), 0.082 * scale,
                                  hx + side * (0.02 + 0.04 * g.y),
                                  0.215 * scale, hy - 0.076 * scale, tilt * 0.7);
    }
    // Leaning: the head carried well off the line of the torso.
    let hx = 0.5 + (g.y - 0.5) * 0.55;
    let hy = 0.62 + 0.09 * g.w;
    return cc_windowlife_Pose(vec2<f32>(hx, hy), 0.084 * scale,
                              hx - side * (0.06 + 0.04 * g.w),
                              0.170 * scale, hy - 0.078 * scale, tilt * 1.3);
}

// Head plus shoulders-and-torso, both soft-edged. A hard-edged version of
// this reads as a sticker; the soft edge is what makes it a person seen
// through glass.
fn cc_windowlife_figure_mask(p: cc_windowlife_Pose, uv: vec2<f32>,
                             fp: f32) -> f32 {
    let au = cc_windowlife_aa_u(fp);
    // Heads are taller than they are wide.
    let d = vec2<f32>(uv.x - p.head.x,
                      (uv.y - p.head.y) / (cc_windowlife_ASPECT * 1.12));
    let head = 1.0 - smoothstep(p.head_r * 0.76,
                                p.head_r * 1.14 + au, length(d));
    // The shoulder line is tilted and starts INSIDE the head, so the body
    // grows out from behind it — a shoulder line below the jaw leaves a gap
    // of light at the neck and turns the whole thing into a lollipop. Full
    // width about 17 cm down, narrowing again toward the waist.
    let dx = uv.x - p.torso_x;
    let below = p.shoulder - uv.y + p.tilt * dx;
    let w = p.torso_w * smoothstep(-0.015, 0.105, below)
                      * (1.0 - 0.16 * smoothstep(0.12, 0.45, below));
    let soft = 0.028 + au;
    let torso = (1.0 - smoothstep(w - soft, w + soft, abs(dx)))
              * smoothstep(-0.02 - au, 0.03 + au, below);
    return clamp(max(head, torso), 0.0, 1.0);
}

// Two points of light where the eyes are. The gaussian is widened by the
// footprint with its integral held fixed, so the dots dim honestly as they
// go sub-pixel instead of flickering, and are gone by 0.18 m/px — this is a
// thing you find by flying close, not a feature of the skyline.
fn cc_windowlife_eyes(p: cc_windowlife_Pose, uv: vec2<f32>,
                      fp: f32) -> vec3<f32> {
    let fade = 1.0 - smoothstep(cc_windowlife_LOD_EYE.x,
                                cc_windowlife_LOD_EYE.y, fp);
    if (fade <= 0.0) {
        return vec3<f32>(0.0);
    }
    let s0 = cc_windowlife_EYE_SIGMA;
    let px = 0.5 * fp / cc_windowlife_PANE_W;
    let s = sqrt(s0 * s0 + px * px);
    let amp = (s0 * s0) / (s * s);      // conserve the integral
    let cy = p.head.y + 0.014 * (p.head_r / 0.085);
    let d1 = vec2<f32>(uv.x - p.head.x + cc_windowlife_EYE_SEP,
                       (uv.y - cy) / cc_windowlife_ASPECT);
    let d2 = vec2<f32>(uv.x - p.head.x - cc_windowlife_EYE_SEP,
                       (uv.y - cy) / cc_windowlife_ASPECT);
    let k = -0.5 / (s * s);
    let g = exp(k * dot(d1, d1)) + exp(k * dot(d2, d2));
    return cc_windowlife_EYE_GAIN * (g * amp * fade);
}

fn cc_windowlife_figure(h: vec4<f32>, uv: vec2<f32>, fp: f32,
                        android: bool) -> vec3<f32> {
    let lod = smoothstep(cc_windowlife_LOD_FIGURE.x,
                         cc_windowlife_LOD_FIGURE.y, fp);
    if (lod >= 1.0) {
        return cc_windowlife_FIGURE_MEAN;   // the eyes average into nothing
    }
    let p = cc_windowlife_pose(h);
    let m = cc_windowlife_figure_mask(p, uv, fp);
    let body = mix(vec3<f32>(1.0), cc_windowlife_BODY_T, m);
    if (!android) {
        return mix(body, cc_windowlife_FIGURE_MEAN, lod);
    }
    // The eyes ride on top of the silhouette and above 1: they are the one
    // thing in this file that adds light rather than taking it.
    let lit = body + cc_windowlife_eyes(p, uv, fp) * m;
    return mix(lit, cc_windowlife_FIGURE_MEAN, lod);
}

fn cc_windowlife_glyph(cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>,
                       fp: f32) -> vec3<f32> {
    let h = cc_windowlife_hash(wh);
    if (h.x < cc_windowlife_T_DRAPE) {
        return vec3<f32>(1.0);      // most windows are just light
    }
    if (h.x < cc_windowlife_T_FIGURE) {
        return cc_windowlife_drape(h, pane_uv, fp);
    }
    return cc_windowlife_figure(h, pane_uv, fp,
                                h.x >= cc_windowlife_T_ANDROID);
}

// --- component: aircars (aircars.wgsl) ---
// aircars — flying cars, frozen mid-flow.
//
// The sky layer of a two-layer traffic system: a sibling component parks the
// cars on the ground, these hold the air above the avenues. There is no time
// here (SPEC rule 4), so this is a long-exposure photograph of living
// traffic — every craft is where its hash says it is, on every frame and from
// every camera, and the streams read as streams because of where they are,
// not because they move.
//
// LAYOUT. Traffic follows the avenue lattice (`city_is_avenue`, every 8
// blocks) but flies off to one side of it and well above. Each avenue carries
// two counter-flowing lanes, one over each verge, ~9 m in from the avenue
// centerline; the two lanes belong to the two cells that flank the avenue —
// the cell whose MIN edge is the avenue owns the lane on its own side, the
// cell whose MAX edge is the avenue owns the other. That ownership rule is
// what keeps every craft geometrically inside the cell that draws it, which
// the DDA requires: a box straddling a cell boundary is tested only from one
// side and disappears from the other. It also makes the flow directions fall
// out right-hand — keep to the right of the avenue you are flying up, and the
// +x verge of a north-south avenue runs north.
//
// Altitude is a two-deck system, ~55 m and ~95 m, +-8 m per craft, chosen by
// the craft's own hash — so a lane is a loose braid rather than a wire, and
// the low deck passes under the high one at every crossing. Over ordinary
// blocks a few percent of cells carry a free flyer at 40-130 m, off-lattice
// and on any heading, which is what stops the air from being a pure grid.
//
// Occupancy rides the cascade the way the streetlights do: a downtown avenue
// is a stream, an outskirts lane is three craft in a kilometre. That is
// `cc.density`, already loaded by the core — free.
//
// LIGHT. The Cloudpunk cue is the UNDERGLOW: an emissive belly panel, cyan /
// magenta / amber, which is the only part of a craft big enough to survive
// past a few hundred metres. From the street you read bellies overhead; from
// 400 m the avenues are strings of colored points above the sodium lattice;
// further out the layer is drifting specks. White nose and red tail lamps
// live in the hull's own shading rather than in geometry of their own, and
// dissolve into their face's mean as the footprint outgrows them.
//
// COST. Cheap gates first, in order: the segment's z-extent against the
// layer's [35, 140] m slab (no hash, no arithmetic beyond two multiply-adds);
// the pixel footprint; then the two avenue tests, which are integer. A cell
// with no avenue edge therefore spends exactly one hash draw — the free-flyer
// coin — and returns. At most three craft touch a cell (one x-lane, one
// y-lane, one free flyer; the two x-edge tests are mutually exclusive at
// avenue period 8), and each craft is two boxes, the second of which drops
// out once it is sub-pixel: <= 6 box tests and 6 pcg2d draws worst case,
// 2 pcg2d in the common one.
//
// CLOSE READ (the SDF, 2026-08-20). Thomas flew the city and the craft "are
// not really clearly cars, they are boxes with a few circles" — true, and the
// far read was never the problem. So the boxes stay exactly as they were
// beyond half a metre per pixel, and inside that gate the box test becomes a
// bounding test only: a hit on the padded box hands off to a sphere trace of
// a vehicle SDF in the craft's own frame (SPEC's SDF-in-a-box pattern). The
// SDF is a rounded lifting body — tapering nose, blunt tail, stub sponsons,
// twin rear nacelles, a canopy bubble smooth-minned onto the deck, a dorsal
// fin and a whip antenna, light housings raised at the sponson tips. If the
// trace misses inside the box the craft misses: rays grazing past the curved
// hull are the whole reason it reads as curved rather than as a slab.
//
// Everything the eye uses to call a shape a vehicle rather than a box is in
// the NORMAL, so the shading is written against it: rim brightness at the
// silhouette, a narrow moon sheen that slides along the crown, the underglow
// falling off as the belly turns into the flank, and the canopy seam found as
// the crease between two SDF parts rather than drawn as a stripe. Panel
// lines, intakes and thruster discs are shading bands in the local frame —
// per SPEC they are cheaper there than in geometry, and at wiper-blade
// calibration they are the right size to be features rather than texture.

// --- craft geometry (a compact wedge, 3.8 x 1.9 x 1.1 m) --------------------
const cc_aircars_HALF_L: f32 = 1.90;
const cc_aircars_HALF_W: f32 = 0.95;
const cc_aircars_HULL_H: f32 = 0.70;
const cc_aircars_CANOPY_H: f32 = 0.40;
const cc_aircars_CANOPY_HALF_L: f32 = 1.10;
const cc_aircars_CANOPY_HALF_W: f32 = 0.60;
const cc_aircars_CANOPY_AFT: f32 = 0.45;  // canopy sits aft: the nose tapers

// --- the close-read SDF -----------------------------------------------------
// Local frame: +x along the direction of travel, +y to port, z measured up
// from the hull bottom (so z = 0 is the belly plane and z = HULL_H the deck).
// Every number below is in metres in that frame, and the whole solid lives
// inside |x| <= 1.88, |y| <= 1.06, 0 <= z <= 1.10 — which is what sets the
// padding on the bounding box.
const cc_aircars_SDF_FP: f32 = 0.50;     // hand-off: boxes beyond this
const cc_aircars_SDF_ITERS: i32 = 24;
// Bounding box half-extents, along-travel and across. The hull inside is
// yawed by up to +-YAW, and the box is NOT: 1.88*cos(5) + 1.06*sin(5) = 1.965
// along and 1.88*sin(5) + 1.06*cos(5) = 1.220 across, plus slack.
const cc_aircars_SDF_PAD_L: f32 = 2.04;
const cc_aircars_SDF_PAD_W: f32 = 1.36;
const cc_aircars_SDF_PAD_ZLO: f32 = 0.06;
const cc_aircars_SDF_PAD_ZHI: f32 = 1.18;
const cc_aircars_YAW: f32 = 0.0873;      // +-5 degrees, per craft
// Housing / lens radius for the nav lamps, and where the thruster discs sit.
const cc_aircars_HOUSE_R: f32 = 0.098;
const cc_aircars_HOUSE_Z: f32 = 0.26;    // housing / sponson axis height
const cc_aircars_LENS_OUT: f32 = 0.60;   // how much of the bump is the lens
// The nacelle axis has to sit far enough outboard that the pod stands PROUD
// of the hull: at |y| = 0.66 the pod's own 0.23 m radius put its outer skin
// at 0.89 against a hull half-width of 0.86, so the engines were swallowed
// and the craft had no visible engines at all from the side — the one item of
// the close read that the first pass silently lost. At 0.80 the pod clears
// the flank by ~0.17 m, which is 19 px at 25 m: a pod on a pylon, which is
// what the eye is looking for when it looks for an engine.
const cc_aircars_NAC_Y: f32 = 0.80;      // nacelle axis, |y|
const cc_aircars_NAC_Z: f32 = 0.52;      // nacelle axis height
const cc_aircars_THRUST_R: f32 = 0.15;   // thruster disc radius
// The chine line: belly below it, flank above. The underglow's side strip is
// a band centred on it, |z - CHINE_Z| running LO to HI.
const cc_aircars_CHINE_Z: f32 = 0.145;
const cc_aircars_CHINE_LO: f32 = 0.018;
const cc_aircars_CHINE_HI: f32 = 0.050;

// --- lanes ------------------------------------------------------------------
const cc_aircars_LANE_OFF: f32 = 9.0;    // lane centerline in from the avenue
const cc_aircars_LANE_JIT: f32 = 4.0;    // +- lateral wander
const cc_aircars_ALONG_JIT: f32 = 34.0;  // +- along-lane wander in its slot
const cc_aircars_Z_LOW: f32 = 55.0;
const cc_aircars_Z_HIGH: f32 = 95.0;
const cc_aircars_Z_JIT: f32 = 8.0;
const cc_aircars_HIGH_CUT: f32 = 0.55;   // hash above this -> the high deck
const cc_aircars_LANE_P: f32 = 0.72;     // slot occupancy at full density
const cc_aircars_DENS_LO: f32 = 0.50;    // occupancy scale, outskirts
const cc_aircars_DENS_HI: f32 = 1.35;    // occupancy scale, downtown
// The window the occupancy ramp spans, in block density. Set against the
// tile's own distribution, which is lognormal and brutally skewed — median
// block 0.045, p75 0.10, p90 0.22 — so a ramp calibrated by eye on the
// megatower district leaves the MODAL city with one craft every 300 m of
// lane, which reads as scattered dust rather than as traffic. Anchored at
// the median instead: a typical avenue carries one craft per ~80 m counting
// both directions, downtown roughly one per 50, the outskirts one per 300.
const cc_aircars_DENS_START: f32 = 0.005;
const cc_aircars_DENS_FULL: f32 = 0.070;
// Free flyers: off-lattice singles over ordinary blocks.
const cc_aircars_FREE_FRAC: f32 = 0.07;
const cc_aircars_FREE_Z_LO: f32 = 40.0;
const cc_aircars_FREE_Z_HI: f32 = 130.0;
const cc_aircars_FREE_MARGIN: f32 = 12.0;
// Segment z-gate: the whole air layer lives in here, hull bottom to canopy.
const cc_aircars_Z_MIN: f32 = 35.0;
const cc_aircars_Z_MAX: f32 = 140.0;

// --- light ------------------------------------------------------------------
// Yardsticks from SPEC rule 5: lit window 3.5, storefront 2.2, lamp pool 0.7.
// The underglow sits between a lamp pool and a storefront — a lit panel, not
// a source you look into — and the nav lamps are point-bright.
const cc_aircars_GLOW_RAD: f32 = 2.5;
// The panel is inset in the belly, not the whole belly: seen from underneath
// at street range a craft has to read as a dark object carrying a light, not
// as a floating rectangle of light. The border keeps a quarter of the
// radiance (the panel spilling onto its own frame), which is also what keeps
// the far-field mean near the full value where the inset is sub-pixel.
const cc_aircars_PANEL_A: f32 = 1.62;    // panel half-length (of 1.90)
const cc_aircars_PANEL_W: f32 = 0.79;    // panel half-width (of 0.95)
const cc_aircars_PANEL_RIM: f32 = 0.25;  // what the border keeps
const cc_aircars_LAMP_RAD: f32 = 6.0;
const cc_aircars_LAMP_R: f32 = 0.12;     // nav-lamp dot radius (m)
const cc_aircars_CANOPY_RAD: f32 = 0.40;
const cc_aircars_HULL_FILL: f32 = 0.05;  // hull albedo against the skyglow
// The panel does not end at the belly seam: it wraps the lower flanks as a
// skirt and dies out toward the shoulder. This is what carries the layer at
// distance — from anywhere but directly underneath, the skirt is the only
// piece of a craft with the craft's own color on it, and from 400 m up a
// lane the strings of light you read are skirts, not bellies. Its mean over
// the 0.7 m flank is what a sub-pixel craft delivers, so it is set against
// the window yardstick (3.5) rather than against the belly's own radiance.
const cc_aircars_SKIRT: f32 = 0.86;
const cc_aircars_SKIRT_LO: f32 = 0.10;   // fully lit below this height (m)
const cc_aircars_SKIRT_HI: f32 = 0.62;   // gone by this one
// Dorsal running light: the same color, dimmer, on the hull's back. Without
// it a craft seen from above is a black chip on a bright street, and the
// whole aerial read of the layer goes with it.
const cc_aircars_DORSAL: f32 = 0.76;
// Close-read light, all of it keyed to the SDF normal. A dark hull against a
// dark city has exactly three ways to show its shape: the silhouette rim, a
// specular slid across the crown by the moon, and the way its own underglow
// dies as the belly turns into the flank. RIM is the loudest of the three and
// the one that does the work — it is the lit edge that says "curved".
const cc_aircars_RIM: f32 = 0.55;
const cc_aircars_HULL_DIFF: f32 = 0.80;  // moonlight on dark metal
const cc_aircars_BOUNCE: f32 = 0.016;    // the panel's spill onto its own hull
const cc_aircars_BAR: f32 = 0.45;        // the chine light bar
const cc_aircars_CABIN: f32 = 0.22;      // how much interior the glass shows
const cc_aircars_SHEEN: f32 = 0.34;      // moon glint, hull
const cc_aircars_GLASS: f32 = 0.55;      // moon glint, canopy (tighter, harder)
const cc_aircars_UPGLOW: f32 = 0.020;    // street uplight caught underneath
const cc_aircars_SEAM_DARK: f32 = 0.80;  // canopy-to-hull seam band
const cc_aircars_LINE_DARK: f32 = 0.70;  // panel lines
const cc_aircars_INTAKE_DARK: f32 = 0.85;
const cc_aircars_THRUST: f32 = 1.75;     // thruster disc radiance
const cc_aircars_THRUST_HALO: f32 = 0.32; // what the nacelle cup keeps

// --- LOD --------------------------------------------------------------------
// Footprint (m/px) where each piece of detail hands over to its own mean. The
// canopy is a 2.2 m silhouette bump on a 3.8 m body: it stops paying for
// itself first. The nav lamps are 0.24 m dots: they fade into the mean
// radiance of the face they sit on, so a distant craft keeps its light and
// loses only the resolution — which is what a long lens does to traffic.
const cc_aircars_CANOPY_FP: f32 = 0.80;
const cc_aircars_LAMP_LOD_START: f32 = 0.35;
const cc_aircars_LAMP_LOD_FULL: f32 = 1.20;
// Where the craft stops being traced at all: 6 m/px puts a whole craft at
// two thirds of a pixel. The emission ramps to zero over the run-up to it
// (cc_aircars_FAR_FADE) so the geometry disappears at the moment it stops
// contributing anything, rather than popping out — the tracer's cutoff and
// the shader's mean meet at the same place. At the harness's 960 px / 65 deg
// this threshold sits past CITY_PROP_RANGE and never fires; it exists for
// low-resolution or very wide-angle frames, where it does.
const cc_aircars_FAR_FP: f32 = 6.0;
const cc_aircars_FAR_FADE: f32 = 3.6;    // emission gone by FAR_FP from here
// The core stops calling cell_props at CITY_PROP_RANGE. A hard population
// edge there would be a ring in any long sightline, so the emission tapers
// over the last quarter of the range instead: the far-field mean of a layer
// this sparse IS near zero, and this is that mean approached rather than
// stepped into.
// Kept short deliberately: a craft is about one pixel across out here and the
// population is sparse, so there is no continuous structure for an edge to
// show up in — all the taper has to do is take the last few craft down, and a
// long taper instead eats the whole mid-field of any shallow aerial view,
// which is exactly where the layer is supposed to read.
const cc_aircars_FADE_START: f32 = 0.92;
const cc_aircars_FADE_FULL: f32 = 1.00;

// Per-craft placement draw. One city_rand4 (two pcg2d) carries everything the
// geometry needs; appearance draws its own hash in the shader, where the cost
// is one per hit pixel rather than one per visited cell.
fn cc_aircars_draw(ci: vec2<i32>, lane: i32) -> vec4<f32> {
    let l = u32(lane);
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x9e3779b9u + l * 0x2545f491u + 0x165667b1u,
        bitcast<u32>(ci.y) * 0x85ebca6bu + l * 0xc2b2ae35u + 0x27d4eb2fu));
}

fn cc_aircars_look(ci: vec2<i32>, lane: i32) -> vec4<f32> {
    let l = u32(lane);
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0xc2b2ae35u + l * 0x9e3779b9u + 0x51ed270bu,
        bitcast<u32>(ci.y) * 0x27d4eb2fu + l * 0x85ebca6bu + 0xdeadbeefu));
}

// Underglow palette: 60% cyan, 25% magenta, 15% amber. Cyan carries the layer
// because it is the one hue the sodium-and-tungsten ground does not already
// own — the traffic reads as a separate system, not as more windows.
fn cc_aircars_glow_color(d: f32) -> vec3<f32> {
    if (d < 0.60) {
        return vec3<f32>(0.16, 0.90, 1.00);
    }
    if (d < 0.85) {
        return vec3<f32>(1.00, 0.20, 0.70);
    }
    return vec3<f32>(1.00, 0.60, 0.16);
}

// --- SDF primitives ---------------------------------------------------------
// All exact or conservative (never over-estimating distance), which is what
// sphere tracing needs: the ellipsoid is iq's second-order bound and the
// smooth-min only ever shortens, so a step of `d` can never overshoot a
// surface.
fn cc_aircars_sd_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0)))
           + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn cc_aircars_sd_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn cc_aircars_sd_ellipsoid(p: vec3<f32>, r: vec3<f32>) -> f32 {
    let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

fn cc_aircars_sd_capsule(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

fn cc_aircars_smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// The vehicle, in its own frame. Returns the combined distance in .x and the
// three parts the shader needs to tell apart in .yzw — hull group, canopy,
// nacelles. One function, so the marcher and the shader can never disagree
// about where the canopy ends; the shader gets the parts for free at the hit
// because it has to evaluate the field there anyway.
//
// Wiper-blade calibration: the features here are the ones a person would name
// looking at the thing — nose, cabin, wing stub, engine pod, fin, aerial,
// light housing. Nothing smaller has geometry; the panel lines and the intake
// slot are shading, which is both cheaper and the right scale.
fn cc_aircars_sdf(q: vec3<f32>) -> vec4<f32> {
    // The port/starboard fold: everything paired is built once in |y|.
    let m = vec3<f32>(q.x, abs(q.y), q.z);

    // Lifting body: a flat rounded slab, a shorter block forward and a sphere
    // at the tip, smooth-minned into a taper rather than modelled as one. The
    // blend radii are deliberately smaller than the parts they join — a big
    // k turns the whole thing back into the lozenge the boxes already were.
    // The body tapers in PLAN as well as in profile: from above a constant-
    // width slab is a suitcase whichever way its nose points. The half-width
    // is a function of x, which costs the field a little Lipschitz slack —
    // hence the 0.98, which keeps every step an under-estimate.
    let wide = 0.66 - 0.16 * smoothstep(-0.20, 1.10, q.x);
    var hull = 0.98 * cc_aircars_sd_rbox(q - vec3<f32>(-0.22, 0.0, 0.36),
                                         vec3<f32>(1.02, wide, 0.14), 0.20);
    // The nose sits LOW and the tail sits high: a wedge in profile. On a
    // symmetric body the eye has no way to tell front from back, and a craft
    // you cannot tell the front of is a pod.
    let nose = cc_aircars_sd_rbox(q - vec3<f32>(1.00, 0.0, 0.29),
                                  vec3<f32>(0.38, 0.20, 0.05), 0.18);
    hull = cc_aircars_smin(hull, nose, 0.18);
    let tip = cc_aircars_sd_sphere(q - vec3<f32>(1.70, 0.0, 0.26), 0.12);
    hull = cc_aircars_smin(hull, tip, 0.16);
    // Blunt tail: the transom the nacelles hang off.
    let tail = cc_aircars_sd_rbox(q - vec3<f32>(-1.34, 0.0, 0.38),
                                  vec3<f32>(0.26, 0.48, 0.09), 0.20);
    hull = cc_aircars_smin(hull, tail, 0.16);
    // The chine: a thin plate a little wider than the hull, running its
    // length. It is the one hard horizontal line on the craft, it splits
    // belly from flank, and the underglow strip lives along it. Without it a
    // rounded hull has no waterline and reads as a pebble.
    let chine = cc_aircars_sd_rbox(q - vec3<f32>(-0.30, 0.0, 0.15),
                                   vec3<f32>(0.88, 0.80, 0.012), 0.055);
    hull = cc_aircars_smin(hull, chine, 0.05);
    // Stub sponsons: the stance. They also carry the nav-light housings out
    // to the widest point, which is where an aircraft puts them.
    // Thin. The first sponson was a 0.3 m bar down the whole flank, and a
    // flank that is one bar with a dot at each end is precisely the box-with-
    // circles this component was sent back to fix. A blade proud of the hull
    // by a few centimetres reads as a wing stub; anything thicker reads as
    // the hull.
    let spon = cc_aircars_sd_rbox(m - vec3<f32>(-0.05, 0.86, 0.26),
                                  vec3<f32>(0.46, 0.055, 0.008), 0.045);
    hull = cc_aircars_smin(hull, spon, 0.05);
    let house = min(
        cc_aircars_sd_sphere(m - vec3<f32>(0.38, 0.93, 0.26),
                             cc_aircars_HOUSE_R),
        cc_aircars_sd_sphere(m - vec3<f32>(-0.52, 0.93, 0.26),
                             cc_aircars_HOUSE_R));
    hull = cc_aircars_smin(hull, house, 0.045);
    // Dorsal fin over the tail, and a whip aerial off the nose deck, offset
    // to port because real hardware is not symmetric.
    // The fin has to grow OUT of the transom, not sit on it. Perched at
    // z = 0.86 with a 0.06 blend it overlapped the tail by three centimetres
    // and rendered as exactly what it was: a rounded slab floating above the
    // hull with daylight under it — the box artifact this whole refit exists
    // to remove, reintroduced at the one place nobody looks. Seated lower,
    // blended at 0.13, and RAKED: the sample's x is pushed forward with
    // height, so the solid leans aft the way a fin does. A shear is not an
    // isometry, so the field is scaled by 1/sqrt(1 + 0.55^2) = 0.876 to stay
    // an under-estimate for the marcher, the same correction the canopy makes.
    let fq = q - vec3<f32>(-1.38, 0.0, 0.74);
    let fin = 0.876 * cc_aircars_sd_rbox(
        vec3<f32>(fq.x + 0.55 * fq.z, fq.y, fq.z),
        vec3<f32>(0.20, 0.010, 0.16), 0.045);
    hull = cc_aircars_smin(hull, fin, 0.13);
    let ant = cc_aircars_sd_capsule(q, vec3<f32>(1.02, 0.22, 0.56),
                                    vec3<f32>(0.96, 0.22, 1.00), 0.028);
    hull = min(hull, ant);

    // Twin nacelles on the rear flanks; their aft caps are the thrusters.
    // Seated aft of the nav housing rather than over it: moved outboard, the
    // pod's 0.23 m radius reached forward far enough to swallow the aft lens
    // and the red lamp came back as a squashed smear. Shorter and further
    // back leaves 0.06 m between pod skin and housing, which the 0.09 blend
    // turns into a pylon root instead of a collision.
    let pods = cc_aircars_sd_rbox(
        m - vec3<f32>(-1.14, cc_aircars_NAC_Y, cc_aircars_NAC_Z),
        vec3<f32>(0.40, 0.045, 0.045), 0.185);
    // Cabin bubble, seated aft of the nose the way the box canopy was — and
    // short enough in x that there is a HOOD in front of it. A bubble that
    // runs into the nose is a fuselage; a bubble with deck fore and aft of it
    // is a cabin, and that is the difference between reading as a car and
    // reading as a pod.
    // The bubble is SHEARED forward with height, which rakes the windshield
    // and gives the cabin a back-slope — the profile everyone reads as a
    // passenger compartment. A shear is not an isometry, so the field is
    // scaled by 1/sqrt(1 + s^2) to stay a valid (under-)estimate for the
    // marcher; s = 0.42 gives 0.922.
    let ck = q - vec3<f32>(-0.28, 0.0, 0.58);
    let canopy = 0.922 * cc_aircars_sd_ellipsoid(
        vec3<f32>(ck.x + 0.42 * ck.z, ck.y, ck.z),
        vec3<f32>(0.76, 0.46, 0.44));

    var d = cc_aircars_smin(hull, pods, 0.09);
    d = cc_aircars_smin(d, canopy, 0.075);
    return vec4<f32>(d, hull, canopy, pods);
}

struct cc_aircars_Craft {
    ok: bool,
    base: vec3<f32>,  // hull bottom, craft centered in xy
    axis: i32,        // 0 = travels along x, 1 = travels along y
    fwd: f32,         // +1 or -1 along that axis
    yaw: f32,         // small per-craft heading offset (rad)
}

fn cc_aircars_no_craft() -> cc_aircars_Craft {
    return cc_aircars_Craft(false, vec3<f32>(0.0), 0, 1.0, 0.0);
}

// The craft's heading as a unit vector in world xy: the lane axis turned by
// the craft's own yaw. Local <-> world is then a rotation about z, so a
// normal transforms with the same two numbers and z is shared outright — the
// shader can read h.normal.z as "up" without converting anything.
fn cc_aircars_frame(v: cc_aircars_Craft) -> vec2<f32> {
    let cd = cos(v.yaw);
    let sd = sin(v.yaw);
    if (v.axis == 0) {
        return v.fwd * vec2<f32>(cd, sd);
    }
    return v.fwd * vec2<f32>(-sd, cd);
}

fn cc_aircars_to_local(f: vec2<f32>, w: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(f.x * w.x + f.y * w.y, f.x * w.y - f.y * w.x, w.z);
}

fn cc_aircars_to_world(f: vec2<f32>, l: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(f.x * l.x - f.y * l.y, f.y * l.x + f.x * l.y, l.z);
}

fn cc_aircars_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// Where craft `lane` of cell `ci` is, if it exists at all.
//   lane 0 — the x-avenue lane (runs north/south)
//   lane 1 — the y-avenue lane (runs east/west)
//   lane 2 — the free flyer
// Deterministic in (ci, cc, lane) alone, so the shader re-derives a craft from
// its hit kind and the core's own CityCell — nothing has to be smuggled
// through CityHit.
fn cc_aircars_craft(ci: vec2<i32>, cc: CityCell, lane: i32)
        -> cc_aircars_Craft {
    let cell = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cell;

    if (lane == 2) {
        // Free flyers only over the built city; open lots keep empty sky.
        if (!cc.built) {
            return cc_aircars_no_craft();
        }
        let r = cc_aircars_draw(ci, lane);
        if (r.x >= cc_aircars_FREE_FRAC) {
            return cc_aircars_no_craft();
        }
        // r.x is uniform on [0, FREE_FRAC): rescaled it is a free draw, so a
        // heading costs no extra hash.
        let q = r.x / cc_aircars_FREE_FRAC;
        let span = cell - 2.0 * cc_aircars_FREE_MARGIN;
        return cc_aircars_Craft(
            true,
            vec3<f32>(cmin.x + cc_aircars_FREE_MARGIN + r.y * span,
                      cmin.y + cc_aircars_FREE_MARGIN + r.z * span,
                      mix(cc_aircars_FREE_Z_LO, cc_aircars_FREE_Z_HI, r.w)),
            select(0, 1, q > 0.5),
            select(-1.0, 1.0, fract(q * 2.0) > 0.5),
            (fract(r.y * 53.0) - 0.5) * 2.0 * cc_aircars_YAW);
    }

    // Avenue lanes. Exactly one of the two edge tests can fire — the avenue
    // period is 8, so consecutive indices are never both avenues — which is
    // why one lane slot per axis per cell is the whole story.
    let k = select(ci.y, ci.x, lane == 0);
    let lo_edge = city_is_avenue(k);
    if (!lo_edge && !city_is_avenue(k + 1)) {
        return cc_aircars_no_craft();
    }
    let r = cc_aircars_draw(ci, lane);
    // Traffic clusters where the city does.
    let p = cc_aircars_LANE_P * mix(cc_aircars_DENS_LO, cc_aircars_DENS_HI,
                                    smoothstep(cc_aircars_DENS_START,
                                               cc_aircars_DENS_FULL,
                                               cc.density));
    if (r.x >= p) {
        return cc_aircars_no_craft();
    }

    // Lateral seat: LANE_OFF in from whichever edge is the avenue, jittered.
    let jit = (r.z - 0.5) * 2.0 * cc_aircars_LANE_JIT;
    let lat = select(cell - cc_aircars_LANE_OFF + jit,
                     cc_aircars_LANE_OFF + jit, lo_edge);
    let along = 0.5 * cell + (r.y - 0.5) * 2.0 * cc_aircars_ALONG_JIT;
    let z = select(cc_aircars_Z_LOW, cc_aircars_Z_HIGH,
                   r.w > cc_aircars_HIGH_CUT)
            + (fract(r.w * 37.0) - 0.5) * 2.0 * cc_aircars_Z_JIT;

    // Yaw is a free re-draw off r.z, the way the deck jitter is off r.w: no
    // craft flies exactly parallel to its lane, and five degrees is the
    // difference between a formation and traffic.
    let yaw = (fract(r.z * 61.0) - 0.5) * 2.0 * cc_aircars_YAW;

    if (lane == 0) {
        // Runs along y. On the avenue's +x verge (this cell's min edge is the
        // avenue) right-hand traffic heads north.
        return cc_aircars_Craft(
            true, vec3<f32>(cmin.x + lat, cmin.y + along, z),
            1, select(-1.0, 1.0, lo_edge), yaw);
    }
    // Runs along x; the same rule rotated.
    return cc_aircars_Craft(
        true, vec3<f32>(cmin.x + along, cmin.y + lat, z),
        0, select(1.0, -1.0, lo_edge), yaw);
}

// Half-extents in xy for a body of the given along/across half-sizes.
fn cc_aircars_extent(axis: i32, hl: f32, hw: f32) -> vec2<f32> {
    return select(vec2<f32>(hw, hl), vec2<f32>(hl, hw), axis == 0);
}

// The close read: bounding box, then sphere-trace the vehicle inside it.
// Called only when the box is worth more than a few pixels; a miss inside the
// box is a miss, which is what carves the curved silhouette out of the slab.
fn cc_aircars_sdf_hit(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t0: f32, t1: f32, v: cc_aircars_Craft, lane: i32,
                      ci: vec2<i32>, fp: f32) -> CityHit {
    let hx = cc_aircars_extent(v.axis, cc_aircars_SDF_PAD_L,
                               cc_aircars_SDF_PAD_W);
    let bmin = vec3<f32>(v.base.xy - hx,
                         v.base.z - cc_aircars_SDF_PAD_ZLO);
    let bmax = vec3<f32>(v.base.xy + hx,
                         v.base.z + cc_aircars_SDF_PAD_ZHI);
    let sb = city_box_hit(o, inv_dir, bmin, bmax);
    let ta = max(max(sb.x, t0), 0.0);
    let tb = min(sb.y, t1);
    if (sb.x > sb.y || ta > tb) {
        return cc_aircars_miss(ci);
    }

    // Ray into the craft's frame once, rather than the sample point every
    // step: the transform is a rotation about z, so |rd| = |dir| and t is
    // still world distance.
    let f = cc_aircars_frame(v);
    let ro = cc_aircars_to_local(f, o - v.base);
    let rd = cc_aircars_to_local(f, dir);
    // Surface tolerance follows the footprint: converge to well under a pixel
    // and no further. The floor keeps the whip aerial from costing the whole
    // iteration budget at arm's length.
    let eps = clamp(0.30 * fp, 0.0035, 0.06);

    var cc_aircars_t = ta;
    var cc_aircars_found = false;
    for (var cc_aircars_i: i32 = 0; cc_aircars_i < cc_aircars_SDF_ITERS;
         cc_aircars_i = cc_aircars_i + 1) {
        let d = cc_aircars_sdf(ro + cc_aircars_t * rd).x;
        if (d < eps) {
            cc_aircars_found = true;
            break;
        }
        cc_aircars_t = cc_aircars_t + d;
        if (cc_aircars_t > tb) {
            break;
        }
    }
    if (!cc_aircars_found || cc_aircars_t > tb) {
        return cc_aircars_miss(ci);
    }

    // Normal by the four-tap tetrahedron gradient, at the tolerance's own
    // scale so a coarse trace gets a correspondingly smoothed normal.
    let q = ro + cc_aircars_t * rd;
    let hs = max(eps, 0.004);
    // The four tetrahedron taps run as a LOOP, not as four unrolled calls,
    // and that is a performance decision, not a style one. Written out, the
    // four calls inline four more copies of the whole vehicle field into the
    // one kernel that also marches the clouds; the register budget is per
    // kernel, so occupancy collapses for EVERY pixel in the frame whether or
    // not a craft is anywhere near it. Measured on the RTX 5080: the unrolled
    // taps cost +67% on the aerial view, where the nearest craft is 7 km away
    // and the gate can never fire. Rolled, one copy serves all four.
    var cc_aircars_g = vec3<f32>(0.0);
    for (var cc_aircars_k: u32 = 0u; cc_aircars_k < 4u;
         cc_aircars_k = cc_aircars_k + 1u) {
        // The tetrahedron's four sign triples, as bit rows: x on taps 0,3;
        // y on 1,3; z on 2,3.
        let cc_aircars_e = 2.0 * vec3<f32>(
            f32((0x9u >> cc_aircars_k) & 1u),
            f32((0xau >> cc_aircars_k) & 1u),
            f32((0xcu >> cc_aircars_k) & 1u)) - 1.0;
        cc_aircars_g = cc_aircars_g + cc_aircars_e
                       * cc_aircars_sdf(q + cc_aircars_e * hs).x;
    }
    let g = cc_aircars_g;

    return CityHit(true, cc_aircars_t, o + cc_aircars_t * dir,
                   cc_aircars_to_world(f, normalize(g)),
                   300 + 4 * lane + 3, ci);
}

// Nearest hit of one craft against [t0, t1]. The kind encodes both the lane
// (so the shader can find the craft again) and the part that was hit:
// 300 + 4 * lane + part, part 0 = hull side/top, 1 = the underglow panel on
// the belly, 2 = the canopy, 3 = an SDF hit (the close read, whose parts the
// shader recovers from the local position and the field itself).
fn cc_aircars_hit_craft(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                        t0: f32, t1: f32, v: cc_aircars_Craft, lane: i32,
                        ci: vec2<i32>, fp: f32) -> CityHit {
    if (fp < cc_aircars_SDF_FP) {
        return cc_aircars_sdf_hit(o, dir, inv_dir, t0, t1, v, lane, ci, fp);
    }
    let hx = cc_aircars_extent(v.axis, cc_aircars_HALF_L, cc_aircars_HALF_W);
    let hmin = vec3<f32>(v.base.xy - hx, v.base.z);
    let hmax = vec3<f32>(v.base.xy + hx, v.base.z + cc_aircars_HULL_H);
    let sh = city_box_hit(o, inv_dir, hmin, hmax);
    let t_hull = select(1e30, sh.x,
                        sh.x <= sh.y && sh.x > t0 && sh.x <= t1);

    let aft = select(vec2<f32>(0.0, -v.fwd * cc_aircars_CANOPY_AFT),
                     vec2<f32>(-v.fwd * cc_aircars_CANOPY_AFT, 0.0),
                     v.axis == 0);
    let cx = cc_aircars_extent(v.axis, cc_aircars_CANOPY_HALF_L,
                               cc_aircars_CANOPY_HALF_W);
    let kmin = vec3<f32>(v.base.xy + aft - cx, v.base.z + cc_aircars_HULL_H);
    let kmax = vec3<f32>(v.base.xy + aft + cx,
                         v.base.z + cc_aircars_HULL_H + cc_aircars_CANOPY_H);
    // The composer's namespace check is line-based, so even this local wears
    // the prefix.
    var cc_aircars_t_canopy = 1e30;
    if (fp < cc_aircars_CANOPY_FP) {
        let sk = city_box_hit(o, inv_dir, kmin, kmax);
        if (sk.x <= sk.y && sk.x > t0 && sk.x <= t1) {
            cc_aircars_t_canopy = sk.x;
        }
    }

    let t = min(t_hull, cc_aircars_t_canopy);
    if (t >= 1e30) {
        return cc_aircars_miss(ci);
    }
    let is_canopy = cc_aircars_t_canopy < t_hull;
    let bmin = select(hmin, kmin, is_canopy);
    let bmax = select(hmax, kmax, is_canopy);
    let pos = o + t * dir;
    let nrm = city_box_normal(pos, bmin, bmax);
    // The belly of the hull is the underglow panel.
    let part = select(select(0, 1, nrm.z < -0.5), 2, is_canopy);
    return CityHit(true, t, pos, nrm, 300 + 4 * lane + part, ci);
}

fn cc_aircars_nearer(a: CityHit, b: CityHit) -> CityHit {
    if (b.hit && (!a.hit || b.t < a.t)) {
        return b;
    }
    return a;
}

// One lane's contribution: place, then test. Both halves early-out cheaply,
// so an absent lane costs an integer compare and an absent craft one hash.
fn cc_aircars_lane_hit(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell,
                       lane: i32, fp: f32) -> CityHit {
    let v = cc_aircars_craft(ci, cc, lane);
    if (!v.ok) {
        return cc_aircars_miss(ci);
    }
    return cc_aircars_hit_craft(o, dir, inv_dir, t0, t1, v, lane, ci, fp);
}

fn cc_aircars_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                          t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    // Gate 1, no hash: does this segment pass through the air layer at all?
    // Most city pixels are looking at a facade or at the road.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_aircars_Z_MAX || max(za, zb) < cc_aircars_Z_MIN) {
        return cc_aircars_miss(ci);
    }
    // Gate 2: a craft smaller than this is not worth a box test.
    let fp = (2.0 * u.cam_origin.w / max(u.params.x, 1.0)) * max(t0, 0.0);
    if (fp > cc_aircars_FAR_FP) {
        return cc_aircars_miss(ci);
    }
    let h0 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 0, fp);
    let h1 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 1, fp);
    let h2 = cc_aircars_lane_hit(o, dir, inv_dir, t0, t1, ci, cc, 2, fp);
    return cc_aircars_nearer(cc_aircars_nearer(h0, h1), h2);
}

// One nav lamp. `a` and `b` are the hit's coordinates in the face's own plane
// relative to the craft, `seat_*` the lamp's seat in the same frame, and
// `span_*` the face's size. Resolved while the dot is bigger than a pixel,
// handed to the face's mean when it is not: dot area over face area is the
// honest sub-pixel value, so the craft keeps its light and loses only the
// resolution.
fn cc_aircars_lamp(a: f32, b: f32, seat_a: f32, seat_b: f32,
                   span_a: f32, span_b: f32, col: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let d = length(vec2<f32>(a - seat_a, b - seat_b));
    let sharp = select(0.0, 1.0, d < cc_aircars_LAMP_R);
    let area = 3.14159265 * cc_aircars_LAMP_R * cc_aircars_LAMP_R;
    let mean = area / max(span_a * span_b, 1e-3);
    let k = smoothstep(cc_aircars_LAMP_LOD_START, cc_aircars_LAMP_LOD_FULL,
                       fp);
    return col * (cc_aircars_LAMP_RAD * mix(sharp, mean, k));
}

// White forward, red aft, on whichever face is looking at you. A craft
// crossing your view still declares which way it is going, because the flank
// carries the same pair near its ends.
fn cc_aircars_navlights(h: CityHit, v: cc_aircars_Craft, fp: f32)
        -> vec3<f32> {
    let rel = h.pos - v.base;
    let a = select(rel.y * v.fwd, rel.x * v.fwd, v.axis == 0);
    let across = select(rel.x, rel.y, v.axis == 0);
    let b = rel.z;
    let seat_b = 0.60 * cc_aircars_HULL_H;
    let white = vec3<f32>(1.00, 0.96, 0.88);
    let red = vec3<f32>(1.00, 0.13, 0.06);
    let n_along = select(h.normal.y * v.fwd, h.normal.x * v.fwd, v.axis == 0);
    let n_across = select(h.normal.x, h.normal.y, v.axis == 0);
    let face_w = 2.0 * cc_aircars_HALF_W;
    let face_l = 2.0 * cc_aircars_HALF_L;

    if (n_along > 0.5) {
        return cc_aircars_lamp(across, b, -0.52, seat_b, face_w,
                               cc_aircars_HULL_H, white, fp)
             + cc_aircars_lamp(across, b, 0.52, seat_b, face_w,
                               cc_aircars_HULL_H, white, fp);
    }
    if (n_along < -0.5) {
        return cc_aircars_lamp(across, b, -0.52, seat_b, face_w,
                               cc_aircars_HULL_H, red, fp)
             + cc_aircars_lamp(across, b, 0.52, seat_b, face_w,
                               cc_aircars_HULL_H, red, fp);
    }
    if (abs(n_across) > 0.5) {
        return cc_aircars_lamp(a, b, 1.55, seat_b, face_l,
                               cc_aircars_HULL_H, white, fp)
             + cc_aircars_lamp(a, b, -1.55, seat_b, face_l,
                               cc_aircars_HULL_H, red, fp);
    }
    return vec3<f32>(0.0);
}

// The close read's shading. Everything here is a function of the SDF normal
// or of the hit's position in the craft's own frame, which is the only reason
// a rounded box full of smooth-mins reads as a machine: the eye recovers the
// surface from how the light falls off across it, not from the outline.
fn cc_aircars_shade_sdf(h: CityHit, v: cc_aircars_Craft, glow: vec3<f32>,
                        bright: f32, fade: f32, fill: vec3<f32>,
                        dir: vec3<f32>, fp: f32, a4: vec4<f32>) -> vec3<f32> {
    let f = cc_aircars_frame(v);
    let q = cc_aircars_to_local(f, h.pos - v.base);
    let n = cc_aircars_to_local(f, h.normal);
    let parts = cc_aircars_sdf(q);
    let ya = abs(q.y);
    let moon = u.sun_dir.xyz;
    let refl = reflect(dir, h.normal);
    let mspec = max(dot(refl, moon), 0.0);

    // Which part of the solid the ray landed on. Soft, because the smooth-min
    // means the surface near a join genuinely belongs to both.
    let w_canopy = 1.0 - smoothstep(-0.02, 0.12,
                                    parts.z - min(parts.y, parts.w));
    // The seam is the crease itself: where the canopy's field and the hull's
    // are equal, the join is a physical joint and gets a dark band. Found,
    // not drawn — it follows the bubble wherever the smooth-min puts it.
    let seam = 1.0 - smoothstep(0.0, 0.11, abs(parts.z - parts.y));

    // --- dark-metal body ---------------------------------------------------
    // This is what has to carry the solidity. Emission alone made a candy
    // lozenge of the first cut: a craft has to read as a dark machine with
    // lights ON it, so the hull gets its own small directional budget and the
    // glow is confined to the panels that are actually panels.
    // A hemisphere: cool skyglow above, sodium city below, blended by n.z.
    // This one gradient does more for solidity than anything else here — it
    // is what every curved surface under a lit street actually looks like,
    // warm underneath and cold on top, and the eye integrates the gradient
    // into a shape without being told.
    let gl = city_glow_sample(h.pos.xy, 3.0);
    // The weights are squared, not linear: a linear hemisphere paints every
    // near-vertical surface the same mid-tone and the gradient stops being a
    // gradient. Squaring keeps the flanks dark and puts the light where the
    // surface actually turns.
    let dn = 0.5 - 0.5 * n.z;
    let sky_amb = fill * ((1.0 - dn) * (1.0 - dn));
    let gnd_amb = CITY_UPLIGHT_COLOR * (cc_aircars_UPGLOW * (0.15 + 2.2 * gl)
                                        * dn * dn);
    var body = cc_aircars_HULL_FILL * sky_amb + gnd_amb;
    // Moonlight on dark metal. Physically this is nearly nothing, and it is
    // kept nearly nothing — but the sign of dot(n, moon) is the only cue that
    // says which way a surface faces, and a hull without it is a cutout.
    body = body + cc_aircars_HULL_DIFF * CITY_MOONLIGHT
                  * max(dot(h.normal, moon), 0.0);
    // The craft's own panel spilling back onto its lower flanks. Small, and
    // the only reason it is here is that a light source hanging under a hull
    // that leaves no trace on the hull looks like a decal.
    body = body + cc_aircars_BOUNCE * mix(glow, vec3<f32>(1.0), 0.45)
                  * (bright * dn);
    // Moon sheen: narrow, so it travels across the crown as the craft turns.
    body = body + cc_aircars_SHEEN * vec3<f32>(0.70, 0.78, 1.00)
                  * pow(mspec, 26.0);
    // Rim: the lit edge that says "curved". Tinted by the craft's own glow,
    // because the belly panel is the brightest thing anywhere near the hull.
    let rim = pow(1.0 - min(abs(dot(h.normal, dir)), 1.0), 3.5);
    body = body + cc_aircars_RIM * rim
                  * (0.06 * glow * bright + vec3<f32>(0.010, 0.011, 0.016));

    // --- underglow, mapped onto the curved lower surface -------------------
    // Two pieces, and the difference between them is the whole look: a broad
    // PANEL on the surface that actually faces the ground, and a narrow STRIP
    // along the chine that is all anyone sees from the side. The box version
    // lit the entire flank because a box has no chine to stop at.
    // Gating on the normal ALONE was the second wrong answer: a rounded nose
    // cap points up over its top half and down over its bottom half, so
    // "faces down" lights the entire nose and the entire tail and the craft
    // goes back to being made of light. Every panel is therefore a normal
    // test AND a footprint in the local frame — which is what a panel is.
    // LOD, and it is the hand-off's whole design. Close up these are BARS —
    // a spine down the deck, a strip along the chine, a panel inset in the
    // belly — because that is what fitted lighting looks like. As the
    // footprint grows toward SDF_FP they widen until they cover the faces
    // they sit on, which is exactly what the box path beyond the gate draws.
    // So the two constructions meet with the same mean radiance per face and
    // the craft does not change brightness when the geometry swaps.
    let k = smoothstep(0.06, cc_aircars_SDF_FP, fp);
    let along = 1.0 - smoothstep(mix(0.95, 1.55, k), mix(1.32, 1.90, k),
                                 abs(q.x));
    let panel = along * (1.0 - smoothstep(mix(0.56, 0.86, k),
                                          mix(0.82, 1.00, k), ya));
    let belly = smoothstep(0.30, 0.85, -n.z) * mix(mix(0.02, 0.30, k), 1.0,
                                                   panel);
    // A BAND centred on the chine, not everything below it: a light bar with
    // dark hull above and below is what makes the panel read as fitted rather
    // than as the craft being made of light.
    // The third gate is on the normal being FLANK-like. Without it the band
    // in z sweeps across the nose's rounded underside — which is tangent to
    // that height over a large area — and the craft grows a glowing snout.
    let strip = (1.0 - smoothstep(cc_aircars_CHINE_LO, cc_aircars_CHINE_HI,
                                  abs(q.z - cc_aircars_CHINE_Z)))
                * (1.0 - smoothstep(0.42, 0.78, abs(n.z)))
                * along * (1.0 - belly);
    // The dorsal light is a spine bar down the middle of the deck, not the
    // whole upper surface. From above it plus the canopy crown is the read.
    let spine = 1.0 - smoothstep(mix(0.14, 0.90, k), mix(0.30, 1.10, k), ya);
    let dorsal = smoothstep(mix(0.62, 0.15, k), mix(0.95, 0.55, k), n.z)
                 * spine
                 * (1.0 - smoothstep(mix(1.05, 1.60, k),
                                     mix(1.45, 1.95, k), abs(q.x)))
                 * cc_aircars_DORSAL;
    let amt = belly + strip * cc_aircars_BAR + dorsal;
    var em = glow * (cc_aircars_GLOW_RAD * bright * amt);

    // --- cabin --------------------------------------------------------------
    // Dark glass with a dim warm interior; the dorsal running light rides
    // only the crown, so from the side the bubble is a window and from above
    // it is still the lit chip the aerial read needs.
    // The interior is visible through the SIDE glass and not through the
    // roof, which is both true and the thing that separates a canopy from a
    // painted panel: a bubble lit evenly all over is a lampshade.
    let cab = vec3<f32>(1.00, 0.74, 0.44)
              * (cc_aircars_CANOPY_RAD * cc_aircars_CABIN
                 * (0.6 + 0.8 * a4.z) * (1.0 - smoothstep(0.15, 0.72, n.z)));
    let canopy_em = cab
                  + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL * bright
                            * smoothstep(0.68, 0.97, n.z)
                            * (1.0 - smoothstep(mix(0.10, 0.62, k), mix(0.26, 0.80, k), ya)))
                  + cc_aircars_GLASS * vec3<f32>(0.55, 0.64, 0.90)
                    * pow(mspec, 110.0);
    em = mix(em, canopy_em, w_canopy);
    // Glass takes far less ambient than painted metal, which is most of what
    // tells a canopy from a roof panel at night.
    body = body * mix(1.0, 0.30, w_canopy);
    em = em * (1.0 - cc_aircars_SEAM_DARK * seam);
    body = body * (1.0 - 0.85 * seam);

    // --- thruster discs -----------------------------------------------------
    // The aft caps of the nacelles, seen only from behind the craft.
    let dr = length(vec2<f32>(ya - cc_aircars_NAC_Y, q.z - cc_aircars_NAC_Z));
    let aft = 1.0 - smoothstep(-1.58, -1.36, q.x);
    let back = smoothstep(0.30, 0.75, -n.x);
    let core = 1.0 - smoothstep(0.55 * cc_aircars_THRUST_R,
                                cc_aircars_THRUST_R, dr);
    let cup = 1.0 - smoothstep(cc_aircars_THRUST_R,
                               cc_aircars_THRUST_R + 0.07, dr);
    em = em + mix(vec3<f32>(1.00, 0.93, 0.84), glow, 0.72)
              * (cc_aircars_THRUST * bright * aft * back
                 * (core + cc_aircars_THRUST_HALO * (cup - core)));

    // --- intake slot --------------------------------------------------------
    // A dark louvred cut in each nose flank. Shading, per SPEC: at this scale
    // geometry would buy nothing the normal does not already give.
    let intake = max(
        (1.0 - smoothstep(0.0, 0.050, abs(q.z - 0.33)))
            * (1.0 - smoothstep(0.24, 0.36, abs(q.x - 0.92)))
            * smoothstep(0.25, 0.60, abs(n.y)),
        // The grille across the nose. A vehicle needs one end that is
        // obviously the front, and at this scale a dark slot facing forward
        // does it more cheaply than any amount of modelled taper.
        (1.0 - smoothstep(0.0, 0.055, abs(q.z - 0.25)))
            * (1.0 - smoothstep(0.26, 0.42, ya))
            * smoothstep(0.40, 0.85, n.x));
    // --- panel lines --------------------------------------------------------
    // Two, both at joints a fabricator would actually have: where the nose
    // section meets the body, and where the tail transom starts.
    // Two transverse joints — nose section to body, body to tail transom —
    // and one longitudinal shoulder crease along the flank. The crease is the
    // single most car-like line on the thing: bodies have a waistline.
    let lines = max(max(1.0 - smoothstep(0.0, 0.045, abs(q.x - 0.52)),
                        1.0 - smoothstep(0.0, 0.045, abs(q.x + 0.98))),
                    (1.0 - smoothstep(0.0, 0.032, abs(q.z - 0.46)))
                        * (1.0 - smoothstep(0.35, 0.62, abs(n.z)))
                        * along)
                * (1.0 - w_canopy);
    let dark = 1.0 - max(cc_aircars_INTAKE_DARK * intake,
                         cc_aircars_LINE_DARK * lines);
    em = em * dark;
    body = body * dark;

    // --- nav lamps in their housings ---------------------------------------
    // White forward, red aft, each a lens on the outboard face of its own
    // raised housing. The lens grows and dims with the footprint so the pair
    // keeps its flux as the housing goes sub-pixel — the craft loses the
    // resolution, not the light.
    // The lens is a cap on the OUTBOARD face of the housing bump, found by
    // where the point sits on the bump rather than by distance to its centre:
    // distance alone lights the whole hemisphere, which is how the first cut
    // grew two headlights the size of the cabin.
    let vf = vec3<f32>(q.x - 0.38, ya - 0.93, q.z - cc_aircars_HOUSE_Z);
    let va = vec3<f32>(q.x + 0.52, ya - 0.93, q.z - cc_aircars_HOUSE_Z);
    let df = length(vf);
    let da = length(va);
    // As the footprint grows the housing goes sub-pixel; open the cap and
    // drop the radiance to hold its flux, so the pair fades rather than
    // flickers.
    let wide = clamp(2.4 * fp, 0.0, 0.55);
    let cut = cc_aircars_LENS_OUT - wide;
    let gain = clamp((1.0 - cc_aircars_LENS_OUT) / (1.0 - cut), 0.20, 1.0);
    let lf = (1.0 - smoothstep(cc_aircars_HOUSE_R, 0.165, df))
             * smoothstep(cut, cut + 0.28, vf.y / max(df, 1e-3));
    let la = (1.0 - smoothstep(cc_aircars_HOUSE_R, 0.165, da))
             * smoothstep(cut, cut + 0.28, va.y / max(da, 1e-3));
    em = em + (vec3<f32>(1.00, 0.96, 0.88) * lf
               + vec3<f32>(1.00, 0.13, 0.06) * la)
              * (cc_aircars_LAMP_RAD * gain);

    return body + em * fade;
}

fn cc_aircars_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let code = h.kind - 300;
    let lane = code / 4;
    let part = code % 4;
    let v = cc_aircars_craft(h.cell, cc, lane);
    let a4 = cc_aircars_look(h.cell, lane);
    let glow = cc_aircars_glow_color(a4.x);
    let bright = 0.70 + 0.70 * a4.y;

    // The two edges of the layer, each approached rather than stepped: the
    // population edge at CITY_PROP_RANGE, and the footprint at which the
    // tracer stops testing craft at all.
    let fade = (1.0 - smoothstep(cc_aircars_FADE_START * CITY_PROP_RANGE,
                                 cc_aircars_FADE_FULL * CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_aircars_FAR_FADE, cc_aircars_FAR_FP, fp));

    // The night fill the core gives every city surface. A hull is dark metal
    // over a dark city: essentially a silhouette with a lit edge.
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(h.pos.xy, 3.0));
    let body = cc_aircars_HULL_FILL * fill;

    if (part == 3) {   // the close read: shade against the SDF's own normal
        return cc_aircars_shade_sdf(h, v, glow, bright, fade, fill, dir, fp,
                                    a4);
    }
    if (part == 1) {   // the underglow panel: the whole point of the layer
        let rel = h.pos - v.base;
        let a = select(rel.y, rel.x, v.axis == 0);
        let across = select(rel.x, rel.y, v.axis == 0);
        let panel =
            (1.0 - smoothstep(cc_aircars_PANEL_A, cc_aircars_HALF_L, abs(a)))
          * (1.0 - smoothstep(cc_aircars_PANEL_W, cc_aircars_HALF_W,
                              abs(across)));
        return body + glow * (cc_aircars_GLOW_RAD * bright * fade
                              * mix(cc_aircars_PANEL_RIM, 1.0, panel));
    }
    if (part == 2) {   // canopy: a warm interior seen through the glass
        // Its BACK is dorsal like the hull's, and that matters more than it
        // sounds: the canopy covers the middle of the top, so from directly
        // above it is most of what a craft shows. Lit warm, the whole aerial
        // read of the layer went with it. This also makes the LOD hand-off
        // exact — when the canopy box drops out the hull top behind it is
        // already the same radiance.
        if (h.normal.z > 0.5) {
            return body + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL
                                  * bright * fade);
        }
        return body + vec3<f32>(1.00, 0.74, 0.44)
                      * (cc_aircars_CANOPY_RAD * (0.6 + 0.8 * a4.z) * fade);
    }
    // Hull. The back carries a dorsal running light; the flanks carry the
    // skirt of the belly panel, e-folding upward from the seam. Nav lamps are
    // drawn on rather than built.
    if (h.normal.z > 0.5) {
        return body + glow * (cc_aircars_GLOW_RAD * cc_aircars_DORSAL
                              * bright * fade);
    }
    let up = max(h.pos.z - v.base.z, 0.0);
    let wrap = 1.0 - smoothstep(cc_aircars_SKIRT_LO, cc_aircars_SKIRT_HI, up);
    let skirt = glow * (cc_aircars_GLOW_RAD * cc_aircars_SKIRT * bright
                        * wrap);
    return body + (skirt + cc_aircars_navlights(h, v, fp)) * fade;
}

// --- component: facadeworks (facadeworks.wgsl) ---
// facadeworks — additive facade detail on top of the core window ladder.
//
// Three layers, in the order they matter to the shot:
//
//   1. VERTICAL NEON SIGNAGE. The cyberpunk cue. A hashed minority of
//      buildings (~12%) carries one narrow vertical sign box mounted on a
//      corner of the main mass, so both walls meeting there show it:
//      saturated amber / cyan / magenta at radiance 5-8, broken into
//      glyph-like blocks that suggest characters without being any script.
//      A mid-rise gets a 12-40 m strip starting 6 m up; a tower gets a
//      ribbon proportional to its own height. This is the layer the whole
//      component exists for — at street range it should read as typography,
//      and from kilometres out it should still be a colored vertical line,
//      which is what makes a skyline read as a CITY and not a lit grid.
//   2. BALCONY BANDS. ~25% of buildings get balcony ledges every 2nd or 3rd
//      storey. We are ADDITIVE, so the ledge itself cannot be drawn (it is a
//      dark band, and there is no multiply here); what we CAN draw is the
//      light that spills past a balcony door, so the ledge reads by its warm
//      underglow line rather than by its shadow. Deliberately faint:
//      balconies are rhythm, not decoration.
//   3. FIRE-ESCAPE ZIGZAGS. Filigree: a stair lattice rim-lit by street
//      light on one wall of ~15% of mid-rises. Sub-pixel almost everywhere,
//      so it is the layer with the most aggressive LOD.
//
// LOD discipline (SPEC rule 3). Every layer states its own mean and blends
// to it as the pixel footprint grows; nothing switches off. Two subtleties
// this file cares about:
//
//   * The sign strip is 1.2 m wide, so past ~2.5 m/px it is thinner than a
//     pixel and a naive gate would make it strobe. Instead the cross-section
//     WIDENS to the pixel while its amplitude drops by the same factor, so
//     the integrated energy across the strip is invariant. That is why a
//     sign is still a clean colored line at 3 km instead of a sparkle.
//   * The core hands us `fp`, not its own `fp_eff` (which divides by the
//     view/normal cosine); we have no `dir` here. So our thresholds are on
//     the un-foreshortened footprint and are set a little conservatively.
//
// The composer's namespace rule covers function-local `var` too, so this
// file is written without mutable locals — which is no loss: every layer is
// a mask times a color.

// --- Vertical neon signage -------------------------------------------------
const cc_facadeworks_SIGN_FRAC: u32 = 31u;      // of 256 buildings (~12%)
const cc_facadeworks_SIGN_HW: f32 = 0.60;      // half-width of the strip (m)
const cc_facadeworks_SIGN_V0: f32 = 6.0;       // strip starts this high (m)
const cc_facadeworks_SEG_PITCH: f32 = 1.15;    // glyph block pitch (m)
const cc_facadeworks_SEG_H: f32 = 0.90;        // lit height within the pitch
const cc_facadeworks_SEG_DUTY: f32 = 0.7826;   // SEG_H / SEG_PITCH
// Mean ink coverage of the 5-stroke glyph alphabet below, each stroke drawn
// with probability 1/2. Integrated over the alphabet rather than guessed:
// this is the value a block collapses to once its strokes go sub-pixel.
const cc_facadeworks_GLYPH_MEAN: f32 = 0.312;
// Both of these are calibrated against exposure 6, where anything over
// ~0.05 radiance is already a clearly visible grey: the first version of
// this halo ran at 0.33 and rendered the sign as a solid tan panel.
const cc_facadeworks_SIGN_HALO_AMP: f32 = 0.014; // wall spill, x the strip mean
const cc_facadeworks_SIGN_HALO_SIG: f32 = 0.85;  // spill sigma (m)
// A dim backing panel behind the glyphs, so a sign block reads as an
// illuminated box with writing on it rather than as free-floating strokes.
const cc_facadeworks_SIGN_PANEL: f32 = 0.005;

// --- Balcony bands ---------------------------------------------------------
const cc_facadeworks_BAL_FRAC: u32 = 64u;      // of 256 buildings (25%)
const cc_facadeworks_BAL_GLOW_V0: f32 = 0.55;  // door spill starts above the
const cc_facadeworks_BAL_GLOW_H: f32 = 0.45;   // 0.5 m ledge, and is this tall
// The brief's 0.6 rendered as a bright hairline rule ruled clean across a
// whole wall — the single most prominent thing on the facade, and the exact
// "stripe" the brief warns against. At exposure 6 a spill that reads as
// spill, not as a drawn line, wants roughly a tenth of that.
const cc_facadeworks_BAL_RADIANCE: f32 = 0.09;
const cc_facadeworks_BAL_COLOR: vec3<f32> = vec3<f32>(1.0, 0.52, 0.22);
const cc_facadeworks_BAL_FLOOR_P: f32 = 0.60;  // storeys whose doors are open
const cc_facadeworks_BAL_UNIT_P: f32 = 0.45;   // doors within such a storey
// One door per window pitch, lighting the middle two thirds of it: what
// breaks the line into a dashed rhythm instead of a rule.
const cc_facadeworks_BAL_DOOR_LO: f32 = 0.18;
const cc_facadeworks_BAL_DOOR_HI: f32 = 0.82;
const cc_facadeworks_BAL_DOOR_FRAC: f32 = 0.64;  // HI - LO
// Mean of the soft vertical profile below over the GLOW_H window.
const cc_facadeworks_BAL_PROFILE_MEAN: f32 = 0.625;
const cc_facadeworks_BAL_MAX_H: f32 = 260.0;   // towers have sealed walls

// --- Fire escapes ----------------------------------------------------------
const cc_facadeworks_FE_FRAC: u32 = 38u;        // of 256 buildings (~15%)
const cc_facadeworks_FE_HW: f32 = 1.50;        // half-width of the column (m)
// 0.30 was the brief's number and it renders as a solid tan scaffold at
// exposure 6 — brighter than most of the windows behind it. This is the
// rim a fire escape catches off a street lamp, so it is set where it reads
// as structure on black and nothing more, and it falls off with height
// because the lamp that lights it is on the ground.
const cc_facadeworks_FE_RADIANCE: f32 = 0.055;
const cc_facadeworks_FE_FALL_H: f32 = 55.0;    // e-fold of the street lamp
const cc_facadeworks_FE_COLOR: vec3<f32> = vec3<f32>(1.0, 0.55, 0.28);
const cc_facadeworks_FE_MEAN: f32 = 0.240;     // ink coverage of the lattice

// Which tier of the building this facade pixel belongs to, and how far the
// wall runs in facade-u. The u-extent is what lets signage and fire escapes
// sit a fixed distance from a CORNER rather than at a hashed offset that
// would sometimes land mid-wall and sometimes off the end of it.
struct cc_facadeworks_Face {
    u0: f32,
    u1: f32,
    z1: f32,      // top of this tier
    base: bool,   // this is tier 1, the main mass
    side: i32,    // 0 = +x wall, 1 = -x, 2 = +y, 3 = -y
}

fn cc_facadeworks_face(cc: CityCell, h: CityHit) -> cc_facadeworks_Face {
    let t2 = cc.tiers >= 2 && h.pos.z > cc.b2min.z;
    let t3 = cc.tiers >= 3 && h.pos.z > cc.b3min.z;
    let bmn = select(select(cc.b1min, cc.b2min, t2), cc.b3min, t3);
    let bmx = select(select(cc.b1max, cc.b2max, t2), cc.b3max, t3);
    // The core's facade tangent, so our uc is exactly the core's uc.
    let tangent = vec2<f32>(-h.normal.y, h.normal.x);
    let a = dot(bmn.xy, tangent);
    let b = dot(bmx.xy, tangent);
    let side = select(
        select(select(3, 2, h.normal.y > 0.5), 1, h.normal.x < -0.5),
        0, h.normal.x > 0.5);
    return cc_facadeworks_Face(min(a, b), max(a, b), bmx.z, !(t2 || t3), side);
}

// One glyph block: up to five strokes from an alphabet of three horizontals
// and two verticals, each present with probability 1/2. Not a script — the
// point is that a stack of them reads as writing at a glance and as nothing
// in particular on inspection, which is what neon signage in a language you
// do not speak looks like.
fn cc_facadeworks_glyph(bits: u32, gu: f32, gv: f32) -> f32 {
    let inl = gu > 0.12 && gu < 0.88;   // horizontal strokes stop short
    let inb = gv > 0.10 && gv < 0.90;   // vertical strokes stop short
    let on = (inl && (bits & 1u) != 0u && abs(gv - 0.86) < 0.10)
          || (inl && (bits & 2u) != 0u && abs(gv - 0.50) < 0.10)
          || (inl && (bits & 4u) != 0u && abs(gv - 0.14) < 0.10)
          || (inb && (bits & 8u) != 0u && abs(gu - 0.20) < 0.075)
          || (inb && (bits & 16u) != 0u && abs(gu - 0.80) < 0.075);
    return select(0.0, 1.0, on);
}

// Layer 1. `s` is the building's structural draw (see cc_facadeworks_detail).
fn cc_facadeworks_sign(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                       uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    // Cheap rejects first: one bit-slice of the building draw, then whether
    // this wall carries the sign, then whether this mass can carry one at
    // all. Signs live on tier 1 only — one that climbed onto a setback would
    // hang off the inset wall's corner instead of the street's.
    //
    // The draw picks a CORNER of the plan, not a wall, and the sign box is
    // mounted on it, so both walls meeting there carry the strip, on
    // whichever of their two u-ends that corner is. One sign, two faces —
    // which is how corner signage actually works, and it is also what makes
    // the layer land at street level: gating on a single wall put a sign in
    // front of the camera roughly never, since 12% of buildings times one
    // wall in four is 3% of the walls in a frame and a street view holds
    // about a dozen.
    //
    // The wall->corner table: wall s has corner (0x0231 >> 4s) & 3 at its u0
    // end and that corner plus one at its u1 end, walking the plan corners
    // (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax) counterclockwise.
    let c0 = i32((0x0231u >> (4u * u32(f.side))) & 3u);
    let corner = i32((s.x >> 8u) & 3u);
    let right_edge = corner == ((c0 + 1) & 3);
    if ((s.x & 255u) >= cc_facadeworks_SIGN_FRAC
        || !(right_edge || corner == c0)
        || !f.base || f.z1 <= 17.0 || (f.u1 - f.u0) <= 6.5) {
        return vec3<f32>(0.0);
    }
    // The sign is sized to the mass it hangs on. A fixed 12-40 m strip
    // starting at 6 m is right for a mid-rise and invisible on everything
    // else: soar is a fly-through whose camera lives well above 46 m, and
    // this city's megatower tier-1 walls are a KILOMETRE of facade that
    // would carry a 40 m mark at the very bottom and nothing above it. So a
    // tall wall takes a length proportional to itself instead — a ribbon
    // running most of the tower, which is the Cloudpunk cue — while a
    // mid-rise keeps the fixed law exactly. The cross-section and the glyph
    // blocks scale as the square root of the length ratio, which holds the
    // character aspect fixed while a tower's sign reads at a tower's size.
    let r_len = city_u01(s.y);
    let tall = mix(0.0, 0.85, smoothstep(60.0, 600.0, f.z1))
               * (0.5 + 0.5 * r_len);
    let len = max(12.0 + 28.0 * r_len, tall * f.z1);
    let gscale = sqrt(clamp(len / 26.0, 1.0, 12.0));
    let hw0 = cc_facadeworks_SIGN_HW * gscale;
    let seg_pitch = cc_facadeworks_SEG_PITCH * gscale;
    let seg_h = cc_facadeworks_SEG_H * gscale;

    let v0 = cc_facadeworks_SIGN_V0;
    let v1 = min(v0 + len, f.z1 - 2.0);
    if (v1 - v0 < 6.0 || vc < v0 - 1.0 || vc > v1 + 1.0) {
        return vec3<f32>(0.0);
    }
    let uanch = select(f.u0 + 1.9 * gscale, f.u1 - 1.9 * gscale, right_edge);
    let du = abs(uc - uanch);
    // The strip's cross-section, widened to the pixel once it is thinner
    // than one and dimmed by the same factor: the integral across u is
    // invariant, so a sign neither strobes nor washes out with range.
    let hw = max(hw0, 0.50 * fp);
    let soft = 0.30 * hw;
    let sig = max(cc_facadeworks_SIGN_HALO_SIG * gscale, 0.65 * fp);
    if (du > max(hw, 2.6 * sig)) {
        return vec3<f32>(0.0);
    }

    // The sign's own draw: color, brightness, and whether a third of its
    // tubes are out. No time in this shader, so a broken sign is permanently
    // broken rather than flickering — the dead blocks are the tell either way.
    let sh = city_rand4(cc.seed ^ vec2<u32>(0x1d8e4a37u, 0x6c9f2b05u));
    // Pushed draws: three explicit slots in city_window_color's palette at a
    // fixed bias, so signage lands on saturated amber / cyan / magenta and
    // never on the fluorescent white that would read as an office window.
    let sdraw = select(select(0.20, 0.89, sh.x > 0.34), 0.97, sh.x > 0.67);
    let scol = city_window_color(sdraw, 0.5);
    let rad = 5.0 + 3.0 * sh.y;
    let dead_frac = select(0.04, 0.35, sh.z < 0.15);

    // Glyph blocks up the strip.
    let sv = vc - v0;
    let iseg = i32(floor(sv / seg_pitch));
    let fsv = sv - f32(iseg) * seg_pitch;
    let sb = pcg2d(vec2<u32>(cc.seed.x ^ 0x51a3f7ddu,
                             bitcast<u32>(iseg) * 0x9e3779b9u));
    let alive = select(0.0, 1.0, city_u01(sb.y) >= dead_frac);
    let gu = (uc - (uanch - hw0)) / (2.0 * hw0);
    let ink = cc_facadeworks_glyph(sb.x, gu, fsv / seg_h);

    // The ladder. Strokes are ~0.2 m, so they go sub-pixel well before the
    // 1.15 m block pitch does: glyph interiors dissolve into the block mean
    // first, then blocks and their gaps dissolve into a continuous line
    // carrying the same mean energy. Both thresholds ride gscale, because a
    // tower's sign is physically bigger and stays resolved further out.
    let l1 = smoothstep(0.30 * gscale, 1.5 * gscale, fp);
    let l2 = smoothstep(1.2 * gscale, 6.0 * gscale, fp);
    let in_seg = select(0.0, 1.0, fsv < seg_h);
    let pan = cc_facadeworks_SIGN_PANEL;
    let lum = mix(ink, cc_facadeworks_GLYPH_MEAN, l1) * (1.0 - pan) + pan;
    let near = lum * in_seg * alive;
    let far = (cc_facadeworks_GLYPH_MEAN * (1.0 - pan) + pan)
              * cc_facadeworks_SEG_DUTY * (1.0 - dead_frac);
    let val = mix(near, far, l2);

    // Profile across the strip (integral 2*hw - soft, hence the amplitude)
    // and the wall spill either side of it, which is what tells the eye the
    // sign is a tube in front of a wall and not a painted rectangle.
    let prof = 1.0 - smoothstep(hw - soft, hw, du);
    let amp = hw0 / max(hw - 0.5 * soft, 1e-3);
    let halo = cc_facadeworks_SIGN_HALO_AMP
               * (cc_facadeworks_SIGN_HALO_SIG * gscale / sig) * far
               * exp(-0.5 * du * du / (sig * sig));
    // Both ends written ascending: smoothstep with edge0 > edge1 is undefined
    // in WGSL, however reasonably a given backend happens to evaluate it.
    let vgate = smoothstep(v0 - 0.3, v0 + 0.3, vc)
                * (1.0 - smoothstep(v1 - 0.3, v1 + 0.3, vc));
    return scol * (rad * vgate * (prof * amp * val + halo));
}

// Two gates: whether this storey's doors are open at all, and which of them
// along it spill. The second is what keeps the line dashed — a continuous
// band ruled across a whole wall reads as a stripe, and balconies are
// supposed to read as rhythm. One door per window pitch, and the spill
// fades top and bottom rather than ending on an edge, because it is light
// falling on a ledge and not a painted line.
fn cc_facadeworks_bal_lit(cc: CityCell, per: i32, uc: f32, vc: f32) -> f32 {
    let ivf = floor(vc / CITY_FLOOR_H);
    let iv = i32(ivf);
    let vloc = vc - ivf * CITY_FLOOR_H;
    if (((iv % per) + per) % per != 0
        || vloc <= cc_facadeworks_BAL_GLOW_V0
        || vloc >= cc_facadeworks_BAL_GLOW_V0 + cc_facadeworks_BAL_GLOW_H) {
        return 0.0;
    }
    let fu = fract(uc / CITY_WIN_PITCH_U);
    if (fu <= cc_facadeworks_BAL_DOOR_LO || fu >= cc_facadeworks_BAL_DOOR_HI) {
        return 0.0;
    }
    let fb = pcg2d(vec2<u32>(cc.seed.x ^ 0x3c6ef372u,
                             bitcast<u32>(iv) * 0x85ebca6bu));
    if (city_u01(fb.x) >= cc_facadeworks_BAL_FLOOR_P) {
        return 0.0;
    }
    let iu = i32(floor(uc / CITY_WIN_PITCH_U));
    let ub = pcg2d(vec2<u32>(fb.y, bitcast<u32>(iu) * 0xc2b2ae35u));
    if (city_u01(ub.x) >= cc_facadeworks_BAL_UNIT_P) {
        return 0.0;
    }
    let q = abs(vloc - (cc_facadeworks_BAL_GLOW_V0
                        + 0.5 * cc_facadeworks_BAL_GLOW_H))
            / (0.5 * cc_facadeworks_BAL_GLOW_H);
    return 1.0 - smoothstep(0.25, 1.0, q);
}

// Layer 2.
fn cc_facadeworks_balcony(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                          uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    if (((s.x >> 11u) & 255u) >= cc_facadeworks_BAL_FRAC
        || cc.height >= cc_facadeworks_BAL_MAX_H) {
        return vec3<f32>(0.0);
    }
    let per = select(2, 3, ((s.x >> 19u) & 1u) != 0u);
    let duty = cc_facadeworks_BAL_GLOW_H * cc_facadeworks_BAL_PROFILE_MEAN
               / (f32(per) * CITY_FLOOR_H);
    let far = duty * cc_facadeworks_BAL_FLOOR_P * cc_facadeworks_BAL_UNIT_P
              * cc_facadeworks_BAL_DOOR_FRAC;
    // A 0.45 m spill at a 7.2-10.8 m pitch is sub-pixel past ~1 m/px, so it
    // dissolves early. Its mean is ~0.0005 radiance — three orders under a
    // lit facade — which is why it can dissolve without leaving a mark.
    let l = smoothstep(0.40, 1.5, fp);
    if (l >= 0.999) {
        return cc_facadeworks_BAL_COLOR * (cc_facadeworks_BAL_RADIANCE * far);
    }
    let near = cc_facadeworks_bal_lit(cc, per, uc, vc);
    return cc_facadeworks_BAL_COLOR
           * (cc_facadeworks_BAL_RADIANCE * mix(near, far, l));
}

// Layer 3.
fn cc_facadeworks_escape(cc: CityCell, f: cc_facadeworks_Face, s: vec2<u32>,
                         uc: f32, vc: f32, fp: f32) -> vec3<f32> {
    // Filigree: fade the pattern into its mean first, then let that mean —
    // 0.013 radiance, a two-hundredth of a lit window — go out over a wide
    // footprint band. Fading a pedestal that small across two octaves of
    // range is not a pop; fading the PATTERN would have been.
    let l2 = smoothstep(2.0, 6.0, fp);
    if (((s.x >> 20u) & 255u) >= cc_facadeworks_FE_FRAC
        || cc.height <= 30.0 || cc.height >= 120.0
        || i32((s.x >> 28u) & 3u) != f.side
        || !f.base || (f.u1 - f.u0) <= 13.0 || l2 >= 0.999) {
        return vec3<f32>(0.0);
    }
    // Set in from the corner far enough to clear a sign strip on the same
    // wall (the two land together on well under 1% of buildings, but where
    // they do they should be neighbours, not the same column).
    let ufe = select(f.u0 + 4.5, f.u1 - 4.5, ((s.x >> 30u) & 1u) != 0u);
    let du = uc - ufe;
    if (abs(du) > cc_facadeworks_FE_HW) {
        return vec3<f32>(0.0);
    }
    let x = (du + cc_facadeworks_FE_HW) / (2.0 * cc_facadeworks_FE_HW);
    let ivf = floor(vc / CITY_FLOOR_H);
    let y = vc / CITY_FLOOR_H - ivf;
    // Stringers alternate direction storey to storey: the switchback is the
    // whole silhouette of a fire escape.
    let xx = select(x, 1.0 - x, ((i32(ivf) % 2) + 2) % 2 == 1);
    let on = abs(y - xx) < 0.055        // stair stringer
          || y < 0.055                  // landing at the floor line
          || abs(x - 0.055) < 0.024     // rails
          || abs(x - 0.945) < 0.024;
    let l1 = smoothstep(0.35, 1.6, fp);
    let val = mix(select(0.0, 1.0, on), cc_facadeworks_FE_MEAN, l1)
              * (1.0 - l2);
    let reach = 0.25 + 0.75 * exp(-max(vc - 6.0, 0.0)
                                  / cc_facadeworks_FE_FALL_H);
    return cc_facadeworks_FE_COLOR
           * (cc_facadeworks_FE_RADIANCE * reach * val);
}

fn cc_facadeworks_detail(cc: CityCell, h: CityHit, uc: f32, vc: f32,
                         fp: f32) -> vec3<f32> {
    // One draw per building for everything structural. pcg2d gives 64 bits;
    // slicing them beats three city_rand4 calls on a hook that runs at every
    // facade pixel, and the continuous draws each layer needs are taken
    // later, behind that layer's own reject.
    //
    //   bits  0-7   sign presence      8-9   sign wall      10  sign edge
    //        11-18  balcony presence    19   balcony pitch
    //        20-27  escape presence   28-29  escape wall     30  escape edge
    // (bit 10 is spare: signage picks a corner, so it needs no edge bit.)
    //   s.y         sign length
    let s = pcg2d(cc.seed ^ vec2<u32>(0x7f4a7c15u, 0x2b3d8e11u));
    let f = cc_facadeworks_face(cc, h);
    return cc_facadeworks_sign(cc, f, s, uc, vc, fp)
         + cc_facadeworks_balcony(cc, f, s, uc, vc, fp)
         + cc_facadeworks_escape(cc, f, s, uc, vc, fp);
}

// --- component: skyway (skyway.wgsl) ---
// skyway — the elevated freeway network.
//
// One system the block grid does not own. Everything else in the city is a
// property of a cell; a freeway is a LINE that runs through thousands of
// them, so it is traced once per ray (`extra_trace`) rather than per visited
// cell, and its whole geometry is closed form: a handful of slab tests
// against a ribbon whose top surface is a piecewise-linear function of the
// distance along it. No marcher, no DDA, no per-cell state.
//
// LATTICE. Routes ride the avenue lattice, sparsely. Blocks are
// `u.ocean_params.x` metres (90 on the shipped tile) and avenues fall every
// `CITY_AVENUE_PERIOD` = 8 blocks, so the finest legal route spacing is 720 m
// — far too dense. The route period is 24 blocks (2160 m), three avenues
// apart, which is what makes the network read as sparse drama rather than as
// a second lattice competing with the streets:
//
//   * an x-running deck on every avenue line with block index y = 0 (mod 24)
//   * a y-running deck on every avenue line with block index x = 8 (mod 24)
//
// The brief asked for the two directions to be offset by 12 blocks. Twelve is
// not a multiple of the avenue period, so a y-route offset by 12 would run
// down the middle of a BLOCK — through the buildings. Eight is the nearest
// offset that keeps both families on avenues (8 and 16 are equidistant from
// 12 and are reflections of each other), so the crossings sit off the
// diagonal without any route ever leaving an avenue. The whole thing is
// defined in world coordinates, so it is endless and identical from every
// camera and on every frame.
//
// Why the avenue matters: `city_cell` insets each plot by CITY_STREET_HALF
// (6 m) plus CITY_AVENUE_EXTRA (10 m) on an avenue boundary, so an avenue is
// a 32 m corridor of guaranteed empty air centred on the line. A 16 m deck on
// the centreline therefore clears the facades by 8 m on each side BY
// CONSTRUCTION, at any height — the flat-roof rule in SPEC's "Respect the
// architecture" is satisfied without a single building test. (Superblocks
// cannot break this either: `merged` groups are anchored on even indices and
// swallow only their odd internal boundary, and avenues are at multiples of
// 8.)
//
// PROFILE. The x-family runs dead flat at 26 m. Where the two families cross,
// the y-family climbs to 34 m and passes over — a real interchange decision,
// and the thing that stops the network being a flat plaid. The rise is
// modelled as the deck's top plane being piecewise linear in the along
// coordinate: flat base, a 200 m linear ramp, 124 m flat at the top, a ramp
// back down. Each piece is an exact slab test in the SHEARED coordinate
// h = z - z_top(a), so the ramps are true sloped slabs, not stair steps, and
// consecutive pieces share their end planes exactly — there is nowhere for a
// gap to open.
//
// Under the raised span the pylons would spear the deck they are crossing, so
// pylons inside CROSS_CLEAR of a crossing are simply absent: the high span
// carries itself across a 110 m gap, which is what a real overpass does.
//
// STRUCTURE. Three things, and no more, because the calibration is wiper
// blades and not tire brands: paired columns on 55 m bays, a pier CAP spanning
// each pair (a column running into the underside of a slab is the detail whose
// absence reads as unbuilt — the crosshead is what turns a row of sticks into
// a colonnade), and an expansion JOINT in the running surface over every bay,
// which is the only thing that says a viaduct is a chain of spans rather than
// a poured ribbon. Bearings, drainage, parapet posts and sign gantries are all
// below that line and are not here.
//
// CANDIDATE ROUTES. The expensive mistake here would be looping over route
// lines. Instead: the whole component lives in z ∈ [0, 35.1], so clip the ray
// to that slab first (this alone rejects every sky ray and most cloud rays
// for free), read off the interval of the lateral coordinate the ray spans
// inside it, and convert that interval directly into a range of route
// indices. For any ray that is not both near-horizontal AND inside the slab
// that range holds 0, 1 or 2 entries. Rays that are — a camera parked on the
// deck looking down it — get the nearest three, ordered by t, which is
// exactly the ordering that matters for an opaque first hit.
//
// LIGHT. The artery read from altitude is NOT the lamp heads: a 0.36 m rail
// at 4 m/px covers a tenth of a pixel and delivers a tenth of its radiance.
// It is what those strings put on the 16 m of DECK — the wash, plus the
// traffic under it — which is what actually makes a lit road photograph as a
// continuous bright ribbon from the air. The string dashes (radiance 5,
// warm white, every 9 m along both rail tops, collapsing into a continuous
// line of the same mean energy once a dash is sub-pixel) are the close read;
// the deck's own mean radiance, about 0.47, is the far one — roughly half
// wash and half frozen traffic.
//
// A note on what "bright" can mean here. At exposure 6 the tone map is
// nearly flat above radiance 1: 0.47 renders at 0.85 display and 1.7 (a
// downtown sodium pool at its peak) at 0.97. No honest deck radiance makes an
// artery outshine a lit avenue, and chasing one only blows out the close
// views. What separates the network from the lattice at a kilometre is
// therefore CONTINUITY and COLOUR — an unbroken near-white ribbon cutting
// diagonally across chains of orange dots, at its own altitude, three avenues
// from the next one. That is also how a real freeway reads from a plane.
//
// COST. Per ray: one z-slab clip (which returns immediately for anything
// looking at the sky), then per family one lateral-interval-to-index
// derivation and at most three corridor rejects, each two divides. A corridor
// the ray actually enters costs, worst case, five envelope slabs (only one of
// which can be non-empty for a ray that is not running along the deck), three
// detail slabs for the piece that hit, and — over at most three bays — one
// cap box and two column boxes each. The common city pixel, looking at a
// facade 2 km from any route, pays the z clip and six corridor rejects and
// nothing else.
//
// Measured on a 5080 at 200 accumulated passes, 960x540, registry entry on vs
// off, three interleaved repetitions per camera (interleaved because the box
// is SHARED and a block of A followed by a block of B measures the drift
// between them; the numbers below are the minimum of three and are contended
// regardless, so read them as an upper bound):
//
//     sky only   0.860 on / 0.850 off        artery aerial  0.470 / 0.460
//     on deck    0.350 on / 0.350 off        under deck     0.380 / 0.370
//
// That is +0.01 s on 200 passes where it is non-zero at all — 50 microseconds
// a pass, and exactly one tick of the harness's 10 ms print granularity. The
// honest reading is that the component's cost is at or below what this rig can
// measure, not that it is precisely 50 us.
//
// One caveat on the last two rows, since it would be easy to quote them as an
// occlusion win: with skyway disabled those cameras are not looking at the
// same scene at all — they are parked in mid-air over an empty avenue — so
// they are not an A/B of anything. An earlier revision of this header claimed
// a canyon view got twice as FAST with the network, from that same confound.
// It does not reproduce, and the claim is withdrawn.

// --- lattice ---------------------------------------------------------------
// In blocks, so the network scales with the tile's own cell size.
const cc_skyway_ROUTE_BLOCKS: f32 = 24.0;   // 3 avenues between routes
const cc_skyway_OFFSET_BLOCKS: f32 = 8.0;   // y-family phase (one avenue)

// --- deck ------------------------------------------------------------------
const cc_skyway_HW: f32 = 8.0;        // half width: a 16 m deck in a 32 m avenue
const cc_skyway_THICK: f32 = 1.4;
const cc_skyway_DECK_Z: f32 = 26.0;   // top of the base deck
const cc_skyway_HIGH_Z: f32 = 34.0;   // top of a raised crossing
const cc_skyway_RAMP: f32 = 200.0;    // linear approach length
const cc_skyway_FLAT_HALF: f32 = 62.0; // half the flat run over a crossing
const cc_skyway_RAIL_H: f32 = 1.1;
const cc_skyway_RAIL_W: f32 = 0.36;
const cc_skyway_PYL_SP: f32 = 55.0;   // pylon bays
const cc_skyway_PYL_HW: f32 = 0.6;    // 1.2 m square
const cc_skyway_PYL_OFF: f32 = 5.2;   // paired, in from the deck edges
// Pier cap: the crosshead the two columns of a bay carry, and what the deck
// actually sits on. A column meeting a slab edge-on is the thing that reads as
// unbuilt — every real viaduct puts a transverse beam in between, and at
// wiper-blades calibration that beam IS the feature (no bearings, no pintles,
// no plaque). It spans the pair and oversails each column by CAP_OS, so the
// colonnade under a deck reads as a row of T's rather than a row of sticks.
const cc_skyway_CAP_H: f32 = 1.05;    // depth of the crosshead
const cc_skyway_CAP_HL: f32 = 1.15;   // half-length along the deck
const cc_skyway_CAP_OS: f32 = 0.55;   // oversail beyond the outer column face
// No pylon within this of a crossing centre: the span it would land in is the
// one the other deck occupies.
const cc_skyway_CROSS_CLEAR: f32 = 22.0;
// The along-range used to choose WHICH crossing a corridor is near. Bounded
// so a near-horizontal ray running kilometres down a deck picks the
// interchange in front of the camera rather than one 15 km away.
const cc_skyway_MC_SPAN: f32 = 1400.0;
// Stands in for "endless" on the along axis. World coordinates are ~1e5 and
// CITY_TRACE_RANGE is 3e4, so this is out of reach without costing precision.
const cc_skyway_FAR: f32 = 1.0e6;
const cc_skyway_Z_TOP: f32 = 35.1;    // HIGH_Z + RAIL_H: the component's slab

// --- light -----------------------------------------------------------------
// SPEC rule 5 yardsticks: lit window 3.5, storefront 2.2, sodium pool 0.7.
const cc_skyway_STR_COLOR: vec3<f32> = vec3<f32>(1.00, 0.85, 0.60);
// The brief's number. It is also, at this exposure, an unfalsifiable one: the
// lit face of a rail lands at radiance 3 either way and the tone map has
// nothing left above 2, so 5 and the 6.5 this file used to carry are the same
// pixel. The rail is 0.36 m of a 17 m cross-section, so the far-field mean
// does not notice either. Kept at the contract value because deviating from it
// buys nothing.
const cc_skyway_STR_RAD: f32 = 5.0;
const cc_skyway_STR_P: f32 = 9.0;     // light-string period along the rail
const cc_skyway_STR_L: f32 = 3.3;     // lit length of each dash
// Where a dash stops being resolvable and becomes a continuous line of the
// same mean energy. A 3.3 m dash is one pixel at ~3.3 m/px, so the hand-off
// straddles the brief's 4 m.
const cc_skyway_STR_FP_LO: f32 = 2.4;
const cc_skyway_STR_FP_HI: f32 = 4.8;
// How much of the string a rail face carries. The strip fixture sits on the
// rail top; the road side sees more of it than the outside does. It is a
// BAND, not the whole face — a 1.1 m barrier lit end to end reads as a broken
// white wall, whereas the same light in the top 0.35 m reads as what it is: a
// dark concrete barrier with a lamp strip running along it.
const cc_skyway_RAIL_FACE_IN: f32 = 0.60;
const cc_skyway_RAIL_FACE_OUT: f32 = 0.34;
const cc_skyway_RAIL_BAND: f32 = 0.35;  // lit depth below the rail top (m)
// The wash the strings throw across the deck. This is half the artery — but
// it is a GRADIENT, not a bar, and the difference is everything. At exposure
// 6 with a white point of 15 the tone map has already spent most of its range
// by radiance 1.5 (0.1 renders at 0.55 display, 1.0 at 0.93, 3.5 at white),
// so a deck lit evenly at 1.3 arrives as a featureless white ribbon with its
// traffic invisible inside it. E-folded over 1.9 m instead, the same deck
// runs 0.95 display under a lamp at the rail down to about 0.2 along the
// centreline: bright margins, dark lanes, and headroom for headlights.
//
// A string of lamps 9 m apart also does not light a road evenly — it scallops
// it, and that rhythm running away toward the vanishing point is most of what
// says "lit road at night" rather than "glowing strip". SCALLOP is the floor
// between lamps, and WASH_AMP is pre-divided by the pattern's own mean (0.857)
// so adding the rhythm does not move the deck's far-field brightness.
const cc_skyway_WASH_AMP: f32 = 1.30;
const cc_skyway_WASH_E: f32 = 1.90;   // e-folding in from the rail (m)
const cc_skyway_SCALLOP: f32 = 0.35;  // between-lamp floor
const cc_skyway_SCALLOP_S: f32 = 2.80; // pool sigma along the deck (m)
const cc_skyway_ASPHALT: f32 = 0.05;  // deck albedo against the skyglow
const cc_skyway_CONCRETE: f32 = 0.055;
// Uplight: the sodium pools on the street 25 m below, thrown back onto the
// underside and the pylons. `city_street_pools` is the same function the
// ground uses, sampled at the point directly below — so the scallops along an
// underside land exactly over the lamps that cause them.
//
// The coefficient is small because what it multiplies is not: `pools` sums
// four lamp terms and carries a x3 factor on an avenue, so it peaks near 6
// right where a deck sits, and `street_scale` multiplies that by up to 2.4
// downtown. The 0.3 the brief suggested puts the soffit at radiance 4 — a
// glowing cream ceiling, brighter than the road on top of it. At 0.010 the
// same geometry peaks near 0.14, which is a dark concrete soffit with warm
// scallops picked out along the lamp line: what the brief actually asked for.
const cc_skyway_UPLIGHT: f32 = 0.010;
const cc_skyway_PYL_UP_E: f32 = 16.0; // uplight e-folds up a pylon (m)
// Transverse girders under the deck. Zero-mean modulation, so it is texture
// and not brightness, and it hands over to flat once it is sub-pixel.
const cc_skyway_RIB_P: f32 = 3.2;
const cc_skyway_RIB: f32 = 0.34;
const cc_skyway_RIB_FP: f32 = 1.20;

// --- lane paint ------------------------------------------------------------
const cc_skyway_PAINT_W: f32 = 0.18;
const cc_skyway_PAINT_AMP: f32 = 0.55;
const cc_skyway_LANE_DIV: f32 = 4.0;  // four 4 m lanes across the 16 m deck
const cc_skyway_EDGE_LINE: f32 = 7.35;
const cc_skyway_CENTRE_LINE: f32 = 0.45;
const cc_skyway_DASH_P: f32 = 12.0;
const cc_skyway_DASH_DUTY: f32 = 0.42;
const cc_skyway_PAINT_FP_LO: f32 = 0.30;
const cc_skyway_PAINT_FP_HI: f32 = 1.10;

// --- expansion joints -------------------------------------------------------
// A viaduct is not a ribbon, it is a chain of spans, and the joint over each
// pier is where that fact becomes visible. One per pylon bay (55 m), the full
// width of the deck, in the running surface and carried down the fascia — the
// same s for both, so the line on the side is the end of the line on top.
//
// Two-tone, because a joint is two things: the GAP (a slot with nothing in it
// — the darkest thing on a lit deck, and the part the eye actually reads) and
// the COMB plates either side of it (bare steel, which under a warm lamp
// strip is a shade brighter than asphalt, not darker). One without the other
// reads as a crack or as a stripe; together they read as hardware.
//
// Derived from s alone, so it does not care whether the bay under it has a
// pylon: segments still butt over a suppressed pier, and the chain stays
// unbroken across a crossing.
// Sized as what a 55 m span actually needs. A span that long moves 40-60 mm
// between summer and winter, which is past what a single sealed gap will take,
// so the joint is a MODULAR one: a wide steel frame with the movement split
// across several seals. That is a genuine 0.7 m of deck hardware — the first
// draft here used 0.38 m, which is a joint for a 20 m span and, at this
// renderer's footprints, one that is never once resolved.
const cc_skyway_JNT_GAP: f32 = 0.11;    // half-width of the sealed slot (m)
const cc_skyway_JNT_PLATE: f32 = 0.35;  // half-width of the whole frame (m)
const cc_skyway_JNT_DARK: f32 = 0.80;   // how much of the wash the slot eats
const cc_skyway_JNT_STEEL: f32 = 0.22;  // plate lift, as a fraction of wash
// Where the 0.70 m frame stops being resolvable, in the ALONG footprint
// (fp_eff, not fp — see the note in the shader). Under the app's default LOD
// slider that puts the last legible joint about 25 m ahead and the pattern
// fully collapsed by 75 m, which is roughly one span and a half: you see the
// joint you are crossing and the one after it, and beyond that the deck is
// honestly smooth rather than dishonestly striped.
const cc_skyway_JNT_FP_LO: f32 = 0.65;
const cc_skyway_JNT_FP_HI: f32 = 2.00;

// --- traffic streaks -------------------------------------------------------
// Four lanes: two per direction, at |lat| = 2 and 6. One hash draw per
// (route, lane, 70 m cell), so with the occupancy below a lane carries a
// trail every ~85 m — inside the brief's 80-200, and dense enough that the
// population's MEAN is a real share of what the deck delivers at a kilometre
// rather than a rounding error on the wash.
const cc_skyway_STK_CELL: f32 = 70.0;
const cc_skyway_STK_L: f32 = 7.0;
const cc_skyway_STK_W: f32 = 0.65;    // lateral sigma (m)
const cc_skyway_STK_P: f32 = 0.85;    // slot occupancy at full district glow
const cc_skyway_STK_TAIL: f32 = 2.2;  // e-folds along the frozen trail
const cc_skyway_STK_HALO: f32 = 0.16;   // glow around the trail core
const cc_skyway_STK_HALO_W: f32 = 2.20; // its sigma (m)
const cc_skyway_STK_FP_LO: f32 = 1.6;
const cc_skyway_STK_FP_HI: f32 = 5.0;
// Headlights blow out; tail lamps must NOT. The tone map desaturates anything
// it clips, so a red at radiance 6 arrives pink-white — the colour that says
// "receding traffic" only survives if the green and blue channels stay off
// the shoulder. Hence a deeply saturated red at a third of the white's
// radiance, which lands at (1.0, 0.64, 0.42) in display: unmistakably red,
// still the brightest thing in its lane.
const cc_skyway_HEAD_COL: vec3<f32> = vec3<f32>(1.00, 0.95, 0.86);
const cc_skyway_HEAD_RAD: f32 = 16.0;
const cc_skyway_TAIL_COL: vec3<f32> = vec3<f32>(1.00, 0.055, 0.018);
const cc_skyway_TAIL_RAD: f32 = 4.5;

// --- hit kinds --------------------------------------------------------------
// kind = 200 + 8*family + part. SPEC allows local in [0, 15]; a stride of 8
// with five parts spends 200..212 of it and leaves the arithmetic to a shift.
// The family bit is what lets the shader rebuild the local frame from nothing
// but the kind and the world position — no per-hit payload to carry.
const cc_skyway_KIND_BASE: i32 = 200;
const cc_skyway_FAM_STRIDE: i32 = 8;
const cc_skyway_P_DECK: i32 = 0;   // running surface
const cc_skyway_P_UNDER: i32 = 1;  // soffit and fascia
const cc_skyway_P_PYL: i32 = 2;    // column
const cc_skyway_P_RAIL: i32 = 3;   // edge barrier
const cc_skyway_P_CAP: i32 = 4;    // pier cap

// The local frame of a route: x = along the deck, y = lateral, z = up. The
// two families differ only by which world axis is "along", so one x/y swap is
// the whole transform — and it is its own inverse, which is why the same call
// maps a normal back to world.
fn cc_skyway_swap(v: vec3<f32>, fam: i32) -> vec3<f32> {
    return select(vec3<f32>(v.y, v.x, v.z), v, fam == 0);
}

// A ray interval being whittled down by slabs. `axis` and `sgn` remember
// which slab last raised the near end, which is the face the ray enters
// through and therefore the one whose normal it wears.
struct cc_skyway_Span {
    t0: f32,
    t1: f32,
    axis: i32,
    sgn: f32,
}

// One slab: keep the part of the interval where a + b*t lies in [lo, hi].
// Works for a sheared coordinate as readily as for a world axis, which is the
// whole trick behind the sloped ramps.
fn cc_skyway_clip(sp: cc_skyway_Span, a: f32, b: f32, lo: f32, hi: f32,
                  axis: i32) -> cc_skyway_Span {
    var r = sp;
    if (abs(b) < 1.0e-9) {
        if (a < lo || a > hi) {
            r.t1 = r.t0 - 1.0;   // parallel and outside: empty forever
        }
        return r;
    }
    let ta = (lo - a) / b;
    let tb = (hi - a) / b;
    let tn = min(ta, tb);
    if (tn > r.t0) {
        r.t0 = tn;
        r.axis = axis;
        r.sgn = select(1.0, -1.0, b > 0.0);
    }
    r.t1 = min(r.t1, max(ta, tb));
    return r;
}

// The entering face's outward normal, in world space. Axes 0 and 1 are the
// along and lateral slabs; axis 2 is the sheared one, whose gradient is
// (-m, 0, 1) in the local frame — that tilt is what makes a ramp shade as a
// ramp rather than as a flat deck at the wrong height.
fn cc_skyway_normal(axis: i32, sgn: f32, m: f32, fam: i32) -> vec3<f32> {
    var nl = vec3<f32>(0.0, 0.0, 1.0);
    if (axis == 0) {
        nl = vec3<f32>(sgn, 0.0, 0.0);
    } else if (axis == 1) {
        nl = vec3<f32>(0.0, sgn, 0.0);
    } else {
        nl = normalize(vec3<f32>(-m, 0.0, 1.0)) * sgn;
    }
    return cc_skyway_swap(nl, fam);
}

struct cc_skyway_Best {
    t: f32,
    nrm: vec3<f32>,
    kind: i32,
}

// The deck's top surface as a function of the along coordinate. The x-family
// never rises; the y-family rises over its crossing with the x-family. Kept
// LINEAR on purpose — it has to agree exactly with the sloped slabs below, or
// the pylons would not meet the underside they hold up.
fn cc_skyway_ztop(a: f32, ac: f32, raised: bool) -> f32 {
    if (!raised) {
        return cc_skyway_DECK_Z;
    }
    let d = abs(a - ac);
    if (d <= cc_skyway_FLAT_HALF) {
        return cc_skyway_HIGH_Z;
    }
    if (d >= cc_skyway_FLAT_HALF + cc_skyway_RAMP) {
        return cc_skyway_DECK_Z;
    }
    return cc_skyway_HIGH_Z
        + (cc_skyway_DECK_Z - cc_skyway_HIGH_Z)
          * (d - cc_skyway_FLAT_HALF) / cc_skyway_RAMP;
}

// One piece of ribbon: the deck slab over [a0, a1] with top plane
// z = zb + m*(a - ab), and the two rails riding on it. An envelope test
// covering deck and rails together gates the three detail tests, so a piece
// the ray never reaches costs three clips and stops.
fn cc_skyway_ribbon(lo3: vec3<f32>, ld3: vec3<f32>, ct0: f32, ct1: f32,
                    a0: f32, a1: f32, ab: f32, zb: f32, m: f32,
                    fam: i32, best: cc_skyway_Best) -> cc_skyway_Best {
    var r = best;
    let h0 = lo3.z - zb - m * (lo3.x - ab);
    let hd = ld3.z - m * ld3.x;
    let seed = cc_skyway_Span(ct0, ct1, -1, 1.0);

    var env = cc_skyway_clip(seed, lo3.x, ld3.x, a0, a1, 0);
    env = cc_skyway_clip(env, lo3.y, ld3.y, -cc_skyway_HW, cc_skyway_HW, 1);
    env = cc_skyway_clip(env, h0, hd, -cc_skyway_THICK, cc_skyway_RAIL_H, 2);
    if (env.t0 > env.t1 || env.t0 >= r.t) {
        return r;
    }

    // The deck itself: everything from the running surface down to the
    // fascia's lower edge.
    var d = cc_skyway_clip(seed, lo3.x, ld3.x, a0, a1, 0);
    d = cc_skyway_clip(d, lo3.y, ld3.y, -cc_skyway_HW, cc_skyway_HW, 1);
    d = cc_skyway_clip(d, h0, hd, -cc_skyway_THICK, 0.0, 2);
    if (d.t0 <= d.t1 && d.axis >= 0 && d.t0 < r.t) {
        let top = d.axis == 2 && d.sgn > 0.0;
        r = cc_skyway_Best(d.t0,
                           cc_skyway_normal(d.axis, d.sgn, m, fam),
                           cc_skyway_KIND_BASE + cc_skyway_FAM_STRIDE * fam
                           + select(1, 0, top));
    }

    // Edge rails, one either side, standing on the deck top.
    for (var side: i32 = 0; side < 2; side = side + 1) {
        let l0 = select(-cc_skyway_HW,
                        cc_skyway_HW - cc_skyway_RAIL_W, side == 1);
        let l1 = select(-cc_skyway_HW + cc_skyway_RAIL_W,
                        cc_skyway_HW, side == 1);
        var q = cc_skyway_clip(seed, lo3.x, ld3.x, a0, a1, 0);
        q = cc_skyway_clip(q, lo3.y, ld3.y, l0, l1, 1);
        q = cc_skyway_clip(q, h0, hd, 0.0, cc_skyway_RAIL_H, 2);
        if (q.t0 <= q.t1 && q.axis >= 0 && q.t0 < r.t) {
            r = cc_skyway_Best(q.t0,
                               cc_skyway_normal(q.axis, q.sgn, m, fam),
                               cc_skyway_KIND_BASE + cc_skyway_FAM_STRIDE * fam
                               + cc_skyway_P_RAIL);
        }
    }
    return r;
}

// Paired pylons, ground to underside. The along position is quantized, so the
// nearest few are found by rounding the ray's own along coordinate rather
// than by searching: take the bay at the entry to the under-deck z-band and
// walk at most three of them in the direction of travel. That is the whole
// colonnade a ray can see before the near pylons occlude the far ones.
fn cc_skyway_pylons(lo3: vec3<f32>, ld3: vec3<f32>, ct0: f32, ct1: f32,
                    ac: f32, raised: bool, fam: i32,
                    best: cc_skyway_Best) -> cc_skyway_Best {
    var r = best;
    var pt0 = ct0;
    var pt1 = ct1;
    let ztop_max = cc_skyway_HIGH_Z - cc_skyway_THICK;
    if (abs(ld3.z) > 1.0e-9) {
        let ta = (0.0 - lo3.z) / ld3.z;
        let tb = (ztop_max - lo3.z) / ld3.z;
        pt0 = max(pt0, min(ta, tb));
        pt1 = min(pt1, max(ta, tb));
    } else if (lo3.z < 0.0 || lo3.z > ztop_max) {
        return r;
    }
    if (pt0 > pt1 || pt0 >= r.t) {
        return r;
    }

    let sa = lo3.x + pt0 * ld3.x;
    let sb = lo3.x + pt1 * ld3.x;
    let na = i32(round(sa / cc_skyway_PYL_SP));
    let nb = i32(round(sb / cc_skyway_PYL_SP));
    let dn = select(-1, 1, nb >= na);
    let seed = cc_skyway_Span(ct0, ct1, -1, 1.0);
    for (var q: i32 = 0; q < 3; q = q + 1) {
        let n = na + q * dn;
        if ((dn > 0 && n > nb) || (dn < 0 && n < nb)) {
            break;
        }
        let s_n = f32(n) * cc_skyway_PYL_SP;
        // The bay a crossing occupies carries no pylon: the high span crosses
        // it clean.
        if (raised && abs(s_n - ac) < cc_skyway_CROSS_CLEAR) {
            continue;
        }
        // The underside at this bay. On a ramp it is sloped, but the cap is
        // only 2.3 m long, so evaluating the profile at the bay centre puts
        // the cap's flat top within 5 cm of it — and that discrepancy is
        // BURIED, because the cap is narrower than the deck on both axes and
        // the deck is the nearer surface wherever the two overlap.
        let soffit = cc_skyway_ztop(s_n, ac, raised) - cc_skyway_THICK;
        let cap_bot = soffit - cc_skyway_CAP_H;

        // The crosshead, spanning the pair.
        let cap_hw = cc_skyway_PYL_OFF + cc_skyway_PYL_HW + cc_skyway_CAP_OS;
        var cp = cc_skyway_clip(seed, lo3.x, ld3.x,
                                s_n - cc_skyway_CAP_HL,
                                s_n + cc_skyway_CAP_HL, 0);
        cp = cc_skyway_clip(cp, lo3.y, ld3.y, -cap_hw, cap_hw, 1);
        cp = cc_skyway_clip(cp, lo3.z, ld3.z, cap_bot, soffit, 2);
        if (cp.t0 <= cp.t1 && cp.axis >= 0 && cp.t0 < r.t) {
            r = cc_skyway_Best(cp.t0,
                               cc_skyway_normal(cp.axis, cp.sgn, 0.0, fam),
                               cc_skyway_KIND_BASE
                               + cc_skyway_FAM_STRIDE * fam
                               + cc_skyway_P_CAP);
        }

        // The columns, ground to the underside of the crosshead.
        for (var side: i32 = 0; side < 2; side = side + 1) {
            let lc = select(-cc_skyway_PYL_OFF, cc_skyway_PYL_OFF, side == 1);
            var p = cc_skyway_clip(seed, lo3.x, ld3.x,
                                   s_n - cc_skyway_PYL_HW,
                                   s_n + cc_skyway_PYL_HW, 0);
            p = cc_skyway_clip(p, lo3.y, ld3.y,
                               lc - cc_skyway_PYL_HW,
                               lc + cc_skyway_PYL_HW, 1);
            p = cc_skyway_clip(p, lo3.z, ld3.z, 0.0, cap_bot, 2);
            if (p.t0 <= p.t1 && p.axis >= 0 && p.t0 < r.t) {
                r = cc_skyway_Best(p.t0,
                                   cc_skyway_normal(p.axis, p.sgn, 0.0, fam),
                                   cc_skyway_KIND_BASE
                                   + cc_skyway_FAM_STRIDE * fam
                                   + cc_skyway_P_PYL);
            }
        }
    }
    return r;
}

// One route line: the corridor reject, then the ribbon pieces and the pylons.
fn cc_skyway_route(o: vec3<f32>, dir: vec3<f32>, fam: i32, lat_c: f32,
                   tz0: f32, tz1: f32, period: f32,
                   best: cc_skyway_Best) -> cc_skyway_Best {
    var r = best;
    let lw = cc_skyway_swap(o, fam);
    let ld3 = cc_skyway_swap(dir, fam);
    // Lateral coordinates measured from the route's own centreline, so every
    // constant below is a half-width and the two families share the code.
    let lo3 = vec3<f32>(lw.x, lw.y - lat_c, lw.z);

    var ct0 = tz0;
    var ct1 = tz1;
    if (abs(ld3.y) > 1.0e-9) {
        let ta = (-cc_skyway_HW - lo3.y) / ld3.y;
        let tb = (cc_skyway_HW - lo3.y) / ld3.y;
        ct0 = max(ct0, min(ta, tb));
        ct1 = min(ct1, max(ta, tb));
    } else if (abs(lo3.y) > cc_skyway_HW) {
        return r;
    }
    if (ct0 > ct1 || ct0 >= r.t) {
        return r;
    }

    let raised = fam == 1;
    if (!raised) {
        // Flat from horizon to horizon.
        return cc_skyway_pylons(
            lo3, ld3, ct0, ct1, 0.0, false, fam,
            cc_skyway_ribbon(lo3, ld3, ct0, ct1,
                             -cc_skyway_FAR, cc_skyway_FAR,
                             0.0, cc_skyway_DECK_Z, 0.0, fam, r));
    }

    // Which crossing this corridor is near. For anything but a ray running
    // along the deck the corridor is metres long and this is exact; for one
    // that is, MC_SPAN picks the interchange ahead of the camera. Farther
    // crossings on the same line stay flat — 2 km down a deck, behind fog and
    // towers, and the pieces still tile the line with no gap.
    let tm = ct0 + 0.5 * min(ct1 - ct0, cc_skyway_MC_SPAN);
    let ac = round((lo3.x + tm * ld3.x) / period) * period;
    let e = cc_skyway_FLAT_HALF;
    let rp = cc_skyway_RAMP;
    let slope = (cc_skyway_HIGH_Z - cc_skyway_DECK_Z) / rp;

    // Five pieces sharing four end planes exactly: base, ramp up, high flat,
    // ramp down, base.
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, -cc_skyway_FAR, ac - e - rp,
                         0.0, cc_skyway_DECK_Z, 0.0, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac - e - rp, ac - e,
                         ac - e - rp, cc_skyway_DECK_Z, slope, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac - e, ac + e,
                         0.0, cc_skyway_HIGH_Z, 0.0, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac + e, ac + e + rp,
                         ac + e, cc_skyway_HIGH_Z, -slope, fam, r);
    r = cc_skyway_ribbon(lo3, ld3, ct0, ct1, ac + e + rp, cc_skyway_FAR,
                         0.0, cc_skyway_DECK_Z, 0.0, fam, r);
    return cc_skyway_pylons(lo3, ld3, ct0, ct1, ac, true, fam, r);
}

// `inv_dir` is the hook's, and deliberately unused: the core builds it with
// 1e30 standing in for a zero component, which is fine for an axis-aligned box
// but not for the sheared slab a ramp needs. Every divide below is guarded at
// the point of use instead.
fn cc_skyway_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
        -> CityHit {
    var miss: CityHit;
    miss.hit = false;
    miss.t = 1e30;
    miss.pos = vec3<f32>(0.0);
    miss.normal = vec3<f32>(0.0, 0.0, 1.0);
    miss.kind = 0;
    miss.cell = vec2<i32>(0);

    // Gate 1, and the one that pays for the component: everything skyway owns
    // lives between the ground and 35.1 m. A ray that never enters that slab
    // — every sky ray, every ray already above the deck and climbing — is
    // done in four operations.
    var tz0 = 1.0e-3;
    var tz1 = CITY_TRACE_RANGE;
    if (abs(dir.z) > 1.0e-9) {
        let ta = (0.0 - o.z) / dir.z;
        let tb = (cc_skyway_Z_TOP - o.z) / dir.z;
        tz0 = max(tz0, min(ta, tb));
        tz1 = min(tz1, max(ta, tb));
    } else if (o.z < 0.0 || o.z > cc_skyway_Z_TOP) {
        return miss;
    }
    if (tz0 >= tz1) {
        return miss;
    }

    let cell = u.ocean_params.x;
    let period = cc_skyway_ROUTE_BLOCKS * cell;
    var best = cc_skyway_Best(1e30, vec3<f32>(0.0, 0.0, 1.0), -1);

    for (var fam: i32 = 0; fam < 2; fam = fam + 1) {
        let ld3 = cc_skyway_swap(dir, fam);
        let lw = cc_skyway_swap(o, fam);
        let phase = select(0.0, cc_skyway_OFFSET_BLOCKS * cell, fam == 1);
        // The lateral interval the ray spans while inside the z slab, widened
        // by the deck half-width, converted straight into route indices. This
        // is the analytic step that replaces a loop over lines.
        let wa = lw.y + tz0 * ld3.y;
        let wb = lw.y + tz1 * ld3.y;
        let w_lo = min(wa, wb) - cc_skyway_HW;
        let w_hi = max(wa, wb) + cc_skyway_HW;
        let m_lo = i32(ceil((w_lo - phase) / period));
        let m_hi = i32(floor((w_hi - phase) / period));
        let n_cand = min(m_hi - m_lo + 1, 3);
        // Nearest first: with an opaque network that ordering is what makes
        // the three-line cap harmless.
        let step = select(-1, 1, ld3.y >= 0.0);
        let m_start = select(m_hi, m_lo, ld3.y >= 0.0);
        for (var k: i32 = 0; k < n_cand; k = k + 1) {
            let lat_c = phase + f32(m_start + k * step) * period;
            best = cc_skyway_route(o, dir, fam, lat_c, tz0, tz1, period, best);
        }
    }

    if (best.kind < 0) {
        return miss;
    }
    var res: CityHit;
    res.hit = true;
    res.t = best.t;
    res.pos = o + best.t * dir;
    res.normal = best.nrm;
    res.kind = best.kind;
    res.cell = vec2<i32>(floor(res.pos.xy / cell));
    return res;
}

// Frozen traffic. One draw per (route family, lane, 70 m cell) places a single
// trail; which side of the centreline the lane is on decides which way it is
// going and therefore whether you are looking at its headlights or its tail
// lamps. Only the two lanes on the shaded point's own side are evaluated —
// the far pair is four metres of a 0.65 m sigma away and contributes e^-19.
// Below the LOD gate the trail is resolved; above it, the population's mean
// over the whole deck, both colours, which is the honest asymptote.
fn cc_skyway_streaks(s: f32, lat: f32, fam: i32, glow: f32, fp: f32)
        -> vec3<f32> {
    let sd = select(0u, 1u, lat >= 0.0);
    let side = select(-1.0, 1.0, lat >= 0.0);
    let n = floor(s / cc_skyway_STK_CELL);
    let ni = bitcast<u32>(i32(n));
    let p = cc_skyway_STK_P
        * mix(0.35, 1.25, smoothstep(0.02, 0.35, glow));
    let col = select(cc_skyway_TAIL_COL, cc_skyway_HEAD_COL, sd == 1u);
    let rad = select(cc_skyway_TAIL_RAD, cc_skyway_HEAD_RAD, sd == 1u);

    var f = 0.0;
    for (var slot: i32 = 0; slot < 2; slot = slot + 1) {
        let key = (u32(fam) * 3u + u32(slot)) * 2u + sd;
        let r = city_rand4(vec2<u32>(
            ni * 0x9e3779b9u + key * 0x51ed270bu + 0x165667b1u,
            (ni * 0x85ebca6bu) ^ (key * 0xc2b2ae35u + 0x27d4eb2fu)));
        if (r.x >= p) {
            continue;
        }
        let s_c = (n + 0.15 + 0.70 * r.y) * cc_skyway_STK_CELL;
        let lane = side * (2.0 + 4.0 * f32(slot));
        // Travel is +along on the lat>0 side, -along on the other; the trail
        // is frozen BEHIND the head, so it points against travel.
        let x = -((s - s_c) * side) / cc_skyway_STK_L;
        if (x < 0.0 || x > 1.0) {
            continue;
        }
        let dl = lat - lane;
        // Core plus halo: a lamp seen on a road surface is a hard bright
        // patch inside a soft glow, and without the second term the trails
        // read as painted dashes rather than as light.
        let core = exp(-dl * dl
                       / (2.0 * cc_skyway_STK_W * cc_skyway_STK_W));
        let halo = cc_skyway_STK_HALO
            * exp(-dl * dl / (2.0 * cc_skyway_STK_HALO_W
                              * cc_skyway_STK_HALO_W));
        f = f + (0.55 + 0.90 * r.w) * exp(-cc_skyway_STK_TAIL * x)
                * (core + halo);
    }

    // The population's mean over the deck: occupancy x the trail's integral
    // along s x its lateral integral, over the cell's area, summed over the
    // two lanes of each colour. Written out rather than fitted, so changing
    // STK_L or STK_W keeps the two ends of the LOD agreeing by construction.
    let along_int = cc_skyway_STK_L * (1.0 - exp(-cc_skyway_STK_TAIL))
                    / cc_skyway_STK_TAIL;
    let lat_int = 2.5066 * (cc_skyway_STK_W
                            + cc_skyway_STK_HALO * cc_skyway_STK_HALO_W);
    let unit = 2.0 * p * along_int * lat_int
               / (cc_skyway_STK_CELL * 2.0 * cc_skyway_HW);
    let mean = (cc_skyway_HEAD_COL * cc_skyway_HEAD_RAD
                + cc_skyway_TAIL_COL * cc_skyway_TAIL_RAD) * unit;

    let k = smoothstep(cc_skyway_STK_FP_LO, cc_skyway_STK_FP_HI, fp);
    return mix(col * (rad * f), mean, k);
}

fn cc_skyway_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let code = h.kind - cc_skyway_KIND_BASE;
    let fam = code / cc_skyway_FAM_STRIDE;
    let part = code % cc_skyway_FAM_STRIDE;
    let cell = u.ocean_params.x;
    let period = cc_skyway_ROUTE_BLOCKS * cell;
    let phase = select(0.0, cc_skyway_OFFSET_BLOCKS * cell, fam == 1);

    let lp = cc_skyway_swap(h.pos, fam);
    let lat = lp.y - (phase + round((lp.y - phase) / period) * period);
    let s = lp.x;

    let glow = city_glow_sample(h.pos.xy, 3.0);
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * glow);

    // A footprint for detail that runs ACROSS the deck rather than along it.
    //
    // `fp` is isotropic — the pixel angle times the range — but a deck is a
    // surface nobody looks at squarely, and at grazing incidence a pixel is
    // long in the along direction and unchanged in the lateral one. The two
    // families of detail here therefore do not share a footprint, and giving
    // them one is the mistake to avoid in both directions: charge the lane
    // lines the grazing factor and the paint dissolves five times too early
    // (their WIDTH is lateral and foreshortens not at all); charge the
    // expansion joints the plain `fp` and a transverse band that is a fifth
    // of a pixel still believes itself resolved, and crawls.
    //
    // So: lateral-extent detail (lane-line widths, the streaks' lateral
    // profile) keeps `fp`; along-extent detail that is thin — the joints —
    // gets this. Clamped at 5x like `city_shade`'s window grid, because at
    // true grazing the exact factor stops meaning anything.
    //
    // Worth knowing when sizing any of these windows: `fp` is NOT the naive
    // pixel footprint. `pixel_angle` is `max(2*tan(fov/2)/width, u.periodic.z)`
    // and the LOD floor wins by 3.6x at 960 px with the app's default slider,
    // so every window below is exercised at 3.6x the range you would guess.
    let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);
    // The deck's top surface here, re-derived from the same profile the
    // tracer used — so heights measured against it (the rail's lamp band, the
    // fascia's depth) hold on the ramps as well as on the flats.
    let ztop = select(cc_skyway_DECK_Z,
                      cc_skyway_ztop(s, round(s / period) * period, true),
                      fam == 1);
    let duty = cc_skyway_STR_L / cc_skyway_STR_P;
    // The dash collapses into a continuous line of the same mean energy —
    // which is what a light string a kilometre off actually is.
    let str_lod = smoothstep(cc_skyway_STR_FP_LO, cc_skyway_STR_FP_HI, fp);
    let dash = select(0.0, 1.0, fract(s / cc_skyway_STR_P) < duty);
    let strength = mix(dash, duty, str_lod);

    // The expansion joint over the nearest pier, as a signed profile: -1 in
    // the slot, +1 on the comb plates, 0 on open deck. One round and two
    // compares, and it is shared by the running surface and the fascia so the
    // two agree on where a segment ends.
    let jd = abs(s - round(s / cc_skyway_PYL_SP) * cc_skyway_PYL_SP);
    let j_raw = select(select(0.0, 1.0, jd < cc_skyway_JNT_PLATE),
                       -1.0, jd < cc_skyway_JNT_GAP);
    // Its area mean over a bay, for the footprint where the assembly stops
    // being resolvable. Written from the same three constants the profile is,
    // so widening the slot cannot drift the two ends of the LOD apart.
    let j_mean = (2.0 * (cc_skyway_JNT_PLATE - cc_skyway_JNT_GAP)
                  - 2.0 * cc_skyway_JNT_GAP) / cc_skyway_PYL_SP;
    let joint = mix(j_raw, j_mean,
                    smoothstep(cc_skyway_JNT_FP_LO, cc_skyway_JNT_FP_HI, fp_eff));
    // Slot eats light, plate adds a little. Both are multipliers on whatever
    // is lighting the surface, so a joint in the dark stays dark.
    let j_mul = 1.0 + cc_skyway_JNT_STEEL * max(joint, 0.0)
                    + cc_skyway_JNT_DARK * min(joint, 0.0);

    if (part == cc_skyway_P_RAIL) {
        var face = 1.0;
        if (h.normal.z < 0.5) {
            let nl = cc_skyway_swap(h.normal, fam);
            let side = select(cc_skyway_RAIL_FACE_OUT,
                              cc_skyway_RAIL_FACE_IN, nl.y * lat < 0.0);
            // Only the top band of the barrier carries the fixture.
            let below = ztop + cc_skyway_RAIL_H - h.pos.z;
            face = side * (1.0 - smoothstep(cc_skyway_RAIL_BAND * 0.5,
                                            cc_skyway_RAIL_BAND, below));
        }
        return cc_skyway_CONCRETE * fill
               + cc_skyway_STR_COLOR * (cc_skyway_STR_RAD * strength * face);
    }

    if (part == cc_skyway_P_DECK) {
        // The wash: the strings' light across the deck, and the reason the
        // network reads as an artery from altitude. Exponential across, and
        // scalloped along under the individual lamps.
        let din = max(cc_skyway_HW - abs(lat), 0.0);
        let q = fract(s / cc_skyway_STR_P - 0.5 * duty);
        let dd = min(q, 1.0 - q) * cc_skyway_STR_P;
        let pool = exp(-dd * dd
                       / (2.0 * cc_skyway_SCALLOP_S * cc_skyway_SCALLOP_S));
        // Between-lamp floor plus pool, handed to the pattern's own mean at
        // the same footprint where the dashes themselves dissolve.
        let pool_mean = 2.5066 * cc_skyway_SCALLOP_S / cc_skyway_STR_P;
        let rhythm = cc_skyway_SCALLOP
            + (1.0 - cc_skyway_SCALLOP) * mix(pool, pool_mean, str_lod);
        let wash = exp(-din / cc_skyway_WASH_E) * rhythm;
        var e = cc_skyway_ASPHALT * fill
                + cc_skyway_STR_COLOR * (cc_skyway_WASH_AMP * wash);

        // Lane paint: a double centre line, dashed lane dividers, solid edge
        // lines. Sub-pixel almost everywhere, so it hands over to its own
        // area mean rather than shimmering.
        let aw = cc_skyway_PAINT_W;
        let al = abs(lat);
        var paint = 0.0;
        paint = paint + select(0.0, 1.0,
                               abs(al - cc_skyway_CENTRE_LINE) < aw);
        paint = paint + select(0.0, 1.0,
                               abs(al - cc_skyway_EDGE_LINE) < aw);
        paint = paint + select(0.0, 1.0,
                               abs(al - cc_skyway_LANE_DIV) < aw
                               && fract(s / cc_skyway_DASH_P)
                                  < cc_skyway_DASH_DUTY);
        let paint_mean = (4.0 * aw + 4.0 * aw
                          + 4.0 * aw * cc_skyway_DASH_DUTY)
                         / (2.0 * cc_skyway_HW);
        let paint_l = mix(paint, paint_mean,
                          smoothstep(cc_skyway_PAINT_FP_LO,
                                     cc_skyway_PAINT_FP_HI, fp));
        e = e + cc_skyway_STR_COLOR
                * (cc_skyway_PAINT_AMP * paint_l * (0.20 + wash));

        // The joint crosses the paint and the wash — a comb plate is bare
        // steel, so the lane line stops at it — but NOT the traffic, which is
        // light thrown onto the deck rather than a property of the deck.
        return e * j_mul + cc_skyway_streaks(s, lat, fam, glow, fp);
    }

    // Underside, fascia, pylons: near-black structure carrying the sodium
    // thrown up at it from the street directly below. The pools are the same
    // ones the ground draws, so an underside's scallops sit exactly over the
    // lamps that make them.
    let pools = city_street_pools(h.pos.xy);
    let district = city_glow_sample(h.pos.xy, 2.0);
    let street_scale = 0.20 + 2.2 * smoothstep(0.02, 0.45, district);
    var up = cc_skyway_UPLIGHT * pools * street_scale;
    if (part == cc_skyway_P_PYL) {
        // Column: brightest at its foot, standing in the pool itself.
        up = up * exp(-max(h.pos.z, 0.0) / cc_skyway_PYL_UP_E);
    } else if (part == cc_skyway_P_CAP) {
        // Crosshead. It sits as high as the soffit but is not the soffit:
        // its own soffit faces the street squarely and catches the same
        // uplight, while its ends and sides are sheer. The 1.35 is the one
        // place this component brightens anything — a projecting beam catches
        // light on three faces where the flat soffit above it catches one,
        // and without it the caps read as holes rather than as hardware.
        up = up * select(0.45, 1.35, h.normal.z < -0.5)
                * exp(-max(h.pos.z, 0.0) / (2.0 * cc_skyway_PYL_UP_E));
    } else if (h.normal.z > -0.5) {
        // Fascia, not soffit: it faces sideways, so it catches much less —
        // and it carries the end of the deck's expansion joint, which is the
        // only thing on this component that says where one span stops and the
        // next starts when you are looking at it from the side.
        up = up * 0.35 * j_mul;
    } else {
        // Soffit: the girders the deck is carried on, as shading rather than
        // as geometry — the read is entirely in the banding. The joint runs
        // across them, and being a real gap it is darker here than the ribs.
        let rib = 1.0 + cc_skyway_RIB
            * select(-1.0, 1.0, fract(s / cc_skyway_RIB_P) < 0.5);
        up = up * mix(rib, 1.0, smoothstep(0.0, cc_skyway_RIB_FP, fp)) * j_mul;
    }
    return cc_skyway_CONCRETE * fill + CITY_LAMP_COLOR * up;
}

// --- component: rooftopworks (rooftopworks.wgsl) ---
// rooftopworks — the machinery on the roofs.
//
// A skyline is not a row of clean rectangles. What separates a real one from
// an extruded-footprint one is the last two metres: water tanks, condenser
// banks, vent stacks, a lit penthouse, a parapet with a strip of light along
// it. From a kilometre out none of it is resolved and all of it matters —
// the roofline goes from a ruled edge to a serrated one, and the eye reads
// "buildings people work in" from the serration alone.
//
// RESPECT THE ARCHITECTURE (SPEC, the rule above all others here). Roof
// furniture needs a flat top to stand on:
//   * arch 0 (setback stack) — the full set, on EVERY tier top. A wedding
//     cake has three roofs, not one, and the lower ones are where the tanks
//     go. The tier standing on a deck is carried as a hole in it, so nothing
//     is ever placed inside the mass above.
//   * arch 2 (growth) — the main roof AND the cantilevered bud tops, which
//     are flat shelves hanging over the street and read beautifully with a
//     tank on them.
//   * arch 3 (tapered shaft) — the crown cap ONLY: the footprint scaled by
//     cc.fscale about the frustum's centre. The podium ledge is a metre of
//     sloped shoulder and gets nothing.
//   * arch 4 (spire crown) — NOTHING. Not one box, and the component returns
//     before it draws a hash.
//
// WHAT IS HERE, in the order it matters to the shot:
//
//   1. MECHANICAL CLUSTERS. 2-4 dark boxes per eligible roof: square water
//      tanks (3 x 3 x 4 m, with the rim band that makes them read as tanks
//      and not as crates), condenser rows (three 2 x 2 x 1 m units in a
//      line, sometimes doubled), vent stacks (1 x 1 x 5 m, in pairs). Dark:
//      their whole job at distance is silhouette. ~30% carry one warm
//      service lamp — a bulkhead light on the housing, radiance 2 — which is
//      what puts a few sparks up on the roofs above the window ladder.
//   2. PARAPET. A real low wall around the top deck of ~55% of buildings
//      (the core only fakes one by brightening the roof albedo near the
//      edge), and on ~12% of roofs overall a strip of light along its
//      coping: cool white, or the building's own house colour where it has
//      one. The wall is a hollow box — two slab tests, not four — and it is
//      deliberately NOT clipped to the DDA cell, because it belongs to the
//      building rather than to the cell: every member cell of a merged
//      superblock derives the same ring and catches whatever part of it
//      falls in its own ray segment.
//   3. PENTHOUSE CROWNS. ~20% of tall flat-topped buildings get a 2-3 storey
//      inset glass box on the roof, lit warm at 1.8. On a merged superblock
//      two of the four quadrant cells may carry one, so a big roof reads as
//      a plant room and a penthouse rather than as one centred lump.
//   4. HELIPADS. On the biggest flat roofs only (a merged superblock, or a
//      footprint over 60 m): a pad slab 0.3 m proud with four warm corner
//      lights at radiance 3. Rare on purpose — a helipad on every tower is
//      a video-game city.
//   5. EXTERNAL PIPEWORK. 1-3 risers, 0.4 m across, running down a facade
//      corner of 30-150 m buildings — the mid-rise band where services are
//      bolted on the outside rather than buried in a core. Dark with a faint
//      rim, gated to close range: it is a texture on the wall, not a
//      silhouette, and past ~230 m there is nothing there to resolve.
//
// DETAIL CALIBRATION (SPEC, global): wiper blades, not tire brands. A tank
// gets its rim band and its legs; a condenser row gets its unit gaps and
// grille bands. Nothing gets a logo, a warning label, or a hatch handle.
//
// COST. Every cell pays: two compares (built, arch 4) and one z-interval
// test against [lowest deck, roof + 12 m]. A cell whose ray segment misses
// that band — most of them, since most city pixels are looking at a wall or
// at the road — costs exactly that. Inside the band each piece has its own
// tighter z window and its own eligibility test BEFORE it draws a hash: the
// parapet's 1.25 m band, the clutter's 5 m one, and a crown that never
// hashes on a building too short to carry one. Worst case is 4 slots x 2
// boxes + 2 parapet + 1 crown = 11 slab tests, and that only on a large roof
// filling the segment; the typical accepted cell runs 3-4. Pipework adds one
// box behind a footprint gate that keeps it inside ~230 m.
//
// Measured on the harness (RTX 5080, CONTENDED — other agents and the pilot
// share the card, so these are upper bounds), 960x540 x 48 passes, A/B by
// toggling `enabled` and interleaving the runs: a downward view over a
// district of roofs costs 0.16 s off / 0.18 s on, and the 900 m `base` view
// 0.28 s off / 0.33 s on — call it 12-18% of the city frame where roofs fill
// the screen. A street-level view, where the roof band is behind the walls,
// measures 0.13 s either way. The parapet is the widest-reaching piece: two
// slab tests on nearly every roof cell a downward ray visits, and that IS
// where the money goes. It buys a wall that is actually there.

// --- kinds (kind_base 600, local 0..15) -------------------------------------
// 600 + slot  mechanical body        604 + slot  its accessory piece
const cc_rooftopworks_K_MECH: i32 = 600;
const cc_rooftopworks_K_ACC: i32 = 604;
const cc_rooftopworks_K_PARAPET: i32 = 608;
const cc_rooftopworks_K_PENT: i32 = 609;
const cc_rooftopworks_K_PAD: i32 = 611;
const cc_rooftopworks_K_PIPE: i32 = 612;

// --- mechanical clusters ----------------------------------------------------
const cc_rooftopworks_MECH_FRAC: f32 = 0.86;   // buildings with any clutter
// Slot occupancy. Slot 0 is unconditional on a deck that fits it; the rest
// thin out, so the modal roof carries two or three objects and a big one
// four. Confetti is the failure mode being avoided here: a roof with a
// uniform sprinkle of boxes reads as noise, a roof with a tank, a condenser
// row and a gap reads as a roof.
const cc_rooftopworks_SLOT_P1: f32 = 0.88;
const cc_rooftopworks_SLOT_P2: f32 = 0.62;
const cc_rooftopworks_SLOT_P3: f32 = 0.34;
const cc_rooftopworks_TANK_CUT: f32 = 0.38;    // type draw: tank below this
const cc_rooftopworks_AC_CUT: f32 = 0.76;      // condenser row below this
const cc_rooftopworks_TANK_HW: f32 = 1.50;     // 3 x 3 m
const cc_rooftopworks_TANK_H: f32 = 4.00;
const cc_rooftopworks_TANK_RIM_HW: f32 = 1.78;
const cc_rooftopworks_TANK_RIM_T: f32 = 0.38;
const cc_rooftopworks_AC_HL: f32 = 3.60;       // three 2 m units, 2.6 pitch
const cc_rooftopworks_AC_HW: f32 = 1.00;
const cc_rooftopworks_AC_H: f32 = 1.00;
const cc_rooftopworks_AC_ROW: f32 = 3.10;      // second row offset
const cc_rooftopworks_AC_P2: f32 = 0.45;       // draw above this: doubled row
const cc_rooftopworks_VENT_HW: f32 = 0.50;
const cc_rooftopworks_VENT_H: f32 = 5.00;
const cc_rooftopworks_VENT2_H: f32 = 3.40;
const cc_rooftopworks_VENT_GAP: f32 = 1.70;
const cc_rooftopworks_EDGE_M: f32 = 1.10;      // keep off the parapet line
// Close-range refinement (SPEC's SDF-in-a-box). A tank is the one object here
// the camera can end up standing next to, and a bare box is exactly what a
// bare box looks like. Inside its envelope, and only inside it, the tank is
// sphere-traced: a rounded tub on four legs with a chamfered rim band. The
// rounding is the point — rays graze past the corners and the silhouette
// stops being a rectangle. 42 m at the harness lens, so the wide scene pays
// one box test and nothing else.
const cc_rooftopworks_SDF_FP: f32 = 0.22;
const cc_rooftopworks_SDF_ITERS: i32 = 24;
const cc_rooftopworks_TANK_LEG: f32 = 0.55;    // stand-off from the deck
const cc_rooftopworks_TANK_R: f32 = 0.20;      // corner rounding
const cc_rooftopworks_LAMP_FRAC: f32 = 0.30;
const cc_rooftopworks_LAMP_RAD: f32 = 2.0;
const cc_rooftopworks_LAMP_SIG: f32 = 0.30;    // source radius (m)
const cc_rooftopworks_LAMP_SPILL: f32 = 1.70;  // what the housing catches (m)
const cc_rooftopworks_LAMP_COL: vec3<f32> = vec3<f32>(1.00, 0.58, 0.24);
const cc_rooftopworks_MECH_ALBEDO: f32 = 0.140;

// --- parapet ----------------------------------------------------------------
const cc_rooftopworks_PAR_FRAC: f32 = 0.55;    // roofs with a real wall
const cc_rooftopworks_PAR_LIT: f32 = 0.21;     // ...of which lit (~12% of roofs)
const cc_rooftopworks_PAR_H: f32 = 1.25;
const cc_rooftopworks_PAR_W: f32 = 0.38;
const cc_rooftopworks_PAR_TRIM: f32 = 0.28;    // lit band under the coping
const cc_rooftopworks_TRIM_RAD: f32 = 1.20;
const cc_rooftopworks_TRIM_COOL: vec3<f32> = vec3<f32>(0.72, 0.86, 1.00);
const cc_rooftopworks_TRIM_PITCH: f32 = 6.50;  // fixture centres (m)
const cc_rooftopworks_TRIM_DUTY: f32 = 0.82;   // lit fraction of the run
// The trim is a 0.28 m band on a 1.25 m wall: past this footprint the band is
// thinner than a pixel and hands over to its mean over the whole wall face,
// which is the same energy spread out. Nothing switches off (SPEC rule 3).
const cc_rooftopworks_TRIM_LOD_START: f32 = 1.60;
const cc_rooftopworks_TRIM_LOD_FULL: f32 = 4.00;
const cc_rooftopworks_TRIM_MEAN_COMP: f32 = 0.20;  // see PENT_MEAN_COMP

// --- penthouse --------------------------------------------------------------
const cc_rooftopworks_PENT_FRAC: f32 = 0.20;
const cc_rooftopworks_PENT_MIN_H: f32 = 105.0; // "tall": tower, not mid-rise
const cc_rooftopworks_PENT_MIN_DECK: f32 = 15.0;
const cc_rooftopworks_PENT_INSET: f32 = 0.28;
const cc_rooftopworks_PENT_RAD: f32 = 1.80;
const cc_rooftopworks_PENT_COL: vec3<f32> = vec3<f32>(1.00, 0.72, 0.42);
const cc_rooftopworks_PENT_PITCH: f32 = 2.10;  // mullion pitch (m)
const cc_rooftopworks_PENT_MULL: f32 = 0.13;   // dark fraction of the pitch
const cc_rooftopworks_PENT_SILL: f32 = 0.80;   // dark spandrel at its foot
const cc_rooftopworks_PENT_LIT: f32 = 0.58;    // bays with the light on
// Tone-map compensation on the far-field means, the same correction the
// core applies to its own window octaves (CITY_MEAN_COMP_BLOCK/_FLAT): a
// mean taken in linear radiance ahead of a Reinhard curve renders CREAM,
// because a lit bay at 1.8 is already near the knee and its average is not.
// Solved rather than guessed: match tone(mean * E) to lit_frac * tone(B * E)
// at E = 6, which puts the coefficient near a quarter.
const cc_rooftopworks_PENT_MEAN_COMP: f32 = 0.26;

// --- helipad ----------------------------------------------------------------
const cc_rooftopworks_PAD_FRAC: f32 = 0.13;
const cc_rooftopworks_PAD_BIG: f32 = 60.0;     // footprint that qualifies
const cc_rooftopworks_PAD_MIN_H: f32 = 55.0;   // and a building worth landing on
const cc_rooftopworks_PAD_T: f32 = 0.30;
const cc_rooftopworks_PAD_LAMP_RAD: f32 = 3.00;
const cc_rooftopworks_PAD_LAMP_SIG: f32 = 0.26;
const cc_rooftopworks_PAD_LAMP_COL: vec3<f32> = vec3<f32>(1.00, 0.66, 0.30);
const cc_rooftopworks_PAD_MARK: f32 = 0.34;    // paint albedo (a ring + bar)

// --- pipework ---------------------------------------------------------------
const cc_rooftopworks_PIPE_FRAC: f32 = 0.45;
const cc_rooftopworks_PIPE_H_LO: f32 = 30.0;
const cc_rooftopworks_PIPE_H_HI: f32 = 150.0;
const cc_rooftopworks_PIPE_D: f32 = 0.40;      // one riser
const cc_rooftopworks_PIPE_GAP: f32 = 0.18;
const cc_rooftopworks_PIPE_OUT: f32 = 0.42;    // stand-off from the wall
const cc_rooftopworks_PIPE_CORNER: f32 = 1.20; // in from the building corner
// Footprint gate: a 0.4 m riser at 1.5 m/px is a quarter pixel of dark line
// on a lit wall and contributes nothing but cost. The emission and the
// contrast ramp to the wall's own value over the run-up, so it dissolves
// rather than pops.
const cc_rooftopworks_PIPE_FP: f32 = 1.20;
const cc_rooftopworks_PIPE_FADE: f32 = 0.75;

// Headroom above the main roof that everything here fits under: a 3-storey
// penthouse is the tallest thing (10.8 m).
const cc_rooftopworks_HEADROOM: f32 = 12.0;

fn cc_rooftopworks_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

// Nearest-hit accumulator over one box. Every piece of geometry in this file
// except the parapet ring goes through here, which is also where the segment
// discipline lives: a hit outside [t0, t1] belongs to another cell's visit
// and must not be returned, because the core's DDA stops the moment a cell
// reports anything.
fn cc_rooftopworks_box(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                       inv_dir: vec3<f32>, t0: f32, t1: f32,
                       bmin: vec3<f32>, bmax: vec3<f32>, kind: i32,
                       ci: vec2<i32>) -> CityHit {
    let s = city_box_hit(o, inv_dir, bmin, bmax);
    if (s.x <= s.y && s.x > max(t0, 1e-3) && s.x <= t1 && s.x < best.t) {
        let p = o + s.x * dir;
        return CityHit(true, s.x, p, city_box_normal(p, bmin, bmax), kind,
                       ci);
    }
    return best;
}

fn cc_rooftopworks_overlap(alo: vec2<f32>, ahi: vec2<f32>,
                           blo: vec2<f32>, bhi: vec2<f32>) -> bool {
    return alo.x < bhi.x && ahi.x > blo.x && alo.y < bhi.y && ahi.y > blo.y;
}

// The DDA cell's own column in world xy. Per-cell-hashed props are placed
// inside it so that the cell that draws a thing is the cell that contains it
// — which is what the DDA requires, and what gives a merged superblock four
// independently furnished quadrants instead of one roof drawn four times.
fn cc_rooftopworks_column(ci: vec2<i32>) -> vec4<f32> {
    let cell = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cell;
    return vec4<f32>(cmin, cmin + vec2<f32>(cell, cell));
}

// --- decks ------------------------------------------------------------------
// A flat top you may stand something on: its rect, its height, and the hole
// punched in it by whatever tier stands on it.
struct cc_rooftopworks_Deck {
    ok: bool,
    lo: vec2<f32>,
    hi: vec2<f32>,
    z: f32,
    hlo: vec2<f32>,
    hhi: vec2<f32>,
}

fn cc_rooftopworks_no_deck() -> cc_rooftopworks_Deck {
    return cc_rooftopworks_Deck(false, vec2<f32>(0.0), vec2<f32>(0.0), 0.0,
                                vec2<f32>(1e30), vec2<f32>(-1e30));
}

fn cc_rooftopworks_mk_deck(lo: vec2<f32>, hi: vec2<f32>, z: f32,
                           hlo: vec2<f32>, hhi: vec2<f32>)
        -> cc_rooftopworks_Deck {
    return cc_rooftopworks_Deck(true, lo, hi, z, hlo, hhi);
}

// Deck `i` of this building, index 0 being the topmost. This function IS the
// architecture rule; everything else in the file only asks it for a rect.
fn cc_rooftopworks_deck(cc: CityCell, i: i32) -> cc_rooftopworks_Deck {
    let nohole_lo = vec2<f32>(1e30);
    let nohole_hi = vec2<f32>(-1e30);
    if (cc.arch == 4) {
        return cc_rooftopworks_no_deck();   // a spire has no roof. Ever.
    }
    if (cc.arch == 3) {
        // The crown cap of the tapered shaft: the frustum's cross-section at
        // its top, which is the base rect scaled by fscale about its centre.
        if (i != 0) {
            return cc_rooftopworks_no_deck();
        }
        let c = 0.5 * (cc.fmin.xy + cc.fmax.xy);
        let hf = 0.5 * (cc.fmax.xy - cc.fmin.xy) * cc.fscale;
        return cc_rooftopworks_mk_deck(c - hf, c + hf, cc.fmax.z,
                                       nohole_lo, nohole_hi);
    }
    if (cc.arch == 2) {
        // Growth: the main roof, plus each cantilevered bud's top. The buds
        // overlap the main mass by a metre at their root and punch no hole
        // in it (they hang below the roof), so no holes here either.
        if (i == 0) {
            return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy,
                                           cc.b1max.z, nohole_lo, nohole_hi);
        }
        if (i == 1 && cc.tiers >= 2) {
            return cc_rooftopworks_mk_deck(cc.b2min.xy, cc.b2max.xy,
                                           cc.b2max.z, nohole_lo, nohole_hi);
        }
        if (i == 2 && cc.tiers >= 3) {
            return cc_rooftopworks_mk_deck(cc.b3min.xy, cc.b3max.xy,
                                           cc.b3max.z, nohole_lo, nohole_hi);
        }
        return cc_rooftopworks_no_deck();
    }
    // Setback stack. Deck 0 is the summit; the ones below it are annular,
    // and carry the tier above as their hole.
    if (cc.tiers == 1) {
        if (i == 0) {
            return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy,
                                           cc.b1max.z, nohole_lo, nohole_hi);
        }
        return cc_rooftopworks_no_deck();
    }
    if (cc.tiers == 2) {
        if (i == 0) {
            return cc_rooftopworks_mk_deck(cc.b2min.xy, cc.b2max.xy,
                                           cc.b2max.z, nohole_lo, nohole_hi);
        }
        if (i == 1) {
            return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy,
                                           cc.b1max.z, cc.b2min.xy,
                                           cc.b2max.xy);
        }
        return cc_rooftopworks_no_deck();
    }
    if (i == 0) {
        return cc_rooftopworks_mk_deck(cc.b3min.xy, cc.b3max.xy, cc.b3max.z,
                                       nohole_lo, nohole_hi);
    }
    if (i == 1) {
        return cc_rooftopworks_mk_deck(cc.b2min.xy, cc.b2max.xy, cc.b2max.z,
                                       cc.b3min.xy, cc.b3max.xy);
    }
    if (i == 2) {
        return cc_rooftopworks_mk_deck(cc.b1min.xy, cc.b1max.xy, cc.b1max.z,
                                       cc.b2min.xy, cc.b2max.xy);
    }
    return cc_rooftopworks_no_deck();
}

// The lowest deck this building owns: the bottom of the z band the whole
// roof section gates on. For a setback stack that is the first tier's top,
// for a growth tower the lower bud, otherwise the roof itself.
fn cc_rooftopworks_deck_low(cc: CityCell) -> f32 {
    var z = cc.height;
    if (cc.arch != 3 && cc.tiers >= 2) {
        z = min(z, min(cc.b1max.z, cc.b2max.z));
        if (cc.tiers >= 3) {
            z = min(z, cc.b3max.z);
        }
    }
    return z;
}

// --- the crown: a penthouse, or a helipad, or neither -----------------------
struct cc_rooftopworks_Crown {
    kind: i32,   // 0 none, 1 penthouse, 2 helipad
    lo: vec2<f32>,
    hi: vec2<f32>,
    z0: f32,
    z1: f32,
}

fn cc_rooftopworks_crown(cc: CityCell, ci: vec2<i32>)
        -> cc_rooftopworks_Crown {
    var res: cc_rooftopworks_Crown;
    res.kind = 0;
    res.lo = vec2<f32>(1e30);
    res.hi = vec2<f32>(-1e30);
    res.z0 = 0.0;
    res.z1 = 0.0;
    let d = cc_rooftopworks_deck(cc, 0);
    if (!d.ok) {
        return res;
    }
    let col = cc_rooftopworks_column(ci);
    let rlo = max(d.lo, col.xy);
    let rhi = min(d.hi, col.zw);
    let size = rhi - rlo;
    if (min(size.x, size.y) < cc_rooftopworks_PENT_MIN_DECK) {
        return res;
    }
    // Eligibility before entropy: a building too short for either a
    // penthouse or a pad never draws a hash. This is the gate that keeps the
    // low-rise majority of the tile out of the crown's cost entirely.
    let foot0 = max(d.hi.x - d.lo.x, d.hi.y - d.lo.y);
    let pad_size = cc.merged || foot0 > cc_rooftopworks_PAD_BIG;
    if (cc.height <= cc_rooftopworks_PENT_MIN_H
        && !(pad_size && cc.height > cc_rooftopworks_PAD_MIN_H)) {
        return res;
    }
    // Building-level draws (shared by every cell of a merged group) decide
    // WHETHER; a cell-level draw decides the shape, so the two penthouses on
    // a superblock are not identical twins.
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x1b56c4e9u, 0x0d2f1a37u));
    let lh = city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x9e3779b9u + 0x2545f491u,
        bitcast<u32>(ci.y) * 0x85ebca6bu + 0x27d4eb2fu));
    // Which quadrant of a superblock this cell is.
    let q = (ci.x & 1) + 2 * (ci.y & 1);

    // Helipad first: it wants the same real estate and it is the rarer thing.
    let pad_q = i32(gh.z * 4.0);
    let pad_here = !cc.merged || q == pad_q;
    if (pad_size && pad_here
        && cc.height > cc_rooftopworks_PAD_MIN_H
        && gh.x < cc_rooftopworks_PAD_FRAC) {
        let half = clamp(0.30 * min(size.x, size.y), 4.5, 11.0);
        let c = mix(rlo + half + 1.0, rhi - half - 1.0,
                    vec2<f32>(0.30 + 0.40 * lh.x, 0.30 + 0.40 * lh.y));
        let plo = c - half;
        let phi = c + half;
        // Not on top of the mast, and not hanging over the hole.
        let clear_mast = !cc.has_mast
            || !cc_rooftopworks_overlap(plo, phi, cc.mast_min.xy - 1.0,
                                        cc.mast_max.xy + 1.0);
        if (clear_mast && !cc_rooftopworks_overlap(plo, phi, d.hlo, d.hhi)) {
            res.kind = 2;
            res.lo = plo;
            res.hi = phi;
            res.z0 = d.z;
            res.z1 = d.z + cc_rooftopworks_PAD_T;
            return res;
        }
    }
    // Penthouse. On a superblock only the two diagonal quadrants may carry
    // one, so a big roof gets at most two and they sit apart.
    let pent_here = !cc.merged || ((ci.x + ci.y) & 1) == 0;
    if (cc.height > cc_rooftopworks_PENT_MIN_H && pent_here
        && gh.y < cc_rooftopworks_PENT_FRAC) {
        let inset = max(cc_rooftopworks_PENT_INSET * min(size.x, size.y), 3.0);
        let plo = rlo + inset;
        let phi = rhi - inset;
        if (min(phi.x - plo.x, phi.y - plo.y) > 6.0
            && !cc_rooftopworks_overlap(plo, phi, d.hlo, d.hhi)) {
            res.kind = 1;
            res.lo = plo;
            res.hi = phi;
            res.z0 = d.z;
            res.z1 = d.z + select(7.2, 10.8, lh.z > 0.45);
            return res;
        }
    }
    return res;
}

// --- mechanical clusters ----------------------------------------------------
struct cc_rooftopworks_Mech {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    has_acc: bool,
    alo: vec3<f32>,
    ahi: vec3<f32>,
    kind: i32,         // 0 tank, 1 condenser row, 2 vent stacks
    lamp: bool,
    seat: vec3<f32>,   // the service lamp's own position
}

fn cc_rooftopworks_no_mech() -> cc_rooftopworks_Mech {
    return cc_rooftopworks_Mech(false, vec3<f32>(0.0), vec3<f32>(0.0), false,
                                vec3<f32>(0.0), vec3<f32>(0.0), 0, false,
                                vec3<f32>(0.0));
}

// Does this building carry rooftop clutter at all? One building-level draw,
// hoisted out of the slot loop so a cell pays it once rather than four times.
fn cc_rooftopworks_mech_on(cc: CityCell) -> bool {
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x3c6ef372u, 0x165667b1u));
    return gh.w < cc_rooftopworks_MECH_FRAC;
}

fn cc_rooftopworks_slot_draw(ci: vec2<i32>, s: i32) -> vec4<f32> {
    let l = u32(s);
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x85ebca6bu + l * 0x9e3779b9u + 0x7feb352du,
        bitcast<u32>(ci.y) * 0xc2b2ae35u + l * 0x51ed270bu + 0x846ca68bu));
}

// Slot `s` of cell `ci`: which deck it stands on, where, and what it is.
// Deterministic in (cc, ci, s), so the shader re-derives the object from the
// hit kind alone — nothing is smuggled through CityHit.
fn cc_rooftopworks_mech(cc: CityCell, ci: vec2<i32>, s: i32,
                        mech_on: bool, cr: cc_rooftopworks_Crown)
        -> cc_rooftopworks_Mech {
    if (!mech_on) {
        return cc_rooftopworks_no_mech();
    }
    let r = cc_rooftopworks_slot_draw(ci, s);
    var p_slot = 1.0;
    if (s == 1) { p_slot = cc_rooftopworks_SLOT_P1; }
    if (s == 2) { p_slot = cc_rooftopworks_SLOT_P2; }
    if (s == 3) { p_slot = cc_rooftopworks_SLOT_P3; }
    if (r.x >= p_slot) {
        return cc_rooftopworks_no_mech();
    }
    // Slots 0 and 1 take the summit; 2 and 3 look for a lower tier top or a
    // bud, and fall back to the summit when the building has none.
    var di = 0;
    if (s == 2) { di = 1; }
    if (s == 3) { di = 2; }
    var d = cc_rooftopworks_deck(cc, di);
    if (!d.ok) {
        d = cc_rooftopworks_deck(cc, 0);
    }
    if (!d.ok) {
        return cc_rooftopworks_no_mech();
    }

    // Type and footprint.
    var hx = vec2<f32>(cc_rooftopworks_TANK_HW);
    var hgt = cc_rooftopworks_TANK_H;
    var ty = 0;
    let axis_y = r.w > 0.5;
    if (r.y >= cc_rooftopworks_TANK_CUT && r.y < cc_rooftopworks_AC_CUT) {
        ty = 1;
        let along = cc_rooftopworks_AC_HL;
        let across = select(cc_rooftopworks_AC_HW,
                            cc_rooftopworks_AC_HW + 0.5 * cc_rooftopworks_AC_ROW,
                            r.z > cc_rooftopworks_AC_P2);
        hx = select(vec2<f32>(along, across), vec2<f32>(across, along),
                    axis_y);
        hgt = cc_rooftopworks_AC_H;
    } else if (r.y >= cc_rooftopworks_AC_CUT) {
        ty = 2;
        let along = cc_rooftopworks_VENT_HW + 0.5 * cc_rooftopworks_VENT_GAP;
        hx = select(vec2<f32>(along, cc_rooftopworks_VENT_HW),
                    vec2<f32>(cc_rooftopworks_VENT_HW, along), axis_y);
        hgt = cc_rooftopworks_VENT_H;
    }

    // Where it may stand: the deck, inset off the parapet line, clipped to
    // this cell's own column.
    let col = cc_rooftopworks_column(ci);
    let m = hx + cc_rooftopworks_EDGE_M + cc_rooftopworks_PAR_W;
    let plo = max(d.lo + m, col.xy + m - cc_rooftopworks_EDGE_M);
    let phi = min(d.hi - m, col.zw - m + cc_rooftopworks_EDGE_M);
    if (plo.x > phi.x || plo.y > phi.y) {
        return cc_rooftopworks_no_mech();   // deck too small for this object
    }
    let c = mix(plo, phi, vec2<f32>(fract(r.z * 13.7), fract(r.w * 7.31)));
    let lo2 = c - hx;
    let hi2 = c + hx;
    // Three exclusions, all cheap rect tests: the tier standing on this deck,
    // the mast, and whatever the crown put here.
    if (cc_rooftopworks_overlap(lo2, hi2, d.hlo - 1.0, d.hhi + 1.0)) {
        return cc_rooftopworks_no_mech();
    }
    if (cc.has_mast && abs(d.z - cc.height) < 0.5
        && cc_rooftopworks_overlap(lo2, hi2, cc.mast_min.xy - 0.8,
                                   cc.mast_max.xy + 0.8)) {
        return cc_rooftopworks_no_mech();
    }
    if (cr.kind != 0 && abs(d.z - cc.height) < 0.5
        && cc_rooftopworks_overlap(lo2, hi2, cr.lo - 1.5, cr.hi + 1.5)) {
        return cc_rooftopworks_no_mech();
    }

    var res: cc_rooftopworks_Mech;
    res.ok = true;
    res.lo = vec3<f32>(lo2, d.z);
    res.hi = vec3<f32>(hi2, d.z + hgt);
    res.kind = ty;
    res.has_acc = false;
    res.alo = vec3<f32>(0.0);
    res.ahi = vec3<f32>(0.0);
    if (ty == 0) {
        // The rim band: a tank without one is a crate.
        res.has_acc = true;
        let rw = vec2<f32>(cc_rooftopworks_TANK_RIM_HW);
        res.alo = vec3<f32>(c - rw,
                            d.z + hgt - cc_rooftopworks_TANK_RIM_T);
        res.ahi = vec3<f32>(c + rw, d.z + hgt + 0.10);
    } else if (ty == 2) {
        // A second, shorter stack beside the first.
        res.has_acc = true;
        let off = select(vec2<f32>(cc_rooftopworks_VENT_GAP, 0.0),
                         vec2<f32>(0.0, cc_rooftopworks_VENT_GAP), axis_y);
        res.alo = vec3<f32>(c + off - cc_rooftopworks_VENT_HW, d.z);
        res.ahi = vec3<f32>(c + off + cc_rooftopworks_VENT_HW,
                            d.z + cc_rooftopworks_VENT2_H);
        res.lo = vec3<f32>(c - off - cc_rooftopworks_VENT_HW, d.z);
        res.hi = vec3<f32>(c - off + cc_rooftopworks_VENT_HW, d.z + hgt);
    }
    // One warm bulkhead light on the housing, on the face the draw picks.
    res.lamp = fract(r.x * 31.7) < cc_rooftopworks_LAMP_FRAC;
    let face = fract(r.y * 17.3);
    var n = vec2<f32>(1.0, 0.0);
    if (face > 0.75) { n = vec2<f32>(-1.0, 0.0); }
    else if (face > 0.50) { n = vec2<f32>(0.0, 1.0); }
    else if (face > 0.25) { n = vec2<f32>(0.0, -1.0); }
    res.seat = vec3<f32>(c + n * (hx + 0.04), d.z + 0.72 * hgt);
    return res;
}

// --- the tank, close up: SDF in a box ---------------------------------------
fn cc_rooftopworks_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0)))
         + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// The tank in its own frame: origin at the centre of its footprint, z from
// the deck it stands on. Tub, rim band, four legs — the functional features
// and nothing below them (SPEC: wiper blades, not tire brands).
fn cc_rooftopworks_tank_sdf(p: vec3<f32>) -> f32 {
    let hw = cc_rooftopworks_TANK_HW;
    let hh = cc_rooftopworks_TANK_H;
    let leg = cc_rooftopworks_TANK_LEG;
    let r = cc_rooftopworks_TANK_R;
    let bz1 = hh - cc_rooftopworks_TANK_RIM_T;
    let body = cc_rooftopworks_rbox(
        p - vec3<f32>(0.0, 0.0, 0.5 * (leg + bz1)),
        vec3<f32>(hw - r, hw - r, max(0.5 * (bz1 - leg) - r, 0.05)), r);
    let rim = cc_rooftopworks_rbox(
        p - vec3<f32>(0.0, 0.0, hh - 0.5 * cc_rooftopworks_TANK_RIM_T),
        vec3<f32>(cc_rooftopworks_TANK_RIM_HW - 0.10,
                  cc_rooftopworks_TANK_RIM_HW - 0.10,
                  0.5 * cc_rooftopworks_TANK_RIM_T - 0.04), 0.07);
    // Four legs, folded into one quadrant by the absolute value.
    let legs = cc_rooftopworks_rbox(
        vec3<f32>(abs(p.xy) - vec2<f32>(hw - 0.32), p.z - 0.5 * leg),
        vec3<f32>(0.13, 0.13, 0.5 * leg), 0.04);
    return min(min(body, rim), legs);
}

fn cc_rooftopworks_tank_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.2e-3, -1.2e-3);
    return normalize(
        e.xyy * cc_rooftopworks_tank_sdf(p + e.xyy)
      + e.yyx * cc_rooftopworks_tank_sdf(p + e.yyx)
      + e.yxy * cc_rooftopworks_tank_sdf(p + e.yxy)
      + e.xxx * cc_rooftopworks_tank_sdf(p + e.xxx));
}

// Sphere-trace the tank between the entry and exit of its envelope box.
// Returns the hit t, or -1 when the ray threads past the rounded hull —
// which is what makes the hull read as rounded.
fn cc_rooftopworks_tank_trace(o: vec3<f32>, dir: vec3<f32>, t_in: f32,
                              t_out: f32, base: vec3<f32>) -> f32 {
    var t = max(t_in, 0.0) + 1.0e-3;
    for (var i: i32 = 0; i < cc_rooftopworks_SDF_ITERS; i = i + 1) {
        if (t > t_out) {
            return -1.0;
        }
        let d = cc_rooftopworks_tank_sdf(o + t * dir - base);
        if (d < 2.0e-3) {
            return t;
        }
        t = t + max(d, 3.0e-3);
    }
    return -1.0;
}

// One mechanical object against the segment: two slab tests far out, the
// sphere-traced hull for a tank close in.
fn cc_rooftopworks_mech_hit(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                            inv_dir: vec3<f32>, t0: f32, t1: f32,
                            m: cc_rooftopworks_Mech, s: i32, ci: vec2<i32>,
                            fp: f32) -> CityHit {
    if (m.kind == 0 && fp < cc_rooftopworks_SDF_FP) {
        let elo = min(m.lo, m.alo);
        let ehi = max(m.hi, m.ahi);
        let sb = city_box_hit(o, inv_dir, elo, ehi);
        if (sb.x > sb.y || sb.y <= max(t0, 1e-3) || sb.x > t1) {
            return best;
        }
        let base = vec3<f32>(0.5 * (m.lo.xy + m.hi.xy), m.lo.z);
        let t = cc_rooftopworks_tank_trace(o, dir, max(sb.x, t0),
                                           min(sb.y, t1), base);
        if (t < 0.0 || t > t1 || t <= max(t0, 1e-3) || t >= best.t) {
            return best;
        }
        let p = o + t * dir;
        // The rim band keeps its own kind, so the shader still gives it the
        // galvanised albedo it had when it was a separate box.
        let is_rim = p.z > m.hi.z - cc_rooftopworks_TANK_RIM_T - 0.02;
        return CityHit(true, t, p,
                       cc_rooftopworks_tank_normal(p - base),
                       select(cc_rooftopworks_K_MECH, cc_rooftopworks_K_ACC,
                              is_rim) + s, ci);
    }
    var r = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1, m.lo, m.hi,
                                cc_rooftopworks_K_MECH + s, ci);
    if (m.has_acc) {
        r = cc_rooftopworks_box(r, o, dir, inv_dir, t0, t1, m.alo, m.ahi,
                                cc_rooftopworks_K_ACC + s, ci);
    }
    return r;
}

// --- the parapet ring -------------------------------------------------------
// A hollow box: outer minus inner, two slab tests. Both rects are the
// BUILDING's, not the cell's — see the header note — and correctness comes
// from accepting only a hit inside this visit's ray segment.
fn cc_rooftopworks_has_parapet(cc: CityCell) -> vec2<f32> {
    // .x: does the wall exist. .y: is its coping lit.
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x9e3779b9u, 0x68e31da4u));
    return vec2<f32>(gh.x, gh.y);
}

fn cc_rooftopworks_parapet(best: CityHit, o: vec3<f32>, dir: vec3<f32>,
                           inv_dir: vec3<f32>, t0: f32, t1: f32,
                           d: cc_rooftopworks_Deck, ci: vec2<i32>)
        -> CityHit {
    let omin = vec3<f32>(d.lo, d.z);
    let omax = vec3<f32>(d.hi, d.z + cc_rooftopworks_PAR_H);
    let so = city_box_hit(o, inv_dir, omin, omax);
    if (so.x > so.y || so.y <= max(t0, 1e-3) || so.x > t1) {
        return best;
    }
    let w = cc_rooftopworks_PAR_W;
    let imin = vec3<f32>(d.lo + w, d.z - 1.0);
    let imax = vec3<f32>(d.hi - w, d.z + cc_rooftopworks_PAR_H + 1.0);
    let si = city_box_hit(o, inv_dir, imin, imax);
    let inner = si.x <= si.y;
    // The ring along the ray is [so.x, so.y] with (si.x, si.y) removed. Its
    // two possible front surfaces are the outer entry and the inner exit.
    let lo_t = max(t0, 1e-3);
    var bt = 1e30;
    var on_inner = false;
    if (!(inner && si.x < so.x && si.y > so.x) && so.x > lo_t && so.x <= t1) {
        bt = so.x;
    }
    if (inner && si.y > so.x && si.y < so.y && si.y > lo_t && si.y <= t1
        && si.y < bt) {
        bt = si.y;
        on_inner = true;
    }
    if (bt >= 1e30 || bt >= best.t) {
        return best;
    }
    let p = o + bt * dir;
    let n = select(city_box_normal(p, omin, omax),
                   -city_box_normal(p, imin, imax), on_inner);
    return CityHit(true, bt, p, n, cc_rooftopworks_K_PARAPET, ci);
}

// --- external pipework ------------------------------------------------------
struct cc_rooftopworks_Pipes {
    ok: bool,
    lo: vec3<f32>,
    hi: vec3<f32>,
    n: i32,
    axis_y: bool,   // the bundle runs along the y wall
}

fn cc_rooftopworks_pipes(cc: CityCell) -> cc_rooftopworks_Pipes {
    var res: cc_rooftopworks_Pipes;
    res.ok = false;
    res.lo = vec3<f32>(0.0);
    res.hi = vec3<f32>(0.0);
    res.n = 1;
    res.axis_y = false;
    // Mid-rise only, and only on the two archetypes whose base mass is a
    // plain vertical box: a riser cannot follow a sloped frustum wall.
    if (cc.arch != 0 && cc.arch != 2) {
        return res;
    }
    if (cc.height < cc_rooftopworks_PIPE_H_LO
        || cc.height > cc_rooftopworks_PIPE_H_HI) {
        return res;
    }
    let gh = city_rand4(cc.seed ^ vec2<u32>(0x27d4eb2fu, 0xb5297a4du));
    if (gh.x >= cc_rooftopworks_PIPE_FRAC) {
        return res;
    }
    let n = 1 + i32(gh.y * 2.999);
    let span = f32(n) * cc_rooftopworks_PIPE_D
             + f32(n - 1) * cc_rooftopworks_PIPE_GAP;
    let b = cc.b1min.xy;
    let bt = cc.b1max.xy;
    let top = cc.b1max.z - 0.8;
    if (top < 8.0) {
        return res;
    }
    let ay = gh.z > 0.5;
    let neg_x = gh.w < 0.5;
    let neg_y = fract(gh.w * 13.0) < 0.5;
    if (ay) {
        // Riding a wall whose normal is +-x, running up beside a corner.
        let xw = select(bt.x, b.x, neg_x);
        let x0 = select(xw, xw - cc_rooftopworks_PIPE_OUT, neg_x);
        let y0 = select(bt.y - cc_rooftopworks_PIPE_CORNER - span,
                        b.y + cc_rooftopworks_PIPE_CORNER, neg_y);
        res.lo = vec3<f32>(x0, y0, 2.0);
        res.hi = vec3<f32>(x0 + cc_rooftopworks_PIPE_OUT, y0 + span, top);
    } else {
        let yw = select(bt.y, b.y, neg_y);
        let y0 = select(yw, yw - cc_rooftopworks_PIPE_OUT, neg_y);
        let x0 = select(bt.x - cc_rooftopworks_PIPE_CORNER - span,
                        b.x + cc_rooftopworks_PIPE_CORNER, neg_x);
        res.lo = vec3<f32>(x0, y0, 2.0);
        res.hi = vec3<f32>(x0 + span, y0 + cc_rooftopworks_PIPE_OUT, top);
    }
    res.ok = true;
    res.n = n;
    res.axis_y = ay;
    return res;
}

// --- the trace hook ---------------------------------------------------------
fn cc_rooftopworks_props_trace(o: vec3<f32>, dir: vec3<f32>,
                               inv_dir: vec3<f32>, t0: f32, t1: f32,
                               ci: vec2<i32>, cc: CityCell) -> CityHit {
    // Gate 0, no hash, no arithmetic: unbuilt lots and spire crowns are not
    // ours. The architecture rule is enforced here first and in the deck
    // function second, so neither can be got around.
    if (!cc.built || cc.arch == 4) {
        return cc_rooftopworks_miss(ci);
    }
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    let z_lo = min(za, zb);
    let z_hi = max(za, zb);
    let fp = max(2.0 * u.cam_origin.w / max(u.params.x, 1.0), u.periodic.z)
             * max(t0, 0.0);
    var best = cc_rooftopworks_miss(ci);

    // Gate 1: the roof band, from the lowest deck this building owns to the
    // headroom above its summit. Most city rays never enter it.
    if (z_hi >= cc_rooftopworks_deck_low(cc) - 1.0
        && z_lo <= cc.height + cc_rooftopworks_HEADROOM) {
        let d0 = cc_rooftopworks_deck(cc, 0);
        let cr = cc_rooftopworks_crown(cc, ci);
        if (d0.ok) {
            // The parapet lives in a 1.25 m band on one deck: a tighter z
            // test than the section gate, and it comes before its hash.
            if (z_hi >= d0.z - 0.2
                && z_lo <= d0.z + cc_rooftopworks_PAR_H + 0.2) {
                let par = cc_rooftopworks_has_parapet(cc);
                if (par.x < cc_rooftopworks_PAR_FRAC) {
                    best = cc_rooftopworks_parapet(best, o, dir, inv_dir,
                                                   t0, t1, d0, ci);
                }
            }
            if (cr.kind != 0) {
                best = cc_rooftopworks_box(
                    best, o, dir, inv_dir, t0, t1,
                    vec3<f32>(cr.lo, cr.z0), vec3<f32>(cr.hi, cr.z1),
                    select(cc_rooftopworks_K_PAD, cc_rooftopworks_K_PENT,
                           cr.kind == 1),
                    ci);
            }
        }
        // Nothing mechanical stands more than 5 m off a deck, so the slot
        // loop gets its own band inside the section gate — the penthouse
        // headroom above it is the crown's business, not the clutter's.
        let mech_on = z_lo <= cc.height + cc_rooftopworks_VENT_H + 0.2
                      && cc_rooftopworks_mech_on(cc);
        for (var s: i32 = 0; s < 4; s = s + 1) {
            let m = cc_rooftopworks_mech(cc, ci, s, mech_on, cr);
            if (m.ok) {
                best = cc_rooftopworks_mech_hit(best, o, dir, inv_dir, t0, t1,
                                                m, s, ci, fp);
            }
        }
    }

    // Gate 2: pipework, close range only.
    if (fp < cc_rooftopworks_PIPE_FP && z_lo <= cc.b1max.z && z_hi >= 1.5) {
        let pp = cc_rooftopworks_pipes(cc);
        if (pp.ok) {
            best = cc_rooftopworks_box(best, o, dir, inv_dir, t0, t1,
                                       pp.lo, pp.hi,
                                       cc_rooftopworks_K_PIPE, ci);
        }
    }
    return best;
}

// --- shading ----------------------------------------------------------------
// A point source seen on a surface, with the energy-preserving widening the
// signage uses: as the footprint grows the spot spreads to a pixel and dims
// by the same factor, so a lamp never strobes and never vanishes.
fn cc_rooftopworks_spot(p: vec3<f32>, seat: vec3<f32>, sig: f32, fp: f32)
        -> f32 {
    let s = max(sig, 0.30 * fp);
    let k = (sig * sig) / (s * s);
    let d = p - seat;
    return k * exp(-dot(d, d) / (2.0 * s * s));
}

fn cc_rooftopworks_fill(p: vec3<f32>) -> vec3<f32> {
    return CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
}

// Dark painted steel against the night: a little skyglow, a little moon, and
// a rim that brightens toward the horizontal because the light on a roof
// comes from the street below and from the city all around it.
fn cc_rooftopworks_dark(h: CityHit, albedo: f32) -> vec3<f32> {
    let fill = cc_rooftopworks_fill(h.pos);
    let side = 1.0 - abs(h.normal.z);
    return albedo * fill * (1.0 + 1.6 * side)
         + albedo * CITY_MOONLIGHT * max(dot(h.normal, u.sun_dir.xyz), 0.0);
}

// The house colour, where the building has one; cool white otherwise.
fn cc_rooftopworks_trim_color(cc: CityCell) -> vec3<f32> {
    if (cc.win_mono >= 0.0) {
        return city_window_color(cc.win_mono, cc.palette_bias);
    }
    return cc_rooftopworks_TRIM_COOL;
}

fn cc_rooftopworks_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    // --- parapet -----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PARAPET) {
        let d0 = cc_rooftopworks_deck(cc, 0);
        var e = cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO * 1.3);
        let par = cc_rooftopworks_has_parapet(cc);
        if (par.y < cc_rooftopworks_PAR_LIT) {
            // A strip of light under the coping. Sharp while the band is
            // wider than a pixel; past that it is the same energy spread
            // over the whole wall face, which is what a real coping light
            // looks like from two kilometres — a line, not a sparkle.
            let up = h.pos.z - d0.z;
            let band = select(
                0.0, 1.0,
                up > cc_rooftopworks_PAR_H - cc_rooftopworks_PAR_TRIM);
            let mean = (cc_rooftopworks_PAR_TRIM / cc_rooftopworks_PAR_H)
                       * cc_rooftopworks_TRIM_MEAN_COMP;
            let k = smoothstep(cc_rooftopworks_TRIM_LOD_START,
                               cc_rooftopworks_TRIM_LOD_FULL, fp);
            let face = select(1.0, 0.45, h.normal.z > 0.5);
            // Runs, not a continuous tube. An unbroken glowing rectangle
            // around every roof reads as wireframe; fixtures on a 6.5 m
            // centre with a dark joint between them read as hardware. The
            // run coordinate is the perimeter direction, taken from which
            // edge of the deck the hit is nearest.
            let dxy = min(h.pos.xy - d0.lo, d0.hi - h.pos.xy);
            let along_y = dxy.x < dxy.y;
            let uc = select(h.pos.x, h.pos.y, along_y);
            let seg = select(1.0, 0.0,
                             fract(uc / cc_rooftopworks_TRIM_PITCH)
                             > cc_rooftopworks_TRIM_DUTY);
            let segk = mix(seg, cc_rooftopworks_TRIM_DUTY,
                           smoothstep(0.9, 2.4, fp));
            e = e + cc_rooftopworks_trim_color(cc)
                    * (cc_rooftopworks_TRIM_RAD * face * segk
                       * mix(band, mean, k));
        }
        return e;
    }

    // --- penthouse ---------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PENT) {
        let cr = cc_rooftopworks_crown(cc, h.cell);
        if (h.normal.z > 0.5) {
            return cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO);
        }
        // Glass on three-and-a-bit sides, with mullions and a dark spandrel
        // at its foot. The colour follows the house where there is one.
        let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
        let tangent = vec2<f32>(-nh.y, nh.x);
        let uc = dot(h.pos.xy, tangent);
        let up = h.pos.z - cr.z0;
        let mull = select(
            1.0, 0.10,
            fract(uc / cc_rooftopworks_PENT_PITCH) < cc_rooftopworks_PENT_MULL);
        let sill = smoothstep(0.0, cc_rooftopworks_PENT_SILL, up);
        // Panes, not a slab. Lit uniformly at 1.8 the crown came out as the
        // brightest thing on the skyline — brighter in the mean than the
        // facade below it, which is backwards. It is a room: most of the
        // glass is lit, some is not, and each bay has its own level.
        let iu = i32(floor(uc / cc_rooftopworks_PENT_PITCH));
        let iv = i32(floor(up / CITY_FLOOR_H));
        let wh = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu) * 0x9e3779b9u) ^ 0x0d2f1a37u,
            cc.seed.y ^ (bitcast<u32>(iv) * 0x85ebca6bu) ^ 0x1b56c4e9u));
        let bay = select(0.0, 0.45 + 1.10 * wh.z,
                         wh.x < cc_rooftopworks_PENT_LIT);
        // Mullions are 13% of the pitch and a bay is 2.1 m: both are
        // sub-pixel past a couple of metres, where the wall settles to its
        // own mean rather than flickering.
        let k = smoothstep(0.7, 2.2, fp);
        let m = mix(mull * bay,
                    (1.0 - 0.9 * cc_rooftopworks_PENT_MULL)
                    * cc_rooftopworks_PENT_LIT
                    * cc_rooftopworks_PENT_MEAN_COMP, k);
        var col = cc_rooftopworks_PENT_COL;
        if (cc.win_mono >= 0.0) {
            col = city_window_color(cc.win_mono, cc.palette_bias);
        }
        return cc_rooftopworks_dark(h, CITY_FACADE_ALBEDO)
             + col * (cc_rooftopworks_PENT_RAD * m * sill);
    }

    // --- helipad -----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PAD) {
        let cr = cc_rooftopworks_crown(cc, h.cell);
        var e = cc_rooftopworks_dark(h, CITY_ROOF_ALBEDO * 0.8);
        if (h.normal.z > 0.5) {
            // Paint: a ring and a bar. Albedo only — at night it is barely
            // there, which is right; the pad is read by its corner lights.
            let c = 0.5 * (cr.lo + cr.hi);
            let half = 0.5 * (cr.hi - cr.lo);
            let q = (h.pos.xy - c) / max(half, vec2<f32>(1e-3));
            let rr = length(q);
            let ring = select(0.0, 1.0, abs(rr - 0.62) < 0.06);
            let bar = select(0.0, 1.0,
                             abs(q.x) < 0.10 && abs(q.y) < 0.34);
            let paint = max(ring, bar) * (1.0 - smoothstep(0.4, 1.4, fp))
                      + 0.22 * smoothstep(0.4, 1.4, fp);
            e = e + vec3<f32>(cc_rooftopworks_PAD_MARK * paint)
                    * cc_rooftopworks_fill(h.pos) * 3.0;
            // Four corner lights, 1 m in from the corners.
            let s = half - 1.0;
            var lamps = 0.0;
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(s.x, s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(-s.x, s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(s.x, -s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            lamps = lamps + cc_rooftopworks_spot(
                h.pos, vec3<f32>(c + vec2<f32>(-s.x, -s.y), cr.z1),
                cc_rooftopworks_PAD_LAMP_SIG, fp);
            e = e + cc_rooftopworks_PAD_LAMP_COL
                    * (cc_rooftopworks_PAD_LAMP_RAD * lamps);
        }
        return e;
    }

    // --- pipework ----------------------------------------------------------
    if (h.kind == cc_rooftopworks_K_PIPE) {
        let pp = cc_rooftopworks_pipes(cc);
        // Across the bundle: each riser reads as a cylinder by shading, a
        // bright sliver where the wall light grazes it and dark at the seam.
        let across = select(h.pos.x - pp.lo.x, h.pos.y - pp.lo.y, pp.axis_y);
        let pitch = cc_rooftopworks_PIPE_D + cc_rooftopworks_PIPE_GAP;
        let fu = clamp(fract(across / pitch) / (cc_rooftopworks_PIPE_D / pitch),
                       0.0, 1.0);
        let round = sqrt(max(1.0 - (2.0 * fu - 1.0) * (2.0 * fu - 1.0), 0.0));
        let face = 1.0 - abs(h.normal.z);
        // The rim: the lit facade behind a riser wraps its edges.
        let k = 1.0 - smoothstep(cc_rooftopworks_PIPE_FADE,
                                 cc_rooftopworks_PIPE_FP, fp);
        let rim = mix(0.35, 1.0, round) * face * k;
        return cc_rooftopworks_dark(h, CITY_FACADE_ALBEDO * 1.4)
             + CITY_PALETTE_MEAN * (0.10 * rim);
    }

    // --- mechanical clusters ------------------------------------------------
    let is_acc = h.kind >= cc_rooftopworks_K_ACC;
    let s = h.kind - select(cc_rooftopworks_K_MECH, cc_rooftopworks_K_ACC,
                            is_acc);
    let m = cc_rooftopworks_mech(cc, h.cell, s,
                                 cc_rooftopworks_mech_on(cc),
                                 cc_rooftopworks_crown(cc, h.cell));
    var alb = cc_rooftopworks_MECH_ALBEDO;
    if (m.kind == 0 && is_acc) {
        alb = alb * 1.35;  // the rim band is galvanised: it catches more
    }
    var e = cc_rooftopworks_dark(h, alb);
    if (m.lamp) {
        // The fixture itself, plus the pool it throws on its own housing.
        let src = cc_rooftopworks_spot(h.pos, m.seat,
                                       cc_rooftopworks_LAMP_SIG, fp);
        let spill = cc_rooftopworks_spot(h.pos, m.seat,
                                         cc_rooftopworks_LAMP_SPILL, fp);
        e = e + cc_rooftopworks_LAMP_COL
                * (cc_rooftopworks_LAMP_RAD * (src + 0.10 * spill));
    }
    // Condenser rows: the unit gaps and the grille bands, drawn rather than
    // built. Three units on a 2.6 m pitch; the seams are dark, the grille a
    // shade lighter than the casing.
    if (m.kind == 1 && !is_acc) {
        let along = select(h.pos.y - m.lo.y, h.pos.x - m.lo.x,
                           (m.hi.x - m.lo.x) > (m.hi.y - m.lo.y));
        let fu = fract(along / 2.60);
        let seam = select(1.0, 0.35, fu < 0.12 || fu > 0.88);
        let grille = select(0.0, 1.0, h.normal.z > 0.5 && fu > 0.30
                            && fu < 0.70);
        // Both are modulations of the casing's own albedo, never additions:
        // a fan grille is lighter metal, not a light. (It was an additive
        // skyglow term once, and every condenser on the roof came out cyan.)
        let k = 1.0 - smoothstep(0.5, 1.6, fp);
        e = e * (mix(1.0, seam, k) * (1.0 + 0.8 * grille * k));
    }
    return e;
}

// --- component: skybridges (skybridges.wgsl) ---
// skybridges — the enclosed walkways that make two towers one address.
//
// THE PROBLEM. Every other piece of this city fits inside one block column,
// which is what lets the DDA test a cell's geometry only while the ray is
// inside that cell. A bridge does not: it spans the street, so half of it
// lives in one column and half in the next. The fix is not to make the
// tracer smarter. It is to make the bridge belong to the EDGE between two
// cells rather than to either cell, and to derive it from data both cells
// can read — so cell A and cell B independently construct the SAME box, to
// the bit. Each then reports whichever part of that box its own ray segment
// contains, and the two halves meet with nothing between them.
//
// The edge id is the whole trick. For the edge between (i,j) and (i+1,j)
// the id is (2i+1, 2j); between (i,j) and (i,j+1) it is (2i, 2j+1). Doubling
// the cell index leaves the even lattice for cells and puts each edge on the
// odd coordinate of the axis it crosses, so the id is a property of the edge
// and is computed identically from either side. A cell tests all four of its
// edges; the neighbour tests the same four ids from its side; the two agree
// by construction rather than by convention.
//
// The other half of the trick is CONTAINMENT. Determinism alone is not
// enough — the box must also lie inside the two columns that test it, or the
// ray would enter it while the DDA is in some third cell that never looks.
// Two rules enforce it: each end face must be within half a block of the
// shared boundary, and the deck's cross-extent is clipped to the edge's own
// row. Everything a bridge needs is then a function of (edge id, cell A,
// cell B), which is also why the shading hook can rebuild the exact span
// from a hit position alone: round the hit to the nearest block boundary on
// the bridge's axis and the two cells fall out.
//
// WHERE THEY GO. Only downtown (both blocks past the density gate, and it is
// the LOWER of the two that decides so both sides agree), only between two
// built buildings, and only where the deck fits under BOTH roofs with 15 m to
// spare. "Under the roof" means under the b1 box — the widest, lowest prim of
// every archetype. That one rule buys the whole architectural constraint the
// city asks for: a bridge lands on a slab or a growth tower's shaft, on a
// tapered shaft only below its podium cap, on a spire tower only below the
// spring line of the spire, and on nothing at all above a setback.
//
// WHAT THEY LOOK LIKE. A 0.5 m deck, a 2.6 m glass band, a thin roof: from a
// kilometre out a lit ribbon strung between two dark towers, which is the
// Cloudpunk image this component exists for. Close up the band resolves into
// mullions every 2.5 m, a scatter of dark bays, and now and then somebody
// walking across at two in the morning.

// kind = BASE + axis + 2 * type, type 0 shell, 1 glass band, 2 truss member.
// The axis rides in the low bit so the type is a plain shift, and both fit
// the 16 local kinds a component owns.
const cc_skybridges_KIND_BASE: i32 = 500;

// --- eligibility ----------------------------------------------------------
// The gate is the first thing evaluated per edge and costs one pcg2d. What
// survives it pays for one neighbour cell fetch; what survives THAT pays for
// three box tests. Measured over a 256x256-cell window on the megatower
// district (tools' probe, 41967 qualifying edges): 59% of gated edges end up
// carrying a bridge, the rest losing on fit — frontage, gap or headroom — so
// the gate is set to 0.118 to land the built share at 7.0% of the edges whose
// two blocks are both built and both past the density floor.
const cc_skybridges_GATE: f32 = 0.118;
const cc_skybridges_MIN_DENSITY: f32 = 0.25;
const cc_skybridges_GAP_MIN: f32 = 8.0;   // a street, not a seam
const cc_skybridges_GAP_MAX: f32 = 62.0;  // beyond this it is a viaduct
const cc_skybridges_Z_MIN: f32 = 25.0;
const cc_skybridges_Z_MAX: f32 = 380.0;   // above this a walkway is a stunt
const cc_skybridges_ROOF_MARGIN: f32 = 15.0;

// --- geometry (metres) ----------------------------------------------------
const cc_skybridges_DECK_T: f32 = 0.5;
const cc_skybridges_GLASS_H: f32 = 2.6;
const cc_skybridges_ROOF_T: f32 = 0.4;
const cc_skybridges_TUBE_H: f32 = 3.5;    // DECK_T + GLASS_H + ROOF_T
const cc_skybridges_HALF_W: f32 = 2.0;    // a 4 m tube: two people wide
const cc_skybridges_HALF_W_GRAND: f32 = 4.0;  // a sky lobby, 8 m across
const cc_skybridges_GRAND_FRAC: f32 = 0.22;
const cc_skybridges_GLASS_INSET: f32 = 0.12;  // deck lip proud of the glass
const cc_skybridges_ROOF_OVER: f32 = 0.18;    // fascia proud of the glass
const cc_skybridges_EMBED: f32 = 1.0;     // driven into each wall, so no end
                                          // can ever float in the gap
const cc_skybridges_ROW_MARGIN: f32 = 1.0;

// The hook's own z gate: the extreme deck is Z_MAX quantized up plus the
// tube, and the lowest is Z_MIN rounded down a storey. Nothing outside this
// band exists, so a segment that misses it costs one compare.
const cc_skybridges_Z_LO: f32 = 20.0;
const cc_skybridges_Z_HI: f32 = 395.0;

// --- the shell ------------------------------------------------------------
// Everything albedo-lit is near-black at night, so the deck and roof cannot
// be made to read by brightening them — only by structure. The underside is
// the surface that matters: from the street a bridge is a dark belly with a
// lamp string on it, and nothing else. Ribs sit on the glazing's own 2.5 m
// pitch, because one tube built once has one rhythm.
const cc_skybridges_SHELL_TINT: f32 = 0.012;  // corridor leak through the shell
const cc_skybridges_SHELL_FILL: f32 = 0.055;
const cc_skybridges_RIB_PITCH: f32 = 2.5;
const cc_skybridges_RIB_HALF: f32 = 0.16;   // a 0.32 m transverse web
const cc_skybridges_SPINE_HALF: f32 = 0.32; // the box girder down the belly
const cc_skybridges_LAMP_PITCH: f32 = 5.0;
const cc_skybridges_LAMP_R: f32 = 0.18;
const cc_skybridges_LAMP_RAD: f32 = 0.85;

// --- the open truss -------------------------------------------------------
// Not every span between two towers is a corridor: some are service crossings
// that never had a wall. The variant keeps the deck and the same three box
// tests, but the middle prim shrinks to a central web and the top prim to a
// chord, so the silhouette is genuinely open — sky on both sides above the
// handrail — and what carries it at night is a lamp string rather than a lit
// room. Rare, because the enclosed walkway is the thing this city is for.
const cc_skybridges_TRUSS_FRAC: f32 = 0.18;
const cc_skybridges_TRUSS_HALF: f32 = 0.30;   // the central web
const cc_skybridges_CHORD_FRAC: f32 = 0.55;   // top chord, as a share of half
const cc_skybridges_STRING_PITCH: f32 = 4.0;
const cc_skybridges_STRING_R: f32 = 0.16;
const cc_skybridges_STRING_RAD: f32 = 2.6;
const cc_skybridges_TRUSS_PITCH: f32 = 3.0;   // one Warren bay
const cc_skybridges_MEMBER_T: f32 = 0.20;     // diagonal, measured vertically
const cc_skybridges_POST_T: f32 = 0.10;
const cc_skybridges_CHORD_T: f32 = 0.17;
// Mean member cover of the web face, integrated off the constants above:
// two chords (2*0.17/2.6), the posts (2*0.10/3.0) and the diagonals
// (2*0.20 of vertical extent per 1.5 m of run, over 2.6 m), less overlap.
const cc_skybridges_TRUSS_COVER: f32 = 0.40;

// --- the band -------------------------------------------------------------
const cc_skybridges_MULLION_PITCH: f32 = 2.5;
const cc_skybridges_MULLION_HALF: f32 = 0.11;  // half the 0.22 m web
const cc_skybridges_GLASS_V0: f32 = 0.26;  // sill, above the deck surface
const cc_skybridges_GLASS_V1: f32 = 2.39;  // head
const cc_skybridges_RADIANCE: f32 = 2.0;
const cc_skybridges_DARK_SEG: f32 = 0.12;  // bays with the lights off
const cc_skybridges_FIG_FRAC: f32 = 0.15;  // bays with somebody in them
const cc_skybridges_BODY_T: f32 = 0.12;
const cc_skybridges_SPANDREL: f32 = 0.04;  // sill/head/mullion transmission

// Footprint windows (m/px) over which each layer blends into its own mean,
// each set by that layer's finest feature: a 0.1 m limb, a 0.22 m mullion
// web, a 2.5 m bay. Past its window a layer is a constant, so the next
// coarser one hands off smoothly and nothing vanishes.
const cc_skybridges_LOD_FIG: vec2<f32> = vec2<f32>(0.25, 1.10);
const cc_skybridges_LOD_MUL: vec2<f32> = vec2<f32>(0.60, 3.00);
const cc_skybridges_LOD_SEG: vec2<f32> = vec2<f32>(3.00, 10.0);
const cc_skybridges_LOD_RIB: vec2<f32> = vec2<f32>(0.50, 2.60);
const cc_skybridges_LOD_LAMP: vec2<f32> = vec2<f32>(0.30, 1.60);

// The band's mean cover, measured off the constants above rather than
// guessed: the glazed fraction of the box face (2.13 / 2.6), the mullion
// duty (1 - 0.22 / 2.5), the lit-bay fraction (1 - 0.12), and the mean
// transmission left by the figures (1 - 0.15 * 0.113 * 0.88, the 0.113 being
// a body's share of a glazed bay). The bay brightness draw has mean 1.
const cc_skybridges_MEAN_COVER: f32 = 0.647;
// Averaging radiance ahead of a compressive tone map runs bright; the same
// compensation the core's octave ladder carries, scaled for a ribbon that
// never fills more than a few pixels.
const cc_skybridges_MEAN_COMP: f32 = 0.80;

// One bridge's whole description. Both hooks build this and must agree.
struct cc_skybridges_Span {
    ok: bool,
    axis: i32,     // 0 the bridge runs along x, 1 along y
    lo: f32,       // start along the axis, inside wall A
    hi: f32,       // end along the axis, inside wall B
    ctr: f32,      // centre across the axis
    half: f32,     // half deck width
    z: f32,        // deck underside
    truss: bool,   // open service crossing rather than an enclosed walkway
    seed: vec2<u32>,
}

// The edge id: even coordinates are cells, the odd one names the axis the
// edge crosses. Computed from the LOWER cell of the pair, which both sides
// can name.
fn cc_skybridges_eid(clo: vec2<i32>, axis: i32) -> vec2<u32> {
    return vec2<u32>(
        bitcast<u32>(2 * clo.x + (1 - axis)) ^ 0x5bf03635u,
        bitcast<u32>(2 * clo.y + axis) ^ 0x9e3779b9u);
}

// The gate. This is the component's hottest instruction by a wide margin —
// four of them run in every cell the DDA visits inside CITY_PROP_RANGE,
// whether or not a bridge is anywhere near — and measuring said the per-cell
// hashing, not the geometry, was two thirds of what the component costs. So
// it is one multiply-xorshift round instead of pcg2d's two. It must be a pure
// function of the edge id, which is the entire determinism contract; it does
// NOT have to be a strong hash, because all it decides is a yes/no on an
// eighth of edges and everything that survives is redrawn with pcg2d.
// One multiply-xorshift round rather than pcg2d's two. Folding a cell's two
// owned edges into the halves of a single hash — four gates for three hashes
// — was tried and measured no faster on the one view whose bridge statistics
// are comparable across the change, so the per-edge hash stays: independent
// gates, and the simpler thing when the complicated thing cannot be shown to
// pay.
fn cc_skybridges_gate(clo: vec2<i32>, axis: i32) -> bool {
    var h = bitcast<u32>(2 * clo.x + (1 - axis)) * 0x9e3779b9u
          ^ bitcast<u32>(2 * clo.y + axis) * 0x85ebca6bu;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    return city_u01(h) < cc_skybridges_GATE;
}

fn cc_skybridges_bmin(axis: i32, a: f32, c: f32, z: f32) -> vec3<f32> {
    if (axis == 0) {
        return vec3<f32>(a, c, z);
    }
    return vec3<f32>(c, a, z);
}

// Everything about the bridge on one edge, from the two cells it joins and
// the edge's own draws. Deterministic in its arguments and nothing else,
// which is the entire contract this component rests on.
fn cc_skybridges_build(clo: vec2<i32>, axis: i32, ca: CityCell, cb: CityCell,
                       d: vec4<f32>) -> cc_skybridges_Span {
    var s: cc_skybridges_Span;
    s.ok = false;
    s.axis = axis;
    s.lo = 0.0; s.hi = 0.0; s.ctr = 0.0; s.half = 0.0; s.z = 0.0;
    s.truss = false;
    s.seed = vec2<u32>(0u);
    if (!ca.built || !cb.built) {
        return s;
    }
    // A merged superblock reports the same seed from every member cell: an
    // edge inside one building is not a bridge, it is a corridor.
    if (ca.seed.x == cb.seed.x && ca.seed.y == cb.seed.y) {
        return s;
    }
    if (min(ca.density, cb.density) <= cc_skybridges_MIN_DENSITY) {
        return s;
    }

    let cell = u.ocean_params.x;
    // Faces along the axis, and the row the edge lives in across it.
    var a_face: f32; var b_face: f32; var bnd: f32;
    var a_lo: f32; var a_hi: f32; var b_lo: f32; var b_hi: f32;
    var row: f32;
    if (axis == 0) {
        bnd = f32(clo.x + 1) * cell;
        a_face = ca.b1max.x; b_face = cb.b1min.x;
        a_lo = ca.b1min.y; a_hi = ca.b1max.y;
        b_lo = cb.b1min.y; b_hi = cb.b1max.y;
        row = f32(clo.y);
    } else {
        bnd = f32(clo.y + 1) * cell;
        a_face = ca.b1max.y; b_face = cb.b1min.y;
        a_lo = ca.b1min.x; a_hi = ca.b1max.x;
        b_lo = cb.b1min.x; b_hi = cb.b1max.x;
        row = f32(clo.x);
    }

    // Containment. Each wall must be within half a block of the boundary:
    // that keeps the whole tube inside the two columns that test it (so no
    // third cell can own the ray where the box begins) and it is also what
    // makes the shading hook's rounding recovery exact.
    let reach = 0.5 * cell - 2.0;
    if (abs(a_face - bnd) > reach || abs(b_face - bnd) > reach) {
        return s;
    }
    let gap = b_face - a_face;
    if (gap < cc_skybridges_GAP_MIN || gap > cc_skybridges_GAP_MAX) {
        return s;
    }

    // Across the axis: the two walls' shared frontage, clipped to the edge's
    // own row so the deck cannot wander into a neighbouring one.
    let c0 = max(max(a_lo, b_lo), row * cell + cc_skybridges_ROW_MARGIN);
    let c1 = min(min(a_hi, b_hi), (row + 1.0) * cell
                                  - cc_skybridges_ROW_MARGIN);
    let frontage = c1 - c0;
    // A service crossing is never also a sky lobby: the two draws are read in
    // order so the variant is decided before the width that depends on it.
    let truss = d.x < cc_skybridges_TRUSS_FRAC;
    var half = cc_skybridges_HALF_W;
    if (!truss && d.w < cc_skybridges_GRAND_FRAC
        && frontage > 2.0 * cc_skybridges_HALF_W_GRAND + 3.0) {
        half = cc_skybridges_HALF_W_GRAND;
    }
    let slack = frontage - 2.0 * (half + cc_skybridges_ROOF_OVER) - 1.0;
    if (slack < 0.0) {
        return s;
    }

    // Height: a storey between 25 m and the lower of the two b1 roofs less
    // its margin. b1 is the base box of every archetype, so this is the
    // "lands on the widest prim" rule stated once and obeyed everywhere.
    let z_top = min(ca.b1max.z, cb.b1max.z) - cc_skybridges_ROOF_MARGIN;
    let z_hi = min(z_top - cc_skybridges_TUBE_H, cc_skybridges_Z_MAX);
    if (z_hi < cc_skybridges_Z_MIN) {
        return s;
    }
    var z = floor((cc_skybridges_Z_MIN + (z_hi - cc_skybridges_Z_MIN) * d.y)
                  / CITY_FLOOR_H) * CITY_FLOOR_H;
    if (z < cc_skybridges_Z_MIN) {
        z = z + CITY_FLOOR_H;
    }

    s.ok = true;
    s.lo = a_face - cc_skybridges_EMBED;
    s.hi = b_face + cc_skybridges_EMBED;
    s.ctr = c0 + half + cc_skybridges_ROOF_OVER + 0.5 + slack * d.z;
    s.half = half;
    s.z = z;
    s.truss = truss;
    s.seed = pcg2d(cc_skybridges_eid(clo, axis) ^ vec2<u32>(0x68e31da4u,
                                                            0xb5297a4du));
    return s;
}

// --- the trace ------------------------------------------------------------
// All four edges of cell ci, cheap hash first. A hit counts only if its t
// falls inside this cell's segment, which is what splits the tube between
// the two cells with no overlap and no gap: the box's entry point lies in
// exactly one column, and that column's segment is the one that contains it.
fn cc_skybridges_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                             t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    // If this cell fails the gate it fails the min() on every one of its
    // edges too, so the neighbour reaches the same verdict.
    if (!cc.built || cc.density <= cc_skybridges_MIN_DENSITY) {
        return res;
    }
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_skybridges_Z_HI || max(za, zb) < cc_skybridges_Z_LO) {
        return res;
    }

    for (var e: i32 = 0; e < 4; e = e + 1) {
        let axis = e >> 1;
        let neg = (e & 1) == 1;
        let stp = select(vec2<i32>(1, 0), vec2<i32>(0, 1), axis == 1);
        let clo = select(ci, ci - stp, neg);
        if (!cc_skybridges_gate(clo, axis)) {
            continue;
        }
        // One extra cell fetch, and only behind the gate.
        let p0 = pcg2d(cc_skybridges_eid(clo, axis));
        let nb = city_cell(select(clo + stp, clo, neg));
        var ca = cc;
        var cb = nb;
        if (neg) {
            ca = nb;
            cb = cc;
        }
        let s = cc_skybridges_build(clo, axis, ca, cb, city_rand4(p0));
        if (!s.ok) {
            continue;
        }

        let zd = s.z;
        let zg = zd + cc_skybridges_DECK_T;
        let zr = zg + cc_skybridges_GLASS_H;
        let zt = zd + cc_skybridges_TUBE_H;
        let gh = s.half - cc_skybridges_GLASS_INSET;
        let rh = s.half + cc_skybridges_ROOF_OVER;
        for (var p: i32 = 0; p < 3; p = p + 1) {
            var w: f32; var z_lo: f32; var z_hi: f32;
            if (p == 0) {
                w = s.half; z_lo = zd; z_hi = zg;        // deck
            } else if (p == 1) {
                // Glass band, or the truss's central web: same prim, same
                // test, a tenth of the width.
                w = select(gh, cc_skybridges_TRUSS_HALF, s.truss);
                z_lo = zg; z_hi = zr;
            } else {
                w = select(rh, s.half * cc_skybridges_CHORD_FRAC, s.truss);
                z_lo = zr; z_hi = zt;                    // roof, or top chord
            }
            let bmin = cc_skybridges_bmin(axis, s.lo, s.ctr - w, z_lo);
            let bmax = cc_skybridges_bmin(axis, s.hi, s.ctr + w, z_hi);
            let hb = city_box_hit(o, inv_dir, bmin, bmax);
            if (hb.x <= hb.y && hb.x > 0.0 && hb.x < res.t
                && hb.x >= t0 - 1e-3 && hb.x <= t1 + 1e-3) {
                res.hit = true;
                res.t = hb.x;
                res.pos = o + hb.x * dir;
                res.normal = city_box_normal(res.pos, bmin, bmax);
                res.cell = ci;
                let side = select(abs(res.normal.y), abs(res.normal.x),
                                  axis == 1);
                // The whole web is a truss member; only the band's vertical
                // faces are glazing (its sill and head are shell).
                var ty = 0;
                if (p == 1) {
                    if (s.truss) { ty = 2; }
                    else if (side > 0.5) { ty = 1; }
                }
                res.kind = cc_skybridges_KIND_BASE + axis + 2 * ty;
            }
        }
    }
    return res;
}

// --- shading --------------------------------------------------------------

fn cc_skybridges_capsule(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, r: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let hh = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * hh) - r;
}

// Somebody crossing. Two capsules — a leaning body and a head — in bay
// metres, which is all a person is worth at the distance a bridge is
// usually seen from, and more than enough to turn a lit tube into an
// inhabited one.
fn cc_skybridges_figure(sh: vec4<f32>, pm: vec2<f32>, fp: f32) -> f32 {
    let sc = 0.90 + 0.22 * sh.w;
    let cx = 0.5 + (cc_skybridges_MULLION_PITCH - 1.0) * sh.y;
    let lean = (sh.z - 0.5) * 0.17;
    let body = cc_skybridges_capsule(
        pm, vec2<f32>(cx, 0.30 * sc), vec2<f32>(cx + lean, 1.40 * sc),
        0.20 * sc);
    let head = cc_skybridges_capsule(
        pm, vec2<f32>(cx + lean * 1.30, 1.58 * sc),
        vec2<f32>(cx + lean * 1.36, 1.65 * sc), 0.115 * sc);
    let aa = 0.03 + 0.5 * fp;
    return 1.0 - smoothstep(-aa, aa, min(body, head));
}

// The tube's shell: deck, roof and fascia. It is albedo-lit, so it is
// near-black and no amount of gain would change that honestly; what makes it
// read is structure. Ribs panelize the whole tube on the glazing pitch; the
// underside additionally carries the box girder and the service lamps, which
// are the only part of a bridge's belly the street ever actually sees.
fn cc_skybridges_shell(h: CityHit, s: cc_skybridges_Span, tint: vec3<f32>,
                       fill: vec3<f32>, moon: vec3<f32>, fp_eff: f32)
        -> vec3<f32> {
    var e = cc_skybridges_SHELL_FILL * fill
            + cc_skybridges_SHELL_TINT * tint
            + 0.045 * CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    if (!s.ok) {
        return e;   // the span could not be rebuilt: flat shell, never magenta
    }
    let along = select(h.pos.x, h.pos.y, s.axis == 1);
    let cross_m = select(h.pos.y, h.pos.x, s.axis == 1) - s.ctr;
    let ua = along - s.lo;
    let aa = min(0.5 * fp_eff, 0.30);
    let lod = smoothstep(cc_skybridges_LOD_RIB.x, cc_skybridges_LOD_RIB.y,
                         fp_eff);

    let rp = ua - floor(ua / cc_skybridges_RIB_PITCH)
             * cc_skybridges_RIB_PITCH;
    let web = min(rp, cc_skybridges_RIB_PITCH - rp);
    let rib = 1.0 - smoothstep(cc_skybridges_RIB_HALF - aa - 0.02,
                               cc_skybridges_RIB_HALF + aa + 0.02, web);
    let rib_l = mix(rib, 2.0 * cc_skybridges_RIB_HALF
                         / cc_skybridges_RIB_PITCH, lod);
    e = e * mix(0.72, 1.75, rib_l);

    if (h.normal.z > -0.5) {
        return e;
    }
    let spine = 1.0 - smoothstep(cc_skybridges_SPINE_HALF - aa - 0.02,
                                 cc_skybridges_SPINE_HALF + aa + 0.02,
                                 abs(cross_m));
    let spine_l = mix(spine,
                      min(cc_skybridges_SPINE_HALF / max(s.half, 0.5), 1.0),
                      lod);
    e = e * mix(1.0, 1.45, spine_l);

    // The lamp string. Sub-pixel it becomes its own mean over one lamp cell,
    // so a belly seen from a kilometre keeps exactly the light it had close.
    let li = floor(ua / cc_skybridges_LAMP_PITCH);
    let lu = ua - (li + 0.5) * cc_skybridges_LAMP_PITCH;
    let dl = length(vec2<f32>(lu, cross_m));
    let laa = clamp(0.5 * fp_eff, 0.02, 0.40);
    let lamp = 1.0 - smoothstep(cc_skybridges_LAMP_R - laa,
                                cc_skybridges_LAMP_R + laa, dl);
    let lamp_mean = 3.14159265 * cc_skybridges_LAMP_R * cc_skybridges_LAMP_R
                    / (cc_skybridges_LAMP_PITCH * 2.0 * max(s.half, 0.5));
    let lamp_l = mix(lamp, lamp_mean,
                     smoothstep(cc_skybridges_LOD_LAMP.x,
                                cc_skybridges_LOD_LAMP.y, fp_eff));
    return e + tint * (cc_skybridges_LAMP_RAD * lamp_l);
}

// The truss's central web. It is only 0.6 m thick, but its SIDE face is the
// whole span by the whole handrail height — the largest surface this variant
// owns — so it is where the truss has to actually look like a truss. The
// Warren lattice is shaded rather than built: members catch the district's
// fill, and the bays between them go to black, which is very nearly what
// looking through a truss at a night street gives you anyway.
fn cc_skybridges_truss(h: CityHit, s: cc_skybridges_Span, tint: vec3<f32>,
                       fill: vec3<f32>, moon: vec3<f32>, fp_eff: f32)
        -> vec3<f32> {
    let lit = 0.10 * fill + 0.010 * tint
              + 0.050 * CITY_MOONLIGHT * max(dot(h.normal, moon), 0.0);
    if (!s.ok) {
        return lit;
    }
    let ua = select(h.pos.x, h.pos.y, s.axis == 1) - s.lo;
    let pv = h.pos.z - s.z - cc_skybridges_DECK_T;
    let aa = min(0.5 * fp_eff, 0.25);
    let hgt = cc_skybridges_GLASS_H;

    // Warren lattice: a triangle wave of diagonals between top and bottom
    // chords, with a vertical post where the diagonals meet.
    let f = ua / cc_skybridges_TRUSS_PITCH;
    let zig = abs(2.0 * (f - floor(f)) - 1.0);
    let diag = 1.0 - smoothstep(cc_skybridges_MEMBER_T - aa,
                                cc_skybridges_MEMBER_T + aa,
                                abs(pv - hgt * zig));
    let vu = abs(ua - round(f) * cc_skybridges_TRUSS_PITCH);
    let post = 1.0 - smoothstep(cc_skybridges_POST_T - aa,
                                cc_skybridges_POST_T + aa, vu);
    let chord = 1.0 - smoothstep(cc_skybridges_CHORD_T - aa,
                                 cc_skybridges_CHORD_T + aa,
                                 min(pv, hgt - pv));
    let mem = mix(max(max(diag, post), chord), cc_skybridges_TRUSS_COVER,
                  smoothstep(cc_skybridges_LOD_RIB.x,
                             cc_skybridges_LOD_RIB.y, fp_eff));
    let e = lit * mem;

    let li = floor(ua / cc_skybridges_STRING_PITCH);
    let lu = ua - (li + 0.5) * cc_skybridges_STRING_PITCH;
    // The string hangs just under the top chord.
    let dl = length(vec2<f32>(lu, pv - cc_skybridges_GLASS_H * 0.82));
    let laa = clamp(0.5 * fp_eff, 0.02, 0.40);
    let lamp = 1.0 - smoothstep(cc_skybridges_STRING_R - laa,
                                cc_skybridges_STRING_R + laa, dl);
    let lamp_mean = 3.14159265 * cc_skybridges_STRING_R
                    * cc_skybridges_STRING_R
                    / (cc_skybridges_STRING_PITCH * cc_skybridges_GLASS_H);
    let lamp_l = mix(lamp, lamp_mean,
                     smoothstep(cc_skybridges_LOD_LAMP.x,
                                cc_skybridges_LOD_LAMP.y, fp_eff));
    return e + tint * (cc_skybridges_STRING_RAD * lamp_l);
}

fn cc_skybridges_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let moon = u.sun_dir.xyz;
    let fill = CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(h.pos.xy, 3.0));
    let k = h.kind - cc_skybridges_KIND_BASE;
    let axis = k & 1;
    let ty = k >> 1;
    let cell = u.ocean_params.x;

    // Rebuild the exact span from the hit. Containment guarantees the tube
    // straddles one block boundary and stays in one row, so rounding the hit
    // on the bridge's axis and flooring it across names the two cells.
    var clo: vec2<i32>;
    var stp: vec2<i32>;
    if (axis == 0) {
        clo = vec2<i32>(i32(round(h.pos.x / cell)) - 1,
                        i32(floor(h.pos.y / cell)));
        stp = vec2<i32>(1, 0);
    } else {
        clo = vec2<i32>(i32(floor(h.pos.x / cell)),
                        i32(round(h.pos.y / cell)) - 1);
        stp = vec2<i32>(0, 1);
    }
    let p0 = pcg2d(cc_skybridges_eid(clo, axis));
    let s = cc_skybridges_build(clo, axis, city_cell(clo),
                                city_cell(clo + stp), city_rand4(p0));

    // The house colour: mostly tungsten corridors, some fluorescent, the
    // odd cyan one. Warm-biased so a bridge reads against the cooler
    // curtain-wall towers it usually connects.
    let bh = city_rand4(s.seed);
    let tint = city_window_color(bh.x * 0.92, 0.75);

    // The footprint a slanted face actually presents: a tube seen end-on
    // smears its detail over far fewer pixels than its distance suggests.
    let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);

    if (ty == 0 || !s.ok) {
        return cc_skybridges_shell(h, s, tint, fill, moon, fp_eff);
    }
    if (ty == 2) {
        return cc_skybridges_truss(h, s, tint, fill, moon, fp_eff);
    }

    // Band coordinates in metres: along the span from wall A, and up from
    // the deck surface.
    let ua = select(h.pos.x, h.pos.y, axis == 1) - s.lo;
    let pv = h.pos.z - s.z - cc_skybridges_DECK_T;

    let ib = i32(floor(ua / cc_skybridges_MULLION_PITCH));
    let sh = city_rand4(vec2<u32>(s.seed.x ^ (bitcast<u32>(ib) * 0x9e3779b9u),
                                  s.seed.y ^ 0x85ebca6bu));
    let bright = 0.55 + 0.90 * sh.x;
    let bay = tint * (cc_skybridges_RADIANCE * bright);

    // Octave 2: the whole ribbon, one colour. Bays average out last.
    let e_far = tint * (cc_skybridges_RADIANCE * cc_skybridges_MEAN_COVER
                        * cc_skybridges_MEAN_COMP);
    // Octave 1: bays still separable, mullions and figures gone.
    let lit_seg = select(1.0, 0.0, sh.y < cc_skybridges_DARK_SEG);
    let e_bay = bay * (cc_skybridges_MEAN_COVER / (1.0 - cc_skybridges_DARK_SEG)
                       * cc_skybridges_MEAN_COMP * lit_seg);

    // Octave 0: the glazing itself.
    let pu = ua - f32(ib) * cc_skybridges_MULLION_PITCH;
    let aa_u = min(0.5 * fp_eff, 0.35);
    let web = min(pu, cc_skybridges_MULLION_PITCH - pu);
    let mull = smoothstep(cc_skybridges_MULLION_HALF - aa_u - 0.02,
                          cc_skybridges_MULLION_HALF + aa_u + 0.02, web);
    let aa_v = min(0.5 * fp_eff, 0.20);
    let glazed = smoothstep(cc_skybridges_GLASS_V0 - aa_v,
                            cc_skybridges_GLASS_V0 + aa_v + 0.03, pv)
               * (1.0 - smoothstep(cc_skybridges_GLASS_V1 - aa_v - 0.03,
                                   cc_skybridges_GLASS_V1 + aa_v, pv));
    var cover = mix(cc_skybridges_SPANDREL, 1.0, mull * glazed);
    if (sh.z < cc_skybridges_FIG_FRAC) {
        let f_lod = smoothstep(cc_skybridges_LOD_FIG.x,
                               cc_skybridges_LOD_FIG.y, fp_eff);
        if (f_lod < 1.0) {
            let fh = city_rand4(s.seed ^ vec2<u32>(bitcast<u32>(ib)
                                                   * 0x27d4eb2fu, 0x165667b1u));
            let m = cc_skybridges_figure(fh, vec2<f32>(pu, pv), fp_eff)
                    * (1.0 - f_lod);
            cover = cover * mix(1.0, cc_skybridges_BODY_T, m);
        }
    }
    let e_near = bay * (cover * lit_seg);

    let b1 = smoothstep(cc_skybridges_LOD_MUL.x, cc_skybridges_LOD_MUL.y,
                        fp_eff);
    let b2 = smoothstep(cc_skybridges_LOD_SEG.x, cc_skybridges_LOD_SEG.y,
                        fp_eff);
    return mix(e_near, mix(e_bay, e_far, b2), b1)
           + 0.02 * fill;
}

// --- component: streetlife (streetlife.wgsl) ---
// streetlife — the city at eye level: the poles the light already comes
// from, the cars parked under them, and the bins behind those.
//
// The core already lights the asphalt: city_street_pools puts a sodium pool
// every CITY_LAMP_SPACING (26 m) along two lines set CITY_LAMP_OFFSET (2.5 m)
// in from each block edge, three times brighter on avenues. Those pools had
// no lamps over them. This component puts the lamps there — on exactly that
// lattice, derived from the same arithmetic, so the light and its source
// cannot drift apart. Everything else here is what stands in that light.
//
// LAYOUT. A cell owns four KERBS, one per plot edge. Each kerb carries, in
// its own across-coordinate:
//   * the lamp line, at the block edge +- CITY_LAMP_OFFSET — the core's
//     lattice, not a new one. A mast at every lattice point inside the cell,
//     with the luminaire arm reaching AWAY from the plot, out over the
//     roadway, which is where a real one hangs.
//   * the parking line, CAR_PARK_OFF in from the plot edge (the kerb proper).
//     Slots on a 7 m lattice in world space, so a run of cars lines up
//     across cell boundaries; ~35% occupied, scaled by the cascade the way
//     the streetlights and the air traffic are, and ~10% of the rest hold a
//     dumpster shoved against the wall instead.
// Every prop lives strictly inside the cell that draws it, because the DDA
// tests a cell only from the side it enters (the rule aircars states). For
// merged 2x2 superblocks the plot edge can fall in a sibling's column; that
// kerb simply has no parking, and the sibling whose column does contain it
// draws those cars.
//
// THE CARS ARE THE POINT. A parked car is the one object in this city the
// camera can stand next to, and boxes with circles on them would say so
// immediately. Inside its bounding box, at fp < CAR_SDF_FP, a car is
// sphere-traced from a real SDF: a tapering lifting-body hull, smooth-min'd
// to a cabin bubble set back from the nose, wheels in cut arches (or a hover
// plenum and intake scallops, 38% of them), mirror stalks, wiper blades
// across the base of the windshield, and lamp housings and a nose intake cut
// as recesses. Detail to wiper-blade level and no further (Thomas,
// 2026-08-20): door seams, shut lines and a rocker crease are SHADING bands
// in the hull's own frame, which is where panel lines belong — no badges, no
// text, nothing that would be noise at 10 m.
//
// Beyond CAR_SDF_FP the hull falls back to two axis-aligned boxes cut to the
// same silhouette (hull + greenhouse); the 5-degree yaw jitter goes with it,
// which at that footprint is a sub-pixel corner.
//
// LIGHT. One lamp, straight overhead, is the entire lighting situation on a
// night street, so the shading is built around it: the incident estimate is
// the core's own asphalt formula (pool x district scale x CITY_LAMP_RADIANCE
// x CITY_LAMP_COLOR) and the direction is the vector to the nearest lamp
// HEAD, recovered from the same lattice. Curvature reads through the
// specular lobe sliding along the hull shoulder — Lambert alone on a dark
// paint at night is nearly flat, and the glint is what says "this surface is
// round". 30% of cars carry an underglow strip; the hue draw is aircars'
// 60/25/15 cyan/magenta/amber, so ground and air read as one traffic system.
//
// COST. In order, each gate cheaper than the one it protects:
//   1. the segment's z range against 12 m — no hash, two multiply-adds, and
//      it rejects the whole hook for every pixel looking at a facade, a roof,
//      a cloud or the sky, which is most of them;
//   2. one slab test per kerb (4), spanning lamp line to parking line;
//   3. inside a live kerb, a second z test that drops the ground props
//      (everything under 1.7 m) for a segment that only passes through the
//      lamp heads;
//   4. the along-range of the segment picks at most 2 pole slots and 4
//      parking slots, walked from the near end with an early break, each one
//      hash-gated before any box test;
//   5. the SDF runs only for a ray that has already entered a car's bounding
//      box at fp < 0.5 m/px.
// Worst case for a ray running the length of a kerb at eye level is 1 slab +
// 2 poles x 2 boxes + 4 slots x 1 box = 9 box tests for that kerb; the three
// other kerbs of the same cell are crossed, not followed, and cost 1-3 each.
// t1 is narrowed by every hit found so far, so later kerbs see shorter
// segments than earlier ones.

// --- the lamp lattice (the core's, restated) --------------------------------
const cc_streetlife_Z_GATE: f32 = 12.0;   // whole-hook z reject
const cc_streetlife_POLE_H: f32 = 9.0;    // mast top
const cc_streetlife_POLE_TOP: f32 = 9.15; // slab ceiling
const cc_streetlife_MAST_R: f32 = 0.085;
const cc_streetlife_ARM_Z: f32 = 8.72;    // the arm's own centre height
const cc_streetlife_ARM_HZ: f32 = 0.065;
const cc_streetlife_ARM_REACH: f32 = 1.35;
const cc_streetlife_HEAD_Z: f32 = 8.56;   // luminaire centre
const cc_streetlife_HEAD_HZ: f32 = 0.115;
const cc_streetlife_HEAD_HA: f32 = 0.17;  // half-extent along the kerb
const cc_streetlife_HEAD_HC: f32 = 0.40;  // half-extent along the arm
// A sodium luminaire seen from underneath is the brightest thing on the
// street by a wide margin — an order over the pool it throws (0.7) and twice
// a lit window (3.5). The housing above it is opaque and near-black, which
// is what stops a row of lamps reading as floating lozenges.
const cc_streetlife_HEAD_RAD: f32 = 6.0;
const cc_streetlife_HEAD_COLOR: vec3<f32> = vec3<f32>(1.0, 0.52, 0.18);
// The housing is a box, so the lens has to be found in the shader: the lower
// lip of each side face is the glass, the rest of the side and the whole top
// are painted aluminium. At a uniform 0.34 of RAD every side face clipped to
// white along with the underside, and a luminaire whose cowl is as bright as
// its lamp is the floating lozenge this was supposed to avoid.
const cc_streetlife_HEAD_SIDE: f32 = 0.62;  // lens lip, fraction of RAD
const cc_streetlife_HEAD_COWL: f32 = 0.030; // painted housing, same units
const cc_streetlife_HEAD_LIP: f32 = 0.072;  // how far the lens runs up (m)
// Galvanised steel, seen at night, lit by a lamp standing on its own axis.
// The Lambert term is deliberately tiny and the grazing edge does the work:
// at MAST_ALB 0.10 the first pass rendered a flat gold bar that read as a
// wooden telegraph pole, because a diffuse fraction of a clipped sodium road
// is a clipped sodium pole. A dark face between two bright edges is both the
// correct photometry for a cylinder under an axial source and the only thing
// that says "round" at this radius (see cc_streetlife_pole_shade).
const cc_streetlife_MAST_FILL: f32 = 0.06;
const cc_streetlife_MAST_ALB: f32 = 0.026;
const cc_streetlife_MAST_EDGE: f32 = 0.105;

// --- parking ----------------------------------------------------------------
// The parking line is set from the LAMP line, not from the plot edge, and
// that is a correction the first renders forced. Parking at the kerb is where
// cars belong on a minor street — the kerb is 3.5 m outboard of the lamps and
// well inside the pool — but an avenue's plot edge is 16 m out (the avenue
// gets CITY_AVENUE_EXTRA on both sides while its lamp lines stay 2.5 m off
// the block edge, so the lamps are effectively a median). Cars parked at an
// avenue kerb sit 12 m from the nearest lamp, where the pool has fallen by
// e^-3.5, and rendered as invisible black shapes on black tarmac. So: a lane
// PARK_LANE outboard of the lamps, pulled in to the kerb whenever the kerb is
// closer than that. Light and source agree by construction, and the kerb slab
// gets narrow enough to be a cheap reject as a side effect.
const cc_streetlife_PARK_LANE: f32 = 3.2;
const cc_streetlife_PARK_KERB: f32 = 0.25;  // clearance from the plot edge
const cc_streetlife_SLOT: f32 = 7.0;
const cc_streetlife_OCC: f32 = 0.35;
const cc_streetlife_BIN_CUT: f32 = 0.945;  // draws above this are dumpsters
// Dumpsters belong at block corners — the alley mouth, the service door —
// far more than they belong in a parking bay, so most of them are placed
// there instead (cc_streetlife_corner_prop) and the kerb keeps only the
// occasional one. A quarter of the corners of a built plot carry one.
const cc_streetlife_CORNER_BIN: f32 = 0.25;
const cc_streetlife_DENS_LO: f32 = 0.55;
const cc_streetlife_DENS_HI: f32 = 1.35;
const cc_streetlife_DENS_START: f32 = 0.005;
const cc_streetlife_DENS_FULL: f32 = 0.070;
const cc_streetlife_YAW: f32 = 0.0873;     // +- 5 degrees
const cc_streetlife_ALONG_JIT: f32 = 0.60;
const cc_streetlife_LAT_JIT: f32 = 0.16;

// Bounding box, in the car's own (along, across) frame. Wide enough for the
// hull yawed 5 degrees and for the mirror stalks; nothing but the reject test
// ever sees it, because the far silhouette is the two proxy boxes below.
const cc_streetlife_BB_A: f32 = 2.60;
const cc_streetlife_BB_C: f32 = 1.36;
const cc_streetlife_BB_Z: f32 = 1.74;

// --- the hull ---------------------------------------------------------------
const cc_streetlife_HL: f32 = 2.32;    // hull half-length
const cc_streetlife_HW: f32 = 0.98;    // hull half-width, at its widest
const cc_streetlife_SILL: f32 = 0.26;  // hull underside
const cc_streetlife_BELT: f32 = 0.94;  // shoulder line
const cc_streetlife_ROOF: f32 = 1.46;
const cc_streetlife_HOVER_CUT: f32 = 0.65;  // draws above this hover
const cc_streetlife_HOVER_LIFT: f32 = 0.14;
const cc_streetlife_AXLE_X: f32 = 1.46;
const cc_streetlife_AXLE_Z: f32 = 0.35;
const cc_streetlife_TRACK: f32 = 0.82;   // wheel centreplane, |y|
const cc_streetlife_TYRE_R: f32 = 0.34;
const cc_streetlife_TYRE_HW: f32 = 0.135; // half the tread width
const cc_streetlife_ARCH_R: f32 = 0.43;
const cc_streetlife_ARCH_HW: f32 = 0.31;  // the arch cuts only the flank skin
const cc_streetlife_ARCH_Z: f32 = 0.29;
const cc_streetlife_RIM_R: f32 = 0.255;

// The SDF is approximate — the plan-form taper and the falling deck make the
// hull's half-extents functions of x, so |grad| runs above 1 near the nose.
// The march steps this fraction of the reported distance, which is what keeps
// it from stepping through the skin.
const cc_streetlife_STEP: f32 = 0.72;
// The march budget, and the reason it is not a compile-time constant. A
// literal bound here is unrolled by the driver, and because cell_props is
// inlined into a 512-iteration DDA loop the unrolled body costs occupancy on
// every city pixel in the frame — including the ones nowhere near a car. The
// measurement: 32 versus 12 iterations moved the `aerial` view, which never
// admits a single car to the SDF at all, from 0.52 s to 0.29 s. Selecting
// between two counts at run time keeps the loop rolled, and doubles as
// honest LOD: a car ten pixels across does not need a 34-step trace.
const cc_streetlife_ITERS: i32 = 34;
const cc_streetlife_ITERS_FAR: i32 = 16;
const cc_streetlife_ITER_FP: f32 = 0.09;

// --- LOD --------------------------------------------------------------------
// Below this footprint a car is sphere-traced; above it, two boxes. 0.5 m/px
// puts a car at ten pixels, which is where a curved shoulder stops being a
// thing you can see and starts being a thing you can only infer.
const cc_streetlife_CAR_SDF_FP: f32 = 0.50;
const cc_streetlife_FINE_FP: f32 = 0.10;   // wipers, and sharp seams
// Where cars and poles stop being traced at all, and where their emission
// has already ramped to zero so nothing pops. Set by cost, not by the eye:
// at fp 2.6 a car is under two pixels long and the sodium road behind it is
// what that pixel was going to be anyway, while every cell inside
// CITY_PROP_RANGE pays for the test. A pole is thinner but taller, and its
// luminaire is the brightest thing on the street, so it runs further.
const cc_streetlife_CAR_FAR_FP: f32 = 2.6;
const cc_streetlife_CAR_FAR_FADE: f32 = 1.6;
const cc_streetlife_POLE_FAR_FP: f32 = 4.0;
const cc_streetlife_POLE_FAR_FADE: f32 = 2.4;
// Seams, shut lines and lamp dots hand over to their own means here; past
// LOD_FULL a car's paint is one colour and its lamp is that lamp's mean over
// the face it sits on, which is what a long lens does to a parked car.
const cc_streetlife_DETAIL_LOD: vec2<f32> = vec2<f32>(0.06, 0.26);
const cc_streetlife_LAMP_LOD: vec2<f32> = vec2<f32>(0.10, 0.45);
// The population edge at CITY_PROP_RANGE, approached rather than stepped.
const cc_streetlife_FADE_START: f32 = 0.92;

// --- car light --------------------------------------------------------------
const cc_streetlife_GLOW_FRAC: f32 = 0.32; // cars carrying an underglow
const cc_streetlife_GLOW_RAD: f32 = 1.1;
const cc_streetlife_LAMP_RAD: f32 = 1.9;
// cc_streetlife_pool returns the core's own asphalt RADIANCE — what city_shade
// emits for the road, with no albedo applied at all. Everything here is a
// fraction of THAT, not of an irradiance, and the fractions are small on
// purpose. A downtown avenue's pool runs near radiance 10, and at exposure 6
// under a Reinhard with white point 15 anything past ~2.5 is white: the road
// under these cars is already clipped, so the whole readable range for a
// painted panel is radiance 0.02 to 0.4. The first pass shaded cars at a
// physical-looking reflectance of the pool and produced pale ceramic
// bathtubs. Dark cars against a hot sodium road is both the correct
// photometry and the shot.
const cc_streetlife_PAINT_GAIN: f32 = 0.10;
const cc_streetlife_GLOSS: f32 = 1.2;
const cc_streetlife_SHEEN: f32 = 0.025;
const cc_streetlife_GLASS_GLOSS: f32 = 4.0;
const cc_streetlife_GLASS_ROAD: f32 = 0.055;
const cc_streetlife_ROAD_BOUNCE: f32 = 0.055;
const cc_streetlife_TYRE_ALB: f32 = 0.0045;
const cc_streetlife_RIM_ALB: f32 = 0.055;
const cc_streetlife_BIN_ALB: f32 = 0.10;

// ---------------------------------------------------------------------------
// SDF primitives
// ---------------------------------------------------------------------------

fn cc_streetlife_rbox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec3<f32>(r);
    return length(max(q, vec3<f32>(0.0)))
         + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Capped cylinder whose axis is y (the car's across direction: wheels, arches
// and intake scallops all share it).
fn cc_streetlife_cyl_y(p: vec3<f32>, rad: f32, h: f32) -> f32 {
    let d = vec2<f32>(length(vec2<f32>(p.x, p.z)) - rad, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn cc_streetlife_seg(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, rad: f32)
        -> f32 {
    let pa = p - a;
    let ba = b - a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t) - rad;
}

fn cc_streetlife_smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn cc_streetlife_smax(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

fn cc_streetlife_miss(ci: vec2<i32>) -> CityHit {
    return CityHit(false, 1e30, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0),
                   0, ci);
}

fn cc_streetlife_nearer(a: CityHit, b: CityHit) -> CityHit {
    if (b.hit && (!a.hit || b.t < a.t)) {
        return b;
    }
    return a;
}

// ---------------------------------------------------------------------------
// The kerb: where a cell's four edges put their lamp line and parking line
// ---------------------------------------------------------------------------

struct cc_streetlife_Side {
    ax: i32,          // 0 = the kerb runs along x, 1 = along y
    lamp_c: f32,      // lamp line, across coordinate
    park_c: f32,      // parking line, across coordinate
    plot_sign: f32,   // which way the plot lies from the kerb
    a_min: f32, a_max: f32,   // the cell's extent along the kerb
    c_min: f32, c_max: f32,   // the cell's extent across it
    park_ok: bool,
}

fn cc_streetlife_side(ci: vec2<i32>, cc: CityCell, side: i32)
        -> cc_streetlife_Side {
    let cellm = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cellm;
    let cmax = cmin + vec2<f32>(cellm);
    var s: cc_streetlife_Side;
    var plot_c: f32;
    if (side == 0) {          // the plot's -x edge; kerb runs along y
        s.ax = 1;
        s.lamp_c = cmin.x + city_lamp_inset(ci.x);
        plot_c = cc.plot_min.x;
        s.plot_sign = 1.0;
    } else if (side == 1) {   // +x edge
        s.ax = 1;
        s.lamp_c = cmax.x - city_lamp_inset(ci.x + 1);
        plot_c = cc.plot_max.x;
        s.plot_sign = -1.0;
    } else if (side == 2) {   // -y edge; kerb runs along x
        s.ax = 0;
        s.lamp_c = cmin.y + city_lamp_inset(ci.y);
        plot_c = cc.plot_min.y;
        s.plot_sign = 1.0;
    } else {                  // +y edge
        s.ax = 0;
        s.lamp_c = cmax.y - city_lamp_inset(ci.y + 1);
        plot_c = cc.plot_max.y;
        s.plot_sign = -1.0;
    }
    // Outboard of the lamps by PARK_LANE, or hard against the kerb if the
    // kerb is nearer than that (see the constant's note).
    let gap = s.plot_sign * (plot_c - s.lamp_c);
    let off = min(cc_streetlife_PARK_LANE,
                  gap - cc_streetlife_BB_C - cc_streetlife_PARK_KERB);
    s.park_c = s.lamp_c + s.plot_sign * off;
    s.a_min = select(cmin.y, cmin.x, s.ax == 0);
    s.a_max = select(cmax.y, cmax.x, s.ax == 0);
    s.c_min = select(cmin.x, cmin.y, s.ax == 0);
    s.c_max = select(cmax.x, cmax.y, s.ax == 0);
    // A merged superblock's plot edge can sit in a sibling's column; that
    // kerb keeps its lamps (they are on the cell's own lattice) and loses its
    // parking to whichever member owns the ground.
    s.park_ok = off > 1.6
             && s.park_c > s.c_min + cc_streetlife_BB_C
             && s.park_c < s.c_max - cc_streetlife_BB_C;
    return s;
}

// The nearest lamp HEAD to a street point, on the core's own lattice: the
// four candidates city_street_pools sums over, and the one that wins. Used
// as the light direction for everything this component shades.
fn cc_streetlife_nearest_lamp(p: vec2<f32>) -> vec3<f32> {
    let cellm = u.ocean_params.x;
    let bx = round(p.x / cellm) * cellm;
    let by = round(p.y / cellm) * cellm;
    let inx = city_lamp_inset(i32(round(p.x / cellm)));
    let iny = city_lamp_inset(i32(round(p.y / cellm)));
    let lx = round(p.x / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
    let ly = round(p.y / CITY_LAMP_SPACING) * CITY_LAMP_SPACING;
    var best = vec2<f32>(bx - inx, ly);
    var bd = 1e30;
    for (var k: i32 = 0; k < 4; k = k + 1) {
        var c: vec2<f32>;
        if (k == 0) {
            c = vec2<f32>(bx - inx, ly);
        } else if (k == 1) {
            c = vec2<f32>(bx + inx, ly);
        } else if (k == 2) {
            c = vec2<f32>(lx, by - iny);
        } else {
            c = vec2<f32>(lx, by + iny);
        }
        let d = dot(c - p, c - p);
        if (d < bd) {
            bd = d;
            best = c;
        }
    }
    return vec3<f32>(best, cc_streetlife_HEAD_Z);
}

// The core's own asphalt radiance at a street point, reused verbatim as the
// incident estimate for anything standing on it. Sharing the formula is the
// point: a car in a pool is exactly as lit as the tarmac it is parked on.
fn cc_streetlife_pool(p: vec2<f32>) -> vec3<f32> {
    let district = city_glow_sample(p, 2.0);
    let scale = 0.20 + 2.2 * smoothstep(0.02, 0.45, district);
    return CITY_LAMP_COLOR
         * (CITY_LAMP_RADIANCE * scale * city_street_pools(p));
}

fn cc_streetlife_fill(p: vec3<f32>) -> vec3<f32> {
    return CITY_SKYGLOW * (0.5 + 1.5 * city_glow_sample(p.xy, 3.0));
}

// ---------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------

struct cc_streetlife_Prop {
    ok: bool,
    bin: bool,
    ctr: vec2<f32>,   // ground point under the prop's centre
    fwd: vec2<f32>,   // the prop's own forward, yaw included
    rgt: vec2<f32>,
    r: vec4<f32>,
}

fn cc_streetlife_no_prop() -> cc_streetlife_Prop {
    return cc_streetlife_Prop(false, false, vec2<f32>(0.0),
                              vec2<f32>(1.0, 0.0), vec2<f32>(0.0, -1.0),
                              vec4<f32>(0.0));
}

fn cc_streetlife_slot_draw(ci: vec2<i32>, side: i32, j: i32) -> vec4<f32> {
    return city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x9e3779b9u + u32(side) * 0x2545f491u
            + bitcast<u32>(j) * 0x85ebca6bu + 0x51ed270bu,
        bitcast<u32>(ci.y) * 0xc2b2ae35u + u32(side) * 0x27d4eb2fu
            + bitcast<u32>(j) * 0x165667b1u + 0x9e3779b9u));
}

// Whatever occupies slot `j` of kerb `side` — deterministic in (cell, side,
// slot) alone, so the shader re-derives a car from its hit position without
// anything being smuggled through CityHit.
fn cc_streetlife_prop(ci: vec2<i32>, cc: CityCell, s: cc_streetlife_Side,
                      side: i32, j: i32) -> cc_streetlife_Prop {
    if (!s.park_ok) {
        return cc_streetlife_no_prop();
    }
    let r = cc_streetlife_slot_draw(ci, side, j);
    let dens = mix(cc_streetlife_DENS_LO, cc_streetlife_DENS_HI,
                   smoothstep(cc_streetlife_DENS_START,
                              cc_streetlife_DENS_FULL, cc.density));
    let occ = cc_streetlife_OCC * dens;
    let bin = r.x > cc_streetlife_BIN_CUT;
    if (r.x >= occ && !bin) {
        return cc_streetlife_no_prop();
    }
    // Slot centre on the world lattice, so a run of cars lines up across the
    // cell boundary rather than restarting inside every block.
    let along = (f32(j) + 0.5) * cc_streetlife_SLOT
              + (r.y - 0.5) * 2.0 * cc_streetlife_ALONG_JIT;
    if (along - cc_streetlife_BB_A < s.a_min
        || along + cc_streetlife_BB_A > s.a_max) {
        return cc_streetlife_no_prop();
    }
    var across = s.park_c + (r.z - 0.5) * 2.0 * cc_streetlife_LAT_JIT;
    if (bin) {
        // Bins get shoved against the wall, not left at the kerb.
        across = s.park_c + s.plot_sign * 0.85;
    }
    var base = vec2<f32>(1.0, 0.0);
    if (s.ax == 1) {
        base = vec2<f32>(0.0, 1.0);
    }
    let perp = vec2<f32>(-base.y, base.x);
    var fwd = base;
    if (!bin) {
        let yaw = (r.w - 0.5) * 2.0 * cc_streetlife_YAW;
        fwd = base * cos(yaw) + perp * sin(yaw);
        if (fract(r.y * 17.31) > 0.5) {
            fwd = -fwd;
        }
    }
    var p: cc_streetlife_Prop;
    p.ok = true;
    p.bin = bin;
    p.ctr = base * along + perp * across;
    p.fwd = fwd;
    p.rgt = vec2<f32>(fwd.y, -fwd.x);
    p.r = r;
    return p;
}

// ---------------------------------------------------------------------------
// The car SDF
// ---------------------------------------------------------------------------
//
// Local frame: +x forward, +y left, z up from the road surface. Everything
// symmetric about the centreline is evaluated once on abs(y), and the wheels
// and lamp housings once on abs(x) too — four wheels for the price of one
// cylinder.

// Two draws of shape per car: an overall scale, and where the cabin sits
// fore and aft. Between them a row of parked cars stops being one model
// repeated — a short car with the cabin back is a coupe, a long one with it
// forward is a saloon, and the eye reads the difference before it reads any
// panel line.
fn cc_streetlife_car_shape(r: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(0.92 + 0.13 * fract(r.z * 7.71),
                     -0.30 + (fract(r.w * 11.37) - 0.5) * 0.34);
}

// Which cars hover. This must NOT be read off r.x, and that it was is the one
// outright bug the salvaged draft carried: r.x is the OCCUPANCY draw, and a
// slot holds a car only where r.x < occ — at most 0.47 even downtown — so a
// hover test of `r.x > 0.62` was unreachable in every cell of the city. Nobody
// would ever have seen it fail; the plenum, the skirt and the intake scallops
// simply never ran. An independent draw off the other three components gives
// the ~35% the file always claimed.
fn cc_streetlife_is_hover(r: vec4<f32>) -> bool {
    return fract(r.y * 43.17 + r.z * 7.31 + r.w * 2.53)
           > cc_streetlife_HOVER_CUT;
}

fn cc_streetlife_car_sdf(p: vec3<f32>, r: vec4<f32>, fine: bool) -> f32 {
    let hover = cc_streetlife_is_hover(r);
    let lift = select(0.0, cc_streetlife_HOVER_LIFT, hover);
    let sh = cc_streetlife_car_shape(r);
    let q = vec3<f32>(p.x, p.y, p.z - lift) / sh.x;
    let cab_c = sh.y;
    let off = cab_c + 0.30;

    // Plan-form: the hull narrows toward both ends, hard at the very tips.
    let xn = clamp(q.x / cc_streetlife_HL, -1.0, 1.0);
    let xn2 = xn * xn;
    let xn4 = xn2 * xn2;
    let hw = cc_streetlife_HW * (1.0 - 0.34 * xn4);
    // Profile: the deck falls away toward the nose and, less, toward the
    // tail. A flat deck is what makes a box read as a box.
    let drop = select(0.08, 0.16, xn > 0.0);
    let deck = cc_streetlife_BELT - drop * xn2;
    let hc = 0.5 * (deck + cc_streetlife_SILL);
    let hh = 0.5 * (deck - cc_streetlife_SILL);
    var d = cc_streetlife_rbox(vec3<f32>(q.x, q.y, q.z - hc),
                               vec3<f32>(cc_streetlife_HL, hw, hh), 0.22);

    // The greenhouse: inset from the shoulder, set back from the nose, and
    // blended only just enough to give it a fillet. A soft blend here is what
    // turned the first pass into a loaf — the shoulder line has to survive.
    // Rounded hard, this reads as a bubble stuck on the deck rather than as
    // a cabin: the roof has to be flat enough to be a roof and the side glass
    // near enough to vertical to be glass.
    let cx = q.x - cab_c;
    var cab = cc_streetlife_rbox(
        vec3<f32>(cx, q.y, q.z - 1.20),
        vec3<f32>(0.92, 0.74, 0.26), 0.10);
    // TUMBLEHOME. The side glass leans in toward the roof, so the greenhouse
    // is a tapered turret rather than a box, and its widest point is the belt
    // line where it meets the shoulder. Without this the cabin read as a loaf
    // of bread set on the deck — full-width, vertical-sided, visibly a second
    // box. One slanted half-space does the whole job; the 0.958 is 1/|grad|,
    // which keeps the march from stepping through the lean.
    cab = cc_streetlife_smax(
        cab, 0.958 * (abs(q.y) - 0.74 + 0.30 * (q.z - 1.02)), 0.07);
    // Windshield and backlight rake: two half-spaces that take the front and
    // rear off the greenhouse, so the cabin is a cabin and not a second box.
    // Cut into the cabin alone — applied to the whole hull the front plane
    // would take the bonnet with it. The planes are placed to MEET THE BELT,
    // not to clip a corner: the first pass's intercepts put the start of the
    // windshield at cx 1.18, outside the cabin box entirely, so the rake took
    // only the top corner and everything below it stayed the box's own
    // vertical wall — which is exactly what the renders showed. Now the glass
    // starts within 5 mm of the shoulder and the roof comes out 1.14 m long
    // by 1.22 wide, which is a car; the first pass's would have been 0.66 by
    // 1.22, which is a bubble canopy.
    cab = cc_streetlife_smax(cab, 0.824 * cx + 0.567 * q.z - 1.339, 0.10);
    cab = cc_streetlife_smax(cab, -0.745 * cx + 0.667 * q.z - 1.361, 0.10);
    d = cc_streetlife_smin(d, cab, 0.11);

    // WHEELS AND ARCHES, both mirrored on abs(y) so there are four of them.
    //
    // The first pass mirrored only on abs(x) and gave the cylinders a
    // half-width of 0.86 — wider than the hull's own 0.98 half-width. That is
    // not four wheels, it is two drums spanning the full track, and the arch
    // that cut them free was 1.10 wide, which bored a tunnel clean through
    // the body. From the side the two errors cancelled and it read correctly;
    // head-on the drum showed under the nose as a hard dark bar with square
    // ends, wider than the car, and it is visible in every frontal frame the
    // draft ever produced. Splitting them fixes the frontal read and makes
    // the arch what an arch actually is: a cut in the outer skin of a flank.
    let qa = vec3<f32>(abs(q.x) - cc_streetlife_AXLE_X,
                       abs(q.y) - cc_streetlife_TRACK,
                       q.z - cc_streetlife_ARCH_Z);
    if (hover) {
        // A plenum instead of wheels, and the arches become intake scallops
        // cut into the flanks — the same silhouette cue read the other way.
        let skirt = cc_streetlife_rbox(
            vec3<f32>(q.x, q.y, q.z - 0.14),
            vec3<f32>(1.94, 0.80, 0.09), 0.08);
        d = cc_streetlife_smin(d, skirt, 0.13);
        let scallop = cc_streetlife_cyl_y(
            vec3<f32>(qa.x, qa.y - 0.10, qa.z), 0.30,
            cc_streetlife_ARCH_HW + 0.10);
        d = cc_streetlife_smax(d, -scallop, 0.05);
    } else {
        let arch = cc_streetlife_cyl_y(
            vec3<f32>(qa.x, qa.y - 0.13, qa.z),
            cc_streetlife_ARCH_R, cc_streetlife_ARCH_HW);
        d = cc_streetlife_smax(d, -arch, 0.035);
        let tyre = cc_streetlife_cyl_y(
            vec3<f32>(qa.x, qa.y, q.z - cc_streetlife_AXLE_Z),
            cc_streetlife_TYRE_R, cc_streetlife_TYRE_HW);
        d = min(d, tyre);
    }

    // Door mirrors, at the base of the A-pillar. A bare capsule reaching from
    // the cabin flank to the hull's widest point — which is what the first
    // pass had — renders as a dark bar floating clear of the car, because a
    // uniform 10 cm cylinder 30 cm long is a stick and nothing about it says
    // mirror. A mirror is a SHORT arm carrying a HOUSING, and the housing is
    // what the eye finds: a flat-backed pod, wider than it is tall.
    let qm = vec3<f32>(q.x - 0.74 - off, abs(q.y), q.z - 1.01);
    d = min(d, cc_streetlife_seg(qm, vec3<f32>(0.0, 0.60, 0.0),
                                 vec3<f32>(0.02, 0.76, 0.03), 0.032));
    d = min(d, cc_streetlife_rbox(qm - vec3<f32>(0.0, 0.855, 0.035),
                                  vec3<f32>(0.075, 0.095, 0.055), 0.038));

    // Lamp housings, cut as recesses at nose and tail, and an intake slot
    // low in the nose. The recess is a LETTERBOX — wider than tall — because
    // the lens the shader paints inside it is an ellipse of the same aspect,
    // and a round lens in a round hole is the headlight shape that made the
    // first pass read as a face with eyes.
    let ql = vec3<f32>(abs(q.x) - 2.15, abs(q.y) - 0.50, q.z - 0.71);
    d = cc_streetlife_smax(
        d, -cc_streetlife_rbox(ql, vec3<f32>(0.13, 0.27, 0.082), 0.04), 0.022);
    let qi = vec3<f32>(q.x - 2.06, q.y, q.z - 0.40);
    d = cc_streetlife_smax(
        d, -cc_streetlife_rbox(qi, vec3<f32>(0.16, 0.44, 0.055), 0.03), 0.03);

    if (fine) {
        // Wiper blades across the base of the windshield. At the footprint
        // that admits them a blade is two or three pixels of hard line on a
        // dark curved reflection, which is exactly what says "windscreen".
        let qw = vec3<f32>(q.x - off, abs(q.y), q.z);
        d = min(d, cc_streetlife_seg(qw, vec3<f32>(0.66, 0.07, 0.925),
                                     vec3<f32>(1.00, 0.52, 0.908), 0.021));
    }
    return d * sh.x;
}

// World point -> the car's own frame.
fn cc_streetlife_to_local(w: vec3<f32>, ctr: vec2<f32>, fwd: vec2<f32>,
                          rgt: vec2<f32>) -> vec3<f32> {
    let rel = w.xy - ctr;
    return vec3<f32>(dot(rel, fwd), dot(rel, rgt), w.z);
}

fn cc_streetlife_car_normal(pl: vec3<f32>, r: vec4<f32>, fine: bool, hh: f32)
        -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * hh;
    let n = vec3<f32>(1.0, -1.0, -1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.x, e.y, e.y), r, fine)
          + vec3<f32>(-1.0, -1.0, 1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.y, e.y, e.x), r, fine)
          + vec3<f32>(-1.0, 1.0, -1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.y, e.x, e.y), r, fine)
          + vec3<f32>(1.0, 1.0, 1.0)
            * cc_streetlife_car_sdf(pl + vec3<f32>(e.x, e.x, e.x), r, fine);
    return normalize(n);
}

// Sphere-trace one car inside a bounding box the ray has already entered.
// A miss inside the box is a real answer, not a failure: rays graze past a
// curved hull, and that is what makes the hull read as curved.
fn cc_streetlife_trace_car(o: vec3<f32>, dir: vec3<f32>, ta: f32, tb: f32,
                           v: cc_streetlife_Prop, ci: vec2<i32>, side: i32,
                           fp: f32) -> CityHit {
    let fine = fp < cc_streetlife_FINE_FP;
    let eps = max(0.0025, 0.30 * fp);
    let iters = select(cc_streetlife_ITERS, cc_streetlife_ITERS_FAR,
                       fp > cc_streetlife_ITER_FP);
    var t = ta + 0.001;
    var got = false;
    for (var i: i32 = 0; i < iters; i = i + 1) {
        let pl = cc_streetlife_to_local(o + t * dir, v.ctr, v.fwd, v.rgt);
        let d = cc_streetlife_car_sdf(pl, v.r, fine);
        if (d < eps) {
            got = true;
            break;
        }
        t = t + d * cc_streetlife_STEP;
        if (t > tb) {
            break;
        }
    }
    if (!got || t > tb) {
        return cc_streetlife_miss(ci);
    }
    let pos = o + t * dir;
    let pl = cc_streetlife_to_local(pos, v.ctr, v.fwd, v.rgt);
    let nl = cc_streetlife_car_normal(pl, v.r, fine, max(0.004, 0.4 * fp));
    let nw = vec3<f32>(v.fwd * nl.x + v.rgt * nl.y, nl.z);
    return CityHit(true, t, pos, nw, 102 + side, ci);
}

// The far read: hull and greenhouse as two axis-aligned boxes cut to the
// SDF's own silhouette. The yaw goes with the SDF, which at this footprint
// is a sub-pixel corner.
fn cc_streetlife_trace_car_box(o: vec3<f32>, inv_dir: vec3<f32>, dir: vec3<f32>,
                               t0: f32, t1: f32, v: cc_streetlife_Prop,
                               ci: vec2<i32>, side: i32) -> CityHit {
    let hover = cc_streetlife_is_hover(v.r);
    let lift = select(0.0, cc_streetlife_HOVER_LIFT, hover);
    let ea = abs(v.fwd) * 2.24 + abs(v.rgt) * 0.90;
    let eb = abs(v.fwd) * 1.06 + abs(v.rgt) * 0.76;
    let amin = vec3<f32>(v.ctr - ea, select(0.02, lift + 0.10, hover));
    let amax = vec3<f32>(v.ctr + ea, lift + 1.00);
    let bmin = vec3<f32>(v.ctr - eb + v.fwd * -0.24, lift + 0.94);
    let bmax = vec3<f32>(v.ctr + eb + v.fwd * -0.24,
                         lift + cc_streetlife_ROOF);
    var best = 1e30;
    var bmn = amin;
    var bmx = amax;
    let ha = city_box_hit(o, inv_dir, amin, amax);
    if (ha.x <= ha.y && ha.y > t0 && ha.x <= t1) {
        best = max(ha.x, t0);
    }
    let hb = city_box_hit(o, inv_dir, bmin, bmax);
    if (hb.x <= hb.y && hb.y > t0 && hb.x <= t1 && max(hb.x, t0) < best) {
        best = max(hb.x, t0);
        bmn = bmin;
        bmx = bmax;
    }
    if (best >= 1e30) {
        return cc_streetlife_miss(ci);
    }
    let pos = o + best * dir;
    return CityHit(true, best, pos, city_box_normal(pos, bmn, bmx),
                   102 + side, ci);
}

// A dumpster at one of the plot's four corners, shoved against the wall a
// couple of metres in from the corner itself. `k` selects the corner: bit 0
// the x wall, bit 1 the y end.
//
// Corners are a per-CELL question, not a per-kerb one, and that is why this
// does not live in cc_streetlife_kerb with the other ground props. The kerb's
// bounding slab runs from the lamp line to the parking line, and on an avenue
// the plot edge is thirteen metres outboard of that — a bin against the wall
// would sit entirely outside the slab the kerb tests, so a kerb-side test
// would silently never fire on exactly the streets that are widest and most
// visible.
//
// Placed strictly inside the drawing cell, per the DDA rule: a merged
// superblock's corner can fall in a sibling's column, and there the sibling
// that owns the ground draws it.
fn cc_streetlife_corner_prop(ci: vec2<i32>, cc: CityCell, k: i32)
        -> cc_streetlife_Prop {
    if (!cc.built) {
        return cc_streetlife_no_prop();
    }
    let r = city_rand4(vec2<u32>(
        bitcast<u32>(ci.x) * 0x27d4eb2fu + u32(k) * 0x9e3779b9u + 0x2f1e3a7bu,
        bitcast<u32>(ci.y) * 0x165667b1u + u32(k) * 0xc2b2ae35u + 0x7feb352du));
    if (r.x > cc_streetlife_CORNER_BIN) {
        return cc_streetlife_no_prop();
    }
    let xlo = (k & 1) == 0;
    let ylo = (k & 2) == 0;
    let wall_x = select(cc.plot_max.x, cc.plot_min.x, xlo);
    let out_x = select(1.0, -1.0, xlo);        // away from the plot
    let corner_y = select(cc.plot_max.y, cc.plot_min.y, ylo);
    let in_y = select(-1.0, 1.0, ylo);         // along the wall, into the plot
    let px = wall_x + out_x * (0.92 + 0.28 * r.z);
    let py = corner_y + in_y * (1.30 + 1.10 * r.y);
    let cellm = u.ocean_params.x;
    let cmin = vec2<f32>(ci) * cellm;
    let cmax = cmin + vec2<f32>(cellm);
    if (px - 0.62 < cmin.x || px + 0.62 > cmax.x
        || py - 0.92 < cmin.y || py + 0.92 > cmax.y) {
        return cc_streetlife_no_prop();
    }
    var p: cc_streetlife_Prop;
    p.ok = true;
    p.bin = true;
    p.ctr = vec2<f32>(px, py);
    p.fwd = vec2<f32>(0.0, 1.0);               // long side along the wall
    p.rgt = vec2<f32>(1.0, 0.0);
    p.r = r;
    return p;
}

fn cc_streetlife_trace_bin(o: vec3<f32>, inv_dir: vec3<f32>, dir: vec3<f32>,
                           t0: f32, t1: f32, v: cc_streetlife_Prop,
                           ci: vec2<i32>, side: i32) -> CityHit {
    let e = abs(v.fwd) * 0.80 + abs(v.rgt) * 0.50;
    let bmin = vec3<f32>(v.ctr - e, 0.0);
    let bmax = vec3<f32>(v.ctr + e, 1.20);
    let hb = city_box_hit(o, inv_dir, bmin, bmax);
    if (hb.x > hb.y || hb.y <= t0 || hb.x > t1) {
        return cc_streetlife_miss(ci);
    }
    let t = max(hb.x, t0);
    let pos = o + t * dir;
    return CityHit(true, t, pos, city_box_normal(pos, bmin, bmax),
                   106 + side, ci);
}

// ---------------------------------------------------------------------------
// One kerb
// ---------------------------------------------------------------------------

fn cc_streetlife_pole(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t0: f32, t1: f32, s: cc_streetlife_Side, a: f32,
                      ci: vec2<i32>, fp: f32) -> CityHit {
    // The arm hangs AWAY from the plot, out over the roadway.
    let arm = -s.plot_sign;
    let head_c = s.lamp_c + arm * cc_streetlife_ARM_REACH;
    var mmin: vec3<f32>;
    var mmax: vec3<f32>;
    var hmin: vec3<f32>;
    var hmax: vec3<f32>;
    if (s.ax == 0) {
        mmin = vec3<f32>(a - cc_streetlife_MAST_R,
                         s.lamp_c - cc_streetlife_MAST_R, 0.0);
        mmax = vec3<f32>(a + cc_streetlife_MAST_R,
                         s.lamp_c + cc_streetlife_MAST_R,
                         cc_streetlife_POLE_H);
        hmin = vec3<f32>(a - cc_streetlife_HEAD_HA,
                         head_c - cc_streetlife_HEAD_HC,
                         cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ);
        hmax = vec3<f32>(a + cc_streetlife_HEAD_HA,
                         head_c + cc_streetlife_HEAD_HC,
                         cc_streetlife_HEAD_Z + cc_streetlife_HEAD_HZ);
    } else {
        mmin = vec3<f32>(s.lamp_c - cc_streetlife_MAST_R,
                         a - cc_streetlife_MAST_R, 0.0);
        mmax = vec3<f32>(s.lamp_c + cc_streetlife_MAST_R,
                         a + cc_streetlife_MAST_R, cc_streetlife_POLE_H);
        hmin = vec3<f32>(head_c - cc_streetlife_HEAD_HC,
                         a - cc_streetlife_HEAD_HA,
                         cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ);
        hmax = vec3<f32>(head_c + cc_streetlife_HEAD_HC,
                         a + cc_streetlife_HEAD_HA,
                         cc_streetlife_HEAD_Z + cc_streetlife_HEAD_HZ);
    }
    var best = 1e30;
    var bmn = mmin;
    var bmx = mmax;
    var kind = 100;
    let hm = city_box_hit(o, inv_dir, mmin, mmax);
    if (hm.x <= hm.y && hm.y > t0 && hm.x <= t1) {
        best = max(hm.x, t0);
    }
    let hh = city_box_hit(o, inv_dir, hmin, hmax);
    if (hh.x <= hh.y && hh.y > t0 && hh.x <= t1 && max(hh.x, t0) < best) {
        best = max(hh.x, t0);
        bmn = hmin;
        bmx = hmax;
        kind = 101;
    }
    // The arm is a 0.13 m bar: it only earns a box test while it is a
    // resolvable line rather than an aliasing one. Past that the mast and the
    // luminaire carry the pole, which is what the eye reads anyway.
    if (fp < 0.30) {
        var amin: vec3<f32>;
        var amax: vec3<f32>;
        let c0 = min(s.lamp_c, head_c);
        let c1 = max(s.lamp_c, head_c);
        if (s.ax == 0) {
            amin = vec3<f32>(a - 0.055, c0,
                             cc_streetlife_ARM_Z - cc_streetlife_ARM_HZ);
            amax = vec3<f32>(a + 0.055, c1,
                             cc_streetlife_ARM_Z + cc_streetlife_ARM_HZ);
        } else {
            amin = vec3<f32>(c0, a - 0.055,
                             cc_streetlife_ARM_Z - cc_streetlife_ARM_HZ);
            amax = vec3<f32>(c1, a + 0.055,
                             cc_streetlife_ARM_Z + cc_streetlife_ARM_HZ);
        }
        let ha = city_box_hit(o, inv_dir, amin, amax);
        if (ha.x <= ha.y && ha.y > t0 && ha.x <= t1 && max(ha.x, t0) < best) {
            best = max(ha.x, t0);
            bmn = amin;
            bmx = amax;
            kind = 100;
        }
    }
    if (best >= 1e30) {
        return cc_streetlife_miss(ci);
    }
    let pos = o + best * dir;
    return CityHit(true, best, pos, city_box_normal(pos, bmn, bmx), kind, ci);
}

// What one kerb found. The sphere trace is deliberately NOT run here: a near
// car is returned as a CANDIDATE and resolved once per cell, after all four
// kerbs have reported. The reason is measured rather than stylistic — see the
// note in cc_streetlife_props_trace.
struct cc_streetlife_Kerb {
    hit: CityHit,      // already resolved: poles, bins, far cars
    car_ok: bool,      // a near car whose bounding box the ray entered
    car: cc_streetlife_Prop,
    side: i32,
    ta: f32,
    tb: f32,           // the box interval to sphere-trace inside
}

fn cc_streetlife_kerb(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell,
                      side: i32, fp: f32) -> cc_streetlife_Kerb {
    var out: cc_streetlife_Kerb;
    out.hit = cc_streetlife_miss(ci);
    out.car_ok = false;
    out.car = cc_streetlife_no_prop();
    out.side = side;
    out.ta = 0.0;
    out.tb = 0.0;

    let s = cc_streetlife_side(ci, cc, side);
    // One slab over the whole kerb: lamp line to parking line, ground to the
    // top of a mast, clipped to the cell's own column. This is the test the
    // wide scene pays, and the only one most cells ever reach.
    let arm_c = s.lamp_c - s.plot_sign * (cc_streetlife_ARM_REACH + 0.5);
    var lo = min(min(s.lamp_c, arm_c), s.park_c - cc_streetlife_BB_C);
    var hi = max(max(s.lamp_c, arm_c), s.park_c + cc_streetlife_BB_C);
    if (!s.park_ok) {
        lo = min(s.lamp_c, arm_c);
        hi = max(s.lamp_c, arm_c);
    }
    lo = max(lo - 0.2, s.c_min);
    hi = min(hi + 0.2, s.c_max);
    if (hi <= lo) {
        return out;
    }
    var bmin: vec3<f32>;
    var bmax: vec3<f32>;
    if (s.ax == 0) {
        bmin = vec3<f32>(s.a_min, lo, 0.0);
        bmax = vec3<f32>(s.a_max, hi, cc_streetlife_POLE_TOP);
    } else {
        bmin = vec3<f32>(lo, s.a_min, 0.0);
        bmax = vec3<f32>(hi, s.a_max, cc_streetlife_POLE_TOP);
    }
    let sb = city_box_hit(o, inv_dir, bmin, bmax);
    let ta = max(sb.x, t0);
    let tb = min(sb.y, t1);
    if (sb.x > sb.y || tb <= ta) {
        return out;
    }

    let pa = o + ta * dir;
    let pb = o + tb * dir;
    let sa = select(pa.y, pa.x, s.ax == 0);
    let sc = select(pb.y, pb.x, s.ax == 0);
    let s_lo = min(sa, sc);
    let s_hi = max(sa, sc);
    let z_lo = min(pa.z, pb.z);
    let fwd_first = select(dir.y, dir.x, s.ax == 0) >= 0.0;

    var res = cc_streetlife_miss(ci);
    var t_end = tb;

    // Poles, on the core's 26 m lattice, restricted to those standing wholly
    // inside this cell.
    if (fp < cc_streetlife_POLE_FAR_FP) {
        let sp = CITY_LAMP_SPACING;
        var j0 = i32(ceil((s_lo - 0.45) / sp));
        var j1 = i32(floor((s_hi + 0.45) / sp));
        j0 = max(j0, i32(ceil((s.a_min + 0.45) / sp)));
        j1 = min(j1, i32(floor((s.a_max - 0.45) / sp)));
        if (j1 >= j0) {
            let jstart = select(j1, j0, fwd_first);
            let jstep = select(-1, 1, fwd_first);
            // Walked from the near end, so the first pole the ray actually
            // strikes is the nearest and the loop is done.
            for (var k: i32 = 0; k < 3; k = k + 1) {
                let j = jstart + k * jstep;
                if (j < j0 || j > j1) {
                    break;
                }
                let hp = cc_streetlife_pole(o, dir, inv_dir, ta, t_end, s,
                                            f32(j) * sp, ci, fp);
                if (hp.hit) {
                    res = cc_streetlife_nearer(res, hp);
                    t_end = min(t_end, res.t);
                    break;
                }
            }
        }
    }

    // Ground props: everything below is under 1.8 m, so a segment that only
    // clips the lamp heads stops here.
    out.hit = res;
    if (!s.park_ok || z_lo > cc_streetlife_BB_Z + 0.1
        || fp > cc_streetlife_CAR_FAR_FP) {
        return out;
    }
    let g0 = i32(floor((s_lo - cc_streetlife_BB_A) / cc_streetlife_SLOT));
    let g1 = i32(floor((s_hi + cc_streetlife_BB_A) / cc_streetlife_SLOT));
    let gstart = select(g1, g0, fwd_first);
    let gstep = select(-1, 1, fwd_first);
    // Same rule as the poles: near end first, stop at the first prop the ray
    // actually strikes. Five iterations is what an empty run costs, and at
    // ~40% occupancy the loop usually ends on the first or the second.
    let near_sdf = fp < cc_streetlife_CAR_SDF_FP;
    for (var k: i32 = 0; k < 5; k = k + 1) {
        let j = gstart + k * gstep;
        if (j < g0 || j > g1) {
            break;
        }
        let v = cc_streetlife_prop(ci, cc, s, side, j);
        if (!v.ok) {
            continue;
        }
        if (v.bin) {
            let hb = cc_streetlife_trace_bin(o, inv_dir, dir, ta, t_end, v,
                                             ci, side);
            if (hb.hit) {
                res = cc_streetlife_nearer(res, hb);
                t_end = min(t_end, res.t);
                break;
            }
            continue;
        }
        // The bounding box: the only cost the wide scene pays for a car.
        let e = abs(v.fwd) * cc_streetlife_BB_A
              + abs(v.rgt) * cc_streetlife_BB_C;
        let cmin = vec3<f32>(v.ctr - e, 0.0);
        let cmax = vec3<f32>(v.ctr + e, cc_streetlife_BB_Z);
        let hc = city_box_hit(o, inv_dir, cmin, cmax);
        if (hc.x > hc.y || hc.y <= ta || hc.x > t_end) {
            continue;
        }
        if (near_sdf) {
            out.car_ok = true;
            out.car = v;
            out.ta = max(hc.x, ta);
            out.tb = min(hc.y, t_end);
            t_end = min(t_end, out.ta);
            break;
        }
        let hit = cc_streetlife_trace_car_box(o, inv_dir, dir, ta, t_end, v,
                                              ci, side);
        if (hit.hit) {
            res = cc_streetlife_nearer(res, hit);
            t_end = min(t_end, res.t);
            break;
        }
    }
    out.hit = res;
    return out;
}

fn cc_streetlife_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                             t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    // WHERE THIS COMPONENT'S COST ACTUALLY IS, measured rather than guessed,
    // so the next reader does not repeat the experiments (RTX 5080, shared,
    // interleaved on/off, 64 accumulated frames at 960x540):
    //
    //     view      off      on     delta
    //     base     0.34 s  0.40 s   +18%
    //     aerial   0.22 s  0.30 s   +36%
    //     horizon  0.31 s  0.40 s   +29%
    //
    // All of it is in this hook, none in the shade hook: unregistering
    // `shade` and leaving `cell_props` reproduced the enabled timings to the
    // centisecond. And none of it is work — on all three views EVERY call
    // leaves at gate 1 below, because the nearest ground within
    // CITY_PROP_RANGE is still a kilometre under the ray. Three attempts to
    // shrink the inlined body moved nothing at all: hoisting the fp cutoff
    // above the four kerb slabs, rolling the four-kerb loop behind a bound
    // the driver cannot fold (the draft's own trick for the march budget),
    // and generating the normal's four tetrahedron taps in a loop instead of
    // spelling them out. Two of the three were reverted for being clutter
    // that bought nothing; the fp cutoff stayed because it is exact.
    //
    // What is left is the gate itself, two fused multiply-adds and two
    // compares, run once per DDA cell within CITY_PROP_RANGE — on the aerial
    // view roughly 26 million times a frame. 0.08 s over 64 frames is about
    // three flops per evaluation, which is the whole of it. This is the floor
    // for ANY cell_props hook in a 512-iteration DDA, not a streetlife
    // problem, and it is not reducible from inside a component.
    //
    // Gate 1, no hash and no memory: does this segment come within reach of
    // the ground at all? Every pixel looking at a facade, a roof, a cloud or
    // the sky leaves here.
    let za = o.z + t0 * dir.z;
    let zb = o.z + t1 * dir.z;
    if (min(za, zb) > cc_streetlife_Z_GATE || max(za, zb) < -0.5) {
        return cc_streetlife_miss(ci);
    }
    let fp = max(2.0 * u.cam_origin.w / max(u.params.x, 1.0), u.periodic.z)
             * max(t0, 0.0);
    // Gate 2: past the pole cutoff nothing this component draws survives, and
    // every prop's emission has already faded to zero before reaching it
    // (POLE_FAR_FADE 2.4 -> 4.0, CAR_FAR_FADE 1.6 -> 2.6). Hoisting the test
    // above the four kerb slabs makes it exact rather than merely cheap: it
    // is the same answer those per-kerb fp tests would have reached, arrived
    // at before any of the four Side structs is built.
    if (fp > cc_streetlife_POLE_FAR_FP) {
        return cc_streetlife_miss(ci);
    }
    var res = cc_streetlife_miss(ci);
    var t_end = t1;
    var cand: cc_streetlife_Kerb;
    var have = false;
    var cand_t = 1e30;
    for (var side: i32 = 0; side < 4; side = side + 1) {
        let k = cc_streetlife_kerb(o, dir, inv_dir, t0, t_end, ci, cc, side,
                                   fp);
        if (k.hit.hit) {
            res = cc_streetlife_nearer(res, k.hit);
            t_end = min(t_end, res.t);
        }
        if (k.car_ok && k.ta < cand_t) {
            cand = k;
            have = true;
            cand_t = k.ta;
            t_end = min(t_end, k.ta);
        }
    }
    // Corner dumpsters. Four hash draws behind a second z gate — a dumpster
    // is 1.2 m tall, so a segment that only passes through the lamp heads
    // leaves here — and a box test only for the quarter of corners that draw.
    // Placed after the kerb loop so t_end is already as short as the kerbs
    // could make it, and before the sphere trace so a bin standing in front
    // of a car correctly stops the car being resolved at all.
    if (cc.built && min(za, zb) < 1.5 && fp < cc_streetlife_CAR_FAR_FP) {
        for (var k: i32 = 0; k < 4; k = k + 1) {
            let b = cc_streetlife_corner_prop(ci, cc, k);
            if (!b.ok) {
                continue;
            }
            let hb = cc_streetlife_trace_bin(o, inv_dir, dir, t0, t_end, b,
                                             ci, k);
            if (hb.hit) {
                res = cc_streetlife_nearer(res, hb);
                t_end = min(t_end, res.t);
            }
        }
    }
    // ONE sphere trace per cell, on the nearest car bounding box any kerb
    // accepted — never one per slot. This is the most consequential
    // structural decision in the file and it was forced by measurement.
    // cell_props is inlined into the core's DDA, whose loop runs up to 512
    // times, and the four-kerb by five-slot loops are small enough that the
    // driver unrolls them: written at the slot, the trace appeared twenty
    // times in that loop body, and the register pressure alone cost 1.0 s of
    // a 1.6 s frame set on the `base` view, where no car is anywhere near
    // the footprint that admits an SDF. Hoisted here it appears once.
    // The price is one artifact: a ray that enters a car's box and then
    // grazes past the hull returns a miss rather than falling through to the
    // car behind it. Parked cars sit 2.4 m apart, so that is a sliver on a
    // silhouette edge, and it buys back the whole rest of the scene.
    if (have && cand_t <= t_end + 1e-4) {
        let hit = cc_streetlife_trace_car(o, dir, cand.ta, cand.tb, cand.car,
                                          ci, cand.side, fp);
        if (hit.hit && (!res.hit || hit.t < res.t)) {
            res = hit;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

// Underglow palette. Deliberately the same draw as aircars' — 60% cyan, 25%
// magenta, 15% amber — so a street of parked cars and the lane of flying ones
// above it are visibly the same fleet.
fn cc_streetlife_glow_color(d: f32) -> vec3<f32> {
    if (d < 0.60) {
        return vec3<f32>(0.16, 0.90, 1.00);
    }
    if (d < 0.85) {
        return vec3<f32>(1.00, 0.20, 0.70);
    }
    return vec3<f32>(1.00, 0.60, 0.16);
}

// Night car paint. Weighted dark on purpose: the road under these cars is a
// clipped sodium wash, so a car is a hole in it with a lit edge, and a
// palette of mid-greys renders a street of pale ceramic bathtubs. One car in
// seven is light enough to be the bright one in the row.
fn cc_streetlife_paint(d: f32) -> vec3<f32> {
    if (d < 0.30) {
        return vec3<f32>(0.055, 0.060, 0.070);  // graphite
    }
    if (d < 0.44) {
        return vec3<f32>(0.320, 0.335, 0.350);  // silver
    }
    if (d < 0.58) {
        return vec3<f32>(0.230, 0.055, 0.055);  // oxblood
    }
    if (d < 0.72) {
        return vec3<f32>(0.045, 0.090, 0.185);  // midnight blue
    }
    if (d < 0.82) {
        return vec3<f32>(0.300, 0.260, 0.170);  // sand
    }
    if (d < 0.93) {
        return vec3<f32>(0.040, 0.155, 0.130);  // deep teal
    }
    return vec3<f32>(0.235, 0.085, 0.150);      // faded plum
}

// A band of width `w` around `x0`, antialiased against the footprint and
// blended to its own mean coverage once the line is sub-pixel — a seam that
// simply vanished would take the panel's mean brightness with it.
fn cc_streetlife_seam(x: f32, x0: f32, w: f32, pitch: f32, fp: f32) -> f32 {
    let e = 0.5 * w + 0.6 * fp;
    let sharp = 1.0 - smoothstep(0.5 * w, e + 1e-4, abs(x - x0));
    let mean = w / max(pitch, 1e-3);
    return mix(sharp, mean,
               smoothstep(cc_streetlife_DETAIL_LOD.x,
                          cc_streetlife_DETAIL_LOD.y, fp));
}

// The footprint the CAR's own detail is resolved at, which is not the one the
// core hands the shade hook.
//
// The core passes `fp = pixel_angle * t`, where pixel_angle is floored by the
// app's view-step LOD slider — tan(0.3 deg) by default, four and a half times
// the actual pixel at 960 px and 60 degrees. That floor is right for what it
// was built for: it stops sub-pixel window LATTICES from moireing as the
// camera moves. A car's wiper blade, rim spoke or lamp lens is not a lattice,
// it is one feature, and blurring it across four pixels throws away detail
// the accumulation would otherwise resolve — the whole reason the LOD floor
// dropped to a quarter pixel in the first place.
//
// So: sharpen toward the true pixel, but never below about one and a half of
// them, which is what keeps a moving 1-spp frame from crawling. With the
// default slider this lands at ~1.5 px; with a fine slider the true-pixel
// term takes over and holds the floor at 1 px.
fn cc_streetlife_fp_px(fp: f32, t: f32) -> f32 {
    return max(0.35 * fp,
               2.0 * u.cam_origin.w / max(u.params.x, 1.0) * max(t, 0.0));
}

// One lamp LENS on a face, resolved while it is bigger than a pixel and
// handed to the face's mean when it is not (aircars' treatment, same
// reasoning). The lens is an ellipse, not a disc, and it is nested inside the
// letterbox recess the SDF cut for it: a circular lens of radius 0.12
// overflowed a housing only 0.16 tall, so it rendered as a white ball with a
// dark eyebrow, and a row of parked cars looked back at the camera. Real
// vehicle lamps are wide and shallow; the aspect alone does most of the work.
const cc_streetlife_LENS_A: f32 = 0.160;   // half-width, along the face
const cc_streetlife_LENS_B: f32 = 0.049;   // half-height
fn cc_streetlife_dot(a: f32, b: f32, sa: f32, sb: f32, span: f32, fp: f32)
        -> f32 {
    let e = vec2<f32>((a - sa) / cc_streetlife_LENS_A,
                      (b - sb) / cc_streetlife_LENS_B);
    let d = length(e);
    // The edge softens in the ellipse's OWN metric, with the footprint
    // normalised by the geometric mean of the two semi-axes and then clamped.
    // Both of the obvious alternatives failed on this shape: dividing fp by
    // the semi-MINOR axis alone inflates it eighteenfold and blew the lens
    // into a lobe covering the whole nose, while normalising by the implicit
    // gradient, (d-1)/|grad d|, looks exact but asymptotes to a constant —
    // LENS_B, 0.056 — far from an eccentric ellipse, so once the footprint
    // crossed that constant the test stopped bounding anything at all and the
    // lens grew vertically without limit. A clamped width in the normalised
    // metric cannot do either: the lens is never more than 1 + w across.
    let w = clamp(fp / sqrt(cc_streetlife_LENS_A * cc_streetlife_LENS_B),
                  0.06, 0.42);
    let sharp = 1.0 - smoothstep(1.0 - w, 1.0 + w, d);
    let mean = 3.14159265 * cc_streetlife_LENS_A * cc_streetlife_LENS_B
             / max(span, 1e-3);
    return mix(sharp, mean,
               smoothstep(cc_streetlife_LAMP_LOD.x,
                          cc_streetlife_LAMP_LOD.y, fp));
}

// How far into the glazing a point on the greenhouse lies, in body units.
//
// The greenhouse is bounded by five planes — windshield, backlight, roof
// rail, belt line, tumblehome flank — and the window surround is the distance
// to the nearest of them. The catch is that the border you are STANDING on is
// at distance zero by definition, so a naive minimum reports "no glass"
// everywhere. Each border is therefore pushed out of the running in
// proportion to how closely the surface normal agrees with the border's own,
// and only when it agrees closely: the ramp starts at dot 0.82, so a
// windshield is excused from its own plane but not from the roof rail it runs
// up to, even though the two are only 55 degrees apart.
//
// Five dot products buys A-pillars, C-pillars, a roof rail and a belt line
// that are all the same band, on every face, with no branch and no per-face
// special case — which is what the first pass tried to get from one `ay`
// bound and one normal test, and got windowless cars instead.
fn cc_streetlife_glass_inset(cx: f32, q: vec3<f32>, nl: vec3<f32>) -> f32 {
    let ay = abs(q.y);
    let sy = select(-1.0, 1.0, q.y >= 0.0);
    let nf = vec3<f32>(0.824, 0.0, 0.567);          // windshield
    let nb = vec3<f32>(-0.745, 0.0, 0.667);         // backlight
    let ns = vec3<f32>(0.0, sy * 0.958, 0.287);     // flank, leaning in
    var m = 1.339 - (nf.x * cx + nf.z * q.z)
          + 12.0 * max(dot(nl, nf) - 0.82, 0.0);
    m = min(m, 1.361 - (nb.x * cx + nb.z * q.z)
               + 12.0 * max(dot(nl, nb) - 0.82, 0.0));
    m = min(m, 1.425 - q.z + 12.0 * max(nl.z - 0.82, 0.0));
    m = min(m, q.z - 1.045 + 12.0 * max(-nl.z - 0.82, 0.0));
    m = min(m, 0.958 * (0.74 - 0.30 * (q.z - 1.02) - ay)
               + 12.0 * max(dot(nl, ns) - 0.82, 0.0));
    return m;
}

fn cc_streetlife_pole_shade(h: CityHit, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let fill = cc_streetlife_fill(h.pos);
    // Two edges to approach, never to step over: the population edge at
    // CITY_PROP_RANGE, and the footprint at which poles stop being traced.
    // The second was declared (POLE_FAR_FADE) and then never applied, so a
    // luminaire at radiance 6 — the brightest thing on the street — simply
    // switched off the instant fp crossed POLE_FAR_FP. That is the one thing
    // the SPEC says outright must not happen: sub-pixel detail dissolves into
    // its own mean, it does not vanish.
    let fade = (1.0 - smoothstep(cc_streetlife_FADE_START * CITY_PROP_RANGE,
                                 CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_streetlife_POLE_FAR_FADE,
                                 cc_streetlife_POLE_FAR_FP, fp));
    if (h.kind == 101) {
        // The luminaire. Its underside is the lamp; its sides are the lens
        // edge; its top is a painted aluminium housing, and dark. A glowing
        // box would read as a floating lozenge, not as a light fitting.
        let district = city_glow_sample(h.pos.xy, 2.0);
        let out = 0.45 + 1.6 * smoothstep(0.02, 0.45, district);
        if (h.normal.z < -0.5) {
            return cc_streetlife_HEAD_COLOR
                   * (cc_streetlife_HEAD_RAD * out * fade);
        }
        if (h.normal.z > 0.5) {
            return 0.10 * fill + vec3<f32>(0.004, 0.003, 0.002);
        }
        // A side face is mostly painted cowl, with the lens showing as a lip
        // along its bottom edge. Uniformly bright, the sides clipped to white
        // with the lamp and the whole fitting became one glowing lozenge —
        // the exact failure the head geometry exists to avoid. Antialiased
        // against fp so the lip fades into the face's own mean rather than
        // strobing once it is thinner than a pixel.
        let zl = cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ;
        let lip = 1.0 - smoothstep(zl + cc_streetlife_HEAD_LIP,
                                   zl + cc_streetlife_HEAD_LIP + 0.5 * fp
                                       + 0.012,
                                   h.pos.z);
        let mean = cc_streetlife_HEAD_LIP
                 / (2.0 * cc_streetlife_HEAD_HZ);
        let k = mix(lip, mean, smoothstep(0.010, 0.075, fp));
        return cc_streetlife_HEAD_COLOR
               * (cc_streetlife_HEAD_RAD * out * fade
                  * mix(cc_streetlife_HEAD_COWL,
                        cc_streetlife_HEAD_SIDE, k))
             + 0.08 * fill;
    }
    // Mast and arm: galvanised steel, lit almost entirely by its own lamp,
    // and more of it the closer to the head — the falloff up the pole is the
    // single cue that says the light is at the top.
    let up = clamp(h.pos.z / cc_streetlife_POLE_H, 0.0, 1.0);
    let near_lamp = 0.12 + 0.88 * up * up;
    let pool = cc_streetlife_pool(h.pos.xy);
    let lamp = cc_streetlife_nearest_lamp(h.pos.xy);
    let l = normalize(lamp - h.pos);
    // The mast is a box because a box is what the DDA wants, but 17 cm of
    // steel is round, and from two metres away a flat-shaded rectangle says
    // so loudly. The silhouette stays square — sub-pixel at any distance you
    // would notice — while the NORMAL is remapped across the width of the
    // face to the cylinder the box circumscribes.
    //
    // The remap has to be driven by WHERE ON THE FACE the hit is, and the
    // draft drove it by the direction to the lamp instead. On a mast the lamp
    // is directly overhead, so that vector is nearly zero in xy, the remap
    // collapsed to the identity, and the note beside it recording that "the
    // remapped normal on its own changed nothing" was reporting a bug rather
    // than a fact about the geometry. Every face then carried ONE normal and
    // one grazing value, which is why a pole came out as a flat gold bar with
    // a dark side rather than as a cylinder with two bright edges.
    //
    // The axis is recoverable exactly: cc_streetlife_nearest_lamp returns the
    // lattice point, which is where the mast stands. The offset of the hit
    // from it, resolved along the face, is the cylinder angle.
    var n = h.normal;
    let rel = h.pos.xy - lamp.xy;
    if (abs(h.normal.z) < 0.5
        && dot(rel, rel) < cc_streetlife_MAST_R * cc_streetlife_MAST_R * 2.9) {
        let tang = vec2<f32>(-h.normal.y, h.normal.x);
        let uu = clamp(dot(rel, tang) / cc_streetlife_MAST_R, -1.0, 1.0);
        n = normalize(vec3<f32>(h.normal.xy * sqrt(max(1.0 - uu * uu, 0.0))
                                + tang * uu, 0.0));
    }
    let lam = 0.30 + 0.70 * max(dot(n, l), 0.0);
    // A vertical pole lit by a lamp on its own axis has almost no shading
    // variation around its circumference — that is the geometry, not a bug,
    // and it is why Lambert alone cannot draw a pole. What makes one look
    // round at night is the grazing edge: the two sides of the cylinder catch
    // the street at glancing incidence and the middle of the face returns
    // almost nothing. So the cylinder is read out through a Fresnel edge —
    // which needs a normal that actually turns across the face, hence the
    // remap above.
    let edge = pow(1.0 - clamp(abs(dot(dir, n)), 0.0, 1.0), 3.0);
    return (cc_streetlife_MAST_FILL * fill
            + cc_streetlife_MAST_ALB * pool * (near_lamp * lam)
            + pool * (cc_streetlife_MAST_EDGE * edge * near_lamp)) * fade;
}

fn cc_streetlife_bin_shade(h: CityHit, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let side = h.kind - 106;
    let fill = cc_streetlife_fill(h.pos);
    let pool = cc_streetlife_pool(h.pos.xy);
    let lamp = cc_streetlife_nearest_lamp(h.pos.xy);
    let l = normalize(lamp - h.pos);
    let lam = 0.22 + 0.78 * max(dot(h.normal, l), 0.0);
    // Steel, painted once and repainted never: a dull olive that the sodium
    // pulls most of the colour out of anyway.
    let body = vec3<f32>(0.20, 0.24, 0.18);
    // The lid, and one rib per side. Both are dark lines, not geometry.
    var k = 1.0;
    if (h.normal.z < 0.5) {
        k = k * (1.0 - 0.55 * cc_streetlife_seam(h.pos.z, 0.98, 0.05, 1.2, fp));
        let along = select(h.pos.y, h.pos.x, side >= 2);
        k = k * (1.0 - 0.35 * cc_streetlife_seam(fract(along * 1.6), 0.5,
                                                 0.07, 1.0, fp));
    }
    return cc_streetlife_BIN_ALB * body * pool * lam * k
         + 0.5 * cc_streetlife_BIN_ALB * body * fill;
}

fn cc_streetlife_car_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    let side = h.kind - 102;
    let s = cc_streetlife_side(h.cell, cc, side);
    // Recover the slot from the hit itself: the car's centre is within 3.1 m
    // of its slot centre and the slots are 7 m apart, so the floor is exact.
    let along = select(h.pos.y, h.pos.x, s.ax == 0);
    let j = i32(floor(along / cc_streetlife_SLOT));
    let v = cc_streetlife_prop(h.cell, cc, s, side, j);
    let r = v.r;
    let hover = cc_streetlife_is_hover(r);
    let lift = select(0.0, cc_streetlife_HOVER_LIFT, hover);

    let sh = cc_streetlife_car_shape(r);
    let cab_c = sh.y;
    let off = cab_c + 0.30;
    // The same body frame the SDF works in, scale and all, so every band
    // below lands on the geometry it names.
    let q = (cc_streetlife_to_local(h.pos, v.ctr, v.fwd, v.rgt)
             - vec3<f32>(0.0, 0.0, lift)) / sh.x;
    let nl = vec3<f32>(dot(h.normal.xy, v.fwd), dot(h.normal.xy, v.rgt),
                       h.normal.z);
    let ay = abs(q.y);
    // Detail resolves at fpd; the distance FADES below still key off the
    // core's fp, because those follow the app's LOD slider by design and a
    // car must not outlive the population edge just because it is sharp.
    let fpd = cc_streetlife_fp_px(fp, h.t);

    let fill = cc_streetlife_fill(h.pos);
    let pool = cc_streetlife_pool(h.pos.xy);
    let lamp = cc_streetlife_nearest_lamp(h.pos.xy);
    let l = normalize(lamp - h.pos);
    let fade = (1.0 - smoothstep(cc_streetlife_FADE_START * CITY_PROP_RANGE,
                                 CITY_PROP_RANGE, h.t))
             * (1.0 - smoothstep(cc_streetlife_CAR_FAR_FADE,
                                 cc_streetlife_CAR_FAR_FP, fp));

    // The one specular lobe is what carries curvature. A dark paint under a
    // single overhead source is nearly flat in Lambert; the glint sliding
    // along the shoulder is the whole read.
    let rf = reflect(dir, h.normal);
    let spec_c = max(dot(rf, l), 0.0);
    let lam = max(dot(h.normal, l), 0.0);

    // Tyres first: matte rubber, and the only part that wants none of the
    // clearcoat treatment. The wheel only reads at all because of the rim
    // face inside it — a black disc on a black car under a dim lamp is
    // nothing, and the first pass had cars that appeared to float.
    let qw = vec3<f32>(abs(q.x) - cc_streetlife_AXLE_X,
                       ay - cc_streetlife_TRACK,
                       q.z - cc_streetlife_AXLE_Z);
    let wheel = cc_streetlife_cyl_y(qw, cc_streetlife_TYRE_R,
                                    cc_streetlife_TYRE_HW);
    if (!hover && wheel < 0.035) {
        let rad = length(vec2<f32>(qw.x, qw.z));
        // The rim: the only part of a wheel that is legible at night, and the
        // first pass gave it a quarter of the tyre's radius, so it never
        // showed and the wheels read as flat pale discs. The face is most of
        // the wheel, as it is on a real one; the tyre is the band around it.
        if (qw.y > 0.085 && rad < cc_streetlife_RIM_R) {
            // Brushed metal, five spokes, a hub, and a rolled lip at the rim
            // edge that catches the road — that lip is what turns a disc into
            // something with depth.
            let ang = atan2(qw.z, qw.x) * 0.795774715;   // turns
            let spoke = cc_streetlife_seam(fract(ang * 5.0), 0.5, 0.30, 1.0,
                                           fpd * 3.0);
            let hub = 1.0 - smoothstep(0.048, 0.070, rad);
            let lip = smoothstep(cc_streetlife_RIM_R - 0.035,
                                 cc_streetlife_RIM_R - 0.008, rad);
            let face = max(max(1.0 - 0.80 * spoke, hub), 0.85 * lip);
            return vec3<f32>(0.94, 0.97, 1.00) * cc_streetlife_RIM_ALB
                   * pool * (0.26 + 0.74 * lam) * face
                 + 0.6 * fill;
        }
        // Rubber. Weathered tyre reflectance is about 0.02 in daylight and
        // less than that here, and it has to come out DARKER than graphite
        // paint or the car floats on four pale coins — which is precisely
        // what the first pass rendered, because 0.012 of a clipped sodium
        // road still beats 0.10 of a 0.055 paint.
        let tread = 1.0 - 0.42 * cc_streetlife_seam(
            fract(atan2(qw.z, qw.x) * 4.6), 0.5, 0.15, 1.0, fpd * 4.0);
        return vec3<f32>(cc_streetlife_TYRE_ALB) * pool * (0.18 + 0.82 * lam)
               * tread
             + 0.16 * fill
             + pool * (0.008 * pow(spec_c, 12.0));
    }

    // GLASS. This test decides whether a car has windows at all, and the
    // salvaged draft's version answered no everywhere in the city — the two
    // renders that motivated the rewrite showed a solid loaf of paint where
    // the cabin should be. Two independent reasons, both worth stating
    // because both are the kind of test that looks obviously right:
    //   * `ay < 0.71` against a cabin whose own half-width is 0.70, inflated
    //     outward by the shoulder's smooth-min: the side glass sat a few
    //     millimetres OUTSIDE its own window, so the flanks were never glass.
    //   * `nl.z < 0.72` on a raked windshield, whose normal is (0.82, 0,
    //     0.57) by construction: the more like a windscreen the windscreen
    //     got, the more certainly it was classified as bodywork.
    // So the region is now the greenhouse's own five-plane interior, inset by
    // a surround (cc_streetlife_glass_inset), with the roof panel taken back
    // out — a car may have a raked screen at every angle, but not a glass
    // roof.
    let cx = q.x - cab_c;
    let roof_face = q.z > 1.33 && nl.z > 0.80;
    let gin = cc_streetlife_glass_inset(cx, q, nl);
    // A B-pillar between the two side lights, while it is wide enough to be a
    // pillar rather than an aliasing line; past 6 cm/px the greenhouse is
    // uniform glass, which is that band's honest mean.
    let b_pillar = select(0.0,
                          1.0 - smoothstep(0.034, 0.062, abs(cx + 0.02)),
                          fpd < 0.06);
    let is_glass = q.z > 1.05 && gin > 0.055 + 0.35 * fpd
                && !roof_face && b_pillar < 0.5;
    if (is_glass) {
        let fres = pow(1.0 - clamp(abs(dot(dir, h.normal)), 0.0, 1.0), 4.0);
        // What a parked car's glass carries at night, in order of how much of
        // it there is: the road it is standing on (bright, and reflected by
        // every window that leans at all), the skyglow, and a wash of the
        // building opposite. Without the road term a dark car's greenhouse
        // is the same value as its paint and the whole cabin stops existing.
        let env = mix(cc_streetlife_GLASS_ROAD * pool,
                      3.0 * fill
                      + CITY_PALETTE_MEAN * (0.05 + 0.22 * cc.lit_frac),
                      clamp(rf.z * 1.8 + 0.45, 0.0, 1.0));
        let glint = pow(spec_c, 220.0) * cc_streetlife_GLASS_GLOSS;
        // Wiper blades cross the glass as hard dark lines; the windshield
        // also carries the demist banding at its base.
        let wip = cc_streetlife_seg(vec3<f32>(q.x - off, ay, q.z),
                                    vec3<f32>(0.66, 0.07, 0.925),
                                    vec3<f32>(1.00, 0.52, 0.908), 0.021);
        let wmask = select(1.0, 0.30, wip < 0.014);
        return ((1.0 + 2.2 * fres) * env + pool * glint) * wmask * fade;
    }

    // Paint.
    var col = cc_streetlife_paint(fract(r.z * 5.17 + r.w * 0.31));
    // Panel lines, in the hull's own frame. A shut line at the door, one at
    // the rear quarter, a rocker crease under the sill, and a hood seam on
    // the deck: functional lines only, nothing that would be noise at 10 m.
    var seam = 1.0;
    if (abs(nl.z) < 0.75) {
        seam = seam * (1.0 - 0.75 * cc_streetlife_seam(cx, 0.86, 0.05,
                                                       1.6, fpd));
        seam = seam * (1.0 - 0.75 * cc_streetlife_seam(cx, -0.78, 0.05,
                                                       1.6, fpd));
        seam = seam * (1.0 - 0.40 * cc_streetlife_seam(q.z, 0.46, 0.06,
                                                       1.0, fpd));
        // Door handles: one recess per door, on the belt line.
        let dh = (1.0 - smoothstep(0.10, 0.17, abs(cx + 0.18)))
               * (1.0 - smoothstep(0.02, 0.045, abs(q.z - 0.80)));
        seam = seam * (1.0 - 0.55 * select(0.0, dh, fpd < 0.05));
    } else {
        seam = seam * (1.0 - 0.55 * cc_streetlife_seam(cx, 1.28, 0.045,
                                                       2.0, fpd));
        seam = seam * (1.0 - 0.55 * cc_streetlife_seam(cx, -0.96, 0.045,
                                                       2.0, fpd));
    }
    var e = cc_streetlife_PAINT_GAIN * col * pool * (0.22 + 0.78 * lam) * seam
          + 1.4 * col * fill;
    // Clearcoat: a tight lobe on the sodium, plus a broad sheen that picks up
    // the whole lit street. Both scale with the pool, so a car parked between
    // lamps stays a silhouette.
    e = e + pool * (cc_streetlife_GLOSS * pow(spec_c, 60.0)
                    + cc_streetlife_SHEEN * pow(spec_c, 6.0));
    // Rim: the skyglow catching the top of a curved shoulder.
    let graze = pow(1.0 - clamp(abs(dot(dir, h.normal)), 0.0, 1.0), 3.0);
    e = e + fill * (2.5 * graze * clamp(nl.z + 0.4, 0.0, 1.0));
    // Road bounce. The single most useful term on the whole car: a vertical
    // flank at night reflects the sodium wash it is parked on, and it does
    // so hardest at grazing incidence and lowest on the body. That is what
    // draws the bright outline around the wheel arch, along the rocker and
    // over the shoulder crease — the specular lobe cannot, because the lamp
    // is overhead and a flank never reflects it toward the camera.
    e = e + pool * (cc_streetlife_ROAD_BOUNCE * graze
                    * clamp(0.55 - 0.75 * nl.z, 0.0, 1.0));

    // Head and tail lamps, in the housings the SDF cut for them.
    let facing = nl.x;
    if (abs(facing) > 0.35) {
        let fwd_face = facing > 0.0;
        let dot_e = cc_streetlife_dot(ay, q.z, 0.50, 0.71, 1.2, fpd);
        // A parked car's lamps are standing lights, not driving beams: white
        // forward, red aft, and both a good deal under the sodium luminaire
        // overhead (radiance 6) rather than clipped alongside it. Tails run
        // brighter than heads because a red lens at this exposure loses two
        // of its three channels.
        let col_l = select(vec3<f32>(1.00, 0.06, 0.03),
                           vec3<f32>(1.00, 0.93, 0.82), fwd_face);
        let rad_l = cc_streetlife_LAMP_RAD * select(1.15, 0.72, fwd_face);
        e = e + col_l * (rad_l * dot_e * fade);
        // A tail light bar on the cars that carry one — the cheapest possible
        // cyberpunk signature, and it survives to a distance the dots do not.
        if (!fwd_face && r.z > 0.55) {
            let bar = (1.0 - smoothstep(0.020, 0.055 + 0.5 * fpd,
                                        abs(q.z - 0.74)))
                    * (1.0 - smoothstep(0.55, 0.72, ay));
            e = e + vec3<f32>(1.00, 0.06, 0.03)
                    * (1.3 * mix(bar, 0.10, smoothstep(0.05, 0.30, fpd))
                       * fade);
        }
    }

    // Underglow: a strip under the sill, or the whole plenum on a hover car.
    let glow_on = r.w < cc_streetlife_GLOW_FRAC;
    if (glow_on) {
        let gc = cc_streetlife_glow_color(fract(r.y * 3.77 + r.z * 0.13));
        let hi_z = select(0.34, 0.30, hover);
        let strip = (1.0 - smoothstep(hi_z - 0.10, hi_z + 0.06, q.z))
                  * clamp(1.0 - abs(nl.z), 0.12, 1.0);
        let amp = select(1.0, 1.7, hover);
        e = e + gc * (cc_streetlife_GLOW_RAD * amp * strip * fade);
    }
    return e;
}

fn cc_streetlife_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    if (h.kind <= 101) {
        return cc_streetlife_pole_shade(h, dir, fp);
    }
    if (h.kind <= 105) {
        return cc_streetlife_car_shade(h, cc, dir, fp);
    }
    return cc_streetlife_bin_shade(h, dir, fp);
}

// --- component: adscreens (adscreens.wgsl) ---
// adscreens — the big screens, and the weird things they are selling.
//
// About one building in twelve over 50 m carries screens — but weighted by
// the cascade, so downtown runs nearer one in five and the outskirts one in
// forty, because advertising clusters where the crowds are and a uniform
// sprinkle left the megatower district (which is the shot) with almost none.
// A chosen building offers each of its vertical box facades to a second draw,
// and the ones that take it get ONE screen — 30-70% of the facade wide (up to
// 86% on a merged superblock, which is where the giants live), aspect
// anywhere from 4:3 to 1:2.5. That is a 10-50 m rectangle on an ordinary
// tower and a 140 m one on a superblock: the loudest thing on the wall.
//
// Content features are sized in METRES, not in fractions of the panel —
// character cells 3-6 m, dead-channel pixel pitch 0.7-1.6 m, test-card bars
// 1.6-4 m. Sizing them as panel fractions instead (which is what this file
// did first) gives a small screen 10 cm noise that is sub-pixel from across
// the street and a superblock 30 m characters, so the same archetype read as
// flat grey on one wall and as three enormous blobs on another.
//
// Six content archetypes, all pure math in screen-local uv, all static:
//   FACE     a giant face suggestion — two eye blobs and a mouth bar over a
//            skin vignette, with a neon pinprick in each eye. Abstracted
//            enough that it reads as a face without ever being one.
//   PRODUCT  a capsule silhouette, luminous, rimmed, on a saturated gradient
//            with a halo bleeding off it. The object is never named.
//   GLYPH    rows of pseudo-ideograms: three axis-aligned strokes per cell,
//            endpoints snapped to a 3x3 lattice out of one pcg2d draw, so the
//            characters have the regularity of writing without being any
//            writing. No real text — the calibration forbids it and it would
//            be worse if it were legible.
//   STRIPES  a test card. Bar count a multiple of three, colours cycling
//            through the screen's three-neon scheme, dark foot band.
//   STATIC   a dead channel: hash noise, a few rows shear-glitched, a few
//            more torn bright. Frozen — this is a broken screen, not a
//            playing one.
//   EYE      iris rings and radial spokes around a black pupil, one glint.
//
// Each screen draws a 3-colour scheme from an 8-hue neon family, and every
// archetype is written as three scalar FIELDS against those three colours:
// content = qa*f.x + qb*f.y + qc*f.z. That factoring is what makes the LOD
// honest — the screen's mean is qa*m.x + qb*m.y + qc*m.z with m the fields'
// own means, so the far-field colour is computed, not guessed.
//
// THE SCREEN IS OPAQUE. The facade hook is additive, so a bright emitter
// dropped on the wall leaves the wall's own lit windows punching through it —
// which is exactly what the first cut of this file did, and it read as a
// projection rather than a panel. So the panel first SUBTRACTS the core's
// window ladder over its whole outer rectangle (cc_adscreens_wall below
// recomputes it, at the core's own fp_eff, which we can form because
// u.cam_origin is in the uniform block) and then adds its own emission. The
// bezel subtracts and adds nothing, so it is a genuinely dark frame rather
// than a strip of bare wall, and the screen sits ON the building.
//
// BRIGHTNESS, and why it is not the brief's 2.5. At exposure 6 against a
// white point of 15, radiance 2.5 lands at display 1.0 EXACTLY: a screen
// normalised to a mean radiance of 2.5 is a wall-sized white rectangle with
// no hue and no content, which is what it rendered as. The tone map is what
// sets this scale, so the normalisation targets the mean colour's LARGEST
// channel (the one that clips first) rather than its luminance, and puts it
// near 0.6-1.4. The content's own contrast then carries highlights — glints,
// hot cores, torn static rows — up past 4, where they clip to white on
// purpose, the way a real LED wall photographs: a white-hot core in a field
// that still has colour.
//
// LOD. Content dissolves into the screen's mean over fp in [0.055, 0.33] x
// the screen's short side: full detail while the short side is wider than
// ~18 px, gone by ~3. The screen itself NEVER goes — past the dissolve it is
// a flat rectangle of exactly the colour its content averaged to, which is
// what a city mega-screen is at two kilometres. Inside that window the four
// periodic generators (glyph strokes, test-card bars, noise cells, iris
// rings) each blend to their OWN mean over their OWN feature size, which is
// far finer than the screen's, and the point highlights are gaussians
// widened by the pixel with their integral held. Nothing aliases on the way
// down and nothing vanishes at the bottom.
//
// THE MEANS ARE PER-SCREEN, NOT ENSEMBLE. A single-feature archetype's mean
// depends strongly on its own hash draw — a fat capsule covers three times
// the panel a thin one does — so an ensemble constant is wrong by 15-25% for
// any particular screen, and that error IS the brightness step between the
// resolved screen and the flat one. Every compact feature here therefore
// carries a closed form in its own draw: capsule and annulus areas, Steiner
// offsets for the rim and halo bands, exact gaussian integrals for the
// glints. The two panel-filling gradients (FACE's vignette, EYE's radial)
// are cubics in the aspect fitted to their erf integrals (max error 0.02%).
// The many-cell archetypes (GLYPH, STRIPES, STATIC) need no per-screen form:
// a screen holds dozens of cells, so the ensemble mean IS the screen's mean —
// except STRIPES' bar brightness, which is drawn per colour index rather than
// per bar so that its per-screen mean stays exact with only 2-5 bars a colour.
//
// Validation, twice over. On the CPU: a numpy port of these functions,
// integrated on a 192^2 quadrature grid per screen over 120 screens x 6
// aspects, against the closed forms. Per-screen error is under 1% on every
// dominant field for aspects 0.55 and wider, rising to ~7% of the panel mean
// at the narrowest 1:2.5 banner, where compact features clip against the
// panel edge and the unclipped areas overstate them. (For contrast, the
// ensemble constants this file started with were off by 2.8x on GLYPH and 3x
// on PRODUCT's halo, and gave FACE's glint a NEGATIVE mean.)
//
// And on the GPU, which is the claim that matters: fp is inversely
// proportional to render width at fixed fov, so rendering one parked camera
// at widths from 1920 down to 120 walks every screen in the frame through
// the entire LOD ladder with the geometry, the occlusion and the hash draws
// all held fixed. Over that 16x sweep the panels' mean displayed colour
// moves by at most 1.3% between adjacent steps and 2.6% end to end, with no
// step at either hand-off. That is the no-pops-on-a-dolly claim, measured
// rather than asserted.

const cc_adscreens_MIN_H: f32 = 50.0;      // shorter buildings get no screen
const cc_adscreens_BLDG_FRAC: f32 = 0.085; // of tall buildings
const cc_adscreens_FACE_FRAC: f32 = 0.55;  // of a chosen building's facades
const cc_adscreens_MIN_FACADE_W: f32 = 14.0;
const cc_adscreens_MIN_SCREEN: f32 = 6.0;
const cc_adscreens_BASE_LO: f32 = 8.0;     // lower edge above the tier base
const cc_adscreens_BASE_HI: f32 = 25.0;
const cc_adscreens_BEZEL_FRAC: f32 = 0.028;
const cc_adscreens_BEZEL_MIN: f32 = 0.40;
// Content -> mean, as a fraction of the screen's short side (m/px).
const cc_adscreens_LOD_LO: f32 = 0.055;
const cc_adscreens_LOD_HI: f32 = 0.33;
// Peak-channel radiance of the screen's MEAN colour (see the header): the
// band a screen's overall level is drawn from.
const cc_adscreens_WANT_LO: f32 = 0.60;
const cc_adscreens_WANT_HI: f32 = 1.40;
// Light the screen throws on the wall around it. A screen this bright with a
// perfectly sharp edge reads as a decal; a little spill is what puts it in
// front of the masonry.
const cc_adscreens_SPILL: f32 = 0.055;
const cc_adscreens_SPILL_SIG: f32 = 1.6;   // metres, scaled by screen size

// --- field means -----------------------------------------------------------
// The panel-filling gradients, as cubics in the aspect ratio: the mean of
// exp(-(a x^2 + b y^2)) over the panel, which is an erf product WGSL has no
// erf for. Fitted over ar in [0.40, 1.34]; max relative error 0.02%.
// FACE's vignette, exp(-3 x^2 - 4.05 y^2):
const cc_adscreens_VIGN_C: vec4<f32> =
    vec4<f32>(0.743370, 0.012539, -0.231555, 0.069387);
// EYE's background, exp(-2.2 (x^2 + y^2)):
const cc_adscreens_EYEBG_C: vec4<f32> =
    vec4<f32>(0.840833, 0.016997, -0.198110, 0.051518);
// GLYPH: mean ink of the 3-stroke alphabet, blanks (1 cell in 8) included.
// Measured over 3000 alphabet draws on a 96^2 grid: 0.1388.
const cc_adscreens_M_GLYPH: f32 = 0.1388;
// STRIPES: the bottom 18% of the panel dimmed to 0.15 -> 0.82 + 0.18*0.15.
const cc_adscreens_STRIPE_FOOT: f32 = 0.847;
// STATIC: a uniform hash averages to a half; 6% of rows are torn bright.
const cc_adscreens_M_STATIC: vec2<f32> = vec2<f32>(0.5, 0.06);
// EYE: E[(0.45 + 0.75*rings)(0.70 + 0.55*spokes)] at rings = spokes = 1/2.
const cc_adscreens_IRIS_MEAN: f32 = 0.80438;
const cc_adscreens_TAU: f32 = 6.2831853;
const cc_adscreens_SQRT_HALF_PI: f32 = 1.2533141;

fn cc_adscreens_cubic(c: vec4<f32>, x: f32) -> f32 {
    return c.x + x * (c.y + x * (c.z + x * c.w));
}

// The advertising palette: saturated where the windows are not. Deliberately
// louder than city_window_color — a screen that shares the facade's spectrum
// stops being a screen.
fn cc_adscreens_neon(k: f32) -> vec3<f32> {
    let i = i32(clamp(k, 0.0, 0.9999) * 8.0);
    if (i == 0) { return vec3<f32>(1.00, 0.13, 0.42); }  // rose
    if (i == 1) { return vec3<f32>(0.10, 0.90, 1.00); }  // cyan
    if (i == 2) { return vec3<f32>(1.00, 0.60, 0.10); }  // amber
    if (i == 3) { return vec3<f32>(0.52, 1.00, 0.20); }  // acid green
    if (i == 4) { return vec3<f32>(0.60, 0.30, 1.00); }  // violet
    if (i == 5) { return vec3<f32>(1.00, 0.22, 0.10); }  // hot red
    if (i == 6) { return vec3<f32>(0.82, 0.90, 1.00); }  // cold white
    return vec3<f32>(1.00, 0.84, 0.32);                  // warm yellow
}

fn cc_adscreens_sd_seg(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * t);
}

// Area of a disc of radius r centred in a strip of half-width w: the iris of
// a narrow vertical banner runs off both sides of it, and an unclipped pi r^2
// would overstate its mean by half.
fn cc_adscreens_disc_clip(r: f32, w: f32) -> f32 {
    let full = 3.14159265 * r * r;
    if (r <= w) {
        return full;
    }
    let seg = r * r * acos(clamp(w / r, -1.0, 1.0))
            - w * sqrt(max(r * r - w * w, 0.0));
    return full - 2.0 * seg;
}

// Does this hit sit on the +nh face of that box? The facade hook gets a
// position and a normal but not which of the building's boxes it belongs to,
// and the screen has to be laid out on THAT box's extent.
fn cc_adscreens_on_face(p: vec3<f32>, nh: vec2<f32>,
                        bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    if (p.z < bmin.z - 0.05 || p.z > bmax.z + 0.05) {
        return false;
    }
    let far_c = select(bmin.xy, bmax.xy, nh > vec2<f32>(0.0));
    if (abs(dot(p.xy, nh) - dot(far_c, nh)) > 0.06) {
        return false;
    }
    let tg = vec2<f32>(-nh.y, nh.x);
    let ua = dot(bmin.xy, tg);
    let ub = dot(bmax.xy, tg);
    let u = dot(p.xy, tg);
    return u > min(ua, ub) - 0.06 && u < max(ua, ub) + 0.06;
}

// --- what the panel covers -------------------------------------------------
//
// The core's window ladder at this point on the wall, recomputed so the panel
// can subtract it. This mirrors the facade branch of city_shade: the same
// lattice, the same style switch, the same two far-field octaves and the same
// fp_eff hand-offs. It is the one place this component reaches into core
// behaviour rather than calling it, and it is duplication only because the
// core's facade emission is inlined in city_shade rather than factored into a
// function a component could call. If it is ever factored out, this goes.
//
// It includes the cc_window_glyph term, because it has to: the core
// multiplies a lit pane by whatever the glyph components put behind the
// glass, and leaving it out left windowlife's occupants standing on the
// panel as bright silhouettes — visible, and the one defect an
// occlusion-only test render still showed. That dispatcher is a composer
// guarantee rather than another component's symbol (it exists, returning
// 1, whatever is registered), and WGSL resolves module-scope functions out
// of order, so calling it forward is safe. With it in, the wall under the
// panel goes exactly black.
fn cc_adscreens_wall(cc: CityCell, uc: f32, vc: f32, fp: f32,
                     fp_eff: f32) -> vec3<f32> {
    let fp_glyph = fp;
    let iu = i32(floor(uc / cc.win_pitch));
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
    let fu = fract(uc / cc.win_pitch);
    let fv = fract(vc / CITY_FLOOR_H);
    let pane = fu > cc.pane_lo.x && fu < cc.pane_hi.x
            && fv > cc.pane_lo.y && fv < cc.pane_hi.y;
    var is_on = wh.x < cc.lit_frac;
    if (cc.win_style == 1) {
        let sh = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu >> 1) * 0x9e3779b9u),
            cc.seed.y ^ (bitcast<u32>(iv) * 0x51ed270bu)));
        is_on = sh.x < cc.lit_frac * 1.15;
    } else if (cc.win_style == 2) {
        let sh = city_rand4(vec2<u32>(
            cc.seed.x ^ (bitcast<u32>(iu) * 0x9e3779b9u),
            cc.seed.y ^ (bitcast<u32>(iv >> 2) * 0x51ed270bu)));
        is_on = sh.x < cc.lit_frac * 1.10;
    } else if (cc.win_style == 3) {
        let fl = city_rand4(vec2<u32>(
            cc.seed.x ^ 0x7feb352du,
            bitcast<u32>(iv) * 0x846ca68bu));
        is_on = fl.x < cc.lit_frac * 1.5 && wh.x < 0.88;
    }
    let lit = pane && !floor_dark && is_on;
    var e_win = vec3<f32>(0.0);
    if (lit) {
        let style_gain = select(
            select(1.0, 0.80, cc.win_style == 1),
            0.55, cc.win_style == 3);
        let bright = (0.25 + 5.0 * pow(wh.z, 7.0)) * style_gain;
        var cdraw = wh.y;
        if (cc.win_mono >= 0.0) {
            cdraw = clamp(cc.win_mono + (wh.y - 0.5) * 0.08, 0.0, 1.0);
        }
        e_win = city_window_color(cdraw, cc.palette_bias)
                * (CITY_WIN_RADIANCE * bright);
        let pane_uv = (vec2<f32>(fu, fv) - cc.pane_lo)
                      / max(cc.pane_hi - cc.pane_lo, vec2<f32>(1e-4));
        e_win = e_win * cc_window_glyph(cc, wh, pane_uv, fp_glyph);
    }
    let e_mean_base = CITY_PALETTE_MEAN
        * (cc.lit_frac * cc.pane_frac * (1.0 - CITY_DARK_FLOOR_FRAC)
           * CITY_WIN_RADIANCE * CITY_WIN_BRIGHT_MEAN);
    let ibu = iu >> 2;
    let ibv = iv >> 2;
    let bh1 = city_rand4(vec2<u32>(
        cc.seed.x ^ (bitcast<u32>(ibu) * 0x85ebca6bu),
        cc.seed.y ^ (bitcast<u32>(ibv) * 0xc2b2ae35u)
    ));
    var block_color = CITY_PALETTE_MEAN;
    if (bh1.y < 0.03) {
        block_color = city_window_color(0.90 + 0.09 * bh1.z, cc.palette_bias);
    }
    let block_var = 0.10 + 1.8 * pow(bh1.x, 5.0);
    let e_block = block_color
        * (cc.lit_frac * cc.pane_frac * (1.0 - CITY_DARK_FLOOR_FRAC)
           * CITY_WIN_RADIANCE * CITY_WIN_BRIGHT_MEAN
           * CITY_MEAN_COMP_BLOCK * block_var);
    let bh2 = city_rand4(vec2<u32>(
        cc.seed.x ^ 0x51ed270bu,
        bitcast<u32>(iv >> 4) * 0x9e3779b9u
    ));
    let band = 0.70 + 0.60 * bh2.x;
    let e_flat = e_mean_base * (CITY_MEAN_COMP_FLAT * band);
    let b1 = smoothstep(CITY_WIN_LOD_START, CITY_WIN_LOD_FULL, fp_eff);
    let b2 = smoothstep(CITY_BLOCK_LOD_START, CITY_BLOCK_LOD_FULL, fp_eff);
    return mix(e_win, mix(e_block, e_flat, b2), b1);
}

// --- the archetypes --------------------------------------------------------
//
// All of them take screen-local coordinates and return the three scalar
// fields that multiply the screen's three colours, plus — where the mean
// depends on the screen's own draw — the closed form of those fields' means.
// px and py are in units of the screen HEIGHT (px = (su - 0.5) * ar), so a
// circle drawn in them is round on the wall whatever the panel's shape. aa is
// half a pixel in those units; ps is the pixel's own half-width, for the
// gaussians that have to conserve their integral rather than shrink out of
// existence.

// A giant face. The eyes are blobs and the mouth is a bar, because a face
// assembled from parts you can name is a cartoon; what makes this one land is
// that the parts are just dark masses in the right places, over a vignette
// that could as easily be a light. The neon pinpricks in the eyes are the
// wrongness: nobody's eyes do that.
fn cc_adscreens_face(ch: vec4<f32>, px: f32, py: f32, aa: f32,
                     ps: f32) -> vec3<f32> {
    let sep = 0.13 + 0.07 * ch.x;
    let ey = 0.10 + 0.06 * ch.y;
    let er = 0.055 + 0.030 * ch.z;
    let my = -0.14 - 0.09 * ch.w;
    let mhw = 0.13 + 0.08 * ch.x;
    let mhh = 0.016 + 0.022 * ch.y;
    // Eyes are a little taller than wide, and the mouth is a slab.
    let d1 = length(vec2<f32>(px + sep, (py - ey) * 0.82));
    let d2 = length(vec2<f32>(px - sep, (py - ey) * 0.82));
    let e1 = 1.0 - smoothstep(er * 0.70, er * 1.15 + aa, d1);
    let e2 = 1.0 - smoothstep(er * 0.70, er * 1.15 + aa, d2);
    let mo = (1.0 - smoothstep(mhw - 0.02 - aa, mhw + 0.02 + aa, abs(px)))
           * (1.0 - smoothstep(mhh - aa, mhh + 0.012 + aa, abs(py - my)));
    let mask = clamp(max(max(e1, e2), mo), 0.0, 1.0);
    let vign = 0.42 + 0.58 * exp(-3.0 * (px * px + 1.35 * py * py));
    let skin = vign * (1.0 - 0.90 * mask);
    // Two point highlights, set off-centre in the eyes so the gaze is not
    // straight out. Widened by the pixel with the integral held, so they dim
    // honestly instead of flickering.
    let s0 = er * 0.26;
    let s = sqrt(s0 * s0 + ps * ps);
    let amp = 3.2 * (s0 * s0) / (s * s);
    let k = -0.5 / (s * s);
    let o = er * 0.28;
    let g1 = vec2<f32>(px + sep - o, py - ey - er * 0.18);
    let g2 = vec2<f32>(px - sep - o, py - ey - er * 0.18);
    let glint = amp * (exp(k * dot(g1, g1)) + exp(k * dot(g2, g2)));
    return vec3<f32>(skin, glint, 0.0);
}

fn cc_adscreens_face_mean(ch: vec4<f32>, ar: f32) -> vec3<f32> {
    let sep = 0.13 + 0.07 * ch.x;
    let ey = 0.10 + 0.06 * ch.y;
    let er = 0.055 + 0.030 * ch.z;
    let my = -0.14 - 0.09 * ch.w;
    let mhw = 0.13 + 0.08 * ch.x;
    let mhh = 0.016 + 0.022 * ch.y;
    let vign = 0.42 + 0.58 * cc_adscreens_cubic(cc_adscreens_VIGN_C, ar);
    // The blobs cut the vignette where they sit, so each area is weighted by
    // the vignette at its own centre, not by the panel average.
    let re = 0.925 * er;
    let a_eye = 3.14159265 * re * re / 0.82;
    let a_mouth = 4.0 * mhw * (mhh + 0.006);
    let ve = 0.42 + 0.58 * exp(-3.0 * (sep * sep + 1.35 * ey * ey));
    let vm = 0.42 + 0.58 * exp(-3.0 * (1.35 * my * my));
    let skin = vign - 0.90 * (2.0 * a_eye * ve + a_mouth * vm) / ar;
    let s0 = er * 0.26;
    let glint = 3.2 * cc_adscreens_TAU * s0 * s0 * 2.0 / ar;
    return vec3<f32>(skin, glint, 0.0);
}

// A product: one luminous capsule, a rim where the silhouette turns away,
// and a halo bleeding into a saturated gradient. Bottle, ampoule, pill,
// canister — the point is that it is never quite any of them.
fn cc_adscreens_product(ch: vec4<f32>, sv: f32, px: f32, py: f32,
                        aa: f32) -> vec3<f32> {
    let th = (ch.x - 0.5) * 0.9;
    let ln = 0.10 + 0.11 * ch.y;
    let r = 0.055 + 0.050 * ch.z;
    let ax = vec2<f32>(sin(th), cos(th));
    let p = vec2<f32>(px, py + 0.02);
    let d = cc_adscreens_sd_seg(p, -ax * ln, ax * ln) - r;
    let body = 1.0 - smoothstep(-aa - 0.004, aa + 0.004, d);
    let rim = exp(-abs(d) * 22.0);
    let bg = (0.18 + 0.82 * pow(1.0 - sv, 1.3)) * 0.55;
    let dh = max(d, 0.0);
    let halo = 1.1 * exp(-dh * dh / (2.0 * 0.075 * 0.075));
    return vec3<f32>(bg, body * 2.4 + rim * 1.1, halo);
}

// Areas by Steiner: a convex body's offset region at distance s has area
// (perimeter + 2 pi s) ds, which integrates the rim's exponential and the
// halo's gaussian in closed form. The inward rim saturates at the inradius.
fn cc_adscreens_product_mean(ch: vec4<f32>, ar: f32) -> vec3<f32> {
    let ln = 0.10 + 0.11 * ch.y;
    let r = 0.055 + 0.050 * ch.z;
    let area = 4.0 * ln * r + 3.14159265 * r * r;
    let peri = 4.0 * ln + 2.0 * 3.14159265 * r;
    let k = 22.0;
    let ekr = exp(-k * r);
    let rim_out = peri / k + 2.0 * 3.14159265 / (k * k);
    let rim_in = peri * (1.0 - ekr) / k
               - 2.0 * 3.14159265 * (1.0 - ekr * (1.0 + k * r)) / (k * k);
    let sig = 0.075;
    let halo = 1.1 * (area + peri * sig * cc_adscreens_SQRT_HALF_PI
                      + 2.0 * 3.14159265 * sig * sig);
    // The background gradient is exact: mean of (0.18 + 0.82 (1-sv)^1.3)*0.55.
    return vec3<f32>(0.29509,
                     (2.4 * area + 1.1 * (rim_out + rim_in)) / ar,
                     halo / ar);
}

// A wall of writing that is not writing. Three axis-aligned strokes per cell
// with endpoints snapped to a 3x3 lattice: that quantization is the whole
// trick — free-floating segments read as scribble, snapped ones read as
// characters, and a character set you cannot read is more unsettling than
// one you can. One cell in eight is blank, which is what gives the rows
// their rhythm.
fn cc_adscreens_glyph_cell(bits: vec2<u32>, cu: f32, cv: f32,
                           aa: f32) -> f32 {
    if ((bits.y & 7u) == 0u) {
        return 0.0;
    }
    let p = vec2<f32>(cu, cv);
    let t = 0.075;
    var m = 0.0;
    for (var k: u32 = 0u; k < 3u; k = k + 1u) {
        let sft = k * 8u;
        let horiz = ((bits.x >> sft) & 1u) == 0u;
        let c = 0.14 + 0.72 * f32((bits.x >> (sft + 1u)) & 3u) / 3.0;
        let s = 0.14 + 0.72 * f32((bits.x >> (sft + 3u)) & 3u) / 3.0;
        let e = min(0.86, s + 0.72 * (0.30 + 0.60
                    * f32((bits.x >> (sft + 5u)) & 3u) / 3.0));
        var a: vec2<f32>;
        var b: vec2<f32>;
        if (horiz) {
            a = vec2<f32>(s, c);
            b = vec2<f32>(e, c);
        } else {
            a = vec2<f32>(c, s);
            b = vec2<f32>(c, e);
        }
        let d = cc_adscreens_sd_seg(p, a, b);
        m = max(m, 1.0 - smoothstep(t - aa, t + aa, d));
    }
    return m;
}

// Iris rings around a black pupil. The staple, and it earns its place: an
// eye at forty metres across is the one image on this wall that looks back.
fn cc_adscreens_eye(ch: vec4<f32>, px: f32, py: f32, aa: f32, ps: f32,
                    damp: f32) -> vec3<f32> {
    let rr = 0.28 + 0.12 * ch.x;
    let pu = rr * (0.24 + 0.14 * ch.y);
    let cx = (ch.z - 0.5) * 0.10;
    let cy = (ch.w - 0.5) * 0.08;
    let q = vec2<f32>(px - cx, py - cy);
    let d = length(q);
    let bg = 0.12 + 0.55 * exp(-2.2 * d * d);
    let nr = 14.0 + 10.0 * ch.y;
    let ns = 14.0 + floor(ch.x * 14.0);
    // Rings and spokes are the finest periodic structure on this screen and
    // blend to their own means (1/2 and 1/2) well before the panel does.
    let rings = mix(0.5 + 0.5 * sin(nr * cc_adscreens_TAU * (d / rr)
                                    + ch.z * cc_adscreens_TAU), 0.5, damp);
    let spokes = mix(0.5 + 0.5 * sin(atan2(q.y, q.x) * ns), 0.5, damp);
    let shape = (1.0 - smoothstep(rr * 0.90, rr + aa, d))
              * smoothstep(pu * 0.85, pu * 1.15 + aa, d);
    let iris = shape * (0.45 + 0.75 * rings) * (0.70 + 0.55 * spokes);
    let s0 = rr * 0.10;
    let s = sqrt(s0 * s0 + ps * ps);
    let amp = 5.0 * (s0 * s0) / (s * s);
    let g = vec2<f32>(q.x + rr * 0.30, q.y - rr * 0.34);
    let glint = amp * exp(-0.5 * dot(g, g) / (s * s));
    return vec3<f32>(bg, iris, glint);
}

fn cc_adscreens_eye_mean(ch: vec4<f32>, ar: f32) -> vec3<f32> {
    let rr = 0.28 + 0.12 * ch.x;
    let pu = rr * (0.24 + 0.14 * ch.y);
    let bg = 0.12 + 0.55 * cc_adscreens_cubic(cc_adscreens_EYEBG_C, ar);
    // The annulus between the two smoothstep midpoints, clipped by the panel
    // sides — a tall banner cuts the iris off well inside its radius.
    let a_ann = cc_adscreens_disc_clip(0.95 * rr, 0.5 * ar)
              - 3.14159265 * pu * pu;
    let s0 = rr * 0.10;
    return vec3<f32>(bg,
                     cc_adscreens_IRIS_MEAN * a_ann / ar,
                     5.0 * cc_adscreens_TAU * s0 * s0 / ar);
}

// --- the hook --------------------------------------------------------------

fn cc_adscreens_facade(cc: CityCell, h: CityHit, uc: f32, vc: f32,
                       fp: f32) -> vec3<f32> {
    if (!cc.built || cc.height < cc_adscreens_MIN_H) {
        return vec3<f32>(0.0);
    }
    // Screens hang on flat vertical wall. A tapered shaft's skin is neither,
    // and the same rule that keeps antennas off a spire keeps screens off it.
    if (abs(h.normal.z) > 0.02) {
        return vec3<f32>(0.0);
    }
    // Which buildings. The base rate is one tall building in twelve, but not
    // spread evenly: screens CLUSTER. A uniform sprinkle put a handful on the
    // whole skyline and none at all in either canyon the harness frames, and
    // that is also not how a city works — the walls that carry advertising
    // are the ones a crowd walks past. So the draw is weighted by the
    // cascade percentile that already sets height and occupancy: downtown
    // runs about 2.5x the base rate, the outskirts a quarter of it, and the
    // megatower district (which is the shot) reliably carries screens.
    let dens_w = 0.30 + 2.20 * smoothstep(0.50, 0.95, cc.rank);
    let bd = city_rand4(cc.seed ^ vec2<u32>(0x1f83d9abu, 0x5be0cd19u));
    if (bd.x >= cc_adscreens_BLDG_FRAC * dens_w) {
        return vec3<f32>(0.0);
    }

    let nh = normalize(h.normal.xy + vec2<f32>(1e-9, 0.0));
    let tg = vec2<f32>(-nh.y, nh.x);
    var bmin = cc.b1min;
    var bmax = cc.b1max;
    var tier: u32 = 0u;
    var found = cc_adscreens_on_face(h.pos, nh, cc.b1min, cc.b1max);
    if (cc.tiers >= 2 && cc_adscreens_on_face(h.pos, nh, cc.b2min, cc.b2max)) {
        bmin = cc.b2min; bmax = cc.b2max; tier = 1u; found = true;
    }
    if (cc.tiers >= 3 && cc_adscreens_on_face(h.pos, nh, cc.b3min, cc.b3max)) {
        bmin = cc.b3min; bmax = cc.b3max; tier = 2u; found = true;
    }
    if (!found) {
        return vec3<f32>(0.0);
    }
    let ua = dot(bmin.xy, tg);
    let ub = dot(bmax.xy, tg);
    let u0 = min(ua, ub);
    let fw = abs(ub - ua);
    if (fw < cc_adscreens_MIN_FACADE_W) {
        return vec3<f32>(0.0);
    }

    var fid: u32 = 0u;
    if (nh.x < -0.5) { fid = 1u; }
    else if (nh.y > 0.5) { fid = 2u; }
    else if (nh.y < -0.5) { fid = 3u; }
    let key = vec2<u32>(cc.seed.x ^ (fid * 0x9e3779b9u) ^ 0x243f6a88u,
                        cc.seed.y ^ (tier * 0x85ebca6bu) ^ 0x13198a2eu);
    let ph = city_rand4(key);
    if (ph.x >= cc_adscreens_FACE_FRAC) {
        return vec3<f32>(0.0);
    }
    let ph2 = city_rand4(vec2<u32>(key.x ^ 0xa4093822u, key.y ^ 0x299f31d0u));

    // Size. A merged superblock's facade is already twice as wide, and it is
    // allowed a bigger fraction of it on top: those are the giants.
    let wlo = select(0.30, 0.35, cc.merged);
    let whi = select(0.70, 0.86, cc.merged);
    var sw = fw * mix(wlo, whi, ph.y);
    let ar = 0.40 + 0.93 * ph.z;          // width : height, 1:2.5 .. 4:3
    var sz = sw / ar;
    // Where the screen's lower edge sits. A mid-rise takes the plain law —
    // 8 to 25 m above the tier it stands on, which is where a screen goes on
    // a building you walk past. A megatower cannot: its tier-1 wall is
    // hundreds of metres of facade, soar's camera flies well above the
    // street, and a 25-m-high mark on a 400 m shaft is a thing no flight
    // ever sees. So a tall wall places its screen proportionally instead,
    // anywhere in the lower two thirds of itself — the same argument
    // facadeworks makes for signage, and the same crossover.
    let wall_h = bmax.z - bmin.z;
    let z_low = cc_adscreens_BASE_LO
              + (cc_adscreens_BASE_HI - cc_adscreens_BASE_LO) * ph.w;
    let z_tall = wall_h * (0.06 + 0.60 * ph.w);
    let z0 = bmin.z + mix(z_low, max(z_tall, z_low),
                          smoothstep(90.0, 320.0, wall_h));
    let avail = bmax.z - 3.0 - z0;
    if (avail < cc_adscreens_MIN_SCREEN) {
        return vec3<f32>(0.0);
    }
    if (sz > avail) {                      // keep the aspect, lose the size
        sw = sw * (avail / sz);
        sz = avail;
    }
    if (sw < cc_adscreens_MIN_SCREEN) {
        return vec3<f32>(0.0);
    }
    let slack = max(fw - 2.0 - sw, 0.0);
    let su = (uc - (u0 + 1.0 + slack * ph2.x)) / sw;
    let sv = (vc - z0) / sz;
    // The spill reaches beyond the panel, so the reject has to as well.
    let spill_sig = clamp(cc_adscreens_SPILL_SIG * (1.0 + 0.02 * min(sw, sz)),
                          0.8, 5.0);
    let mu = 2.6 * spill_sig / sw;
    let mv = 2.6 * spill_sig / sz;
    if (su < -mu || su > 1.0 + mu || sv < -mv || sv > 1.0 + mv) {
        return vec3<f32>(0.0);
    }

    // The view direction, and with it the core's own foreshortened footprint:
    // the panel has to subtract the wall at exactly the LOD the core drew it.
    let dir = normalize(h.pos - u.cam_origin.xyz);
    let fp_eff = fp / clamp(abs(dot(dir, h.normal)), 0.20, 1.0);

    // The outer rectangle: everything inside it is panel or bezel, and the
    // wall behind all of it is covered.
    let au = clamp(0.6 * fp / sw, 0.0015, 0.25);
    let av = clamp(0.6 * fp / sz, 0.0015, 0.25);
    let outer =
        smoothstep(-au, au, su) * (1.0 - smoothstep(1.0 - au, 1.0 + au, su))
        * smoothstep(-av, av, sv) * (1.0 - smoothstep(1.0 - av, 1.0 + av, sv));
    // The bezel: a thin dark frame the screen sits inside, so it reads as
    // mounted on the wall rather than painted into it.
    let bez = max(cc_adscreens_BEZEL_MIN,
                  cc_adscreens_BEZEL_FRAC * min(sw, sz));
    let bu = bez / sw;
    let bv = bez / sz;
    let panel =
        smoothstep(bu - au, bu + au, su)
        * (1.0 - smoothstep(1.0 - bu - au, 1.0 - bu + au, su))
        * smoothstep(bv - av, bv + av, sv)
        * (1.0 - smoothstep(1.0 - bv - av, 1.0 - bv + av, sv));

    // Three colours out of the neon family, never the same one twice.
    let qa0 = cc_adscreens_neon(ph2.z);
    let qb0 = cc_adscreens_neon(fract(ph2.z + 0.19 + 0.60 * ph2.w));
    let qc0 = cc_adscreens_neon(fract(ph2.z + 0.44 + 0.30 * ph.y));
    let ch = city_rand4(vec2<u32>(key.x ^ 0x082efa98u, key.y ^ 0xec4e6c89u));

    let px = (su - 0.5) * ar;
    let py = sv - 0.5;
    let aa = clamp(0.6 * fp / sz, 0.0008, 0.20);
    let ps = 0.5 * fp / sz;

    var f: vec3<f32>;
    var m: vec3<f32>;
    var qa: vec3<f32>;
    var qb: vec3<f32>;
    var qc: vec3<f32>;
    // Not every screen runs at the same level. A dead channel is the dimmest
    // thing on the street, not the brightest; a face lit to full white loses
    // the skin it is supposed to have.
    var lvl = 1.0;
    let ad = ph2.y;
    if (ad < 0.17) {                                   // FACE
        f = cc_adscreens_face(ch, px, py, aa, ps);
        m = cc_adscreens_face_mean(ch, ar);
        qa = mix(vec3<f32>(1.00, 0.68, 0.52), qa0, 0.35);
        qb = qb0;
        qc = vec3<f32>(0.0);
        lvl = 0.72;
    } else if (ad < 0.38) {                            // PRODUCT
        f = cc_adscreens_product(ch, sv, px, py, aa);
        m = cc_adscreens_product_mean(ch, ar);
        qa = qa0 * 0.60;
        qb = mix(qb0, vec3<f32>(1.0), 0.55);
        qc = qc0;
    } else if (ad < 0.59) {                            // GLYPH
        // Characters are sized in METRES, not in panel fractions: a shop
        // screen carries three of them and a superblock carries thirty,
        // which is the difference between a sign and a wall of writing.
        // Counting rows as a fixed fraction of the panel instead made every
        // screen three characters tall, so the small ones read as scribble
        // and the giants as three enormous blobs.
        let cell_m = 3.2 + 3.0 * ch.x;
        let nr = clamp(round(sz / cell_m), 2.0, 40.0);
        let nc = clamp(round(sw / (cell_m * (0.85 + 0.35 * ch.y))),
                       1.0, 60.0);
        let gx = su * nc;
        let gy = sv * nr;
        let ix = i32(floor(gx));
        let iy = i32(floor(gy));
        let bits = pcg2d(vec2<u32>(
            key.x ^ (bitcast<u32>(ix) * 0x9e3779b9u),
            key.y ^ (bitcast<u32>(iy) * 0x85ebca6bu)));
        // Cell size on the wall sets when the strokes stop resolving.
        let cs = min(sw / nc, sz / nr);
        let dmp = smoothstep(0.30 * cs, 1.1 * cs, fp);
        let gm = mix(cc_adscreens_glyph_cell(
                         bits, fract(gx), fract(gy),
                         clamp(0.6 * fp / cs, 0.002, 0.25)),
                     cc_adscreens_M_GLYPH, dmp);
        f = vec3<f32>(0.40 + 0.30 * (1.0 - sv), gm * 2.6, 0.0);
        m = vec3<f32>(0.55, cc_adscreens_M_GLYPH * 2.6, 0.0);
        qa = select(vec3<f32>(0.42, 0.03, 0.06), vec3<f32>(0.04, 0.07, 0.42),
                    ch.w > 0.5);
        qb = qa0;
        qc = vec3<f32>(0.0);
    } else if (ad < 0.70) {                            // STRIPES
        // Bars are metres wide too, and their count stays a multiple of
        // three so each colour gets exactly a third of the panel.
        let bar_m = 1.6 + 2.4 * ch.y;
        let nb = 3.0 * clamp(round(sw / (3.0 * bar_m)), 2.0, 12.0);
        let j = floor(clamp(su, 0.0, 0.99999) * nb);
        let idx = i32(j) - 3 * (i32(j) / 3);
        // One brightness per COLOUR, not per bar: with only 2-5 bars to a
        // colour, a per-bar draw would leave the screen's true mean 15% away
        // from any constant we could write down, and that gap is the LOD
        // step. Per colour it is exact.
        let bh = city_rand4(vec2<u32>(key.x ^ 0x27d4eb2fu, key.y ^ 0x165667b1u));
        let brs = vec3<f32>(0.55 + 0.90 * bh.x, 0.55 + 0.90 * bh.y,
                            0.55 + 0.90 * bh.z);
        let br = select(select(brs.z, brs.y, idx == 1), brs.x, idx == 0);
        let foot = mix(0.15, 1.0, smoothstep(0.18 - av, 0.18 + av, sv));
        let v = br * foot;
        m = brs * (cc_adscreens_STRIPE_FOOT / 3.0);
        let dmp = smoothstep(0.45 * sw / nb, 1.4 * sw / nb, fp);
        f = mix(vec3<f32>(select(0.0, v, idx == 0), select(0.0, v, idx == 1),
                          select(0.0, v, idx == 2)), m, dmp);
        qa = qa0;
        qb = qb0;
        qc = qc0;
    } else if (ad < 0.81) {                            // STATIC
        // The dead channel's pixel pitch is a physical size — a coarse LED
        // module, 0.7-1.6 m — so the grain reads at the same distance on
        // every screen instead of vanishing on the small ones.
        let pitch_m = 0.7 + 0.9 * ch.x;
        let ncx = clamp(round(sw / pitch_m), 6.0, 220.0);
        let ncy = max(1.0, round(sz / pitch_m));
        let row = floor(sv * ncy);
        let rh = city_rand4(vec2<u32>(
            key.x ^ (bitcast<u32>(i32(row)) * 0x9e3779b9u),
            key.y ^ 0x51ed270bu));
        // A tenth of the rows are sheared: a frozen glitch, not a moving one.
        let uu = fract(su + select(0.0, (rh.y - 0.5) * 0.35, rh.x < 0.10));
        let nz = city_rand4(vec2<u32>(
            key.x ^ (bitcast<u32>(i32(floor(uu * ncx))) * 0x85ebca6bu),
            key.y ^ (bitcast<u32>(i32(row)) * 0xc2b2ae35u)));
        m = vec3<f32>(cc_adscreens_M_STATIC, 0.0);
        let cs = min(sw / ncx, sz / ncy);
        let dmp = smoothstep(0.35 * cs, 1.3 * cs, fp);
        f = vec3<f32>(mix(nz.x, m.x, dmp),
                      mix(select(0.0, 1.0, rh.z < 0.06), m.y, dmp), 0.0);
        qa = vec3<f32>(0.78, 0.82, 0.92);
        qb = qa0;
        qc = vec3<f32>(0.0);
        lvl = 0.55;
    } else {                                           // EYE
        let rr = 0.28 + 0.12 * ch.x;
        let ring_p = rr * sz / (14.0 + 10.0 * ch.y);
        let dmp = smoothstep(0.40 * ring_p, 1.3 * ring_p, fp);
        f = cc_adscreens_eye(ch, px, py, aa, ps, dmp);
        m = cc_adscreens_eye_mean(ch, ar);
        qa = qa0 * 0.55;
        qb = qb0;
        qc = vec3<f32>(1.0, 0.98, 0.95);
    }

    // The screen's own mean colour: the three colours against the three field
    // means. It is both the far-field asymptote and the yardstick the content
    // is normalized against, so the level a screen was designed to have
    // survives whatever its palette draw turned out to be. Normalising on the
    // LARGEST channel rather than on luminance is what keeps the hue: the
    // tone map clips channels one at a time, and a screen normalised by
    // luminance drives its dominant channel far past the white point and
    // comes back white.
    let mean_c = max(qa * m.x + qb * m.y + qc * m.z, vec3<f32>(0.0));
    let peak = max(max(mean_c.r, mean_c.g), mean_c.b);
    let want = mix(cc_adscreens_WANT_LO, cc_adscreens_WANT_HI, ph2.x);
    let gain = want * lvl / max(peak, 1e-3);

    let lod = smoothstep(cc_adscreens_LOD_LO * min(sw, sz),
                         cc_adscreens_LOD_HI * min(sw, sz), fp);
    let content = mix(qa * f.x + qb * f.y + qc * f.z, mean_c, lod);

    // Spill: the wall around the panel catches its light. Cheap, and it is
    // what stops the screen reading as a decal printed on the masonry.
    let dxu = max(max(-su, su - 1.0), 0.0) * sw;
    let dyv = max(max(-sv, sv - 1.0), 0.0) * sz;
    let dout = length(vec2<f32>(dxu, dyv));
    let spill = cc_adscreens_SPILL
              * exp(-0.5 * dout * dout / (spill_sig * spill_sig));

    // Opaque: take the wall out over the whole outer rectangle, then put the
    // screen back inside the bezel.
    return content * (gain * panel)
         + mean_c * (gain * spill * (1.0 - outer))
         - cc_adscreens_wall(cc, uc, vc, fp, fp_eff) * outer;
}

fn cc_extra_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let h_skyway = cc_skyway_trace(o, dir, inv_dir);
    if (h_skyway.hit && h_skyway.t < res.t) { res = h_skyway; }
    return res;
}

fn cc_cell_props_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                       t0: f32, t1: f32, ci: vec2<i32>, cc: CityCell)
        -> CityHit {
    var res: CityHit;
    res.hit = false;
    res.t = 1e30;
    let h_aircars = cc_aircars_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_aircars.hit && h_aircars.t < res.t) { res = h_aircars; }
    let h_rooftopworks = cc_rooftopworks_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_rooftopworks.hit && h_rooftopworks.t < res.t) { res = h_rooftopworks; }
    let h_skybridges = cc_skybridges_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_skybridges.hit && h_skybridges.t < res.t) { res = h_skybridges; }
    let h_streetlife = cc_streetlife_props_trace(o, dir, inv_dir, t0, t1, ci, cc);
    if (h_streetlife.hit && h_streetlife.t < res.t) { res = h_streetlife; }
    return res;
}

fn cc_component_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, fp: f32)
        -> vec3<f32> {
    if (h.kind >= 300 && h.kind < 400) {
        return cc_aircars_shade(h, cc, dir, fp);
    }
    if (h.kind >= 200 && h.kind < 300) {
        return cc_skyway_shade(h, cc, dir, fp);
    }
    if (h.kind >= 600 && h.kind < 700) {
        return cc_rooftopworks_shade(h, cc, dir, fp);
    }
    if (h.kind >= 500 && h.kind < 600) {
        return cc_skybridges_shade(h, cc, dir, fp);
    }
    if (h.kind >= 100 && h.kind < 200) {
        return cc_streetlife_shade(h, cc, dir, fp);
    }
    // An unclaimed component kind is a bug, and this is its color.
    return vec3<f32>(1.0, 0.0, 1.0);
}

fn cc_facade_detail(cc: CityCell, h: CityHit, uc: f32, vc: f32, fp: f32)
        -> vec3<f32> {
    var e = vec3<f32>(0.0);
    e = e + cc_facadeworks_detail(cc, h, uc, vc, fp);
    e = e + cc_adscreens_facade(cc, h, uc, vc, fp);
    return e;
}

fn cc_window_glyph(cc: CityCell, wh: vec4<f32>, pane_uv: vec2<f32>, fp: f32)
        -> vec3<f32> {
    var t = vec3<f32>(1.0);
    t = t * cc_windowlife_glyph(cc, wh, pane_uv, fp);
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
        // The city's resolving angle is the same knob as the cloud march's:
        // the larger of the true pixel angle and the view-step LOD angle
        // (u.periodic.z, the app's one LOD slider). Coarsening the slider
        // dissolves windows into blocks exactly as it coarsens the march —
        // one lever, one degrees-not-meters law for the whole scene.
        let pixel_angle = max(2.0 * tan_half_fov / img_w, u.periodic.z);
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
    // Ice-detection mode: 0.0 unless research mode has switched it on, and
    // every branch on it below leaves the math bit-identical when off.
    let ice_on = u.light_cache.w > 0.5;
    // In ice mode most of the aerial haze is given up (select is 1.0 with
    // the mode off): full strength converges every distant cloud onto the
    // one horizon color and the phase information drowns in it.
    let aerial_strength = u.sky_horizon.w * select(1.0, 0.3, ice_on);
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
            if (ice_on) {
                // Distance fades into the alien horizon, not the blue one.
                aer = alien_sky(vec3<f32>(ah, 0.0), sun, vec3<f32>(0.0));
            } else {
                aer = sky_radiance(
                    vec3<f32>(ah, 0.0), sun,
                    u.sky_horizon.xyz, u.sky_bloom.xyz, vec3<f32>(0.0),
                    u.cloud_sun.w
                );
            }
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
                col = col + ice_recolor(transmittance
                            * ocean_shade_dispatch(
                                ocean_hit, dir, sun,
                                sun_cone_dir(sun, sun_frame,
                                             sun_cone_seed, penumbra_tan),
                                t_ocean, shadow_jitter,
                                ocean_slope_seed,
                                jitter_on * jitter_scale),
                            ICE_OCEAN_TINT, ice_on);
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

            // Ice-detection tint for this sample's cloud source terms.
            // Exactly 1.0 with the mode off, so the multiplies below are
            // bit-exact no-ops.
            var phase_tint = vec3<f32>(1.0);
            if (ice_on) {
                phase_tint = ice_tint(ice_fraction_at(p));
            }

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
            col = col + ice_recolor(scatter_weight * ms, phase_tint, ice_on);

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
            col = col + ice_recolor(
                transmittance * d_tau * amb * air_t
                * (amb_w_sky * u.ambient_tint.xyz
                   + amb_w_deep * fill_tint), phase_tint, ice_on);

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
                col = col + ice_recolor(
                    transmittance * d_tau * sky_fill * air_t
                    * (fill_w_sky * u.ambient_tint.xyz
                       + fill_w_deep * fill_tint), phase_tint, ice_on);
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
                col = col + ice_recolor(
                    transmittance * d_tau * high_sun_fill * air_t
                    * vis * u.ambient_tint.xyz, phase_tint, ice_on);
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
                col = col + ice_recolor(
                    transmittance * d_tau * diffuse_beam * air_t
                    * ISO_PHASE * DIFFUSE_BEAM_TINT * u.cloud_sun.xyz,
                    phase_tint, ice_on);
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
                col = col + ice_recolor(
                    transmittance * d_tau * bounce * air_t * BOUNCE_TINT,
                    phase_tint, ice_on);
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
            col = col + ice_recolor(transmittance
                        * ocean_shade_dispatch(
                            ocean_hit, dir, sun,
                            sun_cone_dir(sun, sun_frame, sun_cone_seed,
                                         penumbra_tan),
                            t_ocean, shadow_jitter,
                            ocean_slope_seed, jitter_on * jitter_scale),
                        ICE_OCEAN_TINT, ice_on);
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
        } else if (ice_on) {
            sky = alien_sky(dir, sun, u.sky_disc.xyz);
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
