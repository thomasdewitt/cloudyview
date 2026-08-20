// Every number the look depends on, ported one-for-one from Python.
//
// Sources: cloudyview/look.py (shared with witness), cloudyview/soar/engine.py,
// cloudyview/soar/menu.py, cloudyview/soar/app.py, cloudyview/config.py.
// A value that drifts from its Python twin silently changes the picture, so
// tests/test_web_uniform_parity.py runs this file under node and diffs the
// packed uniform block against engine.write_uniforms. Change one, change both.

"use strict";

// --- look.py: the witness realism package -------------------------------

export const SUN_COLOR = [20.2, 21.0, 22.4];
export const LEGACY_AMBIENT = [0.18, 0.225, 0.33];
export const LEGACY_HORIZON = [0.10, 0.18, 0.38];
export const LEGACY_BLOOM = [0.8, 0.6, 0.3];
export const LEGACY_DISC = [50.0, 45.0, 35.0];

export const SPECTRAL_LIGHTING_STRENGTH = 1.0;
export const ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG = 55.0;
export const ATMOSPHERE_MAX_AIRMASS = 20.0;
export const ATMOSPHERE_RAYLEIGH_OD_550 = 0.10;
export const ATMOSPHERE_AEROSOL_OD_550 = 0.12;
export const ATMOSPHERE_AEROSOL_ANGSTROM = 1.3;
export const ATMOSPHERE_RGB_WAVELENGTHS_NM = [680.0, 550.0, 460.0];
export const SUNSET_HORIZON_RADIANCE = [0.42, 0.20, 0.055];

export const LOW_SUN_SKY_FIELD_STRENGTH = 1.0;
export const CONE_STENCIL_THETA_DEG = 2.0;

export const LIGHT_TRANSFER_SPLIT_STRENGTH = 1.0;
export const LIGHT_TRANSFER_FULL_ELEVATION_DEG = 45.0;
export const LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG = 55.0;

export const AERIAL_PERSPECTIVE_STRENGTH = 1.0;
export const AERIAL_BETA_PER_KM = 0.035;
export const AERIAL_SCALE_HEIGHT_M = 2500.0;
// Whether the aerial haze thins with height at all. Off by default: at a
// 2.5 km scale height an upward ray leaves the haze without ever reaching the
// cutoff optical depth, so nothing bounds its march but the range ceiling.
// Uniform haze is unphysical and caps every ray at one distance — the
// cheapest range lever there is. Mirrors soar_host.ViewState.
export const DEFAULT_HAZE_HEIGHT_DEPENDENT = false;

export const OCEAN_REALISM = 1.0;
export const OCEAN_MIP_BIAS = -0.5;
export const OCEAN_GLINT_STRENGTH = 0.85;
export const OCEAN_GLINT_ROUGHNESS = 0.28;
// How much of the slope variance the normal-mip filter removed is drawn
// stochastically per pixel per frame (iter_008). The remainder stays as
// extra microfacet-lobe width. 1 = fully sampled, 0 = fully analytic.
export const OCEAN_SLOPE_DRAW_FRACTION = 0.5;
export const OCEAN_SKY_SHADOW_FLOOR = 0.75;
export const OCEAN_HAZE_EXTINCTION_PER_KM = 0.012;

// One knob for the whole aerosol story — aerial extinction, the horizon
// wedge, the circumsolar lobe, the haze over the sea. Separate sliders would
// let a viewer build a sky no photograph contains. Every haze-dependent term
// is anchored here: at 0.35 each returns its tuned constant exactly, so the
// default look is unchanged. See look.py for the physics of the ramp, and
// spectral.js for the two functions that consume these.
export const HAZE_ANCHOR = 0.35;
export const DEFAULT_HAZE = 1.0;
export const HAZE_MAX = 2.5;
export const AERIAL_BETA_FLOOR_PER_KM = 0.015;
// The clear end of the haze slider, in the distance a viewer reads rather
// than in the aerosol coordinate: 200 km of e-folding, which is past haze 0
// and so past a Rayleigh-limited sky. HAZE_MIN is derived from it in
// spectral.js. The curve is EXTENDED, not rescaled — see look.py.
export const HAZE_MAX_E_FOLDING_KM = 200.0;

// Haze is also the cheapest performance lever there is, which is why the
// lower tiers ask for more of it. In a periodic domain the view march ends
// where the clear-air transmittance runs out (periodic_march_cap in
// raymarch.wgsl), so thicker air is literally a shorter ray: at haze 1 the
// sea-level e-folding length is 9.0 km, at haze 2 it is 3.5 km, and the cap
// moves in proportion. A tier that cannot afford the distance buys the
// atmosphere that hides it.
//
// The cost of this: tiers no longer converge to the same picture, which the
// hold ladder was built to guarantee. That guarantee is now the Still mode's
// (see DEFAULT_HOLD_MODE) rather than the app's, and two machines that
// auto-pick different tiers will show different weather. Thomas asked for it
// with that trade named (2026-08-14). A hand-set haze is never overridden.
// Values are the aerosol coordinate whose sea-level e-folding length is the
// named distance (hazeFromEFoldingKm in spectral.js — not imported here,
// that would be circular). Retuned 2026-08-18 to Thomas's hand-set panel
// values, much clearer across the board; max mirrors high because whatever
// else max buys, it must never be hazier.
export const DEFAULT_HAZE_BY_TIER = {
  max: -0.0380,      // 70 km
  high: -0.0380,     // 70 km
  medium: 0.4027,    // 25.2 km
  low: 0.6590,       // 15 km
  minimal: 1.0905,   // 8 km
};

// --- engine.py defaults -------------------------------------------------

export const DEFAULT_SUN_AZIMUTH = 20.0;
export const DEFAULT_SUN_ELEVATION = 55.0;
export const DEFAULT_EXPOSURE = 4.0;
export const DEFAULT_G_HG = 0.76;
export const DEFAULT_AMBIENT_STRENGTH = 0.15;
// NB: config.py's witness block carries a different (brighter) ocean
// reflectance. soar uses this one — do not cross the wires.
export const DEFAULT_OCEAN_REFLECTANCE = [0.0020, 0.0045, 0.0126];

export const DEFAULT_GRADIENT_SHADING_STRENGTH = 1.50;
export const DEFAULT_GRADIENT_COARSE_WEIGHT = 0.65;
export const DEFAULT_GRADIENT_COARSE_RADIUS_M = 500.0;
export const DEFAULT_DEEP_SHADOW_MS_SUPPRESSION = 0.90;
export const DEFAULT_AMBIENT_OCCLUSION_STRENGTH = 1.00;
export const DEFAULT_AMBIENT_OCCLUSION_FLOOR = 0.24;
export const DEFAULT_BOUNCE_DEPTH_ATTENUATION = 0.80;

export const PERIODIC_AIR_TAU_CUTOFF = 3.912023005428146;  // -ln(0.02)
export const PERIODIC_MAX_RANGE_M = 4.0e5;
export const STEP_VOXEL_FACTOR = 2.0;
export const DEFAULT_MAX_LIGHT_STEPS = 512;

// The library defaults are 0.0 (exact legacy marching); the app opts in, and
// the browser ships the app's values. The coarser light march is most of what
// makes distance read as haze rather than as a hard dark wall.
export const APP_LIGHT_MARCH_LOD_DEGREES = 1.4;
export const APP_VIEW_STEP_LOD_DEGREES = 0.6;

// Tone-map gamma. 1.4 is witness's reference; 3.08 is what the desktop
// window accidentally showed for years (sRGB swapchain double-encode).
// The default is 1.66 — the value Thomas flies and the one every
// 2026-08-11 look constant was tuned against a real photo at; the look
// and the encode are a matched pair now.
export const TONE_MAP_GAMMA_WITNESS = 1.4;
export const TONE_MAP_GAMMA_AS_FLOWN = 3.08;
export const DEFAULT_TONE_MAP_GAMMA = 1.66;
export const TONE_MAP_GAMMA_LIMITS = [1.0, 4.0];

// The extended-Reinhard white point: the exposed radiance that maps to 1.0,
// and so what decides whether a sunlit face is white or a bright grey it
// cannot escape. A shader const until it became the second knob worth
// reaching for after gamma. Default 15.0 — the tuned value, so nothing moves
// unless the slider does. Below 4 the picture clips; above 40 the shoulder
// is so close to linear that the slider stops doing anything.
export const DEFAULT_TONE_MAP_WHITE_POINT = 15.0;
export const TONE_MAP_WHITE_POINT_LIMITS = [4.0, 40.0];

// Display-space contrast about mid-grey, after the gamma encode. 1.0 is
// exactly the identity — see raymarch.wgsl's tone_map for why that is written
// as a multiply-add. Past 1.6 the sky posterizes and the shadows block up;
// below 0.5 the picture is fog.
export const DEFAULT_CONTRAST = 1.0;
export const CONTRAST_LIMITS = [0.5, 1.6];

// --- auto exposure --------------------------------------------------------
//
// The camera-side half of the 2026-08-14 forward-scattering change. The
// diffraction spike puts thin cloud near the sun orders of magnitude above a
// diffusely-lit base in linear radiance — which is physically right, and
// useless under a fixed exposure, because the tone map clamps the veils to
// white and shows the base at the same grey it always had. What a video
// camera does with that scene is stop down for the highlights, and THAT is
// what makes a toward-sun base read dark. So: meter the linear frame, place
// a high percentile of its luminance just under the tone map's white point,
// and glide the exposure there.
//
// This is HIGHLIGHT PROTECTION, not a full AE loop, and the difference was
// measured before it was chosen: a mid-tone meter (key / log-average) drives
// the dark-base views — the ones the transition retune just darkened on
// purpose — up 3-4x, because to an average meter an intentionally dark scene
// and an underexposure are the same thing. So the tuned DEFAULT_EXPOSURE is
// a CEILING the controller never exceeds; the meter can only stop DOWN from
// it, and only when the highlight statistic would clip. On the judge set
// that leaves every no-sun-in-frame view at exactly the shipped look and
// stops the toward-sun / low-sun frames to ~1, which is where the dark
// bottoms come from.
// Auto is a per-tier default (Thomas, 2026-08-14): on everywhere the meter
// is affordable — which measurement says is everywhere but Minimal, where
// the frame budget is so tight that even the glide's accumulation restarts
// hurt. The meter itself is ~0.1-0.5% of a frame's rays. A hand-touched
// toggle survives tier changes, like haze.
export const AUTO_EXPOSURE_DEFAULT_BY_TIER = {
  max: true, high: true, medium: true, low: true, minimal: false,
};
// Metered on a tiny linear render (TONE_MAP compiled out) of the live view:
// 64x36 = 2304 rays per meter, beneath measurement next to a frame's march.
export const AUTO_EXPOSURE_METER_SIZE = [64, 36];
// Every frame: the 200 ms first cut stepped visibly (Thomas). The real
// cadence is set by the readback round-trip — one meter is in flight at a
// time — and the glide's time constant is computed from measured elapsed
// time, so the feel is frame-rate independent.
export const AUTO_EXPOSURE_INTERVAL_MS = 0;
// The highlight statistic: the MEAN of the pixels above this rank. The mean
// (not the percentile itself) is what keeps the sun's aureole and a bright
// veil from hiding inside the top bin — measured, p95 alone reads the
// toward-sun frame at 3.3 where its top-1% mean is 15.7.
export const AUTO_EXPOSURE_PERCENTILE = 0.99;
// Where the highlight statistic would be placed, as exposed radiance
// relative to the white point, at full response.
export const AUTO_EXPOSURE_HIGHLIGHT_FRACTION = 0.90;
// Response strength: the fraction of the log-distance below the ceiling the
// controller actually applies. Full protection (1.0) put the toward-sun
// STEAM frame 8x down and read as too dark (Thomas, 2026-08-14) — a real
// camera lets the region around the sun clip rather than crush the scene.
// At 0.5 that frame stops ~3.5x down and a mild highlight barely moves.
export const AUTO_EXPOSURE_RESPONSE = 0.5;
// Floor and ceiling of the AUTO range. The ceiling is DEFAULT_EXPOSURE by
// design (see above). The manual slider spans EXPOSURE_LIMITS instead.
export const AUTO_EXPOSURE_LIMITS = [1.0, 4.0];
// The hand slider's range: wider than auto's, because manual means manual.
export const EXPOSURE_LIMITS = [0.5, 16.0];
// Glide, deadband, hysteresis: the loop adapts over ~half a second, stops
// once within the stop band so a parked view's accumulation can converge
// (every exposure step restarts it — exposure is in the scene key), and
// wakes again only past the start band.
export const AUTO_EXPOSURE_TIME_CONSTANT_S = 0.5;
export const AUTO_EXPOSURE_DEADBAND_STOP_LOG2 = 0.05;
export const AUTO_EXPOSURE_DEADBAND_START_LOG2 = 0.25;

export const DEFAULT_MOTION_BLEND_ALPHA = 0.58;
export const DEFAULT_MOTION_BLEND_REFERENCE_FPS = 60.0;

// --- motion smoothing, which the slider states the right way up -----------
//
// The accumulator blends each new march into the picture with weight alpha,
// so alpha is the INVERSE of smoothing: 0.9 keeps almost nothing of the last
// frame, 0.15 keeps most of it. The slider used to show alpha under the label
// "Motion smoothing", so turning it up smoothed less (Thomas, 2026-08-14).
// It now shows smoothing itself, on [0, 1], and this is where the two meet.
//
// The floor is per tier because smoothing is worth more the slower the
// machine: it is the one setting that makes a low frame rate read as motion
// blur rather than as stutter, and it costs nothing at all. Minimal reaches
// twice the blend depth High can (alpha 0.15 against 0.30 — depth goes as
// 1/alpha, so ~6.7 frames against ~3.3).
export const MOTION_ALPHA_AT_ZERO_SMOOTHING = 0.90;
export const MOTION_ALPHA_FLOOR_BY_TIER = {
  max: 0.30, high: 0.30, medium: 0.24, low: 0.20, minimal: 0.15,
};
// Retuned 2026-08-18 to Thomas's hand-set panel values. Medium smooths LESS
// than high on purpose: medium flies at a lower render scale, and smoothing
// stacked on upscale blur reads as smear.
export const DEFAULT_MOTION_SMOOTHING_BY_TIER = {
  // Max is the one tier that does not need smoothing to hide sampling noise:
  // it has already paid for eight samples by the time the frame is shown.
  max: 0.30, high: 0.55, medium: 0.50, low: 0.60, minimal: 0.80,
};

export function motionAlphaForSmoothing(smoothing, tier) {
  const floor = MOTION_ALPHA_FLOOR_BY_TIER[tier];
  if (floor === undefined) throw new Error(`unknown quality tier '${tier}'.`);
  const s = Math.max(0, Math.min(1, Number(smoothing)));
  return MOTION_ALPHA_AT_ZERO_SMOOTHING
    - s * (MOTION_ALPHA_AT_ZERO_SMOOTHING - floor);
}

/** The inverse, for a tier change that must not move the slider's meaning. */
export function motionSmoothingForAlpha(alpha, tier) {
  const floor = MOTION_ALPHA_FLOOR_BY_TIER[tier];
  if (floor === undefined) throw new Error(`unknown quality tier '${tier}'.`);
  const span = MOTION_ALPHA_AT_ZERO_SMOOTHING - floor;
  return Math.max(0, Math.min(
    1, (MOTION_ALPHA_AT_ZERO_SMOOTHING - Number(alpha)) / span));
}

// --- level of detail ------------------------------------------------------
//
// APP_LIGHT_MARCH_LOD_DEGREES and APP_VIEW_STEP_LOD_DEGREES are angular: a
// step is allowed to subtend this much of the view, so it grows with distance
// and the far field costs a fraction of the near one. This multiplies both,
// and it is the second-cheapest tier lever after haze — coarser far-field
// stepping is most of what makes distance read as haze rather than as a wall
// (the note above APP_LIGHT_MARCH_LOD_DEGREES), so it degrades along the
// grain of the look instead of against it. Past ~2.5 the far field starts to
// band on high-contrast tops, which is why the slider stops where it does.
// High and max fly at 0.5 — half the angular step, so the far field is
// marched twice as finely as the tuned constants alone would ask for
// (Thomas, 2026-08-15). The two tiers are equal here on purpose: whatever
// else max buys, it must never march more coarsely than high.
// Lower tiers retuned 2026-08-18 (Thomas's hand-set values). Minimal's
// default sits ON the slider ceiling: coarser is off the table there,
// finer is the only direction that changes anything.
export const DEFAULT_LOD_STRENGTH_BY_TIER = {
  max: 0.5, high: 0.5, medium: 1.05, low: 2.0, minimal: 3.0,
};
// The floor is below the default rather than equal to it: a slider whose
// default sits on its own end stop cannot be used to ask for anything finer,
// and finer is only ever slower — never wrong.
export const LOD_STRENGTH_LIMITS = [0.25, 3.0];

// The strength anything that is not holding a framerate renders at: a still,
// a video frame, witness, and the uniform block's own default. A capture is
// already paying for hundreds of accumulated passes, so it has no business
// inheriting a coarse march chosen to keep flight smooth.
//
// Mirrors soar_host.DEFAULT_LOD_STRENGTH, and tests/test_uniform_parity.py
// compares the two hosts' defaults directly — it caught them disagreeing the
// moment the Python side moved, which is the whole reason this is one named
// constant rather than a 0.5 written into three call sites.
export const DEFAULT_LOD_STRENGTH = 0.5;
export const DEFAULT_MOTION_JITTER_SCALE = 0.65;
export const DEFAULT_MOTION_RESET_ANGLE_DEGREES = 8.0;
export const DEFAULT_MOTION_RESET_TRANSLATION_FRACTION = 0.05;

export const UNIFORM_ROWS = 23;
// DERIVED, never written out. These were two independent literals until a row
// was added and only one of them moved: the packer produced 384 bytes, the
// buffer stayed 368, and every draw failed validation with a black screen.
// One of them is a fact about the block and the other was a copy of it. Kept
// derived after that row was reverted, because the bug was never about the
// row — it was about the copy.
export const UNIFORM_NBYTES = UNIFORM_ROWS * 16;

export const AUTO_FP16_MIN_VOXELS = 256 * 1024 * 1024;

// --- quality ------------------------------------------------------------

// A tier is four numbers, and only the first three vary.
//
// `renderScale` is the FLIGHT scale — what the march runs at while the camera
// is moving. What a held view converges to is the hold ladder below.
//
// `stepFactor` is the view march's step, in voxels. `lightStepFactor` is the
// light march's, and it is a separate axis because the two degrade very
// differently. Measured on a 5080 against the TWPICE view at 640x360, taking
// High as the reference picture:
//
//     light step 2x coarser   1.32x faster   RMSE 0.0011
//     light step 4x coarser   1.51x faster   RMSE 0.0018
//     light step 8x coarser   1.86x faster   RMSE 0.022
//
// The light march is about a third of the frame, and coarsening it costs
// almost nothing visually until 8x. Lowering `maxLightSteps` — the old way of
// making the light march cheap — buys a similar amount but TRUNCATES the
// integration rather than coarsening it, so at a low sun the ray runs out of
// steps before it leaves the cloud and the tier brightens in a way that
// depends on the viewing direction. A coarse step integrates the whole path;
// a low cap integrates part of it and stops. So the cap is now a safety
// ceiling at DEFAULT_MAX_LIGHT_STEPS for every tier, and the light march is
// made cheap by its step instead.
//
// Holding the cap equal across tiers has a second effect worth as much as the
// first: it is what the shader is textually specialized on (see
// specializeShader), so one value means ONE raymarch.wgsl module for the whole
// session. Switching tiers — which the startup probe does several times in its
// first half second, and the quality panel does on a click — no longer
// compiles anything, so there is no stall to hide.
//
// The lower tiers therefore do not look like High, on purpose. They are
// allowed to: their job is to be much faster while the camera moves, and the
// hold ladder converges every tier to High's own sampling the moment you stop,
// so the picture you actually study is tier-independent.
export const QUALITY_PRESETS = {
  // `sppPerFrame` is how many jittered marches go into ONE presented frame.
  // Every tier but Max takes a single one and leans on accumulation across
  // frames instead; Max pays for the noise up front so that a frame is clean
  // while it is still moving. See drawFrame for why that is eight separate
  // submissions rather than one eight-times-longer pass.
  max:    { name: "max",    label: "Max",    renderScale: 1.0,
            stepFactor: 2.0, lightStepFactor: 2.0, sppPerFrame: 8,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  high:   { name: "high",   label: "High",   renderScale: 1.0,
            stepFactor: 2.0, lightStepFactor: 2.0, sppPerFrame: 1,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  medium: { name: "medium", label: "Medium", renderScale: 0.70,
            stepFactor: 2.5, lightStepFactor: 4.0, sppPerFrame: 1,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  low:    { name: "low",    label: "Low",    renderScale: 0.30,
            stepFactor: 3.0, lightStepFactor: 8.0, sppPerFrame: 1,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  // Minimal flies at an eighth, not a quarter. A quarter was chosen against a
  // 5080; a GPU sixty times slower renders minimal's own frame in ~30 ms, and
  // an eighth is four times fewer pixels than a quarter, which brings that
  // back inside a vsync. Plain "Minimal", not the old "smooth stills, rough
  // flight" tag — the name is user-facing in the startup toast on exactly the
  // machines it used to insult as "Potato".
  minimal: { name: "minimal", label: "Minimal",
            renderScale: 0.125, stepFactor: 4.0, lightStepFactor: 12.0,
            sppPerFrame: 1, maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
};
export const QUALITY_TIER_NAMES = ["max", "high", "medium", "low", "minimal"];
// Cheapest first. This is the order the startup probe walks, and the order
// matters for more than tidiness — see AUTO_TIER_COST_RATIO_TO_NEXT.
// Max is deliberately absent: the probe walks this list, so a tier that is
// not in it can only ever be reached by a click. Eight marches a frame is a
// choice a machine should never make on someone's behalf.
export const QUALITY_TIERS_CHEAPEST_FIRST = ["minimal", "low", "medium", "high"];
// The tier a Renderer is born with. In practice the startup probe replaces it
// before the first frame is presented (viewer.loadField sets the probe's
// starting tier before Renderer.init compiles anything); this is the answer
// when there is no measurement, which is what the offline capture paths want.
export const DEFAULT_QUALITY_TIER = "high";
// The floor was 0.25 until minimal's flight scale went below it. Only two
// things read it: renderTargetSize's validation, and the Quality panel's
// slider bounds (whose step is 0.025 so that both ends stay reachable). The
// capture paths never see it — they force 1.0.
export const MIN_RENDER_SCALE = 0.125;
export const MAX_RENDER_SCALE = 1.0;
export const RENDER_SCALE_SLIDER_STEP = 0.025;

// --- the progressive hold ladder -----------------------------------------
//
// Flying and holding still want opposite things. While the camera moves, the
// only thing that matters is that the next frame arrives soon, so the march
// runs at the tier's flight scale. The moment the view is held, latency stops
// mattering and the picture starts to: the renderer climbs this ladder toward
// full resolution, accumulating at each rung, so a held view converges to
// something far better than the tier flew at.
//
// Each rung is {scale, sampling}: the render scale to march at, and the tier
// whose step factors to sample with. A rung whose scale is not strictly above
// the previous rung's is dropped when the ladder is built, so a hand-set
// render scale (the Quality panel's slider) simply shortens the ladder instead
// of contradicting it. Rung 0 is always the flight configuration and is
// implicit — it is not listed here.
//
// The top rung SAMPLES like High for every tier without exception — the tier
// governs flight, and a parked picture is stepped and lit the same wherever
// you flew in from. Its SCALE is capped per tier (0.5 on Minimal, 0.75 on
// Low), which is new and is a real retreat from "every tier converges to the
// same still": those two settle to a picture that is upscaled, not native.
//
// The reason is the upscale itself. Minimal flying at 0.125 and holding at
// 1.0 is a 64x jump in pixels for one frame, and on the hardware that picks
// Minimal that frame is most of a second — which is what the view then has to
// be flown out of (Thomas, on the Mac: "much of the feel of slowness might be
// caused by the still frame upscaling"). 0.5 is 16x, lands inside
// HOLD_MAX_FRAME_MS on that hardware instead of being capped by it, and is
// four times the detail it flew at. The difference between a 0.5 upscale and
// a native still is far smaller than the difference between a settled picture
// and one that never arrives.
//
// Intermediate rungs keep the flight tier's sampling because they are meant to
// be cheap stepping stones: their job is to put something settled on screen
// while the expensive rung is still being paid for, not to be the answer.
// (Before the light-step rework this rule also avoided a mid-hold shader
// compile. That reason is gone — every tier now specializes identically — but
// the rule earns its keep on cost alone.)
// The stepping stones only. The TOP rung is not listed here — it is the
// hold render scale below, sampled like High, which the Quality panel now
// exposes as its own slider next to the flight one. Splitting the two is
// what lets "how sharp it gets when I stop" be a number the user sets rather
// than a table entry they cannot see (Thomas, 2026-08-14).
export const QUALITY_HOLD_RUNGS = {
  max:    [],
  high:   [],
  medium: [],
  low:    [{ scale: 0.50, sampling: "low" }],
  minimal: [{ scale: 0.25, sampling: "minimal" }],
};
// Where each tier converges when it is held: the top rung's scale, at High's
// sampling. Equal to the flight scale on High and Max, which is why their
// ladders collapse to one rung and a hold there merely accumulates.
export const HOLD_RENDER_SCALE_BY_TIER = {
  max: 1.0, high: 1.0, medium: 1.0, low: 0.5, minimal: 0.25,
};
// Accumulated frames to spend on a rung before climbing to the next. Small,
// because the point of the low rungs is to show *something* settled while the
// expensive one is still being paid for; the top rung is where convergence
// actually happens.
export const HOLD_RUNG_FRAMES = 4;

// The ceiling on a single held frame, and the reason the climb is gated by a
// measurement rather than run open-loop.
//
// Every rung doubles the linear scale, so it quadruples the pixels, and the
// top rung changes the sampling on top of that. A machine that settles on
// minimal because minimal flies at 10 ms would reach the top rung at something
// like 10 ms x 64 pixels — half a second in one fragment pass, and on Metal a
// long enough pass is not slow, it is a dead device. So the ladder predicts
// the next rung's cost from the rung it is standing on (see
// Renderer._buildHoldLadder) and refuses to climb past this. A capped ladder
// converges where it stands, which is the whole point: the view still
// settles, it just settles at half resolution on hardware that cannot afford
// full. 400 ms is comfortably inside any GPU watchdog and is a frame you only
// ever wait for while parked.
export const HOLD_MAX_FRAME_MS = 400.0;

// The sample count a held view settles at — PARKED_ACCUM_FRAMES_BY_TIER — is
// defined under "capture" below, next to the capture count it used to be.

// --- the startup auto-tier probe -----------------------------------------
//
// What the probe is aiming at: 40 fps flight (Thomas, 2026-08-18 — was a
// 60 Hz vsync, which auto-picked a tier below what machines could carry).
// AUTO_TIER_MARGIN below still applies on top of this.
export const AUTO_TIER_TARGET_MS = 1000.0 / 40.0;
// The escalation rule, and the safety property that hangs off it.
//
// The probe starts at the cheapest tier and escalates only when the tier it
// just MEASURED predicts that the next tier up will still clear the target
// with margin:
//
//     measured_ms * AUTO_TIER_COST_RATIO_TO_NEXT[tier]
//         < AUTO_TIER_TARGET_MS * AUTO_TIER_MARGIN
//
// so no tier is ever rendered without a measurement below it proving it has
// room. That is not merely tidy on Apple silicon: Metal's GPU watchdog kills
// a fragment pass that runs too long, and a killed pass is a lost device,
// which main.js (rightly) treats as fatal. There is no "try high first and
// back off" shortcut here and there must never be one — backing off would
// mean having already submitted the frame that killed the device.
//
// The ratios are measured, not guessed. RTX 5080, 2560x1440 output, TWPICE
// 256^3, two views (a thick one and an overview), at the tier configurations
// defined above. Re-measured 2026-08-14 when the flight scales moved
// (benchmarking/soar_frame_bench.py --output 2560x1440):
//
//     high     7.48 / 5.32 ms
//     medium   2.52 / 2.35     -> high:   2.98x / 2.27x
//     low      0.654 / 0.614   -> medium: 3.85x / 3.82x
//     minimal  0.332 / 0.403   -> low:    1.97x / 1.52x
//
// The constants below round those UP, which makes the gate harder to pass,
// not easier. Note how unequal the rungs are: a single flat threshold would
// be either too timid at the top or reckless at the bottom, which is why the
// gate is a per-tier ratio rather than one number.
//
// One honest caveat about minimal, and it is why its ratio is 4.0 and not the
// ~2 that was measured. At 320x180 the frame stops being about the march at
// all — command encoding, uniform upload and the blit dominate — so minimal
// measures as barely cheaper than low on this card. That floor is CPU- and
// driver-side, so it does NOT shrink on a slower GPU, where the true ratio is
// nearer the work ratio: 5.8x the pixels times a finer step, call it 7x.
// Taking the measured 2 at face value would let a weak machine escalate on
// the strength of a number that describes this one. 4.0 is the old 4.5 nudged
// by the new scales rather than by the new measurement, deliberately.
//
// That weakness is bounded by the ordering rule rather than by these numbers:
// being one rung too optimistic costs a settle point that is sluggish, not a
// machine that freezes, because the tier above is still never rendered
// without its own completed measurement. The catastrophic case — submitting a
// High frame on an M1 having measured nothing — is prevented by the walk
// being strictly one rung at a time, not by the accuracy of these ratios.
export const AUTO_TIER_COST_RATIO_TO_NEXT = {
  minimal: 4.0,
  low: 4.0,
  medium: 3.0,
};
export const AUTO_TIER_MARGIN = 0.75;
// A first verdict BELOW this tier is re-measured once, this long after it
// settled, before being announced — walking upward from the tier already
// proven. Probe noise is one-sided (startup contention only ever inflates a
// measurement), so a high verdict is self-proving and a low one is the case
// worth a second look; the retry can only raise the answer, and the
// no-frame-at-an-unproven-tier invariant holds throughout.
export const AUTO_TIER_CONFIRM_FROM = "medium";
export const AUTO_TIER_CONFIRM_DELAY_MS = 1000.0;
// Frames per tier: the first is thrown away and the rest are measured. The
// throwaway is not politeness, it is the only way the number means anything —
// the first frame at a tier pays for the pipeline creation (which on Metal
// happens at first use, inside the frame) and the render-target allocation at
// that tier's scale. Measuring it would measure the driver.
export const AUTO_TIER_WARMUP_FRAMES = 1;
// Timed frames per tier, and why it is odd: the estimate is their MEDIAN. A
// mean is dragged by any one stall, and a minimum is optimistic on a machine
// whose frame times are genuinely spread — which is exactly the machine this
// is protecting. Three is the smallest count for which a median means
// anything, and the probe is paid for in real frames the user is watching, so
// it is also as many as is polite. Total cost: 4 frames x 4 tiers, worst case.
export const AUTO_TIER_SAMPLE_FRAMES = 3;
// When even the floor tier cannot hold this, say so out loud. There is
// nothing left to fall back to — the floor is the floor — so the honest
// move is to settle there and tell the user what they are looking at rather
// than let them conclude the app is broken.
export const AUTO_TIER_FLOOR_WARN_MS = 50.0;

// --- the probe's clock, calibrated before it is believed ------------------
//
// The probe's preferred clock is queue-idle round-trips: drain, submit the
// frame, drain again, and the wall time between the drains is the frame's
// GPU work. That is only true when waiting on the queue costs nothing —
// and on Firefox it does not: its WebGPU resolves onSubmittedWorkDone on an
// internal poll cadence rather than on a fence, so the wait reports ~100 ms
// on an RTX 5080 whose actual frame is one or two — which read as "this GPU
// renders 100 ms a frame even at Minimal" while High flew at vsync
// (observed on Thomas's box, 2026-08-11). A clock cannot be trusted on
// reputation; it has to be measured. Before the first probe frame, the
// round-trip is timed on an EMPTY queue: whatever it costs there is the
// clock's own overhead, not the GPU's work. The minimum of a few rounds is
// used because the first may still drain real leftover work — the cleanest
// round is the honest floor.
export const AUTO_TIER_CLOCK_CALIBRATION_ROUNDS = 3;
// An empty-queue round-trip above this means the clock cannot resolve a
// frame budget and the probe switches to its cadence clock (below). Chrome
// measures well under 1 ms here; Firefox's poll cadence is far above it.
export const AUTO_TIER_CLOCK_OVERHEAD_MAX_MS = 4.0;
// The cadence clock: no queue waits at all — just the rAF-to-rAF wall time
// of marched frames. It saturates at the vsync interval, so it cannot see
// headroom below one vsync and cannot PREDICT the next tier the way the
// queue clock's ratios do. What it can do is falsify: a tier whose frames
// arrive on the vsync beat is affordable, one whose frames do not is not.
// So in cadence mode the probe climbs while the current tier holds the
// beat, and steps BACK one rung when it breaks. The one over-step is
// bounded by the adjacent-tier cost ratios (2-4.5x of a vsync-priced
// frame, tens of milliseconds) — brief, and nowhere near watchdog
// territory, which keeps the never-render-the-catastrophic-frame property
// even though this clock cannot predict.
//
// The threshold carries slack over one 60 Hz vsync for compositor jitter.
// On a faster display frames arrive under budget and simply pass sooner.
export const AUTO_TIER_CADENCE_HOLD_MS = AUTO_TIER_TARGET_MS * 1.25;
// The cadence clock reads each frame's time one frame LATE (this frame's
// rAF delta describes the previous frame), so a tier switch needs two
// frames of warm-up before a sample describes the new tier in steady state:
// one for the switch frame itself, one for the first new-tier frame and its
// render-target reallocation.
export const AUTO_TIER_WARMUP_FRAMES_CADENCE = 2;
// --- the in-flight tier governor ------------------------------------------
//
// The startup probe has two blind spots it cannot measure its way out of.
// It runs in the first half-second after load, inside the window where the
// machine is still paying for the load itself — and on Thomas's M1 that
// window is "seconds to tens of seconds", so a steady-state Minimal of
// 1.5-4.2 ms (measured 2026-08-19, TWPICE at 1440p) that clears the
// escalation gate with only ~11% headroom on its worst view reliably fails
// it under startup contention, and the verdict sticks at Minimal on a
// machine that flies Low at 2x margin. And the right tier is not even a
// per-field constant: DYCOMS's thick backlit view costs 18x its overview
// (405 vs 22 ms at Low, same day's numbers), so no one-shot verdict from
// one camera can be right everywhere.
//
// So the probe's verdict is a floor, not a sentence. While the camera is
// actually flying, the governor watches the rAF deltas of marched frames —
// the cadence clock, with the cadence clock's semantics: it saturates at
// vsync, so it cannot predict the tier above, only falsify the tier it is
// on. The rule is the probe's cadence rule transplanted to steady state:
// when the current tier has held the beat for a full window, climb one
// rung; the climbed-to tier is then on trial, and breaking the beat in its
// first window steps back to the rung that held and puts the refused rung
// on a cooldown. The one over-step is bounded by adjacent-tier cost ratios,
// exactly the bound the cadence probe already accepts — so the
// never-render-the-catastrophic-frame property survives: the governor only
// ever steps to a tier adjacent to one that just measured affordable, and
// Max is unreachable (it is not in QUALITY_TIERS_CHEAPEST_FIRST).
//
// Climb-only, apart from the trial step-back. A tier that held its trial
// and later hits a heavy region is left alone: automatic downshift invites
// tier-bouncing at regime boundaries, and an over-tiered stretch is
// sluggish, not broken. The user's hand always wins — a tier picked in the
// Quality panel disables auto-tiering entirely (setQualityTier), and a
// hand-set render scale pauses the governor rather than being stomped by a
// climb.
//
// A window is both a frame count and a wall-time floor: enough frames for
// a median to mean something, enough seconds that one easy glide past a
// gap does not climb onto a tier the next cumulus refutes. The window is
// CONSECUTIVE marched, camera-moving frames — a hold, a park, or a pause
// resets it, because the hold ladder's frames measure the wrong
// configuration (and its top rung can cost 400 ms, which is not evidence
// about flight).
export const GOVERNOR_MIN_FRAMES = 60;
export const GOVERNOR_MIN_SECONDS = 3.0;
// A rung that failed its trial is not retried before this. Long enough to
// outlive the regime that refused it (a thick view flown through), short
// enough that a machine that was merely still warming up gets another look.
export const GOVERNOR_RETRY_COOLDOWN_MS = 60000.0;

// Ceiling on the timestep the camera, bird and recorder integrate over, so a
// stall (a tab in the background, a shader compile) does not fling the camera
// across the domain. It is NOT the clock the FPS readout uses.
export const MAX_SIM_TIMESTEP = 0.1;

// --- camera and flight ---------------------------------------------------

export const DEFAULT_CAMERA = {
  position: [0.0, 0.0, -0.999],   // relative coords; see cameraWorldOrigin
  azimuth: 0.0,
  elevation: 35.0,
  fov: 100.0,
};
export const FOV_LIMITS = [30.0, 110.0];
export const MOUSE_SENS = 0.12;             // degrees per pixel
export const DEFAULT_SPEED = 60.0;          // m/s
export const SPEED_LIMITS = [0.5, 5000.0];
export const SPEED_WHEEL_FACTOR = 1.25;     // per notch
export const ELEVATION_LIMITS = [-89.0, 89.0];
// How close to the sea the camera — and so the bird, which flies off it and
// has no floor of its own — may get. Halved from 25 m on 2026-08-14. It is
// still 1.25 ocean FIF outer scales and better than ten times the surface's
// own peak-to-trough, so nothing here is about clipping through a wave: it is
// how low you are allowed to fly, and lower is better flying.
export const OCEAN_FLOOR_MARGIN_M = 12.5;

// --- sun ------------------------------------------------------------------

export const SUN_PRESETS = [
  { name: "midday", azimuth: 180.0, elevation: 75.0 },
  { name: "golden hour", azimuth: 255.0, elevation: 12.0 },
  { name: "sunset", azimuth: 270.0, elevation: 0.5 },
];
// A periodic light march exits only through the domain top, so a sun at or
// below the horizon has nowhere to go — the engine raises rather than
// rendering something wrong. The slider stops here instead.
export const MIN_SUN_ELEVATION_DEG = 0.5;

export const COMPASS_EDGES = [
  [22.5, "N"], [67.5, "NE"], [112.5, "E"], [157.5, "SE"],
  [202.5, "S"], [247.5, "SW"], [292.5, "W"], [337.5, "NW"],
];

// --- capture --------------------------------------------------------------

export const STILL_ACCUMULATE_FRAMES = 64;
// The still capture's samples-per-pixel slider. The default stays
// STILL_ACCUMULATE_FRAMES; the top is where a 4K still starts costing real
// wall-clock without a visible return on converged scenes.
export const STILL_SAMPLES_LIMITS = [8, 256];
// Accumulated frames at the top of the ladder, with the scene unchanged,
// after which the live loop stops marching.
//
// This used to BE STILL_ACCUMULATE_FRAMES — 64, the same constant a capture
// uses, on the argument that a held view and a still are the same picture and
// must settle at the same point. The argument was right about the picture and
// wrong about the cost. A capture is a thing you asked for and wait on; a held
// view is what happens every time you stop for a moment, and on a slow machine
// 64 marches at the top rung is several seconds of saturated GPU that you then
// have to fly out of — every one of those frames is already submitted when you
// touch a key, and the tail of one is the lag (see docs/soar-bugs.md and the
// note in Renderer.holdTick). The returns are also long gone by then: the
// noise a march removes falls as 1/sqrt(n), so 8 -> 32 halves it and 32 -> 64
// takes another 30%, on top of a picture that is already jitter-averaged.
//
// So parking now stops where the returns stop, sooner on the tiers that can
// least afford not to. A capture still accumulates all 64 — it is explicit,
// and nobody is waiting to fly out of it (Thomas, 2026-08-14).
export const PARKED_ACCUM_FRAMES_BY_TIER = {
  max: 32, high: 32, medium: 24, low: 16, minimal: 8,
};

// Which picture a held view settles to.
//
//   "live"  — the ladder never climbs: what you see when you stop is what you
//             saw while moving, just cleaner (it still accumulates to the cap
//             above, and still parks and sleeps once it gets there).
//   "still" — the hold ladder, climbing toward High sampling whatever tier
//             flew you here. What flying does, and the default.
//
// Live exists because of what the Quality panel is FOR. Every control in it
// changes how the view is drawn while you move, and every one of them was
// invisible there: the panel is open, the camera is not moving, so the ladder
// has already converged the view to High's sampling and the tier buttons do
// nothing you can see. Live holds the flight picture still so the controls
// have something to act on, and it lasts exactly as long as the panel is open
// — close it and stills come back (Thomas, 2026-08-14).
export const HOLD_MODES = ["live", "still"];
export const DEFAULT_HOLD_MODE = "still";
// What the Quality panel opens showing, remembered for the session.
export const DEFAULT_QUALITY_PREVIEW = "live";
export const CAPTURE_SIZE_PRESETS = [
  ["1280 x 720", [1280, 720]],
  ["2K", [1920, 1080]],
  ["4K", [3840, 2160]],
];
export const CAPTURE_SIZE_LIMITS = [64, 7680];
export const DEFAULT_VIDEO_FPS = 60.0;
export const VIDEO_FPS_LIMITS = [12.0, 120.0];
export const DEFAULT_VIDEO_ACCUMULATE = 24;
export const VIDEO_ACCUMULATE_LIMITS = [1, 64];

export const BEHOLD_QUALITY_ROWS = [
  ["Min", "min", "fast preview"],
  ["Low", "low", "draft"],
  ["Medium", "medium", "balanced"],
  ["High", "high", "~1 h"],
  ["Max", "max", "overnight"],
];
export const DEFAULT_BEHOLD_QUALITY = "high";

// --- extinction (optical_depth.py) ---------------------------------------
//
// sigma = (1.5 / (rho * r_eff)) * q * rho_air(z), with q in g/kg and the
// result in m^-1. The 1.5 is 3*Q_ext/4 at the geometric-optics limit Q_ext=2.
// The atmosphere is a fixed isothermal profile — not the field's own.

export const AIR_R = 287.05;             // J/kg/K
export const AIR_T = 280.0;              // K
export const AIR_SCALE_HEIGHT_M = 7000.0;
export const AIR_P0 = 101300.0;          // Pa
export const RE_LIQUID_UM = 10.0;
export const RE_ICE_UM = 30.0;
export const RHO_WATER_G_M3 = 1e6;
export const RHO_ICE_G_M3 = 917e3;
// An ice field whose global maximum is below this is treated as no ice at all.
export const ICE_NEGLIGIBLE_G_KG = 1e-6;

export const SIGMA_LIQUID_PREFACTOR =
  1.5 / (RHO_WATER_G_M3 * RE_LIQUID_UM * 1e-6);   // 0.15 m^2/g
export const SIGMA_ICE_PREFACTOR =
  1.5 / (RHO_ICE_G_M3 * RE_ICE_UM * 1e-6);        // 0.054525... m^2/g

// --- column optical depth (glimpse) ---------------------------------------
//
// The minimap is a glimpse image, and glimpse does NOT integrate the same
// sigma the raymarch uses. Liquid now agrees — both are geometric optics with
// Q_ext = 2 — but ice does not: Ebert & Curry gives 0.0845 m^2/g at
// r_e = 30 um against sigma's geometric-optics 0.0545. So the map cannot be
// derived from the extinction volume — it is accumulated separately, from
// lwc and iwc, on the way past.

// tau = 1.5 * LWP / r_e_um: geometric optics, Q_ext = 2 exactly.
// Petty (2006), "A First Course in Atmospheric Radiation", 2nd ed., Eq. 7.86.
export const TAU_PER_LWP_UM = 1.5;       // m^2 um / g

// tau = IWP * (a + b / r_e_um): Ebert & Curry (1992), JGR 97(D4), 3831-3836,
// doi:10.1029/91JD02472, Table 2 band 1 (0.25-0.7 um), used exactly.
export const EC92_BAND1_A = 3.448e-3;    // m^2 / g
export const EC92_BAND1_B = 2.431;       // m^2 um / g

// Conservative-scattering two-stream reflectance, A = tau / (tau + 2/(1-g)).
// Exactly Eq. 5.51 of Bohren & Clothiaux (2006), "Fundamentals of
// Atmospheric Radiation".
// Unlike beam opacity it keeps contrast from cirrus (tau ~ 1) all the way to
// deep cores (tau ~ 100), which is the whole point of the map.
export const TWO_STREAM_G = 0.85;
export const TWO_STREAM_DENOM = 2.0 / (1.0 - TWO_STREAM_G);   // 13.33...

// --- minimap layout (soar/hud.py) -----------------------------------------

export const MAP_HEIGHT_FRAC = 0.22;
export const MAP_MAX_WIDTH_FRAC = 0.34;
export const MAP_MARGIN_FRAC = 0.025;
export const MAP_OPACITY = 0.74;
// The same ramp basic_render uses, so a glimpse PNG and the corner of the
// flight view read as the same picture — and the same one turbulon-analysis
// plots albedo fields with, so a paper figure does too. Deep ocean blue at
// clear sky through to white at albedo 1, monotone in lightness: the stops
// are what make it read as cloud over water instead of as a two-colour map,
// which is why this is a ramp and not the single blue it used to be
// (0x3a4aa6, lerped straight to white).
//
// Kept byte-for-byte with cloudyview/basic_render.py's `cloud_colors` by
// tests/test_map_ramp_parity.py, because a comment saying "the same ramp" is
// exactly what stops being true.
export const MAP_CLOUD_RAMP = [
  [0.00, [0x06 / 255, 0x1a / 255, 0x3c / 255]],
  [0.18, [0x12 / 255, 0x3f / 255, 0x74 / 255]],
  [0.38, [0x2f / 255, 0x74 / 255, 0xac / 255]],
  [0.60, [0x79 / 255, 0xb0 / 255, 0xd6 / 255]],
  [0.80, [0xc6 / 255, 0xdd / 255, 0xee / 255]],
  [1.00, [1.0, 1.0, 1.0]],
];
// Clear sky, i.e. the bottom of that ramp. The HUD draws its halos and rims
// against it.
export const MAP_SKY_BLUE = MAP_CLOUD_RAMP[0][1];

// The overlay drawn ON the map — camera dot, field-of-view rays, nest
// footprint. One warm colour against a blue-to-white field, and the same
// value as the chrome's --hot and the landing page's --amber, so the marker
// belongs to the app rather than to the picture. Replaces pure red, which
// read as an error state and clashed with the cloud ramp at both ends.
export const MAP_ACCENT = [0xe8 / 255, 0x83 / 255, 0x4a / 255];
