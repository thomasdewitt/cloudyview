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
export const HAZE_MAX = 2.0;
export const AERIAL_BETA_FLOOR_PER_KM = 0.015;

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

export const DEFAULT_MOTION_BLEND_ALPHA = 0.58;
export const DEFAULT_MOTION_BLEND_REFERENCE_FPS = 60.0;
export const DEFAULT_MOTION_JITTER_SCALE = 0.65;
export const DEFAULT_MOTION_RESET_ANGLE_DEGREES = 8.0;
export const DEFAULT_MOTION_RESET_TRANSLATION_FRACTION = 0.05;

export const UNIFORM_ROWS = 23;
export const UNIFORM_NBYTES = 368;

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
  high:   { name: "high",   label: "High",   renderScale: 1.0,
            stepFactor: 2.0, lightStepFactor: 2.0,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  medium: { name: "medium", label: "Medium", renderScale: 0.75,
            stepFactor: 2.5, lightStepFactor: 4.0,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  low:    { name: "low",    label: "Low",    renderScale: 0.60,
            stepFactor: 3.0, lightStepFactor: 8.0,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
  // Potato flies at an eighth, not a quarter. A quarter was chosen against a
  // 5080; a GPU sixty times slower renders potato's own frame in ~30 ms, and
  // an eighth is four times fewer pixels than a quarter, which brings that
  // back inside a vsync. Nothing is lost by it: the hold ladder climbs
  // 0.125 -> 0.25 -> 0.5 -> 1.0 the moment the view is held, so the rough
  // flight scale is only ever what you see while actually moving.
  potato: { name: "potato", label: "Potato — smooth stills, rough flight",
            renderScale: 0.125, stepFactor: 4.0, lightStepFactor: 12.0,
            maxLightSteps: DEFAULT_MAX_LIGHT_STEPS },
};
export const QUALITY_TIER_NAMES = ["high", "medium", "low", "potato"];
// Cheapest first. This is the order the startup probe walks, and the order
// matters for more than tidiness — see AUTO_TIER_COST_RATIO_TO_NEXT.
export const QUALITY_TIERS_CHEAPEST_FIRST = ["potato", "low", "medium", "high"];
// The tier a Renderer is born with. In practice the startup probe replaces it
// before the first frame is presented (viewer.loadField sets the probe's
// starting tier before Renderer.init compiles anything); this is the answer
// when there is no measurement, which is what the offline capture paths want.
export const DEFAULT_QUALITY_TIER = "high";
// The floor was 0.25 until potato's flight scale went below it. Only two
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
// The TOP rung is "high" for every tier without exception, and that is the
// point of the whole mechanism: the tier governs FLIGHT, and a parked picture
// is tier-independent. Whatever you flew here on, stop moving and the view
// converges to High's own sampling at full resolution — the same still. This
// generalizes what the old potato-only parked swap did for one tier.
//
// Intermediate rungs keep the flight tier's sampling because they are meant to
// be cheap stepping stones: their job is to put something settled on screen
// while the expensive rung is still being paid for, not to be the answer.
// (Before the light-step rework this rule also avoided a mid-hold shader
// compile. That reason is gone — every tier now specializes identically — but
// the rule earns its keep on cost alone.)
export const QUALITY_HOLD_LADDERS = {
  high:   [],                                    // flies at 1.0; hold just accumulates
  medium: [{ scale: 1.00, sampling: "high" }],
  low:    [{ scale: 1.00, sampling: "high" }],
  potato: [{ scale: 0.25, sampling: "potato" },
           { scale: 0.50, sampling: "potato" },
           { scale: 1.00, sampling: "high" }],
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
// potato because potato flies at 10 ms would reach the top rung at something
// like 10 ms x 64 pixels — half a second in one fragment pass, and on Metal a
// long enough pass is not slow, it is a dead device. So the ladder predicts
// the next rung's cost from the rung it is standing on (see
// Renderer._buildHoldLadder) and refuses to climb past this. A capped ladder
// converges where it stands, which is the whole point: the view still
// settles, it just settles at half resolution on hardware that cannot afford
// full. 400 ms is comfortably inside any GPU watchdog and is a frame you only
// ever wait for while parked.
export const HOLD_MAX_FRAME_MS = 400.0;

// The frame count that ends a hold — CONVERGED_ACCUM_FRAMES — is defined with
// STILL_ACCUMULATE_FRAMES under "capture" below, because it is the same
// number for the same reason and the two must not drift.

// --- the startup auto-tier probe -----------------------------------------
//
// What the probe is aiming at: one frame inside a 60 Hz vsync.
export const AUTO_TIER_TARGET_MS = 1000.0 / 60.0;
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
// defined above. Per-frame cost is the SLOPE of total time against frame
// count, which cancels setup, upload and readback:
//
//     high     4.75 / 4.59 ms
//     medium   2.55 / 2.50     -> high:   1.87x / 1.83x
//     low      1.59 / 1.45     -> medium: 1.61x / 1.73x
//     potato   0.399 / 0.392   -> low:    3.97x / 3.70x
//
// The constants below round those up for headroom. Note how unequal the rungs
// are: potato to low is a step four times bigger than low to medium. A single
// flat threshold would be either too timid at the top or reckless at the
// bottom, which is why the gate is a per-tier ratio rather than one number.
//
// One honest caveat about potato. It has 23x fewer pixels than low, yet costs
// only ~4x less, because at 320x180 the frame stops being about the march at
// all — command encoding, uniform upload and the blit dominate. That floor is
// mostly CPU- and driver-side, so it does NOT shrink on a slower GPU, and the
// true potato->low ratio there is larger than 4.5. The gate is therefore
// optimistic on exactly the weak hardware it protects.
//
// That is a known and bounded weakness, and it is bounded by the ordering
// rule rather than by this number: being one rung too optimistic costs a
// settle point that is sluggish, not a machine that freezes, because the tier
// above is still never rendered without its own completed measurement. The
// catastrophic case — submitting a High frame on an M1 having measured
// nothing — is prevented by the walk being strictly one rung at a time, not
// by the accuracy of these ratios.
export const AUTO_TIER_COST_RATIO_TO_NEXT = {
  potato: 4.5,
  low: 2.0,
  medium: 2.0,
};
export const AUTO_TIER_MARGIN = 0.75;
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
// renders 100 ms a frame even at Potato" while High flew at vsync
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
export const OCEAN_FLOOR_MARGIN_M = 25.0;   // 2.5 ocean FIF outer scales

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
// Accumulated frames at the hold ladder's top rung, with the scene unchanged,
// after which the live loop stops marching: past this, another march costs a
// full volume traversal to change nothing visible. The same number a still
// capture accumulates, and deliberately the same constant — a held view and a
// still are the same picture, so they must settle at the same point.
export const CONVERGED_ACCUM_FRAMES = STILL_ACCUMULATE_FRAMES;
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
// The same sky-blue -> white ramp basic_render uses, so a glimpse PNG and
// the corner of the flight view read as the same picture.
export const MAP_SKY_BLUE = [0x3a / 255, 0x4a / 255, 0xa6 / 255];
