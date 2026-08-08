// Every number the look depends on, ported one-for-one from Python.
//
// Sources: cloudyview/look.py (shared with witness), cloudyview/soar/engine.py,
// cloudyview/soar/menu.py, cloudyview/soar/app.py, cloudyview/config.py.
// A value that drifts from its Python twin silently changes the picture, so
// tests/test_web_uniform_parity.py runs this file under node and diffs the
// packed uniform block against engine.write_uniforms. Change one, change both.

"use strict";

// --- look.py: the witness realism package -------------------------------

export const SUN_COLOR = [22.0, 21.0, 17.0];
export const LEGACY_AMBIENT = [0.19, 0.225, 0.30];
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

// --- engine.py defaults -------------------------------------------------

export const DEFAULT_SUN_AZIMUTH = 20.0;
export const DEFAULT_SUN_ELEVATION = 55.0;
export const DEFAULT_EXPOSURE = 4.0;
export const DEFAULT_G_HG = 0.76;
export const DEFAULT_AMBIENT_STRENGTH = 0.12;
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

// Tone-map gamma. 1.4 is witness's reference; the desktop window spent years
// presenting through an sRGB swapchain and encoding a second time, so what
// soar actually looked like in flight was ~3.08. The default sits 75% of the
// way from the reference toward that, and this is now the only encode.
export const TONE_MAP_GAMMA_WITNESS = 1.4;
export const TONE_MAP_GAMMA_AS_FLOWN = 3.08;
export const DEFAULT_TONE_MAP_GAMMA = 2.66;
export const TONE_MAP_GAMMA_LIMITS = [1.0, 4.0];

export const DEFAULT_MOTION_BLEND_ALPHA = 0.58;
export const DEFAULT_MOTION_BLEND_REFERENCE_FPS = 60.0;
export const DEFAULT_MOTION_JITTER_SCALE = 0.65;
export const DEFAULT_MOTION_RESET_ANGLE_DEGREES = 8.0;
export const DEFAULT_MOTION_RESET_TRANSLATION_FRACTION = 0.05;

export const UNIFORM_ROWS = 23;
export const UNIFORM_NBYTES = 368;

export const AUTO_FP16_MIN_VOXELS = 256 * 1024 * 1024;

// --- quality ------------------------------------------------------------

export const QUALITY_PRESETS = {
  high:   { name: "high",   label: "High",   renderScale: 1.0,  stepFactor: 2.0, maxLightSteps: 512 },
  medium: { name: "medium", label: "Medium", renderScale: 0.75, stepFactor: 2.5, maxLightSteps: 384 },
  low:    { name: "low",    label: "Low",    renderScale: 0.60, stepFactor: 3.0, maxLightSteps: 256 },
  potato: { name: "potato", label: "Potato — smooth stills, rough flight",
            renderScale: 0.25, stepFactor: 4.0, maxLightSteps: 128 },
};
export const QUALITY_TIER_NAMES = ["high", "medium", "low", "potato"];
export const DEFAULT_QUALITY_TIER = "high";
export const MIN_RENDER_SCALE = 0.25;
export const MAX_RENDER_SCALE = 1.0;
export const AUTO_TIER_TARGET_MS = 1000.0 / 60.0;

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
