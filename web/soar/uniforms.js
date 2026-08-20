// The uniform block: 24 rows of 4 floats, 384 bytes, rebuilt every frame.
//
// A direct port of InteractiveRenderer.write_uniforms. Row order and meaning
// are fixed by raymarch.wgsl, which the browser shares verbatim with the
// desktop — so this file is the whole of the look that lives outside the
// shader. tests/test_web_uniform_parity.py diffs it against Python.

"use strict";

import * as K from "./constants.js";
import {
  directionFromAzimuthElevation, spectralLightingColors,
  effectiveLightTransferSplit, aerialBetaPerKm, oceanHazeExtinctionPerKm,
  HAZE_MIN,
} from "./spectral.js";
import { cameraBasis } from "./camera.js";

const DEG = Math.PI / 180.0;

/** Scaled march resolution. Round-half-up, matching numpy's floor(x + 0.5). */
export function renderTargetSize([w, h], renderScale) {
  const scale = Number(renderScale);
  if (!Number.isFinite(scale)) {
    throw new Error(`render_scale must be finite; got ${renderScale}.`);
  }
  if (!(scale >= K.MIN_RENDER_SCALE && scale <= K.MAX_RENDER_SCALE)) {
    throw new Error(
      `render_scale must be in [${K.MIN_RENDER_SCALE}, ` +
      `${K.MAX_RENDER_SCALE}]; got ${scale}.`);
  }
  if (w < 1 || h < 1) throw new Error(`size must be positive; got ${w}x${h}.`);
  return [
    Math.max(1, Math.floor(w * scale + 0.5)),
    Math.max(1, Math.floor(h * scale + 0.5)),
  ];
}

function unitInterval(name, value) {
  const v = Number(value);
  if (!Number.isFinite(v)) throw new Error(`${name} must be finite; got ${value}.`);
  if (!(v >= 0.0 && v <= 1.0)) {
    throw new Error(`${name} must be in [0, 1]; got ${value}.`);
  }
  return v;
}

/**
 * Pack one frame. `state` carries the scene (field bounds, step sizes, ocean),
 * `view` the camera and per-frame choices. Returns the Float32Array so the
 * caller can upload and key off the same bytes numpy would have produced.
 */
export function packUniforms(state, view) {
  const {
    bmin, bmax, dtView, dtLight, periodic,
    oceanZ, oceanReflectance, oceanFifDx, oceanTileExtent, oceanEnabled,
    oceanMaxLod,
    nested = false, nestBmin, nestBmax, dtViewNest, dtLightNest,
    // The night city (raymarch.wgsl CITY): the ocean-named fields then
    // describe the city tile, row 4 is the moon, and cityOffsetM is where
    // the tile sits in world space (row 8.yz).
    city = false, cityOffsetM = [0.0, 0.0],
  } = state;

  const {
    camera, outputSize, renderSize,
    jitter = true,
    sunAzimuth = K.DEFAULT_SUN_AZIMUTH,
    sunElevation = K.DEFAULT_SUN_ELEVATION,
    exposure = K.DEFAULT_EXPOSURE,
    gHg = K.DEFAULT_G_HG,
    ambientStrength = K.DEFAULT_AMBIENT_STRENGTH,
    gradientShadingStrength = K.DEFAULT_GRADIENT_SHADING_STRENGTH,
    gradientCoarseWeight = K.DEFAULT_GRADIENT_COARSE_WEIGHT,
    gradientCoarseRadiusM = K.DEFAULT_GRADIENT_COARSE_RADIUS_M,
    deepShadowMsSuppression = K.DEFAULT_DEEP_SHADOW_MS_SUPPRESSION,
    ambientOcclusionStrength = K.DEFAULT_AMBIENT_OCCLUSION_STRENGTH,
    ambientOcclusionFloor = K.DEFAULT_AMBIENT_OCCLUSION_FLOOR,
    bounceDepthAttenuation = K.DEFAULT_BOUNCE_DEPTH_ATTENUATION,
    spectralLightingStrength = K.SPECTRAL_LIGHTING_STRENGTH,
    lowSunSkyFieldStrength = K.LOW_SUN_SKY_FIELD_STRENGTH,
    lightTransferSplitStrength = K.LIGHT_TRANSFER_SPLIT_STRENGTH,
    aerialPerspectiveStrength = K.AERIAL_PERSPECTIVE_STRENGTH,
    // One number for the whole aerosol story; no per-term override exists.
    haze = K.DEFAULT_HAZE,
    toneMapWhitePoint = K.DEFAULT_TONE_MAP_WHITE_POINT,
    contrast = K.DEFAULT_CONTRAST,
    oceanRealism = K.OCEAN_REALISM,
    oceanMipBias = K.OCEAN_MIP_BIAS,
    oceanGlintStrength = K.OCEAN_GLINT_STRENGTH,
    oceanGlintRoughness = K.OCEAN_GLINT_ROUGHNESS,
    oceanSlopeDrawFraction = K.OCEAN_SLOPE_DRAW_FRACTION,
    oceanSkyShadowFloor = K.OCEAN_SKY_SHADOW_FLOOR,
    coneStencilThetaDeg = K.CONE_STENCIL_THETA_DEG,
    hazeHeightDependent = K.DEFAULT_HAZE_HEIGHT_DEPENDENT,
    // The app's angles at DEFAULT_LOD_STRENGTH, matching soar_host.ViewState.
    // The viewer always passes these explicitly (its slider scales them); the
    // defaults are what a caller who does not care gets, and they have to be
    // the same number on both sides — see test_uniform_parity.
    lightMarchLodDegrees = K.APP_LIGHT_MARCH_LOD_DEGREES * K.DEFAULT_LOD_STRENGTH,
    viewStepLodDegrees = K.APP_VIEW_STEP_LOD_DEGREES * K.DEFAULT_LOD_STRENGTH,
    toneMapGamma = K.DEFAULT_TONE_MAP_GAMMA,
    frameIndex = 0,
    subpixel = false,
    jitterScale = 1.0,
    // Read tau_sun from the baked sun-tau cache (row 23). Only meaningful
    // when the renderer has a finished bake bound at binding 7; the packer
    // just writes the flag.
    lightCache = false,
    // The vertical sky-visibility march (row 23.z inverts it): off means
    // every consumer of t_sky sees a fully open sky.
    skyProbe = true,
  } = view;

  const [outputW, outputH] = outputSize;
  const [w, h] = renderSize;

  unitInterval("jitter_scale", jitterScale);
  unitInterval("spectral_lighting_strength", spectralLightingStrength);
  unitInterval("low_sun_sky_field_strength", lowSunSkyFieldStrength);
  unitInterval("ocean_realism", oceanRealism);
  unitInterval("ocean_sky_shadow_floor", oceanSkyShadowFloor);
  if (!(Number(haze) >= HAZE_MIN && Number(haze) <= K.HAZE_MAX)) {
    throw new Error(`haze must be in [${HAZE_MIN}, ${K.HAZE_MAX}]; got ${haze}.`);
  }
  {
    const [lo, hi] = K.TONE_MAP_WHITE_POINT_LIMITS;
    if (!(Number(toneMapWhitePoint) >= lo && Number(toneMapWhitePoint) <= hi)) {
      throw new Error(
        `tone_map_white_point must be in [${lo}, ${hi}]; got ${toneMapWhitePoint}.`);
    }
  }
  {
    const [lo, hi] = K.CONTRAST_LIMITS;
    if (!(Number(contrast) >= lo && Number(contrast) <= hi)) {
      throw new Error(`contrast must be in [${lo}, ${hi}]; got ${contrast}.`);
    }
  }
  if (!(aerialPerspectiveStrength >= 0.0)) {
    throw new Error(
      `aerial_perspective_strength must be >= 0; got ${aerialPerspectiveStrength}.`);
  }
  if (!(coneStencilThetaDeg >= 0.0 && coneStencilThetaDeg < 90.0)) {
    throw new Error(
      `cone_stencil_theta_deg must be in [0, 90); got ${coneStencilThetaDeg}.`);
  }
  for (const [name, value] of [
    ["light_march_lod_degrees", lightMarchLodDegrees],
    ["view_step_lod_degrees", viewStepLodDegrees],
  ]) {
    if (!(value >= 0.0 && value < 45.0)) {
      throw new Error(`${name} must be in [0, 45) degrees; got ${value}.`);
    }
  }
  const [gLo, gHi] = K.TONE_MAP_GAMMA_LIMITS;
  if (!(toneMapGamma >= gLo && toneMapGamma <= gHi)) {
    throw new Error(
      `tone_map_gamma must be in [${gLo}, ${gHi}]; got ${toneMapGamma}.`);
  }
  // A periodic light march exits only through the domain top. With the sun at
  // or below the horizon there is no exit, so this is an error rather than a
  // clamp — the picture would be wrong in a way that looks plausible.
  if (periodic && sunElevation <= 0.0) {
    throw new Error(
      "Periodic domains require the sun above the horizon (the light march " +
      `exits only through the domain top); got sun_elevation=${sunElevation}. ` +
      "Turn periodic off for a below-horizon sun.");
  }

  const origin = camera.position;
  const [forward, right, up] = cameraBasis(camera.azimuth, camera.elevation);
  const sun = directionFromAzimuthElevation(sunAzimuth, sunElevation);
  const tanHalfFov = Math.tan(camera.fov * DEG * 0.5);

  // Under CITY the daytime spectral pipeline is bypassed, not scaled: the
  // moon's rows are fixed night values, the low-sun sky field and the
  // light-transfer split are daytime-sun machinery with no night meaning.
  // Mirrors soar_host.pack_uniforms exactly (test_uniform_parity).
  let spec, lowSunSky, lightTransferEff;
  if (city) {
    spec = {
      cloudSun: K.NIGHT_MOON_CLOUD_COLOR, ambient: K.NIGHT_AMBIENT_TINT,
      horizon: K.NIGHT_SKY_HORIZON, bloom: K.NIGHT_MOON_BLOOM,
      disc: K.NIGHT_MOON_DISC,
    };
    lowSunSky = 0.0;
    lightTransferEff = 0.0;
  } else {
    spec = spectralLightingColors(sun, K.SUN_COLOR, spectralLightingStrength);
    lowSunSky = lowSunSkyFieldStrength;
    lightTransferEff = effectiveLightTransferSplit(
      lightTransferSplitStrength, sunElevation);
  }

  const u = new Float32Array(K.UNIFORM_ROWS * 4);
  const row = (i, a, b, c, d) => {
    u[i * 4] = a; u[i * 4 + 1] = b; u[i * 4 + 2] = c; u[i * 4 + 3] = d;
  };

  row(0, origin[0], origin[1], origin[2], tanHalfFov);
  row(1, forward[0], forward[1], forward[2], outputW / outputH);
  row(2, right[0], right[1], right[2], exposure);
  row(3, up[0], up[1], up[2], jitter ? 1.0 : 0.0);
  row(4, sun[0], sun[1], sun[2], frameIndex);
  row(5, bmin[0], bmin[1], bmin[2], dtView);
  row(6, bmax[0], bmax[1], bmax[2], dtLight);
  row(7, w, h, gHg, ambientStrength);
  if (city) {
    row(8, oceanZ, cityOffsetM[0], cityOffsetM[1], 0.0);
  } else {
    row(8, oceanZ, oceanReflectance[0], oceanReflectance[1], oceanReflectance[2]);
  }
  row(9, oceanFifDx, oceanTileExtent, oceanEnabled ? 1.0 : 0.0, oceanMaxLod);
  // x/y are sampling flags, excluded from scene identity; z is haze, which
  // is scene state and is not. See sceneKey.
  row(10, subpixel ? 1.0 : 0.0, jitterScale, haze, toneMapWhitePoint);
  row(11, gradientShadingStrength, deepShadowMsSuppression,
       ambientOcclusionStrength, bounceDepthAttenuation);
  row(12, gradientCoarseWeight, gradientCoarseRadiusM, ambientOcclusionFloor,
       Math.tan(coneStencilThetaDeg * DEG));
  row(13, spec.cloudSun[0], spec.cloudSun[1], spec.cloudSun[2],
       lowSunSky);
  row(14, spec.ambient[0], spec.ambient[1], spec.ambient[2], lightTransferEff);
  row(15, spec.horizon[0], spec.horizon[1], spec.horizon[2],
       aerialPerspectiveStrength);
  row(16, spec.bloom[0], spec.bloom[1], spec.bloom[2],
       aerialBetaPerKm(haze) * 1e-3);          // w: beta0 in m^-1
  // A scale height of 0 is the shader's signal for "no height profile";
  // see raymarch.wgsl's sky_disc.w.
  row(17, spec.disc[0], spec.disc[1], spec.disc[2],
      hazeHeightDependent ? K.AERIAL_SCALE_HEIGHT_M : 0.0);
  row(18, oceanRealism, oceanMipBias, oceanGlintStrength, oceanGlintRoughness);
  row(19, oceanSlopeDrawFraction, oceanHazeExtinctionPerKm(haze) * 1e-3,
       oceanSkyShadowFloor, contrast);
  row(20, periodic ? 1.0 : 0.0,
       Math.tan(lightMarchLodDegrees * DEG),
       Math.tan(viewStepLodDegrees * DEG),
       toneMapGamma);
  if (nested) {
    row(21, nestBmin[0], nestBmin[1], nestBmin[2], dtViewNest);
    row(22, nestBmax[0], nestBmax[1], nestBmax[2], dtLightNest);
  }
  // Row 23.y is the bake slice index; the renderer's bake pass writes it
  // into its own copy of the block, never through here.
  row(23, lightCache ? 1.0 : 0.0, 0.0, skyProbe ? 0.0 : 1.0, 0.0);
  return u;
}

/**
 * The scene-identity key that decides whether temporal accumulation continues
 * or restarts. Two components are zeroed out on purpose:
 *
 *   row 4.w  — the frame index only decorrelates jitter seeds. If it counted,
 *              every frame would read as a scene change and nothing would
 *              ever converge.
 *   row 10.x/y — the sampling flags are OUTPUTS of the accumulation decision.
 *              Including them would make the key self-referential. Only those
 *              two: row 10.z is haze, which is scene state like any other and
 *              must restart the average when it moves.
 *
 * maxLightSteps and the output size are folded in because both change the
 * image without appearing in the buffer.
 */
export function sceneKey(u, maxLightSteps, outputW, outputH) {
  const copy = new Float32Array(u);
  copy[4 * 4 + 3] = 0.0;
  copy[10 * 4] = 0.0; copy[10 * 4 + 1] = 0.0;
  const key = new Uint8Array(copy.byteLength + 12);
  key.set(new Uint8Array(copy.buffer), 0);
  new DataView(key.buffer).setUint32(copy.byteLength, maxLightSteps, true);
  new DataView(key.buffer).setUint32(copy.byteLength + 4, outputW, true);
  new DataView(key.buffer).setUint32(copy.byteLength + 8, outputH, true);
  return key;
}

export function keysEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Frame-rate-independent EMA: alpha is defined per frame at referenceFps. */
export function motionAlphaForDt(alpha, referenceFps, deltaSeconds) {
  if (deltaSeconds == null) return alpha;
  if (deltaSeconds <= 0 || alpha === 0.0 || alpha === 1.0) return alpha;
  return 1.0 - Math.pow(1.0 - alpha, deltaSeconds * referenceFps);
}

/**
 * The tier to escalate to after measuring `tier` at `measuredMs`, or null to
 * settle where we are.
 *
 * This replaces an earlier `chooseQualityTier`, which took a frame time for
 * every tier and picked the best one that fit. That shape cannot be used
 * here, and the reason is worth keeping: to fill in its argument you have to
 * have already rendered a frame at every tier, including the one that is too
 * expensive — and rendering one frame that is too expensive is precisely the
 * failure being avoided (a multi-second fragment pass freezes the machine and
 * can lose the device to the GPU watchdog). So the decision is made one rung
 * at a time, upward, and each step is justified by a measurement taken below
 * it. See AUTO_TIER_COST_RATIO_TO_NEXT for the numbers and the argument.
 */
export function escalateQualityTier(tier, measuredMs) {
  const order = K.QUALITY_TIERS_CHEAPEST_FIRST;
  const at = order.indexOf(tier);
  if (at < 0) throw new Error(`unknown quality tier '${tier}'.`);
  if (!(Number.isFinite(measuredMs) && measuredMs >= 0)) {
    throw new Error(
      `measured frame time must be finite and >= 0; got ${measuredMs}.`);
  }
  if (at === order.length - 1) return null;          // nothing dearer to try
  const ratio = K.AUTO_TIER_COST_RATIO_TO_NEXT[tier];
  if (!(ratio > 0)) {
    throw new Error(`no cost ratio recorded for quality tier '${tier}'.`);
  }
  const budget = K.AUTO_TIER_TARGET_MS * K.AUTO_TIER_MARGIN;
  return measuredMs * ratio < budget ? order[at + 1] : null;
}
