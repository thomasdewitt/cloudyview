// The uniform block: rows of four floats, rebuilt every frame.
//
// The Uniforms struct in shader/uniforms.wgsl declares the layout and says
// what each slot is. The Python host packs the same block
// (cloudyview/soar/uniforms.py), and tests/test_uniform_parity.py diffs the
// two byte for byte.

"use strict";

import * as K from "./constants.js";
import {
  directionFromAzimuthElevation, spectralLightingColors,
  effectiveLightTransferSplit, aerialBetaPerKm, oceanHazeExtinctionPerKm,
  HAZE_MIN,
} from "./spectral.js";
import { cameraBasis } from "./camera.js";

const DEG = Math.PI / 180.0;

/**
 * Where a slot sits in the packed block, as an index into its Float32Array,
 * for the code that reads or rewrites one slot of a block already packed.
 */
export const SLOT = Object.freeze({
  CAM_ORIGIN: 0,                  // row 0, xyz
  CAM_FORWARD: 4,                 // row 1, xyz
  FRAME_INDEX: 4 * 4 + 3,
  DT_LIGHT: 6 * 4 + 3,
  SCENE_CLOCK: 16 * 4 + 2,
  LENS_DT: 16 * 4 + 3,
  SUBPIXEL_JITTER: 17 * 4 + 2,
  JITTER_SCALE: 17 * 4 + 3,
  BAKE_SLICE: 20 * 4 + 3,
});

/**
 * The auto-exposure meter's ray grid for an output of `[w, h]`: the output's
 * shape at about AUTO_EXPOSURE_METER_RAYS rays, rounded as renderTargetSize
 * rounds. The shader takes its aspect from the OUTPUT (row 1.w) and its
 * pixel angle from the grid's width, so a grid of any other shape has
 * oblong pixels: the meter's fixed 64x36 did, everywhere but 16:9.
 */
export function meterSize([w, h]) {
  if (!(w >= 1 && h >= 1)) {
    throw new Error(`size must be positive; got ${w}x${h}.`);
  }
  const scale = Math.sqrt(K.AUTO_EXPOSURE_METER_RAYS / (w * h));
  return [Math.max(1, Math.floor(w * scale + 0.5)),
          Math.max(1, Math.floor(h * scale + 0.5))];
}

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
    // The night city: the ocean-named fields then describe the city tile,
    // the sun is the moon, and cityOffsetM is where the tile sits in world
    // space.
    city = false, cityOffsetM = [0.0, 0.0],
    // Explicit override of the surface offset; null means the scene's
    // static offset (cityOffsetM under CITY, the origin otherwise). Mirrors
    // SceneState.surface_offset_m.
    surfaceOffsetM = null,
    // The period the surface offset folds at: null is the tile extent,
    // every surface's rule. Mirrors SceneState.surface_period_m.
    surfacePeriodM = null,
    // The surface's clock (its meta.json clock_period_s): a period, the
    // scene clock packed modulo it so float32 never loses the cadence (the
    // city); null for a surface where nothing moves, packing 0 at any scene
    // time (the sea). Undefined until the caller says, and a
    // running clock with it undefined throws. Mirrors
    // SceneState.clock_period_s (UNSTATED / None / a number).
    clockPeriodS,
  } = state;

  const {
    camera, outputSize, renderSize,
    jitter = true,
    sunAzimuth = K.DEFAULT_SUN_AZIMUTH,
    sunElevation = K.DEFAULT_SUN_ELEVATION,
    exposure = K.DEFAULT_EXPOSURE,
    spectralLightingStrength = K.SPECTRAL_LIGHTING_STRENGTH,
    lightTransferSplitStrength = K.LIGHT_TRANSFER_SPLIT_STRENGTH,
    // One number for the whole aerosol story; no per-term override exists.
    haze = K.DEFAULT_HAZE,
    toneMapWhitePoint = K.DEFAULT_TONE_MAP_WHITE_POINT,
    contrast = K.DEFAULT_CONTRAST,
    hazeHeightDependent = K.DEFAULT_HAZE_HEIGHT_DEPENDENT,
    // The app's angles at DEFAULT_LOD_STRENGTH, matching the Python host's
    // ViewState.
    // The viewer always passes these explicitly (its slider scales them); the
    // defaults are what a caller who does not care gets, and they have to be
    // the same number on both sides — see test_uniform_parity.
    lightMarchLodDegrees = K.APP_LIGHT_MARCH_LOD_DEGREES * K.DEFAULT_LOD_STRENGTH,
    viewStepLodDegrees = K.APP_VIEW_STEP_LOD_DEGREES * K.DEFAULT_LOD_STRENGTH,
    toneMapGamma = K.DEFAULT_TONE_MAP_GAMMA,
    frameIndex = 0,
    subpixel = false,
    jitterScale = 1.0,
    // Read tau_sun from the baked sun-tau cache. Only meaningful
    // when the renderer has a finished bake bound at binding 7; the packer
    // just writes the flag.
    lightCache = false,
    // The vertical sky-visibility march: off means
    // every consumer of t_sky sees a fully open sky.
    skyProbe = true,
    // Ice-detection false-color mode. Only meaningful
    // when a real ice-fraction texture is bound at binding 8; the packer
    // just writes the flag, same contract as lightCache.
    iceMode = false,
    // The scene clock, seconds: where a surface with a clock (the city's
    // steam) is in its motion.
    sceneTime = 0.0,
    // The city's steam drawn after the accumulation at the live clock
    // (city_steam_live, fs_steam_live in the shader): the live
    // view's, and its auto-exposure meter's. Every still, video and replay
    // leaves it off and marches the steam.
    liveSteam = false,
    // The city's roof clutter density: CITY_ROOF_DENSITY_BY_TIER
    // at the flight's tier, 1 as tuned. Packed under CITY only.
    cityRoofDensity = 1.0,
  } = view;

  const [outputW, outputH] = outputSize;
  const [w, h] = renderSize;

  unitInterval("jitter_scale", jitterScale);
  unitInterval("spectral_lighting_strength", spectralLightingStrength);
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
  // moon's rows are fixed night values, and the light-transfer split is
  // daytime-sun machinery with no night meaning.
  let spec, lightTransferEff;
  if (city) {
    spec = {
      cloudSun: K.NIGHT_MOON_CLOUD_COLOR, ambient: K.NIGHT_AMBIENT_TINT,
      horizon: K.NIGHT_SKY_HORIZON, bloom: K.NIGHT_MOON_BLOOM,
      disc: K.NIGHT_MOON_DISC,
    };
    lightTransferEff = 0.0;
  } else {
    spec = spectralLightingColors(sun, K.SUN_COLOR, spectralLightingStrength);
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
  if (nested) {
    row(7, nestBmin[0], nestBmin[1], nestBmin[2], dtViewNest);
    row(8, nestBmax[0], nestBmax[1], nestBmax[2], dtLightNest);
  }
  row(9, spec.cloudSun[0], spec.cloudSun[1], spec.cloudSun[2], lightTransferEff);
  row(10, spec.ambient[0], spec.ambient[1], spec.ambient[2], haze);
  row(11, spec.horizon[0], spec.horizon[1], spec.horizon[2],
      aerialBetaPerKm(haze) * 1e-3);          // beta0 in m^-1
  // A scale height of 0 is the shader's signal for "no height profile".
  row(12, spec.bloom[0], spec.bloom[1], spec.bloom[2],
      hazeHeightDependent ? K.AERIAL_SCALE_HEIGHT_M : 0.0);
  row(13, spec.disc[0], spec.disc[1], spec.disc[2],
      oceanHazeExtinctionPerKm(haze) * 1e-3);
  // The surface: the water's reflectance (none over the city) and its z.
  if (city) {
    row(14, 0.0, 0.0, 0.0, oceanZ);
  } else {
    row(14, oceanReflectance[0], oceanReflectance[1], oceanReflectance[2],
        oceanZ);
  }
  row(15, oceanFifDx, oceanTileExtent, oceanEnabled ? 1.0 : 0.0, oceanMaxLod);
  // The surface offset: where the surface tile sits in the world THIS FRAME
  // renders, folded into [0, surface period), the period being the
  // surface's tile (surfacePeriodM). The shader derives the city frame (and
  // the water uv) as world minus this, and because everything on a surface
  // is periodic in its tile, only the offset mod the tile can matter. A
  // camera (or replay pose) that carries its own tile phase defines the
  // offset as cam.xy minus surfacePosition, so a cloud-period fold of the
  // camera moves the offset with it and the surface never jumps.
  // fold(origin) is computed FIRST and surfacePosition subtracted after: at
  // zero drift fold(origin) is the very expression surfacePosition was
  // derived from, so the difference is the literal 0.0 and a day scene's
  // water reads subtract exactly zero. A camera with no surface frame
  // writes the static offset, folded.
  const period = surfacePeriodM ?? oceanTileExtent;
  const foldTile = (v) => {
    const r = v % period;
    return r < 0 ? r + period : r;
  };
  let surfaceOff;
  if (camera.surfacePosition) {
    surfaceOff = [
      foldTile(foldTile(origin[0]) - camera.surfacePosition[0]),
      foldTile(foldTile(origin[1]) - camera.surfacePosition[1]),
    ];
  } else {
    const stat = surfaceOffsetM ?? (city ? cityOffsetM : [0.0, 0.0]);
    surfaceOff = [foldTile(stat[0]), foldTile(stat[1])];
  }
  if (!Number.isFinite(Number(sceneTime))) {
    throw new Error(`scene_time must be finite; got ${sceneTime}.`);
  }
  row(16, surfaceOff[0], surfaceOff[1], sceneClock(sceneTime, clockPeriodS),
      0.0);                                   // w: the lens-water time step
  // The two sampling flags (SLOT.SUBPIXEL_JITTER, SLOT.JITTER_SCALE) are
  // excluded from scene identity; see sceneKey.
  row(17, w, h, subpixel ? 1.0 : 0.0, jitterScale);
  row(18, toneMapWhitePoint, toneMapGamma, contrast, iceMode ? 1.0 : 0.0);
  // The view-step angle floors at a QUARTER of the render's pixel angle.
  // Not the full pixel: the accumulation is jittered, so a converged image
  // resolves sub-pixel, and a floor at exactly one pixel dissolves the very
  // detail accumulation would have antialiased (Thomas, in flight: pixel-
  // matched still read substantially coarser than a pixel). A quarter is
  // where the return genuinely ends. An exact 0 stays 0 — the "LOD off,
  // fixed dt" sentinel, unreachable from the slider.
  const pixelTan = 2.0 * tanHalfFov / w;
  let viewStepTan = Math.tan(viewStepLodDegrees * DEG);
  if (viewStepTan > 0.0) viewStepTan = Math.max(viewStepTan, 0.25 * pixelTan);
  // The city's two: its roof clutter density and whether its steam is live.
  // Zero on a day scene, which reads neither.
  let roofDensity = 0.0;
  if (city) {
    const [lo, hi] = K.CITY_ROOF_DENSITY_RANGE;
    if (!(cityRoofDensity >= lo && cityRoofDensity <= hi)) {
      throw new RangeError(
        `cityRoofDensity must be in [${lo}, ${hi}] (the far mean's measured `
        + `range); got ${cityRoofDensity}.`);
    }
    roofDensity = cityRoofDensity;
  }
  row(19, Math.tan(lightMarchLodDegrees * DEG), viewStepTan, roofDensity,
      city && liveSteam ? 1.0 : 0.0);
  // The host's copies of three compile-time switches, so that changing one
  // changes the scene key. w is the bake slice index, which the renderer's
  // bake pass writes into its own copy of the block.
  row(20, periodic ? 1.0 : 0.0, lightCache ? 1.0 : 0.0,
      skyProbe ? 0.0 : 1.0, 0.0);
  return u;
}

/**
 * The scene-identity key that decides whether temporal accumulation continues
 * or restarts. Three slots are zeroed out on purpose:
 *
 *   the frame index — it only decorrelates jitter seeds. If it counted,
 *       every frame would read as a scene change and nothing would ever
 *       converge.
 *   the two sampling flags (sub-pixel jitter, jitter scale) — they are
 *       OUTPUTS of the accumulation decision. Including them would make the
 *       key self-referential.
 *
 * The output size is folded in because it changes the image without
 * appearing in the buffer.
 */
/**
 * The scene clock a surface reads: only where something reads it (a
 * surface with a period: the city's steam), wrapped there in float64,
 * before the narrowing to float32. A surface with no clock packs 0 whatever
 * the scene time, so its accumulation never restarts for it.
 */
export function sceneClock(sceneTime, clockPeriodS) {
  if (sceneTime === 0) return 0.0;
  if (clockPeriodS === undefined) {
    throw new Error("a running clock needs the surface's clockPeriodS — " +
                    "its meta.json clock_period_s: a period, or null for " +
                    "a surface with no clock.");
  }
  if (clockPeriodS === null) return 0.0;
  if (!(clockPeriodS > 0)) {
    throw new Error("clockPeriodS must be positive or null; got " +
                    `${clockPeriodS}.`);
  }
  const clock = sceneTime % clockPeriodS;
  return clock < 0 ? clock + clockPeriodS : clock;
}

export function sceneKey(u, outputW, outputH) {
  const copy = new Float32Array(u);
  copy[SLOT.FRAME_INDEX] = 0.0;
  copy[SLOT.SUBPIXEL_JITTER] = 0.0;
  copy[SLOT.JITTER_SCALE] = 0.0;
  const key = new Uint8Array(copy.byteLength + 8);
  key.set(new Uint8Array(copy.buffer), 0);
  new DataView(key.buffer).setUint32(copy.byteLength, outputW, true);
  new DataView(key.buffer).setUint32(copy.byteLength + 4, outputH, true);
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
