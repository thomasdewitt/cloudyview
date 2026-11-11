// Viewer methods (settings). The look and quality setters the menu and the
// keys call: sun, tone map, haze, LOD, hold mode, tier, render scale, lens
// water, periodic domain.

"use strict";

import * as K from "../constants.js";
import { mod360, HAZE_MIN } from "../spectral.js";

// Lens water's setting, per browser: default on. Storage can be missing or
// refuse (a private window); the setting then simply is not remembered.
const LENS_WATER_KEY = "soar.lensWater";
export function readLensWater() {
  try {
    return localStorage.getItem(LENS_WATER_KEY) !== "off";
  } catch {
    return true;
  }
}
function writeLensWater(on) {
  try {
    localStorage.setItem(LENS_WATER_KEY, on ? "on" : "off");
  } catch {
    // Not remembered; the switch still works for this session.
  }
}

export const settings = {
  setSun({ azimuth, elevation, zenith }) {
    if (azimuth != null) this.sunAzimuth = mod360(azimuth);
    if (zenith != null) elevation = 90.0 - zenith;
    if (elevation != null) {
      this.sunElevation = Math.min(
        90.0, Math.max(K.MIN_SUN_ELEVATION_DEG, elevation));
    }
    // The baked sun-tau cache is a function of the sun; while it re-bakes
    // the march runs live, so dragging the sun is correct throughout and
    // merely slower until the fresh cache lands. The texture is reused, so
    // this never re-risks the load-time memory probe.
    this.renderer.invalidateLightCache();
    this.renderer.startLightBake(this._viewKwargs());
    this._wake("sun");
  },

  /**
   * J: flip the sky probe. The same setting as the Advanced panel's row, so
   * flipping it by key also marks the quality Custom.
   */
  toggleSkyProbe() {
    const on = !this.skyProbeOn;
    this.setSkyProbe(on);
    this.markQualityCustom();
    this.ui.say(on
      ? "Sky probe on — measured sky visibility in shadow."
      : "Sky probe off — shadowed samples assume open sky.", 4);
  },

  /** K: flip the sun-tau cache (on always means the /2 bake). Same setting
   *  as the Advanced panel's row; marks the quality Custom. */
  toggleLightCache() {
    const on = !this.lightCacheOn;
    this.setLightCache(on);
    this.markQualityCustom();
    this.ui.say(on
      ? "Light cache on — baked sun shadows."
      : "Light cache off — live sun march.", 4);
  },

  setToneMapGamma(gamma) {
    const [lo, hi] = K.TONE_MAP_GAMMA_LIMITS;
    if (!(gamma >= lo && gamma <= hi)) {
      throw new Error(`tone_map_gamma must be in [${lo}, ${hi}]; got ${gamma}.`);
    }
    this.toneMapGamma = gamma;
    this.renderer.resetAccumulation();
    this._wake("tone-map gamma");
  },

  setToneMapWhitePoint(whitePoint) {
    const [lo, hi] = K.TONE_MAP_WHITE_POINT_LIMITS;
    if (!(whitePoint >= lo && whitePoint <= hi)) {
      throw new Error(
        `tone_map_white_point must be in [${lo}, ${hi}]; got ${whitePoint}.`);
    }
    this.toneMapWhitePoint = whitePoint;
    this.renderer.resetAccumulation();
    this._wake("tone-map white point");
  },

  setContrast(contrast) {
    const [lo, hi] = K.CONTRAST_LIMITS;
    if (!(contrast >= lo && contrast <= hi)) {
      throw new Error(`contrast must be in [${lo}, ${hi}]; got ${contrast}.`);
    }
    this.contrast = contrast;
    this.renderer.resetAccumulation();
    this._wake("contrast");
  },

  /**
   * How much aerosol is in the air, 0 to 1. Unlike the gamma above this is
   * not a display choice — it changes what the march computes, so the frames
   * already averaged are of a different sky and cannot be kept.
   */
  setHaze(haze, { byHand = true } = {}) {
    if (!(haze >= HAZE_MIN && haze <= K.HAZE_MAX)) {
      throw new Error(`haze must be in [${HAZE_MIN}, ${K.HAZE_MAX}]; got ${haze}.`);
    }
    if (byHand) this._byHand.add("haze");
    this.haze = haze;
    this.renderer.resetAccumulation();
    this._wake("haze");
  },

  /**
   * Whether that aerosol thins with height.
   *
   * Off is uniform haze — the sea-level extinction at every altitude, which
   * is unphysical and is the cheapest range lever in the renderer: an
   * exponential atmosphere at a 2.5 km scale height lets an upward ray leave
   * the haze without ever reaching the cutoff optical depth, so only the
   * range ceiling bounds it. Changes what the march computes, so the
   * accumulated frames are of a different sky and cannot be kept.
   */
  setHazeHeightDependent(on, { byHand = true } = {}) {
    if (byHand) this._byHand.add("hazeHeightDependent");
    this.hazeHeightDependent = Boolean(on);
    this.renderer.resetAccumulation();
    this._wake("haze profile");
  },

  /**
   * How coarsely the far field is stepped, as a multiple of the tuned angular
   * LOD. Cheap distance, and it degrades along the grain of the look.
   */
  setLodStrength(strength, { byHand = true } = {}) {
    const [lo, hi] = K.LOD_STRENGTH_LIMITS;
    if (!(strength >= lo && strength <= hi)) {
      throw new Error(`LOD strength must be in [${lo}, ${hi}]; got ${strength}.`);
    }
    if (byHand) this._byHand.add("lod");
    this.lodStrength = strength;
    this.renderer.resetAccumulation();
    this._wake("level of detail");
  },

  /**
   * Live or still: what a held view settles to. See HOLD_MODES.
   *
   * Held on the viewer as well as on the renderer, because there are windows
   * with no renderer in them: _releaseField nulls it, and the panels that go
   * up during a load — which group? which units? — drive this from the UI on
   * their way up, which threw and turned a three-group file into "Could not
   * open this field" (2026-08-14). Remembering it here rather than skipping
   * the call means the mode survives the gap instead of depending on an
   * argument about which panels can be open while a field loads.
   */
  setHoldMode(mode) {
    this._holdMode = mode;
    this.renderer?.setHoldMode(mode);
    this._wake("hold mode");
  },

  /** The mode a renderer built from now on should start in. */
  get holdMode() { return this._holdMode ?? K.DEFAULT_HOLD_MODE; },

  /**
   * The tier's own defaults for the three settings that have them, skipping
   * any the user has set by hand. Motion smoothing is carried across in
   * SMOOTHING units rather than alpha, so a tier change moves the floor
   * without moving where the slider sits.
   */
  _applyTierDefaults(tier) {
    if (!this._byHand.has("haze")) {
      this.setHaze(K.DEFAULT_HAZE_BY_TIER[tier], { byHand: false });
    }
    if (!this._byHand.has("autoExposure")) {
      this.setAutoExposure(K.AUTO_EXPOSURE_DEFAULT_BY_TIER[tier] ?? false,
                           { byHand: false });
    }
    if (!this._byHand.has("lod")) {
      this.setLodStrength(
        K.lodStrengthByTier(this.scene?.city)[tier], { byHand: false });
    }
    const smoothing = this._byHand.has("smoothing")
      ? this.motionSmoothing
      : K.DEFAULT_MOTION_SMOOTHING_BY_TIER[tier];
    this.setMotionSmoothing(smoothing, { byHand: false });
  },

  /** Where the motion-smoothing slider sits, in its own units. */
  get motionSmoothing() {
    return K.motionSmoothingForAlpha(
      this.renderer.motionBlendAlpha, this.renderer.qualityTier);
  },

  /**
   * The Quality panel's tier buttons. Choosing by hand ends the automatic
   * choice for the session — including on the next field loaded, because
   * having the app overrule a deliberate choice a minute later is worse than
   * carrying a stale one.
   */
  /**
   * The Quality panel's Auto button: hand the choice back to measurement
   * after a manual pick ended it. Re-runs the startup probe's walk from the
   * floor rather than trusting the hand-picked tier — the probe is the only
   * mechanism that can move DOWN (the governor climbs), and its walk is the
   * one path already proven never to render an unaffordable frame.
   */
  enableAutoTier() {
    if (this._autoTier && !this.qualityCustom) return;
    this._autoTier = true;
    this._resetQualityOverrides();
    this._applyTierDefaults(this.qualityTier ?? K.DEFAULT_QUALITY_TIER);
    this._beginAutoTier();
    this._wake("auto tier");
  },

  setQualityTier(tier) {
    this._autoTier = false;
    this._probe = null;
    this._governor = null;
    clearTimeout(this._confirmTimer);
    this._confirmTimer = null;
    this.qualityTier = tier;
    this._resetQualityOverrides();
    this.renderer.setQualityTier(tier);
    this._applyTierDefaults(tier);
    this._wake("quality tier");
  },

  /**
   * Clicking a preset (or Auto) means the preset, whole: every Advanced
   * setting returns to its default — the per-tier ones through
   * _applyTierDefaults after the byHand marks are cleared here, the global
   * ones directly — and the Custom chip goes away.
   */
  _resetQualityOverrides() {
    this._byHand.clear();
    const r = this.renderer;
    r.lightCacheMode = "auto";
    r.skyProbeMode = "auto";
    r.parkedSppOverride = null;
    this.setToneMapGamma(K.DEFAULT_TONE_MAP_GAMMA);
    this.setToneMapWhitePoint(K.DEFAULT_TONE_MAP_WHITE_POINT);
    this.setContrast(K.DEFAULT_CONTRAST);
    this.setHazeHeightDependent(K.DEFAULT_HAZE_HEIGHT_DEPENDENT);
    // Directly, not through setExposure — that setter reads a drag on the
    // slider and turns auto-exposure off, which the tier default about to be
    // applied may immediately turn back on.
    this.exposure = K.DEFAULT_EXPOSURE;
    this.qualityCustom = false;
  },

  /** Any hand-change in the Advanced panel (or K/J) lands here. */
  markQualityCustom() {
    this.qualityCustom = true;
  },

  /** The flight-time sky probe the current settings ask for. */
  get skyProbeOn() {
    const r = this.renderer;
    if (r.skyProbeMode !== "auto") return r.skyProbeMode === "on";
    return Boolean(K.QUALITY_PRESETS[r.qualityTier].skyProbe);
  },

  get lightCacheOn() {
    const r = this.renderer;
    if (r.lightCacheMode !== "auto") return r.lightCacheMode === "on";
    return Boolean(K.QUALITY_PRESETS[r.qualityTier].lightCache);
  },

  get parkedSpp() {
    return this.renderer.parkedSppOverride
      ?? K.PARKED_ACCUM_FRAMES_BY_TIER[this.renderer.qualityTier];
  },

  setSkyProbe(on) {
    this.renderer.skyProbeMode = on ? "on" : "off";
    this.renderer.resetAccumulation();
    this._marchPending = true;
    this._wake("sky probe");
  },

  setLightCache(on) {
    this.renderer.lightCacheMode = on ? "on" : "off";
    this.renderer.resetAccumulation();
    this._marchPending = true;
    this._wake("light cache");
  },

  setParkedSpp(spp) {
    this.renderer.parkedSppOverride = spp;
    this._marchPending = true;
    this._wake("parked spp");
  },

  /** The Quality panel's render-scale sliders, by hand: what the march runs
   *  at while the camera moves, and what a held view climbs to. */
  setRenderScale(scale) {
    this.renderer.setRenderScale(scale);
    this._wake("render scale");
  },

  setHoldRenderScale(scale) {
    this.renderer.setHoldRenderScale(scale);
    this._wake("hold render scale");
  },

  /**
   * The Quality panel's motion-smoothing slider, in smoothing units: 0 is
   * none, 1 is as deep as this tier allows. The renderer still holds an
   * alpha, which runs the other way — see motionAlphaForSmoothing.
   */
  setMotionSmoothing(smoothing, { byHand = true } = {}) {
    if (byHand) this._byHand.add("smoothing");
    this.renderer.motionBlendAlpha = K.motionAlphaForSmoothing(
      smoothing, this.renderer.qualityTier);
    this._wake("motion smoothing");
  },

  /** The field-of-view slider, which is a camera change like any other. */
  setFov(fov) {
    this.camera.setFov(fov);
    this._wake("field of view");
  },

  /** Lens water on or off; off dries the lens at once. */
  setLensWater(on) {
    this.lensWater = Boolean(on);
    writeLensWater(this.lensWater);
    if (!this.lensWater) this.renderer?.resetLens();
    // Switched on while parked in cloud, the lens has wetting to do before
    // any read-back can say so: keep the loop up until one does.
    else if (this.renderer) this.renderer.lensSettled = false;
    this._wake("lens water");
  },

  togglePeriodic() {
    this.renderer.setPeriodic(!this.renderer.periodic);
    this.camera.periodic = this.renderer.periodic;
    this.camera.constrain();
    // The baked sun-tau cache marched its shadows under the OLD wrap: baked
    // non-periodic, the sun ray tapers into zero at the domain wall, and
    // reading that cache in a periodic fly paints a lighting seam on every
    // lateral face. Same recovery as setSun: live march while the fresh
    // bake lands.
    this.renderer.invalidateLightCache();
    this.renderer.startLightBake(this._viewKwargs());
    this.ui.say(`Periodic domain ${this.renderer.periodic ? "on" : "off"}.`);
    this._wake("periodic");
  },
};
