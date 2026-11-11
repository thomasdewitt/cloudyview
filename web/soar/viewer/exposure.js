// Viewer methods (exposure). Exposure by hand, and the auto-exposure loop.

"use strict";

import * as K from "../constants.js";

/**
 * The auto-exposure target for a metered highlight: the exposure that places
 * the highlight statistic AUTO_EXPOSURE_HIGHLIGHT_FRACTION under the white
 * point, softened by AUTO_EXPOSURE_RESPONSE and clamped to the auto range.
 * `highlight` is exposure-independent (the meter's march has the tone map —
 * and with it the exposure multiply — compiled out), so the result is an
 * absolute exposure, not a correction.
 *
 * The ONE statement of the formula: _aeTick glides toward it, a capture
 * applies it directly, and tools/prebake_demos.py restates it in Python
 * (auto_exposure — change one, change both). Callers own the non-positive-
 * highlight case (the live loop ignores the reading; a capture keeps
 * DEFAULT_EXPOSURE), so a meter that saw no light at all is an error here,
 * not a silent 4.0.
 */
export function autoExposureTarget(highlight, whitePoint) {
  if (!(highlight > 0)) {
    throw new Error(`autoExposureTarget needs a positive highlight; got ${highlight}.`);
  }
  const [lo, hi] = K.AUTO_EXPOSURE_LIMITS;
  const full = K.AUTO_EXPOSURE_HIGHLIGHT_FRACTION * whitePoint / highlight;
  return Math.min(hi, Math.max(
    lo, full >= hi ? hi : hi * Math.pow(full / hi, K.AUTO_EXPOSURE_RESPONSE)));
}

export const exposure = {
  /**
   * Video-camera exposure. On, the linear meter drives `exposure` (see
   * _aeTick); off, it freezes wherever the meter left it — manual mode on a
   * camera inherits the metered setting, and snapping back to the default
   * would visibly re-expose the very picture the user just chose to pin.
   * A hand toggle is final: no tier change flips it back.
   */
  setAutoExposure(on, { byHand = true } = {}) {
    if (byHand) this._byHand.add("autoExposure");
    this.autoExposure = Boolean(on);
    this._aeSettled = false;
    if (this.autoExposure) {
      this._aeLastMeterMs = 0;
      this._wake("auto exposure");
    }
  },

  /** The hand slider. Moving it is choosing manual, as on a camera body. */
  setExposure(exposure) {
    const [lo, hi] = K.EXPOSURE_LIMITS;
    if (!(exposure >= lo && exposure <= hi)) {
      throw new Error(`exposure must be in [${lo}, ${hi}]; got ${exposure}.`);
    }
    if (this.autoExposure) this.setAutoExposure(false);
    this.exposure = exposure;
    this.renderer.resetAccumulation();
    this._wake("exposure");
  },

  /**
   * One metering step per frame (the one-readback-in-flight guard in
   * meterLuminance is the real throttle), folded into the exposure with a
   * fixed-time-constant glide over MEASURED elapsed time, so the feel does
   * not depend on the frame rate. Skipped while the startup probe is timing
   * frames (its queue drains must not measure meter work) — captures never
   * reach here, since _frame short-circuits on them.
   *
   * The glide stops inside the stop band and does not restart until the
   * target leaves the (wider) start band: every exposure step is a scene-key
   * change that restarts accumulation, so without the hysteresis a parked
   * view would chase metering noise forever instead of converging.
   */
  _aeTick(nowMs, outputSize) {
    if (!this.autoExposure || this._probe) return;
    if (nowMs - this._aeLastMeterMs < K.AUTO_EXPOSURE_INTERVAL_MS) return;
    this._aeLastMeterMs = nowMs;
    const renderer = this.renderer;
    // The live picture's steam is not in its accumulation, so not in what
    // sets its exposure either: a meter that saw it move would move the
    // exposure, and the exposure is in the scene key. Where the steam is
    // marched (Low, Minimal) the meter marches it too.
    renderer.meterLuminance(outputSize, { ...this._viewKwargs(),
                                          liveSteam: renderer.liveSteamActive })
      .catch((err) => {
        // A readback abandoned because its renderer was torn down (a field
        // change, leaving the viewer) is not a failure of anything live.
        // Anything else is, and stops rendering as loudly as a draw would.
        if (!this._disposed && this.renderer === renderer) {
          this._failRendering(err);
        }
        return null;
      })
      .then((highlight) => {
        if (!(highlight > 0) || this._disposed || !this.autoExposure) return;
        // The one statement of the target formula (a capture applies the
        // same one directly, with no glide) — see autoExposureTarget.
        const target = autoExposureTarget(highlight, this.toneMapWhitePoint);
        const err = Math.abs(Math.log2(target / this.exposure));
        if (this._aeSettled) {
          if (err < K.AUTO_EXPOSURE_DEADBAND_START_LOG2) return;
          this._aeSettled = false;
        } else if (err < K.AUTO_EXPOSURE_DEADBAND_STOP_LOG2) {
          this._aeSettled = true;
          return;
        }
        // Elapsed time is measured between APPLIED steps — readbacks can
        // skip frames under load, and the glide should not slow with them.
        const applyMs = performance.now();
        const elapsedS = Math.min(
          0.25, Math.max(0.001, (applyMs - (this._aeLastApplyMs ?? applyMs
                                            - 16.7)) / 1000));
        this._aeLastApplyMs = applyMs;
        const alpha = 1 - Math.exp(
          -elapsedS / K.AUTO_EXPOSURE_TIME_CONSTANT_S);
        this.exposure *= Math.pow(target / this.exposure, alpha);
        this.ui?.refreshExposure?.(this.exposure);
        // The new exposure reaches the shader through _viewKwargs on the
        // next frame; it is in the scene key, so accumulation restarts by
        // itself. A sleeping loop needs the nudge.
        this._wake("auto exposure");
      });
  },
};
