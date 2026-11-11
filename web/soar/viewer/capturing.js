// Viewer methods (capturing). Stills, track recording and track video: what a
// capture marches, and the three captures themselves.

"use strict";

import * as K from "../constants.js";
import { cameraWorldOrigin, worldToRelative } from "../camera.js";
import {
  trackPayload, resampleTrack, resampledFrameCount, knocksDue,
  MAX_TRACK_SECONDS,
} from "../track.js";
import {
  renderStill, imageDataToPng, download, timestampedName, createOfflineTarget,
  renderAccumulated, captureFrames, Readback,
} from "../capture.js";
import { autoExposureTarget } from "./exposure.js";

export const capturing = {
  /**
   * Whether any of the three look settings a capture tier owns is hand-set.
   * When it is, the capture panels select "Custom" by default: the session's
   * values verbatim, because a named tier is the PURE preset and would
   * silently discard exactly what the user just set.
   */
  get _lookIsCustom() {
    return this._byHand.has("haze") || this._byHand.has("lod")
      || this._byHand.has("autoExposure");
  },

  /** The still panel's tier. Stored null = not explicitly picked. */
  get captureStillTier() {
    return this._captureStillTier ?? (this._lookIsCustom ? "custom" : "max");
  },

  set captureStillTier(tier) { this._captureStillTier = tier; },

  get captureVideoTier() {
    return this._captureVideoTier ?? (this._lookIsCustom ? "custom" : "high");
  },

  set captureVideoTier(tier) { this._captureVideoTier = tier; },

  /**
   * The QUALITY_PRESETS tier a capture at `tier` renders with. "Custom" is
   * the session's look on the LIVE tier's lighting method and parked spp —
   * there is no custom row in the preset table, deliberately.
   */
  capturePresetTier(tier) {
    if (tier === "custom") return this.qualityTier ?? K.DEFAULT_QUALITY_TIER;
    if (!(tier in K.QUALITY_PRESETS)) {
      throw new Error(`unknown capture tier '${tier}'.`);
    }
    return tier;
  },

  /**
   * Everything a capture at `tier` marches that is not the camera: the tier
   * name it will be recorded under, the preset that supplies lighting method
   * and spp, and the look — haze, LOD, exposure.
   *
   * A NAMED tier is the pure parked preset, full stop:
   * the tier's haze default and its parked LOD — the tier table over clouds,
   * CITY_PARKED_LOD_STRENGTH over the city — with the session's hand-set
   * values deliberately NOT mixed in. A hand-set look selects the Custom
   * tier instead, which carries it verbatim, and a named tier means the
   * same picture on every machine and every session.
   *
   * `meter` says the exposure is still to be measured: the tier's AE default
   * is on, so _meterCaptureExposure replaces `exposure` with the converged
   * target before the march. Custom never meters — the session's exposure is
   * already where the live loop (or the hand) put it. hazeHeightDependent is
   * not tier-varying, so the session value rides through either way.
   */
  _captureLook(tier) {
    if (tier === "custom") {
      return {
        tier: "custom", preset: this.capturePresetTier(tier),
        haze: this.haze, hazeHeightDependent: this.hazeHeightDependent,
        lodStrength: this.lodStrength,
        exposure: this.exposure, meter: false,
      };
    }
    this.capturePresetTier(tier);   // validates the name
    return {
      tier, preset: tier,
      haze: K.DEFAULT_HAZE_BY_TIER[tier],
      hazeHeightDependent: this.hazeHeightDependent,
      lodStrength: this.scene?.city
        ? K.CITY_PARKED_LOD_STRENGTH : K.DEFAULT_LOD_STRENGTH_BY_TIER[tier],
      exposure: K.DEFAULT_EXPOSURE,
      meter: Boolean(K.AUTO_EXPOSURE_DEFAULT_BY_TIER[tier]),
    };
  },

  /** The per-frame view kwargs a capture at `look` marches. */
  _captureViewKwargs(look) {
    return {
      ...this._viewKwargs({ lodStrength: look.lodStrength,
                            tier: look.preset }),
      haze: look.haze,
      hazeHeightDependent: look.hazeHeightDependent,
      exposure: look.exposure,
    };
  },

  /**
   * The exposure a capture applies, decided ONCE at capture start — no
   * glide, no deadband: nobody watches a capture converge, and for a video a
   * per-frame meter would flicker, so one metered still exposure holds for
   * the whole file. Metered with the CAPTURE's view kwargs (its haze and
   * LOD change the luminance), against the live renderer state. A live
   * meter's readback may still be in
   * flight (one is ever allowed; that is the only null); it always resolves,
   * so wait it out rather than skipping the metering. A meter that fails
   * throws, and fails the capture with it. A meter that saw no light keeps
   * DEFAULT_EXPOSURE, the live loop's own ignore rule.
   */
  async _meterCaptureExposure(look, size, overrides = {}) {
    if (!look.meter) return look.exposure;
    // The meter reads the light cache's sky visibility, so the cache is
    // finished first: the exposure must not depend on how far a bake had got.
    await this.renderer.finishLightBake();
    for (;;) {
      const highlight = await this.renderer.meterLuminance(
        size, { ...this._captureViewKwargs(look), ...overrides });
      if (highlight !== null) {
        return highlight > 0
          ? autoExposureTarget(highlight, this.toneMapWhitePoint)
          : K.DEFAULT_EXPOSURE;
      }
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
  },

  /**
   * R: start or stop recording the flight path.
   *
   * What is recorded is the track, not the pixels — where the camera was and
   * when, once per rendered frame. The video is made afterwards by flying
   * that path again at an exact frame rate with the accumulation converged,
   * which is a picture the live view never shows you and a screen recording
   * cannot produce.
   */
  toggleTrackRecording() {
    if (this.recorder.recording) { this._finishTrackRecording(); return; }
    this.recorder.start();
    // The track's lens: the switch as it stands now, recorded in the header
    // and flown by the video; and a dry start, as every replay starts dry.
    this._trackLensWater = this.lensWater;
    this._trackPausedS = 0.0;
    this.renderer.resetLens();
    // A recording holds the loop awake outright — a track's samples are its
    // timeline, and a sleeping loop would record a pause as nothing at all.
    this._wake("track recording");
    this.ui.say(
      "Recording the flight path. R again to stop; it stops itself after " +
      `${Math.round(MAX_TRACK_SECONDS / 60)} minutes of flying.`, 4);
  },

  /** Stop recording and offer what was caught. */
  _finishTrackRecording(reachedLimit = false) {
    const samples = this.recorder.stop();
    if (samples.length < 2) {
      this.ui.say("Too short to be a track — nothing recorded.", 3);
      return;
    }
    this.pause();
    if (reachedLimit) {
      this.ui.say(
        `That is ${Math.round(MAX_TRACK_SECONDS / 60)} minutes of flying, ` +
        "which is the longest track cloudyview records. Here is what it " +
        "caught.", 6);
    }
    this.ui.open("track", { samples, knocks: this.recorder.knocks });
  },

  /** How many video frames a track becomes at the chosen rate. */
  trackFrameCount(samples) {
    return resampledFrameCount(samples, this.videoFps);
  },

  /**
   * Fly a recorded track again, offline, and encode it.
   *
   * Every output frame is rendered with the accumulation converged — the
   * picture the live view only reaches when you stop moving — and stamped
   * with the timestamp the track says it has, not the one the clock says.
   * That is the whole argument for WebCodecs over screen capture: a frame
   * may take six seconds to converge and the video is still 30 seconds long.
   */
  async renderTrackVideo(samples, knocks = []) {
    if (this._capturing) return;
    const { VideoWriter, evenSize } = await import("../video.js");

    // Even before the offscreen texture is made, not after: H.264 refuses odd
    // dimensions, and every frame has to be the size the encoder was told.
    const size = evenSize(this.captureDimensions());
    // Over the city the positions go straight between samples: those are
    // the moves the collider cleared (track.js linear()).
    const frames = resampleTrack(samples, this.videoFps,
                                 { periodic: this.renderer.periodic,
                                   linearPosition: Boolean(this.scene.city) });

    this._capturing = true;
    this._wake("capture");
    this._videoAbort = false;
    // The replay flies the flyer from the track's first pose, not from
    // wherever the live flight left it.
    this.flyer?.calm();
    this.ui.close();
    this.ui.showProgress("Choosing a codec…", 0);

    let writer = null, target = null, readback = null, offline = false;
    try {
      writer = new VideoWriter({
        width: size[0], height: size[1], fps: this.videoFps,
      });
      const chosen = await writer.init();
      for (const warning of writer.warnings ?? []) console.warn(warning);

      // The capture's look, decided once for the whole video — including
      // the exposure, metered on the first frame and then HELD
      // (_trackLook).
      const look = await this._trackLook(size, frames[0]);

      target = createOfflineTarget(this.device, size, "soar-video");
      // One staging buffer for the whole video, like the target: made per
      // frame, their freed memory outran the driver's reclaiming on long
      // videos (capture.js Readback).
      readback = new Readback(this.device, size);
      await this.renderer.beginOffline(look.preset);
      offline = true;
      const framesPerVideoFrame = captureFrames(look.preset);
      // A video's lens starts dry and steps by the resampled track's own
      // frame interval — what `soar TRACK.json` steps it by — never by how
      // long a frame took to render; on or off as the track was flown (the
      // header says the same).
      this.renderer.resetLens();
      const lensWater = this._trackLensWater ?? this.lensWater;

      const t0 = performance.now();
      for (let i = 0; i < frames.length; i++) {
        if (this._videoAbort) throw new Error("Cancelled.");
        const pose = this._trackPose(frames[i]);
        const overlays = this.captureOverlays
          ? this._offlineOverlays(pose, size, 1.0 / this.videoFps, look,
                                  knocksDue(knocks,
                                            i > 0 ? frames[i - 1].t : -Infinity,
                                            frames[i].t))
          : [];
        await renderAccumulated(
          this.renderer, target.view, size,
          // The frame's own scene clock: the city's steam moves as it
          // moved in flight, one video frame at a time, however many passes
          // each frame accumulates.
          { ...this._captureViewKwargs(look),
            camera: pose, frameIndex: i * 1024,
            sceneTime: frames[i].sceneTime },
          framesPerVideoFrame, overlays, null,
          lensWater
            ? { dt: i > 0 ? frames[i].t - frames[i - 1].t : 0.0 } : null);
        await writer.addFrame(await readback.read(target.texture), i);

        const done = (i + 1) / frames.length;
        const rate = (i + 1) / ((performance.now() - t0) / 1000);
        const eta = (frames.length - i - 1) / Math.max(rate, 1e-6);
        this.ui.showProgress(
          `Frame ${i + 1} of ${frames.length} · ${chosen.label} · ` +
          `${eta > 90 ? `${(eta / 60).toFixed(1)} min` : `${eta.toFixed(0)} s`}` +
          " left — Esc to cancel", done);
      }

      this.ui.showProgress("Finishing the file…", 1);
      const blob = await writer.finish();
      download(blob, timestampedName("cloudyview_soar", writer.fileExtension));
      this.ui.say(
        `Saved ${frames.length} frames as ${writer.fileExtension} ` +
        `(${(blob.size / 1e6).toFixed(1)} MB, ${chosen.label}).`, 6);
    } catch (err) {
      await writer?.abort();
      this.ui.say(this._videoAbort
        ? "Video cancelled; nothing was saved."
        : `Could not make the video: ${err.message}`, 6);
    } finally {
      target?.texture.destroy();
      readback?.destroy();
      // The renderer is gone when the session was disposed mid-render — the
      // abort flag is what woke this loop up. There is then no state left to
      // restore, and reaching for it would throw inside a finally.
      if (offline && this.renderer) this.renderer.endOffline();
      // The video stepped the lens along the track; the live flight resumes
      // with a dry one rather than the replay's last drops.
      this.renderer?.resetLens();
      this.ui.hideProgress();
      this._capturing = false;
      // And back to the live flight, which the replay's last pose is not.
      this.flyer?.calm();
      // The capture owned the renderer's quality state and its accumulation;
      // whatever the live view was converging towards is gone, and the loop
      // may have been asleep when the capture began.
      this._wake("capture finished");
      this._videoAbort = false;
      this._lastTime = performance.now();
    }
  },

  /**
   * The camera pose a resampled track frame renders from. The frame's
   * surface offset rides on the pose the same way the camera origin does —
   * packUniforms derives u.surface_offset from a pose that carries a surfacePosition
   * (world metres, like the live camera's). A v1 track has no surface
   * frame: over a city the frame's world xy folds against the static offset
   * instead, and over the day ocean the null pose frame makes packUniforms
   * write the static offset folded (exactly 0.0) — the un-shifted surface
   * either way, which is what those recordings meant.
   */
  _trackPose(frame) {
    const position = cameraWorldOrigin(
      frame.position, this.scene.bmin, this.scene.bmax);
    const sp = frame.surfacePosition;
    return {
      position,
      azimuth: frame.azimuth,
      elevation: frame.elevation,
      fov: frame.fov,
      surfacePosition: sp
        ? [sp[0] * this.scene.surfacePeriod, sp[1] * this.scene.surfacePeriod]
        : (this.scene.cityPosition(position)?.slice(0, 2) ?? null),
    };
  },

  /** The overlays an offline frame draws, laid out for that frame's size.
   *  `look` is the CAPTURE's, so the flyer is exposed like the picture it
   *  flies in and the minimap carries that picture's haze, not the session's
   *  it was launched from. */
  _offlineOverlays(pose, size, dt, look, knocks = []) {
    const haze = look.haze;
    const overlays = [];
    if (this.flyer && this.flyerEnabled) {
      this.flyer.update(dt, pose);
      // A crash in the recorded flight: the pose already carries the
      // bounce, fall and shake; the tumble is the flyer's, handed over here
      // at the age it has on this frame.
      for (const knock of knocks) this.flyer.knock(knock);
      this.flyer.writeUniforms(pose, size, {
        sunAzimuth: this.sunAzimuth, sunElevation: this.sunElevation,
        exposure: look.exposure,
        toneMapGamma: this.toneMapGamma,
        toneMapWhitePoint: this.toneMapWhitePoint,
        contrast: this.contrast,
      });
      overlays.push((enc, view, format) =>
        this.flyer.encodePass(enc, view, format, size));
    }
    if (this.minimap && this.minimapMode !== "off") {
      this.minimap.update({
        ...pose,
        relativePosition: () => worldToRelative(
          pose.position, this.scene.bmin, this.scene.bmax),
      }, this.scene, size, false, haze);
      overlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }
    return overlays;
  },

  /**
   * A track video's look, decided once for the whole file: the video tier's
   * look, with the exposure metered at frame 0's own pose and clock — not
   * the live camera, which sits wherever the flight ended — and then HELD:
   * a per-frame meter would flicker, and a video wants a still exposure the
   * way a camera on a tripod does. The browser's video and the header of
   * the saved track (which `soar TRACK.json` renders from) both come from
   * here, so the two videos of one flight share an exposure.
   */
  async _trackLook(size, frame0) {
    const look = this._captureLook(this.captureVideoTier);
    look.exposure = await this._meterCaptureExposure(
      look, size, { camera: this._trackPose(frame0),
                    sceneTime: frame0.sceneTime });
    return look;
  },

  /** The track as a file, which `soar TRACK.json` renders to video.
   *  Its header describes the video — the video tier's look, metered as
   *  renderTrackVideo meters it — rather than a still of the live view. */
  async downloadTrack(samples, knocks = []) {
    const { evenSize } = await import("../video.js");
    const size = evenSize(this.captureDimensions());
    const [frame0] = resampleTrack(
      samples, this.videoFps, { periodic: this.renderer.periodic,
                                linearPosition: Boolean(this.scene.city) });
    const look = await this._trackLook(size, frame0);
    const payload = trackPayload(
      this.renderMetadata(size, look,
                          { lensWater: this._trackLensWater ?? this.lensWater }),
      samples, knocks, this.scene.surfacePeriod);
    download(new Blob([JSON.stringify(payload)], { type: "application/json" }),
             timestampedName("cloudyview_track", ".json"));
    this.ui.say(`Saved ${samples.length} samples.`, 3);
  },

  /** The capture resolution: an explicit choice, or whatever the window is. */
  captureDimensions() {
    return this.captureSize ?? [this.canvas.width, this.canvas.height];
  },

  async saveScreenshot() {
    const overlays = this.captureOverlays;
    if (this._capturing) return;
    this._capturing = true;
    this._wake("capture");
    const size = this.captureDimensions();
    this.ui.close();
    this.ui.showProgress(
      `Rendering a ${size[0]}x${size[1]} still…`, 0);
    // A still is rendered at the capture size, not the window's, so the
    // overlays are re-laid-out for it. The bird holds the pose it had when
    // the button was pressed rather than flying on through the accumulation.
    const stillOverlays = [];
    const flyerDrawn = overlays && this.flyer && this.flyerEnabled;
    if (flyerDrawn) {
      stillOverlays.push((enc, view, format) =>
        this.flyer.encodePass(enc, view, format, size));
    }
    // The capture's look, not the session's: a named tier is the pure parked
    // preset (Custom carries the session's values). Built before the minimap
    // below so the overlay draws the haze the picture actually has.
    const look = this._captureLook(this.captureStillTier);
    if (overlays && this.minimap && this.minimapMode !== "off") {
      this.minimap.update(this.camera, this.scene, size, false, look.haze);
      stillOverlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }
    try {
      look.exposure = await this._meterCaptureExposure(look, size);
      if (flyerDrawn) {
        // After the metering, so the flyer is exposed like the picture.
        this.flyer.writeUniforms(this.camera, size, {
          sunAzimuth: this.sunAzimuth, sunElevation: this.sunElevation,
          exposure: look.exposure,
          toneMapGamma: this.toneMapGamma,
          toneMapWhitePoint: this.toneMapWhitePoint,
          contrast: this.contrast,
        });
      }
      const image = await renderStill(
        this.device, this.renderer,
        this._captureViewKwargs(look), size,
        look.preset, stillOverlays,
        (fraction) => this.ui.showProgress(
          `Rendering a ${size[0]}x${size[1]} still…`, fraction));
      this.ui.showProgress("Encoding…", 0.95);
      const blob = await imageDataToPng(image, this.renderMetadata(size, look));
      download(blob, timestampedName("cloudyview_soar", ".png"));
      this.ui.say(`Saved a ${size[0]}x${size[1]} still.`, 3);
    } catch (err) {
      this.ui.say(`Could not save the still: ${err.message}`, 5);
    } finally {
      this.ui.hideProgress();
      this._capturing = false;
      // The capture owned the renderer's quality state and its accumulation;
      // whatever the live view was converging towards is gone, and the loop
      // may have been asleep when the capture began.
      this._wake("capture finished");
      this._lastTime = performance.now();
    }
  },
};
