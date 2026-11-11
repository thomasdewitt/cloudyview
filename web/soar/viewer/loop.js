// Viewer methods (loop). The frame loop: waking and sleeping it, and one
// frame.

"use strict";

import * as K from "../constants.js";

// The longest single step the live lens takes (s): see _frame.
const LENS_MAX_STEP_S = 1.0;

/** The roofs' clutter density at a tier: a tier with none is an error, not
 *  a default (the far mean is measured at each tier's own density). */
function cityRoofDensity(tier) {
  const d = K.QUALITY_PRESETS[tier].cityRoofDensity;
  if (d === undefined) {
    throw new Error(`no roof clutter density for tier '${tier}'.`);
  }
  return d;
}

export const loop = {
  /**
   * The one way the frame loop is started again after it has slept, and the
   * one place to look for the answer to "what wakes it".
   *
   * Everything that changes the picture, the overlays, or the size of either
   * calls this. In order:
   *
   *   camera input   keydown, keyup, mouse move under the lock, wheel,
   *                  the field-of-view slider, a click-to-travel on the
   *                  fullscreen minimap, losing or taking the pointer
   *   scene state    sun, quality tier, render scale, motion smoothing,
   *                  tone-map gamma, periodic domain, removing the nest
   *   overlays       minimap mode, bird on/off, the stats readout's own mode
   *   geometry       canvas resize and devicePixelRatio change, via the
   *                  observers in _bindInput — the size check inside _frame
   *                  cannot see either while the loop is asleep
   *   captures       track recording and video/still capture, which hold the
   *                  loop awake for their duration
   *
   * Safe to call at any time and from any state; waking an already-awake loop
   * costs one boolean.
   */
  _wake(reason) {
    if (this.stop || this._disposed) return;
    this._marchPending = true;
    if (!this._sleeping) return;
    this._sleeping = false;
    this._wakeReason = reason;
    this._startLoop();
  },

  /**
   * The LOD the LIVE view marches this frame. The flight slider, except over
   * a parked city: there a held view sharpens to CITY_PARKED_LOD_STRENGTH —
   * the quarter-pixel floor the city's high/max fly at — because the city's
   * lower flight tiers trade windows for frame rate and a parked view has no
   * frame to protect. Only when the slider is not hand-set (a hand-set
   * slider is the user's, moving or parked) and the hold mode is not "live"
   * (what you flew on is what you keep). The LOD angles sit in the scene key
   * (u.light_lod_tan, u.view_lod_tan), so the flip restarts accumulation by itself, once,
   * at the moment of parking — which parking does anyway.
   */
  _liveLodStrength() {
    if (this.scene?.city && this.holdMode !== "live"
        && !this._byHand.has("lod")
        && this.renderer && !this.renderer.cameraMoving) {
      return K.CITY_PARKED_LOD_STRENGTH;
    }
    return this.lodStrength;
  },

  /**
   * The per-frame view state.
   *
   * `lodStrength` is overridable because an offline render should not
   * inherit a march chosen to keep flight smooth — captures pass their
   * tier's parked LOD (see _captureLook). Everything else follows the live
   * view; a capture overrides haze and exposure on top, in
   * _captureViewKwargs.
   */
  _viewKwargs({ lodStrength = this._liveLodStrength(),
                tier = this.qualityTier ?? K.DEFAULT_QUALITY_TIER } = {}) {
    return {
      camera: this.camera,
      jitter: true,
      sunAzimuth: this.sunAzimuth,
      sunElevation: this.sunElevation,
      exposure: this.exposure,
      toneMapGamma: this.toneMapGamma,
      toneMapWhitePoint: this.toneMapWhitePoint,
      contrast: this.contrast,
      haze: this.haze,
      hazeHeightDependent: this.hazeHeightDependent,
      lightMarchLodDegrees: K.APP_LIGHT_MARCH_LOD_DEGREES * lodStrength,
      viewStepLodDegrees: K.APP_VIEW_STEP_LOD_DEGREES * lodStrength,
      frameIndex: this.frameIndex,
      sceneTime: this.sceneTime,
      // The roofs' clutter follows the tier (its cityRoofDensity);
      // packed under CITY only.
      cityRoofDensity: cityRoofDensity(tier),
    };
  },

  /**
   * Start the one frame loop, retiring whatever was running before it.
   *
   * The only place a loop is ever started. Everything that wants the picture
   * moving again — the first load, a replacement field, the end of a capture —
   * comes through here, so "how many loops are running" has one answer.
   */
  _startLoop() {
    if (this.stop || this._disposed) return;
    this._stopLoop();
    const generation = this._loopGeneration;
    this._lastTime = performance.now();
    this._sleeping = false;
    this._marchPending = true;
    // Whatever the last marched frame cost, it was measured before a sleep of
    // unknown length at a tier that may since have changed. The hold ladder
    // waits for a fresh one rather than climbing on it.
    this._lastMarchMs = null;
    this._raf = requestAnimationFrame(() => this._frame(generation));
  },

  /** Retire the running loop, including a frame currently mid-await. */
  _stopLoop() {
    this._loopGeneration += 1;
    if (this._raf !== null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  },

  /** The live loop's one way to stop on an error: loudly, and for good. */
  _failRendering(err) {
    this.stop = true;
    this.onFailure?.("Rendering stopped.", String(err.message || err),
                     err.advice || "");
  },

  /** Whether the frame that is asking is still the one that should be running. */
  _loopAlive(generation) {
    return !this.stop && !this._disposed
        && generation === this._loopGeneration;
  },

  async _frame(generation) {
    this._raf = null;
    if (!this._loopAlive(generation)) return;
    // A capture owns the renderer while it runs — the live loop would fight
    // it for the accumulation buffer and neither picture would converge.
    if (this._capturing) {
      this._raf = requestAnimationFrame(() => this._frame(generation));
      return;
    }
    const now = performance.now();
    // Two clocks. `elapsed` is what the frame actually took; `dt` is what the
    // simulation is allowed to believe, clamped so one long stall cannot
    // teleport the camera through the field. Only the camera, the bird and
    // the track recorder get the clamped one — the FPS readout gets the truth,
    // or it saturates at exactly 1/CLAMP and reports the same number at every
    // quality tier no matter how much slower the frame really was.
    const elapsed = (now - this._lastTime) / 1000;
    const dt = Math.min(elapsed, K.MAX_SIM_TIMESTEP);
    this._lastTime = now;
    // How long the previous frame took, kept only when that frame actually
    // marched. This is the rAF clock rather than a queue wait, and that is
    // fine for what it is used for: rAF deltas saturate at the vsync interval
    // and so understate a FAST frame, but a frame that took 300 ms shows up
    // as 300 ms. The hold ladder's gate only ever asks "is this too slow to
    // climb", which is the direction this clock is honest in.
    if (this._prevFrameMarched) this._lastMarchMs = elapsed * 1000;

    // The flight, in the order collide.js lays out: the episode's clock,
    // the player's move, the flyer placed off the camera, and then the
    // whole frame's motion made legal against the city. The flyer is
    // updated here rather than with its overlay below because the collider
    // needs it where it is going to be drawn.
    if (!this.paused) {
      this.collider?.advance(dt, this.camera);
      this.camera.move(dt);
    }
    this.camera.constrain();
    const flyer = this.flyer && this.flyerEnabled ? this.flyer : null;
    flyer?.update(this.paused ? 0 : dt, this.camera);
    if (!this.paused && this.collider) {
      const knock = this.collider.resolve(dt, this.camera, flyer);
      if (knock) this.recorder.noteKnock(knock);
    }

    // "Moving" is an exact comparison against the previous frame's view, not
    // a velocity: it is what lets the hold ladder start climbing the instant
    // you stop, rather than after a timeout.
    const signature = this.camera.signature();
    const cameraMoved =
      this._lastSignature !== null && signature !== this._lastSignature;
    this.renderer.setCameraMoving(cameraMoved);
    this._lastSignature = signature;
    // The clock runs on the simulation's dt (the camera's clock, so a stall
    // cannot jump the steam either). Where the steam is drawn live
    // (Renderer.liveSteamActive: the city at Medium and up) it runs paused
    // or parked too, so the steam keeps going when the flight is paused: the
    // live view draws the steam after the
    // accumulation and keeps the clock out of the scene key, so a parked
    // picture converges under steam that moves. At Low and Minimal the steam
    // is marched and the clock is in the key, so it runs only with the
    // flight or a recording, and a parked picture converges under still
    // steam. A recording samples the clock with the camera, so a replay's
    // steam moves as it did.
    const clockRuns = this.renderer.liveSteamActive
      || (!this.paused && (cameraMoved || this.recorder.recording));
    if (clockRuns) this.sceneTime += dt;
    // Lens water keeps wall-clock time, parked or not: it goes on wetting in
    // cloud and drying out of it while you hover. It is drawn after the
    // accumulation, so a changing lens never disturbs the parked picture's
    // convergence. Capped at a second, which only bites after a hidden tab
    // or a stall (a wake resets the frame clock, and the loop does not
    // sleep until the lens has settled); the law is exact for any step.
    this._lensDt = Math.min(elapsed, LENS_MAX_STEP_S);

    if (this._probe) {
      // The probe measures the FLIGHT configuration, because what decides
      // whether a tier is usable is what it costs while the camera moves.
      // The camera is by definition still at load, so without this the hold
      // ladder would climb underneath the probe and it would measure the
      // converged picture instead.
      this.renderer.setCameraMoving(true);
    } else {
      this.renderer.holdTick(this._lastMarchMs);
    }

    // A few planes of the sun-tau bake per frame, ahead of the march so the
    // frame that completes it is also the first to read it. The reads flip
    // on inside drawFrame the moment the bake finishes, and a parked view's
    // average restarts on the cached method there (drawFrame's methodKey);
    // the explicit march-pending is for a parked view, which otherwise has
    // no reason to march again. Inside the frame's failure path: a bake
    // pipeline that failed to compile throws here, and that must stop the
    // renderer loudly like any other pipeline failure, not end the loop
    // without a word.
    if (this.renderer.lightBakePending && !this._probe) {
      try {
        if (this.renderer.stepLightBake()) this._marchPending = true;
      } catch (err) {
        this._failRendering(err);
        return;
      }
    }

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const outW = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const outH = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== outW || this.canvas.height !== outH) {
      this.canvas.width = outW;
      this.canvas.height = outH;
    }

    // Bird first, minimap second: the bird lives in the scene and takes the
    // scene's depth, the map is chrome laid over the finished picture.
    const overlays = [];
    if (flyer) {
      flyer.writeUniforms(this.camera, [outW, outH], {
        sunAzimuth: this.sunAzimuth, sunElevation: this.sunElevation,
        exposure: this.exposure,
        toneMapGamma: this.toneMapGamma,
        toneMapWhitePoint: this.toneMapWhitePoint,
        contrast: this.contrast,
      });
      overlays.push((enc, view, format) =>
        flyer.encodePass(enc, view, format, [outW, outH]));
    }
    if (this.minimap && this.minimapMode !== "off") {
      this.minimap.update(this.camera, this.scene, [outW, outH],
                          this.minimapMode === "full", this.haze);
      overlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }

    // Marching is the expensive half — a full traversal of the volume for
    // every pixel. Presenting is a blit. Once the view has converged there is
    // nothing left for a march to add, so the loop keeps presenting (the bird
    // still flies over it) and stops marching.
    let march = this._marchPending
      || !this.renderer.converged
      || !this.renderer.canPresentLast;
    const probing = Boolean(this._probe);
    // The probe's first act is to time its own stopwatch — see
    // _chooseProbeClock. Two clocks come out of that:
    //
    //   queue    drain, submit, drain: the wall time between the drains is
    //            this frame's GPU work. Honest wherever waiting on the
    //            queue is a fence, which is measured, not assumed —
    //            Firefox's onSubmittedWorkDone is a ~100 ms poll cadence
    //            that read a 5080's one-millisecond Minimal as a slideshow.
    //   cadence  no waits at all: the rAF delta of marched frames. It
    //            saturates at the vsync interval, so it cannot predict the
    //            next tier — but it can falsify the current one, and the
    //            probe's cadence rule (climb while the beat holds, step
    //            back when it breaks) needs nothing more.
    if (probing && this._probe.clock === null) {
      await this._chooseProbeClock();
      if (!this._loopAlive(generation)) return;
    }
    const queueClock = probing && this._probe?.clock === "queue";
    // Tracks the probe WITHIN this frame: it goes false the moment a queue
    // wait fails, so the rest of the frame stops pretending to time
    // anything. See _awaitQueueIdle, which also demotes the clock.
    let measuring = probing;
    let submittedAt = 0;
    try {
      if (queueClock && measuring) {
        // Start the clock on an idle queue. Without this the measurement
        // includes however much of the PREVIOUS frame was still outstanding
        // when this one was encoded, which on the first probe frames — the
        // ones that matter most, at the floor — is most of what is being
        // timed. Drain, then start counting.
        measuring = await this._awaitQueueIdle();
        submittedAt = performance.now();
      }
      if (march) {
        this._marchPending = false;
        // Deliberately a thunk: see Renderer.drawFrame. The swapchain texture
        // must not be acquired until every await inside the draw has resolved.
        const marched = await this.renderer.drawFrame(
          () => this.context.getCurrentTexture().createView(), this.canvasFormat,
          [outW, outH], this._viewKwargs(),
          { deltaSeconds: dt, overlays, waitForPipeline: false,
            lens: this.lensWater ? { dt: this._lensDt ?? 0.0 } : null,
            liveSteam: true });
        if (!marched) {
          // The shader variant this frame needs is still compiling (a tier,
          // bake or toggle moved the lighting specialization; see
          // Renderer._rayPipeline). Nothing was drawn and the canvas keeps
          // its last picture. Asking again next frame rather than waiting
          // inside the draw is what keeps a compile from holding a frame
          // open while the field or a capture changes underneath it. The
          // frame did not march, so no clock may count it.
          this._marchPending = true;
          march = false;
          measuring = false;
        }
      } else {
        this.renderer.presentLast(
          this.context.getCurrentTexture().createView(), this.canvasFormat,
          [outW, outH], overlays,
          this.lensWater ? { dt: this._lensDt ?? 0.0 } : null,
          { sceneTime: this.sceneTime });
      }
      if (queueClock && measuring) {
        // Close of the window the drain above opened: what sits between the
        // two clocks is this frame's GPU work and nothing else. Serializing
        // the pipeline is what makes the number honest, and is affordable
        // for the dozen-odd frames the probe lasts. (Timestamp queries would
        // avoid the stall, but Safari commonly does not expose the feature.)
        measuring = await this._awaitQueueIdle();
      }
    } catch (err) {
      this._failRendering(err);
      return;
    }
    // The draw is the loop's one long await. A field replaced (or the viewer
    // left) while it was in flight means this frame belongs to a session that
    // no longer exists — and everything below it touches state that release
    // has already torn down.
    if (!this._loopAlive(generation)) return;
    if (measuring && this._probe) {
      this._probeFrame(this._probe.clock === "queue"
        // The queue clock's own round-trip cost rides on every reading;
        // calibration measured it, so take it back off.
        ? Math.max(0, performance.now() - submittedAt - this._probe.overheadMs)
        // This frame's rAF delta describes the PREVIOUS frame — the cadence
        // warm-up count accounts for the lag.
        : elapsed * 1000);
    } else if (!this._probe) {
      // The governor only ever hears about consecutive marched frames flown
      // with a moving camera — the flight configuration, which is what a
      // tier verdict is about. Everything else (a hold climb, a park, a
      // pause) resets its window rather than polluting it.
      if (march && cameraMoved && !this.paused) {
        this._governorFrame(elapsed * 1000);
      } else {
        this._governorReset();
      }
    }

    // Fire-and-forget: the meter renders ~5k rays on its own uniform buffer
    // and folds its answer in whenever the readback lands.
    this._aeTick(now, [outW, outH]);
    // Likewise the dart-tip condensate probe (ice-detection mode).
    this._iceTick(now);

    // Sampled after the frame it describes, so the track records what was on
    // screen rather than what was about to be. The clock is the flight's own
    // accumulated time, not the wall's: see TrackRecorder.
    // A pause is cut out of the track, and so is the steam's time through
    // it: the replay's clock runs on from where it stopped, rather than
    // jumping by the pause in one frame.
    if (this.recorder.recording && this.paused && clockRuns) {
      this._trackPausedS += dt;
    }
    if (this.recorder.recording && !this.paused) {
      this.recorder.advance(dt);
      this.recorder.sample(this.camera, this.sceneTime - this._trackPausedS);
      if (this.recorder.full) this._finishTrackRecording(true);
    }

    this.frameIndex += 1;
    // Both the readout and the hold ladder's cost gate describe MARCHED
    // frames only. Folding presented-only frames in would report the cost of
    // a blit as the cost of the picture, and would tell the ladder it could
    // afford a rung it cannot.
    this._prevFrameMarched = march;
    if (march) {
      this._fpsAcc += elapsed; this._fpsN += 1;
      if (this._fpsAcc >= 0.5) {
        this._fps = this._fpsN / this._fpsAcc;
        this._fpsAcc = 0; this._fpsN = 0;
      }
    }
    const converged = this.renderer.converged;
    this.ui.drawStats({
      fps: this._fps, camera: this.camera,
      tier: this.renderer.qualityTier, renderScale: this.renderer.renderScale,
      frame: this.frameIndex,
      minimap: Boolean(this.minimap && this.minimapMode !== "off"),
      flyer: this.flyerLabel,
      recording: this.recorder.recording,
      showSpeed: performance.now() / 1000 < this.camera.speedFlashUntil,
      parked: !march,
      converged,
      probing,
      autoTier: this._autoTier,
      wakeReason: this._wakeReason,
      spp: this.renderer.accumCount,
      holdRung: this.renderer.holdRung,
      holdRungCount: this.renderer.holdRungCount,
      holdCapped: this.renderer.holdCapped,
      // The camera's own surface frame, not a recompute from the world
      // position — the recompute is what can drift once the two frames fold
      // independently. Scene-gated: day cameras carry a surface frame too
      // (the ocean's), and that one is not a city readout.
      cityPosition: this.scene.city && this.camera.surfacePosition
        ? [...this.camera.surfacePosition, this.camera.position[2]]
        : null,
    });

    // Sleep on a converged view.
    //
    // Nothing here is coupled to pause or to the pointer lock: the test is
    // purely "is the picture finished", and what ends it is purely the input
    // enumerated in _wake. The pause state machine is where input bugs live,
    // and this must not become part of it.
    //
    // The bird is the usual reason a converged view keeps a loop at all — it
    // flies whether or not the camera does — but a blit is not a march, so
    // the GPU still goes quiet. When it is off (or the view is paused, which
    // stops it), nothing is left to draw and the loop stops outright. That
    // matters most on the machines that struggle: a fanless laptop that keeps
    // re-marching a view nobody is moving heats itself into thermal
    // throttling, so a parked picture is not merely wasteful, it makes the
    // NEXT flight slower.
    const animating = this._overlaysAnimate() || this.recorder.recording
      || this._capturing || probing || this.renderer.lightBakePending
      // A lens still wetting or drying is motion too.
      || (this.lensWater && !this.renderer.lensSettled)
      // And the city's live steam, which keeps the live clock, wherever any
      // can be in view.
      || Boolean(this.renderer.liveSteamActive
                 && this.camera.position[2] < K.LIVE_STEAM_CEILING_M);
    if (converged && !this._marchPending && !animating) {
      this._sleeping = true;
      return;                    // no rAF is scheduled; only _wake starts one
    }

    this._raf = requestAnimationFrame(() => this._frame(generation));
  },

  /**
   * Whether anything drawn over the finished picture is still moving.
   *
   * Only the bird is, and only while the view is unpaused — _frame hands it a
   * dt of zero when paused, so a paused bird is a still image. A city crash
   * (collide.js) is also motion nobody is steering: the fall, the bounce and
   * the shake play out over a second whether a flyer is up or not. The minimap
   * changes when the camera or the mode does, and both of those go through
   * _wake, so it does not hold the loop open.
   */
  _overlaysAnimate() {
    return Boolean(this.flyer && this.flyerEnabled && !this.paused)
      || Boolean(this.collider?.active && !this.paused);
  },
};
