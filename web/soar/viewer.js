// The viewer: owns the scene, the frame loop, and the input surface.
//
// This is the browser's answer to FlyThroughApp. It keeps world metres and
// meteorological angles directly and only converts to the relative convention
// at the edges (reproduction commands, metadata).

"use strict";

import * as K from "./constants.js";
import { Renderer } from "./renderer.js";
import { FlightCamera, cameraBasis } from "./camera.js";
import { viewSpansDomainEdge } from "./field.js";
import { loadDemoScene, loadOceanTile } from "./scene.js";
import { Minimap } from "./minimap.js";
import { Bird } from "./bird.js";
import {
  TrackRecorder, trackPayload, resampleTrack, resampledFrameCount,
  MAX_TRACK_SECONDS,
} from "./track.js";
import { UI } from "./ui.js";
import { mod360 } from "./spectral.js";
import { escalateQualityTier } from "./uniforms.js";
import {
  renderStill, imageDataToPng, download, timestampedName,
  createOfflineTarget, beginOfflineRender, endOfflineRender, renderAccumulated,
  readBack,
} from "./capture.js";
import { cameraWorldOrigin, worldToRelative } from "./camera.js";

const OCEAN_URL = "ocean";

/**
 * The canvas must not present through an sRGB format. raymarch.wgsl's tone
 * map already gamma-encodes, and an sRGB swapchain encodes again — which is
 * exactly the bug that made the desktop window render at an effective gamma
 * near 3.1 for years without anyone naming it. Gamma happens once, in the
 * shader, where the slider can reach it.
 */
export function presentFormat(preferred) {
  return preferred.endsWith("-srgb")
    ? preferred.slice(0, -"-srgb".length)
    : preferred;
}

class Viewer {
  constructor(device, canvas, uiRoot) {
    this.device = device;
    this.canvas = canvas;
    this.uiRoot = uiRoot;

    this.sunAzimuth = K.DEFAULT_SUN_AZIMUTH;
    this.sunElevation = K.DEFAULT_SUN_ELEVATION;
    this.toneMapGamma = K.DEFAULT_TONE_MAP_GAMMA;
    this.toneMapWhitePoint = K.DEFAULT_TONE_MAP_WHITE_POINT;
    this.contrast = K.DEFAULT_CONTRAST;
    this.haze = K.DEFAULT_HAZE;
    this.beholdQuality = K.DEFAULT_BEHOLD_QUALITY;
    this.beholdField = "outer";   // "outer" or "nest": behold renders one
    // witness is the default because it reproduces this exact view — nests,
    // wrap and image settings — where behold path-traces one bare field.
    this.terminalRenderer = "witness";
    this.captureSize = null;
    this.stillSamples = K.STILL_ACCUMULATE_FRAMES;
    this.videoFps = K.DEFAULT_VIDEO_FPS;
    this.videoAccumulate = K.DEFAULT_VIDEO_ACCUMULATE;

    this.birdEnabled = true;
    // "corner" | "full" | "off". Fullscreen frees the mouse so the map can
    // be clicked to travel; M cycles through all three.
    this.minimapMode = "corner";
    this.recorder = new TrackRecorder();
    this.paused = true;
    this.captured = false;
    // Sparse bricks are a manual switch, never inferred from the field: no
    // occupancy threshold, no memory heuristic, nothing that could choose
    // differently on two machines and make a measurement unrepeatable.
    this.brickedRequested = false;
    this.brickShape = [8, 8, 8];
    this._source = null;
    this._keepCameraOnNextLoad = false;
    this.frameIndex = 0;
    this._fpsAcc = 0; this._fpsN = 0; this._fps = null;
    this._lastSignature = null;
    this._discardNextPointerMove = false;

    // Quality is measured, not assumed. `qualityTier` is the answer for this
    // session — set by the startup probe, or by the user in the Quality
    // panel, which also switches `_autoTier` off for good: having chosen once
    // by hand, they should not have the choice taken back on the next field.
    this.qualityTier = null;
    this._autoTier = true;
    this._probe = null;
    // One-shot second opinion on a low auto-tier verdict — see
    // _settleAutoTier and AUTO_TIER_CONFIRM_FROM.
    this._confirmTimer = null;
    this._confirmed = false;

    // The loop sleeps on a converged view (see _frame) and only these three
    // fields say so. `_sleeping` means no rAF is pending and only _wake can
    // start one; `_marchPending` means the next frame must re-march whatever
    // the accumulation state thinks; `_lastMarchMs` is what the last marched
    // frame cost, which is what gates the hold ladder's next rung.
    this._sleeping = false;
    this._marchPending = true;
    this._lastMarchMs = null;
    this._prevFrameMarched = false;
    this._wakeReason = null;
    this._resizeObserver = null;
    this._loadNotes = null;
    this._loadNotesUntil = 0;

    // The frame loop is identified by a generation, not by a boolean.
    //
    // A frame is async: it can be sitting in an await when the field is
    // replaced. Suppressing it with a flag that a later load clears again
    // revives it, and since every live frame schedules its own successor the
    // result is two permanent loops sharing one renderer — which a hidden tab
    // makes deterministic rather than merely possible, because the suppressed
    // callback cannot run and retire itself while the load is in flight.
    // Bumping the generation orphans the old frame for good.
    this._loopGeneration = 0;
    this._raf = null;
    // `stop` is fatal and one-way: a lost device or a failed draw. Nothing
    // clears it, so no later load can resurrect a loop on a dead device.
    this.stop = false;
    this._disposed = false;
    this._listeners = new AbortController();
  }

  get sunZenith() { return 90.0 - this.sunElevation; }

  get sunCompass() {
    const a = mod360(this.sunAzimuth);
    for (const [edge, label] of K.COMPASS_EDGES) if (a < edge) return label;
    return "N";
  }

  get isFullscreen() { return Boolean(document.fullscreenElement); }

  /** Whether soar is still choosing the quality tier by measurement. */
  get autoTier() { return this._autoTier; }

  get sourceLabel() {
    return this.scene.title
      ? `${this.scene.title} — ${this.scene.sourceName}`
      : (this.scene.sourceName ?? "cloud field");
  }

  // --- setup ---------------------------------------------------------------

  async start(source, progress) {
    this.shaderSource = await (await fetch("raymarch.wgsl")).text();
    this.hudSource = await (await fetch("hud.wgsl")).text();
    this.birdSource = await (await fetch("bird.wgsl")).text();
    this.progress = progress;

    // The UI is built before the field loads, because loading a file asks
    // questions — which group, what units — and those are menu panels.
    this.ui = new UI(this.uiRoot, this);
    this.ui.statsMode = "subtle";

    this.context = this.canvas.getContext("webgpu");
    this.canvasFormat = presentFormat(navigator.gpu.getPreferredCanvasFormat());
    this.context.configure({
      device: this.device, format: this.canvasFormat, alphaMode: "opaque",
    });

    await this.loadField(source, progress);
    this._bindInput();
    this.paused = false;
    this._startLoop();
  }

  /**
   * Release everything the current field owns, once the GPU has finished with
   * it. Destroying a texture that submitted commands still reference is legal
   * by the letter of the spec and segfaults browsers in practice, so the
   * barrier is not optional.
   */
  async _releaseField() {
    this._stopLoop();
    if (!this.scene) return;
    try {
      await this.device.queue.onSubmittedWorkDone();
    } catch {
      // Same shape as dispose(): a device already lost has no queue to
      // drain, and that is precisely the case where the destroys below
      // matter least and must not throw.
    }
    this.renderer?.destroy();
    this.minimap?.destroy();
    this.bird?.destroy();
    this.scene.destroy();
    this.scene = null;
    this.renderer = null;
    this.minimap = null;
    this.bird = null;
    this.frameIndex = 0;
    this._lastSignature = null;
    this.sunAzimuth = K.DEFAULT_SUN_AZIMUTH;
    this.sunElevation = K.DEFAULT_SUN_ELEVATION;
  }

  /**
   * Load (or replace) the field. Everything resident on the GPU for the old
   * one is released first — a second field would otherwise sit alongside the
   * first, and these are gigabytes.
   *
   * The new field is built into locals and only committed once all of it
   * exists. A shader that fails to compile half-way through would otherwise
   * leave a volume texture with no owner and no reference: gigabytes that
   * only a garbage collector knows about, released whenever it feels like it.
   */
  async loadField(source, progress) {
    await this._releaseField();
    // Remembered so the brick switch can rebuild the same field the other
    // way round without asking the user to open it again.
    this._source = source;

    let scene = null, renderer = null, minimap = null, bird = null;
    let sun = null, minimapProblem = null, birdProblem = null;
    try {
      // The ocean is a patch of sea surface, not anything about the data, so
      // it survives a change of field — and belongs to the viewer, which is
      // what disposes of it.
      const ocean = async () => (this._ocean ??=
        await loadOceanTile(this.device, OCEAN_URL));

      if (source.kind === "demo") {
        scene = await loadDemoScene(
          this.device, source.base, ocean,
          (stage, fraction) => progress(stage, fraction),
          { bricked: this.brickedRequested, brick: this.brickShape });
        sun = scene.sun ?? null;
      } else {
        const { loadFileScene } = await import("./ingest/index.js");
        scene = await loadFileScene(
          this.device, source.file, {
            ocean,
            progress,
            ask: (question) => this._ask(question),
          });
      }

      renderer = new Renderer(this.device, this.shaderSource, scene,
                              { canvasFormat: this.canvasFormat });
      if (scene.periodicDefault === false) renderer.setPeriodic(false);
      // Choose the tier BEFORE init(), because init() is what compiles and
      // validates the shader, and the shader is specialized on the tier's
      // maxLightSteps. Setting it afterwards would compile a module for a
      // tier no frame is ever drawn at, and would make the probe's first
      // frame pay for a second compile.
      renderer.setQualityTier(this._startingQualityTier());
      progress("Compiling the shader…", 0.97);
      await renderer.init();

      // The map and the bird are overlays, not the picture. A GPU that cannot
      // hold one (or a field too wide for a 2D texture) is a reason to fly
      // without it and say so, not a reason to fail the load. Said now rather
      // than only when the menu is next opened.
      const map = new Minimap(this.device, {
        albedo: scene.albedo, shape: scene.albedoShape,
      });
      try {
        minimap = await map.init(this.canvasFormat, this.hudSource);
      } catch (err) {
        map.destroy();
        minimap = null;
        minimapProblem = String(err?.message || err);
      }
      const flyer = new Bird(this.device, {
        volumeView: scene.volumeView, sampler: renderer.volSampler,
        bmin: scene.bmin, bmax: scene.bmax,
      });
      try {
        await flyer.init(this.canvasFormat, this.birdSource);
        bird = flyer;
      } catch (err) {
        flyer.destroy();
        bird = null;
        birdProblem = String(err?.message || err);
      }
    } catch (err) {
      bird?.destroy();
      minimap?.destroy();
      renderer?.destroy();
      scene?.destroy();
      throw err;
    }

    this.scene = scene;
    this.renderer = renderer;
    this.minimap = minimap;
    this.bird = bird;
    this._minimapProblem = minimapProblem;
    this._birdProblem = birdProblem;
    if (sun) {
      this.sunAzimuth = sun.azimuth;
      this.sunElevation = sun.elevation;
    }
    // A brick rebuild is the SAME field seen the same way, so the camera
    // survives it — otherwise the switch would move the view and the A/B it
    // exists to serve would compare two different pictures.
    const keep = this._keepCameraOnNextLoad ? this.camera : null;
    this._keepCameraOnNextLoad = false;
    this.camera = new FlightCamera(scene.bmin, scene.bmax,
                                   { periodic: renderer.periodic });
    if (keep) {
      this.camera.position = keep.position;
      this.camera.azimuth = keep.azimuth;
      this.camera.elevation = keep.elevation;
      this.camera.speed = keep.speed;
    }

    this.ui.setSubtitle(this.sourceLabel);

    // Everything the load has to admit to, in ONE toast. say() replaces
    // whatever is on screen rather than queueing, so four calls in a row
    // would show the fourth and quietly lose the other three — which is the
    // same silence these messages exist to break.
    const notes = [];
    if (scene.nestNote) notes.push(scene.nestNote);
    if (minimapProblem) notes.push(`Flying without the minimap: ${minimapProblem}`);
    if (birdProblem) notes.push(`Flying without the bird: ${birdProblem}`);
    if (scene.skipped?.length) {
      notes.push(
        `${scene.skipped.length} group(s) in this file could not be read, ` +
        `and were not offered:\n${scene.skipped.join("\n")}`);
    }
    // Kept as well as said, because the auto-quality result lands a second or
    // so later and say() replaces rather than queues; _announceAutoTier
    // repeats these alongside it rather than wiping them.
    if (notes.length) {
      const seconds = 6 + 3 * notes.length;
      this._loadNotes = notes.join("\n\n");
      this._loadNotesUntil = performance.now() + seconds * 1000;
      this.ui.say(this._loadNotes, seconds);
    } else {
      this._loadNotes = null;
      this._loadNotesUntil = 0;
    }

    this._beginAutoTier();
  }

  // --- choosing a quality tier by measurement ------------------------------

  /**
   * The tier the first frame of a newly loaded field is drawn at.
   *
   * When the probe is going to run this is the FLOOR, and that is a safety
   * property rather than a preference. The deployed demo field is
   * 1024x1024x206; a high-tier frame of it at Retina resolution is most of a
   * second of fragment work that nothing can preempt, which on a Mac freezes
   * the whole machine (the compositor shares the GPU) and can lose the device
   * to Metal's command-buffer watchdog. The old default of "high, and never
   * adapt" is exactly that frame, submitted before anything has been
   * measured. So: start at the floor, and never render a tier that a
   * completed measurement below it has not shown to be affordable.
   */
  _startingQualityTier() {
    if (!this._autoTier) return this.qualityTier ?? K.DEFAULT_QUALITY_TIER;
    return K.QUALITY_TIERS_CHEAPEST_FIRST[0];
  }

  /** Arm the probe for a freshly loaded field, unless the user opted out. */
  _beginAutoTier() {
    clearTimeout(this._confirmTimer);
    this._confirmTimer = null;
    this._confirmed = false;
    if (!this._autoTier) { this._probe = null; return; }
    this._armProbe(K.QUALITY_TIERS_CHEAPEST_FIRST[0]);
  }

  _armProbe(tier) {
    this._probe = {
      kind: "tier",
      tier, frame: 0, samples: [],
      // Which stopwatch to believe — decided by _chooseProbeClock at the
      // first probe frame, by measuring the stopwatch itself. "queue" is the
      // drain-submit-drain GPU clock; "cadence" is rAF wall time.
      clock: null,
      overheadMs: 0,
      // The tier climbed FROM into the current one, so the cadence clock can
      // step back a rung when the tier it climbed to breaks the beat.
      climbedFrom: null,
      // Median measured at each tier that finished sampling, so settling on
      // a stepped-back tier reports that tier's own number.
      medians: {},
    };
  }

  /**
   * Decide which clock the probe may believe, by timing the clock itself.
   *
   * See the AUTO_TIER_CLOCK_* block in constants.js for the whole argument.
   * The short of it: onSubmittedWorkDone is only a GPU fence on some
   * browsers. On Firefox it resolves on an internal poll cadence — ~100 ms
   * on a machine whose actual frame is one or two — and a probe that
   * believed it concluded a 5080 could not afford Minimal (then "potato"). So the round-trip
   * is measured here on an EMPTY queue: anything it costs is the clock's
   * own overhead. The minimum of a few rounds, because the first may still
   * be draining real work from the load.
   *
   * `?probeclock=queue|cadence` in the URL forces the choice, for testing
   * either path on a browser whose calibration would pick the other.
   */
  async _chooseProbeClock() {
    const probe = this._probe;
    const forced = new URLSearchParams(location.search).get("probeclock");
    if (forced === "queue" || forced === "cadence") {
      probe.clock = forced; this._probeClockUsed = probe.clock;
      console.info(`soar: probe clock forced to '${forced}' by ?probeclock=`);
      return;
    }
    // One calibration per session. The verdict is a property of the browser
    // and device, not of the field, so a hot-swapped file reuses it — which
    // also keeps queue waits (a crash lottery on Firefox, see bug 14) off
    // the reload path.
    if (this._probeClockVerdict) {
      probe.clock = this._probeClockVerdict.clock; this._probeClockUsed = probe.clock;
      probe.overheadMs = this._probeClockVerdict.overheadMs;
      return;
    }
    const remember = () => {
      this._probeClockVerdict = {
        clock: probe.clock, overheadMs: probe.overheadMs,
      };
    };
    const rounds = [];
    try {
      for (let i = 0; i < K.AUTO_TIER_CLOCK_CALIBRATION_ROUNDS; i++) {
        const t0 = performance.now();
        await this.device.queue.onSubmittedWorkDone();
        rounds.push(performance.now() - t0);
      }
    } catch (err) {
      // The queue clock does not even run here (headless Chrome rejects it
      // outright for swapchain frames). The cadence clock owes it nothing.
      probe.clock = "cadence"; this._probeClockUsed = probe.clock;
      remember();
      console.info(
        `soar: onSubmittedWorkDone rejects on this browser ` +
        `(${err?.message || err}); probing by frame cadence.`);
      return;
    }
    const overhead = Math.min(...rounds);
    if (overhead <= K.AUTO_TIER_CLOCK_OVERHEAD_MAX_MS) {
      probe.clock = "queue"; this._probeClockUsed = probe.clock;
      probe.overheadMs = overhead;
      remember();
    } else {
      probe.clock = "cadence"; this._probeClockUsed = probe.clock;
      remember();
      console.info(
        `soar: waiting on an EMPTY queue costs ${overhead.toFixed(1)} ms ` +
        `here, so onSubmittedWorkDone is a poll cadence, not a fence — ` +
        `probing by frame cadence instead.`);
    }
  }

  /**
   * Account for one probe frame. `ms` is wall time across a frame that was
   * followed by device.queue.onSubmittedWorkDone(), so it is that frame's GPU
   * time — rAF deltas cannot be used here, because they saturate at the
   * vsync interval and are pipelined besides.
   *
   * Strictly one rung at a time: the escalation below happens only after a
   * full measurement has come back, and the next frame is the first one drawn
   * at the new tier. No frame is ever in flight at a tier whose affordability
   * is still being established.
   */
  _probeFrame(ms) {
    const probe = this._probe;
    if (!probe) return;
    probe.frame += 1;
    const warmup = probe.clock === "cadence"
      ? K.AUTO_TIER_WARMUP_FRAMES_CADENCE
      : K.AUTO_TIER_WARMUP_FRAMES;
    if (probe.frame <= warmup) return;
    // A benchmark borrows the probe's timing path rather than starting a
    // second one, because everything hard about timing a frame here is
    // already solved in it: which stopwatch this browser can be believed
    // about (onSubmittedWorkDone is a ~100 ms poll on Firefox, not a fence),
    // what that stopwatch costs on an empty queue, and how many frames to
    // throw away first. It also holds the renderer in its FLIGHT
    // configuration, which is what stops the hold ladder from converging the
    // view out from under a stationary measurement and timing a blit.
    if (probe.kind === "bench") {
      probe.samples.push(ms);
      if (performance.now() < probe.until) return;
      this._probe = null;
      probe.resolve(probe.samples);
      return;
    }
    probe.samples.push(ms);
    if (probe.samples.length < K.AUTO_TIER_SAMPLE_FRAMES) return;

    // The median, not the mean and not the minimum. A mean is dragged by any
    // one stall — a GC pause, the compositor taking the GPU — and every such
    // contamination is upward, so the mean of three is biased high. The
    // minimum over-corrects the other way: on a machine whose frame times are
    // genuinely spread, the best of three is a frame time it will rarely hit
    // again, and escalating on it is exactly the wrong call on exactly the
    // hardware this protects. The middle sample is neither.
    const measured = [...probe.samples].sort((a, b) => a - b)[
      Math.floor(probe.samples.length / 2)];
    probe.medians[probe.tier] = measured;

    const climb = (next) => {
      probe.climbedFrom = probe.tier;
      probe.tier = next;
      probe.frame = 0;
      probe.samples = [];
      this.renderer.setQualityTier(next);
    };

    if (probe.clock === "cadence") {
      // The cadence clock cannot predict the next tier (it saturates at the
      // vsync interval), only falsify the current one: frames arriving on
      // the beat mean this tier is affordable, frames off it mean it is not.
      // So: climb while the beat holds, and when the tier climbed TO breaks
      // it, step back to the one that held. See AUTO_TIER_CADENCE_HOLD_MS.
      const order = K.QUALITY_TIERS_CHEAPEST_FIRST;
      const at = order.indexOf(probe.tier);
      const holds = measured <= K.AUTO_TIER_CADENCE_HOLD_MS;
      if (holds && at < order.length - 1) {
        climb(order[at + 1]);
        return;
      }
      let tier = probe.tier;
      if (!holds && probe.climbedFrom) {
        tier = probe.climbedFrom;
        this.renderer.setQualityTier(tier);
      }
      this._probe = null;
      this._settleAutoTier(tier, probe.medians[tier] ?? measured, "cadence");
      return;
    }

    const next = escalateQualityTier(probe.tier, measured);
    if (next) {
      climb(next);
      return;
    }
    this._probe = null;
    this._settleAutoTier(probe.tier, measured, "queue");
  }

  /**
   * Wait for the GPU queue to go idle, for timing. True if it did.
   *
   * This is the probe's clock and it is allowed to fail. `onSubmittedWorkDone`
   * is core WebGPU, but it is not universally honoured: headless Chrome
   * rejects it with "A valid external Instance reference no longer exists"
   * for every frame that touched the canvas swapchain — reproducible in
   * twenty lines with no soar involved — and a driver that behaves like that
   * in the field is entirely possible.
   *
   * What must NOT happen is what happened before this existed: the rejection
   * reaching the frame's own catch, which treats a throw as a lost device and
   * stops rendering for good. A stopwatch that breaks is not a renderer that
   * broke. So the failure is caught here and ends the probe instead — and
   * ending the probe means staying at the tier last PROVEN affordable, which
   * is the floor unless a completed measurement already justified more. No
   * measurement, no escalation: that is the same invariant the whole probe
   * runs on, arriving at the same answer from the other direction.
   */
  async _awaitQueueIdle() {
    try {
      await this.device.queue.onSubmittedWorkDone();
      return true;
    } catch (err) {
      // The queue clock died mid-probe. That is a broken stopwatch, not a
      // broken renderer — and the probe still has its other clock. Demote to
      // cadence, restart the current tier's samples (they were taken with a
      // clock that just proved untrustworthy), and carry on climbing.
      const probe = this._probe;
      if (probe) {
        probe.clock = "cadence"; this._probeClockUsed = probe.clock;
        probe.frame = 0;
        probe.samples = [];
        console.warn(
          `soar: the queue clock failed mid-probe ` +
          `(${err?.message || err}); continuing by frame cadence.`);
      }
      return false;
    }
  }

  /** Record the probe's verdict and say it out loud. */
  _settleAutoTier(tier, measuredMs, clock) {
    this.qualityTier = tier;
    // A low verdict gets one silent second opinion before it is announced.
    // Probe noise is one-sided: load turbulence — the volume upload's tail,
    // shader warmup, GC, the compositor — can only make a machine look
    // SLOWER than it is, never faster. So a verdict at or above
    // AUTO_TIER_CONFIRM_FROM proves itself, while a 5080 that measured as
    // Minimal (seen ~10% of loads) deserves re-measuring once the
    // turbulence has passed. The retry walks upward from the tier already
    // proven — same rung-at-a-time invariant, so it can only raise the
    // answer, and it never runs when the user has picked a tier by hand.
    const order = K.QUALITY_TIERS_CHEAPEST_FIRST;
    if (order.indexOf(tier) < order.indexOf(K.AUTO_TIER_CONFIRM_FROM)
        && !this._confirmed) {
      this._confirmed = true;
      console.info(
        `soar: auto quality ${tier} measured ${measuredMs.toFixed(2)} ms ` +
        `(${clock} clock) — low verdict, re-confirming in ` +
        `${K.AUTO_TIER_CONFIRM_DELAY_MS} ms`);
      this._confirmTimer = setTimeout(() => {
        this._confirmTimer = null;
        if (!this._autoTier || this._probe || this._disposed) return;
        this._armProbe(this.qualityTier);
        this._wake("auto-tier confirm");
      }, K.AUTO_TIER_CONFIRM_DELAY_MS);
      return;
    }
    const label = K.QUALITY_PRESETS[tier].label.split(" —")[0];
    console.info(
      `soar: auto quality ${tier} — ${measuredMs.toFixed(2)} ms/frame probed ` +
      `at ${this.canvas.width}x${this.canvas.height} (${clock} clock)`);

    const floor = K.QUALITY_TIERS_CHEAPEST_FIRST[0];
    // No silent misery: when even the floor is over budget there is nothing
    // to fall back to, so say what the machine is doing instead of letting
    // the user conclude the app is broken. The parked view is still worth
    // having on such a GPU, which is worth saying too.
    const line = (tier === floor && measuredMs > K.AUTO_TIER_FLOOR_WARN_MS)
      ? `This GPU renders ${measuredMs.toFixed(0)} ms a frame even at ` +
        `${label} — the lowest quality soar has. Expect a slideshow while ` +
        "you fly. Stop moving and the picture still sharpens and converges."
      : `Auto quality: ${label} — change any time in the menu.`;

    if (this._loadNotes && performance.now() < this._loadNotesUntil) {
      this.ui.say(`${this._loadNotes}\n\n${line}`, 8);
    } else {
      this.ui.say(line, tier === floor ? 8 : 5);
    }
  }

  /**
   * Put a question from the loader on screen and wait for the answer.
   *
   * Nothing here is guessed. Which group holds the field, which two nest, and
   * what the condensate units are when the file does not say — a wrong guess
   * on the last one is off by a factor of a thousand and still looks like a
   * cloud.
   */
  _ask(question) {
    return new Promise((resolve, reject) => {
      this.setLoadingVisible?.(false);
      const done = (value) => {
        this.ui.close();
        this.setLoadingVisible?.(true);
        resolve(value);
      };
      const cancel = () => {
        this.ui.close();
        // `cancelled` is what the catch sites test for — a deliberate Back
        // is not a load failure and must not be reported as one (bug 11).
        // A marker property, not a message string: wording changes.
        const err = new Error("Cancelled before the field was loaded.");
        err.cancelled = true;
        reject(err);
      };
      if (question.panel === "groups") {
        this.ui.open("groups", {
          groups: question.groups, pairs: question.pairs,
          filename: question.filename,
          onPick: (group) => done({ group }),
          onPickPair: (pair) => done({ pair }),
          onCancel: cancel,
        });
      } else if (question.panel === "units") {
        this.ui.open("units", {
          variables: question.variables, filename: question.filename,
          onPick: (units) => done({ units }),
          onCancel: cancel,
        });
      } else {
        reject(new Error(`unknown question '${question.panel}'`));
      }
    });
  }

  // --- waking the loop -----------------------------------------------------

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
  }

  // --- input ---------------------------------------------------------------

  _bindInput() {
    const canvas = this.canvas;
    // Every listener below outlives its own statement, and half of them are on
    // `document`, which outlives the viewer. Registered against one signal so
    // dispose() can take them all off in a line — otherwise a viewer that has
    // been left behind is still reachable from the document, and so is its
    // scene, its renderer, and its device.
    const { signal } = this._listeners;

    canvas.addEventListener("click", (e) => {
      if (this.ui.isOpen) return;
      if (this.minimapMode === "full" && !this.captured && this.minimap) {
        // The canvas backbuffer and its CSS box differ by the render scale;
        // the map rect lives in backbuffer pixels.
        const hit = this.minimap.worldXYFromPixel(
          e.offsetX * (canvas.width / canvas.clientWidth),
          e.offsetY * (canvas.height / canvas.clientHeight),
          this.scene);
        if (hit) {
          this.camera.position[0] = hit[0];
          this.camera.position[1] = hit[1];
          this.camera.constrain();
          this._wake("minimap travel");
          return;
        }
        // Clicking off the map is done with it: back to the corner, flying.
        this.setMinimapMode("corner");
        return;
      }
      if (!this.captured) this._requestCapture();
    }, { signal });

    document.addEventListener("pointerlockchange", () => {
      this._wake("pointer lock");
      const wasCaptured = this.captured;
      this.captured = document.pointerLockElement === canvas;
      // The jump to the window centre arrives as one enormous movement event.
      // Swallowing it is the difference between taking the mouse and having
      // the view snap somewhere random.
      this._discardNextPointerMove = true;
      this._syncChrome();
      if (this.captured) {
        // Holding the pointer IS flying — enforced here as an invariant
        // rather than trusted to every path that can ask for the lock.
        // Without it, anything that leaves "paused, menu closed, mouse free"
        // (a screenshot closes the menu with no resume — bug 2's sibling;
        // app switching can drop and re-grant the lock) lets the next click
        // capture the pointer while paused: the view still pans, because
        // look() is deliberately not gated on pause, but WASD, the bird's
        // clock and the recorder are all frozen. Observed live 2026-08-11.
        if (this.paused) {
          this.ui.close();
          this.paused = false;
          this._lastTime = performance.now();
          this._syncChrome();
        }
        return;
      }

      this.camera.keys.clear();
      // Escape releases the pointer lock in the browser itself, and Firefox
      // ALSO delivers the keydown. Handling both turned one press into
      // pause-then-resume — the menu appeared and vanished in a frame. So the
      // lock loss is the single source of truth for pausing, and the keydown
      // that caused it is ignored for a moment afterwards.
      this._lockLostAt = performance.now();
      if (wasCaptured && !this._tabRelease && !this.paused) this.pause();
      this._tabRelease = false;
    }, { signal });

    // Firefox refuses a re-lock for about a second after the user pressed
    // Escape, so resuming cannot rely on it. Say what to do instead.
    document.addEventListener("pointerlockerror", () => {
      if (!this.paused) this.ui.say("Click the view to take the mouse.", 3);
    }, { signal });

    document.addEventListener("mousemove", (e) => {
      if (!this.captured) return;
      if (this._discardNextPointerMove) {
        this._discardNextPointerMove = false;
        return;
      }
      this.camera.look(e.movementX, e.movementY);
      this._wake("mouse look");
    }, { signal });

    document.addEventListener("wheel", (e) => {
      if (!this.captured) return;
      this.camera.scrollSpeed(e.deltaY, performance.now() / 1000);
      this._wake("wheel");
    }, { passive: true, signal });

    document.addEventListener("keydown", (e) => this._onKeyDown(e), { signal });
    document.addEventListener("keyup", (e) => {
      this.camera.keys.delete(e.key.toLowerCase());
      this._wake("keyup");
    }, { signal });

    window.addEventListener("blur", () => {
      this.camera.keys.clear();
      this._wake("blur");
    }, { signal });

    // The canvas's size check lives inside _frame, which does not run while
    // the loop is asleep — so the loop has to be woken from outside it. One
    // observer for the CSS box (window resize, entering fullscreen, the
    // sidebar of a devtools pane opening) and one media query for the device
    // pixel ratio (dragging the window to a display with different scaling,
    // or changing the browser's zoom), which fires no resize at all.
    this._resizeObserver = new ResizeObserver(() => this._wake("resize"));
    this._resizeObserver.observe(this.canvas);
    this._watchDevicePixelRatio(signal);
  }

  /**
   * There is no devicePixelRatio event. The idiom is a media query pinned to
   * the current value, which stops matching the moment it changes; it is
   * one-shot, so it re-arms itself against the new value.
   */
  _watchDevicePixelRatio(signal) {
    const arm = () => {
      if (this._disposed) return;
      const query = window.matchMedia(
        `(resolution: ${window.devicePixelRatio}dppx)`);
      query.addEventListener("change", () => {
        this._wake("devicePixelRatio");
        arm();
      }, { once: true, signal });
    };
    arm();
  }

  /**
   * Keys are only for things done in flight. Everything reachable from the
   * menu is reachable by clicking it, so nothing here needs a second binding.
   */
  _onKeyDown(e) {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    // Before the switch, and unconditionally: every key here either moves the
    // camera or changes what is drawn, and the ones that do neither cost one
    // frame to find that out.
    this._wake("keydown");

    if (key === "Escape") {
      // Escape reaches us by three routes that can all fire for one physical
      // press: the browser releasing the pointer lock, the keydown itself,
      // and auto-repeat if the key is held a moment. Any two of them in
      // sequence read as close-then-open, which is the menu "popping right
      // back up". So: no repeats, and one Escape action per 350 ms whatever
      // the source.
      // A video render owns the GPU for minutes. Escape is the way out of it,
      // and must not also open the menu on top of the progress bar.
      if (this._capturing) { this._videoAbort = true; return; }
      const now = performance.now();
      if (e.repeat) return;
      if (now - (this._lockLostAt ?? -1e9) < 400) return;
      if (now - (this._lastEscapeAt ?? -1e9) < 350) return;
      this._lastEscapeAt = now;
      if (this.ui.isOpen) this.ui.back(false);
      else this.pause();
      return;
    }
    if (this.ui.isOpen) return;

    switch (key) {
      case "Tab":
        e.preventDefault();
        if (this.captured) {
          this._tabRelease = true;
          document.exitPointerLock();
        } else {
          this._requestCapture();
        }
        return;
      case "f": this.toggleFullscreen(); return;
      case "F3": e.preventDefault(); this.ui.cycleStats(); return;
      case "b": this.toggleBird(); return;
      case "m": this.toggleMinimap(); return;
      case "r": this.toggleTrackRecording(); return;
      // Through pause(), not ui.open() directly: the dialog needs the mouse,
      // so the pointer must be released first — opening it under a held lock
      // left the camera turning behind an unusable panel (bug 1).
      case "F12": e.preventDefault(); this.pause("capture"); return;
      default: break;
    }
    if ("wasdc ".includes(key)) this.camera.keys.add(key);
    if (key === "Shift") this.camera.keys.add("shift");
  }

  // --- state changes -------------------------------------------------------

  /**
   * Keep the on-screen chrome consistent with what the mouse is doing.
   *
   * The toolbar is for when the mouse is free but the menu is closed — after
   * Tab, or before the first click. While flying there is nothing to click,
   * and while the menu is open the menu has everything.
   */
  _syncChrome() {
    const viewer = this.canvas.parentElement;
    viewer.classList.toggle("captured", this.captured);
    viewer.classList.toggle("menu-open", Boolean(this.ui?.isOpen));
  }

  pause(panel = "main") {
    this.paused = true;
    this.camera.keys.clear();
    if (this.captured) document.exitPointerLock();
    this.ui.open(panel);
    this._syncChrome();
    // Pausing does not put the loop to sleep and resuming does not wake it:
    // sleep is decided purely by whether the picture is finished. This wake
    // is only here because pausing stops the bird, which changes what the
    // next frame has to draw.
    this._wake("pause");
  }

  /**
   * Ask for the pointer once, quietly. Chromium returns a promise that
   * rejects inside the post-Escape cooldown; the pointerlockerror listener
   * already tells the user what to do, so the rejection itself is noise.
   */
  _requestCapture() {
    const p = this.canvas.requestPointerLock?.();
    if (p?.catch) p.catch(() => {});
  }

  resume({ capture = true } = {}) {
    this.ui.close();
    this.paused = false;
    // A fullscreen map under a re-taken pointer is a screen you can't leave.
    if (this.minimapMode === "full") this.minimapMode = "corner";
    this._lastTime = performance.now();
    this._syncChrome();
    if (capture) this._requestCapture();
    else this.ui.say("Click the view to take the mouse.", 2);
    this._wake("resume");
  }

  setSun({ azimuth, elevation, zenith }) {
    if (azimuth != null) this.sunAzimuth = mod360(azimuth);
    if (zenith != null) elevation = 90.0 - zenith;
    if (elevation != null) {
      this.sunElevation = Math.min(
        90.0, Math.max(K.MIN_SUN_ELEVATION_DEG, elevation));
    }
    this._wake("sun");
  }

  setToneMapGamma(gamma) {
    const [lo, hi] = K.TONE_MAP_GAMMA_LIMITS;
    if (!(gamma >= lo && gamma <= hi)) {
      throw new Error(`tone_map_gamma must be in [${lo}, ${hi}]; got ${gamma}.`);
    }
    this.toneMapGamma = gamma;
    this.renderer.resetAccumulation();
    this._wake("tone-map gamma");
  }

  setToneMapWhitePoint(whitePoint) {
    const [lo, hi] = K.TONE_MAP_WHITE_POINT_LIMITS;
    if (!(whitePoint >= lo && whitePoint <= hi)) {
      throw new Error(
        `tone_map_white_point must be in [${lo}, ${hi}]; got ${whitePoint}.`);
    }
    this.toneMapWhitePoint = whitePoint;
    this.renderer.resetAccumulation();
    this._wake("tone-map white point");
  }

  setContrast(contrast) {
    const [lo, hi] = K.CONTRAST_LIMITS;
    if (!(contrast >= lo && contrast <= hi)) {
      throw new Error(`contrast must be in [${lo}, ${hi}]; got ${contrast}.`);
    }
    this.contrast = contrast;
    this.renderer.resetAccumulation();
    this._wake("contrast");
  }

  /**
   * How much aerosol is in the air, 0 to 1. Unlike the gamma above this is
   * not a display choice — it changes what the march computes, so the frames
   * already averaged are of a different sky and cannot be kept.
   */
  setHaze(haze) {
    if (!(haze >= 0.0 && haze <= K.HAZE_MAX)) {
      throw new Error(`haze must be in [0, ${K.HAZE_MAX}]; got ${haze}.`);
    }
    this.haze = haze;
    this.renderer.resetAccumulation();
    this._wake("haze");
  }

  /**
   * The Quality panel's tier buttons. Choosing by hand ends the automatic
   * choice for the session — including on the next field loaded, because
   * having the app overrule a deliberate choice a minute later is worse than
   * carrying a stale one.
   */
  setQualityTier(tier) {
    this._autoTier = false;
    this._probe = null;
    clearTimeout(this._confirmTimer);
    this._confirmTimer = null;
    this.qualityTier = tier;
    this.renderer.setQualityTier(tier);
    this._wake("quality tier");
  }

  /** The Quality panel's render-scale slider: the flight scale, by hand. */
  setRenderScale(scale) {
    this.renderer.setRenderScale(scale);
    this._wake("render scale");
  }

  /** The Quality panel's motion-smoothing slider. */
  setMotionBlendAlpha(alpha) {
    this.renderer.motionBlendAlpha = alpha;
    this._wake("motion smoothing");
  }

  /** The field-of-view slider, which is a camera change like any other. */
  setFov(fov) {
    this.camera.setFov(fov);
    this._wake("field of view");
  }

  /**
   * Store the field as one dense volume, or as sparse bricks.
   *
   * A manual switch and nothing else. There is no gate that looks at the
   * field and decides for you: the two prior attempts at empty-space skipping
   * were reverted on measurements taken on one machine, and the whole reason
   * this is a switch is so the same view can be timed both ways on YOURS.
   *
   * Bricking is decided when the field is built, so this rebuilds it — and
   * carries the camera across, because comparing one view two ways is the
   * point and a moved camera would compare two pictures instead.
   */
  async toggleBricks() {
    if (this._brickSwitchBusy || !this._source) return;
    if (this._source.kind !== "demo") {
      this.ui.say("Bricks are only built for the bundled fields so far. An "
                  + "opened file still loads dense — the ingest path has to "
                  + "learn to build them as it reads.");
      return;
    }
    this._brickSwitchBusy = true;
    this.brickedRequested = !this.brickedRequested;
    this._keepCameraOnNextLoad = true;
    try {
      await this.loadField(this._source, this.progress);
      const stats = this.scene?.brickStats;
      if (stats) {
        const share = 100 * stats.occupiedBricks / stats.totalBricks;
        const saving = stats.denseTexels / Math.max(stats.atlasTexels, 1);
        this.ui.say(
          `Bricks on — ${stats.occupiedBricks.toLocaleString()} of ` +
          `${stats.totalBricks.toLocaleString()} bricks hold cloud ` +
          `(${share.toFixed(1)}%), atlas ${saving.toFixed(1)}x smaller than ` +
          "the dense volume.");
      } else {
        this.ui.say("Bricks off — one dense volume.");
      }
    } catch (err) {
      // The rebuild failed, so the field is gone rather than merely
      // unbricked. Say which, and put the flag back so a retry is the other
      // way round rather than the same way again.
      this.brickedRequested = !this.brickedRequested;
      this.ui.say(`Could not rebuild the field: ${err.message}`);
      throw err;
    } finally {
      this._brickSwitchBusy = false;
      // _releaseField stopped the loop and nothing else restarts it.
      this._startLoop();
    }
  }

  /** Brick extent in voxels. Rebuilds the field when bricks are on. */
  async setBrickShape(axis, value) {
    const next = [...this.brickShape];
    next["xyz".indexOf(axis)] = Math.max(1, Math.round(value));
    if (next.join() === this.brickShape.join()) return;
    this.brickShape = next;
    if (!this.brickedRequested || this._brickSwitchBusy) return;
    this._brickSwitchBusy = true;
    this._keepCameraOnNextLoad = true;
    try {
      await this.loadField(this._source, this.progress);
      const s = this.scene?.brickStats;
      if (s) {
        this.ui.say(
          `Bricks ${this.brickShape.join("x")} — ` +
          `${(100 * s.occupiedBricks / s.totalBricks).toFixed(1)}% occupied, ` +
          `atlas ${(s.denseTexels / Math.max(s.atlasTexels, 1)).toFixed(1)}x ` +
          "smaller than dense.");
      }
    } finally {
      this._brickSwitchBusy = false;
      this._startLoop();
    }
  }

  /**
   * Time the view in front of you, for `seconds`, as it is right now.
   *
   * Resolves to the per-frame milliseconds the probe path measured. The
   * camera must not move while this runs — it is timing one picture, and a
   * hand on the keyboard makes the number describe a different one every
   * frame.
   */
  _timeCurrentView(seconds) {
    return new Promise((resolve) => {
      let done = false;
      const finish = (samples) => {
        if (done) return;
        done = true;
        clearTimeout(watchdog);
        this._probe = null;
        resolve(samples);
      };
      // A measurement that waits on frames must not wait forever. If the loop
      // stops for any reason the window did not anticipate — a lost device, a
      // backgrounded tab throttling rAF to nothing — this ends the run with
      // whatever it collected instead of leaving the app looking hung, which
      // is exactly what it looked like the first time.
      const watchdog = setTimeout(() => {
        console.warn(
          `soar: benchmark window of ${seconds}s ended by watchdog after ` +
          `${this._probe?.samples.length ?? 0} frames — the loop stopped ` +
          "producing them.");
        finish(this._probe?.samples ?? []);
      }, (seconds + 5) * 1000);
      this._probe = {
        kind: "bench", frame: 0, samples: [], clock: null, overheadMs: 0,
        until: performance.now() + seconds * 1000,
        resolve: finish,
      };
      // _startLoop, not _wake. _releaseField stops the loop by bumping the
      // generation, but it does not set _sleeping — so after a field rebuild
      // there is no loop running AND _wake believes there is, returns early,
      // and starts nothing. The first run of a benchmark only ever worked
      // because the auto-tier probe happened to start one during that load;
      // by the second run auto-tier had settled and nothing did, so the
      // window collected zero frames and the watchdog ended it.
      this._startLoop();
    });
  }

  /**
   * Time this exact view with bricks off and with bricks on, and report both.
   *
   * The two runs are separated by a rebuild of the field, which is the only
   * honest way to compare them: bricking is decided at build time. The camera
   * is carried across, so the two numbers describe one picture rather than
   * two. Nothing is inferred and nothing is chosen — the answer goes to you,
   * because the answer is known to differ by machine and the two prior
   * attempts at this were killed by a measurement taken on one card.
   */
  async benchmarkBricks({ seconds = 5 } = {}) {
    if (this._brickSwitchBusy || !this._source) return null;
    if (this._source.kind !== "demo") {
      this.ui.say("Bricks are only built for the bundled fields so far.");
      return null;
    }
    const wasBricked = this.brickedRequested;
    this._brickSwitchBusy = true;
    const runs = [];
    try {
      for (const bricked of [false, true]) {
        this.brickedRequested = bricked;
        this._keepCameraOnNextLoad = true;
        // Timed and reported, but deliberately OUTSIDE the window below.
        // Building an atlas is a one-off cost paid when the field is opened;
        // folding it into a per-frame number would be a category error. It is
        // still worth knowing, because on a field this decides nothing else
        // does — a minute to build is a fact about whether you would ever
        // switch this on, whatever the frame time says.
        const builtAt = performance.now();
        await this.loadField(this._source, this.progress);
        const buildMs = performance.now() - builtAt;
        // A rebuilt field starts on rung 0 — the flight configuration, which
        // at Minimal is an eighth of the pixels. That is not the picture you
        // were looking at when you asked for the measurement, and it would
        // also put the two runs on different rungs. Both are pinned to the
        // top, which is where a still view ends up anyway.
        this.renderer.pinTopRung();
        this.ui.say(
          `Measuring ${bricked ? `bricks ${this.brickShape.join("x")}`
                               : "dense"} at ` +
          `${(this.renderer.renderScale * 100).toFixed(0)}% scale… ` +
          `${seconds}s, hands off.`,
          seconds + 1);
        const samples = await this._timeCurrentView(seconds);
        // The median, for the reason the tier probe uses one: a mean is
        // dragged by any single stall — a GC pause, the compositor taking
        // the GPU — and every such contamination is upward.
        const sorted = [...samples].sort((a, b) => a - b);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        runs.push({
          bricked,
          brick: bricked ? [...this.brickShape] : null,
          frames: sorted.length,
          medianMs: sorted[Math.floor(sorted.length / 2)] ?? NaN,
          meanMs: samples.reduce((a, b) => a + b, 0) / (samples.length || 1),
          stats: this.scene?.brickStats ?? null,
          buildMs,
          // Recorded per run, not read afterwards: the finally block rebuilds
          // the field, and a renderer read after that describes the restored
          // one rather than the measured one.
          tier: this.renderer.qualityTier,
          scale: this.renderer.renderScale,
          rung: `${this.renderer.holdRung + 1}/${this.renderer.holdRungCount}`,
          pixels: [
            Math.round(this.canvas.clientWidth * dpr * this.renderer.renderScale),
            Math.round(this.canvas.clientHeight * dpr * this.renderer.renderScale),
          ],
          clock: this._probeClockUsed ?? "?",
        });
      }
    } finally {
      this.brickedRequested = wasBricked;
      this._keepCameraOnNextLoad = true;
      await this.loadField(this._source, this.progress).catch(() => {});
      this._brickSwitchBusy = false;
      // Nothing else is going to. loadField builds a field; it does not start
      // the loop — the caller does, and at startup that caller is init(). Left
      // out, the app sits on the last frame of the last measurement looking
      // exactly like it has hung, which is what it did.
      this._startLoop();
    }
    if (runs.length === 2) this._reportBrickBenchmark(runs, seconds);
    return runs;
  }

  _reportBrickBenchmark(runs, seconds) {
    const [dense, brick] = runs;
    const ratio = brick.medianMs / Math.max(dense.medianMs, 1e-9);
    const fps = (ms) => (ms > 0 ? 1000 / ms : 0);
    const s = brick.stats;
    // The full table to the console, where it can be selected and pasted
    // into the results file; the verdict on screen, where it is read.
    const lines = [
      `soar brick benchmark — ${this.sourceLabel}`,
      `  tier ${dense.tier}, hold rung ${dense.rung}, render scale ` +
      `${dense.scale} (${dense.pixels[0]}x${dense.pixels[1]} marched), ` +
      `${seconds}s per run, ${dense.clock} clock`,
      `  camera ${[...this.camera.position].map((v) => v.toFixed(0)).join(", ")} ` +
      `az ${this.camera.azimuth.toFixed(1)} el ${this.camera.elevation.toFixed(1)}`,
      `  dense           median ${dense.medianMs.toFixed(3)} ms ` +
      `(${fps(dense.medianMs).toFixed(0)} fps), mean ` +
      `${dense.meanMs.toFixed(3)}, ${dense.frames} frames`,
      `  bricks ${brick.brick.join("x").padEnd(9)} median ` +
      `${brick.medianMs.toFixed(3)} ms (${fps(brick.medianMs).toFixed(0)} fps)` +
      `, mean ${brick.meanMs.toFixed(3)}, ${brick.frames} frames`,
      `  bricked is ${ratio.toFixed(2)}x ` +
      `${ratio >= 1 ? "SLOWER" : "faster"} than dense`,
      // One-off, and NOT part of the per-frame numbers above. Reported
      // because a field that takes a minute to brick is one you would never
      // switch this on for, whatever its frame time turns out to be.
      `  field build (one-off, not in the frame times above): dense ` +
      `${(dense.buildMs / 1000).toFixed(1)}s, bricked ` +
      `${(brick.buildMs / 1000).toFixed(1)}s`,
    ];
    if (s) {
      const saving = s.denseTexels / Math.max(s.atlasTexels, 1);
      lines.push(
        `  ${s.occupiedBricks} of ${s.totalBricks} bricks occupied ` +
        `(${(100 * s.occupiedBricks / s.totalBricks).toFixed(2)}%), atlas ` +
        `${saving.toFixed(2)}x ${saving >= 1 ? "smaller" : "LARGER"} than dense`);
      if (saving < 1) {
        lines.push(
          "  NOTE: the atlas is bigger than the dense volume. At this " +
          "occupancy the aprons cost more than the empty bricks save, so " +
          "there is nothing for bricking to win here on memory either — try " +
          "a larger brick, or a sparser field.");
      }
    }
    console.log(lines.join("\n"));
    this.ui.say(
      `${brick.brick.join("x")}: dense ${dense.medianMs.toFixed(2)} ms vs ` +
      `bricked ${brick.medianMs.toFixed(2)} ms — ${ratio.toFixed(2)}x ` +
      `${ratio >= 1 ? "slower" : "faster"}. Full table in the console.`,
      12);
  }

  togglePeriodic() {
    this.renderer.setPeriodic(!this.renderer.periodic);
    this.camera.periodic = this.renderer.periodic;
    this.camera.constrain();
    this.ui.say(`Periodic domain ${this.renderer.periodic ? "on" : "off"}.`);
    this._wake("periodic");
  }

  toggleMinimap() {
    if (!this.minimap) {
      this.ui.say(
        this._minimapProblem
          ? `No minimap for this field: ${this._minimapProblem}`
          : "There is no minimap for this field.", 5);
      return;
    }
    const order = ["corner", "full", "off"];
    this.setMinimapMode(order[(order.indexOf(this.minimapMode) + 1) % 3]);
  }

  setMinimapMode(mode) {
    const wasFull = this.minimapMode === "full";
    this.minimapMode = mode;
    this._wake("minimap mode");
    if (mode === "full") {
      // The map needs a visible cursor to be clickable.
      if (this.captured) {
        this._tabRelease = true;
        document.exitPointerLock();
      }
      this.ui.say("Minimap fullscreen — click to travel, M to dismiss.", 4);
    } else {
      this.ui.say(`Minimap ${mode === "corner" ? "on" : "off"}.`);
      if (wasFull && !this.paused && !this.ui.isOpen && !this.captured) {
        this._requestCapture();
      }
    }
  }

  toggleBird() {
    if (!this.bird) {
      this.ui.say(
        this._birdProblem
          ? `No bird for this field: ${this._birdProblem}`
          : "There is no bird for this field.", 5);
      return;
    }
    this.birdEnabled = !this.birdEnabled;
    this.ui.say(`Bird ${this.birdEnabled ? "on" : "off"}.`);
    // Both directions matter: turning it on gives the loop something to
    // animate again, turning it off lets a converged view sleep outright.
    this._wake("bird");
  }

  toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  }

  removeNest() {
    if (this.scene.removeNest()) {
      this.renderer.refreshBindGroup();
      this.renderer.resetAccumulation();
      this.frameIndex = 0;
      this.ui.say("Nested field removed.");
      this._wake("nest removed");
    }
    this.ui.open("main");
  }

  /**
   * Give everything back: the frame loop, the listeners, the GPU.
   *
   * Idempotent and safe to call from any state, because every path out of the
   * viewer ends here — leaving by the menu, backing out of a failure, and a
   * boot that threw half-built. Ordered deliberately: the loop stops before
   * anything is destroyed, and the queue is drained before any texture is,
   * because a frame submitted a millisecond ago may still be reading the
   * volume this is about to free.
   */
  async dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this.stop = true;
    this._stopLoop();
    clearTimeout(this._confirmTimer);
    this._confirmTimer = null;
    this._videoAbort = true;
    this._listeners.abort();
    // Not covered by the abort signal: ResizeObserver has no signal option,
    // and it holds the canvas, which holds this viewer.
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    if (this.captured) document.exitPointerLock();

    try {
      await this.device.queue.onSubmittedWorkDone();
    } catch {
      // A device already lost has no queue to drain, and that is precisely
      // the case where the destroys below matter least and must not throw.
    }
    this.renderer?.destroy();
    this.minimap?.destroy();
    this.bird?.destroy();
    this.scene?.destroy();
    this._ocean?.texture?.destroy();
    this.renderer = null;
    this.minimap = null;
    this.bird = null;
    this.scene = null;
    this._ocean = null;

    // Unconfigure before destroy: the swapchain holds textures on this device,
    // and leaving it configured against a destroyed device is the state
    // Firefox's own error message calls "Queue[Id] does not exist".
    this.context?.unconfigure();
    this.device.destroy();
  }

  async leave() {
    await this.dispose();
    location.reload();
  }

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
    // A recording holds the loop awake outright — a track's samples are its
    // timeline, and a sleeping loop would record a pause as nothing at all.
    this._wake("track recording");
    this.ui.say(
      "Recording the flight path. R again to stop; it stops itself after " +
      `${Math.round(MAX_TRACK_SECONDS / 60)} minutes of flying.`, 4);
  }

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
    this.ui.open("track", { samples });
  }

  /** How many video frames a track becomes at the chosen rate. */
  trackFrameCount(samples) {
    return resampledFrameCount(samples, this.videoFps);
  }

  /**
   * Fly a recorded track again, offline, and encode it.
   *
   * Every output frame is rendered with the accumulation converged — the
   * picture the live view only reaches when you stop moving — and stamped
   * with the timestamp the track says it has, not the one the clock says.
   * That is the whole argument for WebCodecs over screen capture: a frame
   * may take six seconds to converge and the video is still 30 seconds long.
   */
  async renderTrackVideo(samples) {
    if (this._capturing) return;
    const { VideoWriter, evenSize } = await import("./video.js");

    // Even before the offscreen texture is made, not after: H.264 refuses odd
    // dimensions, and every frame has to be the size the encoder was told.
    const size = evenSize(this.captureDimensions());
    const frames = resampleTrack(samples, this.videoFps,
                                 { periodic: this.renderer.periodic });

    this._capturing = true;
    this._wake("capture");
    this._videoAbort = false;
    this.ui.close();
    this.ui.showProgress("Choosing a codec…", 0);

    let writer = null, target = null, saved = null;
    try {
      writer = new VideoWriter({
        width: size[0], height: size[1], fps: this.videoFps,
      });
      const chosen = await writer.init();
      for (const warning of writer.warnings ?? []) console.warn(warning);

      target = createOfflineTarget(this.device, size, "soar-video");
      saved = beginOfflineRender(this.renderer);

      const t0 = performance.now();
      for (let i = 0; i < frames.length; i++) {
        if (this._videoAbort) throw new Error("Cancelled.");
        const pose = {
          position: cameraWorldOrigin(
            frames[i].position, this.scene.bmin, this.scene.bmax),
          azimuth: frames[i].azimuth,
          elevation: frames[i].elevation,
          fov: frames[i].fov,
        };
        const overlays = this._offlineOverlays(pose, size, 1.0 / this.videoFps);
        await renderAccumulated(
          this.renderer, target.view, size,
          { ...this._viewKwargs(), camera: pose, frameIndex: i * 1024 },
          this.videoAccumulate, overlays);
        await writer.addFrame(
          await readBack(this.device, target.texture, size[0], size[1]), i);

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
      // The renderer is gone when the session was disposed mid-render — the
      // abort flag is what woke this loop up. There is then no state left to
      // restore, and reaching for it would throw inside a finally.
      if (saved && this.renderer) endOfflineRender(this.renderer, saved);
      this.ui.hideProgress();
      this._capturing = false;
      // The capture owned the renderer's quality state and its accumulation;
      // whatever the live view was converging towards is gone, and the loop
      // may have been asleep when the capture began.
      this._wake("capture finished");
      this._videoAbort = false;
      this._lastTime = performance.now();
    }
  }

  /** The overlays an offline frame draws, laid out for that frame's size. */
  _offlineOverlays(pose, size, dt) {
    const overlays = [];
    if (this.bird && this.birdEnabled) {
      this.bird.update(dt, pose);
      this.bird.writeUniforms(pose, size, {
        sunAzimuth: this.sunAzimuth, sunElevation: this.sunElevation,
        toneMapGamma: this.toneMapGamma,
        toneMapWhitePoint: this.toneMapWhitePoint,
        contrast: this.contrast,
      });
      overlays.push((enc, view, format) =>
        this.bird.encodePass(enc, view, format, size));
    }
    if (this.minimap && this.minimapMode !== "off") {
      this.minimap.update({
        ...pose,
        relativePosition: () => worldToRelative(
          pose.position, this.scene.bmin, this.scene.bmax),
      }, this.scene, size);
      overlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }
    return overlays;
  }

  /** The track as a file, readable by cloudyview's own `render_track`. */
  downloadTrack(samples) {
    const size = this.captureDimensions();
    const payload = trackPayload(this.renderMetadata(size), samples);
    download(new Blob([JSON.stringify(payload)], { type: "application/json" }),
             timestampedName("cloudyview_track", ".json"));
    this.ui.say(`Saved ${samples.length} samples.`, 3);
  }

  /** The capture resolution: an explicit choice, or whatever the window is. */
  captureDimensions() {
    return this.captureSize ?? [this.canvas.width, this.canvas.height];
  }

  /** What made this picture, so the picture can be made again. */
  renderMetadata(size) {
    const rel = this.camera.relativePosition();
    return {
      schema: "cloudyview.render.v1",
      source: {
        path: this.scene.sourceName ?? null,
        ice_path: this.scene.iceSourceName ?? null,
        liquid_var: this.scene.liquidVar ?? null,
        ice_var: this.scene.iceVar ?? null,
      },
      camera: {
        position: rel,
        azimuth: this.camera.azimuth,
        elevation: this.camera.elevation,
        fov: this.camera.fov,
      },
      sun: { azimuth: this.sunAzimuth, elevation: this.sunElevation },
      render: {
        renderer: "soar-web",
        size,
        tier: this.renderer.qualityTier,
        render_scale: 1.0,
        step_factor: this.renderer.stepFactor,
        max_light_steps: this.renderer.maxLightSteps,
        periodic: this.renderer.periodic,
        accumulate_frames: K.STILL_ACCUMULATE_FRAMES,
        tone_map_gamma: this.toneMapGamma,
        tone_map_white_point: this.toneMapWhitePoint,
        contrast: this.contrast,
        haze: this.haze,
        light_march_lod_degrees: K.APP_LIGHT_MARCH_LOD_DEGREES,
        view_step_lod_degrees: K.APP_VIEW_STEP_LOD_DEGREES,
      },
      timestamp: new Date().toISOString(),
      // witness, not behold: the still was rendered by the witness/soar
      // renderer, so this is the command that reproduces THIS image —
      // nests, wrap and image controls included.
      reproduction_command: this.witnessCommand(),
    };
  }

  async saveScreenshot({ overlays = true } = {}) {
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
    if (overlays && this.bird && this.birdEnabled) {
      this.bird.writeUniforms(this.camera, size, {
        sunAzimuth: this.sunAzimuth, sunElevation: this.sunElevation,
        toneMapGamma: this.toneMapGamma,
        toneMapWhitePoint: this.toneMapWhitePoint,
        contrast: this.contrast,
      });
      stillOverlays.push((enc, view, format) =>
        this.bird.encodePass(enc, view, format, size));
    }
    if (overlays && this.minimap && this.minimapMode !== "off") {
      this.minimap.update(this.camera, this.scene, size);
      stillOverlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }
    try {
      const image = await renderStill(
        this.device, this.renderer, this._viewKwargs(), size,
        this.stillSamples, stillOverlays);
      this.ui.showProgress("Encoding…", 0.95);
      const blob = await imageDataToPng(image, this.renderMetadata(size));
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
  }

  /** Open a netCDF file and fly it, releasing the current one. */
  async pickFile() {
    this.ui.close();
    let file = null;
    try {
      if (window.showOpenFilePicker) {
        const [handle] = await window.showOpenFilePicker({
          types: [{ description: "netCDF cloud field",
                    accept: { "application/x-netcdf": [".nc", ".nc4", ".cdf"],
                              "application/x-hdf5": [".h5", ".hdf5"] } }],
        });
        file = await handle.getFile();
      } else {
        file = await new Promise((resolve) => {
          const input = document.createElement("input");
          input.type = "file";
          input.accept = ".nc,.nc4,.cdf,.h5,.hdf5";
          input.addEventListener("change",
            () => resolve(input.files?.[0] ?? null), { once: true });
          input.addEventListener("cancel", () => resolve(null), { once: true });
          input.click();
        });
      }
    } catch {
      file = null;   // the picker was dismissed
    }
    if (!file) {
      // Dismissing the picker with no field loaded happens one way: a Back
      // from the group/units question already released the field, and then
      // the re-offered picker was dismissed too. There is nothing to resume
      // into, so this is a full exit to the start page.
      if (!this.scene) { this.leave(); return; }
      this.paused ? this.ui.open("main") : this.resume();
      return;
    }

    this.paused = true;
    this.setLoadingVisible?.(true);
    try {
      await this.loadField({ kind: "file", file }, this.progress);
      this.setLoadingVisible?.(false);
      this.paused = false;
      this._startLoop();
    } catch (err) {
      // Back from the group/units question returns to choosing a file — the
      // usual reason is the wrong file, and the picker is where that gets
      // fixed (bug 11). The field is already released by then, so the
      // no-file branch above handles a second dismissal.
      if (err?.cancelled) { this.setLoadingVisible?.(false); return this.pickFile(); }
      this.onFailure?.("Could not open this field.",
                       String(err?.message || err), err?.advice || "");
    }
  }

  /** True when the view sees wrapped copies — behold's volume does not tile. */
  viewSpansDomainEdge() {
    return viewSpansDomainEdge(
      this.camera.position,
      cameraBasis(this.camera.azimuth, this.camera.elevation),
      this.camera.fov,
      this.canvas.width / this.canvas.height,
      this.scene.bmin, this.scene.bmax);
  }

  /** Group name of the nested level, for the menu and the behold command. */
  get nestName() {
    return this.scene?.nestGroup ?? null;
  }

  /**
   * Which of the loaded fields the behold command names, and the box its
   * coordinates are measured in.
   *
   * behold renders one field from one group, so a nested scene has to be
   * asked about; the outer field is the answer until it is. The box travels
   * with the group because the relative position means "this far across THIS
   * field" — quoting the outer domain's fraction at a nest a fortieth of its
   * width puts the camera kilometres from where it was framed.
   */
  beholdTarget() {
    if (this.scene?.nested && this.beholdField === "nest") {
      return {
        group: this.scene.nestGroup,
        bmin: this.scene.nestBmin,
        bmax: this.scene.nestBmax,
      };
    }
    return {
      group: this.scene?.groupPath ?? null,
      bmin: this.scene?.bmin,
      bmax: this.scene?.bmax,
    };
  }

  beholdGroup() {
    return this.beholdTarget().group;
  }

  beholdCommand() {
    const { group, bmin, bmax } = this.beholdTarget();
    const rel = worldToRelative(this.camera.position, bmin, bmax);
    const n = (v) => Number(v).toPrecision(12).replace(/\.?0+$/, "");
    // A browser never learns where the file it was handed lives, so the
    // path here is a name to be completed in the terminal, not a path.
    const source = this.scene.sourceName ?? "<your-file.nc>";
    return [
      "behold", source, this.beholdQuality, "--gpu",
      ...(group ? ["--group", group] : []),
      "--camera-position", n(rel[0]), n(rel[1]), n(rel[2]),
      "--camera-azimuth", n(this.camera.azimuth),
      "--camera-elevation", n(this.camera.elevation),
      "--fov", n(this.camera.fov),
      "--sun-azimuth", n(this.sunAzimuth),
      "--sun-elevation", n(this.sunElevation),
    ].join(" ");
  }

  /**
   * Unlike behold, witness drives the exact WGSL this view was drawn with,
   * so its command reproduces the view completely: both nested levels, the
   * periodic wrap, and the image controls — which are display state a
   * default render would not carry. The image controls are always written
   * out rather than only when non-default, because "exactly this view" must
   * survive a change of defaults.
   */
  witnessCommand() {
    const n = (v) => Number(v).toPrecision(12).replace(/\.?0+$/, "");
    const source = this.scene.sourceName ?? "<your-file.nc>";
    const rel = worldToRelative(
      this.camera.position, this.scene.bmin, this.scene.bmax);
    return [
      "witness", source,
      "--size", this.canvas.width, this.canvas.height,
      ...(this.scene.groupPath ? ["--group", this.scene.groupPath] : []),
      ...(this.scene.nested && this.scene.nestGroup
          ? ["--nest-group", this.scene.nestGroup] : []),
      "--camera-position", n(rel[0]), n(rel[1]), n(rel[2]),
      "--camera-azimuth", n(this.camera.azimuth),
      "--camera-elevation", n(this.camera.elevation),
      "--fov", n(this.camera.fov),
      "--sun-azimuth", n(this.sunAzimuth),
      "--sun-elevation", n(this.sunElevation),
      ...(this.renderer.periodic ? ["--periodic"] : []),
      "--gamma", n(this.toneMapGamma),
      "--white-point", n(this.toneMapWhitePoint),
      "--contrast", n(this.contrast),
      "--haze", n(this.haze),
    ].join(" ");
  }

  // --- the loop ------------------------------------------------------------

  _viewKwargs() {
    return {
      camera: this.camera,
      jitter: true,
      sunAzimuth: this.sunAzimuth,
      sunElevation: this.sunElevation,
      toneMapGamma: this.toneMapGamma,
      toneMapWhitePoint: this.toneMapWhitePoint,
      contrast: this.contrast,
      haze: this.haze,
      lightMarchLodDegrees: K.APP_LIGHT_MARCH_LOD_DEGREES,
      viewStepLodDegrees: K.APP_VIEW_STEP_LOD_DEGREES,
      frameIndex: this.frameIndex,
    };
  }

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
  }

  /** Retire the running loop, including a frame currently mid-await. */
  _stopLoop() {
    this._loopGeneration += 1;
    if (this._raf !== null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  /** Whether the frame that is asking is still the one that should be running. */
  _loopAlive(generation) {
    return !this.stop && !this._disposed
        && generation === this._loopGeneration;
  }

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

    if (!this.paused) this.camera.move(dt);
    this.camera.constrain();

    // "Moving" is an exact comparison against the previous frame's view, not
    // a velocity: it is what lets the hold ladder start climbing the instant
    // you stop, rather than after a timeout.
    const signature = this.camera.signature();
    this.renderer.setCameraMoving(
      this._lastSignature !== null && signature !== this._lastSignature);
    this._lastSignature = signature;

    if (this._probe?.kind === "bench") {
      // A benchmark is the mirror image of the tier probe. The probe wants
      // the flight configuration, because what decides whether a tier is
      // usable is what it costs while you are moving. A benchmark wants the
      // configuration you are LOOKING at — you ran it from a paused menu, in
      // front of a view that has long since climbed to full render scale —
      // so the ladder is left exactly where the pinning at bench start put
      // it, and holdTick is not called, so nothing moves it mid-measurement.
      //
      // Marching is forced instead: a still view converges within a couple of
      // seconds and then stops marching, so a five-second window that did not
      // do this would time a blit for most of its length and report a frame
      // rate that belongs to the compositor.
      this._marchPending = true;
    } else if (this._probe) {
      // The probe measures the FLIGHT configuration, because what decides
      // whether a tier is usable is what it costs while the camera moves.
      // The camera is by definition still at load, so without this the hold
      // ladder would climb underneath the probe and it would measure the
      // converged picture instead.
      this.renderer.setCameraMoving(true);
    } else {
      this.renderer.holdTick(this._lastMarchMs);
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
    if (this.bird && this.birdEnabled) {
      this.bird.update(this.paused ? 0 : dt, this.camera);
      this.bird.writeUniforms(this.camera, [outW, outH], {
        sunAzimuth: this.sunAzimuth, sunElevation: this.sunElevation,
        toneMapGamma: this.toneMapGamma,
        toneMapWhitePoint: this.toneMapWhitePoint,
        contrast: this.contrast,
      });
      overlays.push((enc, view, format) =>
        this.bird.encodePass(enc, view, format, [outW, outH]));
    }
    if (this.minimap && this.minimapMode !== "off") {
      this.minimap.update(this.camera, this.scene, [outW, outH],
                          this.minimapMode === "full");
      overlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }

    // Marching is the expensive half — a full traversal of the volume for
    // every pixel. Presenting is a blit. Once the view has converged there is
    // nothing left for a march to add, so the loop keeps presenting (the bird
    // still flies over it) and stops marching.
    const march = this._marchPending
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
        await this.renderer.drawFrame(
          () => this.context.getCurrentTexture().createView(), this.canvasFormat,
          [outW, outH], this._viewKwargs(),
          { deltaSeconds: dt, overlays });
      } else {
        this.renderer.presentLast(
          this.context.getCurrentTexture().createView(), this.canvasFormat,
          [outW, outH], overlays);
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
      this.stop = true;
      this.onFailure?.("Rendering stopped.", String(err.message || err),
                       err.advice || "");
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
    }

    // Sampled after the frame it describes, so the track records what was on
    // screen rather than what was about to be. The clock is the flight's own
    // accumulated time, not the wall's: see TrackRecorder.
    if (this.recorder.recording && !this.paused) {
      this.recorder.advance(dt);
      this.recorder.sample(this.camera);
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
      bird: Boolean(this.bird && this.birdEnabled),
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
    });

    // Sleep on a converged view.
    //
    // Nothing here is coupled to pause or to the pointer lock: the test is
    // purely "is the picture finished", and what ends it is purely the input
    // enumerated in _wake. Bugs 1-2 in docs/soar-bugs.md live in the pause
    // state machine and this must not become a third.
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
      || this._capturing || probing;
    if (converged && !this._marchPending && !animating) {
      this._sleeping = true;
      return;                    // no rAF is scheduled; only _wake starts one
    }

    this._raf = requestAnimationFrame(() => this._frame(generation));
  }

  /**
   * Whether anything drawn over the finished picture is still moving.
   *
   * Only the bird is, and only while the view is unpaused — _frame hands it a
   * dt of zero when paused, so a paused bird is a still image. The minimap
   * changes when the camera or the mode does, and both of those go through
   * _wake, so it does not hold the loop open.
   */
  _overlaysAnimate() {
    return Boolean(this.bird && this.birdEnabled && !this.paused);
  }
}

export async function boot({ device, source, progress, onReady, onFailure,
                             setLoadingVisible, register }) {
  const canvas = document.getElementById("view");
  const uiRoot = document.getElementById("ui");
  uiRoot.replaceChildren();
  const viewer = new Viewer(device, canvas, uiRoot);
  viewer.onFailure = onFailure;
  viewer.setLoadingVisible = setLoadingVisible;
  // Handed over before loading starts, not after: the device can be lost
  // during the very allocation that loading performs, and whoever is
  // watching for that needs to be able to stop this viewer.
  register?.(viewer);
  try {
    await viewer.start(source, progress);
  } catch (err) {
    // A boot that threw half-way is still holding whatever it managed to
    // build — most of a field, a device, a configured swapchain. The caller
    // is about to show a failure panel, not to keep this viewer.
    await viewer.dispose();
    throw err;
  }
  onReady?.();
  return viewer;
}
