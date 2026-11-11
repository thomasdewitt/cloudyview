// The viewer: owns the scene, the frame loop, and the input surface.
//
// It keeps world metres and meteorological angles directly and only converts
// to the relative convention at the edges (reproduction commands, metadata).
//
// This file holds the class, its constructor (every field the viewer has)
// and start(). The methods are in viewer/, one file per concern:
//   load.js       loading a field and releasing it, the file picker, dispose
//   loop.js       the frame loop: _wake, _frame, sleeping
//   input.js      keys, mouse, pointer lock, pause and resume
//   autotier.js   choosing the quality tier: the startup probe, the governor
//   settings.js   sun, tone map, haze, LOD, tier and the other setters
//   exposure.js   exposure and the auto-exposure loop
//   ice.js        ice-detection mode
//   overlays.js   the minimap's mode and which flyer is up
//   capturing.js  stills, track recording, track video
//   commands.js   render metadata and the witness/behold commands

"use strict";

import * as K from "./constants.js";
import { loadShaders } from "./shader.js";
import { TrackRecorder } from "./track.js";
import { UI } from "./ui.js";
import { mod360 } from "./spectral.js";
import { load } from "./viewer/load.js";
import { autotier } from "./viewer/autotier.js";
import { loop } from "./viewer/loop.js";
import { input } from "./viewer/input.js";
import { readLensWater, settings } from "./viewer/settings.js";
import { exposure } from "./viewer/exposure.js";
import { ice } from "./viewer/ice.js";
import { FLYERS, overlays } from "./viewer/overlays.js";
import { capturing } from "./viewer/capturing.js";
import { commands } from "./viewer/commands.js";

export { autoExposureTarget } from "./viewer/exposure.js";

/**
 * The canvas must not present through an sRGB format. The shader's tone
 * map already gamma-encodes, and an sRGB swapchain encodes again, for an
 * effective gamma near 3.1 that nothing on screen names. Gamma happens
 * once, in the shader, where the slider can reach it.
 */
export function presentFormat(preferred) {
  return preferred.endsWith("-srgb")
    ? preferred.slice(0, -"-srgb".length)
    : preferred;
}

// Exported for tests, which drive its prototype methods against stubs —
// see tests/app/test_soar_viewer_state.py. boot() is still the way to build one.
export class Viewer {
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
    this.hazeHeightDependent = K.DEFAULT_HAZE_HEIGHT_DEPENDENT;
    this._holdMode = K.DEFAULT_HOLD_MODE;
    // The video-camera loop: exposure glides toward what the linear meter
    // asks for (see constants.js, AUTO_EXPOSURE_*). `exposure` is the live
    // value the frames are drawn with either way; turning auto off freezes
    // it where it is, the way a camera's manual mode inherits the metered
    // setting. Auto is a tier default (off below High) until touched by
    // hand; the pre-probe initial state matches the default tier's.
    this.exposure = K.DEFAULT_EXPOSURE;
    this.autoExposure =
      K.QUALITY_PRESETS[K.DEFAULT_QUALITY_TIER].autoExposure ?? false;
    this._aeLastMeterMs = 0;
    this._aeSettled = false;
    this.lodStrength = K.QUALITY_PRESETS[K.DEFAULT_QUALITY_TIER].lodStrength;
    // Three settings carry a per-tier default (haze, motion smoothing, LOD
    // strength) and the tier is chosen by measurement, so it can change under
    // the user twice in the first second. A setting the user has touched is
    // theirs from then on and no tier change moves it again.
    this._byHand = new Set();
    this.beholdQuality = K.DEFAULT_BEHOLD_QUALITY;
    this.beholdField = "outer";   // "outer" or "nest": behold renders one
    // witness is the default because it reproduces this exact view — nests,
    // wrap and image settings — where behold path-traces one bare field.
    this.terminalRenderer = "witness";
    this.captureSize = null;
    this.videoFps = K.DEFAULT_VIDEO_FPS;
    // Captures pick their own tier — a still leans expensive, a video has
    // hundreds of frames to pay for. A capture at a
    // named tier is what a PARKED view at that tier converges to, at the
    // capture size: the tier's lighting method, parked spp and
    // look defaults, sampled like the top hold rung. Stored null = "not
    // explicitly picked", which the captureStillTier/-VideoTier getters
    // resolve to the usual defaults — or to "custom" (the session's hand-set
    // look, verbatim) the moment any of haze/lod/autoExposure is by hand.
    this._captureStillTier = null;
    this._captureVideoTier = null;
    // Whether a capture — still or video — draws the flyer and the minimap
    // over the clouds. One switch for both overlays, shared by both panels:
    // a capture either shows the flight apparatus or it doesn't.
    this.captureOverlays = true;
    // Lens water (lens.wgsl): drops on the lens after flying through cloud,
    // in the live view and in videos; stills and captures are always dry.
    this.lensWater = readLensWater();
    // True once any Advanced quality setting was hand-changed: the tier row
    // shows a selected "Custom" chip, and the governor stands down. Clicking
    // any preset (or Auto) resets every Advanced setting and clears this.
    this.qualityCustom = false;

    // Ice-detection mode (I in flight): false-color ice fraction plus the
    // dart-tip condensate readout. Off on every load, and its volume is not
    // read until the first activation — see toggleIceMode.
    this.iceModeOn = false;
    // The scene whose ice read is in flight, or null. A scene rather than a
    // flag: the read outlives the field it was started for, and a bare
    // boolean cleared by whichever of the two finished first let a second
    // read start for the new scene and race the first one's attachment.
    this._iceLoadingFor = null;
    // Bumped to discard a pending activation (a mode switch out of research).
    this._iceRequest = 0;
    this._iceProbeBuf = null;        // 16-byte readback buffer, made lazily
    this._iceProbePending = false;
    this._iceProbeAt = 0;

    this.flyerEnabled = true;
    // Which of the two flyers is up. They share a render pass and differ in
    // mesh, shading and — mostly — in how they move. B cycles them in FLYERS
    // order and then off; the first of FLYERS is what you get on load.
    this.flyerKind = FLYERS[0].kind;
    // "corner" | "full" | "off". Fullscreen frees the mouse so the map can
    // be clicked to travel; M cycles through all three.
    this.minimapMode = "corner";
    this.recorder = new TrackRecorder();
    // Built after the first field loads, and null on every path that never
    // gets that far. Everything that notifies it does so optionally.
    this.tutorial = null;
    this.paused = true;
    this.captured = false;
    this.frameIndex = 0;
    // The scene clock (s), u.scene_clock, read by what moves in the
    // scene (the city's steam; the sea has no clock and packs 0). It runs only while the view is flying or a track is being
    // recorded, never while paused — so a parked view is a still scene and
    // its accumulation converges, and a held shot keeps the steam it was
    // held on. Captures take it from the view (a still) or from the track
    // (a video), never from the wall.
    this.sceneTime = 0.0;
    // Paused time inside a recording, cut from the clock its track records.
    this._trackPausedS = 0.0;
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
    // The in-flight tier governor — armed when the probe settles, fed one
    // rAF delta per marched camera-moving frame. See the GOVERNOR_* block
    // in constants.js for the whole argument.
    this._governor = null;

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

    // Everything below is filled in later. It is listed here so that this
    // constructor names every field the viewer has.
    // By boot():
    this.onFailure = null;
    this.setLoadingVisible = null;
    // By start(): the shader sources, the UI, the canvas context.
    this.shaders = null;
    this.lensSource = null;
    this.hudSource = null;
    this.birdSource = null;
    this.dartSource = null;
    this.progress = null;
    this.ui = null;
    this.context = null;
    this.canvasFormat = null;
    // By loadField(): one field's scene and what is built for it.
    this.scene = null;
    this.renderer = null;
    this.camera = null;
    this.minimap = null;
    this.flyers = null;
    this.collider = null;
    this._minimapProblem = null;
    this._flyerProblem = null;
    // The surface tiles, loaded on first use and kept for the session.
    this._ocean = null;
    this._cityTile = null;
    // The frame clock, and the lens water's last step (loop.js).
    this._lastTime = null;
    this._lensDt = null;
    // Pointer-lock and Escape bookkeeping (input.js).
    this._tabRelease = false;
    this._lockLostAt = null;
    this._lastEscapeAt = null;
    // The probe clock's verdict, kept for the session (autotier.js).
    this._probeClockVerdict = null;
    // When auto exposure last applied a step (exposure.js).
    this._aeLastApplyMs = null;
    // A capture in progress, its cancel flag, and the lens setting the last
    // track was recorded with (capturing.js).
    this._capturing = false;
    this._videoAbort = false;
    this._trackLensWater = null;
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

  async start(source, progress) {
    const shaders = await loadShaders(new URL("./shader/", import.meta.url));
    this.shaders = shaders.modules;
    this.lensSource = shaders.lens;
    this.hudSource = await (await fetch("hud.wgsl")).text();
    this.birdSource = await (await fetch("bird.wgsl")).text();
    this.dartSource = await (await fetch("dart.wgsl")).text();
    this.progress = progress;

    // The UI is built before the field loads, because loading a file asks
    // questions — which group, what units — and those are menu panels.
    this.ui = new UI(this.uiRoot, this);
    // Off until asked for. The overlay is a diagnostic — fps, loop state,
    // tier — and the two numbers a flyer wants without asking (speed, and
    // whether a track is recording) are in the corner chip regardless.
    this.ui.statsMode = "off";

    this.context = this.canvas.getContext("webgpu");
    this.canvasFormat = presentFormat(navigator.gpu.getPreferredCanvasFormat());
    this.context.configure({
      device: this.device, format: this.canvasFormat, alphaMode: "opaque",
    });

    await this.loadField(source, progress);
    this._bindInput();
    this.paused = false;
    this._startLoop();

    // After the field, because the first prompt puts the camera somewhere and
    // there is no camera until a field has been loaded. Built whatever the
    // entry point or the mode; `maybeStart` is what decides it has been run
    // before.
    const { Tutorial } = await import("./tutorial.js");
    this.tutorial = new Tutorial(this.uiRoot, this);
    this.tutorial.maybeStart();
  }
}

// The rest of the class is in viewer/, one file per concern. Each file
// exports an object of methods (and getters), and all of them are installed
// on Viewer.prototype here, so every one is called as viewer.method() and
// sees the same `this`.
for (const methods of [
  load, autotier, loop, input, settings, exposure,
  ice, overlays, capturing, commands,
]) {
  Object.defineProperties(Viewer.prototype,
                          Object.getOwnPropertyDescriptors(methods));
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
