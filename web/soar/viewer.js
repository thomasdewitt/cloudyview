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
import { TrackRecorder, trackPayload, resampleTrack } from "./track.js";
import { UI } from "./ui.js";
import { mod360 } from "./spectral.js";
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
    this.beholdQuality = K.DEFAULT_BEHOLD_QUALITY;
    this.captureSize = null;
    this.videoFps = K.DEFAULT_VIDEO_FPS;
    this.videoAccumulate = K.DEFAULT_VIDEO_ACCUMULATE;

    this.birdEnabled = true;
    this.minimapEnabled = true;
    this.recorder = new TrackRecorder();
    this.paused = true;
    this.captured = false;
    this.frameIndex = 0;
    this._fpsAcc = 0; this._fpsN = 0; this._fps = null; this._frameMs = null;
    this._lastSignature = null;
    this._discardNextPointerMove = false;
  }

  get sunZenith() { return 90.0 - this.sunElevation; }

  get sunCompass() {
    const a = mod360(this.sunAzimuth);
    for (const [edge, label] of K.COMPASS_EDGES) if (a < edge) return label;
    return "N";
  }

  get isFullscreen() { return Boolean(document.fullscreenElement); }

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
    this._lastTime = performance.now();
    this.paused = false;
    requestAnimationFrame(() => this._frame());
  }

  /**
   * Load (or replace) the field. Everything resident on the GPU for the old
   * one is released first — a second field would otherwise sit alongside the
   * first, and these are gigabytes.
   */
  async loadField(source, progress) {
    if (this.scene) {
      this.stop = true;
      await this.device.queue.onSubmittedWorkDone();
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

    if (source.kind === "demo") {
      this.scene = await loadDemoScene(
        this.device, source.base, OCEAN_URL,
        (stage, fraction) => progress(stage, fraction));
      if (this.scene.sun) {
        this.sunAzimuth = this.scene.sun.azimuth;
        this.sunElevation = this.scene.sun.elevation;
      }
    } else {
      const { loadFileScene } = await import("./ingest/index.js");
      this.scene = await loadFileScene(
        this.device, source.file, {
          // The ocean is a patch of sea surface, not anything about the
          // data, so it survives a change of field.
          ocean: async () => (this._ocean ??=
            await loadOceanTile(this.device, OCEAN_URL)),
          progress,
          ask: (question) => this._ask(question),
        });
    }

    this.renderer = new Renderer(this.device, this.shaderSource, this.scene,
                                 { canvasFormat: this.canvasFormat });
    if (this.scene.periodicDefault === false) this.renderer.setPeriodic(false);
    progress("Compiling the shader…", 0.97);
    await this.renderer.init();
    this.camera = new FlightCamera(this.scene.bmin, this.scene.bmax,
                                   { periodic: this.renderer.periodic });

    // The map and the bird are overlays, not the picture. A GPU that cannot
    // hold one (or a field too wide for a 2D texture) is a reason to fly
    // without it and say so, not a reason to fail the load.
    try {
      this.minimap = await new Minimap(this.device, {
        albedo: this.scene.albedo, shape: this.scene.albedoShape,
      }).init(this.canvasFormat, this.hudSource);
    } catch (err) {
      this.minimap = null;
      this._minimapProblem = String(err?.message || err);
    }
    try {
      this.bird = new Bird(this.device, {
        volumeView: this.scene.volumeView, sampler: this.renderer.volSampler,
        bmin: this.scene.bmin, bmax: this.scene.bmax,
      });
      await this.bird.init(this.canvasFormat, this.birdSource);
    } catch (err) {
      this.bird = null;
      this._birdProblem = String(err?.message || err);
    }

    this.ui.setSubtitle(this.sourceLabel);
    if (this.scene.nestNote) this.ui.say(this.scene.nestNote, 8);
    this.stop = false;
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
        reject(new Error("Cancelled before the field was loaded."));
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

  // --- input ---------------------------------------------------------------

  _bindInput() {
    const canvas = this.canvas;

    canvas.addEventListener("click", () => {
      if (!this.captured && !this.ui.isOpen) canvas.requestPointerLock();
    });

    document.addEventListener("pointerlockchange", () => {
      const wasCaptured = this.captured;
      this.captured = document.pointerLockElement === canvas;
      // The jump to the window centre arrives as one enormous movement event.
      // Swallowing it is the difference between taking the mouse and having
      // the view snap somewhere random.
      this._discardNextPointerMove = true;
      this._syncChrome();
      if (this.captured) return;

      this.camera.keys.clear();
      // Escape releases the pointer lock in the browser itself, and Firefox
      // ALSO delivers the keydown. Handling both turned one press into
      // pause-then-resume — the menu appeared and vanished in a frame. So the
      // lock loss is the single source of truth for pausing, and the keydown
      // that caused it is ignored for a moment afterwards.
      this._lockLostAt = performance.now();
      if (wasCaptured && !this._tabRelease && !this.paused) this.pause();
      this._tabRelease = false;
    });

    // Firefox refuses a re-lock for about a second after the user pressed
    // Escape, so resuming cannot rely on it. Say what to do instead.
    document.addEventListener("pointerlockerror", () => {
      if (!this.paused) this.ui.say("Click the view to take the mouse.", 3);
    });

    document.addEventListener("mousemove", (e) => {
      if (!this.captured) return;
      if (this._discardNextPointerMove) {
        this._discardNextPointerMove = false;
        return;
      }
      this.camera.look(e.movementX, e.movementY);
    });

    document.addEventListener("wheel", (e) => {
      if (!this.captured) return;
      this.camera.scrollSpeed(e.deltaY, performance.now() / 1000);
    }, { passive: true });

    document.addEventListener("keydown", (e) => this._onKeyDown(e));
    document.addEventListener("keyup", (e) => {
      this.camera.keys.delete(e.key.toLowerCase());
    });

    window.addEventListener("blur", () => this.camera.keys.clear());
  }

  /**
   * Keys are only for things done in flight. Everything reachable from the
   * menu is reachable by clicking it, so nothing here needs a second binding.
   */
  _onKeyDown(e) {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;

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
      if (this.ui.isOpen) this.ui.back();
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
          this.canvas.requestPointerLock();
        }
        return;
      case "f": this.toggleFullscreen(); return;
      case "F3": e.preventDefault(); this.ui.cycleStats(); return;
      case "b": this.toggleBird(); return;
      case "m": this.toggleMinimap(); return;
      case "r": this.toggleTrackRecording(); return;
      case "F12": e.preventDefault(); this.ui.open("capture"); return;
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
   * and while the menu is open the menu has everything, so a floating "menu"
   * button that only appears once the menu is already open is just noise.
   */
  _syncChrome() {
    const viewer = this.canvas.parentElement;
    viewer.classList.toggle("captured", this.captured);
    viewer.classList.toggle("menu-open", Boolean(this.ui?.isOpen));
  }

  pause() {
    this.paused = true;
    this.camera.keys.clear();
    if (this.captured) document.exitPointerLock();
    this.ui.open("main");
    this._syncChrome();
  }

  resume() {
    this.ui.close();
    this.paused = false;
    this._lastTime = performance.now();
    this._syncChrome();
    this.canvas.requestPointerLock();
  }

  setSun({ azimuth, elevation, zenith }) {
    if (azimuth != null) this.sunAzimuth = mod360(azimuth);
    if (zenith != null) elevation = 90.0 - zenith;
    if (elevation != null) {
      this.sunElevation = Math.min(
        90.0, Math.max(K.MIN_SUN_ELEVATION_DEG, elevation));
    }
  }

  setToneMapGamma(gamma) {
    const [lo, hi] = K.TONE_MAP_GAMMA_LIMITS;
    if (!(gamma >= lo && gamma <= hi)) {
      throw new Error(`tone_map_gamma must be in [${lo}, ${hi}]; got ${gamma}.`);
    }
    this.toneMapGamma = gamma;
    this.renderer.resetAccumulation();
  }

  setQualityTier(tier) {
    this.renderer.setQualityTier(tier);
  }

  togglePeriodic() {
    this.renderer.setPeriodic(!this.renderer.periodic);
    this.camera.periodic = this.renderer.periodic;
    this.camera.constrain();
    this.ui.say(`Periodic domain ${this.renderer.periodic ? "on" : "off"}.`);
  }

  toggleMinimap() {
    if (!this.minimap) {
      this.ui.say(
        this._minimapProblem
          ? `No minimap for this field: ${this._minimapProblem}`
          : "There is no minimap for this field.", 5);
      return;
    }
    this.minimapEnabled = !this.minimapEnabled;
    this.ui.say(`Minimap ${this.minimapEnabled ? "on" : "off"}.`);
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
    }
    this.ui.open("main");
  }

  leave() {
    this.stop = true;
    this.scene.destroy();
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
    if (this.recorder.recording) {
      const samples = this.recorder.stop();
      if (samples.length < 2) {
        this.ui.say("Too short to be a track — nothing recorded.", 3);
        return;
      }
      this.pause();
      this.ui.open("track", { samples });
      return;
    }
    this.recorder.start(performance.now() / 1000);
    this.ui.say("Recording the flight path. R again to stop.", 3);
  }

  /** How many video frames a track becomes at the chosen rate. */
  trackFrameCount(samples) {
    return resampleTrack(samples, this.videoFps,
                         { periodic: this.renderer.periodic }).length;
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
      if (saved) endOfflineRender(this.renderer, saved);
      this.ui.hideProgress();
      this._capturing = false;
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
      });
      overlays.push((enc, view, format) =>
        this.bird.encodePass(enc, view, format, size));
    }
    if (this.minimap && this.minimapEnabled) {
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
        light_march_lod_degrees: K.APP_LIGHT_MARCH_LOD_DEGREES,
        view_step_lod_degrees: K.APP_VIEW_STEP_LOD_DEGREES,
      },
      timestamp: new Date().toISOString(),
      reproduction_command: this.beholdCommand(),
    };
  }

  async saveScreenshot({ overlays = true } = {}) {
    if (this._capturing) return;
    this._capturing = true;
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
      });
      stillOverlays.push((enc, view, format) =>
        this.bird.encodePass(enc, view, format, size));
    }
    if (overlays && this.minimap && this.minimapEnabled) {
      this.minimap.update(this.camera, this.scene, size);
      stillOverlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }
    try {
      const image = await renderStill(
        this.device, this.renderer, this._viewKwargs(), size,
        K.STILL_ACCUMULATE_FRAMES, stillOverlays);
      this.ui.showProgress("Encoding…", 0.95);
      const blob = await imageDataToPng(image, this.renderMetadata(size));
      download(blob, timestampedName("cloudyview_soar", ".png"));
      this.ui.say(`Saved a ${size[0]}x${size[1]} still.`, 3);
    } catch (err) {
      this.ui.say(`Could not save the still: ${err.message}`, 5);
    } finally {
      this.ui.hideProgress();
      this._capturing = false;
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
    if (!file) { this.paused ? this.ui.open("main") : this.resume(); return; }

    this.paused = true;
    this.setLoadingVisible?.(true);
    try {
      await this.loadField({ kind: "file", file }, this.progress);
      this.setLoadingVisible?.(false);
      this._lastTime = performance.now();
      this.paused = false;
      requestAnimationFrame(() => this._frame());
    } catch (err) {
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

  beholdCommand() {
    const rel = this.camera.relativePosition();
    const n = (v) => Number(v).toPrecision(12).replace(/\.?0+$/, "");
    const source = this.scene.sourceName ?? "<your-file.nc>";
    return [
      "behold", source, this.beholdQuality, "--gpu",
      "--camera-position", n(rel[0]), n(rel[1]), n(rel[2]),
      "--camera-azimuth", n(this.camera.azimuth),
      "--camera-elevation", n(this.camera.elevation),
      "--fov", n(this.camera.fov),
      "--sun-azimuth", n(this.sunAzimuth),
      "--sun-elevation", n(this.sunElevation),
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
      lightMarchLodDegrees: K.APP_LIGHT_MARCH_LOD_DEGREES,
      viewStepLodDegrees: K.APP_VIEW_STEP_LOD_DEGREES,
      frameIndex: this.frameIndex,
    };
  }

  async _frame() {
    if (this.stop) return;
    // A capture owns the renderer while it runs — the live loop would fight
    // it for the accumulation buffer and neither picture would converge.
    if (this._capturing) {
      requestAnimationFrame(() => this._frame());
      return;
    }
    const now = performance.now();
    const dt = Math.min((now - this._lastTime) / 1000, 0.1);
    this._lastTime = now;

    if (!this.paused) this.camera.move(dt);
    this.camera.constrain();

    // "Moving" is an exact comparison against the previous frame's view, not
    // a velocity: it is what lets Potato swap to full quality the instant you
    // stop, rather than after a timeout.
    const signature = this.camera.signature();
    this.renderer.setCameraMoving(
      this._lastSignature !== null && signature !== this._lastSignature);
    this._lastSignature = signature;

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
      });
      overlays.push((enc, view, format) =>
        this.bird.encodePass(enc, view, format, [outW, outH]));
    }
    if (this.minimap && this.minimapEnabled) {
      this.minimap.update(this.camera, this.scene, [outW, outH]);
      overlays.push((enc, view, format) =>
        this.minimap.encodePass(enc, view, format));
    }

    try {
      // Deliberately a thunk: see Renderer.drawFrame. The swapchain texture
      // must not be acquired until every await inside the draw has resolved.
      await this.renderer.drawFrame(
        () => this.context.getCurrentTexture().createView(), this.canvasFormat,
        [outW, outH], this._viewKwargs(),
        { deltaSeconds: dt, overlays });
    } catch (err) {
      this.stop = true;
      this.onFailure?.("Rendering stopped.", String(err.message || err),
                       err.advice || "");
      return;
    }

    // Sampled after the frame it describes, so the track records what was on
    // screen rather than what was about to be.
    if (this.recorder.recording && !this.paused) {
      this.recorder.sample(now / 1000, this.camera);
    }

    this.frameIndex += 1;
    this._fpsAcc += dt; this._fpsN += 1;
    if (this._fpsAcc >= 0.5) {
      const meanDt = this._fpsAcc / this._fpsN;
      this._fps = 1 / meanDt;
      this._frameMs = meanDt * 1000;
      this._fpsAcc = 0; this._fpsN = 0;
    }
    this.ui.drawStats({
      fps: this._fps, frameMs: this._frameMs, camera: this.camera,
      tier: this.renderer.qualityTier, renderScale: this.renderer.renderScale,
      frame: this.frameIndex,
      minimap: Boolean(this.minimap && this.minimapEnabled),
      bird: Boolean(this.bird && this.birdEnabled),
      recording: this.recorder.recording,
      showSpeed: performance.now() / 1000 < this.camera.speedFlashUntil,
    });

    requestAnimationFrame(() => this._frame());
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
  await viewer.start(source, progress);
  onReady?.();
  return viewer;
}
