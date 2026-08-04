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
import { UI } from "./ui.js";
import { mod360 } from "./spectral.js";
import {
  renderStill, imageDataToPng, download, timestampedName,
} from "./capture.js";

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
    const shaderSource = await (await fetch("raymarch.wgsl")).text();

    if (source.kind === "demo") {
      this.scene = await loadDemoScene(
        this.device, source.base, OCEAN_URL,
        (stage, fraction) => progress(stage, fraction));
      if (this.scene.sun) {
        this.sunAzimuth = this.scene.sun.azimuth;
        this.sunElevation = this.scene.sun.elevation;
      }
    } else {
      // Ingest of an arbitrary netCDF lands here; see ingest/.
      const { loadFileScene } = await import("./ingest/index.js");
      this.scene = await loadFileScene(
        this.device, source.file, {
          ocean: () => loadOceanTile(this.device, OCEAN_URL),
          progress,
          ask: (question) => this._ask(question),
        });
    }

    this.context = this.canvas.getContext("webgpu");
    this.canvasFormat = presentFormat(navigator.gpu.getPreferredCanvasFormat());
    this.context.configure({
      device: this.device, format: this.canvasFormat, alphaMode: "opaque",
    });

    this.renderer = new Renderer(this.device, shaderSource, this.scene,
                                 { canvasFormat: this.canvasFormat });
    this.camera = new FlightCamera(this.scene.bmin, this.scene.bmax,
                                   { periodic: this.renderer.periodic });
    this.ui = new UI(this.uiRoot, this);
    this.ui.statsMode = "subtle";
    this.ui.setSubtitle(this.sourceLabel);

    this._bindInput();
    this._lastTime = performance.now();
    this.paused = false;
    requestAnimationFrame(() => this._frame());
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
      this.canvas.parentElement.classList.toggle("captured", this.captured);
      if (this.captured) return;

      this.camera.keys.clear();
      // While the pointer is locked the browser eats Escape to release it, so
      // the keydown handler never sees it. Losing the lock IS the pause —
      // except when Tab asked for it, which frees the cursor without leaving
      // the flight.
      if (wasCaptured && !this._tabRelease && !this.paused) this.pause();
      this._tabRelease = false;
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
      // Pointer lock eats the first Escape itself; this fires on the second,
      // or immediately when the mouse was already free.
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
      case "b":
        this.birdEnabled = !this.birdEnabled;
        this.ui.say(`Bird ${this.birdEnabled ? "on" : "off"}.`, 1.2);
        return;
      case "m":
        this.minimapEnabled = !this.minimapEnabled;
        this.ui.say(`Minimap ${this.minimapEnabled ? "on" : "off"}.`, 1.2);
        return;
      case "r": this.toggleTrackRecording(); return;
      case "F12": e.preventDefault(); this.ui.open("capture"); return;
      default: break;
    }
    if ("wasdc ".includes(key)) this.camera.keys.add(key);
    if (key === "Shift") this.camera.keys.add("shift");
  }

  // --- state changes -------------------------------------------------------

  pause() {
    this.paused = true;
    this.camera.keys.clear();
    if (this.captured) document.exitPointerLock();
    this.ui.open("main");
  }

  resume() {
    this.ui.close();
    this.paused = false;
    this._lastTime = performance.now();
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

  toggleTrackRecording() {
    this.ui.say("Track recording is not wired up yet.", 2);
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
    try {
      const image = await renderStill(
        this.device, this.renderer, this._viewKwargs(), size,
        K.STILL_ACCUMULATE_FRAMES);
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
      // `overlays` will select the bird and minimap once those land; the
      // still is clouds-only until then, which is what both buttons give.
      void overlays;
    }
  }

  pickFile() {
    this.ui.say("Opening another file is not wired up yet.", 2);
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

    try {
      await this.renderer.drawFrame(
        this.context.getCurrentTexture().createView(), this.canvasFormat,
        [outW, outH], this._viewKwargs(), { deltaSeconds: dt });
    } catch (err) {
      this.stop = true;
      this.onFailure?.("Rendering stopped.", String(err.message || err),
                       err.advice || "");
      return;
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
      frame: this.frameIndex, minimap: this.minimapEnabled,
      bird: this.birdEnabled, recording: false,
      showSpeed: performance.now() / 1000 < this.camera.speedFlashUntil,
    });

    requestAnimationFrame(() => this._frame());
  }
}

export async function boot({ device, source, progress, onReady, onFailure }) {
  const canvas = document.getElementById("view");
  const uiRoot = document.getElementById("ui");
  uiRoot.replaceChildren();
  const viewer = new Viewer(device, canvas, uiRoot);
  viewer.onFailure = onFailure;
  await viewer.start(source, progress);
  onReady?.();
  return viewer;
}
