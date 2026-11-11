// Viewer methods (load). Loading a field and letting it go: the surface
// tiles, the scene, the renderer and the overlays built for it, the loader's
// questions, the file picker, and dispose.

"use strict";

import * as K from "../constants.js";
import { Renderer } from "../renderer.js";
import { FlightCamera } from "../camera.js";
import { loadDemoScene, loadOceanTile } from "../scene.js";
import { Minimap } from "../minimap.js";
import { CityTile, CityLots } from "../citylots.js";
import { CityProps } from "../cityprops.js";
import { CityCollider } from "../collide.js";
import { FLYERS, destroyFlyers } from "./overlays.js";

const OCEAN_URL = "ocean";
const CITY_URL = "city";

export const load = {
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
    destroyFlyers(this.flyers);
    this.scene.destroy();
    this.scene = null;
    this.renderer = null;
    this.minimap = null;
    this.flyers = null;
    this.collider = null;
    this.frameIndex = 0;
    this._lastSignature = null;
    this.sunAzimuth = K.DEFAULT_SUN_AZIMUTH;
    this.sunElevation = K.DEFAULT_SUN_ELEVATION;
  },

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
  /**
   * The surface tiles are not anything about the data — the ocean is a
   * patch of sea surface, the city a patch of city — so they survive a
   * change of field and belong to the viewer, which is what disposes of
   * them. Each is loaded on first use and cached for the session.
   */
  async _surfaceTile(kind) {
    if (kind === "city") {
      // keepCells: the minimap draws the city out of the tile's own mip
      // 0, the flight collider builds its buildings from it too, and the
      // tile is the only place those bytes ever arrive. Held
      // for the session with the tile itself (8 MB beside a multi-GB
      // volume) rather than re-downloaded per field.
      return (this._cityTile ??=
        await loadOceanTile(this.device, CITY_URL, undefined,
                            { keepCells: true,
                              schema: K.SURFACE_TILE_SCHEMAS.city }));
    }
    return (this._ocean ??=
      await loadOceanTile(this.device, OCEAN_URL, undefined,
                          { schema: K.SURFACE_TILE_SCHEMAS.ocean }));
  },

  async loadField(source, progress) {
    await this._releaseField();

    let scene = null, renderer = null, minimap = null, flyers = null;
    let sun = null, minimapProblem = null, flyerProblem = null;
    try {
      const surface = (kind) => this._surfaceTile(kind);

      if (source.kind === "demo") {
        scene = await loadDemoScene(
          this.device, source.base, surface,
          (stage, fraction) => progress(stage, fraction),
          // Cyberpunk mode flies the same demos every other mode does and
          // stands them on the city; the landing page is what decides that,
          // so it arrives with the source rather than being read here.
          { surfaceKind: source.surfaceKind ?? null });
        sun = scene.sun ?? null;
      } else {
        const { loadFileScene } = await import("../ingest/index.js");
        scene = await loadFileScene(
          this.device, source.file, {
            ocean: () => surface("ocean"),
            progress,
            ask: (question) => this._ask(question),
          });
      }

      renderer = new Renderer(this.device, this.shaders, scene,
                              { canvasFormat: this.canvasFormat,
                                lensSource: this.lensSource });
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
      // Over a city the map IS the city: same image the shader builds from,
      // in the tile's frame, replacing the cloud map outright. See Minimap.
      const map = new Minimap(this.device, {
        albedo: scene.albedo, shape: scene.albedoShape,
        cityCells: scene.cityCells ?? null,
      });
      try {
        minimap = await map.init(this.canvasFormat, this.hudSource);
      } catch (err) {
        map.destroy();
        minimap = null;
        minimapProblem = String(err?.message || err);
      }
      // Both flyers are built up front rather than on demand. Each is a few
      // thousand triangles and one small pipeline, and building them together
      // means switching between them is instant and — more to the point —
      // that a GPU which cannot hold one says so now, at load, rather than
      // the first time someone presses B.
      const resources = {
        volumeView: scene.volumeView, sampler: renderer.volSampler,
        bmin: scene.bmin, bmax: scene.bmax,
      };
      const built = {};
      for (const { kind, Species, source } of FLYERS) {
        const flyer = new Species(this.device, resources);
        try {
          await flyer.init(this.canvasFormat, this[source]);
          built[kind] = flyer;
        } catch (err) {
          flyer.destroy();
          built[kind] = null;
          // One message covers both: they share a pass, so whatever stops one
          // stops the other, and two identical notes help nobody.
          flyerProblem ??= String(err?.message || err);
        }
      }
      flyers = built;
    } catch (err) {
      destroyFlyers(flyers);
      minimap?.destroy();
      renderer?.destroy();
      scene?.destroy();
      throw err;
    }

    this.scene = scene;
    this.renderer = renderer;
    // Ice-detection mode does not carry across a change of field. The new
    // field's fraction has not been read — that is the point of reading it on
    // demand — so the mode starts off and the next I pays for it.
    this.iceModeOn = false;
    // Any read still in flight is for the scene it names, not for this one,
    // and toggleIceMode's guard compares against the scene rather than being
    // cleared here.
    renderer.iceMode = false;
    this._syncIceReadout();
    // A renderer is born in DEFAULT_HOLD_MODE; if the UI asked for another
    // one while there was no renderer to ask, it gets it now.
    renderer.setHoldMode(this.holdMode);
    this.minimap = minimap;
    this.flyers = flyers;
    // If the one that was up last time did not build for this field, start on
    // one that did rather than on a null nothing draws and nobody mentions.
    if (!this.flyer) this.flyerKind = this.availableFlyers[0] ?? this.flyerKind;
    this._minimapProblem = minimapProblem;
    this._flyerProblem = flyerProblem;
    if (sun) {
      this.sunAzimuth = sun.azimuth;
      this.sunElevation = sun.elevation;
    }
    this.camera = new FlightCamera(scene.bmin, scene.bmax,
                                   { periodic: renderer.periodic,
                                     start: scene.startCamera,
                                     // EVERY scene has a second periodic
                                     // frame — the surface tile: the city's
                                     // block grid, or the day ocean's wave
                                     // patch. The camera tracks its position
                                     // in it alongside the cloud one (see
                                     // FlightCamera's surfacePosition), so
                                     // neither the city nor the water jumps
                                     // when the cloud frame folds ("the
                                     // ocean should never jump" — Thomas,
                                     // 2026-09-01).
                                     // Folded at the surface's own period:
                                     // the tile for the city, 16 tiles
                                     // for the sea (constants.js
                                     // SURFACE_FRAME_TILES).
                                     surfaceTileExtent: scene.surfacePeriod,
                                     surfaceOffsetM:
                                       scene.city ? scene.cityOffsetM
                                                  : [0.0, 0.0],
                                     // Streets, not a cloud field: the city
                                     // is the only scene with anything
                                     // within 20 m of the camera, so it
                                     // flies slower too.
                                     // And the street is solid ground
                                     // the collider owns, not a floor.
                                     ...(scene.city
                                       ? { speed: K.CITY_DEFAULT_SPEED,
                                           floorZ: null }
                                       : {}) });
    // Over the night city the buildings are solid (collide.js), and so is
    // what stands among them — freeways, lamps, parked cars, skybridges,
    // roof ridges (cityprops.js): the same things the shader raises, stood
    // up on the CPU from the same tile. The day scenes have nothing to fly
    // into and get no collider at all.
    const cityTile = scene.city ? new CityTile(scene.cityCells) : null;
    this.collider = scene.city
      ? new CityCollider(new CityLots(cityTile, new CityProps(cityTile)),
                         scene.oceanTileExtent)
      : null;

    this.ui.setSubtitle(this.sourceLabel);

    // Everything the load has to admit to, in ONE toast. say() replaces
    // whatever is on screen rather than queueing, so four calls in a row
    // would show the fourth and quietly lose the other three — which is the
    // same silence these messages exist to break.
    const notes = [];
    // First, because it is the one that changes what you are looking at
    // rather than what is missing from it. A field whose axes were taken by
    // position renders a completely plausible cloud with x and z swapped;
    // the only defence against that is saying so. Never suppressed.
    if (scene.assumptions?.length) notes.push(scene.assumptions.join("\n"));
    if (scene.nestNote) notes.push(scene.nestNote);
    if (minimapProblem) notes.push(`Flying without the minimap: ${minimapProblem}`);
    if (flyerProblem) notes.push(`Flying without a flyer: ${flyerProblem}`);
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

    // The sun-shadow cache is a standing resource now — every tier but Max
    // reads it — so it is allocated HERE, where failing is still a clean
    // load rejection: the allocation is the memory probe (WebGPU has no
    // free-VRAM query), and a field whose cache does not fit fails like any
    // other GPU-OOM instead of degrading silently.
    await this.renderer.allocateLightCache();
    this.renderer.startLightBake(this._viewKwargs());

    this._beginAutoTier();
  },

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
      } else if (question.panel === "coordUnits") {
        this.ui.open("coordUnits", {
          coords: question.coords, filename: question.filename,
          onPick: (coordUnits) => done({ coordUnits }),
          onCancel: cancel,
        });
      } else if (question.panel === "gridSpacing") {
        this.ui.open("gridSpacing", {
          axes: question.axes, filename: question.filename,
          group: question.group,
          onPick: (gridSpacing, zFirst) => done({ gridSpacing, zFirst }),
          onCancel: cancel,
        });
      } else if (question.panel === "axes") {
        // Detection could not tell the dimensions apart. This is the reason
        // a failed load is no longer a dead end: the panel is offered
        // INSTEAD of the error, not after it.
        this.ui.open("axes", {
          dims: question.dims, filename: question.filename,
          reason: question.reason,
          onPick: (axes) => done({ axes }),
          onCancel: cancel,
        });
      } else if (question.panel === "variable") {
        this.ui.open("variable", {
          role: question.role, title: question.title,
          status: question.status, variables: question.variables,
          filename: question.filename, group: question.group,
          offerFile: question.offerFile,
          // null is a real answer for the ice role — "No ice" — and has to
          // reach the loader as null rather than as a cancellation.
          onPick: (variable) => done({ variable }),
          // The other first-class ice answer: it is in a second file.
          onPickFile: (chosen) => done({ file: chosen }),
          onCancel: cancel,
        });
      } else if (question.panel === "timestep") {
        this.ui.open("timestep", {
          timeDim: question.timeDim, filename: question.filename,
          group: question.group,
          onPick: (timestep) => done({ timestep }),
          onCancel: cancel,
        });
      } else {
        reject(new Error(`unknown question '${question.panel}'`));
      }
    });
  },

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
  },

  removeNest() {
    if (this.scene.removeNest()) {
      this.renderer.refreshBindGroup();
      // The baked sun-tau cache marched its shadows through the nest (the
      // light march integrates it at its own resolution), and a bake still
      // in flight would finish its remaining slices without it. Same
      // recovery as setSun: live march while the fresh bake lands.
      this.renderer.invalidateLightCache();
      this.renderer.startLightBake(this._viewKwargs());
      this.renderer.resetAccumulation();
      this.frameIndex = 0;
      this.ui.say("Nested field removed.");
      this._wake("nest removed");
    }
    this.ui.open("main");
  },

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
    this.tutorial?.destroy();
    this.tutorial = null;
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
    destroyFlyers(this.flyers);
    this._iceProbeBuf?.destroy();
    this._iceProbeBuf = null;
    this.scene?.destroy();
    this._ocean?.texture?.destroy();
    this._cityTile?.texture?.destroy();
    this.renderer = null;
    this.minimap = null;
    this.flyers = null;
    this.scene = null;
    this._ocean = null;
    this._cityTile = null;

    // Unconfigure before destroy: the swapchain holds textures on this device,
    // and leaving it configured against a destroyed device is the state
    // Firefox's own error message calls "Queue[Id] does not exist".
    this.context?.unconfigure();
    this.device.destroy();
  },

  async leave() {
    await this.dispose();
    location.reload();
  },
};
