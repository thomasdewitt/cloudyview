// Opening a netCDF file from the user's machine.
//
// The file is never uploaded and never fully read. h5wasm mounts the File
// through Emscripten's WORKERFS and libhdf5 pulls the chunks it needs, so a
// multi-gigabyte run costs tens of megabytes of wasm heap. That only works
// off the main thread — Emscripten's filesystem is synchronous and browsers
// forbid synchronous file access on the main thread — hence the worker and
// the small RPC below.

"use strict";

import {
  Scene, createVolumeTexture, writeVolumeSlab, createNestDummy,
  UPLOAD_DRAIN_BYTES,
} from "../scene.js";
import { volumeAABB, minVoxelSize, domainExtent, nestablePairs } from "../field.js";
import { SPEC_FLOOR_TEXTURE_3D } from "../gpu.js";
import { T } from "./strings.js";

/** Promise-shaped calls over postMessage, plus a channel for streamed data. */
class WorkerLink {
  constructor(onEvent) {
    this.worker = new Worker(new URL("./worker.js", import.meta.url),
                             { type: "module" });
    this.pending = new Map();
    this.nextId = 1;
    this.worker.onmessage = ({ data }) => {
      if (data.type) { onEvent(data); return; }
      const entry = this.pending.get(data.id);
      if (!entry) return;
      this.pending.delete(data.id);
      if (data.ok) entry.resolve(data.result);
      else {
        const err = new Error(data.error);
        err.advice = data.advice;
        // Structured cloning drops an Error's own properties, so the worker
        // sends this alongside rather than on the error. It is what turns
        // "could not tell which dimensions are x, y and z" from a dead end
        // into the manual-assignment panel.
        err.axisChoice = data.axisChoice || null;
        entry.reject(err);
      }
    };
    this.worker.onerror = (event) => {
      const err = new Error(
        `The file reader crashed: ${event.message || "unknown error"}`);
      for (const { reject } of this.pending.values()) reject(err);
      this.pending.clear();
    };
  }

  call(op, args, transfer) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, op, ...args }, transfer || []);
    });
  }

  /** Fire-and-forget, for the slab acks: no id, no reply, no promise. */
  notify(op, args) { this.worker.postMessage({ op, ...args }); }

  close() { this.worker.terminate(); }
}

/**
 * Upload one level's extinction field, slab by slab as the worker produces
 * it, so the JS heap never holds the whole volume.
 *
 * `iceOnly` is the deferred ice-fraction pass (loadIceVolume): the same
 * stream carrying only slabs tagged field:"ice", so no extinction texture is
 * allocated and no minimap arrives.
 */
function levelReceiver(device, label, onProgress, ack, { iceOnly = false } = {}) {
  const state = { texture: null, dims: null, geometry: null,
                  volume: null, tiles: 0, slabsDone: 0, error: null,
                  queuedBytes: 0,
                  iceTexture: null, iceSlabs: 0, iceSlabsDone: 0 };

  const step = async (message) => {
    if (message.type === "geometry") {
      state.geometry = message;
      state.tiles = message.slabs;
      // The volume is sized to the occupied z band, which is not known until
      // the file has been read, so allocation waits for the `volume` message
      // below. What CAN be settled now is whether the crop could possibly
      // help: x and y are untouched by it, so a field that overflows the
      // texture limit laterally overflows it however empty its sky is. Saying
      // so here rather than after the read is the difference between a
      // sentence and several minutes of decompression followed by the same
      // sentence.
      const [nx, ny] = message.description.shape;
      const cap = device.limits.maxTextureDimension3D;
      const worst = Math.max(nx, ny);
      if (worst > cap) {
        const axis = nx >= ny ? "x" : "y";
        const err = new Error(
          `This field needs ${worst} texels on ${axis}; this browser allows ` +
          `${cap}. Cropping empty sky cannot help — it only ever shrinks z.`);
        // No tool named here on purpose. This used to point at
        // tools/decimate_field.py, which has since been deleted, and advice
        // that names a file the reader cannot find is worse than advice that
        // says what to do — box-averaging is exact for extinction (it is
        // linear in the mixing ratios), so anyone can do it in four lines.
        err.advice = cap <= SPEC_FLOOR_TEXTURE_3D
          ? "Chrome reports the WebGPU spec minimum of 2048 no matter what " +
            "the card can do; Firefox reports the hardware's real limit. Or " +
            `box-average the field down by ${Math.ceil(worst / cap)}x on ` +
            `${axis} first.`
          : `Box-average the field down by ${Math.ceil(worst / cap)}x on ` +
            `${axis} first.`;
        throw err;
      }
    } else if (message.type === "read") {
      onProgress?.(message.done, Math.round(message.done * state.tiles),
                   state.tiles, "read");
    } else if (message.type === "volume") {
      state.volume = message;
      state.slabs = message.slabs;
      state.dims = message.shape.slice();
      if (!iceOnly) {
        state.texture = await createVolumeTexture(
          device, state.dims, `the field in ${label}`);
      }
      // Ice-detection mode: the ice-fraction volume, same shape, half the
      // texel. NOTE this still adds half the field's size again in video
      // memory, which is why it is loaded on demand rather than with the
      // field. r8unorm because the fraction is a [0, 1] quantity read
      // through a color ramp, where 1/255 steps are invisible — at the
      // price of clamping the negative/NaN condensate fp16 passes through,
      // which the worker does explicitly. Same format the demo bake writes
      // (tools/prebake_demos.py), so both paths give the same picture.
      state.iceSlabs = message.iceSlabs ?? 0;
      if (state.iceSlabs > 0) {
        state.iceTexture = await createVolumeTexture(
          device, state.dims, `the ice fraction in ${label}`, "r8unorm");
      }
    } else if (message.type === "slab") {
      const target = message.field === "ice" ? state.iceTexture : state.texture;
      if (!target) {
        throw new Error(
          "A slab arrived before the volume texture existed — the upload " +
          "queue is out of order.");
      }
      writeVolumeSlab(device, target, message.data,
                      message.origin, message.size);
      if (message.field === "ice") state.iceSlabsDone += 1;
      else state.slabsDone += 1;
      state.queuedBytes += message.data.byteLength;

      // Bound the staging memory a multi-gigabyte field can pile up — see
      // UPLOAD_DRAIN_BYTES in scene.js, which the demo path's whole-field
      // upload chunks by as well.
      if (state.queuedBytes >= UPLOAD_DRAIN_BYTES) {
        state.queuedBytes = 0;
        await device.queue.onSubmittedWorkDone();
      }
      // The ice-only pass counts its own slabs — the extinction counters stay
      // at zero there, and reporting them read as "part 0 of 0".
      onProgress?.(message.done,
                   iceOnly ? state.iceSlabsDone : state.slabsDone,
                   iceOnly ? state.iceSlabs : state.slabs, "upload");
    } else if (message.type === "map") {
      state.albedo = message.data;
      state.albedoShape = message.shape;
    }
  };

  // Messages arrive faster than they can be handled, and handling the first
  // one is genuinely slow: allocating the volume texture goes through an
  // error scope, which resolves only after the GPU has caught up. Handling
  // them concurrently meant slabs arriving during that window were written
  // to a texture that did not exist yet — the write threw into a promise
  // nobody was holding, the slab was silently lost, and the field came out
  // with untouched regions in it. Hence one at a time, in order, with the
  // failure kept rather than dropped.
  // Fires the moment the first failure is recorded, so the caller can give up
  // then rather than at the end of the read.
  //
  // Everything that can fail above is fatal, and the first thing to run is the
  // volume allocation. A field too large for the card therefore FAILS in the
  // first second — but the error used to be read only after the extinction RPC
  // resolved, and that RPC resolves when the worker has sent the whole field.
  // So an 8-gigavoxel run spent several minutes decompressing, transposing and
  // posting slabs into a texture that had never existed, and only then said it
  // was out of memory. The catch is unchanged; what is new is that somebody is
  // listening while it still matters.
  let announceFailure;
  const failed = new Promise((_, reject) => { announceFailure = reject; });
  // The success path never awaits this, and an unobserved rejection is a
  // console error in its own right. A catch here marks it handled without
  // making `failed` itself resolve.
  failed.catch(() => {});

  let chain = Promise.resolve();
  return {
    state,
    failed,
    handle(message) {
      if (message.label !== label) return chain;
      // Read now, while the buffer is still attached: `step` hands it to
      // writeTexture and the ack must survive that.
      const credit = message.type === "slab" ? (message.bytes ?? 0) : 0;
      chain = chain
        .then(() => step(message))
        .catch((err) => { state.error ??= err; announceFailure(err); })
        // Unconditionally, including after a failure. The worker is blocked
        // waiting for this; withholding it because the upload went wrong
        // turns a load that should report an error into one that hangs.
        .then(() => { if (credit) ack(credit); });
      return chain;
    },
    settled: () => chain,
  };
}

/**
 * Read a cloud field out of a local netCDF file and build a Scene from it.
 *
 * `ask` is how the questions io.py asks on a terminal get asked here: which
 * group, whether to nest two of them, and what the condensate units are when
 * the file does not say. None of them are guessed — a field rendered in the
 * wrong units looks entirely plausible and is off by a thousand.
 */
export async function loadFileScene(
  device, file, { ocean, progress, ask, slabBudget } = {},
) {
  const receivers = new Map();
  const link = new WorkerLink((message) => {
    receivers.get(message.label)?.handle(message);
  });

  // A throw anywhere below — a refused nest, a map that never arrived, an
  // ocean that would not load — used to leave whatever had already been
  // allocated to the garbage collector, which is a poor custodian of several
  // gigabytes of video memory. The catch at the end gives it all back: every
  // volume texture is reachable through `receivers` from the moment it is
  // created, and the nest stand-in through this.
  let nestDummy = null;
  try {
    progress("Reading the file structure…", 0.02);

    // Everything the user has settled that detection could not, carried into
    // every later describe: which variable is the cloud water, which is the
    // ice, and which storage axis is x, y and z. One object, because a
    // describe that saw the axis answer but not the variable answer would
    // resolve a different variable's dimensions and could disagree.
    let choice = null;
    // Guesses that WERE made and stuck — stated on screen at the end of the
    // load rather than only in the console (see the toast below). A detection
    // that had to fall back to position is not a failure and does not stop
    // the load, but it is never allowed to be invisible.
    const assumptions = [];

    let opened;
    try {
      opened = await link.call("open", { file });
    } catch (err) {
      // No dead end. If the only thing missing is which dimension is which,
      // ask — the same question shape as the group and units panels — and
      // open the file again with the answer.
      if (!err.axisChoice) throw err;
      const answer = await ask({
        panel: "axes", filename: file.name,
        dims: err.axisChoice.dims, reason: err.message,
      });
      choice = { axes: answer.axes };
      assumptions.push(T.axesByHand(["x", "y", "z"].map((a) =>
        `${a} = ${err.axisChoice.dims.find(
          (d) => d.axis === answer.axes[a]).name}`)));
      opened = await link.call("open", { file, choice });
    }
    const { groups, problems } = opened;
    if (problems?.length) {
      // Kept, not just logged. Offering three of a file's five groups with no
      // explanation is how someone concludes the tool cannot read their data.
      console.warn("cloudyview: groups skipped:\n" + problems.join("\n"));
    }

    // Which level, or which two.
    //
    // Only groups that described fully can be ranked for nesting: a group
    // still waiting on a variable answer has no coordinates yet, so there is
    // no box to compare. It stays on offer as a single level.
    const pairs = nestablePairs(groups.filter((g) => g.coords).map((g) => {
      const { bmin, bmax, spacing } = domainExtent(
        g.coords.x, g.coords.y, g.coords.z);
      return { name: g.path, bmin, bmax, spacing };
    }));

    let chosen = [groups[0]];
    if (groups.length > 1) {
      const answer = await ask({
        panel: "groups", filename: file.name,
        groups: groups.map((g) => g.path), pairs,
      });
      chosen = answer.pair
        ? answer.pair.map((p) => groups.find((g) => g.path === p))
        : [groups.find((g) => g.path === answer.group)];
    }

    // Which variable is the liquid condensate, and which is the ice.
    //
    // Inference gets first go and is silent when it works. A miss is a
    // question — never a refusal, never a guess — and the chooser lists
    // EVERY three-dimensional variable in the group rather than only names
    // the condensate lists already know. That is the whole point: a file
    // whose water is called something else, or a file of temperature with no
    // water at all, is a question this can ask instead of an error it has to
    // report.
    //
    // Asked once and applied to every chosen level, because a nested pair is
    // one model's output written twice and the same variable means the same
    // thing in both. The describe call below is what rejects a name a group
    // really does not have.
    const inferredLiquid = chosen[0].liquidVar;
    // Offered only for a single level: one attached file cannot be the ice
    // for two different grids.
    const mayAttachIce = chosen.length === 1;
    let iceFile = null;

    for (const role of ["liquid", "ice"]) {
      const flag = role === "liquid" ? "needsLiquidChoice" : "needsIceChoice";
      const level = chosen.find((g) => g[flag]);
      if (!level) continue;
      // Composed here rather than in the panel, so that what the load says
      // about itself lives beside the decisions it is reporting.
      const status = role === "liquid"
        ? (level.inferredIce
            ? [T.inferredIce(level.inferredIce), T.noLiquid]
            : [T.noneInferred])
        : [inferredLiquid ? T.inferredLiquid(inferredLiquid) : null, T.noIce]
            .filter(Boolean);
      const answer = await ask({
        panel: "variable", role, filename: file.name, group: level.path,
        title: role === "liquid" ? T.askLiquid : T.askIce,
        variables: level.variables, status,
        offerFile: role === "ice" && mayAttachIce,
      });
      choice = { ...(choice ?? {}) };
      if (role === "liquid") {
        choice.liquidVar = answer.variable;
      } else if (answer.file) {
        // The ice is in a second file: this group has none, and the attach
        // below is what supplies it.
        iceFile = answer.file;
        choice.iceVar = null;
      } else {
        choice.iceVar = answer.variable;         // may be null: "No ice"
      }
      // Re-describe with the answer, so everything downstream — dimensions,
      // coordinates, units, chunking — comes from the variable the user
      // actually picked rather than from the one detection guessed.
      for (let i = 0; i < chosen.length; i++) {
        chosen[i] = await link.call(
          "describe", { group: chosen[i].path, choice });
      }
    }

    // Which timestep. Silent on the single-step files that are most of them;
    // a multi-step file used to be refused outright here.
    const stepped = chosen.find((g) => g.needsTimestepChoice);
    if (stepped) {
      const answer = await ask({
        panel: "timestep", filename: file.name, group: stepped.path,
        timeDim: stepped.timeDim,
      });
      choice = { ...(choice ?? {}), timestep: answer.timestep };
      for (let i = 0; i < chosen.length; i++) {
        chosen[i] = await link.call(
          "describe", { group: chosen[i].path, choice });
      }
    }

    for (const g of chosen) {
      for (const note of g.assumptions ?? []) {
        if (!assumptions.includes(note)) assumptions.push(note);
      }
    }

    // Units, once, covering every condensate variable of every level chosen —
    // ice as well as liquid. A variable whose units the file does not declare
    // is a question, never an inference from its neighbour.
    let units = null;
    const unknown = [];
    for (const g of chosen) {
      if (!g.unitsKnown) unknown.push(g.liquidVar);
      if (g.iceVar && !g.iceUnitsKnown) unknown.push(g.iceVar);
    }
    if (unknown.length) {
      units = (await ask({
        panel: "units", filename: file.name, variables: [...new Set(unknown)],
      })).units;
    }

    // A warm-looking file plus the ice somebody has beside it.
    //
    // Asked HERE, at load, and not when the ice-detection mode is first
    // pressed (Thomas, 2026-08-22). The reason is not interface tidiness: the
    // extinction this load is about to compute INCLUDES ice, so a second file
    // that turned up later would mean the field on screen had been rendered
    // without its ice all along and the fraction would then be measured
    // against a denominator that did not contain it. Attaching at load makes
    // one field out of the two files; attaching afterwards would make two.
    //
    // Skipping is a first-class answer, not a nag dismissed: most fields are
    // warm and have no ice anywhere to attach.
    //
    // Single-level loads only. A nested pair is two grids, so one attached
    // file cannot be the ice for both, and asking twice for a case nobody has
    // yet asked for is a guess about an interface rather than about data.
    //
    // Reached from the ice VARIABLE question rather than from a second panel
    // of its own: "which variable is the ice" and "the ice is in another
    // file" are answers to one question, and asking them one after the other
    // made every warm field answer the same thing twice.
    let iceFrom = null;
    if (iceFile) {
      progress("Reading the ice file's structure…", 0.03);
      const iceOpen = await link.call("openIce", { file: iceFile });
      let iceLevel = iceOpen.groups[0];
      if (iceOpen.groups.length > 1) {
        const pick = await ask({
          panel: "groups", filename: iceFile.name, pairs: [],
          groups: iceOpen.groups.map((g) => g.path),
        });
        iceLevel = iceOpen.groups.find((g) => g.path === pick.group);
      }
      // Pinned at the same step as the field it is joining.
      let iceChoice = choice && "timestep" in choice
        ? { timestep: choice.timestep } : null;
      if (iceLevel.needsIceChoice) {
        const pick = await ask({
          panel: "variable", role: "ice", filename: iceFile.name,
          group: iceLevel.path, title: T.askIce,
          variables: iceLevel.variables, status: [T.noIce],
          offerFile: false,
        });
        if (!pick.variable) {
          throw new Error(T.noIceInFile(iceFile.name));
        }
        iceChoice = { ...(iceChoice ?? {}), iceVar: pick.variable };
        iceLevel = await link.call(
          "describeIce", { group: iceLevel.path, choice: iceChoice });
      }
      let iceUnits = null;
      if (!iceLevel.iceUnitsKnown) {
        iceUnits = (await ask({
          panel: "units", filename: iceFile.name,
          variables: [iceLevel.iceVar],
        })).units;
      }
      // Checked BEFORE anything is read: a grid mismatch is a sentence, not
      // a wasted pass over two multi-gigabyte files. It is checked again in
      // the read itself — see worker.js — because that read happens later
      // and nothing in between keeps the two descriptions in step.
      const attached = await link.call("iceGrid", {
        group: iceLevel.path, choice: iceChoice,
        waterGroup: chosen[0].path, waterChoice: choice,
        filename: iceFile.name,
      });
      iceFrom = {
        group: iceLevel.path, choice: iceChoice,
        units: iceUnits, filename: iceFile.name,
      };
      for (const note of attached.assumptions ?? []) {
        const said = T.inFile(iceFile.name, note);
        if (!assumptions.includes(said)) assumptions.push(said);
      }
      console.info(
        `cloudyview: ice attached from '${iceFile.name}' — ` +
        `'${attached.iceVar}' on the same ${attached.shape.join(" x ")} grid.`);
    }

    for (const level of chosen) {
      if (level.chunks) {
        const chunkVoxels = level.chunks.reduce((a, b) => a * b, 1);
        if (chunkVoxels > 0 && chunkVoxels < 4096) {
          console.warn(
            `cloudyview: '${level.path}' is stored in ${level.chunks} chunks, ` +
            "which is small enough that HDF5's fixed 1 MB chunk cache will " +
            "thrash. Reading will be slow. h5wasm exposes no way to enlarge " +
            "the cache from here.");
        }
      }
    }

    // Outer level first, then the nest.
    const built = [];
    for (const [index, level] of chosen.entries()) {
      const label = level.path || "(root)";
      const share = 0.9 / chosen.length;
      // Two phases now, and they are worth naming separately: the read is the
      // slow one and ends at the crop, the upload is what the old bar showed.
      // Each gets half the level's share so neither appears to stall.
      const receiver = levelReceiver(device, label, (done, n, of, phase) =>
        progress(
          `${phase === "read" ? "Reading" : "Uploading"} ${label} — ` +
          `part ${n} of ${of}` +
          (level.shape ? ` (${level.shape.join(" x ")} cells)` : ""),
          0.05 + share * (index + (phase === "read" ? done : 1 + done) / 2)),
        (bytes) => link.notify("ack", { bytes }));
      receivers.set(label, receiver);
      progress(`Reading ${label}…`, 0.05 + share * index);
      // Whichever comes first: the worker finishing, or the upload failing.
      // Losing the race throws out of here, and the catch below releases what
      // was allocated while `finally` terminates the worker mid-read — which
      // is the whole point, since there is nothing left to receive its output.
      await Promise.race([
        // No ice FRACTION here. It is a second volume the size of this one
        // and most flights never ask for it, so it is read on demand
        // (loadIceVolume) rather than paid for by everybody at load. The ice
        // itself is read now either way — it is part of the extinction.
        link.call("extinction", {
          group: level.path, units, label, slabBudget, choice,
          // Only the outer level. iceFrom is offered for single-level loads
          // (see above), so `index` is 0 whenever it is set; the guard is
          // there so that a later nested case cannot pick it up by accident.
          iceFrom: index === 0 ? iceFrom : null,
        }),
        receiver.failed,
      ]);
      // The RPC resolving means the worker has SENT everything, not that we
      // have finished writing it. Wait for the upload queue to drain, and
      // surface anything it swallowed.
      await receiver.settled();
      if (receiver.state.error) throw receiver.state.error;
      if (receiver.state.slabsDone !== receiver.state.slabs) {
        throw new Error(
          `Only ${receiver.state.slabsDone} of ${receiver.state.slabs} parts ` +
          `of '${label}' reached the GPU. The field would have holes in it.`);
      }
      if (!receiver.state.volume) {
        throw new Error(
          `The occupied extent of '${label}' never arrived, so nothing was ` +
          "ever allocated to put it in.");
      }
      const crop = receiver.state.volume;
      if (crop.zCropped) {
        const [lo, hi] = crop.zCrop;
        const source = receiver.state.geometry.description.shape[2];
        console.info(
          `cloudyview: '${label}' cropped to z ${lo}–${hi} of ${source} ` +
          `(${(100 * (1 - crop.shape[2] / source)).toFixed(0)}% of the ` +
          "vertical held no cloud). The domain box and the march follow the " +
          "crop; the ocean stays at z = 0.");
      }
      if (!receiver.state.albedo) {
        throw new Error(`The overhead map for '${label}' never arrived.`);
      }
      built.push({ level, receiver });
    }

    progress("Loading the ocean surface…", 0.95);
    // The ocean belongs to the session, not to this scene, so it is not in
    // the cleanup below: the caller keeps it across a change of field.
    const oceanTile = await ocean();

    const outer = built[0];
    // The cropped coordinates, not the file's: these are what put bmin.z and
    // bmax.z on the cloud instead of on the top of the model domain, and that
    // shrunken box is where the speed comes from.
    const outerShape = outer.receiver.state.volume.shape;
    const { bmin, bmax } = volumeAABB(
      outer.receiver.state.volume.coords.x,
      outer.receiver.state.volume.coords.y,
      outer.receiver.state.volume.coords.z);

    const scene = new Scene(device, {
      volumeTexture: outer.receiver.state.texture,
      volumeView: outer.receiver.state.texture.createView(),
      shape: outerShape,
      bmin, bmax,
      minVoxelM: minVoxelSize(outerShape, bmin, bmax),
      oceanView: oceanTile.view,
      oceanFifDx: oceanTile.dx,
      oceanTileExtent: oceanTile.tileExtent,
      oceanMaxLod: oceanTile.maxLod,
      // The minimap always shows the OUTER domain; the nest is drawn on it as
      // a footprint rectangle, which is why only this level's map is kept.
      albedo: outer.receiver.state.albedo,
      albedoShape: outer.receiver.state.albedoShape,
      _nest: null,
      _nestDummy: (nestDummy = createNestDummy(device)),
      // A file's domain is NOT known to be periodic. nest_a in a nested
      // run is a 32 km box inside a 2048 km parent; tiling it laterally
      // invents structure that looks exactly like data — a line of
      // identical clouds marching to the horizon. Off unless asked for.
      periodicDefault: false,
      sourceName: file.name,
      // Kept so the behold hand-off can name this group again: a file of
      // several groups renders the wrong field without it.
      groupPath: outer.level.path || null,
      title: outer.level.path ? `group ${outer.level.path}` : null,
      liquidVar: outer.level.liquidVar,
      iceVar: outer.level.iceVar,
      // Ice-detection mode: what it would take to read the ice fraction, not
      // the fraction itself. Null when the field carries no ice variable, and
      // then the viewer says so instead of offering the mode. Holding the
      // File is holding a handle, not the bytes — the same one this load
      // already streamed through.
      // A second file attached at load counts exactly as much as the field's
      // own ice variable: the extinction on screen already includes it, so
      // the mode is available and the fraction is one more pass — over two
      // files instead of one. `kind` is what tells the viewer which loader
      // to use; a demo's prebaked fraction needs no HDF5 reader at all.
      iceSource: (outer.level.iceVar || iceFrom)
        ? { kind: "netcdf", file, iceFile, iceFrom, choice,
            group: outer.level.path, units,
            label: outer.level.path || "(root)" }
        : null,
      // Why the mode is not on offer, when it is not. Distinguishes "this
      // file has no ice" from "you were offered an ice file and skipped",
      // because the second one has an obvious remedy and the first does not.
      iceSkipped: !outer.level.iceVar && !iceFrom && chosen.length === 1,
    });

    // Guesses, stated. Never a silent detection: a field whose axes were
    // taken by position renders a perfectly plausible cloud with x and z
    // swapped, and the only defence is that the person was told.
    scene.assumptions = assumptions;

    if (built.length > 1) {
      const inner = built[1];
      const innerShape = inner.receiver.state.volume.shape;
      const box = volumeAABB(
        inner.receiver.state.volume.coords.x,
        inner.receiver.state.volume.coords.y,
        inner.receiver.state.volume.coords.z);
      // The nest tapers to zero at its edges even in a periodic domain: that
      // taper IS how it blends out into the coarse field around it.
      const report = scene.attachNest({
        texture: inner.receiver.state.texture,
        bmin: box.bmin, bmax: box.bmax,
        minVoxelM: minVoxelSize(innerShape, box.bmin, box.bmax),
        name: inner.level.path,
      });
      if (report.clipped) console.warn(`cloudyview: ${report.clipped}`);
      scene.nestNote = report.clipped;
    }

    // Groups this file has that could not be read. Carried out so the viewer
    // can say so on screen rather than only in the console.
    scene.skipped = problems ?? [];

    progress("Ready.", 1);
    return scene;
  } catch (err) {
    for (const receiver of receivers.values()) {
      receiver.state.texture?.destroy();
      receiver.state.iceTexture?.destroy();
    }
    nestDummy?.destroy();
    throw err;
  } finally {
    link.close();
  }
}

/**
 * Read just the ice-fraction volume of a field already on screen.
 *
 * The deferred half of the ice-detection mode: nothing about it is paid for
 * until somebody asks for it, and the price when they do is one more pass
 * over the same file. The crop is not passed in — the pass re-derives it from
 * the same occupancy test the extinction read used, so the two bands agree by
 * construction and the texture lines up texel for texel with the field it
 * tints.
 *
 * `source` is the scene's `iceSource`. Returns the texture; the caller owns
 * it from then on.
 */
export async function loadIceVolume(device, source, { progress, slabBudget } = {}) {
  const { file, iceFile, iceFrom, choice, group, units, label } = source;
  const receivers = new Map();
  const link = new WorkerLink((message) => {
    receivers.get(message.label)?.handle(message);
  });
  const receiver = levelReceiver(
    device, label,
    (done, n, of, phase) => progress?.(
      phase === "read"
        ? `Reading the ice fraction — part ${Math.round(done * (of || 1))}` +
          ` of ${of || "?"}`
        : `Uploading the ice fraction — part ${n} of ${of}`,
      phase === "read" ? 0.5 * done : 0.5 + 0.5 * done),
    (bytes) => link.notify("ack", { bytes }),
    { iceOnly: true });
  receivers.set(label, receiver);

  try {
    progress?.("Opening the file again…", 0.0);
    // A fresh worker, so the mounts and the h5wasm handles this pass needs
    // are its own — including the attached ice file, which the load-time
    // worker opened and then terminated with itself.
    await link.call("open", { file, choice });
    if (iceFrom) {
      progress?.("Opening the ice file again…", 0.0);
      await link.call("openIce", { file: iceFile });
    }
    await Promise.race([
      link.call("extinction",
                { group, units, label, slabBudget, choice, iceFrom,
                  iceOnly: true }),
      receiver.failed,
    ]);
    await receiver.settled();
    if (receiver.state.error) throw receiver.state.error;
    if (!receiver.state.iceTexture) {
      throw new Error(
        `No ice fraction ever arrived for '${label}', so there is nothing to ` +
        "show. The field's ice variable read as empty.");
    }
    if (receiver.state.iceSlabsDone !== receiver.state.iceSlabs) {
      throw new Error(
        `Only ${receiver.state.iceSlabsDone} of ${receiver.state.iceSlabs} ` +
        `ice-fraction parts of '${label}' reached the GPU. The ice field ` +
        "would have holes in it.");
    }
    progress?.("Ready.", 1);
    return receiver.state.iceTexture;
  } catch (err) {
    receiver.state.iceTexture?.destroy();
    throw err;
  } finally {
    link.close();
  }
}
