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
  Scene, createVolumeTexture, writeVolumeSlab, writeGhostBorder,
  createNestDummy,
} from "../scene.js";
import { volumeAABB, minVoxelSize, domainExtent, nestablePairs } from "../field.js";

// How much uploaded data may be in flight before waiting for the GPU.
const UPLOAD_DRAIN_BYTES = 64 * 1024 * 1024;

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
 */
function levelReceiver(device, label, onProgress, ack) {
  const state = { texture: null, padded: null, faces: null, geometry: null,
                  slabsDone: 0, error: null, queuedBytes: 0 };

  const step = async (message) => {
    if (message.type === "geometry") {
      state.geometry = message;
      state.slabs = message.slabs;
      const [nx, ny, nz] = message.description.shape;
      state.padded = [nx + 2, ny + 2, nz + 2];
      state.texture = await createVolumeTexture(
        device, state.padded, `the field in ${label}`);
    } else if (message.type === "slab") {
      if (!state.texture) {
        throw new Error(
          "A slab arrived before the volume texture existed — the upload " +
          "queue is out of order.");
      }
      writeVolumeSlab(device, state.texture, message.data,
                      message.origin, message.size);
      state.slabsDone += 1;
      state.queuedBytes += message.data.byteLength;

      // queue.writeTexture copies into driver-owned staging memory that is
      // only reclaimed once the GPU has consumed it. Nothing here submits
      // work, so without a barrier every slab of a multi-gigabyte field
      // piles up at once — which on a 3.5 GB variable exhausts system memory
      // and takes the device, and then the browser, down with it. Draining
      // periodically bounds that to roughly one barrier's worth.
      if (state.queuedBytes >= UPLOAD_DRAIN_BYTES) {
        state.queuedBytes = 0;
        await device.queue.onSubmittedWorkDone();
      }
      onProgress?.(message.done, state.slabsDone, state.slabs);
    } else if (message.type === "faces") {
      state.faces = message.faces;
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
  let chain = Promise.resolve();
  return {
    state,
    handle(message) {
      if (message.label !== label) return chain;
      // Read now, while the buffer is still attached: `step` hands it to
      // writeTexture and the ack must survive that.
      const credit = message.type === "slab" ? (message.bytes ?? 0) : 0;
      chain = chain
        .then(() => step(message))
        .catch((err) => { state.error ??= err; })
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
    const { groups, problems } = await link.call("open", { file });
    if (problems?.length) {
      // Kept, not just logged. Offering three of a file's five groups with no
      // explanation is how someone concludes the tool cannot read their data.
      console.warn("cloudyview: groups skipped:\n" + problems.join("\n"));
    }

    // Which level, or which two.
    const pairs = nestablePairs(groups.map((g) => {
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
      const receiver = levelReceiver(device, label, (done, n, of) =>
        progress(
          `Reading ${label} — part ${n} of ${of}` +
          (level.shape ? ` (${level.shape.join(" x ")} cells)` : ""),
          0.05 + share * (index + done)),
        (bytes) => link.notify("ack", { bytes }));
      receivers.set(label, receiver);
      progress(`Reading ${label}…`, 0.05 + share * index);
      await link.call("extinction",
                      { group: level.path, units, label, slabBudget });
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
      if (!receiver.state.faces) {
        throw new Error(`The wrap faces for '${label}' never arrived.`);
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
    const { bmin, bmax } = volumeAABB(
      outer.receiver.state.geometry.coords.x,
      outer.receiver.state.geometry.coords.y,
      outer.receiver.state.geometry.coords.z);

    writeGhostBorder(device, outer.receiver.state.texture,
                     outer.receiver.state.faces, true,
                     outer.receiver.state.padded);

    const scene = new Scene(device, {
      volumeTexture: outer.receiver.state.texture,
      volumeView: outer.receiver.state.texture.createView(),
      padded: outer.receiver.state.padded,
      shape: outer.level.shape,
      bmin, bmax,
      minVoxelM: minVoxelSize(outer.level.shape, bmin, bmax),
      oceanView: oceanTile.view,
      oceanFifDx: oceanTile.dx,
      oceanTileExtent: oceanTile.tileExtent,
      oceanMaxLod: oceanTile.maxLod,
      _faces: outer.receiver.state.faces,
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
    });

    if (built.length > 1) {
      const inner = built[1];
      const box = volumeAABB(
        inner.receiver.state.geometry.coords.x,
        inner.receiver.state.geometry.coords.y,
        inner.receiver.state.geometry.coords.z);
      // The nest keeps a zero border even in a periodic domain: that taper
      // IS how it blends out into the coarse field at its own edges.
      const report = scene.attachNest({
        texture: inner.receiver.state.texture,
        bmin: box.bmin, bmax: box.bmax,
        minVoxelM: minVoxelSize(inner.level.shape, box.bmin, box.bmax),
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
    }
    nestDummy?.destroy();
    throw err;
  } finally {
    link.close();
  }
}
