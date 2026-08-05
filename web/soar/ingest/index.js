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

  close() { this.worker.terminate(); }
}

/**
 * Upload one level's extinction field, slab by slab as the worker produces
 * it, so the JS heap never holds the whole volume.
 */
function levelReceiver(device, label, onProgress) {
  const state = { texture: null, padded: null, faces: null, geometry: null,
                  pending: Promise.resolve() };
  return {
    state,
    async handle(message) {
      if (message.label !== label) return;
      if (message.type === "geometry") {
        state.geometry = message;
        const [nx, ny, nz] = message.description.shape;
        state.padded = [nx + 2, ny + 2, nz + 2];
        state.texture = await createVolumeTexture(
          device, state.padded, `the field in ${label}`);
      } else if (message.type === "slab") {
        writeVolumeSlab(device, state.texture, message.data,
                        message.origin, message.size);
        onProgress?.(message.done);
      } else if (message.type === "faces") {
        state.faces = message.faces;
      }
    },
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
export async function loadFileScene(device, file, { ocean, progress, ask }) {
  const receivers = new Map();
  const link = new WorkerLink((message) => {
    receivers.get(message.label)?.handle(message);
  });

  try {
    progress("Reading the file structure…", 0.02);
    const { groups, problems } = await link.call("open", { file });
    if (problems?.length) {
      console.warn("cloudyview: groups skipped:\n" + problems.join("\n"));
    }

    // Which level, or which two.
    const pairs = nestablePairs(groups.map((g) => {
      const { bmin, bmax, dx } = domainExtent(
        g.coords.x, g.coords.y, g.coords.z);
      return { name: g.path, bmin, bmax, dx };
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

    // Units, once, covering every level chosen.
    let units = null;
    const unknown = chosen.filter((g) => !g.unitsKnown).map((g) => g.liquidVar);
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
      const receiver = levelReceiver(device, label, (done) =>
        progress(`Reading ${label}…`, 0.05 + share * (index + done)));
      receivers.set(label, receiver);
      progress(`Reading ${label}…`, 0.05 + share * index);
      await link.call("extinction", { group: level.path, units, label });
      built.push({ level, receiver });
    }

    progress("Loading the ocean surface…", 0.95);
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
      _nest: null,
      _nestDummy: createNestDummy(device),
      sourceName: file.name,
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

    progress("Ready.", 1);
    return scene;
  } finally {
    link.close();
  }
}
