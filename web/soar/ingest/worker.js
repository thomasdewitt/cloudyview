// The ingest worker.
//
// It has to be a worker: Emscripten's filesystem is synchronous, and browsers
// forbid synchronous file access on the main thread. WORKERFS is what makes
// this a viewer rather than an uploader — libhdf5 reads the chunks it needs
// straight out of the File, so a 40 GB run costs a few tens of megabytes of
// wasm heap and never leaves the machine.
//
// Measured 2026-08-04: a 537 MB chunked netCDF-4, mid-file sub-box read in
// 4 ms, wasm heap flat at 19 MB across mount, open and slice, in both Firefox
// and Chrome, with no cross-origin isolation.

"use strict";

import * as h5wasm from "../vendor/h5wasm/hdf5_hl.js";
import {
  describeGroup, findLiquidWaterGroups, decoderFor, unitsMultiplier,
  attrString,
} from "./netcdf.js";
import {
  rhoAirTable, sigmaAt, cellThickness, opticalDepthFromWaterPaths,
  twoStreamAlbedo, iceExtinctionFraction,
} from "../optical.js";
import {
  pluginsNeeded, installPlugins, assertFiltersSupported,
} from "./filters.js";
import { makeHalfWriter } from "../half.js";
import { markOccupiedPlanes, occupiedBand } from "../zcrop.js";

const MOUNT = "/local";
// Bytes of decoded source per slab, per variable.
//
// Kept modest because libhdf5 runs in a 32-bit wasm heap and a filtered read
// needs a whole-chunk scratch buffer besides. (An earlier comment here
// blamed this budget for a field that came back all zeros. That was wrong —
// the cause was an unregistered compression filter; see filters.js. The
// smaller budget is still the right default, but it fixed nothing.)
//
// 64 MB, not the 16 MB it was: below one whole chunk the budget stops being
// modest and starts being expensive. A STEAM parent chunked 128x64x1000
// float32 is 32.8 MB, so 16 MB split z in half and every chunk was inflated
// TWICE per variable — 134 GB of decompression to deliver 67. 64 MB fits that
// chunk whole with room to grow z, and keeps the fp16 slab it produces at
// 16.4 MB, under MAX_OUTSTANDING_BYTES, so two are still in flight at once.
const SLAB_BUDGET_BYTES = 64 * 1024 * 1024;

// The most a single chunk may drag the budget up to (see `chunkFloor` below).
// A file chunked larger than this gets split and re-inflated, because a
// 32-bit heap holding two of these plus libhdf5's own scratch is the failure
// that takes the whole worker with it.
const SLAB_BUDGET_CEILING_BYTES = 256 * 1024 * 1024;

let ready = null;
let root = null;
let mounted = false;

// --- helpers ---------------------------------------------------------------

const strictlyDescending = (c) => {
  for (let i = 1; i < c.length; i++) if (!(c[i] - c[i - 1] < 0)) return false;
  return c.length > 1;
};

function post(message, transfer) {
  self.postMessage(message, transfer || []);
}

// --- backpressure ----------------------------------------------------------

// Bytes of slab data allowed to be posted but not yet consumed.
//
// postMessage does not block and does not care whether anyone is listening.
// The main thread consumes a slab, uploads it, and periodically waits for the
// GPU; while it waits, and while the tab is in the background and throttled,
// this loop would happily decode and post the rest of the field. Each posted
// slab keeps its transferred buffer alive in the browser's message queue, so
// on a multi-gigabyte field "switch to another app during the load" became
// "the whole volume is resident in the message queue" — and then neither the
// content process nor the GPU process survives it. Kept below the consumer's
// own drain threshold so the pipeline still overlaps.
const MAX_OUTSTANDING_BYTES = 32 * 1024 * 1024;

let outstandingBytes = 0;
let creditWaiter = null;

/** The main thread has finished with `bytes` worth of slab. */
function returnCredit(bytes) {
  outstandingBytes = Math.max(0, outstandingBytes - bytes);
  if (creditWaiter && outstandingBytes < MAX_OUTSTANDING_BYTES) {
    const resume = creditWaiter;
    creditWaiter = null;
    resume();
  }
}

/**
 * Wait until there is room for another slab, then account for it.
 * One producer, so one waiter — there is never a queue of these.
 */
async function spendCredit(bytes) {
  while (outstandingBytes >= MAX_OUTSTANDING_BYTES) {
    await new Promise((resolve) => { creditWaiter = resolve; });
  }
  outstandingBytes += bytes;
}

/** Forget any outstanding accounting — a fresh read starts from zero. */
function resetCredit() {
  outstandingBytes = 0;
  const resume = creditWaiter;
  creditWaiter = null;
  resume?.();
}

async function ensureReady() {
  ready ??= h5wasm.ready;
  return ready;
}

/** How many elements a hyperslab request should return. */
function slabVoxels(storageShape, ranges) {
  let n = 1;
  for (let i = 0; i < storageShape.length; i++) {
    const r = ranges[i];
    n *= (!r || r.length === 0) ? storageShape[i] : (r[1] - r[0]);
  }
  return n;
}

/**
 * Read a hyperslab and insist it is the size that was asked for.
 *
 * libhdf5 in wasm can return a short or empty buffer rather than failing
 * when a read does not fit the heap. Silently accepting that produces a
 * field of zeros, which renders as a clear sky and reads as "the tool does
 * not work" rather than "the read failed".
 */
function readSlice(dataset, ranges, expected, name) {
  const values = dataset.slice(ranges);
  if (!values || values.length !== expected) {
    throw new Error(
      `Reading '${name}' returned ${values ? values.length : 0} values where ` +
      `${expected} were requested. The HDF5 reader could not satisfy a read ` +
      "this large in the browser's memory.");
  }
  return values;
}

let installedPlugins = new Set();

/** Walk every 3-D dataset's filter list and fetch the plugins it implies. */
async function installNeededPlugins() {
  const module = await ensureReady();
  const filterLists = [];
  const visit = (group) => {
    for (const key of group.keys()) {
      const item = group.get(key);
      if (!item) continue;
      if (item.type === "Group") { visit(item); continue; }
      if (item.shape?.length >= 3 && item.filters) filterLists.push(item.filters);
    }
  };
  visit(root);
  const { plugins, unknown } = pluginsNeeded(filterLists);
  if (unknown !== null) {
    console.warn(
      `cloudyview: unrecognized HDF5 filter id ${unknown}; loading every ` +
      "decompression plugin rather than guessing.");
  }
  if (plugins.length) {
    installedPlugins = await installPlugins(module, plugins, installedPlugins);
  }
}

// --- operations ------------------------------------------------------------

async function open({ file }) {
  const { FS } = await ensureReady();
  if (!mounted) {
    try { FS.mkdir(MOUNT); } catch { /* already there */ }
    FS.mount(FS.filesystems.WORKERFS, { files: [file] }, MOUNT);
    mounted = true;
  }
  root = new h5wasm.File(`${MOUNT}/${file.name}`, "r");

  // Compression filters first: reading a dataset whose filter is missing
  // does not fail, it returns zeros. Opening the file is enough to see which
  // filters it uses — that is metadata, and costs no decompression.
  await installNeededPlugins();

  const paths = findLiquidWaterGroups(root);
  if (!paths.length) {
    throw new Error(
      "No cloud water field in this file. cloudyview looks for a variable " +
      "named one of: qc, QC, ql, QL, LWC, clw, cloud_liquid_water_mixing_" +
      "ratio, liquid_water_content, lwc — with three spatial dimensions.");
  }

  const groups = [];
  const problems = [];
  for (const path of paths) {
    try {
      groups.push(describeGroup(root, path));
    } catch (err) {
      // A group with unusable coordinates just drops out of the list, but
      // say so — silently offering fewer choices is how people conclude the
      // tool cannot read their file.
      problems.push(`${path || "(root)"}: ${err.message}`);
    }
  }
  if (!groups.length) {
    throw new Error(
      `Found cloud water, but no group could be read.\n${problems.join("\n")}`);
  }
  return { groups, problems, filename: file.name };
}

/**
 * Stream one group's extinction field to the main thread, laid out for the
 * texture.
 *
 * The transpose happens here rather than on the GPU. netCDF stores (z, y, x)
 * with x fastest; the texture wants z fastest. r16float is not a storage
 * texture format and copyBufferToTexture's 256-byte row rule does not fit an
 * nz*2 row, so a compute path would cost either double the memory or a lot of
 * padding. queue.writeTexture has no such rule, and the per-voxel work is a
 * multiply-add against a precomputed rho_air(z) — no exp per voxel.
 *
 * `iceOnly` runs the same read for the ice-fraction volume alone (the
 * ice-detection mode, loaded on demand rather than with the field). It posts
 * only the ice slabs and no minimap, but still computes sigma per voxel and
 * still derives the z crop from it — so the band it posts into is the same
 * band the extinction pass established, by construction rather than by a
 * remembered number the two passes could disagree about.
 */
async function extinction({ group, units, label, slabBudget,
                            iceOnly = false }) {
  const description = describeGroup(root, group);
  const handle = group ? root.get(group) : root;
  const dataset = handle.get(description.liquidVar);
  const iceDataset = description.iceVar ? handle.get(description.iceVar) : null;
  // Ice-detection mode: a volume of per-voxel ice extinction fraction, which
  // only the on-demand pass produces. The extinction itself always includes
  // ice, whoever is reading. The caller checks for the variable before it
  // gets here, so an ice-only pass over a group without one is a bug rather
  // than a fact about the file.
  if (iceOnly && !iceDataset) {
    throw new Error(
      `'${label}' has no ice mixing ratio variable, so there is no ice ` +
      "fraction to read.");
  }

  // Each condensate variable takes its OWN declared units, falling back only
  // to what the user was asked for — never to the other variable's. An ice
  // field silently given the liquid field's multiplier is wrong by a factor
  // of a thousand and renders as a perfectly plausible sky.
  const multiplierFor = (declared, name) => {
    const m = unitsMultiplier(declared) ?? unitsMultiplier(units ?? null);
    if (m === null) throw new Error(`No units on ${name} and none supplied.`);
    return m;
  };
  const multiplier = multiplierFor(description.units, description.liquidVar);
  const iceMultiplier = iceDataset
    ? multiplierFor(attrString(iceDataset.attrs, "units"), description.iceVar)
    : 0.0;

  assertFiltersSupported(dataset.filters, installedPlugins, description.liquidVar);
  if (iceDataset) {
    assertFiltersSupported(
      iceDataset.filters, installedPlugins, description.iceVar);
  }

  const decode = decoderFor(dataset.attrs);
  const decodeIce = iceDataset ? decoderFor(iceDataset.attrs) : null;

  const [nx, ny, nz] = description.shape;
  const axis = description.storageAxis;         // field axis -> storage axis
  const storageShape = description.storageShape;

  // Ascending coordinates, and the flip that gets us there.
  const flip = {};
  const coords = {};
  for (const a of ["x", "y", "z"]) {
    const values = Float64Array.from(description.coords[a]);
    flip[a] = strictlyDescending(values);
    coords[a] = flip[a] ? values.slice().reverse() : values;
  }

  // rho_air is a function of height alone, so it is a table, not a per-voxel
  // exp. Fixed isothermal atmosphere, matching optical_depth.py exactly.
  const rhoAir = rhoAirTable(coords.z);

  // The minimap's column integrals, accumulated on the way past.
  //
  // glimpse does not integrate sigma — it integrates water paths and applies
  // empirical path-to-tau relations, which weight ice nearly twice as heavily
  // as the extinction volume does. So the map cannot be derived from the
  // texture; it has to come from lwc and iwc directly, which is free here and
  // a second full read of the file anywhere else. float32 accumulators, like
  // numpy's, over a few hundred same-magnitude terms.
  const dz = cellThickness(coords.z);
  const lwpColumn = new Float32Array(nx * ny);
  const iwpColumn = new Float32Array(nx * ny);

  // Read in TILES aligned to the file's own HDF5 chunking.
  //
  // A hyperslab that covers part of a chunk still costs the whole chunk's
  // decompression. Slabbing one axis at a time therefore re-decompresses the
  // same chunk once per slab that crosses it: on a 2048x2048x211 field with
  // 128x64x211 chunks, a four-plane slab decompresses ~221 MB to deliver 8,
  // and does it again 31 more times as x advances. Tiling on chunk
  // boundaries reads each chunk exactly once.
  const fieldExtent = { x: nx, y: ny, z: nz };
  const spatialVoxels = nx * ny * nz;

  const chunkExtent = {};
  for (const a of ["x", "y", "z"]) {
    const c = description.chunks?.[axis[a]];
    chunkExtent[a] = Math.max(1, Math.min(c || fieldExtent[a], fieldExtent[a]));
  }
  // The budget's floor is one whole chunk, whatever was asked for. HDF5 has
  // to inflate the entire chunk to satisfy any part of it, so a budget under
  // that size buys no memory — the scratch buffer is the same either way —
  // and costs a re-inflation of every chunk for every piece it was split
  // into. The ceiling is what stops a pathologically-chunked file from
  // turning that floor into a heap exhaustion.
  //
  // Only for a CHUNKED dataset. A contiguous one has no chunk to inflate:
  // libhdf5 reads the hyperslab and nothing else, so the floor would buy
  // nothing and would ask for the whole variable in one read.
  const chunkFloor = description.chunks
    ? chunkExtent.x * chunkExtent.y * chunkExtent.z * 4 : 0;
  const budget = Math.min(
    Math.max(slabBudget || SLAB_BUDGET_BYTES, chunkFloor),
    SLAB_BUDGET_CEILING_BYTES);
  const tile = { x: chunkExtent.x, y: chunkExtent.y, z: chunkExtent.z };
  const tileBytes = () => tile.x * tile.y * tile.z * 4;
  // A single chunk larger than the budget has to be split — halve the longest
  // side until it fits, which at least keeps the pieces compact.
  while (tileBytes() > budget) {
    const a = ["x", "y", "z"].reduce((p, q) => (tile[q] > tile[p] ? q : p));
    if (tile[a] <= 1) break;
    tile[a] = Math.ceil(tile[a] / 2);
  }
  // Otherwise take whole extra chunks while they still fit. z first: it is
  // the fastest axis in the output, so growing it lengthens the runs written
  // per row rather than fragmenting them.
  for (let grew = true; grew; ) {
    grew = false;
    for (const a of ["z", "y", "x"]) {
      if (tile[a] >= fieldExtent[a]) continue;
      const next = Math.min(fieldExtent[a], tile[a] + chunkExtent[a]);
      if ((tileBytes() / tile[a]) * next <= budget) { tile[a] = next; grew = true; }
    }
  }

  const tileCount = Math.ceil(nx / tile.x) * Math.ceil(ny / tile.y)
                  * Math.ceil(nz / tile.z);

  let finiteNonZero = 0;
  let nonFinite = 0;

  // Which z planes hold anything at all, and the slabs waiting on the answer.
  //
  // A cloud field is mostly empty sky: measured over the demo set, 8% of the
  // z extent is vacuum on a STEAM parent, 19% on TWP-ICE LPT, 35% on the FIF
  // cascade, 40% on CM1, and 75% on DYCOMS, whose deck occupies 137 of 531
  // levels. Sizing the volume texture to the file's z extent pays for all of
  // it — in memory, and in a march that crosses the vacuum sample by sample
  // because nothing tells it there is nothing there. Cropping to the occupied
  // band costs one comparison per texel here and returned 3.6x on the DYCOMS
  // source at high tier (v8 4.47 -> 1.23 ms), plus 3.8x of its memory.
  //
  // The band cannot be known before the last tile is read, so the texture
  // cannot be allocated before then either, and the slabs are held until it
  // exists. That is the one place soar keeps a whole field in host memory,
  // and it is bounded by the same budget that decides the field is loadable
  // at all: a volume that fits on the card as r16float fits in the RAM of any
  // machine that has such a card, and one that does not fit is exactly the
  // case this crop exists to rescue.
  const zOccupied = new Uint8Array(nz);
  const staged = [];

  post({ type: "geometry", label, description,
         coords: { x: Array.from(coords.x), y: Array.from(coords.y),
                   z: Array.from(coords.z) },
         flip,
         // So the caller can say "part 3 of 512" rather than showing a bar
         // that sits still for ten seconds at a time.
         slabs: tileCount,
         tile: [tile.x, tile.y, tile.z],
         chunk: [chunkExtent.x, chunkExtent.y, chunkExtent.z],
         voxels: spatialVoxels });

  const AXES = ["x", "y", "z"];
  let tilesDone = 0;
  for (let x0 = 0; x0 < nx; x0 += tile.x) {
  for (let y0 = 0; y0 < ny; y0 += tile.y) {
  for (let z0 = 0; z0 < nz; z0 += tile.z) {
    const base = { x: x0, y: y0, z: z0 };
    const local = {
      x: Math.min(tile.x, nx - x0),
      y: Math.min(tile.y, ny - y0),
      z: Math.min(tile.z, nz - z0),
    };

    // Field indices map to storage indices through the flip, which reverses
    // the range as well as the direction.
    const ranges = storageShape.map(() => []);
    for (const dropped of description.droppedAxes) ranges[dropped] = [0, 1];
    const rangeStart = {};
    for (const a of AXES) {
      const n = fieldExtent[a];
      const lo = flip[a] ? n - (base[a] + local[a]) : base[a];
      ranges[axis[a]] = [lo, lo + local[a]];
      rangeStart[a] = lo;
    }

    const expected = slabVoxels(storageShape, ranges);
    const raw = readSlice(dataset, ranges, expected, description.liquidVar);
    const rawIce = iceDataset
      ? readSlice(iceDataset, ranges, expected, description.iceVar) : null;

    const slabShape = storageShape.slice();
    for (const a of AXES) slabShape[axis[a]] = local[a];
    for (const dropped of description.droppedAxes) slabShape[dropped] = 1;
    const strides = new Array(slabShape.length);
    let acc = 1;
    for (let i = slabShape.length - 1; i >= 0; i--) {
      strides[i] = acc; acc *= slabShape[i];
    }

    const out = makeHalfWriter(local.x * local.y * local.z);
    const outIce = iceOnly
      ? makeHalfWriter(local.x * local.y * local.z) : null;
    const idx = new Array(slabShape.length).fill(0);
    let o = 0;
    for (let lx = 0; lx < local.x; lx++) {
      const gx = base.x + lx;
      idx[axis.x] = (flip.x ? nx - 1 - gx : gx) - rangeStart.x;
      for (let ly = 0; ly < local.y; ly++) {
        const gy = base.y + ly;
        idx[axis.y] = (flip.y ? ny - 1 - gy : gy) - rangeStart.y;
        const column = gx * ny + gy;
        for (let lz = 0; lz < local.z; lz++) {
          const gz = base.z + lz;
          idx[axis.z] = (flip.z ? nz - 1 - gz : gz) - rangeStart.z;

          let flat = 0;
          for (let i = 0; i < idx.length; i++) flat += idx[i] * strides[i];

          let q = raw[flat];
          if (decode) q = decode(q);
          let qi = 0.0;
          if (rawIce) {
            qi = rawIce[flat];
            if (decodeIce) qi = decodeIce(qi);
          }
          const lwc = q * multiplier;
          const iwc = qi * iceMultiplier;
          const sigma = sigmaAt(lwc, iwc, rhoAir[gz]);
          if (Number.isFinite(sigma)) { if (sigma !== 0) finiteNonZero += 1; }
          else nonFinite += 1;
          if (outIce) outIce.set(o, iceExtinctionFraction(lwc, iwc));
          out.set(o++, sigma);

          const thickness = rhoAir[gz] * dz[gz];
          lwpColumn[column] += lwc * thickness;
          iwpColumn[column] += iwc * thickness;
        }
      }
    }
    const bytes = out.bytes();
    // Occupancy is read off the STORED fp16, never off the f64 sigma. The
    // crop has to mean "every plane outside this band is exactly zero in the
    // texture", and a sigma small enough to flush to fp16 zero is zero as far
    // as the renderer is ever going to know. Extinction is non-negative, so
    // the only zero bit pattern in play is 0x0000.
    //
    // z is the fastest axis of `out` — o = (lx * local.y + ly) * local.z + lz
    // — so the plane a texel belongs to is its index modulo the tile depth.
    markOccupiedPlanes(bytes, z0, local.z, zOccupied);
    // The occupancy above is the last thing an ice-only pass wants sigma for;
    // holding on to the buffer as well would double the staged heap to carry
    // slabs nobody is going to send.
    staged.push({ x0, y0, z0, local: { ...local },
                  bytes: iceOnly ? null : bytes,
                  iceBytes: outIce ? outIce.bytes() : null });
    post({ type: "read", label, done: (++tilesDone) / tileCount });
  } } }

  // A field that is entirely zero apart from a scattering of infinities is
  // not a cloud field, it is a failed read. Say so instead of rendering it.
  // Checked here, before anything is sized to the result: a crop computed
  // from a failed read would be a second, more confusing error.
  if (finiteNonZero === 0) {
    throw new Error(
      `'${description.liquidVar}' read as ${spatialVoxels} values that are ` +
      `all zero (plus ${nonFinite} non-finite). That is a failed read rather ` +
      "than an empty sky — the HDF5 reader returned buffers it never filled.");
  }

  // The occupied band, and the crop that follows from it. finiteNonZero > 0
  // got us past the check above, so a throw here means a real field whose
  // every value flushed to fp16 zero — which occupiedBand says better than a
  // repeat of it would.
  const { lo: zLo, hi: zHi, count: zCount, cropped } = occupiedBand(zOccupied);

  // Coordinates follow the crop; everything downstream derives the domain box
  // from them, so this is what actually moves bmin.z/bmax.z onto the cloud.
  const zCoords = Array.from(coords.z.slice(zLo, zHi + 1));
  const keptSlabs = staged.reduce(
    (n, s) => n + (s.z0 + s.local.z > zLo && s.z0 <= zHi ? 1 : 0), 0);
  post({
    type: "volume", label,
    shape: [nx, ny, zCount],
    zCrop: [zLo, zHi],
    zCropped: cropped,
    coords: { x: Array.from(coords.x), y: Array.from(coords.y), z: zCoords },
    // Only the slabs that survive the crop are sent, so this is the count the
    // receiver checks against — not the tile count of the read.
    slabs: iceOnly ? 0 : keptSlabs,
    // Ice-fraction slabs ride alongside, same tiling, tagged field: "ice".
    iceSlabs: iceOnly ? keptSlabs : 0,
  });

  // z is the fastest axis, so a z sub-range is a stride-copy of runs.
  const trimZ = (bytes, slab, lo, depth) => {
    if (depth === slab.local.z) return bytes;
    const trimmed = new Uint16Array(slab.local.x * slab.local.y * depth);
    const offset = lo - slab.z0;
    let o = 0;
    for (let c = 0; c < slab.local.x * slab.local.y; c++) {
      const src = c * slab.local.z + offset;
      for (let k = 0; k < depth; k++) trimmed[o++] = bytes[src + k];
    }
    return trimmed;
  };

  // Post what the crop kept, clipping the tiles that straddle its edges.
  let sent = 0;
  const sendable = staged.length;
  for (let i = 0; i < sendable; i++) {
    const slab = staged[i];
    staged[i] = null;                     // release as we go, not at the end
    const lo = Math.max(slab.z0, zLo);
    const hi = Math.min(slab.z0 + slab.local.z - 1, zHi);
    if (hi < lo) continue;
    const depth = hi - lo + 1;
    sent += 1;
    const fields = iceOnly ? [] : ["sigma"];
    if (slab.iceBytes) fields.push("ice");
    for (const field of fields) {
      const bytes = trimZ(field === "ice" ? slab.iceBytes : slab.bytes,
                          slab, lo, depth);
      // Read before the transfer: posting detaches the buffer, and the ack
      // that comes back has to be able to name the same number.
      const slabBytes = bytes.byteLength;
      await spendCredit(slabBytes);
      // Voxel i is texel i — nothing is padded — and z is measured from the
      // crop's floor rather than the file's.
      post({
        type: "slab", label, field,
        origin: [lo - zLo, slab.y0, slab.x0],
        size: [depth, slab.local.y, slab.local.x],  // texture is (w=z, h=y, d=x)
        done: sent / sendable,
        bytes: slabBytes,
        data: bytes,
      }, [bytes.buffer]);
    }
  }
  // Nothing more is read for the domain edges. A doubly periodic field used
  // to need its four opposite faces uploaded as a ghost ring so that
  // filtering across the wrap seam saw the far side; the renderer now wraps
  // in the sampler (raymarch.wgsl sample_level), which is the same fetch
  // without the four extra hyperslabs, the extra texels, or the two hosts
  // that had to agree on where the faces went.

  // The minimap image, in glimpse's orientation: (ny, nx) with east to the
  // right and north up. The read already delivered ascending coordinates, so
  // glimpse's conditional flips are behind us and only the transpose is left.
  // An ice-only pass is a second read of a field already on screen, minimap
  // and all; rebuilding the same image to throw it away is the one piece of
  // work it can honestly skip.
  if (!iceOnly) {
    const albedo = new Float32Array(ny * nx);
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        albedo[iy * nx + ix] = twoStreamAlbedo(opticalDepthFromWaterPaths(
          lwpColumn[ix * ny + iy], iwpColumn[ix * ny + iy]));
      }
    }
    post({ type: "map", label, shape: [ny, nx], data: albedo }, [albedo.buffer]);
  }

  return { label, shape: [nx, ny, zCount], zCrop: [zLo, zHi], sourceZ: nz };
}

/**
 * Read raw values at given STORAGE indices, with no interpretation at all —
 * no unit conversion, no transpose, no flips, no extinction. Purely "what
 * does libhdf5 hand back for this element". If these disagree with the same
 * indices in Python, the fault is under us; if they agree, it is ours.
 */
async function probe({ group, variable, points, box }) {
  const handle = group ? root.get(group) : root;
  const dataset = handle.get(variable);
  assertFiltersSupported(dataset.filters, installedPlugins, variable);
  const shape = dataset.shape;
  const values = [];
  for (const point of points) {
    const ranges = point.map((i) => [i, i + 1]);
    const got = dataset.slice(ranges);
    values.push({ index: point, value: got && got.length ? got[0] : null,
                  returned: got ? got.length : 0 });
  }
  let boxReport = null;
  if (box) {
    const got = dataset.slice(box.map(([a, b]) => [a, b]));
    let sum = 0, nonzero = 0, nonFinite = 0;
    for (let i = 0; i < got.length; i++) {
      const v = got[i];
      if (!Number.isFinite(v)) { nonFinite += 1; continue; }
      sum += v; if (v !== 0) nonzero += 1;
    }
    const expected = box.reduce((n, [a, b]) => n * (b - a), 1);
    boxReport = { box, returned: got.length, expected, sum, nonzero, nonFinite,
                  first: got[0] ?? null, last: got[got.length - 1] ?? null };
  }
  return { variable, shape, dtype: dataset.dtype,
           chunks: dataset.metadata?.chunks ?? null, values, box: boxReport };
}

self.onmessage = async (event) => {
  // Acks carry no id and expect no reply: they are the consumer saying it has
  // room for more, which is the only thing keeping the loop above in step
  // with a main thread that may be throttled or waiting on the GPU.
  if (event.data?.op === "ack") { returnCredit(event.data.bytes); return; }

  const { id, op, ...args } = event.data;
  try {
    let result;
    if (op === "open") result = await open(args);
    else if (op === "extinction") { resetCredit(); result = await extinction(args); }
    else if (op === "probe") result = await probe(args);
    else throw new Error(`unknown ingest operation '${op}'`);
    post({ id, ok: true, result });
  } catch (err) {
    post({ id, ok: false, error: String(err?.message || err),
           advice: err?.advice || "" });
  }
};
