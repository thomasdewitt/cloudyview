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
import { rhoAirTable, sigmaAt } from "../optical.js";
import {
  pluginsNeeded, installPlugins, assertFiltersSupported,
} from "./filters.js";

const MOUNT = "/local";
// Bytes of decoded source per slab, per variable.
//
// Kept modest because libhdf5 runs in a 32-bit wasm heap and a filtered read
// needs a whole-chunk scratch buffer besides. (An earlier comment here
// blamed this budget for a field that came back all zeros. That was wrong —
// the cause was an unregistered compression filter; see filters.js. The
// smaller budget is still the right default, but it fixed nothing.)
const SLAB_BUDGET_BYTES = 16 * 1024 * 1024;

let ready = null;
let root = null;
let mounted = false;

// --- fp16 ------------------------------------------------------------------

const HAS_F16 = typeof Float16Array !== "undefined";
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/** IEEE binary32 to binary16, with the same round-to-nearest-even numpy uses. */
function toHalf(value) {
  _f32[0] = value;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf / NaN
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;                            // overflow
  if (e <= 0) {
    if (e < -10) return sign;                                     // underflow
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    if ((mant >>> (shift - 1)) & 1) half += 1;                    // round
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  if ((mant >>> 12) & 1) half += 1;
  return sign | half;
}

function makeHalfWriter(length) {
  if (HAS_F16) {
    const view = new Float16Array(length);
    return { store: view, set: (i, v) => { view[i] = v; },
             bytes: () => new Uint16Array(view.buffer) };
  }
  const view = new Uint16Array(length);
  return { store: view, set: (i, v) => { view[i] = toHalf(v); },
           bytes: () => view };
}

// --- helpers ---------------------------------------------------------------

const strictlyDescending = (c) => {
  for (let i = 1; i < c.length; i++) if (!(c[i] - c[i - 1] < 0)) return false;
  return c.length > 1;
};

function post(message, transfer) {
  self.postMessage(message, transfer || []);
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
 * Stream one group's extinction field to the main thread, ghost-padded and
 * laid out for the texture.
 *
 * The transpose happens here rather than on the GPU. netCDF stores (z, y, x)
 * with x fastest; the texture wants z fastest. r16float is not a storage
 * texture format and copyBufferToTexture's 256-byte row rule does not fit an
 * (nz+2)*2 row, so a compute path would cost either double the memory or a
 * lot of padding. queue.writeTexture has no such rule, and the per-voxel work
 * is a multiply-add against a precomputed rho_air(z) — no exp per voxel.
 */
async function extinction({ group, units, label, slabBudget }) {
  const description = describeGroup(root, group);
  const handle = group ? root.get(group) : root;
  const dataset = handle.get(description.liquidVar);
  const iceDataset = description.iceVar ? handle.get(description.iceVar) : null;

  const multiplier = unitsMultiplier(
    description.units ?? units ?? null) ?? unitsMultiplier(units);
  if (multiplier === null) {
    throw new Error(
      `No units on ${description.liquidVar} and none supplied.`);
  }
  const iceMultiplier = iceDataset
    ? (unitsMultiplier(attrString(iceDataset.attrs, "units")) ??
       unitsMultiplier(units) ?? multiplier)
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

  // Chunk along the slowest SPATIAL storage axis — the contiguous direction
  // on disk, so each read is chunk-aligned rather than striped across the
  // file. Storage axis 0 is usually the time dimension, which has already
  // been dropped and is not a field axis at all.
  const fieldExtent = { x: nx, y: ny, z: nz };
  const chunkField = ["x", "y", "z"]
    .map((a) => [a, axis[a]])
    .sort((p, q) => p[1] - q[1])[0][0];
  const chunkAxis = axis[chunkField];
  const spatialVoxels = nx * ny * nz;
  const perIndexBytes = (spatialVoxels / fieldExtent[chunkField]) * 4;
  const chunkLength = Math.max(
    1, Math.min(storageShape[chunkAxis],
                Math.floor((slabBudget || SLAB_BUDGET_BYTES)
                            / Math.max(perIndexBytes, 1))));

  let finiteNonZero = 0;
  let nonFinite = 0;

  const total = fieldExtent[chunkField];
  post({ type: "geometry", label, description,
         coords: { x: Array.from(coords.x), y: Array.from(coords.y),
                   z: Array.from(coords.z) },
         flip,
         // So the caller can say "slab 3 of 11" rather than showing a bar
         // that sits still for ten seconds at a time.
         slabs: Math.ceil(total / chunkLength),
         voxels: spatialVoxels });
  for (let start = 0; start < total; start += chunkLength) {
    const stop = Math.min(start + chunkLength, total);
    // Field indices map to storage indices through the flip.
    const storageRange = flip[chunkField]
      ? [total - stop, total - start]
      : [start, stop];

    const ranges = storageShape.map(() => []);
    for (const dropped of description.droppedAxes) ranges[dropped] = [0, 1];
    ranges[chunkAxis] = storageRange;

    const expected = slabVoxels(storageShape, ranges);
    const raw = readSlice(dataset, ranges, expected, description.liquidVar);
    const rawIce = iceDataset
      ? readSlice(iceDataset, ranges, expected, description.iceVar) : null;

    const local = { x: nx, y: ny, z: nz };
    local[chunkField] = stop - start;
    const out = makeHalfWriter(local.x * local.y * local.z);

    // Storage strides over the slab we just read.
    const slabShape = storageShape.slice();
    slabShape[chunkAxis] = storageRange[1] - storageRange[0];
    for (const dropped of description.droppedAxes) slabShape[dropped] = 1;
    const strides = new Array(slabShape.length);
    let acc = 1;
    for (let i = slabShape.length - 1; i >= 0; i--) {
      strides[i] = acc; acc *= slabShape[i];
    }

    const base = { x: 0, y: 0, z: 0 };
    base[chunkField] = start;
    const idx = new Array(slabShape.length).fill(0);

    let o = 0;
    for (let lx = 0; lx < local.x; lx++) {
      const gx = base.x + lx;
      idx[axis.x] = flip.x ? nx - 1 - gx : gx;
      if (chunkField === "x") idx[axis.x] -= storageRange[0];
      for (let ly = 0; ly < local.y; ly++) {
        const gy = base.y + ly;
        idx[axis.y] = flip.y ? ny - 1 - gy : gy;
        if (chunkField === "y") idx[axis.y] -= storageRange[0];
        for (let lz = 0; lz < local.z; lz++) {
          const gz = base.z + lz;
          idx[axis.z] = flip.z ? nz - 1 - gz : gz;
          if (chunkField === "z") idx[axis.z] -= storageRange[0];

          let flat = 0;
          for (let i = 0; i < idx.length; i++) flat += idx[i] * strides[i];

          let q = raw[flat];
          if (decode) q = decode(q);
          let qi = 0.0;
          if (rawIce) {
            qi = rawIce[flat];
            if (decodeIce) qi = decodeIce(qi);
          }
          const sigma = sigmaAt(q * multiplier, qi * iceMultiplier, rhoAir[gz]);
          if (Number.isFinite(sigma)) { if (sigma !== 0) finiteNonZero += 1; }
          else nonFinite += 1;
          out.set(o++, sigma);
        }
      }
    }

    const bytes = out.bytes();
    // Ghost ring: original voxel i lands on texel i+1.
    const origin = [1, 1, 1];
    origin[0] = chunkField === "z" ? start + 1 : 1;
    origin[1] = chunkField === "y" ? start + 1 : 1;
    origin[2] = chunkField === "x" ? start + 1 : 1;
    post({
      type: "slab", label,
      origin,
      size: [local.z, local.y, local.x],   // texture is (w=z, h=y, d=x)
      done: stop / total,
      data: bytes,
    }, [bytes.buffer]);
  }
  // The four lateral ghost planes, for periodic wrapping. Read separately as
  // four thin hyperslabs rather than accumulated during the sweep: each is
  // one plane out of the whole field, so the cost is noise and the code is
  // something a person can check.
  const plane = (fieldAxis, fieldIndex) => {
    const storageIndex = flip[fieldAxis]
      ? fieldExtent[fieldAxis] - 1 - fieldIndex : fieldIndex;
    const ranges = storageShape.map(() => []);
    for (const dropped of description.droppedAxes) ranges[dropped] = [0, 1];
    ranges[axis[fieldAxis]] = [storageIndex, storageIndex + 1];
    const raw = dataset.slice(ranges);
    const rawI = iceDataset ? iceDataset.slice(ranges) : null;
    // The two axes that remain, in field order.
    const rest = ["x", "y", "z"].filter((a) => a !== fieldAxis);
    const shape = rest.map((a) => fieldExtent[a]);
    const slabShape = storageShape.slice();
    slabShape[axis[fieldAxis]] = 1;
    for (const dropped of description.droppedAxes) slabShape[dropped] = 1;
    const strides = new Array(slabShape.length);
    let acc = 1;
    for (let i = slabShape.length - 1; i >= 0; i--) {
      strides[i] = acc; acc *= slabShape[i];
    }
    const out = new Float64Array(shape[0] * shape[1]);
    const idx = new Array(slabShape.length).fill(0);
    idx[axis[fieldAxis]] = 0;
    let o = 0;
    for (let a = 0; a < shape[0]; a++) {
      idx[axis[rest[0]]] = flip[rest[0]] ? shape[0] - 1 - a : a;
      for (let b = 0; b < shape[1]; b++) {
        idx[axis[rest[1]]] = flip[rest[1]] ? shape[1] - 1 - b : b;
        let flat = 0;
        for (let i = 0; i < idx.length; i++) flat += idx[i] * strides[i];
        let q = raw[flat];
        if (decode) q = decode(q);
        let qi = 0.0;
        if (rawI) {
          qi = rawI[flat];
          if (decodeIce) qi = decodeIce(qi);
        }
        // rest is in field order, so z is the second axis unless the fixed
        // axis IS z, in which case this plane has no z to index.
        const zIndex = rest[1] === "z" ? b : (rest[0] === "z" ? a : 0);
        out[o++] = sigmaAt(q * multiplier, qi * iceMultiplier, rhoAir[zIndex]);
      }
    }
    return { values: out, shape };
  };

  // engine._ghost_face_arrays: the opposite face, with corners that wrap in
  // both x and y because they are the trilinear support near a domain corner.
  const buildX = (source) => {          // source is (ny, nz)
    const face = makeHalfWriter((ny + 2) * (nz + 2));
    const at = (iy, iz) => source.values[iy * nz + iz];
    for (let iz = 0; iz < nz; iz++) {
      for (let iy = 0; iy < ny; iy++) {
        face.set((iy + 1) * (nz + 2) + iz + 1, at(iy, iz));
      }
      face.set(0 * (nz + 2) + iz + 1, at(ny - 1, iz));
      face.set((ny + 1) * (nz + 2) + iz + 1, at(0, iz));
    }
    return face.bytes();
  };
  const buildY = (source) => {          // source is (nx, nz)
    const face = makeHalfWriter((nx + 2) * (nz + 2));
    const at = (ix, iz) => source.values[ix * nz + iz];
    for (let iz = 0; iz < nz; iz++) {
      for (let ix = 0; ix < nx; ix++) {
        face.set((ix + 1) * (nz + 2) + iz + 1, at(ix, iz));
      }
      face.set(0 * (nz + 2) + iz + 1, at(nx - 1, iz));
      face.set((nx + 1) * (nz + 2) + iz + 1, at(0, iz));
    }
    return face.bytes();
  };

  // A field that is entirely zero apart from a scattering of infinities is
  // not a cloud field, it is a failed read. Say so instead of rendering it.
  if (finiteNonZero === 0) {
    throw new Error(
      `'${description.liquidVar}' read as ${spatialVoxels} values that are ` +
      `all zero (plus ${nonFinite} non-finite). That is a failed read rather ` +
      "than an empty sky — the HDF5 reader returned buffers it never filled.");
  }

  const faces = {
    x_lo: buildX(plane("x", nx - 1)),
    x_hi: buildX(plane("x", 0)),
    y_lo: buildY(plane("y", ny - 1)),
    y_hi: buildY(plane("y", 0)),
  };
  post({ type: "faces", label, faces },
       Object.values(faces).map((f) => f.buffer));

  return { label, shape: [nx, ny, nz] };
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
  const { id, op, ...args } = event.data;
  try {
    let result;
    if (op === "open") result = await open(args);
    else if (op === "extinction") result = await extinction(args);
    else if (op === "probe") result = await probe(args);
    else throw new Error(`unknown ingest operation '${op}'`);
    post({ id, ok: true, result });
  } catch (err) {
    post({ id, ok: false, error: String(err?.message || err),
           advice: err?.advice || "" });
  }
};
