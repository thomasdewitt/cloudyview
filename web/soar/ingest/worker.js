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

/**
 * IEEE binary32 to binary16, round-to-nearest-EVEN — the mode numpy and the
 * native Float16Array both use. An earlier version rounded half away from
 * zero, which diverged systematically.
 *
 * Only reached on browsers with WebGPU but without Float16Array (Chrome
 * 113-134; Firefox has had it since well before its WebGPU release), so this
 * is close to dead code already.
 *
 * Residual: measured against the native path over 137k values it differs on
 * 4 of them, always by one fp16 ULP. Those are double-rounding cases — this
 * goes f64 to f32 to f16 where the native path goes f64 to f16 directly.
 * Eliminating them needs a round-to-odd intermediate, which is not worth the
 * code for a 3e-5 disagreement one ULP below a quantization the renderer has
 * already applied. Stated rather than hidden.
 */
function toHalf(value) {
  _f32[0] = value;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf / NaN
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;                            // overflow
  if (e <= 0) {
    if (e < -10) return sign;                                     // underflow
    const m = mant | 0x800000;
    const shift = 14 - e;
    let half = m >>> shift;
    const rest = m & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rest > halfway || (rest === halfway && (half & 1))) half += 1;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (half & 1))) half += 1;
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
  const budget = slabBudget || SLAB_BUDGET_BYTES;

  const chunkExtent = {};
  for (const a of ["x", "y", "z"]) {
    const c = description.chunks?.[axis[a]];
    chunkExtent[a] = Math.max(1, Math.min(c || fieldExtent[a], fieldExtent[a]));
  }
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
    const idx = new Array(slabShape.length).fill(0);
    let o = 0;
    for (let lx = 0; lx < local.x; lx++) {
      const gx = base.x + lx;
      idx[axis.x] = (flip.x ? nx - 1 - gx : gx) - rangeStart.x;
      for (let ly = 0; ly < local.y; ly++) {
        const gy = base.y + ly;
        idx[axis.y] = (flip.y ? ny - 1 - gy : gy) - rangeStart.y;
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
          const sigma = sigmaAt(q * multiplier, qi * iceMultiplier, rhoAir[gz]);
          if (Number.isFinite(sigma)) { if (sigma !== 0) finiteNonZero += 1; }
          else nonFinite += 1;
          out.set(o++, sigma);
        }
      }
    }
    const bytes = out.bytes();
    // Ghost ring: original voxel i lands on texel i+1.
    post({
      type: "slab", label,
      origin: [z0 + 1, y0 + 1, x0 + 1],
      size: [local.z, local.y, local.x],   // texture is (w=z, h=y, d=x)
      done: (++tilesDone) / tileCount,
      data: bytes,
    }, [bytes.buffer]);
  } } }
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
    const expected = slabVoxels(storageShape, ranges);
    const raw = readSlice(dataset, ranges, expected, description.liquidVar);
    const rawI = iceDataset
      ? readSlice(iceDataset, ranges, expected, description.iceVar) : null;
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
