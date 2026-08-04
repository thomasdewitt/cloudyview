// Resident GPU resources for one scene: the extinction volume, an optional
// finer nest, the ocean normal tile, and the geometry the uniform block needs.
//
// The volume's texture layout is the one piece here worth stating plainly.
// A C-order (nx, ny, nz) array already has z varying fastest, so it maps onto
// a 3D texture of width nz+2, height ny+2, depth nx+2 with no reshuffling at
// all; raymarch.wgsl swizzles its sample coordinates to match. The +2 is a
// ghost ring — original voxel i becomes texel i+1 — so hardware trilinear
// filtering supplies witness's linear taper into the zero border instead of
// clamping the edge voxel outward.

"use strict";

import { volumeAABB, minVoxelSize, validateNestContainment } from "./field.js";
import { guardAllocation, volumeFits } from "./gpu.js";

const FACE_NAMES = ["x_lo", "x_hi", "y_lo", "y_hi"];

async function fetchBytes(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not fetch ${url} (HTTP ${response.status}).`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  if (!onProgress || !total || !response.body) {
    return new Uint8Array(await response.arrayBuffer());
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

/**
 * The ocean surface: a periodic ~100 m tile of multifractal normals with a
 * renormalized mip chain. Field-independent, so it ships with the tool and is
 * loaded once per session.
 */
export async function loadOceanTile(device, baseUrl, onProgress) {
  const meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  const texture = device.createTexture({
    label: "ocean-fif-normals",
    size: [meta.n, meta.n, 1],
    format: "rgba16float",
    mipLevelCount: meta.mips,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  for (let level = 0; level < meta.mips; level++) {
    const bytes = await fetchBytes(`${baseUrl}/fif_mip${level}.bin`);
    const n = Math.max(1, meta.n >> level);
    device.queue.writeTexture(
      { texture, mipLevel: level },
      bytes,
      { bytesPerRow: n * 8, rowsPerImage: n },
      [n, n, 1],
    );
    onProgress?.((level + 1) / meta.mips);
  }
  return {
    view: texture.createView(),
    texture,
    dx: meta.dx_m,
    tileExtent: meta.tile_extent_m,
    maxLod: meta.mips - 1,
  };
}

/** A zero-initialized r16float volume texture of the padded shape. */
export async function createVolumeTexture(device, padded, label) {
  const fit = volumeFits(device.limits, padded);
  if (!fit.ok) {
    const err = new Error(fit.message);
    err.advice = fit.advice;
    throw err;
  }
  const [px, py, pz] = padded;
  return guardAllocation(device, label || "the cloud volume", fit.bytes, () =>
    device.createTexture({
      label: label || "soar-volume",
      size: [pz, py, px],           // width=nz+2, height=ny+2, depth=nx+2
      dimension: "3d",
      format: "r16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.COPY_SRC,
    }));
}

/**
 * Write a slab of the interior. `data` is fp16 laid out with z fastest, then
 * y, then x — the same order the texture wants, so this is a straight copy.
 * queue.writeTexture is used rather than a staging buffer on purpose: the
 * 256-byte bytesPerRow rule applies only to buffer-to-texture copies, and
 * (nz+2)*2 is almost never a multiple of 256.
 */
export function writeVolumeSlab(device, texture, data, origin, size) {
  device.queue.writeTexture(
    { texture, origin: { x: origin[0], y: origin[1], z: origin[2] } },
    data,
    { bytesPerRow: size[0] * 2, rowsPerImage: size[1] },
    size,
  );
}

/**
 * Fill (or clear) the four lateral ghost planes.
 *
 * Periodic domains take their x/y ghost texels from the OPPOSITE face so that
 * filtering across the wrap seam is exact; z always keeps the zero taper,
 * because the vertical is never periodic. Non-periodic writes zeros of the
 * same shape, which is why toggling does not need the volume re-uploaded.
 */
export function writeGhostBorder(device, texture, faces, periodic, padded) {
  const [px, py, pz] = padded;
  const zeros = (n) => new Uint16Array(n);
  const xPlane = (name) => periodic ? faces[name] : zeros(py * pz);
  const yPlane = (name) => periodic ? faces[name] : zeros(px * pz);

  const write = (data, origin, size, rowsPerImage) =>
    device.queue.writeTexture(
      { texture, origin: { x: origin[0], y: origin[1], z: origin[2] } },
      data, { bytesPerRow: pz * 2, rowsPerImage }, size);

  write(xPlane("x_lo"), [0, 0, 0], [pz, py, 1], py);
  write(xPlane("x_hi"), [0, 0, px - 1], [pz, py, 1], py);
  write(yPlane("y_lo"), [0, 0, 0], [pz, 1, px], 1);
  write(yPlane("y_hi"), [0, py - 1, 0], [pz, 1, px], 1);
}

/** Split the packed faces.bin blob into its four planes. */
export function unpackFaces(bytes, padded) {
  const [px, py, pz] = padded;
  const sizes = { x_lo: py * pz, x_hi: py * pz, y_lo: px * pz, y_hi: px * pz };
  const words = new Uint16Array(
    bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  const faces = {};
  let offset = 0;
  for (const name of FACE_NAMES) {
    faces[name] = words.subarray(offset, offset + sizes[name]);
    offset += sizes[name];
  }
  return faces;
}

export class Scene {
  constructor(device, parts) {
    this.device = device;
    Object.assign(this, parts);
  }

  get nested() { return Boolean(this._nest); }
  get nestBmin() { return this._nest?.bmin ?? [0, 0, 0]; }
  get nestBmax() { return this._nest?.bmax ?? [0, 0, 0]; }
  get minVoxelNestM() { return this._nest?.minVoxelM ?? this.minVoxelM; }
  get nestView() { return (this._nest?.texture ?? this._nestDummy).createView(); }

  writeGhostBorder(periodic) {
    if (!this._faces) return;
    writeGhostBorder(
      this.device, this.volumeTexture, this._faces, periodic, this.padded);
  }

  /**
   * Attach a finer level. The nest is ALWAYS padded with a zero border, even
   * in a periodic domain: that taper is how it blends out into the coarse
   * field at its own edges, which is what witness does per level.
   */
  attachNest(nest) {
    const report = validateNestContainment(
      this.bmin, this.bmax, nest.bmin, nest.bmax);
    this._nest = nest;
    return report;
  }

  removeNest() {
    const had = Boolean(this._nest);
    this._nest?.texture?.destroy();
    this._nest = null;
    return had;
  }

  /** Fraction of the outer domain's volume the nest covers. */
  nestCoverageFraction() {
    if (!this._nest) return 0.0;
    let f = 1.0;
    for (let i = 0; i < 3; i++) {
      const span = this.bmax[i] - this.bmin[i];
      const overlap = Math.max(0.0,
        Math.min(this.bmax[i], this._nest.bmax[i]) -
        Math.max(this.bmin[i], this._nest.bmin[i]));
      f *= span > 0 ? overlap / span : 0.0;
    }
    return f;
  }

  destroy() {
    this.volumeTexture?.destroy();
    this._nest?.texture?.destroy();
    this._nestDummy?.destroy();
  }
}

/** The 1x1x1 stand-in bound when there is no nest, so one layout serves both. */
export function createNestDummy(device) {
  const texture = device.createTexture({
    label: "soar-nest-absent",
    size: [1, 1, 1], dimension: "3d", format: "r16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture }, new Uint16Array(1), { bytesPerRow: 2, rowsPerImage: 1 },
    [1, 1, 1]);
  return texture;
}

/**
 * Build the demo scene from the exported payload.
 * `progress(stage, fraction)` reports download and upload separately because
 * the download is the slow part on a first visit and the upload is the slow
 * part afterwards, and saying so is better than one bar that stalls.
 */
export async function loadDemoScene(device, baseUrl, oceanUrl, progress) {
  progress?.("Downloading the cloud field…", 0);
  const meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  const padded = meta.volume.padded_dims_xyz;

  const volumeBytes = await fetchBytes(
    `${baseUrl}/volume.bin`, (f) => progress?.("Downloading the cloud field…", f * 0.8));
  const facesBytes = await fetchBytes(`${baseUrl}/faces.bin`);

  progress?.("Loading the ocean surface…", 0.85);
  const ocean = await loadOceanTile(device, oceanUrl);

  progress?.("Uploading to the GPU…", 0.92);
  const volumeTexture = await createVolumeTexture(
    device, padded, "the demo cloud field");
  const [px, py, pz] = padded;
  writeVolumeSlab(
    device, volumeTexture,
    new Uint16Array(volumeBytes.buffer, volumeBytes.byteOffset,
                    volumeBytes.byteLength / 2),
    [0, 0, 0], [pz, py, px]);

  const faces = unpackFaces(facesBytes, padded);
  writeGhostBorder(device, volumeTexture, faces, true, padded);

  const bmin = meta.volume.bmin;
  const bmax = meta.volume.bmax;
  const scene = new Scene(device, {
    volumeTexture,
    volumeView: volumeTexture.createView(),
    padded,
    shape: meta.volume.shape_xyz,
    bmin, bmax,
    minVoxelM: minVoxelSize(meta.volume.shape_xyz, bmin, bmax),
    oceanView: ocean.view,
    oceanFifDx: ocean.dx,
    oceanTileExtent: ocean.tileExtent,
    oceanMaxLod: ocean.maxLod,
    _faces: faces,
    _nest: null,
    _nestDummy: createNestDummy(device),
    title: meta.title,
    description: meta.description,
    sourceName: meta.source,
    sun: meta.sun,
  });
  progress?.("Ready.", 1);
  return scene;
}

export { volumeAABB, minVoxelSize };
