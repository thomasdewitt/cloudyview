// Resident GPU resources for one scene: the extinction volume, an optional
// finer nest, the ocean normal tile, and the geometry the uniform block needs.
//
// The volume's texture layout is the one piece here worth stating plainly.
// A C-order (nx, ny, nz) array already has z varying fastest, so it maps onto
// a 3D texture of width nz, height ny, depth nx with no reshuffling at all;
// raymarch.wgsl swizzles its sample coordinates to match. Nothing is padded:
// voxel i is texel i. The boundary behaviour the ghost ring used to buy —
// witness's linear taper into zero, and the periodic wrap — is a sampler
// address mode plus an analytic window in the shader (sample_level), which
// is exact and does not cost a 2048-cell axis the two texels that put it over
// a browser's 3D texture limit.

"use strict";

import { volumeAABB, minVoxelSize, voxelSizes, validateNestContainment }
  from "./field.js";
import {
  guardAllocation, volumeFits, retireAfterSubmittedWork,
} from "./gpu.js";

async function fetchBytes(url, onProgress, decompress = null) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not fetch ${url} (HTTP ${response.status}).`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  if (!onProgress || !total || !response.body) {
    const raw = await response.arrayBuffer();
    if (!decompress) return new Uint8Array(raw);
    return new Uint8Array(await new Response(
      new Blob([raw]).stream().pipeThrough(new DecompressionStream(decompress))
    ).arrayBuffer());
  }
  // Progress counts bytes off the wire, so it has to be measured BEFORE the
  // decompressor — content-length is the compressed size, and counting
  // inflated bytes against it would run the bar to several hundred percent.
  let received = 0;
  const counted = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      received += chunk.length;
      onProgress(received / total);
      controller.enqueue(chunk);
    },
  }));
  const stream = decompress
    ? counted.pipeThrough(new DecompressionStream(decompress))
    : counted;
  return new Uint8Array(await new Response(stream).arrayBuffer());
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

/** A zero-initialized r16float volume texture for an (nx, ny, nz) field. */
export async function createVolumeTexture(device, shape, label) {
  const fit = volumeFits(device.limits, shape);
  if (!fit.ok) {
    const err = new Error(fit.message);
    err.advice = fit.advice;
    throw err;
  }
  const [nx, ny, nz] = shape;
  return guardAllocation(device, label || "the cloud volume", fit.bytes, () =>
    device.createTexture({
      label: label || "soar-volume",
      size: [nz, ny, nx],           // width=nz, height=ny, depth=nx
      dimension: "3d",
      format: "r16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.COPY_SRC,
    }));
}

/**
 * Write a slab of the field. `data` is fp16 laid out with z fastest, then y,
 * then x — the same order the texture wants, so this is a straight copy.
 * queue.writeTexture is used rather than a staging buffer on purpose: the
 * 256-byte bytesPerRow rule applies only to buffer-to-texture copies, and
 * nz*2 is almost never a multiple of 256.
 */
export function writeVolumeSlab(device, texture, data, origin, size) {
  device.queue.writeTexture(
    { texture, origin: { x: origin[0], y: origin[1], z: origin[2] } },
    data,
    { bytesPerRow: size[0] * 2, rowsPerImage: size[1] },
    size,
  );
}

// queue.writeTexture copies into driver-owned staging memory that is only
// reclaimed once the GPU has consumed it. Nothing in an upload loop submits
// work, so without a barrier every slab of a multi-gigabyte field piles up at
// once — which on a 3.5 GB field exhausts system memory and takes the device,
// and then the browser, down with it. Draining every this-many bytes bounds
// the pile to roughly one barrier's worth.
//
// It is also the chunk size for a whole-field upload below, which makes it do
// double duty: no single writeTexture call can then approach the 2 GB ceiling
// a browser puts on one source view.
export const UPLOAD_DRAIN_BYTES = 64 * 1024 * 1024;

/**
 * Upload a whole field, in x-plane runs.
 *
 * NOT one writeTexture call. Firefox rejects a source view larger than 2 GB
 * outright ("ArrayBufferView ... larger than 2 GB"), which the 4.2 GB STEAM
 * desert field walked straight into — and even under the ceiling, one call
 * with several gigabytes behind it is the staging-memory problem above in its
 * worst form. The file path never hit either, because ingest has always
 * uploaded slab by slab; this is the demo path learning the same lesson.
 *
 * x is the texture's depth axis (width=nz, height=ny, depth=nx), so a run of
 * x planes is contiguous in the source and needs no repacking.
 */
export async function writeWholeVolume(device, texture, words, shape,
                                       onProgress) {
  const [nx, ny, nz] = shape;
  const perPlane = ny * nz;                       // elements in one x plane
  const planes = Math.max(1, Math.floor(UPLOAD_DRAIN_BYTES / (perPlane * 2)));
  for (let x0 = 0; x0 < nx; x0 += planes) {
    const depth = Math.min(planes, nx - x0);
    writeVolumeSlab(
      device, texture,
      words.subarray(x0 * perPlane, (x0 + depth) * perPlane),
      [0, 0, x0], [nz, ny, depth]);
    await device.queue.onSubmittedWorkDone();
    onProgress?.((x0 + depth) / nx);
  }
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
  /** NetCDF group the nest was read from, null when it came from the root. */
  get nestGroup() { return this._nest?.name || null; }

  /**
   * Attach a finer level. The nest ALWAYS tapers to zero at its own edges,
   * even in a periodic domain: that taper is how it blends out into the
   * coarse field around it, which is what witness does per level.
   */
  attachNest(nest) {
    const report = validateNestContainment(
      this.bmin, this.bmax, nest.bmin, nest.bmax,
      voxelSizes(this.shape, this.bmin, this.bmax));
    this._nest = nest;
    return report;
  }

  /**
   * Unbind the nest and release it once the GPU is done with it.
   *
   * The caller rebuilds the bind group immediately, but frames submitted
   * before that are still sampling this texture. Destroying it now is how a
   * "remove nest" click becomes a crash a few milliseconds later.
   */
  removeNest() {
    const had = Boolean(this._nest);
    const texture = this._nest?.texture ?? null;
    this._nest = null;
    if (texture) retireAfterSubmittedWork(this.device, texture);
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
 *
 * `ocean` is a thunk rather than a URL for the same reason the file loader
 * takes one: the sea surface is field-independent and belongs to the session,
 * so loading the demo after a file (or the other way round) reuses the tile
 * already on the card instead of allocating a second one and abandoning it.
 */
export async function loadDemoScene(device, baseUrl, ocean, progress) {
  progress?.("Downloading the cloud field…", 0);
  const meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  // A pre-v5 bake ships a ghost-padded volume and a faces.bin, which this
  // renderer would read two texels out of register on every axis. Say so
  // rather than rendering a subtly wrong field.
  if (meta.volume.padded_dims_xyz) {
    throw new Error(
      `The '${meta.id}' demo was baked ghost-padded (schema ` +
      `${meta.schema}); soar now uploads fields unpadded. Re-bake it with ` +
      "tools/prebake_demos.py --skip-still.");
  }
  const shape = meta.volume.shape_xyz;

  // The volume ships gzipped. It is fp16 either way — the texture is
  // r16float — so this is purely wire size, and DecompressionStream means a
  // dumb static host works without any Content-Encoding negotiation.
  const volumeFile = meta.volume.file ?? "volume.bin";
  const gzipped = meta.volume.compression === "gzip";
  const volumeBytes = await fetchBytes(
    `${baseUrl}/${volumeFile}`,
    (f) => progress?.("Downloading the cloud field…", f * 0.8),
    gzipped ? "gzip" : null);
  const mapBytes = await fetchBytes(`${baseUrl}/map.bin`);

  progress?.("Loading the ocean surface…", 0.85);
  const oceanTile = await ocean();

  progress?.("Uploading to the GPU…", 0.92);
  const volumeTexture = await createVolumeTexture(
    device, shape, "the demo cloud field");
  // From here the volume exists on the card and nothing else holds it. Any
  // throw before the Scene takes ownership has to give it back.
  let nestDummy = null;
  try {
    const words = new Uint16Array(volumeBytes.buffer, volumeBytes.byteOffset,
                                  volumeBytes.byteLength / 2);
    await writeWholeVolume(
      device, volumeTexture, words, shape,
      (f) => progress?.("Uploading to the GPU…", 0.92 + 0.07 * f));
    nestDummy = createNestDummy(device);

    const bmin = meta.volume.bmin;
    const bmax = meta.volume.bmax;
    const scene = new Scene(device, {
      volumeTexture,
      volumeView: volumeTexture.createView(),
      shape,
      bmin, bmax,
      minVoxelM: minVoxelSize(shape, bmin, bmax),
      oceanView: oceanTile.view,
      oceanFifDx: oceanTile.dx,
      oceanTileExtent: oceanTile.tileExtent,
      oceanMaxLod: oceanTile.maxLod,
      albedo: new Float32Array(
        mapBytes.buffer, mapBytes.byteOffset, mapBytes.byteLength / 4),
      albedoShape: meta.map.shape_yx,
      _nest: null,
      _nestDummy: nestDummy,
      // From the bake, because it is a property of the field rather than a
      // preference: TWP-ICE is a 102 km crop out of a larger run, and
      // wrapping it repeats a squall line. Older bakes have no `periodic`
      // key and every one of them tiled, so absent means true.
      periodicDefault: meta.periodic ?? true,
      title: meta.title,
      description: meta.description,
      sourceName: meta.source,
      sun: meta.sun,
      // Where the landing page's still was taken from, so flight can open
      // there — see Camera.applyStart and viewer boot.
      startCamera: meta.still?.camera ?? null,
    });
    progress?.("Ready.", 1);
    return scene;
  } catch (err) {
    volumeTexture.destroy();
    nestDummy?.destroy();
    throw err;
  }
}

export { volumeAABB, minVoxelSize };
