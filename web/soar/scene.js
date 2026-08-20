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
 * Open a URL as a stream of decompressed bytes, without ever buffering the
 * whole body. Progress counts bytes off the wire (compressed), for the same
 * reason as in fetchBytes above.
 */
async function fetchDecompressedStream(url, onProgress, decompress = null) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not fetch ${url} (HTTP ${response.status}).`);
  }
  if (!response.body) {
    throw new Error(`Could not fetch ${url}: the response has no body.`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  let received = 0;
  let stream = response.body;
  if (onProgress && total) {
    stream = stream.pipeThrough(new TransformStream({
      transform(chunk, controller) {
        received += chunk.length;
        onProgress(received / total);
        controller.enqueue(chunk);
      },
    }));
  }
  return decompress
    ? stream.pipeThrough(new DecompressionStream(decompress))
    : stream;
}

/**
 * A surface tile: the ocean's periodic ~100 m patch of multifractal normals,
 * or the night city's density tile — same rgba16float mip-chain byte format,
 * different meta key for the texel size (the ocean's dx_m is the city's
 * cell_m). Field-independent, so each ships with the tool and is loaded once
 * per session.
 */
export async function loadOceanTile(device, baseUrl, onProgress) {
  const meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  const texture = device.createTexture({
    label: `${baseUrl}-surface-tile`,
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
  const dx = meta.dx_m ?? meta.cell_m;
  if (!(dx > 0)) {
    throw new Error(
      `${baseUrl}/meta.json names neither dx_m nor cell_m; the tile has ` +
      "no texel size.");
  }
  return {
    view: texture.createView(),
    texture,
    dx,
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
 * Upload a whole field straight off a byte stream, one slab at a time.
 *
 * NOT one writeTexture call. Firefox rejects a source view larger than 2 GB
 * outright ("ArrayBufferView ... larger than 2 GB"), which the 4.2 GB STEAM
 * desert field walked straight into. x is the texture's depth axis
 * (width=nz, height=ny, depth=nx), so a run of x planes is contiguous in
 * the source and needs no repacking.
 *
 * The demo path used to inflate the entire gzipped volume into one JS
 * ArrayBuffer and only then upload — which for the 4.15 GB desert field,
 * on top of the 4.15 GB texture, is more than an 8 GB machine has, and
 * Chrome kills the fetch mid-stream with a bare "Failed to fetch". Here the
 * JS heap never holds more than one UPLOAD_DRAIN_BYTES slab: the texture is
 * the only multi-gigabyte allocation, which is what lets a field of nearly
 * the machine's memory load at all.
 *
 * `stream` yields the DECOMPRESSED bytes, fp16, z fastest (the texture's
 * own layout, see writeVolumeSlab). Throws if the stream ends short or runs
 * long against `shape`.
 */
export async function streamWholeVolume(device, texture, stream, shape,
                                        onProgress) {
  const [nx, ny, nz] = shape;
  const perPlaneBytes = ny * nz * 2;              // one x plane, fp16
  const planes = Math.max(1, Math.floor(UPLOAD_DRAIN_BYTES / perPlaneBytes));
  const slab = new Uint8Array(planes * perPlaneBytes);
  let filled = 0;                                 // bytes waiting in `slab`
  let x0 = 0;                                     // next x plane to write

  const flush = async () => {
    const depth = filled / perPlaneBytes;         // integral by construction
    if (x0 + depth > nx) {
      throw new Error(
        `the volume stream carried more than the ${nx} x planes the ` +
        "metadata promised — wrong file behind the URL?");
    }
    writeVolumeSlab(
      device, texture,
      new Uint16Array(slab.buffer, 0, filled / 2),
      [0, 0, x0], [nz, ny, depth]);
    await device.queue.onSubmittedWorkDone();
    x0 += depth;
    filled = 0;
    onProgress?.(x0 / nx);
  };

  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let off = 0;
      while (off < value.length) {
        const take = Math.min(value.length - off, slab.length - filled);
        slab.set(value.subarray(off, off + take), filled);
        filled += take;
        off += take;
        if (filled === slab.length) await flush();
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (filled % perPlaneBytes !== 0) {
    throw new Error(
      `the volume stream ended mid-plane: ${filled % perPlaneBytes} stray ` +
      `bytes after ${x0} of ${nx} x planes — truncated download?`);
  }
  if (filled) await flush();
  if (x0 !== nx) {
    throw new Error(
      `the volume stream carried ${x0} x planes where the metadata ` +
      `promised ${nx} — truncated download?`);
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
 * `surface` is a thunk rather than a URL for the same reason the file loader
 * takes one: the tiles are field-independent and belong to the session, so
 * loading the demo after a file (or the other way round) reuses the tile
 * already on the card instead of allocating a second one and abandoning it.
 * It takes the surface kind — "ocean", or "city" for a night-city demo
 * (meta.surface) — because which tile a demo stands on is the demo's call.
 */
export async function loadDemoScene(device, baseUrl, surface, progress) {
  progress?.("Downloading the cloud field…", 0);
  const meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  const surfaceKind = meta.surface ?? "ocean";
  if (surfaceKind !== "ocean" && surfaceKind !== "city") {
    throw new Error(
      `The '${meta.id}' demo asks for surface '${surfaceKind}', which this ` +
      "viewer does not know. It knows 'ocean' and 'city'.");
  }
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

  // The texture is created BEFORE the download so the volume can stream
  // straight onto the card slab by slab (streamWholeVolume): the JS heap
  // never holds the inflated field, which for the multi-gigabyte demos is
  // the difference between loading and the browser running out of memory.
  // It also means a field too large for this GPU fails here, in one
  // sentence, before a single volume byte is downloaded.
  const volumeTexture = await createVolumeTexture(
    device, shape, "the demo cloud field");
  // From here the volume exists on the card and nothing else holds it. Any
  // throw before the Scene takes ownership has to give it back.
  let nestDummy = null;
  try {
    // The volume ships gzipped. It is fp16 either way — the texture is
    // r16float — so this is purely wire size, and DecompressionStream means
    // a dumb static host works without any Content-Encoding negotiation.
    const volumeFile = meta.volume.file ?? "volume.bin";
    const gzipped = meta.volume.compression === "gzip";
    try {
      await streamWholeVolume(
        device, volumeTexture,
        await fetchDecompressedStream(
          `${baseUrl}/${volumeFile}`,
          (f) => progress?.("Downloading the cloud field…", f * 0.9),
          gzipped ? "gzip" : null),
        shape);
    } catch (err) {
      // The browser's own message for a stream that dies is a bare
      // "Failed to fetch", which reads like a server problem. On a machine
      // with little free memory it usually is not one.
      const gb = ((meta.volume.bytes_uncompressed
                   ?? shape[0] * shape[1] * shape[2] * 2) / 2 ** 30);
      const wrapped = new Error(
        `Downloading '${meta.id}' failed midway: ` +
        `${String(err && err.message || err)}`);
      wrapped.advice =
        `This field is ${gb.toFixed(1)} GB unpacked. If the connection is ` +
        "fine, the machine likely ran short of memory — a coarse variant " +
        "of the same field is in the list, or retry after closing other " +
        "tabs.";
      throw wrapped;
    }

    const mapBytes = await fetchBytes(`${baseUrl}/${meta.map.file ?? "map.bin"}`);
    progress?.(surfaceKind === "city"
               ? "Raising the city…" : "Loading the ocean surface…", 0.95);
    const surfaceTile = await surface(surfaceKind);
    nestDummy = createNestDummy(device);

    const bmin = meta.volume.bmin;
    const bmax = meta.volume.bmax;
    const scene = new Scene(device, {
      volumeTexture,
      volumeView: volumeTexture.createView(),
      shape,
      bmin, bmax,
      minVoxelM: minVoxelSize(shape, bmin, bmax),
      oceanView: surfaceTile.view,
      oceanFifDx: surfaceTile.dx,
      oceanTileExtent: surfaceTile.tileExtent,
      oceanMaxLod: surfaceTile.maxLod,
      city: surfaceKind === "city",
      cityOffsetM: meta.city?.offset_m ?? [0.0, 0.0],
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
