// Resident GPU resources for one scene: the extinction volume, an optional
// finer nest, the surface (the sea's, the city's or the desert's
// tile), and the geometry the uniform block needs.
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
import * as K from "./constants.js";
import { worldToRelative } from "./camera.js";
import { fromHalf, HAS_F16, toHalf } from "./half.js";
import { buildDesertBounds, LAYERS as DESERT_BOUNDS_LAYERS }
  from "./desert_bounds.js";

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

// Channels per texel of each surface-tile format (soar_host's
// SURFACE_FORMAT_CHANNELS): the ocean and the city are rgba16float, the
// desert rg16float.
const SURFACE_FORMAT_CHANNELS = { rgba16float: 4, rg16float: 2 };

/**
 * One stored mip level -> the interleaved texel bytes the texture wants.
 * A level is either the texels themselves, or (encoding
 * "planar-byte-shuffle", the desert's) each channel's plane with every half
 * float's low bytes before its high bytes, the arrangement that lets gzip
 * see a height field's smoothness. Twin of soar_host.decode_surface_level.
 */
export function decodeSurfaceLevel(bytes, side, meta) {
  const format = meta.format ?? "rgba16float";
  const channels = SURFACE_FORMAT_CHANNELS[format];
  if (!channels) throw new Error(`unknown surface-tile format '${format}'.`);
  const texels = side * side;
  if (bytes.byteLength !== texels * channels * 2) {
    throw new Error(
      `a ${side}x${side} ${format} level is ${texels * channels * 2} bytes; ` +
      `this one is ${bytes.byteLength}.`);
  }
  const encoding = meta.encoding ?? "interleaved";
  if (encoding === "interleaved") return bytes;
  if (encoding !== "planar-byte-shuffle") {
    throw new Error(`unknown surface-tile encoding '${encoding}'.`);
  }
  const out = new Uint8Array(bytes.byteLength);
  for (let c = 0; c < channels; c++) {
    const lo = bytes.subarray((2 * c) * texels, (2 * c + 1) * texels);
    const hi = bytes.subarray((2 * c + 1) * texels, (2 * c + 2) * texels);
    for (let i = 0, o = 2 * c; i < texels; i++, o += 2 * channels) {
      out[o] = lo[i];
      out[o + 1] = hi[i];
    }
  }
  return out;
}

/** Row 8.yzw under DESERT, from the desert tile's meta (soar_host twin). */
export function desertReliefM(meta) {
  for (const key of ["height_min_m", "height_max_m", "band_max_m"]) {
    if (!Number.isFinite(meta?.[key])) {
      throw new Error(`the desert tile's meta.json has no ${key}.`);
    }
  }
  return [meta.height_min_m, meta.height_max_m, meta.band_max_m];
}

/**
 * Rows 25..46 under DESERT (raymarch.wgsl desert_tile), from the desert
 * tile's meta: its per-level tables four levels to a row, its per-level
 * colour correction a level to a row, then (bounce albedo, 0).
 * Twin of soar_host.desert_tile_rows.
 */
export function desertTileRows(meta) {
  const mips = meta?.mips;
  if (mips !== 12) {
    throw new Error(`the desert tile has ${mips} mips; the shader's tables ` +
                    "are laid out for 12.");
  }
  const rows = [];
  for (const key of ["slope_max", "envelope_m", "slope_variance_removed"]) {
    const table = meta[key];
    if (!Array.isArray(table) || table.length !== mips) {
      throw new Error(`the desert tile's meta.json ${key} is not ${mips} ` +
                      "levels.");
    }
    for (let i = 0; i < mips; i += 4) rows.push(table.slice(i, i + 4));
  }
  const correction = meta.albedo_correction;
  if (!Array.isArray(correction) || correction.length !== mips) {
    throw new Error("the desert tile's meta.json albedo_correction is not " +
                    `${mips} levels.`);
  }
  for (const rgb of correction) rows.push([rgb[0], rgb[1], rgb[2], 0.0]);
  const bounce = meta.bounce_albedo;
  if (!Array.isArray(bounce) || bounce.length !== 3) {
    throw new Error("the desert tile's meta.json has no bounce_albedo.");
  }
  rows.push([bounce[0], bounce[1], bounce[2], 0.0]);
  if (rows.length !== K.DESERT_TILE_ROWS) {
    throw new Error(`desertTileRows made ${rows.length} rows; the shader ` +
                    `reads ${K.DESERT_TILE_ROWS}.`);
  }
  return rows;
}

/**
 * The camera's floor over the desert: a function of the surface-frame
 * position (tile metres, the camera's surfacePosition) returning the lowest
 * z the camera may take. The tile's own height, read bilinearly the way the
 * GPU reads it (texel i centred at (i + 1/2) cells, periodic), plus the most
 * the sub-tile octaves can add (band_max_m times raymarch.wgsl's
 * DESERT_OCTAVE_MARGIN) and the clearance. From the same level-0 bytes the
 * shader samples, so the floor is under the drawn ground by construction.
 */
export const DESERT_OCTAVE_MARGIN = 0.4942;       // raymarch.wgsl's, a true bound

export function desertFloorAt(tile, groundZ = 0.0) {
  const { heights, meta } = tile;
  if (!heights) throw new Error("the desert tile was loaded without heights.");
  const n = meta.n;
  const cell = meta.cell_m;
  const lift = meta.band_max_m * DESERT_OCTAVE_MARGIN
    + K.DESERT_FLOOR_CLEARANCE_M + groundZ;
  const at = (i, j) =>
    fromHalf(heights[(((j % n) + n) % n) * n + (((i % n) + n) % n)]);
  return ([x, y]) => {
    const gx = x / cell - 0.5;
    const gy = y / cell - 0.5;
    const i = Math.floor(gx);
    const j = Math.floor(gy);
    const fx = gx - i;
    const fy = gy - j;
    const h = (at(i, j) * (1 - fx) + at(i + 1, j) * fx) * (1 - fy)
            + (at(i, j + 1) * (1 - fx) + at(i + 1, j + 1) * fx) * fy;
    return h + lift;
  };
}

/**
 * A binding-3 surface tile: the ocean's wave normals (rgba16float), the
 * night city's density tile (rgba16float) or the desert's eroded terrain
 * (rg16float, planar-byte-shuffled) — one mip-chain loader, each tile's meta
 * saying its format, files, encoding and compression. The texel size is
 * dx_m (the ocean's) or cell_m.
 * Field-independent, so each ships with the tool and is loaded once per
 * session.
 *
 * `schema` is the meta.json schema the caller's shader decodes
 * (constants.js SURFACE_TILE_SCHEMAS); anything else is refused. The deploy
 * copies tile directories through unfingerprinted, so the cache-safety is
 * here instead: meta.json is always revalidated, and the bins are fetched
 * under the schema as a version query, so a schema bump is a new URL for
 * every edge and browser cache while a stale meta.json fails loudly rather
 * than pairing a new shader with an old tile.
 *
 * keepHeights holds the desert's level-0 height channel (as half-float bits)
 * for the camera's floor — the same bytes the shader reads.
 *
 * bounds builds the desert's bounds pyramid from every level's heights
 * (desert_bounds.js; the trace steps on it) and returns it as `seaView`,
 * the binding-9 view the desert reads, with its texture as
 * `boundsTexture`. onStage("bounds") is called as that build begins, for a
 * caller that names its steps (onProgress carries the download's fraction).
 * boundsBuilder, if given, builds it instead of buildDesertBounds on this
 * thread — the viewer passes buildDesertBoundsInWorker; it must return (or
 * resolve to) the same bytes.
 */
export async function loadOceanTile(device, baseUrl, onProgress,
                                    { keepCells = false, keepHeights = false,
                                      bounds = false, schema,
                                      onStage = null,
                                      boundsBuilder = null } = {}) {
  if (!schema) {
    throw new Error(`loadOceanTile(${baseUrl}) needs the schema it expects.`);
  }
  const meta = await (await fetch(`${baseUrl}/meta.json`,
                                  { cache: "no-cache" })).json();
  if (meta.schema !== schema) {
    throw new Error(
      `${baseUrl}/meta.json is schema ${JSON.stringify(meta.schema)}; this ` +
      `soar reads ${schema}. The tile and the app are from different ` +
      "releases — reload the page; if it persists the deploy is mixed.");
  }
  const version = `?v=${encodeURIComponent(meta.schema)}`;
  let cells = null;
  let heights = null;
  const boundsLevels = [];
  let boundsTexture = null;
  const format = meta.format ?? "rgba16float";
  const channels = SURFACE_FORMAT_CHANNELS[format];
  const files = meta.files
    ?? Array.from({ length: meta.mips }, (_, k) => `fif_mip${k}.bin`);
  // Checked before anything is allocated, and outside the try below: the
  // return reads them.
  const dx = meta.dx_m ?? meta.cell_m;
  if (!(dx > 0)) {
    throw new Error(
      `${baseUrl}/meta.json names neither dx_m nor cell_m; the tile has ` +
      "no texel size.");
  }
  const clockPeriod = tileClockPeriod(meta, baseUrl);
  // The surface is the one whose schema this is (checked above).
  const surface = Object.keys(K.SURFACE_TILE_SCHEMAS)
    .find((k) => K.SURFACE_TILE_SCHEMAS[k] === schema);
  if (!surface) {
    throw new Error(`loadOceanTile(${baseUrl}): ${schema} is no surface's ` +
                    "schema (constants.js SURFACE_TILE_SCHEMAS).");
  }
  const texture = device.createTexture({
    label: `${baseUrl}-surface-tile`,
    size: [meta.n, meta.n, 1],
    format,
    mipLevelCount: meta.mips,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  // From here the tile exists on the card and nothing else holds it: a
  // download or decode that fails part-way must give it back, or every
  // retry abandons another mip chain.
  try {
    for (let level = 0; level < meta.mips; level++) {
      const n = Math.max(1, meta.n >> level);
      const url = `${baseUrl}/${files[level]}${version}`;
      const stored = meta.compression === "gzip"
        ? new Uint8Array(await new Response(
            await fetchDecompressedStream(url, null, "gzip")).arrayBuffer())
        : await fetchBytes(url);
      const bytes = decodeSurfaceLevel(stored, n, meta);
      if (bounds) {
        const halves = new Uint16Array(
          bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        const h = new Float32Array(n * n);
        for (let i = 0; i < n * n; i++) h[i] = fromHalf(halves[i * channels]);
        boundsLevels.push(h);
      }
      if (keepHeights && level === 0) {
        const halves = new Uint16Array(
          bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        heights = new Uint16Array(n * n);
        for (let i = 0; i < n * n; i++) heights[i] = halves[i * channels];
      }
      // The city minimap draws the same cascade the shader raises buildings
      // out of, and the flight collider (collide.js) stands the same
      // buildings up again on the CPU, so both read the SAME bytes rather
      // than fetching anything of their own: level 0 is one texel per block,
      // which is the resolution the districts actually have. Decoded on the
      // way past and the fp16 dropped; all four channels city_cell reads —
      // density, rank, and the lot and street words — are kept.
      if (keepCells && level === 0) {
        const halves = new Uint16Array(
          bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        const density = new Float32Array(n * n);
        const rank = new Float32Array(n * n);
        const lot = new Float32Array(n * n);
        const street = new Float32Array(n * n);
        for (let i = 0; i < n * n; i++) {
          density[i] = fromHalf(halves[i * 4]);
          rank[i] = fromHalf(halves[i * 4 + 1]);
          lot[i] = fromHalf(halves[i * 4 + 2]);
          street[i] = fromHalf(halves[i * 4 + 3]);
        }
        cells = { n, cellM: meta.dx_m ?? meta.cell_m, density, rank, lot,
                  street };
      }
      device.queue.writeTexture(
        { texture, mipLevel: level },
        bytes,
        { bytesPerRow: n * channels * 2, rowsPerImage: n },
        [n, n, 1],
      );
      onProgress?.((level + 1) / meta.mips);
    }
    if (keepCells && !cells) {
      throw new Error(
        `${baseUrl} was asked for its block cells and has no mip 0 to take ` +
        "them from.");
    }
    if (bounds) {
      onStage?.("bounds");
      const pyramid = await (boundsBuilder ?? buildDesertBounds)(
        boundsLevels, dx);
      const side = Math.round(Math.sqrt(pyramid[0].length / 4
                                        / DESERT_BOUNDS_LAYERS));
      boundsTexture = device.createTexture({
        label: `${baseUrl}-bounds`,
        size: [side, side, DESERT_BOUNDS_LAYERS],
        format: "rgba16float",
        mipLevelCount: pyramid.length,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      pyramid.forEach((bits, level) => {
        const m = Math.max(1, side >> level);
        device.queue.writeTexture(
          { texture: boundsTexture, mipLevel: level }, bits,
          { bytesPerRow: m * 8, rowsPerImage: m },
          [m, m, DESERT_BOUNDS_LAYERS]);
      });
    }
  } catch (err) {
    texture.destroy();
    boundsTexture?.destroy();
    throw err;
  }
  return {
    view: texture.createView(),
    texture,
    dx,
    tileExtent: meta.tile_extent_m,
    // A tile-bound surface is a pure function of its tile, so its surface
    // frame folds at a whole number of tiles, invisibly (constants.js
    // SURFACE_FRAME_TILES: the sea's 16, the district's one).
    surfacePeriod: K.SURFACE_FRAME_TILES[surface] * meta.tile_extent_m,
    // The surface's clock: the period its animation repeats at (the city's
    // steam), where the scene clock wraps, or null where nothing moves.
    clockPeriod,
    maxLod: meta.mips - 1,
    cells,
    heights,
    seaView: boundsTexture?.createView({ dimension: "2d-array" }) ?? null,
    boundsTexture,
    meta,
  };
}

/** A tile's clock_period_s: a positive period, or null for a surface with
 *  no clock. Missing is an error, not a surface without a clock
 *  (soar_host.surface_clock_period_s is the twin). */
export function tileClockPeriod(meta, baseUrl) {
  if (!("clock_period_s" in meta)) {
    throw new Error(`${baseUrl}/meta.json has no clock_period_s (a period ` +
                    "in seconds, or null for a surface with no clock).");
  }
  const period = meta.clock_period_s;
  if (period !== null && !(period > 0)) {
    throw new Error(`${baseUrl}/meta.json clock_period_s must be positive ` +
                    `or null; got ${JSON.stringify(period)}.`);
  }
  return period;
}

// The two volume formats, and the bytes a texel of each costs. Extinction is
// r16float: it spans orders of magnitude and has no bounded range to quantize
// against. The ice fraction is r8unorm — a [0, 1] quantity read through a
// color ramp, where 1/255 steps are invisible and the halving is a whole
// field's worth of video memory, since the fraction sits BESIDE the
// extinction rather than replacing it. Both the demo bake
// (tools/prebake_demos.py) and the NetCDF ingest quantize the same way, so a
// field flown as a demo and the same field opened from disk agree.
export const VOLUME_TEXEL_BYTES = { r16float: 2, r8unorm: 1 };

/** A zero-initialized volume texture for an (nx, ny, nz) field. */
export async function createVolumeTexture(device, shape, label,
                                          format = "r16float") {
  const texelBytes = VOLUME_TEXEL_BYTES[format];
  if (!texelBytes) {
    throw new Error(
      `createVolumeTexture does not know the format '${format}'; it knows ` +
      `${Object.keys(VOLUME_TEXEL_BYTES).join(" and ")}.`);
  }
  const fit = volumeFits(device.limits, shape, texelBytes);
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
      format,
      // RENDER_ATTACHMENT is never drawn into. It is here for how Chrome
      // zero-fills a texture the first time a partial write touches it.
      // Dawn's D3D12 backend (TextureD3D12.cpp, ClearTexture) has two ways
      // to do that: a render-target clear when the resource allows one, and
      // otherwise a copy from a staging buffer it allocates for the WHOLE
      // texture at once — 256-aligned rows x height x depth, in one
      // Dawn_DynamicUploaderStaging buffer. The first 64 MB slab written
      // into a 2048x2048x512 fp16 field therefore asked Windows for a
      // 4294967296-byte staging buffer, twice D3D12's 2147483648 cap, and
      // the upload died with "Buffer size exceeds the max buffer size
      // limit" (bug report, RTX 5060, 2026-09-13; the same field loads on
      // Metal, whose buffers are allowed to be that large). Allowing the
      // render-target path costs nothing and makes the clear a
      // ClearRenderTargetView over the depth slices, no staging at all.
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
           | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    }));
}

/**
 * Write a slab of the field. `data` is laid out with z fastest, then y, then
 * x — the same order the texture wants, so this is a straight copy. The row
 * pitch comes from the view's own element size rather than a constant 2:
 * fp16 extinction arrives as a Uint16Array, the uint8 ice fraction as a
 * Uint8Array, and both go through here.
 * queue.writeTexture is used rather than a staging buffer on purpose: the
 * 256-byte bytesPerRow rule applies only to buffer-to-texture copies, and
 * nz*2 is almost never a multiple of 256.
 */
export function writeVolumeSlab(device, texture, data, origin, size) {
  device.queue.writeTexture(
    { texture, origin: { x: origin[0], y: origin[1], z: origin[2] } },
    data,
    { bytesPerRow: size[0] * data.BYTES_PER_ELEMENT,
      rowsPerImage: size[1] },
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
 * `stream` yields the DECOMPRESSED bytes, z fastest (the texture's own
 * layout, see writeVolumeSlab). `texelBytes` says which volume this is: 2
 * for the fp16 extinction field, 1 for the uint8 ice fraction beside it.
 * Throws if the stream ends short or runs long against `shape`.
 */
export async function streamWholeVolume(device, texture, stream, shape,
                                        onProgress, texelBytes = 2) {
  const [nx, ny, nz] = shape;
  const perPlaneBytes = ny * nz * texelBytes;     // one x plane
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
    // A writeTexture the browser refuses is reported asynchronously, as an
    // uncaptured validation error that main.js can only call "a bug in
    // cloudyview". Catch it here, where the field and the numbers are
    // known, and say what was being uploaded and how big it is.
    device.pushErrorScope("validation");
    writeVolumeSlab(
      device, texture,
      texelBytes === 2 ? new Uint16Array(slab.buffer, 0, filled / 2)
                       : new Uint8Array(slab.buffer, 0, filled),
      [0, 0, x0], [nz, ny, depth]);
    const refused = await device.popErrorScope();
    if (refused) {
      const gb = (nx * ny * nz * texelBytes) / 2 ** 30;
      const err = new Error(
        `The GPU refused the upload of x planes ${x0}-${x0 + depth - 1} ` +
        `(of ${nx}) into the ${nx}x${ny}x${nz} volume: ${refused.message}`);
      err.advice =
        `This volume is ${gb.toFixed(1)} GB on the card. The slab itself ` +
        `was ${(filled / 2 ** 20).toFixed(0)} MB, so the number the ` +
        "browser objects to is one it chose for the whole texture, not " +
        "something the page asked for. A coarser variant of the same " +
        "field is in the list.";
      throw err;
    }
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

/**
 * A world position in the city tile's own frame, or null off a city scene.
 *
 * This is raymarch.wgsl's city_glow_sample mapping and nothing else:
 *
 *     uv = (xy - u.ocean.yz) / u.ocean_params.y
 *
 * with the tile sampler repeating, i.e. only the fractional part of uv
 * selects a block. So the city coordinate of a camera is
 * `(worldXY - cityOffsetM) mod tileExtent`, folded into `[0, tileExtent)`,
 * and z is world z unchanged (the tile is 2D; CITY_GROUND_Z is world).
 *
 * The fold is what makes the number worth showing. Relative coordinates
 * measure across the CLOUD box, so the same relative triple lands in a
 * different district under every differently-sized field; the tile frame is
 * fixed in world metres and periodic, so this triple names one place in the
 * city whatever field is flying over it.
 *
 * Callers must pass the scene's own `cityOffsetM` and `oceanTileExtent` —
 * the same two values uniforms.js writes into rows 8 and 9 — so the readout
 * and the shader cannot disagree.
 */
export function cityFramePosition(position, cityOffsetM, tileExtentM) {
  if (!(tileExtentM > 0)) return null;
  const fold = (v, offset) => {
    const t = (v - offset) % tileExtentM;
    return t < 0 ? t + tileExtentM : t;
  };
  return [fold(position[0], cityOffsetM[0]),
          fold(position[1], cityOffsetM[1]),
          position[2]];
}

/**
 * Cyberpunk's opening view, in the loaded field's own relative frame.
 *
 * CITY_START_CAMERA is pinned in city metres because that is the frame the
 * view lives in — a street canyon is a place in the TILE, not a fraction of
 * whatever cloud box is flying over it. The conversion here places the
 * CLOUD frame: offset + city metres, made relative to this field's box. The
 * camera then derives its own surface frame from that world position
 * (FlightCamera._resyncSurface), which recovers positionCityM folded at the
 * tile — so both frames open on the street the capture named, whatever the
 * domain size.
 */
function cityStartCamera(bmin, bmax) {
  const [cx, cy, cz] = K.CITY_START_CAMERA.positionCityM;
  const world = [K.CITY_TILE_OFFSET_M[0] + cx, K.CITY_TILE_OFFSET_M[1] + cy, cz];
  return {
    position: worldToRelative(world, bmin, bmax),
    azimuth: K.CITY_START_CAMERA.azimuth,
    elevation: K.CITY_START_CAMERA.elevation,
    fov: K.CITY_START_CAMERA.fov,
  };
}

export class Scene {
  constructor(device, parts) {
    this.device = device;
    Object.assign(this, parts);
  }

  /**
   * Where a world position is in the city, or null when this is not a city
   * scene. See cityFramePosition — the ocean scenes have no second frame and
   * must not grow a meaningless one.
   */
  cityPosition(position) {
    if (!this.city) return null;
    return cityFramePosition(position, this.cityOffsetM, this.oceanTileExtent);
  }

  get nested() { return Boolean(this._nest); }
  get nestBmin() { return this._nest?.bmin ?? [0, 0, 0]; }
  get nestBmax() { return this._nest?.bmax ?? [0, 0, 0]; }
  get minVoxelNestM() { return this._nest?.minVoxelM ?? this.minVoxelM; }
  get nestView() { return (this._nest?.texture ?? this._nestDummy).createView(); }
  /** NetCDF group the nest was read from, null when it came from the root. */
  get nestGroup() { return this._nest?.name || null; }

  /**
   * The ice-fraction volume for the ice-detection mode, or null until
   * somebody asks for it. A getter for the same reason nestView is one: the
   * bind group is rebuilt from the scene, and a view cached beside a texture
   * that has since been attached is a frame sampling the stand-in.
   */
  get iceView() { return this.iceTexture?.createView() ?? null; }

  /**
   * Whether the ice fraction is here or could be fetched. False is a fact
   * about the field — a demo bake carries no ice variable and neither do
   * plenty of files — not a failure, and the menu says so in those words.
   */
  get iceAvailable() { return Boolean(this.iceTexture || this.iceSource); }

  /**
   * Why the ice fraction is not on offer, or null when it is. A fact about
   * the data rather than a failure, and since the demos gained their own
   * prebaked ice volumes it is the SAME fact in both cases: the source this
   * field came from carried no ice. (It used to be two — a bake threw the
   * phase split away — and that is no longer true of any case that has one
   * to throw.)
   */
  get iceNote() {
    if (this.iceAvailable) return null;
    if (this.prebaked) {
      return "This demo has no ice data — its source field is liquid only, " +
             "so there is no phase split to show. The other cases carry one.";
    }
    // An upload that was OFFERED a second file and declined it gets a
    // different sentence from one that was never asked: the first has an
    // obvious remedy and saying "this file carries none" would be true and
    // useless. The offer is made at load and only at load, because the
    // extinction on screen already includes whatever ice there is — see
    // ingest/index.js.
    return this.iceSkipped
      ? "This field has no ice data. You were offered an ice file when it " +
        "loaded and continued without one — load the field again to attach " +
        "an ice file on the same grid."
      : "This field has no ice data — ice detection needs an ice mixing " +
        "ratio variable (QI / IWC) and this file carries none.";
  }

  /** Take ownership of a lazily-read ice-fraction volume (loadIceVolume). */
  attachIce(texture) {
    this.iceTexture?.destroy();
    this.iceTexture = texture;
  }

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
    this.iceTexture?.destroy();
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
 * It takes the surface kind — "ocean", "desert", or "city" for the night
 * city — because which tile a demo stands on is the demo's call
 * (meta.surface)…
 *
 * …unless the PAGE overrides it. `surfaceKind` does that, for two callers:
 * cyberpunk mode, which flies the same demos every other mode does standing
 * them on the city (the alternative was a parallel set of city-surfaced
 * bakes, the same volumes twice on disk to carry one boolean), and
 * ?surface= (main.js), which stands a demo on either day surface.
 */
export async function loadDemoScene(device, baseUrl, surface, progress,
                                    { surfaceKind: forcedSurface = null } = {}) {
  progress?.("Downloading the cloud field…", 0);
  const meta = await (await fetch(`${baseUrl}/meta.json`)).json();
  const surfaceKind = forcedSurface ?? meta.surface ?? "ocean";
  if (!["ocean", "desert", "city"].includes(surfaceKind)) {
    throw new Error(
      `The '${meta.id}' demo asks for surface '${surfaceKind}', which this ` +
      "viewer does not know. It knows 'ocean', 'desert' and 'city'.");
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
      // An error that already knows what to advise (a refused writeTexture,
      // see streamWholeVolume) keeps its own; the memory guess below is for
      // the bare "Failed to fetch" that knows nothing.
      wrapped.advice = err?.advice ||
        `This field is ${gb.toFixed(1)} GB unpacked. If the connection is ` +
        "fine, the machine likely ran short of memory — a coarse variant " +
        "of the same field is in the list, or retry after closing other " +
        "tabs.";
      throw wrapped;
    }

    const mapBytes = await fetchBytes(`${baseUrl}/${meta.map.file ?? "map.bin"}`);
    progress?.({ city: "Raising the city…",
                 desert: "Laying out the desert…" }[surfaceKind]
               ?? "Loading the ocean surface…", 0.95);
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
      seaView: surfaceTile.seaView ?? null,
      oceanFifDx: surfaceTile.dx,
      oceanTileExtent: surfaceTile.tileExtent,
      surfacePeriod: surfaceTile.surfacePeriod,
      clockPeriod: surfaceTile.clockPeriod,
      oceanMaxLod: surfaceTile.maxLod,
      city: surfaceKind === "city",
      desert: surfaceKind === "desert",
      desertReliefM: surfaceKind === "desert"
        ? desertReliefM(surfaceTile.meta) : null,
      desertTileRows: surfaceKind === "desert"
        ? desertTileRows(surfaceTile.meta) : null,
      desertFloorAt: surfaceKind === "desert"
        ? desertFloorAt(surfaceTile) : null,
      // The city tile's per-block cascade, for the minimap to draw the city
      // with. Null for an ocean scene, and null is what makes the minimap
      // build the cloud map instead — see Minimap's constructor.
      cityCells: surfaceTile.cells,
      // The demo's own offset when the demo asked for a city; the mode's when
      // the mode did, because a borrowed demo has no opinion about where a
      // city it never expected should sit under it.
      cityOffsetM: meta.city?.offset_m ?? (
        forcedSurface === "city" ? K.CITY_TILE_OFFSET_M : [0.0, 0.0]),
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
      // A bake is extinction and nothing else: the liquid and ice mixing
      // ratios went into it and cannot be taken back out. Said here rather
      // than inferred from an absent variable name, because the ice-detection
      // mode has to tell the two cases apart to explain either.
      prebaked: true,
      // …which is why the phase split ships as its OWN file when the source
      // had one. What it takes to fetch, not the fraction itself: it is a
      // third of a gigabyte on the larger cases and most flights never press
      // I. `kind` is what tells toggleIceMode which loader to hand this to —
      // a demo needs no NetCDF reader, and pulling h5wasm in to tint a baked
      // field would be the download all over again. Absent `ice` block means
      // the source carried no ice variable (DYCOMS), and iceAvailable is
      // then false and the menu says why.
      iceSource: meta.ice ? {
        kind: "demo",
        url: `${baseUrl}/${meta.ice.file}`,
        shape,
        compression: meta.ice.compression === "gzip" ? "gzip" : null,
        format: meta.ice.format,
      } : null,
      // Under a forced city the demo's sun is a desert afternoon and the
      // scene's light is a moon, so the mode's moon replaces it. A demo that
      // asked for the city itself carries a moon in this field already.
      sun: forcedSurface === "city" ? K.CITY_MOON : meta.sun,
      // Where the landing page's still was taken from, so flight can open
      // there — see Camera.applyStart and viewer boot. Under a forced city
      // the mode's own opening view replaces it, for the same reason the
      // mode's moon replaces the demo's sun: the demo's camera is a daylight
      // aerial over cloud tops, and cyberpunk is a street. The mode's view
      // is pinned in the city frame (see CITY_START_CAMERA) and converted to
      // THIS field's relative frame here, where the box is known.
      startCamera: forcedSurface === "city"
        ? cityStartCamera(bmin, bmax) : (meta.still?.camera ?? null),
    });
    progress?.("Ready.", 1);
    return scene;
  } catch (err) {
    volumeTexture.destroy();
    nestDummy?.destroy();
    throw err;
  }
}

/**
 * Fetch a demo's prebaked ice fraction and upload it.
 *
 * The demo counterpart of ingest's loadIceVolume, and the same bargain: a
 * second volume the size of the field, paid for only when somebody asks for
 * the ice-detection mode. `source` is the scene's `iceSource` when its kind
 * is "demo". Returns the texture; the caller owns it from then on.
 *
 * No crop is negotiated and none is possible: the bake wrote this volume
 * from the same array the extinction came out of, with the same window and
 * the same z band, so the two line up texel for texel by construction. The
 * shape here is the SCENE's — if the file behind the URL disagrees,
 * streamWholeVolume says so rather than uploading a shifted field.
 */
export async function loadDemoIceVolume(device, source, { progress } = {}) {
  if (source.format !== "r8unorm") {
    throw new Error(
      `This demo's ice fraction is baked as '${source.format}'; the viewer ` +
      "uploads r8unorm. Re-bake it with tools/prebake_demos.py --ice-only.");
  }
  progress?.("Downloading the ice fraction…", 0);
  const texture = await createVolumeTexture(
    device, source.shape, "the demo ice fraction", "r8unorm");
  try {
    await streamWholeVolume(
      device, texture,
      await fetchDecompressedStream(
        source.url,
        (f) => progress?.("Downloading the ice fraction…", f * 0.9),
        source.compression),
      source.shape,
      (f) => progress?.("Uploading the ice fraction…", 0.9 + 0.1 * f),
      VOLUME_TEXEL_BYTES.r8unorm);
  } catch (err) {
    texture.destroy();
    throw err;
  }
  progress?.("Ready.", 1);
  return texture;
}

export { volumeAABB, minVoxelSize };
