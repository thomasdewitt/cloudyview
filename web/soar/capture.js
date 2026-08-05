// Stills and video.
//
// A still is not one frame. The live view converges as you sit still, and a
// single-frame capture cannot reproduce what you were looking at when you
// decided to press the button — so a capture re-renders the same camera many
// times and averages, exactly as the desktop does.

"use strict";

import * as K from "./constants.js";

const STILL_FORMAT = "rgba8unorm";

/**
 * Render `frames` accumulated passes at `size` into an offscreen target and
 * read the result back as an ImageData.
 *
 * Render scale is forced to 1.0 for the duration: a still has no framerate to
 * protect, and capturing the flight's downscaled target would be a strictly
 * worse picture for no gain. Restored afterwards so the live view is
 * untouched, which also means the accumulation the viewer had built up is
 * spent — the caller resets it.
 */
export async function renderStill(device, renderer, view, size, frames,
                                  overlays = null) {
  const [w, h] = size;
  if (w > device.limits.maxTextureDimension2D ||
      h > device.limits.maxTextureDimension2D) {
    throw new Error(
      `${w}x${h} is larger than this browser's ${device.limits.maxTextureDimension2D} ` +
      "texture limit. Choose a smaller capture size.");
  }
  const texture = device.createTexture({
    label: "soar-still",
    size: [w, h], format: STILL_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const savedScale = renderer.flightRenderScale;
  const savedTier = renderer.qualityTier;
  try {
    renderer.setQualityTier("high");
    renderer.setRenderScale(1.0);
    renderer.resetAccumulation();
    const targetView = texture.createView();
    for (let i = 0; i < frames; i++) {
      await renderer.drawFrame(
        targetView, STILL_FORMAT, size,
        { ...view, frameIndex: (view.frameIndex ?? 0) + i },
        // Every pass clears the target before blitting into it, so drawing
        // the overlays each time composites them once, not sixty-four times.
        { deltaSeconds: null, overlays });
    }
    return await readBack(device, texture, w, h);
  } finally {
    texture.destroy();
    renderer.setQualityTier(savedTier);
    renderer.setRenderScale(savedScale);
    renderer.resetAccumulation();
  }
}

/** Copy a texture to the CPU, undoing the 256-byte row padding. */
export async function readBack(device, texture, w, h) {
  const rowBytes = Math.ceil((w * 4) / 256) * 256;
  const buffer = device.createBuffer({
    size: rowBytes * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow: rowBytes, rowsPerImage: h },
    [w, h, 1]);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buffer.getMappedRange());
  const image = new ImageData(w, h);
  for (let y = 0; y < h; y++) {
    image.data.set(
      src.subarray(y * rowBytes, y * rowBytes + w * 4), y * w * 4);
  }
  for (let i = 3; i < image.data.length; i += 4) image.data[i] = 255;
  buffer.unmap();
  buffer.destroy();
  return image;
}

// --- PNG metadata ---------------------------------------------------------

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * Splice a tEXt chunk into a PNG.
 *
 * canvas.toBlob gives no way to attach metadata, and the metadata is the
 * point: a screenshot that cannot tell you which field, camera and sun made
 * it is a pretty picture rather than a record. Same key and payload as the
 * desktop's render_metadata, so the two are readable by the same tools.
 */
export function embedPngText(pngBytes, keyword, text) {
  const encoder = new TextEncoder();
  const key = encoder.encode(keyword);
  const value = encoder.encode(text);
  const dataLength = key.length + 1 + value.length;

  const chunk = new Uint8Array(12 + dataLength);
  const dv = new DataView(chunk.buffer);
  dv.setUint32(0, dataLength);
  chunk.set(encoder.encode("tEXt"), 4);
  chunk.set(key, 8);
  chunk[8 + key.length] = 0;
  chunk.set(value, 9 + key.length);
  dv.setUint32(8 + dataLength, crc32(chunk.subarray(4, 8 + dataLength)));

  // After IHDR (which is always the first chunk, 8 signature + 25 bytes).
  const insertAt = 8 + 25;
  const out = new Uint8Array(pngBytes.length + chunk.length);
  out.set(pngBytes.subarray(0, insertAt), 0);
  out.set(chunk, insertAt);
  out.set(pngBytes.subarray(insertAt), insertAt + chunk.length);
  return out;
}

export async function imageDataToPng(image, metadata) {
  const canvas = new OffscreenCanvas(image.width, image.height);
  canvas.getContext("2d").putImageData(image, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  if (!metadata) return blob;
  const bytes = new Uint8Array(await blob.arrayBuffer());
  return new Blob(
    [embedPngText(bytes, "cloudyview.render_metadata",
                  JSON.stringify(metadata))],
    { type: "image/png" });
}

/** Hand a blob to the browser's download machinery. */
export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function timestampedName(prefix, suffix) {
  const now = new Date();
  const p = (v, n = 2) => String(v).padStart(n, "0");
  const stamp = `${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}` +
    `_${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}` +
    `_${p(now.getMilliseconds(), 3)}`;
  return `${prefix}_${stamp}${suffix}`;
}

// --- video ----------------------------------------------------------------

/**
 * Is frame-exact video encoding available here?
 *
 * WebCodecs is the path that matters: it takes the timestamp I give it, so a
 * 30-second flight is 30 seconds of video whether each frame took 16 ms or
 * six seconds to converge. Screen capture cannot do that.
 */
export async function videoSupport(width, height, fps) {
  if (typeof VideoEncoder === "undefined") {
    return { kind: "none",
             why: "This browser has no WebCodecs VideoEncoder." };
  }
  // Even dimensions are a hard H.264 requirement in every implementation.
  const w = width & ~1, h = height & ~1;
  for (const codec of ["avc1.640034", "avc1.640028", "vp09.00.10.08", "vp8"]) {
    const config = {
      codec, width: w, height: h,
      bitrate: 24_000_000, framerate: fps,
      latencyMode: "quality",
      ...(codec.startsWith("avc1") ? { avc: { format: "avc" } } : {}),
    };
    try {
      const support = await VideoEncoder.isConfigSupported(config);
      if (support.supported) {
        return { kind: "webcodecs", config: support.config ?? config };
      }
    } catch { /* try the next codec */ }
  }
  return { kind: "none",
           why: "No codec this browser offers could be configured." };
}
