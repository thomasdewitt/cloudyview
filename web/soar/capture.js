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
export async function renderStill(device, renderer, view, size, tier,
                                  overlays = null, onProgress = null) {
  const target = createOfflineTarget(device, size, "soar-still");
  const saved = beginOfflineRender(renderer, tier);
  try {
    await renderAccumulated(
      renderer, target.view, size, view, captureFrames(tier), overlays,
      onProgress && ((done, total) =>
        onProgress(STILL_MARCH_SHARE * done / total)));
    // The read-back is where the GPU is actually waited on. It gets the rest
    // of the bar rather than a stage name of its own — the caller says what
    // is happening once, and the bar says how far along it is.
    onProgress?.(STILL_MARCH_SHARE);
    return await readBack(device, target.texture, size[0], size[1]);
  } finally {
    target.texture.destroy();
    endOfflineRender(renderer, saved);
  }
}

/** How much of a still's progress bar the marching owns; the rest is the
 *  read-back and the PNG encode. */
const STILL_MARCH_SHARE = 0.85;

/**
 * One offscreen colour target for an offline render.
 *
 * Separate from renderStill because a video makes hundreds of frames at the
 * same size, and allocating and destroying a 4K texture per frame would
 * dominate the render.
 */
export function createOfflineTarget(device, size, label) {
  const [w, h] = size;
  const limit = device.limits.maxTextureDimension2D;
  if (w > limit || h > limit) {
    throw new Error(
      `${w}x${h} is larger than this browser's ${limit} texture limit. ` +
      "Choose a smaller capture size.");
  }
  const texture = device.createTexture({
    label: label || "soar-offline",
    size: [w, h], format: STILL_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  return { texture, view: texture.createView(), format: STILL_FORMAT };
}

/**
 * Accumulation passes for a capture at `tier`: the tier's parked sample
 * count, in presented frames — Max reaches its 32 samples in 4 passes of 8.
 */
export function captureFrames(tier) {
  const preset = K.QUALITY_PRESETS[tier];
  if (!preset) throw new Error(`unknown capture tier '${tier}'.`);
  return Math.max(1, Math.round(
    K.PARKED_ACCUM_FRAMES_BY_TIER[tier] / (preset.sppPerFrame ?? 1)));
}

/**
 * Configure the renderer for an offline render at `tier`, returning what to
 * restore. A capture is the tier's FLIGHT configuration marched at the
 * capture resolution — pure preset, no session overrides — with spp from
 * the parked table (Thomas, 2026-08-20). Pure preset is also what keeps a
 * capture reproducible from the CLI: witness --soar-tier applies the same
 * table with nothing invisible mixed in.
 */
export function beginOfflineRender(renderer, tier = "high") {
  const saved = { scale: renderer.flightRenderScale,
                  tier: renderer.qualityTier,
                  lightCacheMode: renderer.lightCacheMode,
                  skyProbeMode: renderer.skyProbeMode,
                  parkedSppOverride: renderer.parkedSppOverride };
  renderer.setQualityTier(tier);
  renderer.lightCacheMode = "auto";
  renderer.skyProbeMode = "auto";
  renderer.parkedSppOverride = null;
  renderer.setRenderScale(1.0);
  // Most tiers read the sun-tau cache, and a capture must not race a
  // half-done bake: the frames before completion would light differently
  // from the frames after. Finish it here — the capture owns the GPU.
  while (renderer.lightBakePending) renderer.stepLightBake(64);
  renderer.resetAccumulation();
  return saved;
}

export function endOfflineRender(renderer, saved) {
  renderer.setQualityTier(saved.tier);
  renderer.setRenderScale(saved.scale);
  renderer.lightCacheMode = saved.lightCacheMode;
  renderer.skyProbeMode = saved.skyProbeMode;
  renderer.parkedSppOverride = saved.parkedSppOverride;
  renderer.resetAccumulation();
}

/**
 * Accumulate `frames` jittered passes of one camera into `targetView`.
 *
 * `onProgress(done, total)` is optional, and asking for it changes how the
 * loop runs: drawFrame only SUBMITS work, so without a yield the whole
 * accumulation is one unbroken run of microtasks and the browser cannot paint
 * until the last pass is in — which is why a still's progress bar used to sit
 * at 0 and then jump to done. Reporting therefore comes with a wait for the
 * next paint, which is also what paces the loop honestly: a frame callback
 * cannot arrive faster than the compositor can present, and the compositor is
 * behind the same GPU doing the marching. (The same feedback the tier probe's
 * cadence clock runs on.)
 *
 * The wait is time-gated rather than per-pass. A fast card can finish a pass
 * in less than a frame interval, and stopping for a paint after every one of
 * them would cost more than the marching does; twice a second is enough for a
 * bar to read as moving, and cheap enough that the account of the work cannot
 * measurably slow the work.
 */
export async function renderAccumulated(renderer, targetView, size, view,
                                        frames, overlays = null,
                                        onProgress = null) {
  renderer.resetAccumulation();
  let lastReport = performance.now();
  for (let i = 0; i < frames; i++) {
    await renderer.drawFrame(
      targetView, STILL_FORMAT, size,
      { ...view, frameIndex: (view.frameIndex ?? 0) + i },
      // Every pass clears the target before blitting into it, so drawing the
      // overlays each time composites them once, not sixty-four times.
      { deltaSeconds: null, overlays });
    if (!onProgress) continue;
    const now = performance.now();
    if (now - lastReport < PROGRESS_INTERVAL_MS && i < frames - 1) continue;
    lastReport = now;
    onProgress(i + 1, frames);
    await nextPaint();
  }
}

const PROGRESS_INTERVAL_MS = 500;

/**
 * Wait for the browser to paint — or don't, if it isn't going to.
 *
 * A hidden tab is served no frame callbacks at all, so waiting for one there
 * would park a capture the moment its user switched away to wait for it, and
 * a timer instead of the callback runs into background throttling. A tab
 * nobody is looking at has no bar to redraw either, so the honest thing is to
 * stop waiting: the capture runs the way it did before there was a bar.
 */
const nextPaint = () => {
  const doc = globalThis.document;
  if (doc?.visibilityState === "hidden") return Promise.resolve();
  return new Promise((resolve) => {
    // Switching away mid-wait is the same situation arriving a moment later,
    // and the pending frame callback would never come.
    const done = () => {
      doc?.removeEventListener("visibilitychange", done);
      resolve();
    };
    doc?.addEventListener("visibilitychange", done, { once: true });
    requestAnimationFrame(done);
  });
};

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

// Codec selection lives in video.js, which decides by trial-encoding a real
// frame rather than by asking isConfigSupported — a browser can answer
// "supported" and then produce nothing.
