// Device acquisition and the honest-failure floor.
//
// Everything downstream assumes a working device with known limits, so this
// module owns the four ways that assumption breaks: no WebGPU at all, no
// adapter, a field that cannot fit, and a device that dies mid-flight. Each
// one gets a message naming the actual number involved. A blank canvas is
// never an acceptable outcome here.

"use strict";

// Limits the renderer actually consumes. We ask for the adapter's own
// reported value on each, because "as much as the machine will give" is the
// whole point — a 2 GB field is the use case, not the edge case. Requesting
// a limit the adapter does not report throws at requestDevice(), so these
// are read from the adapter rather than hardcoded.
export const NEEDED_LIMITS = [
  "maxTextureDimension2D",       // render targets, incl. high-res stills
  "maxTextureDimension3D",       // the volume — the binding constraint
  "maxBufferSize",               // staging for slice uploads
  "maxStorageBufferBindingSize", // extinction compute
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
];

// Dawn (Chrome/Edge) clamps maxTextureDimension3D to the spec floor whatever
// the hardware can do; wgpu (Firefox) passes the real value through. Measured
// 2026-08-04 on an RTX 5080: Chrome 2048, Firefox 16384. Not a bug we can
// route around — it decides which browser opens a full-resolution LES field.
// It is a cell count, not a texel count with something subtracted: 2048 is a
// common LES lateral size and 2046 is not, so soar spends no texel on
// anything but data (scene.js, raymarch.wgsl sample_level).
export const SPEC_FLOOR_TEXTURE_3D = 2048;

export class WebGPUUnavailable extends Error {
  constructor(message, detail) {
    super(message);
    this.name = "WebGPUUnavailable";
    this.detail = detail || "";
  }
}

export class AllocationFailed extends Error {
  constructor(message, { label, bytes } = {}) {
    super(message);
    this.name = "AllocationFailed";
    this.label = label;
    this.bytes = bytes;
  }
}

/** Rough browser identification — only ever used to pick advice text. */
export function browserGuess(ua = navigator.userAgent) {
  if (/Firefox\//.test(ua)) return "firefox";
  if (/Edg\//.test(ua)) return "edge";
  if (/Chrome\//.test(ua)) return "chrome";
  if (/Safari\//.test(ua)) return "safari";
  return "unknown";
}

/**
 * Why WebGPU is missing, and what this particular person can do about it.
 * Returned as {title, body} of plain text — the caller renders it.
 */
export function missingWebGPUAdvice(ua = navigator.userAgent) {
  const browser = browserGuess(ua);
  const linux = /Linux|X11/.test(ua) && !/Android/.test(ua);
  if (browser === "firefox") {
    return {
      title: "Firefox has WebGPU, but it is switched off here.",
      body:
        "Open about:config, accept the warning, search for " +
        "dom.webgpu.enabled and set it to true, then reload this page. " +
        (linux
          ? "On Linux you may also need gfx.webgpu.force-enabled if your " +
            "GPU is not on the allowlist yet."
          : ""),
    };
  }
  if (browser === "chrome" || browser === "edge") {
    return {
      title: "This build of Chrome is not exposing WebGPU.",
      body: linux
        ? "WebGPU on Linux still needs flags. Launch with " +
          "--enable-unsafe-webgpu --enable-features=Vulkan " +
          "--ozone-platform=x11 (Vulkan and Wayland are mutually " +
          "exclusive, so x11 is not optional here), or use Firefox with " +
          "dom.webgpu.enabled, which needs no command line."
        : "Update to Chrome 113 or newer, and check that hardware " +
          "acceleration is on in Settings → System. In a virtual machine " +
          "or over remote desktop there may be no GPU to reach.",
    };
  }
  if (browser === "safari") {
    return {
      title: "This version of Safari has no WebGPU.",
      body: "Safari 26 or newer supports it. On older versions, enable " +
            "WebGPU under Develop → Feature Flags.",
    };
  }
  return {
    title: "This browser has no WebGPU.",
    body: "Chrome 113+, Edge 113+, Firefox 141+, or Safari 26+, on a " +
          "machine with a real GPU.",
  };
}

/**
 * Get an adapter, or throw WebGPUUnavailable with advice attached.
 * Kept separate from the device so the landing page can report what this
 * machine can open before anything large is allocated.
 */
export async function acquireAdapter() {
  if (!navigator.gpu) {
    const advice = missingWebGPUAdvice();
    throw new WebGPUUnavailable(advice.title, advice.body);
  }
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
  } catch (err) {
    throw new WebGPUUnavailable(
      "Asking for a GPU adapter failed.", String(err && err.message || err));
  }
  if (!adapter) {
    throw new WebGPUUnavailable(
      "WebGPU is present but no GPU adapter was offered.",
      "The GPU may be blocklisted by the browser, disabled in settings, or " +
      "absent (a VM or remote session). In Chrome, chrome://gpu says which.");
  }
  return adapter;
}

/**
 * Create the device, requesting the adapter's full value on every limit the
 * renderer uses. Any limit the adapter does not report is skipped rather
 * than defaulted, so this never throws for asking too much.
 */
export async function acquireDevice(adapter) {
  const requiredLimits = {};
  for (const name of NEEDED_LIMITS) {
    const value = adapter.limits[name];
    if (typeof value === "number" && Number.isFinite(value)) {
      requiredLimits[name] = value;
    }
  }
  const requiredFeatures = [];
  if (adapter.features.has("float32-filterable")) {
    requiredFeatures.push("float32-filterable");
  }
  let device;
  try {
    device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
  } catch (err) {
    throw new WebGPUUnavailable(
      "The GPU adapter refused to create a device.",
      String(err && err.message || err));
  }
  return device;
}

/** Snapshot of the limits that decide what this machine can open. */
export function limitsSummary(adapter) {
  const l = adapter.limits;
  return {
    maxTextureDimension3D: l.maxTextureDimension3D,
    maxTextureDimension2D: l.maxTextureDimension2D,
    maxBufferSize: l.maxBufferSize,
    clampedToSpecFloor: l.maxTextureDimension3D <= SPEC_FLOOR_TEXTURE_3D,
    vendor: adapter.info?.vendor || "",
    architecture: adapter.info?.architecture || "",
    description: adapter.info?.description || "",
  };
}

/**
 * Can a field of this shape become a 3D texture here?
 *
 * `shape` is [nx, ny, nz] in field-axis order — the field itself, with no
 * border: soar uploads one texel per voxel and does its boundary work in the
 * shader, so a 2048-cell axis needs exactly 2048 texels and clears the spec
 * floor rather than missing it by two. The texture is created transposed (see
 * scene.js), so every axis must clear the same limit and which one is which
 * does not matter. Returns {ok, message, advice}.
 */
export function volumeFits(limits, shape, bytesPerVoxel = 2) {
  const cap = limits.maxTextureDimension3D;
  const worst = Math.max(shape[0], shape[1], shape[2]);
  if (worst <= cap) {
    const bytes = shape[0] * shape[1] * shape[2] * bytesPerVoxel;
    return { ok: true, bytes, message: "", advice: "" };
  }
  const axis = ["x", "y", "z"][shape.indexOf(worst)];
  const clamped = cap <= SPEC_FLOOR_TEXTURE_3D;
  const factor = Math.ceil(worst / cap);
  return {
    ok: false,
    bytes: 0,
    message:
      `This field needs ${worst} texels on ${axis} (${shape[0]}x${shape[1]}` +
      `x${shape[2]} cells); this browser allows ${cap}.`,
    advice: clamped
      ? "Chrome reports the WebGPU spec minimum of 2048 no matter what the " +
        "card can do. Firefox reports the hardware's real limit — on this " +
        `field that is the difference between opening it and not. Or ` +
        `decimate by ${factor}x, or crop to ${cap} cells on ${axis}.`
      : `Decimate by ${factor}x or crop to ${cap} cells on ${axis}.`,
  };
}

/**
 * Run an allocation inside error scopes so an OOM is a sentence rather than
 * a dead canvas. WebGPU reports allocation failure asynchronously, so the
 * texture object exists and looks fine until this await resolves.
 */
export async function guardAllocation(device, label, bytes, fn) {
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let result;
  try {
    result = fn();
  } catch (err) {
    await device.popErrorScope();
    await device.popErrorScope();
    throw err;
  }
  const validation = await device.popErrorScope();
  const oom = await device.popErrorScope();
  if (validation || oom) {
    // The object exists — the failure is only reported here, asynchronously.
    // Throwing without releasing it leaves gigabytes reachable from nothing
    // but a garbage collector, which is exactly the wrong thing to do at the
    // moment the GPU has just said it is out of memory.
    releaseAllocation(result);
  }
  if (validation) {
    throw new AllocationFailed(
      `${label} was rejected: ${validation.message}`, { label, bytes });
  }
  if (oom) {
    const gb = bytes ? ` (${(bytes / 1e9).toFixed(2)} GB)` : "";
    throw new AllocationFailed(
      `The GPU ran out of memory allocating ${label}${gb}. Close other ` +
      "tabs and GPU-heavy applications, or try a field with a smaller " +
      "grid size.",
      { label, bytes });
  }
  return result;
}

/** Destroy whatever an allocation thunk produced: one object, or a bag of them. */
function releaseAllocation(result) {
  if (!result || typeof result !== "object") return;
  if (typeof result.destroy === "function") { result.destroy(); return; }
  for (const value of Object.values(result)) {
    if (value && typeof value.destroy === "function") value.destroy();
  }
}

/**
 * Destroy textures and buffers only once the GPU has finished with the work
 * already submitted.
 *
 * Destroying a resource an in-flight command buffer still references is legal
 * by the letter of the spec and is the kind of thing that takes a browser's
 * GPU process down rather than raising — see Renderer._targetsFor, which has
 * always waited. Anything replaced mid-flight (a depth buffer on a resize, a
 * nest that was just unbound) needs the same barrier.
 */
export function retireAfterSubmittedWork(device, ...resources) {
  const live = resources.filter(Boolean);
  if (!live.length) return Promise.resolve();
  const release = () => {
    for (const resource of live) {
      // A device destroyed while this was pending has already reclaimed
      // everything on it; destroy() then has nothing to do and may object.
      try { resource.destroy(); } catch { /* the device took it first */ }
    }
  };
  return device.queue.onSubmittedWorkDone().then(release, release);
}

/**
 * Wire up the two asynchronous ways a device reports trouble. `onLost` fires
 * for driver resets and OOM kills; `onError` for anything no error scope
 * caught. Both are fatal to the current session, so the caller shows a
 * message and stops the frame loop rather than limping on.
 */
export function watchDevice(device, { onLost, onError } = {}) {
  device.lost.then((info) => {
    // "destroyed" is our own teardown — not a failure.
    if (info.reason === "destroyed") return;
    onLost?.(
      info.message ||
        "The GPU device was lost. This usually means a driver reset or an " +
        "out-of-memory kill.",
      info.reason);
  });
  device.addEventListener("uncapturederror", (event) => {
    console.error("WebGPU:", event.error);
    onError?.(String(event.error?.message || event.error));
  });
}
