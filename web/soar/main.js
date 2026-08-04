// Entry point: decide what this browser can do, take a field from the user,
// and hand it to the viewer.
//
// Landing and viewer are one page rather than two. A File chosen here is a
// live handle; navigating would throw it away and there is no way to pass an
// unuploaded multi-gigabyte file across a page load. So the landing hides and
// the viewer takes over.

"use strict";

import {
  acquireAdapter, acquireDevice, limitsSummary, watchDevice,
  WebGPUUnavailable, SPEC_FLOOR_TEXTURE_3D, browserGuess,
} from "./gpu.js";

const el = (id) => document.getElementById(id);

const dom = {
  landing: el("landing"),
  viewer: el("viewer"),
  header: el("site-header"),
  capability: el("capability"),
  demo: el("choice-demo"),
  open: el("choice-open"),
  fileInput: el("file-input"),
  loading: el("loading"),
  stage: el("loading-stage"),
  bar: el("loading-bar"),
  failure: el("loading-failure"),
  failureTitle: el("failure-title"),
  failureBody: el("failure-body"),
  failureAdvice: el("failure-advice"),
  failureBack: el("failure-back"),
};

// --- the loading takeover -------------------------------------------------

export function showLoading(stage) {
  dom.loading.hidden = false;
  dom.loading.classList.remove("failed");
  dom.stage.textContent = stage;
  dom.bar.style.width = "0%";
}

export function progress(stage, fraction) {
  dom.stage.textContent = stage;
  if (typeof fraction === "number") {
    dom.bar.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  }
}

export function hideLoading() {
  dom.loading.hidden = true;
}

/**
 * Every failure path lands here. A blank canvas is not an outcome: the title
 * says what broke, the body says why, and the advice says what to do about
 * it — which for the interesting failures is a real, specific instruction.
 */
export function showFailure(title, body, advice) {
  dom.loading.hidden = false;
  dom.loading.classList.add("failed");
  dom.stage.textContent = "";
  dom.failureTitle.textContent = title;
  dom.failureBody.textContent = body || "";
  dom.failureAdvice.textContent = advice || "";
  console.error(title, body, advice);
}

dom.failureBack.addEventListener("click", () => {
  hideLoading();
  dom.viewer.hidden = true;
  dom.header.style.display = "";
});

// --- capability check -----------------------------------------------------

let gpu = null;         // {adapter, limits} once probed
let gpuProbe = null;    // the in-flight probe, so we only run it once

async function probeGPU() {
  const adapter = await acquireAdapter();
  const limits = limitsSummary(adapter);
  gpu = { adapter, limits };
  return gpu;
}

function fieldSizeSentence(limits) {
  const cap = limits.maxTextureDimension3D;
  // The volume is ghost-padded by one texel per side before upload.
  const cells = cap - 2;
  const voxels = (cells / 1e3) ** 3;
  return `Fields up to <b>${cells}&thinsp;&times;&thinsp;${cells}` +
         `&thinsp;&times;&thinsp;${cells}</b> cells ` +
         `(${voxels.toFixed(1)} billion voxels) fit on one axis here.`;
}

async function renderCapability() {
  try {
    const { limits } = await (gpuProbe ??= probeGPU());
    const parts = [];
    const card = limits.description || limits.vendor || "your GPU";
    parts.push(`WebGPU is available on <b>${card}</b>. ` +
               fieldSizeSentence(limits));
    if (limits.clampedToSpecFloor) {
      parts.push(
        `<span class="detail">This is Chrome's fixed ceiling of ` +
        `${SPEC_FLOOR_TEXTURE_3D} rather than anything about your card — ` +
        `it reports the WebGPU minimum whatever the hardware can do. ` +
        `Firefox reports the real limit, which on a modern card is ` +
        `usually 16384. If a field of yours is refused here, that is the ` +
        `reason and Firefox is the fix.</span>`);
    }
    dom.capability.className = limits.clampedToSpecFloor ? "warn" : "good";
    dom.capability.innerHTML = parts.join(" ");
  } catch (err) {
    dom.capability.className = "bad";
    const detail = err instanceof WebGPUUnavailable ? err.detail : String(err);
    dom.capability.innerHTML =
      `<b>${err.message}</b><span class="detail">${detail}</span>`;
    dom.demo.disabled = true;
    dom.open.disabled = true;
  }
}

// --- picking a field ------------------------------------------------------

// Demo data is fetched from the cloudyview repository rather than shipped
// beside the page, so the site stays a thin static folder. Pinned to a tag:
// a deployed page must not change under a push to master.
export const DEMO_BASE_URL =
  "https://raw.githubusercontent.com/thomasddewitt/cloudyview/web-demo-v1/web/demo";

// Until the repo is public, a sibling demo/ folder wins if it exists. This is
// also how development works with no network.
export const DEMO_LOCAL_URL = "../demo";

async function resolveDemoBase() {
  try {
    const probe = await fetch(`${DEMO_LOCAL_URL}/meta.json`, { method: "HEAD" });
    if (probe.ok) return DEMO_LOCAL_URL;
  } catch { /* no local copy; fall through to the pinned remote */ }
  return DEMO_BASE_URL;
}

async function enterViewer(source) {
  dom.viewer.hidden = false;
  dom.header.style.display = "none";
  showLoading("Starting WebGPU…");
  try {
    const { adapter } = await (gpuProbe ??= probeGPU());
    const device = await acquireDevice(adapter);
    watchDevice(device, {
      onLost: (message) => showFailure(
        "The GPU device was lost.", message,
        "Reload the page to start over. If it keeps happening on the same " +
        "field, it is probably running out of video memory — try a coarser " +
        "level or close other GPU-heavy tabs."),
      onError: (message) => console.error("uncaptured:", message),
    });
    const { boot } = await import("./viewer.js");
    await boot({ device, adapter, source, progress, onReady: hideLoading,
                 onFailure: showFailure });
  } catch (err) {
    if (err instanceof WebGPUUnavailable) {
      showFailure(err.message, err.detail, "");
    } else {
      showFailure("Could not open this field.",
                  String(err && err.message || err), err?.advice || "");
    }
  }
}

dom.demo.addEventListener("click", async () => {
  const base = await resolveDemoBase();
  enterViewer({ kind: "demo", base });
});

dom.open.addEventListener("click", () => dom.fileInput.click());
dom.fileInput.addEventListener("change", () => {
  const file = dom.fileInput.files?.[0];
  if (file) enterViewer({ kind: "file", file });
});

// Drop anywhere on the landing page.
for (const type of ["dragenter", "dragover"]) {
  document.addEventListener(type, (e) => {
    if (!dom.viewer.hidden) return;
    e.preventDefault();
    dom.open.classList.add("dropping");
  });
}
for (const type of ["dragleave", "drop"]) {
  document.addEventListener(type, (e) => {
    if (type === "drop") e.preventDefault();
    dom.open.classList.remove("dropping");
  });
}
document.addEventListener("drop", (e) => {
  if (!dom.viewer.hidden) return;
  const file = e.dataTransfer?.files?.[0];
  if (file) enterViewer({ kind: "file", file });
});

renderCapability();

// A URL of the form ?demo goes straight in — useful for linking to the thing
// itself rather than to the page that offers it.
if (new URLSearchParams(location.search).has("demo")) {
  resolveDemoBase().then((base) => enterViewer({ kind: "demo", base }));
}

export { browserGuess };
