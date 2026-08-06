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
  rail: el("rail"),
  open: el("choice-open"),
  fileInput: el("file-input"),
  reel: [el("reel-a"), el("reel-b")],
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

// Backing out of a failure is a full exit, not a hidden div.
//
// The viewer behind this panel still owns a device, a swapchain, a volume of
// several gigabytes, and a set of listeners on `document` that keep all of it
// reachable. Hiding it and then opening another field stacked a second device
// on the same canvas — and the watcher for the first one was still live,
// pointed at whatever `viewer` happened to mean by then.
dom.failureBack.addEventListener("click", async () => {
  hideLoading();
  dom.viewer.hidden = true;
  dom.header.style.display = "";
  await endSession();
});

// --- capability check -----------------------------------------------------

let gpu = null;         // {adapter, limits} once probed
let gpuProbe = null;    // the in-flight probe, so we only run it once
let viewer = null;      // set once the viewer boots, for error reporting

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
    dom.open.disabled = true;
    for (const button of dom.rail.querySelectorAll(".demo")) {
      button.disabled = true;
    }
    capabilityFailed = true;
  }
}

// --- picking a field ------------------------------------------------------

// Demo fields are hosted rather than served from the cloudyview repository:
// the set runs to a few hundred megabytes and binaries in git are forever.
// They live beside the site on the same static host.
export const DEMOS_BASE_URL = "https://thomasddewitt.com/thought-cloud/soar-demos";

// A sibling demos/ folder wins if it exists — that is how a local bake is
// tried out, and how development works with no network.
export const DEMOS_LOCAL_URL = "../demos";

async function resolveDemosRoot() {
  try {
    const probe = await fetch(`${DEMOS_LOCAL_URL}/index.json`, { method: "HEAD" });
    if (probe.ok) return DEMOS_LOCAL_URL;
  } catch { /* no local bake; fall through to the hosted set */ }
  return DEMOS_BASE_URL;
}

/** Tear down whatever session is running, and wait for it. */
async function endSession() {
  const going = viewer;
  viewer = null;
  await going?.dispose();
}

/**
 * Open a field in a viewer of its own.
 *
 * Serialized against itself. Double-clicking Demo used to start two of these
 * — the network probe for the demo base is awaited before anything is hidden,
 * so both got that far — and each acquired its own GPUDevice and configured
 * the same canvas with it. One canvas cannot belong to two devices, and the
 * loser's device-lost watcher would then stop the winner's viewer.
 */
let entering = null;
function enterViewer(source) {
  entering = (entering ?? Promise.resolve())
    .catch(() => {})
    .then(() => enterViewerOnce(source));
  return entering;
}

async function enterViewerOnce(source) {
  await endSession();
  dom.viewer.hidden = false;
  dom.header.style.display = "none";
  showLoading("Starting WebGPU…");
  let device = null;
  try {
    const { adapter } = await (gpuProbe ??= probeGPU());
    device = await acquireDevice(adapter);
    // `session` rather than the module-level `viewer`: this closure must act
    // on the viewer that owns THIS device, whatever has happened since. An
    // old device dying is not a reason to stop a new session.
    const session = { viewer: null };
    watchDevice(device, {
      onLost: (message) => {
        // Stop FIRST. Every frame after this would call into a queue whose
        // device no longer exists, and Firefox does not treat that as an
        // error — it crashes the process ("Queue[Id] does not exist").
        if (session.viewer) session.viewer.stop = true;
        if (session.viewer !== viewer) return;   // a session already replaced
        showFailure(
          "The GPU device was lost.", message,
          "Reload the page to start over. If it keeps happening on the same " +
          "field, it is probably running out of video memory — try a coarser " +
          "level or close other GPU-heavy tabs.");
      },
      // WebGPU reports validation asynchronously. gpu.js calls both of these
      // fatal, and it is right: an uncaptured validation error means the
      // command stream is not what this code thinks it is, so the picture is
      // undefined and every later frame compounds it. Stop and say so rather
      // than carry on submitting into a state nobody can reason about.
      onError: (message) => {
        console.error("uncaptured:", message);
        if (session.viewer) session.viewer.stop = true;
        if (session.viewer !== viewer) return;
        showFailure(
          "The GPU rejected a command.", message,
          "This is a bug in cloudyview rather than anything you did. The " +
          "picture after this point would be undefined, so rendering has " +
          "stopped. Reload the page to start over.");
      },
    });
    const { boot } = await import("./viewer.js");
    viewer = await boot({
      device, adapter, source, progress,
      onReady: hideLoading, onFailure: showFailure,
      // The loader asks questions (which group, what units) and those are
      // menu panels, which live under this overlay.
      setLoadingVisible: (visible) => { dom.loading.hidden = !visible; },
      register: (v) => { viewer = v; session.viewer = v; },
    });
    session.viewer = viewer;
  } catch (err) {
    // boot() disposes the viewer it could not finish, which takes the device
    // with it; a failure before boot leaves the device here to release.
    if (viewer) await endSession();
    else device?.destroy();
    if (err instanceof WebGPUUnavailable) {
      showFailure(err.message, err.detail, "");
    } else {
      showFailure("Could not open this field.",
                  String(err && err.message || err), err?.advice || "");
    }
  }
}

// --- the demo rail --------------------------------------------------------
//
// Each demo carries one still rendered from its own field by
// tools/prebake_demos.py. Hovering a demo cross-fades it behind the page, so
// the choice is made by looking at a sky rather than by reading a label.

let capabilityFailed = false;
let reelActive = 1;      // starts at 1 so the first show() lands on reel-a
let reelShown = null;
let previewHold = null;

/** Cross-fade the backdrop to a demo's still, decoding before it is shown. */
async function showReel(demo, root) {
  if (!demo?.still || reelShown === demo.id) return;
  const next = 1 - reelActive;
  const image = dom.reel[next];
  const src = `${root}/${demo.base}/${demo.still}`;
  if (image.dataset.src !== src) {
    image.dataset.src = src;
    image.src = src;
    // Wait for `load`, not `decode()`. decode() is the nicer primitive —
    // it resolves when the bitmap is ready to paint, so the fade never
    // reveals a half-drawn image — but it does not settle at all in a
    // backgrounded tab, which left the backdrop black with nothing logged.
    // `load` fires either way, and an image that errors resolves too so a
    // missing still costs one blank fade rather than a stuck reel.
    if (!image.complete) {
      await new Promise((resolve) => {
        image.addEventListener("load", resolve, { once: true });
        image.addEventListener("error", resolve, { once: true });
      });
    }
  }
  image.classList.add("on");
  dom.reel[reelActive].classList.remove("on");
  reelActive = next;
  reelShown = demo.id;
}

function formatBytes(n) {
  if (!n) return "";
  return n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`;
}

async function buildRail() {
  const root = await resolveDemosRoot();
  let index;
  try {
    const response = await fetch(`${root}/index.json`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    index = await response.json();
  } catch (err) {
    // No silent empty rail: if the list cannot be had, say so, because the
    // page without it offers no way in but your own file.
    dom.rail.innerHTML =
      `<p class="rail-loading">The demo list could not be loaded (${err.message}). ` +
      `Opening your own file still works.</p>`;
    return;
  }

  dom.rail.replaceChildren();
  for (const demo of index.demos) {
    const button = document.createElement("button");
    button.className = "demo";
    button.type = "button";
    button.disabled = capabilityFailed;
    button.innerHTML =
      `<span class="name"></span><span class="field"></span>` +
      `<span class="size"></span>`;
    button.querySelector(".name").textContent = demo.title;
    button.querySelector(".field").textContent = demo.field ?? "";
    button.querySelector(".size").textContent = formatBytes(demo.bytes);
    button.title = demo.description ?? "";

    const enter = () => {
      clearTimeout(previewHold);
      document.body.classList.add("previewing");
      for (const other of dom.rail.children) other.classList.toggle("live", other === button);
      showReel(demo, root);
    };
    const leave = () => {
      clearTimeout(previewHold);
      previewHold = setTimeout(() => {
        document.body.classList.remove("previewing");
        for (const other of dom.rail.children) other.classList.remove("live");
      }, 350);
    };
    button.addEventListener("mouseenter", enter);
    button.addEventListener("focus", enter);
    button.addEventListener("mouseleave", leave);
    button.addEventListener("blur", leave);
    button.addEventListener("click", () =>
      enterViewer({ kind: "demo", base: `${root}/${demo.base}` }));

    dom.rail.appendChild(button);
  }

  // Something is always on screen, so the page is a sky rather than a form.
  if (index.demos.length) await showReel(index.demos[0], root);

  // Warm the rest so the first hover cross-fades instead of flashing empty.
  for (const demo of index.demos.slice(1)) {
    if (demo.still) new Image().src = `${root}/${demo.base}/${demo.still}`;
  }
}

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
buildRail();

// ?demo=<id> goes straight in — useful for linking to a particular field
// rather than to the page that offers them. Bare ?demo takes the first.
const wanted = new URLSearchParams(location.search).get("demo");
if (wanted !== null) {
  resolveDemosRoot().then(async (root) => {
    const index = await (await fetch(`${root}/index.json`)).json();
    const demo = index.demos.find((d) => d.id === wanted) ?? index.demos[0];
    if (demo) enterViewer({ kind: "demo", base: `${root}/${demo.base}` });
  });
}

export { browserGuess };
