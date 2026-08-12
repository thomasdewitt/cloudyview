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
  WebGPUUnavailable, browserGuess,
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
    dom.capability.className = limits.clampedToSpecFloor ? "warn" : "good";
    dom.capability.innerHTML = parts.join(" ");
  } catch (err) {
    dom.capability.className = "bad";
    const detail = err instanceof WebGPUUnavailable ? err.detail : String(err);
    dom.capability.innerHTML =
      `<b>${err.message}</b><span class="detail">${detail}</span>`;
    // Deliberately NOT `disabled`. A dead button that swallows the click
    // tells you nothing, and the readout explaining why is at the far bottom
    // of the page where a hand on the demo rail is not looking. Leave them
    // live and answer on click, in front of the thing that was clicked.
    capabilityError = { message: err.message, detail };
    capabilityFailed = true;
    dom.rail.classList.add("unavailable");
    dom.open.classList.add("unavailable");
  }
}

/**
 * The click answer when there is no GPU to render with. Same words as the
 * bottom readout, promoted to the full failure panel so the reason arrives
 * where the click did.
 */
function reportNoGPU() {
  showFailure(
    capabilityError.message,
    capabilityError.detail,
    "Soar needs WebGPU with a working GPU adapter. Chrome or Edge 113+, " +
    "Firefox 141+, or Safari 26 on a machine with a GPU it is allowed to " +
    "use. Remote desktops and virtual machines usually cannot offer one.");
}

// --- picking a field ------------------------------------------------------

// The demo set sits inside the app, because a thought-cloud folder is one
// self-contained artifact: `soar/` is copied to the site whole, and there is
// no second location to keep in step and nothing to configure per host. The
// bytes are still derived and still gitignored — see .gitignore and
// tools/prebake_demos.py — they are simply staged with everything else.
//
// Relative, and deliberately not absolute: the same folder has to work from
// file://, from a local http.server, and from the site, without a build-time
// substitution or a hostname compiled into the app.
export const DEMOS_URL = "./demos";

/**
 * Accept either index shape and return a list of groups.
 *
 * v2 is grouped. v1 was one flat list, and a stale bake on disk or a stale
 * copy still sitting on the host must not blank the page, so it normalises
 * into a single untitled group. Anything else is not a demo index at all —
 * throw, and let the caller say the list could not be loaded, because an
 * empty rail looks like "there are no demos" rather than like a fault.
 */
function normalizeIndex(index) {
  const schema = index?.schema;
  if (schema === "soar.demos.v2") {
    if (!Array.isArray(index.groups)) throw new Error("v2 index has no groups");
    return index.groups.map((group) => ({ ...group, demos: group.demos ?? [] }));
  }
  if (schema === "soar.demos.v1") {
    if (!Array.isArray(index.demos)) throw new Error("v1 index has no demos");
    return [{ id: "demos", title: "", demos: index.demos }];
  }
  throw new Error(`unrecognised index schema ${JSON.stringify(schema ?? null)}`);
}

async function fetchIndex(root) {
  const response = await fetch(`${root}/index.json`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return normalizeIndex(await response.json());   // throws on an unknown shape
}

/**
 * The demo set, read once and shared.
 *
 * There is one root now, so there is nothing to choose between and no probe:
 * a failure here is a real failure and the caller says so on the page. This
 * used to pick between a local bake and a hosted copy, which meant a HEAD
 * whose 200 proved nothing — plenty of hosts answer 200 with an HTML error
 * page for any path they do not recognise. Folding the demos into the app
 * deleted the question rather than answering it better.
 */
let demoIndex = null;
function resolveDemos() {
  return (demoIndex ??= (async () => (
    { root: DEMOS_URL, groups: await fetchIndex(DEMOS_URL) }
  ))());
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
    if (err?.cancelled) {
      // Back on the group/units question during the very first load. There
      // is no viewer to return to, so this is a quiet walk back to the start
      // page — a deliberate choice, not a failure panel (bug 11).
      hideLoading();
      dom.viewer.hidden = true;
      dom.header.style.display = "";
      return;
    }
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
let capabilityError = null;
let reelActive = 1;      // starts at 1 so the first show() lands on reel-a
let reelShown = null;
let reelGeneration = 0;
let previewHold = null;

/** Cross-fade the backdrop to a demo's still, decoding before it is shown. */
async function showReel(demo, root) {
  if (!demo?.still || reelShown === demo.id) return;
  // Claim the reel before the first await. Two calls in flight used to pick
  // the same buffer and the loser hid the element the winner had just shown,
  // leaving the backdrop black (docs/soar-bugs.md entry 8). `reelShown` set
  // here dedupes repeat hovers on one card; the generation token below drops
  // every call but the newest — the last hover is the only one whose result
  // anyone wants.
  const generation = ++reelGeneration;
  reelShown = demo.id;
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
  if (generation !== reelGeneration) return;
  image.classList.add("on");
  dom.reel[reelActive].classList.remove("on");
  reelActive = next;
}

/** One card. Hovering it previews its field behind the whole page. */
function demoCard(demo, root) {
  const button = document.createElement("button");
  button.className = "demo";
  button.type = "button";
  // The description rides inside the card and is revealed by growing it,
  // rather than living in a `title` tooltip. A tooltip is the OS's typeface
  // at the OS's timing in a box this page has no say over, and it lands next
  // to the cursor rather than on the thing it describes.
  button.innerHTML =
    `<span class="name"></span><span class="field"></span>` +
    `<span class="desc"><span class="desc-inner"></span></span>`;
  button.querySelector(".name").textContent = demo.title;
  button.querySelector(".field").textContent = demo.field ?? "";
  button.querySelector(".desc-inner").textContent = demo.description ?? "";

  // `live` is cleared across every card on the page, not just this card's
  // group: the preview is one backdrop shared by all of them.
  const enter = () => {
    clearTimeout(previewHold);
    document.body.classList.add("previewing");
    for (const other of dom.rail.querySelectorAll(".demo")) {
      other.classList.toggle("live", other === button);
    }
    showReel(demo, root);
  };
  const leave = () => {
    clearTimeout(previewHold);
    previewHold = setTimeout(() => {
      document.body.classList.remove("previewing");
      for (const other of dom.rail.querySelectorAll(".demo")) {
        other.classList.remove("live");
      }
    }, 350);
  };
  button.addEventListener("mouseenter", enter);
  button.addEventListener("focus", enter);
  button.addEventListener("mouseleave", leave);
  button.addEventListener("blur", leave);
  button.addEventListener("click", () =>
    capabilityFailed
      ? reportNoGPU()
      : enterViewer({ kind: "demo", base: `${root}/${demo.base}` }));
  return button;
}

/**
 * One labelled section of the rail.
 *
 * The heading is the page's existing rail-head idiom — small caps and a
 * hairline running to the edge — so a second group reads as more of the same
 * page rather than as a new component. A group with no demos yet says so
 * with a strip of text; a disabled button would look like something broken
 * rather than something not written yet.
 */
function renderGroup(group, root) {
  const section = document.createElement("section");
  section.className = "rail-group";

  // A v1 index normalises to one untitled group, which wants no heading at
  // all — the page it came from had exactly one rail and said "Demos" above
  // it in the markup.
  if (group.title) {
    const head = document.createElement("div");
    head.className = "rail-head";
    const label = document.createElement("span");
    label.textContent = group.title;
    const rule = document.createElement("span");
    rule.className = "rule";
    head.append(label, rule);
    // A one-line description of the group, when there is one to give, would
    // sit here between the heading and the cards.
    section.appendChild(head);
  }

  const rail = document.createElement("div");
  rail.className = "rail";
  if (group.status === "coming-soon" && !group.demos.length) {
    const soon = document.createElement("p");
    soon.className = "soon";
    soon.textContent = "Coming soon";
    rail.appendChild(soon);
  } else {
    for (const demo of group.demos) rail.appendChild(demoCard(demo, root));
  }
  section.appendChild(rail);
  return section;
}

async function buildRail() {
  let root, groups;
  try {
    ({ root, groups } = await resolveDemos());
  } catch (err) {
    // No silent empty rail: if the list cannot be had, say so, because the
    // page without it offers no way in but your own file.
    const note = document.createElement("p");
    note.className = "rail-loading";
    note.textContent = `The demo list could not be loaded (${err.message}). ` +
                       `Opening your own file still works.`;
    dom.rail.replaceChildren(note);
    return;
  }

  dom.rail.replaceChildren(...groups.map((group) => renderGroup(group, root)));

  // The reel is one backdrop for the whole rail, so it warms across groups.
  const all = groups.flatMap((group) => group.demos);
  // Something is always on screen, so the page is a sky rather than a form.
  if (all.length) await showReel(all[0], root);

  // Warm the rest so the first hover cross-fades instead of flashing empty.
  for (const demo of all.slice(1)) {
    if (demo.still) new Image().src = `${root}/${demo.base}/${demo.still}`;
  }
}

dom.open.addEventListener("click", () =>
  capabilityFailed ? reportNoGPU() : dom.fileInput.click());
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
// The id is looked up across every group: which group a demo sits in is a
// presentation choice on the landing page and links must not depend on it.
const wanted = new URLSearchParams(location.search).get("demo");
if (wanted !== null) {
  resolveDemos().then(({ root, groups }) => {
    const all = groups.flatMap((group) => group.demos);
    const demo = all.find((d) => d.id === wanted) ?? all[0];
    if (demo) enterViewer({ kind: "demo", base: `${root}/${demo.base}` });
    // buildRail shares this promise and reports the failure on the page, so
    // there is nothing to say here beyond not raising it twice.
  }).catch(() => {});
}

export { browserGuess };
