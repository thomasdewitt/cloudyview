// Entry point: decide what this browser can do, take a field from the user,
// and hand it to the viewer.
//
// Landing and viewer are one page rather than two. A File chosen here is a
// live handle; navigating would throw it away and there is no way to pass an
// unuploaded multi-gigabyte file across a page load. So the landing hides and
// the viewer takes over.

"use strict";

import {
  acquireAdapter, acquireDevice, limitsSummary, watchDevice, volumeFits,
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
  modeToggle: el("mode-toggle"),
  flyLess: el("fly-less"),
  flyMore: el("fly-more"),
  flyMoreWarn: el("fly-more-warn"),
  reel: [el("reel-a"), el("reel-b")],
  cyberBack: el("reel-cyberpunk"),
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
  // One texel per cell, so the limit IS the cell count.
  const cells = cap;
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
    // All three .own buttons, not just the file picker: they are one control
    // in three instances and there is no GPU for any of them.
    for (const button of [dom.open, dom.flyLess, dom.flyMore]) {
      button.classList.add("unavailable");
    }
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
    await (gpuProbe ??= probeGPU());   // the capability gate, run once
    // A FRESH adapter for every session: WebGPU adapters are single-use —
    // requestDevice on one that already made a device throws ("adapter is
    // consumed"). Reusing the probe's adapter across sessions meant the
    // second session after a Back — e.g. retrying a smaller demo after a
    // failed load — could never start until a reload.
    device = await acquireDevice(await acquireAdapter());
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
          "field, it is probably running out of video memory — try a field " +
          "with a smaller grid size, or close other GPU-heavy tabs.");
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
      device, source, progress,
      onReady: hideLoading, onFailure: showFailure,
      // The loader asks questions (which group, what units) and those are
      // menu panels, which live under this overlay.
      setLoadingVisible: (visible) => { dom.loading.hidden = !visible; },
      // window.soar is a read-only debug handle — the only way to inspect
      // the probe/governor state machines from a console or a headless
      // driver, since everything real is module-scoped.
      register: (v) => { viewer = v; session.viewer = v; window.soar = v; },
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

// The cyberpunk backdrop: one offline render of the view a cyberpunk flight
// opens on (constants.js CITY_START_CAMERA), over the C1=0.07 city tile and
// under the mode's moon, at the max preset. It ships with the city tile it is
// a picture of rather than with a demo, because it is a picture of the MODE:
// no demo owns it, and the field under it is the same field the pair loads.
const CYBERPUNK_BACKDROP = "city/landing.webp";

/**
 * The mode's own backdrop, in place of the demo reel.
 *
 * The reel is not stopped or rewound — CSS covers it while the body carries
 * mode-cyberpunk, and uncovers it on the way out, so leaving cyberpunk shows
 * whatever the hover had left on top. All this does is decline to download a
 * quarter of a megabyte until somebody asks for the mode.
 */
function showCyberpunkBackdrop(on) {
  if (!on || dom.cyberBack.src) return;
  dom.cyberBack.src = CYBERPUNK_BACKDROP;
}

/**
 * The numbers a case shows, from the volume block of its own meta.json.
 *
 * Horizontal spacing rather than all three: the vertical is usually
 * different, and the grid line right next to it already says the shape. If
 * x and y disagree the pair is printed, because picking one would be a lie.
 */
function caseNumbers(meta) {
  const v = meta?.volume;
  const n = v?.shape_xyz, lo = v?.bmin, hi = v?.bmax;
  if (!Array.isArray(n) || !Array.isArray(lo) || !Array.isArray(hi)) return null;
  const span = [0, 1, 2].map((i) => hi[i] - lo[i]);
  const dx = span[0] / n[0], dy = span[1] / n[1];
  const metres = (m) => (m < 10 ? Math.round(m * 10) / 10 : Math.round(m));
  const across = span[0] / 1000;
  return {
    grid: `${n[0]} × ${n[1]} × ${n[2]}`,
    res: Math.abs(dx - dy) / dx > 0.01
      ? `${metres(dx)} × ${metres(dy)} m` : `${metres(dx)} m`,
    domain: `${across < 10 ? Math.round(across * 10) / 10 : Math.round(across)} km`,
  };
}

/**
 * One case. Hovering it previews its field behind the whole page.
 *
 * `numbers` may be null — see buildRail for why that is rendered as a card
 * without a data line rather than as a card that is missing.
 */
function demoCard(demo, root, numbers) {
  const button = document.createElement("button");
  button.className = "demo";
  button.type = "button";
  // The regime and the numbers are always visible rather than revealed on
  // hover: they are the reason the list is a list of these particular cases,
  // and comparing them across rows is the point — DYCOMS resolves 5 m over
  // 3.2 km where TWP-ICE resolves 100 m over 102 km. A number you have to
  // hover to see cannot be compared with the one above it.
  //
  // Anything past the numbers is optional and hidden until the row is hovered
  // or focused, when the card grows to hold it. Most cases have nothing here
  // and the block is not built at all — an empty expansion that opens onto
  // nothing is worse than a row that simply does not open.
  //
  // This used to be `button.title`, i.e. the browser's own tooltip: a pale
  // box in the OS's font that lands wherever the pointer is, after whatever
  // delay the OS feels like, on top of the still the hover exists to reveal.
  const more = [
    demo.warning ? `<span class="warn"></span>` : "",
    demo.description ? `<span class="note"></span>` : "",
  ].join("");
  button.innerHTML =
    `<span class="top"><span class="name"></span><span class="res"></span></span>` +
    `<span class="field"></span>` +
    (numbers
      ? `<span class="data"><span class="grid"></span>` +
        `<span><span class="domain"></span> <span class="u">across</span></span></span>`
      : "") +
    (more ? `<span class="more"><span class="more-in">${more}</span></span>` : "");
  button.querySelector(".name").textContent = demo.title;
  button.querySelector(".field").textContent = demo.field ?? "";
  if (numbers) {
    button.querySelector(".res").textContent = numbers.res;
    button.querySelector(".grid").textContent = numbers.grid;
    button.querySelector(".domain").textContent = numbers.domain;
  }
  // textContent, not innerHTML: these two strings come out of a baked JSON
  // file rather than out of this source, and the rest of the card is built
  // the same way.
  if (demo.warning) button.querySelector(".warn").textContent = demo.warning;
  if (demo.description) button.querySelector(".note").textContent = demo.description;

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
      : enterViewer(demoSource(root, demo)));
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
function renderGroup(group, root, numbersById) {
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

  if (group.status === "coming-soon" && !group.demos.length) {
    const soon = document.createElement("p");
    soon.className = "soon";
    soon.textContent = "Coming soon";
    section.appendChild(soon);
  } else {
    for (const demo of group.demos) {
      section.appendChild(demoCard(demo, root, numbersById.get(demo.id)));
    }
  }
  return section;
}

/**
 * The per-case numbers, fetched alongside the index.
 *
 * They live in each demo's meta.json rather than in the group index, which
 * carries only id/title/field/description/warning/base/bytes/still. Reading
 * the metas is deliberate over adding the fields to the index: the index is
 * baked output, and hoisting them into it would mean re-baking the demo set
 * to see them. These files are about a kilobyte each, the app fetches them
 * on load anyway, and asking early warms that fetch.
 *
 * A meta that will not load resolves to null rather than rejecting. That is
 * not a silent fallback dressed up: a case whose meta.json is missing cannot
 * be opened either, and the loader has real failure copy for it, so the row
 * stays clickable and answers on click — the same choice the no-GPU rail
 * makes. What must not happen is one unreadable file blanking the whole
 * list. The console says which one.
 */
const metaById = new Map();

async function fetchNumbers(demos, root) {
  const pairs = await Promise.all(demos.map(async (demo) => {
    const url = `${root}/${demo.base}/meta.json`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const meta = await res.json();
      // Kept whole, not just the derived numbers: the fly-pair gate below
      // needs volume.shape_xyz, and this is already the one fetch of it.
      metaById.set(demo.id, meta);
      return [demo.id, caseNumbers(meta)];
    } catch (err) {
      console.warn(`soar: no numbers for "${demo.id}" — ${url}: ${err.message}`);
      return [demo.id, null];
    }
  }));
  return new Map(pairs);
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

  // The reel is one backdrop for the whole rail, so it warms across groups.
  const all = groups.flatMap((group) => group.demos);
  const numbersById = await fetchNumbers(all, root);

  dom.rail.replaceChildren(
    ...groups.map((group) => renderGroup(group, root, numbersById)));

  // Something is always on screen, so the page is a sky rather than a form.
  if (all.length) await showReel(all[0], root);

  // Warm the rest so the first hover cross-fades instead of flashing empty.
  for (const demo of all.slice(1)) {
    if (demo.still) new Image().src = `${root}/${demo.base}/${demo.still}`;
  }
}

// --- modes ----------------------------------------------------------------
//
// One page, three audiences. `basic` is the default and the one most visitors
// get: no rail, no file picker, two buttons and a sky. `research` is the page
// as it was, plus those two buttons. `cyberpunk` is basic's page — the same
// two buttons, the same absent rail — flying the same two demos at night over
// the multifractal city.
//
// It is chosen HERE and nowhere else. The pause menu offers basic and
// research to each other because those two differ only in which controls a
// menu carries, and a menu can change that mid-flight; cyberpunk is a
// different scene compiled into the shader and standing on a different
// surface tile, which is a load, not a preference. So the way out of it is
// the way into it — the start page (see ui.js MODES.cyberpunk.landingOnly).

// The labels say "mode" out loud. A row of three bare adjectives over a
// button that says "Fly less detailed demo" reads as a property of the demo;
// the word is what makes the control name what it switches.
const MODES = [
  { id: "research", label: "research mode" },
  { id: "basic", label: "basic mode" },
  { id: "cyberpunk", label: "cyberpunk mode" },
];

const MODE_KEY = "soar.mode";
const DEFAULT_MODE = "basic";

/** The stored mode, if it is one we still offer. */
function storedMode() {
  let saved = null;
  // Private windows and file:// in some browsers throw on access rather than
  // returning null, and a page that will not load because of a preference is
  // a worse page than one that forgets the preference.
  try { saved = localStorage.getItem(MODE_KEY); } catch { /* no storage */ }
  return MODES.find((m) => m.id === saved)?.id ?? DEFAULT_MODE;
}

let mode = storedMode();

/**
 * Put the sliding block over the selected segment.
 *
 * Measured rather than declared: the three labels are different lengths, so
 * the geometry only exists once the segments have been laid out in the real
 * face. transform rather than `left` because it is the property that moves
 * without asking for layout on every frame of the slide.
 */
function placeThumb() {
  const thumb = dom.modeToggle.querySelector(".thumb");
  const current = dom.modeToggle.querySelector('.mode[aria-checked="true"]');
  if (!thumb || !current) return;
  thumb.style.width = `${current.offsetWidth}px`;
  thumb.style.transform = `translateX(${current.offsetLeft}px)`;
}

/**
 * Everything mode-dependent is one body class and CSS; the pair, which two of
 * the three modes want, and the thumb, whose geometry CSS cannot know, are
 * the two things set here.
 */
function applyMode(next, { save = true } = {}) {
  // Re-clicking the segment already lit changes nothing, and must not arm the
  // switch transition either: `switched` is what makes the rail glide, so a
  // no-op click was buying a reveal animation for controls that never moved.
  const changed = next !== mode;
  mode = next;
  // Before the mode class, not after. It marks this as a switch rather than a
  // page load — which is what tells the newly revealed controls to use the
  // short reveal instead of the intro, and what arms the rail's glide at all
  // (see body.switched in style.css). Set it after the layout change and the
  // very first switch of a session is asking the engine to transition on a
  // rule that was not there when the change was made.
  if (save && changed) document.body.classList.add("switched");
  for (const m of MODES) document.body.classList.toggle(`mode-${m.id}`, m.id === next);
  // Every mode wants the pair; it starts hidden only so the stack does not
  // reflow between first paint and the stored mode landing. WHICH two cases
  // it loads is the mode's — a no-op until the index lands, after which the
  // switch is a lookup.
  dom.flyLess.hidden = false;
  dom.flyMore.hidden = false;
  refreshFlyPair();
  showCyberpunkBackdrop(next === "cyberpunk");
  for (const button of dom.modeToggle.children) {
    if (button.dataset.mode === undefined) continue;   // the thumb
    button.setAttribute("aria-checked", String(button.dataset.mode === next));
  }
  placeThumb();
  if (!save) return;
  try { localStorage.setItem(MODE_KEY, next); } catch { /* no storage */ }
}

function buildModeToggle() {
  const thumb = document.createElement("span");
  thumb.className = "thumb";
  dom.modeToggle.replaceChildren(thumb, ...MODES.map((m) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mode";
    button.dataset.mode = m.id;
    button.setAttribute("role", "radio");
    const label = document.createElement("span");
    label.textContent = m.label;
    button.appendChild(label);
    button.addEventListener("click", () => applyMode(m.id));
    return button;
  }));
  applyMode(mode, { save: false });   // reflect the stored value, don't re-store

  // The thumb is placed before the webfont arrives, so the segments it was
  // measured against are the fallback face's width. Re-place it when the real
  // one lands and on any resize; `.ready` is added after the first placement
  // so none of that first correction is animated — a block sliding in from
  // the left edge on load would look like the toggle was being demonstrated.
  const settle = () => {
    placeThumb();
    dom.modeToggle.classList.add("ready");
  };
  requestAnimationFrame(settle);
  document.fonts?.ready.then(placeThumb).catch(() => {});
  addEventListener("resize", placeThumb);
}

// The entry animations belong to the first paint only. Left on the elements
// they re-run, with their hold, every time a mode switch reveals one — see
// body.intro in style.css. 2.7 s clears the longest (1.6 s at 1.1 s).
document.body.classList.add("intro");
setTimeout(() => document.body.classList.remove("intro"), 2700);

// --- the two-button way in ------------------------------------------------

// The coarse/full twin. Ids from demos/index.json rather than two more copies
// of the fields: a pair IS one case at its two bakes, and the rail shows the
// same two rows in research mode.
//
// Cyberpunk flies its own pair (Thomas, 2026-08-22). The daylight pair is the
// 51.2 km desert domain, which under a city is 51 km of streets to cross at
// 20 m/s; marine congestus is a fifth of that box and its cloud base sits
// where a skyline can reach it. Two ids, not two bakes: the same volumes the
// rail lists, stood on the city by demoSource.
const FLY_PAIRS = {
  research: { less: "desert-coarse", more: "desert" },
  basic: { less: "desert-coarse", more: "desert" },
  cyberpunk: { less: "marine-congestus-coarse", more: "marine-congestus" },
};

/** The two ids this mode's buttons load. */
function flyPair() {
  const pair = FLY_PAIRS[mode];
  if (!pair) throw new Error(`no fly pair for mode '${mode}'.`);
  return pair;
}

/**
 * What flying this demo means, in the mode the page is in.
 *
 * Read at click time rather than at wire time: the buttons are wired once,
 * when the index resolves, and the mode can be switched any number of times
 * after that without the pair being rebuilt.
 */
function demoSource(root, demo) {
  return {
    kind: "demo",
    base: `${root}/${demo.base}`,
    ...(mode === "cyberpunk" ? { surfaceKind: "city" } : {}),
  };
}

/** An id the manifest does not have is an error on the button, not a swap. */
function flyMissing(button, id) {
  button.classList.add("missing");
  button.disabled = true;
  button.querySelector(".warn").textContent =
    `The demo "${id}" is not in the demo list.`;
  console.error(`soar: fly-pair case "${id}" missing from the demo index`);
}

/**
 * Wire the pair, and decide whether the full-resolution one can be opened.
 *
 * Deliberately hung off the work the rail already does — resolveDemos() is
 * the shared index promise and metaById is filled by the meta fetch the rail
 * makes anyway — so this adds no request and nothing blocks on it. The
 * buttons are live from the moment the index resolves; the limits check only
 * ever takes one away, and it lands whenever the adapter probe does.
 */
// The resolved index, once, so a mode switch re-reads the pair without
// re-fetching anything. Null until it lands (or for good, if it fails).
let flyIndex = null;
let railSettled = null;

async function wireFlyPair(railTask) {
  railSettled = railTask.catch(() => {});
  let root, groups;
  try {
    ({ root, groups } = await resolveDemos());
  } catch (err) {
    for (const button of [dom.flyLess, dom.flyMore]) {
      button.classList.add("missing");
      button.disabled = true;
    }
    dom.flyMoreWarn.textContent =
      `The demo list could not be loaded (${err.message}).`;
    el("fly-less-warn").textContent =
      `The demo list could not be loaded (${err.message}).`;
    return;
  }
  flyIndex = { root, all: groups.flatMap((group) => group.demos) };

  // Wired once and for both modes: the handler resolves the id at CLICK time,
  // the same reading demoSource does of the surface. A listener per mode
  // switch would stack, and removing them again is a second thing to keep
  // right about a button whose behaviour is one lookup.
  for (const which of ["less", "more"]) {
    const button = which === "less" ? dom.flyLess : dom.flyMore;
    button.addEventListener("click", () => {
      if (capabilityFailed) return reportNoGPU();
      const demo = flyDemo(which);
      // Only reachable if the index lost a case between refresh and click;
      // refreshFlyPair has already disabled the button for a missing id.
      if (!demo) return;
      return enterViewer(demoSource(flyIndex.root, demo));
    });
  }
  await refreshFlyPair();
}

/** This mode's demo for one of the two buttons, or undefined. */
function flyDemo(which) {
  return flyIndex?.all.find((d) => d.id === flyPair()[which]);
}

/**
 * Put this mode's pair on the two buttons: the ids, the caution, and whether
 * the full-resolution one fits on this GPU.
 *
 * Re-run on every mode switch, because all three of those are per-case and
 * cyberpunk's case is not basic's. It only ever reads state that is already
 * in hand — the index, the metas the rail fetched, the adapter probe — so a
 * switch costs no request.
 */
async function refreshFlyPair() {
  if (!flyIndex) return;
  for (const which of ["less", "more"]) {
    const button = which === "less" ? dom.flyLess : dom.flyMore;
    const demo = flyDemo(which);
    if (!demo) {
      flyMissing(button, flyPair()[which]);
      continue;
    }
    button.classList.remove("missing");
    button.disabled = false;
    // The caution the rail card carries, in the same words and from the same
    // field of the same file — not a second sentence about the same fact.
    // Cleared when this mode's case has none, or the previous mode's warning
    // would sit under the new one's label.
    button.querySelector(".warn").textContent = demo.warning ?? "";
  }
  const more = flyDemo("more");
  if (!more) return;

  // Now the harder claim: not "this is heavy" but "this will not fit". Both
  // halves can be late (the meta comes with the rail, the limits with the
  // adapter probe) and neither is on the path to a first paint.
  await railSettled;
  const shape = metaById.get(more.id)?.volume?.shape_xyz;
  // No meta means no claim. fetchNumbers already logged it, and a case whose
  // meta is unreadable fails loudly in the loader — the same choice the rail
  // makes for a row with no numbers.
  if (!Array.isArray(shape) || shape.length !== 3) return;
  let limits;
  try {
    ({ limits } = await (gpuProbe ??= probeGPU()));
  } catch {
    return;   // renderCapability owns this failure; capabilityFailed is set
  }
  const fits = volumeFits(limits, shape);
  // A mode switch during the awaits above: this verdict is about a case that
  // is no longer on the button, and writing it there would caution about the
  // wrong field.
  if (flyDemo("more")?.id !== more.id) return;
  if (fits.ok) return;
  dom.flyMore.disabled = true;
  dom.flyMoreWarn.textContent = `${fits.message} ${fits.advice}`;
}

dom.open.addEventListener("click", () =>
  capabilityFailed ? reportNoGPU() : dom.fileInput.click());
dom.fileInput.addEventListener("change", () => {
  const file = dom.fileInput.files?.[0];
  // Reset before loading: the input is persistent, and browsers fire no
  // "change" when the selection is unchanged — without this, re-picking the
  // same file after a Back from the group/units question (bug 11) or a
  // failed load silently did nothing.
  dom.fileInput.value = "";
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

buildModeToggle();
renderCapability();
wireFlyPair(buildRail());

// ?demo=<id> goes straight in — useful for linking to a particular field
// rather than to the page that offers them. Bare ?demo takes the first.
// The id is looked up across every group: which group a demo sits in is a
// presentation choice on the landing page and links must not depend on it.
const wanted = new URLSearchParams(location.search).get("demo");
if (wanted !== null) {
  resolveDemos().then(({ root, groups }) => {
    const all = groups.flatMap((group) => group.demos);
    const demo = all.find((d) => d.id === wanted) ?? all[0];
    // Through demoSource like every other way in, so the stored mode decides
    // the surface here too — a raw source flew the daytime ocean while the
    // in-flight UI said Cyberpunk.
    if (demo) enterViewer(demoSource(root, demo));
    // buildRail shares this promise and reports the failure on the page, so
    // there is nothing to say here beyond not raising it twice.
  }).catch(() => {});
}

export { browserGuess };
