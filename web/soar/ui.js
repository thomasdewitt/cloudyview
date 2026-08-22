// The overlay: menu, panels, readouts, prompts.
//
// Menus are click-only. Keys are reserved for things you do WHILE flying —
// where reaching for a menu would mean losing the shot — so the desktop's key
// state machine does not come across. Esc, Tab, F, F3, B, M, R and P are
// the whole keyboard surface once you are in the air.

"use strict";

import * as K from "./constants.js";
import { hazeEFoldingKm, hazeFromEFoldingKm, HAZE_MIN } from "./spectral.js";

/** Where the mode lives between sessions. Shared with tutorial.js. */
export const MODE_KEY = "soar.mode";

/**
 * What each mode's pause menu may show, as data.
 *
 * `items: null` means every item. The alternative — a conditional at each
 * item — spreads one decision across a dozen call sites, and the thing being
 * decided is a list.
 *
 * The tutorial and the way back to the start page are on every list: neither
 * is a research control, and a mode with no exit is not a mode.
 *
 * The labels carry the word "mode" — the toggle stands where "Paused" used to
 * and has to say what it switches, not just which of three.
 */
export const MODES = {
  basic: {
    label: "Basic mode",
    available: true,
    items: ["sun", "capture", "controls", "tutorial", "leave"],
  },
  research: { label: "Research mode", available: true, items: null },
  cyberpunk: {
    label: "Cyberpunk mode",
    available: false,
    note: "in development",
    items: ["sun", "capture", "controls", "tutorial", "leave"],
  },
};

export const DEFAULT_MODE = "basic";

/**
 * The stored mode, or the default.
 *
 * localStorage is writable by hand and survives a rename here, so an
 * unreadable value is a fact about the browser rather than an impossible
 * state: it is said out loud and replaced. A mode named in CODE that this
 * table does not have is a bug, and setMode throws on it.
 */
export function readMode() {
  let stored = null;
  try {
    stored = localStorage.getItem(MODE_KEY);
  } catch {
    return DEFAULT_MODE;         // storage denied (private mode, file://)
  }
  if (stored === null) return DEFAULT_MODE;
  if (MODES[stored]?.available) return stored;
  console.warn(`soar: stored mode '${stored}' is not available; using ` +
               `'${DEFAULT_MODE}'.`);
  return DEFAULT_MODE;
}

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};

/**
 * A labelled slider row that reports live and commits continuously.
 *
 * `label` is a string, or a list of strings and nodes when the label carries
 * a control of its own — the exposure row's auto toggle rides inside its
 * parentheses rather than spending a row of a panel meant to be looked past.
 */
function sliderRow(label, { min, max, step, value, format, onInput }) {
  const row = el("div", "row");
  const tag = el("label");
  tag.append(...(Array.isArray(label) ? label : [label]));
  row.append(tag);
  const input = el("input");
  input.type = "range";
  input.min = min; input.max = max; input.step = step; input.value = value;
  const readout = el("span", "value", format(value));
  input.addEventListener("input", () => {
    const v = Number(input.value);
    readout.textContent = format(v);
    onInput(v);
  });
  row.append(input, readout);
  row.setValue = (v) => { input.value = v; readout.textContent = format(v); };
  return row;
}

// Panels whose controls change (or capture) the picture dock to the bottom
// edge instead of the centre — anything where one would want to SEE the view
// while the panel is up (Thomas, 2026-08-11). Navigation and questions stay
// centred.
const DOCKED_PANELS = new Set(["quality", "sun", "terminal", "capture", "track"]);

/**
 * A row of mutually exclusive chips. An option may carry a third element,
 * `{ disabled, title }` — a segment that names something the build does not
 * do yet has to be visible to be honest about, and unclickable to be true.
 */
function segmented(options, isOn, onPick) {
  const wrap = el("div", "segmented");
  const buttons = options.map(([label, value, opts = {}]) => {
    const b = el("button", null, label);
    if (opts.disabled) {
      b.disabled = true;
      if (opts.title) b.title = opts.title;
    } else {
      b.addEventListener("click", () => onPick(value));
    }
    b.dataset.value = String(value);
    return b;
  });
  wrap.append(...buttons);
  wrap.refresh = () => {
    for (const b of buttons) b.classList.toggle("on", isOn(b.dataset.value));
  };
  wrap.refresh();
  return wrap;
}

/** A length in the unit that reads at its own magnitude. */
function distance(metres) {
  return Math.abs(metres) < 1000.0
    ? `${metres.toFixed(0)} m`
    : `${(metres / 1000.0).toFixed(2)} km`;
}

function item(label, note, onClick, { disabled = false } = {}) {
  const b = el("button", "item");
  b.append(el("span", null, label));
  if (note) b.append(el("span", "note", note));
  if (disabled) b.disabled = true;
  else b.addEventListener("click", onClick);
  return b;
}

export class UI {
  /** `app` is the viewer; see viewer.js for the surface used here. */
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.panel = null;          // null while flying
    // What the Quality panel shows while it is open, remembered for the
    // session. It is a preview, not a setting: only the panel turns it on and
    // leaving the panel always turns it off. See HOLD_MODES.
    this._qualityPreview = K.DEFAULT_QUALITY_PREVIEW;
    // Which menu the pause screen offers. Persisted, so a research session
    // stays a research session across a reload.
    this.mode = readMode();
    this._build();
  }

  /**
   * Switch modes. Takes effect on the next menu build, which is immediate —
   * the caller re-opens the panel it is standing in.
   *
   * A mode this table does not have, or one that is not built yet, is a bug
   * in the caller rather than something to interpret.
   */
  setMode(mode) {
    const spec = MODES[mode];
    if (!spec) throw new Error(`no such mode: ${mode}`);
    if (!spec.available) throw new Error(`mode '${mode}' is not available`);
    if (this.mode === mode) return;
    this.mode = mode;
    try {
      localStorage.setItem(MODE_KEY, mode);
    } catch {
      // Storage denied. The mode still applies to this session; only its
      // memory is lost, and there is nothing to do about that here.
    }
    this.app.tutorial?.onModeChange(mode);
  }

  /** Whether the current mode's menu carries this item. */
  allows(key) {
    const allow = MODES[this.mode].items;
    return allow === null || allow.includes(key);
  }

  _build() {
    const root = this.root;

    this.hint = el("div", "panel");
    this.hint.id = "hint";
    this.hint.innerHTML =
      "<b>Click</b> to fly &nbsp;·&nbsp; <b>WASD</b> move &nbsp;·&nbsp; " +
      "<b>Space / Shift</b> up, down &nbsp;·&nbsp; <b>scroll</b> speed " +
      "&nbsp;·&nbsp; <b>Esc</b> menu" +
      '<div class="sub"></div>';
    this.hintSub = this.hint.querySelector(".sub");

    this.stats = el("div", "panel");
    this.stats.id = "stats";
    this.stats.hidden = true;

    // The speed readout is its own corner chip rather than a line of the
    // stats overlay: speed flashes for a second and a half after the wheel
    // moves, whatever the overlay is doing, and the tutorial points at it.
    // fps lives in the stats overlay and nowhere else.
    this.speed = el("div", "panel");
    this.speed.id = "speed";
    this.speed.hidden = true;

    this.toolbar = el("div", "panel");
    this.toolbar.id = "toolbar";
    // No "menu" chip here. It only ever existed as a rescue for the state
    // "mouse free, menu closed, Escape swallowed" (bug 2); Escape now pauses
    // reliably from the keydown route, so the net came down rather than
    // staying up to hide the next regression.
    this.fsButton = el("button", "chip", "⛶ fullscreen");
    this.fsButton.addEventListener("click", () => this.app.toggleFullscreen());
    this.toolbar.append(this.fsButton);

    this.menu = el("div", "panel");
    this.menu.id = "menu";
    this.menu.hidden = true;

    this.toast = el("div", "panel");
    this.toast.id = "toast";

    this.progress = el("div", "panel");
    this.progress.id = "viewer-progress";
    this.progress.hidden = true;
    this.progress.innerHTML =
      '<div class="stage"></div><div class="bar"><span></span></div>';

    // #speed before #stats: viewer.css lifts the overlay off the chip with a
    // sibling selector, and a sibling selector only looks forwards.
    root.append(this.hint, this.speed, this.stats, this.toolbar, this.progress,
                this.menu, this.toast);
  }

  // --- transient messages --------------------------------------------------

  say(message, seconds = 2.5) {
    this.toast.textContent = message;
    this.toast.classList.add("show");
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(
      () => this.toast.classList.remove("show"), seconds * 1000);
  }

  showProgress(stage, fraction) {
    this.progress.hidden = false;
    this.progress.querySelector(".stage").textContent = stage;
    this.progress.querySelector(".bar span").style.width =
      `${Math.max(0, Math.min(1, fraction ?? 0)) * 100}%`;
  }

  hideProgress() { this.progress.hidden = true; }

  setSubtitle(text) { this.hintSub.textContent = text; }

  // --- stats ---------------------------------------------------------------

  /**
   * Off, or the whole instrument. Two states, not three.
   *
   * The middle rung used to be a one-line "42 fps · 180 m/s", which is two
   * numbers that now have homes of their own: the speed chip is always in the
   * corner, and fps is a diagnostic. A diagnostic that is half on is a third
   * state to reason about and answers no question the full one does not.
   */
  cycleStats() {
    this.statsMode = this.statsMode === "full" ? "off" : "full";
    this.stats.hidden = this.statsMode === "off";
    return this.statsMode;
  }

  /**
   * The readout, which has to be honest about a loop that stops.
   *
   * An fps number describes frames the renderer is choosing to produce. Once
   * the view has converged it produces none — the loop either presents the
   * finished picture for the flyer to cross, or stops outright — and the
   * last number measured then sits there describing nothing. So a converged
   * view reports what it actually is: parked, and how many samples deep.
   * Same for the startup probe, whose frames are deliberately serialized
   * against the GPU queue and so have a frame rate that means nothing at all.
   */
  drawStats({ fps, camera, tier, renderScale, frame, minimap, flyer,
              recording, showSpeed, parked = false, converged = false,
              probing = false, autoTier = false, wakeReason = null, spp = 0,
              holdRung = 0, holdRungCount = 1, holdCapped = false }) {
    // The corner chip: speed while it is flashing, and the recording dot for
    // as long as a track is being taken — the one piece of flight state with
    // no other home once the stats overlay is off.
    const rec = recording ? "● REC" : "";
    const speed = showSpeed ? `${camera.speed.toFixed(0)} m/s` : "";
    this.speed.textContent = [rec, speed].filter(Boolean).join(" · ");
    this.speed.hidden = !(rec || speed);

    if (this.statsMode !== "full") {
      this.stats.hidden = true;
      return;
    }
    this.stats.hidden = false;
    // `fps` is null until the first half-second of marching has been
    // averaged, which on a slow machine is a while.
    const rate = fps == null ? "—" : `${fps.toFixed(0)} fps`;
    // Which half of the loop is running, in the words the code uses for it.
    const loop = probing ? `probing ${tier}`
      : converged ? (parked ? "parked · presenting only" : "parked")
      : holdCapped ? `holding — capped at rung ${holdRung + 1}/${holdRungCount}`
      : holdRung > 0 ? `holding — rung ${holdRung + 1}/${holdRungCount}`
      : "flying";
    const rel = camera.relativePosition();
    // Where the camera actually is, in the units the field is measured in.
    // The relative triple below is the reproduction commands' convention and
    // means nothing on its own; a metre is a metre.
    const rows = [
      ["fps", converged ? `parked · ${spp} spp` : rate],
      ["loop", loop +
               (converged && wakeReason ? ` · last woke: ${wakeReason}` : "")],
      ["pos", ["x", "y", "z"]
        .map((axis, i) => `${axis} ${distance(camera.position[i])}`)
        .join(" · ")],
      ["rel", `(${rel.map((v) => v.toFixed(2)).join(", ")})`],
      ["view", `az ${camera.azimuth.toFixed(0)}° · el ` +
               `${camera.elevation.toFixed(0)}° · fov ${camera.fov.toFixed(0)}°`],
      ["speed", `${camera.speed.toFixed(0)} m/s`],
      ["tier", `${tier}${autoTier ? " (auto)" : ""} · ` +
               `${renderScale.toFixed(2)}x`],
      ["flags", `map ${minimap ? "on" : "off"} · flyer ${flyer}` +
                (recording ? " · REC" : "")],
      ["frame", `${frame}`],
    ];
    this.stats.innerHTML = rows
      .map(([k, v]) => `<div><span class="k">${k}</span> ${v}</div>`)
      .join("");
  }

  // --- menu ----------------------------------------------------------------

  get isOpen() { return !this.menu.hidden; }

  /**
   * Track the meter on the open Quality panel's exposure slider. Called by
   * the viewer whenever auto exposure applies a step; a detached row (panel
   * closed or rebuilt) takes the value harmlessly.
   */
  refreshExposure(value) {
    this._exposureRow?.setValue(value);
  }

  close() {
    this.menu.hidden = true;
    this.panel = null;
    this._syncQualityPreview();
    // Every open/close keeps the chrome classes honest. saveScreenshot used
    // to close the menu with no _syncChrome, leaving `menu-open` describing
    // a state the app was no longer in (bug 2's sibling).
    this.app._syncChrome?.();
  }

  /**
   * Esc: back out one level, or resume from the top.
   *
   * Resuming from the Escape KEY must not grab the pointer: Firefox honours
   * the request and then its own Escape handling exits the fresh lock, which
   * reads as lock loss and re-opens the menu — the "menu flashes away and
   * right back" bug. So the key resumes without capture (a click takes the
   * mouse); the clickable Resume items keep the grab, a click is a clean
   * gesture.
   */
  back(capture = true) {
    if (this.panel && this.panel !== "main") this.open("main");
    else this.app.resume({ capture });
  }

  open(name, context) {
    this.panel = name;
    this.menu.hidden = false;
    // Panels whose controls change the picture dock to the bottom edge so
    // the picture stays visible while it is being tuned — a centred box
    // sits exactly on top of the thing the sliders move.
    this.menu.classList.toggle("dock-bottom", DOCKED_PANELS.has(name));
    // The pause menu is two columns; every other centred panel is one.
    this.menu.classList.toggle("wide", name === "main");
    this._syncQualityPreview();
    this.menu.replaceChildren();
    const build = this[`_panel_${name}`];
    if (!build) throw new Error(`no such panel: ${name}`);
    this.app._syncChrome?.();
    try {
      build.call(this, context);
    } catch (err) {
      // A panel that throws half-way through leaves a blank box and no clue.
      console.error(`panel '${name}' failed:`, err);
      this.menu.append(this._header("error", "This panel failed to build"));
      this.menu.append(el("div", "row", String(err.message || err)));
      this.menu.append(el("div", "divider"));
      this.menu.append(item("Back to flying", null, () => this.app.resume()));
    }
  }

  /** A panel head. `kicker` may be null where it would only restate the
   *  title — two lines saying one thing is worse than one (Thomas,
   *  2026-08-14). */
  /**
   * Hold the flight picture still while — and only while — the Quality panel
   * is up, so its controls have something visible to act on. Driven from the
   * panel's lifecycle rather than from a setting, because a preview that
   * outlived the panel would quietly cost every later hold its still.
   */
  _syncQualityPreview() {
    this.app.setHoldMode?.(
      this.panel === "quality" ? this._qualityPreview : "still");
  }

  _header(kicker, title) {
    const h = el("div", "panel-head");
    if (kicker) h.append(el("h3", null, kicker));
    h.append(el("p", "menu-title", title));
    return h;
  }

  _backButton(label = "Back") {
    return item(label, null, () => this.open("main"));
  }

  /**
   * The mode toggle, which stands where the word "Paused" used to.
   *
   * "Paused" was the state and the menu around it already said so. What the
   * menu cannot say for itself is which of several menus it is, and that is
   * the one thing worth a control on the title line.
   */
  _modeToggle() {
    const toggle = segmented(
      Object.entries(MODES).map(([name, spec]) => [
        spec.label, name,
        spec.available ? {} : { disabled: true, title: spec.note },
      ]),
      (name) => name === this.mode,
      (name) => { this.setMode(name); this.open("main"); });
    toggle.classList.add("mode");
    toggle.dataset.menuKey = "mode";
    return toggle;
  }

  /**
   * The pause menu: two columns, and Resume sits in the title line.
   *
   * Which rows appear is the mode's business and nothing else's — every row
   * is built into a table keyed by name and then filtered once, so the modes
   * differ by a list in MODES rather than by conditionals scattered through
   * here. Left column is how the picture looks; right column is the field you
   * are flying and what you take away from it. Leaving is neither, so it sits
   * alone at the foot (Thomas, 2026-08-14).
   */
  _panel_main() {
    const app = this.app;
    const m = this.menu;

    const head = el("div", "menu-head");
    head.append(this._modeToggle());
    const resume = el("button", "resume");
    resume.append(el("span", null, "Resume"));
    resume.append(el("kbd", null, "Esc"));
    resume.addEventListener("click", () => app.resume());
    head.append(resume);
    m.append(head);

    const cols = el("div", "menu-cols");
    const appearance = el("div", "col");
    const session = el("div", "col");
    cols.append(appearance, session);
    m.append(cols);

    // Every row, by name. The key is also what the tutorial's spotlight
    // aims at, so renaming one moves both.
    const add = (col, key, node) => {
      if (!this.allows(key)) return;
      node.dataset.menuKey = key;
      col.append(node);
    };

    // "Current: …", like the file row, and named where it has a name: the
    // panel's own presets are the vocabulary, so a sun sitting on one is
    // reported as that rather than as the angle it happens to be (the same
    // test the panel's segmented control uses to light a segment).
    const sunPreset = K.SUN_PRESETS.find(
      (p) => Math.abs(app.sunElevation - p.elevation) < 0.05
             && Math.abs((app.sunAzimuth - p.azimuth) % 360.0) < 0.05);
    add(appearance, "sun", item(
      "Time of day…",
      `Current: ${sunPreset
        ? sunPreset.name
        : `${app.sunZenith.toFixed(0)}° zenith, sun to the ${app.sunCompass}`}`,
      () => this.open("sun")));
    // The tier's name and nothing else. The render scale, and whether the
    // machine chose the tier, are the quality panel's business (Thomas,
    // 2026-08-14).
    add(appearance, "quality", item(
      "Quality…",
      K.QUALITY_PRESETS[app.renderer.qualityTier].label.split(" —")[0],
      () => this.open("quality")));
    // Whether the field wraps is a fact about how the scene looks — an
    // endless sheet or a box in empty air — so it sits with the other two
    // that change the picture, above the overlay and window rows.
    add(appearance, "periodic",
        item(`Periodic domain: ${app.renderer.periodic ? "on" : "off"}`,
             "wrap the field laterally",
             () => { app.togglePeriodic(); this.open("main"); }));
    // No minimap, flyer or fullscreen row. Each was a row whose whole content
    // was a state and the key that cycles it, and the Controls panel is where
    // the keys are — so they are stated there, once, with their availability
    // (Thomas, 2026-08-22). Nothing is lost but three rows off the menu: the
    // keys work from the flight, which is where you are when you want them.

    // What is loaded belongs on the control that would replace it, not in a
    // block of its own at the foot (Thomas, 2026-08-14).
    const open = item("Open a NetCDF file…", `Current: ${app.sourceLabel}`,
                      () => app.pickFile());
    if (app.scene.nested) {
      const coverage = app.scene.nestCoverageFraction();
      open.append(el("span", "note",
        `+ nest, ${(app.renderer.dtView / app.renderer.dtViewNest).toFixed(0)}` +
        `× finer, covering ${(coverage * 100).toFixed(0)}% of the domain` +
        (coverage > 0.75 ? " — little of the outer field is visible" : "")));
    }
    add(session, "open", open);
    if (app.scene.nested) {
      add(session, "removeNest",
          item(`Remove the nest (${app.nestName ?? "nested field"})`,
               null, () => app.removeNest()));
    }
    // The panel does video too, and says so; the row is named for the thing
    // people open it to do. The ellipsis is the menu's mark for a row that
    // opens a panel rather than acting.
    add(session, "capture",
        item("Capture a still…", null, () => this.open("capture")));
    add(session, "terminal", item("Render this view in a terminal…", null,
                                  () => this.open("terminal")));
    add(session, "controls", item("Controls", null, () => this.open("controls")));
    // Up here rather than at the foot of the Controls panel, where it was: the
    // walkthrough is a thing you do with this session, like Capture and
    // Controls, and nobody opens a key-binding list looking for it.
    add(session, "tutorial",
        item("Replay the tutorial", null, () => this.app.tutorial?.replay()));
    const fov = sliderRow("Field of view", {
      min: K.FOV_LIMITS[0], max: K.FOV_LIMITS[1], step: 1,
      value: app.camera.fov, format: (v) => `${v.toFixed(0)}°`,
      onInput: (v) => app.setFov(v),
    });
    fov.classList.add("tight");
    add(session, "fov", fov);

    // Headers only over a column that got rows: basic mode empties neither,
    // but a mode list that did would otherwise leave a heading over nothing.
    if (appearance.childElementCount) {
      appearance.prepend(el("h3", null, "appearance"));
    }
    if (session.childElementCount) session.prepend(el("h3", null, "session"));

    m.append(el("div", "divider"));
    const foot = el("div", "menu-foot");
    if (this.allows("leave")) {
      const leave = item("Back to the start page", null, () => app.leave());
      leave.dataset.menuKey = "leave";
      foot.append(leave);
    }
    m.append(foot);
  }

  _panel_quality() {
    const app = this.app;
    const r = app.renderer;
    const m = this.menu;
    m.append(this._header("performance", "Quality"));

    // The tiers and the preview share one line. This panel is docked at the
    // bottom to be looked PAST, and every row it does not spend is sky
    // (Thomas, 2026-08-14) — which is also why the prose that used to explain
    // each of these is gone.
    const head = el("div", "seg-row");
    // Everything here goes through the viewer rather than straight at the
    // renderer, because every one of these changes the picture and the frame
    // loop may be asleep in front of it — the viewer's setters are what wake
    // it. Reaching past them would leave a slider that moves and a view that
    // does not.
    //
    // Auto leads the tier buttons: picking a tier by hand ends the automatic
    // choice for the session, and this is the way back. A hand-changed
    // Advanced setting grows a selected "Custom" chip on the right, and any
    // preset (Auto included) resets every Advanced setting and clears it.
    let tierRow;
    const buildTierRow = () => segmented(
      [["Auto", "__auto"],
       ...K.QUALITY_TIER_NAMES.map(
         (n) => [K.QUALITY_PRESETS[n].label.split(" —")[0], n]),
       ...(app.qualityCustom ? [["Custom", "__custom"]] : [])],
      (v) => app.qualityCustom
        ? v === "__custom"
        : (v === "__auto" ? app.autoTier : v === r.qualityTier),
      (v) => {
        if (v === "__custom") return;      // a state, not a choice
        if (v === "__auto") app.enableAutoTier();
        else app.setQualityTier(v);
        this.open("quality");
      });
    tierRow = buildTierRow();
    head.append(tierRow);
    m.append(head);
    // Called from every Advanced control's first hand-change: flips the
    // custom state and re-renders the tier row IN PLACE — the panel must not
    // rebuild wholesale under a hand still dragging a slider.
    const noteCustom = () => {
      if (app.qualityCustom) return;
      app.markQualityCustom();
      const fresh = buildTierRow();
      tierRow.replaceWith(fresh);
      tierRow = fresh;
    };

    m.append(item(this._advancedOpen ? "Advanced ▾" : "Advanced ▸",
                  this._advancedOpen ? null
                    : "preview, render scales, lighting method, image controls",
                  () => {
                    this._advancedOpen = !this._advancedOpen;
                    this.open("quality");
                  }));
    if (!this._advancedOpen) {
      m.append(el("div", "divider"));
      m.append(this._backButton());
      return;
    }

    const show = el("div", "seg-group");
    show.append(el("h3", null, "show"));
    show.append(segmented(
      [["Live", "live"], ["Still", "still"]],
      (v) => v === this._qualityPreview,
      (v) => { this._qualityPreview = v; this.open("quality"); }));
    m.append(show);

    // The lighting method, spelled as the two switches the presets set: on
    // for the cache always means the /2 bake (there is no other resolution).
    const boolRow = (label, get, set) => {
      const b = el("button", "mini");
      const paint = () => {
        b.textContent = get() ? "on" : "off";
        b.classList.toggle("on", get());
      };
      paint();
      b.addEventListener("click", () => { set(!get()); paint(); noteCustom(); });
      const row = el("div", "row");
      row.append(`${label}: `, b);
      return row;
    };
    m.append(boolRow("Light march cache",
                     () => app.lightCacheOn, (v) => app.setLightCache(v)));
    m.append(boolRow("Sky probe",
                     () => app.skyProbeOn, (v) => app.setSkyProbe(v)));

    m.append(el("div", "divider"));
    // Two columns while docked at the bottom: the panel exists to be looked
    // PAST — every row it saves is sky the sliders are actually changing.
    const sliders = el("div", "slider-grid");
    m.append(sliders);
    // Two scales, because the tier has two: what it marches at while you
    // move, and what a held view climbs to. The step has to divide the floor:
    // at 0.05 the slider could not reach 0.125 at all, and the bottom of its
    // own range was unselectable.
    const scaleRow = (label, value, onInput) => sliderRow(label, {
      min: K.MIN_RENDER_SCALE, max: K.MAX_RENDER_SCALE,
      step: K.RENDER_SCALE_SLIDER_STEP,
      value, format: (v) => `${v.toFixed(3)}x`, onInput,
    });
    sliders.append(scaleRow("Render scale, moving", r.flightRenderScale,
                            (v) => { app.setRenderScale(v); noteCustom(); }));
    sliders.append(scaleRow("Render scale, still", r.holdRenderScale,
                            (v) => { app.setHoldRenderScale(v); noteCustom(); }));
    // Samples a parked view settles at; also what a capture at this tier
    // would spend, unless the capture panel's own tier says otherwise.
    sliders.append(sliderRow("Parked samples", {
      min: K.PARKED_SPP_LIMITS[0], max: K.PARKED_SPP_LIMITS[1], step: 1,
      value: app.parkedSpp, format: (v) => `${v}`,
      onInput: (v) => { app.setParkedSpp(v); noteCustom(); },
    }));
    // Smoothing, not alpha — up is more. The tier sets how far up goes.
    sliders.append(sliderRow("Motion smoothing", {
      min: 0.0, max: 1.0, step: 0.01, value: app.motionSmoothing,
      format: (v) => v.toFixed(2),
      onInput: (v) => { app.setMotionSmoothing(v); noteCustom(); },
    }));
    // Exposure is a slider full stop, with auto as its mode (Thomas,
    // 2026-08-14) — so auto is a word inside the label rather than a row of
    // its own. Dragging the slider is choosing manual: setExposure turns
    // auto off, and the toggle has to be repainted where it stands, since
    // nothing rebuilds the panel under a hand that is still on the slider.
    const auto = el("button", "mini");
    const paintAuto = () => {
      auto.textContent = app.autoExposure ? "on" : "off";
      auto.classList.toggle("on", app.autoExposure);
    };
    paintAuto();
    auto.addEventListener("click", () => {
      app.setAutoExposure(!app.autoExposure);
      paintAuto();
      noteCustom();
    });
    const exposureRow = sliderRow(["Exposure (auto: ", auto, ")"], {
      min: K.EXPOSURE_LIMITS[0], max: K.EXPOSURE_LIMITS[1], step: 0.05,
      value: app.exposure, format: (v) => v.toFixed(2),
      onInput: (v) => { app.setExposure(v); paintAuto(); noteCustom(); },
    });
    sliders.append(exposureRow);
    this._exposureRow = exposureRow;
    sliders.append(sliderRow("Tone-map gamma", {
      min: K.TONE_MAP_GAMMA_LIMITS[0], max: K.TONE_MAP_GAMMA_LIMITS[1],
      step: 0.01, value: app.toneMapGamma, format: (v) => v.toFixed(2),
      onInput: (v) => { app.setToneMapGamma(v); noteCustom(); },
    }));
    // Read out as the distance it means. The knob drives four terms and is
    // not a length, but this is the length it implies at sea level, and it is
    // what the periodic march cap is derived from.
    // The profile rides inside the label the way auto does on Exposure: it is
    // a mode of this slider, not a control of its own. Off means the
    // sea-level extinction applies at every altitude — unphysical, and the
    // cheapest range lever there is, because an exponential atmosphere lets
    // an upward ray leave the haze and march to the ceiling.
    const profile = el("button", "mini");
    const paintProfile = () => {
      profile.textContent = app.hazeHeightDependent ? "on" : "off";
      profile.classList.toggle("on", app.hazeHeightDependent);
    };
    paintProfile();
    profile.addEventListener("click", () => {
      app.setHazeHeightDependent(!app.hazeHeightDependent);
      paintProfile();
      noteCustom();
    });
    // Scaled in the distance it reads out, logarithmically, rather than in
    // the aerosol coordinate. The range it has to cover is 2.5 km to 200 km —
    // eighty-fold — and linear travel in `haze` spends more than half its
    // length between 2 and 5 km of visibility, where consecutive positions
    // are indistinguishable, and crosses the whole clear end in a few pixels.
    // Log travel gives every position the same RATIO of change.
    //
    // The slider still runs clear-to-murky left-to-right, which is why the
    // mapping is inverted: it is labelled Haze, and dragging right must add
    // haze. Position is a plain 0..1; only the readout is a physical number.
    const hazeKm = (p) => K.HAZE_MAX_E_FOLDING_KM
      * (hazeEFoldingKm(K.HAZE_MAX) / K.HAZE_MAX_E_FOLDING_KM) ** p;
    const hazePosition = (haze) =>
      Math.log(hazeEFoldingKm(haze) / K.HAZE_MAX_E_FOLDING_KM)
      / Math.log(hazeEFoldingKm(K.HAZE_MAX) / K.HAZE_MAX_E_FOLDING_KM);
    sliders.append(sliderRow(["Haze (height dependent: ", profile, ")"], {
      min: 0.0, max: 1.0, step: 0.002, value: hazePosition(app.haze),
      format: (p) => {
        const km = hazeKm(p);
        return `${km >= 100 ? km.toFixed(0) : km.toFixed(1)} km`;
      },
      onInput: (p) => {
        app.setHaze(
          Math.min(K.HAZE_MAX, Math.max(HAZE_MIN, hazeFromEFoldingKm(hazeKm(p)))));
        noteCustom();
      },
    }));
    sliders.append(sliderRow("Level of detail", {
      min: K.LOD_STRENGTH_LIMITS[0], max: K.LOD_STRENGTH_LIMITS[1],
      step: 0.05, value: app.lodStrength,
      format: (v) => `${v.toFixed(2)}x`,
      onInput: (v) => { app.setLodStrength(v); noteCustom(); },
    }));
    sliders.append(sliderRow("White point", {
      min: 4.0, max: 40.0, step: 0.5, value: app.toneMapWhitePoint,
      format: (v) => v.toFixed(1),
      onInput: (v) => { app.setToneMapWhitePoint(v); noteCustom(); },
    }));
    sliders.append(sliderRow("Contrast", {
      min: 0.5, max: 1.6, step: 0.01, value: app.contrast,
      format: (v) => v.toFixed(2),
      onInput: (v) => { app.setContrast(v); noteCustom(); },
    }));
    m.append(el("div", "divider"));
    m.append(this._backButton());
  }

  _panel_sun() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header("time of day", "Sun"));

    m.append(segmented(
      K.SUN_PRESETS.map((p) => [
        p.name[0].toUpperCase() + p.name.slice(1), p.name]),
      (name) => {
        const p = K.SUN_PRESETS.find((q) => q.name === name);
        return p && Math.abs(app.sunElevation - p.elevation) < 0.05
          && Math.abs((app.sunAzimuth - p.azimuth) % 360.0) < 0.05;
      },
      (name) => {
        const p = K.SUN_PRESETS.find((q) => q.name === name);
        app.setSun({ azimuth: p.azimuth, elevation: p.elevation });
        this.open("sun");
      }));

    m.append(el("div", "divider"));
    const readout = el("div", "row");
    const refresh = () => {
      readout.textContent =
        `zenith ${app.sunZenith.toFixed(1)}° · elevation ` +
        `${app.sunElevation.toFixed(1)}° · azimuth ` +
        `${app.sunAzimuth.toFixed(0)}° (${app.sunCompass})`;
    };
    m.append(sliderRow("Solar zenith", {
      min: 0, max: 90 - K.MIN_SUN_ELEVATION_DEG, step: 0.5,
      value: app.sunZenith, format: (v) => `${v.toFixed(1)}°`,
      onInput: (v) => { app.setSun({ zenith: v }); refresh(); },
    }));
    m.append(sliderRow("Solar azimuth", {
      min: 0, max: 360, step: 1, value: app.sunAzimuth,
      format: (v) => `${v.toFixed(0)}°`,
      onInput: (v) => { app.setSun({ azimuth: v }); refresh(); },
    }));
    refresh();
    m.append(readout);
    m.append(el("div", "divider"));
    m.append(this._backButton());
  }

  _panel_controls() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header(null, "Controls"));
    const section = (title, rows) => {
      m.append(el("h3", null, title));
      for (const [key, what] of rows) {
        const row = el("div", "row");
        row.append(el("label", null, key));
        row.append(el("span", null, what));
        m.append(row);
      }
    };
    section("Flying", [
      ["W / S", "forward and back along the view"],
      ["A / D", "strafe"],
      ["Space", "climb"],
      ["Shift / C", "descend"],
      ["mouse", "look — Tab releases it, click takes it back"],
      ["scroll", "flight speed"],
    ]);
    // F-keys are Fn-gated on most laptops, so every one of them has a plain
    // key that does the same thing. F3 and ` are one binding, not two modes.
    //
    // The minimap and flyer keys carry their state and, when the field cannot
    // have one, the reason — this is the only place either is said now that
    // the menu rows are gone, and a field that cannot carry a flyer still owes
    // an explanation of why.
    section("While flying", [
      ["Esc", "menu"],
      ["F", "fullscreen"],
      ["F3 or `", "stats overlay on/off"],
      ["B", app.availableFlyers.length
        ? `flyer — paper dart, swift, off (now: ${app.flyerLabel})`
        : `flyer — ${app._flyerProblem ?? "not available for this field"}`],
      ["M", app.minimap
        ? `minimap — corner, fullscreen, off (now: ${
            { corner: "corner", full: "fullscreen", off: "off" }[app.minimapMode]})`
        : `minimap — ${app._minimapProblem ?? "not available for this field"}`],
      ["R", "record a flight track — R again to stop"],
      ["P", "capture a still"],
      ...(this.allows("quality")
        ? [["K", "light cache on/off (goes Custom; a preset resets)"],
           ["J", "sky probe on/off (goes Custom; a preset resets)"]]
        : []),
    ]);
    // Replay lives in the pause menu's session column now — see _panel_main.
    m.append(this._backButton());
  }

  _panel_terminal() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header(null, "Render in a terminal"));

    m.append(segmented(
      [["witness — exactly this view", "witness"],
       ["behold — path traced", "behold"]],
      (v) => v === app.terminalRenderer,
      (v) => { app.terminalRenderer = v; this.open("terminal"); }));

    m.append(el("div", "row",
      "Run this command in the folder where the file is saved."));
    if (app.terminalRenderer === "behold") {
      // witness drives the same WGSL as this view, so nothing is absent
      // from its frame and no outer/nest choice exists to offer; behold
      // needs its caveats and its outer/nest choice.
      if (app.renderer.periodic && app.viewSpansDomainEdge()) {
        m.append(el("div", "row",
          "This view spans a domain edge. behold does not tile, so its " +
          "frame will differ from what you see here."));
      }
      if (app.scene?.nested) {
        // Two fields on screen and behold renders one, so which is wanted
        // cannot be read off the view — the camera sees both.
        m.append(el("div", "row",
          "behold renders one field, not a nested pair. The other one will " +
          "be absent from its frame."));
        m.append(segmented(
          [[`Outer (${app.scene.groupPath || "root group"})`, "outer"],
           [`Nested (${app.scene.nestGroup || "root group"})`, "nest"]],
          (v) => v === app.beholdField,
          (v) => { app.beholdField = v; this.open("terminal"); }));
      }
      m.append(segmented(
        K.BEHOLD_QUALITY_ROWS.map(([label, value]) => [label, value]),
        (v) => v === app.beholdQuality,
        (v) => { app.beholdQuality = v; this.open("terminal"); }));
    }

    const command = app.terminalRenderer === "witness"
      ? app.witnessCommand() : app.beholdCommand();
    const box = el("div", "row");
    box.style.display = "block";
    box.style.fontFamily = "var(--ui-mono)";
    box.style.fontSize = "11.5px";
    box.style.wordBreak = "break-all";
    box.textContent = command;
    m.append(box);
    m.append(item("Copy the command", null, async () => {
      try {
        await navigator.clipboard.writeText(command);
        this.say("Copied to the clipboard.");
      } catch (err) {
        this.say(`Clipboard unavailable (${err.message}).\n` +
                 "The command is printed to the console.");
        console.log(command);
      }
    }));
    m.append(el("div", "divider"));
    m.append(this._backButton());
  }

  /** What R offers when you stop recording — including every video option.
   *  Video settings live HERE, not in the capture panel: they only mean
   *  anything once a track exists (Thomas, 2026-08-11). */
  _panel_track({ samples }) {
    const app = this.app;
    const m = this.menu;
    const duration = samples[samples.length - 1][0];
    m.append(this._header("track", "Flight recorded"));

    const summary = el("div", "row");
    m.append(summary);

    m.append(el("div", "divider"));
    m.append(el("h3", null, "size"));
    m.append(segmented(
      [["Window", "window"],
       ...K.CAPTURE_SIZE_PRESETS.map(([label, size]) => [label, size.join("x")])],
      (v) => v === (app.captureSize ? app.captureSize.join("x") : "window"),
      (v) => {
        app.captureSize = v === "window" ? null : v.split("x").map(Number);
        this.open("track", { samples });
      }));

    // Whether video works at all is a question only video.js can answer, and
    // it answers it by encoding a frame — so the button is built optimistically
    // and the capability line is filled in once the module has loaded.
    const render = item("Render it to video", "",
                        () => app.renderTrackVideo(samples));
    const refresh = () => {
      let frames = null, problem = "";
      try {
        frames = app.trackFrameCount(samples);
      } catch (err) {
        problem = ` ${String(err.message || err)}`;
      }
      summary.textContent =
        `${samples.length} samples over ${duration.toFixed(1)} seconds` +
        (frames === null
          ? `.${problem}`
          : `, which becomes ${frames} frames at ` +
            `${app.videoFps.toFixed(0)} fps.`);
      if (render.disabled) return;   // the capability line owns the note
      const note = render.querySelector(".note");
      if (note) {
        note.textContent =
          `${K.PARKED_ACCUM_FRAMES_BY_TIER[app.captureVideoTier]} samples ` +
          `per pixel at the ` +
          `${K.QUALITY_PRESETS[app.captureVideoTier].label.split(" —")[0]} ` +
          `preset, ${app.videoFps.toFixed(0)} fps`;
      }
    };
    m.append(sliderRow("Frame rate", {
      min: K.VIDEO_FPS_LIMITS[0], max: K.VIDEO_FPS_LIMITS[1], step: 1,
      value: app.videoFps, format: (v) => `${v.toFixed(0)} fps`,
      onInput: (v) => { app.videoFps = v; refresh(); },
    }));
    // The video's tier: each rendered frame is that preset's flight
    // configuration at the capture size, accumulated to its parked spp.
    // Refreshed in place — reopening the panel would drop its {samples}
    // context, which is the recording itself.
    m.append(el("h3", null, "quality"));
    const tierSeg = segmented(
      K.QUALITY_TIER_NAMES.map(
        (n) => [K.QUALITY_PRESETS[n].label.split(" —")[0], n]),
      (v) => v === app.captureVideoTier,
      (v) => { app.captureVideoTier = v; refresh(); tierSeg.refresh(); });
    m.append(tierSeg);

    m.append(el("div", "divider"));
    m.append(render);
    refresh();
    import("./video.js").then(({ videoCapabilities }) => {
      const caps = videoCapabilities();
      if (caps.available) return;
      render.disabled = true;
      const note = render.querySelector(".note");
      if (note) note.textContent = caps.why;
    });
    m.append(item("Save the track", "a .json cloudyview's render_track reads",
                  () => { app.downloadTrack(samples); this.open("main"); }));
    m.append(item("Discard it", null, () => this.open("main")));
  }

  /** Stills only. Video options live in the track panel, which appears when
   *  a recording stops — they mean nothing before a track exists. */
  _panel_capture() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header("capture", "Still image"));

    m.append(el("h3", null, "size"));
    m.append(segmented(
      [["Window", "window"],
       ...K.CAPTURE_SIZE_PRESETS.map(([label, size]) => [label, size.join("x")])],
      (v) => v === (app.captureSize ? app.captureSize.join("x") : "window"),
      (v) => {
        app.captureSize = v === "window" ? null : v.split("x").map(Number);
        this.open("capture");
      }));

    // A capture picks its own tier — the still leans expensive by default —
    // and its spp is that tier's parked sample count. All settings are the
    // preset's flight configuration, marched at the capture size; witness
    // --soar-tier reproduces exactly this from the CLI.
    m.append(el("h3", null, "quality"));
    m.append(segmented(
      K.QUALITY_TIER_NAMES.map(
        (n) => [K.QUALITY_PRESETS[n].label.split(" —")[0], n]),
      (v) => v === app.captureStillTier,
      (v) => { app.captureStillTier = v; this.open("capture"); }));

    m.append(el("div", "divider"));
    m.append(item("Save a still",
                  `${K.PARKED_ACCUM_FRAMES_BY_TIER[app.captureStillTier]} ` +
                  "samples per pixel, then a PNG download · shortcut: P",
                  () => app.saveScreenshot({ overlays: true })));
    m.append(item("Save a still, clouds only", "no bird, no minimap",
                  () => app.saveScreenshot({ overlays: false })));

    m.append(el("div", "row",
      "For video: press R while flying to record a track."));
    m.append(el("div", "divider"));
    m.append(this._backButton());
  }

  /** Which of several groups in one file to open, and whether to nest. */
  _panel_groups({ groups, pairs, filename, onPick, onPickPair, onCancel }) {
    const m = this.menu;
    m.append(this._header("open file", "Which group?"));
    m.append(el("div", "row", filename));
    if (pairs.length) {
      m.append(el("div", "row",
        pairs.length === 1
          ? `'${pairs[0][1]}' lies inside '${pairs[0][0]}' and is finer — ` +
            "they can be rendered together, the finer one taking over where " +
            "it covers."
          : "These nest — two levels render together, the finer taking over " +
            "where it covers:"));
      for (const [outer, inner] of pairs) {
        m.append(item(`${outer}  +  ${inner}`, "both, nested",
                      () => onPickPair([outer, inner])));
      }
      m.append(el("div", "divider"));
      m.append(el("h3", null, "or just one"));
    } else {
      m.append(el("div", "row",
        "The root group holds no cloud field. These groups do:"));
    }
    for (const g of groups) m.append(item(g || "(root)", null, () => onPick(g)));
    m.append(el("div", "divider"));
    m.append(item("Back", "choose a different file", onCancel));
  }

  /** Condensate units, when the file does not say. */
  _panel_units({ variables, filename, onPick, onCancel }) {
    const m = this.menu;
    m.append(this._header("open file", "Which units?"));
    m.append(el("div", "row", filename));
    m.append(el("div", "row",
      `No units attribute on ${variables.join(", ")}. Mixing ratios are ` +
      "usually kg/kg; SAM-style output is g/kg."));
    m.append(item("g/kg", null, () => onPick("g/kg")));
    m.append(item("kg/kg", null, () => onPick("kg/kg")));
    m.append(el("div", "divider"));
    m.append(item("Back", "choose a different file", onCancel));
  }

  _panel_message({ kicker, title, body, advice, actions }) {
    const m = this.menu;
    m.append(this._header(kicker, title));
    m.append(el("div", "row", body));
    if (advice) m.append(el("div", "row", advice));
    m.append(el("div", "divider"));
    for (const [label, onClick] of actions) m.append(item(label, null, onClick));
  }
}
