// The overlay: menu, panels, readouts, prompts.
//
// Menus are click-only. Keys are reserved for things you do WHILE flying —
// where reaching for a menu would mean losing the shot — so the desktop's key
// state machine does not come across. Esc, Tab, F, F3, B, M, R and F12 are
// the whole keyboard surface once you are in the air.

"use strict";

import * as K from "./constants.js";
import { hazeEFoldingKm, hazeFromEFoldingKm, HAZE_MIN } from "./spectral.js";

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

function segmented(options, isOn, onPick) {
  const wrap = el("div", "segmented");
  const buttons = options.map(([label, value]) => {
    const b = el("button", null, label);
    b.addEventListener("click", () => onPick(value));
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
    this._build();
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

    root.append(this.hint, this.stats, this.toolbar, this.progress,
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

  /** subtle -> expanded -> hidden, and round again. */
  cycleStats() {
    const order = ["subtle", "expanded", "hidden"];
    this.statsMode = order[(order.indexOf(this.statsMode ?? "subtle") + 1)
                           % order.length];
    this.stats.hidden = this.statsMode === "hidden";
    this.stats.classList.toggle("compact", this.statsMode === "subtle");
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
  drawStats({ fps, frameMs, camera, tier, renderScale, frame, minimap, flyer,
              recording, showSpeed, parked = false, converged = false,
              probing = false, autoTier = false, wakeReason = null, spp = 0,
              holdRung = 0, holdRungCount = 1, holdCapped = false }) {
    // `fps` is null until the first half-second of marching has been
    // averaged, which on a slow machine is a while — but the probe is running
    // in exactly that window and has something to say, so a null rate is no
    // longer on its own a reason to hide.
    if (this.statsMode === "hidden" || (fps == null && !probing && !converged)) {
      this.stats.hidden = true;
      return;
    }
    this.stats.hidden = false;
    const rate = fps == null
      ? "—"
      : `${fps.toFixed(0)} fps · ${frameMs.toFixed(1)} ms`;
    if (this.statsMode !== "expanded") {
      const speed = showSpeed ? ` · ${camera.speed.toFixed(0)} m/s` : "";
      const rec = recording ? "● " : "";
      const state = probing ? "measuring quality…"
        : converged ? `parked · ${spp} spp`
        : rate;
      this.stats.textContent = `${rec}${state}${speed}`;
      return;
    }
    // Which half of the loop is running, in the words the code uses for it.
    const loop = probing ? `probing ${tier}`
      : converged ? (parked ? "parked · presenting only" : "parked")
      : holdCapped ? `holding — capped at rung ${holdRung + 1}/${holdRungCount}`
      : holdRung > 0 ? `holding — rung ${holdRung + 1}/${holdRungCount}`
      : "flying";
    const rel = camera.relativePosition();
    const rows = [
      ["fps", converged ? `parked · ${spp} spp` : rate],
      ["loop", loop +
               (converged && wakeReason ? ` · last woke: ${wakeReason}` : "")],
      ["pos", `(${rel.map((v) => v.toFixed(2)).join(", ")})`],
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
   * The pause menu: two columns, and Resume sits in the title line.
   *
   * "Paused" is the state and Resume is the way out of it, so the way out
   * belongs on the same line as the state rather than as the first of a dozen
   * identical rows. Left column is how the picture looks; right column is the
   * field you are flying and what you take away from it. Leaving is neither,
   * so it sits alone at the foot (Thomas, 2026-08-14).
   */
  _panel_main() {
    const app = this.app;
    const m = this.menu;

    const head = el("div", "menu-head");
    head.append(el("p", "menu-title", "Paused"));
    const resume = el("button", "resume");
    resume.append(el("span", null, "Resume"));
    resume.append(el("kbd", null, "Esc"));
    resume.addEventListener("click", () => app.resume());
    head.append(resume);
    m.append(head);

    const cols = el("div", "menu-cols");
    const appearance = el("div", "col");
    const session = el("div", "col");
    appearance.append(el("h3", null, "appearance"));
    session.append(el("h3", null, "session"));
    cols.append(appearance, session);
    m.append(cols);

    appearance.append(item("Time of day…",
                           `${app.sunZenith.toFixed(0)}° zenith`,
                           () => this.open("sun")));
    // The tier's name and nothing else. The render scale, and whether the
    // machine chose the tier, are the quality panel's business (Thomas,
    // 2026-08-14).
    appearance.append(item(
      "Quality…",
      K.QUALITY_PRESETS[app.renderer.qualityTier].label.split(" —")[0],
      () => this.open("quality")));
    // Whether the field wraps is a fact about how the scene looks — an
    // endless sheet or a box in empty air — so it sits with the other two
    // that change the picture, above the overlay and window rows.
    appearance.append(item(`Periodic domain: ${app.renderer.periodic ? "on" : "off"}`,
                           "wrap the field laterally",
                           () => { app.togglePeriodic(); this.open("main"); }));
    appearance.append(item(
      `Minimap: ${!app.minimap ? "off" : { corner: "on", full: "fullscreen", off: "off" }[app.minimapMode]}`,
      app.minimap
        ? "Keyboard shortcut: M"
        : (app._minimapProblem ?? "not available for this field"),
      () => {
        app.toggleMinimap();
        // Landing on fullscreen, the whole point is the map this menu would
        // cover — and "click to travel" cannot work under it (bug 12). The
        // menu closes (mouse stays free for the map; Escape brings it back);
        // the other modes re-open it so the row's label updates.
        if (app.minimapMode === "full") this.close();
        else this.open("main");
      }));
    // Like the minimap row above: the label carries the state and the key
    // cycles it, because there are three states and a checkbox has two. The
    // second branch of the subtitle has to survive — a field that cannot
    // carry a flyer still owes an explanation of why.
    appearance.append(item(
      `Flyer: ${app.flyerLabel}`,
      app.availableFlyers.length
        ? "Keyboard shortcut: B"
        : (app._flyerProblem ?? "not available for this field"),
      () => { app.cycleFlyer(); this.open("main"); }));
    appearance.append(item(app.isFullscreen ? "Exit fullscreen" : "Enter fullscreen",
                           "Keyboard shortcut: F",
                           () => { app.toggleFullscreen(); this.open("main"); }));

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
    session.append(open);
    if (app.scene.nested) {
      session.append(item(`Remove the nest (${app.nestName ?? "nested field"})`,
                          null, () => app.removeNest()));
    }
    session.append(item("Capture…", null, () => this.open("capture")));
    session.append(item("Render this view in a terminal…", null,
                        () => this.open("terminal")));
    session.append(item("Controls", null, () => this.open("controls")));
    const fov = sliderRow("Field of view", {
      min: K.FOV_LIMITS[0], max: K.FOV_LIMITS[1], step: 1,
      value: app.camera.fov, format: (v) => `${v.toFixed(0)}°`,
      onInput: (v) => app.setFov(v),
    });
    fov.classList.add("tight");
    session.append(fov);

    m.append(el("div", "divider"));
    const foot = el("div", "menu-foot");
    foot.append(item("Back to the start page", null, () => app.leave()));
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
    head.append(segmented(
      K.QUALITY_TIER_NAMES.map((n) => [K.QUALITY_PRESETS[n].label.split(" —")[0], n]),
      (v) => v === r.qualityTier,
      (v) => { app.setQualityTier(v); this.open("quality"); }));
    const show = el("div", "seg-group");
    show.append(el("h3", null, "show"));
    show.append(segmented(
      [["Live", "live"], ["Still", "still"]],
      (v) => v === this._qualityPreview,
      (v) => { this._qualityPreview = v; this.open("quality"); }));
    head.append(show);
    m.append(head);
    if (r.qualityIsCustom) {
      m.append(el("div", "row", "Render scale has been set by hand, so this " +
                               "tier is running custom."));
    }

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
                            (v) => app.setRenderScale(v)));
    sliders.append(scaleRow("Render scale, still", r.holdRenderScale,
                            (v) => app.setHoldRenderScale(v)));
    // Smoothing, not alpha — up is more. The tier sets how far up goes.
    sliders.append(sliderRow("Motion smoothing", {
      min: 0.0, max: 1.0, step: 0.01, value: app.motionSmoothing,
      format: (v) => v.toFixed(2),
      onInput: (v) => app.setMotionSmoothing(v),
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
    });
    const exposureRow = sliderRow(["Exposure (auto: ", auto, ")"], {
      min: K.EXPOSURE_LIMITS[0], max: K.EXPOSURE_LIMITS[1], step: 0.05,
      value: app.exposure, format: (v) => v.toFixed(2),
      onInput: (v) => { app.setExposure(v); paintAuto(); },
    });
    sliders.append(exposureRow);
    this._exposureRow = exposureRow;
    sliders.append(sliderRow("Tone-map gamma", {
      min: K.TONE_MAP_GAMMA_LIMITS[0], max: K.TONE_MAP_GAMMA_LIMITS[1],
      step: 0.01, value: app.toneMapGamma, format: (v) => v.toFixed(2),
      onInput: (v) => app.setToneMapGamma(v),
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
      onInput: (p) => app.setHaze(
        Math.min(K.HAZE_MAX, Math.max(HAZE_MIN, hazeFromEFoldingKm(hazeKm(p))))),
    }));
    sliders.append(sliderRow("Level of detail", {
      min: K.LOD_STRENGTH_LIMITS[0], max: K.LOD_STRENGTH_LIMITS[1],
      step: 0.05, value: app.lodStrength,
      format: (v) => `${v.toFixed(2)}x`,
      onInput: (v) => app.setLodStrength(v),
    }));
    sliders.append(sliderRow("White point", {
      min: 4.0, max: 40.0, step: 0.5, value: app.toneMapWhitePoint,
      format: (v) => v.toFixed(1),
      onInput: (v) => app.setToneMapWhitePoint(v),
    }));
    sliders.append(sliderRow("Contrast", {
      min: 0.5, max: 1.6, step: 0.01, value: app.contrast,
      format: (v) => v.toFixed(2),
      onInput: (v) => app.setContrast(v),
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
    section("While flying", [
      ["Esc", "menu"],
      ["F", "fullscreen"],
      ["F3", "stats: brief, full, off"],
      ["B", "flyer — paper dart, swift, off"],
      ["M", "minimap"],
      ["R", "record a flight track"],
      ["F12", "screenshot"],
    ]);
    m.append(el("div", "divider"));
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
          `${app.videoAccumulate} samples per pixel, at ` +
          `${app.videoFps.toFixed(0)} fps`;
      }
    };
    m.append(sliderRow("Frame rate", {
      min: K.VIDEO_FPS_LIMITS[0], max: K.VIDEO_FPS_LIMITS[1], step: 1,
      value: app.videoFps, format: (v) => `${v.toFixed(0)} fps`,
      onInput: (v) => { app.videoFps = v; refresh(); },
    }));
    m.append(sliderRow("Samples per pixel", {
      min: K.VIDEO_ACCUMULATE_LIMITS[0], max: K.VIDEO_ACCUMULATE_LIMITS[1],
      step: 1, value: app.videoAccumulate, format: (v) => `${v}`,
      onInput: (v) => { app.videoAccumulate = v; refresh(); },
    }));

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

    m.append(el("div", "divider"));
    const save = item("Save a still", "",
                      () => app.saveScreenshot({ overlays: true }));
    const refresh = () => {
      const note = save.querySelector(".note");
      if (note) {
        note.textContent =
          `${app.stillSamples} samples per pixel, then a PNG download`;
      }
    };
    m.append(sliderRow("Samples per pixel", {
      min: K.STILL_SAMPLES_LIMITS[0], max: K.STILL_SAMPLES_LIMITS[1],
      step: 1, value: app.stillSamples, format: (v) => `${v}`,
      onInput: (v) => { app.stillSamples = v; refresh(); },
    }));
    m.append(save);
    refresh();
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
