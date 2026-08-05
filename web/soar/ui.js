// The overlay: menu, panels, readouts, prompts.
//
// Menus are click-only. Keys are reserved for things you do WHILE flying —
// where reaching for a menu would mean losing the shot — so the desktop's key
// state machine does not come across. Esc, Tab, F, F3, B, M, R and F12 are
// the whole keyboard surface once you are in the air.

"use strict";

import * as K from "./constants.js";

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};

/** A labelled slider row that reports live and commits continuously. */
function sliderRow(label, { min, max, step, value, format, onInput }) {
  const row = el("div", "row");
  row.append(el("label", null, label));
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
    this.menuButton = el("button", "chip", "☰ menu");
    this.menuButton.addEventListener("click", () => this.open("main"));
    this.fsButton = el("button", "chip", "⛶ fullscreen");
    this.fsButton.addEventListener("click", () => this.app.toggleFullscreen());
    this.toolbar.append(this.menuButton, this.fsButton);

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

  drawStats({ fps, frameMs, camera, tier, renderScale, frame, minimap, bird,
              recording, showSpeed }) {
    if (this.statsMode === "hidden" || fps == null) {
      this.stats.hidden = true;
      return;
    }
    this.stats.hidden = false;
    if (this.statsMode !== "expanded") {
      const speed = showSpeed ? ` · ${camera.speed.toFixed(0)} m/s` : "";
      const rec = recording ? "● " : "";
      this.stats.textContent =
        `${rec}${fps.toFixed(0)} fps · ${frameMs.toFixed(1)} ms${speed}`;
      return;
    }
    const rel = camera.relativePosition();
    const rows = [
      ["fps", `${fps.toFixed(1)} · ${frameMs.toFixed(1)} ms`],
      ["pos", `(${rel.map((v) => v.toFixed(2)).join(", ")})`],
      ["view", `az ${camera.azimuth.toFixed(0)}° · el ` +
               `${camera.elevation.toFixed(0)}° · fov ${camera.fov.toFixed(0)}°`],
      ["speed", `${camera.speed.toFixed(0)} m/s`],
      ["tier", `${tier} · ${renderScale.toFixed(2)}x`],
      ["flags", `map ${minimap ? "on" : "off"} · bird ${bird ? "on" : "off"}` +
                (recording ? " · REC" : "")],
      ["frame", `${frame}`],
    ];
    this.stats.innerHTML = rows
      .map(([k, v]) => `<div><span class="k">${k}</span> ${v}</div>`)
      .join("");
  }

  // --- menu ----------------------------------------------------------------

  get isOpen() { return !this.menu.hidden; }

  close() {
    this.menu.hidden = true;
    this.panel = null;
  }

  /** Esc: back out one level, or resume from the top. */
  back() {
    if (this.panel && this.panel !== "main") this.open("main");
    else this.app.resume();
  }

  open(name, context) {
    this.panel = name;
    this.menu.hidden = false;
    this.menu.replaceChildren();
    const build = this[`_panel_${name}`];
    if (!build) throw new Error(`no such panel: ${name}`);
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

  _header(kicker, title) {
    const h = el("div");
    h.append(el("h3", null, kicker));
    h.append(el("p", "menu-title", title));
    return h;
  }

  _backButton(label = "Back") {
    return item(label, null, () => this.open("main"));
  }

  _panel_main() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header("cloudyview", "Paused"));

    m.append(item("Resume", "Esc", () => app.resume()));
    m.append(item("Open a file…", "netCDF, read on this machine",
                  () => app.pickFile()));
    if (app.scene.nested) {
      m.append(item(`Remove the nest (${app.nestName ?? "nested field"})`,
                    null, () => app.removeNest()));
    }
    m.append(item("Time of day…",
                  `${app.sunZenith.toFixed(0)}° zenith`,
                  () => this.open("sun")));
    m.append(item("Quality…",
                  `${K.QUALITY_PRESETS[app.renderer.qualityTier].label}` +
                  `, ${app.renderer.renderScale.toFixed(2)}x`,
                  () => this.open("quality")));
    m.append(item("Capture…", "still image or a flight video",
                  () => this.open("capture")));
    m.append(item("Render this view in behold…",
                  "the path-traced command, for a terminal",
                  () => this.open("behold")));
    m.append(item("Controls", null, () => this.open("controls")));
    m.append(item(
      `Minimap: ${app.minimap && app.minimapEnabled ? "on" : "off"}`,
      app.minimap
        ? "M — the overhead albedo map, top right"
        : (app._minimapProblem ?? "not available for this field"),
      () => { app.toggleMinimap(); this.open("main"); }));
    m.append(item(
      `Bird: ${app.bird && app.birdEnabled ? "on" : "off"}`,
      app.bird
        ? "B — a swift, flying ahead of you"
        : (app._birdProblem ?? "not available for this field"),
      () => { app.toggleBird(); this.open("main"); }));
    m.append(item(`Periodic domain: ${app.renderer.periodic ? "on" : "off"}`,
                  "wrap the field laterally, so flight never runs out",
                  () => { app.togglePeriodic(); this.open("main"); }));
    m.append(item(app.isFullscreen ? "Exit fullscreen" : "Enter fullscreen",
                  "F", () => { app.toggleFullscreen(); this.open("main"); }));
    m.append(item("Back to the start page", null, () => app.leave()));

    m.append(el("div", "divider"));
    const fov = sliderRow("Field of view", {
      min: K.FOV_LIMITS[0], max: K.FOV_LIMITS[1], step: 1,
      value: app.camera.fov, format: (v) => `${v.toFixed(0)}°`,
      onInput: (v) => app.camera.setFov(v),
    });
    m.append(fov);

    const source = el("div", "row");
    source.style.display = "block";
    source.append(el("h3", null, "field"));
    source.append(el("div", null, app.sourceLabel));
    if (app.scene.nested) {
      const coverage = app.scene.nestCoverageFraction();
      source.append(el("div", null,
        `+ nest, ${(app.renderer.dtView / app.renderer.dtViewNest).toFixed(0)}` +
        `× finer, covering ${(coverage * 100).toFixed(0)}% of the domain` +
        (coverage > 0.75 ? " — little of the outer field is visible" : "")));
    }
    m.append(source);
  }

  _panel_quality() {
    const app = this.app;
    const r = app.renderer;
    const m = this.menu;
    m.append(this._header("performance", "Quality"));

    const tiers = segmented(
      K.QUALITY_TIER_NAMES.map((n) => [K.QUALITY_PRESETS[n].label.split(" —")[0], n]),
      (v) => v === r.qualityTier,
      (v) => { app.setQualityTier(v); this.open("quality"); });
    m.append(tiers);
    if (r.qualityIsCustom) {
      m.append(el("div", "row", "Render scale has been set by hand, so this " +
                               "tier is running custom."));
    }

    m.append(el("div", "divider"));
    m.append(sliderRow("Render scale", {
      min: K.MIN_RENDER_SCALE, max: K.MAX_RENDER_SCALE, step: 0.05,
      value: r.flightRenderScale, format: (v) => `${v.toFixed(2)}x`,
      onInput: (v) => r.setRenderScale(v),
    }));
    m.append(sliderRow("Motion smoothing", {
      min: 0.3, max: 0.9, step: 0.01, value: r.motionBlendAlpha,
      format: (v) => v.toFixed(2),
      onInput: (v) => { r.motionBlendAlpha = v; },
    }));
    m.append(sliderRow("Tone-map gamma", {
      min: K.TONE_MAP_GAMMA_LIMITS[0], max: K.TONE_MAP_GAMMA_LIMITS[1],
      step: 0.01, value: app.toneMapGamma, format: (v) => v.toFixed(2),
      onInput: (v) => app.setToneMapGamma(v),
    }));
    m.append(el("div", "row",
      `${K.TONE_MAP_GAMMA_WITNESS} is witness's own value — darker, with a ` +
      `harder far field. Higher lifts distance into haze; ` +
      `${K.TONE_MAP_GAMMA_AS_FLOWN} is what the desktop window used to ` +
      `render by encoding gamma twice. The default sits between them.`));

    if (r.qualityTier === "potato") {
      m.append(el("div", "row",
        "Potato switches to exact High sampling the moment the camera " +
        "stops, then accumulates a smooth still."));
    }
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
    if (app.sunElevation <= K.MIN_SUN_ELEVATION_DEG + 1e-6) {
      m.append(el("div", "row",
        "At the horizon. The sun cannot go below it while the domain is " +
        "periodic — the light march exits through the domain top."));
    }
    m.append(el("div", "divider"));
    m.append(this._backButton());
  }

  _panel_controls() {
    const m = this.menu;
    m.append(this._header("cloudyview", "Controls"));
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
      ["B", "bird"],
      ["M", "minimap"],
      ["R", "record a flight track"],
      ["F12", "screenshot"],
    ]);
    m.append(el("div", "row",
      "Everything else lives in the menu — there is nothing you can only " +
      "reach by keyboard."));
    m.append(el("div", "divider"));
    m.append(this._backButton());
  }

  _panel_behold() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header("render in behold", "Command for this view"));
    m.append(el("div", "row",
      "Path tracing runs in a terminal, not in a tab. This command " +
      "reproduces exactly what you are looking at — camera, sun, field and " +
      "all."));
    if (app.renderer.periodic && app.viewSpansDomainEdge()) {
      m.append(el("div", "row",
        "This view spans a domain edge. behold does not tile, so its frame " +
        "will differ from what you see here."));
    }
    m.append(segmented(
      K.BEHOLD_QUALITY_ROWS.map(([label, value]) => [label, value]),
      (v) => v === app.beholdQuality,
      (v) => { app.beholdQuality = v; this.open("behold"); }));

    const command = app.beholdCommand();
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

  /** What R offers when you stop recording. */
  _panel_track({ samples }) {
    const app = this.app;
    const m = this.menu;
    const duration = samples[samples.length - 1][0];
    m.append(this._header("track", "Flight recorded"));

    let frames = null, problem = null;
    try {
      frames = app.trackFrameCount(samples);
    } catch (err) {
      problem = String(err.message || err);
    }
    m.append(el("div", "row",
      `${samples.length} samples over ${duration.toFixed(1)} seconds` +
      (frames === null
        ? "."
        : `, which becomes ${frames} frames at ` +
          `${app.videoFps.toFixed(0)} fps.`)));
    if (problem) m.append(el("div", "row", problem));

    m.append(el("div", "divider"));
    // Whether video works at all is a question only video.js can answer, and
    // it answers it by encoding a frame — so the button is built optimistically
    // and the capability line is filled in once the module has loaded.
    const render = item(
      "Render it to video",
      `${app.videoAccumulate} accumulated passes per frame, ` +
      `at ${app.videoFps.toFixed(0)} fps`,
      () => app.renderTrackVideo(samples));
    m.append(render);
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

  _panel_capture() {
    const app = this.app;
    const m = this.menu;
    m.append(this._header("capture", "Still or video"));

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
    m.append(item("Save a still", `${K.STILL_ACCUMULATE_FRAMES} accumulated ` +
                  "passes, then a PNG download",
                  () => app.saveScreenshot({ overlays: true })));
    m.append(item("Save a still, clouds only", "no bird, no minimap",
                  () => app.saveScreenshot({ overlays: false })));

    m.append(el("div", "divider"));
    m.append(el("h3", null, "flight video"));
    m.append(sliderRow("Frame rate", {
      min: K.VIDEO_FPS_LIMITS[0], max: K.VIDEO_FPS_LIMITS[1], step: 1,
      value: app.videoFps, format: (v) => `${v.toFixed(0)} fps`,
      onInput: (v) => { app.videoFps = v; },
    }));
    m.append(sliderRow("Passes per frame", {
      min: K.VIDEO_ACCUMULATE_LIMITS[0], max: K.VIDEO_ACCUMULATE_LIMITS[1],
      step: 1, value: app.videoAccumulate, format: (v) => `${v}`,
      onInput: (v) => { app.videoAccumulate = v; },
    }));
    m.append(el("div", "row",
      "Press R while flying to record a track. Stop it and the track is " +
      "re-rendered here at full convergence — every frame clean, however " +
      "long that takes — and downloaded as a video."));
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
    m.append(item("Cancel", null, onCancel));
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
    m.append(item("Cancel", null, onCancel));
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
