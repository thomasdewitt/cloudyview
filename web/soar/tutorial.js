// The first flight: a prompt at a time, cleared by doing the thing.
//
// Two chapters and a coda. The flight chapter is do-to-advance — each prompt
// names one control and disappears the moment that control is used, so the
// tutorial is never a wall of text in front of a view and never asks to be
// read twice. The menu chapter cannot work that way (clicking Capture opens
// Capture, which is not where the walkthrough is going next), so those steps
// dim the screen around the item being named and wait for OK. The research
// coda runs once, the first time research mode is switched on.
//
// Nothing here reaches into the viewer's input handling: viewer.js notifies
// `onInput(kind, detail)` from the handlers it already has, and this decides
// whether that was the thing the current step asked for.

"use strict";

export const DONE_KEY = "soar.tutorialDone";
// Not part of the shared contract, and needed for the same reason that one
// is: "the first time research mode is activated" is a fact that has to
// outlive the session it happened in.
export const RESEARCH_DONE_KEY = "soar.tutorialResearchDone";

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};

/** localStorage, which a browser is allowed to refuse outright. */
function flag(key) {
  try {
    return localStorage.getItem(key) === "1";
  } catch {
    // Storage denied (private mode, file://). The tutorial then runs every
    // session, which is the harmless direction: it is skippable in one click
    // and it is the alternative to never running at all.
    return false;
  }
}

function setFlag(key) {
  try {
    localStorage.setItem(key, "1");
  } catch {
    // See flag(). Nothing to do, and nothing worth failing a flight over.
  }
}

/** The ghost keycaps under the WASD prompt. */
function keycaps() {
  const wrap = el("div", "keycaps");
  for (const row of [["W"], ["A", "S", "D"]]) {
    const line = el("div", "keycap-row");
    for (const cap of row) line.append(el("span", "keycap", cap));
    wrap.append(line);
  }
  return wrap;
}

/**
 * The flight chapter. `when` is asked of every input the viewer reports; the
 * step clears the first time it answers true.
 */
const FLIGHT_STEPS = [
  {
    text: "Click once to take the mouse. Then just move it to look around. " +
          "Esc gives it back.",
    when: (kind) => kind === "capture",
  },
  {
    text: "Use WASD keys to move horizontally",
    graphic: keycaps,
    when: (kind, key) => kind === "key" && "wasd".includes(key),
  },
  {
    text: "Space to fly up",
    when: (kind, key) => kind === "key" && key === " ",
  },
  {
    text: "Left Shift to fly down",
    when: (kind, key) => kind === "key" && key === "Shift",
  },
  {
    text: "Scroll to change speed",
    // The one flight step with a spotlight: the readout it points at is in
    // the corner and would otherwise never be noticed.
    spot: "#speed",
    spotText: "Speed is displayed here",
    pinSpeed: true,
    when: (kind) => kind === "wheel",
  },
  {
    text: "Hit Esc to view the menu",
    // The pause menu specifically. P also pauses, straight onto the capture
    // panel, and the next step points at a row of the main one.
    when: (kind, panel) => kind === "menu" && panel === "main",
  },
];

/** The menu chapter: OK-advanced, each one dimming all but its own item. */
const MENU_STEPS = [
  {
    menuKey: "capture",
    text: "Capture a still of the current view (shortcut: P). Stills are " +
          "rendered at higher quality. Flight videos may be saved by " +
          "pressing R during flight to start the recording and R again to " +
          "stop.",
  },
  { menuKey: "sun", text: "Sun location can be changed." },
  { menuKey: "controls", text: "View flight controls." },
  {
    menuKey: "mode",
    text: "Try other modes. Research mode allows more visual control, more " +
          "cloud volumes including custom uploads, and false color " +
          "diagnostics. Cyberpunk mode replaces the surface with a " +
          "multifractal city.",
  },
];

/**
 * The research coda. Every step is gated on its menu item actually being
 * there — the menu is built from a per-mode allowlist and a field without a
 * nest has no nest row, so a step whose item is absent is dropped rather
 * than pointed at empty screen.
 */
const RESEARCH_STEPS = [
  {
    menuKey: "quality",
    text: "Custom control over ray marching and quality settings.",
  },
  {
    menuKey: "terminal",
    text: "Give a terminal command to render this view programmatically " +
          "using the cloudyview CLI.",
  },
  { menuKey: "periodic", text: "Wrap the domain horizontally." },
  {
    menuKey: "open",
    text: "Supply your own cloud volume in a NetCDF file. Liquid and ice " +
          "variables are auto-detected and converted to extinction fields.",
  },
];

export class Tutorial {
  /** `root` is #ui; `app` is the viewer. */
  constructor(root, app) {
    this.root = root;
    this.app = app;
    this.steps = null;          // the running chapter, or null
    this.index = 0;
    this._raf = null;
    this._pendingResearch = false;
    this._build();
  }

  get active() { return this.steps !== null; }

  _build() {
    this.spot = el("div", "tutorial-spot");
    this.spot.hidden = true;
    this.spotLabel = el("div", "tutorial-spot-label");
    this.spotLabel.hidden = true;

    this.box = el("div", "panel tutorial-box");
    this.box.hidden = true;
    this.body = el("p", "tutorial-text");
    this.graphic = el("div", "tutorial-graphic");
    this.actions = el("div", "tutorial-actions");
    this.ok = el("button", "tutorial-ok", "OK");
    this.ok.addEventListener("click", () => this._next());
    this.skip = el("button", "tutorial-skip", "Skip tutorial");
    this.skip.addEventListener("click", () => this.skipAll());
    this.actions.append(this.ok, this.skip);
    this.box.append(this.body, this.graphic, this.actions);

    this.root.append(this.spot, this.spotLabel, this.box);
  }

  // --- starting and stopping ----------------------------------------------

  /** Run the whole thing, unless it has been run (or skipped) before. */
  maybeStart() {
    if (flag(DONE_KEY)) return;
    this.replay();
  }

  /** From the menu's Replay item, and from maybeStart. */
  replay() {
    this.app.ui.close();
    if (this.app.paused) this.app.resume({ capture: false });
    this.app.tutorialSpawn();
    this._run([...FLIGHT_STEPS, ...MENU_STEPS]);
  }

  _run(steps) {
    if (!steps.length) throw new Error("a tutorial chapter with no steps");
    this.steps = steps;
    this.index = 0;
    this.app.canvas.parentElement.classList.add("tutorial");
    this._show();
  }

  /** Every way out of a chapter, including the last step's OK. */
  end() {
    this.steps = null;
    this._unpinSpeed();
    this.box.hidden = true;
    this.spot.hidden = true;
    this.spotLabel.hidden = true;
    this.app.canvas.parentElement.classList.remove("tutorial");
    if (this._raf !== null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
    if (this._pendingResearch) {
      this._pendingResearch = false;
      this._startResearch();
    }
  }

  /** The Skip affordance: this chapter, and every chapter, for good. */
  skipAll() {
    this._pendingResearch = false;
    setFlag(DONE_KEY);
    setFlag(RESEARCH_DONE_KEY);
    this.end();
  }

  destroy() {
    this._pendingResearch = false;
    clearTimeout(this._researchTimer);
    this._researchTimer = null;
    this.end();
    this.spot.remove();
    this.spotLabel.remove();
    this.box.remove();
  }

  // --- the running step ----------------------------------------------------

  get step() { return this.steps?.[this.index] ?? null; }

  _next() {
    if (!this.active) return;
    this._unpinSpeed();
    this.index += 1;
    if (this.index >= this.steps.length) {
      // Reaching the end of a chapter is the only thing that marks it done —
      // the flight and menu chapters run as one list, so this fires once.
      setFlag(this.steps === this._researchSteps ? RESEARCH_DONE_KEY : DONE_KEY);
      this.end();
      return;
    }
    this._show();
  }

  _show() {
    const step = this.step;
    this.body.textContent = step.text;
    this.graphic.replaceChildren();
    if (step.graphic) this.graphic.append(step.graphic());
    this.graphic.hidden = !step.graphic;
    // Do-to-advance steps have no button to press; there is nothing for OK to
    // mean when the prompt clears itself.
    this.ok.hidden = Boolean(step.when);
    this.box.hidden = false;
    if (step.pinSpeed) this._pinSpeed();
    this._aim();
  }

  /**
   * Point the spotlight at this step's target and keep it there.
   *
   * Re-measured every frame while a spotlight is up: the menu is rebuilt
   * wholesale by several of its own rows, the window resizes, and a rect
   * measured once would be a hole over nothing.
   */
  _aim() {
    const step = this.step;
    const selector = step?.spot
      ?? (step?.menuKey ? `#menu [data-menu-key="${step.menuKey}"]` : null);
    if (this._raf !== null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
    if (!selector) {
      this.spot.hidden = true;
      this.spotLabel.hidden = true;
      this.box.classList.remove("top");
      return;
    }
    // A spotlit step puts its box at the top of the window: the menu owns the
    // middle and the speed readout owns the bottom corner, and the box must
    // not sit on the thing it is pointing at.
    this.box.classList.add("top");
    const track = () => {
      this._raf = null;
      if (!this.active) return;
      const target = document.querySelector(selector);
      if (!target) {
        // The item is not on screen right now — a submenu is up, or the row
        // belongs to a mode that has just been switched away from. Wait for
        // it rather than dimming the screen around nothing.
        this.spot.hidden = true;
        this.spotLabel.hidden = true;
      } else {
        const r = target.getBoundingClientRect();
        const pad = 6;
        Object.assign(this.spot.style, {
          left: `${r.left - pad}px`,
          top: `${r.top - pad}px`,
          width: `${r.width + 2 * pad}px`,
          height: `${r.height + 2 * pad}px`,
        });
        this.spot.hidden = false;
        if (step.spotText) {
          this.spotLabel.textContent = step.spotText;
          this.spotLabel.hidden = false;
          // Below the ring, unless there is no room below it — the speed
          // readout is 12 px off the bottom of the window, and a label under
          // that one is a label nobody sees. The label does not wrap (a
          // two-line caption pushed back over the ring it labels), so it is
          // measured and pulled left of the window edge instead.
          const below = r.bottom + pad + 8;
          const room = below + 24 < window.innerHeight;
          this.spotLabel.style.top = room
            ? `${below}px` : `${r.top - pad - 20}px`;
          const width = this.spotLabel.offsetWidth;
          this.spotLabel.style.left =
            `${Math.max(8, Math.min(r.left - pad,
                                    window.innerWidth - 8 - width))}px`;
        } else {
          this.spotLabel.hidden = true;
        }
      }
      this._raf = requestAnimationFrame(track);
    };
    track();
  }

  /**
   * Hold the speed readout on screen for as long as the step points at it.
   *
   * The chip is a flash — a second and a half after the wheel moves — and
   * `speedFlashUntil` is the whole of that rule, so extending it is how the
   * chip is held rather than a second visibility flag for drawStats to
   * reconcile with the first.
   */
  _pinSpeed() {
    this._speedPinned = true;
    this.app.camera.speedFlashUntil = Infinity;
    this.app._wake("tutorial");
  }

  _unpinSpeed() {
    if (!this._speedPinned) return;
    this._speedPinned = false;
    // Not zero: the step is cleared by a scroll, and the flash that scroll
    // earned should still run its course.
    this.app.camera.speedFlashUntil = performance.now() / 1000 + 1.5;
    this.app._wake("tutorial");
  }

  // --- what the viewer reports ---------------------------------------------

  /** Called from viewer.js's own input handlers. See the module note. */
  onInput(kind, detail) {
    const step = this.step;
    if (!step?.when) return;
    if (step.when(kind, detail)) this._next();
  }

  /**
   * Research mode, switched on. The coda runs once — and after whatever
   * chapter is already running, because the mode toggle is the last thing the
   * menu chapter points at and clicking it there is the expected thing to do.
   */
  onModeChange(mode) {
    if (mode !== "research" || flag(RESEARCH_DONE_KEY)) return;
    if (this.active) { this._pendingResearch = true; return; }
    this._startResearch();
  }

  _startResearch() {
    // One turn late: the caller switches the mode and then rebuilds the menu,
    // so the rows these steps aim at do not exist yet. A timer rather than a
    // frame, because rAF does not fire in a tab that is not being drawn and
    // this has to be a turn, not a repaint.
    clearTimeout(this._researchTimer);
    this._researchTimer = setTimeout(() => {
      this._researchTimer = null;
      if (flag(RESEARCH_DONE_KEY) || this.active) return;
      const steps = RESEARCH_STEPS.filter(
        (s) => document.querySelector(`#menu [data-menu-key="${s.menuKey}"]`));
      if (!steps.length) return;   // nothing this field and mode can show
      this._researchSteps = steps;
      this._run(steps);
    }, 0);
  }
}
