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

/**
 * The ghost keycaps under the WASD prompt.
 *
 * Each carries the direction it flies as well as its letter: the shape of the
 * cluster says "these four go together" but not which way any of them goes,
 * and the arrow is how everyone who has not played this kind of game already
 * knows reads a key like this.
 */
function keycaps() {
  const wrap = el("div", "keycaps");
  const arrows = { W: "↑", A: "←", S: "↓", D: "→" };
  for (const row of [["W"], ["A", "S", "D"]]) {
    const line = el("div", "keycap-row");
    for (const cap of row) {
      const key = el("span", "keycap");
      key.append(el("span", "cap-key", cap), el("span", "cap-arrow", arrows[cap]));
      line.append(key);
    }
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
    text: "Click once and move the mouse to look around. " +
          "Escape returns the mouse.",
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
  {
    menuKey: "controls",
    text: "View keyboard shortcuts and flight controls.",
  },
  {
    // Two modes, one sentence each: a list, because that is what it is. Read
    // as a paragraph they ran together into one long claim about "modes".
    //
    // Only the first is a switch on the toggle this step points at. The
    // second says where it is instead, because a reader who has just been
    // shown a toggle will look for it there and it is not going to be there.
    menuKey: "mode",
    text: "Try other modes:",
    bullets: [
      "Research mode allows more visual control, more cloud volumes " +
      "including custom uploads, and false color diagnostics.",
      "Cyberpunk mode flies these same demos at night over a multifractal " +
      "city. It is chosen from the start page.",
    ],
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
  // Second, not last: bringing your own field is the reason to be in research
  // mode at all, and a walkthrough that reaches it after two rows about how
  // the picture is drawn buries the offer (Thomas, 2026-08-22).
  {
    menuKey: "open",
    text: "Supply your own cloud volume in a NetCDF file. Liquid and ice " +
          "variables are auto-detected and converted to extinction fields.",
  },
  {
    menuKey: "terminal",
    text: "Give a terminal command to render this view programmatically " +
          "using the cloudyview CLI.",
  },
  {
    menuKey: "ice",
    text: "Activate false colors indicating cloud ice fraction (magenta for " +
          "liquid, cyan for ice).",
  },
  { menuKey: "periodic", text: "Wrap the domain horizontally." },
  {
    menuKey: "leave",
    text: "From the start page, fly other cloud fields from STEAM or common " +
          "LES cloud models.",
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
    this._forceResearch = false;   // an explicit replay asked for the coda
    this._reserved = null;      // the room the menu is keeping for the box
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
    this.list = el("ul", "tutorial-list");
    this.graphic = el("div", "tutorial-graphic");
    this.note = el("div", "tutorial-note",
                   "Try this after finishing the tutorial.");
    this.note.hidden = true;
    this.actions = el("div", "tutorial-actions");
    this.ok = el("button", "tutorial-ok", "OK");
    this.ok.addEventListener("click", () => this._next());
    this.skip = el("button", "tutorial-skip", "Skip tutorial");
    this.skip.addEventListener("click", () => this.skipAll());
    this.actions.append(this.ok, this.skip);
    this.box.append(this.body, this.list, this.graphic, this.note, this.actions);

    this.root.append(this.spot, this.spotLabel, this.box);

    // While a chapter is running the menu is a diagram, not a control. Every
    // menu step names a row that opens a panel, and opening one takes the
    // walkthrough somewhere it was not going: clicking Capture at the Capture
    // step left the capture panel up, the spotlight aimed at a row that was no
    // longer there, and no way on but Skip (Thomas, 2026-08-22).
    // Capture phase, so the row's own handler never runs — stopping the event
    // here stops it before the target phase.
    // Resume is the exception, and has to be: a flight step interrupted by
    // Escape (see onInput) puts this prompt under an open menu, and clicking
    // Resume is how the reader gets back to the flight the prompt is about.
    this.root.addEventListener("click", (event) => {
      if (!this.active) return;
      const menu = document.getElementById("menu");
      if (!menu || menu.hidden || !menu.contains(event.target)) return;
      if (event.target.closest(".resume")) return;
      event.preventDefault();
      event.stopPropagation();
      this.refuse();
    }, true);
  }

  // --- starting and stopping ----------------------------------------------

  /** Run the whole thing, unless it has been run (or skipped) before. */
  maybeStart() {
    if (flag(DONE_KEY)) return;
    this.replay();
  }

  /**
   * From the menu's Replay item, and from maybeStart.
   *
   * In research mode "the tutorial" includes the research coda — asking for
   * the walkthrough from a menu with the research rows in it and being shown
   * only the rows basic mode has would be answering a different question.
   * Queued rather than concatenated: RESEARCH_STEPS are filtered against the
   * menu rows that actually exist, and at this moment the menu is closed and
   * the flight has not started, so the filter would drop all of them. `end()`
   * runs it, by which point the menu chapter has the menu open.
   */
  replay() {
    this.app.ui.close();
    if (this.app.paused) this.app.resume({ capture: false });
    this.app.tutorialSpawn();
    this._pendingResearch = this.app.ui.mode === "research";
    // An explicit replay overrides "already seen once" — that flag is what
    // stops the coda running twice on its own, not a ban on asking for it.
    this._forceResearch = this._pendingResearch;
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
    // The menu keeps room for a prompt only while there is one — see
    // _dockBox and `#viewer.tutorial #menu` in viewer.css.
    this._reserved = null;
    document.documentElement.style.removeProperty("--tutorial-dock");
    this.box.style.top = "";
    this.box.classList.remove("docked");
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
    this._forceResearch = false;
    setFlag(DONE_KEY);
    setFlag(RESEARCH_DONE_KEY);
    this.end();
  }

  destroy() {
    this._pendingResearch = false;
    this._forceResearch = false;
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
    // Entering the menu chapter means the pause menu is up and showing its
    // main panel — the last flight step is cleared by exactly that, and every
    // step here dims the screen around one of that panel's rows. A pause onto
    // any other panel is a place this chapter cannot point at, so the flight
    // chapter starts over instead of advancing blind. Every source of such a
    // pause is refused while a chapter runs (TUTORIAL_REFUSED in viewer.js);
    // this is the guard that keeps the silent void unreachable rather than a
    // path anything is expected to take.
    if (this.step.menuKey && this.steps !== this._researchSteps
        && this.app.ui.panel !== "main") {
      this._restartFlight();
      return;
    }
    this._show();
  }

  _show() {
    const step = this.step;
    this.body.textContent = step.text;
    this.list.replaceChildren(
      ...(step.bullets ?? []).map((line) => el("li", null, line)));
    this.list.hidden = !step.bullets;
    // The note answers one click, not the whole chapter.
    this.note.hidden = true;
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
    if (!step) {
      this._setDocked(false);
      this.spot.hidden = true;
      this.spotLabel.hidden = true;
      return;
    }
    const track = () => {
      this._raf = null;
      if (!this.active) return;
      // Whenever the menu is up the box hangs under it — see _dockBox. That is
      // every menu step, and also a flight step during the restart in onInput,
      // where the prompt would otherwise be over the menu the reader has to
      // click. Asked per frame rather than per step, because the menu opens
      // and closes under a flight step without the step changing.
      // A spotlight is not by itself a reason to move: the speed step points
      // at the bottom-right corner with no menu open, so its prompt stays
      // where every other flight prompt is — the middle.
      const docked = Boolean(step.menuKey) || this._menuOpen();
      this._setDocked(docked);
      const target = selector ? document.querySelector(selector) : null;
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
          // measured and pulled left of the window edge instead. Its height is
          // measured too rather than assumed: the caption is set large enough
          // to be noticed now, and a fixed 20 px lift put it over the ring.
          const height = this.spotLabel.offsetHeight;
          const below = r.bottom + pad + 8;
          const room = below + height + 4 < window.innerHeight;
          this.spotLabel.style.top = room
            ? `${below}px` : `${r.top - pad - height - 6}px`;
          const width = this.spotLabel.offsetWidth;
          this.spotLabel.style.left =
            `${Math.max(8, Math.min(r.left - pad,
                                    window.innerWidth - 8 - width))}px`;
        } else {
          this.spotLabel.hidden = true;
        }
      }
      // A step with neither a spotlight nor a menu under it has nothing left
      // to watch, and holding a frame callback open for it would keep the
      // window ticking for the rest of the step.
      if (!selector && !docked) return;
      this._raf = requestAnimationFrame(track);
    };
    track();
  }

  /**
   * Escape, offered to the tutorial before the viewer acts on it.
   *
   * The click interceptor in _build makes the menu inert while a chapter runs,
   * and Escape is a way of clicking Resume without the button: it closed the
   * menu out from under the step that was pointing at it, silently. Same
   * answer as a row: not yet, and here is the line that says so.
   *
   * Only with the menu actually up. Escape from the flight has to keep
   * pausing — it is the browser's own way out of a pointer lock, and a flight
   * step's interruption rule (see onInput) is written against that pause.
   *
   * @returns {boolean} true if the tutorial swallowed the key.
   */
  interceptEscape() {
    if (!this.active || !this._menuOpen()) return false;
    return this.refuse();
  }

  /**
   * "Not yet" — the one answer to anything the running step did not ask for.
   *
   * Shared by the click interceptor, interceptEscape and the viewer's keyboard
   * refusal (see TUTORIAL_REFUSED in viewer.js) so that a shortcut, a row and
   * Escape all say the same sentence in the same place.
   *
   * @returns {boolean} true if a chapter is running and was told to say so.
   */
  refuse() {
    if (!this.active) return false;
    this.note.hidden = false;
    return true;
  }

  /** Is the pause menu on screen? */
  _menuOpen() {
    const menu = document.getElementById("menu");
    return Boolean(menu) && !menu.hidden;
  }

  /** The box hangs under the menu, or sits in the middle of the window. */
  _setDocked(on) {
    this.box.classList.toggle("docked", on);
    if (on) { this._dockBox(); return; }
    if (this._reserved === null) return;
    this._reserved = null;
    this.box.style.top = "";
    document.documentElement.style.removeProperty("--tutorial-dock");
  }

  /**
   * Put the box under the menu rather than over it.
   *
   * The menu chapter talks about rows of the menu, so it cannot cover it: the
   * box used to sit at the top of the window, which is where a tall menu's own
   * top edge is. Under the menu is the one place that is out of the way and
   * still obviously attached to what is being pointed at.
   *
   * The menu keeps room for it — see `#viewer.tutorial #menu` in viewer.css —
   * so this normally lands as written; the clamp is for a window short enough
   * that the reservation is not enough, where the last resort is a box the
   * reader can read rather than one tidily off the bottom edge.
   */
  _dockBox() {
    const gap = 14;
    const height = this.box.offsetHeight;
    // Reserved before the menu is measured, because the menu's max-height is
    // a function of it. Written only when it changes: this runs every frame,
    // and a custom property set to the value it already has still costs a
    // style recalculation of everything under it.
    const reserve = `${Math.round(height + gap + 12)}px`;
    if (this._reserved !== reserve) {
      this._reserved = reserve;
      document.documentElement.style.setProperty("--tutorial-dock", reserve);
    }
    const menu = document.getElementById("menu");
    const bottom = menu && !menu.hidden
      ? menu.getBoundingClientRect().bottom : 24 - gap;
    const top = Math.min(bottom + gap, window.innerHeight - height - 12);
    this.box.style.top = `${Math.max(12, top)}px`;
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
    if (step.when(kind, detail)) { this._next(); return; }
    // The flight chapter, interrupted. Every one of its steps is cleared by
    // flying, and a paused flight cannot fly: pressing Escape at the WASD
    // prompt — or switching apps, which drops the pointer lock and pauses the
    // same way — left a step up that no key could satisfy and no button could
    // advance, so the only way on was Skip (Thomas, 2026-08-22).
    // The chapter starts over rather than stepping forward. Its first prompt
    // is "click once and move the mouse", which is exactly what the reader is
    // about to do to get back into the flight, and Resume then clears it.
    if (kind === "menu") this._restartFlight();
  }

  /** The first prompt of the flight chapter again, with the reason showing. */
  _restartFlight() {
    this.index = 0;
    this._unpinSpeed();
    this._show();
    // _show clears the note, so this says it after: the interruption is the
    // same "not yet" as clicking a row of the menu, and a walkthrough that
    // silently rewound would look broken rather than deliberate.
    this.note.hidden = false;
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
      // The flag is what keeps the coda from running twice on its own; an
      // explicit replay asked for it, so it is cleared rather than obeyed.
      const forced = this._forceResearch;
      this._forceResearch = false;
      if ((flag(RESEARCH_DONE_KEY) && !forced) || this.active) return;
      const steps = RESEARCH_STEPS.filter(
        (s) => document.querySelector(`#menu [data-menu-key="${s.menuKey}"]`));
      if (!steps.length) return;   // nothing this field and mode can show
      this._researchSteps = steps;
      this._run(steps);
    }, 0);
  }
}
