// Viewer methods (input). Keys, mouse and the pointer lock; pause and resume.

"use strict";

import * as K from "../constants.js";

// Every key _onKeyDown's switch answers, and nothing else. It is the set the
// tutorial refuses while a chapter is running: each of these opens a panel or
// cycles a mode, and a walkthrough that has just told the reader to press
// Space cannot survive being somewhere else. Escape is deliberately absent —
// it has its own rule (Tutorial.interceptEscape), because one flight step
// asks for it.
const TUTORIAL_REFUSED = new Set(
  ["Tab", "f", "F3", "`", "b", "i", "m", "r", "k", "p", "F12"]);

export const input = {
  _bindInput() {
    const canvas = this.canvas;
    // Every listener below outlives its own statement, and half of them are on
    // `document`, which outlives the viewer. Registered against one signal so
    // dispose() can take them all off in a line — otherwise a viewer that has
    // been left behind is still reachable from the document, and so is its
    // scene, its renderer, and its device.
    const { signal } = this._listeners;

    canvas.addEventListener("click", (e) => {
      if (this.ui.isOpen) return;
      if (this.minimapMode === "full" && !this.captured && this.minimap) {
        // The canvas backbuffer and its CSS box differ by the render scale;
        // the map rect lives in backbuffer pixels.
        const hit = this.minimap.worldXYFromPixel(
          e.offsetX * (canvas.width / canvas.clientWidth),
          e.offsetY * (canvas.height / canvas.clientHeight),
          this.scene, this.camera);
        if (hit) {
          // An absolute reposition, so the camera re-derives its surface
          // frame too — the map names a place in the tile, and teleport's
          // fold against the static offset is exactly the place named.
          this.camera.teleport(hit[0], hit[1]);
          this._wake("minimap travel");
          return;
        }
        // Clicking off the map is done with it: back to the corner, flying.
        this.setMinimapMode("corner");
        return;
      }
      if (!this.captured) this._requestCapture();
    }, { signal });

    document.addEventListener("pointerlockchange", () => {
      this._wake("pointer lock");
      const wasCaptured = this.captured;
      this.captured = document.pointerLockElement === canvas;
      // The jump to the window centre arrives as one enormous movement event.
      // Swallowing it is the difference between taking the mouse and having
      // the view snap somewhere random.
      this._discardNextPointerMove = true;
      this._syncChrome();
      if (this.captured) {
        // Holding the pointer IS flying — enforced here as an invariant
        // rather than trusted to every path that can ask for the lock.
        // Without it, anything that leaves "paused, menu closed, mouse free"
        // (a screenshot closes the menu with no resume;
        // app switching can drop and re-grant the lock) lets the next click
        // capture the pointer while paused: the view still pans, because
        // look() is deliberately not gated on pause, but WASD, the bird's
        // clock and the recorder are all frozen.
        if (this.paused) {
          this.ui.close();
          this.paused = false;
          this._lastTime = performance.now();
          this._syncChrome();
        }
        this.tutorial?.onInput("capture");
        return;
      }

      this.camera.keys.clear();
      // Escape releases the pointer lock in the browser itself, and Firefox
      // ALSO delivers the keydown. Handling both turned one press into
      // pause-then-resume — the menu appeared and vanished in a frame. So the
      // lock loss is the single source of truth for pausing, and the keydown
      // that caused it is ignored for a moment afterwards.
      this._lockLostAt = performance.now();
      if (wasCaptured && !this._tabRelease && !this.paused) this.pause();
      this._tabRelease = false;
    }, { signal });

    // Firefox refuses a re-lock for about a second after the user pressed
    // Escape, so resuming cannot rely on it. Say what to do instead.
    document.addEventListener("pointerlockerror", () => {
      if (!this.paused) this.ui.say("Click to look around.", 3);
    }, { signal });

    document.addEventListener("mousemove", (e) => {
      if (!this.captured) return;
      if (this._discardNextPointerMove) {
        this._discardNextPointerMove = false;
        return;
      }
      this.camera.look(e.movementX, e.movementY);
      this._wake("mouse look");
    }, { signal });

    document.addEventListener("wheel", (e) => {
      if (!this.captured) return;
      this.camera.scrollSpeed(e.deltaY, performance.now() / 1000);
      this._wake("wheel");
      this.tutorial?.onInput("wheel");
    }, { passive: true, signal });

    document.addEventListener("keydown", (e) => this._onKeyDown(e), { signal });
    document.addEventListener("keyup", (e) => {
      this.camera.keys.delete(e.key.toLowerCase());
      this._wake("keyup");
    }, { signal });

    window.addEventListener("blur", () => {
      this.camera.keys.clear();
      this._wake("blur");
    }, { signal });

    // The canvas's size check lives inside _frame, which does not run while
    // the loop is asleep — so the loop has to be woken from outside it. One
    // observer for the CSS box (window resize, entering fullscreen, the
    // sidebar of a devtools pane opening) and one media query for the device
    // pixel ratio (dragging the window to a display with different scaling,
    // or changing the browser's zoom), which fires no resize at all.
    this._resizeObserver = new ResizeObserver(() => this._wake("resize"));
    this._resizeObserver.observe(this.canvas);
    this._watchDevicePixelRatio(signal);
  },

  /**
   * There is no devicePixelRatio event. The idiom is a media query pinned to
   * the current value, which stops matching the moment it changes; it is
   * one-shot, so it re-arms itself against the new value.
   */
  _watchDevicePixelRatio(signal) {
    const arm = () => {
      if (this._disposed) return;
      const query = window.matchMedia(
        `(resolution: ${window.devicePixelRatio}dppx)`);
      query.addEventListener("change", () => {
        this._wake("devicePixelRatio");
        arm();
      }, { once: true, signal });
    };
    arm();
  },

  /**
   * Keys are only for things done in flight. Everything reachable from the
   * menu is reachable by clicking it, so nothing here needs a second binding.
   */
  _onKeyDown(e) {
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    // Before the switch, and unconditionally: every key here either moves the
    // camera or changes what is drawn, and the ones that do neither cost one
    // frame to find that out.
    this._wake("keydown");

    // While the walkthrough runs, the flight is the walkthrough's. Every key
    // in the switch below either opens a panel the tutorial is not expecting
    // (P/F12 onto capture, R's track panel once a recording stops) or cycles
    // state a later prompt was written against, and the tutorial's menu is
    // already inert to clicks — a shortcut past it would strand the reader
    // with only Skip live. Refused at the source, with the
    // prompt's own "not yet" note; the flight keys below the switch — WASD,
    // Space, Shift — and the wheel are untouched, as is every key here the
    // moment the tutorial ends.
    if (this.tutorial?.active && TUTORIAL_REFUSED.has(key)) {
      e.preventDefault();
      this.tutorial.refuse();
      return;
    }

    if (key === "Escape") {
      // Escape reaches us by three routes that can all fire for one physical
      // press: the browser releasing the pointer lock, the keydown itself,
      // and auto-repeat if the key is held a moment. Any two of them in
      // sequence read as close-then-open, which is the menu "popping right
      // back up". So: no repeats, and one Escape action per 350 ms whatever
      // the source.
      // A video render owns the GPU for minutes. Escape is the way out of it,
      // and must not also open the menu on top of the progress bar.
      if (this._capturing) { this._videoAbort = true; return; }
      const now = performance.now();
      if (e.repeat) return;
      if (now - (this._lockLostAt ?? -1e9) < 400) return;
      if (now - (this._lastEscapeAt ?? -1e9) < 350) return;
      this._lastEscapeAt = now;
      // With the menu up and a walkthrough running, the menu belongs to the
      // walkthrough — closing it by key is the same "not yet" as clicking one
      // of its rows, and the tutorial says so. It swallows nothing while the
      // menu is closed: Escape from the flight still pauses, which is what a
      // flight step's own interruption rule is written against.
      if (this.tutorial?.interceptEscape()) return;
      if (this.ui.isOpen) this.ui.back(false);
      else this.pause();
      return;
    }
    if (this.ui.isOpen) return;

    switch (key) {
      case "Tab":
        e.preventDefault();
        if (this.captured) {
          this._tabRelease = true;
          document.exitPointerLock();
        } else {
          this._requestCapture();
        }
        return;
      case "f": this.toggleFullscreen(); return;
      // One binding, two keys. F-keys are Fn-gated on most laptop keyboards,
      // so an F-key alone is a shortcut half the machines cannot press.
      case "F3":
      case "`": e.preventDefault(); this.ui.cycleStats(); return;
      case "b": this.cycleFlyer(); return;
      // Ice detection is a research control too, and it is the only one whose
      // key can start a load — so the gate is here, before anything is asked
      // of the file.
      case "i": if (this.ui.allows("ice")) this.toggleIceMode(); return;
      case "m": this.toggleMinimap(); return;
      case "r": this.toggleTrackRecording(); return;
      // Quality is the research mode's business; in basic the tier is
      // whatever the probe measured, and this key would move it.
      case "k": if (this.ui.allows("quality")) this.toggleLightCache(); return;
      // Through pause(), not ui.open() directly: the dialog needs the mouse,
      // so the pointer must be released first — opening it under a held lock
      // leaves the camera turning behind an unusable panel.
      //
      // P is the binding; F12 stays a silent alias because it was the one
      // people learned, and because the browser's own F12 is a devtools key
      // this preventDefault is worth keeping in front of.
      case "p":
      case "F12": e.preventDefault(); this.pause("capture"); return;
      default: break;
    }
    if ("wasdc ".includes(key)) this.camera.keys.add(key);
    if (key === "Shift") this.camera.keys.add("shift");
    this.tutorial?.onInput("key", key);
  },

  /**
   * Keep the on-screen chrome consistent with what the mouse is doing.
   *
   * The toolbar is for when the mouse is free but the menu is closed — after
   * Tab, or before the first click. While flying there is nothing to click,
   * and while the menu is open the menu has everything.
   */
  _syncChrome() {
    const viewer = this.canvas.parentElement;
    viewer.classList.toggle("captured", this.captured);
    viewer.classList.toggle("menu-open", Boolean(this.ui?.isOpen));
  },

  pause(panel = "main") {
    this.paused = true;
    this.camera.keys.clear();
    if (this.captured) document.exitPointerLock();
    this.ui.open(panel);
    this._syncChrome();
    this.tutorial?.onInput("menu", panel);
    // Pausing does not put the loop to sleep and resuming does not wake it:
    // sleep is decided purely by whether the picture is finished. This wake
    // is only here because pausing stops the bird, which changes what the
    // next frame has to draw.
    this._wake("pause");
  },

  /**
   * Ask for the pointer once, quietly. Chromium returns a promise that
   * rejects inside the post-Escape cooldown; the pointerlockerror listener
   * already tells the user what to do, so the rejection itself is noise.
   */
  _requestCapture() {
    const p = this.canvas.requestPointerLock?.();
    if (p?.catch) p.catch(() => {});
  },

  resume({ capture = true } = {}) {
    this.ui.close();
    this.paused = false;
    // A fullscreen map under a re-taken pointer is a screen you can't leave.
    if (this.minimapMode === "full") this.minimapMode = "corner";
    this._lastTime = performance.now();
    this._syncChrome();
    if (capture) this._requestCapture();
    else this.ui.say("Click to look around.", 2);
    this._wake("resume");
  },

  toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  },

  /**
   * Put the camera where the tutorial opens.
   *
   * On the sea: down on the water, looking at the horizon, at whatever x/y
   * the loaded view had, so it is still this field's own place. Low is the
   * point: the first prompt asks for the mouse, and a beginner who moves it
   * wants the sea and the sky to swing past — a start high above the field
   * looks the same whichever way it is turned. In the city: the start
   * street's canyon (CITY_TUTORIAL_CAMERA), lit towers on both sides.
   */
  tutorialSpawn() {
    if (this.scene.city) {
      // A city camera always carries its tile-frame position; without one
      // there is no street to put it in.
      const s = this.camera.surfacePosition;
      if (!s) {
        throw new Error("tutorialSpawn: a city scene's camera has no "
                        + "surfacePosition");
      }
      // Reached through the camera's one move path, so the cloud and tile
      // frames move together; by jump, not displace, because this is an
      // absolute reposition (a replay from the menu puts it mid-flight) and
      // the collider must not sweep it as one frame of flight from wherever
      // the player was, stopping at the first wall (camera.js jump).
      const pose = K.CITY_TUTORIAL_CAMERA;
      const [tx, ty, tz] = pose.positionCityM;
      this.camera.jump(tx - s[0], ty - s[1], tz - this.camera.position[2]);
      this.camera.azimuth = pose.azimuth;
      this.camera.elevation = pose.elevation;
    } else {
      this.camera.position[2] = K.OCEAN_FLOOR_MARGIN_M * 4.0;
      this.camera.elevation = 0.0;
    }
    // The look was just written absolutely: no shake is on it any more.
    this.camera.shake = [0.0, 0.0];
    this.camera.constrain();
    this._wake("tutorial spawn");
  },
};
