// Viewer methods (zen). Zen mode: the flight flies itself (../zen.js), in
// fullscreen, with the map away, and the controls do nothing but leave it.

"use strict";

import { CityZen, CloudZen } from "../zen.js";

export const zen = {
  /** Whether zen mode is on. It stays on, held still, under a pause. */
  get zenOn() { return this._zen !== null; },

  /**
   * Hand the flight to zen from where the camera is.
   *
   * What zen changes besides the flight is kept, so that leaving puts it
   * back: the minimap's mode, and fullscreen if zen is what entered it.
   * The flyer is left as it is.
   */
  startZen() {
    let flight;
    try {
      flight = this.scene.city
        ? new CityZen(this.camera, this.collider.city)
        : new CloudZen(this.camera);
    } catch (err) {
      this.ui.say(`Zen mode cannot start here: ${err.message}`, 5);
      return;
    }
    const enteredFullscreen = !document.fullscreenElement;
    this._zen = { flight, minimapMode: this.minimapMode, enteredFullscreen };
    this.camera.keys.clear();
    this.minimapMode = "off";
    if (enteredFullscreen) {
      // Refused when the browser does not count this as the user's own act;
      // zen then flies in the window.
      document.documentElement.requestFullscreen().catch(() => {});
    }
    // A paused flight with the menu closed (after a still) starts flying.
    if (this.paused) {
      this.paused = false;
      this._lastTime = performance.now();
    }
    this.canvas.parentElement.classList.add("zen");
    this.ui.say("Zen mode. Z or Esc to leave.", 4);
    this.tutorial?.onInput("zen");
    this._wake("zen");
  },

  /** Give the flight back, where zen has taken it. A no-op when zen is off. */
  endZen() {
    const was = this._zen;
    if (!was) return;
    this._zen = null;
    this.minimapMode = was.minimapMode;
    if (was.enteredFullscreen && document.fullscreenElement) {
      document.exitFullscreen();
    }
    this.canvas.parentElement.classList.remove("zen");
    this._wake("zen");
  },
};
