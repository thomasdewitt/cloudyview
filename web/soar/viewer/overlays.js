// Viewer methods (overlays). The minimap's mode and which flyer is up.

"use strict";

import { Bird } from "../bird.js";
import { Dart } from "../dart.js";

/**
 * The flyers, in the order B cycles them — and so the first is the default.
 *
 * One table rather than an order here, a name there and a build loop
 * somewhere else: three lists that have to agree is three chances for them
 * not to, and the only thing any of them encodes is "which flyers, in what
 * order".
 *
 * The dart leads. It is the one that reads at a glance — pale against sky
 * where the swift is a dark speck, and its phugoid says "thrown object"
 * before you have looked at the shape — so it is the better first impression
 * of a renderer someone has just opened.
 */
export const FLYERS = [
  { kind: "dart", Species: Dart, name: "paper dart", source: "dartSource" },
  { kind: "bird", Species: Bird, name: "swift", source: "birdSource" },
];

/** Release both flyers, whichever of them got built. */
export function destroyFlyers(flyers) {
  for (const flyer of Object.values(flyers ?? {})) flyer?.destroy();
}

export const overlays = {
  toggleMinimap() {
    if (!this.minimap) {
      this.ui.say(
        this._minimapProblem
          ? `No minimap for this field: ${this._minimapProblem}`
          : "There is no minimap for this field.", 5);
      return;
    }
    const order = ["corner", "full", "off"];
    this.setMinimapMode(order[(order.indexOf(this.minimapMode) + 1) % 3]);
  },

  setMinimapMode(mode) {
    const wasFull = this.minimapMode === "full";
    this.minimapMode = mode;
    this._wake("minimap mode");
    if (mode === "full") {
      // The map needs a visible cursor to be clickable.
      if (this.captured) {
        this._tabRelease = true;
        document.exitPointerLock();
      }
      this.ui.say("Minimap fullscreen — click to travel, M to dismiss.", 4);
    } else {
      this.ui.say(`Minimap ${mode === "corner" ? "on" : "off"}.`);
      if (wasFull && !this.paused && !this.ui.isOpen && !this.captured) {
        this._requestCapture();
      }
    }
  },

  /**
   * The flyer currently up, or null if there is none for this field.
   *
   * Everything downstream — the frame loop, the stills, the stats readout —
   * only ever wants the active one, and asking through a getter means there
   * is no second copy of "which flyer" to be assigned, forgotten, and left
   * pointing at a destroyed object.
   */
  get flyer() {
    return this.flyers?.[this.flyerKind] ?? null;
  },

  /** The flyers that built, in cycle order. */
  get availableFlyers() {
    return FLYERS.map((f) => f.kind).filter((kind) => this.flyers?.[kind]);
  },

  /** What is up, for the menu row and the stats readout: a name, or "off". */
  get flyerLabel() {
    return this.flyer && this.flyerEnabled
      ? FLYERS.find((f) => f.kind === this.flyerKind).name : "off";
  },

  /**
   * B: off -> paper dart -> swift -> off, per FLYERS.
   *
   * One key for both the choice and the switch, because there are two of them
   * and there is not going to be a third. Turning one on gives the loop
   * something to animate again; turning them off lets a converged view sleep
   * outright, so both directions have to wake it.
   */
  cycleFlyer() {
    // Only the ones that actually built. They share a pass, so in practice
    // either both are here or neither is — but cycling onto a flyer that
    // failed to compile would draw nothing and say nothing, which is the one
    // outcome worth ruling out by construction.
    const available = this.availableFlyers;
    if (!available.length) {
      this.ui.say(
        this._flyerProblem
          ? `No flyer for this field: ${this._flyerProblem}`
          : "There is no flyer for this field.", 5);
      return;
    }
    const at = available.indexOf(this.flyerKind);
    if (!this.flyerEnabled) {
      this.flyerEnabled = true;
      this.flyerKind = available[0];
    } else if (at < 0 || at === available.length - 1) {
      this.flyerEnabled = false;
    } else {
      this.flyerKind = available[at + 1];
    }
    const name = FLYERS.find((f) => f.kind === this.flyerKind).name;
    this.ui.say(this.flyerEnabled
      ? `${name[0].toUpperCase()}${name.slice(1)}.`
      : "No flyer.");
    this._wake("flyer");
  },
};
