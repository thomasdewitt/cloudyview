// Camera basis, the relative<->world coordinate convention, and the flight
// controller. Ports cloudyview/camera.py plus FlyThroughApp's movement half.

"use strict";

import {
  MOUSE_SENS, DEFAULT_SPEED, SPEED_LIMITS, SPEED_WHEEL_FACTOR,
  ELEVATION_LIMITS, OCEAN_FLOOR_MARGIN_M, FOV_LIMITS, DEFAULT_CAMERA,
} from "./constants.js";
import { directionFromAzimuthElevation, mod360 } from "./spectral.js";

const DEG = Math.PI / 180.0;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Orthonormal camera basis: [forward, right, up].
 *
 * `right` is the closed form cos/-sin rather than a cross product with world
 * up. That is deliberate and load-bearing: the cross-product construction
 * degenerates within a couple of degrees of straight up or down and snaps the
 * horizon over as you fly through vertical. Do not "simplify" it.
 */
export function cameraBasis(azimuthDeg, elevationDeg) {
  const forward = directionFromAzimuthElevation(azimuthDeg, elevationDeg);
  const az = azimuthDeg * DEG;
  const right = [Math.cos(az), -Math.sin(az), 0.0];
  const up = [
    right[1] * forward[2] - right[2] * forward[1],
    right[2] * forward[0] - right[0] * forward[2],
    right[0] * forward[1] - right[1] * forward[0],
  ];
  const n = Math.hypot(up[0], up[1], up[2]);
  return [forward, right, [up[0] / n, up[1] / n, up[2] / n]];
}

/**
 * Relative camera coordinates to absolute metres.
 *
 * x and y span the domain AABB over [-1, 1]. z does NOT: it is anchored to the
 * physical surface, so rel z = -1 is sea level and rel z = +1 is the top of
 * the data. An elevated domain therefore keeps its real altitude. Treating z
 * like x and y is the classic way to put the camera underground.
 */
export function cameraWorldOrigin(rel, bmin, bmax) {
  return [
    bmin[0] + (rel[0] + 1.0) * 0.5 * (bmax[0] - bmin[0]),
    bmin[1] + (rel[1] + 1.0) * 0.5 * (bmax[1] - bmin[1]),
    (rel[2] + 1.0) * 0.5 * bmax[2],
  ];
}

/** The inverse — what the reproduction commands and metadata record. */
export function worldToRelative(pos, bmin, bmax) {
  return [
    2.0 * (pos[0] - bmin[0]) / (bmax[0] - bmin[0]) - 1.0,
    2.0 * (pos[1] - bmin[1]) / (bmax[1] - bmin[1]) - 1.0,
    2.0 * pos[2] / bmax[2] - 1.0,
  ];
}

/**
 * Flight state and the controls that move it. World metres and meteorological
 * angles throughout; relative coordinates appear only at the edges.
 */
export class FlightCamera {
  constructor(bmin, bmax, { periodic = true } = {}) {
    this.bmin = bmin;
    this.bmax = bmax;
    this.periodic = periodic;
    this.position = cameraWorldOrigin(DEFAULT_CAMERA.position, bmin, bmax);
    this.azimuth = DEFAULT_CAMERA.azimuth;
    this.elevation = DEFAULT_CAMERA.elevation;
    this.fov = DEFAULT_CAMERA.fov;
    this.speed = DEFAULT_SPEED;
    this.keys = new Set();
    this.speedFlashUntil = 0;
    this.constrain();
  }

  reset() {
    this.position = cameraWorldOrigin(DEFAULT_CAMERA.position, this.bmin, this.bmax);
    this.azimuth = DEFAULT_CAMERA.azimuth;
    this.elevation = DEFAULT_CAMERA.elevation;
    this.speed = DEFAULT_SPEED;
    this.constrain();
  }

  /** A tuple that changes exactly when the view does. */
  signature() {
    return `${this.position[0]},${this.position[1]},${this.position[2]},` +
           `${this.azimuth},${this.elevation},${this.fov}`;
  }

  look(dx, dy) {
    this.azimuth = mod360(this.azimuth + dx * MOUSE_SENS);
    this.elevation = clamp(
      this.elevation - dy * MOUSE_SENS, ELEVATION_LIMITS[0], ELEVATION_LIMITS[1]);
  }

  /** Wheel notches scale flight speed geometrically. */
  scrollSpeed(deltaY, now) {
    const notches = -deltaY / 100.0;
    this.speed = clamp(
      this.speed * Math.pow(SPEED_WHEEL_FACTOR, notches),
      SPEED_LIMITS[0], SPEED_LIMITS[1]);
    this.speedFlashUntil = now + 1.5;
  }

  setFov(fov) {
    this.fov = clamp(fov, FOV_LIMITS[0], FOV_LIMITS[1]);
  }

  move(dt) {
    if (this.keys.size === 0) return false;
    const [f, r] = cameraBasis(this.azimuth, this.elevation);
    const d = this.speed * Math.min(dt, 0.1);
    const p = this.position;
    const k = this.keys;
    if (k.has("w")) { p[0] += f[0] * d; p[1] += f[1] * d; p[2] += f[2] * d; }
    if (k.has("s")) { p[0] -= f[0] * d; p[1] -= f[1] * d; p[2] -= f[2] * d; }
    if (k.has("a")) { p[0] -= r[0] * d; p[1] -= r[1] * d; }
    if (k.has("d")) { p[0] += r[0] * d; p[1] += r[1] * d; }
    if (k.has(" ")) p[2] += d;
    if (k.has("shift") || k.has("c")) p[2] -= d;
    this.constrain();
    return true;
  }

  /**
   * Keep the camera above the ocean, and — in a periodic domain — inside the
   * box by wrapping rather than stopping. Crossing a lateral face puts you at
   * the opposite one; flight is endless.
   */
  constrain() {
    const p = this.position;
    if (this.periodic) {
      for (const i of [0, 1]) {
        const extent = this.bmax[i] - this.bmin[i];
        p[i] = this.bmin[i] +
          (((p[i] - this.bmin[i]) % extent) + extent) % extent;
      }
    }
    p[2] = Math.max(p[2], OCEAN_FLOOR_MARGIN_M);
  }

  relativePosition() {
    return worldToRelative(this.position, this.bmin, this.bmax);
  }
}
