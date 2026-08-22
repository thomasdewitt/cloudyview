// Camera basis, the relative<->world coordinate convention, and the flight
// controller. Ports cloudyview/camera.py plus FlyThroughApp's movement half.

"use strict";

import {
  MOUSE_SENS, DEFAULT_SPEED, SPEED_LIMITS, SPEED_WHEEL_FACTOR,
  VERTICAL_SPEED_FRACTION,
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
/**
 * Where flight begins.
 *
 * A demo baked since 2026-08-14 carries the camera its landing-page still was
 * taken from, so clicking a case opens on the picture that was clicked rather
 * than on a generic view of the same field. Absent is normal — older bakes
 * and any user's own file have none, and those get DEFAULT_CAMERA.
 *
 * Present but malformed is NOT normal: it means a bake wrote a camera nobody
 * can use, and quietly flying somewhere else would hide that for as long as
 * nobody happened to compare the still with the view. So it raises.
 */
function startCamera(start) {
  if (start == null) return DEFAULT_CAMERA;
  const { position, azimuth, elevation, fov } = start;
  const ok = Array.isArray(position) && position.length === 3
    && position.every((v) => Number.isFinite(v))
    && [azimuth, elevation].every((v) => Number.isFinite(v));
  if (!ok) {
    throw new Error(
      "This field carries a start camera that cannot be read " +
      `(${JSON.stringify(start)}). Re-bake it with tools/prebake_demos.py.`);
  }
  return {
    position, azimuth, elevation,
    fov: Number.isFinite(fov) ? fov : DEFAULT_CAMERA.fov,
  };
}

export class FlightCamera {
  constructor(bmin, bmax, { periodic = true, start = null,
                            wrapExtent = null,
                            // The speed this session opens at, and the one R
                            // returns to. A scene's, not a global: 1500 m/s
                            // crosses a cloud field and goes through a city
                            // block before the frame lands (see
                            // CITY_DEFAULT_SPEED).
                            speed = DEFAULT_SPEED } = {}) {
    this.bmin = bmin;
    this.bmax = bmax;
    this.periodic = periodic;
    // Fold distance per lateral axis, when it is not the box's own extent.
    // The night city needs one: the shader wraps the CLOUDS at the domain
    // width wherever the camera is, but the city is read in absolute world
    // coordinates — folding the camera at the cloud period teleported the
    // city by a fraction of a tile every crossing. Folding at a common
    // multiple of both periods instead makes both resets invisible.
    this.wrapExtent = wrapExtent;
    // Held so `reset` returns to this field's own opening view rather than to
    // the global default — on a demo those are different places, and the one
    // worth going back to is the one the page promised.
    this.start = startCamera(start);
    this.position = cameraWorldOrigin(this.start.position, bmin, bmax);
    this.azimuth = this.start.azimuth;
    this.elevation = this.start.elevation;
    this.fov = this.start.fov;
    this.startSpeed = speed;
    this.speed = this.startSpeed;
    this.keys = new Set();
    this.speedFlashUntil = 0;
    this.constrain();
  }

  reset() {
    this.position = cameraWorldOrigin(this.start.position, this.bmin, this.bmax);
    this.azimuth = this.start.azimuth;
    this.elevation = this.start.elevation;
    this.speed = this.startSpeed;
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

  /**
   * WASD is the ground plane; Space and Shift are the only way to change
   * altitude.
   *
   * W used to walk the full view direction, so looking up and holding W
   * climbed — which meant altitude was coupled to where you happened to be
   * pointing and could not be held while looking at anything. Splitting the
   * two lets you fly level across the domain and look wherever you like.
   *
   * The horizontal forward comes from the azimuth directly rather than by
   * flattening the view vector, which would shrink to nothing (and then
   * normalize to noise) as the view approached vertical.
   */
  move(dt) {
    if (this.keys.size === 0) return false;
    const [, r] = cameraBasis(this.azimuth, this.elevation);
    const f = directionFromAzimuthElevation(this.azimuth, 0.0);
    const d = this.speed * Math.min(dt, 0.1);
    const p = this.position;
    const k = this.keys;
    if (k.has("w")) { p[0] += f[0] * d; p[1] += f[1] * d; }
    if (k.has("s")) { p[0] -= f[0] * d; p[1] -= f[1] * d; }
    if (k.has("a")) { p[0] -= r[0] * d; p[1] -= r[1] * d; }
    if (k.has("d")) { p[0] += r[0] * d; p[1] += r[1] * d; }
    const dz = d * VERTICAL_SPEED_FRACTION;
    if (k.has(" ")) p[2] += dz;
    if (k.has("shift") || k.has("c")) p[2] -= dz;
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
        const extent = this.wrapExtent?.[i] ?? (this.bmax[i] - this.bmin[i]);
        p[i] = this.bmin[i] +
          (((p[i] - this.bmin[i]) % extent) + extent) % extent;
      }
    }
    p[2] = Math.max(p[2], OCEAN_FLOOR_MARGIN_M);
  }

  relativePosition() {
    // Folded into the box whatever the wrap extent: the minimap and every
    // relative readout describe the cloud domain, and a camera nine periods
    // out is still over the same cloud.
    const p = [...this.position];
    if (this.periodic) {
      for (const i of [0, 1]) {
        const extent = this.bmax[i] - this.bmin[i];
        p[i] = this.bmin[i] +
          (((p[i] - this.bmin[i]) % extent) + extent) % extent;
      }
    }
    return worldToRelative(p, this.bmin, this.bmax);
  }
}
