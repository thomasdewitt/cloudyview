// Zen mode's flight: where the camera goes with nobody at the controls.
//
// Two walks, one per kind of scene.
//
//   CloudZen  over a cloud field. Level flight, at a speed set by the field's
//             grid (ZEN_CLOUD_CELLS_PER_S), on a curve
//             whose turn rate wanders, never tighter than a circle flown in
//             ZEN_CLOUD_CIRCLE_S, while the altitude wanders inside a band
//             that straddles the bottom of the cloud layer. It flies on
//             without end, so the field has to wrap: the viewer turns the
//             periodic domain on for a zen flight (viewer/zen.js).
//   CityZen   over the night city, at ZEN_CITY_SPEED. Above the buildings
//             it flies as CloudZen does, on a wandering curve; then it lines
//             up over a street, sinks into it, flies it straight at street
//             level, and climbs back out; and so on, about half the time
//             each. Every leg is swept against the city (citylots.js,
//             cityprops.js) before it is flown: the curve is drawn again
//             until it clears the towers, and the altitudes are ones that
//             clear the skybridges, freeway decks and lamps. No leg is
//             taken unless a full circle is clear from its end, so the
//             flight never goes where it could not turn round.
//
// Both are flown the same way: step(dt, camera) moves the camera through its
// own move path (FlightCamera.displace) and sets its look, level. Neither
// touches the DOM or the GPU, so node can fly them (tests/app/test_zen.py).

"use strict";

import { directionFromAzimuthElevation, mod360 } from "./spectral.js";

const DEG = Math.PI / 180.0;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
/** 0 at x <= 0, 1 at x >= 1, with no slope at either end. */
const ease = (x) => { const t = clamp(x, 0.0, 1.0); return t * t * (3.0 - 2.0 * t); };
/** An angle difference folded into [-180, 180). */
const fold180 = (deg) => mod360(deg + 180.0) - 180.0;

// --- what both walks share --------------------------------------------------

// How long a wandering value takes to ease from one random level to the
// next: drawn uniformly from this range each time, in seconds.
export const ZEN_DRIFT_S = [30.0, 60.0];
// The flight eases up to its speed over this long, so that starting zen from
// a hover is not a jolt.
export const ZEN_EASE_IN_S = 3.0;
// The look is level in zen. Whatever elevation the camera had when zen
// started decays to zero with this time constant.
export const ZEN_LEVEL_S = 0.8;

// --- over a cloud field -----------------------------------------------------

// The speed, in grid cells a second: the flight crosses this many of the
// field's horizontal cells each second, so a fine field is flown slowly and
// a coarse one fast, and the detail the field resolves goes by at one rate.
export const ZEN_CLOUD_CELLS_PER_S = 10.0;
// The tightest turn: a full circle would take this long. The turn rate
// wanders between that to the left and that to the right.
export const ZEN_CLOUD_CIRCLE_S = 300.0;
// The altitude band, as a fraction of the cloud layer's depth, centred on
// the layer's bottom: half of it is clear air under the cloud, half is the
// bottom of the cloud.
export const ZEN_CLOUD_BAND_FRACTION = 2.0 / 3.0;
// The band never comes lower than this above the sea.
export const ZEN_CLOUD_MIN_ALTITUDE_M = 50.0;
// The most the altitude changes between two levels, as a fraction of the
// distance flown between them.
export const ZEN_CLOUD_CLIMB = 0.1;

// --- over the city ----------------------------------------------------------

export const ZEN_CITY_SPEED = 6.0;               // m/s
// Above the buildings: the altitude range, the most the altitude changes
// between two levels as a fraction of the distance flown between them, and
// how far the flight is kept from anything solid. The turn rate wanders as
// it does over cloud (ZEN_CLOUD_CIRCLE_S).
export const ZEN_CITY_ALTITUDE_M = [40.0, 500.0];
export const ZEN_CITY_CLIMB = 0.25;
export const ZEN_CITY_CLEAR_M = 8.0;
// When no gentle curve is clear of the towers ahead, tighter ones are
// tried, to these turn rates in turn (degrees per second), and then closer
// passes (metres from anything solid).
export const ZEN_CITY_AVOID_RATES = [4.0, 10.0];
export const ZEN_CITY_AVOID_CLEAR_M = [3.0, 1.0];
// The way out kept in hand above the buildings: a full circle at this turn
// rate from the end of the leg, eased into over this many seconds. A leg is
// only taken if its circle is clear, and the circle is what is flown when
// no next leg is.
export const ZEN_CITY_CIRCLE_RATE = 10.0;        // degrees per second
export const ZEN_CITY_CIRCLE_EASE_S = 3.0;
// In a street: the altitude range above it, and how far the flight is kept
// from anything solid (a street is 12 m from wall to wall).
export const ZEN_CITY_STREET_ALTITUDE_M = [4.0, 25.0];
export const ZEN_CITY_STREET_CLEAR_M = 1.0;
// The slope of the descent into a street and of the climb out of it, as
// height over distance, averaged over the leg.
export const ZEN_CITY_SINK = 0.25;
// How long the flight stays above the buildings before it looks for a
// street, and how long it stays in one, in seconds: drawn from these ranges.
// A street that ends sooner ends the stay sooner.
export const ZEN_CITY_ABOVE_S = [50.0, 100.0];
export const ZEN_CITY_STREET_S = [180.0, 360.0];
// Lining up over a street: the flight steers for a point this far ahead on
// the street's line, turning no faster than this.
export const ZEN_CITY_ALIGN_AHEAD_M = 60.0;
export const ZEN_CITY_ALIGN_RATE = 5.0;          // degrees per second
// A leg is checked this far past its end, held as it ends, so that the next
// one does not begin against an obstacle.
export const ZEN_CITY_LOOKAHEAD_M = 120.0;
// When the curves drawn at their usual length are all blocked, short legs
// are tried: this long, in seconds, and checked this far past their end.
export const ZEN_CITY_SHORT_S = [8.0, 15.0];
export const ZEN_CITY_SHORT_AHEAD_M = 40.0;
// Planning a leg sweeps the city some hundreds of times. It is done a little
// each frame, for at most this long, while the leg before it is flown.
export const ZEN_CITY_PLAN_MS = 2.0;
// How far apart the points of a planned leg are.
export const ZEN_CITY_STEP_M = 2.0;

/**
 * A value that wanders. It eases from one level to the next as time u runs
 * (seconds) and draws the next level when it arrives: `draw(v, u)` is given
 * the level just reached and when, and returns [next level, when it is
 * reached].
 */
class Drift {
  constructor(u, v, draw) {
    this.draw = draw;
    this.u0 = u;
    this.v0 = v;
    [this.v1, this.u1] = draw(v, u);
  }

  at(u) {
    while (u >= this.u1) {
      this.u0 = this.u1;
      this.v0 = this.v1;
      [this.v1, this.u1] = this.draw(this.v0, this.u0);
    }
    const x = (u - this.u0) / (this.u1 - this.u0);
    return this.v0 + (this.v1 - this.v0) * ease(x);
  }
}

/** Bring the look level: the elevation decays to zero and then is zero. */
function levelLook(camera, dt) {
  const e = camera.elevation * Math.exp(-dt / ZEN_LEVEL_S);
  camera.elevation = Math.abs(e) < 0.01 ? 0.0 : e;
}

/**
 * The cloud walk's altitude band [low, high] for a field whose data spans
 * bmin[2]..bmax[2]. The volume is cropped to the planes that hold cloud
 * (zcrop.js), so those two are the cloud layer's bottom and top.
 */
export function cloudBand(bmin, bmax) {
  const half = 0.5 * ZEN_CLOUD_BAND_FRACTION * (bmax[2] - bmin[2]);
  const high = Math.max(bmin[2] + half, ZEN_CLOUD_MIN_ALTITUDE_M);
  const low = Math.min(Math.max(bmin[2] - half, ZEN_CLOUD_MIN_ALTITUDE_M), high);
  return [low, high];
}

/**
 * The cloud walk's speed, m/s, over a field of `shape` cells spanning
 * bmin..bmax: ZEN_CLOUD_CELLS_PER_S of its horizontal cells a second.
 */
export function cloudSpeed(bmin, bmax, shape) {
  const cell = 0.5 * ((bmax[0] - bmin[0]) / shape[0] + (bmax[1] - bmin[1]) / shape[1]);
  return ZEN_CLOUD_CELLS_PER_S * cell;
}

export class CloudZen {
  /**
   * `camera` is the FlightCamera zen takes over, where it is now, and
   * `speed` the speed to fly at, m/s (cloudSpeed).
   */
  constructor(camera, speed, random = Math.random) {
    if (!(speed > 0)) throw new Error(`CloudZen needs a speed; got ${speed}.`);
    this.speed = speed;
    const span = () => ZEN_DRIFT_S[0] + random() * (ZEN_DRIFT_S[1] - ZEN_DRIFT_S[0]);
    const [low, high] = cloudBand(camera.bmin, camera.bmax);
    this.band = [low, high];
    this.time = 0.0;
    this.heading = camera.azimuth;
    // Degrees per second, positive clockwise seen from above. It starts at
    // zero: the flight leaves along the heading it had.
    const maxRate = 360.0 / ZEN_CLOUD_CIRCLE_S;
    this.turnRate = new Drift(0.0, 0.0, (_, t) =>
      [(2.0 * random() - 1.0) * maxRate, t + span()]);
    // Each new level is a height drawn evenly from the band, approached no
    // faster than the climb limit allows.
    this.altitude = new Drift(0.0, camera.position[2], (z, t) => {
      const dt = span();
      const most = ZEN_CLOUD_CLIMB * speed * dt;
      const target = low + random() * (high - low);
      return [z + clamp(target - z, -most, most), t + dt];
    });
  }

  step(dt, camera) {
    this.time += dt;
    const speed = this.speed * ease(this.time / ZEN_EASE_IN_S);
    this.heading = mod360(this.heading + this.turnRate.at(this.time) * dt);
    const f = directionFromAzimuthElevation(this.heading, 0.0);
    const d = speed * dt;
    camera.displace(f[0] * d, f[1] * d,
                    this.altitude.at(this.time) - camera.position[2]);
    camera.azimuth = this.heading;
    levelLook(camera, dt);
  }
}

// --- the street grid --------------------------------------------------------
//
// The streets run along the block grid's lines. A NODE is a grid corner,
// (i, j) at (i * cell, j * cell); a street joins two neighbouring nodes. The
// tile's street word (tools/city_lots.py) says which do: bit 0 of block
// (i, j) is the street up its -x edge, from node (i, j) to (i, j + 1), and
// bit 2 the street along its -y edge, from node (i, j) to (i + 1, j).

// The four ways along the grid, counter-clockwise: +x, +y, -x, -y.
const WAYS = [[1, 0], [0, 1], [-1, 0], [0, -1]];
const WAY_AZIMUTH = [90.0, 0.0, 270.0, 180.0];

/** Whether a street leaves node (i, j) by `way`. */
export function streetFrom(tile, i, j, way) {
  // The block whose -x or -y edge the street is.
  const bi = way === 2 ? i - 1 : i;
  const bj = way === 3 ? j - 1 : j;
  const n = tile.n;
  const word = tile.street[(((bj % n) + n) % n) * n + (((bi % n) + n) % n)] + 2048;
  return ((word >>> (way % 2 === 0 ? 2 : 0)) & 1) === 1;
}

export class CityZen {
  /**
   * `camera` is the FlightCamera zen takes over, where it is now, and
   * `lots` the city it flies in (citylots.js CityLots, with its props).
   */
  constructor(camera, lots, random = Math.random) {
    this.lots = lots;
    this.tile = lots.tile;
    this.random = random;
    this.time = 0.0;
    // The camera in the city frame. It is the camera's own surfacePosition,
    // so the legs are laid in the frame the renderer and the collider fold
    // the city in (see _follow).
    this.pos = [camera.surfacePosition[0], camera.surfacePosition[1]];
    // The legs to fly, in order: each a list of points [x, y, z, azimuth]
    // ZEN_CITY_STEP_M apart. `s` is how far along the first the flight is.
    this.legs = [];
    this.s = 0.0;
    // Where the last planned leg ends, and the turn rate it ends with.
    this.tip = { x: this.pos[0], y: this.pos[1], z: camera.position[2],
                 azimuth: camera.azimuth, rate: 0.0 };
    // Metres planned so far, and the distance at which the present stay
    // (above the buildings, or in a street) is due to end.
    this.planned = 0.0;
    this.until = this._stay(ZEN_CITY_ABOVE_S);
    // In a street: the way it runs, the line it runs along (the grid line's
    // x or y, metres), and the climb out that has been checked from the end
    // of the last planned leg. Null above the buildings.
    this.street = null;
    // Above the buildings: the circle that is clear from the tip
    // ({points, rate, clear}, see _circle), flown when no next leg is clear.
    // `circled` counts how often it was.
    this.circle = null;
    this.circled = 0;
    // With no circle in hand either (zen began boxed in), the flight rises
    // where it is and plans again (see step). `stuck` counts how often.
    this.rising = false;
    this.stuck = 0;
    this._planAt = 0.0;
    // The planning under way (a generator, see _work), or null, and whether
    // the last one found nothing clear.
    this._job = null;
    this.failed = false;
  }

  _between([low, high]) { return low + this.random() * (high - low); }

  /** The planned distance at which a stay of `range` seconds, begun now, ends. */
  _stay(range) { return this.planned + ZEN_CITY_SPEED * this._between(range); }

  // --- checking a path against the city --------------------------------------

  /**
   * Whether a path keeps `clear` metres from everything solid. It is swept
   * as chords `stride` points long: short enough that a curve stays within
   * centimetres of them.
   *
   * This and everything below that calls it are generators: each `yield`
   * is a place the work can stop for the frame (see _work), and the value
   * returned at the end is the answer.
   */
  *_clear(points, clear, stride = 4) {
    // A sweep that starts within reach of a solid does not report it, so
    // the first point is asked about by itself.
    const [x, y, z] = points[0];
    if (this.lots.inside([x, y, z], clear) !== null) return false;
    let swept = 0;
    for (let k = 0; k < points.length - 1; k += stride) {
      const p = points[k];
      const q = points[Math.min(k + stride, points.length - 1)];
      if (this.lots.sweep([p[0], p[1], p[2]], [q[0], q[1], q[2]], clear) !== null) {
        return false;
      }
      swept += 1;
      if (swept % 6 === 0) yield;
    }
    return true;
  }

  /** `count` more points straight on from a point, level. */
  _straightOn(from, count) {
    const [x, y, z, azimuth] = from;
    const f = directionFromAzimuthElevation(azimuth, 0.0);
    const out = [[x, y, z, azimuth]];
    for (let k = 1; k <= count; k++) {
      out.push([x + k * ZEN_CITY_STEP_M * f[0], y + k * ZEN_CITY_STEP_M * f[1],
                z, azimuth]);
    }
    return out;
  }

  /**
   * Whether every stretch of a street's line that the points fly below the
   * buildings' range (ZEN_CITY_ALTITUDE_M[0]) is a street.
   */
  _onStreet(points, way, line) {
    const cell = this.tile.cellM;
    const along = way % 2 === 0 ? 0 : 1;
    const across = Math.round(line / cell);
    let last = null;
    for (const p of points) {
      if (p[2] >= ZEN_CITY_ALTITUDE_M[0]) continue;
      const k = Math.floor(p[along] / cell);
      if (k === last) continue;
      last = k;
      const open = along === 0 ? streetFrom(this.tile, k, across, 0)
                               : streetFrom(this.tile, across, k, 1);
      if (!open) return false;
    }
    return true;
  }

  // --- above the buildings -----------------------------------------------------

  /**
   * The way out from a point: level, the turn rate eased from `rate` to
   * ZEN_CITY_CIRCLE_RATE one way or the other, and then once right round.
   * {points, rate, clear} for the first way that keeps `clear` from
   * everything, or null. Its end is on the same circle, so the same circle
   * is the way out from there too.
   */
  *_circle(from, rate, clear) {
    const step = ZEN_CITY_STEP_M;
    const dt = step / ZEN_CITY_SPEED;
    const easing = Math.round(ZEN_CITY_CIRCLE_EASE_S / dt);
    const round = Math.round(360.0 / ZEN_CITY_CIRCLE_RATE / dt);
    // The way it is already turning first.
    const first = rate !== 0.0 ? Math.sign(rate) : (this.random() < 0.5 ? 1.0 : -1.0);
    for (const side of [first, -first]) {
      const full = side * ZEN_CITY_CIRCLE_RATE;
      const points = [[...from]];
      let [px, py, , az] = from;
      for (let k = 1; k <= easing + round; k++) {
        const turn = (rate + (full - rate) * ease(k / easing)) * dt;
        const f = directionFromAzimuthElevation(az + 0.5 * turn, 0.0);
        px += step * f[0];
        py += step * f[1];
        az = mod360(az + turn);
        points.push([px, py, from[2], az]);
      }
      if (yield* this._clear(points, clear)) return { points, rate: full, clear };
    }
    return null;
  }

  /**
   * A leg of the wandering curve from the tip: the turn rate eases to a new
   * random rate within `maxRate`, the altitude to a new height in the
   * range, over a random time: {points, rate, circle}. Null if what was
   * drawn is not clear, or no circle is clear from its end.
   */
  *_curve(maxRate, clear, climb, span = ZEN_DRIFT_S,
         lookahead = ZEN_CITY_LOOKAHEAD_M) {
    const { x, y, z, azimuth, rate } = this.tip;
    const [low, high] = ZEN_CITY_ALTITUDE_M;
    const step = ZEN_CITY_STEP_M;
    const seconds = this._between(span);
    const count = Math.max(2, Math.round(seconds * ZEN_CITY_SPEED / step));
    const rate1 = (2.0 * this.random() - 1.0) * maxRate;
    const most = climb * count * step;
    const z1 = z + clamp(this._between([low, high]) - z, -most, most);
    const points = [[x, y, z, azimuth]];
    let px = x, py = y, az = azimuth;
    const dt = step / ZEN_CITY_SPEED;
    // The leg, and then the lookahead flown on at the rate and height it
    // ends with.
    const ahead = Math.round(lookahead / step);
    for (let k = 1; k <= count + ahead; k++) {
      const t = Math.min(k / count, 1.0);
      const turn = (rate + (rate1 - rate) * ease(t)) * dt;
      const f = directionFromAzimuthElevation(az + 0.5 * turn, 0.0);
      px += step * f[0];
      py += step * f[1];
      az = mod360(az + turn);
      points.push([px, py, z + (z1 - z) * ease(t), az]);
    }
    if (!(yield* this._clear(points, clear))) return null;
    const circle = yield* this._circle(points[count], rate1, clear);
    if (circle === null) return null;
    return { points: points.slice(0, count + 1), rate: rate1, circle };
  }

  /**
   * The next leg above the buildings: the gentle curve if one is clear;
   * else straight on for a short way; else tighter turns and steeper
   * climbs; else short legs that pass closer. Null when nothing is clear.
   */
  *_planAbove() {
    const gentle = 360.0 / ZEN_CLOUD_CIRCLE_S;
    const steep = 2.0 * ZEN_CITY_CLIMB;
    const tightest = ZEN_CITY_AVOID_RATES[ZEN_CITY_AVOID_RATES.length - 1];
    const ladder = [
      [gentle, ZEN_CITY_CLEAR_M, ZEN_CITY_CLIMB, ZEN_DRIFT_S, ZEN_CITY_LOOKAHEAD_M],
      [0.0, ZEN_CITY_CLEAR_M, steep, ZEN_CITY_SHORT_S, ZEN_CITY_LOOKAHEAD_M],
      ...ZEN_CITY_AVOID_RATES.map((r) => [
        r, ZEN_CITY_CLEAR_M, steep, ZEN_DRIFT_S, ZEN_CITY_LOOKAHEAD_M]),
      [tightest, ZEN_CITY_CLEAR_M, steep, ZEN_CITY_SHORT_S, ZEN_CITY_SHORT_AHEAD_M],
      ...ZEN_CITY_AVOID_CLEAR_M.map((c) => [
        tightest, c, steep, ZEN_CITY_SHORT_S, ZEN_CITY_SHORT_AHEAD_M]),
      ...ZEN_CITY_AVOID_CLEAR_M.map((c) => [
        0.0, c, steep, ZEN_CITY_SHORT_S, ZEN_CITY_SHORT_AHEAD_M]),
    ];
    for (const rung of ladder) {
      for (let k = 0; k < 8; k++) {
        const leg = yield* this._curve(...rung);
        if (leg !== null) return leg;
      }
    }
    return null;
  }

  // --- into a street -----------------------------------------------------------

  /**
   * A curve from the tip onto a street's line, ending on it and along it,
   * at altitude `z1`. It steers for a point ZEN_CITY_ALIGN_AHEAD_M ahead on
   * the line. Null if it has not settled on the line within a kilometre
   * and a half.
   */
  _lineUp(way, line, z1) {
    const { x, y, z, azimuth, rate } = this.tip;
    const step = ZEN_CITY_STEP_M;
    const dt = step / ZEN_CITY_SPEED;
    const u = WAYS[way];
    const axis = WAY_AZIMUTH[way];
    const points = [[x, y, z, azimuth]];
    let px = x, py = y, az = azimuth, r = rate;
    const limit = Math.round(1500.0 / step);
    for (let k = 1; k <= limit; k++) {
      // How far left of the line, facing along it.
      const left = way % 2 === 0 ? (py - line) * u[0] : -(px - line) * u[1];
      const want = axis + clamp(
        Math.atan(left / ZEN_CITY_ALIGN_AHEAD_M) / DEG, -45.0, 45.0);
      const ask = clamp(fold180(want - az) / 3.0,
                        -ZEN_CITY_ALIGN_RATE, ZEN_CITY_ALIGN_RATE);
      r += (ask - r) * Math.min(dt / 1.5, 1.0);
      const turn = r * dt;
      const f = directionFromAzimuthElevation(az + 0.5 * turn, 0.0);
      px += step * f[0];
      py += step * f[1];
      az = mod360(az + turn);
      points.push([px, py, 0.0, az]);
      if (Math.abs(left) < 0.05 && Math.abs(fold180(az - axis)) < 0.2
          && Math.abs(r) < 0.05) break;
      if (k === limit) return null;
    }
    // The last few centimetres: five points that end exactly on the line,
    // exactly along it.
    const last = points[points.length - 1];
    for (let k = 1; k <= 5; k++) {
      const w = k / 5;
      const on = way % 2 === 0
        ? [last[0] + k * step * u[0], last[1] + (line - last[1]) * w]
        : [last[0] + (line - last[0]) * w, last[1] + k * step * u[1]];
      points.push([on[0], on[1], 0.0, mod360(last[3] + fold180(axis - last[3]) * w)]);
    }
    const n = points.length - 1;
    points.forEach((p, k) => { p[2] = z + (z1 - z) * ease(k / n); });
    return points;
  }

  /**
   * A straight leg along a street's line from `from`, `length` metres,
   * easing to altitude `z1`.
   */
  _along(from, length, z1) {
    const count = Math.max(2, Math.round(length / ZEN_CITY_STEP_M));
    const points = this._straightOn(from, count);
    points.forEach((p, k) => { p[2] = from[2] + (z1 - from[2]) * ease(k / count); });
    return points;
  }

  /**
   * The climb out of a street from `from`: up along the street to a height
   * from which ZEN_CITY_LOOKAHEAD_M straight on, and a circle, are clear by
   * the margin kept above the buildings. {points, circle}, or null if no
   * height in the range will do.
   */
  *_climbOut(from, way, line) {
    const ahead = Math.round(ZEN_CITY_LOOKAHEAD_M / ZEN_CITY_STEP_M);
    const [low, high] = ZEN_CITY_ALTITUDE_M;
    for (const top of [low + 20.0, 100.0, 150.0, 220.0, 320.0, high - 20.0]) {
      if (top <= from[2]) continue;
      const points = this._along(from, (top - from[2]) / ZEN_CITY_SINK, top);
      if (!this._onStreet(points, way, line)) continue;
      if (!(yield* this._clear(points, ZEN_CITY_STREET_CLEAR_M, 8))) continue;
      const end = points[points.length - 1];
      if (!(yield* this._clear(this._straightOn(end, ahead), ZEN_CITY_CLEAR_M, 8))) {
        continue;
      }
      const circle = yield* this._circle(end, 0.0, ZEN_CITY_CLEAR_M);
      if (circle !== null) return { points, circle };
    }
    return null;
  }

  /**
   * A straight leg in a street from `from` to street-level altitude `z1`,
   * with the climb out from its end: {points, climb}, or null if the leg
   * leaves the street, touches something, or has no clear climb out.
   */
  *_streetLeg(from, way, line, length, z1) {
    const points = this._along(from, length, z1);
    const ahead = this._straightOn(
      points[points.length - 1], Math.round(40.0 / ZEN_CITY_STEP_M));
    if (!this._onStreet(points, way, line) || !this._onStreet(ahead, way, line)) {
      return null;
    }
    if (!(yield* this._clear(points, ZEN_CITY_STREET_CLEAR_M, 8))
        || !(yield* this._clear(ahead, ZEN_CITY_STREET_CLEAR_M, 8))) return null;
    const climb = yield* this._climbOut(points[points.length - 1], way, line);
    return climb === null ? null : { points, climb };
  }

  /**
   * Leave the wandering curve for a street: a curve onto the line of a
   * street that runs roughly the way the flight is heading, then the
   * descent into it. The legs are queued and true returned; false when no
   * street near enough can be lined up over and sunk into.
   */
  *_enterStreet() {
    const cell = this.tile.cellM;
    const { x, y, z, azimuth } = this.tip;
    const [low] = ZEN_CITY_ALTITUDE_M;
    const [streetLow, streetHigh] = ZEN_CITY_STREET_ALTITUDE_M;
    // The grid ways within 60 degrees of the heading, nearest first.
    const ways = [0, 1, 2, 3]
      .map((w) => [w, Math.abs(fold180(WAY_AZIMUTH[w] - azimuth))])
      .filter(([, off]) => off < 60.0)
      .sort((a, b) => a[1] - b[1]);
    for (const [way] of ways) {
      const across = way % 2 === 0 ? y : x;
      const k0 = Math.round(across / cell);
      for (const dk of [0, 1, -1, 2, -2, 3, -3]) {
        const line = (k0 + dk) * cell;
        // Come down toward the rooftops on the way, as far as the wander's
        // own climb limit lets it; if that is not clear, level.
        const flat = this._lineUp(way, line, z);
        yield;
        if (flat === null) continue;
        const most = ZEN_CITY_CLIMB * (flat.length - 1) * ZEN_CITY_STEP_M;
        const lower = Math.max(z - most, low + 20.0);
        for (const zLine of lower < z ? [lower, z] : [z]) {
          const curve = zLine === z ? flat : this._lineUp(way, line, zLine);
          if (!(yield* this._clear(curve, ZEN_CITY_CLEAR_M))) continue;
          const from = curve[curve.length - 1];
          for (let k = 0; k < 6; k++) {
            const zStreet = this._between([streetLow, streetHigh]);
            const sink = yield* this._streetLeg(
              from, way, line, (zLine - zStreet) / ZEN_CITY_SINK, zStreet);
            if (sink === null) continue;
            this._queue(curve, 0.0);
            this._queue(sink.points, 0.0);
            this.circle = null;
            this.street = { way, line, climb: sink.climb };
            this.until = this._stay(ZEN_CITY_STREET_S);
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Zen began down among the buildings. If the flight is in a street and
   * facing more along it than across it, it takes that street from where
   * it is: a curve onto the centre line, then the street as any other.
   * True if so.
   */
  *_joinStreet() {
    const cell = this.tile.cellM;
    const { x, y, z, azimuth } = this.tip;
    const [streetLow, streetHigh] = ZEN_CITY_STREET_ALTITUDE_M;
    const f = directionFromAzimuthElevation(azimuth, 0.0);
    // The nearest grid line each way, nearest first.
    const lines = [
      { ways: [0, 2], line: Math.round(y / cell) * cell, off: y },
      { ways: [1, 3], line: Math.round(x / cell) * cell, off: x },
    ].sort((a, b) => Math.abs(a.off - a.line) - Math.abs(b.off - b.line));
    for (const { ways, line } of lines) {
      const way = ways.find((w) => f[0] * WAYS[w][0] + f[1] * WAYS[w][1] > 0.0);
      if (way === undefined) continue;
      const curve = this._lineUp(way, line, z);
      yield;
      if (curve === null || !this._onStreet(curve, way, line)
          || !(yield* this._clear(curve, ZEN_CITY_STREET_CLEAR_M))) continue;
      const leg = yield* this._streetLeg(
        curve[curve.length - 1], way, line, 80.0, clamp(z, streetLow, streetHigh));
      if (leg === null) continue;
      this._queue(curve, 0.0);
      this._queue(leg.points, 0.0);
      this.street = { way, line, climb: leg.climb };
      this.until = this._stay(ZEN_CITY_STREET_S);
      return true;
    }
    return false;
  }

  /**
   * The next leg in the street: on along it at a new street-level altitude
   * while the stay lasts and the street does, else the climb out.
   */
  *_planStreet() {
    const { way, line, climb } = this.street;
    const { x, y, z, azimuth } = this.tip;
    const from = [x, y, z, azimuth];
    if (this.planned < this.until) {
      const length = ZEN_CITY_SPEED * this._between(ZEN_DRIFT_S);
      const most = ZEN_CITY_CLIMB * length;
      const levels = [];
      for (let k = 0; k < 6; k++) {
        const target = this._between(ZEN_CITY_STREET_ALTITUDE_M);
        levels.push(z + clamp(target - z, -most, most));
      }
      levels.push(z);
      for (const z1 of levels) {
        const leg = yield* this._streetLeg(from, way, line, length, z1);
        if (leg === null) continue;
        this._queue(leg.points, 0.0);
        this.street.climb = leg.climb;
        return true;
      }
    }
    this._queue(climb.points, 0.0);
    this.circle = climb.circle;
    this.street = null;
    this.until = this._stay(ZEN_CITY_ABOVE_S);
    return true;
  }

  // --- the queue ---------------------------------------------------------------

  /** Add a leg; the tip moves to its end, with turn rate `rate`. */
  _queue(points, rate) {
    this.legs.push(points);
    this.planned += (points.length - 1) * ZEN_CITY_STEP_M;
    const [x, y, z, azimuth] = points[points.length - 1];
    this.tip = { x, y, z, azimuth, rate };
  }

  /** Plan one more leg from the tip. False when nothing is clear from it. */
  *_planning() {
    if (this.street !== null) return yield* this._planStreet();
    if (this.planned >= this.until && (yield* this._enterStreet())) return true;
    // Zen has only just begun, low: a street the flight is already in comes
    // before a way up out of it.
    const begun = this.circle === null && this.tip.z < ZEN_CITY_ALTITUDE_M[0];
    if (begun && (yield* this._joinStreet())) return true;
    const leg = yield* this._planAbove();
    if (leg !== null) {
      this._queue(leg.points, leg.rate);
      this.circle = leg.circle;
      return true;
    }
    // Nothing drawn was clear. Round the circle kept for this, and draw
    // again from its end. With no circle, there is nothing to fly.
    if (this.circle === null) return false;
    const { points, rate, clear } = this.circle;
    this._queue(points, rate);
    this.circled += 1;
    this.circle = yield* this._circle(points[points.length - 1], rate, clear);
    return true;
  }

  /**
   * Keep two legs in hand, the one being flown and the one after it, by
   * planning for at most ZEN_CITY_PLAN_MS of this frame. When nothing is
   * clear from the tip, planning starts again half a second later.
   */
  _work() {
    if (this.legs.length >= 2 || this.time < this._planAt) return;
    const stop = performance.now() + ZEN_CITY_PLAN_MS;
    do {
      if (this._job === null) this._job = this._planning();
      const { done, value } = this._job.next();
      if (!done) continue;
      this._job = null;
      if (!value) {
        this._planAt = this.time + 0.5;
        this.failed = true;
        return;
      }
      this.failed = false;
    } while (this.legs.length < 2 && performance.now() < stop);
  }

  // --- flying it ---------------------------------------------------------------

  /**
   * Take the camera's position as it now is, and keep the legs in the
   * camera's frame: the camera's surfacePosition folds at the tile, and
   * when it does everything planned is moved by the same whole tile.
   */
  _follow(camera) {
    const extent = this.tile.n * this.tile.cellM;
    const sp = camera.surfacePosition;
    for (const i of [0, 1]) {
      const shift = Math.round((sp[i] - this.pos[i]) / extent) * extent;
      if (shift === 0) continue;
      // What a plan under way has worked out is in the old frame.
      this._job = null;
      for (const leg of this.legs) for (const p of leg) p[i] += shift;
      this.tip[i === 0 ? "x" : "y"] += shift;
      if (this.circle !== null) for (const p of this.circle.points) p[i] += shift;
      if (this.street !== null) {
        for (const p of this.street.climb.points) p[i] += shift;
        for (const p of this.street.climb.circle.points) p[i] += shift;
        // A street's line is an x for a street that runs along y.
        if ((this.street.way % 2 === 0) === (i === 1)) this.street.line += shift;
      }
    }
    this.pos = [sp[0], sp[1]];
  }

  step(dt, camera) {
    this.time += dt;
    this._follow(camera);
    levelLook(camera, dt);
    const speed = ZEN_CITY_SPEED * ease(this.time / ZEN_EASE_IN_S)
                * camera.control;
    const lengthOf = (leg) => (leg.length - 1) * ZEN_CITY_STEP_M;
    if (this.rising) {
      // Nothing was clear from where the flight was: it rises for half a
      // second, holds while it plans from there, and rises again if that
      // found nothing either.
      if (this._job === null && this.time >= this._planAt) {
        this.tip = { x: this.pos[0], y: this.pos[1], z: camera.position[2],
                     azimuth: camera.azimuth, rate: 0.0 };
        this.until = Math.max(this.until, this.planned);
      }
      this._work();
      if (this.legs.length > 0) {
        this.rising = false;
        this.s = 0.0;
      } else if (this._job === null) {
        camera.displace(0.0, 0.0, speed * dt);
      }
      return;
    }
    this._work();
    const spent = this.legs.length === 0
      || (this.legs.length === 1 && this.s >= lengthOf(this.legs[0]));
    if (spent) {
      // Out of legs. While the plan for the next is still being worked out
      // the flight holds where it is; if nothing was clear, it rises.
      if (this._job === null && this.failed) {
        this.rising = true;
        this.stuck += 1;
        this.legs = [];
        this.street = null;
        this.circle = null;
      }
      if (this.legs.length === 0) return;
    }
    this.s += speed * dt;
    let length = lengthOf(this.legs[0]);
    while (this.s >= length && this.legs.length > 1) {
      this.s -= length;
      this.legs.shift();
      length = lengthOf(this.legs[0]);
    }
    const leg = this.legs[0];
    const at = Math.min(this.s, length) / ZEN_CITY_STEP_M;
    const k = Math.min(Math.floor(at), leg.length - 2);
    const w = at - k;
    const a = leg[k], b = leg[k + 1];
    // Onto the leg. A crash (collide.js) leaves the camera off it; it comes
    // back at a bounded speed, not in one frame.
    let dx = a[0] + (b[0] - a[0]) * w - this.pos[0];
    let dy = a[1] + (b[1] - a[1]) * w - this.pos[1];
    let dz = a[2] + (b[2] - a[2]) * w - camera.position[2];
    const off = Math.hypot(dx, dy, dz);
    const most = 3.0 * ZEN_CITY_SPEED * dt;
    if (off > most) {
      dx *= most / off; dy *= most / off; dz *= most / off;
    }
    camera.displace(dx, dy, dz);
    this.pos[0] += dx;
    this.pos[1] += dy;
    camera.azimuth = mod360(a[3] + fold180(b[3] - a[3]) * w);
  }
}
