// Zen mode's flight: where the camera goes with nobody at the controls.
//
// Two walks, one per kind of scene.
//
//   CloudZen  over a cloud field. Level flight at ZEN_CLOUD_SPEED on a curve
//             whose turn rate wanders, never tighter than a circle flown in
//             ZEN_CLOUD_CIRCLE_S, while the altitude wanders inside a band
//             that straddles the bottom of the cloud layer.
//   CityZen   over the night city. It follows the street grid at
//             ZEN_CITY_SPEED, at each intersection going straight on or
//             turning left or right, while the altitude wanders between the
//             street and ZEN_CITY_ALTITUDE_M[1]. Every stretch is swept
//             against the city (citylots.js, cityprops.js) before it is
//             flown, so the altitude it picks is one that clears the
//             skybridges, freeway decks and lamps on the way.
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
/** The compass azimuth, in degrees, of a horizontal direction. */
const azimuthOf = (dx, dy) => mod360(Math.atan2(dx, dy) / DEG);

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

export const ZEN_CLOUD_SPEED = 500.0;            // m/s
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
// A field that does not wrap has edges. Past this fraction of the way from
// the middle to an edge the flight is turned back toward the middle, on a
// circle that fits in half of what is left. That turn is never faster than
// ZEN_CLOUD_TURN_BACK_RATE; in a field too small for 500 m/s at that rate,
// the flight is slower instead.
export const ZEN_CLOUD_TURN_BACK_FROM = 0.7;
export const ZEN_CLOUD_TURN_BACK_RATE = 6.0;     // degrees per second

// --- over the city ----------------------------------------------------------

export const ZEN_CITY_SPEED = 8.0;               // m/s
// At an intersection: left with this probability, right with this
// probability, straight on with the rest. A way that is not a street is not
// offered, and the others share its probability.
export const ZEN_CITY_TURN_P = 0.1;
// The lowest and highest the flight goes, above the street.
export const ZEN_CITY_ALTITUDE_M = [4.0, 500.0];
// The radius of a turn from one street into another.
export const ZEN_CITY_TURN_RADIUS_M = 12.0;
// The most the altitude changes between two levels, as a fraction of the
// distance flown between them: inside the altitude range, and on the way
// into it from outside (zen started high over the city).
export const ZEN_CITY_CLIMB = 0.25;
export const ZEN_CITY_CLIMB_OUTSIDE = 1.0;
// How far every stretch of the route is kept from anything solid.
export const ZEN_CITY_CLEAR_M = 1.0;
// A stretch is checked this far past its end, level, so that the next one
// does not begin against an obstacle.
export const ZEN_CITY_LOOKAHEAD_M = 40.0;
// The view turns toward the route's direction with this time constant, and
// no faster than this rate.
export const ZEN_CITY_YAW_S = 0.5;
export const ZEN_CITY_YAW_RATE = 60.0;           // degrees per second
// How far around the camera a street is looked for when zen starts, in
// blocks.
export const ZEN_CITY_SEARCH_BLOCKS = 64;

/**
 * A value that wanders. It eases from one level to the next along a running
 * coordinate u (seconds, or metres flown) and draws the next level when it
 * arrives: `draw(v, u)` is given the level just reached and where, and
 * returns [next level, where it is reached].
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

export class CloudZen {
  /** `camera` is the FlightCamera zen takes over, where it is now. */
  constructor(camera, random = Math.random) {
    const span = () => ZEN_DRIFT_S[0] + random() * (ZEN_DRIFT_S[1] - ZEN_DRIFT_S[0]);
    const [low, high] = cloudBand(camera.bmin, camera.bmax);
    this.band = [low, high];
    this.time = 0.0;
    this.heading = camera.azimuth;
    this.speed = ZEN_CLOUD_SPEED;
    // In a field that does not wrap: the radius of the turn back toward the
    // middle, and the speed that radius can be flown at.
    this.turnBackRadius = null;
    if (!camera.periodic) {
      const half = Math.min(camera.bmax[0] - camera.bmin[0],
                            camera.bmax[1] - camera.bmin[1]) / 2.0;
      this.turnBackRadius = 0.5 * (1.0 - ZEN_CLOUD_TURN_BACK_FROM) * half;
      this.speed = Math.min(
        ZEN_CLOUD_SPEED, ZEN_CLOUD_TURN_BACK_RATE * DEG * this.turnBackRadius);
    }
    // Degrees per second, positive clockwise seen from above. It starts at
    // zero: the flight leaves along the heading it had.
    const maxRate = 360.0 / ZEN_CLOUD_CIRCLE_S;
    this.turnRate = new Drift(0.0, 0.0, (_, t) =>
      [(2.0 * random() - 1.0) * maxRate, t + span()]);
    // Each new level is a height drawn evenly from the band, approached no
    // faster than the climb limit allows.
    this.altitude = new Drift(0.0, camera.position[2], (z, t) => {
      const dt = span();
      const most = ZEN_CLOUD_CLIMB * this.speed * dt;
      const target = low + random() * (high - low);
      return [z + clamp(target - z, -most, most), t + dt];
    });
  }

  step(dt, camera) {
    this.time += dt;
    const speed = this.speed * ease(this.time / ZEN_EASE_IN_S);
    let rate = this.turnRate.at(this.time);
    if (this.turnBackRadius !== null) rate = this._turnBack(camera, rate);
    this.heading = mod360(this.heading + rate * dt);
    const f = directionFromAzimuthElevation(this.heading, 0.0);
    const d = speed * dt;
    camera.displace(f[0] * d, f[1] * d,
                    this.altitude.at(this.time) - camera.position[2]);
    camera.azimuth = this.heading;
    levelLook(camera, dt);
  }

  /**
   * In a field that does not wrap, the turn rate near an edge: the wander's
   * own in the middle of the field, and toward the middle of the field once
   * the flight is past ZEN_CLOUD_TURN_BACK_FROM of the way out, at the rate
   * that flies a circle of turnBackRadius.
   */
  _turnBack(camera, rate) {
    const { bmin, bmax, position: p } = camera;
    const centre = [0, 1].map((i) => 0.5 * (bmin[i] + bmax[i]));
    const out = Math.max(
      ...[0, 1].map((i) => Math.abs(p[i] - centre[i]) / (0.5 * (bmax[i] - bmin[i]))));
    const from = ZEN_CLOUD_TURN_BACK_FROM;
    if (out <= from) return rate;
    // Wholly the turn back by half way through the margin.
    const weight = ease((out - from) / (0.5 * (1.0 - from)));
    const most = this.speed / this.turnBackRadius / DEG;
    const off = fold180(
      azimuthOf(centre[0] - p[0], centre[1] - p[1]) - this.heading);
    const home = most * clamp(off / 30.0, -1.0, 1.0);
    return rate * (1.0 - weight) + home * weight;
  }
}

// --- the street grid --------------------------------------------------------
//
// The streets run along the block grid's lines. A NODE is a grid corner,
// (i, j) at (i * cell, j * cell); a street joins two neighbouring nodes. The
// tile's street word (tools/city_lots.py) says which do: bit 0 of block
// (i, j) is the street up its -x edge, from node (i, j) to (i, j + 1), and
// bit 2 the street along its -y edge, from node (i, j) to (i + 1, j).

// The four ways out of a node, counter-clockwise: +x, +y, -x, -y.
const WAYS = [[1, 0], [0, 1], [-1, 0], [0, -1]];
const back = (way) => (way + 2) % 4;

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
   * Raises when no street is near enough to follow.
   */
  constructor(camera, lots, random = Math.random) {
    this.lots = lots;
    this.tile = lots.tile;
    this.random = random;
    this.time = 0.0;
    this.heading = camera.azimuth;
    // The camera in the city frame. It is the camera's own surfacePosition,
    // so the route is laid in the frame the renderer and the collider fold
    // the city in (see _follow).
    this.pos = [camera.surfacePosition[0], camera.surfacePosition[1]];
    // The route: lines and quarter-circle arcs end to end, each with the
    // metres flown at its start and end (s0, s1). `s` is how far along it
    // the flight is.
    this.pieces = [];
    this.s = 0.0;
    // The route ends at `end`, on the street that leads to `node` by `way`.
    this.end = null;
    this.node = null;
    this.way = 0;
    this.altitude = null;
    // How many stretches were flown level because no altitude was clear.
    this.blocked = 0;
    // While the way onto the nearest street is not clear, the flight rises
    // where it is and looks again (see step).
    this.rising = false;
    this._lookAt = 0.0;
    this._begin(camera);
  }

  // --- the route -------------------------------------------------------------

  /** Lay the route's first line, from the camera onto the nearest street. */
  _begin(camera) {
    const cell = this.tile.cellM;
    const p = this.pos;
    const street = this._nearestStreet(p);
    if (street === null) {
      throw new Error(
        `No street within ${ZEN_CITY_SEARCH_BLOCKS} blocks to follow.`);
    }
    const f = directionFromAzimuthElevation(camera.azimuth, 0.0);
    // Along the street, the way the camera is already facing.
    let way = street.way;
    let from = [street.i, street.j];
    if (f[0] * WAYS[way][0] + f[1] * WAYS[way][1] < 0.0) {
      from = [from[0] + WAYS[way][0], from[1] + WAYS[way][1]];
      way = back(way);
    }
    const u = WAYS[way];
    const node = [from[0] + u[0], from[1] + u[1]];
    const corner = [node[0] * cell, node[1] * cell];
    // Meet the street's centre line some way ahead of the nearest point on
    // it, so that the first line comes onto it at a shallow angle. Not past
    // the next node: what happens there is the route's to decide.
    const lead = clamp(1.5 * street.distance + 10.0, 10.0, 60.0);
    const reach = (corner[0] - street.point[0]) * u[0]
                + (corner[1] - street.point[1]) * u[1];
    const ahead = Math.min(lead, reach);
    const join = [street.point[0] + ahead * u[0], street.point[1] + ahead * u[1]];
    const z = camera.position[2];
    if (this.lots.sweep([p[0], p[1], z], [join[0], join[1], z],
                        ZEN_CITY_CLEAR_M) !== null) {
      this.rising = true;
      return;
    }
    this.rising = false;
    this.pieces = [];
    this.s = 0.0;
    this.end = [p[0], p[1]];
    this._line(join);
    this.node = node;
    this.way = way;
    this.altitude = new Drift(0.0, z, (z0, s0) => this._nextAltitude(z0, s0));
  }

  /** The street nearest a point: its node, way, nearest point and distance. */
  _nearestStreet(p) {
    const cell = this.tile.cellM;
    const ci = Math.round(p[0] / cell), cj = Math.round(p[1] / cell);
    let best = null;
    // Ring by ring outward, stopping once no further ring can hold a nearer
    // street than the one in hand.
    for (let ring = 0; ring <= ZEN_CITY_SEARCH_BLOCKS; ring++) {
      if (best !== null && (ring - 2) * cell > best.distance) break;
      for (let j = cj - ring; j <= cj + ring; j++) {
        for (let i = ci - ring; i <= ci + ring; i++) {
          if (Math.max(Math.abs(i - ci), Math.abs(j - cj)) !== ring) continue;
          for (const way of [0, 1]) {
            if (!streetFrom(this.tile, i, j, way)) continue;
            const u = WAYS[way];
            const along = clamp(
              (p[0] - i * cell) * u[0] + (p[1] - j * cell) * u[1], 0.0, cell);
            const point = [i * cell + along * u[0], j * cell + along * u[1]];
            const distance = Math.hypot(p[0] - point[0], p[1] - point[1]);
            if (best === null || distance < best.distance) {
              best = { i, j, way, point, distance };
            }
          }
        }
      }
    }
    return best;
  }

  get _routeEnd() {
    return this.pieces.length ? this.pieces[this.pieces.length - 1].s1 : 0.0;
  }

  /** Add a straight line from the route's end to `to`. */
  _line(to) {
    const from = this.end;
    const length = Math.hypot(to[0] - from[0], to[1] - from[1]);
    if (length > 1e-6) {
      const s0 = this._routeEnd;
      this.pieces.push({
        s0, s1: s0 + length, from: [from[0], from[1]], to: [to[0], to[1]],
        azimuth: azimuthOf(to[0] - from[0], to[1] - from[1]),
      });
    }
    this.end = [to[0], to[1]];
  }

  /**
   * Carry the route through its next node: straight across, or round a
   * quarter circle into the street to the left or right, or — at a dead end
   * — back the way it came.
   */
  _extend() {
    const cell = this.tile.cellM;
    const u = WAYS[this.way];
    const corner = [this.node[0] * cell, this.node[1] * cell];
    const reach = (corner[0] - this.end[0]) * u[0]
                + (corner[1] - this.end[1]) * u[1];
    const next = this._choose(this.node, this.way);
    if (next === this.way) {
      this._line(corner);
    } else if (next === back(this.way)) {
      // A dead end. The route turns round where it stands, short of the
      // node; the view comes round at ZEN_CITY_YAW_RATE.
      const stop = Math.max(reach - ZEN_CITY_TURN_RADIUS_M, 0.0);
      this._line([this.end[0] + stop * u[0], this.end[1] + stop * u[1]]);
      this.node = [this.node[0] - u[0], this.node[1] - u[1]];
      this.way = next;
      return;
    } else {
      const v = WAYS[next];
      const r = Math.min(ZEN_CITY_TURN_RADIUS_M, reach);
      this._line([corner[0] - r * u[0], corner[1] - r * u[1]]);
      if (r > 1e-6) {
        const s0 = this._routeEnd;
        // Counter-clockwise when the new way is to the left of the old.
        const turn = u[0] * v[1] - u[1] * v[0] > 0.0 ? 1.0 : -1.0;
        this.pieces.push({
          s0, s1: s0 + 0.5 * Math.PI * r, radius: r, turn,
          centre: [corner[0] - r * u[0] + r * v[0],
                   corner[1] - r * u[1] + r * v[1]],
          // The angle, from the centre, of the point the arc starts at.
          angle: Math.atan2(-v[1], -v[0]),
        });
      }
      this.end = [corner[0] + r * v[0], corner[1] + r * v[1]];
    }
    this.node = [this.node[0] + WAYS[next][0], this.node[1] + WAYS[next][1]];
    this.way = next;
  }

  /**
   * Which way to leave `node`, having arrived by `way`: straight, left or
   * right by ZEN_CITY_TURN_P, among the ways that are streets. A way whose
   * street ends at a dead end is passed over while another is open.
   * With none open, the way back.
   */
  _choose(node, way) {
    const open = (n, w) => streetFrom(this.tile, n[0], n[1], w);
    const leadsOn = (w) => {
      const m = [node[0] + WAYS[w][0], node[1] + WAYS[w][1]];
      return [0, 1, 2, 3].some((e) => e !== back(w) && open(m, e));
    };
    const p = ZEN_CITY_TURN_P;
    let ways = [[way, 1.0 - 2.0 * p], [(way + 1) % 4, p], [(way + 3) % 4, p]]
      .filter(([w]) => open(node, w));
    if (ways.length === 0) return back(way);
    const onward = ways.filter(([w]) => leadsOn(w));
    if (onward.length > 0) ways = onward;
    let pick = this.random() * ways.reduce((sum, [, weight]) => sum + weight, 0.0);
    for (const [w, weight] of ways) {
      pick -= weight;
      if (pick < 0.0) return w;
    }
    return ways[ways.length - 1][0];
  }

  /** The route's [x, y, azimuth] at `s` metres along it. */
  _at(s) {
    while (this._routeEnd < s) this._extend();
    let piece = this.pieces[0];
    for (const candidate of this.pieces) {
      piece = candidate;
      if (s < candidate.s1) break;
    }
    const along = clamp(s - piece.s0, 0.0, piece.s1 - piece.s0);
    if (piece.radius === undefined) {
      const x = along / (piece.s1 - piece.s0);
      return [piece.from[0] + x * (piece.to[0] - piece.from[0]),
              piece.from[1] + x * (piece.to[1] - piece.from[1]),
              piece.azimuth];
    }
    const a = piece.angle + piece.turn * along / piece.radius;
    return [piece.centre[0] + piece.radius * Math.cos(a),
            piece.centre[1] + piece.radius * Math.sin(a),
            azimuthOf(-piece.turn * Math.sin(a), piece.turn * Math.cos(a))];
  }

  // --- the altitude ----------------------------------------------------------

  /**
   * The next altitude level after `z0` at `s0` metres along the route, and
   * where it is reached: [z1, s1].
   *
   * A level is a height drawn evenly from ZEN_CITY_ALTITUDE_M, approached
   * no faster than the climb limit, and it is taken only if the stretch
   * that reaches it is clear (_clear). Several are tried; then flying
   * level, then small steps up and down; then all of that over shorter
   * stretches. If nothing is clear the stretch is flown level and counted
   * in `blocked`, and the collider (collide.js) is what stops the flight.
   */
  _nextAltitude(z0, s0) {
    const [low, high] = ZEN_CITY_ALTITUDE_M;
    const inside = z0 >= low && z0 <= high;
    const random = this.random;
    const drift = ZEN_DRIFT_S[0] + random() * (ZEN_DRIFT_S[1] - ZEN_DRIFT_S[0]);
    for (const span of [ZEN_CITY_SPEED * drift, 120.0, 60.0]) {
      const most = span * (inside ? ZEN_CITY_CLIMB : ZEN_CITY_CLIMB_OUTSIDE);
      const within = (z) => (inside ? clamp(z, low, high) : z);
      const levels = [];
      for (let k = 0; k < 8; k++) {
        const target = low + random() * (high - low);
        levels.push(z0 + clamp(target - z0, -most, most));
      }
      levels.push(z0);
      for (const step of [5.0, 10.0, 20.0, 40.0, 80.0, 160.0]) {
        if (step > most) break;
        levels.push(within(z0 + step), within(z0 - step));
      }
      for (const z1 of levels) {
        if (this._clear(s0, s0 + span, z0, z1)) return [z1, s0 + span];
      }
    }
    this.blocked += 1;
    return [z0, s0 + 60.0];
  }

  /**
   * Whether the route from s0 to s1, easing from altitude z0 to z1, and
   * then ZEN_CITY_LOOKAHEAD_M further at z1, keeps ZEN_CITY_CLEAR_M from
   * everything solid.
   */
  _clear(s0, s1, z0, z1) {
    const far = s1 + ZEN_CITY_LOOKAHEAD_M;
    while (this._routeEnd < far) this._extend();
    // Short enough chords that neither an arc nor the altitude's curve
    // strays far from them.
    const lineStep = Math.min(16.0, (s1 - s0) / 12.0);
    const point = (s) => {
      const [x, y] = this._at(s);
      return [x, y, z0 + (z1 - z0) * ease((s - s0) / (s1 - s0))];
    };
    let s = s0;
    let p = point(s);
    for (const piece of this.pieces) {
      if (piece.s1 <= s0 || piece.s0 >= far) continue;
      const step = piece.radius === undefined ? lineStep : 3.0;
      const stop = Math.min(piece.s1, far);
      while (s < stop) {
        s = Math.min(s + step, stop);
        const q = point(s);
        if (this.lots.sweep(p, q, ZEN_CITY_CLEAR_M) !== null) return false;
        p = q;
      }
    }
    return true;
  }

  // --- flying it ---------------------------------------------------------------

  /**
   * Take the camera's position as it now is, and keep the route in the
   * camera's frame: the camera's surfacePosition folds at the tile, and
   * when it does the route is moved by the same whole tile.
   */
  _follow(camera) {
    const extent = this.tile.n * this.tile.cellM;
    const sp = camera.surfacePosition;
    for (const i of [0, 1]) {
      const tiles = Math.round((sp[i] - this.pos[i]) / extent);
      if (tiles === 0) continue;
      const shift = tiles * extent;
      for (const piece of this.pieces) {
        for (const key of ["from", "to", "centre"]) {
          if (piece[key]) piece[key][i] += shift;
        }
      }
      if (this.end) this.end[i] += shift;
      if (this.node) this.node[i] += tiles * this.tile.n;
    }
    this.pos = [sp[0], sp[1]];
  }

  step(dt, camera) {
    this.time += dt;
    this._follow(camera);
    levelLook(camera, dt);
    const speed = ZEN_CITY_SPEED * ease(this.time / ZEN_EASE_IN_S)
                * camera.control;
    if (this.rising) {
      camera.displace(0.0, 0.0, speed * dt);
      if (this.time >= this._lookAt) {
        this._lookAt = this.time + 0.5;
        this._begin(camera);
      }
      return;
    }
    this.s += speed * dt;
    const [x, y, azimuth] = this._at(this.s);
    // Onto the route. A crash (collide.js) leaves the camera off it; it
    // comes back at a bounded speed, not in one frame.
    let dx = x - this.pos[0];
    let dy = y - this.pos[1];
    let dz = this.altitude.at(this.s) - camera.position[2];
    const off = Math.hypot(dx, dy, dz);
    const most = 3.0 * ZEN_CITY_SPEED * dt;
    if (off > most) {
      dx *= most / off; dy *= most / off; dz *= most / off;
    }
    camera.displace(dx, dy, dz);
    this.pos[0] += dx;
    this.pos[1] += dy;
    const turn = fold180(azimuth - this.heading)
               * (1.0 - Math.exp(-dt / ZEN_CITY_YAW_S));
    const mostTurn = ZEN_CITY_YAW_RATE * dt;
    this.heading = mod360(this.heading + clamp(turn, -mostTurn, mostTurn));
    camera.azimuth = this.heading;
    // What is behind is done with. The piece under the flight stays.
    while (this.pieces.length > 1 && this.pieces[0].s1 < this.s - 1.0) {
      this.pieces.shift();
    }
  }
}
