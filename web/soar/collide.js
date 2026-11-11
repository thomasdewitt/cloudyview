// The night city is solid: flight collision and what a collision does to you.
//
// Before this, flying into a building made it vanish. The renderer traces
// every building from the outside, so a camera inside one sees straight
// through its walls; nothing stopped the camera getting there. Thomas
// (2026-09-23): "the buildings currently just disappear when you fly into
// them, but probably the paper airplane should be rocked, like you loose
// control for a sec and it falls a bit."
//
// So this module does two things.
//
//   STOP. Every frame, the motion the flight made is swept against the city
//   (citylots.js and cityprops.js, the renderer's own buildings and what
//   stands among them, mirrored on the CPU) and cut
//   back where it would have come within a body's radius of something.
//   Only the component INTO the surface is removed, so a glancing touch
//   slides along the wall rather than sticking to it.
//
//   KNOCK. A hard enough arrival starts an EPISODE, about a second of lost
//   control: the flight bounces back off the surface, the controls go
//   nearly dead and come back over the second half, the aeroplane drops
//   five to fifteen metres, the view is shaken, and the flyer tumbles
//   (Flyer.knock). A landing — a roof or the street, hit from above — is a
//   shorter stagger with a hop and no drop, since there is nowhere to fall.
//
// --- What collides: the camera, the flyer, and the line between them -----
//
// Two bodies have to stay out of the buildings and they fail differently.
// The CAMERA inside a building is the original bug: the building disappears.
// The FLYER inside one is subtler and as bad — flyer.js draws it over the
// finished frame with its own depth buffer, so a dart 2.6 m ahead of a
// camera parked against a wall is drawn IN FRONT of the wall it has gone
// through, which reads as the wall not being there. And the plane is what
// hits first: it flies ahead of the camera by its `distance`, so a camera-
// only test would let the dart bury itself before anything happened.
//
// So three segments are swept each frame, and the first that is blocked is
// resolved, until none is:
//
//   the camera's path      last frame's eye -> this frame's
//   the flyer's path       last frame's plane -> this frame's (this includes
//                          the plane swinging round when you look round —
//                          stopped by a wall like anything else, though
//                          only the flight's own motion counts as a crash)
//   the tether             this frame's eye -> this frame's plane: nothing
//                          may stand between you and the thing you are
//                          following, or it is drawn over that thing
//
// Every correction moves camera and flyer together — the flyer is placed off
// the camera, so moving one without the other would just be undone next
// frame. With no flyer up, the camera is the only body.
//
// --- The frame ------------------------------------------------------------
//
// All of this happens in the CITY frame: x, y are the camera's
// surfacePosition (the tile frame raymarch.wgsl builds the city in, see
// FlightCamera), z is world z. Positions between frames are unwrapped
// against the previous one, since a frame moves at most a few hundred
// metres and the tile is 92 km across.
//
// --- Order within a frame (viewer/loop.js) ----------------------------------
//
//   collider.advance(dt, camera)   the episode's clock: controls damped, the
//                                  bounce and the fall applied, shake taken
//                                  back off so the move reads the true look
//   camera.move(dt)                the player's move, damped
//   flyer.update(dt, camera)       the flyer placed off the camera
//   collider.resolve(dt, camera, flyer)
//                                  sweep, clip, maybe start an episode, and
//                                  put this frame's shake on for the render
//
// Everything that is a function of time is a closed form of the episode
// clock and each frame applies the difference between its values at the
// frame's two ends, so the total bounce, drop and shake do not depend on
// the frame rate.
//
// The day scenes have none of this: the viewer builds a collider for a city
// scene and for nothing else, and every hook it adds elsewhere is inert
// without one.

"use strict";

// --- Bodies -------------------------------------------------------------------
// Each body is a sphere, and nothing comes inside its radius. The camera's
// is the renderer's near-field comfort, not a hitbox; the flyer's is about
// its own size about the point it is placed at, so a dart stopped by a wall
// touches it with its nose (the sheet is 0.3 m long). Both must stay under
// citylots.js's half-metre limit.
const CAMERA_RADIUS_M = 0.35;
const FLYER_RADIUS_M = 0.20;
// How far past a surface a body is backed out when it is stopped: enough
// that next frame's sweep starts clear of the surface it is resting on.
const CLEARANCE_M = 0.01;

// --- Impact ---------------------------------------------------------------------
// Speed INTO the surface below which a contact is a scrape — clipped, no
// episode. Anything more is a crash, graded by severity, which is 1 at
// IMPACT_FULL_SPEED and above: 25 m/s is a little over the city's opening
// speed (CITY_DEFAULT_SPEED, 20 m/s), so flying straight into a wall at the
// speed the session starts at is very nearly the worst it gets.
const IMPACT_MIN_SPEED = 3.0;    // m/s
const IMPACT_FULL_SPEED = 25.0;  // m/s
// The bounce: a fraction of the arrival speed, back along the normal,
// dying away with time constant BOUNCE_TAU and tapered to rest by the
// episode's end (Episode.displacement) — so it carries you a little under
// v * RESTITUTION * BOUNCE_TAU metres back in total (2.7 m off a 20 m/s
// wall; 1.75 m until 2026-11, when Thomas asked for the crash to rock the
// flight harder). Capped, because at 500 m/s a proportional bounce would
// throw you across a block.
const RESTITUTION = 0.50;
const BOUNCE_MAX = 14.0;         // m/s
const BOUNCE_TAU = 0.30;         // s

// --- The episode --------------------------------------------------------------
// [severity 0, severity 1]. A wall is the full loss of control; a landing is
// a stumble.
const WALL_EPISODE_S = [0.75, 1.15];
const WALL_DROP_M = [5.0, 15.0];
const LANDING_EPISODE_S = [0.35, 0.60];
// The controls: nearly dead for the first CONTROL_RETURN of the episode,
// then eased back to full by its end.
const CONTROL_FLOOR = 0.08;
const CONTROL_RETURN = 0.45;

// --- The shake ----------------------------------------------------------------
// A lurch of a few degrees at two or three hertz, dying away; exactly zero
// at the episode's end, so the look you had is the look you get back.
// Landings shake mostly in elevation. (It was a degree or two at four or
// five hertz until 2026-11 — a buzz, which Thomas found too gentle: "Paper
// airplane could be a bit more 'rocked' when it hits a building". A slower
// swing reads as the craft being knocked rather than the camera vibrating.)
const SHAKE_AZ_DEG = 3.4;
const SHAKE_EL_DEG = 2.6;
const SHAKE_AZ_HZ = 3.1;
const SHAKE_EL_HZ = 2.3;
const SHAKE_DECAY = 2.4;         // e-folds over the episode
const LANDING_SHAKE_AZ = 0.35;

// Blocked segments resolved per frame before giving up and stopping dead.
// A slide needs one, a corner two; four is a corner with the flyer's tether
// also against something.
const MAX_CLIPS = 4;
const MAX_DT = 0.1;              // s — the viewer's MAX_SIM_TIMESTEP

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const lerp = (a, b, t) => a + (b - a) * t;
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function smoothstep(lo, hi, x) {
  const t = clamp((x - lo) / (hi - lo), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

/**
 * One loss of control. Its clock runs from the impact; everything it does
 * to the flight is a closed form of that clock.
 */
class Episode {
  constructor({ severity, landing, bounce }) {
    this.severity = severity;
    this.landing = landing;
    this.t = 0.0;
    this.duration = lerp(...(landing ? LANDING_EPISODE_S : WALL_EPISODE_S),
                         severity);
    this.drop = landing ? 0.0 : lerp(...WALL_DROP_M, severity);
    this.bounce = bounce;               // m/s at the impact, off the surface
    // The drop's progress when a roof or the street stopped it, or null
    // while it is still falling.
    this.landedAt = null;
  }

  get done() { return this.t >= this.duration; }

  /** Player authority at episode time t, 0..1. */
  control(t) {
    const u = t / this.duration;
    return lerp(CONTROL_FLOOR, 1.0, smoothstep(CONTROL_RETURN, 1.0, u));
  }

  /**
   * Fraction of the drop fallen by time t: accelerating away from the
   * impact like anything let go of, then easing out as the aeroplane
   * recovers — zero vertical speed at both ends, so neither the knock nor
   * the recovery is a step.
   */
  _fallen(t) {
    const u = clamp(t / this.duration, 0.0, 1.0);
    const g = u * u * (3.0 - 2.0 * u);
    return this.landedAt === null ? g : Math.min(g, this.landedAt);
  }

  /** The bounce and the fall between episode times t0 and t1 (metres). */
  displacement(t0, t1) {
    // The bounce's velocity, b (e^(-t/tau) - E) / (1 - E) with
    // E = e^(-T/tau): the full bounce at the impact, exactly zero at the
    // episode's end. A plain exponential still carried 14-31% of itself
    // when a landing's 0.35-0.6 s ran out, and stopped dead there (a
    // severity-1 landing let go of 1.7 m/s upward in one frame).
    const E = Math.exp(-this.duration / BOUNCE_TAU);
    const k = (BOUNCE_TAU * (Math.exp(-t0 / BOUNCE_TAU)
                             - Math.exp(-t1 / BOUNCE_TAU))
               - E * (t1 - t0)) / (1.0 - E);
    const fall = this.drop * (this._fallen(t1) - this._fallen(t0));
    return [this.bounce[0] * k, this.bounce[1] * k, this.bounce[2] * k - fall];
  }

  /** [azimuth, elevation] shake in degrees at episode time t. */
  shake(t) {
    const u = clamp(t / this.duration, 0.0, 1.0);
    const env = this.severity * Math.exp(-SHAKE_DECAY * u) * (1.0 - u);
    const az = SHAKE_AZ_DEG * (this.landing ? LANDING_SHAKE_AZ : 1.0);
    return [az * env * Math.sin(2.0 * Math.PI * SHAKE_AZ_HZ * t),
            SHAKE_EL_DEG * env
              * Math.sin(2.0 * Math.PI * SHAKE_EL_HZ * t + 1.1)];
  }

  /**
   * A further contact while the episode runs: it is clipped like any other,
   * and it takes from the episode whatever pushes INTO that surface — the
   * bounce's component (off the far wall of a narrow street), and, if the
   * surface is a floor, the rest of the fall.
   */
  touch(normal) {
    const into = dot(this.bounce, normal);
    if (into < 0.0) this.bounce = sub(this.bounce, scale(normal, into));
    if (normal[2] > 0.7 && this.landedAt === null) {
      this.landedAt = this._fallen(this.t);
    }
  }
}

/**
 * Collision and its consequences, for one city.
 *
 * `city` is anything with the two queries citylots.js CityLots answers, in
 * the unwrapped city frame, for a sphere of radius r:
 *
 *   sweep(p0, p1, r, eye) -> {t, point, normal} | null  first surface met
 *   inside(p, r, eye)     -> {top} | null                a solid it overlaps
 *
 * `eye` is where this frame is seen from: the city folds its queries there,
 * as the renderer folds its camera (CityLots.sweep); a city that does not
 * fold may ignore it.
 *
 * and `tileExtentM` is the period the camera's surfacePosition folds at.
 */
export class CityCollider {
  constructor(city, tileExtentM) {
    if (typeof city?.sweep !== "function"
        || typeof city?.inside !== "function") {
      throw new Error("CityCollider needs a city with sweep() and inside().");
    }
    if (!(tileExtentM > 0)) {
      throw new Error(`CityCollider needs the tile extent; got ${tileExtentM}.`);
    }
    this.city = city;
    this.extent = tileExtentM;
    this.episode = null;
    // The last resolved (valid) positions, city frame. Null until the first
    // resolve seeds them.
    this._cam = null;
    this._flyer = null;
    this._flyerShown = null;   // the flyer _flyer is the position of
    this._epoch = null;
  }

  /** Whether an episode is running — the view has something to animate. */
  get active() { return this.episode !== null; }

  /**
   * Start of the frame, before the player's move: advance the episode,
   * damp the controls for the move, take the shake back off (so the move
   * and the flyer read the look the player actually has), and apply the
   * episode's own motion.
   */
  advance(dt, camera) {
    camera.setShake(0.0, 0.0);
    const e = this.episode;
    if (e === null || camera.epoch !== this._epoch) {
      camera.control = 1.0;
      return;
    }
    const t0 = e.t;
    const t1 = Math.min(t0 + clamp(dt, 0.0, MAX_DT), e.duration);
    e.t = t1;
    camera.control = e.control(0.5 * (t0 + t1));
    const [dx, dy, dz] = e.displacement(t0, t1);
    camera.displace(dx, dy, dz);
  }

  /**
   * End of the frame's motion: make it legal, react to what it hit, and put
   * this frame's shake on.
   *
   * `flyer` is the flyer that is up (placed off this frame's camera
   * already), or null. Returns the knock this frame started —
   * {severity, side, duration, landing}, what Flyer.knock takes — or null,
   * so a track being recorded can carry it (track.js).
   */
  resolve(dt, camera, flyer) {
    dt = clamp(dt, 1e-4, MAX_DT);
    if (camera.epoch !== this._epoch || this._cam === null) {
      this._seed(camera, flyer);
      return null;
    }
    const c0 = this._cam;
    const c1 = this._cityPosition(camera);
    const offset = flyer ? sub(flyer.position, camera.position) : null;
    // A flyer that was not the one up last frame — just switched on, or
    // the other species (the swift flies 4.6 m out, the dart 2.6) — starts
    // on its tether: there is no path from where the other one was, and
    // the tether sweep is what catches it if it came up through a wall.
    // Whatever it was doing when it was last drawn is over.
    if (flyer && flyer !== this._flyerShown) flyer.calm();
    const f0 = flyer ? (flyer === this._flyerShown ? this._flyer
                                                   : add(c0, offset))
                     : null;

    let corr = [0.0, 0.0, 0.0];
    let contact = null;
    let resolved = false;
    for (let i = 0; i < MAX_CLIPS && !resolved; i++) {
      const cam = add(c1, corr);
      const block = this._firstBlock(c0, cam, f0,
                                     offset && add(cam, offset));
      if (block === null) {
        resolved = true;
        break;
      }
      contact ??= block;
      // Back out along the normal to where the body touches the surface:
      // the tangential part of the motion survives, which is the slide.
      corr = add(corr, scale(block.normal,
        dot(block.normal, sub(block.point, block.end)) + CLEARANCE_M));
    }
    if (!resolved) {
      // Wedged — every clip uncovered another surface. Stay where the last
      // legal frame was; that position is clear by construction.
      corr = sub(c0, c1);
    }
    this._move(camera, flyer, corr);

    // The guarantee that matters most — the eye never ends a frame inside a
    // building — checked on the position the camera actually holds now,
    // after its own constrain().
    const cam = this._cityPosition(camera);
    if (this.city.sweep(c0, cam, CAMERA_RADIUS_M, cam) !== null
        || this.city.inside(cam, CAMERA_RADIUS_M) !== null) {
      this._move(camera, flyer, sub(c0, cam));
    }

    const knock = contact !== null ? this._contact(contact, dt, flyer) : null;
    if (this.episode !== null && this.episode.done) this.episode = null;
    if (this.episode === null) camera.control = 1.0;
    const [saz, sel] = this.episode ? this.episode.shake(this.episode.t)
                                    : [0.0, 0.0];
    camera.setShake(saz, sel);
    this._remember(camera, flyer);
    return knock;
  }

  // ---------------------------------------------------------------------------

  /**
   * The first blocked segment, as {point, normal, end, arrival}: the body's
   * centre where it meets a surface, the surface's outward normal, the
   * moving centre's end position (the one to back out), and the FLIGHT's
   * displacement over the frame — the camera's, whichever body touched.
   *
   * The flight's and not the body's because the flyer also moves when you
   * only look round: it hangs 2.6 m out along the view, so a quick glance
   * sweeps it through a few metres a frame. Beside a wall that glance
   * pushes it into the wall, and it is stopped there like anything else —
   * but it has not flown into anything, and a crash for looking at a
   * building would make hovering beside one impossible.
   */
  _firstBlock(c0, c1, f0, f1) {
    const arrival = sub(c1, c0);
    // The two paths are the same frame's motion, parameterized alike, so
    // the one blocked EARLIER is what the flight met first — and it decides
    // what kind of crash it was (the dart into a wall before the eye comes
    // down on a roof is a wall).
    // Every query of the frame folds at this frame's eye, as the renderer
    // that draws it does (CityLots.sweep).
    const cam = this.city.sweep(c0, c1, CAMERA_RADIUS_M, c1);
    const fly = f1 === null ? null
                            : this.city.sweep(f0, f1, FLYER_RADIUS_M, c1);
    if (cam !== null && (fly === null || cam.t <= fly.t)) {
      return { point: cam.point, normal: cam.normal, end: c1, arrival };
    }
    if (fly !== null) {
      return { point: fly.point, normal: fly.normal, end: f1, arrival };
    }
    // Both paths clear: the tether, which is a line held at one instant.
    const tether = f1 === null ? null
                               : this.city.sweep(c1, f1, FLYER_RADIUS_M, c1);
    if (tether !== null) {
      return { point: tether.point, normal: tether.normal, end: f1, arrival };
    }
    return null;
  }

  /**
   * A contact: a scrape, a crash, or something that happens mid-episode.
   * Returns the knock a crash hands the flyer, or null.
   */
  _contact(block, dt, flyer) {
    if (this.episode !== null) {
      this.episode.touch(block.normal);
      return null;
    }
    const n = block.normal;
    const speed = -dot(n, block.arrival) / dt;
    if (speed < IMPACT_MIN_SPEED) return null;
    const severity = clamp(speed / IMPACT_FULL_SPEED, 0.0, 1.0);
    const landing = n[2] > 0.7;
    const e = new Episode({
      severity, landing,
      bounce: scale(n, Math.min(RESTITUTION * speed, BOUNCE_MAX)) });
    this.episode = e;
    // Which way the aeroplane is flung: the side the wall turned it
    // towards, from the arrival's sense round the normal. Head-on has no
    // side worth reading and gets one anyway.
    const across = n[0] * block.arrival[1] - n[1] * block.arrival[0];
    const knock = { severity, landing, duration: e.duration,
                    side: across < 0.0 ? -1.0 : 1.0 };
    flyer?.knock(knock);
    return knock;
  }

  /**
   * Move camera and flyer together by a city-frame (= world) vector.
   *
   * The flyer moves by what the camera's world position ACTUALLY did, not
   * by `d`: a correction that carries the camera across the cloud domain's
   * edge comes back folded by a whole cloud period (constrain()), and a
   * flyer moved by `d` alone would be left a domain-width from the camera
   * it hangs off — drawn nowhere this frame, and next frame swept from
   * fifty kilometres away.
   */
  _move(camera, flyer, d) {
    if (d[0] === 0.0 && d[1] === 0.0 && d[2] === 0.0) return;
    const before = [...camera.position];
    camera.displace(d[0], d[1], d[2]);
    flyer?.shift(sub(camera.position, before));
  }

  /** The camera in the city frame, unwrapped against the last position. */
  _cityPosition(camera) {
    const sp = camera.surfacePosition;
    if (!sp) {
      throw new Error(
        "CityCollider needs a camera that keeps the tile frame " +
        "(FlightCamera surfaceTileExtent).");
    }
    const z = camera.position[2];
    if (this._cam === null) return [sp[0], sp[1], z];
    const t = this.extent;
    const near = (v, ref) => ref + (v - ref) - t * Math.round((v - ref) / t);
    return [near(sp[0], this._cam[0]), near(sp[1], this._cam[1]), z];
  }

  _remember(camera, flyer) {
    const sp = camera.surfacePosition;
    this._cam = [sp[0], sp[1], camera.position[2]];
    this._flyer = flyer
      ? add(this._cam, sub(flyer.position, camera.position)) : null;
    this._flyerShown = flyer;
    this._epoch = camera.epoch;
  }

  /**
   * A new start: the first frame, a reset, a map click. There is no path
   * to sweep, only a place to be — and if the place is inside a building
   * (a map click onto a tower), the place is its roof.
   */
  _seed(camera, flyer) {
    this.episode = null;
    camera.control = 1.0;
    camera.setShake(0.0, 0.0);
    flyer?.calm();
    this._cam = null;
    for (let i = 0; ; i++) {
      const p = this._cityPosition(camera);
      const solid = this.city.inside(p, CAMERA_RADIUS_M);
      if (solid === null) break;
      if (i === 4) {
        throw new Error(
          `The camera at city ${p.map((v) => v.toFixed(1))} is inside a ` +
          `building whose roof (${solid.top} m) does not get it out.`);
      }
      this._move(camera, flyer, [0.0, 0.0,
        solid.top + CAMERA_RADIUS_M + CLEARANCE_M - p[2]]);
    }
    // And the flyer off whatever it was placed into — a reset or a map
    // click beside a wall, looking at it — by backing both off along the
    // tether's blocking face, as resolve() would, as long as that move is
    // itself clear for the camera.
    for (let i = 0; flyer && i < MAX_CLIPS; i++) {
      const cam = this._cityPosition(camera);
      const fly = add(cam, sub(flyer.position, camera.position));
      const hit = this.city.sweep(cam, fly, FLYER_RADIUS_M);
      if (hit === null) break;
      const corr = scale(hit.normal,
        dot(hit.normal, sub(hit.point, fly)) + CLEARANCE_M);
      const to = add(cam, corr);
      if (this.city.sweep(cam, to, CAMERA_RADIUS_M) !== null
          || this.city.inside(to, CAMERA_RADIUS_M, cam) !== null) break;
      this._move(camera, flyer, corr);
    }
    this._remember(camera, flyer);
  }
}
