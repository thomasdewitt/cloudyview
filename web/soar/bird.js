// The bird: a common swift, the organic one of soar's two flying subjects.
//
// The flight behaviour began as a port of the desktop's bird.py; the geometry
// and shading did not. `birdmesh.js` and `bird.wgsl` rebuilt the animal from
// anatomy and are native to this build — unlike raymarch.wgsl, which crossed
// over verbatim.
//
// A common swift at its real size — procedural mesh, no assets — that flies a
// few metres ahead of and below the camera. It flaps (faster when slow,
// tucking into a glide above a speed threshold), banks into turns, pitches
// with climbs and descents, and bobs with its own wingbeat. All state is
// exponentially smoothed so it feels alive rather than bolted to the camera.
//
// Everything that is not specific to being a bird — the GPU resources, the
// camera-relative matrices, the uniform block, the pass, the placement ahead
// of the camera — lives in flyer.js and is shared with the paper dart. What
// is left here is the swift: its wingbeat, and how a bird banks.
//
// `setStatic(camera, t, overrides)` replaces `update` for a deterministic
// pose (offscreen stills), matching Bird.set_static.

"use strict";

import { Flyer, attitudeFrame, clamp, mod2pi, smoothstep,
         TWO_PI, DEG } from "./flyer.js";
import { buildBirdMesh } from "./birdmesh.js";

const SHADER_URL = new URL("./bird.wgsl", import.meta.url);

// --- Placement (metres, camera-relative) -----------------------------------
//
// The bird is life-size. The old one was scaled to a 1.8 m wingspan and flown
// 8.5 m out, which subtended twelve degrees — a swift the size of an
// albatross, a quarter of the way across the screen. A real one at 4.6 m
// subtends five, which is about ninety pixels at 1280 across a 60 degree
// view: small enough to be a bird, large enough that the primaries separate.
const DISTANCE = 4.6;         // ahead of the camera along the smoothed view direction
const DROP = 1.15;            // below the view center (screen-space, via camera up)
const SCALE = 1.0;            // life size: 0.40 m span, 0.165 m bill to tail

// --- Animation --------------------------------------------------------------
const FLAP_AMPLITUDE = 0.62;  // rad of wingtip rotation about the body axis
const REST_DIHEDRAL = 0.10;   // rad, wings-slightly-raised carry angle while flapping
const GLIDE_DIHEDRAL = 0.18;  // rad, the stiff shallow V of a glide
// A swift's wingbeat is mostly hand. The wrist adds its own flex a quarter
// cycle behind the shoulder, which is what makes the stroke look driven
// rather than waved, and the hand supinates on the upstroke so it slices.
const WRIST_FLEX = 0.30;      // rad, extra bend carried by the hand alone
const WRIST_LAG = 1.15;       // rad of phase, hand behind shoulder
const HAND_TWIST = 0.34;      // rad of supination at the top of the upstroke
const TAIL_SPREAD_MAX = 0.55; // fraction of extra width in a hard turn

const FLAP_HZ_SLOW = 4.2;     // wingbeat when hovering / slow
const FLAP_HZ_FAST = 2.2;     // wingbeat just below the glide threshold
const GLIDE_LO = 70.0;        // m/s: flap starts fading into a glide
const GLIDE_HI = 110.0;       // m/s: fully tucked into the glide
const BOB_AMPLITUDE = 0.10;   // m, body bob coupled to the wingbeat
const IDLE_BOB = 0.05;        // m, slow ambient bob so a parked bird still breathes
const IDLE_BOB_HZ = 0.45;

// --- Feel (exponential smoothing time constants, seconds) -------------------
const TAU_HEADING = 0.22;     // view/heading lag: the bird swings through turns
const TAU_BANK = 0.30;
const TAU_PITCH = 0.35;
const TAU_SPEED = 0.30;
const TAU_FLAP_AMP = 0.45;    // blend between flapping and glide

const BANK_PER_DEG_S = 0.40;  // deg of roll per deg/s of heading rate
const BANK_MAX = 50.0;        // deg
const PITCH_MAX = 25.0;       // deg
const BODY_PITCH = 8.0;       // deg, resting nose-up attitude (slow-flight posture;
                              // also presents more wing area to the from-below view)

/**
 * How strongly the sun comes through a backlit feather vane, and how much sky
 * the oiled plumage catches at grazing angles. Both are the knobs to reach for
 * first if the bird sits wrong against a particular sky.
 */
export class Bird extends Flyer {
  static TRANSMISSION_GAIN = 0.85;
  static SHEEN_GAIN = 0.55;

  constructor(device, resources) {
    super(device, resources, {
      label: "bird",
      shaderUrl: SHADER_URL,
      mesh: buildBirdMesh({ scale: SCALE }),
      distance: DISTANCE,
      drop: DROP,
    });

    this.flapPhase = 0.0;              // rad
    this.flapAmp = FLAP_AMPLITUDE;
    this.flapAngle = REST_DIHEDRAL;    // rad, current wing angle
    this.wristFlex = 0.0;              // rad, extra bend carried by the hand
    this.handTwist = 0.0;              // rad, supination of the hand
    this.tailSpread = 0.0;             // 0 closed, 1 fully fanned
  }

  /** Local->world rotation columns for the current attitude. */
  _frame() {
    return attitudeFrame(this.heading, this.pitch + BODY_PITCH, this.bank);
  }

  /** The slots bird.wgsl reads as its own: flap angle, wrist, twist, tail. */
  _species(u) {
    u[59] = this.flapAngle;
    u[72] = this.wristFlex; u[73] = this.handTwist;
    u[74] = this.tailSpread; u[75] = 0.0;
  }

  // ------------------------------------------------------------------
  // Animation
  // ------------------------------------------------------------------

  /** Advance the wingbeat; returns the body bob (m). */
  _flap(dt) {
    const glide = smoothstep(this._speed, GLIDE_LO, GLIDE_HI);
    const ampTarget = FLAP_AMPLITUDE * (1.0 - glide);
    const k = 1.0 - Math.exp(-dt / TAU_FLAP_AMP);
    this.flapAmp += (ampTarget - this.flapAmp) * k;

    const v = Math.min(this._speed / GLIDE_LO, 1.0);
    const hz = FLAP_HZ_SLOW + (FLAP_HZ_FAST - FLAP_HZ_SLOW) * v;
    this.flapPhase = mod2pi(this.flapPhase + TWO_PI * hz * dt);

    const center = REST_DIHEDRAL + (GLIDE_DIHEDRAL - REST_DIHEDRAL) * glide;
    this.flapAngle = center + this.flapAmp * Math.sin(this.flapPhase);

    const ampFrac = this.flapAmp / FLAP_AMPLITUDE;

    // The hand lags the shoulder and adds its own flex, and supinates through
    // the upstroke — nose-down as the wing rises, so it slices rather than
    // pushing the bird back down. Both fade out with the flap into a glide,
    // where the wing goes stiff.
    this.wristFlex = WRIST_FLEX * ampFrac * Math.sin(this.flapPhase - WRIST_LAG);
    this.handTwist = HAND_TWIST * ampFrac * Math.cos(this.flapPhase);

    return BOB_AMPLITUDE * ampFrac * Math.sin(this.flapPhase - 1.2)
         + IDLE_BOB * Math.sin(TWO_PI * IDLE_BOB_HZ * this._clock);
  }

  /**
   * Per-frame dynamics from the live camera. Everything else — banking,
   * pitch, glide/flap blend, bob — is derived and smoothed here.
   */
  update(dt, camera) {
    dt = clamp(dt, 1e-4, 0.1);
    this._clock += dt;

    const tracked = this._track(dt, camera, {
      tauHeading: TAU_HEADING, tauSpeed: TAU_SPEED });
    if (tracked === null) {
      // First frame: snap, no dynamics.
      this._place(camera.position, this._flap(dt));
      return;
    }
    const { daz, vel } = tracked;

    // Bank into turns: roll follows the smoothed heading rate.
    const headingRate = daz / dt;   // deg/s
    const bankTarget = clamp(headingRate * BANK_PER_DEG_S, -BANK_MAX, BANK_MAX);
    const kb = 1.0 - Math.exp(-dt / TAU_BANK);
    this.bank += (bankTarget - this.bank) * kb;

    // Pitch with climb/descent.
    const hSpeed = Math.max(Math.hypot(vel[0], vel[1]), 15.0);
    const pitchTarget = clamp(
      Math.atan2(this._vz, hSpeed) / DEG, -PITCH_MAX, PITCH_MAX);
    const kp = 1.0 - Math.exp(-dt / TAU_PITCH);
    this.pitch += (pitchTarget - this.pitch) * kp;

    // The tail fans in a turn — it is the rudder and the airbrake, and a
    // swift's fork opens visibly whenever it changes direction.
    const spreadTarget = TAIL_SPREAD_MAX
      * Math.min(1.0, Math.abs(this.bank) / BANK_MAX);
    this.tailSpread += (spreadTarget - this.tailSpread) * kb;

    this._place(camera.position, this._flap(dt));
  }

  /**
   * Deterministic pose for offscreen rendering (no dynamics).
   *
   * The bird cruises along the view direction: steady wingbeat at phase
   * 2*pi*FLAP_HZ_FAST*t, climbing/descending with the camera elevation as if
   * flying that trajectory, unless `bank`/`pitch` (deg) or `flapPhase` (rad)
   * override it.
   */
  setStatic(camera, t = 0.0, { bank = null, pitch = null, flapPhase = null } = {}) {
    this.heading = camera.azimuth;
    this.viewElevation = camera.elevation;
    this.bank = bank === null ? 0.0 : Number(bank);
    this.pitch = pitch === null
      ? clamp(0.6 * camera.elevation, -PITCH_MAX, PITCH_MAX)
      : Number(pitch);
    this.flapAmp = FLAP_AMPLITUDE;
    this.flapPhase = flapPhase === null
      ? mod2pi(TWO_PI * FLAP_HZ_FAST * t)
      : Number(flapPhase);
    this.flapAngle = REST_DIHEDRAL + this.flapAmp * Math.sin(this.flapPhase);
    this.wristFlex = WRIST_FLEX * Math.sin(this.flapPhase - WRIST_LAG);
    this.handTwist = HAND_TWIST * Math.cos(this.flapPhase);
    this.tailSpread = 0.0;
    this._place(camera.position, BOB_AMPLITUDE * Math.sin(this.flapPhase - 1.2));
  }
}
