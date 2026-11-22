// The dart: a thrown sheet of A4, the inorganic one of soar's two flyers.
//
// Shares everything structural with the bird through flyer.js. What is here
// is how a paper aeroplane moves, and that is where it has to differ, because
// the two are only really told apart in motion. A swift is powered: it beats,
// and every beat is a decision. A dart is a thrown object with no engine and
// almost no damping, and everything it does after it leaves your hand is a
// consequence of that:
//
//   - It flies STEADY. In straight flight it holds its attitude and its
//     height: no porpoising, no rocking. What moves it is what the flight
//     does, a turn, a climb, a crash.
//   - The WOBBLE. A dart's keel is its only fin and it is a small one, so
//     directional stability is marginal and every turn rings a lightly damped
//     roll-yaw oscillation that takes a second or two to settle. A bird damps
//     this out with its tail inside a single beat.
//   - FLUTTER. At speed the unsupported trailing edge buzzes. It is a few
//     millimetres at twenty hertz, it is the only fast motion on the
//     aircraft, and it is entirely absent from a bird.
//   - It flies slightly WING DOWN, always, because dartmesh.js folds it
//     slightly crooked. The geometry and the trim agree on which way, which
//     is the sort of thing nobody notices and everybody believes.

"use strict";

import { Flyer, clamp, mod2pi, TWO_PI, DEG } from "./flyer.js";
import { buildDartMesh, DEFAULT_WEAR } from "./dartmesh.js";

const SHADER_URL = new URL("./dart.wgsl", import.meta.url);

// --- Placement (metres, camera-relative) -----------------------------------
//
// A dart is 0.174 m across where the swift is 0.40, so at the bird's 4.6 m it
// would subtend two degrees and read as a speck. It is flown closer instead —
// which is also the honest framing, since you are behind something you threw,
// not following a bird — and at 2.6 m the two subtend within a degree of each
// other. Keeping them the same size on screen is what lets the choice between
// them be a choice about character rather than about composition.
const DISTANCE = 2.6;
const DROP = 0.62;
const SCALE = 1.0;            // real A4: 0.174 m span, 0.297 m nose to tail

// --- The wobble -------------------------------------------------------------
const WOBBLE_PERIOD = 0.85;       // s, roll-yaw ring after a disturbance
const WOBBLE_DEG = 4.2;           // deg of roll at full excitation
const TAU_WOBBLE = 1.5;           // s, how long the ring takes to die
const WOBBLE_PER_DEG_S = 0.020;   // excitation per deg/s of heading rate

// --- Flutter ----------------------------------------------------------------
const FLUTTER_HZ_STILL = 9.0;
const FLUTTER_HZ_FAST = 25.0;
const FLUTTER_FULL_MS = 70.0;     // m/s at which flutter reaches full amplitude

// --- Trim and structure -----------------------------------------------------
const BODY_PITCH = -1.6;          // deg: a dart glides very slightly nose-down
// It was folded crooked (see FOLD_DEG in dartmesh.js), so it flies crooked.
const BANK_TRIM = 1.8;            // deg, standing right-wing-down
const FLEX_PER_G = 0.115;         // rad of tip-up per g of load above one
const FLEX_LIMIT = 0.20;          // rad; paper gives, but only so far
const KEEL_SWAY = 0.055;          // rad of keel deflection at full yaw rate

// --- Feel (exponential smoothing time constants, seconds) -------------------
const TAU_HEADING = 0.26;         // a thrown object follows the view lazily
const TAU_BANK = 0.22;            // but rolls quickly, having no inertia to speak of
const TAU_PITCH = 0.40;
const TAU_SPEED = 0.30;
const TAU_FLEX = 0.18;
const TAU_KEEL = 0.30;

const BANK_PER_DEG_S = 0.52;      // deg of roll per deg/s of heading rate
const BANK_MAX = 58.0;            // deg — it will go further over than a bird
const PITCH_MAX = 28.0;           // deg

/**
 * How strongly the sun comes through backlit paper, and how much sky the
 * surface catches at grazing angles.
 *
 * The transmission gain is high and the sheen gain is low, which is the whole
 * difference between paper and a feather: paper passes a lot of light and
 * reflects almost nothing specular. Turning the sheen up is the fastest way
 * to make this look like laminated plastic.
 */
export class Dart extends Flyer {
  static TRANSMISSION_GAIN = 1.15;
  static SHEEN_GAIN = 0.30;

  /**
   * `wear` is how thrown this one is: 0 folds a crisp sheet, 1 the desk-worn
   * dart. It is a fold-time property, not a uniform — the rumples, the
   * blunted nose and the bent tip are geometry — so changing it means
   * rebuilding the mesh, which is why it is a construction argument.
   */
  constructor(device, resources, { wear = DEFAULT_WEAR } = {}) {
    super(device, resources, {
      label: "dart",
      shaderUrl: SHADER_URL,
      mesh: buildDartMesh({ scale: SCALE, wear }),
      distance: DISTANCE,
      drop: DROP,
    });

    this.wobblePhase = 0.0;       // rad
    this.wobbleAmp = 0.0;         // 0..1
    this.flutterPhase = 0.0;      // rad
    this.flutterAmp = 0.0;        // 0..1
    this.wingFlex = 0.0;          // rad, tip up
    this.keelSway = 0.0;          // rad
  }

  /** Local->world rotation columns for the current attitude. */
  _frame() {
    return this._attitude(this.course, this.pitch + BODY_PITCH,
                          this.bank + BANK_TRIM);
  }

  /**
   * A crash sets the roll-yaw ring going at full strength too: the tumble
   * dies away into the dart's own wobble rather than into stillness, which
   * is how a real one recovers — rocking, a second or two after it has its
   * nose back.
   */
  knock(args) {
    super.knock(args);
    this.wobbleAmp = 1.0;
  }

  /** The slots dart.wgsl reads as its own: wing flex, flutter, keel. */
  _species(u) {
    u[59] = this.wingFlex;
    u[72] = this.flutterPhase; u[73] = this.flutterAmp;
    u[74] = this.keelSway; u[75] = 0.0;
  }

  // ------------------------------------------------------------------
  // Animation
  // ------------------------------------------------------------------

  /**
   * Advance the roll-yaw ring. `excite` is the magnitude of the heading rate
   * in deg/s; a turn sets it going and it takes TAU_WOBBLE to die away.
   */
  _wobble(dt, excite) {
    this.wobblePhase = mod2pi(this.wobblePhase + TWO_PI * dt / WOBBLE_PERIOD);
    const target = Math.min(1.0, Math.abs(excite) * WOBBLE_PER_DEG_S);
    const k = 1.0 - Math.exp(-dt / TAU_WOBBLE);
    // Rings up fast and decays slowly: a disturbance is felt at once.
    this.wobbleAmp += (target - this.wobbleAmp) * (target > this.wobbleAmp ? 0.5 : k);
    return WOBBLE_DEG * this.wobbleAmp * Math.sin(this.wobblePhase);
  }

  /** Advance the trailing-edge buzz. Amplitude goes as speed squared. */
  _flutter(dt) {
    const v = Math.min(this._speed / FLUTTER_FULL_MS, 1.0);
    this.flutterAmp = v * v;
    const hz = FLUTTER_HZ_STILL + (FLUTTER_HZ_FAST - FLUTTER_HZ_STILL) * v;
    this.flutterPhase = mod2pi(this.flutterPhase + TWO_PI * hz * dt);
  }

  /** Per-frame dynamics from the live camera. */
  update(dt, camera) {
    dt = clamp(dt, 1e-4, 0.1);
    this._clock += dt;

    const tracked = this._track(dt, camera, {
      tauHeading: TAU_HEADING, tauSpeed: TAU_SPEED });
    if (tracked === null) {
      this._flutter(dt);
      this._place(camera.position, 0.0);
      return;
    }
    // The rate the FLIGHT PATH is turning at — a thrown sheet rolls and rings
    // because its course changed, not because you looked somewhere else.
    const { dcourse, vel } = tracked;
    const headingRate = dcourse / dt;   // deg/s

    // Roll: the steady part follows the turn, and the ring is laid over it.
    const bankTarget = clamp(headingRate * BANK_PER_DEG_S, -BANK_MAX, BANK_MAX);
    const kb = 1.0 - Math.exp(-dt / TAU_BANK);
    this._bankSteady = (this._bankSteady ?? 0.0)
                     + (bankTarget - (this._bankSteady ?? 0.0)) * kb;
    this.bank = clamp(this._bankSteady + this._wobble(dt, headingRate),
                      -BANK_MAX - WOBBLE_DEG, BANK_MAX + WOBBLE_DEG);

    // Pitch: the climb the camera is actually doing.
    const hSpeed = Math.max(Math.hypot(vel[0], vel[1]), 15.0);
    const pitchTarget = clamp(
      Math.atan2(this._vz, hSpeed) / DEG, -PITCH_MAX, PITCH_MAX);
    const kp = 1.0 - Math.exp(-dt / TAU_PITCH);
    this._pitchSteady = (this._pitchSteady ?? 0.0)
                      + (pitchTarget - (this._pitchSteady ?? 0.0)) * kp;
    this.pitch = this._pitchSteady;

    // Wing flex follows the load: bank pulls g in a turn.
    const turnG = 1.0 / Math.max(Math.cos(this.bank * DEG), 0.35);
    const flexTarget = clamp(FLEX_PER_G * (turnG - 1.0), 0.0, FLEX_LIMIT);
    const kf = 1.0 - Math.exp(-dt / TAU_FLEX);
    this.wingFlex += (flexTarget - this.wingFlex) * kf;

    // The keel is the only fin it has, and it lags the yaw.
    const keelTarget = KEEL_SWAY * clamp(headingRate / 60.0, -1.0, 1.0);
    const kk = 1.0 - Math.exp(-dt / TAU_KEEL);
    this.keelSway += (keelTarget - this.keelSway) * kk;

    this._flutter(dt);
    this._place(camera.position, 0.0);
  }
}
