// The dart: a thrown sheet of A4, the inorganic one of soar's two flyers.
//
// Shares everything structural with the bird through flyer.js. What is here
// is how a paper aeroplane moves, and that is where it has to differ, because
// the two are only really told apart in motion. A swift is powered: it beats,
// and every beat is a decision. A dart is a thrown object with no engine and
// almost no damping, and everything it does after it leaves your hand is a
// consequence of that:
//
//   - The PHUGOID. The signature of every paper plane ever thrown. It trades
//     speed for height and back on a slow cycle a couple of seconds long —
//     nose up, climb, slow, nose down, dive, gain speed, repeat — and because
//     paper has essentially no pitch damping the cycle never dies out. If you
//     only implement one thing here, this is the one; it is what makes a
//     viewer say "paper aeroplane" before they have looked at the object.
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
//
// `setStatic(camera, t, overrides)` replaces `update` for a deterministic
// pose, matching the bird's.

"use strict";

import { Flyer, attitudeFrame, clamp, mod2pi, TWO_PI, DEG } from "./flyer.js";
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

// --- The phugoid ------------------------------------------------------------
const PHUGOID_PERIOD = 2.15;      // s
const PHUGOID_PITCH_DEG = 7.5;    // amplitude of the pitch swing
const PHUGOID_BOB_M = 0.085;      // and of the height it trades for
// Altitude lags attitude by a quarter cycle: the aeroplane is highest a
// quarter cycle AFTER it is most nose-up, because it has to climb first.
const PHUGOID_BOB_LAG = Math.PI / 2;
// A real one is not a metronome — the cycle wanders in strength.
const PHUGOID_WANDER_HZ = 0.19;

// --- The wobble -------------------------------------------------------------
const WOBBLE_PERIOD = 0.85;       // s, roll-yaw ring after a disturbance
const WOBBLE_DEG = 4.2;           // deg of roll at full excitation
const TAU_WOBBLE = 1.5;           // s, how long the ring takes to die
const WOBBLE_PER_DEG_S = 0.020;   // excitation per deg/s of heading rate
const WOBBLE_IDLE = 0.10;         // never quite still, even in smooth air

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

    this.phugoidPhase = 0.0;      // rad
    this.wobblePhase = 0.0;       // rad
    this.wobbleAmp = WOBBLE_IDLE; // 0..1
    this.flutterPhase = 0.0;      // rad
    this.flutterAmp = 0.0;        // 0..1
    this.wingFlex = 0.0;          // rad, tip up
    this.keelSway = 0.0;          // rad
  }

  /** Local->world rotation columns for the current attitude. */
  _frame() {
    return attitudeFrame(this.heading, this.pitch + BODY_PITCH,
                         this.bank + BANK_TRIM);
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
   * Advance the phugoid. Returns {pitch, bob}: the degrees of nose-up it
   * contributes now, and the metres of height that buys.
   *
   * The two are a quarter cycle apart and that is not a stylistic choice —
   * it is what a phugoid is. Put them in phase and the aeroplane looks like
   * it is being waved rather than flying.
   */
  _phugoid(dt) {
    this.phugoidPhase = mod2pi(
      this.phugoidPhase + TWO_PI * dt / PHUGOID_PERIOD);
    const wander = 0.72 + 0.28 * Math.sin(TWO_PI * PHUGOID_WANDER_HZ * this._clock);
    return {
      pitch: PHUGOID_PITCH_DEG * wander * Math.sin(this.phugoidPhase),
      bob: PHUGOID_BOB_M * wander
         * Math.sin(this.phugoidPhase - PHUGOID_BOB_LAG),
    };
  }

  /**
   * Advance the roll-yaw ring. `excite` is the magnitude of the heading rate
   * in deg/s; a turn sets it going and it takes TAU_WOBBLE to die back to the
   * idle level it never drops below.
   */
  _wobble(dt, excite) {
    this.wobblePhase = mod2pi(this.wobblePhase + TWO_PI * dt / WOBBLE_PERIOD);
    const target = Math.min(
      1.0, WOBBLE_IDLE + Math.abs(excite) * WOBBLE_PER_DEG_S);
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
      const { bob } = this._phugoid(dt);
      this._flutter(dt);
      this._place(camera.position, bob);
      return;
    }
    const { daz, vel } = tracked;
    const headingRate = daz / dt;   // deg/s

    // Roll: the steady part follows the turn, and the ring is laid over it.
    const bankTarget = clamp(headingRate * BANK_PER_DEG_S, -BANK_MAX, BANK_MAX);
    const kb = 1.0 - Math.exp(-dt / TAU_BANK);
    this._bankSteady = (this._bankSteady ?? 0.0)
                     + (bankTarget - (this._bankSteady ?? 0.0)) * kb;
    this.bank = clamp(this._bankSteady + this._wobble(dt, headingRate),
                      -BANK_MAX - WOBBLE_DEG, BANK_MAX + WOBBLE_DEG);

    // Pitch: the climb the camera is actually doing, plus the phugoid.
    const phugoid = this._phugoid(dt);
    const hSpeed = Math.max(Math.hypot(vel[0], vel[1]), 15.0);
    const pitchTarget = clamp(
      Math.atan2(this._vz, hSpeed) / DEG, -PITCH_MAX, PITCH_MAX);
    const kp = 1.0 - Math.exp(-dt / TAU_PITCH);
    this._pitchSteady = (this._pitchSteady ?? 0.0)
                      + (pitchTarget - (this._pitchSteady ?? 0.0)) * kp;
    this.pitch = clamp(this._pitchSteady + phugoid.pitch,
                       -PITCH_MAX - PHUGOID_PITCH_DEG,
                       PITCH_MAX + PHUGOID_PITCH_DEG);

    // Wing flex follows the load. Bank pulls g in a turn; the phugoid pulls
    // it at the bottom of every cycle, which is why the wings breathe with
    // the porpoise even in still air.
    const turnG = 1.0 / Math.max(Math.cos(this.bank * DEG), 0.35);
    const phugoidG = -0.55 * Math.sin(this.phugoidPhase - PHUGOID_BOB_LAG);
    const flexTarget = clamp(
      FLEX_PER_G * (turnG - 1.0 + phugoidG), -FLEX_LIMIT * 0.5, FLEX_LIMIT);
    const kf = 1.0 - Math.exp(-dt / TAU_FLEX);
    this.wingFlex += (flexTarget - this.wingFlex) * kf;

    // The keel is the only fin it has, and it lags the yaw.
    const keelTarget = KEEL_SWAY * clamp(headingRate / 60.0, -1.0, 1.0);
    const kk = 1.0 - Math.exp(-dt / TAU_KEEL);
    this.keelSway += (keelTarget - this.keelSway) * kk;

    this._flutter(dt);
    this._place(camera.position, phugoid.bob);
  }

  /**
   * Deterministic pose for offscreen rendering (no dynamics). `t` walks the
   * phugoid, so a sequence of stills shows the porpoise.
   */
  setStatic(camera, t = 0.0, { bank = null, pitch = null, phugoidPhase = null } = {}) {
    this.heading = camera.azimuth;
    this.viewElevation = camera.elevation;
    this._clock = t;
    this.phugoidPhase = phugoidPhase === null
      ? mod2pi(TWO_PI * t / PHUGOID_PERIOD)
      : Number(phugoidPhase);
    const wander = 0.72 + 0.28 * Math.sin(TWO_PI * PHUGOID_WANDER_HZ * t);
    const swing = PHUGOID_PITCH_DEG * wander * Math.sin(this.phugoidPhase);
    const bob = PHUGOID_BOB_M * wander
              * Math.sin(this.phugoidPhase - PHUGOID_BOB_LAG);

    this._bankSteady = bank === null ? 0.0 : Number(bank);
    this.bank = this._bankSteady;
    this._pitchSteady = pitch === null
      ? clamp(0.6 * camera.elevation, -PITCH_MAX, PITCH_MAX)
      : Number(pitch);
    this.pitch = this._pitchSteady + (pitch === null ? swing : 0.0);

    this.wobbleAmp = WOBBLE_IDLE;
    this.wobblePhase = 0.0;
    this.flutterAmp = 0.35;
    this.flutterPhase = mod2pi(TWO_PI * 14.0 * t);
    this.wingFlex = FLEX_PER_G
      * (-0.55 * Math.sin(this.phugoidPhase - PHUGOID_BOB_LAG));
    this.keelSway = 0.0;
    this._place(camera.position, bob);
  }
}
