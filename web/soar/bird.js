// The bird: the small animated flying subject that leads the fly-through.
//
// The flight behaviour began as a port of the desktop's bird.py; the geometry
// and shading did not. `birdmesh.js` and `bird.wgsl` rebuilt the animal from
// anatomy and are native to this build — unlike raymarch.wgsl, which crossed
// over verbatim.
//
// A common swift at its real size — procedural mesh, no assets — that
// flies a few metres ahead of and below the camera. It flaps (faster when
// slow, tucking into a glide above a speed threshold), banks into turns,
// pitches with climbs and descents, and bobs with its own wingbeat. All state
// is exponentially smoothed so it feels alive rather than bolted to the
// camera, and its fragment shader attenuates by the extinction field so it
// fades naturally when it flies into cloud.
//
// --- Public API -----------------------------------------------------------
//
//   const bird = new Bird(device, { volumeView, sampler, bmin, bmax });
//   await bird.init(targetFormat[, shaderSource]);
//   bird.update(dtSeconds, camera);
//   bird.writeUniforms(camera, [outW, outH], {
//     sunAzimuth, sunElevation, exposure, ambientStrength });
//   bird.encodePass(commandEncoder, targetView, targetFormat, [outW, outH]);
//   bird.destroy();
//
// `camera` is anything carrying {position (world metres), azimuth, elevation,
// fov (deg)} — FlightCamera is one, and its `position` array may keep being
// mutated afterwards, so nothing here retains a reference to it. `dtSeconds`
// is wall clock. `setStatic(camera, t, overrides)` replaces `update` for a
// deterministic pose (offscreen stills), matching Bird.set_static.
//
// The pass loads rather than clears its colour attachment, so it must be
// encoded after whatever painted the frame; it carries its own small depth
// buffer purely for self-occlusion (a wing in front of the body).

"use strict";

import { DEFAULT_EXPOSURE, DEFAULT_AMBIENT_STRENGTH,
         DEFAULT_SUN_AZIMUTH, DEFAULT_SUN_ELEVATION,
         DEFAULT_TONE_MAP_GAMMA,
         SPECTRAL_LIGHTING_STRENGTH } from "./constants.js";
import { cameraBasis } from "./camera.js";
import { directionFromAzimuthElevation, mod360,
         spectralLightingColors } from "./spectral.js";
import { buildBirdMesh, FLOATS_PER_VERTEX } from "./birdmesh.js";
import { retireAfterSubmittedWork } from "./gpu.js";

const SHADER_URL = new URL("./bird.wgsl", import.meta.url);

const UNIFORM_NBYTES = 3 * 64 + 7 * 16;   // 3 mat4 + 7 vec4
const UNIFORM_FLOATS = UNIFORM_NBYTES / 4;
const DEG = Math.PI / 180.0;

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
const NEAR = 0.5, FAR = 400.0;

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

// --- Look -------------------------------------------------------------------
// How strongly the sun comes through a backlit feather vane, and how much sky
// the oiled plumage catches at grazing angles. Both are the knobs to reach for
// first if the bird sits wrong against a particular sky.
const TRANSMISSION_GAIN = 0.85;
const SHEEN_GAIN = 0.55;
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

const DEPTH_FORMAT = "depth24plus";

const TWO_PI = 2.0 * Math.PI;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// Python's floor-mod. JS `%` truncates, which would let a phase set from a
// negative override stay negative and drift out of step with the desktop.
const mod2pi = (a) => ((a % TWO_PI) + TWO_PI) % TWO_PI;

function smoothstep(x, lo, hi) {
  const t = clamp((x - lo) / (hi - lo), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}


// --- 4x4 matrices ----------------------------------------------------------
//
// Held row-major (index 4*row + col) because that is how the Python reads,
// and transposed once on the way into the uniform buffer — WGSL's mat4x4
// storage is column-major.

function mat4Multiply(a, b) {
  const out = new Float64Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0.0;
      for (let i = 0; i < 4; i++) s += a[4 * r + i] * b[4 * i + c];
      out[4 * r + c] = s;
    }
  }
  return out;
}

function writeColumnMajor(out, offset, m) {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) out[offset + 4 * c + r] = m[4 * r + c];
  }
}

/**
 * World->clip matrix matching the raymarcher's ray construction: horizontal
 * FOV, clip +y = image top, WebGPU depth in [0, 1].
 */
function perspectiveVP(origin, camera, [w, h]) {
  const [forward, right, up] = cameraBasis(camera.azimuth, camera.elevation);
  const aspect = w / h;
  const f = 1.0 / Math.tan(camera.fov * DEG * 0.5);

  const view = new Float64Array(16);
  view[0] = right[0]; view[1] = right[1]; view[2] = right[2];
  view[3] = -dot3(right, origin);
  view[4] = up[0]; view[5] = up[1]; view[6] = up[2];
  view[7] = -dot3(up, origin);
  view[8] = -forward[0]; view[9] = -forward[1]; view[10] = -forward[2];
  view[11] = dot3(forward, origin);
  view[15] = 1.0;

  const proj = new Float64Array(16);
  proj[0] = f;
  proj[5] = f * aspect;
  proj[10] = FAR / (NEAR - FAR);
  proj[11] = NEAR * FAR / (NEAR - FAR);
  proj[14] = -1.0;
  return mat4Multiply(proj, view);
}

/**
 * Local (x right, y forward, z up) -> world rotation, returned as its three
 * columns [right, forward, up].
 *
 * heading is a met azimuth (0 = N, 90 = E); positive bank rolls right (into a
 * rightward turn); positive pitch climbs.
 */
function birdRotation(headingDeg, pitchDeg, bankDeg) {
  const fwd = directionFromAzimuthElevation(headingDeg, pitchDeg);
  // cross(fwd, world_up) with world_up = +z.
  let right = [fwd[1], -fwd[0], 0.0];
  let nrm = Math.hypot(right[0], right[1], right[2]);
  if (nrm < 1e-6) {   // pitched vertical; pick any horizontal right
    right = [1.0, 0.0, 0.0];
    nrm = 1.0;
  }
  right = [right[0] / nrm, right[1] / nrm, right[2] / nrm];
  const up = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];
  const b = bankDeg * DEG;
  const cb = Math.cos(b), sb = Math.sin(b);
  const rightB = [
    cb * right[0] - sb * up[0],
    cb * right[1] - sb * up[1],
    cb * right[2] - sb * up[2],
  ];
  const upB = [
    cb * up[0] + sb * right[0],
    cb * up[1] + sb * right[1],
    cb * up[2] + sb * right[2],
  ];
  return [rightB, fwd, upB];
}

/**
 * GPU resources + animation state for the flying subject.
 *
 * Shares the scene's resident sigma texture and sampler with the raymarcher
 * and draws as a second, tiny raster pass: own depth buffer for
 * self-occlusion, alpha-blended over the finished volume frame.
 */
export class Bird {
  /**
   * `volumeView` and `sampler` are the raymarcher's own (Scene.volumeView,
   * Renderer.volSampler); `bmin`/`bmax` are the outer domain AABB in metres,
   * which is what the shader's occlusion march clips against. The nest, if
   * any, is deliberately not consulted — the desktop bird samples the outer
   * level only, and this is an attenuation estimate, not the picture.
   */
  constructor(device, { volumeView, sampler, bmin, bmax }) {
    if (!device) throw new Error("Bird needs a GPUDevice.");
    if (!volumeView || !sampler) {
      throw new Error(
        "Bird needs the raymarcher's volume view and sampler: its fragment " +
        "stage attenuates by the same extinction field.");
    }
    this.device = device;
    this.volumeView = volumeView;
    this.sampler = sampler;
    this.bmin = [...bmin];
    this.bmax = [...bmax];

    const { data, vertexCount, stride } = buildBirdMesh({ scale: SCALE });
    this.nVertices = vertexCount;
    this.vertexStride = stride;
    this._vbuf = device.createBuffer({
      label: "bird-mesh",
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this._vbuf.getMappedRange()).set(data);
    this._vbuf.unmap();

    this._ubuf = device.createBuffer({
      label: "bird-uniforms",
      size: UNIFORM_NBYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._uniforms = new Float32Array(UNIFORM_NBYTES / 4);

    // Declared explicitly, never `layout: "auto"`. An auto layout is derived
    // from the bindings an entry point happens to use, and the mismatch it
    // produces is reported asynchronously — a silent black frame.
    this._bindGroupLayout = device.createBindGroupLayout({
      label: "bird",
      entries: [
        { binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" } },
      ],
    });
    this._pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this._bindGroupLayout],
    });
    this._refreshBindGroup();

    this._shader = null;
    this._pipelines = new Map();   // target format -> pipeline
    this._depth = null;            // {w, h, texture, view}

    // --- Animation state (world units / degrees / radians as noted) ---
    this.position = [0.0, 0.0, 0.0];   // world metres
    this.heading = 0.0;                // smoothed met azimuth, deg
    this.viewElevation = 0.0;          // smoothed camera elevation, deg
    this.bank = 0.0;                   // deg, + rolls right
    this.pitch = 0.0;                  // deg, + climbs
    this.flapPhase = 0.0;              // rad
    this.flapAmp = FLAP_AMPLITUDE;
    this.flapAngle = REST_DIHEDRAL;    // rad, current wing angle
    this.wristFlex = 0.0;              // rad, extra bend carried by the hand
    this.handTwist = 0.0;              // rad, supination of the hand
    this.tailSpread = 0.0;             // 0 closed, 1 fully fanned
    this._speed = 0.0;                 // smoothed m/s
    this._vz = 0.0;                    // smoothed vertical velocity m/s
    this._clock = 0.0;
    this._prevOrigin = null;           // for velocity estimation
  }

  _refreshBindGroup() {
    this._bindGroup = this.device.createBindGroup({
      label: "bird",
      layout: this._bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this._ubuf, offset: 0,
                                  size: UNIFORM_NBYTES } },
        { binding: 1, resource: this.volumeView },
        { binding: 2, resource: this.sampler },
      ],
    });
  }

  /**
   * Point the bird at a different field without recompiling its shader.
   * A new scene brings a new texture and new bounds, and the old view refers
   * to a texture the scene has already destroyed.
   */
  setVolume({ volumeView, bmin, bmax }) {
    if (!volumeView) throw new Error("Bird.setVolume needs a volume view.");
    this.volumeView = volumeView;
    this.bmin = [...bmin];
    this.bmax = [...bmax];
    this._refreshBindGroup();
  }

  // ------------------------------------------------------------------
  // Setup
  // ------------------------------------------------------------------

  /**
   * Compile the shader and build the pipeline for `targetFormat`, asking up
   * front for both failures WebGPU otherwise reports asynchronously. Pass
   * `shaderSource` to skip the fetch (a bundled build, or a test).
   */
  async init(targetFormat, shaderSource = null) {
    if (typeof targetFormat !== "string") {
      throw new Error(`Bird.init needs a target texture format; got ${targetFormat}.`);
    }
    let source = shaderSource;
    if (source === null) {
      const response = await fetch(SHADER_URL);
      if (!response.ok) {
        throw new Error(
          `Could not fetch ${SHADER_URL} (HTTP ${response.status}).`);
      }
      source = await response.text();
    }
    this._shader = this.device.createShaderModule({ label: "bird", code: source });

    const problems = [];
    const info = await this._shader.getCompilationInfo?.();
    for (const message of info?.messages ?? []) {
      if (message.type !== "error") continue;
      problems.push(`line ${message.lineNum}:${message.linePos} — ${message.message}`);
    }
    if (problems.length) {
      throw new Error(`bird.wgsl did not compile.\n${problems.join("\n")}`);
    }

    this.device.pushErrorScope("validation");
    this._pipelineFor(targetFormat);
    const error = await this.device.popErrorScope();
    if (error) {
      throw new Error(`Setting up the bird pipeline failed: ${error.message}`);
    }
  }

  // ------------------------------------------------------------------
  // Animation
  // ------------------------------------------------------------------

  /** Anchor the bird ahead of / below the smoothed view direction. */
  _place(origin, bob) {
    const [forward, , up] = cameraBasis(this.heading, this.viewElevation);
    this.position = [
      origin[0] + forward[0] * DISTANCE - up[0] * DROP,
      origin[1] + forward[1] * DISTANCE - up[1] * DROP,
      origin[2] + forward[2] * DISTANCE - up[2] * DROP + bob,
    ];
  }

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
   *
   * `camera.position` is the flight controller's own live array, so the
   * previous origin is copied rather than referenced: keeping the reference
   * would make the velocity estimate identically zero.
   */
  update(dt, camera) {
    const origin = camera.position;
    const azimuth = camera.azimuth;
    const elevation = camera.elevation;
    dt = clamp(dt, 1e-4, 0.1);
    this._clock += dt;

    if (this._prevOrigin === null) {
      // First frame: snap, no dynamics.
      this._prevOrigin = [...origin];
      this.heading = azimuth;
      this.viewElevation = elevation;
      this._place(origin, this._flap(dt));
      return;
    }

    // Velocity estimate (smoothed).
    const vel = [
      (origin[0] - this._prevOrigin[0]) / dt,
      (origin[1] - this._prevOrigin[1]) / dt,
      (origin[2] - this._prevOrigin[2]) / dt,
    ];
    this._prevOrigin = [...origin];
    const ks = 1.0 - Math.exp(-dt / TAU_SPEED);
    this._speed += (Math.hypot(vel[0], vel[1], vel[2]) - this._speed) * ks;
    this._vz += (vel[2] - this._vz) * ks;

    // Heading/view lag: the bird swings through turns and settles.
    const kh = 1.0 - Math.exp(-dt / TAU_HEADING);
    const daz = mod360(azimuth - this.heading + 180.0) - 180.0;
    this.heading = mod360(this.heading + daz * kh);
    this.viewElevation += (elevation - this.viewElevation) * kh;

    // Bank into turns: roll follows the smoothed heading rate.
    const headingRate = daz * kh / dt;   // deg/s
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

    this._place(origin, this._flap(dt));
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

  // ------------------------------------------------------------------
  // GPU
  // ------------------------------------------------------------------

  /** Pack matrices + params for the current pose and enqueue the upload. */
  writeUniforms(camera, outputSize, {
    sunAzimuth = DEFAULT_SUN_AZIMUTH,
    sunElevation = DEFAULT_SUN_ELEVATION,
    exposure = DEFAULT_EXPOSURE,
    ambientStrength = DEFAULT_AMBIENT_STRENGTH,
    toneMapGamma = DEFAULT_TONE_MAP_GAMMA,
    spectralStrength = SPECTRAL_LIGHTING_STRENGTH,
    transmissionGain = TRANSMISSION_GAIN,
    sheenGain = SHEEN_GAIN,
  } = {}) {
    const [w, h] = outputSize;
    if (!(w >= 1 && h >= 1)) {
      throw new Error(`Bird.writeUniforms needs a positive size; got ${w}x${h}.`);
    }
    const origin = camera.position;
    const vp = perspectiveVP(origin, camera, outputSize);
    const [right, fwd, up] = birdRotation(
      this.heading, this.pitch + BODY_PITCH, this.bank);

    // model = rotation with the bird's world position in the last column;
    // nrot is the same rotation with no translation, for the normals.
    const model = new Float64Array(16);
    const nrot = new Float64Array(16);
    for (let r = 0; r < 3; r++) {
      model[4 * r] = right[r]; model[4 * r + 1] = fwd[r]; model[4 * r + 2] = up[r];
      model[4 * r + 3] = this.position[r];
      nrot[4 * r] = right[r]; nrot[4 * r + 1] = fwd[r]; nrot[4 * r + 2] = up[r];
    }
    model[15] = 1.0;
    nrot[15] = 1.0;

    const sun = directionFromAzimuthElevation(sunAzimuth, sunElevation);
    // The same spectral shift the clouds get, from the same function: at a
    // low sun the beam reddens and the fill goes blue, and the bird has to
    // move with it or it reads as a sticker.
    const light = spectralLightingColors(sun, undefined, spectralStrength);

    const u = this._uniforms;
    writeColumnMajor(u, 0, vp);
    writeColumnMajor(u, 16, model);
    writeColumnMajor(u, 32, nrot);
    u[48] = origin[0]; u[49] = origin[1]; u[50] = origin[2]; u[51] = exposure;
    u[52] = sun[0]; u[53] = sun[1]; u[54] = sun[2]; u[55] = ambientStrength;
    u[56] = this.bmin[0]; u[57] = this.bmin[1]; u[58] = this.bmin[2];
    u[59] = this.flapAngle;
    u[60] = this.bmax[0]; u[61] = this.bmax[1]; u[62] = this.bmax[2];
    u[63] = toneMapGamma;
    u[64] = light.cloudSun[0]; u[65] = light.cloudSun[1];
    u[66] = light.cloudSun[2]; u[67] = transmissionGain;
    u[68] = light.ambient[0]; u[69] = light.ambient[1];
    u[70] = light.ambient[2]; u[71] = sheenGain;
    u[72] = this.wristFlex; u[73] = this.handTwist;
    u[74] = this.tailSpread; u[75] = 0.0;
    this.device.queue.writeBuffer(this._ubuf, 0, u);
  }

  _pipelineFor(targetFormat) {
    let pipeline = this._pipelines.get(targetFormat);
    if (!pipeline) {
      if (this._shader === null) {
        throw new Error("Bird.init() must be awaited before the bird draws.");
      }
      pipeline = this.device.createRenderPipeline({
        label: `bird(${targetFormat})`,
        layout: this._pipelineLayout,
        vertex: {
          module: this._shader,
          entryPoint: "vs_main",
          buffers: [{
            arrayStride: FLOATS_PER_VERTEX * 4,
            attributes: [
              { format: "float32x3", offset: 0, shaderLocation: 0 },   // pos
              { format: "float32x3", offset: 12, shaderLocation: 1 },  // normal
              { format: "float32", offset: 24, shaderLocation: 2 },    // span
              { format: "float32", offset: 28, shaderLocation: 3 },    // chord
              { format: "float32", offset: 32, shaderLocation: 4 },    // part
              { format: "float32", offset: 36, shaderLocation: 5 },    // feather
            ],
          }],
        },
        primitive: {
          topology: "triangle-list",
          // Thin shell viewed from both sides; the fragment shader flips
          // normals toward the camera.
          cullMode: "none",
        },
        depthStencil: {
          format: DEPTH_FORMAT,
          depthWriteEnabled: true,
          depthCompare: "less",
        },
        fragment: {
          module: this._shader,
          entryPoint: "fs_main",
          targets: [{
            format: targetFormat,
            blend: {
              color: { srcFactor: "src-alpha",
                       dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one",
                       dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          }],
        },
      });
      this._pipelines.set(targetFormat, pipeline);
    }
    return pipeline;
  }

  /** Tiny per-target depth buffer for bird self-occlusion (cached). */
  _depthView(w, h) {
    if (this._depth === null || this._depth.w !== w || this._depth.h !== h) {
      // Not destroyed on the spot: a resize (or a move to a screen with a
      // different DPI) happens between frames that are still in flight, and
      // this texture is a live attachment in them.
      if (this._depth) retireAfterSubmittedWork(this.device, this._depth.texture);
      const texture = this.device.createTexture({
        label: "bird-depth",
        size: [w, h, 1],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this._depth = { w, h, texture, view: texture.createView() };
    }
    return this._depth.view;
  }

  /**
   * Encode the bird raster pass over an already-rendered frame. The colour
   * attachment loads; the depth attachment is scratch, so it clears on entry
   * and is discarded on exit rather than stored.
   */
  encodePass(commandEncoder, targetView, targetFormat, outputSize) {
    const [w, h] = outputSize;
    if (!(w >= 1 && h >= 1)) {
      throw new Error(`Bird.encodePass needs a positive size; got ${w}x${h}.`);
    }
    const pass = commandEncoder.beginRenderPass({
      label: "bird",
      colorAttachments: [{
        view: targetView, loadOp: "load", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this._depthView(w, h),
        depthLoadOp: "clear",
        depthStoreOp: "discard",
        depthClearValue: 1.0,
      },
    });
    pass.setPipeline(this._pipelineFor(targetFormat));
    pass.setBindGroup(0, this._bindGroup);
    pass.setVertexBuffer(0, this._vbuf);
    pass.draw(this.nVertices);
    pass.end();
  }

  /** Release everything this object allocated. Pipelines go with the object. */
  destroy() {
    this._depth?.texture.destroy();
    this._depth = null;
    this._vbuf?.destroy();
    this._ubuf?.destroy();
  }
}
