// The flying subject: everything a flyer needs that is not the flyer.
//
// Soar carries two of these — a common swift and a paper dart — and the parts
// that differ between a bird and a sheet of A4 are exactly the interesting
// parts: the mesh, the shading, and how the thing moves. Everything else is
// identical and always was: a small raster pass after the volume march, its
// own depth buffer for self-occlusion, alpha blending over the finished
// frame, attenuation by the scene's own extinction field, and one uniform
// block of matrices and light.
//
// So that lives here, and `Bird` and `Dart` are the two hundred lines that
// actually differ. A subclass supplies:
//
//   MESH        {data, vertexCount, stride, attributes} from its mesh module
//   SHADER_URL  its own .wgsl, which must declare the SAME uniform block
//   LABEL       for buffer/pipeline labels and error messages
//   _frame()    -> [right, forward, up], the local->world rotation columns
//   _species(u) writes the slots whose meaning is the subclass's own:
//               bmin.w and the whole of `anim`. See the two shaders.
//
// --- Camera-relative, and it must stay that way ---------------------------
//
// The model matrix translates by (position - camera position) and the view
// matrix has no translation at all. That subtraction happens in JS, in the
// doubles the flight state is kept in, and only its result — a few metres —
// is ever narrowed to float32.
//
// Doing it the obvious way instead is what broke the bird. Both the model
// translation and the view matrix carried absolute world coordinates, so the
// shader computed `model * p` as (1e6 + 0.008) and then `vp * wp` as
// (1e6 - 1e6): two catastrophic cancellations in a row. A flyer is tens of
// centimetres across with millimetre features, finer than a float32 ulp at
// those magnitudes, so the mesh snapped onto a world-axis lattice and
// rasterized as three or four stray slivers. Nothing else in soar noticed,
// because nothing else has millimetre geometry — the clouds and the ocean are
// metres to hundreds of metres and quantize invisibly.
//
// It took flying out to reach, which is why it read as random rather than
// positional: Camera.constrain bounds x and y only in a periodic domain and
// never bounds altitude, so world coordinates are unbounded and a few minutes
// at speed puts you hundreds of kilometres from the origin. Camera-relative
// removes the failure mode rather than moving the distance at which it
// appears — every quantity below is of order the flyer's own size, where
// float32 resolves a fraction of a micron.
//
// --- Public API -----------------------------------------------------------
//
//   const flyer = new Bird(device, { volumeView, sampler, bmin, bmax });
//   await flyer.init(targetFormat[, shaderSource]);
//   flyer.update(dtSeconds, camera);
//   flyer.writeUniforms(camera, [outW, outH], { sunAzimuth, ... });
//   flyer.encodePass(commandEncoder, targetView, targetFormat, [outW, outH]);
//   flyer.destroy();
//
// `camera` is anything carrying {position (world metres), azimuth, elevation,
// fov (deg)} — FlightCamera is one, and its `position` array may keep being
// mutated afterwards, so nothing here retains a reference to it.
//
// The pass loads rather than clears its colour attachment, so it must be
// encoded after whatever painted the frame.

"use strict";

import { DEFAULT_EXPOSURE, DEFAULT_AMBIENT_STRENGTH,
         DEFAULT_SUN_AZIMUTH, DEFAULT_SUN_ELEVATION,
         DEFAULT_TONE_MAP_GAMMA,
         DEFAULT_TONE_MAP_WHITE_POINT, DEFAULT_CONTRAST,
         SPECTRAL_LIGHTING_STRENGTH } from "./constants.js";
import { cameraBasis } from "./camera.js";
import { directionFromAzimuthElevation, mod360,
         spectralLightingColors } from "./spectral.js";
import { retireAfterSubmittedWork } from "./gpu.js";

export const UNIFORM_NBYTES = 3 * 64 + 8 * 16;   // 3 mat4 + 8 vec4
export const DEG = Math.PI / 180.0;

const NEAR = 0.5, FAR = 400.0;
const DEPTH_FORMAT = "depth24plus";

export const TWO_PI = 2.0 * Math.PI;
export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Python's floor-mod. JS `%` truncates, which lets a negative phase stay so. */
export const mod2pi = (a) => ((a % TWO_PI) + TWO_PI) % TWO_PI;

export function smoothstep(x, lo, hi) {
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
 * Camera-relative view->clip, matching the raymarcher's ray construction:
 * horizontal FOV, clip +y = image top, WebGPU depth in [0, 1].
 *
 * The view matrix carries NO translation, and that is the whole point — see
 * the note at the top of this file.
 */
function perspectiveVP(camera, [w, h]) {
  const [forward, right, up] = cameraBasis(camera.azimuth, camera.elevation);
  const aspect = w / h;
  const f = 1.0 / Math.tan(camera.fov * DEG * 0.5);

  // Rows 0-2 are the camera basis; the fourth column stays zero.
  const view = new Float64Array(16);
  view[0] = right[0]; view[1] = right[1]; view[2] = right[2];
  view[4] = up[0]; view[5] = up[1]; view[6] = up[2];
  view[8] = -forward[0]; view[9] = -forward[1]; view[10] = -forward[2];
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
 * columns [right, forward, up]. Shared because a swift and a dart bank and
 * pitch the same way; only what drives the angles differs.
 *
 * heading is a met azimuth (0 = N, 90 = E); positive bank rolls right (into a
 * rightward turn); positive pitch climbs.
 */
export function attitudeFrame(headingDeg, pitchDeg, bankDeg) {
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
  return [
    [cb * right[0] - sb * up[0], cb * right[1] - sb * up[1],
     cb * right[2] - sb * up[2]],
    fwd,
    [cb * up[0] + sb * right[0], cb * up[1] + sb * right[1],
     cb * up[2] + sb * right[2]],
  ];
}

/**
 * GPU resources and the shared half of a flyer's state.
 *
 * Shares the scene's resident sigma texture and sampler with the raymarcher
 * and draws as a second, tiny raster pass: own depth buffer for
 * self-occlusion, alpha-blended over the finished volume frame.
 */
export class Flyer {
  /**
   * `volumeView` and `sampler` are the raymarcher's own (Scene.volumeView,
   * Renderer.volSampler); `bmin`/`bmax` are the outer domain AABB in metres,
   * which is what the shader's occlusion march clips against. The nest, if
   * any, is deliberately not consulted — this is an attenuation estimate, not
   * the picture.
   *
   * `spec` is the subclass's own {mesh, shaderUrl, label, distance, drop}.
   */
  constructor(device, { volumeView, sampler, bmin, bmax }, spec) {
    const label = spec.label;
    if (!device) throw new Error(`${label} needs a GPUDevice.`);
    if (!volumeView || !sampler) {
      throw new Error(
        `${label} needs the raymarcher's volume view and sampler: its ` +
        "fragment stage attenuates by the same extinction field.");
    }
    this.device = device;
    this.volumeView = volumeView;
    this.sampler = sampler;
    this.bmin = [...bmin];
    this.bmax = [...bmax];

    this.label = label;
    this.shaderUrl = spec.shaderUrl;
    this.distance = spec.distance;
    this.drop = spec.drop;

    const { data, vertexCount, stride, attributes } = spec.mesh;
    this.nVertices = vertexCount;
    this.vertexStride = stride;
    this._attributes = attributes;
    this._vbuf = device.createBuffer({
      label: `${label}-mesh`,
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this._vbuf.getMappedRange()).set(data);
    this._vbuf.unmap();

    this._ubuf = device.createBuffer({
      label: `${label}-uniforms`,
      size: UNIFORM_NBYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._uniforms = new Float32Array(UNIFORM_NBYTES / 4);

    // Declared explicitly, never `layout: "auto"`. An auto layout is derived
    // from the bindings an entry point happens to use, and the mismatch it
    // produces is reported asynchronously — a silent black frame.
    this._bindGroupLayout = device.createBindGroupLayout({
      label,
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

    // --- Shared flight state (world units / degrees as noted) ---
    this.position = [0.0, 0.0, 0.0];   // world metres
    this.heading = 0.0;                // smoothed met azimuth, deg
    this.viewElevation = 0.0;          // smoothed camera elevation, deg
    this.bank = 0.0;                   // deg, + rolls right
    this.pitch = 0.0;                  // deg, + climbs
    this._speed = 0.0;                 // smoothed m/s
    this._vz = 0.0;                    // smoothed vertical velocity m/s
    this._clock = 0.0;
    this._prevOrigin = null;           // for velocity estimation
  }

  _refreshBindGroup() {
    this._bindGroup = this.device.createBindGroup({
      label: this.label,
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
   * Point the flyer at a different field without recompiling its shader.
   * A new scene brings a new texture and new bounds, and the old view refers
   * to a texture the scene has already destroyed.
   */
  setVolume({ volumeView, bmin, bmax }) {
    if (!volumeView) throw new Error(`${this.label}.setVolume needs a volume view.`);
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
      throw new Error(
        `${this.label}.init needs a target texture format; got ${targetFormat}.`);
    }
    let source = shaderSource;
    if (source === null) {
      const response = await fetch(this.shaderUrl);
      if (!response.ok) {
        throw new Error(
          `Could not fetch ${this.shaderUrl} (HTTP ${response.status}).`);
      }
      source = await response.text();
    }
    this._shader = this.device.createShaderModule(
      { label: this.label, code: source });

    const problems = [];
    const info = await this._shader.getCompilationInfo?.();
    for (const message of info?.messages ?? []) {
      if (message.type !== "error") continue;
      problems.push(`line ${message.lineNum}:${message.linePos} — ${message.message}`);
    }
    if (problems.length) {
      throw new Error(`${this.label} did not compile.\n${problems.join("\n")}`);
    }

    this.device.pushErrorScope("validation");
    this._pipelineFor(targetFormat);
    const error = await this.device.popErrorScope();
    if (error) {
      throw new Error(
        `Setting up the ${this.label} pipeline failed: ${error.message}`);
    }
  }

  // ------------------------------------------------------------------
  // Animation
  // ------------------------------------------------------------------

  /** Anchor the flyer ahead of / below the smoothed view direction. */
  _place(origin, bob) {
    const [forward, , up] = cameraBasis(this.heading, this.viewElevation);
    this.position = [
      origin[0] + forward[0] * this.distance - up[0] * this.drop,
      origin[1] + forward[1] * this.distance - up[1] * this.drop,
      origin[2] + forward[2] * this.distance - up[2] * this.drop + bob,
    ];
  }

  /**
   * Track the camera: velocity, heading lag, and the vertical rate. Returns
   * the smoothed heading change in degrees this step, which is what banking
   * is driven from — subclasses bank differently and this is the raw material
   * they share.
   *
   * Returns null on the first frame, where there is nothing to differentiate
   * and the flyer snaps into place instead.
   */
  _track(dt, camera, { tauHeading, tauSpeed }) {
    const origin = camera.position;
    if (this._prevOrigin === null) {
      this._prevOrigin = [...origin];
      this.heading = camera.azimuth;
      this.viewElevation = camera.elevation;
      return null;
    }
    const vel = [
      (origin[0] - this._prevOrigin[0]) / dt,
      (origin[1] - this._prevOrigin[1]) / dt,
      (origin[2] - this._prevOrigin[2]) / dt,
    ];
    this._prevOrigin = [...origin];
    const ks = 1.0 - Math.exp(-dt / tauSpeed);
    this._speed += (Math.hypot(vel[0], vel[1], vel[2]) - this._speed) * ks;
    this._vz += (vel[2] - this._vz) * ks;

    const kh = 1.0 - Math.exp(-dt / tauHeading);
    const daz = mod360(camera.azimuth - this.heading + 180.0) - 180.0;
    this.heading = mod360(this.heading + daz * kh);
    this.viewElevation += (camera.elevation - this.viewElevation) * kh;
    return { daz: daz * kh, vel, kh };
  }

  // ------------------------------------------------------------------
  // GPU
  // ------------------------------------------------------------------

  /**
   * Pack matrices + light for the current pose and enqueue the upload. The
   * slots whose meaning is the subclass's own — bmin.w and `anim` — are
   * filled by `_species`.
   */
  writeUniforms(camera, outputSize, {
    sunAzimuth = DEFAULT_SUN_AZIMUTH,
    sunElevation = DEFAULT_SUN_ELEVATION,
    exposure = DEFAULT_EXPOSURE,
    ambientStrength = DEFAULT_AMBIENT_STRENGTH,
    toneMapGamma = DEFAULT_TONE_MAP_GAMMA,
    toneMapWhitePoint = DEFAULT_TONE_MAP_WHITE_POINT,
    contrast = DEFAULT_CONTRAST,
    spectralStrength = SPECTRAL_LIGHTING_STRENGTH,
    transmissionGain = null,
    sheenGain = null,
  } = {}) {
    const [w, h] = outputSize;
    if (!(w >= 1 && h >= 1)) {
      throw new Error(
        `${this.label}.writeUniforms needs a positive size; got ${w}x${h}.`);
    }
    const origin = camera.position;
    const vp = perspectiveVP(camera, outputSize);
    const [right, fwd, up] = this._frame();

    // model = rotation with the flyer's offset FROM THE CAMERA in the last
    // column — differenced here in doubles, never in the shader; nrot is the
    // same rotation with no translation, for the normals.
    const model = new Float64Array(16);
    const nrot = new Float64Array(16);
    for (let r = 0; r < 3; r++) {
      model[4 * r] = right[r]; model[4 * r + 1] = fwd[r]; model[4 * r + 2] = up[r];
      model[4 * r + 3] = this.position[r] - origin[r];
      nrot[4 * r] = right[r]; nrot[4 * r + 1] = fwd[r]; nrot[4 * r + 2] = up[r];
    }
    model[15] = 1.0;
    nrot[15] = 1.0;

    const sun = directionFromAzimuthElevation(sunAzimuth, sunElevation);
    // The same spectral shift the clouds get, from the same function: at a
    // low sun the beam reddens and the fill goes blue, and the flyer has to
    // move with it or it reads as a sticker.
    const light = spectralLightingColors(sun, undefined, spectralStrength);

    const u = this._uniforms;
    writeColumnMajor(u, 0, vp);
    writeColumnMajor(u, 16, model);
    writeColumnMajor(u, 32, nrot);
    u[48] = origin[0]; u[49] = origin[1]; u[50] = origin[2]; u[51] = exposure;
    u[52] = sun[0]; u[53] = sun[1]; u[54] = sun[2]; u[55] = ambientStrength;
    u[56] = this.bmin[0]; u[57] = this.bmin[1]; u[58] = this.bmin[2];
    u[60] = this.bmax[0]; u[61] = this.bmax[1]; u[62] = this.bmax[2];
    u[63] = toneMapGamma;
    u[64] = light.cloudSun[0]; u[65] = light.cloudSun[1];
    u[66] = light.cloudSun[2];
    u[67] = transmissionGain ?? this.constructor.TRANSMISSION_GAIN;
    u[68] = light.ambient[0]; u[69] = light.ambient[1];
    u[70] = light.ambient[2];
    u[71] = sheenGain ?? this.constructor.SHEEN_GAIN;
    u[76] = toneMapWhitePoint; u[77] = contrast; u[78] = 0.0; u[79] = 0.0;
    // u[59] (bmin.w) and u[72..75] (anim) are the subclass's.
    this._species(u);
    this.device.queue.writeBuffer(this._ubuf, 0, u);
  }

  _pipelineFor(targetFormat) {
    let pipeline = this._pipelines.get(targetFormat);
    if (!pipeline) {
      if (this._shader === null) {
        throw new Error(
          `${this.label}.init() must be awaited before it draws.`);
      }
      pipeline = this.device.createRenderPipeline({
        label: `${this.label}(${targetFormat})`,
        layout: this._pipelineLayout,
        vertex: {
          module: this._shader,
          entryPoint: "vs_main",
          buffers: [{
            arrayStride: this.vertexStride,
            attributes: this._attributes,
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

  /** Tiny per-target depth buffer for self-occlusion (cached). */
  _depthView(w, h) {
    if (this._depth === null || this._depth.w !== w || this._depth.h !== h) {
      // Not destroyed on the spot: a resize (or a move to a screen with a
      // different DPI) happens between frames that are still in flight, and
      // this texture is a live attachment in them.
      if (this._depth) retireAfterSubmittedWork(this.device, this._depth.texture);
      const texture = this.device.createTexture({
        label: `${this.label}-depth`,
        size: [w, h, 1],
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this._depth = { w, h, texture, view: texture.createView() };
    }
    return this._depth.view;
  }

  /**
   * Encode the flyer's raster pass over an already-rendered frame. The colour
   * attachment loads; the depth attachment is scratch, so it clears on entry
   * and is discarded on exit rather than stored.
   */
  encodePass(commandEncoder, targetView, targetFormat, outputSize) {
    const [w, h] = outputSize;
    if (!(w >= 1 && h >= 1)) {
      throw new Error(
        `${this.label}.encodePass needs a positive size; got ${w}x${h}.`);
    }
    const pass = commandEncoder.beginRenderPass({
      label: this.label,
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

// Re-exported so a subclass gets its whole vocabulary from one import.
export { mod360 };
