// The render loop's GPU half: shader specialization, the three-pass march /
// accumulate / present chain, and the temporal accumulation state machine.
//
// Ported from InteractiveRenderer.encode_pass. The accumulation logic is the
// part worth reading twice — it is what makes a parked camera converge to a
// clean still while a moving one stays smooth instead of boiling.

"use strict";

import * as K from "./constants.js";
import { packUniforms, sceneKey, keysEqual, renderTargetSize,
         motionAlphaForDt } from "./uniforms.js";
import { guardAllocation } from "./gpu.js";

// The march's output for one frame. Half precision is plenty for a single
// sample: its round-off is ~0.03 8-bit levels and it is averaged away.
const SAMPLE_FORMAT = "rgba16float";

// The running mean. This one is a *feedback* buffer — this frame's output is
// next frame's input — so its round-off does not average away, it integrates.
// Measured on this box (NVIDIA/Vulkan), the f32 -> f16 render-target store
// truncates toward zero rather than rounding to nearest: mean error -0.469
// ulp, and the WebGPU spec does not pin the rounding mode, so no
// implementation is obliged to do better. Fed back through
// `prev*i/(i+1) + s/(i+1)` that half-ulp integrates into a systematic
// darkening that grows with frame count (~10 8-bit levels by 1024 frames; see
// the lighting journal, iter_012). Full precision removes it: an f32 store is
// exact for a value the pass just computed in f32.
//
// rgba32float is renderable but NOT filterable in core WebGPU, which is why
// the accumulate pass and the present pass below both read it with
// textureLoad and declare `unfilterable-float`. Nothing here needs a
// filtering sampler, so no optional feature is involved.
const ACCUM_FORMAT = "rgba32float";

const ACCUM_SHADER = `
struct AccumUniforms {
    prev_weight: f32,
    sample_weight: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(0) var<uniform> au: AccumUniforms;
@group(0) @binding(1) var sample_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_tex: texture_2d<f32>;
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(frag_pos.xy);
    let s = textureLoad(sample_tex, xy, 0);
    if (au.prev_weight <= 0.0) {
        return vec4<f32>(s.rgb, 1.0);
    }
    let prev = textureLoad(prev_tex, xy, 0);
    return vec4<f32>(prev.rgb * au.prev_weight + s.rgb * au.sample_weight, 1.0);
}
`;

// Two present paths, matching the engine: an exact texel copy when the render
// target is already the output size, and a bilinear upscale when it is not.
//
// The upscale is done by hand from four textureLoads rather than by a
// filtering sampler, because its source is the rgba32float accumulator and
// core WebGPU cannot filter that. Same taps, same weights, same result as the
// hardware's bilinear; it just costs three extra loads on a pass that runs
// once per frame.
//
// This pass is also the ONLY place a float value in this renderer becomes an
// 8-bit one: the march writes float, accumulation averages float,
// and the canvas is a plain (non-srgb) unorm8 swapchain, so the hardware
// quantizes exactly what fs_main/fs_exact return. That makes it the one
// correct place for the dither — see `DITHER_WGSL`, and see
// cloudyview/basic_render.py's quantize_uint8 for witness's matching encode.
// Anything dithered earlier would be averaged back out by accumulation.
const DITHER_WGSL = `
// TPDF dither, ~1 LSB, for the float -> unorm8 present.
//
// A smooth ramp (the low-sun sky above all, open water, a cloud flank)
// quantizes to its own iso-contours, because the rounding error is a
// deterministic function of the value: constant along a contour, jumping a
// level across it. That is Mach banding, and it is made here, not in the
// march. Adding zero-mean noise before the rounding decorrelates the error
// from the signal; drawing it TPDF (sum of two uniforms) makes the error's
// variance signal-independent too, so the ramp carries an even grain rather
// than a grain that pulses with the contours.
//
// The pattern is a pure function of pixel position and does not advance with
// time: it must be *static*, or the accumulated still would average it away
// (restoring the bands) and a parked camera would shimmer.
//
// Near the 0 and 1 rails the dither tapers to nothing so a clipped highlight
// or a true black stays exactly clipped. Clipping the dithered value instead
// would bias exactly where there is no quantization error to hide.
// Two independent uniforms per call, from pcg2d. The input here is an exact
// integer pixel coordinate, and the usual float sine/fract hashes are lattice
// sequences on integer input — measurably biased (the first version of this
// cost 0.03 LSB of mean brightness). An integer bit-mix has no such structure.
// The three channels are decorrelated by an offset in the input, so each
// stream keeps its own mean of 0.5: a shared stream would tint the noise.
fn dither_hash2(p: vec2<u32>) -> vec2<f32> {
    var v = p * 1664525u + vec2<u32>(1013904223u);
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    v.x = v.x + v.y * 1664525u;
    v.y = v.y + v.x * 1664525u;
    v = v ^ (v >> vec2<u32>(16u));
    return vec2<f32>(v) * (1.0 / 4294967296.0);
}

fn dither_present(rgb: vec3<f32>, frag_xy: vec2<f32>) -> vec3<f32> {
    let lsb = 1.0 / 255.0;
    let v = clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let taper = clamp(min(v, vec3<f32>(1.0) - v) / lsb, vec3<f32>(0.0),
                      vec3<f32>(1.0));
    let p = vec2<u32>(frag_xy);
    let hr = dither_hash2(p);
    let hg = dither_hash2(p + vec2<u32>(17u, 23u));
    let hb = dither_hash2(p + vec2<u32>(41u, 59u));
    let tpdf = vec3<f32>(hr.x + hr.y, hg.x + hg.y, hb.x + hb.y)
             - vec3<f32>(1.0);
    return v + tpdf * lsb * taper;
}
`;

const PRESENT_SHADER = `
${DITHER_WGSL}
@group(0) @binding(0) var src_tex: texture_2d<f32>;

// Clamp-to-edge bilinear, by hand. Matches a linear sampler's texel space:
// the texel centre of texel i is at (i + 0.5) / size.
fn sample_bilinear(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(src_tex, 0));
    let p = uv * size - vec2<f32>(0.5);
    let base = floor(p);
    let f = p - base;
    let hi = vec2<i32>(size) - vec2<i32>(1);
    let i0 = clamp(vec2<i32>(base), vec2<i32>(0), hi);
    let i1 = clamp(vec2<i32>(base) + vec2<i32>(1), vec2<i32>(0), hi);
    let c00 = textureLoad(src_tex, vec2<i32>(i0.x, i0.y), 0).rgb;
    let c10 = textureLoad(src_tex, vec2<i32>(i1.x, i0.y), 0).rgb;
    let c01 = textureLoad(src_tex, vec2<i32>(i0.x, i1.y), 0).rgb;
    let c11 = textureLoad(src_tex, vec2<i32>(i1.x, i1.y), 0).rgb;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) & 1) * 4.0 - 1.0;
    var out: VOut;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}
@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let rgb = sample_bilinear(in.uv);
    return vec4<f32>(dither_present(rgb, floor(in.position.xy)), 1.0);
}
@fragment
fn fs_exact(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let rgb = textureLoad(src_tex, vec2<i32>(frag_pos.xy), 0).rgb;
    return vec4<f32>(dither_present(rgb, floor(frag_pos.xy)), 1.0);
}
`;

/**
 * Bake the three compile-time constants into raymarch.wgsl.
 *
 * These are textual replacements rather than WGSL `override`s on purpose:
 * folding them at compile time removes the branches from the march's hot
 * loops entirely, and MAX_LIGHT_STEPS bounds a loop, which an overridable
 * value cannot do. Each sentinel must appear exactly once — a shader that
 * quietly failed to specialize would render the wrong thing at full speed.
 */
/**
 * Run whatever wants to draw on top of the finished frame — the minimap, the
 * bird. Each is `(encoder, targetView, targetFormat)` and encodes its own
 * load-op-"load" pass; nulls are skipped so a caller can pass a fixed-shape
 * list of things that are individually on or off.
 */
function encodeOverlays(overlays, encoder, targetView, targetFormat) {
  if (!overlays) return;
  for (const overlay of overlays) {
    overlay?.(encoder, targetView, targetFormat);
  }
}

export function specializeShader(source, { periodic, nested, maxLightSteps }) {
  if (!(maxLightSteps >= 1 && maxLightSteps <= K.DEFAULT_MAX_LIGHT_STEPS)) {
    throw new Error(
      `max_light_steps must be in [1, ${K.DEFAULT_MAX_LIGHT_STEPS}]; ` +
      `got ${maxLightSteps}.`);
  }
  const swaps = [
    ["const PERIODIC_DOMAIN: bool = true;",
     `const PERIODIC_DOMAIN: bool = ${periodic ? "true" : "false"};`],
    ["const NESTED: bool = false;",
     `const NESTED: bool = ${nested ? "true" : "false"};`],
    ["const MAX_LIGHT_STEPS: i32 = 512;",
     `const MAX_LIGHT_STEPS: i32 = ${maxLightSteps};`],
  ];
  let out = source;
  for (const [sentinel, replacement] of swaps) {
    const count = out.split(sentinel).length - 1;
    if (count !== 1) {
      throw new Error(
        `raymarch.wgsl must contain "${sentinel}" exactly once; found ` +
        `${count}. The shader and this host are out of step.`);
    }
    out = out.replace(sentinel, replacement);
  }
  return out;
}

export class Renderer {
  /**
   * `scene` supplies the resident GPU resources and the field geometry:
   * {volumeView, nestView, oceanView, bmin, bmax, minVoxelM, minVoxelNestM,
   *  nested, nestBmin, nestBmax, oceanFifDx, oceanTileExtent, oceanMaxLod}.
   */
  constructor(device, shaderSource, scene, { canvasFormat }) {
    this.device = device;
    this.shaderSource = shaderSource;
    this.scene = scene;
    this.canvasFormat = canvasFormat;

    this.periodic = true;
    this.qualityTier = K.DEFAULT_QUALITY_TIER;
    this._flightRenderScale = K.QUALITY_PRESETS[this.qualityTier].renderScale;
    this._cameraMoving = false;
    this.renderScale = this._flightRenderScale;
    this.stepFactor = K.QUALITY_PRESETS[this.qualityTier].stepFactor;
    this.maxLightSteps = K.QUALITY_PRESETS[this.qualityTier].maxLightSteps;

    this.motionBlendAlpha = K.DEFAULT_MOTION_BLEND_ALPHA;
    this.motionResetTranslationM = K.DEFAULT_MOTION_RESET_TRANSLATION_FRACTION *
      Math.min(scene.bmax[0] - scene.bmin[0], scene.bmax[1] - scene.bmin[1]);

    this._modules = new Map();
    this._pipelines = new Map();
    this._targets = null;
    this._resetAccumulation();

    this.uniformBuf = device.createBuffer({
      label: "soar-uniforms",
      size: K.UNIFORM_NBYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.accumUniformBuf = device.createBuffer({
      label: "soar-accum-uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.volSampler = device.createSampler({
      addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
      addressModeW: "clamp-to-edge", magFilter: "linear", minFilter: "linear",
    });
    this.oceanSampler = device.createSampler({
      addressModeU: "repeat", addressModeV: "repeat",
      magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
    });
    this.rayLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "3d" } },
      ],
    });
    this.rayPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.rayLayout],
    });
    this.refreshBindGroup();

    // One layout for both present entry points: a single unfilterable-float
    // texture and no sampler. `unfilterable-float` is the weaker requirement,
    // so it accepts the rgba16float sample target as well as the rgba32float
    // accumulator, and neither entry point filters any more.
    this.blitLayout = device.createBindGroupLayout({
      label: "present",
      entries: [{
        binding: 0, visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "unfilterable-float", viewDimension: "2d" },
      }],
    });
    this.blitPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.blitLayout],
    });

    // Explicit, not `layout: "auto"`: auto would infer a filterable-float
    // entry for prev_tex, and the rgba32float accumulator cannot satisfy it.
    // WebGPU reports that asynchronously, so the symptom would be a black
    // canvas and one console line.
    this.accumLayout = device.createBindGroupLayout({
      label: "accumulate",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      ],
    });

    const accumModule = device.createShaderModule({ code: ACCUM_SHADER });
    const presentModule = device.createShaderModule({ code: PRESENT_SHADER });
    this.presentModule = presentModule;
    this._shaderModules = [accumModule, presentModule];
    this.accumPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.accumLayout],
      }),
      vertex: { module: accumModule, entryPoint: "vs_main" },
      fragment: { module: accumModule, entryPoint: "fs_main",
                  targets: [{ format: ACCUM_FORMAT }] },
      primitive: { topology: "triangle-list" },
    });
    this._blitPipelines = new Map();
  }

  /**
   * Compile and validate before the first frame.
   *
   * WebGPU hands back a shader module and a pipeline whether or not they are
   * valid, and reports the failure asynchronously — so the default symptom of
   * a broken shader or a mismatched layout is a black canvas and a line in
   * the console. This asks the questions up front and throws with the WGSL
   * error text, which the failure panel then shows.
   */
  async init() {
    this._shaderModules.push(this._module(
      this.periodic, this.scene.nested, this.maxLightSteps));

    const problems = [];
    for (const module of this._shaderModules) {
      const info = await module.getCompilationInfo?.();
      for (const message of info?.messages ?? []) {
        if (message.type !== "error") continue;
        problems.push(
          `line ${message.lineNum}:${message.linePos} — ${message.message}`);
      }
    }
    if (problems.length) {
      throw new Error(`The shader did not compile.\n${problems.join("\n")}`);
    }

    this.device.pushErrorScope("validation");
    // The march only ever targets the float intermediate now; everything
    // reaches the canvas through the (dithering) present pass.
    this._rayPipeline(SAMPLE_FORMAT);
    this._blitPipeline(this.canvasFormat, true);
    this._blitPipeline(this.canvasFormat, false);
    const error = await this.device.popErrorScope();
    if (error) {
      throw new Error(`Setting up the render pipelines failed: ${error.message}`);
    }
  }

  /**
   * Rebuild the bind group against the scene's current textures. Must be
   * called whenever a nest is attached or removed — binding 5 otherwise still
   * points at the 1x1x1 stand-in and the nest renders as nothing at all.
   */
  refreshBindGroup() {
    const scene = this.scene;
    this.rayBindGroup = this.device.createBindGroup({
      label: "soar-raymarch",
      layout: this.rayLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: scene.volumeView },
        { binding: 2, resource: this.volSampler },
        { binding: 3, resource: scene.oceanView },
        { binding: 4, resource: this.oceanSampler },
        { binding: 5, resource: scene.nestView },
      ],
    });
  }

  // --- quality ------------------------------------------------------------

  get flightRenderScale() { return this._flightRenderScale; }

  get qualityIsCustom() {
    return this._flightRenderScale !==
      K.QUALITY_PRESETS[this.qualityTier].renderScale;
  }

  setQualityTier(tier, cameraMoving = null) {
    if (!(tier in K.QUALITY_PRESETS)) {
      throw new Error(`unknown quality tier '${tier}'.`);
    }
    if (cameraMoving !== null) this._cameraMoving = Boolean(cameraMoving);
    this.qualityTier = tier;
    this._flightRenderScale = K.QUALITY_PRESETS[tier].renderScale;
    this._applyEffectiveQuality();
  }

  setRenderScale(scale) {
    renderTargetSize([1, 1], scale);   // validates, throws on a bad value
    this._flightRenderScale = Number(scale);
    this._applyEffectiveQuality();
  }

  setCameraMoving(moving) {
    moving = Boolean(moving);
    if (moving === this._cameraMoving) return;
    this._cameraMoving = moving;
    this._applyEffectiveQuality();
  }

  /**
   * Potato is the only tier that behaves differently parked: it swaps to
   * High's sampling AND forces full render scale, so a still converges
   * properly instead of accumulating a quarter-resolution smear.
   */
  _applyEffectiveQuality() {
    const preset = K.QUALITY_PRESETS[this.qualityTier];
    let effective = preset;
    if (this.qualityTier === "potato" && !this._cameraMoving) {
      effective = K.QUALITY_PRESETS.high;
    }
    const renderScale =
      (effective.name === "high" && this.qualityTier === "potato")
        ? 1.0 : this._flightRenderScale;
    const changed = renderScale !== this.renderScale
      || effective.stepFactor !== this.stepFactor
      || effective.maxLightSteps !== this.maxLightSteps;
    this.renderScale = renderScale;
    this.stepFactor = effective.stepFactor;
    this.maxLightSteps = effective.maxLightSteps;
    if (changed) this._resetAccumulation();
  }

  setPeriodic(periodic) {
    periodic = Boolean(periodic);
    if (periodic === this.periodic) return;
    this.periodic = periodic;
    // The ghost texels must agree with the shader's branch; the caller
    // rewrites the border. Accumulation resets by itself — row 20.x is in
    // the scene-identity key.
    this.scene.writeGhostBorder?.(periodic);
  }

  get dtView() { return this.scene.minVoxelM * this.stepFactor; }
  get dtLight() { return this.dtView; }
  get dtViewNest() {
    return (this.scene.minVoxelNestM ?? this.scene.minVoxelM) * this.stepFactor;
  }
  get dtLightNest() { return this.dtViewNest; }

  // --- pipelines ----------------------------------------------------------

  _module(periodic, nested, maxLightSteps) {
    const key = `${periodic}|${nested}|${maxLightSteps}`;
    let module = this._modules.get(key);
    if (!module) {
      module = this.device.createShaderModule({
        label: `raymarch(${key})`,
        code: specializeShader(this.shaderSource,
                               { periodic, nested, maxLightSteps }),
      });
      this._modules.set(key, module);
    }
    return module;
  }

  _rayPipeline(targetFormat) {
    const key = `${targetFormat}|${this.periodic}|${this.scene.nested}|` +
                `${this.maxLightSteps}`;
    let pipeline = this._pipelines.get(key);
    if (!pipeline) {
      const module = this._module(
        this.periodic, this.scene.nested, this.maxLightSteps);
      pipeline = this.device.createRenderPipeline({
        label: `raymarch(${key})`,
        layout: this.rayPipelineLayout,
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint: "fs_main",
                    targets: [{ format: targetFormat }] },
        primitive: { topology: "triangle-list" },
      });
      this._pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  /**
   * The present pass, in two flavours: an exact texel copy when the render
   * target is already the output size, and a bilinear upscale when it is not.
   *
   * Both share one explicit layout — one unfilterable-float texture, no
   * sampler. Explicit rather than `layout: "auto"` because auto derives the
   * layout from the bindings an entry point actually uses and infers
   * filterable float for a plain `texture_2d<f32>`, which the rgba32float
   * accumulator cannot satisfy; WebGPU reports that asynchronously, so the
   * only symptom would be a black canvas and a console line.
   */
  _blitPipeline(targetFormat, exact) {
    const key = `${targetFormat}|${exact}`;
    let pipeline = this._blitPipelines.get(key);
    if (!pipeline) {
      pipeline = this.device.createRenderPipeline({
        label: `present(${key})`,
        layout: this.blitPipelineLayout,
        vertex: { module: this.presentModule, entryPoint: "vs_main" },
        fragment: { module: this.presentModule,
                    entryPoint: exact ? "fs_exact" : "fs_main",
                    targets: [{ format: targetFormat }] },
        primitive: { topology: "triangle-list" },
      });
      this._blitPipelines.set(key, pipeline);
    }
    return pipeline;
  }

  async _targetsFor(w, h) {
    if (this._targets && this._targets.w === w && this._targets.h === h) {
      return this._targets;
    }
    // Frames already submitted may still be reading these. Destroying a
    // texture out from under in-flight commands is legal by the letter of the
    // spec, but it is the kind of thing that segfaults a browser rather than
    // raising, and a resize happens once, not per frame — so wait.
    if (this._targets) await this.device.queue.onSubmittedWorkDone();
    for (const t of [this._targets?.sample, this._targets?.accumA,
                     this._targets?.accumB]) {
      t?.destroy();
    }
    this._lastPresented = null;   // it pointed into what we just destroyed
    // 8 bytes for the half-precision sample target, 16 each for the two
    // full-precision accumulators.
    const bytes = w * h * (8 + 16 + 16);
    const make = (label, format) => this.device.createTexture({
      label, size: [w, h], format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.COPY_SRC,
    });
    this._targets = await guardAllocation(
      this.device, `${w}x${h} render targets`, bytes, () => ({
        w, h,
        sample: make("soar-sample", SAMPLE_FORMAT),
        accumA: make("soar-accum-a", ACCUM_FORMAT),
        accumB: make("soar-accum-b", ACCUM_FORMAT),
      }));
    this._resetAccumulation();
    return this._targets;
  }

  // --- accumulation -------------------------------------------------------

  _resetAccumulation() {
    this._accumKey = null;
    this._accumCount = 0;
    this._accumIndex = 0;
    this._accumMotion = false;
    this._accumLastOrigin = null;
    this._accumLastForward = null;
    this.lastMotionReset = false;
  }

  resetAccumulation() { this._resetAccumulation(); }

  _motionDeltaExceedsReset(origin, forward) {
    if (!this._accumLastOrigin || !this._accumLastForward) return true;
    const d = [
      origin[0] - this._accumLastOrigin[0],
      origin[1] - this._accumLastOrigin[1],
      origin[2] - this._accumLastOrigin[2],
    ];
    const translation = Math.hypot(d[0], d[1], d[2]);
    let dot = 0;
    for (let i = 0; i < 3; i++) dot += forward[i] * this._accumLastForward[i];
    const angle = Math.acos(Math.min(1, Math.max(-1, dot))) * 180 / Math.PI;
    return translation > this.motionResetTranslationM
        || angle > K.DEFAULT_MOTION_RESET_ANGLE_DEGREES;
  }

  /**
   * Decide this frame's blend weights. Three cases, and the middle one is the
   * subtle one: when the camera stops, the smeared motion buffer is thrown
   * away rather than averaged into the still, otherwise the converged image
   * keeps a ghost of the approach.
   */
  _accumulationPlan(u, deltaSeconds) {
    const currentKey = sceneKey(
      u, this.maxLightSteps, this._outputW, this._outputH);
    const origin = [u[0], u[1], u[2]];
    const forward = [u[4], u[5], u[6]];

    const prevCount = this._accumCount;
    let nextCount = prevCount + 1;
    let prevWeight = prevCount === 0 ? 0.0 : prevCount / nextCount;
    let sampleWeight = prevCount === 0 ? 1.0 : 1.0 / nextCount;
    let subpixel = prevCount >= 1;
    let jitterScale = 1.0;
    this.lastMotionReset = false;

    if (this._accumKey === null) {
      this._accumKey = currentKey;
      this._accumMotion = false;
      subpixel = false;
    } else if (keysEqual(this._accumKey, currentKey)) {
      if (this._accumMotion) {
        this._accumCount = 0; this._accumIndex = 0; this._accumMotion = false;
        prevWeight = 0.0; sampleWeight = 1.0; nextCount = 1; subpixel = false;
      }
    } else {
      const alpha = motionAlphaForDt(
        this.motionBlendAlpha, K.DEFAULT_MOTION_BLEND_REFERENCE_FPS,
        deltaSeconds);
      const resetForJump = this._motionDeltaExceedsReset(origin, forward);
      this._accumKey = currentKey;
      if (alpha < 1.0 && !resetForJump) {
        prevWeight = 1.0 - alpha; sampleWeight = alpha; nextCount = 1;
        subpixel = true; jitterScale = K.DEFAULT_MOTION_JITTER_SCALE;
        this._accumMotion = true;
      } else {
        this._accumCount = 0; this._accumIndex = 0;
        prevWeight = 0.0; sampleWeight = 1.0; nextCount = 1;
        subpixel = false; this._accumMotion = false;
        this.lastMotionReset = true;
      }
    }

    this._accumLastOrigin = origin;
    this._accumLastForward = forward;
    return { prevWeight, sampleWeight, nextCount, subpixel, jitterScale };
  }

  // --- the frame ----------------------------------------------------------

  /**
   * March, accumulate, present. `view` is everything packUniforms takes bar
   * the sampling flags, which this method owns.
   *
   * `target` may be a texture view or a function returning one, and for the
   * canvas it MUST be the function. A swapchain texture is only valid until
   * the end of the animation-frame callback that asked for it, and this
   * method awaits the render targets first — an await that goes to the GPU
   * and back whenever the canvas has just been resized. Taking the view
   * before that await meant drawing into a texture the compositor had
   * already reclaimed, which is not an exception, it is a crashed browser.
   */
  async drawFrame(target, targetFormat, outputSize, view,
                  { deltaSeconds = null, accumulate = true,
                    overlays = null } = {}) {
    const [outputW, outputH] = outputSize;
    this._outputW = outputW;
    this._outputH = outputH;
    const renderSize = renderTargetSize(outputSize, this.renderScale);
    const targets = await this._targetsFor(renderSize[0], renderSize[1]);
    const targetView = typeof target === "function" ? target() : target;

    // Pack once to learn the scene key, then re-pack with the sampling flags
    // the plan chose. Row 10 is excluded from the key precisely so this is
    // safe — the flags cannot change the decision that produced them.
    const state = this._sceneState();
    let u = packUniforms(state, { ...view, outputSize, renderSize });
    const plan = accumulate && view.jitter !== false
      ? this._accumulationPlan(u, deltaSeconds)
      : null;
    if (plan) {
      u = packUniforms(state, {
        ...view, outputSize, renderSize,
        subpixel: plan.subpixel, jitterScale: plan.jitterScale,
      });
    } else {
      this._resetAccumulation();
    }
    this.device.queue.writeBuffer(this.uniformBuf, 0, u);

    const encoder = this.device.createCommandEncoder();
    const pass = (viewRef) => encoder.beginRenderPass({
      colorAttachments: [{
        view: viewRef, loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });

    if (!plan) {
      // No accumulation: march at a float intermediate and present from it.
      // The march could target the canvas directly when the sizes match, and
      // used to — but then the hardware would quantize the march's own output
      // and this frame would be the one path in the renderer that skips the
      // present pass's dither. One full-screen blit is cheaper than a second
      // encode point.
      const exact = renderSize[0] === outputW && renderSize[1] === outputH;
      const p = pass(targets.sample.createView());
      p.setPipeline(this._rayPipeline(SAMPLE_FORMAT));
      p.setBindGroup(0, this.rayBindGroup);
      p.draw(3);
      p.end();
      this._encodeBlit(encoder, targets.sample, targetView, targetFormat,
                       exact);
      encodeOverlays(overlays, encoder, targetView, targetFormat);
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    this.device.queue.writeBuffer(this.accumUniformBuf, 0, new Float32Array(
      [plan.prevWeight, plan.sampleWeight, 0, 0]));

    let p = pass(targets.sample.createView());
    p.setPipeline(this._rayPipeline(SAMPLE_FORMAT));
    p.setBindGroup(0, this.rayBindGroup);
    p.draw(3);
    p.end();

    const prevTex = this._accumIndex === 0 ? targets.accumA : targets.accumB;
    const outTex = this._accumIndex === 0 ? targets.accumB : targets.accumA;
    p = pass(outTex.createView());
    p.setPipeline(this.accumPipeline);
    p.setBindGroup(0, this.device.createBindGroup({
      layout: this.accumLayout,
      entries: [
        { binding: 0, resource: { buffer: this.accumUniformBuf } },
        { binding: 1, resource: targets.sample.createView() },
        { binding: 2, resource: prevTex.createView() },
      ],
    }));
    p.draw(3);
    p.end();

    this._encodeBlit(encoder, outTex, targetView, targetFormat,
                     renderSize[0] === outputW && renderSize[1] === outputH);
    encodeOverlays(overlays, encoder, targetView, targetFormat);
    this.device.queue.submit([encoder.finish()]);

    this._accumIndex = 1 - this._accumIndex;
    this._accumCount = plan.nextCount;
    this._lastPresented = outTex;
  }

  /**
   * Overlays are encoded into the SAME command buffer as the frame they sit
   * on, after the blit that put it on screen. A separate submit would work
   * but would let the compositor pick up the bare frame in between, which
   * reads as the minimap flickering.
   */
  _encodeBlit(encoder, srcTex, targetView, targetFormat, exact) {
    const entries = [{ binding: 0, resource: srcTex.createView() }];
    const p = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView, loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    p.setPipeline(this._blitPipeline(targetFormat, exact));
    p.setBindGroup(0, this.device.createBindGroup({
      layout: this.blitLayout, entries,
    }));
    p.draw(3);
    p.end();
  }

  /** Re-present the last accumulated frame without marching again. */
  presentLast(targetView, targetFormat, outputSize) {
    if (!this._lastPresented || !this._targets) return false;
    const encoder = this.device.createCommandEncoder();
    this._encodeBlit(
      encoder, this._lastPresented, targetView, targetFormat,
      this._targets.w === outputSize[0] && this._targets.h === outputSize[1]);
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  /** Release the render targets. Pipelines and modules go with the object. */
  destroy() {
    for (const t of [this._targets?.sample, this._targets?.accumA,
                     this._targets?.accumB]) {
      t?.destroy();
    }
    this._targets = null;
    this._lastPresented = null;
    this.uniformBuf?.destroy();
    this.accumUniformBuf?.destroy();
  }

  _sceneState() {
    const s = this.scene;
    return {
      bmin: s.bmin, bmax: s.bmax,
      dtView: this.dtView, dtLight: this.dtLight,
      periodic: this.periodic,
      oceanZ: 0.0,
      oceanReflectance: K.DEFAULT_OCEAN_REFLECTANCE,
      oceanFifDx: s.oceanFifDx,
      oceanTileExtent: s.oceanTileExtent,
      oceanEnabled: true,
      oceanMaxLod: s.oceanMaxLod,
      nested: s.nested,
      nestBmin: s.nestBmin,
      nestBmax: s.nestBmax,
      dtViewNest: this.dtViewNest,
      dtLightNest: this.dtLightNest,
    };
  }
}
