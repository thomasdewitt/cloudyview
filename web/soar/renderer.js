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

const ACCUM_FORMAT = "rgba16float";

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
const PRESENT_SHADER = `
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
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
    return vec4<f32>(textureSampleLevel(src_tex, src_samp, in.uv, 0.0).rgb, 1.0);
}
@fragment
fn fs_exact(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureLoad(src_tex, vec2<i32>(frag_pos.xy), 0).rgb, 1.0);
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
    this.blitSampler = device.createSampler({
      magFilter: "linear", minFilter: "linear",
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

    const accumModule = device.createShaderModule({ code: ACCUM_SHADER });
    const presentModule = device.createShaderModule({ code: PRESENT_SHADER });
    this.presentModule = presentModule;
    this.accumPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: accumModule, entryPoint: "vs_main" },
      fragment: { module: accumModule, entryPoint: "fs_main",
                  targets: [{ format: ACCUM_FORMAT }] },
      primitive: { topology: "triangle-list" },
    });
    this._blitPipelines = new Map();
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

  _blitPipeline(targetFormat, exact) {
    const key = `${targetFormat}|${exact}`;
    let pipeline = this._blitPipelines.get(key);
    if (!pipeline) {
      pipeline = this.device.createRenderPipeline({
        label: `present(${key})`,
        layout: "auto",
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
    for (const t of [this._targets?.sample, this._targets?.accumA,
                     this._targets?.accumB]) {
      t?.destroy();
    }
    this._lastPresented = null;   // it pointed into what we just destroyed
    const bytes = w * h * 8 * 3;
    const make = (label) => this.device.createTexture({
      label, size: [w, h], format: ACCUM_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
           | GPUTextureUsage.COPY_SRC,
    });
    this._targets = await guardAllocation(
      this.device, `${w}x${h} render targets`, bytes, () => ({
        w, h,
        sample: make("soar-sample"),
        accumA: make("soar-accum-a"),
        accumB: make("soar-accum-b"),
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
   */
  async drawFrame(targetView, targetFormat, outputSize, view,
                  { deltaSeconds = null, accumulate = true } = {}) {
    const [outputW, outputH] = outputSize;
    this._outputW = outputW;
    this._outputH = outputH;
    const renderSize = renderTargetSize(outputSize, this.renderScale);
    const targets = await this._targetsFor(renderSize[0], renderSize[1]);

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
      // No accumulation: march straight at the output (through a scaled
      // intermediate when the render size differs).
      if (renderSize[0] === outputW && renderSize[1] === outputH) {
        const p = pass(targetView);
        p.setPipeline(this._rayPipeline(targetFormat));
        p.setBindGroup(0, this.rayBindGroup);
        p.draw(3);
        p.end();
      } else {
        let p = pass(targets.sample.createView());
        p.setPipeline(this._rayPipeline(ACCUM_FORMAT));
        p.setBindGroup(0, this.rayBindGroup);
        p.draw(3);
        p.end();
        this._encodeBlit(encoder, targets.sample, targetView, targetFormat,
                         false);
      }
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    this.device.queue.writeBuffer(this.accumUniformBuf, 0, new Float32Array(
      [plan.prevWeight, plan.sampleWeight, 0, 0]));

    let p = pass(targets.sample.createView());
    p.setPipeline(this._rayPipeline(ACCUM_FORMAT));
    p.setBindGroup(0, this.rayBindGroup);
    p.draw(3);
    p.end();

    const prevTex = this._accumIndex === 0 ? targets.accumA : targets.accumB;
    const outTex = this._accumIndex === 0 ? targets.accumB : targets.accumA;
    p = pass(outTex.createView());
    p.setPipeline(this.accumPipeline);
    p.setBindGroup(0, this.device.createBindGroup({
      layout: this.accumPipeline.getBindGroupLayout(0),
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
    this.device.queue.submit([encoder.finish()]);

    this._accumIndex = 1 - this._accumIndex;
    this._accumCount = plan.nextCount;
    this._lastPresented = outTex;
  }

  _encodeBlit(encoder, srcTex, targetView, targetFormat, exact) {
    const p = encoder.beginRenderPass({
      colorAttachments: [{
        view: targetView, loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    p.setPipeline(this._blitPipeline(targetFormat, exact));
    p.setBindGroup(0, this.device.createBindGroup({
      layout: this._blitPipeline(targetFormat, exact).getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: this.blitSampler },
      ],
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
