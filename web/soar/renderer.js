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
import { guardAllocation, retireAfterSubmittedWork } from "./gpu.js";
import { desertReliefM, desertTileRows } from "./scene.js";

// Frame-index spacing between the samples of one frame. The index seeds the
// jitter and the dither, and the viewer's own index steps by 1 per frame, so
// a stride of 1 here would make sample s of this frame the twin of sample 0
// of the frame s later — correlated noise that averages to a pattern.
const SPP_FRAME_INDEX_STRIDE = 4096;

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
// The surface pre-pass target: (radiance, t) of the traced city or desert
// under each render pixel (raymarch.wgsl fs_surface). t runs to 1e30 on a
// miss, so it is float32; the march only textureLoad()s it.
const SURFACE_FORMAT = "rgba32float";
// The veil the first and last surface stages write beside the surface
// (raymarch.wgsl veil_pack): half-float radiance and opacity and an f32
// depth, packed as uints so no float path touches the bits.
const VEIL_FORMAT = "rgba32uint";
// The optical depths the full desert's first stage writes in the veil's
// place (K.TAUS_STAGE), f32 so its flora stage rebuilds the very same light
// (raymarch.wgsl desert_light_from). soar_host.TAUS_FORMAT is the twin.
const TAUS_FORMAT = "rg32float";

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
/** The lighting method the sun-tau bake is compiled with; see _bakePipeline. */
const BAKE_LIGHTING = Object.freeze({ lightCache: false, skyProbe: true });

/** Whether a _pipeline answer is still a compile in flight. */
function isPending(pipelineOrPromise) {
  return typeof pipelineOrPromise?.then === "function";
}

function encodeOverlays(overlays, encoder, targetView, targetFormat) {
  if (!overlays) return;
  for (const overlay of overlays) {
    overlay?.(encoder, targetView, targetFormat);
  }
}

/**
 * Where each of `n` chained surface stages writes and what it reads:
 * [writes, reads] with "final" the surface the march reads and "a"/"b" the
 * two intermediates the chain alternates between (reads null: the first
 * stage). soar_host.surface_stage_plan is the twin.
 */
export function surfaceStagePlan(n) {
  const plan = [];
  for (let i = 0; i < n; i++) {
    plan.push([i === n - 1 ? "final" : "ab"[i % 2],
               i === 0 ? null : "ab"[(i - 1) % 2]]);
  }
  return plan;
}

/** A surface stage's colour targets: the surface, and beside it the veil
 *  for a VEIL_STAGES entry or the optical depths for K.TAUS_STAGE. */
function stageFormats(entry) {
  if (K.VEIL_STAGES.includes(entry)) return [SURFACE_FORMAT, VEIL_FORMAT];
  if (entry === K.TAUS_STAGE) return [SURFACE_FORMAT, TAUS_FORMAT];
  return SURFACE_FORMAT;
}

// Section stripping (raymarch.wgsl's `// @section` markers): a module keeps
// the compiled surface's sections and the other surfaces' stubs, so it
// carries no code of a surface it cannot show — another surface's dead code
// had been moving this one's codegen. soar_host.strip_sections is the twin;
// tests/test_wgsl_sections.py pins the two to the same bytes.
const SECTION_NAMES = ["sea", "city", "desert"];
const SECTION_MARKER = /^\/\/ @(section|end|stub|end-stub) (\w+)\s*$/;

/**
 * `source` with every section but `keep`'s blanked, and `keep`'s stub
 * blanked. Lines are blanked, not removed, so a compile error's line number
 * is still the file's. A source with no markers passes through; one with
 * any must be complete (every surface: at least one section, exactly one
 * stub; no nesting; every block closed).
 */
export function stripSections(source, keep) {
  if (!SECTION_NAMES.includes(keep)) {
    throw new Error(`unknown section ${keep}`);
  }
  const lines = source.split("\n");
  const marks = [];
  lines.forEach((l, i) => {
    const m = SECTION_MARKER.exec(l);
    if (m) marks.push([i, m[1], m[2]]);
  });
  if (marks.length === 0) return source;
  const seen = Object.fromEntries(
    SECTION_NAMES.map((n) => [n, { section: 0, stub: 0 }]));
  const out = lines.slice();
  let open = null;                     // [kind, name, line]
  for (const [i, kind, name] of marks) {
    if (!SECTION_NAMES.includes(name)) {
      throw new Error(`raymarch.wgsl line ${i + 1}: unknown section ` +
                      `${name}; the sections are ${SECTION_NAMES}.`);
    }
    if (kind === "section" || kind === "stub") {
      if (open) {
        throw new Error(`raymarch.wgsl line ${i + 1}: @${kind} ${name} ` +
                        `opens inside @${open[0]} ${open[1]} (line ` +
                        `${open[2] + 1}); blocks do not nest.`);
      }
      open = [kind, name, i];
      seen[name][kind] += 1;
      continue;
    }
    const want = kind === "end" ? "section" : "stub";
    if (!open || open[0] !== want || open[1] !== name) {
      throw new Error(`raymarch.wgsl line ${i + 1}: @${kind} ${name} ` +
                      `closes nothing open as @${want} ${name}.`);
    }
    const drop = want === "section" ? name !== keep : name === keep;
    if (drop) {
      for (let j = open[2] + 1; j < i; j++) out[j] = "";
    }
    open = null;
  }
  if (open) {
    throw new Error(`raymarch.wgsl line ${open[2] + 1}: @${open[0]} ` +
                    `${open[1]} is never closed.`);
  }
  for (const name of SECTION_NAMES) {
    const n = seen[name];
    if (n.section < 1 || n.stub !== 1) {
      throw new Error(`raymarch.wgsl: ${name} has ${n.section} section ` +
                      `block(s) and ${n.stub} stub block(s); a marked ` +
                      "shader needs at least one and exactly one.");
    }
  }
  return out.join("\n");
}

export function specializeShader(source,
                                 { periodic, nested, maxLightSteps,
                                   toneMap = true, city = false,
                                   desert = false, desertLite = false,
                                   desertFloraCut = 1.0,
                                   lightCache, skyProbe }) {
  if (!(maxLightSteps >= 1 && maxLightSteps <= K.DEFAULT_MAX_LIGHT_STEPS)) {
    throw new Error(
      `max_light_steps must be in [1, ${K.DEFAULT_MAX_LIGHT_STEPS}]; ` +
      `got ${maxLightSteps}.`);
  }
  // The lighting method has no default here on purpose: it is the same pair
  // of flags packUniforms writes into row 23, and a caller that forgot to
  // say which method it compiled would render one method under the other's
  // scene key. See Renderer.lighting.
  if (typeof lightCache !== "boolean" || typeof skyProbe !== "boolean") {
    throw new Error(
      "specializeShader needs the lighting method: lightCache and skyProbe " +
      `as booleans; got ${lightCache} and ${skyProbe}.`);
  }
  const swaps = [
    ["const PERIODIC_DOMAIN: bool = true;",
     `const PERIODIC_DOMAIN: bool = ${periodic ? "true" : "false"};`],
    ["const NESTED: bool = false;",
     `const NESTED: bool = ${nested ? "true" : "false"};`],
    ["const MAX_LIGHT_STEPS: i32 = 512;",
     `const MAX_LIGHT_STEPS: i32 = ${maxLightSteps};`],
    // The auto-exposure meter compiles the encode out and reads linear HDR —
    // the same switch witness's return_linear uses (soar_host.specialize).
    ["const TONE_MAP: bool = true;",
     `const TONE_MAP: bool = ${toneMap ? "true" : "false"};`],
    ["const LIGHT_CACHE: bool = false;",
     `const LIGHT_CACHE: bool = ${lightCache ? "true" : "false"};`],
    ["const SKY_PROBE: bool = true;",
     `const SKY_PROBE: bool = ${skyProbe ? "true" : "false"};`],
  ];
  // Demanded only when asked for, like soar_host.specialize: a shader that
  // predates the night city still serves every daytime render.
  if (city && desert) {
    throw new Error("a scene stands on one surface: city and desert are " +
                    "both set.");
  }
  if (city) {
    swaps.push(["const CITY: bool = false;", "const CITY: bool = true;"]);
  }
  if (desert) {
    swaps.push(["const DESERT: bool = false;", "const DESERT: bool = true;"]);
  }
  // The simplified desert, demanded (like CITY and DESERT) only when turned
  // on, and only ever over a desert.
  if (desertLite) {
    if (!desert) {
      throw new Error("desertLite without desert: the simplified desert is " +
                      "a desert specialization.");
    }
    swaps.push(["const DESERT_LITE: bool = false;",
                "const DESERT_LITE: bool = true;"]);
  }
  // The flora's explicit cut, the same way: demanded only when it departs
  // from the shader's declared 1.0, and only ever over a desert.
  if (!(Number.isFinite(desertFloraCut) && desertFloraCut > 0)) {
    throw new Error(`desertFloraCut must be a positive number; got ` +
                    `${desertFloraCut}.`);
  }
  if (desertFloraCut !== 1.0) {
    if (!desert) {
      throw new Error("desertFloraCut without desert: the flora's cut is " +
                      "a desert specialization.");
    }
    swaps.push(["const DESERT_FLORA_CUT: f32 = 1.0;",
                `const DESERT_FLORA_CUT: f32 = ${wgslFloat(desertFloraCut)};`]);
  }
  let out = stripSections(source, city ? "city" : (desert ? "desert" : "sea"));
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

/**
 * The one cache key for a raymarch.wgsl specialization: every field of
 * specializeShader's argument, in a fixed order. Every module and pipeline
 * cache in this file keys on it (plus the entry point and target format),
 * so a new specialization axis is added here once rather than threaded
 * through each cache's own string.
 */
export function shaderVariantKey(spec) {
  const surface = spec.city ? "city" : (spec.desert ? "desert" : "ocean");
  return [spec.periodic, spec.nested, spec.maxLightSteps, spec.toneMap,
          surface, Boolean(spec.desertLite), spec.desertFloraCut ?? 1.0,
          spec.lightCache, spec.skyProbe].join("|");
}

/** A number as a WGSL f32 literal: always with a decimal point. */
function wgslFloat(x) {
  return Number.isInteger(x) ? x.toFixed(1) : String(x);
}

export class Renderer {
  /**
   * `scene` supplies the resident GPU resources and the field geometry:
   * {volumeView, nestView, oceanView, seaView, bmin, bmax, minVoxelM,
   *  minVoxelNestM, nested, nestBmin, nestBmax, oceanFifDx, oceanTileExtent,
   *  surfacePeriod, oceanMaxLod}. oceanView (binding 3, the city's or the
   *  desert's tile) and seaView (binding 9, the moving sea) are each null
   *  on the scenes that do not read them.
   */
  constructor(device, shaderSource, scene, { canvasFormat }) {
    this.device = device;
    this.shaderSource = shaderSource;
    this.scene = scene;
    this.canvasFormat = canvasFormat;

    this.periodic = true;
    this.qualityTier = K.DEFAULT_QUALITY_TIER;
    this._flightRenderScale = K.QUALITY_PRESETS[this.qualityTier].renderScale;
    this._holdRenderScale = K.HOLD_RENDER_SCALE_BY_TIER[this.qualityTier];
    this._cameraMoving = false;
    this._holdLadder = null;
    this._holdRung = 0;
    this._holdCapped = false;
    this._holdMode = K.DEFAULT_HOLD_MODE;
    this.renderScale = null;
    this.stepFactor = null;
    this.lightStepFactor = null;
    this.maxLightSteps = null;
    this._resetAccumulation();          // _applyRung resets it again; be defined first
    this._applyEffectiveQuality();

    this.motionBlendAlpha = K.DEFAULT_MOTION_BLEND_ALPHA;
    this.motionResetTranslationM = K.DEFAULT_MOTION_RESET_TRANSLATION_FRACTION *
      Math.min(scene.bmax[0] - scene.bmin[0], scene.bmax[1] - scene.bmin[1]);

    this._modules = new Map();
    this._pipelines = new Map();
    this._pendingPipelines = new Map();
    this._failedPipelines = new Map();
    this._prefetchArmed = false;          // see prefetchPipelines
    this._targets = null;
    // Scratch for the per-frame accumulation weights; see drawFrame.
    this._accumWeights = new Float32Array(4);
    this._resetAccumulation();
    // Offline renders cap the GPU queue at two marches; see _submit.
    this.boundedQueue = false;
    this._queueFences = [];

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
    // The periodic read of the same volume. Texture axes: u indexes field z
    // (never periodic), v field y, w field x. Repeat on the two lateral axes
    // is what makes filtering exact across the wrap seam now that no ghost
    // ring is uploaded — see raymarch.wgsl sample_level.
    this.volWrapSampler = device.createSampler({
      addressModeU: "clamp-to-edge", addressModeV: "repeat",
      addressModeW: "repeat", magFilter: "linear", minFilter: "linear",
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
        { binding: 6, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "3d" } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d-array" } },
      ],
    });
    // Group 1: the surface pre-pass's targets — the final surface, which
    // only the march reads; the stage before's, which each later stage
    // reads; the veil, read by the march; and the desert's optical depths,
    // read by its flora stage. Every raymarch pipeline shares this one
    // layout, so every pass sets group 1: stand-ins everywhere but those,
    // which bind that target size's own.
    this.surfaceLayout = device.createBindGroupLayout({
      label: "soar-surface-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
        // The veil (raymarch.wgsl veil_buf), read by the march.
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "uint", viewDimension: "2d" } },
        // The desert's optical depths (surface_taus), read by its flora
        // stage.
        { binding: 3, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      ],
    });
    this.rayPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.rayLayout, this.surfaceLayout],
    });
    this._surfaceDummy = device.createTexture({
      label: "soar-surface-dummy", size: [1, 1], format: SURFACE_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this._veilDummy = device.createTexture({
      label: "soar-veil-dummy", size: [1, 1], format: VEIL_FORMAT,
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });
    this.surfaceDummyGroup = device.createBindGroup({
      label: "soar-surface-dummy",
      layout: this.surfaceLayout,
      entries: [
        { binding: 0, resource: this._surfaceDummy.createView() },
        { binding: 1, resource: this._surfaceDummy.createView() },
        { binding: 2, resource: this._veilDummy.createView() },
        { binding: 3, resource: this._surfaceDummy.createView() },
      ],
    });
    // The sun-tau cache (binding 7) starts as a 1x1x1 zero stand-in, like the
    // nest's. Whether it is read is a per-tier preset choice (see
    // QUALITY_PRESETS.lightCache), with lightCacheMode/skyProbeMode as
    // whole-session overrides for A/B ("auto" follows the rung; "on"/"off"
    // force). The flag actually packed into row 23 is wanted AND baked — an
    // unfinished bake reads the live march, never a half-written cache.
    this._lightTauDummy = device.createTexture({
      label: "soar-light-tau-dummy", size: [1, 1, 1], dimension: "3d",
      format: "r16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this._lightTauDummy }, new Uint16Array([0]),
      { bytesPerRow: 2, rowsPerImage: 1 }, [1, 1, 1]);
    // Binding 8 (ice-detection mode): the scene's ice-fraction
    // volume when it has one, this zero stand-in otherwise. The mode flag is
    // packed into row 23.w only when a real texture is bound.
    // r8unorm, the one format the ice fraction is stored in anywhere — the
    // demo bake writes it (tools/prebake_demos.py), the NetCDF ingest
    // quantizes to it (ingest/worker.js), and the real texture is created as
    // it. The binding takes either, since the layout only asks for a float
    // sample type; matching it anyway is so that "what format is the ice
    // fraction" has one answer rather than one answer and an exception.
    this._iceDummy = device.createTexture({
      label: "soar-ice-frac-dummy", size: [1, 1, 1], dimension: "3d",
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this._iceDummy }, new Uint8Array([0]),
      { bytesPerRow: 1, rowsPerImage: 1 }, [1, 1, 1]);
    this.iceMode = false;
    // Bindings 3 and 9 are the two surface slots: the city's or the
    // desert's tile at 3, the moving sea at 9. A scene fills the one it
    // reads; the other gets this 1x1 zero stand-in, viewed as a 2D texture
    // or as a one-layer array. (Its own field: it once shared
    // _surfaceDummy's name, which orphaned group 1's stand-in above.)
    this._tileDummy = device.createTexture({
      label: "soar-tile-dummy", size: [1, 1, 1], format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this._tileDummy }, new Uint16Array(4),
      { bytesPerRow: 8, rowsPerImage: 1 }, [1, 1, 1]);
    this._lightTauTex = null;
    this._lightBake = null;           // in-flight bake state, or null
    this.lightCacheMode = "auto";
    this.skyProbeMode = "auto";
    // Advanced-panel override for the parked sample count; null follows
    // PARKED_ACCUM_FRAMES_BY_TIER.
    this.parkedSppOverride = null;
    this._lightCacheReady = false;
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
    this._shaderModules.push(this._module(this._shaderSpec()));

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

    // The march only ever targets the float intermediate now; everything
    // reaches the canvas through the (dithering) present pass. Awaited, so a
    // pipeline that fails to build rejects here with its own message — and
    // with it the scene's surface pre-pass, which every march needs.
    await this._rayPipeline(SAMPLE_FORMAT, this.lighting);
    for (const sp of this._stagePipelines(this.lighting)) await sp;
    this.device.pushErrorScope("validation");
    this._blitPipeline(this.canvasFormat, true);
    this._blitPipeline(this.canvasFormat, false);
    const error = await this.device.popErrorScope();
    if (error) {
      throw new Error(`Setting up the render pipelines failed: ${error.message}`);
    }
    this._prefetchArmed = true;
    this.prefetchPipelines();
  }

  /** Binding 3's stand-in over the ocean, whose sea is at binding 9. */
  _tileDummyView() {
    return this._tileDummy.createView();
  }

  /** Binding 9's stand-in under the city and the desert: one layer. */
  _seaDummyView() {
    return this._tileDummy.createView({ dimension: "2d-array" });
  }

  /**
   * Rebuild the bind group against the scene's current textures. Must be
   * called whenever a nest is attached or removed — binding 5 otherwise still
   * points at the 1x1x1 stand-in and the nest renders as nothing at all.
   */
  refreshBindGroup() {
    // A nest attached or removed moves the NESTED specialization.
    this.prefetchPipelines();
    const scene = this.scene;
    this.rayBindGroup = this.device.createBindGroup({
      label: "soar-raymarch",
      layout: this.rayLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: scene.volumeView },
        { binding: 2, resource: this.volSampler },
        { binding: 3, resource: scene.oceanView ?? this._tileDummyView() },
        { binding: 4, resource: this.oceanSampler },
        { binding: 5, resource: scene.nestView },
        { binding: 6, resource: this.volWrapSampler },
        { binding: 7, resource:
            (this._lightTauTex ?? this._lightTauDummy).createView() },
        { binding: 8, resource:
            scene.iceView ?? this._iceDummy.createView() },
        { binding: 9, resource: scene.seaView ?? this._seaDummyView() },
      ],
    });
  }

  /**
   * Build and validate the pipeline for a day surface — "ocean" or
   * "desert" — WITHOUT standing the scene on it. The first half of an
   * in-flight switch: everything that can fail or wait happens here, while
   * the scene keeps rendering what it stands on, so a shader that will not
   * compile says so and leaves the flight as it was.
   */
  async prepareSurface(kind) {
    if (this.scene.city) {
      throw new Error("the night city has no other surface to switch to.");
    }
    if (kind !== "ocean" && kind !== "desert") {
      throw new Error(`unknown day surface '${kind}'; expected 'ocean' or ` +
                      "'desert'.");
    }
    // Every entry the target surface draws with, at the current lighting
    // and tier, on the specialization the first frame after commitSurface
    // will ask for: the desert's tier axes (lite, the flora's cut) are the
    // TARGET surface's, not the scene's — the scene is not touched.
    const desert = kind === "desert";
    const spec = this._shaderSpec({ desert });
    try {
      await this._pipeline("fs_main", SAMPLE_FORMAT, spec);
      if (desert) {
        const entries = spec.desertLite ? K.SURFACE_STAGES_LITE
                                        : K.SURFACE_STAGES.desert;
        for (const entry of entries) {
          await this._pipeline(entry, stageFormats(entry), spec);
        }
      }
    } catch (err) {
      throw new Error(`The ${kind} shader failed to build: ` +
                      `${err?.message || err}`);
    }
  }

  /**
   * The second half: stand the scene on `kind`, whose pipeline
   * prepareSurface has already built. Synchronous, so the caller can move
   * the camera onto the new surface's frame and floor in the same tick and
   * no frame is ever drawn half-switched. `tile` is the loaded surface tile
   * (scene.js loadOceanTile).
   */
  commitSurface(kind, tile) {
    // Everything that can throw, before anything is changed: a tile whose
    // meta lacks the relief or its tables must leave the scene standing
    // where it was.
    const relief = kind === "desert" ? desertReliefM(tile.meta) : null;
    const tables = kind === "desert" ? desertTileRows(tile.meta) : null;
    const s = this.scene;
    s.desert = kind === "desert";
    // The desert's tile at binding 3, or the sea at binding 9 — see
    // loadOceanTile / loadSea. Whichever this surface does not use is
    // null and binds the stand-in.
    s.oceanView = tile.view;
    s.seaView = tile.seaView ?? null;
    s.oceanFifDx = tile.dx;
    s.oceanTileExtent = tile.tileExtent;
    s.surfacePeriod = tile.surfacePeriod;
    s.clockPeriod = tile.clockPeriod ?? null;
    s.oceanMaxLod = tile.maxLod;
    s.desertReliefM = relief;
    s.desertTileRows = tables;
    this.prefetchPipelines();
    this.refreshBindGroup();
    this._resetAccumulation();
  }

  /** Ice mode actually reaches the shader only with a real ice texture. */
  get iceModeActive() {
    return Boolean(this.iceMode && this.scene.iceView);
  }

  // --- quality ------------------------------------------------------------

  get flightRenderScale() { return this._flightRenderScale; }

  get holdRenderScale() { return this._holdRenderScale; }

  get qualityIsCustom() {
    return this._flightRenderScale
        !== K.QUALITY_PRESETS[this.qualityTier].renderScale
      || this._holdRenderScale
        !== K.HOLD_RENDER_SCALE_BY_TIER[this.qualityTier];
  }

  setQualityTier(tier, cameraMoving = null) {
    if (!(tier in K.QUALITY_PRESETS)) {
      throw new Error(`unknown quality tier '${tier}'.`);
    }
    if (cameraMoving !== null) this._cameraMoving = Boolean(cameraMoving);
    this.qualityTier = tier;
    // Both scales come from the tier. A hand-set scale does not survive a
    // tier change, and never has — the tier is the coarser choice.
    this._flightRenderScale = K.QUALITY_PRESETS[tier].renderScale;
    this._holdRenderScale = K.HOLD_RENDER_SCALE_BY_TIER[tier];
    this._applyEffectiveQuality();
    this.prefetchPipelines();
  }

  setRenderScale(scale) {
    renderTargetSize([1, 1], scale);   // validates, throws on a bad value
    this._flightRenderScale = Number(scale);
    this._applyEffectiveQuality();
  }

  /** What a held view climbs to. Below the flight scale it climbs nowhere. */
  setHoldRenderScale(scale) {
    renderTargetSize([1, 1], scale);
    this._holdRenderScale = Number(scale);
    this._applyEffectiveQuality();
  }

  /** Whether the camera moved this frame — the viewer's parked-vs-flying
   *  question (the parked city LOD reads it; see viewer._liveLodStrength). */
  get cameraMoving() { return this._cameraMoving; }

  setCameraMoving(moving) {
    moving = Boolean(moving);
    if (moving === this._cameraMoving) return;
    this._cameraMoving = moving;
    // Moving is always rung 0. Stopping does NOT jump up the ladder — it
    // starts at the flight rung and climbs from there, so the first held
    // frame costs exactly what the last flown one did.
    this._holdRung = 0;
    this._holdCapped = false;
    this._applyRung();
  }

  // --- the hold ladder ----------------------------------------------------

  /**
   * Build the rungs this tier can climb while the view is held.
   *
   * Rung 0 is the flight configuration — the tier's own sampling at whatever
   * render scale is in force, which is the preset's unless the Quality
   * panel's slider has overridden it. Above it come QUALITY_HOLD_RUNGS'
   * stepping stones and then the hold scale itself, minus any that would not
   * actually be an improvement on the rung below. That last rule is what
   * makes both hand-set scales behave: raise the flight scale to the hold
   * scale, or lower the hold scale to the flight scale, and the ladder
   * collapses to a single rung — "hold just accumulates", which is exactly
   * right in either case.
   */
  _buildHoldLadder() {
    const preset = K.QUALITY_PRESETS[this.qualityTier];
    const rungs = [{
      scale: this._flightRenderScale,
      stepFactor: preset.stepFactor,
      lightStepFactor: preset.lightStepFactor,
      maxLightSteps: preset.maxLightSteps,
      costRatio: 1.0,
    }];
    const steps = [
      // Stepping stones only count while they are on the way. Dragging the
      // hold slider down to (or below) the flight scale must collapse the
      // ladder, and it would not if an intermediate rung were still allowed
      // to climb past the destination.
      ...K.QUALITY_HOLD_RUNGS[this.qualityTier].filter(
        (step) => step.scale < this._holdRenderScale),
      // The top rung is the hold scale, sampled like High: whatever tier flew
      // you here, a settled picture is stepped and lit the same. A hold scale
      // at or below the flight scale simply drops it, which collapses the
      // ladder to "hold just accumulates" — the honest reading of asking for
      // a still no sharper than the flight.
      { scale: this._holdRenderScale, sampling: "high" },
    ];
    for (const step of steps) {
      const below = rungs[rungs.length - 1];
      if (!(step.scale > below.scale)) continue;
      const sampling = K.QUALITY_PRESETS[step.sampling];
      if (!sampling) {
        throw new Error(
          `hold ladder for '${this.qualityTier}' names an unknown sampling ` +
          `tier '${step.sampling}'.`);
      }
      rungs.push({
        scale: step.scale,
        stepFactor: sampling.stepFactor,
        lightStepFactor: sampling.lightStepFactor,
        maxLightSteps: sampling.maxLightSteps,
        // What this rung is expected to cost relative to the one below it.
        // Pixels go as the square of the scale. The view march takes a step
        // every stepFactor voxels, so halving the step factor doubles the
        // steps. The light march does the same, and is about a third of the
        // frame — hence the weighted third/two-thirds split rather than
        // treating the two step factors alike. Crude, and deliberately so: it
        // is a gate, not a model, and it only has to be right to within the
        // factor of safety in HOLD_MAX_FRAME_MS.
        costRatio: (step.scale / below.scale) ** 2
                 * ((2 / 3) * (below.stepFactor / sampling.stepFactor)
                    + (1 / 3) * (below.lightStepFactor
                                 / sampling.lightStepFactor)),
      });
    }
    this._holdLadder = rungs;
    this._holdRung = 0;
    this._holdCapped = false;
  }

  _applyEffectiveQuality() {
    this._buildHoldLadder();
    this._applyRung();
  }

  /** Put the current rung's numbers in force, restarting accumulation if they moved. */
  _applyRung() {
    const rung = this._holdLadder[this._holdRung];
    const changed = rung.scale !== this.renderScale
      || rung.stepFactor !== this.stepFactor
      || rung.lightStepFactor !== this.lightStepFactor
      || rung.maxLightSteps !== this.maxLightSteps;
    this.renderScale = rung.scale;
    this.stepFactor = rung.stepFactor;
    this.lightStepFactor = rung.lightStepFactor;
    this.maxLightSteps = rung.maxLightSteps;
    if (changed) this._resetAccumulation();
  }

  /**
   * One live frame's worth of hold progress. The live loop calls this and
   * nothing else does — the offline capture paths drive drawFrame directly
   * and must never climb, because their output has to be reproducible.
   *
   * Climbing throws away the rung below rather than upscaling it into the new
   * accumulator. That is a deliberate choice and it is safe: the first frame
   * after a reset is drawn unjittered (subpixel = false, see
   * _accumulationPlan), which is a clean, complete image — the same frame the
   * view already shows every time a turn ends. So the switch reads as
   * coarse-then-sharp, never as a flash or a black frame. Seeding the new
   * accumulator from the old one would need a second, non-dithering upscale
   * pipeline and a reordering of _targetsFor's destroy-then-create, for a
   * transition that is already not visible as a regression.
   *
   * `lastMarchMs` is how long the previous marched frame took. The climb is
   * gated on it: a rung is only taken when the rung below it, measured,
   * predicts the new one will land inside HOLD_MAX_FRAME_MS. Nothing is ever
   * rendered at a cost that has not been justified from below — the same
   * invariant the startup tier probe runs on, and for the same reason.
   */
  holdTick(lastMarchMs = null) {
    if (this._holdMode === "live") return;   // what you flew on is what you keep
    if (this._cameraMoving || this._holdCapped) return;
    if (this._holdRung >= this._holdLadder.length - 1) return;
    if (this._accumCount < K.HOLD_RUNG_FRAMES) return;
    const next = this._holdLadder[this._holdRung + 1];
    if (lastMarchMs === null) return;     // no measurement yet: not next frame's problem
    if (lastMarchMs * next.costRatio > K.HOLD_MAX_FRAME_MS) {
      // This machine cannot afford the rest of the ladder. Stop here and let
      // the view converge where it stands rather than never converging (and
      // so never sleeping) on exactly the hardware that most needs the sleep.
      // A one-off stall can trip this spuriously; the cost is a still that
      // settles a rung low, and it clears the moment the camera moves again.
      this._holdCapped = true;
      return;
    }
    this._holdRung += 1;
    this._applyRung();
  }

  /** Which rung is in force, and how many there are. For the stats readout. */
  get holdRung() { return this._holdRung; }
  get holdRungCount() { return this._holdLadder.length; }
  /** True when the ladder stopped short because the next rung was too dear. */
  get holdCapped() { return this._holdCapped; }

  /**
   * The view is settled: as high up the ladder as this machine will go, no
   * motion blend left in the buffer, and enough samples that another march
   * would change nothing anyone can see. This is the live loop's signal to
   * stop marching.
   */
  get converged() {
    const atTop = this._holdMode === "live"
      || this._holdRung === this._holdLadder.length - 1
      || this._holdCapped;
    return !this._cameraMoving && atTop && !this._accumMotion
      && this._accumCount >= this.parkedAccumFrames;
  }

  /** Jittered marches per presented frame — 1 everywhere but Max. */
  get sppPerFrame() {
    return K.QUALITY_PRESETS[this.qualityTier].sppPerFrame ?? 1;
  }

  /** Samples a held view settles at, which is a property of the tier. */
  get parkedAccumFrames() {
    // In live hold mode the point is to LOOK like flight, and flight is
    // sppPerFrame samples, not a converged average — accumulating further
    // was exactly the "show Live still goes to parked" bug (Thomas,
    // 2026-08-20): the resolution dropped but the picture then settled into
    // a clean still anyway. Converge at the flight sample count instead; the
    // loop still sleeps, holding a frame that looks like the one you fly on.
    if (this._holdMode === "live") return this.sppPerFrame;
    return this.parkedSppOverride
      ?? K.PARKED_ACCUM_FRAMES_BY_TIER[this.qualityTier];
  }

  get holdMode() { return this._holdMode; }

  /**
   * Live or still. Switching to live drops straight back to the flight rung —
   * the point of the mode is that nothing above it is ever rendered, and that
   * has to include the rung already in force when the switch is made.
   */
  setHoldMode(mode) {
    if (!K.HOLD_MODES.includes(mode)) {
      throw new Error(`unknown hold mode '${mode}'.`);
    }
    if (mode === this._holdMode) return;
    this._holdMode = mode;
    if (mode === "live") {
      this._holdRung = 0;
      this._holdCapped = false;
      this._applyRung();
      // Explicit, not only via _applyRung's changed-check: on a tier whose
      // flight and hold configurations coincide (High, Max) the rung numbers
      // do not move, and without this the converged still would survive into
      // a mode whose whole point is the un-converged flight look.
      this._resetAccumulation();
    }
  }

  /** Samples blended into the picture on screen. The stats readout's "spp". */
  get accumCount() { return this._accumCount; }

  /** Whether there is a finished frame to re-present without marching. */
  get canPresentLast() {
    return Boolean(this._lastPresented && this._targets);
  }

  setPeriodic(periodic) {
    periodic = Boolean(periodic);
    if (periodic === this.periodic) return;
    this.periodic = periodic;
    // Nothing to re-upload: the wrap is a sampler and a shader branch, not
    // texels. Accumulation resets by itself — row 20.x is in the
    // scene-identity key — and the specialized module is picked per frame.
    this.prefetchPipelines();
  }

  // The view march's step and the light march's are separate numbers now; see
  // QUALITY_PRESETS for the measurements that separated them. They are equal
  // at High, and at every tier's top hold rung, which is High's sampling — so
  // a converged still is stepped identically whatever tier flew there.
  get dtView() { return this.scene.minVoxelM * this.stepFactor; }
  get dtLight() { return this.scene.minVoxelM * this.lightStepFactor; }
  get dtViewNest() {
    return (this.scene.minVoxelNestM ?? this.scene.minVoxelM) * this.stepFactor;
  }
  get dtLightNest() {
    return (this.scene.minVoxelNestM ?? this.scene.minVoxelM)
      * this.lightStepFactor;
  }

  // --- pipelines ----------------------------------------------------------

  /**
   * Everything the shader is specialized on for a march from this renderer
   * now — specializeShader's argument. `lighting` is the method (see
   * `lighting`); the surface is a property of the scene, which every caller
   * inherits here rather than threading it. The city is constant for a
   * renderer's whole life; ocean and desert can trade places in flight
   * (prepareSurface / commitSurface), which is why the surface is in every
   * key.
   */
  _shaderSpec({ toneMap = true, lighting = this.lighting,
                tier = this.qualityTier,
                desert = Boolean(this.scene.desert),
                desertLite = this._desertLiteFor(tier, desert),
                desertFloraCut = this._desertFloraCutFor(tier, desert) } = {}) {
    return {
      periodic: this.periodic,
      nested: Boolean(this.scene.nested),
      maxLightSteps: this.maxLightSteps,
      toneMap,
      city: Boolean(this.scene.city),
      desert,
      desertLite,
      desertFloraCut,
      lightCache: lighting.lightCache,
      skyProbe: lighting.skyProbe,
    };
  }

  _module(spec) {
    const key = shaderVariantKey(spec);
    let module = this._modules.get(key);
    if (!module) {
      module = this.device.createShaderModule({
        label: `raymarch(${key})`,
        code: specializeShader(this.shaderSource, spec),
      });
      this._modules.set(key, module);
    }
    return module;
  }

  /** The march pipeline for `lighting` at the current specialization. */
  _rayPipeline(targetFormat, lighting, tier = this.qualityTier) {
    return this._pipeline("fs_main", targetFormat,
                          this._shaderSpec({ lighting, tier }));
  }

  /**
   * Whether `tier` flies the simplified desert (DESERT_LITE_TIERS): a
   * specialization of the desert scene only, keyed on the tier like the
   * lighting method, so captures (which set the parked tier) and prefetch
   * (which walks the neighbouring tiers) pick the same variant flight does.
   */
  _desertLiteFor(tier, desert = Boolean(this.scene.desert)) {
    return desert && K.DESERT_LITE_TIERS.includes(tier);
  }

  /** The flora's explicit cut at `tier` (DESERT_FLORA_CUT_BY_TIER), keyed
   *  like _desertLiteFor; 1 off the desert, where it means nothing.
   *  `desert` defaults to the scene's surface; prepareSurface passes the
   *  surface it is preparing. */
  _desertFloraCutFor(tier, desert = Boolean(this.scene.desert)) {
    if (!desert) return 1.0;
    const cut = K.DESERT_FLORA_CUT_BY_TIER[tier];
    if (cut === undefined) {
      throw new Error(`no flora cut for tier ${tier}`);
    }
    return cut;
  }

  /** Whether this scene's opaque surface is traced by the pre-pass. */
  _hasSurfacePass() {
    return Boolean(this.scene.city || this.scene.desert);
  }

  /**
   * The surface stages this scene runs at `tier`, in order (SURFACE_STAGES;
   * the simplified desert's first alone), as raymarch.wgsl entries; none
   * over the sea.
   */
  _stageEntries(tier = this.qualityTier) {
    if (!this._hasSurfacePass()) return [];
    if (this._desertLiteFor(tier)) return K.SURFACE_STAGES_LITE;
    return K.SURFACE_STAGES[this.scene.city ? "city" : "desert"];
  }

  /**
   * Each stage's pipeline for `lighting` at `tier`, in order, on the same
   * specialization as the march (same module, so no extra compile) — the
   * pipeline itself, or a promise of it while it compiles.
   */
  _stagePipelines(lighting, tier = this.qualityTier) {
    const spec = this._shaderSpec({ lighting, tier });
    return this._stageEntries(tier).map(
      (entry) => this._pipeline(entry, stageFormats(entry), spec));
  }

  /** Whether the stages at `tier` include a refinement (fs_surface_refine):
   *  the city's facade shell, the full desert's near plants. */
  _hasRefineStage(tier = this.qualityTier) {
    return this._stageEntries(tier).includes("fs_surface_refine");
  }

  /** Whether the stages include the city's street (fs_surface_street). */
  _hasStreetStage(tier = this.qualityTier) {
    return this._stageEntries(tier).includes("fs_surface_street");
  }

  /**
   * A raymarch.wgsl render pipeline — entry point `entryPoint` into
   * `targetFormat`, for the specialization `spec` — as the pipeline itself
   * once it is built, or a promise of it while it is compiling (isPending
   * tells which). Every raymarch.wgsl pipeline that can be compiled while
   * flying goes through here, so they share one cache, one key and one
   * failure rule.
   *
   * Compiled with createRenderPipelineAsync, and asked for ahead of need
   * (prefetchPipelines), because the lighting method is a specialization: a
   * tier switch, a bake finishing, or a sky-probe toggle can each need a
   * shader this session has not compiled yet, and a synchronous compile of
   * fs_main is a visible stall — long ones on Metal. The live loop never
   * waits on one (see drawFrame's waitForPipeline).
   *
   * A pipeline that fails to build is kept, per key, and THROWN by every
   * later request for that key — synchronously, so the frame that needs it
   * stops loudly — and never replaced by another pipeline: none would render
   * the same picture. A failure in a prefetch the session never uses is
   * logged when it happens and is otherwise harmless.
   *
   * Every compile is also a chance to meet a driver's compiler bugs: bug 22
   * (docs/soar-bugs.md) is Firefox+NVIDIA segfaulting on compiling a
   * specialization, and the lighting method multiplies the specializations a
   * session compiles. Prefetch is kept to the tiers a session can actually
   * reach next for that reason, not only for time.
   */
  _pipeline(entryPoint, targetFormat, spec) {
    // One format, or an array of them for an entry point with several
    // outputs (the surface stages that write the veil).
    const formats = [].concat(targetFormat);
    const key = `${entryPoint}|${formats.join("+")}|${shaderVariantKey(spec)}`;
    const ready = this._pipelines.get(key);
    if (ready) return ready;
    const failed = this._failedPipelines.get(key);
    if (failed) throw failed;
    let pending = this._pendingPipelines.get(key);
    if (!pending) {
      const module = this._module(spec);
      pending = this.device.createRenderPipelineAsync({
        label: `raymarch(${key})`,
        layout: this.rayPipelineLayout,
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint,
                    targets: formats.map((format) => ({ format })) },
        primitive: { topology: "triangle-list" },
      }).then((pipeline) => {
        this._pipelines.set(key, pipeline);
        this._pendingPipelines.delete(key);
        return pipeline;
      }, (err) => {
        this._pendingPipelines.delete(key);
        const failure = new Error(
          `The raymarch shader (${key}) failed to build: ` +
          `${err?.message || err}`);
        this._failedPipelines.set(key, failure);
        console.error(`soar: ${failure.message}`);
        throw failure;
      });
      this._pendingPipelines.set(key, pending);
    }
    return pending;
  }

  /**
   * Start compiling, in the background, every march pipeline this session
   * can reach from here without a reload: the current tier and its two
   * neighbours in QUALITY_TIER_NAMES (the probe, the governor and a click
   * all move one rung at a time), each with its cache ready and with its
   * bake still in flight. Called on init and whenever the specialization
   * moves under the renderer (tier, periodicity, surface). Compiles already
   * done or in flight are not repeated.
   */
  prefetchPipelines() {
    // Not before init(): the constructor and the viewer's setup move the
    // tier, periodicity and nest before the first frame, and prefetching for
    // each passing state would compile variants no frame will use — each
    // compile a chance at bug 22.
    if (!this._prefetchArmed) return;
    // The current tier first: compiles run roughly in request order, and
    // the variant a frame is waiting on must not queue behind a neighbour's.
    const order = K.QUALITY_TIER_NAMES;
    const at = order.indexOf(this.qualityTier);
    const reachable = [this.qualityTier, order[at - 1], order[at + 1]]
      .filter(Boolean);
    for (const tier of reachable) {
      for (const cacheReady of [true, false]) {
        const lighting = this._lightingFor(tier, cacheReady);
        try {
          // A promise while compiling: its rejection is recorded (and
          // logged) by _rayPipeline itself, so the prefetch just lets go.
          const p = this._rayPipeline(SAMPLE_FORMAT, lighting, tier);
          if (isPending(p)) p.catch(() => {});
          for (const sp of this._stagePipelines(lighting, tier)) {
            if (isPending(sp)) sp.catch(() => {});
          }
        } catch {
          // Already failed and recorded; thrown again if it is ever needed.
        }
      }
    }
    // The bake's one pipeline (see _bakePipeline): every tier but Max bakes
    // on the first sun, and a sun change re-bakes.
    try {
      const bake = this._bakePipeline();
      if (isPending(bake)) bake.catch(() => {});
    } catch {
      // Recorded; thrown by the bake that needs it.
    }
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

  /**
   * The render targets for one march size, POOLED per size rather than
   * destroyed and reallocated on every change.
   *
   * The hold ladder made target-size changes routine — every rung climb is
   * one — where they used to be a rare resize. The old destroy-and-drain
   * path cost a full `onSubmittedWorkDone()` wait per change, which on
   * Firefox is a ~100 ms poll rather than a fence (see the probe-clock
   * notes in constants.js), so a single hold's climb stalled for most of a
   * second. Worse than slow: Firefox's WebGPU runs in its MAIN process on
   * this stack, and its queue lifetime tracking is assert-happy — "Queue[Id]
   * is no longer alive" takes down the whole browser (docs/soar-bugs.md 14).
   * Every drain-and-destroy is a spin of that wheel. A pool spins it only on
   * a genuine output resize.
   *
   * Cost: the ladder's sizes sum to ~1.33x of the full-size targets alone
   * (1 + 1/4 + 1/16 + ...), so the pool's ceiling is about a third more
   * memory than the old worst case, held instead of churned.
   */
  async _targetsFor(w, h) {
    if (this._targets && this._targets.w === w && this._targets.h === h) {
      return this._targets;
    }
    this._targetPool ??= new Map();
    const key = `${w}x${h}`;
    let targets = this._targetPool.get(key);
    if (!targets) {
      // A genuine new size. If the pool has grown past the ladder's worst
      // case, something is resizing repeatedly (a window drag) — drain once
      // and drop the lot rather than accumulating textures forever. This is
      // the ONLY path left with a queue wait on it.
      if (this._targetPool.size >= 6) {
        await this.device.queue.onSubmittedWorkDone();
        for (const old of this._targetPool.values()) {
          old.sample.destroy(); old.accumA.destroy(); old.accumB.destroy();
          for (const tex of old.surface?.textures ?? []) tex.destroy();
        }
        this._targetPool.clear();
      }
      // 8 bytes for the half-precision sample target, 16 each for the two
      // full-precision accumulators.
      const bytes = w * h * (8 + 16 + 16);
      const make = (label, format) => this.device.createTexture({
        label: `${label}-${key}`, size: [w, h], format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
             | GPUTextureUsage.COPY_SRC,
      });
      targets = await guardAllocation(
        this.device, `${key} render targets`, bytes, () => ({
          w, h,
          sample: make("soar-sample", SAMPLE_FORMAT),
          accumA: make("soar-accum-a", ACCUM_FORMAT),
          accumB: make("soar-accum-b", ACCUM_FORMAT),
        }));
      // The two accumulation bind groups, built once per target set rather
      // than once per frame. Index i is the group to use when _accumIndex is
      // i: it reads the sample target and the accumulator NOT being written
      // this frame. Ordered to match the outTex choice in drawFrame.
      targets.accumBindGroups = [targets.accumA, targets.accumB].map(
        (prevTex) => this.device.createBindGroup({
          layout: this.accumLayout,
          entries: [
            { binding: 0, resource: { buffer: this.accumUniformBuf } },
            { binding: 1, resource: targets.sample.createView() },
            { binding: 2, resource: prevTex.createView() },
          ],
        }));
      this._targetPool.set(key, targets);
    }
    this._targets = targets;
    // The last presented frame lives in some OTHER size's accumulator now.
    // It still exists (pooled, not destroyed), but presentLast sizes its
    // blit off _targets, so re-presenting it would upscale with the wrong
    // dimensions. Null it: the first frame at the new size is unjittered and
    // clean, exactly as the old destroy path behaved.
    this._lastPresented = null;
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
   *
   * Resolves true when it marched. With `waitForPipeline: false` (the live
   * loop) it resolves false instead of waiting when the shader variant this
   * frame needs is still compiling — nothing marched, nothing presented, no
   * state touched.
   */
  async drawFrame(target, targetFormat, outputSize, view,
                  { deltaSeconds = null, accumulate = true,
                    overlays = null, waitForPipeline = true } = {}) {
    const [outputW, outputH] = outputSize;
    this._outputW = outputW;
    this._outputH = outputH;
    // The march pipeline, the surface pipelines (pre-pass, refine, street) and
    // the render targets for the state as it stands after the LAST await:
    // from the end of this loop to the first submit nothing awaits, so the
    // specialization, the lighting method, the targets and the uniforms
    // below all describe one moment. The method is read once — it picks the
    // pipelines and it is what row 23 packs — or a bake finishing between
    // two reads would march one method under the other's scene key. The
    // pair resolves together, so a frame never encodes one without the
    // other. A variant still compiling is waited out only when the caller
    // asks (offline renders, which own the renderer); the live loop gets
    // `false` back, marches nothing, keeps its last picture and asks again
    // next frame, so no frame of it is ever suspended on a compile while the
    // field or a capture changes under it. Fetching the targets can itself
    // await (a resize drains the pool; the surface target is made on first
    // need), so the loop only exits once the variant, the method and the
    // render size it chose are still the renderer's after those awaits.
    let lighting;
    let rayPipeline;
    let surfacePipelines;
    let renderSize;
    let targets;
    for (;;) {
      lighting = this.lighting;
      const p = this._rayPipeline(SAMPLE_FORMAT, lighting);
      const entries = this._stageEntries();
      const sps = this._stagePipelines(lighting);
      const pending = [p, ...sps].filter((q) => q && isPending(q));
      if (pending.length > 0) {
        if (!waitForPipeline) {
          // Recorded; thrown when next requested.
          for (const q of pending) q.catch(() => {});
          return false;
        }
        await Promise.all(pending);
        continue;
      }
      const variant = shaderVariantKey(this._shaderSpec({ lighting }));
      renderSize = renderTargetSize(outputSize, this.renderScale);
      targets = await this._targetsFor(renderSize[0], renderSize[1]);
      // A scene with a traced surface needs the set's surface target(s),
      // made on first need (e.g. after setSurface to the desert).
      await this._ensureSurfaceTarget(targets);
      const lightingNow = this.lighting;
      const sizeNow = renderTargetSize(outputSize, this.renderScale);
      if (lightingNow.lightCache === lighting.lightCache
          && lightingNow.skyProbe === lighting.skyProbe
          && shaderVariantKey(this._shaderSpec({ lighting })) === variant
          && sizeNow[0] === renderSize[0] && sizeNow[1] === renderSize[1]) {
        rayPipeline = p;
        surfacePipelines = entries.map((entry, k) => [entry, sps[k]]);
        break;
      }
    }
    const targetView = typeof target === "function" ? target() : target;

    // Pack ONCE, learn the scene key from it, then write the sampling flags
    // the plan chose straight into row 10's x and y. Those two components are
    // excluded from the key precisely so this is safe — the flags are outputs
    // of the decision that produced them and cannot change it. The rest of
    // row 10 (z = haze) is scene state and is left alone.
    //
    // This used to re-pack the whole block instead, which meant running the
    // spectral sky solve, the light-transfer split and two dozen rows of
    // trigonometry a second time every frame to alter two floats. Nothing
    // else in the block depends on subpixel or jitterScale, and nothing here
    // touches row 10's z or w.
    const state = this._sceneState();
    // The effective flags, not the wishes: a wanted-but-unbaked cache reads
    // the live march. Row 23 is in the scene key, so a bake completing, a
    // rung switching methods or an override is a scene change: never
    // averaged with the old method, and with a parked camera a clean
    // restart (see methodKey below).
    const u = packUniforms(state, {
      ...view, outputSize, renderSize,
      ...lighting,
      iceMode: this.iceModeActive,
      desertLite: this._desertLiteFor(this.qualityTier),
    });
    // A change of lighting method under a parked camera restarts the
    // average instead of blending into it. Any scene-key change with a
    // still camera takes the one-frame motion blend (_accumulationPlan),
    // which suits a value that glides — exposure, haze — but a method switch
    // is a step: a bake finishing, a tier changing methods or a K/J toggle
    // replaces one lighting model with another, and the blend would hold a
    // ~42% ghost of the old one for a frame before the average restarted.
    // In flight the key changes every frame anyway, and the motion blend is
    // the point, so a moving camera is left to it.
    const methodKey = `${lighting.lightCache}|${lighting.skyProbe}`;
    if (this._accumMethod !== undefined && methodKey !== this._accumMethod
        && !this._cameraMoving) {
      this._resetAccumulation();
    }
    this._accumMethod = methodKey;
    const plan = accumulate && view.jitter !== false
      ? this._accumulationPlan(u, deltaSeconds)
      : null;
    if (plan) {
      u[10 * 4] = plan.subpixel ? 1.0 : 0.0;
      u[10 * 4 + 1] = plan.jitterScale;
    } else {
      this._resetAccumulation();
    }
    this.device.queue.writeBuffer(this.uniformBuf, 0, u);

    const encoder = this.device.createCommandEncoder();

    if (!plan) {
      // No accumulation: march at a float intermediate and present from it.
      // The march could target the canvas directly when the sizes match, and
      // used to — but then the hardware would quantize the march's own output
      // and this frame would be the one path in the renderer that skips the
      // present pass's dither. One full-screen blit is cheaper than a second
      // encode point.
      const exact = renderSize[0] === outputW && renderSize[1] === outputH;
      this._encodeMarch(encoder, targets, this.rayBindGroup, rayPipeline,
                        surfacePipelines);
      this._encodeBlit(encoder, targets.sample, targetView, targetFormat,
                       exact);
      encodeOverlays(overlays, encoder, targetView, targetFormat);
      await this._submit(encoder.finish());
      return true;
    }

    // How many jittered marches go into this one presented frame. Every tier
    // but Max takes a single one; see QUALITY_PRESETS.
    //
    // They are separate submissions rather than one long pass, and that is the
    // safety property, not an implementation detail: each command buffer is
    // exactly one ordinary frame's worth of work, so eight samples cost eight
    // frames of GPU time and never one pass eight times longer. Metal's
    // watchdog kills a pass by duration, so the loop is what lets a tier this
    // dear exist at all without risking the device. (It also means the
    // uniform buffer can be rewritten between samples, which is what
    // decorrelates their jitter.)
    const samples = accumulate ? Math.max(1, this.sppPerFrame) : 1;
    let outTex = null;
    for (let s = 0; s < samples; s++) {
      const step = s === 0 ? plan : this._extraSamplePlan(plan.jitterScale);
      if (s > 0) {
        // Only the three components that differ between samples of one view,
        // for the reason the full repack is avoided above. None of them is in
        // the scene key, so the key computed once still describes all of them.
        u[4 * 4 + 3] = (view.frameIndex ?? 0) + s * SPP_FRAME_INDEX_STRIDE;
        u[10 * 4] = step.subpixel ? 1.0 : 0.0;
        u[10 * 4 + 1] = step.jitterScale;
        this.device.queue.writeBuffer(this.uniformBuf, 0, u);
      }
      // One scratch array for the blend weights, refilled rather than rebuilt.
      // writeBuffer copies out of it synchronously, so reusing it is safe.
      this._accumWeights[0] = step.prevWeight;
      this._accumWeights[1] = step.sampleWeight;
      this.device.queue.writeBuffer(this.accumUniformBuf, 0, this._accumWeights);

      const enc = s === 0 ? encoder : this.device.createCommandEncoder();
      const sPass = (viewRef) => enc.beginRenderPass({
        colorAttachments: [{
          view: viewRef, loadOp: "clear", storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });

      // The surface pass runs per SAMPLE, after this sample's uniform
      // rewrite: its ray is this sample's ray.
      this._encodeMarch(enc, targets, this.rayBindGroup, rayPipeline,
                        surfacePipelines);

      // The accumulator ping-pongs between two textures, so there are exactly
      // two bind groups for the whole life of a target set. They were being
      // built from scratch every frame — along with three texture views — for
      // a pair of resources that only change when the render target is
      // reallocated, which is where they are now built instead.
      outTex = this._accumIndex === 0 ? targets.accumB : targets.accumA;
      const p = sPass(outTex.createView());
      p.setPipeline(this.accumPipeline);
      p.setBindGroup(0, targets.accumBindGroups[this._accumIndex]);
      p.draw(3);
      p.end();

      // Only the last sample reaches the screen: the ones before it are the
      // picture being built, and presenting each would be a strobe.
      if (s === samples - 1) {
        this._encodeBlit(enc, outTex, targetView, targetFormat,
                         renderSize[0] === outputW && renderSize[1] === outputH);
        encodeOverlays(overlays, enc, targetView, targetFormat);
      }
      await this._submit(enc.finish());

      this._accumIndex = 1 - this._accumIndex;
      this._accumCount = step.nextCount;
    }
    this._lastPresented = outTex;
    return true;
  }

  /**
   * Submit one march — and, for an offline render, never let more than two
   * sit in the GPU's queue.
   *
   * The compositor shares the GPU, and it is not patient. Chrome on Linux
   * with Vulkan compositing gives vkAcquireNextImageKHR two seconds
   * (ui/ozone/platform/x11/vulkan_surface_x11.cc, a workaround for an X
   * server vblank bug); a present that sits behind more than that much
   * queued work times out, Chrome calls the swapchain hung, and it
   * restarts the GPU process — which loses every WebGPU device with "A
   * valid external Instance reference no longer exists". The video loop
   * used to queue a whole video frame's marches before the read-back
   * synced: 32 of them at Max, 1.5–2.5 s at 1080p on a 5080, so a heavier
   * frame crossed the line and the render died an hour in (Thomas,
   * 2026-09-01: seven minutes in, at the same signature in his journal).
   * Metal's command-buffer watchdog is the same shape of limit.
   *
   * Two in flight rather than one: the GPU always has the next march ready
   * when the current one ends, so this costs no bubble — the CPU just
   * waits for the march before last instead of racing ahead. The queue is
   * then at most two marches deep, and a single march is the unit nothing
   * here can split; one longer than a second would need tiling, which
   * this does not do. onSubmittedWorkDone promises resolve in submission
   * order, so the oldest fence is always the one to wait on.
   *
   * Offline only. The live loop draws into a swapchain texture that is
   * valid until the end of its animation-frame callback, and an await in
   * the middle of that frame is exactly the crash drawFrame's doc
   * describes; the live loop bounds its depth by drawing once per frame.
   */
  async _submit(commandBuffer) {
    if (!this.boundedQueue) {
      this.device.queue.submit([commandBuffer]);
      return;
    }
    while (this._queueFences.length >= 2) await this._queueFences.shift();
    this.device.queue.submit([commandBuffer]);
    this._queueFences.push(this.device.queue.onSubmittedWorkDone());
  }

  /**
   * A further jittered sample of the SAME view inside one presented frame.
   *
   * Not _accumulationPlan: that one asks whether the scene changed, and
   * within a frame it has not — asked twice it would read the unchanged key
   * as "motion has just ended" and throw the samples already taken away.
   * This is the plain running average, continuing whatever the frame's first
   * sample decided (a motion blend, or a fresh start).
   */
  _extraSamplePlan(jitterScale) {
    const nextCount = this._accumCount + 1;
    return {
      prevWeight: this._accumCount / nextCount,
      sampleWeight: 1.0 / nextCount,
      nextCount, subpixel: true, jitterScale,
    };
  }

  /**
   * One march into `targets.sample`, preceded — in a scene with a traced
   * surface — by its surface stages (K.SURFACE_STAGES: the first,
   * fs_surface or the full desert's fs_surface_desert, then the later
   * ones, each reading the one before) into `targets.surface`, which the
   * march then reads at group 1. `surfaces` is the stages as [entry,
   * pipeline] in order (_stageEntries), empty over the sea. They use the
   * same uniforms, so the stages trace exactly the rays the march
   * composites. Every march in the renderer goes through here or through
   * the meter, which mirrors it on its own targets.
   */
  _encodeMarch(encoder, targets, rayBindGroup, rayPipeline, surfaces) {
    let marchGroup = this.surfaceDummyGroup;
    if (surfaces.length > 0) {
      const surface = targets.surface;
      // The stages this frame runs, chained by surfaceStagePlan: each later
      // stage reads the one before at group 1 through the two
      // intermediates, the last writes the surface the march reads, a
      // VEIL_STAGES entry writes the veil beside it, and K.TAUS_STAGE the
      // optical depths K.TAUS_READER reads. No stage binds what it writes,
      // and a stage without the target it needs throws rather than read a
      // stand-in.
      const slot = {
        final: [surface.view, null],
        a: [surface.plainView, surface.afterPlain],
        b: [surface.midView, surface.afterMid],
      };
      const plan = surfaceStagePlan(surfaces.length);
      const stages = surfaces.map(([entry, pipeline], k) => {
        const [writes, reads] = plan[k];
        const views = [slot[writes][0]];
        if (K.VEIL_STAGES.includes(entry)) views.push(surface.veilView);
        if ((entry === K.TAUS_STAGE || entry === K.TAUS_READER)
            && !surface.tausView) {
          throw new Error(`${entry} needs the desert's optical-depth ` +
                          "target, and this size's surface set has none " +
                          "(_ensureSurfaceTarget was not awaited for it).");
        }
        if (entry === K.TAUS_STAGE) views.push(surface.tausView);
        return [pipeline, reads === null ? null : slot[reads][1], views];
      });
      for (const [pipeline, group1, views] of stages) {
        const sp = encoder.beginRenderPass({
          colorAttachments: views.map((view, k) => ({
            view, loadOp: "clear", storeOp: "store",
            clearValue: k === 0 ? { r: 0, g: 0, b: 0, a: 1e30 }
                                : { r: 0, g: 0, b: 0, a: 0 },
          })),
        });
        sp.setPipeline(pipeline);
        sp.setBindGroup(0, rayBindGroup);
        sp.setBindGroup(1, group1 ?? this.surfaceDummyGroup);
        sp.draw(3);
        sp.end();
      }
      marchGroup = surface.group;
    }
    const p = encoder.beginRenderPass({
      colorAttachments: [{
        view: targets.sample.createView(), loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    p.setPipeline(rayPipeline);
    p.setBindGroup(0, rayBindGroup);
    p.setBindGroup(1, marchGroup);
    p.draw(3);
    p.end();
  }

  /**
   * Give a target set the surface targets the ACTIVE stage chain writes and
   * the march's group 1 over them: the final surface and the veil, and the
   * intermediates surfaceStagePlan alternates between (none for one stage,
   * one for two, both for more). A set is pooled by size, and the scene can
   * gain a traced surface or a longer chain after the set was made
   * (setSurface, a tier change). Grow-only: a set that already holds enough
   * keeps them, so a size used only at Low (the simplified desert's single
   * stage) never allocates an intermediate, and after a size's first
   * growth a tier change never reallocates it. Allocated under the same
   * out-of-memory guard as the rest of the set. Awaited by every caller
   * BEFORE it resolves pipelines and encodes, and it returns only once the
   * set satisfies the tier live at that moment — the tier can change during
   * the allocation's await — so _encodeMarch only ever reads it.
   */
  async _ensureSurfaceTarget(targets) {
    // A set covers a chain when it holds as many intermediates and, for the
    // full desert's (K.TAUS_STAGE), the optical depths.
    const covers = (set, nMid, taus) =>
      set.nMid >= nMid && (set.taus || !taus);
    for (;;) {
      if (!this._hasSurfacePass()) return;
      const entries = this._stageEntries();
      const nMid = Math.min(Math.max(entries.length - 1, 0), 2);
      const taus = entries.includes(K.TAUS_STAGE);
      const old = targets.surface;
      if (old && covers(old, nMid, taus)) return;
      // Grow-only: the new set keeps whatever the old one held.
      const made = await this._makeSurfaceSet(
        targets, Math.max(nMid, old?.nMid ?? 0), taus || Boolean(old?.taus));
      // Another caller may have grown the set during the await: keep the
      // one that covers the other, retire the other once the GPU is done
      // with any frame that still reads it. Then check the (possibly
      // changed) tier again.
      const cur = targets.surface;
      const [keep, drop] = cur && covers(cur, made.nMid, made.taus)
        ? [cur, made] : [made, cur];
      targets.surface = keep;
      if (drop) {
        for (const tex of drop.textures) {
          retireAfterSubmittedWork(this.device, tex);
        }
      }
    }
  }

  /** A surface set of `targets`' size with `nMid` intermediates, and the
   *  desert's optical depths if `taus`. */
  async _makeSurfaceSet(targets, nMid, taus) {
    const later = nMid >= 1;
    const both = nMid >= 2;
    const label = `soar-surface-${targets.w}x${targets.h}`;
    const count = 2 + (later ? 1 : 0) + (both ? 1 : 0);
    return guardAllocation(
      this.device, `${targets.w}x${targets.h} surface target`,
      targets.w * targets.h * (16 * count + (taus ? 8 : 0)), () => {
        const make = (name, format) => this.device.createTexture({
          label: `${label}-${name}`, size: [targets.w, targets.h], format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT
               | GPUTextureUsage.TEXTURE_BINDING,
        });
        const tex = make("final", SURFACE_FORMAT);
        const veil = make("veil", VEIL_FORMAT);
        const plain = later ? make("plain", SURFACE_FORMAT) : null;
        const mid = both ? make("mid", SURFACE_FORMAT) : null;
        const tausTex = taus ? make("taus", TAUS_FORMAT) : null;
        const dummy = this._surfaceDummy.createView();
        const veilDummy = this._veilDummy.createView();
        const group1 = (final, prev, veilView, tausView) =>
          this.device.createBindGroup({
            label, layout: this.surfaceLayout,
            entries: [{ binding: 0, resource: final },
                      { binding: 1, resource: prev },
                      { binding: 2, resource: veilView },
                      { binding: 3, resource: tausView }],
          });
        // A later stage's group: what it reads, and the optical depths
        // wherever they exist (the flora stage reads them; the others
        // ignore binding 3).
        const after = (prevTex) => group1(dummy, prevTex.createView(),
                                          veilDummy,
                                          tausTex?.createView() ?? dummy);
        return {
          nMid,
          taus,
          tex,
          textures: [tex, veil, plain, mid, tausTex].filter(Boolean),
          view: tex.createView(),
          veilView: veil.createView(),
          plainView: plain?.createView() ?? null,
          midView: mid?.createView() ?? null,
          tausView: tausTex?.createView() ?? null,
          // Group 1 for a stage reading the plain or the middle surface.
          afterPlain: plain ? after(plain) : null,
          afterMid: mid ? after(mid) : null,
          // The march reads the final surface and the veil.
          group: group1(tex.createView(), dummy, veil.createView(), dummy),
        };
      });
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

  /**
   * Re-present the last accumulated frame without marching again, drawing
   * `overlays` over it as drawFrame would.
   *
   * This is the cheap half of the live loop once the view has converged: the
   * volume is not traversed at all, and the frame costs one full-screen blit
   * plus whatever the overlays are. The bird is why the loop still runs at
   * all in that state — it flies whether or not the camera does.
   */
  presentLast(targetView, targetFormat, outputSize, overlays = null) {
    if (!this.canPresentLast) return false;
    const encoder = this.device.createCommandEncoder();
    this._encodeBlit(
      encoder, this._lastPresented, targetView, targetFormat,
      this._targets.w === outputSize[0] && this._targets.h === outputSize[1]);
    encodeOverlays(overlays, encoder, targetView, targetFormat);
    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  // --- the sun-tau cache -----------------------------------------------------
  //
  // A 3D texture of sun optical depth over the outer AABB, baked one field-x
  // plane per draw (see raymarch.wgsl fs_bake_light) and read by the cloud
  // march instead of a per-sample sun march while lightCacheEnabled. The bake
  // spreads across frames — the viewer calls stepLightBake with a slice
  // budget — and the packed row-23 flag stays off until the last slice lands,
  // so an unfinished cache is never read. A sun change invalidates and, if
  // the toggle is on, the viewer starts a fresh bake.

  get lightBakePending() { return this._lightBake !== null; }

  get lightCacheReady() { return this._lightCacheReady; }

  /**
   * The lighting method a march from this renderer runs now: the flags
   * packUniforms writes into row 23 AND the constants the march pipeline is
   * specialized on (raymarch.wgsl LIGHT_CACHE, SKY_PROBE), from this one
   * getter so the two can never disagree.
   *
   * What this frame actually does: hand override, else the SELECTED tier's
   * preset — never the hold ladder's sampling tier. Parking sharpens the
   * sampling toward High's step factors, but it must not switch the
   * lighting method underneath: the sky probe is never on below High, flying
   * or parked (Thomas, 2026-08-20), and Max's parked march stays live. A
   * wanted cache is read only once its bake has finished.
   */
  get lighting() {
    return this._lightingFor(this.qualityTier, this._lightCacheReady);
  }

  /** The method `tier` runs with the cache ready or not — see `lighting`. */
  _lightingFor(tier, cacheReady) {
    const preset = K.QUALITY_PRESETS[tier];
    const wanted = this.lightCacheMode === "auto"
      ? preset.lightCache
      : this.lightCacheMode === "on";
    return {
      lightCache: Boolean(wanted && cacheReady),
      skyProbe: this.skyProbeMode === "auto"
        ? Boolean(preset.skyProbe)
        : this.skyProbeMode === "on",
    };
  }

  invalidateLightCache() {
    // The abandoned bake's slice texture, not just the reference to it: a
    // sun drag invalidates once per event, and each orphaned slice held its
    // cz*cy r16float until the tab closed. Fenced — the bake's last copy
    // into the cache may still be in flight.
    if (this._lightBake) {
      retireAfterSubmittedWork(this.device, this._lightBake.sliceTex);
    }
    this._lightBake = null;
    this._lightCacheReady = false;
  }

  /**
   * Allocate the cache texture, with the allocation itself as the memory
   * probe: WebGPU exposes no free-VRAM query, so the only honest test for
   * "does the cache fit next to this volume" is to ask for it and listen on
   * an out-of-memory error scope. Throws a plain-language error when it
   * does not fit — the loader treats that like any other GPU-OOM and
   * rejects the field, rather than silently degrading (Thomas, 2026-08-20).
   * Same-size reallocation is skipped, so sun re-bakes never re-probe.
   */
  async allocateLightCache(divisor = K.LIGHT_CACHE_DIVISOR) {
    const shape = this.scene.shape;          // [nx, ny, nz], field axes
    if (!shape) {
      throw new Error("the scene carries no shape; cannot size the cache.");
    }
    const [nx, ny, nz] = shape;
    const cx = Math.max(1, Math.ceil(nx / divisor));
    const cy = Math.max(1, Math.ceil(ny / divisor));
    const cz = Math.max(1, Math.ceil(nz / divisor));
    const d = this._lightCacheDims;
    if (this._lightTauTex && d && d.cx === cx && d.cy === cy && d.cz === cz) {
      return;
    }
    this.device.pushErrorScope("out-of-memory");
    const tex = this.device.createTexture({
      label: "soar-light-tau", size: [cz, cy, cx], dimension: "3d",
      format: "r16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const oom = await this.device.popErrorScope();
    if (oom) {
      tex.destroy();
      const mb = (cx * cy * cz * 2 / 2 ** 20).toFixed(0);
      const err = new Error(
        `This GPU cannot hold the sun-shadow cache (${mb} MB at 1/` +
        `${divisor} resolution) alongside the volume.`);
      err.advice = "Try a field with a smaller grid size, or close other " +
        "GPU-heavy tabs and applications.";
      throw err;
    }
    this._lightCacheReady = false;
    // The outgoing cache may still be read by frames already submitted.
    retireAfterSubmittedWork(this.device, this._lightTauTex);
    this._lightTauTex = tex;
    this._lightCacheDims = { cx, cy, cz };
    this.refreshBindGroup();
  }

  startLightBake(view) {
    if (!this._lightTauTex || !this._lightCacheDims) {
      throw new Error("allocateLightCache() must succeed before a bake.");
    }
    const { cx, cy, cz } = this._lightCacheDims;
    this._lightCacheReady = false;
    this._bakeUniformBuf ??= this.device.createBuffer({
      label: "soar-bake-uniforms", size: K.UNIFORM_NBYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sliceTex = this.device.createTexture({
      label: "soar-light-tau-slice", size: [cz, cy], format: "r16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const u = packUniforms(this._sceneState(), {
      ...view, outputSize: [cz, cy], renderSize: [cz, cy],
      subpixel: false, jitterScale: 1.0,
    });
    // Bake at High's light step whatever tier is flying (row 6.w is the
    // step the bake's marches take). The bake is paid once, so it takes the
    // finest quadrature any tier would use — which also makes the cache
    // tier-independent: governor climbs, hold-ladder rungs and tier clicks
    // never stale it. Only the sun does.
    u[6 * 4 + 3] = this.scene.minVoxelM * K.QUALITY_PRESETS.high.lightStepFactor;
    this._lightBake = {
      next: 0, cx, cy, cz, sliceTex, u,
      bindGroup: this.device.createBindGroup({
        label: "soar-light-bake",
        layout: this.rayLayout,
        entries: [
          { binding: 0, resource: { buffer: this._bakeUniformBuf } },
          { binding: 1, resource: this.scene.volumeView },
          { binding: 2, resource: this.volSampler },
          { binding: 3, resource:
              this.scene.oceanView ?? this._tileDummyView() },
          { binding: 4, resource: this.oceanSampler },
          { binding: 5, resource: this.scene.nestView },
          { binding: 6, resource: this.volWrapSampler },
          { binding: 7, resource: this._lightTauTex.createView() },
          { binding: 8, resource:
              this.scene.iceView ?? this._iceDummy.createView() },
          { binding: 9, resource:
              this.scene.seaView ?? this._seaDummyView() },
        ],
      }),
    };
  }

  /**
   * The bake pipeline's compile while it is still in flight, else null — for
   * a caller that must finish a bake synchronously-in-spirit (a capture) and
   * so has to yield until the pipeline exists. Throws a failed compile.
   */
  bakePipelinePending() {
    const pipeline = this._bakePipeline();
    return isPending(pipeline) ? pipeline : null;
  }

  /** Bake up to `slices` planes; true when the cache became complete. */
  stepLightBake(slices = K.LIGHT_CACHE_BAKE_SLICES_PER_FRAME) {
    const b = this._lightBake;
    if (!b) return false;
    // Still compiling: no slices this frame. The frames keep marching live
    // (the cache is not ready) and the bake resumes when the pipeline is.
    const pipeline = this._bakePipeline();
    if (isPending(pipeline)) {
      pipeline.catch(() => {});    // recorded; thrown when next requested
      return false;
    }
    const sliceView = b.sliceTex.createView();
    const end = Math.min(b.next + Math.max(1, slices), b.cx);
    for (let ix = b.next; ix < end; ix++) {
      b.u[23 * 4 + 1] = ix;
      this.device.queue.writeBuffer(this._bakeUniformBuf, 0, b.u);
      const encoder = this.device.createCommandEncoder();
      const p = encoder.beginRenderPass({
        colorAttachments: [{
          view: sliceView, loadOp: "clear", storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      p.setPipeline(pipeline);
      p.setBindGroup(0, b.bindGroup);
      p.setBindGroup(1, this.surfaceDummyGroup);
      p.draw(3);
      p.end();
      encoder.copyTextureToTexture(
        { texture: b.sliceTex, origin: [0, 0, 0] },
        { texture: this._lightTauTex, origin: [0, 0, ix] },
        [b.cz, b.cy, 1]);
      this.device.queue.submit([encoder.finish()]);
    }
    b.next = end;
    if (b.next >= b.cx) {
      // Fenced, not destroyed outright: the copy of the last slice into the
      // cache was submitted a moment ago and may still reference it.
      retireAfterSubmittedWork(this.device, b.sliceTex);
      this._lightBake = null;
      this._lightCacheReady = true;
      return true;
    }
    return false;
  }

  /**
   * The sun-tau bake's pipeline, through the same async cache as the march.
   * fs_bake_light reads neither lighting constant, so it is built on ONE
   * fixed method — the shader's declared default — rather than on flight's:
   * keyed on flight's, a tier change or probe toggle mid-bake would compile
   * a second bake pipeline identical to the first.
   */
  _bakePipeline() {
    // fs_bake_light reads neither the lighting method nor the desert's
    // detail (lite, flora cut), so it is built on one fixed variant of each.
    return this._pipeline("fs_bake_light", "r16float",
                          this._shaderSpec({ lighting: BAKE_LIGHTING,
                                             desertLite: false,
                                             desertFloraCut: 1.0 }));
  }


  // --- the auto-exposure meter ---------------------------------------------

  /**
   * One tiny linear-radiance render of `view`, read back and reduced to the
   * luminance at AUTO_EXPOSURE_PERCENTILE. Returns null when a previous
   * meter's readback is still in flight (one is ever allowed), or while its
   * pipelines are still compiling: either way the AE loop just tries again
   * next interval.
   *
   * The march is the real shader with TONE_MAP compiled out, at
   * AUTO_EXPOSURE_METER_SIZE — the same footprint-per-cost trade renderScale
   * makes, so ~5k rays per meter against ~1M for a frame. It gets its own
   * uniform buffer and target: the main frame's buffer is rewritten inside
   * drawFrame's multi-sample loop, and sharing it would race the meter's
   * submit against those rewrites.
   */
  async meterLuminance(outputSize, view) {
    if (this._meterBusy) return null;
    this._meterBusy = true;
    try {
      const [mw, mh] = K.AUTO_EXPOSURE_METER_SIZE;
      if (!this._meterTex) {
        this._meterUniformBuf = this.device.createBuffer({
          label: "soar-meter-uniforms",
          size: K.UNIFORM_NBYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._meterTex = this.device.createTexture({
          label: "soar-meter", size: [mw, mh], format: ACCUM_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        // 16 bytes per rgba32float texel; the row is 256-byte aligned by
        // construction (96 * 16 = 1536).
        this._meterReadBuf = this.device.createBuffer({
          label: "soar-meter-read", size: mw * 16 * mh,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
      }
      // The meter's lighting is the live march with the probe — what its
      // packUniforms call below has always packed (no lighting flags given,
      // so the defaults). Named here so the module is specialized on the
      // same method the row carries.
      // Its targets first: this is the meter's last await, so the
      // specialization, pipelines and uniforms below all describe one moment.
      this._meterTargets ??= { w: mw, h: mh, sample: this._meterTex };
      await this._ensureSurfaceTarget(this._meterTargets);
      const lighting = { lightCache: false, skyProbe: true };
      const spec = this._shaderSpec({ toneMap: false, lighting });
      // Its march and its surface pre-pass (city/desert), on that
      // specialization, through the same async cache as the frame's: a
      // first meter in a city scene would otherwise compile the whole city
      // trace on the main thread, right after load, on the live AE path.
      // While either is still compiling this interval has no answer; the
      // next one asks again, and a pipeline that failed throws from here.
      const pipeline = this._pipeline("fs_main", ACCUM_FORMAT, spec);
      const surfaces = this._stageEntries().map(
        (entry) => [entry, this._pipeline(entry, stageFormats(entry), spec)]);
      const compiling = [pipeline, ...surfaces.map(([, sp]) => sp)]
        .filter((q) => q && isPending(q));
      if (compiling.length > 0) {
        for (const q of compiling) q.catch(() => {});
        return null;
      }
      // Real outputSize for the aspect, meter size for the ray grid — the
      // same split renderScale relies on. One deterministic sample.
      const u = packUniforms(this._sceneState(), {
        ...view, outputSize, renderSize: [mw, mh],
        subpixel: false, jitterScale: 1.0, ...lighting,
        desertLite: this._desertLiteFor(this.qualityTier),
      });
      this.device.queue.writeBuffer(this._meterUniformBuf, 0, u);
      const bindGroup = this.device.createBindGroup({
        label: "soar-meter",
        layout: this.rayLayout,
        entries: [
          { binding: 0, resource: { buffer: this._meterUniformBuf } },
          { binding: 1, resource: this.scene.volumeView },
          { binding: 2, resource: this.volSampler },
          { binding: 3, resource:
              this.scene.oceanView ?? this._tileDummyView() },
          { binding: 4, resource: this.oceanSampler },
          { binding: 5, resource: this.scene.nestView },
          { binding: 6, resource: this.volWrapSampler },
          { binding: 7, resource:
              (this._lightTauTex ?? this._lightTauDummy).createView() },
          { binding: 8, resource:
              this.scene.iceView ?? this._iceDummy.createView() },
          { binding: 9, resource:
              this.scene.seaView ?? this._seaDummyView() },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      this._encodeMarch(encoder, this._meterTargets, bindGroup, pipeline,
                        surfaces);
      encoder.copyTextureToBuffer(
        { texture: this._meterTex },
        { buffer: this._meterReadBuf, bytesPerRow: mw * 16 },
        [mw, mh]);
      this.device.queue.submit([encoder.finish()]);

      await this._meterReadBuf.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(this._meterReadBuf.getMappedRange());
      const lums = new Float32Array(mw * mh);
      for (let i = 0; i < mw * mh; i++) {
        lums[i] = 0.2126 * data[i * 4]
                + 0.7152 * data[i * 4 + 1]
                + 0.0722 * data[i * 4 + 2];
      }
      this._meterReadBuf.unmap();
      lums.sort();
      // The highlight statistic: mean of everything above the rank. See
      // AUTO_EXPOSURE_PERCENTILE for why the mean and not the percentile.
      const rank = Math.min(
        lums.length - 1,
        Math.floor(K.AUTO_EXPOSURE_PERCENTILE * lums.length));
      let sum = 0.0;
      for (let i = rank; i < lums.length; i++) sum += lums[i];
      return sum / (lums.length - rank);
    } finally {
      // Errors are the caller's: null means only "not now" (a reading in
      // flight, or its pipelines compiling — both end). A meter that failed
      // used to return null as well, and a capture waiting for a reading
      // then asked again forever.
      this._meterBusy = false;
    }
  }

  /** Release the render targets. Pipelines and modules go with the object. */
  destroy() {
    this._meterTex?.destroy();
    for (const tex of this._meterTargets?.surface?.textures ?? []) {
      tex.destroy();
    }
    this._meterReadBuf?.destroy();
    this._meterUniformBuf?.destroy();
    this._meterTex = this._meterReadBuf = this._meterUniformBuf = null;
    this._meterTargets = null;
    this._surfaceDummy?.destroy();
    for (const targets of this._targetPool?.values() ?? []) {
      targets.sample.destroy();
      targets.accumA.destroy();
      targets.accumB.destroy();
      for (const tex of targets.surface?.textures ?? []) tex.destroy();
    }
    this._targetPool?.clear();
    this._targets = null;
    this._lastPresented = null;
    this.uniformBuf?.destroy();
    this.accumUniformBuf?.destroy();
    this._iceDummy?.destroy();
    this._tileDummy?.destroy();
    this._veilDummy?.destroy();
    // The sun-tau cache is volume/8 bytes — hundreds of MB on a large field
    // — and none of these were released here before: every change of field
    // leaked the lot. Fenced, because a frame or a bake submitted just
    // before teardown may still reference them.
    retireAfterSubmittedWork(
      this.device, this._lightTauTex, this._lightTauDummy,
      this._bakeUniformBuf, this._lightBake?.sliceTex);
    this._lightTauTex = this._lightTauDummy = this._bakeUniformBuf = null;
    this._lightBake = null;
    this._lightCacheReady = false;
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
      surfacePeriodM: s.surfacePeriod,
      clockPeriodS: s.clockPeriod ?? null,
      oceanEnabled: true,
      oceanMaxLod: s.oceanMaxLod,
      nested: s.nested,
      nestBmin: s.nestBmin,
      nestBmax: s.nestBmax,
      dtViewNest: this.dtViewNest,
      dtLightNest: this.dtLightNest,
      city: Boolean(s.city),
      cityOffsetM: s.cityOffsetM ?? [0.0, 0.0],
      desert: Boolean(s.desert),
      desertReliefM: s.desert ? s.desertReliefM : null,
      desertTileRows: s.desert ? s.desertTileRows : null,
    };
  }
}
