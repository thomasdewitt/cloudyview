// soar web — WebGPU browser build of cloudyview's interactive renderer.
//
// The raymarch WGSL is copied verbatim from the desktop app (the shader is
// the shared artifact); this host only mirrors the Python engine's
// per-frame work: camera basis, uniform packing (rows 0-4, 7.xy, 10 — all
// other rows come from meta.json's template, dumped from a real
// InteractiveRenderer so look constants can never drift), temporal
// accumulation, and flight controls.
//
// Conventions (see cloudyview/camera.py): meteorological axes — +x east,
// +y north, +z up; azimuth clockwise from north; elevation from horizon.

"use strict";

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
`;

const DEG = Math.PI / 180.0;
const OCEAN_FLOOR_MARGIN_M = 50.0;
const SPEED_DEFAULT = 60.0;      // m/s, wheel-scalable
const MOVING_BLEND_PREV = 0.45;  // EMA while flying (motion smoothing)
const STILL_ACCUM_MAX = 256;     // running-average cap when parked

function forwardFrom(azDeg, elDeg) {
  const az = azDeg * DEG, el = elDeg * DEG;
  const ce = Math.cos(el);
  return [Math.sin(az) * ce, Math.cos(az) * ce, Math.sin(el)];
}

function cameraBasis(azDeg, elDeg) {
  // Port of Camera.basis(): world-up reference unless nearly vertical.
  const f = forwardFrom(azDeg, elDeg);
  let upRef = [0, 0, 1];
  if (Math.abs(f[2]) > 0.999) upRef = [0, 1, 0];
  const r = [
    f[1] * upRef[2] - f[2] * upRef[1],
    f[2] * upRef[0] - f[0] * upRef[2],
    f[0] * upRef[1] - f[1] * upRef[0],
  ];
  const rn = Math.hypot(...r);
  const right = r.map((v) => v / rn);
  const u = [
    right[1] * f[2] - right[2] * f[1],
    right[2] * f[0] - right[0] * f[2],
    right[0] * f[1] - right[1] * f[0],
  ];
  const un = Math.hypot(...u);
  return [f, right, u.map((v) => v / un)];
}

async function fetchBin(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  return new Uint16Array(await resp.arrayBuffer());  // fp16 payloads
}

const SELFTEST = new URLSearchParams(location.search).has("selftest");
function slog(stage, extra) {
  if (!SELFTEST) return;
  fetch("/selftest-log", {
    method: "POST",
    body: JSON.stringify({ stage, ...(extra || {}) }),
  }).catch(() => {});
}
window.addEventListener("error", (e) =>
  slog("window-error", { error: String(e.message) }));
window.addEventListener("unhandledrejection", (e) =>
  slog("unhandled-rejection", { error: String(e.reason) }));

async function main() {
  const canvas = document.getElementById("view");
  const hud = document.getElementById("hud");
  const overlay = document.getElementById("overlay");
  const fail = (msg) => {
    overlay.innerHTML = `<div class="msg">${msg}</div>`;
    throw new Error(msg);
  };

  if (!navigator.gpu) {
    fail("This browser has no WebGPU. Chrome/Edge 113+, Firefox 141+, " +
         "or Safari 26 — on a machine with a GPU.");
  }
  overlay.querySelector(".msg").textContent = "downloading cloud field…";

  const [meta, volumeData, wgsl] = await Promise.all([
    fetch("demo/meta.json").then((r) => r.json()),
    fetchBin("demo/volume.bin"),
    fetch("raymarch.wgsl").then((r) => r.text()),
  ]);
  const fifMips = [];
  for (let i = 0; i < meta.fif.mips; i++) {
    fifMips.push(await fetchBin(`demo/fif_mip${i}.bin`));
  }

  overlay.querySelector(".msg").textContent = "starting WebGPU…";
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) fail("WebGPU adapter unavailable (GPU blocklisted?).");
  const device = await adapter.requestDevice();
  const gpuErrors = [];
  device.addEventListener("uncapturederror", (e) => {
    gpuErrors.push(String(e.error?.message || e.error));
    console.error("WebGPU:", e.error);
    slog("gpu-error", { error: gpuErrors[gpuErrors.length - 1] });
  });
  slog("device-ok");
  const context = canvas.getContext("webgpu");
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: "opaque" });

  // --- Volume texture: (w=nz, h=ny, d=nx), fp16, tight rows are fine for
  // queue.writeTexture (the 256-byte rule only binds buffer->texture copies).
  const [px, py, pz] = meta.volume.padded_dims_xyz;
  const volTex = device.createTexture({
    size: [pz, py, px],
    dimension: "3d",
    format: "r16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: volTex }, volumeData,
    { bytesPerRow: pz * 2, rowsPerImage: py }, [pz, py, px],
  );

  const fifTex = device.createTexture({
    size: [meta.fif.n, meta.fif.n, 1],
    format: "rgba16float",
    mipLevelCount: meta.fif.mips,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  for (let i = 0; i < meta.fif.mips; i++) {
    const n = Math.max(1, meta.fif.n >> i);
    device.queue.writeTexture(
      { texture: fifTex, mipLevel: i }, fifMips[i],
      { bytesPerRow: n * 8, rowsPerImage: n }, [n, n, 1],
    );
  }

  const volSampler = device.createSampler({
    addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
    addressModeW: "clamp-to-edge",
    magFilter: "linear", minFilter: "linear",
  });
  const oceanSampler = device.createSampler({
    addressModeU: "repeat", addressModeV: "repeat",
    magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
  });

  // --- Uniforms: template from the Python renderer; we own a few rows.
  const uniform = new Float32Array(21 * 4);
  uniform.set(meta.uniform_template.flat());
  const uniformBuf = device.createBuffer({
    size: uniform.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const accumUniformBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const rayModule = device.createShaderModule({ code: wgsl });
  const accumModule = device.createShaderModule({ code: ACCUM_SHADER });
  const presentModule = device.createShaderModule({ code: PRESENT_SHADER });

  const rayLayout = device.createBindGroupLayout({
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
    ],
  });
  const rayPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [rayLayout] }),
    vertex: { module: rayModule, entryPoint: "vs_main" },
    fragment: {
      module: rayModule, entryPoint: "fs_main",
      targets: [{ format: "rgba16float" }],
    },
    primitive: { topology: "triangle-list" },
  });
  const rayBindGroup = device.createBindGroup({
    layout: rayLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: volTex.createView() },
      { binding: 2, resource: volSampler },
      { binding: 3, resource: fifTex.createView() },
      { binding: 4, resource: oceanSampler },
    ],
  });

  const accumPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: accumModule, entryPoint: "vs_main" },
    fragment: {
      module: accumModule, entryPoint: "fs_main",
      targets: [{ format: "rgba16float" }],
    },
    primitive: { topology: "triangle-list" },
  });
  const presentPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: presentModule, entryPoint: "vs_main" },
    fragment: {
      module: presentModule, entryPoint: "fs_main",
      targets: [{ format: canvasFormat }],
    },
    primitive: { topology: "triangle-list" },
  });

  // --- Render targets (recreated on resize / scale change).
  let targets = null;
  function makeTargets(w, h) {
    const mk = () => device.createTexture({
      size: [w, h], format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT
           | GPUTextureUsage.TEXTURE_BINDING,
    });
    targets = { w, h, sample: mk(), accumA: mk(), accumB: mk(), flip: false };
  }

  // --- Camera / flight state.
  const bmin = meta.volume.bmin, bmax = meta.volume.bmax;
  const spanX = bmax[0] - bmin[0], spanY = bmax[1] - bmin[1];
  const rel = meta.camera_default.position;
  const state = {
    pos: [
      bmin[0] + (rel[0] + 1) * 0.5 * spanX,
      bmin[1] + (rel[1] + 1) * 0.5 * spanY,
      (rel[2] + 1) * 0.5 * bmax[2],
    ],
    az: meta.camera_default.azimuth,
    el: meta.camera_default.elevation,
    fov: meta.camera_default.fov,
    speed: SPEED_DEFAULT,
    keys: new Set(),
    captured: false,
    lod: true,
    renderScale: 1.0,
    frame: 0,
    stillFrames: 0,
    lastCam: "",
  };
  // Distance-LOD template values (row 20 y/z) for the L toggle.
  const lodY = uniform[20 * 4 + 1], lodZ = uniform[20 * 4 + 2];

  canvas.addEventListener("click", () => {
    if (!state.captured) canvas.requestPointerLock();
  });
  const topbar = document.getElementById("topbar");
  document.addEventListener("pointerlockchange", () => {
    state.captured = document.pointerLockElement === canvas;
    hud.classList.toggle("hidden", state.captured);
    topbar.classList.toggle("hidden", state.captured);
  });
  const QUALITY_STEPS = [
    ["full", 1.0], ["balanced", 0.75], ["fast", 0.5],
  ];
  let qualityIdx = 0;
  const qualityBtn = document.getElementById("qualitybtn");
  qualityBtn.addEventListener("click", () => {
    qualityIdx = (qualityIdx + 1) % QUALITY_STEPS.length;
    const [label, scale] = QUALITY_STEPS[qualityIdx];
    state.renderScale = scale;
    qualityBtn.textContent = `quality: ${label}`;
  });
  document.getElementById("fsbtn").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  });
  document.addEventListener("mousemove", (e) => {
    if (!state.captured) return;
    state.az = (state.az + e.movementX * 0.12) % 360;
    state.el = Math.max(-89, Math.min(89, state.el - e.movementY * 0.12));
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "l" || e.key === "L") { state.lod = !state.lod; return; }
    state.keys.add(e.key.toLowerCase());
  });
  document.addEventListener("keyup", (e) => state.keys.delete(e.key.toLowerCase()));
  document.addEventListener("wheel", (e) => {
    state.speed = Math.max(2, Math.min(2000,
      state.speed * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  }, { passive: true });

  function move(dt) {
    const k = state.keys;
    if (!state.captured || k.size === 0) return;
    const [f, r] = cameraBasis(state.az, state.el);
    const d = state.speed * dt;
    const p = state.pos;
    if (k.has("w")) { p[0] += f[0] * d; p[1] += f[1] * d; p[2] += f[2] * d; }
    if (k.has("s")) { p[0] -= f[0] * d; p[1] -= f[1] * d; p[2] -= f[2] * d; }
    if (k.has("a")) { p[0] -= r[0] * d; p[1] -= r[1] * d; }
    if (k.has("d")) { p[0] += r[0] * d; p[1] += r[1] * d; }
    if (k.has(" ")) p[2] += d;
    if (k.has("shift") || k.has("c")) p[2] -= d;
    // Periodic domain: wrap x/y; keep a soft floor above the ocean.
    p[0] = bmin[0] + ((p[0] - bmin[0]) % spanX + spanX) % spanX;
    p[1] = bmin[1] + ((p[1] - bmin[1]) % spanY + spanY) % spanY;
    p[2] = Math.max(OCEAN_FLOOR_MARGIN_M, Math.min(p[2], 2.5 * bmax[2]));
  }

  function writeUniforms(w, h, outW, outH) {
    const [f, r, u] = cameraBasis(state.az, state.el);
    const U = uniform;
    U[0] = state.pos[0]; U[1] = state.pos[1]; U[2] = state.pos[2];
    U[3] = Math.tan(state.fov * DEG * 0.5);
    U[4] = f[0]; U[5] = f[1]; U[6] = f[2]; U[7] = outW / outH;
    U[8] = r[0]; U[9] = r[1]; U[10] = r[2];        // U[11] exposure: template
    U[12] = u[0]; U[13] = u[1]; U[14] = u[2]; U[15] = 1.0;   // jitter on
    U[16 + 3] = state.frame;                        // sun xyz from template
    U[7 * 4 + 0] = w; U[7 * 4 + 1] = h;
    U[10 * 4 + 0] = 1.0;                            // subpixel jitter
    U[10 * 4 + 1] = 0.65;
    U[20 * 4 + 1] = state.lod ? lodY : 0.0;
    U[20 * 4 + 2] = state.lod ? lodZ : 0.0;
    device.queue.writeBuffer(uniformBuf, 0, U);
  }

  const fpsEl = document.getElementById("fps");
  const selftest = SELFTEST;
  const selftestT0 = performance.now();
  slog("pipelines-ready");
  let lastT = performance.now();
  let fpsAcc = 0, fpsN = 0;

  function frame() {
    const now = performance.now();
    const dt = Math.min((now - lastT) / 1000, 0.1);
    lastT = now;
    move(dt);

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const outW = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const outH = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== outW || canvas.height !== outH) {
      canvas.width = outW; canvas.height = outH;
    }
    const w = Math.max(1, Math.floor(outW * state.renderScale));
    const h = Math.max(1, Math.floor(outH * state.renderScale));
    if (!targets || targets.w !== w || targets.h !== h) makeTargets(w, h);

    const camKey = state.pos.map((v) => v.toFixed(2)).join(",")
      + `|${state.az.toFixed(2)}|${state.el.toFixed(2)}|${w}x${h}|${state.lod}`;
    const still = camKey === state.lastCam;
    state.lastCam = camKey;
    state.stillFrames = still ? Math.min(state.stillFrames + 1, STILL_ACCUM_MAX) : 0;

    writeUniforms(w, h, outW, outH);
    const n = state.stillFrames;
    const prevW = still ? (n > 0 ? (still && n < STILL_ACCUM_MAX ? n / (n + 1) : 1 - 1 / STILL_ACCUM_MAX) : 0.0)
                        : MOVING_BLEND_PREV;
    device.queue.writeBuffer(accumUniformBuf, 0,
      new Float32Array([state.frame === 0 ? 0 : prevW, 1 - (state.frame === 0 ? 0 : prevW), 0, 0]));

    const enc = device.createCommandEncoder();
    const passDesc = (view) => ({
      colorAttachments: [{
        view, loadOp: "clear", storeOp: "store",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });

    // 1) raymarch -> sample
    let pass = enc.beginRenderPass(passDesc(targets.sample.createView()));
    pass.setPipeline(rayPipeline);
    pass.setBindGroup(0, rayBindGroup);
    pass.draw(3);
    pass.end();

    // 2) accumulate: prev = flip ? A : B, out = the other
    const prevTex = targets.flip ? targets.accumA : targets.accumB;
    const outTex = targets.flip ? targets.accumB : targets.accumA;
    targets.flip = !targets.flip;
    pass = enc.beginRenderPass(passDesc(outTex.createView()));
    pass.setPipeline(accumPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: accumPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: accumUniformBuf } },
        { binding: 1, resource: targets.sample.createView() },
        { binding: 2, resource: prevTex.createView() },
      ],
    }));
    pass.draw(3);
    pass.end();

    // 3) present (bilinear upscale to canvas). Skipped in selftest:
    // headless Chrome cannot composite the WebGPU canvas, and touching
    // getCurrentTexture there stalls the whole loop.
    if (!selftest) {
      pass = enc.beginRenderPass(
        passDesc(context.getCurrentTexture().createView()));
      pass.setPipeline(presentPipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: presentPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: outTex.createView() },
          { binding: 1, resource: oceanSampler },
        ],
      }));
      pass.draw(3);
      pass.end();
    }
    device.queue.submit([enc.finish()]);

    state.frame += 1;
    fpsAcc += dt; fpsN += 1;
    if (fpsAcc >= 0.5) {
      fpsEl.textContent = `${(fpsN / fpsAcc).toFixed(0)} fps`;
      fpsAcc = 0; fpsN = 0;
    }
    schedule(frame);
    // Self-test harness (?selftest): after 90 real frames, read the
    // OFFSCREEN accum texture back through a buffer copy and POST it —
    // deliberately bypassing canvas presentation, which headless Chrome
    // cannot composite (ProduceSkia mailbox errors) even when the render
    // pipeline itself is healthy.
    if (selftest && state.frame === 90) {
      selftestCapture(outTex, targets.w, targets.h).catch((e) =>
        fetch("/selftest-log", {
          method: "POST",
          body: JSON.stringify({ error: String(e), gpuErrors }),
        })
      );
    }
  }

  async function selftestCapture(srcTex, w, h) {
    const fps = 90 / ((performance.now() - selftestT0) / 1000);
    const shot = device.createTexture({
      size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const shotPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: presentModule, entryPoint: "vs_main" },
      fragment: { module: presentModule, entryPoint: "fs_main",
                  targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list" },
    });
    const rowBytes = Math.ceil((w * 4) / 256) * 256;
    const buf = device.createBuffer({
      size: rowBytes * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: shot.createView(), loadOp: "clear",
                           storeOp: "store" }],
    });
    pass.setPipeline(shotPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: shotPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: oceanSampler },
      ],
    }));
    pass.draw(3);
    pass.end();
    enc.copyTextureToBuffer(
      { texture: shot }, { buffer: buf, bytesPerRow: rowBytes,
                           rowsPerImage: h }, [w, h, 1],
    );
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const img = new ImageData(w, h);
    for (let y = 0; y < h; y++) {
      img.data.set(src.subarray(y * rowBytes, y * rowBytes + w * 4), y * w * 4);
    }
    for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;
    const c2d = document.createElement("canvas");
    c2d.width = w; c2d.height = h;
    c2d.getContext("2d").putImageData(img, 0, 0);
    const blob = await new Promise((res) => c2d.toBlob(res, "image/png"));
    await fetch("/selftest-shot", { method: "POST", body: blob });
    await fetch("/selftest-log", {
      method: "POST",
      body: JSON.stringify({
        fps: Number(fps.toFixed(1)), size: [w, h], gpuErrors,
      }),
    });
  }

  // rAF depends on canvas compositing, which headless lacks — selftest
  // drives the loop with timers instead.
  const schedule = selftest
    ? (f) => setTimeout(f, 0)
    : (f) => requestAnimationFrame(f);
  overlay.remove();
  schedule(frame);
}

main().catch((err) => {
  console.error(err);
  slog("main-failed", { error: String(err.message || err) });
  const overlay = document.getElementById("overlay");
  if (overlay && !overlay.querySelector(".msg")?.textContent.includes("WebGPU")) {
    overlay.innerHTML = `<div class="msg">failed: ${err.message}</div>`;
  }
});
