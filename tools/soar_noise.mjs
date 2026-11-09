// How noisy one frame is: the per-pixel spread of single-sample frames at a
// pinned view, in the browser's own renderer, for one tree.
//
// A flying soar frame is one sample per pixel blended into the motion
// average, and a parked one starts there, so the grain a viewer sees in
// flight is the spread of one-sample frames. This draws N of them with no
// accumulation (frame index 0..N-1, the scene clock held), reads each back
// in display values, and reports the per-pixel standard deviation over the
// N: its mean and percentiles over the image and over the lower half (in a
// street view, the ground), in 8-bit levels — the grain of a flying frame.
// And how far the mean of the first EARLY (8) frames is from the N-frame
// mean (RMS over the lower half): how settled a view is a moment after the
// camera stops, which stratified draws improve where they cannot change a
// single frame's spread. It also writes the N-frame mean
// and the standard deviation (float32, rows top-down, mean.f32 / std.f32,
// luma), so two trees can be compared: the
// noise by the numbers, and an estimator's exactness by the means (two
// unbiased estimators' N-frame means differ by noise of order std/sqrt(N)).
//
//   node tools/soar_noise.mjs --web DIR --out DIR [--cdp PORT]
//        [--size WxH] [--frames N] [--tier high] [--pose-png FILE] [--turn DEG]
//
// The view is the cyberpunk city's start at eye level, looking down a wet
// street, or with --pose-png a soar screenshot's own city pose and clock
// (its render metadata); --turn adds DEG to its azimuth. Auto-exposure is off and the exposure pinned (the
// screenshot's, or 2.0). Run it under tools/gpulock;
// compare trees inside ONE lock, interleaving if timing matters (it does
// not here: the numbers are of the picture, not of the clock).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { openTab, serveTree } from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf(n); return i < 0 ? d : argv[i + 1]; };
const webDir = flag("--web", null);
const out = flag("--out", null);
const cdpPort = flag("--cdp", "9333");
const [W, H] = flag("--size", "768x432").split("x").map(Number);
const N = Number(flag("--frames", "64"));
const tier = flag("--tier", "high");
const posePng = flag("--pose-png", null);
const turn = Number(flag("--turn", "0"));
if (!webDir || !out || !(N >= 2) || !(W > 0 && H > 0)) {
  console.error("usage: node tools/soar_noise.mjs --web DIR --out DIR " +
                "[--cdp PORT] [--size WxH] [--frames N] [--tier T] " +
                "[--pose-png FILE]");
  process.exit(2);
}
mkdirSync(out, { recursive: true });

/** The render metadata a soar screenshot carries (its tEXt chunk). */
function pngMetadata(path) {
  const b = readFileSync(path);
  for (let i = 8; i < b.length;) {
    const n = b.readUInt32BE(i);
    if (b.toString("latin1", i + 4, i + 8) === "tEXt") {
      const text = b.toString("latin1", i + 8, i + 8 + n);
      const cut = text.indexOf("\0");
      if (text.slice(0, cut) === "cloudyview.render_metadata") {
        return JSON.parse(text.slice(cut + 1));
      }
    }
    i += 12 + n;
  }
  throw new Error(`${path} carries no cloudyview.render_metadata`);
}

let view = { city: null, x: 5013, y: 17456, z: 2, az: 90, el: -4, fov: null };
let clock = 100.0;
let exposure = 2.0;
if (posePng) {
  const md = pngMetadata(posePng);
  if (!md.city) throw new Error(`${posePng} is not a city screenshot`);
  view = { city: md.city.position_m.slice(0, 2), z: md.city.position_m[2],
           az: md.camera.azimuth, el: md.camera.elevation, fov: md.camera.fov };
  clock = md.render.scene_time;
  exposure = md.render.exposure;
}

const server = await serveTree(resolve(webDir));
let tab = null;
try {
  tab = await openTab(cdpPort);
  await tab.send("Emulation.setDeviceMetricsOverride",
                 { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await tab.goto(`${server.base}/soar/`);
  await tab.eval(`localStorage.clear();
    localStorage.setItem("soar.mode", "cyberpunk");
    localStorage.setItem("soar.tutorialDone", "1");
    localStorage.setItem("soar.lensWater", "off"); true`);
  await tab.goto(`${server.base}/soar/`);
  await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
  await tab.eval(`document.getElementById("fly-less").click(); true`);
  await tab.waitFor(`(() => { const L = document.getElementById("loading");
      if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
      return window.soar?.scene && L.hidden; })()`, 180000);
  await tab.eval(`(() => { const s = window.soar, c = s.camera, v = ${JSON.stringify(view)};
    s.setQualityTier(${JSON.stringify(tier)});
    // Pinned, not metered: the meter reads the noisy picture, so two trees
    // whose noise differs would meter to different exposures before it is
    // switched off, and every mean would differ by that.
    s.autoExposure = false; s.exposure = ${exposure};
    if (v.city) {
      c.jump(v.city[0] - c.surfacePosition[0], v.city[1] - c.surfacePosition[1],
                 v.z - c.position[2]);
    } else {
      c.jump(v.x - c.position[0], v.y - c.position[1], v.z - c.position[2]);
    }
    if (v.fov) c.fov = v.fov;
    c.azimuth = v.az + ${turn}; c.elevation = v.el; return true; })()`);
  // A tier change prefetches its neighbours too: dozens of city compiles.
  await tab.waitFor(`(window.soar.renderer._pendingPipelines?.size ?? 0) === 0
      && !window.soar.renderer.lightBakePending`, 900000);
  const res = await tab.eval(`(async () => {
    const s = window.soar, r = s.renderer, d = s.device;
    s._capturing = true; await new Promise((ok) => setTimeout(ok, 300));
    r.setCameraMoving(false);
    const tex = d.createTexture({ size: [${W}, ${H}], format: s.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const buf = d.createBuffer({ size: ${W * H * 4},
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const n = ${W * H};
    const sum = new Float64Array(n), sq = new Float64Array(n);
    const early = new Float64Array(n);
    const bgra = s.canvasFormat.startsWith("bgra");
    try {
      for (let i = 0; i < ${N}; i++) {
        await r.drawFrame(tex.createView(), s.canvasFormat, [${W}, ${H}],
          { ...s._viewKwargs(), frameIndex: i, sceneTime: ${clock} },
          { accumulate: false });
        const enc = d.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: ${W * 4} },
                                [${W}, ${H}]);
        d.queue.submit([enc.finish()]);
        await buf.mapAsync(GPUMapMode.READ);
        const px = new Uint8Array(buf.getMappedRange());
        for (let p = 0, q = 0; p < n; p++, q += 4) {
          const rr = bgra ? px[q + 2] : px[q], bb = bgra ? px[q] : px[q + 2];
          const y = 0.2126 * rr + 0.7152 * px[q + 1] + 0.0722 * bb;
          sum[p] += y; sq[p] += y * y;
          if (i < 8) early[p] += y;
        }
        buf.unmap();
      }
    } finally { tex.destroy(); buf.destroy(); s._capturing = false; }
    const mean = new Float32Array(n), std = new Float32Array(n);
    let e2 = 0, ne = 0;
    for (let p = 0; p < n; p++) {
      if (p >= ${Math.floor(H / 2) * W}) {
        const de = early[p] / 8 - sum[p] / ${N}; e2 += de * de; ne += 1;
      }
      const m = sum[p] / ${N};
      mean[p] = m;
      std[p] = Math.sqrt(Math.max(sq[p] / ${N} - m * m, 0) * ${N} / ${N - 1});
    }
    const b64 = (a) => { const u8 = new Uint8Array(a.buffer); let bin = "";
      for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode(...u8.subarray(i, i + 0x8000));
      return btoa(bin); };
    return { mean: b64(mean), std: b64(std), early: Math.sqrt(e2 / ne) };
  })()`);
  const mean = new Float32Array(Buffer.from(res.mean, "base64").buffer.slice(0));
  const std = new Float32Array(Buffer.from(res.std, "base64").buffer.slice(0));
  writeFileSync(join(out, "mean.f32"), Buffer.from(mean.buffer));
  writeFileSync(join(out, "std.f32"), Buffer.from(std.buffer));
  const stats = (from) => {
    const a = Array.from(std.subarray(from * W)).sort((x, y) => x - y);
    const q = (f) => a[Math.min(a.length - 1, Math.floor(f * a.length))];
    const avg = a.reduce((x, y) => x + y, 0) / a.length;
    return { mean: +avg.toFixed(3), p50: +q(0.5).toFixed(2), p90: +q(0.9).toFixed(2),
             p99: +q(0.99).toFixed(2) };
  };
  const summary = { tree: resolve(webDir), size: [W, H], frames: N, tier, clock, exposure,
                    pose: posePng ?? "start street", turn, all: stats(0),
                    lower_half: stats(Math.floor(H / 2)),
                    early8_rms_lower_half: +res.early.toFixed(3) };
  writeFileSync(join(out, "noise.json"), JSON.stringify(summary, null, 1));
  console.log(JSON.stringify(summary));
  await tab.close(); tab = null;
  server.close();
} catch (err) {
  console.error(`soar_noise: ${err.message}`);
  await tab?.close();
  server.close();
  process.exit(err.code ?? 1);
}
