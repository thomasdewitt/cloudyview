// The sea's speckle as a viewer meets it: in flight, on the first frames
// after the camera stops, and converged, in the browser's own renderer and
// through its own accumulation plan (renderer._accumulationPlan: the motion
// blend while flying; on stopping, the motion buffer thrown away and the
// accumulation restarted from one unjittered sample — the frame the speckle
// flashes on).
//
//   node tools/sea_speckle_qa.mjs --web DIR --out DIR [--cdp PORT]
//        [--size WxH] [--tier high] [--fps 60] [--views a,b]
//
// Each view is the marine-congestus-coarse demo over the sea, at a pinned
// camera and sun (the Python ablation's views, temp/speckle in the ocean
// worktree): low toward a low sun, low across it, and a cruise at 600 m.
// Per view: a glide of GLIDE frames at 1 m a frame along the view azimuth
// (the camera moving, so the real motion blend), the last of them read back
// as `VIEW-flight`; then the camera held, and park frames 1..4 read back as
// `VIEW-park1..4`; then held until the renderer calls the view converged —
// the tier's parked sample count at the top of its hold ladder, climbed as
// the live loop climbs it (holdTick, fed each frame's measured time) —
// `VIEW-converged`, the picture a parked viewer settles to.
// --fps sets the glide's frame time, which the motion blend is scaled by
// (motionAlphaForDt): 60 is a viewer at 60 Hz, a slower machine keeps more
// history in flight.
// Each is raw RGBA8 (rows top-down), with meta.json saying the size and
// format; tools/sea_speckle_sheet.py turns them into a sheet.
//
// The tier is set through the viewer (setQualityTier: the startup probe and
// its governor off for good) and checked in force before anything is drawn;
// auto-exposure is off and the exposure pinned, lens water off, the live
// loop held while drawing. Everything here draws on the GPU: run it under
// tools/gpulock (board rules 10-11), one tree per run. Needs web/soar/demos
// with marine-congestus-coarse (a reflinked copy, not a link) and a QA
// Chrome with a hardware adapter on --cdp (tools/soar_cdp.mjs).

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { openTab, serveTree } from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const flag = (n, d) => { const i = argv.indexOf(n); return i < 0 ? d : argv[i + 1]; };
const webDir = flag("--web", null);
const out = flag("--out", null);
const cdpPort = flag("--cdp", "9333");
const [W, H] = flag("--size", "960x540").split("x").map(Number);
const tier = flag("--tier", "high");
const fps = Number(flag("--fps", "60"));
const only = flag("--views", null)?.split(",") ?? null;
if (!webDir || !out || !(W > 0 && H > 0) || !(fps > 0)) {
  console.error("usage: node tools/sea_speckle_qa.mjs --web DIR --out DIR " +
                "[--cdp PORT] [--size WxH] [--tier T] [--fps N] [--views a,b]");
  process.exit(2);
}
mkdirSync(out, { recursive: true });

const DEMO = "marine-congestus-coarse";
// The clearest 3 km of the demo (tools/contact_sheet.py OCEAN_CLEAR), so the
// sea is in sun, not under its deck.
const CLEAR = [17105.0, 1405.0];
const VIEWS = {
  toward_sun: { z: 30, az: 250, el: -6, fov: 70, sun: [250, 10] },
  cross_sun: { z: 30, az: 160, el: -4, fov: 70, sun: [250, 10] },
  cruise: { z: 600, az: 200, el: -5, fov: 70, sun: [250, 25] },
};
const GLIDE = 24;
const PARK = 4;
// A cap on the hold, not a target: the renderer's own converged flag ends it.
const HOLD_LIMIT = 512;
const EXPOSURE = 4.0;      // witness.render_nested's, so the Python ablation compares

const server = await serveTree(webDir);
let tab = null;
const done = async (code) => { await tab?.close(); server.close(); process.exit(code); };
try {
  tab = await openTab(cdpPort);
  await tab.send("Emulation.setDeviceMetricsOverride",
                 { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await tab.goto(`${server.base}/soar/`);
  await tab.eval(`localStorage.clear();
    localStorage.setItem("soar.mode", "research");
    localStorage.setItem("soar.tutorialDone", "1");
    localStorage.setItem("soar.tutorialResearchDone", "1");
    localStorage.setItem("soar.lensWater", "off"); true`);
  await tab.goto(`${server.base}/soar/?demo=${DEMO}`);
  await tab.waitFor(`(() => { const L = document.getElementById("loading");
      if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
      return window.soar?.scene && L.hidden; })()`, 180000);
  if (await tab.eval(`Boolean(window.soar.scene.city)`)) {
    throw new Error(`${DEMO} did not load over the sea`);
  }
  const inForce = await tab.eval(`(() => { const s = window.soar;
    s.setQualityTier(${JSON.stringify(tier)});
    s.setExposure(${EXPOSURE}); s.minimapMode = "off";
    return { viewer: s.qualityTier, renderer: s.renderer.qualityTier,
             auto: s.autoTier, alpha: s.renderer.motionBlendAlpha }; })()`);
  if (inForce.viewer !== tier || inForce.renderer !== tier || inForce.auto) {
    throw new Error(`tier not in force: ${JSON.stringify(inForce)}`);
  }
  console.log(`tree ${server.root}, tier ${tier} in force ` +
              `(motion blend alpha ${inForce.alpha})`);

  await tab.eval(`window.__seaQa = {
    async settle() { const r = window.soar.renderer;
      while ((r._pendingPipelines?.size ?? 0) > 0 || r.lightBakePending) {
        if (r.lightBakePending) r.stepLightBake(64);
        await window.soar.device.queue.onSubmittedWorkDone();
        await new Promise((ok) => setTimeout(ok, 50));
      } return true; },
    place(v) { const s = window.soar, c = s.camera;
      c.displace(${CLEAR[0]} - c.position[0], ${CLEAR[1]} - c.position[1],
                 v.z - c.position[2]);
      c.azimuth = v.az; c.elevation = v.el; c.fov = v.fov;
      s.setSun({ azimuth: v.sun[0], elevation: v.sun[1] });
      return { position: c.position, bmin: s.scene.bmin }; },
    async readback(tex) { const d = window.soar.device;
      // Rows padded to the 256-byte pitch a texture-to-buffer copy needs,
      // then stripped back to W * 4.
      const pitch = Math.ceil(${W * 4} / 256) * 256;
      const buf = d.createBuffer({ size: pitch * ${H},
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: pitch },
                              [${W}, ${H}]);
      d.queue.submit([enc.finish()]); await buf.mapAsync(GPUMapMode.READ);
      const padded = new Uint8Array(buf.getMappedRange());
      const bytes = new Uint8Array(${W * H * 4});
      for (let y = 0; y < ${H}; y++) {
        bytes.set(padded.subarray(y * pitch, y * pitch + ${W * 4}), y * ${W * 4});
      }
      buf.unmap(); buf.destroy();
      let bin = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      }
      return btoa(bin); },
    // The glide, the stop and the convergence, the live loop held off.
    async run(v) { const s = window.soar, r = s.renderer, d = s.device, c = s.camera;
      s._capturing = true; await new Promise((ok) => setTimeout(ok, 300));
      const tex = d.createTexture({ size: [${W}, ${H}], format: s.canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const a = v.az * Math.PI / 180;
      const step = [Math.sin(a), Math.cos(a)];
      const shots = {}, spp = {};
      let frame = 0;
      const draw = async () => {
        await r.drawFrame(tex.createView(), s.canvasFormat, [${W}, ${H}],
          { ...s._viewKwargs(), frameIndex: frame++, sceneTime: 0.0 },
          { deltaSeconds: 1 / ${fps} });
      };
      try {
        r.resetAccumulation();
        for (let i = 0; i < ${GLIDE}; i++) {
          c.displace(step[0], step[1], 0.0);
          r.setCameraMoving(true);
          await draw();
        }
        shots.flight = await this.readback(tex); spp.flight = r.accumCount;
        r.setCameraMoving(false);
        let k = 0;
        while (!r.converged) {
          if (++k > ${HOLD_LIMIT}) throw new Error("the held view never converged");
          const t0 = performance.now();
          await draw();
          await d.queue.onSubmittedWorkDone();
          const ms = performance.now() - t0;
          if (k <= ${PARK}) {
            shots["park" + k] = await this.readback(tex); spp["park" + k] = r.accumCount;
          }
          r.holdTick(ms);
        }
        shots.converged = await this.readback(tex); spp.converged = r.accumCount;
        spp.held = k;
        return { shots, spp, format: s.canvasFormat, renderScale: r.renderScale };
      } finally { tex.destroy(); s._capturing = false; s._wake?.("sea-qa"); } },
  }; true`);

  const meta = { width: W, height: H, tier, views: {} };
  for (const [name, v] of Object.entries(VIEWS)) {
    if (only && !only.includes(name)) continue;
    const at = await tab.eval(`window.__seaQa.place(${JSON.stringify(v)})`);
    await tab.eval(`window.__seaQa.settle()`);
    const res = await tab.eval(`window.__seaQa.run(${JSON.stringify(v)})`);
    for (const [shot, b64] of Object.entries(res.shots)) {
      writeFileSync(join(out, `${name}-${shot}.rgba`), Buffer.from(b64, "base64"));
    }
    meta.format = res.format;
    meta.views[name] = { ...v, start: at.position, spp: res.spp,
                         renderScale: res.renderScale };
    console.log(`${name}: from ${at.position.map((p) => p.toFixed(1)).join(",")} ` +
                `spp ${JSON.stringify(res.spp)} render scale ${res.renderScale}`);
  }
  writeFileSync(join(out, "meta.json"), JSON.stringify(meta, null, 1));
  await done(0);
} catch (err) {
  console.error(`sea_speckle_qa: ${err.message}`);
  await done(2);
}
