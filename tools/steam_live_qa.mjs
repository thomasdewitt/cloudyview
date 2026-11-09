// The live view's steam (raymarch.wgsl fs_steam_live), held against the
// marched steam every still and video draws, in the browser's own renderer.
//
// The live view leaves the city's steam out of its accumulation and draws it
// after, at the live clock, so it keeps moving over a parked picture. That
// composite is approximate (steam_live_city lists how), and this is what
// measures it, on two pinned views of the cyberpunk city's start: STREET, at
// eye level with a plume ahead, and ABOVE, at 70 m looking steeply down at
// the same plume.
//
//   node tools/steam_live_qa.mjs --out DIR [--web DIR] [--cdp PORT]
//        [--size WxH] [--tier high] [--fog SIGMA] STEP...
//
// Steps, run in order on one page:
//   strip    the REAL live loop, parked on each view: wait for convergence,
//            then six screenshots ~0.9 s of clock apart, each logged with the
//            scene clock and the accumulated spp (strip-VIEW-k.png,
//            strip.json). Shows the steam moving over a converged picture.
//   compare  at two fixed clocks, the same accumulation (64 frames) drawn
//            with the live steam and with the marched steam, plus the bare
//            accumulator under the live one: VIEW-T-{live,exact,bare}.rgba
//            and compare.json (size, format). tools/steam_live_compare.py
//            turns them into numbers and a contact sheet.
//   timing   per-frame GPU time, blocks of frames submitted back to back and
//            drained once at each end, interleaved over 8 rounds: an
//            accumulating frame with the steam marched (M) and live (L), a
//            converged present with the live steam (P) and without (B), and
//            M2/L2/P2 repeating M/L/P as the null. Medians, ms.
// --fog SIGMA writes a QA-only fog bank (SIGMA /m, 25-200 m, 1.2 km square
// around the views) into the GPU volume before the steps and re-bakes the
// sun-tau cache: cloud between the camera and the steam, which the shipped
// demos almost never put within the steam's 360 m (their cloud bases sit
// near 1 km). It is invisible from eye level, where the field (z >= 25 m)
// never lies between the camera and a plume; ABOVE looks through ~64 m of it.
//
// --views a,b runs only the named views (street, above, png).
// --pose-png FILE adds a third view, "png": the city pose a soar screenshot
// carries in its render metadata (city.position_m, azimuth, elevation; the
// camera placed by its city-frame position, so the street under it is the
// screenshot's), compared at the screenshot's own clock — a reported bug,
// re-rendered with and without its steam.
//
// Auto-exposure is switched off first (it would move the exposure, which is
// in the scene key, between the two halves of a comparison), the hint panel
// and the minimap are hidden, and lens water is off.
//
// Everything here draws on the GPU: run it under tools/gpulock (board rules
// 10-11), and for timing check the card inside the lock first:
//   tools/gpulock bash -c "uv run python tools/benchmarking/gpu_idle.py && \
//     node tools/steam_live_qa.mjs --out temp/steam-qa --size 768x432 timing"
// Needs web/soar/demos (a reflinked copy, not a link) and a QA Chrome with a
// hardware adapter on --cdp (see tools/soar_cdp.mjs).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { openTab, serveTree } from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const flags = new Map();
const steps = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i].startsWith("--")) flags.set(argv[i], argv[++i]);
  else steps.push(argv[i]);
}
const here = dirname(fileURLToPath(import.meta.url));
const out = flags.get("--out");
const webDir = resolve(flags.get("--web") ?? join(here, "..", "web"));
const cdpPort = flags.get("--cdp") ?? "9333";
const [W, H] = (flags.get("--size") ?? "960x540").split("x").map(Number);
const tier = flags.get("--tier") ?? "high";
const fog = flags.has("--fog") ? Number(flags.get("--fog")) : null;
const posePng = flags.get("--pose-png") ?? null;
const KNOWN = new Set(["strip", "compare", "timing"]);
if (!out || steps.length === 0 || steps.some((s) => !KNOWN.has(s))
    || !(W > 0 && H > 0) || (fog !== null && !(fog > 0))) {
  console.error("usage: node tools/steam_live_qa.mjs --out DIR [--web DIR] " +
                "[--cdp PORT] [--size WxH] [--tier T] [--fog SIGMA] " +
                "strip|compare|timing ...");
  process.exit(2);
}
mkdirSync(out, { recursive: true });

// World positions at the cyberpunk start (its first case, fly-less).
const VIEWS = {
  street: { x: 5013, y: 17456, z: 2, az: 90, el: 0 },
  above: { x: 4990, y: 17456, z: 70, az: 90, el: -45 },
};
const CLOCKS = [100.0, 101.5];       // the compare step's fixed scene clocks

/** The render metadata a soar screenshot carries (its tEXt chunk). */
function pngMetadata(path) {
  const b = readFileSync(path);
  for (let i = 8; i < b.length;) {
    const n = b.readUInt32BE(i);
    const type = b.toString("latin1", i + 4, i + 8);
    if (type === "tEXt") {
      const text = b.toString("latin1", i + 8, i + 8 + n);
      const [key, value] = [text.slice(0, text.indexOf("\0")),
                            text.slice(text.indexOf("\0") + 1)];
      if (key === "cloudyview.render_metadata") return JSON.parse(value);
    }
    i += 12 + n;
  }
  throw new Error(`${path} carries no cloudyview.render_metadata`);
}
let pngClock = null;
const only = flags.has("--views") ? flags.get("--views").split(",") : null;
if (posePng) {
  const md = pngMetadata(posePng);
  if (!md.city) throw new Error(`${posePng} is not a city screenshot`);
  const [cx, cy, cz] = md.city.position_m;
  VIEWS.png = { city: [cx, cy], z: cz, az: md.camera.azimuth,
                el: md.camera.elevation, fov: md.camera.fov };
  pngClock = md.render.scene_time;
}
const COMPARE_FRAMES = 64;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const views = () => Object.entries(VIEWS).filter(([n]) => !only || only.includes(n));
const server = await serveTree(webDir);
let tab = null;
try {
  tab = await openTab(cdpPort);
  await tab.send("Emulation.setDeviceMetricsOverride",
                 { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  await tab.goto(`${server.base}/soar/`);
  await tab.eval(`localStorage.clear();
    localStorage.setItem("soar.mode", "cyberpunk");
    localStorage.setItem("soar.tutorialDone", "1");
    localStorage.setItem("soar.tutorialResearchDone", "1");
    localStorage.setItem("soar.lensWater", "off"); true`);
  await tab.goto(`${server.base}/soar/`);
  await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
  await tab.eval(`document.getElementById("fly-less").click(); true`);
  await tab.waitFor(`(() => { const L = document.getElementById("loading");
      if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
      return window.soar?.scene && L.hidden; })()`, 180000);
  await tab.eval(`(() => { const s = window.soar;
    s.setQualityTier(${JSON.stringify(tier)});
    s.autoExposure = false; s.minimapMode = "off";
    const hint = [...document.querySelectorAll("body *")].filter((e) =>
      e.children.length > 0 && e.innerText?.includes("to fly")
      && e.innerText.length < 400).pop();
    if (hint) hint.style.visibility = "hidden";
    return true; })()`);
  // Setting the tier prefetches it and its two neighbours, each lighting
  // method (dozens of city compiles when the probe had chosen another tier).
  await tab.waitFor(`(window.soar.renderer._pendingPipelines?.size ?? 0) === 0
      && !window.soar.renderer.lightBakePending`, 900000);

  // Page-side helpers.
  await tab.eval(`window.__steamQa = {
    place(v) { const s = window.soar, c = s.camera;
      if (v.city) {
        // By the city frame: displace moves the camera's surface frame with it.
        c.displace(v.city[0] - c.surfacePosition[0], v.city[1] - c.surfacePosition[1],
                   v.z - c.position[2]);
        if (v.fov) c.fov = v.fov;
      } else {
        c.displace(v.x - c.position[0], v.y - c.position[1], v.z - c.position[2]);
      }
      c.azimuth = v.az; c.elevation = v.el; s._wake?.("steam-qa"); return true; },
    half(x) { const f = new Float32Array([x]), i = new Uint32Array(f.buffer)[0];
      return ((i >>> 16) & 0x8000) | ((((i >>> 23) & 0xff) - 112) << 10)
             | ((i & 0x7fffff) >>> 13); },
    async fog(x, y, halfM, ztop, sigma) { const s = window.soar, sc = s.scene;
      const [nx, , nz] = sc.shape;
      const dx = (sc.bmax[0] - sc.bmin[0]) / nx, dz = (sc.bmax[2] - sc.bmin[2]) / nz;
      const i0 = Math.floor((x - sc.bmin[0] - halfM) / dx);
      const j0 = Math.floor((y - sc.bmin[1] - halfM) / dx);
      const n = Math.ceil(2 * halfM / dx), kz = Math.ceil((ztop - sc.bmin[2]) / dz);
      // The volume texture is (z, y, x): z fastest (scene.js writeVolumeSlab).
      s.device.queue.writeTexture(
        { texture: sc.volumeTexture, origin: { x: 0, y: j0, z: i0 } },
        new Uint16Array(kz * n * n).fill(this.half(sigma)),
        { bytesPerRow: kz * 2, rowsPerImage: n }, [kz, n, n]);
      const r = s.renderer;
      r.invalidateLightCache(); r.startLightBake(s._viewKwargs());
      while (r.lightBakePending) {
        r.stepLightBake(64); await s.device.queue.onSubmittedWorkDone();
      }
      return { i0, j0, n, kz }; },
    async readback(tex) { const d = window.soar.device;
      const buf = d.createBuffer({ size: ${W * H * 4},
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: ${W * 4} },
                              [${W}, ${H}]);
      d.queue.submit([enc.finish()]); await buf.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(buf.getMappedRange()).slice();
      buf.unmap(); buf.destroy();
      let bin = "";
      for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      }
      return btoa(bin); },
    // One accumulation at the fixed clock T, the live loop held off: the
    // live steam or the marched; for the live one also the bare accumulator
    // (presentLast with no steam).
    async render(live, T, frames) { const s = window.soar, r = s.renderer, d = s.device;
      s._capturing = true; await new Promise((ok) => setTimeout(ok, 300));
      r.setCameraMoving(false); r.resetAccumulation();
      const tex = d.createTexture({ size: [${W}, ${H}], format: s.canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      try {
        for (let i = 0; i < frames; i++) {
          await r.drawFrame(tex.createView(), s.canvasFormat, [${W}, ${H}],
            { ...s._viewKwargs(), frameIndex: i, sceneTime: T },
            { liveSteam: live, deltaSeconds: 1 / 60 });
        }
        const img = await this.readback(tex);
        let bare = null;
        if (live) {
          r.presentLast(tex.createView(), s.canvasFormat, [${W}, ${H}]);
          bare = await this.readback(tex);
        }
        return { img, bare, format: s.canvasFormat, spp: r.accumCount,
                 renderScale: r.renderScale };
      } finally { tex.destroy(); s._capturing = false; s._wake?.("steam-qa"); } },
  }; true`);

  const save = (name, b64) => writeFileSync(join(out, name), Buffer.from(b64, "base64"));
  if (fog !== null) {
    const f = await tab.eval(`window.__steamQa.fog(${VIEWS.street.x}, ${VIEWS.street.y}, 600, 200, ${fog})`);
    console.log(`fog ${fog} /m written: ${JSON.stringify(f)}`);
  }

  for (const step of steps) {
    if (step === "strip") {
      const log = [];
      for (const [name, v] of views()) {
        await tab.eval(`window.__steamQa.place(${JSON.stringify(v)})`);
        await tab.waitFor(`window.soar.renderer.converged`, 120000);
        for (let k = 0; k < 6; k++) {
          const st = await tab.eval(`({ clock: window.soar.sceneTime,
            spp: window.soar.renderer.accumCount,
            converged: window.soar.renderer.converged })`);
          const shot = await tab.send("Page.captureScreenshot", { format: "png" });
          save(`strip-${name}-${k}.png`, shot.data);
          log.push({ view: name, k, ...st });
          console.log(`strip ${name} ${k} clock ${st.clock.toFixed(2)} s ` +
                      `spp ${st.spp}${st.converged ? "" : " (not converged)"}`);
          await sleep(700);
        }
      }
      writeFileSync(join(out, "strip.json"), JSON.stringify(log, null, 1));
    } else if (step === "compare") {
      let meta = null;
      for (const [name, v] of views()) {
        await tab.eval(`window.__steamQa.place(${JSON.stringify(v)})`);
        await sleep(1500);
        console.log(`compare ${name} camera ${await tab.eval(`JSON.stringify({
          position: window.soar.camera.position, city: window.soar.camera.surfacePosition,
          azimuth: window.soar.camera.azimuth, elevation: window.soar.camera.elevation })`)}`);
        for (const T of name === "png" ? [pngClock] : CLOCKS) {
          const lv = await tab.eval(`window.__steamQa.render(true, ${T}, ${COMPARE_FRAMES})`);
          const ex = await tab.eval(`window.__steamQa.render(false, ${T}, ${COMPARE_FRAMES})`);
          const stem = `${name}-T${T}`;
          save(`${stem}-live.rgba`, lv.img);
          save(`${stem}-bare.rgba`, lv.bare);
          save(`${stem}-exact.rgba`, ex.img);
          meta = { width: W, height: H, format: lv.format, frames: COMPARE_FRAMES,
                   renderScale: lv.renderScale, tier, fog };
          console.log(`compare ${stem}: ${lv.spp}/${ex.spp} spp, render scale ` +
                      `${lv.renderScale}`);
        }
      }
      writeFileSync(join(out, "compare.json"), JSON.stringify(meta, null, 1));
    } else if (step === "timing") {
      for (const [name, v] of views()) {
        await tab.eval(`window.__steamQa.place(${JSON.stringify(v)})`);
        await sleep(1500);
        const res = await tab.eval(`(async () => {
          const s = window.soar, r = s.renderer, d = s.device;
          s._capturing = true; await new Promise((ok) => setTimeout(ok, 300));
          r.setCameraMoving(false);
          const tex = d.createTexture({ size: [${W}, ${H}], format: s.canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
          const view = (i) => ({ ...s._viewKwargs(), frameIndex: i, sceneTime: 100 + i / 60 });
          const t = { M: [], L: [], M2: [], L2: [], P: [], B: [], P2: [] };
          let fi = 0;
          const draw = (live) => r.drawFrame(tex.createView(), s.canvasFormat,
            [${W}, ${H}], view(fi++), { liveSteam: live });
          const march = async (live, key) => {
            r.resetAccumulation();
            for (let i = 0; i < 4; i++) await draw(live);
            await d.queue.onSubmittedWorkDone(); const t0 = performance.now();
            for (let i = 0; i < 24; i++) await draw(live);
            await d.queue.onSubmittedWorkDone(); t[key].push((performance.now() - t0) / 24);
          };
          const present = async (steam, key) => {
            await d.queue.onSubmittedWorkDone(); const t0 = performance.now();
            for (let i = 0; i < 48; i++) {
              r.presentLast(tex.createView(), s.canvasFormat, [${W}, ${H}], null, null,
                            steam ? { sceneTime: 100 + i / 60 } : null);
            }
            await d.queue.onSubmittedWorkDone(); t[key].push((performance.now() - t0) / 48);
          };
          try {
            for (let round = 0; round < 8; round++) {
              await march(false, "M"); await march(true, "L"); await march(false, "M2");
              // The presents re-present a LIVE march: the steam pass needs one.
              await march(true, "L2");
              await present(true, "P"); await present(false, "B"); await present(true, "P2");
            }
          } finally { tex.destroy(); s._capturing = false; }
          const med = (a) => { const b = [...a].sort((x, y) => x - y);
                               return b[Math.floor(b.length / 2)]; };
          return Object.fromEntries(Object.entries(t).map(([k, a]) => [k, med(a)]));
        })()`);
        const f = (k) => res[k].toFixed(3);
        const inForce = await tab.eval(`({ tier: window.soar.renderer.qualityTier,
          scale: window.soar.renderer.renderScale })`);
        if (inForce.tier !== tier) {
          throw new Error(`timed at tier ${inForce.tier}, not the ${tier} asked for`);
        }
        console.log(`timing ${name} ${W}x${H} tier ${inForce.tier} (scale ${inForce.scale}) ms/frame: marched ${f("M")} live ` +
                    `${f("L")} (null ${f("M2")}/${f("L2")}) | parked present: ` +
                    `steam ${f("P")} bare ${f("B")} (null ${f("P2")}) | live/marched ` +
                    `${(res.L / res.M).toFixed(3)}, null ${(res.M2 / res.M).toFixed(3)}`);
      }
    }
  }
  await tab.close(); tab = null;
  server.close();
} catch (err) {
  console.error(`steam_live_qa: ${err.message}`);
  await tab?.close();
  server.close();
  process.exit(err.code ?? 1);
}
