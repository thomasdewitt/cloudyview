// The city's first surface stage in the browser, as the fragment pass
// (fs_surface) and as the compute pass (cs_surface, the ?surfacecs
// experiment, raymarch.wgsl), interleaved with a null (the fragment pass
// again), on the renderer's own targets, bind group and uniforms for a view.
// Measurement only.
//
//   tools/gpulock node tools/surface_cs_timer.mjs --web DIR [--cdp PORT]
//        [--passes N] [--rounds R] [--query surfacecs=1]
//
// DIR is a checkout's web/ with soar/demos present (cyberpunk mode flies
// marine-congestus-coarse over the city). CDP_PORT is the QA Chrome
// (tools/qa_chrome.sh). Per view, the camera is put at the night-city
// harness's pose (xy a fraction of the scene box, z in metres); one still at
// low (2560x1440) is rendered with _encodeMarch recording what it was given
// (the targets come out at the capture size); then R rounds of N
// first-stage passes per variant in one submit each, timed from submit to
// onSubmittedWorkDone (the device has no timestamp queries), the first round
// warmup. Prints the median ms a pass per variant, the ratios and the
// rounds' spread. Run it on an otherwise idle card
// (tools/benchmarking/gpu_idle.py --outside-lock first).
import { openTab, requireHardwareAdapter, serveTree } from "./soar_cdp.mjs";
import { requireGpulock } from "./gpulock_guard.mjs";

// It drives the QA Chrome's WebGPU: refuse outside tools/gpulock before
// anything else runs (gpulock_guard.mjs).
requireGpulock("tools/surface_cs_timer.mjs");

const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i < 0 ? fallback : argv[i + 1];
};
const webDir = option("--web", null);
const port = option("--cdp", "9333");
const N = Number(option("--passes", "40"));
const ROUNDS = Number(option("--rounds", "7"));
const query = option("--query", "surfacecs=1");
if (!webDir || !(N > 0) || !(ROUNDS > 1)) {
  console.error("usage: tools/gpulock node tools/surface_cs_timer.mjs --web DIR " +
                "[--cdp PORT] [--passes N] [--rounds R>1] [--query surfacecs=1]");
  process.exit(2);
}
const VIEWS = {
  aerial: { xy: [0.50, 0.50], z: 3300.0, elevation: -28.0, azimuth: 45.0, fov: 65.0 },
  horizon: { xy: [0.80, 0.72], z: 1600.0, elevation: -4.0, azimuth: 227.0, fov: 68.0 },
  street: { xy: [0.60, 0.50], z: 60.0, elevation: 25.0, azimuth: 265.0, fov: 80.0 },
};

const server = await serveTree(webDir);
const tab = await openTab(port);
try {
  await tab.goto(`${server.base}/`);
  console.log(`adapter: ${await requireHardwareAdapter(tab)}`);
  await tab.goto(`${server.base}/soar/?${query}`);
  await tab.eval(`localStorage.clear();
    localStorage.setItem("soar.mode", "cyberpunk");
    localStorage.setItem("soar.tutorialDone", "1");
    localStorage.setItem("soar.tutorialResearchDone", "1"); true`);
  await tab.goto(`${server.base}/soar/?${query}`);
  await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
  await tab.eval(`document.getElementById("fly-less").click(); true`);
  await tab.waitFor(`(() => { const L = document.getElementById("loading");
    if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
    return window.soar?.scene && L.hidden; })()`, 180000);
  await tab.waitFor(`(window.soar.renderer._pendingPipelines?.size ?? 0) === 0
    && !window.soar.renderer.lightBakePending`, 180000);
  const cs = await tab.eval(`JSON.stringify(window.soar.renderer.surfaceCS)`);
  console.log(`surfaceCS ${cs}; N ${N} passes a submit, ${ROUNDS} rounds ` +
              "(the first is warmup)");
  for (const [name, v] of Object.entries(VIEWS)) {
    const out = await tab.eval(`(async () => {
      const s = window.soar, r = s.renderer, dev = s.device;
      s._capturing = true;
      const cam = s.camera, sc = s.scene;
      const want = [sc.bmin[0] + ${v.xy[0]} * (sc.bmax[0] - sc.bmin[0]),
                    sc.bmin[1] + ${v.xy[1]} * (sc.bmax[1] - sc.bmin[1]), ${v.z}];
      cam.jump(want[0] - cam.position[0], want[1] - cam.position[1],
               want[2] - cam.position[2]);
      cam.azimuth = ${v.azimuth}; cam.elevation = ${v.elevation};
      cam.setFov(${v.fov});
      // Record what the still's march is given, and its lighting.
      let rec = null, lighting = null;
      const sp = r._stagePipelines.bind(r), em = r._encodeMarch.bind(r);
      r._stagePipelines = (l) => { lighting = l; return sp(l); };
      r._encodeMarch = (enc, targets, rbg, rp, surfaces) => {
        rec = { targets, rbg, surfaces, lighting };
        return em(enc, targets, rbg, rp, surfaces);
      };
      const cap = await import("${server.base}/soar/capture.js");
      const look = s._captureLook("low");
      look.exposure = 2.0;
      const view = { ...s._captureViewKwargs(look), frameIndex: 0, sceneTime: 0.0 };
      await cap.renderStill(dev, r, view, [2560, 1440], "low");
      r._stagePipelines = sp; r._encodeMarch = em;
      if (!rec) throw new Error("no march recorded");
      const surf = rec.targets.surface;
      const spec = r._shaderSpec({ lighting: rec.lighting });
      const frag = await r._pipeline("fs_surface", ["rgba32float", "rgba32uint"], spec);
      const comp = rec.surfaces[0][1];
      if (!(comp instanceof GPUComputePipeline)) {
        throw new Error("the first stage is not compute: is ?surfacecs set?");
      }
      // The slot the first stage writes in a three-stage chain: "a".
      const plainView = surf.plainView ?? surf.view;
      const csGroup = surf.csGroups.a ?? surf.csGroups.final;
      const [wx, wy] = r.surfaceCS;
      const W = rec.targets.w, H = rec.targets.h;
      const encode = (enc, kind) => {
        if (kind === "cs") {
          const cp = enc.beginComputePass();
          cp.setPipeline(comp); cp.setBindGroup(0, rec.rbg);
          cp.setBindGroup(1, r.surfaceCSEmptyGroup); cp.setBindGroup(2, csGroup);
          cp.dispatchWorkgroups(Math.ceil(W / wx), Math.ceil(H / wy));
          cp.end();
        } else {
          const p = enc.beginRenderPass({ colorAttachments: [
            { view: plainView, loadOp: "clear", storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 1e30 } },
            { view: surf.veilView, loadOp: "clear", storeOp: "store",
              clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
          p.setPipeline(frag); p.setBindGroup(0, rec.rbg);
          p.setBindGroup(1, r.surfaceDummyGroup); p.draw(3); p.end();
        }
      };
      const kinds = { frag: "frag", cs: "cs", null: "frag" };
      const t = { frag: [], cs: [], null: [] };
      for (let k = 0; k < ${ROUNDS}; k++) {
        for (const label of ["frag", "cs", "null"]) {
          const enc = dev.createCommandEncoder();
          for (let i = 0; i < ${N}; i++) encode(enc, kinds[label]);
          await dev.queue.onSubmittedWorkDone();
          const t0 = performance.now();
          dev.queue.submit([enc.finish()]);
          await dev.queue.onSubmittedWorkDone();
          if (k > 0) t[label].push((performance.now() - t0) / ${N});
        }
      }
      const med = (a) => { const b = [...a].sort((x, y) => x - y);
                           return b[Math.floor(b.length / 2)]; };
      const spread = (a) => Math.max(...a) / Math.min(...a) - 1;
      return JSON.stringify({
        W, H, frag: med(t.frag), cs: med(t.cs), null: med(t.null),
        spread: Math.max(spread(t.frag), spread(t.cs), spread(t.null)) });
    })()`);
    const o = JSON.parse(out);
    console.log(`${name.padEnd(8)} ${o.W}x${o.H}  frag ${o.frag.toFixed(3)} ms  ` +
                `cs ${o.cs.toFixed(3)} ms  null ${o.null.toFixed(3)} ms  ` +
                `cs/frag ${(o.cs / o.frag).toFixed(3)}  ` +
                `null/frag ${(o.null / o.frag).toFixed(3)}  ` +
                `round spread ${(100 * o.spread).toFixed(1)}%`);
  }
} finally {
  await tab.close();
  server.close();
}
