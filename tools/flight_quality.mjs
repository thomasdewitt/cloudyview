// How good is a frame while flying? A scripted flight through the browser's
// own renderer, scored against what the same tier converges to at the same
// camera pose.
//
//   node tools/flight_quality.mjs --web web [--cdp 9333] [--demo twpice|cyberpunk]
//        [--tiers low,minimal] [--paths cruise,strafe,pan] [--size 1280x720]
//        [--frames 90] [--every 15] [--fps 60] [--scale S] [--query "k=v&k=v"]
//        [--set "renderer.prop=value;..."] [--save DIR] [--json]
//
// The live loop is held off (the viewer's capture flag) and drawFrame is
// called once per scripted frame, as the loop would call it for a moving
// camera, at exactly 1/fps apart. Every `--every`th frame is read back. The
// reference for a frame is the same view, parked: render scale 1, the tier's
// own step sizes and look, REF_FRAMES accumulated. So the score is what
// motion and the tier's render scale cost, and nothing else.
//
// Per frame it prints
//   rmse   root-mean-square difference from the reference, in 8-bit levels
//   edge   the same over the images' gradient magnitudes: blur and noise
//          both raise it
//   sharp  mean gradient magnitude over the reference's: below 1 is blur,
//          above 1 is noise or ringing
//   flick  rms of (this frame minus the one before) minus the same step of
//          the references: what changes between frames that should not
// and per path the mean of each, plus ms/frame for the flight (GPU time for
// `--frames` frames without the read-backs, fenced once at the end).
//
// --scale flies the tier at another render scale than its own (the
// reference is still the tier's sampling at scale 1): what a tier would look
// like marching fewer or more pixels.
// --set assigns properties on the viewer (`window.soar`) after the tier is
// chosen, for A/B without a code change: "renderer.motionBlendAlpha=0.3".
// --save writes path-tier-NNN-{flight,ref}.png.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { openTab, requireHardwareAdapter, serveTree } from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i < 0 ? fallback : argv[i + 1];
};
const webDir = option("--web", null);
const cdpPort = option("--cdp", "9333");
const demo = option("--demo", "twpice");
const tiers = option("--tiers", "low").split(",");
const paths = option("--paths", "cruise,strafe,pan").split(",");
const [w, h] = option("--size", "1280x720").split("x").map(Number);
const frames = Number(option("--frames", "90"));
const every = Number(option("--every", "15"));
const fps = Number(option("--fps", "60"));
const query = option("--query", "");
const flightScale = option("--scale", null);
const sets = option("--set", "");
const saveDir = option("--save", null);
const asJson = argv.includes("--json");
const REF_FRAMES = 48;
if (!webDir) {
  console.error("usage: node tools/flight_quality.mjs --web DIR [options]; " +
                "see the head of this file");
  process.exit(2);
}
if (saveDir) mkdirSync(saveDir, { recursive: true });

// Each path, from the demo's opening view: metres a second along the
// camera's forward and right, and degrees a second of azimuth. `speed` is a
// multiple of the flight speed the scene opens at.
const PATHS = {
  cruise: { forward: 1.0, right: 0.0, yaw: 0.0 },
  strafe: { forward: 0.0, right: 1.0, yaw: 0.0 },
  pan: { forward: 0.3, right: 0.0, yaw: 25.0 },
  turn: { forward: 0.0, right: 0.0, yaw: 25.0 },
  still: { forward: 0.0, right: 0.0, yaw: 0.0 },
  // A cruise that stops dead at frame `stopAt` and holds, with the hold
  // ladder running as the live loop runs it. Scored 1, 4, 8, 16 and 32
  // frames after the stop: what stopping does to the picture.
  stop: { forward: 1.0, right: 0.0, yaw: 0.0, stopAt: 40 },
};

const server = await serveTree(webDir);
let tab = null;
const done = async (code) => { await tab?.close(); server.close(); process.exit(code); };
try {
  tab = await openTab(cdpPort);
  await tab.goto(`${server.base}/`);
  const adapter = await requireHardwareAdapter(tab);
  await tab.goto(`${server.base}/soar/`);
  // --demo cyberpunk is the night city, entered as pinned_still enters it.
  const city = demo === "cyberpunk";
  await tab.eval(`localStorage.clear();
    localStorage.setItem("soar.mode", ${JSON.stringify(city ? "cyberpunk" : "research")});
    localStorage.setItem("soar.tutorialDone", "1");
    localStorage.setItem("soar.tutorialResearchDone", "1"); true`);
  if (city) {
    await tab.goto(`${server.base}/soar/${query ? `?${query}` : ""}`);
    await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
    await tab.eval(`document.getElementById("fly-less").click(); true`);
  } else {
    await tab.goto(`${server.base}/soar/?demo=${demo}${query ? `&${query}` : ""}`);
  }
  await tab.waitFor(`(() => { const L = document.getElementById("loading");
    if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
    return window.soar?.scene && L.hidden; })()`, 180000);
  if (!asJson) {
    console.log(`flight quality: ${demo}, ${w}x${h}, ${frames} frames at ` +
                `${fps} fps (${adapter})${sets ? `, set ${sets}` : ""}` +
                `${flightScale ? `, scale ${flightScale}` : ""}` +
                `${query ? `, query ${query}` : ""}`);
  }

  for (const tier of tiers) {
    for (const pathName of paths) {
      const path = PATHS[pathName];
      if (!path) throw new Error(`unknown path '${pathName}'`);
      const r = await tab.eval(`(async () => {
        const s = window.soar;
        const cap = await import("${server.base}/soar/capture.js");
        const camMod = await import("${server.base}/soar/camera.js");
        const W = ${w}, H = ${h}, N = ${frames}, EVERY = ${every};
        const dt = 1 / ${fps};
        const path = ${JSON.stringify(path)};
        s._capturing = true;                 // the live loop stays off
        const target = cap.createOfflineTarget(s.device, [W, H], "flight-quality");
        const readback = new cap.Readback(s.device, [W, H]);
        const R = s.renderer;
        try {
          s.setQualityTier(${JSON.stringify(tier)});
          if (${JSON.stringify(flightScale)} !== null) {
            s.setRenderScale(Number(${JSON.stringify(flightScale)}));
          }
          for (const kv of ${JSON.stringify(sets)}.split(";").filter(Boolean)) {
            const [key, value] = kv.split("=");
            const parts = key.trim().split(".");
            let obj = s;
            for (const p of parts.slice(0, -1)) obj = obj[p];
            obj[parts.at(-1)] = JSON.parse(value);
          }
          await new Promise((ok) => setTimeout(ok, 50));
          while ((R._pendingPipelines?.size ?? 0) > 0 || R.lightBakePending) {
            if (R.lightBakePending) R.stepLightBake();
            await new Promise((ok) => setTimeout(ok, 20));
          }
          const cam = s.camera;
          const start = { pos: camMod.cameraWorldOrigin(
                              cam.start.position, cam.bmin, cam.bmax),
                          az: cam.start.azimuth, el: cam.start.elevation };
          const [fwd, right] = camMod.cameraBasis(start.az, start.el);
          const speed = cam.startSpeed;
          const stopAt = path.stopAt ?? Infinity;
          const pose = (k) => {
            const t = Math.min(k, stopAt) * dt;
            const a = path.forward * speed * t, b = path.right * speed * t;
            // From the start by one displacement, through the camera's own
            // move path: it keeps a second position, in the surface tile's
            // frame, and a position assigned directly would leave the city
            // standing still under moving clouds.
            cam.position = [...start.pos];
            cam._resyncSurface();
            cam.displace(a * fwd[0] + b * right[0], a * fwd[1] + b * right[1],
                         a * fwd[2] + b * right[2]);
            cam.azimuth = (start.az + path.yaw * t) % 360;
            cam.elevation = start.el;
          };
          const viewAt = (k) => ({ ...s._viewKwargs(), frameIndex: k,
                                   sceneTime: k * dt });
          const fly = async (onFrame) => {
            R.resetAccumulation();
            R.setCameraMoving(true);
            for (let k = 0; k < N; k++) {
              pose(k);
              if (k > stopAt) {
                // Parked, as the live loop parks: the ladder climbs on a
                // frame time that always affords it.
                R.setCameraMoving(false);
                R.holdTick(5.0);
              }
              await R.drawFrame(target.view, target.format, [W, H], viewAt(k),
                                { deltaSeconds: dt });
              if (onFrame) await onFrame(k);
            }
          };
          const grab = async () => (await readback.read(target.texture)).data;

          // The flight, with read-backs.
          const flown = [];
          const scored = (k) => (path.stopAt
            ? [1, 4, 8, 16, 32].includes(k - stopAt)
            : k > 0 && (k % EVERY === 0 || k === N - 1));
          await fly(async (k) => {
            // A scored frame, and the one before it for the flicker score.
            if (scored(k) || scored(k + 1)) {
              flown.push({ k, px: await grab(), scored: scored(k) });
            }
          });
          // The flight again, timed: no read-backs, one fence at the end.
          await s.device.queue.onSubmittedWorkDone();
          const t0 = performance.now();
          await fly(null);
          await s.device.queue.onSubmittedWorkDone();
          const msPerFrame = (performance.now() - t0) / N;

          // The references: each read-back pose, parked, at scale 1.
          const flightScale = R.flightRenderScale;
          const reprojectOn = R.reproject.on;
          R.reproject.on = false;      // references are averaged in place
          R.setCameraMoving(false);
          R.setRenderScale(1.0);
          const out = [];
          const luma = (px, n) => {
            const y = new Float32Array(n);
            for (let i = 0; i < n; i++) {
              y[i] = 0.2126 * px[4 * i] + 0.7152 * px[4 * i + 1]
                   + 0.0722 * px[4 * i + 2];
            }
            return y;
          };
          const grad = (y) => {
            const g = new Float32Array(W * H);
            for (let j = 1; j < H - 1; j++) {
              for (let i = 1; i < W - 1; i++) {
                const o = j * W + i;
                const gx = y[o + 1] - y[o - 1], gy = y[o + W] - y[o - W];
                g[o] = Math.hypot(gx, gy) * 0.5;
              }
            }
            return g;
          };
          const png = async (px) => {
            const canvas = new OffscreenCanvas(W, H);
            canvas.getContext("2d").putImageData(
              new ImageData(new Uint8ClampedArray(px), W, H), 0, 0);
            const blob = await canvas.convertToBlob({ type: "image/png" });
            const bytes = new Uint8Array(await blob.arrayBuffer());
            let bin = "";
            for (let i = 0; i < bytes.length; i += 0x8000) {
              bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
            }
            return btoa(bin);
          };
          for (const f of flown) {
            pose(f.k);
            R.resetAccumulation();
            const view = viewAt(f.k);
            for (let j = 0; j < ${REF_FRAMES}; j++) {
              await R.drawFrame(target.view, target.format, [W, H],
                                { ...view, frameIndex: j }, {});
            }
            const ref = await grab();
            f.ref = ref;
            if (!f.scored) continue;
            // Flicker: how far this frame's change from the last one is from
            // the change the references make over the same step. A flight
            // that only lags or blurs scores low here; one that boils scores
            // high.
            const prev = flown.find((g) => g.k === f.k - 1);
            let fe = 0;
            for (let i = 0; i < ref.length; i += 4) {
              for (let c = 0; c < 3; c++) {
                const d = (f.px[i + c] - prev.px[i + c])
                        - (ref[i + c] - prev.ref[i + c]);
                fe += d * d;
              }
            }
            let se = 0;
            for (let i = 0; i < ref.length; i += 4) {
              for (let c = 0; c < 3; c++) {
                const d = f.px[i + c] - ref[i + c];
                se += d * d;
              }
            }
            const n = W * H;
            const gf = grad(luma(f.px, n)), gr = grad(luma(ref, n));
            let ge = 0, sf = 0, sr = 0;
            for (let i = 0; i < n; i++) {
              const d = gf[i] - gr[i];
              ge += d * d; sf += gf[i]; sr += gr[i];
            }
            out.push({ k: f.k, rmse: Math.sqrt(se / (3 * n)),
                       edge: Math.sqrt(ge / n), sharp: sf / sr,
                       flick: Math.sqrt(fe / (3 * n)),
                       flight: ${JSON.stringify(Boolean(saveDir))} ? await png(f.px) : null,
                       ref: ${JSON.stringify(Boolean(saveDir))} ? await png(ref) : null });
          }
          R.setRenderScale(flightScale);
          R.reproject.on = reprojectOn;
          return { msPerFrame, frames: out,
                   renderSize: [R._targets?.w, R._targets?.h] };
        } finally {
          readback.destroy();
          target.texture.destroy();
          s._capturing = false;
        }
      })()`);
      const mean = (key) => r.frames.reduce((a, f) => a + f[key], 0) / r.frames.length;
      for (const f of r.frames) {
        if (saveDir) {
          const stem = `${pathName}-${tier}-${String(f.k).padStart(3, "0")}`;
          writeFileSync(join(saveDir, `${stem}-flight.png`), Buffer.from(f.flight, "base64"));
          writeFileSync(join(saveDir, `${stem}-ref.png`), Buffer.from(f.ref, "base64"));
        }
        delete f.flight; delete f.ref;
      }
      if (PATHS[pathName].stopAt && !asJson) {
        console.log(`${tier.padEnd(8)}${pathName.padEnd(8)}frames after the stop: ` +
          r.frames.map((f) => `+${f.k - PATHS[pathName].stopAt} rmse ` +
                              `${f.rmse.toFixed(2)} sharp ${f.sharp.toFixed(2)}`).join(" | "));
      }
      const row = { demo, tier, path: pathName, ms: +r.msPerFrame.toFixed(2),
                    rmse: +mean("rmse").toFixed(3), edge: +mean("edge").toFixed(3),
                    sharp: +mean("sharp").toFixed(3),
                    flick: +mean("flick").toFixed(3), frames: r.frames };
      if (asJson) {
        console.log(JSON.stringify(row));
      } else {
        console.log(`${tier.padEnd(8)}${pathName.padEnd(8)}` +
                    `rmse ${row.rmse.toFixed(2).padStart(6)}  ` +
                    `edge ${row.edge.toFixed(2).padStart(6)}  ` +
                    `sharp ${row.sharp.toFixed(2).padStart(5)}  ` +
                    `flick ${row.flick.toFixed(2).padStart(6)}  ` +
                    `${row.ms.toFixed(2).padStart(7)} ms/frame`);
      }
    }
  }
  await done(0);
} catch (err) {
  console.error(`flight_quality: ${err.message}`);
  await done(err.code ?? 2);
}
