// A still pinned in every input — camera, frame index, size, tier, exposure,
// scene clock — rendered by the browser's own renderer, hashed byte for
// byte, so two trees can be compared exactly.
//
// Why: tools/soar_shader_ab.py drives one Python host against two shader
// revisions, which cannot compare a change that also moves the hosts (a new
// binding, a new pass). This compares the
// whole browser build — hosts and shader together — so run it once per tree
// and diff the lines.
//
//   node tools/pinned_still.mjs --web DIR [--cdp PORT]
//        [--scenes city,day] [--tiers high,low,max] [--size 640x360]
//        [--save DIR]
//
// DIR is a checkout's web/ directory WITH soar/demos present (the symlink
// into the site's demo store): the day scene is the TWP-ICE demo over the
// sea, the city is cyberpunk mode's first case. The tree is served here on a
// fresh port (tools/soar_cdp.mjs) and its shader hash is printed with the
// stills. CDP_PORT is a Chrome with --remote-debugging-port and a hardware
// WebGPU adapter (see tests/lint/test_tint_compile.py). The live loop is held off
// while each still renders (the viewer's own capture flag), and every
// pipeline and the sun-tau bake are settled first, so nothing but the tree
// can move a byte.
//
// Output: one line per still, "scene/tier sha256[:16] mean-rgb camera". --save
// DIR also writes each still there as scene-tier.png (the exact bytes the
// hash is of), for looking at what a changed hash changed.
// Exit 0; 2 when it cannot run.

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
const scenes = option("--scenes", "city,day").split(",");
const tiers = option("--tiers", "high,low,max").split(",");
const [w, h] = option("--size", "640x360").split("x").map(Number);
const saveDir = option("--save", null);
// Extra query parameters for every soar page (a flag the viewer reads at
// load).
const query = option("--query", "");
const q = (sep) => (query ? `${sep}${query}` : "");
if (saveDir) mkdirSync(saveDir, { recursive: true });
if (!webDir) {
  console.error("usage: node tools/pinned_still.mjs --web DIR [--cdp PORT] " +
                "[--scenes city,day] [--tiers high,low,max] [--size WxH] " +
                "[--save DIR]");
  process.exit(2);
}

const SCENES = {
  city: { mode: "cyberpunk", demo: null },
  day: { mode: "research", demo: "twpice" },
};

const server = await serveTree(webDir);
let tab = null;
const done = async (code) => { await tab?.close(); server.close(); process.exit(code); };
try {
  tab = await openTab(cdpPort);
  await tab.goto(`${server.base}/`);
  const adapter = await requireHardwareAdapter(tab);
  console.log(`tree ${server.root} (${adapter})`);
  for (const name of scenes) {
    const sc = SCENES[name];
    if (!sc) throw new Error(`unknown scene '${name}' (city, day)`);
    await tab.goto(`${server.base}/soar/${q("?")}`);
    await tab.eval(`localStorage.clear();
      localStorage.setItem("soar.mode", ${JSON.stringify(sc.mode)});
      localStorage.setItem("soar.tutorialDone", "1");
      localStorage.setItem("soar.tutorialResearchDone", "1"); true`);
    if (sc.demo) {
      await tab.goto(`${server.base}/soar/?demo=${sc.demo}${q("&")}`);
    } else {
      await tab.goto(`${server.base}/soar/${q("?")}`);
      await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
      await tab.eval(`document.getElementById("fly-less").click(); true`);
    }
    await tab.waitFor(`(() => { const L = document.getElementById("loading");
      if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
      return window.soar?.scene && L.hidden; })()`, 180000);
    await tab.waitFor(`(window.soar.renderer._pendingPipelines?.size ?? 0) === 0
      && !window.soar.renderer.lightBakePending`, 180000);
    for (const tier of tiers) {
      const r = await tab.eval(`(async () => {
        const s = window.soar;
        const cap = await import("${server.base}/soar/capture.js");
        s._capturing = true;               // the live loop stays off
        try {
          const look = s._captureLook(${JSON.stringify(tier)});
          look.exposure = 2.0;
          const view = { ...s._captureViewKwargs(look), frameIndex: 0,
                         sceneTime: 0.0 };
          const img = await cap.renderStill(s.device, s.renderer, view,
                                            [${w}, ${h}], ${JSON.stringify(tier)});
          const d = await crypto.subtle.digest("SHA-256", img.data);
          let sum = 0;
          for (let i = 0; i < img.data.length; i += 4) {
            sum += img.data[i] + img.data[i + 1] + img.data[i + 2];
          }
          let png = null;
          if (${JSON.stringify(Boolean(saveDir))}) {
            const canvas = new OffscreenCanvas(img.width, img.height);
            canvas.getContext("2d").putImageData(img, 0, 0);
            const blob = await canvas.convertToBlob({ type: "image/png" });
            const bytes = new Uint8Array(await blob.arrayBuffer());
            let bin = "";
            for (let i = 0; i < bytes.length; i += 0x8000) {
              bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
            }
            png = btoa(bin);
          }
          return { png, sha: [...new Uint8Array(d)].slice(0, 8)
                     .map((b) => b.toString(16).padStart(2, "0")).join(""),
                   mean: (sum / (img.data.length / 4) / 3).toFixed(4),
                   cam: s.camera.position.map((v) => v.toFixed(2)).join(",") };
        } finally { s._capturing = false; }
      })()`);
      console.log(`${name}/${tier} ${r.sha} ${r.mean} ` +
                  r.cam);
      if (saveDir) {
        writeFileSync(join(saveDir, `${name}-${tier}.png`),
                      Buffer.from(r.png, "base64"));
      }
    }
  }
  await done(0);
} catch (err) {
  console.error(`pinned_still: ${err.message}`);
  await done(err.code ?? 2);
}
