// A still pinned in every input — camera, frame index, size, tier, exposure,
// scene clock — rendered by the browser's own renderer, hashed byte for
// byte, so two trees can be compared exactly.
//
// Why: tools/soar_shader_ab.py drives one Python host against two shader
// revisions, which cannot compare a change that also moves the hosts (a new
// binding, a new pass: the surface pre-pass, 2026-09-23). This compares the
// whole browser build — hosts and shader together — so run it once per tree
// and diff the lines.
//
//   node tools/pinned_still.mjs --web DIR [--cdp PORT]
//        [--scenes city,day,desert] [--tiers high,low,max] [--size 640x360]
//
// DIR is a checkout's web/ directory WITH soar/demos present (the symlink
// into the site's demo store): the day scene is the TWP-ICE demo over the
// sea, the desert is the same demo over the desert (?surface=desert), the
// city is cyberpunk mode's first case. The desert's tier picks its variant
// as flight does (renderer._desertLiteFor): the default tiers take the full
// ground at high and max and the LITE ground at low, and each desert line
// says which it drew. The tree is served here on a
// fresh port (tools/soar_cdp.mjs) and its shader hash is printed with the
// stills. CDP_PORT is a Chrome with --remote-debugging-port and a hardware
// WebGPU adapter (see tests/test_tint_compile.py). The live loop is held off
// while each still renders (the viewer's own capture flag), and every
// pipeline and the sun-tau bake are settled first, so nothing but the tree
// can move a byte.
//
// Output: one line per still, "scene/tier sha256[:16] mean-rgb camera",
// the desert's tier suffixed "+lite" where it drew the LITE ground.
// Exit 0; 2 when it cannot run.

import { join } from "node:path";
import { openTab, serveTree, servedShaderHash } from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i < 0 ? fallback : argv[i + 1];
};
const webDir = option("--web", null);
const cdpPort = option("--cdp", "9333");
const scenes = option("--scenes", "city,day,desert").split(",");
const tiers = option("--tiers", "high,low,max").split(",");
const [w, h] = option("--size", "640x360").split("x").map(Number);
if (!webDir) {
  console.error("usage: node tools/pinned_still.mjs --web DIR [--cdp PORT] " +
                "[--scenes city,day,desert] [--tiers high,low,max] [--size WxH]");
  process.exit(2);
}

const SCENES = {
  city: { mode: "cyberpunk", demo: null },
  day: { mode: "research", demo: "twpice" },
  // Low over the ground, looking ahead and down, so the stills hold the near
  // stones and plants (the refine stage) as well as the far forms.
  desert: { mode: "research", demo: "twpice", surface: "desert",
            ground: { up: 3.0, elevation: -8.0 } },
};

const server = await serveTree(webDir);
let tab = null;
const done = async (code) => { await tab?.close(); server.close(); process.exit(code); };
try {
  const shader = await servedShaderHash(server.base,
                                        join(server.root, "soar", "raymarch.wgsl"));
  console.log(`tree ${server.root} (raymarch.wgsl sha256 ${shader})`);
  tab = await openTab(cdpPort);
  for (const name of scenes) {
    const sc = SCENES[name];
    if (!sc) throw new Error(`unknown scene '${name}' (city, day, desert)`);
    await tab.goto(`${server.base}/soar/`);
    await tab.eval(`localStorage.clear();
      localStorage.setItem("soar.mode", ${JSON.stringify(sc.mode)});
      localStorage.setItem("soar.tutorialDone", "1");
      localStorage.setItem("soar.tutorialResearchDone", "1"); true`);
    if (sc.demo) {
      const surface = sc.surface ? `&surface=${sc.surface}` : "";
      await tab.goto(`${server.base}/soar/?demo=${sc.demo}${surface}`);
    } else {
      await tab.goto(`${server.base}/soar/`);
      await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
      await tab.eval(`document.getElementById("fly-less").click(); true`);
    }
    await tab.waitFor(`(() => { const L = document.getElementById("loading");
      if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
      return window.soar?.scene && L.hidden; })()`, 180000);
    // The surface asked for, or no still: a ?surface the tree ignores would
    // otherwise hash the sea as the desert.
    const desert = await tab.eval(`Boolean(window.soar.scene.desert)`);
    if (desert !== (sc.surface === "desert")) {
      throw new Error(`scene '${name}' loaded ${desert ? "the desert" : "no desert"}`);
    }
    if (sc.ground) {
      await tab.eval(`(() => { const cam = window.soar.camera;
        const floor = window.soar.scene.desertFloorAt(cam.surfacePosition);
        cam.displace(0.0, 0.0, floor + ${sc.ground.up} - cam.position[2]);
        cam.elevation = ${sc.ground.elevation};
        return true; })()`);
    }
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
          return { sha: [...new Uint8Array(d)].slice(0, 8)
                     .map((b) => b.toString(16).padStart(2, "0")).join(""),
                   mean: (sum / (img.data.length / 4) / 3).toFixed(4),
                   cam: s.camera.position.map((v) => v.toFixed(2)).join(","),
                   // A tree from before DESERT_LITE has no lite ground.
                   lite: s.renderer._desertLiteFor?.(${JSON.stringify(tier)})
                         ?? false };
        } finally { s._capturing = false; }
      })()`);
      console.log(`${name}/${tier}${r.lite ? "+lite" : ""} ${r.sha} ${r.mean} ` +
                  r.cam);
    }
  }
  await done(0);
} catch (err) {
  console.error(`pinned_still: ${err.message}`);
  await done(err.code ?? 2);
}
