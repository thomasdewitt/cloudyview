// Compile every WGSL module soar ships with a browser's own WGSL compiler
// (Chrome's Tint), in every specialization the app can ask for.
//
// Why this exists: naga — the compiler behind wgpu, so behind the Python host
// and the whole test suite — is more permissive than the WGSL spec in places.
// It accepted `a ^ b * c`, which the spec gives no precedence and Tint
// refuses, and master (abbc9dc, 2026-09-23) shipped a shader no Chrome, Edge
// or Safari could build while every test passed. A green suite says nothing
// about whether a browser can compile the shader; this does.
//
//   node tools/tint_check.mjs --web DIR [--cdp PORT]
//   node tools/tint_check.mjs BASE_URL --expect FILE [--cdp PORT]
//
// --web serves DIR (a checkout's web/ directory) itself, on a port the
// kernel picks, so no other session's server can answer in its place. A
// BASE_URL someone else serves is accepted only with --expect FILE, the
// raymarch.wgsl it must be serving. Either way the served shader is fetched
// and compared byte for byte with the local one before anything compiles —
// a port collision once had the gate compile another session's tree and
// report green (2026-09-23) — and its hash is printed for a merge request to
// quote. The page imports soar/renderer.js from the served tree, so the
// specialization is the app's own. --cdp (default 9333) is a Chrome with
// --remote-debugging-port and a hardware WebGPU adapter (on the Fedora box:
// --enable-features=Vulkan --ozone-platform=x11). Compile only — nothing is
// drawn.
//
// Exit 0 clean; 1 on any compile error; 2 when it cannot run at all (no
// Chrome, no adapter); 3 when the served shader is not the expected one.

import { join } from "node:path";
import { openTab, requireHardwareAdapter, serveTree, servedShaderHash }
  from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const option = (name) => {
  const i = argv.indexOf(name);
  if (i < 0) return null;
  const v = argv[i + 1];
  argv.splice(i, 2);
  return v;
};
const webDir = option("--web");
let expectFile = option("--expect");
const cdpPort = option("--cdp") ?? "9333";
let base = argv[0] ?? null;
if (!webDir === !base) {
  console.error("usage: node tools/tint_check.mjs --web DIR [--cdp PORT]\n" +
                "       node tools/tint_check.mjs BASE_URL --expect FILE " +
                "[--cdp PORT]");
  process.exit(2);
}
if (base && !expectFile) {
  console.error("tint_check: a BASE_URL needs --expect FILE (the raymarch.wgsl " +
                "it must be serving); prefer --web DIR, which serves it " +
                "here on a fresh port.");
  process.exit(2);
}

let server = null;
if (webDir) {
  server = await serveTree(webDir);
  base = server.base;
  expectFile = join(server.root, "soar", "raymarch.wgsl");
}
let tab = null;
const done = async (code) => {
  await tab?.close();
  server?.close();
  process.exit(code);
};
const fail = async (err) => {
  console.error(`tint_check: ${err.message}`);
  await done(err.code ?? 2);
};

// The served shader must be the local one, byte for byte.
let shaderHash;
try {
  shaderHash = await servedShaderHash(base, expectFile);
  tab = await openTab(cdpPort);
  // A same-origin page to import the app's modules from; the app itself is
  // not booted.
  await tab.goto(`${base}/`);
  await requireHardwareAdapter(tab);
} catch (err) {
  await fail(err);
}

const expression = `(async () => {
  const base = ${JSON.stringify(base)};
  // A Chrome just launched while other sessions hold VRAM can answer the
  // first requests with no adapter and then settle (seen 2026-09-23), so a
  // few seconds of retries before calling it absent.
  let adapter = null;
  for (let i = 0; i < 6 && !adapter; i++) {
    if (i) await new Promise((r) => setTimeout(r, 1000));
    adapter = await navigator.gpu?.requestAdapter();
  }
  if (!adapter) return { fatal: "no WebGPU adapter in this Chrome (6 tries over 5 s)" };
  const device = await adapter.requestDevice();
  const R = await import(base + "/soar/renderer.js");
  const K = await import(base + "/soar/constants.js");
  const text = async (p) => {
    const r = await fetch(base + "/soar/" + p, { cache: "no-store" });
    if (!r.ok) throw new Error(p + ": HTTP " + r.status);
    return r.text();
  };
  const compile = async (label, code) => {
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label, code });
    const info = await module.getCompilationInfo();
    const scoped = await device.popErrorScope();
    const say = (m) => m.lineNum + ":" + m.linePos + " " + m.message;
    const errors = info.messages.filter((m) => m.type === "error").map(say);
    const warnings = info.messages.filter((m) => m.type === "warning")
      .map(say);
    if (scoped && !errors.length) errors.push(scoped.message);
    return { label, errors, warnings };
  };
  const raymarch = await text("raymarch.wgsl");
  // Every lighting recipe the app can reach: each tier's light-step cap,
  // crossed with the sun-tau cache and the sky probe on and off — the tiers
  // pick three of those four, and the hand toggles reach the fourth. A host
  // whose specializeShader takes fewer options ignores the rest, and the
  // identical sources that leaves are compiled once (below).
  const caps = [...new Set(Object.values(K.QUALITY_PRESETS)
    .map((p) => p.maxLightSteps ?? K.DEFAULT_MAX_LIGHT_STEPS))];
  const recipes = [];
  for (const maxLightSteps of caps) {
    for (const lightCache of [false, true]) {
      for (const skyProbe of [false, true]) {
        recipes.push({ maxLightSteps, lightCache, skyProbe });
      }
    }
  }
  const out = [];
  const done = new Set();
  // Two desert-only axes, which specializeShader refuses on any other
  // surface: the lite ground (DESERT_LITE, the low tiers') and the full
  // ground's flora cut (DESERT_FLORA_CUT, each value a tier asks for; the
  // lite ground ignores it). A tree from before either axis has no table,
  // and serves the desert at the defaults.
  const cuts = [...new Set(Object.entries(K.DESERT_FLORA_CUT_BY_TIER ?? {})
    .filter(([tier]) => !(K.DESERT_LITE_TIERS ?? []).includes(tier))
    .map(([, cut]) => cut).concat([1.0]))];
  const surfaces = [["ocean", false, 1.0], ["city", false, 1.0],
                    ...cuts.map((cut) => ["desert", false, cut]),
                    ["desert", true, 1.0]];
  for (const [surface, desertLite, desertFloraCut] of surfaces) {
    for (const nested of [false, true]) {
      for (const periodic of [false, true]) {
        for (const toneMap of [true, false]) {
          for (const recipe of recipes) {
            const label = [surface + (desertLite ? "-lite" : "")
                           + (desertFloraCut !== 1.0 ? "-cut" + desertFloraCut
                                                     : ""),
                           nested ? "nested" : "flat",
                           periodic ? "periodic" : "open",
                           toneMap ? "tonemap" : "linear",
                           "steps" + recipe.maxLightSteps
                           + (recipe.lightCache ? "+cache" : "")
                           + (recipe.skyProbe ? "+probe" : "")].join("/");
            let code;
            try {
              code = R.specializeShader(raymarch, {
                ...recipe, periodic, nested, toneMap,
                city: surface === "city", desert: surface === "desert",
                ...(desertLite ? { desertLite } : {}),
                ...(desertFloraCut !== 1.0 ? { desertFloraCut } : {}) });
            } catch (err) {
              out.push({ label, errors: ["specialize: " + err.message],
                         warnings: [] });
              continue;
            }
            if (done.has(code)) continue;
            done.add(code);
            out.push(await compile("raymarch " + label, code));
          }
        }
      }
    }
  }
  for (const f of ["bird.wgsl", "dart.wgsl", "hud.wgsl"]) {
    try { out.push(await compile(f, await text(f))); }
    catch (err) { out.push({ label: f, errors: [err.message], warnings: [] }); }
  }
  // The present pass, which carries lens water (lens.wgsl is not a module
  // on its own: the present shader includes it and provides lens_src).
  try {
    out.push(await compile("present + lens.wgsl",
                           R.presentShader(await text("lens.wgsl"))));
  } catch (err) {
    out.push({ label: "present + lens.wgsl", errors: [err.message],
               warnings: [] });
  }
  device.destroy();
  return { out, adapter: [adapter.info?.vendor, adapter.info?.architecture]
                           .filter(Boolean).join(" ") };
})()`;

let result;
try {
  result = await tab.eval(expression);
} catch (err) {
  await fail(err);
}
if (result.fatal) await fail(new Error(result.fatal));
let failed = 0;
const seen = new Set();
for (const m of result.out) {
  if (m.errors.length) {
    failed += 1;
    console.log(`FAIL ${m.label}`);
    for (const e of m.errors) console.log(`   ${e}`);
  }
  for (const w of m.warnings) {
    if (!seen.has(w)) { seen.add(w); console.log(`warn ${m.label}: ${w}`); }
  }
}
console.log(`tint_check: ${result.out.length} modules, ${failed} failed ` +
            `(raymarch.wgsl sha256 ${shaderHash}; ${result.adapter})`);
await done(failed ? 1 : 0);
