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

// The sweep runs as ONE evaluate per module (with its cs_surface pipelines),
// not one evaluate for all of it: each has its own deadline, a slow or hung
// compile fails naming its module, and the progress (and each module's time)
// goes to stderr. As one evaluate the whole sweep (68 modules and the
// pipelines, ~39 s on the 5080) could pass soar_cdp's 10-minute eval
// deadline under Metal on the M1 with no shader at fault (loose-ends, review
// of c17d79b). The setup evaluate below builds the plan and parks run(i) on
// the page; the order, the de-duplication and the once-only 16x16 pipeline
// are the single sweep's, so the results are the same.
const SETUP_DEADLINE_MS = 10 * 60 * 1000;   // adapter, imports, specializing
const JOB_DEADLINE_MS = 10 * 60 * 1000;     // one module and its pipelines
const setup = `(async () => {
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
  // The adapter's compute limits, so the city's cs_surface (the ?surfacecs
  // experiment) can be built at its 512-invocation workgroup below.
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: {
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: L.maxComputeWorkgroupSizeY } });
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
    return { label, errors, warnings, module };
  };
  // cs_surface as the hosts build it (renderer.js _surfaceComputePipeline):
  // Tint validates every entry when the module is made, but its backend,
  // the workgroup overrides and the device limits meet the entry only at
  // pipeline creation, which no other check here does for it.
  const csPipeline = async (label, module, [wx, wy]) => {
    const errors = [];
    if (wx * wy > device.limits.maxComputeInvocationsPerWorkgroup) {
      errors.push("this adapter allows " +
                  device.limits.maxComputeInvocationsPerWorkgroup +
                  " invocations, not " + (wx * wy));
    } else {
      try {
        await device.createComputePipelineAsync({
          label, layout: "auto",
          compute: { module, entryPoint: "cs_surface",
                     constants: { SURFACE_CS_WX: wx, SURFACE_CS_WY: wy } } });
      } catch (err) {
        errors.push(String(err?.message || err));
      }
    }
    return { label, errors, warnings: [], pipeline: true };
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
  const jobs = [];
  const seen = new Set();
  for (const surface of ["ocean", "city"]) {
    for (const nested of [false, true]) {
      for (const periodic of [false, true]) {
        for (const toneMap of [true, false]) {
          for (const recipe of recipes) {
            const label = [surface,
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
                city: surface === "city" });
            } catch (err) {
              jobs.push({ label, error: "specialize: " + err.message });
              continue;
            }
            if (seen.has(code)) continue;
            seen.add(code);
            jobs.push({ label: "raymarch " + label, tail: label, code,
                        city: surface === "city" });
          }
        }
      }
    }
  }
  for (const f of ["bird.wgsl", "dart.wgsl", "hud.wgsl"]) {
    jobs.push({ label: f, file: f });
  }
  // The present pass, which carries lens water (lens.wgsl is not a module
  // on its own: the present shader includes it and provides lens_src).
  jobs.push({ label: "present + lens.wgsl", present: true });
  let csSmall = false;     // the 16x16 workgroup, built once
  const run = async (i) => {
    const j = jobs[i];
    if (j.error) return [{ label: j.label, errors: [j.error], warnings: [] }];
    let code = j.code;
    try {
      if (j.file) code = await text(j.file);
      if (j.present) code = R.presentShader(await text("lens.wgsl"));
    } catch (err) {
      return [{ label: j.label, errors: [err.message], warnings: [] }];
    }
    const m = await compile(j.label, code);
    const res = [m];
    if (j.city && !m.errors.length) {
      res.push(await csPipeline("cs_surface 32x16 " + j.tail, m.module,
                                [32, 16]));
      if (!csSmall) {
        csSmall = true;
        res.push(await csPipeline("cs_surface 16x16 " + j.tail, m.module,
                                  [16, 16]));
      }
    }
    for (const r of res) delete r.module;
    return res;
  };
  window.__tint = { run, device };
  return { jobs: jobs.map((j) => j.label),
           adapter: [adapter.info?.vendor, adapter.info?.architecture]
                      .filter(Boolean).join(" ") };
})()`;

let plan;
try {
  plan = await tab.eval(setup, SETUP_DEADLINE_MS);
} catch (err) {
  await fail(err);
}
if (plan.fatal) await fail(new Error(plan.fatal));
const out = [];
for (let i = 0; i < plan.jobs.length; i++) {
  const t0 = Date.now();
  try {
    out.push(...await tab.eval(`window.__tint.run(${i})`, JOB_DEADLINE_MS));
  } catch (err) {
    await fail(new Error(`${plan.jobs[i]}: ${err.message}`));
  }
  console.error(`tint_check: [${i + 1}/${plan.jobs.length}] ${plan.jobs[i]} ` +
                `${Date.now() - t0} ms`);
}
await tab.eval("(window.__tint.device.destroy(), true)").catch(() => {});
const result = { out, adapter: plan.adapter };
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
const pipelines = result.out.filter((m) => m.pipeline).length;
console.log(`tint_check: ${result.out.length - pipelines} modules and ` +
            `${pipelines} cs_surface pipelines, ${failed} failed ` +
            `(raymarch.wgsl sha256 ${shaderHash}; ${result.adapter})`);
await done(failed ? 1 : 0);
