// Compile every shader soar ships with Chrome's own WGSL compiler (Tint).
//
// The Python host and every pytest go through naga, which accepts some WGSL
// the spec does not (`a ^ b * c` once shipped that way, and no browser could
// build the shader while the suite stayed green). This serves a checkout's
// web/ directory, opens a tab in a Chrome with a DevTools port, and creates
// each shader module there: every surface's module under every combination
// of the compile-time switches, the overlay shaders, and the present pass.
//
//   node tools/tint_check.mjs [--web DIR] [--cdp PORT]
//
// DIR defaults to this checkout's web/. PORT (default 9333) is a Chrome
// started with --remote-debugging-port and a hardware WebGPU adapter, e.g.
//
//   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
//       --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/soar-cdp
//
// Prints each failure with Tint's message, then a one-line summary.
// Exit 0 when everything compiles, 1 when something does not, 2 when the
// check could not run.

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { openTab, requireHardwareAdapter, serveTree } from "./soar_cdp.mjs";

const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i < 0 ? fallback : argv[i + 1];
};
const here = dirname(fileURLToPath(import.meta.url));
const webDir = option("--web", join(here, "..", "web"));
const cdpPort = option("--cdp", "9333");

const server = await serveTree(webDir);
let tab = null;
const done = async (code) => {
  await tab?.close();
  server.close();
  process.exit(code);
};
const fail = async (err) => {
  console.error(`tint_check: ${err.message}`);
  await done(err.code ?? 2);
};

let adapter;
try {
  tab = await openTab(cdpPort);
  // A same-origin page to import the app's modules from; the app itself is
  // not booted.
  await tab.goto(`${server.base}/`);
  adapter = await requireHardwareAdapter(tab);
} catch (err) {
  await fail(err);
}

// One evaluate builds the list of modules and parks run(i) on the page; each
// module then compiles in an evaluate of its own, so a slow or hung compile
// fails naming its module.
const SETUP_DEADLINE_MS = 5 * 60 * 1000;
const JOB_DEADLINE_MS = 5 * 60 * 1000;
const setup = `(async () => {
  const base = ${JSON.stringify(server.base)} + "/soar/";
  const device = await (await navigator.gpu.requestAdapter()).requestDevice();
  const S = await import(base + "shader.js");
  const R = await import(base + "renderer.js");
  const text = async (p) => {
    const r = await fetch(base + p, { cache: "no-store" });
    if (!r.ok) throw new Error(p + ": HTTP " + r.status);
    return r.text();
  };
  const loaded = await S.loadShaders(new URL("shader/", base));
  const shaders = loaded.modules;
  const jobs = [];
  const onOff = [false, true];
  for (const surface of Object.keys(shaders)) {
    for (const nested of onOff) for (const periodic of onOff)
    for (const toneMap of onOff) for (const lightCache of onOff)
    for (const skyProbe of onOff) {
      const spec = { periodic, nested, toneMap, lightCache, skyProbe };
      jobs.push({
        label: surface + " " + Object.entries(spec)
          .filter(([, on]) => on).map(([k]) => k).join("+"),
        code: () => S.specializeShader(shaders[surface], spec) });
    }
  }
  for (const f of ["bird.wgsl", "dart.wgsl", "hud.wgsl"]) {
    jobs.push({ label: f, code: () => text(f) });
  }
  // lens.wgsl is not a module on its own: the present shader includes it.
  jobs.push({ label: "present + shader/lens.wgsl",
              code: () => R.presentShader(loaded.lens) });
  const run = async (i) => {
    const { label, code } = jobs[i];
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label, code: await code() });
    const info = await module.getCompilationInfo();
    const scoped = await device.popErrorScope();
    const errors = info.messages.filter((m) => m.type === "error")
      .map((m) => m.lineNum + ":" + m.linePos + " " + m.message);
    if (scoped && !errors.length) errors.push(scoped.message);
    return errors;
  };
  window.__tint = { run, device };
  return jobs.map((j) => j.label);
})()`;

let labels;
try {
  labels = await tab.eval(setup, SETUP_DEADLINE_MS);
} catch (err) {
  await fail(err);
}
let failed = 0;
const t0 = Date.now();
for (let i = 0; i < labels.length; i++) {
  let errors;
  try {
    errors = await tab.eval(`window.__tint.run(${i})`, JOB_DEADLINE_MS);
  } catch (err) {
    await fail(new Error(`${labels[i]}: ${err.message}`));
  }
  if (errors.length) {
    failed += 1;
    console.log(`FAIL ${labels[i]}`);
    for (const e of errors) console.log(`   ${e}`);
  }
}
await tab.eval("(window.__tint.device.destroy(), true)").catch(() => {});
console.log(`tint_check: ${labels.length} modules, ${failed} failed ` +
            `(${adapter}; ${((Date.now() - t0) / 1000).toFixed(0)} s)`);
await done(failed ? 1 : 0);
