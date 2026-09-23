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
//   node tools/tint_check.mjs BASE_URL [CDP_PORT]
//
// BASE_URL is a static server over a checkout's web/ directory (the page that
// is driven imports soar/renderer.js from it, so the specialization is the
// app's own); CDP_PORT (default 9333) is a Chrome started with
// --remote-debugging-port and a hardware WebGPU adapter (on the Fedora box:
// --enable-features=Vulkan --ozone-platform=x11). Compile only — nothing is
// drawn. Exit 0 clean, 1 on any compile error, 2 when it cannot run at all.

const [base, cdpPort = "9333"] = process.argv.slice(2);
if (!base) {
  console.error("usage: node tools/tint_check.mjs BASE_URL [CDP_PORT]");
  process.exit(2);
}

async function page(port) {
  let targets;
  try {
    targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  } catch (err) {
    console.error(`tint_check: no Chrome DevTools endpoint on port ${port}`);
    process.exit(2);
  }
  // A tab of its own, closed at the end: the same Chrome may be driving a
  // QA pass in its other tabs, and navigating theirs would break it.
  void targets;
  const t = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`,
                               { method: "PUT" })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
  let id = 0;
  const waiting = new Map();
  const events = [];
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && waiting.has(m.id)) {
      const { ok, bad } = waiting.get(m.id);
      waiting.delete(m.id);
      m.error ? bad(new Error(JSON.stringify(m.error))) : ok(m.result);
    } else if (m.method) {
      for (const f of events) f(m);
    }
  };
  const send = (method, params = {}) => new Promise((ok, bad) => {
    waiting.set(++id, { ok, bad });
    ws.send(JSON.stringify({ id, method, params }));
  });
  await send("Page.enable");
  const close = () => fetch(`http://127.0.0.1:${port}/json/close/${t.id}`)
    .catch(() => {});
  return { ws, send, events, close };
}

const { ws, send, events, close } = await page(cdpPort);
// A same-origin page to import the app's modules from. The directory listing
// is enough; the app itself is not booted.
const loaded = new Promise((ok) => events.push((m) => {
  if (m.method === "Page.loadEventFired") ok();
}));
await send("Page.navigate", { url: `${base}/` });
await loaded;

const expression = `(async () => {
  const base = ${JSON.stringify(base)};
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { fatal: "no WebGPU adapter in this Chrome" };
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
  // The lighting recipes the quality tiers can ask for, deduplicated. A host
  // whose specializeShader takes fewer options ignores the rest, and the
  // identical sources that leaves are compiled once.
  const recipes = [];
  for (const p of Object.values(K.QUALITY_PRESETS)) {
    const r = { maxLightSteps: p.maxLightSteps ?? K.DEFAULT_MAX_LIGHT_STEPS,
                lightCache: Boolean(p.lightCache),
                skyProbe: Boolean(p.skyProbe) };
    if (!recipes.some((q) => JSON.stringify(q) === JSON.stringify(r))) {
      recipes.push(r);
    }
  }
  const out = [];
  const done = new Set();
  for (const surface of ["ocean", "city", "desert"]) {
    for (const nested of [false, true]) {
      for (const periodic of [false, true]) {
        for (const toneMap of [true, false]) {
          for (const recipe of recipes) {
            const label = [surface, nested ? "nested" : "flat",
                           periodic ? "periodic" : "open",
                           toneMap ? "tonemap" : "linear",
                           "steps" + recipe.maxLightSteps
                           + (recipe.lightCache ? "+cache" : "")
                           + (recipe.skyProbe ? "+probe" : "")].join("/");
            let code;
            try {
              code = R.specializeShader(raymarch, {
                ...recipe, periodic, nested, toneMap,
                city: surface === "city", desert: surface === "desert" });
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
  device.destroy();
  return { out, adapter: [adapter.info?.vendor, adapter.info?.architecture]
                           .filter(Boolean).join(" ") };
})()`;

const r = await send("Runtime.evaluate",
                     { expression, awaitPromise: true, returnByValue: true });
ws.close();
await close();
if (r.exceptionDetails) {
  console.error("tint_check: " + (r.exceptionDetails.exception?.description
                                  ?? r.exceptionDetails.text));
  process.exit(2);
}
const result = r.result.value;
if (result.fatal) {
  console.error(`tint_check: ${result.fatal}`);
  process.exit(2);
}
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
            `(${base}; ${result.adapter})`);
process.exit(failed ? 1 : 0);
