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

import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { extname, join, resolve, sep } from "node:path";

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

const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript",
                ".wgsl": "text/plain", ".json": "application/json",
                ".html": "text/html", ".css": "text/css" };
let server = null;
if (webDir) {
  const root = resolve(webDir);
  server = createServer(async (req, res) => {
    try {
      const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
      if (path === "/") {
        // The page the modules are imported into. It must be a real 200
        // document: Chrome replaces an empty 404 with its own error page,
        // which has no WebGPU.
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end("<!doctype html><title>tint_check</title>");
        return;
      }
      const file = resolve(join(root, path));
      if (file !== root && !file.startsWith(root + sep)) throw new Error("outside");
      // Symlinks are followed: web/soar/raymarch.wgsl is one, into the
      // package, and that is the file the app really serves.
      const body = await readFile(await realpath(file));
      res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "text/html" });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  base = `http://127.0.0.1:${server.address().port}`;
  expectFile = join(root, "soar", "raymarch.wgsl");
}
const done = (code) => { server?.close(); process.exit(code); };

// The served shader must be the local one, byte for byte.
const sha = (buf) => createHash("sha256").update(buf).digest("hex");
const local = await readFile(await realpath(expectFile));
let served;
try {
  const r = await fetch(`${base}/soar/raymarch.wgsl`, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  served = Buffer.from(await r.arrayBuffer());
} catch (err) {
  console.error(`tint_check: could not fetch ${base}/soar/raymarch.wgsl ` +
                `(${err.message})`);
  done(2);
}
if (sha(served) !== sha(local)) {
  console.error(`tint_check: ${base} serves a raymarch.wgsl that is NOT ` +
                `${expectFile} (served sha256 ${sha(served).slice(0, 16)}, ` +
                `local ${sha(local).slice(0, 16)}). Another tree is being ` +
                "served on that port; nothing was compiled.");
  done(3);
}
const shaderHash = sha(served).slice(0, 16);

async function page(port) {
  let targets;
  try {
    targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  } catch (err) {
    console.error(`tint_check: no Chrome DevTools endpoint on port ${port}`);
    done(2);
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
  done(2);
}
const result = r.result.value;
if (result.fatal) {
  console.error(`tint_check: ${result.fatal}`);
  done(2);
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
            `(raymarch.wgsl sha256 ${shaderHash}; ${result.adapter})`);
done(failed ? 1 : 0);
