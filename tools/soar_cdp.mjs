// Drive soar in a real Chrome over the DevTools protocol, against a tree this
// process serves itself. Shared by tools/tint_check.mjs and
// tools/pinned_still.mjs. The Chrome is tools/qa_chrome.sh's: headless and
// cut off from the desktop, so a check never takes focus or appears over
// someone's apps (Thomas, mid-game, 2026-11-08), and requireHardwareAdapter
// refuses one that renders in software.
//
// serveTree(dir) serves a checkout's web/ directory on a port the kernel
// picks, so no other session's server can answer in its place, and answers
// "/" with a blank page for modules to be imported into. openTab(cdpPort)
// opens a tab of its own (the same Chrome may be driving other passes in
// other tabs), with the HTTP cache off: a served tree can move between
// commits, and a cached module mixed with fresh ones is a build that never
// existed.

import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { extname, join, resolve, sep } from "node:path";

const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript",
                ".wgsl": "text/plain", ".json": "application/json",
                ".html": "text/html", ".css": "text/css",
                ".webp": "image/webp", ".png": "image/png" };

export async function serveTree(webDir) {
  const root = resolve(webDir);
  const server = createServer(async (req, res) => {
    try {
      const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
      if (path === "/") {
        // A real 200 document: Chrome replaces an empty 404 with its own
        // error page, which has no WebGPU.
        res.writeHead(200, { "Content-Type": "text/html" });
        res.end("<!doctype html><title>soar</title>");
        return;
      }
      let file = resolve(join(root, path));
      if (file !== root && !file.startsWith(root + sep)) throw new Error("outside");
      if (path.endsWith("/")) file = join(file, "index.html");
      // Symlinks are followed: web/soar/raymarch.wgsl is one, into the
      // package, and web/soar/demos usually is too.
      const body = await readFile(await realpath(file));
      res.writeHead(200, { "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end();
    }
  });
  await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
  return { base: `http://127.0.0.1:${server.address().port}`, root,
           close: () => server.close() };
}

export const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");

/** The served raymarch.wgsl, checked byte for byte against `expectFile`. */
export async function servedShaderHash(base, expectFile) {
  const local = await readFile(await realpath(expectFile));
  const r = await fetch(`${base}/soar/raymarch.wgsl`, { cache: "no-store" });
  if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status}`), { code: 2 });
  const served = Buffer.from(await r.arrayBuffer());
  if (sha256(served) !== sha256(local)) {
    throw Object.assign(new Error(
      `${base} serves a raymarch.wgsl that is NOT ${expectFile} (served ` +
      `sha256 ${sha256(served).slice(0, 16)}, local ` +
      `${sha256(local).slice(0, 16)}). Another tree is being served there.`),
      { code: 3 });
  }
  return sha256(served).slice(0, 16);
}

// A software rasterizer passes for WebGPU (Chrome falls back to SwiftShader
// when the Vulkan flags are missing or wrong, and Mesa's llvmpipe/lavapipe
// answer as a "GPU" too), and every hash and timing taken on one is fiction
// against the hardware stills. `info` is a GPUAdapterInfo-like object plus
// the adapter's isFallbackAdapter; the reason it is software, or null.
export function softwareAdapter(info) {
  if (!info) return "no WebGPU adapter";
  if (info.isFallbackAdapter) return "the fallback adapter";
  const id = [info.vendor, info.architecture, info.description]
    .filter(Boolean).join(" ");
  const sw = /swiftshader|llvmpipe|lavapipe|softpipe/i.exec(id);
  return sw ? `a software rasterizer (${id})` : null;
}

/** The adapter a page of `tab` gets, as "vendor architecture"; throws (code
 *  2, "cannot run") when there is none or it is software (softwareAdapter).
 *  The page must be a secure context — a served tree, or localhost. */
export async function requireHardwareAdapter(tab) {
  const info = await tab.eval(`(async () => {
    // A Chrome just launched while other sessions hold VRAM can answer the
    // first requests with no adapter and then settle (seen 2026-09-23).
    let a = null;
    for (let i = 0; i < 6 && !a; i++) {
      if (i) await new Promise((r) => setTimeout(r, 1000));
      a = await navigator.gpu?.requestAdapter();
    }
    if (!a) return null;
    const i = a.info ?? {};
    return { vendor: i.vendor, architecture: i.architecture,
             description: i.description,
             isFallbackAdapter: a.isFallbackAdapter ?? i.isFallbackAdapter };
  })()`);
  const why = softwareAdapter(info);
  if (why) {
    throw Object.assign(new Error(
      `this Chrome gives ${why}, not the hardware GPU; launch it with ` +
      "tools/qa_chrome.sh (or --enable-features=Vulkan and a working " +
      "ANGLE/ozone setup)"), { code: 2 });
  }
  return [info.vendor, info.architecture].filter(Boolean).join(" ");
}

/** What the page threw, from a Runtime.evaluate result's exceptionDetails. */
function thrownMessage(details) {
  return details.exception?.description ?? details.text;
}

/**
 * Poll `evaluate(expression)` — one CDP Runtime.evaluate, resolving to its
 * result or rejecting on a protocol error — until it gives a truthy value,
 * and return that value.
 *
 * Three answers, kept apart:
 *   - a truthy value: done;
 *   - a falsy value, or a protocol error (no execution context: the page is
 *     navigating, its context was destroyed, it is not up yet): not ready,
 *     poll again;
 *   - the expression THREW (exceptionDetails): a failure the expression is
 *     reporting, so rethrown at once with the page's message.
 * Until 2026-11-09 the poll swallowed all three alike (`.catch(() =>
 * undefined)`), so a failure check inside a waited expression — pinned_
 * still's `if (L.classList.contains("failed") ...) throw new Error(L.
 * innerText)` — could never fire: ?surfacecs=2 showed its failure panel and
 * the tool timed out 181 s later with the reason lost. A waited expression
 * must therefore not throw while merely not ready (use ?. on what may not
 * exist yet); every one in tools/ is written that way.
 */
export async function pollUntil(evaluate, expression, timeoutMs = 60000,
                                intervalMs = 250) {
  const t0 = Date.now();
  let last = null;
  for (;;) {
    let r = null;
    try {
      r = await evaluate(expression);
    } catch (err) {
      last = err;
    }
    if (r?.exceptionDetails) {
      const short = expression.length > 120
        ? `${expression.slice(0, 117)}...` : expression;
      throw new Error(`the page threw while waiting for ${short}: ` +
                      thrownMessage(r.exceptionDetails));
    }
    const v = r?.result?.value;
    if (v) return v;
    if (Date.now() - t0 > timeoutMs) {
      throw new Error(`timed out waiting for ${expression}` +
                      (last ? ` (last protocol error: ${last.message})` : ""));
    }
    await new Promise((ok) => setTimeout(ok, intervalMs));
  }
}

export async function openTab(cdpPort) {
  let t;
  try {
    t = await (await fetch(`http://127.0.0.1:${cdpPort}/json/new?about:blank`,
                           { method: "PUT" })).json();
  } catch {
    throw Object.assign(
      new Error(`no Chrome DevTools endpoint on port ${cdpPort}`), { code: 2 });
  }
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
  let id = 0;
  const waiting = new Map();
  const listeners = [];
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.id && waiting.has(m.id)) {
      const { ok, bad } = waiting.get(m.id);
      waiting.delete(m.id);
      m.error ? bad(new Error(JSON.stringify(m.error))) : ok(m.result);
    } else if (m.method) {
      for (const f of listeners) f(m);
    }
  };
  const send = (method, params = {}) => new Promise((ok, bad) => {
    waiting.set(++id, { ok, bad });
    ws.send(JSON.stringify({ id, method, params }));
  });
  await send("Page.enable");
  await send("Network.enable");
  await send("Network.setCacheDisabled", { cacheDisabled: true });
  const evaluate = (expression) => send("Runtime.evaluate",
    { expression, awaitPromise: true, returnByValue: true, userGesture: true });
  const tab = {
    send,
    async eval(expression) {
      const r = await evaluate(expression);
      if (r.exceptionDetails) throw new Error(thrownMessage(r.exceptionDetails));
      return r.result.value;
    },
    async goto(url) {
      const loaded = new Promise((ok) => listeners.push((m) => {
        if (m.method === "Page.loadEventFired") ok();
      }));
      await send("Page.navigate", { url });
      await loaded;
    },
    /** pollUntil on this tab: a throw in `expression` fails at once. */
    waitFor(expression, timeoutMs = 60000) {
      return pollUntil(evaluate, expression, timeoutMs);
    },
    async close() {
      for (const s of SIGNALS) process.off(s, onSignal);
      ws.close();
      await fetch(`http://127.0.0.1:${cdpPort}/json/close/${t.id}`).catch(() => {});
    },
  };
  // A tool killed mid-run (a timeout, ^C) must not leave its soar tab
  // animating in the shared Chrome: an orphan once spoiled every timing
  // taken outside the GPU lock (2026-09-24).
  const SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"];
  const onSignal = async (s) => {
    await tab.close();
    process.exit(128 + { SIGHUP: 1, SIGINT: 2, SIGTERM: 15 }[s]);
  };
  for (const s of SIGNALS) process.once(s, onSignal);
  return tab;
}
