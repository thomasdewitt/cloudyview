// Drive soar in a real Chrome over the DevTools protocol, against a tree this
// process serves itself. Shared by tools/tint_check.mjs and
// tools/pinned_still.mjs.
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
  const tab = {
    send,
    async eval(expression) {
      const r = await send("Runtime.evaluate",
                           { expression, awaitPromise: true,
                             returnByValue: true, userGesture: true });
      if (r.exceptionDetails) {
        throw new Error(r.exceptionDetails.exception?.description
                        ?? r.exceptionDetails.text);
      }
      return r.result.value;
    },
    async goto(url) {
      const loaded = new Promise((ok) => listeners.push((m) => {
        if (m.method === "Page.loadEventFired") ok();
      }));
      await send("Page.navigate", { url });
      await loaded;
    },
    async waitFor(expression, timeoutMs = 60000) {
      const t0 = Date.now();
      for (;;) {
        const v = await tab.eval(expression).catch(() => undefined);
        if (v) return v;
        if (Date.now() - t0 > timeoutMs) {
          throw new Error(`timed out waiting for ${expression}`);
        }
        await new Promise((r) => setTimeout(r, 250));
      }
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
