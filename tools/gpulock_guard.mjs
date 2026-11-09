// Refuse to open a GPU device outside tools/gpulock: the .mjs tools' copy of
// tools/gpulock_guard.py, which says why and what "under the lock" means
// (the SOAR_GPULOCK marker tools/gpulock exports, AND the lock held by an
// ancestor of this process, per /proc/locks).
// Call requireGpulock() before the tool opens a tab or requests an adapter.

import { readFileSync, statSync } from "node:fs";

export const LOCK = "/tmp/soar-gpu.lock";
export const EXIT_UNLOCKED = 3;   // not a tool's own "cannot run" (2)

/** The pids holding a FLOCK on `path` now (blocked waiters excluded). */
export function lockHolders(path = LOCK) {
  let st;
  try { st = statSync(path, { bigint: true }); } catch { return new Set(); }
  // glibc's major()/minor() of a Linux dev_t.
  const d = st.dev;
  const major = Number(((d >> 8n) & 0xfffn) | ((d >> 32n) & ~0xfffn));
  const minor = Number((d & 0xffn) | ((d >> 12n) & ~0xffn));
  const hex = (n) => n.toString(16).padStart(2, "0");
  const key = `${hex(major)}:${hex(minor)}:${st.ino}`;
  const pids = new Set();
  for (const line of readFileSync("/proc/locks", "utf8").split("\n")) {
    // "N: FLOCK ADVISORY WRITE pid maj:min:ino start end"; a waiter's line
    // carries "->" after the index.
    const f = line.trim().split(/\s+/);
    if (f.length >= 6 && f[1] === "FLOCK" && f[5] === key) pids.add(Number(f[4]));
  }
  return pids;
}

/** This process and its ancestors, nearest first. */
export function lineage(pid = process.pid) {
  const chain = [];
  while (pid > 0 && !chain.includes(pid)) {
    chain.push(pid);
    let status;
    try { status = readFileSync(`/proc/${pid}/status`, "utf8"); } catch { break; }
    const m = status.match(/^PPid:\s+(\d+)/m);
    if (!m) break;
    pid = Number(m[1]);
  }
  return chain;
}

/** Why this process is not under tools/gpulock, or null when it is. */
export function unlockedReason(env = process.env, path = LOCK) {
  if (env.SOAR_GPULOCK_EXEMPT === "pytest") return null;
  if (env.SOAR_GPULOCK !== path) return "SOAR_GPULOCK is not set";
  const holders = lockHolders(path);
  if (holders.size === 0) return `SOAR_GPULOCK is set but ${path} is not held (a stale export?)`;
  const mine = lineage();
  if (![...holders].some((p) => mine.includes(p))) {
    return `SOAR_GPULOCK is set but ${path} is held by pid ${Math.min(...holders)}, ` +
           "which is not this command's tools/gpulock";
  }
  return null;
}

/** Exit EXIT_UNLOCKED, before any device is opened, unless locked. */
export function requireGpulock(tool, argv = process.argv.slice(2)) {
  if (argv.includes("-h") || argv.includes("--help")) return;  // usage opens nothing
  const why = unlockedReason();
  if (why === null) return;
  console.error(`${tool}: run this under tools/gpulock: it opens a GPU device ` +
                `(${why}). For example:\n    tools/gpulock node ${tool} ...`);
  process.exit(EXIT_UNLOCKED);
}
