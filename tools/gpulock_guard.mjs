// Refuse to open a GPU device outside tools/gpulock: the .mjs tools' copy of
// tools/gpulock_guard.py, which says why and what "under the lock" means
// (the SOAR_GPULOCK marker tools/gpulock exports, AND SOAR_GPULOCK_PID an
// ancestor of this process that is a running tools/gpulock), read from the
// same `ps` table on Linux and macOS alike.
// Call requireGpulock() before the tool opens a tab or requests an adapter.

import { spawnSync } from "node:child_process";
import { basename } from "node:path";

export const LOCK = "/tmp/soar-gpu.lock";
export const EXIT_UNLOCKED = 3;   // not a tool's own "cannot run" (2)

/** Map pid -> [ppid, command] for every process, from ps. */
export function processTable() {
  const r = spawnSync("ps", ["-A", "-ww", "-o", "pid=,ppid=,command="],
                      { encoding: "utf8", maxBuffer: 64 << 20 });
  if (r.error || r.status !== 0) {
    throw new Error(`gpulock_guard: ps failed: ${r.error?.message ?? r.stderr}`);
  }
  return parsePs(r.stdout);
}

/** `ps -o pid=,ppid=,command=` output -> Map pid -> [ppid, command]. */
export function parsePs(out) {
  const table = new Map();
  for (const line of out.split("\n")) {
    const m = line.match(/^\s*(\d+)\s+(\d+)\s?(.*)$/);
    if (m) table.set(Number(m[1]), [Number(m[2]), m[3]]);
  }
  return table;
}

/** `pid` (this process by default) and its ancestors, nearest first. */
export function lineage(table, pid = process.pid) {
  const chain = [];
  while (pid > 0 && !chain.includes(pid)) {
    chain.push(pid);
    if (!table.has(pid)) break;
    pid = table.get(pid)[0];
  }
  return chain;
}

/** Whether a ps command line is tools/gpulock running. */
export function isLauncher(command) {
  return command.split(/\s+/).slice(0, 3).some((w) => basename(w) === "gpulock");
}

/** Why this process is not under tools/gpulock, or null when it is. */
export function unlockedReason(env = process.env, path = LOCK, table = null) {
  if (env.SOAR_GPULOCK_EXEMPT === "pytest") return null;
  if (env.SOAR_GPULOCK !== path) return "SOAR_GPULOCK is not set";
  if (!/^\d+$/.test(env.SOAR_GPULOCK_PID ?? "")) {
    return "SOAR_GPULOCK_PID is not set (a tools/gpulock from before 2026-11-09?)";
  }
  const pid = Number(env.SOAR_GPULOCK_PID);
  table = table ?? processTable();
  if (!lineage(table).includes(pid)) {
    return `SOAR_GPULOCK_PID ${pid} is not an ancestor of this process (a stale export?)`;
  }
  if (!isLauncher((table.get(pid) ?? [0, ""])[1])) {
    return `SOAR_GPULOCK_PID ${pid} is not tools/gpulock`;
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
