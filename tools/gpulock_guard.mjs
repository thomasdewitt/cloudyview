// Refuse to open a GPU device outside tools/gpulock: the .mjs tools' copy of
// tools/gpulock_guard.py, which says why and what "under the lock" means
// (the SOAR_GPULOCK marker tools/gpulock exports, AND the lock held now).
// Call requireGpulock() before the tool opens a tab or requests an adapter.

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

export const LOCK = "/tmp/soar-gpu.lock";
export const EXIT_UNLOCKED = 3;   // not a tool's own "cannot run" (2)

/** Whether someone holds the GPU lock now (a non-blocking flock fails). */
export function lockHeld(path = LOCK) {
  if (!existsSync(path)) return false;
  const r = spawnSync("flock", ["-n", path, "true"]);
  if (r.error) throw new Error(`gpulock_guard: cannot run flock: ${r.error.message}`);
  return r.status !== 0;
}

/** Why this process is not under tools/gpulock, or null when it is. */
export function unlockedReason(env = process.env, path = LOCK) {
  if (env.SOAR_GPULOCK_EXEMPT === "pytest") return null;
  if (env.SOAR_GPULOCK !== path) return "SOAR_GPULOCK is not set";
  if (!lockHeld(path)) return `SOAR_GPULOCK is set but ${path} is not held (a stale export?)`;
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
