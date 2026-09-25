#!/usr/bin/env python
"""Refuse to time anything on a GPU that is doing someone else's work.

There is one GPU on this box and many sessions. tools/gpulock serializes the
ones that take it, but a Chrome tab left flying, a pass run outside the lock
or a python process still holding a device skews every timestamp without
leaving a trace in the numbers. On 2026-09-24 two measurements of the same
change disagreed by ~60 points and nobody could say which was contaminated.
Timing tools call require_idle() before they create their own device, and
check_quiet() when they finish.

Two kinds of finding, over a few seconds of nvidia-smi samples (the fleet's
policy, 2026-09-24):

- FATAL, a neighbour: any process outside the desktop (DESKTOP) above
  IDLE_SM_PCT of SM time, or holding a GPU context at all (Chrome excepted:
  the QA harness parks an idle browser, and if it draws, the SM test
  catches it). require_idle() waits for these to go, then exits;
  check_quiet() exits non-zero with CONTAMINATED — discard the numbers.
- WARNING, the desktop drawing: gnome-shell or the Claude app above
  DESKTOP_SM_PCT, or device utilization pmon attributes to nobody (the
  compositor under the clock lock). This is Thomas's own screen and cannot
  be switched off; under the 180 MHz lock an ordinary redraw takes a third
  of the card. Interleaving spreads it over every variant alike, so the
  RATIOS stand when the null control holds within ~1% and the rounds agree;
  the absolute ms under the lock are pessimistic. Tools report it, with
  that verdict, rather than failing.

    uv run python tools/benchmarking/gpu_idle.py      # exit 0 idle, 1 not
"""

import os
import subprocess
import sys
import time

IDLE_UTIL_PCT = 3
# Under the M1-proxy clock lock (180 MHz) the compositor's few frames take
# ten times as long, and the device reads 6-7% busy with nothing but the
# desktop on it, below what pmon attributes to any process.
IDLE_UTIL_PCT_LOCKED = 10
IDLE_SM_PCT = 2
DESKTOP_SM_PCT = 15
# Processes that always hold a context on this desktop, the Claude app
# (claude-desktop) among them.
DESKTOP = {"gnome-shell", "Xwayland", "gnome-software", "ptyxis",
           "gnome-remote-de", "mutter-x11-fram", "claude-desktop"}


def _smi(*args) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True,
                          text=True, check=True).stdout


def _processes():
    """(pid, name, sm%) for every process with a context, one pmon sample.
    sm% is None when pmon reports '-' (no activity in the sample)."""
    rows = []
    for line in _smi("pmon", "-c", "1", "-s", "u").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        sm = None if f[3] == "-" else int(f[3])
        rows.append((int(f[1]), f[-1], sm))
    return rows


def _full_name(pid: int, short: str) -> str:
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return short


def clock_mode() -> str:
    """The clock mode the caller declared (tools that lock the clocks set
    GPU_CLOCK_MODE), for the header of every table."""
    return os.environ.get("GPU_CLOCK_MODE",
                          "stock (no clock lock declared by the caller)")


def survey(seconds: float = 3.0, exclude=()) -> tuple[list[str], list[str]]:
    """(fatal, warnings) sampled over `seconds`; both empty when idle.
    `exclude`: pids to ignore besides this process (which is always
    ignored: the caller's own tail is not a neighbour)."""
    exclude = set(exclude) | {os.getpid()}
    fatal, warn = set(), set()
    limit = (IDLE_UTIL_PCT_LOCKED if clock_mode().startswith("locked")
             else IDLE_UTIL_PCT)
    t_end = time.monotonic() + seconds
    while True:
        util = int(_smi("--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits").strip())
        known = 0
        for pid, short, sm in _processes():
            known += sm or 0
            if pid in exclude:
                continue
            name = _full_name(pid, short)
            if name in DESKTOP:
                if (sm or 0) > DESKTOP_SM_PCT:
                    warn.add(f"desktop: pid {pid} ({name}) drawing at "
                             f"{sm}% SM")
                continue
            if sm is not None and sm > IDLE_SM_PCT:
                fatal.add(f"pid {pid} ({name}) at {sm}% SM")
            if "chrome" not in name:
                fatal.add(f"pid {pid} ({name}) holds a GPU context")
        if util - known > limit:
            warn.add(f"device {util}% busy, {util - known}% of it "
                     f"attributed to no process (> {limit}%)")
        if time.monotonic() >= t_end:
            return sorted(fatal), sorted(warn)
        time.sleep(0.5)


def require_idle(wait_s: float = 300.0) -> list[str]:
    """Block until no neighbour is on the card, up to `wait_s`, then exit
    loudly. Returns the desktop warnings seen at the start."""
    t_end = time.monotonic() + wait_s
    while True:
        fatal, warn = survey()
        if not fatal:
            return warn
        if time.monotonic() >= t_end:
            sys.exit("GPU is not idle, refusing to time:\n  "
                     + "\n  ".join(fatal))
        print("waiting for an idle GPU: " + "; ".join(fatal), file=sys.stderr,
              flush=True)
        time.sleep(10)


def check_quiet() -> list[str]:
    """After a run (own device still open): exit loudly if a neighbour is on
    the card, which may have skewed the numbers just printed. Returns the
    desktop warnings, for the caller's verdict."""
    time.sleep(1.0)            # let the caller's own last passes drain
    fatal, warn = survey(seconds=2.0)
    if fatal:
        sys.exit("CONTAMINATED: another process was on the GPU at the end "
                 "of the run; discard these numbers:\n  " + "\n  ".join(fatal))
    return warn


if __name__ == "__main__":
    fatal, warn = survey()
    for w in warn:
        print("WARNING " + w)
    print("idle" if not fatal else "NOT idle:\n  " + "\n  ".join(fatal))
    sys.exit(1 if fatal else 0)
