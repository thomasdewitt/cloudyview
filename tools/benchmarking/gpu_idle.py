#!/usr/bin/env python
"""Refuse to time anything on a GPU that is doing someone else's work.

There is one GPU on this box and many sessions. tools/gpulock serializes the
ones that take it, but a Chrome tab left flying, a pass run outside the lock
or a python process still holding a device skews every timestamp without
leaving a trace in the numbers. On 2026-09-24 two measurements of the same
change disagreed by ~60 points and nobody could say which was contaminated.
Timing tools call require_idle() before they create their own device, and
check_quiet() when they finish, so a run either starts on an idle card and
stays that way or fails loudly.

Idle means, over a few seconds of samples:
- no process outside the desktop (DESKTOP) shows more than IDLE_SM_PCT of
  SM time (nvidia-smi pmon), and the device's utilization, less what the
  desktop draws, stays at or under IDLE_UTIL_PCT;
- the desktop itself stays under DESKTOP_SM_PCT: the Claude app draws a few
  percent while any session streams text, which cannot be switched off and
  is small, but a scrolled or animating window is not;
- no process other than the desktop's and Chrome holds a GPU context.
  Chrome is let through because the QA harness parks an idle browser; if it
  is not idle, the SM test catches it.

    uv run python tools/benchmarking/gpu_idle.py      # exit 0 idle, 1 not
"""

import os
import subprocess
import sys
import time

IDLE_UTIL_PCT = 3
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


def problems(seconds: float = 3.0, exclude=()) -> list[str]:
    """What stands between this card and an idle one, sampled over
    `seconds`; empty when idle. `exclude`: pids to ignore (the caller's own,
    once it holds a device)."""
    exclude = set(exclude) | {os.getpid()}
    found = set()
    t_end = time.monotonic() + seconds
    while True:
        util = int(_smi("--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits").strip())
        desktop = 0
        for pid, short, sm in _processes():
            if pid in exclude:
                desktop += sm or 0     # the caller's own tail, not a neighbour
                continue
            name = _full_name(pid, short)
            if name in DESKTOP:
                desktop += sm or 0
                if (sm or 0) > DESKTOP_SM_PCT:
                    found.add(f"pid {pid} ({name}) drawing at {sm}% SM")
                continue
            if sm is not None and sm > IDLE_SM_PCT:
                found.add(f"pid {pid} ({name}) at {sm}% SM")
            if "chrome" not in name:
                found.add(f"pid {pid} ({name}) holds a GPU context")
        if util - desktop > IDLE_UTIL_PCT:
            found.add(f"device utilization {util}% of which the desktop "
                      f"and the caller {desktop}% (> {IDLE_UTIL_PCT}% "
                      f"besides them)")
        if time.monotonic() >= t_end:
            return sorted(found)
        time.sleep(0.5)


def clock_mode() -> str:
    """The clock mode the caller declared (tools that lock the clocks set
    GPU_CLOCK_MODE), for the header of every table."""
    return os.environ.get("GPU_CLOCK_MODE",
                          "stock (no clock lock declared by the caller)")


def require_idle(wait_s: float = 300.0) -> None:
    """Block until the card is idle, up to `wait_s`; then exit loudly."""
    t_end = time.monotonic() + wait_s
    while True:
        p = problems()
        if not p:
            return
        if time.monotonic() >= t_end:
            sys.exit("GPU is not idle, refusing to time:\n  " + "\n  ".join(p))
        print("waiting for an idle GPU: " + "; ".join(p), file=sys.stderr,
              flush=True)
        time.sleep(10)


def check_quiet() -> None:
    """After a run (own device still open): exit loudly if anyone else used
    the card meanwhile, which may have skewed the numbers just printed."""
    time.sleep(1.0)            # let the caller's own last passes drain
    p = problems(seconds=2.0)
    if p:
        sys.exit("CONTAMINATED: the GPU was not idle at the end of the run; "
                 "discard these numbers:\n  " + "\n  ".join(p))


if __name__ == "__main__":
    p = problems()
    print("idle" if not p else "NOT idle:\n  " + "\n  ".join(p))
    sys.exit(1 if p else 0)
