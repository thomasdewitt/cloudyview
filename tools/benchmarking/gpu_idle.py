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
  IDLE_SM_PCT of SM time, or holding a GPU context at all — except an idle
  browser (BROWSERS): the QA harness parks a Chrome, and Thomas flies soar
  in his own Firefox on this box (2026-09-25), which is his to leave open;
  his Steam client (a Chromium UI, steamwebhelper) is the same case
  (2026-11-08: idle at 0% SM, it refused a whole timing lock).
  A browser that DRAWS is a neighbour like any other; one driven over
  DevTools (the QA Chrome) is named by whoever drives it (driven_by). require_idle() waits
  briefly for these to go (WAIT_S), then exits; a Watch samples all through
  the run, and
  check_quiet() exits non-zero with CONTAMINATED if the watch or its own
  last look saw one — discard the numbers.
- WARNING, the desktop drawing: gnome-shell or the Claude app above
  DESKTOP_SM_PCT, or device utilization pmon attributes to nobody (the
  compositor under the clock lock). This is Thomas's own screen and cannot
  be switched off; under the 180 MHz lock an ordinary redraw takes a third
  of the card. Interleaving spreads it over every variant alike, so the
  RATIOS stand when the null control holds within ~1% and the rounds agree;
  the absolute ms under the lock are pessimistic. Tools report it, with
  that verdict, rather than failing.

    uv run python tools/benchmarking/gpu_idle.py      # exit 0 idle, 1 not

Check BEFORE queueing for tools/gpulock as well as inside it. Inside the
lock nothing worth waiting for can arrive (a teammate's GPU job is queued
behind you), so a neighbour there is outside-lock work or an app that will
not leave soon, and every second spent waiting holds the whole queue: on
2026-11-08 an idle Steam client made one timing lock refuse six blocks of
five minutes each. So require_idle() waits WAIT_S, not minutes, and a
timing command is preceded by the pre-check, which ignores the lock
holder's own work (LOCK; its tree under the flock that holds it):

    uv run python tools/benchmarking/gpu_idle.py --outside-lock && \
        tools/gpulock uv run python tools/benchmarking/soar_pass_timer.py ...

The pre-check is only a fast fail: the card can change while you queue, so
the check inside the lock stays the one that counts. It fails closed on one
case: the holder's QA Chrome drawing reads as a neighbour, because the CDP
scripts only drive that Chrome and it does not run under the flock.
"""

import os
import re
import subprocess
import sys
import time

IDLE_UTIL_PCT = 3
# How long require_idle() waits for a neighbour to leave, inside the lock.
WAIT_S = 30.0
# tools/gpulock's lock file: the flock holding it, and everything under it, is
# the lock holder's work (what --outside-lock ignores).
LOCK = "/tmp/soar-gpu.lock"
# Under the M1-proxy clock lock (180 MHz) the compositor's few frames take
# ten times as long, and the device reads 6-7% busy with nothing but the
# desktop on it, below what pmon attributes to any process.
IDLE_UTIL_PCT_LOCKED = 10
IDLE_SM_PCT = 2
DESKTOP_SM_PCT = 15
# Processes that always hold a context on this desktop, the Claude app
# (claude-desktop) among them.
# Browsers whose idle context is allowed; their drawing is not (see above).
BROWSERS = ("chrome", "firefox", "steam", "steamwebhelper")
DESKTOP = {"gnome-shell", "Xwayland", "gnome-software", "ptyxis",
           "gnome-remote-de", "mutter-x11-fram", "claude-desktop", "nautilus"}


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
        if not f[1].isdigit():
            continue              # pmon's placeholder row: no process at all
        sm = None if f[3] == "-" else int(f[3])
        rows.append((int(f[1]), f[-1], sm))
    return rows


def _cmdline(pid: int) -> list[str]:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return []
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def _ppid(pid: int) -> int:
    try:
        stat = open(f"/proc/{pid}/stat").read()
    except OSError:
        return 0
    return int(stat.rsplit(")", 1)[1].split()[1])


def _ss_clients(port: int) -> str:
    return subprocess.run(
        ["ss", "-Htnp", "state", "established", f"( dport = :{port} )"],
        capture_output=True, text=True).stdout


def driven_by(pid: int) -> str:
    """Who is driving a browser that draws: the DevTools port of the browser
    `pid` belongs to (its GPU process is a grandchild of the browser, which
    carries --remote-debugging-port), and every process connected to that
    port, by pid, command line and working directory — the QA Chrome is one
    browser that every session's CDP scripts drive, so its own pid names
    nobody. "" when no ancestor has a DevTools port (Thomas's own Firefox)."""
    port, p = None, pid
    for _ in range(8):
        # Searched in the joined line: Chrome rewrites its argv into one
        # space-separated string.
        m = re.search(r"--remote-debugging-port=(\d+)", " ".join(_cmdline(p)))
        if m or p <= 1:
            port = int(m.group(1)) if m else None
            break
        p = _ppid(p)
    if port is None:
        return ""
    clients = []
    for m in re.finditer(r"pid=(\d+)", _ss_clients(port)):
        c = int(m.group(1))
        try:
            cwd = os.readlink(f"/proc/{c}/cwd")
        except OSError:
            cwd = "?"
        line = " ".join(" ".join(_cmdline(c)).split())[:120]
        clients.append(f"pid {c} `{line}` in {cwd}")
    if not clients:
        return (f"; DevTools :{port} has no client connected (a tab left "
                "running, or someone at the window)")
    return (f"; driven over DevTools :{port} by "
            + ", ".join(sorted(set(clients))))


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


def _descendants(root: int) -> set[int]:
    """`root` and every process under it, from /proc's parent links."""
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry}/stat").read()
        except OSError:
            continue              # exited between the listing and the read
        # The field after the parenthesised name (which may hold spaces or
        # parentheses itself) is the state, then the parent's pid.
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    tree, todo = set(), [root]
    while todo:
        pid = todo.pop()
        tree.add(pid)
        todo.extend(children.get(pid, ()))
    return tree


def survey(seconds: float = 3.0, exclude=(),
           outside_lock: bool = False) -> tuple[list[str], list[str]]:
    """(fatal, warnings) sampled over `seconds`; both empty when idle.
    `outside_lock`: the lock holder's work (lock_work()) is not a
    neighbour either — the pre-check before queueing.
    `exclude`: pids to ignore besides this process and everything it
    started (always ignored: the caller's own work is not a neighbour). The
    whole tree, not just this pid, because a caller that times a child
    through `uv run` has the GPU held by uv's child, its grandchild — a
    watch that excluded only its own pid and the child's flagged the timed
    harness itself as a neighbour (2026-09-25). The price: a child the
    caller left running from an earlier variant is its own work too, and is
    not caught here — a caller that times variants as subprocesses must
    reap each before starting the next."""
    fatal, warn = set(), set()
    limit = (IDLE_UTIL_PCT_LOCKED if clock_mode().startswith("locked")
             else IDLE_UTIL_PCT)
    t_end = time.monotonic() + seconds
    while True:
        util = int(_smi("--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits").strip())
        # The tree on both sides of the pmon sample, each time: the caller's
        # children come and go during a run, and one started or ended just
        # as pmon looked is still the caller's.
        # The same for the lock holder's tree under outside_lock: a holder
        # running a loop starts new GPU processes all the time.
        own = _descendants(os.getpid())
        if outside_lock:
            own |= lock_work()
        rows = _processes()
        own |= _descendants(os.getpid()) | set(exclude)
        if outside_lock:
            own |= lock_work()
        known = 0
        for pid, short, sm in rows:
            known += sm or 0
            if pid in own:
                continue
            name = _full_name(pid, short)
            if name in DESKTOP:
                if (sm or 0) > DESKTOP_SM_PCT:
                    warn.add(f"desktop: pid {pid} ({name}) drawing at "
                             f"{sm}% SM")
                continue
            if sm is not None and sm > IDLE_SM_PCT:
                culprit = (driven_by(pid)
                           if any(b in name for b in BROWSERS) else "")
                fatal.add(f"pid {pid} ({name}) at {sm}% SM{culprit}")
            if not any(b in name for b in BROWSERS):
                fatal.add(f"pid {pid} ({name}) holds a GPU context")
        if util - known > limit:
            warn.add(f"device {util}% busy, {util - known}% of it "
                     f"attributed to no process (> {limit}%)")
        if time.monotonic() >= t_end:
            return sorted(fatal), sorted(warn)
        time.sleep(0.5)


def require_idle(wait_s: float = WAIT_S) -> list[str]:
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


class Watch:
    """Samples the card all through a run on a background thread, so a
    neighbour that draws for a while mid-run (a browser tab brought to the
    front) is caught, not only one still there at the end:
    `with Watch() as w: ...` then check_quiet(w). What the caller runs
    meanwhile, a subprocess under `uv run` included, is its own work, not a
    neighbour (survey)."""

    def __init__(self):
        import threading
        self.fatal = set()
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        # A watch that dies must say so: a dead thread with an empty
        # `fatal` would pass the run as watched and clean (loose-ends'
        # review: nvidia-smi failing, a pmon row it cannot parse).
        try:
            while not self._stop.is_set():
                fatal, _ = survey(seconds=0.0)
                self.fatal.update(fatal)
                self._stop.wait(0.5)
        except Exception as e:  # noqa: BLE001 — kept, and failed on below
            self.error = e

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def check_quiet(watch: "Watch | None" = None) -> list[str]:
    """After a run (own device still open): exit loudly if a neighbour is on
    the card, or `watch` saw one during the run, either of which may have
    skewed the numbers just printed. Returns the desktop warnings, for the
    caller's verdict."""
    if watch is not None and watch.error is not None:
        sys.exit(f"CONTAMINATED: the watch failed ({watch.error!r}); the run "
                 "was not watched, so these numbers cannot be trusted.")
    time.sleep(1.0)            # let the caller's own last passes drain
    fatal, warn = survey(seconds=2.0)
    fatal = sorted(set(fatal) | (watch.fatal if watch else set()))
    if fatal:
        sys.exit("CONTAMINATED: another process was on the GPU during or "
                 "at the end of the run; discard these numbers:\n  "
                 + "\n  ".join(fatal))
    return warn


def lock_work() -> set[int]:
    """Every process under a flock on LOCK: the GPU work of whoever holds
    tools/gpulock (the flocks still queued have nothing under them yet)."""
    work = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            argv = open(f"/proc/{entry}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if (argv and os.path.basename(argv[0]) == b"flock"
                and LOCK.encode() in argv):
            work |= _descendants(int(entry))
    return work


if __name__ == "__main__":
    outside = "--outside-lock" in sys.argv[1:]
    fatal, warn = survey(outside_lock=outside)
    for w in warn:
        print("WARNING " + w)
    print("idle" if not fatal else "NOT idle:\n  " + "\n  ".join(fatal))
    sys.exit(1 if fatal else 0)
