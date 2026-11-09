"""Refuse to open a GPU device outside tools/gpulock.

Unlocked GPU tools bit four timing runs in two days (2026-11-08/09:
compose_city in three sessions, soar_regcount once): a device another
session opens mid-run shares the card and turns the run's numbers into
fiction, and a rule in a comment did not stop it. So the tools that open a
device call require_gpulock() before they request an adapter, and stop
with a clear message when they are not under the lock.

"Under the lock" is two facts, both checked:
  - the marker tools/gpulock exports to its command, SOAR_GPULOCK = the
    lock file's path (it survives flock and systemd-run --scope);
  - the lock is held by an ANCESTOR of this process: /proc/locks names the
    pid holding the FLOCK on that file, and under tools/gpulock that is
    flock(1), the command's parent (systemd-run --scope execs in place).
    "Held by someone" is not enough: this box's lock is usually held by
    another session, which is exactly when an unlocked run spoils its
    timing, and a marker set by hand would then pass (loose-ends, review
    of 4b86576, found it passing).

The library and pytest do not call this: Thomas runs those directly, and
the library's own CLIs (witness, behold, glimpse, soar) stay unguarded by
design, bound only by the rule. A test that drives one of these tools as a
child (test_tint_compile, tint_check) passes SOAR_GPULOCK_EXEMPT=pytest for
that child; nothing else may.

tools/gpulock_guard.mjs is the same check for the .mjs tools, and
tools/qa_chrome.sh does it in bash for the QA Chrome's start.
"""

import os
import sys

LOCK = "/tmp/soar-gpu.lock"
# Distinct from a tool's own "cannot run" (2), so a caller that skips on 2
# does not quietly skip a refusal.
EXIT_UNLOCKED = 3


def lock_holders(path: str = LOCK) -> set:
    """The pids holding a FLOCK on `path` now (blocked waiters excluded)."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return set()
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    pids = set()
    with open("/proc/locks") as f:
        for line in f:
            # "N: FLOCK ADVISORY WRITE pid maj:min:ino start end"; a waiter's
            # line carries "->" after the index.
            fields = line.split()
            if (len(fields) >= 6 and fields[1] == "FLOCK"
                    and fields[5] == key):
                pids.add(int(fields[4]))
    return pids


def lineage(pid: int = None) -> list:
    """This process and its ancestors, nearest first."""
    pid = os.getpid() if pid is None else pid
    chain = []
    while pid > 0 and pid not in chain:
        chain.append(pid)
        try:
            with open(f"/proc/{pid}/status") as f:
                ppid = next(int(l.split()[1]) for l in f
                            if l.startswith("PPid:"))
        except (FileNotFoundError, StopIteration):
            break
        pid = ppid
    return chain


def unlocked_reason(env=None, path: str = LOCK):
    """Why this process is not under tools/gpulock, or None when it is."""
    env = os.environ if env is None else env
    if env.get("SOAR_GPULOCK_EXEMPT") == "pytest":
        return None
    if env.get("SOAR_GPULOCK") != path:
        return "SOAR_GPULOCK is not set"
    holders = lock_holders(path)
    if not holders:
        return (f"SOAR_GPULOCK is set but {path} is not held "
                "(a stale export?)")
    if not holders & set(lineage()):
        return (f"SOAR_GPULOCK is set but {path} is held by pid "
                f"{min(holders)}, which is not this command's tools/gpulock")
    return None


def require_gpulock(tool: str, argv=None, example=None) -> None:
    """Exit EXIT_UNLOCKED, before any device is opened, unless locked.

    Called before a tool's first device work: first in main(), before the
    argument parse, wherever the tool always opens a device (so a missing
    required argument cannot end the tool before this does); after the
    parse where only some modes do (prebake_demos, benchmark.py). A -h or
    --help in the arguments passes, since a usage text opens nothing.
    """
    argv = sys.argv[1:] if argv is None else argv
    if "-h" in argv or "--help" in argv:
        return
    why = unlocked_reason()
    if why is None:
        return
    example = example or f"tools/gpulock uv run python {tool} ..."
    print(f"{tool}: run this under tools/gpulock: it opens a GPU device "
          f"({why}). For example:\n    {example}",
          file=sys.stderr)
    sys.exit(EXIT_UNLOCKED)


if __name__ == "__main__":
    # The check for a shell tool (tools/qa_chrome.sh start):
    #   python3 tools/gpulock_guard.py TOOL "EXAMPLE COMMAND"
    # exits EXIT_UNLOCKED, with the message, unless under the lock. Stdlib
    # only, so the system python3 runs it without the project env.
    require_gpulock(sys.argv[1], argv=[], example=sys.argv[2])
