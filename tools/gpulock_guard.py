"""Refuse to open a GPU device outside tools/gpulock.

Unlocked GPU tools bit four timing runs in two days (2026-11-08/09:
compose_city in three sessions, soar_regcount once): a device another
session opens mid-run shares the card and turns the run's numbers into
fiction, and a rule in a comment did not stop it. So the tools that open a
device call require_gpulock() first, before they request an adapter, and
stop with a clear message when they are not under the lock.

"Under the lock" is two facts, both checked:
  - the marker tools/gpulock exports to its command, SOAR_GPULOCK = the
    lock file's path (it survives flock and systemd-run --scope);
  - the lock itself is held right now: a non-blocking flock of that file
    fails. A marker without the lock is a stale export in a shell, not a
    lock.

The library and pytest do not call this: Thomas runs those directly. A test
that drives one of these tools as a child (test_tint_compile, tint_check)
passes SOAR_GPULOCK_EXEMPT=pytest for that child; nothing else may.

tools/gpulock_guard.mjs is the same check for the .mjs tools.
"""

import fcntl
import os
import sys

LOCK = "/tmp/soar-gpu.lock"
# Distinct from a tool's own "cannot run" (2), so a caller that skips on 2
# does not quietly skip a refusal.
EXIT_UNLOCKED = 3


def lock_held(path: str = LOCK) -> bool:
    """Whether someone holds the GPU lock now (a non-blocking flock fails)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def unlocked_reason(env=None, path: str = LOCK):
    """Why this process is not under tools/gpulock, or None when it is."""
    env = os.environ if env is None else env
    if env.get("SOAR_GPULOCK_EXEMPT") == "pytest":
        return None
    if env.get("SOAR_GPULOCK") != path:
        return "SOAR_GPULOCK is not set"
    if not lock_held(path):
        return (f"SOAR_GPULOCK is set but {path} is not held "
                "(a stale export?)")
    return None


def require_gpulock(tool: str, argv=None) -> None:
    """Exit EXIT_UNLOCKED, before any device is opened, unless locked.

    Called first in a tool's main(), before its argument parse (so a
    missing required argument cannot end the tool before this does); a
    -h/--help in the arguments passes, since a usage text opens nothing.
    """
    argv = sys.argv[1:] if argv is None else argv
    if "-h" in argv or "--help" in argv:
        return
    why = unlocked_reason()
    if why is None:
        return
    print(f"{tool}: run this under tools/gpulock: it opens a GPU device "
          f"({why}). For example:\n"
          f"    tools/gpulock uv run python {tool} ...",
          file=sys.stderr)
    sys.exit(EXIT_UNLOCKED)
