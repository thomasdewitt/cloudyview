"""Refuse to open a GPU device outside tools/gpulock.

Unlocked GPU tools bit four timing runs in two days (2026-11-08/09:
compose_city in three sessions, soar_regcount once): a device another
session opens mid-run shares the card and turns the run's numbers into
fiction, and a rule in a comment did not stop it. So the tools that open a
device call require_gpulock() before they request an adapter, and stop
with a clear message when they are not under the lock.

"Under the lock" is two facts, both checked:
  - the marker tools/gpulock exports to its command, SOAR_GPULOCK = the
    lock file's path;
  - SOAR_GPULOCK_PID, the launcher's own pid, which it also exports, is an
    ANCESTOR of this process and is a running tools/gpulock. The launcher
    starts the command only once it holds the lock, so that is the lock held
    by this command's own launcher. "Held by someone" would not be enough:
    the lock is usually held by another session, which is exactly when an
    unlocked run spoils its timing, and a marker set by hand would then pass
    (loose-ends, review of 4b86576, found it passing).

Read from one `ps -A -ww -o pid=,ppid=,command=` table, which procps (Linux)
and BSD ps (macOS) both answer: one code path everywhere. Until 2026-11-09
the second check read /proc/locks and /proc/<pid>/status, which macOS does
not have, so on Thomas's M1 every guarded tool refused.

The library and pytest do not call this: Thomas runs those directly, and
the library's own CLIs (witness, behold, glimpse, soar) stay unguarded by
design, bound only by the rule. A test that drives one of these tools as a
child (test_tint_compile, tint_check) passes SOAR_GPULOCK_EXEMPT=pytest for
that child; nothing else may.

tools/gpulock_guard.mjs is the same check for the .mjs tools, and
tools/qa_chrome.sh does it in bash for the QA Chrome's start.
"""

import os
import subprocess
import sys

LOCK = "/tmp/soar-gpu.lock"
# Distinct from a tool's own "cannot run" (2), so a caller that skips on 2
# does not quietly skip a refusal.
EXIT_UNLOCKED = 3


def process_table() -> dict:
    """{pid: (ppid, command)} for every process, from ps."""
    return parse_ps(subprocess.run(
        ["ps", "-A", "-ww", "-o", "pid=,ppid=,command="],
        capture_output=True, text=True, check=True).stdout)


def parse_ps(out: str) -> dict:
    """`ps -o pid=,ppid=,command=` output -> {pid: (ppid, command)}."""
    table = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            table[int(parts[0])] = (int(parts[1]),
                                    parts[2] if len(parts) > 2 else "")
    return table


def lineage(table: dict, pid: int = None) -> list:
    """`pid` (this process by default) and its ancestors, nearest first."""
    pid = os.getpid() if pid is None else pid
    chain = []
    while pid > 0 and pid not in chain:
        chain.append(pid)
        if pid not in table:
            break
        pid = table[pid][0]
    return chain


def is_launcher(command: str) -> bool:
    """Whether a ps command line is tools/gpulock running: the script is
    among its first words (`python3 .../tools/gpulock ...`, under whatever
    interpreter path the platform shows)."""
    return any(os.path.basename(w) == "gpulock" for w in command.split()[:3])


def unlocked_reason(env=None, path: str = LOCK, table: dict = None):
    """Why this process is not under tools/gpulock, or None when it is."""
    env = os.environ if env is None else env
    if env.get("SOAR_GPULOCK_EXEMPT") == "pytest":
        return None
    return launcher_reason(env, path, table)


def launcher_reason(env=None, path: str = LOCK, table: dict = None):
    """Why no tools/gpulock among this process's ancestors holds `path`, or
    None when one does: unlocked_reason without the pytest exemption, which
    is also what a nested tools/gpulock asks before taking the lock again."""
    env = os.environ if env is None else env
    if env.get("SOAR_GPULOCK") != path:
        return "SOAR_GPULOCK is not set"
    pid = env.get("SOAR_GPULOCK_PID", "")
    if not pid.isdigit():
        return ("SOAR_GPULOCK_PID is not set (a tools/gpulock from before "
                "2026-11-09?)")
    pid = int(pid)
    table = process_table() if table is None else table
    if pid not in lineage(table):
        return (f"SOAR_GPULOCK_PID {pid} is not an ancestor of this process "
                "(a stale export?)")
    if not is_launcher(table.get(pid, (0, ""))[1]):
        return f"SOAR_GPULOCK_PID {pid} is not tools/gpulock"
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
