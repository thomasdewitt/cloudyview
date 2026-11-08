"""gpu_idle's idea of a neighbour: the caller's own process tree is not one.

A timing tool that runs its harness as a subprocess through `uv run` has
the GPU held by uv's child — the caller's grandchild. A watch that excluded
only its own pid and the child's flagged the timed run itself as a
neighbour and failed it CONTAMINATED (2026-09-25). No GPU needed: the
pmon sample is stood in for.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "benchmarking"))
import gpu_idle  # noqa: E402

# A child that starts a child and reports its pid, as `uv run python ...`
# leaves the python under uv.
SPAWNER = ("import subprocess, sys, time; "
           "p = subprocess.Popen([sys.executable, '-c', "
           "'import time; time.sleep(60)']); "
           "print(p.pid, flush=True); p.wait()")


@pytest.fixture
def grandchild():
    child = subprocess.Popen([sys.executable, "-c", SPAWNER],
                             stdout=subprocess.PIPE, text=True)
    pid = int(child.stdout.readline())
    yield pid
    subprocess.run(["kill", str(pid)], check=False)
    child.wait(timeout=10)
    child.stdout.close()


def _pmon_says(monkeypatch, rows):
    """survey() over a pmon sample of `rows` (pid, name, sm%), device idle.
    Returns the list the samples taken are counted in."""
    samples = []
    def processes():
        samples.append(1)
        return rows
    monkeypatch.setattr(gpu_idle, "_processes", processes)
    monkeypatch.setattr(gpu_idle, "_smi", lambda *a: "0\n")
    return samples


def test_the_callers_grandchild_is_in_its_tree(grandchild):
    assert grandchild in gpu_idle._descendants(os.getpid())
    assert os.getpid() in gpu_idle._descendants(os.getpid())
    # Downward only: the caller is not in its grandchild's tree.
    assert os.getpid() not in gpu_idle._descendants(grandchild)


def test_a_grandchild_drawing_is_not_a_neighbour(monkeypatch, grandchild):
    """The caller's grandchild at 90% SM passes; a process outside the tree
    (the caller's own parent, above it) at the same load is fatal — the
    control, so the test can fail."""
    parent = os.getppid()
    _pmon_says(monkeypatch, [(grandchild, "python3", 90),
                             (parent, "python3", 90)])
    fatal, _ = gpu_idle.survey(seconds=0.0)
    assert not any(f"pid {grandchild} " in f for f in fatal), fatal
    assert any(f.startswith(f"pid {parent} ") for f in fatal), fatal


def test_a_watch_does_not_flag_the_run_it_watches(monkeypatch, grandchild):
    samples = _pmon_says(monkeypatch, [(grandchild, "python3", 90)])
    with gpu_idle.Watch() as w:
        deadline = time.monotonic() + 10.0
        while not samples and time.monotonic() < deadline:
            time.sleep(0.01)
    assert samples, "the watch never sampled"
    assert w.error is None and not w.fatal, (w.error, w.fatal)


def test_only_the_direct_pid_would_flag_it(monkeypatch, grandchild):
    """Without the tree (the old rule: this pid alone) the same sample is
    fatal: what the tree exclusion is for."""
    _pmon_says(monkeypatch, [(grandchild, "python3", 90)])
    monkeypatch.setattr(gpu_idle, "_descendants", lambda root: {root})
    fatal, _ = gpu_idle.survey(seconds=0.0)
    assert any(f"pid {grandchild} " in f for f in fatal), fatal


# pids no process has: the names come from pmon's column, as for a process
# that exited between the sample and the /proc read.
STEAM, HELPER, GAME = 2**22 + 11, 2**22 + 12, 2**22 + 13


def test_thomass_idle_steam_is_not_a_neighbour(monkeypatch):
    """His Steam client holds contexts at 0% SM (2026-11-08); like his idle
    Firefox it is allowed. Any other idle context stays fatal — the control."""
    _pmon_says(monkeypatch, [(STEAM, "steam", None),
                             (HELPER, "steamwebhelper", 0),
                             (GAME, "somegame", None)])
    fatal, _ = gpu_idle.survey(seconds=0.0)
    assert fatal == [f"pid {GAME} (somegame) holds a GPU context"], fatal


def test_steam_drawing_is_a_neighbour(monkeypatch):
    _pmon_says(monkeypatch, [(HELPER, "steamwebhelper", 40)])
    fatal, _ = gpu_idle.survey(seconds=0.0)
    assert fatal == [f"pid {HELPER} (steamwebhelper) at 40% SM"], fatal
