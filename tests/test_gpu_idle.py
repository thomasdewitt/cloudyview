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


def test_the_lock_holders_work_is_not_a_neighbour_outside_the_lock(
        monkeypatch, tmp_path):
    """--outside-lock's view: what runs under the flock holding the GPU lock
    is the holder's work (a pre-check before queueing would otherwise always
    see it and fail), including a worker the holder starts only after the
    check has begun (a holder running a loop does that all the time); a
    process outside it is still fatal — the control."""
    lock = tmp_path / "gpu.lock"
    monkeypatch.setattr(gpu_idle, "LOCK", str(lock))
    late = SPAWNER.replace("import subprocess", "import subprocess, time; "
                           "time.sleep(0.5); import subprocess", 1)
    # Detached (setsid -f): a holder under this process would be the
    # caller's own tree, excluded whatever lock_work() says.
    holder = subprocess.Popen(["setsid", "-f", "flock", "-x", str(lock),
                               sys.executable, "-c", late],
                              stdout=subprocess.PIPE, text=True)
    outsider = os.getppid()
    seen = []

    def processes():
        # The pmon sample: taken after survey started, by which time the
        # holder's worker exists (readline blocks until it is spawned).
        if not seen:
            seen.append(int(holder.stdout.readline()))
        return [(seen[0], "python3", 50), (outsider, "python3", 50)]

    monkeypatch.setattr(gpu_idle, "_processes", processes)
    monkeypatch.setattr(gpu_idle, "_smi", lambda *a: "0\n")
    try:
        before = gpu_idle.lock_work()
        fatal, _ = gpu_idle.survey(seconds=0.0, outside_lock=True)
        worker = seen[0]
        assert worker not in before        # it really started late
        assert worker not in gpu_idle._descendants(os.getpid())
        assert not any(f"pid {worker} " in f for f in fatal), fatal
        assert any(f.startswith(f"pid {outsider} ") for f in fatal), fatal
        # A snapshot taken before the worker existed is what the review
        # caught: with it the worker reads as a neighbour.
        monkeypatch.setattr(gpu_idle, "lock_work", lambda: before)
        fatal, _ = gpu_idle.survey(seconds=0.0, outside_lock=True)
        assert any(f"pid {worker} " in f for f in fatal), fatal
    finally:
        if seen:
            subprocess.run(["kill", str(seen[0])], check=False)
        holder.wait(timeout=10)
        holder.stdout.close()


# A Chrome as /proc shows it: the GPU process under the zygote under the
# browser, whose argv Chrome rewrites into ONE space-joined string.
GPU, ZYGOTE, BROWSER, DRIVER = 2**22 + 21, 2**22 + 22, 2**22 + 23, 2**22 + 24
CHROME = {
    GPU: (ZYGOTE, ["/opt/google/chrome/chrome --type=gpu-process"]),
    ZYGOTE: (BROWSER, ["/opt/google/chrome/chrome --type=zygote"]),
    BROWSER: (1, ["/opt/google/chrome/chrome --user-data-dir=/q "
                  "--remote-debugging-port=9333 about:blank"]),
    DRIVER: (1, ["node", "tools/tint_check.mjs", "--web", "web"]),
}


def _chrome(monkeypatch, ss):
    monkeypatch.setattr(gpu_idle, "_cmdline",
                        lambda pid: CHROME.get(pid, (0, []))[1])
    monkeypatch.setattr(gpu_idle, "_ppid",
                        lambda pid: CHROME.get(pid, (0,))[0])
    ports = []
    def clients(port):
        ports.append(port)
        return ss
    monkeypatch.setattr(gpu_idle, "_ss_clients", clients)
    return ports


def test_a_drawing_qa_chrome_names_who_drives_it(monkeypatch):
    """The QA Chrome is one browser every session drives over CDP, so its
    pid names nobody (2026-11-08: an afternoon to find that a merge gate's
    tint_check had drawn in two timing locks). The neighbour line names the
    process connected to its DevTools port."""
    ports = _chrome(monkeypatch, "0 0 127.0.0.1:40060 127.0.0.1:9333 "
                                 f'users:(("node",pid={DRIVER},fd=21))\n')
    _pmon_says(monkeypatch, [(GPU, "chrome", 40)])
    fatal, _ = gpu_idle.survey(seconds=0.0)
    assert ports == [9333]
    assert len(fatal) == 1
    assert fatal[0].startswith(f"pid {GPU} (chrome) at 40% SM;")
    assert f"pid {DRIVER} `node tools/tint_check.mjs --web web`" in fatal[0]


def test_an_undriven_chrome_says_so(monkeypatch):
    _chrome(monkeypatch, "")
    assert "no client connected" in gpu_idle.driven_by(GPU)


def test_a_browser_with_no_devtools_port_names_nobody(monkeypatch):
    """Thomas's Firefox has no DevTools port; nothing to add, nothing run."""
    ports = _chrome(monkeypatch, "")
    assert gpu_idle.driven_by(DRIVER) == "" and ports == []
