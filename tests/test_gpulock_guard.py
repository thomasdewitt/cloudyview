"""The tools that open a GPU device refuse to run outside tools/gpulock.

Unlocked device-opening tools spoiled four timing runs in two days
(2026-11-08/09: compose_city in three sessions, soar_regcount once), and a
rule in a comment did not stop it. tools/gpulock now exports SOAR_GPULOCK
(the lock's path) to its command, and each tool below calls
tools/gpulock_guard (.py or .mjs) first, which refuses, exit 3, unless the
marker is set AND the lock is held.

Cheap, no GPU: every tool here is run without the marker, and must refuse
BEFORE it requests an adapter. The Python tools run with a stand-in `wgpu`
and `mitsuba` on the path whose adapter and variant entry points raise, so
a tool that reached for a device first would exit 1 with that error, not 3;
the .mjs tools run with node's network poisoned, so one that reached for
Chrome's DevTools (where its adapter is) would fail the same way.

The library and pytest are not guarded (Thomas runs them directly); a test
that drives a guarded tool passes SOAR_GPULOCK_EXEMPT=pytest to that child
(test_tint_compile).
"""

import fcntl
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"

PY_TOOLS = [
    "tools/compose_city.py",
    "tools/soar_regcount.py",
    "tools/soar_shader_ab.py",
    "tools/light_cache_ab.py",
    "tools/city_graffiti_harness.py",
    "tools/night_city_harness.py",
    "tools/contact_sheet.py",
    "tools/prebake_demos.py",
    "tools/benchmarking/soar_pass_timer.py",
    "tools/benchmarking/soar_frame_bench.py",
    "tools/benchmarking/benchmark.py",      # its default, CUDA, backend
]
JS_TOOLS = [
    "tools/tint_check.mjs",
    "tools/pinned_still.mjs",
    "tools/steam_live_qa.mjs",
    "tools/sea_speckle_qa.mjs",
    "tools/soar_noise.mjs",
    "tools/surface_cs_timer.mjs",
]
# What in a tool's source means it opens a device (directly, through the
# renderer host or witness or behold, through Mitsuba's CUDA variant, through
# the QA Chrome's WebGPU, or by running one of the library's CLIs), and the files that may say so without
# the guard: the shared CDP library the .mjs tools call, the guard itself,
# and the exporter, which only encodes tiles and never renders. The library
# CLIs themselves (witness, behold, glimpse, soar) stay unguarded by design.
OPENS = re.compile(r"request_adapter|SoarRenderer\(|\bwitness\(|"
                   r"render_nested\(|\bbehold\b|import mitsuba|"
                   r"import wgpu|from wgpu|openTab\(|"
                   r"[\"'](witness|behold|glimpse|soar)[\"']\s*[,\]]")  # a CLI
# A shell tool that starts a browser starts its GPU process.
OPENS_SH = re.compile(r"google-chrome|chromium")
SH_TOOLS = ["tools/qa_chrome.sh"]
NOT_TOOLS = {"tools/soar_cdp.mjs", "tools/gpulock_guard.py",
             "tools/gpulock_guard.mjs", "tools/export_web_assets.py"}

EXIT_UNLOCKED = 3

STANDIN = '''
class _Any:
    def __getattr__(self, name):
        return _Any()
    def __call__(self, *a, **k):
        return _Any()
    def __iter__(self):
        return iter(())
    def __or__(self, other):
        return self
    __ror__ = __or__


def __getattr__(name):
    if name in {BLOCKED}:
        raise RuntimeError("STANDIN: a device was requested before the guard")
    return _Any()
'''


def _unlocked_env(extra_path=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("SOAR_GPULOCK", "SOAR_GPULOCK_EXEMPT")}
    if extra_path:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(extra_path)] + ([env["PYTHONPATH"]] if "PYTHONPATH" in env
                                 else []))
    return env


@pytest.fixture(scope="module")
def standins(tmp_path_factory):
    d = tmp_path_factory.mktemp("standins")
    for mod, blocked in (("wgpu", {"gpu"}),
                         ("mitsuba", {"set_variant", "variants"})):
        (d / mod).mkdir()
        (d / mod / "__init__.py").write_text(
            STANDIN.replace("{BLOCKED}", repr(blocked)))
    return d


@pytest.mark.parametrize("tool", PY_TOOLS)
def test_a_python_tool_refuses_unlocked(tool, standins):
    proc = subprocess.run([sys.executable, str(REPO / tool)], cwd=REPO,
                          env=_unlocked_env(standins), capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == EXIT_UNLOCKED, proc.stderr[-2000:]
    assert "run this under tools/gpulock: it opens a GPU device" in proc.stderr
    assert "STANDIN" not in proc.stderr


POISON = """
import http from "node:http";
import net from "node:net";
const no = () => { throw new Error("POISON: the network was touched before the guard"); };
http.request = no; http.get = no; net.connect = no; net.createConnection = no;
globalThis.fetch = no; globalThis.WebSocket = class { constructor() { no(); } };
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("tool", JS_TOOLS)
def test_a_js_tool_refuses_unlocked(tool, tmp_path):
    poison = tmp_path / "poison.mjs"
    poison.write_text(POISON)
    proc = subprocess.run(
        ["node", "--import", poison.as_uri(), str(REPO / tool),
         "--web", str(REPO / "web")],
        cwd=REPO, env=_unlocked_env(), capture_output=True, text=True,
        timeout=60)
    assert proc.returncode == EXIT_UNLOCKED, proc.stderr[-2000:]
    assert "run this under tools/gpulock: it opens a GPU device" in proc.stderr
    assert "POISON" not in proc.stderr


def test_every_tool_that_opens_a_device_is_guarded():
    guarded = set(PY_TOOLS) | set(JS_TOOLS) | set(SH_TOOLS)
    for p in sorted(TOOLS.rglob("*")):
        if p.suffix not in (".py", ".mjs", ".sh"):
            continue
        rel = p.relative_to(REPO).as_posix()
        src = p.read_text()
        opens = OPENS_SH if p.suffix == ".sh" else OPENS
        if rel in NOT_TOOLS or not opens.search(src):
            continue
        assert rel in guarded, f"{rel} opens a GPU device and is not guarded"
        call = {".py": f'require_gpulock("{rel}"',
                ".mjs": f'requireGpulock("{rel}")',
                ".sh": f'gpulock_guard.py" {rel}'}[p.suffix]
        assert call in src, f"{rel} does not call the guard ({call})"


def test_the_qa_chrome_start_refuses_unlocked(tmp_path):
    env = _unlocked_env()
    env["QA_DIR"] = str(tmp_path)
    try:
        proc = subprocess.run([str(TOOLS / "qa_chrome.sh"), "start", "9447"],
                              cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=60)
    finally:
        # Should the guard ever let it through, no Chrome is left running.
        subprocess.run([str(TOOLS / "qa_chrome.sh"), "stop", "9447"],
                       cwd=REPO, env=env, capture_output=True, timeout=30)
    assert proc.returncode == EXIT_UNLOCKED, proc.stderr[-2000:]
    assert "run this under tools/gpulock: it opens a GPU device" in proc.stderr
    assert not any(tmp_path.iterdir()), "it got as far as making a profile"


def _guard():
    spec = importlib.util.spec_from_file_location("gpulock_guard",
                                                  TOOLS / "gpulock_guard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _launcher():
    """tools/gpulock as a module (it is a Python script with no suffix)."""
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("gpulock_launcher", str(TOOLS / "gpulock"))
    spec = importlib.util.spec_from_loader("gpulock_launcher", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def _locked(lock, *cmd, wait="30", **kw):
    """Run `cmd` under tools/gpulock on the test's own lock file."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("SOAR_GPULOCK", "SOAR_GPULOCK_PID", "GPULOCK_MEM")}
    env.update(GPULOCK_FILE=str(lock), GPULOCK_WAIT=wait)
    return subprocess.run([str(TOOLS / "gpulock"), *cmd], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60, **kw)


def test_the_lock_path_is_one_constant():
    lock = _launcher().LOCK
    assert _guard().LOCK == lock
    assert f'export const LOCK = "{lock}";' in (
        TOOLS / "gpulock_guard.mjs").read_text()


GUARD_PY = ("import sys; sys.path.insert(0, 'tools'); import gpulock_guard "
            "as g, os; print(os.environ['SOAR_GPULOCK'], "
            "os.environ['SOAR_GPULOCK_PID'], os.getppid(), "
            "g.unlocked_reason(path=os.environ['SOAR_GPULOCK']))")


def test_under_the_launcher_the_guard_passes(tmp_path):
    lock = tmp_path / "lock"
    r = _locked(lock, sys.executable, "-c", GUARD_PY)
    assert r.returncode == 0, r.stderr
    marker, pid, ppid, why = r.stdout.split()
    assert marker == str(lock) and why == "None", r.stdout
    # The launcher is the command's parent (systemd-run --scope execs in
    # place on Linux; elsewhere the command is its child directly).
    assert pid == ppid, r.stdout
    if shutil.which("node") is not None:
        js = (f"import {{ unlockedReason }} from "
              f"{(TOOLS / 'gpulock_guard.mjs').as_uri()!r};"
              "console.log(unlockedReason(process.env, process.env.SOAR_GPULOCK));")
        r = _locked(lock, "node", "--input-type=module", "-e", js)
        assert r.returncode == 0 and r.stdout.strip() == "null", r


def test_outside_the_launcher_the_guard_refuses(tmp_path):
    g = _guard()
    lock = str(tmp_path / "lock")
    assert g.unlocked_reason({}, lock) == "SOAR_GPULOCK is not set"
    assert "SOAR_GPULOCK_PID is not set" in g.unlocked_reason(
        {"SOAR_GPULOCK": lock}, lock)
    # An ancestor, but not a launcher: this test's own parent.
    why = g.unlocked_reason({"SOAR_GPULOCK": lock,
                             "SOAR_GPULOCK_PID": str(os.getppid())}, lock)
    assert why == f"SOAR_GPULOCK_PID {os.getppid()} is not tools/gpulock", why
    # Not an ancestor at all: a pid no process has.
    why = g.unlocked_reason({"SOAR_GPULOCK": lock,
                             "SOAR_GPULOCK_PID": "4194000"}, lock)
    assert "is not an ancestor of this process" in why, why


def test_a_launcher_that_is_not_an_ancestor_does_not_pass(tmp_path):
    # The usual state of this box: another session's launcher holds the
    # lock. A marker set by hand, even naming that launcher, must not pass
    # (loose-ends, review of 4b86576). Here the launcher is this test's own
    # child, holding the test's own lock file.
    lock = tmp_path / "lock"
    env = {k: v for k, v in os.environ.items() if k != "GPULOCK_MEM"}
    env.update(GPULOCK_FILE=str(lock))
    holder = subprocess.Popen([str(TOOLS / "gpulock"), "sleep", "60"],
                              env=env, stderr=subprocess.DEVNULL)
    try:
        g = _guard()
        for _ in range(500):
            if g.is_launcher(g.process_table().get(holder.pid, (0, ""))[1]):
                break
            time.sleep(0.01)
        marked = {"SOAR_GPULOCK": str(lock),
                  "SOAR_GPULOCK_PID": str(holder.pid)}
        why = g.unlocked_reason(marked, str(lock))
        assert why and "is not an ancestor of this process" in why, why
        if shutil.which("node") is not None:
            js = subprocess.run(
                ["node", "--input-type=module", "-e",
                 f"import {{ unlockedReason }} from "
                 f"{(TOOLS / 'gpulock_guard.mjs').as_uri()!r};"
                 f"console.log(unlockedReason({json.dumps(marked)}, "
                 f"{str(lock)!r}));"],
                capture_output=True, text=True, check=True, timeout=30)
            assert "is not an ancestor of this process" in js.stdout, js.stdout
    finally:
        holder.terminate()
        holder.wait(timeout=30)


def test_the_launcher_times_out_with_75(tmp_path):
    lock = tmp_path / "lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        t0 = time.monotonic()
        r = _locked(lock, "true", wait="0.3")
        assert r.returncode == 75, r
        assert "gave up after 0.3 s" in r.stderr
        assert time.monotonic() - t0 < 10
    finally:
        os.close(fd)


def test_the_command_does_not_inherit_the_lock(tmp_path):
    # A background child of the command (the QA Chrome) cannot keep the
    # lock: the launcher's descriptor stays in the launcher.
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("needs /proc to list descriptors")
    lock = tmp_path / "lock"
    probe = ("import os; print(sum(os.path.realpath(f'/proc/self/fd/{f}') == "
             f"{str(lock)!r} for f in os.listdir('/proc/self/fd')))")
    r = _locked(lock, sys.executable, "-c", probe)
    assert r.returncode == 0 and r.stdout.strip() == "0", r


def test_a_signal_to_the_launcher_reaches_the_command(tmp_path):
    lock = tmp_path / "lock"
    env = {k: v for k, v in os.environ.items() if k != "GPULOCK_MEM"}
    env.update(GPULOCK_FILE=str(lock))
    p = subprocess.Popen([str(TOOLS / "gpulock"), sys.executable, "-c",
                          "import time; print('up', flush=True); time.sleep(60)"],
                         env=env, stdout=subprocess.PIPE, text=True,
                         stderr=subprocess.DEVNULL)
    assert p.stdout.readline().strip() == "up"
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 128 + signal.SIGTERM
    # And the lock is free again.
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


# --- Darwin, mocked (there is no Mac here) -------------------------------

def test_darwin_runs_uncapped_and_says_so(capsys):
    launcher = _launcher()
    assert launcher.command(["node", "x.mjs"], platform="darwin",
                            env={}) == ["node", "x.mjs"]
    assert capsys.readouterr().err.strip() == "gpulock: no memory cap on darwin"
    linux = launcher.command(["node", "x.mjs"], platform="linux", env={})
    assert linux[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryMax=16G" in linux and linux[-2:] == ["node", "x.mjs"]


def test_darwin_refuses_a_memory_cap_it_cannot_honour():
    with pytest.raises(SystemExit) as e:
        _launcher().command(["node"], platform="darwin",
                            env={"GPULOCK_MEM": "8G"})
    assert "GPULOCK_MEM=8G was set" in str(e.value)
    assert "no memory cap on darwin" in str(e.value)


def _darwin_ps(me, parent):
    """`ps -A -ww -o pid=,ppid=,command=` as macOS's BSD ps prints it: a
    framework Python running tools/gpulock, node under it, this test's pid
    as the tool under node."""
    return "\n".join([
        "    1     0 /sbin/launchd",
        "  412     1 /System/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal",
        "  415   412 -zsh",
        "  900   415 /Library/Frameworks/Python.framework/Versions/3.12/Resources/"
        "Python.app/Contents/MacOS/Python tools/gpulock node tools/pinned_still.mjs --web web",
        f"{parent:5d}   900 node tools/pinned_still.mjs --web web",
        f"{me:>5} {parent:5d} python3 -c import gpulock_guard",
        "  950   415 /Library/Frameworks/Python.framework/Versions/3.12/Resources/"
        "Python.app/Contents/MacOS/Python tools/gpulock sleep 60",
    ]) + "\n"


def test_darwin_ps_output_passes_under_its_launcher_only():
    g = _guard()
    me = os.getpid()
    table = g.parse_ps(_darwin_ps(me, 901))
    env = {"SOAR_GPULOCK": g.LOCK, "SOAR_GPULOCK_PID": "900"}
    assert g.unlocked_reason(env, table=table) is None
    # Another launcher on the same machine, not an ancestor.
    why = g.unlocked_reason({**env, "SOAR_GPULOCK_PID": "950"}, table=table)
    assert "is not an ancestor" in why, why
    # An ancestor that is not a launcher.
    why = g.unlocked_reason({**env, "SOAR_GPULOCK_PID": "415"}, table=table)
    assert why == "SOAR_GPULOCK_PID 415 is not tools/gpulock", why
    if shutil.which("node") is not None:
        # The same table with node's own pid where this test's stands: the
        # guard walks up from the process it runs in.
        js = subprocess.run(
            ["node", "--input-type=module", "-e",
             f"import {{ parsePs, unlockedReason, LOCK }} from "
             f"{(TOOLS / 'gpulock_guard.mjs').as_uri()!r};"
             "const t = parsePs(process.argv[1].replace('{ME}', "
             "String(process.pid).padStart(5)));"
             "const e = (pid) => ({ SOAR_GPULOCK: LOCK, SOAR_GPULOCK_PID: pid });"
             "console.log(JSON.stringify([unlockedReason(e('900'), LOCK, t),"
             " unlockedReason(e('950'), LOCK, t), unlockedReason(e('415'), LOCK, t)]));",
             _darwin_ps("{ME}", 901)],
            capture_output=True, text=True, timeout=30)
        assert js.returncode == 0, js.stderr
        ok, other, shell = json.loads(js.stdout)
        assert ok is None, ok
        assert "is not an ancestor" in other, other
        assert shell == "SOAR_GPULOCK_PID 415 is not tools/gpulock", shell


def test_pytest_may_exempt_the_tool_it_drives():
    assert _guard().unlocked_reason({"SOAR_GPULOCK_EXEMPT": "pytest"}) is None
    assert _guard().unlocked_reason({"SOAR_GPULOCK_EXEMPT": "1"}) is not None
