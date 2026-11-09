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
import os
import re
import shutil
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


def test_gpulock_exports_the_marker_to_its_command():
    src = (TOOLS / "gpulock").read_text()
    lock = re.search(r'^LOCK=(\S+)$', src, re.M).group(1)
    export = src.index('export SOAR_GPULOCK="$LOCK"')
    assert export < src.index("exec flock")
    spec = importlib.util.spec_from_file_location("gpulock_guard",
                                                  TOOLS / "gpulock_guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    assert guard.LOCK == lock
    js = (TOOLS / "gpulock_guard.mjs").read_text()
    assert f'export const LOCK = "{lock}";' in js


def _guard():
    spec = importlib.util.spec_from_file_location("gpulock_guard",
                                                  TOOLS / "gpulock_guard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_marker_needs_the_lock_held(tmp_path):
    # On a lock file of the test's own, so no other session's lock matters.
    g = _guard()
    lock = tmp_path / "lock"
    marked = {"SOAR_GPULOCK": str(lock)}
    assert g.unlocked_reason({}, str(lock)) == "SOAR_GPULOCK is not set"
    assert "not held" in g.unlocked_reason(marked, str(lock))   # no file
    lock.touch()
    assert "not held" in g.unlocked_reason(marked, str(lock))   # free
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert g.unlocked_reason(marked, str(lock)) is None
        assert g.unlocked_reason({"SOAR_GPULOCK": "/elsewhere"},
                                 str(lock)) == "SOAR_GPULOCK is not set"
        if shutil.which("node") is not None:
            js = subprocess.run(
                ["node", "--input-type=module", "-e",
                 f"import {{ unlockedReason }} from "
                 f"{(TOOLS / 'gpulock_guard.mjs').as_uri()!r};"
                 f"console.log(JSON.stringify(["
                 f"unlockedReason({{SOAR_GPULOCK: {str(lock)!r}}}, {str(lock)!r}),"
                 f"unlockedReason({{}}, {str(lock)!r})]));"],
                capture_output=True, text=True, check=True, timeout=30)
            assert js.stdout.strip() == '[null,"SOAR_GPULOCK is not set"]'
    finally:
        os.close(fd)
    if shutil.which("node") is not None:
        js = subprocess.run(
            ["node", "--input-type=module", "-e",
             f"import {{ unlockedReason }} from "
             f"{(TOOLS / 'gpulock_guard.mjs').as_uri()!r};"
             f"console.log(unlockedReason({{SOAR_GPULOCK: {str(lock)!r}}}, "
             f"{str(lock)!r}));"],
            capture_output=True, text=True, check=True, timeout=30)
        assert "not held" in js.stdout


def test_a_lock_another_process_holds_does_not_pass(tmp_path):
    # The usual state of this box: some other session holds the lock. A
    # marker set by hand must not pass on that (loose-ends, review of
    # 4b86576): only a holder among this process's ancestors counts. Here
    # the holder is a flock(1) this test starts, a child, not an ancestor.
    if shutil.which("flock") is None:
        pytest.skip("needs flock(1)")
    g = _guard()
    lock = tmp_path / "lock"
    lock.touch()
    holder = subprocess.Popen(["flock", str(lock), "sleep", "60"])
    try:
        for _ in range(200):
            if g.lock_holders(str(lock)):
                break
            time.sleep(0.01)
        assert g.lock_holders(str(lock)) == {holder.pid}
        why = g.unlocked_reason({"SOAR_GPULOCK": str(lock)}, str(lock))
        assert why and f"held by pid {holder.pid}" in why, why
        if shutil.which("node") is not None:
            js = subprocess.run(
                ["node", "--input-type=module", "-e",
                 f"import {{ unlockedReason }} from "
                 f"{(TOOLS / 'gpulock_guard.mjs').as_uri()!r};"
                 f"console.log(unlockedReason({{SOAR_GPULOCK: {str(lock)!r}}}, "
                 f"{str(lock)!r}));"],
                capture_output=True, text=True, check=True, timeout=30)
            assert f"held by pid {holder.pid}" in js.stdout, js.stdout
    finally:
        holder.kill()
        holder.wait()


def test_pytest_may_exempt_the_tool_it_drives():
    assert _guard().unlocked_reason({"SOAR_GPULOCK_EXEMPT": "pytest"}) is None
    assert _guard().unlocked_reason({"SOAR_GPULOCK_EXEMPT": "1"}) is not None
