"""Every shader soar ships must compile in a browser, not only under wgpu.

naga (wgpu's compiler, which every other test here goes through) accepts
some WGSL the spec does not — `a ^ b * c` shipped on master (abbc9dc,
2026-09-23) and no Chrome, Edge or Safari could build the shader while the
suite stayed green. This drives tools/tint_check.mjs, which compiles every
specialization with Chrome's own compiler (Tint) through the DevTools
protocol.

It needs a Chrome with --remote-debugging-port (CLOUDYVIEW_CDP_PORT, default
9333) and a hardware WebGPU adapter. On the Fedora box:

    google-chrome --user-data-dir=/tmp/soar-cdp --remote-debugging-port=9333 \
        --enable-features=Vulkan --ozone-platform=x11

Without one it skips, and says why — it never passes vacuously.
"""

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CDP_PORT = os.environ.get("CLOUDYVIEW_CDP_PORT", "9333")


def _cdp_reachable() -> bool:
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{CDP_PORT}/json/version", timeout=1).read()
        return True
    except OSError:
        return False


def test_every_specialization_compiles_under_tint():
    """The tool serves this checkout's web/ itself, on a fresh port, and
    checks the served shader is this tree's before compiling anything."""
    if shutil.which("node") is None:
        pytest.skip("needs node")
    if not _cdp_reachable():
        pytest.skip(f"no Chrome DevTools endpoint on port {CDP_PORT} "
                    "(see this module's docstring for the Chrome command)")
    proc = subprocess.run(
        ["node", str(REPO / "tools" / "tint_check.mjs"),
         "--web", str(REPO / "web"), "--cdp", CDP_PORT],
        capture_output=True, text=True, timeout=600,
        # pytest is not held to tools/gpulock (Thomas runs it directly), so
        # the tool it drives is exempted for this child (gpulock_guard).
        env={**os.environ, "SOAR_GPULOCK_EXEMPT": "pytest"})
    if proc.returncode == 2:
        pytest.skip("tint_check could not run: "
                    + (proc.stderr or proc.stdout).strip()[-400:])
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert " 0 failed" in proc.stdout
