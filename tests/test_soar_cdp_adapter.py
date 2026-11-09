"""soar_cdp refuses a Chrome that renders WebGPU in software.

Chrome falls back to SwiftShader when its Vulkan flags are missing or wrong,
and Mesa's llvmpipe/lavapipe answer as a GPU too; every still hashed and
pass timed on one is fiction against the hardware ones. tint_check and
pinned_still call requireHardwareAdapter, whose verdict is softwareAdapter's,
pinned here on the adapter infos those rasterizers report. Needs node.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOAR_CDP = REPO / "tools" / "soar_cdp.mjs"

CASES = {
    "rtx5080": ({"vendor": "nvidia", "architecture": "blackwell"}, False),
    "swiftshader": ({"vendor": "google", "architecture": "swiftshader",
                     "description": "SwiftShader Device (Subzero)"}, True),
    "fallback": ({"vendor": "", "architecture": "",
                  "isFallbackAdapter": True}, True),
    "llvmpipe": ({"vendor": "mesa", "architecture": "",
                  "description": "llvmpipe (LLVM 19.1.0, 256 bits)"}, True),
    "lavapipe": ({"vendor": "mesa", "architecture": "lavapipe"}, True),
    "none": (None, True),
}


def test_software_adapters_are_refused():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    js = textwrap.dedent("""
        import { softwareAdapter } from "%s";
        const cases = JSON.parse(process.argv[1]);
        const out = {};
        for (const [k, [info]] of Object.entries(cases)) {
          out[k] = softwareAdapter(info);
        }
        console.log(JSON.stringify(out));
    """) % SOAR_CDP.as_uri()
    got = json.loads(subprocess.run(
        ["node", "--input-type=module", "-e", js, json.dumps(CASES)],
        capture_output=True, text=True, check=True).stdout)
    for name, (_, software) in CASES.items():
        assert (got[name] is not None) == software, (name, got[name])
