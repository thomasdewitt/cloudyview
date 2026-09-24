"""Removing a nest re-bakes the sun-tau cache.

The light march integrates the fine nest at its own resolution, so a cache
baked with a nest carries its shadows; a bake still in flight when the nest
goes would finish its remaining slices without it. removeNest therefore does
what setSun and togglePeriodic do: invalidate, then start a fresh bake, after
the bind group has dropped the nest. Driven under node against a stub viewer.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

_JS = textwrap.dedent("""
    import { Viewer } from "%s";
    const run = (hadNest) => {
      const calls = [];
      const self = {
        scene: { removeNest: () => hadNest },
        renderer: {
          refreshBindGroup: () => calls.push("refreshBindGroup"),
          invalidateLightCache: () => calls.push("invalidateLightCache"),
          startLightBake: () => calls.push("startLightBake"),
          resetAccumulation: () => calls.push("resetAccumulation"),
        },
        ui: { say() {}, open() {} },
        _viewKwargs: () => ({}),
        _wake() {},
      };
      Viewer.prototype.removeNest.call(self);
      return calls;
    };
    process.stdout.write(JSON.stringify({ removed: run(true), none: run(false) }));
""") % (REPO / "web" / "soar" / "viewer.js").as_posix()


@pytest.fixture(scope="module")
def calls():
    proc = subprocess.run(["node", "--input-type=module", "-e", _JS],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_removing_a_nest_rebakes_after_the_bind_group_drops_it(calls):
    c = calls["removed"]
    assert "invalidateLightCache" in c and "startLightBake" in c
    assert c.index("refreshBindGroup") < c.index("invalidateLightCache") \
        < c.index("startLightBake")


def test_no_nest_no_rebake(calls):
    assert calls["none"] == []
