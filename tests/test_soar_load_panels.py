"""Settings the UI drives while a field is still loading.

Opening a local file asks questions — which group, which units — and those
questions are menu panels. So the UI raises panels during a window in which
_releaseField has already nulled the renderer and the new one does not exist
yet, and anything a panel drives on its way up meets a half-built viewer.

The Quality panel's live preview is driven exactly that way (UI.open and
UI.close both sync it), and it went straight at the renderer. Loading a
multi-group file therefore died with "can't access property setHoldMode,
this.renderer is null", which the app reported as "Could not open this
field" — a nesting fix from the same morning looked like it had broken file
loading, when what had broken was the panel machinery underneath it. Demos
never showed it: a demo asks no questions, so no panel goes up mid-load
(Thomas, 2026-08-14).

The fix is not a skipped call. The mode is remembered on the viewer and
applied to whichever renderer appears, so it survives the gap rather than
depending on an argument about which panels can be open while a field loads.
That is what these pin, on the real prototype method against stubs — no GPU
and no DOM are involved in either half.

Skips only without node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VIEWER_JS = REPO / "web" / "soar" / "viewer.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not VIEWER_JS.exists(),
    reason="needs node and web/soar/viewer.js")


_JS = textwrap.dedent("""
    import { Viewer } from "%s";
    import * as K from "%s";

    // The half-built viewer of a load in progress: no renderer, and the
    // wake machinery present because setHoldMode ends by calling it.
    const halfBuilt = () => {
      const v = Object.create(Viewer.prototype);
      v.renderer = null;
      v._holdMode = K.DEFAULT_HOLD_MODE;
      v._sleeping = false;
      v.stop = false;
      v._disposed = false;
      v._marchPending = false;
      return v;
    };

    const out = {};

    // 1. What a load-time panel does, which used to throw.
    const loading = halfBuilt();
    try {
      loading.setHoldMode("live");
      out.duringLoad = { threw: false, remembered: loading._holdMode };
    } catch (err) {
      out.duringLoad = { threw: true, message: String(err.message || err) };
    }

    // 2. And the value a renderer is handed is defined even on a viewer
    // whose constructor never ran — the field and the getter's default say
    // the same thing, so no build order can feed it undefined.
    out.defaultOnABareObject = Object.create(Viewer.prototype).holdMode;

    // 3. The renderer that turns up afterwards is told, not left behind.
    const applied = [];
    loading.renderer = { setHoldMode: (m) => applied.push(m) };
    loading.setHoldMode(loading._holdMode);
    out.appliedAfterwards = applied;

    // 4. With a renderer present it still forwards, and still remembers.
    const flying = halfBuilt();
    const seen = [];
    flying.renderer = { setHoldMode: (m) => seen.push(m) };
    flying.setHoldMode("still");
    out.whileFlying = { forwarded: seen, remembered: flying._holdMode };

    process.stdout.write(JSON.stringify(out));
""") % (VIEWER_JS.as_posix(),
        (REPO / "web" / "soar" / "constants.js").as_posix())


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_a_panel_may_set_the_hold_mode_with_no_renderer_present(result):
    during = result["duringLoad"]
    assert during["threw"] is False, during.get("message")
    assert during["remembered"] == "live"


def test_the_mode_survives_the_gap_and_reaches_the_next_renderer(result):
    """Remembered, not skipped: the setting outlives the load."""
    assert result["appliedAfterwards"] == ["live"]


def test_the_mode_handed_to_a_new_renderer_is_never_undefined(result):
    """setHoldMode throws on an unknown mode, so an uninitialised field here
    would fail every load rather than only the ones that ask questions."""
    assert result["defaultOnABareObject"] == "still"


def test_a_renderer_that_is_present_is_still_told_directly(result):
    flying = result["whileFlying"]
    assert flying["forwarded"] == ["still"]
    assert flying["remembered"] == "still"
