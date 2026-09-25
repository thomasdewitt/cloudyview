"""The surface intermediates grow with the active stage chain, never shrink.

Both hosts hold, per render size, the intermediates the ACTIVE surface
stage chain alternates between (surface_stage_plan): none for the simplified
desert's single stage at Low, two for the full desert's three at High.
Grow-only: a session that stays at Low never allocates the two it does not
use, and after the first growth a tier flip never reallocates. What is
pinned here, on a counting stand-in for the GPU device (no GPU needed):

- Low allocates no intermediate; Low -> High grows once; High -> Low ->
  High allocates nothing more.
- A retired set's textures are destroyed only after the queue reports the
  submitted work done — never while a frame in flight may still read them.

The node half skips without node.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import soar_host
from cloudyview.soar_host import SoarRenderer, ViewState

REPO = Path(__file__).resolve().parents[1]
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"


class _Tex:
    def __init__(self, log, fmt):
        self.fmt = fmt
        log.append(fmt)

    def create_view(self):
        return self


class _Device:
    def __init__(self):
        self.made = []

    def create_texture(self, *, size, dimension, format, usage):
        return _Tex(self.made, format)

    def create_bind_group(self, *, layout, entries):
        return ("group", len(entries))


def _renderer(surface):
    import wgpu
    r = object.__new__(SoarRenderer)
    r.wgpu = wgpu
    r.device = _Device()
    r.surface = surface
    r._targets = None
    r._surface_dummy = _Tex([], "dummy")
    r._veil_dummy = _Tex([], "dummy")
    r._surface_layout = None
    return r


def _view(lite):
    return ViewState(camera_position=[0.0, 0.0, 1.0], azimuth=0.0,
                     elevation=0.0, fov=60.0, output_size=(8, 8),
                     render_size=(8, 8), desert_lite=lite)


def test_python_host_grows_once_and_never_shrinks():
    r = _renderer("desert")
    surf = soar_host.SURFACE_FORMAT
    t = r._ensure_targets((8, 8), _view(lite=True))
    assert t["plain"] is None and t["mid"] is None
    first = len(r.device.made)
    full = len(soar_host.SURFACE_STAGES["desert"])
    r._ensure_targets((8, 8), _view(lite=False))
    grown = len(r.device.made) - first
    assert grown == min(full - 1, 2), grown
    assert r.device.made[first:] == [surf] * grown
    after = len(r.device.made)
    for lite in (True, False, True, False):
        r._ensure_targets((8, 8), _view(lite=lite))
    assert len(r.device.made) == after


def test_python_host_without_a_view_holds_the_longest_chain():
    r = _renderer("city")
    t = r._ensure_targets((8, 8))
    assert t["plain"] is not None and t["mid"] is not None


_JS = textwrap.dedent("""
    import * as K from "%s";
    import { Renderer } from "%s";

    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 16, TEXTURE_BINDING: 4 };
    const made = [];
    let destroyed = 0;
    let pendingDone = [];
    const device = {
      pushErrorScope() {}, async popErrorScope() { return null; },
      createTexture(d) {
        const t = { label: d.label, createView: () => ({}),
                    destroy: () => { destroyed++; } };
        made.push(d.label);
        return t;
      },
      createBindGroup() { return {}; },
      queue: { onSubmittedWorkDone: () => new Promise((res) => {
        pendingDone.push(res); }) },
    };
    const r = Object.create(Renderer.prototype);
    r.device = device;
    r.scene = { desert: true };
    r._surfaceDummy = { createView: () => ({}) };
    r._veilDummy = { createView: () => ({}) };
    r.surfaceLayout = {};
    const targets = { w: 8, h: 8 };
    const out = { steps: [] };
    for (const tier of ["low", "high", "low", "high", "minimal", "max"]) {
      r.qualityTier = tier;
      const before = made.length;
      const destroyedBefore = destroyed;
      await r._ensureSurfaceTarget(targets);
      out.steps.push({ tier, made: made.length - before,
                       nMid: targets.surface.nMid,
                       destroyedNow: destroyed - destroyedBefore,
                       retirePending: pendingDone.length });
    }
    // The GPU reports the submitted work done: only now are retired
    // textures destroyed.
    for (const res of pendingDone) res();
    await new Promise((res) => setTimeout(res, 0));
    out.destroyedAfterDone = destroyed;
    out.lowTextures = made.slice(0, out.steps[0].made);
    console.log(JSON.stringify(out));
""") % ((REPO / "web" / "soar" / "constants.js").as_uri(), RENDERER_JS.as_uri())


@pytest.fixture(scope="module")
def js():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _JS],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_browser_host_grows_once_on_a_tier_flip(js):
    low, high, *rest = js["steps"]
    # Low: the final surface and the veil only.
    assert low["nMid"] == 0 and low["made"] == 2, low
    assert not any("plain" in t or "mid" in t for t in js["lowTextures"])
    # Low -> High: one growth, to the full desert's intermediates.
    full = len(soar_host.SURFACE_STAGES["desert"])
    assert high["nMid"] == min(full - 1, 2) and high["made"] > 0, high
    # Every flip after that: nothing made.
    for step in rest:
        assert step["made"] == 0, step


def test_retired_textures_wait_for_the_gpu(js):
    # The growth retired Low's set, but destroyed nothing while the queue
    # had not reported its work done ...
    assert all(s["destroyedNow"] == 0 for s in js["steps"])
    assert js["steps"][1]["retirePending"] >= 1
    # ... and destroyed Low's two textures once it had.
    assert js["destroyedAfterDone"] == 2
