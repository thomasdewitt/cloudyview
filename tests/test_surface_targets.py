"""The surface intermediates a size's stage chain needs, held and retired.

Both hosts hold, per render size, the intermediates the surface stage chain
alternates between (surface_stage_plan): one for each stage after the first,
up to two. Grow-only: a set that already holds enough is kept. What is
pinned here, on a counting stand-in for the GPU device (no GPU needed):

- The Python host holds the city's chain at once, and a second ensure of the
  same size allocates nothing.
- The browser's set is the final surface, the veil and the intermediates,
  made once per size.
- A set made and then not kept (two ensures of one size racing, each
  allocating: the later finds the other's set already covering its chain)
  is retired, and its textures are destroyed only after the queue reports
  the submitted work done — never while a frame in flight may still read
  them.

(Until 2026-11-09 the desert's tier-dependent chain drove these: its Low ran
one stage, its High three with an optical-depth target. The desert left this
line, and the city's chain is the same at every tier.)

The node half skips without node.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import soar_host
from cloudyview.soar_host import SoarRenderer

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
    r.surface_cs = None
    r._targets = None
    r._surface_dummy = _Tex([], "dummy")
    r._veil_dummy = _Tex([], "dummy")
    r._surface_layout = None
    return r


def test_python_host_without_a_view_holds_the_longest_chain():
    r = _renderer("city")
    t = r._ensure_targets((8, 8))
    assert t["plain"] is not None and t["mid"] is not None


def test_python_host_allocates_a_size_once():
    r = _renderer("city")
    r._ensure_targets((8, 8))
    first = len(r.device.made)
    r._ensure_targets((8, 8))
    assert len(r.device.made) == first


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
    r.scene = { city: true };
    r._surfaceDummy = { createView: () => ({}) };
    r._veilDummy = { createView: () => ({}) };
    r.surfaceLayout = {};
    const out = {};
    const targets = { w: 8, h: 8 };
    await r._ensureSurfaceTarget(targets);
    out.first = { made: made.length, nMid: targets.surface.nMid };
    await r._ensureSurfaceTarget(targets);
    out.again = made.length - out.first.made;
    // Two ensures of one fresh size, racing: both allocate; the later keeps
    // the earlier's set and retires its own.
    const raced = { w: 16, h: 16 };
    const before = made.length;
    await Promise.all([r._ensureSurfaceTarget(raced),
                       r._ensureSurfaceTarget(raced)]);
    out.raced = { made: made.length - before, destroyedNow: destroyed,
                  retirePending: pendingDone.length };
    // The GPU reports the submitted work done: only now are the retired
    // textures destroyed.
    for (const res of pendingDone) res();
    await new Promise((res) => setTimeout(res, 0));
    out.destroyedAfterDone = destroyed;
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


def test_browser_host_makes_a_size_once(js):
    full = len(soar_host.SURFACE_STAGES["city"])
    # The final surface, the veil, and the chain's intermediates.
    assert js["first"]["nMid"] == min(full - 1, 2), js["first"]
    assert js["first"]["made"] == 2 + min(full - 1, 2), js["first"]
    assert js["again"] == 0


def test_retired_textures_wait_for_the_gpu(js):
    per_set = js["first"]["made"]
    # Both racers allocated a whole set; the one not kept is retired but
    # destroyed nothing while the queue had not reported its work done ...
    assert js["raced"]["made"] == 2 * per_set, js["raced"]
    assert js["raced"]["destroyedNow"] == 0, js["raced"]
    assert js["raced"]["retirePending"] >= 1, js["raced"]
    # ... and destroyed that set's textures once it had.
    assert js["destroyedAfterDone"] == per_set
