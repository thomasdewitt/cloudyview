"""The surface intermediates grow with the active stage chain, never shrink.

Both hosts hold, per render size, the intermediates the ACTIVE surface
stage chain alternates between (surface_stage_plan): none for the simplified
desert's single stage at Low, one for each stage after the first beyond
that, up to two — and the full desert's optical-depth target (TAUS_STAGE),
which only its chain writes. Grow-only: a size used only at Low never
allocates an intermediate or the optical depths, and after a size's first
growth a tier flip never reallocates it. What is pinned here, on a
counting stand-in for the GPU device (no GPU needed):

- Low allocates no intermediate and no optical-depth target; Low -> High
  grows once, by both; High -> Low -> High allocates nothing more.
- A retired set's textures are destroyed only after the queue reports the
  submitted work done — never while a frame in flight may still read them.
- The tier can rise DURING the allocation's await (the meter is
  fire-and-forget; the governor or the menu can change tier meanwhile):
  the ensure still returns a set that holds every slot the chain live at
  that moment writes or reads (loose-ends' race, 2026-09-25), the optical
  depths included.
- A stage chain that needs the optical depths on a set without them
  raises / throws rather than bind a stand-in.
- A set made and then not kept (two ensures of one size racing, each
  allocating: the later finds the other's set already covering its chain)
  is retired the same way, after the GPU is done — the city's chain, which
  is the same at every tier, retires only by this path.

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
    grown = r.device.made[first:]
    assert sorted(grown) == sorted(
        [surf] * min(full - 1, 2) + [soar_host.TAUS_FORMAT]), grown
    after = len(r.device.made)
    for lite in (True, False, True, False):
        r._ensure_targets((8, 8), _view(lite=lite))
    assert len(r.device.made) == after


def test_python_host_refuses_the_flora_stage_without_its_depths():
    # A set grown for Low only has no optical-depth target: the full
    # desert's chain on it raises rather than bind a stand-in.
    r = _renderer("desert")
    t = r._ensure_targets((8, 8), _view(lite=True))
    t["plain"] = t["mid"] = _Tex([], soar_host.SURFACE_FORMAT)
    with pytest.raises(RuntimeError, match="optical-depth"):
        r._surface_stages(t, _view(lite=False))


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
    const tausOf = () => made.filter((l) => l.endsWith("-taus")).length;
    for (const tier of ["low", "high", "low", "high", "minimal", "max"]) {
      r.qualityTier = tier;
      const before = made.length;
      const destroyedBefore = destroyed;
      await r._ensureSurfaceTarget(targets);
      out.steps.push({ tier, made: made.length - before,
                       nMid: targets.surface.nMid, taus: tausOf(),
                       destroyedNow: destroyed - destroyedBefore,
                       retirePending: pendingDone.length });
    }
    // A set without the optical depths, handed the full desert's chain:
    // the encoder throws rather than bind a stand-in.
    try {
      r._encodeMarch({}, { surface: { ...targets.surface, tausView: null } },
                     {}, {}, K.SURFACE_STAGES.desert.map((e) => [e, {}]));
      out.missingTaus = "encoded";
    } catch (e) {
      out.missingTaus = String(e.message);
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
    assert low["nMid"] == 0 and low["made"] == 2 and low["taus"] == 0, low
    assert not any("plain" in t or "mid" in t for t in js["lowTextures"])
    # Low -> High: one growth, to the full desert's intermediates and its
    # optical depths.
    full = len(soar_host.SURFACE_STAGES["desert"])
    assert high["nMid"] == min(full - 1, 2) and high["made"] > 0, high
    assert high["taus"] == 1, high
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


def test_browser_host_refuses_the_flora_stage_without_its_depths(js):
    assert "optical-depth" in js["missingTaus"], js["missingTaus"]


_JS_RACE = textwrap.dedent("""
    import * as K from "%s";
    import { Renderer, surfaceStagePlan } from "%s";

    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 16, TEXTURE_BINDING: 4 };
    const r = Object.create(Renderer.prototype);
    let flipTo = null;          // the tier to switch to inside the await
    const device = {
      pushErrorScope() {},
      async popErrorScope() {
        if (flipTo) { r.qualityTier = flipTo; flipTo = null; }
        return null;
      },
      createTexture(d) { return { label: d.label, createView: () => ({ of: d.label }),
                                  destroy() {} }; },
      createBindGroup() { return { group: true }; },
      queue: { onSubmittedWorkDone: () => Promise.resolve() },
    };
    r.device = device;
    r.scene = { desert: true };
    r._surfaceDummy = { createView: () => ({}) };
    r._veilDummy = { createView: () => ({}) };
    r.surfaceLayout = {};
    const out = [];
    // Every slot the live chain writes or reads must exist.
    const check = (targets) => {
      const s = targets.surface;
      const slot = { final: [s.view, null], a: [s.plainView, s.afterPlain],
                     b: [s.midView, s.afterMid] };
      const entries = r._stageEntries();
      const bad = [];
      surfaceStagePlan(entries.length).forEach(([w, rd], k) => {
        if (!slot[w][0]) bad.push(`stage ${k} writes ${w}: no view`);
        if (rd !== null && !slot[rd][1]) bad.push(`stage ${k} reads ${rd}: no group`);
      });
      if (entries.includes(K.TAUS_STAGE) && !s.tausView) {
        bad.push("the chain writes the optical depths: no view");
      }
      return { tier: r.qualityTier, entries, nMid: s.nMid, bad };
    };
    for (const [start, during] of [["low", "high"], ["low", "max"],
                                   ["high", "low"], ["minimal", "medium"]]) {
      const targets = { w: 8, h: 8 };
      r.qualityTier = start;
      flipTo = during;
      await r._ensureSurfaceTarget(targets);
      out.push({ start, during, ...check(targets) });
    }
    console.log(JSON.stringify(out));
""") % ((REPO / "web" / "soar" / "constants.js").as_uri(), RENDERER_JS.as_uri())


def test_a_tier_rise_during_the_allocation_still_leaves_every_slot():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _JS_RACE],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for case in json.loads(out.stdout):
        assert case["tier"] == case["during"], case
        assert not case["bad"], case


_JS_RACE_RETIRE = textwrap.dedent("""
    import { Renderer } from "%s";

    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 16, TEXTURE_BINDING: 4 };
    const made = [];
    let destroyed = 0;
    let pendingDone = [];
    const device = {
      pushErrorScope() {}, async popErrorScope() { return null; },
      createTexture(d) {
        made.push(d.label);
        return { label: d.label, createView: () => ({}),
                 destroy: () => { destroyed++; } };
      },
      createBindGroup() { return {}; },
      queue: { onSubmittedWorkDone: () => new Promise((res) => {
        pendingDone.push(res); }) },
    };
    const r = Object.create(Renderer.prototype);
    r.device = device;
    r.scene = { city: true };
    r.qualityTier = "high";
    r._surfaceDummy = { createView: () => ({}) };
    r._veilDummy = { createView: () => ({}) };
    r.surfaceLayout = {};
    const one = { w: 8, h: 8 };
    await r._ensureSurfaceTarget(one);
    const perSet = made.length;
    // Two ensures of one fresh size, racing: both allocate; the later keeps
    // the earlier's set and retires its own.
    const raced = { w: 16, h: 16 };
    await Promise.all([r._ensureSurfaceTarget(raced),
                       r._ensureSurfaceTarget(raced)]);
    const out = { perSet, made: made.length - perSet, destroyedNow: destroyed,
                  retirePending: pendingDone.length };
    for (const res of pendingDone) res();
    await new Promise((res) => setTimeout(res, 0));
    out.destroyedAfterDone = destroyed;
    console.log(JSON.stringify(out));
""") % RENDERER_JS.as_uri()


def test_a_set_lost_to_a_race_is_retired_after_the_gpu():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e",
                          _JS_RACE_RETIRE],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    # Both racers allocated a whole set; the one not kept is retired but
    # destroyed nothing while the queue had not reported its work done ...
    assert got["made"] == 2 * got["perSet"], got
    assert got["destroyedNow"] == 0, got
    assert got["retirePending"] >= 1, got
    # ... and destroyed that set's textures once it had.
    assert got["destroyedAfterDone"] == got["perSet"], got
