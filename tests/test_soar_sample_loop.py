"""Max's eight-samples-per-frame loop, which no test here can render.

The tier that marches eight times for every frame it shows is browser-side
code on a GPU this suite cannot reach, so what is checkable is the shape of
what it submits — and the shape is where the failure modes are:

- eight samples must be eight SUBMISSIONS, not one pass eight times longer.
  Metal kills a fragment pass by duration, so a tier this dear only exists
  safely because each command buffer is one ordinary frame's worth of work.
- only the last sample may reach the screen. Presenting each would strobe.
- the blend weights must be a running mean over the samples of THIS frame
  (1, 1/2, 2/3, ...), not eight restarts and not eight full-weight writes.
- the samples must be decorrelated. The frame index seeds both the jitter and
  the dither, and the viewer's own index steps by one per frame, so a stride
  of one would make sample s the twin of a later frame's first sample.
- and every other tier must still submit exactly once, unchanged.

Renderer is driven through a stub device: the loop's arithmetic and encode
order are pure, and everything they touch (queue writes, encoders, passes) is
recorded rather than executed. Skips only without node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not RENDERER_JS.exists(),
    reason="needs node and web/soar/renderer.js")


_JS = textwrap.dedent("""
    import { Renderer } from "%s";
    import * as K from "%s";

    const log = [];
    const texture = (label) => ({ label, createView: () => ({ of: label }) });

    // Enough of a device to record what a frame would have submitted.
    const device = {
      limits: { maxTextureDimension2D: 8192 },
      queue: {
        writeBuffer(buf, offset, data) {
          // Row 4.w is the frame index (jitter and dither seed); row 10.x/y
          // are the sampling flags. Those three are what a sample changes.
          log.push(buf.label === "uniform"
            ? { write: "uniform", frameIndex: data[19],
                subpixel: data[40], jitterScale: data[41] }
            : { write: buf.label, data: Array.from(data.slice(0, 4)) });
        },
        submit(buffers) { log.push({ submit: buffers.length }); },
      },
      createCommandEncoder: () => ({
        beginRenderPass: ({ colorAttachments }) => {
          log.push({ pass: colorAttachments[0].view.of });
          return { setPipeline() {}, setBindGroup() {}, draw() {}, end() {} };
        },
        finish: () => ({}),
      }),
      createBindGroup: () => ({}),
    };

    const targets = {
      w: 8, h: 8,
      sample: texture("sample"),
      accumA: texture("accumA"),
      accumB: texture("accumB"),
      accumBindGroups: [{}, {}],
    };

    function stubRenderer(tier) {
      const r = Object.create(Renderer.prototype);
      r.device = device;
      r.scene = { bmin: [0, 0, 0], bmax: [1000, 1000, 1000], minVoxelM: 100 };
      r.periodic = false;
      r.qualityTier = tier;
      r._flightRenderScale = K.QUALITY_PRESETS[tier].renderScale;
      r._cameraMoving = false;
      r._holdLadder = null;
      r._holdRung = 0;
      r._holdCapped = false;
      r._holdMode = K.DEFAULT_HOLD_MODE;
      r.renderScale = null; r.stepFactor = null;
      r.lightStepFactor = null; r.maxLightSteps = null;
      r._resetAccumulation();
      r._applyEffectiveQuality();
      r.motionBlendAlpha = K.DEFAULT_MOTION_BLEND_ALPHA;
      r.motionResetTranslationM = 50.0;
      r._accumWeights = new Float32Array(4);
      r.uniformBuf = { label: "uniform" };
      r.accumUniformBuf = { label: "accum" };
      r.rayBindGroup = {};
      r.accumPipeline = {};
      // The GPU half, stubbed at the seams drawFrame uses.
      r._targetsFor = async () => targets;
      r._rayPipeline = () => ({});
      r._blitPipeline = () => ({});
      r._sceneState = () => ({
        bmin: r.scene.bmin, bmax: r.scene.bmax,
        dtView: r.dtView, dtLight: r.dtLight, periodic: false,
        oceanZ: 0, oceanReflectance: K.DEFAULT_OCEAN_REFLECTANCE,
        oceanFifDx: 1, oceanTileExtent: 1, oceanEnabled: false, oceanMaxLod: 0,
      });
      r._encodeBlit = (enc, src, view, fmt, exact) => {
        log.push({ blit: src.label });
      };
      return r;
    }

    const camera = {
      position: [0, 0, 500], azimuth: 0, elevation: 0, fov: 60,
    };

    async function frame(r, frameIndex, dx) {
      camera.position = [dx, 0, 500];
      await r.drawFrame(
        { of: "canvas" }, "rgba8unorm", [8, 8],
        { camera, frameIndex, sunAzimuth: 20, sunElevation: 55 },
        { deltaSeconds: 1 / 60 });
    }

    const runs = {};
    for (const tier of ["high", "max"]) {
      log.length = 0;
      const r = stubRenderer(tier);
      await frame(r, 0, 0);            // first frame of a held view
      const first = log.slice();
      log.length = 0;
      await frame(r, 1, 0);            // second, same view: it accumulates
      runs[tier] = {
        spp: r.sppPerFrame,
        first, second: log.slice(),
        accumCount: r.accumCount,
      };
    }

    process.stdout.write(JSON.stringify(runs));
""") % (RENDERER_JS.as_posix(),
        (REPO / "web" / "soar" / "constants.js").as_posix())


@pytest.fixture(scope="module")
def runs():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def submits(entries):
    return [e for e in entries if "submit" in e]


def blits(entries):
    return [e for e in entries if "blit" in e]


def accum_weights(entries):
    return [e["data"][:2] for e in entries if e.get("write") == "accum"]


def frame_indices(entries):
    """Row 4.w of each uniform write — the jitter and dither seed."""
    return [e["frameIndex"] for e in entries if e.get("write") == "uniform"]


def test_every_other_tier_still_submits_one_frame_at_a_time(runs):
    high = runs["high"]
    assert high["spp"] == 1
    assert len(submits(high["first"])) == 1
    assert len(blits(high["first"])) == 1


def test_max_submits_one_command_buffer_per_sample(runs):
    """Eight frames of work, never one pass eight times longer."""
    mx = runs["max"]
    assert mx["spp"] == 8
    assert len(submits(mx["first"])) == 8
    assert all(e["submit"] == 1 for e in submits(mx["first"]))


def test_only_the_last_sample_reaches_the_screen(runs):
    mx = runs["max"]
    for entries in (mx["first"], mx["second"]):
        assert len(blits(entries)) == 1
        # ...and it is the final thing encoded before the last submit.
        assert entries.index(blits(entries)[0]) > entries.index(submits(entries)[-2])


def test_the_samples_of_one_frame_are_a_running_mean(runs):
    weights = accum_weights(runs["max"]["first"])
    assert len(weights) == 8
    assert weights[0] == pytest.approx([0.0, 1.0])          # a clean start
    for k, (prev, new) in enumerate(weights[1:], start=1):
        assert prev == pytest.approx(k / (k + 1)), k
        assert new == pytest.approx(1 / (k + 1)), k


def test_a_second_frame_of_the_same_view_keeps_accumulating(runs):
    """32 parked samples is four frames at Max, not four restarts."""
    mx = runs["max"]
    weights = accum_weights(mx["second"])
    assert weights[0][0] > 0.0            # continues rather than restarting
    assert mx["accumCount"] == 16


def test_the_samples_of_one_frame_do_not_share_a_jitter_seed(runs):
    seeds = frame_indices(runs["max"]["first"])
    assert len(seeds) == 8
    assert len(set(seeds)) == len(seeds)
    # Strided far enough that a later frame's samples cannot collide either.
    steps = sorted(b - a for a, b in zip(seeds, seeds[1:]))
    assert min(steps) > 64
