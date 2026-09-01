"""A still's progress bar has to move at the speed of the actual work.

Two bugs shaped this contract. First the bar did not move at all:
`renderAccumulated` submitted every accumulation pass in one unbroken run of
microtasks, so the browser had no chance to paint between them. Then it moved
at the wrong speed: drawFrame only SUBMITS work, so a loop that counts
submissions counts the CPU racing ahead of the GPU — the bar reached the
march's whole share almost immediately and then sat there while the
read-back's mapAsync waited out all the actual marching (ebc1065).

So this pins the reporting contract of the real module, under node, with a
stub renderer and a stub GPU queue:

- a report is made after every pass, and only after the GPU sync for that
  pass (`queue.onSubmittedWorkDone`) has resolved — `done` counts passes the
  GPU has FINISHED, and the await is also what frees the event loop so the
  browser can paint the bar at all,
- the final report is always the last pass, so the bar reaches the end of
  the marching stage,
- asking for no progress (the video path, which paces itself on a read-back
  per frame) never touches the device and never waits,
- and a still's fractions give the march STILL_MARCH_SHARE of the bar, per
  finished pass, with the read-back owning the rest.

No GPU is needed to count callbacks, which is the point. Skips only without
node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CAPTURE_JS = REPO / "web" / "soar" / "capture.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not CAPTURE_JS.exists(),
    reason="needs node and web/soar/capture.js")


_JS = textwrap.dedent("""
    import { renderAccumulated, renderStill } from "%s";

    // Enough of WebGPU for the offline target and the read-back.
    globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, COPY_SRC: 2 };
    globalThis.GPUBufferUsage = { COPY_DST: 1, MAP_READ: 2 };
    globalThis.GPUMapMode = { READ: 1 };
    globalThis.ImageData = class {
      constructor(w, h) {
        this.width = w; this.height = h;
        this.data = new Uint8ClampedArray(w * h * 4);
      }
    };
    const makeDevice = () => ({
      syncs: 0,
      limits: { maxTextureDimension2D: 8192 },
      createTexture: () => ({ createView: () => ({}), destroy() {} }),
      createBuffer: ({ size }) => ({
        async mapAsync() {}, getMappedRange: () => new ArrayBuffer(size),
        unmap() {}, destroy() {},
      }),
      createCommandEncoder: () => ({
        copyTextureToBuffer() {}, finish: () => ({}),
      }),
      queue: {
        submit() {},
        async onSubmittedWorkDone() { this.device.syncs += 1; },
      },
    });
    const stubDevice = () => {
      const device = makeDevice();
      device.queue.device = device;   // so the sync can count itself
      return device;
    };

    const stubRenderer = (device) => ({
      device,
      passes: 0,
      // A report must only happen after the sync for its pass has resolved.
      syncsAtReport: [],
      resetAccumulation() {},
      setQualityTier() {},
      setRenderScale() {},
      flightRenderScale: 1.0,
      qualityTier: "high",
      lightCacheMode: "auto",
      skyProbeMode: "auto",
      parkedSppOverride: null,
      lightBakePending: false,
      stepLightBake() { return true; },
      async drawFrame() { this.passes += 1; },
    });

    const run = async (frames, { report = true } = {}) => {
      const device = stubDevice();
      const renderer = stubRenderer(device);
      const reports = [];
      await renderAccumulated(
        renderer, {}, [8, 8], { frameIndex: 0 }, frames, null,
        report ? (done, total) => {
          reports.push([done, total]);
          renderer.syncsAtReport.push(device.syncs);
        } : null);
      return { reports, passes: renderer.passes, syncs: device.syncs,
               syncsAtReport: renderer.syncsAtReport };
    };

    const reported = await run(8);
    // The video path, which asks for nothing and must wait for nothing.
    const silent = await run(8, { report: false });

    // The device with no renderer attached: the silent path must never
    // reach for renderer.device at all — pin that by omitting it.
    const bare = stubRenderer(undefined);
    await renderAccumulated(
      bare, {}, [8, 8], { frameIndex: 0 }, 8, null, null);
    const silentNoDevice = { passes: bare.passes };

    // And the whole still, to pin the fractions the bar is given.
    // "minimal" is 8 parked samples of one pass each — the recipe comes from
    // the tier rather than a frame count argument.
    const device = stubDevice();
    const fractions = [];
    await renderStill(
      device, stubRenderer(device), { frameIndex: 0 }, [4, 3], "minimal",
      null, (fraction) => fractions.push(fraction));

    process.stdout.write(JSON.stringify(
      { reported, silent, silentNoDevice, fractions }));
""") % CAPTURE_JS.as_posix()


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_every_pass_is_reported(result):
    reported = result["reported"]
    assert reported["passes"] == 8
    assert reported["reports"] == [[i, 8] for i in range(1, 9)]


def test_each_report_waits_for_its_gpu_sync(result):
    """The sync IS the fix: `done` counts passes the GPU has finished, and
    the await is what frees the event loop so the bar can be painted."""
    reported = result["reported"]
    assert reported["syncs"] == 8
    assert reported["syncsAtReport"] == list(range(1, 9))


def test_the_march_ends_on_a_full_report(result):
    assert result["reported"]["reports"][-1] == [8, 8]


def test_no_progress_asked_for_means_no_waiting(result):
    silent = result["silent"]
    assert silent["passes"] == 8
    assert silent["reports"] == []
    assert silent["syncs"] == 0
    # And the device is never even reached for — the video path's renderer
    # need not carry one.
    assert result["silentNoDevice"]["passes"] == 8


def test_a_still_reports_a_fraction_per_pass_and_then_the_read_back(result):
    """A bar and one sentence from the caller — the stages do not narrate."""
    fractions = result["fractions"]
    # minimal's 8 parked samples: one report per finished pass, then the
    # read-back's report at the top of the march's share.
    assert len(fractions) == 9
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(0.95 / 8)
    # The marching owns 0.95 of the bar; the read-back and the PNG encode own
    # the rest, and the read-back is the one that actually waits on the GPU.
    assert fractions[-2] == pytest.approx(0.95)
    assert fractions[-1] == pytest.approx(0.95)
