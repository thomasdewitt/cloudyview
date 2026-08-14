"""A still's progress bar has to move while the still is being made.

It did not. `renderAccumulated` submitted every accumulation pass in one
unbroken run of microtasks, so the browser had no chance to paint between
them: the bar showed 0 for the whole march and then jumped to done. Nothing
was wrong with the picture, only with the account of it — which is exactly the
class of bug that is invisible to an image golden.

So this pins the reporting contract of the real module, under node, with a
stub renderer whose passes cost a controlled amount of fake time:

- every report is followed by a wait for a paint (that wait is the fix),
- the final report is always the last pass, so the bar reaches the end of the
  marching stage,
- reporting is time-gated (twice a second), not per-pass — a card fast enough
  to finish a pass inside a frame interval must not be made to stop for the
  compositor after each one, so that the account of the work cannot slow the
  work,
- and asking for no progress (the video path, which paces itself on a
  read-back per frame) still submits with no waits at all.

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

    // A clock the test drives, so "a pass costs 600 ms" is a fact rather than
    // a hope about the machine the suite runs on.
    let clock = 0;
    Object.defineProperty(globalThis, "performance", {
      value: { now: () => clock }, configurable: true, writable: true,
    });
    let paints = 0;
    globalThis.requestAnimationFrame = (fn) => {
      paints += 1;
      // A real frame callback lands on a later task, not a later microtask —
      // and a microtask would not let a browser paint, so it would not be a
      // fair stand-in for one.
      setTimeout(() => fn(clock), 0);
      return paints;
    };

    const stubRenderer = (passMs) => ({
      passes: 0,
      resetAccumulation() {},
      setQualityTier() {},
      setRenderScale() {},
      flightRenderScale: 1.0,
      qualityTier: "high",
      async drawFrame() { this.passes += 1; clock += passMs; },
    });

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
    const stubDevice = {
      limits: { maxTextureDimension2D: 8192 },
      createTexture: () => ({ createView: () => ({}), destroy() {} }),
      createBuffer: ({ size }) => ({
        async mapAsync() {}, getMappedRange: () => new ArrayBuffer(size),
        unmap() {}, destroy() {},
      }),
      createCommandEncoder: () => ({
        copyTextureToBuffer() {}, finish: () => ({}),
      }),
      queue: { submit() {} },
    };

    const run = async (frames, passMs, { report = true, hidden = false } = {}) => {
      clock = 0; paints = 0;
      globalThis.document = {
        visibilityState: hidden ? "hidden" : "visible",
        addEventListener() {}, removeEventListener() {},
      };
      const renderer = stubRenderer(passMs);
      const reports = [];
      await renderAccumulated(
        renderer, {}, [8, 8], { frameIndex: 0 }, frames, null,
        report ? (done, total) => reports.push([done, total]) : null);
      return { reports, paints, passes: renderer.passes };
    };

    // A slow card: every pass costs longer than the reporting interval.
    const slow = await run(8, 600);
    // A fast one: eight passes inside a single interval.
    const fast = await run(8, 1);
    // The video path, which asks for nothing and must wait for nothing.
    const silent = await run(8, 600, { report: false });
    // A tab nobody is looking at: no frame callbacks are coming, so none are
    // waited for, and the capture finishes anyway.
    const hidden = await run(8, 600, { hidden: true });

    // And the whole still, to pin the fractions the bar is given.
    clock = 0; paints = 0;
    globalThis.document = { visibilityState: "visible",
                            addEventListener() {}, removeEventListener() {} };
    const fractions = [];
    await renderStill(
      stubDevice, stubRenderer(600), { frameIndex: 0 }, [4, 3], 6, null,
      (fraction) => fractions.push(fraction));

    process.stdout.write(
      JSON.stringify({ slow, fast, silent, hidden, fractions }));
""") % CAPTURE_JS.as_posix()


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_every_pass_is_reported_when_each_one_is_slow(result):
    slow = result["slow"]
    assert slow["passes"] == 8
    assert slow["reports"] == [[i, 8] for i in range(1, 9)]


def test_each_report_waits_for_a_paint(result):
    """The wait IS the fix: without it the bar cannot be redrawn."""
    for case in ("slow", "fast"):
        assert result[case]["paints"] == len(result[case]["reports"]), case


def test_the_march_always_ends_on_a_full_report(result):
    for case in ("slow", "fast"):
        reports = result[case]["reports"]
        assert reports[-1] == [8, 8], case
        counts = [done for done, _ in reports]
        assert counts == sorted(counts), case


def test_reporting_is_time_gated_not_per_pass(result):
    """Eight passes inside one interval cost one stop, not eight."""
    fast = result["fast"]
    assert fast["passes"] == 8
    assert fast["reports"] == [[8, 8]]


def test_a_hidden_tab_is_never_waited_on(result):
    """No frame callback is coming, and background timers are throttled — so
    a capture whose user switched away still finishes."""
    hidden = result["hidden"]
    assert hidden["passes"] == 8
    assert hidden["paints"] == 0
    assert hidden["reports"][-1] == [8, 8]


def test_no_progress_asked_for_means_no_waiting(result):
    silent = result["silent"]
    assert silent["passes"] == 8
    assert silent["reports"] == []
    assert silent["paints"] == 0


def test_a_still_reports_a_bare_fraction_per_pass_and_then_the_read_back(result):
    """A bar and one sentence from the caller — the stages do not narrate."""
    fractions = result["fractions"]
    assert len(fractions) == 7          # six passes, then the read-back
    assert fractions == sorted(fractions)
    assert fractions[0] == pytest.approx(0.85 / 6)
    # The marching owns 0.85 of the bar; the read-back and the PNG encode own
    # the rest, and the read-back is the one that actually waits on the GPU.
    assert fractions[-2] == pytest.approx(0.85)
    assert fractions[-1] == pytest.approx(0.85)
