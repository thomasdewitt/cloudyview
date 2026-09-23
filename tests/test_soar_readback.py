"""A video reads every frame back through ONE staging buffer.

The 2026-08-31 VRAM OOM on a long in-browser video: `readBack` created and
destroyed a frame-sized MAP_READ buffer per frame (33 MB at 4K), and a
destroyed buffer's memory returns to the driver only when the browser gets
round to it. `capture.Readback` makes the buffer once per capture. This pins,
under node with a stub device whose copy writes a known padded image:

- N reads through one Readback allocate exactly one staging buffer, and
  destroy() frees it once,
- every read returns exactly the texture's pixels — row padding undone,
  alpha forced to 255 — so the change moves no output byte, including on a
  width whose rows need padding and on a buffer that has been used before,
- a second read while one is in flight refuses rather than reusing a
  buffer that is still mapped-pending.

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
CAPTURE_JS = REPO / "web" / "soar" / "capture.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="needs node")

_JS = textwrap.dedent("""
    import { Readback } from "%s";

    globalThis.GPUBufferUsage = { COPY_DST: 1, MAP_READ: 2 };
    globalThis.GPUMapMode = { READ: 1 };
    globalThis.ImageData = class {
      constructor(w, h) {
        this.width = w; this.height = h;
        this.data = new Uint8ClampedArray(w * h * 4);
      }
    };

    // A texture is its frame number; its texel (x, y) channel c holds a
    // byte that depends on all four, alpha included (which must be forced).
    const texel = (f, x, y, c) => (f * 31 + x * 7 + y * 13 + c * 57) & 255;
    const device = {
      buffers: 0, destroyed: 0,
      createBuffer({ size }) {
        device.buffers += 1;
        const mem = new ArrayBuffer(size);
        return {
          mem, async mapAsync() { await null; },
          getMappedRange: () => mem, unmap() {},
          destroy() { device.destroyed += 1; },
        };
      },
      createCommandEncoder: () => ({
        copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [w, h]) {
          const dst = new Uint8Array(buffer.mem);
          dst.fill(0xEE);                    // padding is garbage, not zeros
          for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++)
              for (let c = 0; c < 4; c++)
                dst[y * bytesPerRow + x * 4 + c] = texel(texture.frame, x, y, c);
        },
        finish: () => ({}),
      }),
      queue: { submit() {} },
    };

    const [w, h] = [37, 5];                  // 148-byte rows, padded to 256
    const rb = new Readback(device, [w, h]);
    let exact = true;
    for (let f = 0; f < 4; f++) {
      const img = await rb.read({ frame: f });
      if (img.width !== w || img.height !== h) exact = false;
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++)
          for (let c = 0; c < 4; c++) {
            const want = c === 3 ? 255 : texel(f, x, y, c);
            if (img.data[(y * w + x) * 4 + c] !== want) exact = false;
          }
    }
    const buffersAfterReads = device.buffers;

    let refused = null;
    const first = rb.read({ frame: 9 });
    try { await rb.read({ frame: 10 }); } catch (e) { refused = e.message; }
    await first;
    const afterRefusal = await rb.read({ frame: 11 });   // still usable

    rb.destroy();
    process.stdout.write(JSON.stringify({
      exact, buffersAfterReads, destroyed: device.destroyed, refused,
      usableAfter: afterRefusal.data[3] === 255,
    }));
""") % CAPTURE_JS.as_posix()


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_one_staging_buffer_for_every_frame(result):
    assert result["buffersAfterReads"] == 1
    assert result["destroyed"] == 1


def test_reads_return_the_texture_exactly(result):
    assert result["exact"]


def test_an_overlapping_read_refuses(result):
    assert result["refused"] and "in flight" in result["refused"]
    assert result["usableAfter"]
