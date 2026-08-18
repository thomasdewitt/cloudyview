"""A whole-field upload must arrive whole, and in pieces a browser will take.

The demo path used to hand the entire field to one queue.writeTexture call.
That is fine at 400 MB and fatal at 4 GB: Firefox rejects a source view larger
than 2 GB outright, so the STEAM desert case (2048 x 2048 x 495, 4.15 GB) could
not be opened at all — while the same field loaded from its netCDF worked,
because ingest has always uploaded slab by slab.

So scene.js chunks it. Chunking a volume upload is index arithmetic against a
texture whose axes are permuted (width=nz, height=ny, depth=nx), which is
exactly the kind of code that is obviously right and off by one: a wrong
subarray window writes the field shifted, and a shifted extinction field still
renders a perfectly plausible cloud. Nothing downstream would report it.

Since 2026-08-18 the demo path STREAMS the volume (streamWholeVolume): the
decompressed bytes go from the network straight into slab-sized writeTexture
calls, so the JS heap never holds the whole field — which is what lets a
4.15 GB demo load on an 8 GB machine at all. The chunking arithmetic is the
same, with one extra hazard: network chunks land at arbitrary offsets
relative to planes and slabs, so the driver below feeds the stream in
prime-sized pieces to put a chunk edge everywhere.

This drives the real streamWholeVolume under node against a stub device that
records every call, then checks the calls TILE the volume: each x plane written
once, in order, with the source bytes that plane actually holds. It also pins
the two properties the chunking exists for — no call over the ceiling, and a
drain between chunks so staging memory cannot pile up unbounded — and that a
truncated stream fails loudly instead of leaving a silently short field.

Needs node. No GPU: this is arithmetic over a recorded call list.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCENE_JS = REPO / "web" / "soar" / "scene.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not SCENE_JS.exists(),
    reason="needs node and web/soar/scene.js")

# The limit that made this necessary. Firefox's message names 2 GB; the chunk
# size is far below it, and the assertion is that we stay under the real cap
# rather than that we hit any particular chunk size.
FIREFOX_VIEW_CEILING = 2 * 1024**3

_JS = textwrap.dedent("""
    import { streamWholeVolume, UPLOAD_DRAIN_BYTES } from "%s";

    const [nx, ny, nz] = JSON.parse(process.env.SHAPE);
    const truncate = Number(process.env.TRUNCATE || 0);

    // Each element fingerprints its own flat index, so a slab taken from the
    // wrong window shows up as values that do not match where it landed.
    //
    // Modulo a PRIME, not masked to 16 bits: a plane here is a power of two
    // elements, so `i & 0xffff` is zero at the start of every chunk and the
    // fingerprint cannot see a window that starts in the wrong place. It read
    // as a passing test while asserting nothing — 65521 is the largest prime
    // under 2**16 and shares no factor with any plane size.
    const words = new Uint16Array(nx * ny * nz);
    for (let i = 0; i < words.length; i++) words[i] = i %% 65521;

    const calls = [];
    let drains = 0;
    const device = {
      queue: {
        writeTexture(dst, data, layout, size) {
          calls.push({
            originZ: dst.origin.z, originX: dst.origin.x, originY: dst.origin.y,
            size, bytes: data.byteLength,
            bytesPerRow: layout.bytesPerRow, rowsPerImage: layout.rowsPerImage,
            // What the GPU would actually receive, folded to something small:
            // three samples across the view, and its length.
            first: data[0], mid: data[data.length >> 1],
            last: data[data.length - 1], length: data.length,
            drainsBefore: drains,
          });
        },
        onSubmittedWorkDone() { drains++; return Promise.resolve(); },
      },
    };

    // The network hands the decompressor whatever chunk sizes it likes, so
    // feed the stream in PRIME-sized pieces: chunk edges land at every
    // offset relative to planes and slabs, which is where a wrong copy
    // window would hide.
    const bytes = new Uint8Array(
      words.buffer, 0, words.byteLength - truncate);
    const CHUNK = 65521;
    const stream = new ReadableStream({
      start(controller) {
        for (let off = 0; off < bytes.length; off += CHUNK) {
          controller.enqueue(
            bytes.subarray(off, Math.min(off + CHUNK, bytes.length)));
        }
        controller.close();
      },
    });

    const progress = [];
    await streamWholeVolume(device, {}, stream, [nx, ny, nz],
                            (f) => progress.push(f));
    process.stdout.write(JSON.stringify(
      { calls, drains, progress, drainBytes: UPLOAD_DRAIN_BYTES }));
""") % SCENE_JS.as_posix()


def upload(shape, tmp_path, truncate=0):
    script = tmp_path / "drive.mjs"
    script.write_text(_JS)
    out = subprocess.run(
        ["node", str(script)], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "SHAPE": json.dumps(list(shape)),
             "TRUNCATE": str(truncate)})
    if out.returncode != 0:
        raise AssertionError(f"node failed:\n{out.stderr}")
    return json.loads(out.stdout)


# Shapes chosen for the ways the arithmetic goes wrong. Three of them are
# sized to actually cross the chunk boundary — at 64 MB a chunk, a toy field
# goes in one call and tests nothing about chunking at all, which is how the
# first draft of this file passed while asserting nothing.
SHAPES = [
    ("one call is enough", (4, 8, 16)),
    ("the chunk divides nx", (256, 512, 512)),          # 134 MB -> 2 chunks
    ("nx not a multiple of the chunk", (301, 512, 512)),  # 158 MB -> 3 chunks
    ("a wide plane, few per chunk", (9, 2048, 2048)),   # 8.4 MB a plane
]


@pytest.mark.parametrize("name,shape", SHAPES,
                         ids=[s[0].replace(" ", "-") for s in SHAPES])
def test_slabs_tile_the_volume_exactly(name, shape, tmp_path):
    nx, ny, nz = shape
    result = upload(shape, tmp_path)
    calls = result["calls"]
    assert calls, "nothing was uploaded"

    covered = 0
    for call in calls:
        # The texture's axes are permuted: a slab is [width=nz, height=ny,
        # depth=<x planes>] written at origin.z = the first x plane.
        assert call["originX"] == 0 and call["originY"] == 0
        assert call["originZ"] == covered, "slabs are out of order or overlap"
        w, h, d = call["size"]
        assert (w, h) == (nz, ny), "a slab must span the full x plane"
        assert call["bytesPerRow"] == nz * 2
        assert call["rowsPerImage"] == ny
        # The source view has to be the bytes of exactly those planes — which
        # is the check that a chunked upload exists to get right, and the one
        # nothing downstream could report: a window taken from the wrong
        # offset writes a shifted field, and a shifted extinction field still
        # renders a perfectly plausible cloud.
        start = covered * ny * nz
        assert call["length"] == d * ny * nz
        assert call["first"] == start % 65521
        assert call["mid"] == (start + (call["length"] >> 1)) % 65521
        assert call["last"] == (start + call["length"] - 1) % 65521
        covered += d
    assert covered == nx, f"{covered} of {nx} planes were written"


@pytest.mark.parametrize("name,shape", SHAPES,
                         ids=[s[0].replace(" ", "-") for s in SHAPES])
def test_no_call_approaches_the_view_ceiling(name, shape, tmp_path):
    result = upload(shape, tmp_path)
    for call in result["calls"]:
        assert call["bytes"] < FIREFOX_VIEW_CEILING
        # And under the drain budget too, except where one x plane is already
        # bigger than it — which cannot be split further along this axis.
        nx, ny, nz = shape
        assert call["bytes"] <= max(result["drainBytes"], ny * nz * 2)


def test_the_queue_is_drained_between_chunks(tmp_path):
    """Otherwise staging memory for a multi-GB field piles up all at once."""
    result = upload((256, 512, 512), tmp_path)
    assert len(result["calls"]) > 1, "this shape should need several chunks"
    for i, call in enumerate(result["calls"]):
        assert call["drainsBefore"] == i, "a chunk went out without a barrier"
    assert result["drains"] == len(result["calls"])


def test_progress_runs_to_one(tmp_path):
    result = upload((301, 512, 512), tmp_path)
    assert result["progress"], "no progress was reported"
    assert result["progress"] == sorted(result["progress"])
    assert result["progress"][-1] == 1.0


@pytest.mark.parametrize("drop,expected", [
    # Mid-plane: stray bytes that tile no plane at all.
    (7, "mid-plane"),
    # Whole planes short: every byte tiles, but the field is short.
    (512 * 512 * 2 * 3, "promised"),
], ids=["mid-plane", "whole-planes-short"])
def test_a_truncated_stream_fails_loudly(drop, expected, tmp_path):
    """A short download must throw, not leave a silently truncated field."""
    with pytest.raises(AssertionError, match=expected):
        upload((256, 512, 512), tmp_path, truncate=drop)
