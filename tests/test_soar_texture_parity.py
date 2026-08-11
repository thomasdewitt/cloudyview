"""The Python host and the browser must build the same TEXTURE, not just the
same uniforms.

tests/test_uniform_parity.py has diffed the 368-byte uniform block against the
browser's own packUniforms under node since the desktop app was deleted. It
covered the only surface anybody had thought could drift. It could not see the
one that had already drifted: for the entire life of the periodic renderer the
browser filled a periodic field's lateral ghost ring from the opposite faces
and the Python host shipped zeros there, so every `witness --periodic` render
— and all eight soar golden images — tapered into a boundary that is not
there. Measured at up to 0.23 on the judge views, which is twice the golden
suite's own max-diff gate, and nothing said a word.

So this does for texture construction what that test does for uniforms: it
runs the browser's actual modules under node, on the same synthetic field, and
byte-diffs the result against the Python host's. Not a reimplementation of
either — web/soar/ghost.js is the code ingest/worker.js and scene.js call, and
cloudyview.soar_host.write_wrap_ghosts is what upload_volume calls.

The field is small and deliberately awkward: nonzero on every face and in
every corner (so a wrong face, a transposed axis, or a missing corner all
show), fp16 subnormals (so a lossy round-trip shows), and a z column that
must stay zero at both ends (the vertical is not periodic and must not wrap).

Values are pre-quantized to fp16 before either side sees them. That is not
weakening the test — it removes one known and documented difference (the
browser's fp16 fallback rounds f64 -> f32 -> f16 and so double-rounds about 3
values in 100000) that is not what this test is about, and node 22+ uses the
native Float16Array anyway.

Skips when node is unavailable. It does NOT skip without a GPU: no GPU is
needed to lay out an array, which is the point.
"""

import base64
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview import soar_host as sh

REPO = Path(__file__).resolve().parents[1]
GHOST_JS = REPO / "web" / "soar" / "ghost.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not GHOST_JS.exists(),
    reason="needs node and web/soar/ghost.js")


def synthetic_field(nx=7, ny=5, nz=4):
    """A field whose every face, edge and corner carries a distinct value."""
    rng = np.random.default_rng(20260811)
    sigma = rng.random((nx, ny, nz)) * 0.05
    # Distinct, exactly-representable markers on the four lateral faces, so a
    # swapped or mirrored face is unmistakable rather than merely unequal.
    sigma[0, :, :] = 0.125
    sigma[-1, :, :] = 0.25
    sigma[:, 0, :] = 0.5
    sigma[:, -1, :] = 0.75
    # Corners last, so they win over the faces they sit on.
    sigma[0, 0, :] = 1.5
    sigma[0, -1, :] = 1.75
    sigma[-1, 0, :] = 2.5
    sigma[-1, -1, :] = 3.5
    # fp16 subnormals: below the smallest normal (6.1e-5) and above the
    # smallest subnormal (6.0e-8). These survive the cast as nonzero and are
    # exactly the values an RCE-like hazy field is full of.
    sigma[2, 2, :] = 1e-7
    sigma[3, 1, :] = 3e-6
    # Quantize once, so both hosts start from values fp16 represents exactly.
    return np.asarray(np.asarray(sigma, np.float16), np.float64)


_JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    import { buildXFace, buildYFace, ghostPlanes, applyGhostFaces }
      from "%s";
    import { makeHalfWriter } from "%s";

    const { nx, ny, nz, values } = JSON.parse(
      readFileSync(process.env.FIELD_FILE, "utf8"));
    const padded = [nx + 2, ny + 2, nz + 2];
    const [px, py, pz] = padded;

    // The interior, exactly as ingest/worker.js streams it: original voxel i
    // at texel i + 1, x-major with z fastest.
    const vol = makeHalfWriter(px * py * pz);
    for (let x = 0; x < nx; x++)
      for (let y = 0; y < ny; y++)
        for (let z = 0; z < nz; z++)
          vol.set(((x + 1) * py + (y + 1)) * pz + (z + 1),
                  values[(x * ny + y) * nz + z]);
    const data = vol.bytes();

    // The four faces, from the worker's own builders on the worker's own
    // slices: plane("x", nx - 1) is an (ny, nz) slab, plane("y", ...) an
    // (nx, nz) one.
    const xSlice = (ix) => {
      const out = new Float64Array(ny * nz);
      for (let y = 0; y < ny; y++)
        for (let z = 0; z < nz; z++)
          out[y * nz + z] = values[(ix * ny + y) * nz + z];
      return out;
    };
    const ySlice = (iy) => {
      const out = new Float64Array(nx * nz);
      for (let x = 0; x < nx; x++)
        for (let z = 0; z < nz; z++)
          out[x * nz + z] = values[(x * ny + iy) * nz + z];
      return out;
    };
    const faces = {
      x_lo: buildXFace(xSlice(nx - 1), ny, nz),
      x_hi: buildXFace(xSlice(0), ny, nz),
      y_lo: buildYFace(ySlice(ny - 1), nx, nz),
      y_hi: buildYFace(ySlice(0), nx, nz),
    };
    applyGhostFaces(data, padded, faces);

    process.stdout.write(JSON.stringify({
      padded,
      planes: ghostPlanes(padded),
      volume: Buffer.from(data.buffer, data.byteOffset,
                          data.byteLength).toString("base64"),
    }));
""") % (GHOST_JS.as_posix(), (REPO / "web" / "soar" / "half.js").as_posix())


@pytest.fixture(scope="module")
def field():
    return synthetic_field()


@pytest.fixture(scope="module")
def browser_volume(field, tmp_path_factory):
    """The padded volume the browser would have on the card, run under node."""
    nx, ny, nz = field.shape
    payload = json.dumps({"nx": nx, "ny": ny, "nz": nz,
                          "values": field.ravel(order="C").tolist()})
    field_file = tmp_path_factory.mktemp("texparity") / "field.json"
    field_file.write_text(payload)
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "FIELD_FILE": str(field_file)})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    out = json.loads(proc.stdout)
    volume = np.frombuffer(base64.b64decode(out["volume"]), np.float16)
    out["volume"] = volume.reshape(out["padded"])
    return out


def test_periodic_ghost_ring_matches_the_browser(field, browser_volume):
    """The whole padded texture, texel for texel, both hosts."""
    # The interior first, then the ring — which is the order upload_volume
    # sees it in too.
    ours = np.zeros(np.array(field.shape) + 2, np.float16)
    ours[1:-1, 1:-1, 1:-1] = field
    sh.write_wrap_ghosts(ours)

    theirs = browser_volume["volume"]
    assert ours.shape == theirs.shape

    same = ours.view(np.uint16) == theirs.view(np.uint16)
    if not same.all():
        bad = np.argwhere(~same)
        detail = "\n".join(
            f"    texel {tuple(int(i) for i in ix)}: python={float(ours[tuple(ix)])!r} "
            f"js={float(theirs[tuple(ix)])!r}" for ix in bad[:12])
        raise AssertionError(
            f"the two hosts build different volume textures — "
            f"{len(bad)} of {ours.size} texels differ:\n{detail}\n"
            "  This is the shape of the 2026-08-11 divergence: the browser "
            "wraps the lateral ghost ring and the Python host must too.")


def test_interior_is_untouched_by_the_ring(field, browser_volume):
    """The wrap writes ghosts, never data. A ring that reached one texel too
    far would corrupt the field's own outer voxels, and every other test here
    would still pass."""
    theirs = browser_volume["volume"]
    assert np.array_equal(np.asarray(theirs[1:-1, 1:-1, 1:-1], np.float64),
                          field)


def test_vertical_ghosts_stay_zero(browser_volume):
    """z is never periodic: both z ghost planes must remain the taper."""
    v = browser_volume["volume"]
    assert not v[:, :, 0].any(), "the bottom z ghost plane picked up data"
    assert not v[:, :, -1].any(), "the top z ghost plane picked up data"


def test_corners_wrap_in_both_axes(field, browser_volume):
    """A corner ghost texel is the trilinear support of a sample near a domain
    corner, so it must come from the diagonally opposite corner column — the
    one placement a naive four-plane write gets wrong."""
    v = browser_volume["volume"]
    nx, ny, nz = field.shape
    for (gx, gy), (sx, sy) in {(0, 0): (nx - 1, ny - 1),
                               (0, -1): (nx - 1, 0),
                               (-1, 0): (0, ny - 1),
                               (-1, -1): (0, 0)}.items():
        got = np.asarray(v[gx, gy, 1:-1], np.float64)
        want = field[sx, sy]
        assert np.array_equal(got, want), (
            f"corner ghost column ({gx}, {gy}) should mirror field voxel "
            f"({sx}, {sy}); got {got} want {want}")


def test_demo_bake_agrees_with_the_host(field):
    """tools/export_web_assets bakes faces.bin for the demo with its own
    implementation, so it is a third place this can drift. Applied to a zero
    ring it must reproduce write_wrap_ghosts exactly."""
    from tools.export_web_assets import _ghost_face_arrays

    faces = _ghost_face_arrays(np.asarray(field, np.float16))
    baked = np.zeros(np.array(field.shape) + 2, np.float16)
    baked[1:-1, 1:-1, 1:-1] = field
    baked[0] = faces["x_lo"]
    baked[-1] = faces["x_hi"]
    baked[:, 0:1, :] = faces["y_lo"]
    baked[:, -1:, :] = faces["y_hi"]

    ours = np.zeros(np.array(field.shape) + 2, np.float16)
    ours[1:-1, 1:-1, 1:-1] = field
    sh.write_wrap_ghosts(ours)
    assert np.array_equal(baked.view(np.uint16), ours.view(np.uint16))


def test_ghost_plane_geometry_matches(browser_volume, field):
    """The origins and sizes scene.js hands writeTexture, checked against the
    padded shape rather than assumed."""
    px, py, pz = browser_volume["padded"]
    assert [px, py, pz] == [n + 2 for n in field.shape]
    planes = {p["name"]: p for p in browser_volume["planes"]}
    assert planes["x_lo"]["origin"] == [0, 0, 0]
    assert planes["x_hi"]["origin"] == [0, 0, px - 1]
    assert planes["y_lo"]["origin"] == [0, 0, 0]
    assert planes["y_hi"]["origin"] == [0, py - 1, 0]
    assert planes["x_lo"]["size"] == [pz, py, 1]
    assert planes["y_lo"]["size"] == [pz, 1, px]
    # x first, then y: that ordering is what makes the corners wrap twice.
    assert [p["name"] for p in browser_volume["planes"]] == [
        "x_lo", "x_hi", "y_lo", "y_hi"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
