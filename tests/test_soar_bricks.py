"""Brick decomposition must be exactly value-preserving.

web/soar/bricks.js packs a sparse field's occupied 8^3 bricks (plus a 1-voxel
apron each) into an atlas behind a page table. The whole design rests on one
invariant: EVERY atlas texel — interior and apron alike — equals the dense
field's value at that texel's global coordinate, with x/y wrapped when the
field is periodic and zero past the z faces, exactly the semantics ghost.js
gives the dense texture's ghost ring. If that holds, hardware trilinear
filtering inside any single brick is exact everywhere including across brick
seams and the periodic wrap, and a bricked renderer that keeps the sample
lattice must reproduce the dense renderer to the bit.

So the test is a round trip: numpy builds a dense reference deliberately
shaped to catch layout bugs (ragged extents that leave partial edge bricks,
occupied bricks adjacent across brick seams so aprons need REAL neighbour
data, occupied voxels on every domain face and corner so the periodic wrap
is exercised, and isolated single voxels); node feeds it to the real
bricks.js in ingest's tile order with tile sizes that do not align to the
brick grid; Python then checks every atlas texel and every page-table entry
against the reference. Exact equality, no tolerance.

Runs bricks.js under node like test_soar_texture_parity.py runs ghost.js.
Skips when node is unavailable; needs no GPU.
"""

import base64
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
BRICKS_JS = REPO / "web" / "soar" / "bricks.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not BRICKS_JS.exists(),
    reason="needs node and web/soar/bricks.js")


def reference_field(nx=37, ny=22, nz=13, seed=20260812):
    """Sparse field shaped to catch every layout bug at once."""
    rng = np.random.default_rng(seed)
    f = np.zeros((nx, ny, nz), dtype=np.float32)
    # A blob spanning several bricks, so adjacent occupied bricks must trade
    # real apron data across their shared faces, edges and corners.
    blob = np.s_[5:min(20, nx), 3:min(14, ny), 2:min(9, nz)]
    f[blob] = rng.random(f[blob].shape).astype(np.float32) * 0.05
    # Punch holes so occupied bricks are partially filled.
    f[8:min(12, nx), 6:min(10, ny), 4:min(6, nz)] = 0.0
    # Occupied voxels on every face and every corner: the periodic apron must
    # fetch across the wrap, and the z faces must stay empty beyond the grid.
    f[0, :, :] = rng.random((ny, nz)).astype(np.float32) * 0.01 + 0.001
    f[-1, :, :] = rng.random((ny, nz)).astype(np.float32) * 0.01 + 0.001
    f[:, 0, 0] = 0.002
    f[:, -1, -1] = 0.003
    # Isolated single voxels, one deep inside a brick and one at a brick corner.
    if (nx, ny, nz) >= (33, 18, 12):
        f[29, 17, 11] = 0.7
        f[32, 16, 8] = 0.9
    return f


def node_script():
    """The node driver: feed a field to bricks.js in ingest's tile order."""
    return textwrap.dedent("""
        import { createBrickBuilder } from %r;
        import { readFileSync } from "node:fs";

        const p = JSON.parse(readFileSync(process.argv[2], "utf8"));
        const [nx, ny, nz] = p.dims;
        const field = new Float32Array(
            Uint8Array.from(atob(p.field), c => c.charCodeAt(0)).buffer);

        const b = createBrickBuilder(
            { dims: p.dims, brick: p.brick, periodic: p.periodic });
        const [tx, ty, tz] = p.tile;
        // ingest/worker.js tile order: x outer, y middle, z inner.
        for (let x0 = 0; x0 < nx; x0 += tx) {
          for (let y0 = 0; y0 < ny; y0 += ty) {
            for (let z0 = 0; z0 < nz; z0 += tz) {
              const sx = Math.min(tx, nx - x0);
              const sy = Math.min(ty, ny - y0);
              const sz = Math.min(tz, nz - z0);
              const vals = new Float32Array(sx * sy * sz);
              let i = 0;
              for (let lx = 0; lx < sx; lx++)
                for (let ly = 0; ly < sy; ly++)
                  for (let lz = 0; lz < sz; lz++, i++)
                    vals[i] = field[((x0 + lx) * ny + y0 + ly) * nz + z0 + lz];
              b.addTile([x0, y0, z0], [sx, sy, sz], vals);
            }
          }
        }
        const out = b.finalize();
        const b64 = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength)
                              .toString("base64");
        console.log(JSON.stringify({
          pageDims: out.pageDims,
          pageTable: b64(out.pageTable),
          atlasDims: out.atlasDims,
          atlas: b64(out.atlas),
          atlasBrickGrid: out.atlasBrickGrid,
          stats: out.stats,
        }));
    """) % (str(BRICKS_JS),)


def run_node(field, brick, periodic, tile, tmp_path):
    nx, ny, nz = field.shape
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({
        "dims": [nx, ny, nz],
        "brick": list(brick),
        "periodic": periodic,
        "tile": list(tile),
        "field": base64.b64encode(field.astype("<f4").tobytes()).decode(),
    }))
    proc = subprocess.run(
        ["node", "--input-type=module", "-", str(payload_path)],
        input=node_script(), capture_output=True, text=True, cwd=REPO,
        timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    page = np.frombuffer(base64.b64decode(out["pageTable"]),
                         dtype="<u4").reshape(out["pageDims"])
    atlas = np.frombuffer(base64.b64decode(out["atlas"]),
                          dtype="<f4").reshape(out["atlasDims"])
    return page, atlas, out


def expected_value(field, x, y, z, periodic):
    """ghost.js semantics: x/y wrap when periodic, z out of range is zero."""
    nx, ny, nz = field.shape
    if periodic:
        x %= nx
        y %= ny
    if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
        return 0.0
    return float(field[x, y, z])


def check_round_trip(field, brick, periodic, tile, tmp_path):
    page, atlas, out = run_node(field, brick, periodic, tile, tmp_path)
    nx, ny, nz = field.shape
    bx, by, bz = brick
    px, py, pz = bx + 2, by + 2, bz + 2
    ax, ay, az = out["atlasBrickGrid"]

    # Page-table truth: a brick has a slot exactly when the renderer can ever
    # sample a non-zero value inside it — which is NOT the same as holding a
    # non-zero voxel.
    #
    # Trilinear filtering at a point in brick c has support [floor(g),
    # floor(g)+1], so a sample anywhere in the brick can reach one voxel past
    # its far face. That overhang is the taper a cloud edge fades through. A
    # brick just outside the cloud that had no slot would answer 0 there and
    # cut the taper off at a brick boundary — visible as a hard edge, and
    # worse, it would tell the empty-space skip it may leap a region the dense
    # march finds cloud in. So the test is over the SAMPLING SUPPORT, wrapped
    # in x/y when the field is periodic and clipped in z where it is not.
    gx, gy, gz = out["pageDims"]
    nx, ny, nz = field.shape
    for cx in range(gx):
        for cy in range(gy):
            for cz in range(gz):
                xs = [(cx * bx + i) % nx if periodic else cx * bx + i
                      for i in range(bx + 1)]
                ys = [(cy * by + i) % ny if periodic else cy * by + i
                      for i in range(by + 1)]
                xs = [v for v in xs if 0 <= v < nx]
                ys = [v for v in ys if 0 <= v < ny]
                zs = [v for v in (cz * bz + i for i in range(bz + 1))
                      if 0 <= v < nz]
                reachable = bool(
                    xs and ys and zs
                    and (field[np.ix_(xs, ys, zs)] != 0).any())
                assert (page[cx, cy, cz] != 0) == reachable, \
                    f"page table wrong at brick ({cx},{cy},{cz})"

    # Every texel of every occupied brick's padded slot — interior AND apron —
    # equals the reference at its global coordinate. Exact, no tolerance.
    ids = {}
    for cx in range(gx):
        for cy in range(gy):
            for cz in range(gz):
                if page[cx, cy, cz]:
                    ids[int(page[cx, cy, cz]) - 1] = (cx, cy, cz)
    assert len(ids) == out["stats"]["occupiedBricks"]

    for s, (cx, cy, cz) in ids.items():
        sx, sy, sz = s // (ay * az), (s // az) % ay, s % az
        slot = atlas[sx * px:(sx + 1) * px,
                     sy * py:(sy + 1) * py,
                     sz * pz:(sz + 1) * pz]
        for ox in range(px):
            for oy in range(py):
                for oz in range(pz):
                    want = expected_value(field, cx * bx + ox - 1,
                                          cy * by + oy - 1,
                                          cz * bz + oz - 1, periodic)
                    got = float(slot[ox, oy, oz])
                    assert got == want, (
                        f"brick ({cx},{cy},{cz}) texel ({ox},{oy},{oz}): "
                        f"got {got}, want {want}")

    # Unreferenced atlas texels (slack slots in the packing grid) stay zero.
    used = np.zeros((ax, ay, az), dtype=bool)
    for s in ids:
        used[s // (ay * az), (s // az) % ay, s % az] = True
    for ux in range(ax):
        for uy in range(ay):
            for uz in range(az):
                if not used[ux, uy, uz]:
                    slack = atlas[ux * px:(ux + 1) * px,
                                  uy * py:(uy + 1) * py,
                                  uz * pz:(uz + 1) * pz]
                    assert not slack.any()


@pytest.mark.parametrize("brick", [(8, 8, 8), (8, 8, 4), (4, 4, 4)])
def test_round_trip_periodic(brick, tmp_path):
    """Awkward extents, misaligned tiles, periodic wrap: exact round trip."""
    check_round_trip(reference_field(), brick, True, (16, 9, 13), tmp_path)


def test_round_trip_nonperiodic(tmp_path):
    """Same field without the wrap: domain-face aprons are zero instead."""
    check_round_trip(reference_field(), (8, 8, 8), False, (16, 9, 13), tmp_path)


def test_round_trip_single_tile(tmp_path):
    """The whole field as one tile must agree with the tiled build."""
    field = reference_field(nx=17, ny=11, nz=9)
    check_round_trip(field, (8, 8, 8), True, (17, 11, 9), tmp_path)


def test_brick_straddling_tiles(tmp_path):
    """A 1-wide z tile slices every brick; straddled bricks must assemble."""
    field = reference_field(nx=16, ny=16, nz=8)
    check_round_trip(field, (8, 8, 8), True, (5, 7, 1), tmp_path)
