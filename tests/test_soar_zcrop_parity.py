"""The browser and the Python host must crop to the SAME z band.

Cropping empty sky off a field moves bmin.z and bmax.z, and the domain box is
what cameras place against and what the march sweeps. So a browser that crops
to planes 216-352 and a `witness` that crops to 215-352 do not render slightly
different images of one scene — they render two different scenes, and the
golden suite cannot see it because the field it pins (the TWP-ICE subvolume)
is already tight in z and crops to nothing at all.

That is the same shape of hole tests/test_soar_texture_parity.py was written
to close: a rule duplicated in two hosts, agreeing by inspection until it did
not. So the rule lives in one pure module, web/soar/zcrop.js, and this runs it
under node against cloudyview.witness.crop_empty_z on the same fields.

The fields are chosen for the ways a band rule goes wrong: empty sky at one
end, at both, at neither; a single occupied plane (which must widen, because
the AABB takes its outer half-cells from a gap that a one-plane field does not
have); values that are nonzero in f64 but flush to zero in fp16 (which must
NOT hold a plane open, since the texture is r16float and the renderer will
never see them); and a field that is empty everywhere (which must raise on
both sides rather than return a degenerate range).

The node side slabs the field the way ingest does — z fastest, tiled at a
depth that does not divide nz — because markOccupiedPlanes recovers a texel's
plane from its index modulo the slab depth, and a tiling that divides evenly
would let an off-by-one in that arithmetic pass.

Skips when node is unavailable. Needs no GPU: this is arithmetic on an array.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview.witness import crop_empty_z

REPO = Path(__file__).resolve().parents[1]
ZCROP_JS = REPO / "web" / "soar" / "zcrop.js"
HALF_JS = REPO / "web" / "soar" / "half.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not ZCROP_JS.exists(),
    reason="needs node and web/soar/zcrop.js")


_JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    import { markOccupiedPlanes, occupiedBand } from "%s";
    import { makeHalfWriter } from "%s";

    const { nx, ny, nz, tileZ, values } = JSON.parse(
      readFileSync(process.env.FIELD_FILE, "utf8"));

    // Exactly ingest's sweep: tile the field, quantize each tile to the fp16
    // it will be stored as, and mark planes off the STORED bytes.
    const occupied = new Uint8Array(nz);
    for (let z0 = 0; z0 < nz; z0 += tileZ) {
      const depth = Math.min(tileZ, nz - z0);
      const out = makeHalfWriter(nx * ny * depth);
      let o = 0;
      for (let x = 0; x < nx; x++)
        for (let y = 0; y < ny; y++)
          for (let lz = 0; lz < depth; lz++)
            out.set(o++, values[(x * ny + y) * nz + z0 + lz]);
      markOccupiedPlanes(out.bytes(), z0, depth, occupied);
    }

    let result;
    try {
      result = occupiedBand(occupied);
    } catch (err) {
      result = { error: err.message };
    }
    process.stdout.write(JSON.stringify(result));
""") % (ZCROP_JS.as_posix(), HALF_JS.as_posix())


def browser_band(sigma, tile_z, tmp_path):
    """Run web/soar/zcrop.js over `sigma` the way the ingest worker would."""
    nx, ny, nz = sigma.shape
    field_file = tmp_path / "field.json"
    field_file.write_text(json.dumps({
        "nx": nx, "ny": ny, "nz": nz, "tileZ": tile_z,
        "values": [float(v) for v in sigma.reshape(-1)],
    }))
    script = tmp_path / "drive.mjs"
    script.write_text(_JS)
    out = subprocess.run(
        ["node", str(script)], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "FIELD_FILE": str(field_file)})
    if out.returncode != 0:
        raise AssertionError(f"node failed:\n{out.stderr}")
    return json.loads(out.stdout)


def field_with_occupied(nz, occupied_planes, nx=5, ny=3, seed=20260813):
    """A field whose given z planes hold cloud and whose others hold nothing."""
    rng = np.random.default_rng(seed)
    sigma = np.zeros((nx, ny, nz), dtype=np.float64)
    for z in occupied_planes:
        sigma[:, :, z] = rng.random((nx, ny)) * 0.05 + 1e-3
    return sigma


# (name, nz, occupied planes) — the ways a band rule goes wrong.
CASES = [
    ("empty above", 20, list(range(0, 12))),
    ("empty below", 20, list(range(8, 20))),
    ("empty at both ends", 40, list(range(13, 27))),
    ("nothing empty", 12, list(range(0, 12))),
    ("one plane, at the floor", 16, [0]),
    ("one plane, mid field", 16, [9]),
    ("one plane, at the ceiling", 16, [15]),
    ("a gap in the middle stays inside the band", 24, [3, 4, 18, 19]),
    ("occupied only on the outermost planes", 24, [0, 23]),
]


@pytest.mark.parametrize("name,nz,planes",
                         CASES, ids=[c[0].replace(" ", "-") for c in CASES])
def test_hosts_agree_on_the_band(name, nz, planes, tmp_path):
    sigma = field_with_occupied(nz, planes)
    _, _, (lo, hi) = crop_empty_z(sigma, np.arange(nz, dtype=np.float64) * 40.0)
    # A tile depth that does not divide nz, so the plane-from-index arithmetic
    # is exercised on ragged final tiles rather than on whole ones.
    band = browser_band(sigma, 7, tmp_path)
    assert "error" not in band, band.get("error")
    assert (band["lo"], band["hi"]) == (lo, hi), (
        f"{name}: browser cropped to {band['lo']}-{band['hi']}, "
        f"Python to {lo}-{hi}")
    assert band["count"] == hi - lo + 1
    assert band["cropped"] == (hi - lo + 1 < nz)


def test_a_single_occupied_plane_widens_on_both_hosts(tmp_path):
    """One plane is widened to two — the AABB needs a gap to size itself."""
    sigma = field_with_occupied(16, [9])
    _, z, (lo, hi) = crop_empty_z(sigma, np.arange(16, dtype=np.float64) * 40.0)
    assert (lo, hi) == (9, 10)
    assert z.size == 2
    assert browser_band(sigma, 7, tmp_path)["count"] == 2


def test_values_below_the_fp16_floor_do_not_hold_a_plane_open(tmp_path):
    """A plane that stores as all-zero is empty, whatever f64 says.

    2**-26 is nonzero in f64 and rounds to zero in fp16, so a host that judged
    emptiness on the f64 sigma would keep these planes and one that judged on
    the stored texel would drop them. That disagreement is the whole reason
    the rule is written against the stored value.
    """
    sigma = field_with_occupied(20, list(range(8, 12)))
    sigma[:, :, 0] = 2.0 ** -26
    sigma[:, :, 19] = 2.0 ** -26
    _, _, (lo, hi) = crop_empty_z(sigma, np.arange(20, dtype=np.float64) * 40.0)
    assert (lo, hi) == (8, 11), "the fp16-invisible planes were kept"
    band = browser_band(sigma, 7, tmp_path)
    assert (band["lo"], band["hi"]) == (8, 11)


def test_a_value_just_above_the_fp16_floor_does_hold_a_plane_open(tmp_path):
    """The other side of the same threshold, so it is a floor and not a fence."""
    sigma = field_with_occupied(20, list(range(8, 12)))
    # The smallest positive fp16 subnormal: stores exactly, so it counts.
    sigma[:, :, 2] = 2.0 ** -24
    _, _, (lo, hi) = crop_empty_z(sigma, np.arange(20, dtype=np.float64) * 40.0)
    assert (lo, hi) == (2, 11)
    band = browser_band(sigma, 7, tmp_path)
    assert (band["lo"], band["hi"]) == (2, 11)


def test_an_entirely_empty_field_raises_on_both_hosts(tmp_path):
    """No band to crop to. Both sides say so rather than returning nonsense."""
    sigma = np.zeros((5, 3, 12), dtype=np.float64)
    with pytest.raises(ValueError, match="empty sky"):
        crop_empty_z(sigma, np.arange(12, dtype=np.float64) * 40.0)
    band = browser_band(sigma, 7, tmp_path)
    assert "error" in band
    assert "empty sky" in band["error"]


def test_the_band_is_reached_the_same_way_however_the_field_is_tiled(tmp_path):
    """Tiling is an ingest detail; the band it produces must not be one."""
    sigma = field_with_occupied(31, list(range(6, 23)))
    _, _, (lo, hi) = crop_empty_z(sigma, np.arange(31, dtype=np.float64) * 40.0)
    for tile_z in (1, 2, 5, 8, 16, 31, 64):
        band = browser_band(sigma, tile_z, tmp_path)
        assert (band["lo"], band["hi"]) == (lo, hi), f"tile depth {tile_z}"
