"""
No street lamp stands out in a cross street.

The lamp lattice (city/layout.wgsl city_lamp_lattice) runs along a street
without regard to the streets that cross it, and until 2026-11-25 a station
that fell where a cross street runs put its mast in that street's
carriageway: in the first 300 x 300 blocks of the shipped tile, 75,422 of
916,764 masts stood in a carriageway (a minor street's pole up to 11.5 m out
into an avenue) and 24,735 more in the road where the kerb's corner arc cuts
the sidewalk away. city_lamp_stands leaves those stations out, in the
renderer and in web/soar/cityprops.js alike.

This reads the masts the collider's mirror stands (cityprops.js, held to the
renderer by test_city_props_mirror) over a downtown stretch and an ordinary
one, and checks each against the street geometry (city/street.wgsl
city_kerb_rect: the kerb 3 m off a minor street's centreline, 11.5 m off an
avenue's, the corner a CITY_KERB_R = 3 m quarter circle).

Skips without node.
"""

import json
import shutil
import subprocess

import pytest

from tests.conftest import REPO

SOAR = REPO / "web" / "soar"
TILE_DIR = REPO / "cloudyview" / "soar" / "city"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

JS = """
import { readFileSync } from "node:fs";
import { CityTile, cityCell, isAvenue } from "%(soar)s/citylots.js";
import { CityProps, cc_streetlife_MAST_R, cc_streetlife_POLE_H }
  from "%(soar)s/cityprops.js";
const spec = JSON.parse(readFileSync(0, "utf8"));
const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
const b = readFileSync("%(tile)s/fif_mip0.bin");
const tile = CityTile.fromHalfBytes(
  new Uint8Array(b.buffer, b.byteOffset, b.byteLength), meta.n, meta.cell_m);
const props = new CityProps(tile);
const C = meta.cell_m, R = cc_streetlife_MAST_R, KERB_R = 3.0;
const kerb = (k) => (isAvenue(k) ? 11.5 : 3.0);
let masts = 0, road = 0, arc = 0;
const [x0, y0, n] = spec.blocks;
for (let cy = y0; cy < y0 + n; cy++) for (let cx = x0; cx < x0 + n; cx++) {
  const cc = cityCell(tile, cx, cy);
  for (const s of props.cellSolids(cc, cx, cy, 0).solids) {
    if (s.source !== "lamp") continue;
    const lo = [0, 0, 0], hi = [0, 0, 0];
    for (const [a, bb, c, d] of s.planes) {
      const ax = a ? 0 : bb ? 1 : 2;
      if ((a || bb || c) > 0) hi[ax] = d; else lo[ax] = -d;
    }
    if (!(lo[2] === 0 && hi[2] === cc_streetlife_POLE_H)) continue;  // the mast
    masts += 1;
    const x = 0.5 * (lo[0] + hi[0]), y = 0.5 * (lo[1] + hi[1]);
    const kx = Math.round(x / C), ky = Math.round(y / C);
    const dx = Math.abs(x - kx * C), dy = Math.abs(y - ky * C);
    if (dx - R < kerb(kx) || dy - R < kerb(ky)) { road += 1; continue; }
    const ox = kerb(kx) + KERB_R, oy = kerb(ky) + KERB_R;
    if (dx < ox && dy < oy && Math.hypot(ox - dx, oy - dy) > KERB_R - R) arc += 1;
  }
}
console.log(JSON.stringify({ masts, road, arc }));
""" % {"soar": SOAR, "tile": TILE_DIR}


@pytest.mark.parametrize("blocks", [(180, 760, 60), (0, 0, 60)],
                         ids=["downtown", "tile-corner"])
def test_no_lamp_mast_stands_in_a_roadway(blocks):
    out = subprocess.run(["node", "--input-type=module", "-e", JS],
                         input=json.dumps({"blocks": blocks}),
                         capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["masts"] > 10_000
    assert r["road"] == 0
    assert r["arc"] == 0
