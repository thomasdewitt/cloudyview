"""Zen mode's flight (web/soar/zen.js), flown in node.

Over a cloud field: the altitude stays in the band that straddles the bottom
of the cloud layer and spends about half its time under the cloud where
there is room under it, the turn is never tighter than a circle in
ZEN_CLOUD_CIRCLE_S, the look ends level, and in a field that does not wrap
the flight stays inside it.

Over the city, through the shipped tile with everything that stands in it
(citylots.js, cityprops.js): the flight keeps to the street grid's centre
lines and its turns, never comes within a body's reach of anything solid,
stays between the street and the ceiling, never needs a stretch flown
without a clear altitude, and comes down into that range when started high
above it. The street word's reading is held to the lot mirror's.

Needs node; no GPU.
"""
import json
import shutil
import subprocess
import textwrap

import pytest

from tests.conftest import REPO

SOAR = REPO / "web" / "soar"
TILE_DIR = REPO / "cloudyview" / "soar" / "city"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

PRELUDE = """
import { readFileSync } from "node:fs";
import { FlightCamera } from "%(soar)s/camera.js";
import { CityTile, CityLots, cityCell } from "%(soar)s/citylots.js";
import { CityProps } from "%(soar)s/cityprops.js";
import * as Z from "%(soar)s/zen.js";

const spec = JSON.parse(readFileSync(0, "utf8"));

// A seeded generator, so a failure names a flight that can be flown again.
let seed = spec.seed >>> 0;
const random = () => {
  seed = (seed + 0x6D2B79F5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const DT = 1 / 60;
const fold = (deg) => ((deg + 540) %% 360) - 180;
""" % {"soar": SOAR}


def _node(body, spec):
    out = subprocess.run(
        ["node", "--input-type=module", "-e", PRELUDE + textwrap.dedent(body)],
        input=json.dumps(spec), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


CLOUD_JS = """
const cam = new FlightCamera(spec.bmin, spec.bmax, { periodic: spec.periodic });
cam.elevation = 25;
const zen = new Z.CloudZen(cam, random);
const settle = Math.round(240 / DT), steps = Math.round(spec.minutes * 60 / DT);
const centre = [0, 1].map((i) => 0.5 * (spec.bmin[i] + spec.bmax[i]));
const half = [0, 1].map((i) => 0.5 * (spec.bmax[i] - spec.bmin[i]));
let lastAz = cam.azimuth, maxRate = 0, zmin = Infinity, zmax = -Infinity;
let under = 0, out = 0, moved = 0, last = [...cam.position];
for (let k = 0; k < steps; k++) {
  zen.step(DT, cam);
  maxRate = Math.max(maxRate, Math.abs(fold(cam.azimuth - lastAz)) / DT);
  lastAz = cam.azimuth;
  out = Math.max(out, ...[0, 1].map(
    (i) => Math.abs(cam.position[i] - centre[i]) / half[i]));
  if (k === settle) last = [...cam.position];
  if (k === settle + 60) {
    moved = Math.hypot(cam.position[0] - last[0], cam.position[1] - last[1]);
  }
  if (k < settle) continue;
  zmin = Math.min(zmin, cam.position[2]);
  zmax = Math.max(zmax, cam.position[2]);
  if (cam.position[2] < spec.bmin[2]) under += 1;
}
console.log(JSON.stringify({
  band: zen.band, zmin, zmax, under: under / (steps - settle), maxRate, out,
  elevation: cam.elevation, speed: moved, circle: Z.ZEN_CLOUD_CIRCLE_S,
  nominal: Z.ZEN_CLOUD_SPEED, backRate: Z.ZEN_CLOUD_TURN_BACK_RATE,
}));
"""

# The demo set's shapes: a thin deck, a layer on the sea, and a deep one
# that starts high (web/soar/demos/*/meta.json bmin, bmax).
FIELDS = {
    "dycoms": ([-2.5, -2.5, 537.5], [3197.5, 3197.5, 887.5]),
    "congestus": ([-5.0, -5.0, 25.0], [20475.0, 20475.0, 3875.0]),
    "stratified": ([-50.0, -50.0, 7011.4], [102350.0, 102350.0, 14934.0]),
}


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_the_cloud_walk_keeps_its_band_and_its_gentle_turn(name):
    bmin, bmax = FIELDS[name]
    r = _node(CLOUD_JS, {"seed": 11, "bmin": bmin, "bmax": bmax,
                         "periodic": True, "minutes": 90})
    low, high = r["band"]
    depth = bmax[2] - bmin[2]
    assert high == pytest.approx(bmin[2] + depth / 3)
    assert low == pytest.approx(max(bmin[2] - depth / 3, 50.0))
    assert low - 1e-6 <= r["zmin"] and r["zmax"] <= high + 1e-6
    # It uses the band, not one height in it.
    assert r["zmax"] - r["zmin"] > 0.6 * (high - low)
    # Half the band is under the cloud where the band is not cut by the sea.
    if low > 50.0:
        assert 0.3 < r["under"] < 0.7
    assert r["maxRate"] <= 360.0 / r["circle"] + 1e-6
    assert r["elevation"] == 0.0
    assert r["speed"] == pytest.approx(r["nominal"], rel=1e-3)


@pytest.mark.parametrize("extent", [102400.0, 3200.0])
def test_the_cloud_walk_stays_in_a_field_that_does_not_wrap(extent):
    r = _node(CLOUD_JS, {"seed": 5, "bmin": [0.0, 0.0, 500.0],
                         "bmax": [extent, extent, 900.0],
                         "periodic": False, "minutes": 90})
    assert r["out"] < 1.05
    assert r["maxRate"] <= r["backRate"] + 1e-6
    # A field too small to turn in at 500 m/s is flown slower.
    assert r["speed"] <= r["nominal"] * (1 + 1e-3)
    if extent < 10000.0:
        assert r["speed"] < 0.2 * r["nominal"]


CITY_JS = """
const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
const b = readFileSync("%(tile)s/fif_mip0.bin");
const tile = CityTile.fromHalfBytes(
  new Uint8Array(b.buffer, b.byteOffset, b.byteLength), meta.n, meta.cell_m);
const lots = new CityLots(tile, new CityProps(tile));
const EXTENT = meta.n * meta.cell_m, CELL = meta.cell_m;
""" % {"tile": TILE_DIR}

CITY_FLIGHT_JS = CITY_JS + """
const cam = new FlightCamera([0, 0, 25], [20480, 20480, 3875], {
  periodic: true, surfaceTileExtent: EXTENT, floorZ: null });
cam.jump(spec.start[0] - cam.surfacePosition[0],
         spec.start[1] - cam.surfacePosition[1],
         spec.start[2] - cam.position[2]);
cam.azimuth = spec.azimuth;
cam.elevation = -20;
const zen = new Z.CityZen(cam, lots, random);
const here = () => [cam.surfacePosition[0], cam.surfacePosition[1],
                    cam.position[2]];
const steps = Math.round(spec.minutes * 60 / DT), settle = Math.round(20 / DT);
let prev = here(), hits = 0, offGrid = 0, zmin = Infinity, zmax = -Infinity;
let maxYaw = 0, lastAz = cam.azimuth, flown = 0, zEnd = 0;
for (let k = 0; k < steps; k++) {
  zen.step(DT, cam);
  const p = here();
  // Unwrapped against the last position: the tile folds under the flight.
  for (const i of [0, 1]) {
    if (p[i] - prev[i] > EXTENT / 2) p[i] -= EXTENT;
    if (p[i] - prev[i] < -EXTENT / 2) p[i] += EXTENT;
  }
  if (lots.sweep(prev, p, spec.reach) !== null) hits += 1;
  flown += Math.hypot(p[0] - prev[0], p[1] - prev[1]);
  prev = here();
  maxYaw = Math.max(maxYaw, Math.abs(fold(cam.azimuth - lastAz)) / DT);
  lastAz = cam.azimuth;
  if (k < settle) continue;
  // On a street's centre line, or within a turn's radius of a grid node.
  const off = [0, 1].map((i) => Math.abs(p[i] - Math.round(p[i] / CELL) * CELL));
  if (Math.min(off[0], off[1]) > 1e-6
      && Math.max(off[0], off[1]) > Z.ZEN_CITY_TURN_RADIUS_M + 1e-6) offGrid += 1;
  zmin = Math.min(zmin, p[2]);
  zmax = Math.max(zmax, p[2]);
  zEnd = p[2];
}
console.log(JSON.stringify({
  hits, offGrid, blocked: zen.blocked, rising: zen.rising, zmin, zmax, zEnd,
  flown, maxYaw, elevation: cam.elevation, range: Z.ZEN_CITY_ALTITUDE_M,
  speed: Z.ZEN_CITY_SPEED, yawRate: Z.ZEN_CITY_YAW_RATE,
}));
"""

# The cyberpunk start camera's place (constants.js CITY_START_CAMERA), a
# place by the tile's seam, and downtown among the towers.
CITY_STARTS = {
    "start camera": ([24240.74, 68827.51, 26.88], 315.6),
    "tile seam": ([92100.0, 68850.0, 30.0], 90.0),
    "downtown": ([17393.0, 61646.0, 40.0], 200.0),
}


@pytest.mark.parametrize("name", sorted(CITY_STARTS))
def test_the_city_walk_follows_the_streets_and_touches_nothing(name):
    start, azimuth = CITY_STARTS[name]
    r = _node(CITY_FLIGHT_JS, {"seed": 3, "start": start, "azimuth": azimuth,
                               "minutes": 40, "reach": 0.6})
    assert r["rising"] is False
    assert r["hits"] == 0
    assert r["blocked"] == 0
    assert r["offGrid"] == 0
    low, high = r["range"]
    assert low - 1e-6 <= r["zmin"] and r["zmax"] <= high + 1e-6
    assert r["zmax"] - r["zmin"] > 100.0
    assert r["flown"] == pytest.approx(r["speed"] * 40 * 60, rel=0.01)
    assert r["maxYaw"] <= r["yawRate"] + 1e-6
    assert r["elevation"] == 0.0


def test_the_city_walk_comes_down_from_above_its_ceiling():
    r = _node(CITY_FLIGHT_JS, {"seed": 9, "start": [24240.0, 68827.0, 2000.0],
                               "azimuth": 90.0, "minutes": 15, "reach": 0.6})
    low, high = r["range"]
    assert r["hits"] == 0 and r["blocked"] == 0
    assert low <= r["zEnd"] <= high


def test_the_street_word_is_read_as_the_lot_mirror_reads_it():
    """A street leaves a node along a block edge exactly where the lot
    mirror says that edge carries one (cityCell's `streets` bits: -x, +x,
    -y, +y), over a few thousand blocks."""
    r = _node(CITY_JS + """
    let checked = 0, wrong = 0;
    for (let j = 300; j < 360; j++) for (let i = 500; i < 560; i++) {
      const s = cityCell(tile, i, j).streets;
      const pairs = [
        [(s >>> 0) & 1, Z.streetFrom(tile, i, j, 1)],        // -x edge
        [(s >>> 1) & 1, Z.streetFrom(tile, i + 1, j, 1)],    // +x edge
        [(s >>> 2) & 1, Z.streetFrom(tile, i, j, 0)],        // -y edge
        [(s >>> 3) & 1, Z.streetFrom(tile, i, j + 1, 0)],    // +y edge
        [(s >>> 0) & 1, Z.streetFrom(tile, i, j + 1, 3)],    // -x, from its far end
        [(s >>> 2) & 1, Z.streetFrom(tile, i + 1, j, 2)],    // -y, from its far end
      ];
      for (const [bit, street] of pairs) {
        checked += 1;
        if (Boolean(bit) !== street) wrong += 1;
      }
    }
    console.log(JSON.stringify({ checked, wrong }));
    """, {"seed": 0})
    assert r["checked"] == 6 * 3600 and r["wrong"] == 0
