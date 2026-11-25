"""Zen mode's flight (web/soar/zen.js), flown in node.

Over a cloud field: the altitude stays in the band that straddles the bottom
of the cloud layer and spends about half its time under the cloud where
there is room under it, the turn is never tighter than a circle in
ZEN_CLOUD_CIRCLE_S, the look ends level, and the speed is
ZEN_CLOUD_CELLS_PER_S of the field's grid cells a second.

Over the city, through the shipped tile with everything that stands in it
(citylots.js, cityprops.js): the flight never comes within a body's reach of
anything solid and never has to rise out of a place it planned itself into;
below the buildings' range it is on a street's centre line, flying straight
along it; it spends something like half its time at street level; and it
comes down into its range when started high above it. The street word's
reading is held to the lot mirror's.

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
const nominal = Z.cloudSpeed(spec.bmin, spec.bmax, spec.shape);
const zen = new Z.CloudZen(cam, nominal, random);
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
  nominal, cells: Z.ZEN_CLOUD_CELLS_PER_S,
}));
"""

# The demo set's shapes: a thin deck, a layer on the sea, and a deep one
# that starts high (web/soar/demos/*/meta.json bmin, bmax, shape_xyz).
FIELDS = {
    "dycoms": ([-2.5, -2.5, 537.5], [3197.5, 3197.5, 887.5], [640, 640, 140]),
    "congestus": ([-5.0, -5.0, 25.0], [20475.0, 20475.0, 3875.0],
                  [2048, 2048, 385]),
    "stratified": ([-50.0, -50.0, 7011.4], [102350.0, 102350.0, 14934.0],
                   [1024, 1024, 300]),
}


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_the_cloud_walk_keeps_its_band_and_its_gentle_turn(name):
    bmin, bmax, shape = FIELDS[name]
    r = _node(CLOUD_JS, {"seed": 11, "bmin": bmin, "bmax": bmax,
                         "shape": shape, "periodic": True, "minutes": 90})
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
    # So many cells a second, whatever the grid: slow on DYCOMS's 5 m, fast
    # on the cirrus field's 100 m.
    cell = (bmax[0] - bmin[0]) / shape[0]
    assert r["nominal"] == pytest.approx(r["cells"] * cell)
    assert r["speed"] == pytest.approx(r["nominal"], rel=1e-3)


DART_JS = """
import { Dart } from "%(soar)s/dart.js";
// The dart's flight dynamics without its GPU half: Dart.update reads and
// writes only these fields (flyer.js's constructor sets them beside the
// buffers and pipelines, which need a device).
const dart = Object.assign(Object.create(Dart.prototype), {
  position: [0, 0, 0], heading: 0, course: 0, viewElevation: 0, bank: 0,
  pitch: 0, _speed: 0, _vz: 0, _clock: 0, _prevOrigin: null, _knock: null,
  distance: 2.6, drop: 0.3, wingFlex: 0, keelSway: 0, wobblePhase: 0,
  wobbleAmp: 0, flutterPhase: 0, flutterAmp: 0 });
const cam = new FlightCamera(spec.bmin, spec.bmax, { periodic: true });
cam.position = [spec.bmax[0] - 200, 0.5 * (spec.bmin[1] + spec.bmax[1]), 700];
cam.azimuth = 80;
cam.elevation = 0;
const zen = new Z.CloudZen(cam, Z.cloudSpeed(spec.bmin, spec.bmax, spec.shape),
                           random);
let wraps = 0, maxBank = 0, maxStep = 0, last = [...cam.position];
for (let k = 0; k < Math.round(spec.minutes * 60 / DT); k++) {
  zen.step(DT, cam);
  const before = dart.course;
  dart.update(DT, cam);
  if ([0, 1].some((i) => Math.abs(cam.position[i] - last[i]) > 1000)) wraps += 1;
  last = [...cam.position];
  if (k < 60) continue;
  maxBank = Math.max(maxBank, Math.abs(dart.bank));
  maxStep = Math.max(maxStep, Math.abs(fold(dart.course - before)));
}
console.log(JSON.stringify({ wraps, maxBank, maxStep }));
""" % {"soar": SOAR}


def test_the_dart_crosses_the_periodic_wrap_without_a_kick():
    """The camera's x and y fold at the domain's faces; the dart's course
    and bank come from the camera's motion, which must not see the fold.
    Measured before the fix: a 10 degree course jump in one frame at every
    wrap and the bank rung to 8 degrees, a kick sideways."""
    bmin, bmax, shape = FIELDS["dycoms"]
    r = _node(DART_JS, {"seed": 4, "bmin": bmin, "bmax": bmax,
                        "shape": shape, "minutes": 5})
    assert r["wraps"] >= 3
    # The zen turn is at most 1.2 degrees a second: well under a degree of
    # bank, and a course that moves a fiftieth of a degree a frame.
    assert r["maxBank"] < 1.0
    assert r["maxStep"] < 0.05


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
const steps = Math.round(spec.minutes * 60 / DT), settle = Math.round(120 / DT);
const BELOW = Z.ZEN_CITY_ALTITUDE_M[0], STREET = Z.ZEN_CITY_STREET_ALTITUDE_M;
let prev = here(), hits = 0, zmax = -Infinity, zEnd = 0, maxYaw = 0;
let lastAz = cam.azimuth, flown = 0;
let low = 0, lowOffStreet = 0, lowTurning = 0, lowOutOfRange = 0, atStreet = 0;
let highInFirstMinute = 0;
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
  const yaw = Math.abs(fold(cam.azimuth - lastAz)) / DT;
  lastAz = cam.azimuth;
  zEnd = p[2];
  if (k < 60 / DT && p[2] >= BELOW) highInFirstMinute += 1;
  if (k < settle) continue;
  maxYaw = Math.max(maxYaw, yaw);
  zmax = Math.max(zmax, p[2]);
  if (p[2] <= STREET[1] + 1e-6) atStreet += 1;
  if (p[2] >= BELOW) continue;
  // Below the buildings' range: on a grid line that is a street here,
  // heading along it, not turning, and no lower than the street range.
  low += 1;
  const node = [0, 1].map((i) => Math.round(p[i] / CELL));
  const off = [0, 1].map((i) => Math.abs(p[i] - node[i] * CELL));
  const alongX = off[1] < off[0];
  const at = Math.floor(p[alongX ? 0 : 1] / CELL);
  const street = alongX ? Z.streetFrom(tile, at, node[1], 0)
                        : Z.streetFrom(tile, node[0], at, 1);
  if (Math.min(off[0], off[1]) > 1e-6 || !street) lowOffStreet += 1;
  if (yaw > 1e-6 || Math.abs(fold(cam.azimuth - (alongX ? 90 : 0))) % 180 > 1e-6) {
    lowTurning += 1;
  }
  if (p[2] < STREET[0] - 1e-6) lowOutOfRange += 1;
}
console.log(JSON.stringify({
  hits, stuck: zen.stuck, rising: zen.rising, zmax, zEnd, flown, maxYaw,
  low, lowOffStreet, lowTurning, lowOutOfRange, highInFirstMinute,
  atStreet: atStreet / (steps - settle), elevation: cam.elevation,
  range: Z.ZEN_CITY_ALTITUDE_M, speed: Z.ZEN_CITY_SPEED,
  avoid: Z.ZEN_CITY_AVOID_RATES,
}));
"""

# The cyberpunk start camera's place (constants.js CITY_START_CAMERA), a
# place by the tile's seam, and a place in the middle of the tile, high.
CITY_STARTS = {
    "start camera": ([24240.74, 68827.51, 60.0], 315.6),
    "tile seam": ([92100.0, 68850.0, 120.0], 90.0),
    "mid tile": ([46080.0, 46080.0, 300.0], 10.0),
}


@pytest.mark.parametrize("name", sorted(CITY_STARTS))
def test_the_city_walk_curves_above_and_flies_the_streets_straight(name):
    start, azimuth = CITY_STARTS[name]
    r = _node(CITY_FLIGHT_JS, {"seed": 3, "start": start, "azimuth": azimuth,
                               "minutes": 60, "reach": 0.6})
    assert r["hits"] == 0
    assert r["stuck"] == 0 and r["rising"] is False
    # It does go down to the streets, and what it does there is fly straight
    # along one.
    assert r["low"] > 0
    assert r["lowOffStreet"] == 0
    assert r["lowTurning"] == 0
    assert r["lowOutOfRange"] == 0
    assert 0.25 < r["atStreet"] < 0.75
    assert r["zmax"] <= r["range"][1] + 1e-6
    assert r["flown"] == pytest.approx(r["speed"] * 60 * 60, rel=0.01)
    assert r["maxYaw"] <= max(r["avoid"]) + 1e-6
    assert r["elevation"] == 0.0


def test_the_city_walk_started_in_a_street_takes_that_street():
    """Zen begun low in a street, facing along it and off its centre line,
    flies that street first: it does not rise out of it in the first minute,
    and it is never left with nothing to fly."""
    # 5 m up: under the lamp heads (8.6 m), which reach in over the kerb.
    r = _node(CITY_FLIGHT_JS, {"seed": 5, "start": [24195.0, 68851.5, 5.0],
                               "azimuth": 265.0, "minutes": 30, "reach": 0.6})
    assert r["hits"] == 0 and r["stuck"] == 0
    assert r["highInFirstMinute"] == 0
    assert r["lowOffStreet"] == 0 and r["lowTurning"] == 0


def test_the_city_walk_comes_down_from_above_its_ceiling():
    r = _node(CITY_FLIGHT_JS, {"seed": 9, "start": [24240.0, 68827.0, 2000.0],
                               "azimuth": 90.0, "minutes": 25, "reach": 0.6})
    assert r["hits"] == 0 and r["stuck"] == 0
    assert r["zEnd"] <= r["range"][1]


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
