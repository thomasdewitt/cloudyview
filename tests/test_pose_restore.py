"""A capture's pose, handed back to the camera, is the pose it was taken at.

The QA tools stand the camera at a soar screenshot's pose (tools/
steam_live_qa.mjs and tools/soar_noise.mjs, --pose-png): the capture's
city.position_m is the camera's surfacePosition and its z (viewer.js
_renderMetadata), and the tools move the camera by the difference. They
did it with camera.displace, which is a move of flight, so the city
collider's next resolve swept the jump — tens of kilometres from wherever
the camera stood — and stopped at the first wall on the way. Thomas's
screenshots at z 14.5 m and 6.9 m came back at 294 m and 910 m, over other
streets (2026-11-09). They now move by camera.jump, which bumps the epoch
the collider tracks the flight by.

This flies the real FlightCamera and CityCollider in node over a stand-in
city whose wall stands between the spawn and the capture's place, runs the
frames the viewer runs after the jump, and requires the pose to come back
to the metres — the round trip of the metadata through the restore. The
negative control is displace, which must not.

Needs node; no GPU.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOAR = REPO / "web" / "soar"
TOOLS = REPO / "tools"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

SCRIPT = textwrap.dedent("""
import { FlightCamera, worldToRelative } from "%(soar)s/camera.js";
import { CityCollider } from "%(soar)s/collide.js";

const EXTENT = 92160.0;
// One wall across the whole way from the spawn to the capture's place,
// the only thing in this city; and the street. The collider works in the
// city frame (surfacePosition), so the wall stands 500 m east of where the
// spawn's world x = 500 lands in it.
const SX = ((500.0 - 79780.0) %% EXTENT + EXTENT) %% EXTENT;
const WALL = [[SX + 500.0, -1e6, 0.0], [SX + 510.0, 1e6, 500.0]];
const city = {
  sweep(p0, p1, r) {
    const d = [0, 1, 2].map((k) => p1[k] - p0[k]);
    let best = null;
    if (d[2] < 0 && p0[2] >= r) {
      const t = (r - p0[2]) / d[2];
      if (t <= 1) best = { t, normal: [0, 0, 1] };
    }
    const lo = WALL[0].map((v) => v - r), hi = WALL[1].map((v) => v + r);
    if (Math.abs(d[0]) > 1e-12) {
      const t = ((d[0] > 0 ? lo[0] : hi[0]) - p0[0]) / d[0];
      const q = [0, 1, 2].map((k) => p0[k] + t * d[k]);
      if (t >= 0 && t <= 1 && q[1] >= lo[1] && q[1] <= hi[1]
          && q[2] >= lo[2] && q[2] <= hi[2] && (best === null || t < best.t)) {
        best = { t, normal: [d[0] > 0 ? -1 : 1, 0, 0] };
      }
    }
    if (best === null) return null;
    return { t: best.t, normal: best.normal,
             point: [0, 1, 2].map((k) => p0[k] + best.t * d[k]) };
  },
  inside(p, r) {
    return p[0] > WALL[0][0] - r && p[0] < WALL[1][0] + r
        && p[2] < WALL[1][2] + r ? {} : null;
  },
};

// What the viewer writes into a capture (_renderMetadata's city block).
const metadata = (camera) => ({
  position_m: [...camera.surfacePosition, camera.position[2]] });

// The flight loop's per-frame order (viewer.js), no keys held.
function frames(camera, collider, n) {
  for (let i = 0; i < n; i++) {
    collider.advance(1 / 60, camera);
    camera.move(1 / 60);
    camera.constrain();
    collider.resolve(1 / 60, camera, null);
  }
}

function restore(how, capture) {
  const BMIN = [-30000, -30000, 0], BMAX = [30000, 30000, 6000];
  const camera = new FlightCamera(BMIN, BMAX, {
    start: { position: worldToRelative([500.0, 2000.0, 309.0], BMIN, BMAX),
             azimuth: 90, elevation: 0, fov: 90 },
    surfaceTileExtent: EXTENT, surfaceOffsetM: [79780.0, 47970.0],
    speed: 20, floorZ: null });
  const collider = new CityCollider(city, EXTENT);
  frames(camera, collider, 3);                      // seeded, flying
  const [x, y, z] = capture.position_m;
  const s = camera.surfacePosition;
  // The tools' own expression (steam_live_qa place, soar_noise).
  camera[how](x - s[0], y - s[1], z - camera.position[2]);
  frames(camera, collider, 30);
  return metadata(camera).position_m;
}

// Where the capture was taken: east of the wall, low over a street.
const [x0, y0] = [500.0, 2000.0];
const capture = { position_m: [(x0 + 1500.0 - 79780.0 + 4 * EXTENT) %% EXTENT,
                               (y0 + 25.0 - 47970.0 + 4 * EXTENT) %% EXTENT,
                               6.9] };
console.log(JSON.stringify({ capture: capture.position_m,
                             jump: restore("jump", capture),
                             displace: restore("displace", capture) }));
""")


def _run():
    out = subprocess.run(
        ["node", "--input-type=module", "-e", SCRIPT % {"soar": SOAR}],
        capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_captures_pose_round_trips():
    r = _run()
    for k in range(3):
        assert abs(r["jump"][k] - r["capture"][k]) < 1e-6, r


def test_displace_is_swept_and_does_not():
    # The negative control: the old placement, swept into the wall.
    r = _run()
    assert max(abs(r["displace"][k] - r["capture"][k])
               for k in range(3)) > 100.0, r


def test_the_tutorial_spawn_places_by_jump():
    # The viewer's own absolute city reposition: the tutorial's spawn, which
    # a replay from the menu makes mid-flight (tutorial.js -> tutorialSpawn).
    src = (SOAR / "viewer" / "input.js").read_text()
    start = src.index("  tutorialSpawn() {")
    body = src[start:src.index("\n  },\n", start)]
    assert "this.camera.jump(" in body and "displace(" not in body
