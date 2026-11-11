"""Flying into the night city: stopped at the wall, knocked, and let go.

web/soar/collide.js is the flight collider: a building flown into stops the
flight and rocks the flyer, which loses control for a moment and falls a
little. These tests fly the real FlightCamera through the real collider,
frame by frame in node, in the order viewer/loop.js runs them, with a
stand-in flyer placed the way flyer.js places one (ahead of the camera by
its distance, hanging below by its drop):

  - into a wall of known position, asserting that neither the camera nor
    the flyer ever gets inside it, that the flight bounces back off it,
    that the episode lasts about a second with the controls dead and then
    back, that it drops some metres, and that the shake leaves the look
    exactly where it was;
  - the same crash at three frame rates, which must agree;
  - a dive into the street, which is a landing: shorter, no drop, never
    below the ground;
  - a crash low enough that the fall would go through the street;
  - at 2 km/s into a mast 1.4 m across, which a point test would miss;
  - into a real building of the shipped tile, through citylots.js;
  - into an elevated freeway's deck and a lamp post's mast, through
    cityprops.js.

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
TILE_DIR = REPO / "cloudyview" / "soar" / "city"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

# A stand-in city of boxes over a ground plane, answering the two queries
# collide.js asks of citylots.js — deliberately written independently of
# citylots.js, so these tests are about the collider and nothing else.
BOX_CITY_JS = """
class BoxCity {
  constructor(boxes) { this.boxes = boxes; }
  sweep(p0, p1, r) {
    const d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let best = null;
    if (d[2] < 0 && p0[2] >= r) {
      const t = (r - p0[2]) / d[2];
      if (t <= 1) best = { t, normal: [0, 0, 1] };
    }
    for (const [blo, bhi] of this.boxes) {
      const lo = blo.map((v) => v - r), hi = bhi.map((v) => v + r);
      let t0 = -Infinity, t1 = Infinity, ax = -1;
      let miss = false;
      for (let k = 0; k < 3; k++) {
        if (Math.abs(d[k]) < 1e-12) {
          if (p0[k] < lo[k] || p0[k] > hi[k]) { miss = true; break; }
          continue;
        }
        let a = (lo[k] - p0[k]) / d[k], b = (hi[k] - p0[k]) / d[k];
        if (a > b) [a, b] = [b, a];
        if (a > t0) { t0 = a; ax = k; }
        t1 = Math.min(t1, b);
      }
      if (miss || ax < 0 || t0 > t1 || t0 < 0 || t0 > 1) continue;
      if (best === null || t0 < best.t) {
        const n = [0, 0, 0]; n[ax] = d[ax] > 0 ? -1 : 1;
        best = { t: t0, normal: n };
      }
    }
    if (best === null) return null;
    return { t: best.t, normal: best.normal,
             point: [p0[0] + best.t * d[0], p0[1] + best.t * d[1],
                     p0[2] + best.t * d[2]] };
  }
  inside(p, r) {
    if (p[2] < r) return { top: 0 };
    for (const [lo, hi] of this.boxes) {
      if (p.every((v, k) => v > lo[k] - r && v < hi[k] + r)) {
        return { top: hi[2] };
      }
    }
    return null;
  }
}
"""

# The flight loop, as viewer/loop.js _frame runs it, with a stand-in flyer.
FLY_JS = """
import { FlightCamera, cameraBasis, worldToRelative } from "%(soar)s/camera.js";
import { CityCollider } from "%(soar)s/collide.js";
import { TrackRecorder, resampleTrack, knocksDue } from "%(soar)s/track.js";
import { CityTile, CityLots, cityCell } from "%(soar)s/citylots.js";
import { CityProps } from "%(soar)s/cityprops.js";
import { readFileSync } from "node:fs";
%(box_city)s

const EXTENT = 92160.0;

// flyer.js _place, without the smoothing: ahead by `distance` along the
// look (turned by `yaw` degrees, to stand in for a flyer whose smoothed
// heading lags the view), below by `drop`.
class StandInFlyer {
  constructor(distance = 2.6, yaw = 0.0) {
    this.position = [0, 0, 0]; this.knocks = []; this.calms = 0;
    this.distance = distance; this.yaw = yaw;
  }
  update(dt, camera) {
    const [f, , u] = cameraBasis(camera.azimuth + this.yaw, camera.elevation);
    const o = camera.position;
    this.position = [0, 1, 2].map(
      (i) => o[i] + this.distance * f[i] - 0.62 * u[i]);
  }
  shift(d) { this.position = this.position.map((v, i) => v + d[i]); }
  knock(args) { this.knocks.push(args); }
  calm() { this.calms += 1; }
}

function fly({ city, start, azimuth, elevation, speed, keys, fps, seconds,
               releaseOnKnock = true,
               // The CLOUD domain, which the camera's world position folds
               // at — independent of the city tile's frame.
               domain = [[-200000, -200000, 0], [200000, 200000, 6000]],
               videoFps = null,
               // Mouse look between frames, deg/s of azimuth.
               turn = 0.0,
               // [[fromFrame, distance, yaw], ...]: which flyer is up when.
               flyers = [[0, 2.6, 0.0]],
               // [frame, x, y]: a map click.
               teleport = null }) {
  const [BMIN, BMAX] = domain;
  const recorder = new TrackRecorder();
  recorder.start();
  const camera = new FlightCamera(BMIN, BMAX, {
    start: { position: worldToRelative(start, BMIN, BMAX),
             azimuth, elevation, fov: 90 },
    surfaceTileExtent: EXTENT, surfaceOffsetM: [0, 0], speed,
    floorZ: null });
  const collider = new CityCollider(city, EXTENT);
  const shown = flyers.map(([from, dist, yaw]) =>
    [from, new StandInFlyer(dist, yaw)]);
  const flyerAt = (i) => shown.filter(([from]) => from <= i).pop()[1];
  let flyer = flyerAt(0);
  const knocks = [];
  for (const k of keys) camera.keys.add(k);
  const dt = 1 / fps;
  const frames = [];
  for (let i = 0; i < Math.round(seconds * fps); i++) {
    camera.azimuth = ((camera.azimuth + turn * dt) %% 360 + 360) %% 360;
    if (teleport && teleport[0] === i) camera.teleport(teleport[1], teleport[2]);
    flyer = flyerAt(i);
    collider.advance(dt, camera);
    camera.move(dt);
    camera.constrain();
    flyer.update(dt, camera);
    const knock = collider.resolve(dt, camera, flyer);
    if (knock) { recorder.noteKnock(knock); knocks.push(knock); }
    recorder.advance(dt);
    // v3: the scene clock rides with the flight, as the viewer's does
    // while a track records.
    recorder.sample(camera, recorder.elapsed);
    if (releaseOnKnock && knocks.length) camera.keys.clear();
    const sp = camera.surfacePosition;
    const cam = [sp[0], sp[1], camera.position[2]];
    const fly = [0, 1, 2].map((k) =>
      cam[k] + flyer.position[k] - camera.position[k]);
    frames.push({
      t: (i + 1) * dt, cam, fly,
      az: camera.azimuth, el: camera.elevation,
      control: camera.control, active: collider.active,
      tether: Math.hypot(...[0, 1, 2].map(
        (k) => flyer.position[k] - camera.position[k])),
      camInside: city.inside(cam, 0.35) !== null,
      flyInside: city.inside(fly, 0.2) !== null,
    });
  }
  const out = { frames, knocks,
                calms: shown.map(([, f]) => f.calms),
                recorded: { samples: recorder.samples,
                            knocks: recorder.knocks } };
  if (videoFps !== null) {
    // What the video pass hands the flyer, frame by frame.
    const video = resampleTrack(recorder.samples, videoFps,
                                { linearPosition: true });
    out.replayKnocks = video.map((f, i) => knocksDue(
      recorder.knocks, i > 0 ? video[i - 1].t : -Infinity, f.t));
    out.videoT = video.map((f) => f.t);
    out.videoCam = video.map((f) => [f.surfacePosition[0] * EXTENT,
                                     f.surfacePosition[1] * EXTENT,
                                     (f.position[2] + 1) * 0.5 * BMAX[2]]);
  }
  return out;
}

const spec = JSON.parse(readFileSync(0, "utf8"));
let city;
if (spec.city === "tile") {
  const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
  const b = readFileSync("%(tile)s/fif_mip0.bin");
  const tile = CityTile.fromHalfBytes(
    new Uint8Array(b.buffer, b.byteOffset, b.byteLength),
    meta.n, meta.cell_m);
  city = new CityLots(tile);
  // The tallest plain slab near the requested cell, approached square-on
  // at a third of its height from 40 m off its west face — through clear
  // air: the start must be outside every building and the first thing the
  // approach meets must be this slab's own (shell-grown) face.
  let pick = null;
  const startOf = (c) => [c.b1min[0] - 40,
                          0.5 * (c.b1min[1] + c.b1max[1]), c.height / 3];
  for (let cy = spec.near[1] - 20; cy < spec.near[1] + 20; cy++) {
    for (let cx = spec.near[0] - 20; cx < spec.near[0] + 20; cx++) {
      const c = cityCell(tile, cx, cy);
      if (!(c.built && c.tiers === 1 && !c.has_frustum && c.height > 60
            && (pick === null || c.height > pick.height))) continue;
      const s0 = startOf(c);
      if (city.inside(s0)) continue;
      const hit = city.sweep(s0, [c.b1min[0] + 1.0, s0[1], s0[2]]);
      if (hit === null || hit.point[0] < c.b1min[0] - 2.0) continue;
      pick = c;
    }
  }
  if (pick === null) throw new Error("no clear slab near the requested cell");
  spec.start = startOf(pick);
  spec.wallX = pick.b1min[0];
} else if (spec.city === "props") {
  // The shipped tile with what stands among its buildings (cityprops.js).
  const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
  const b = readFileSync("%(tile)s/fif_mip0.bin");
  const tile = CityTile.fromHalfBytes(
    new Uint8Array(b.buffer, b.byteOffset, b.byteLength),
    meta.n, meta.cell_m);
  city = new CityLots(tile, new CityProps(tile));
  if (spec.mast) {
    // A lamp mast near the requested cell, approached level at 4 m, below
    // the arm, from 12 m off along whichever axis meets the mast first from
    // clear air. The mast's centre is read off its box's planes.
    const [cx0, cy0] = spec.mast;
    let found = null;
    for (let cy = cy0; cy < cy0 + 4 && found === null; cy++) {
      for (let cx = cx0; cx < cx0 + 4 && found === null; cx++) {
        const props = city.props.cellSolids(cityCell(tile, cx, cy), cx, cy, 0);
        for (const sd of props.solids) {
          if (sd.source !== "lamp" || sd.top < 8.9) continue;
          const pl = sd.planes;
          const c = [0.5 * (pl[0][3] - pl[1][3]), 0.5 * (pl[2][3] - pl[3][3]), 4.0];
          for (const [ux, uy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const s0 = [c[0] - 12 * ux, c[1] - 12 * uy, 4.0];
            if (city.inside(s0, 0.35)) continue;
            const hit = city.sweep(s0, c, 0.2);
            if (hit && hit.source === "lamp") {
              found = { s0, c, az: (Math.atan2(ux, uy) * 180 / Math.PI + 360) %% 360 };
              break;
            }
          }
          if (found) break;
        }
      }
    }
    if (found === null) throw new Error("no clear lamp mast near the requested cell");
    spec.start = found.s0;
    spec.azimuth = found.az;
    spec.mastAt = found.c;
  }
} else {
  city = new BoxCity(spec.boxes);
}
process.stdout.write(JSON.stringify({ ...fly({ ...spec, city }), wallX: spec.wallX,
                                     mastAt: spec.mastAt }));
""" % {"soar": SOAR.as_posix(), "tile": TILE_DIR.as_posix(),
       "box_city": BOX_CITY_JS}


def fly(**spec):
    out = subprocess.run(["node", "--input-type=module", "--eval", FLY_JS],
                         input=json.dumps(spec), capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# A wall 30 m thick facing west at x = 50, 500 m tall and 2 km long.
WALL = [[[50.0, -1000.0, 0.0], [80.0, 1000.0, 500.0]]]
EAST = 90.0


def _episode(frames):
    """(first, last) index of the frames during which an episode ran."""
    on = [i for i, f in enumerate(frames) if f["active"]]
    assert on, "no episode ever started"
    assert on == list(range(on[0], on[-1] + 1)), "the episode restarted"
    return on[0], on[-1]


def test_a_wall_stops_bounces_knocks_and_lets_go():
    r = fly(boxes=WALL, start=[0.0, 0.0, 60.0], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=5.0)
    frames = r["frames"]
    # Never inside, either of them, and never past the face.
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert max(f["fly"][0] for f in frames) <= 50.0 - 0.2 + 1e-9
    assert max(f["cam"][0] for f in frames) <= 50.0 - 0.2 + 1e-9

    # One knock: a wall, not a landing, severe at 20 m/s (0.8 of full).
    assert len(r["knocks"]) == 1
    knock = r["knocks"][0]
    assert knock["landing"] is False
    assert knock["severity"] == pytest.approx(0.8, abs=0.05)

    first, last = _episode(frames)
    hit = frames[first]
    # The plane, not the camera, met the wall: it flies 2.6 m ahead, and
    # stops with its 0.2 m body touching the face (plus the 1 cm the
    # collider backs out by).
    assert hit["fly"][0] == pytest.approx(50.0 - 0.21, abs=1e-6)
    assert hit["cam"][0] == pytest.approx(50.0 - 0.21 - 2.6, abs=1e-6)
    # About a second of lost control.
    duration = frames[last]["t"] - hit["t"]
    assert 0.8 <= duration <= 1.2
    assert duration == pytest.approx(knock["duration"], abs=2 / 60)
    # A bounce back off the wall, a few metres (2.7 here; 1.75 before the
    # 2026-11 ask for a harder rock, which this lower bound now holds).
    back = hit["cam"][0] - min(f["cam"][0] for f in frames[first:last + 1])
    assert 2.5 < back < 3.5
    # A drop of five to fifteen metres (13 m at this severity).
    drop = hit["cam"][2] - frames[last]["cam"][2]
    assert 5.0 <= drop <= 15.0
    assert drop == pytest.approx(5.0 + 10.0 * knock["severity"], abs=0.05)
    # The controls: nearly dead early, fully back after.
    early = frames[first + 6]
    assert early["control"] < 0.15
    assert frames[last + 1]["control"] == 1.0
    # The view was rocked — by several degrees (3.6 here; about 1.7 before
    # the 2026-11 ask for a harder rock) — and is exactly where it was
    # afterwards.
    shaken = max(abs(f["az"] - EAST) + abs(f["el"])
                 for f in frames[first:last + 1])
    assert 3.0 < shaken < 5.0
    assert frames[last + 1]["az"] == pytest.approx(EAST, abs=1e-9)
    assert frames[last + 1]["el"] == pytest.approx(0.0, abs=1e-9)
    assert frames[-1]["az"] == pytest.approx(EAST, abs=1e-9)


def test_the_crash_does_not_depend_on_the_frame_rate():
    runs = {fps: fly(boxes=WALL, start=[0.0, 0.0, 60.0], azimuth=EAST,
                     elevation=0.0, speed=20.0, keys=["w"], fps=fps,
                     seconds=5.0)
            for fps in (30, 60, 144)}
    summary = {}
    for fps, r in runs.items():
        frames = r["frames"]
        first, last = _episode(frames)
        hit = frames[first]
        summary[fps] = (
            hit["cam"][2] - frames[-1]["cam"][2],                    # drop
            hit["cam"][0] - frames[-1]["cam"][0],                    # bounce
            frames[last]["t"] - hit["t"],                            # length
        )
    drop60, back60, len60 = summary[60]
    for fps, (drop, back, length) in summary.items():
        # The impact itself lands on a frame boundary, which moves the
        # arrival speed (and so severity) by at most one frame's travel.
        assert drop == pytest.approx(drop60, rel=0.03), fps
        assert back == pytest.approx(back60, rel=0.05), fps
        assert length == pytest.approx(len60, abs=2.5 / 30), fps


def test_diving_into_the_street_is_a_landing():
    # Looking 40 degrees down, descending, going forward: the flyer, hung
    # below and ahead, meets the asphalt first.
    r = fly(boxes=[], start=[0.0, 0.0, 12.0], azimuth=EAST, elevation=-40.0,
            speed=20.0, keys=["w", "shift"], fps=60, seconds=3.0)
    frames = r["frames"]
    assert min(f["fly"][2] for f in frames) >= 0.2 - 1e-9
    assert min(f["cam"][2] for f in frames) >= 0.35 - 1e-9
    assert len(r["knocks"]) == 1 and r["knocks"][0]["landing"] is True
    first, last = _episode(frames)
    assert frames[last]["t"] - frames[first]["t"] < 0.65
    # A hop, not a fall: nowhere lower to go.
    assert frames[last]["cam"][2] >= frames[first]["cam"][2] - 1e-6
    assert max(f["cam"][2] for f in frames[first:last + 1]) \
        > frames[first]["cam"][2] + 0.3


def test_the_fall_stops_at_the_street():
    # Hit low: the 13 m drop would go 7 m under the asphalt.
    r = fly(boxes=WALL, start=[0.0, 0.0, 6.0], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=5.0)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert min(f["fly"][2] for f in frames) >= 0.2 - 1e-9
    assert min(f["cam"][2] for f in frames) >= 0.35 - 1e-9
    assert len(r["knocks"]) == 1   # landing mid-fall is not a second crash


def test_two_km_per_second_into_a_mast_does_not_tunnel():
    mast = [[[400.0, -0.7, 0.0], [401.4, 0.7, 300.0]]]
    r = fly(boxes=mast, start=[0.0, 0.0, 100.0], azimuth=EAST, elevation=0.0,
            speed=2000.0, keys=["w"], fps=60, seconds=1.0)
    frames = r["frames"]
    # 33 m a frame, and the mast is 1.4 m across.
    assert max(f["fly"][0] for f in frames) <= 400.0 - 0.2 + 1e-9
    assert max(f["cam"][0] for f in frames) <= 400.0
    assert len(r["knocks"]) == 1 and r["knocks"][0]["severity"] == 1.0


def test_a_real_building_of_the_shipped_tile():
    # Downtown, near where the cyberpunk session opens (CITY_START_CAMERA
    # is at city (17393, 61646), cell (193, 684)).
    r = fly(city="tile", near=[193, 684], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=5.0)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert max(f["fly"][0] for f in frames) <= r["wallX"] - 0.2 + 1e-6
    assert len(r["knocks"]) == 1
    _episode(frames)


def test_an_elevated_freeway_is_solid():
    # Level at 25.3 m, north across the x-route on y = 51840 (block row
    # 576), from inside its avenue corridor: the plane meets the deck's
    # fascia (its south face at y = 51832, from 24.6 to 26 m).
    r = fly(city="props", start=[28010.0, 51826.0, 25.3], azimuth=0.0,
            elevation=0.0, speed=20.0, keys=["w"], fps=60, seconds=3.0)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert max(f["fly"][1] for f in frames) <= 51832.0 - 0.2 + 1e-6
    assert len(r["knocks"]) == 1 and r["knocks"][0]["landing"] is False


CREST_JS = """
import { readFileSync } from "node:fs";
import { CityTile, CityLots } from "%(soar)s/citylots.js";
import { CityProps } from "%(soar)s/cityprops.js";
const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
const b = readFileSync("%(tile)s/fif_mip0.bin");
const tile = CityTile.fromHalfBytes(
  new Uint8Array(b.buffer, b.byteOffset, b.byteLength), meta.n, meta.cell_m);
const city = new CityLots(tile, new CityProps(tile));
const segs = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(segs.map(([p0, p1]) => {
  const hit = city.sweep(p0, p1, 0.35);
  return hit && { t: hit.t, normal: hit.normal, source: hit.source };
})));
""" % {"soar": SOAR.as_posix(), "tile": TILE_DIR.as_posix()}


def test_a_raised_freeway_is_smooth_to_fly_along():
    # Resting on the high span of the y-route at x = 26640 over its crossing
    # at y = 51840 (deck top 34 m; the camera's radius and the collider's
    # clearance above it), and flying level along the deck past either
    # crest, where the ramps begin: no wall (a ramp's grown end face would
    # stand 1.4 cm proud of the flat's). Then along the low
    # deck into the foot of a ramp: met, but as a road rising, not a wall.
    x, ac, rest = 26640.0, 51840.0, 34.0 + 0.35 + 0.01
    segs = [[[x, ac - 40, rest], [x, ac - 80, rest]],
            [[x, ac + 40, rest], [x, ac + 80, rest]],
            [[x, ac - 300, 26.36], [x, ac - 250, 26.36]]]
    out = json.loads(subprocess.run(
        ["node", "--input-type=module", "--eval", CREST_JS],
        input=json.dumps(segs), capture_output=True, text=True,
        check=True).stdout)
    assert out[0] is None and out[1] is None, out
    foot = out[2]
    assert foot is not None and foot["source"] == "skyway", out
    assert foot["normal"][2] > 0.99, out


def test_a_lamp_post_is_solid():
    # A street lamp's 0.17 m mast, met level at 4 m under its arm.
    r = fly(city="props", mast=[312, 576], azimuth=0.0, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=3.0)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    mx, my = r["mastAt"][:2]
    closest = min(max(abs(f["fly"][0] - mx), abs(f["fly"][1] - my))
                  for f in frames)
    assert closest >= 0.085 + 0.2 - 1e-6
    assert len(r["knocks"]) == 1


CROWN_JS = """
import { readFileSync } from "node:fs";
import { CityTile, CityLots, cityCell, shellDepth } from "%(soar)s/citylots.js";
import { CityProps } from "%(soar)s/cityprops.js";
const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
const b = readFileSync("%(tile)s/fif_mip0.bin");
const tile = CityTile.fromHalfBytes(
  new Uint8Array(b.buffer, b.byteOffset, b.byteLength), meta.n, meta.cell_m);
const city = new CityLots(tile, new CityProps(tile));
// The first penthouse crown in the tile, on a building of its own.
let c = null, cx = 0, cy = 0;
for (cy = 0; cy < meta.n && !c; cy++) {
  for (cx = 0; cx < meta.n; cx++) {
    const cc = cityCell(tile, cx, cy);
    if (cc.crown.kind === 1 && !cc.merged && !cc.has_mast) { c = cc; break; }
  }
}
const k = c.crown, r = 0.35;
const y = 0.5 * (k.lo[1] + k.hi[1]);
const z = 0.5 * (k.z0 + k.z1);
// Level, at the crown's mid-height, west to east across the roof into it;
// and the same line one metre over the crown's top.
const into = city.sweep([k.lo[0] - 6.0, y, z], [k.hi[0], y, z], r);
const over = city.sweep([k.lo[0] - 6.0, y, k.z1 + r + 1.0],
                        [k.hi[0], y, k.z1 + r + 1.0], r);
process.stdout.write(JSON.stringify({
  crown: k, shell: shellDepth(c), height: c.height,
  into: into && { x: k.lo[0] - 6.0 + into.t * (k.hi[0] - k.lo[0] + 6.0),
                  normal: into.normal, source: into.source },
  over: over && { source: over.source } }));
""" % {"soar": SOAR.as_posix(), "tile": TILE_DIR.as_posix()}


def test_a_penthouse_crown_is_solid():
    # The crown is core massing (city/facade.wgsl city_crown): flown into level
    # at its mid-height from over the roof, it is met at its west wall grown
    # by the tower's relief and the camera's radius — and a metre over its
    # top, nothing is.
    out = json.loads(subprocess.run(
        ["node", "--input-type=module", "--eval", CROWN_JS],
        capture_output=True, text=True, check=True).stdout)
    k = out["crown"]
    assert k["z0"] == out["height"]
    into = out["into"]
    assert into is not None and into["source"] == "building", out
    assert into["normal"] == [-1.0, 0.0, 0.0], out
    assert abs(into["x"] - (k["lo"][0] - out["shell"] - 0.35)) < 1e-3, out
    assert out["over"] is None, out


def test_a_correction_across_the_cloud_domain_edge_keeps_the_flyer_on():
    # A 200 m cloud domain, so the camera's world x folds at +-100 while the
    # city frame runs straight on. The flight crosses the fold and is thrown
    # back across it by the wall: the flyer must stay on its 2.7 m tether
    # (it once kept the unfolded correction and landed a domain away).
    # The first frame's move carries the camera from 99.9 to 100.23
    # (folded to -99.77); the wall at 102.8 throws it back by 0.24 m, to
    # 99.99, across the fold again.
    r = fly(boxes=[[[102.8, -1000.0, 0.0], [130.0, 1000.0, 500.0]]],
            start=[99.9, 0.0, 60.0], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=2.0,
            domain=[[-100.0, -100.0, 0.0], [100.0, 100.0, 6000.0]])
    frames = r["frames"]
    assert len(r["knocks"]) == 1
    assert max(f["tether"] for f in frames) < 3.0
    assert not any(f["camInside"] or f["flyInside"] for f in frames)


def test_a_recorded_crash_replays_with_its_tumble():
    r = fly(boxes=WALL, start=[0.0, 0.0, 60.0], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=5.0, videoFps=30)
    frames, rec = r["frames"], r["recorded"]
    first, last = _episode(frames)
    # The knock is on the track at the time of the first sample showing it.
    assert len(rec["knocks"]) == 1
    tk, severity, side, duration, landing = rec["knocks"][0]
    assert tk == pytest.approx(rec["samples"][first][0], abs=1e-12)
    assert (severity, side, duration, landing) == pytest.approx(
        (r["knocks"][0]["severity"], r["knocks"][0]["side"],
         r["knocks"][0]["duration"], 0.0))
    # The shake is in the samples themselves: the recorded look moves.
    az = [s[4] for s in rec["samples"][first:last + 1]]
    assert max(abs(a - EAST) for a in az) > 0.3
    # The video pass hands the flyer that knock exactly once, on the first
    # frame at or after it, aged by how far past it that frame falls.
    handed = [(i, k) for i, ks in enumerate(r["replayKnocks"]) for k in ks]
    assert len(handed) == 1
    i, k = handed[0]
    assert 0.0 <= k["age"] < 1.0 / 30 + 1e-9
    assert r["videoT"][i] - k["age"] == pytest.approx(tk, abs=1e-9)
    assert k["landing"] is False and k["duration"] == duration


def test_python_reads_a_track_that_carries_knocks(tmp_path):
    from cloudyview.track import load_track, resample_track
    r = fly(boxes=WALL, start=[0.0, 0.0, 60.0], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=4.0)
    payload = {"schema": "cloudyview.track.v3", "header": {},
               "samples": r["recorded"]["samples"],
               "knocks": r["recorded"]["knocks"]}
    path = tmp_path / "crash.json"
    path.write_text(json.dumps(payload))
    _, samples, _ = load_track(path)
    # v3: the surface frame, then the scene clock, which ran with the flight.
    assert samples.shape[1] == 10
    assert samples[:, 9] == pytest.approx(samples[:, 0])
    assert len(resample_track(samples, 30.0, periodic=True,
                              linear_position=True)) > 100


def test_starting_with_the_flyer_in_a_wall_backs_off_before_the_first_frame():
    # Parked 2 m from a wall, looking at it: the flyer (2.6 m ahead) is
    # placed inside it. The very first frame must already show it outside.
    r = fly(boxes=WALL, start=[48.0, 0.0, 60.0], azimuth=EAST,
            elevation=0.0, speed=20.0, keys=[], fps=60, seconds=0.5)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert frames[0]["fly"][0] == pytest.approx(50.0 - 0.21, abs=1e-6)
    assert r["knocks"] == []   # a placement, not a crash


def test_starting_inside_a_building_puts_you_on_its_roof():
    r = fly(boxes=WALL, start=[60.0, 0.0, 60.0], azimuth=EAST,
            elevation=0.0, speed=20.0, keys=[], fps=60, seconds=0.5)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    # Lifted until the eye clears the roof, then until the flyer — hanging
    # 0.62 m below it — does: the flyer rests on the roof.
    assert frames[0]["fly"][2] == pytest.approx(500.0 + 0.21, abs=1e-6)
    assert frames[0]["cam"][2] == pytest.approx(500.0 + 0.21 + 0.62,
                                                abs=1e-6)


@pytest.mark.parametrize("start,roof", [
    # A 266 m tower with no crown and no mast (cell (241, 680)): its top_z
    # carries a crown's 10.8 m of headroom all the same, and the camera
    # must land on the roof, not on that bound.
    ([21735.96, 61254.10, 256.4], 266.4),
    # A 112 m tower with a 7.2 m crown (cell (221, 680)), started under the
    # crown: lifted onto the roof, which is inside the crown, then onto the
    # crown's top.
    ([19931.26, 61266.38, 101.6], 118.8)])
def test_starting_inside_a_tower_puts_you_on_what_it_carries(start, roof):
    r = fly(city="props", start=start, azimuth=EAST, elevation=0.0,
            speed=20.0, keys=[], fps=60, seconds=0.5)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert frames[0]["fly"][2] == pytest.approx(roof + 0.21, abs=1e-3)


def test_looking_round_beside_a_wall_is_not_a_crash():
    # Hovering 2 m off a wall, facing along it, then swinging round to face
    # it at 360 deg/s: the flyer sweeps into the wall at ~15 m/s. It is
    # stopped there (and the camera backed off), but nothing flew into
    # anything, so there is no knock.
    r = fly(boxes=WALL, start=[48.0, 0.0, 60.0], azimuth=0.0,
            elevation=0.0, speed=20.0, keys=[], fps=60, seconds=0.25,
            turn=360.0)
    frames = r["frames"]
    assert not any(f["camInside"] or f["flyInside"] for f in frames)
    assert max(f["fly"][0] for f in frames) <= 50.0 - 0.2 + 1e-9
    assert r["knocks"] == []


@pytest.mark.parametrize("speed,fps,release", [
    (20.0, 60, True), (5000.0, 60, True),
    # Held into the wall at the fastest the wheel allows, on a machine
    # drawing 10 frames a second: a spline through these samples replayed
    # the camera 34 m through the wall.
    (5000.0, 10, False)])
def test_a_replayed_crash_stays_out_of_the_wall(speed, fps, release):
    # The video pass interpolates the recorded poses, over the city in
    # straight lines between them (track.js linear()) — each of those
    # lines is a move the collider found clear.
    wall = [[[3000.0, -1000.0, 0.0], [3030.0, 1000.0, 500.0]]]
    start = [2900.0, 0.0, 60.0] if speed < 100 else [0.0, 0.0, 60.0]
    r = fly(boxes=wall, start=start, azimuth=EAST, elevation=0.0,
            speed=speed, keys=["w"], fps=fps, seconds=7.0 if speed < 100
            else 4.0, videoFps=30, releaseOnKnock=release)
    assert len(r["knocks"]) >= 1
    live = max(f["cam"][0] for f in r["frames"])
    replay = max(c[0] for c in r["videoCam"])
    assert replay <= live + 1e-6
    # And the replayed flyer, placed 2.6 m ahead of that camera, is short
    # of the face too.
    assert replay + 2.6 <= 3000.0 - 0.2 + 1e-6


def test_switching_flyers_does_not_sweep_from_one_to_the_other():
    # A swift 4.6 m out to the east, then a dart 2.6 m out to the north
    # (standing in for a smoothed heading that lags the view), with a mast
    # between where the one was and where the other is, but on neither
    # tether. The camera is parked: nothing moved, nothing may be hit.
    mast = [[[2.4, 0.3, 0.0], [3.8, 1.7, 400.0]]]
    r = fly(boxes=mast, start=[0.0, 0.0, 200.0], azimuth=EAST,
            elevation=0.0, speed=20.0, keys=[], fps=60, seconds=0.5,
            flyers=[[0, 4.6, 0.0], [10, 2.6, -90.0]])
    frames = r["frames"]
    assert all(abs(f["cam"][0]) < 1e-9 and abs(f["cam"][1]) < 1e-9
               for f in frames)
    assert r["knocks"] == []
    assert r["calms"][1] >= 1, "the new flyer was not told to start afresh"


def test_the_first_surface_met_decides_the_crash():
    # Flying east and down: the dart, ahead, meets a wall a sixth of the
    # way through the frame; the eye would come down on a roof below it
    # only at four fifths. That is a wall crash, not a landing.
    boxes = [[[2.95, -50.0, 0.0], [40.0, 50.0, 300.0]],     # the wall
             [[-50.0, -50.0, 0.0], [1.0, 50.0, 9.75]]]      # the roof
    r = fly(boxes=boxes, start=[0.0, 0.0, 10.5], azimuth=EAST,
            elevation=0.0, speed=60.0, keys=["w", "shift"], fps=60,
            seconds=0.2)
    assert len(r["knocks"]) == 1
    assert r["knocks"][0]["landing"] is False


def test_a_map_click_ends_the_tumble():
    # Crash, then a map click a few frames into the episode: the episode
    # is over, and the flyer is told so.
    r = fly(boxes=WALL, start=[0.0, 0.0, 60.0], azimuth=EAST, elevation=0.0,
            speed=20.0, keys=["w"], fps=60, seconds=3.0,
            teleport=[150, -500.0, 0.0])
    frames = r["frames"]
    first, _ = _episode(frames)
    assert first < 150
    assert not frames[150]["active"]
    assert frames[150]["control"] == 1.0
    assert frames[150]["az"] == pytest.approx(EAST, abs=1e-9)
    assert r["calms"][0] >= 2   # the first frame's seed, and the click's


FLYER_JS = """
import { Flyer } from "%(soar)s/flyer.js";
const f = Object.create(Flyer.prototype);
f.position = [1, 2, 3]; f._prevOrigin = [0, 0, 0]; f._knock = null;
f.knock({ severity: 1, side: 1, duration: 1, landing: false });
const knocked = f._knock !== null;
// A correction that crossed a cloud fold moves both what is drawn and the
// origin the next velocity is measured from.
f.shift([-200, 0, 0.5]);
const after = { position: f.position, prev: f._prevOrigin };
f.calm();
process.stdout.write(JSON.stringify({ knocked, after,
  calmed: f._knock === null && f._prevOrigin === null }));
""" % {"soar": SOAR.as_posix()}


def test_flyer_shift_moves_its_velocity_origin_and_calm_forgets():
    out = subprocess.run(["node", "--input-type=module", "--eval", FLYER_JS],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["knocked"]
    assert r["after"]["position"] == [-199, 2, 3.5]
    assert r["after"]["prev"] == [-200, 0, 0.5]
    assert r["calmed"]


SWEEP_VS_POINTS_JS = """
import { readFileSync } from "node:fs";
import { CityTile, CityLots, cellSolids, insideSolid, cityCell }
  from "%(soar)s/citylots.js";
const meta = JSON.parse(readFileSync("%(tile)s/meta.json", "utf8"));
const b = readFileSync("%(tile)s/fif_mip0.bin");
const city = new CityLots(CityTile.fromHalfBytes(
  new Uint8Array(b.buffer, b.byteOffset, b.byteLength), meta.n, meta.cell_m));
// Solid at p for radius r, by brute force over the 5 x 5 blocks about it —
// wider than any growth reaches, and sharing nothing with the walk.
const solidAt = (p, r) => {
  if (p[2] < r) return true;
  const cx = Math.floor(p[0] / meta.cell_m), cy = Math.floor(p[1] / meta.cell_m);
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const cc = city.cell(cx + dx, cy + dy);
      if (cc.built && cellSolids(cc, r).some((s) => insideSolid(p, s))) {
        return true;
      }
    }
  }
  return false;
};
let seed = 20260923;
const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0)
                   / 2 ** 32);
const out = { segments: 0, hits: 0, disagree: [], edgeBuds: 0 };
const check = (p0, p1, r) => {
  const d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const len = Math.hypot(...d);
  out.segments++;
  const hit = city.sweep(p0, p1, r);
  // The reference: the first of points 2 cm apart that are solid.
  const steps = Math.ceil(len / 0.02);
  let first = null;
  for (let i = 1; i <= steps && first === null; i++) {
    const t = i / steps;
    if (solidAt([p0[0] + t * d[0], p0[1] + t * d[1], p0[2] + t * d[2]],
                r)) first = t * len;
  }
  if (hit) out.hits++;
  const agree = hit === null ? first === null
    : first !== null && Math.abs(hit.t * len - first) < 0.03;
  if (!agree) out.disagree.push({ p0, p1, r, sweep: hit && hit.t * len,
                                  points: first });
  // inside() on the segment's end is the same question asked the short way.
  const end = city.inside(p1, r) !== null;
  if (end !== solidAt(p1, r)) out.disagree.push({ p1, r, inside: end });
};

// The case the neighbour test exists for: a bud clamped half a metre
// inside its lot, whose shell reaches into the next block's column. A
// segment down that next column, a metre and a bit off the bud's face,
// meets the shell without its centre ever entering the bud's block.
for (let cy = 640; cy < 730 && out.edgeBuds < 40; cy++) {
  for (let cx = 150; cx < 240 && out.edgeBuds < 40; cx++) {
    const c = cityCell(city.tile, cx, cy);
    if (!(c.built && c.arch === 2)) continue;
    const z = 0.5 * (c.b2min[2] + c.b2max[2]);
    const my = 0.5 * (c.b2min[1] + c.b2max[1]);
    if (Math.abs(c.b2max[0] - (c.lot_max[0] - 0.5)) < 1e-6) {
      const x = c.lot_max[0] + 0.7;      // 1.2 m off the face, next block
      check([x, my - 40, z], [x, my + 40, z], 0.2);
      out.edgeBuds++;
    } else if (Math.abs(c.b2min[0] - (c.lot_min[0] + 0.5)) < 1e-6) {
      const x = c.lot_min[0] - 0.7;
      check([x, my - 40, z], [x, my + 40, z], 0.2);
      out.edgeBuds++;
    }
  }
}

while (out.segments < 800 + out.edgeBuds) {
  const r = rnd() < 0.5 ? 0.2 : 0.35;
  // Downtown, around where the cyberpunk session opens, low enough to be
  // among the buildings.
  const p0 = [17393 + (rnd() - 0.5) * 4000, 61646 + (rnd() - 0.5) * 4000,
              rnd() * 150];
  if (solidAt(p0, r)) continue;
  const len = Math.exp(rnd() * Math.log(250));
  const th = rnd() * 2 * Math.PI, ph = (rnd() - 0.5) * 0.8;
  const d = [Math.cos(th) * Math.cos(ph) * len,
             Math.sin(th) * Math.cos(ph) * len, Math.sin(ph) * len];
  check(p0, [p0[0] + d[0], p0[1] + d[1], p0[2] + d[2]], r);
}
process.stdout.write(JSON.stringify(out));
""" % {"soar": SOAR.as_posix(), "tile": TILE_DIR.as_posix()}


def test_the_sweep_meets_what_the_points_say_is_solid():
    # The walk (a DDA over blocks, each testing its eight neighbours for
    # the facade shell's reach, stopping at the first hit) and inside()
    # against the plainest possible reference — points 2 cm apart, each
    # tested against every solid of the 5 x 5 blocks about it — on the
    # shipped tile's downtown.
    out = subprocess.run(["node", "--input-type=module", "--eval",
                          SWEEP_VS_POINTS_JS], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["hits"] > 100, "the sample barely touched a building"
    assert r["edgeBuds"] >= 20, "too few lot-edge buds to test the shell reach"
    assert r["disagree"] == [], r["disagree"][:3]
