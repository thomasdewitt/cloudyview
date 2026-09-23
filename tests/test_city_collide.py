"""Flying into the night city: stopped at the wall, knocked, and let go.

web/soar/collide.js is the flight collider (Thomas, 2026-09-23: "the
buildings currently just disappear when you fly into them, but probably the
paper airplane should be rocked, like you loose control for a sec and it
falls a bit"). These tests fly the real FlightCamera through the real
collider, frame by frame in node, in the order viewer.js runs them, with a
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
  - into a real building of the shipped tile, through citylots.js.

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

# The flight loop, as viewer.js _frame runs it, with a stand-in flyer.
FLY_JS = """
import { FlightCamera, cameraBasis, worldToRelative } from "%(soar)s/camera.js";
import { CityCollider } from "%(soar)s/collide.js";
import { TrackRecorder, resampleTrack, knocksDue } from "%(soar)s/track.js";
import { CityTile, CityLots, cityCell } from "%(soar)s/citylots.js";
import { readFileSync } from "node:fs";
%(box_city)s

const EXTENT = 92160.0;

// flyer.js _place, without the smoothing: ahead by `distance` along the
// look, below by `drop`.
class StandInFlyer {
  constructor() { this.position = [0, 0, 0]; this.knocks = []; }
  update(dt, camera) {
    const [f, , u] = cameraBasis(camera.azimuth, camera.elevation);
    const o = camera.position;
    this.position = [0, 1, 2].map((i) => o[i] + 2.6 * f[i] - 0.62 * u[i]);
  }
  shift(d) { this.position = this.position.map((v, i) => v + d[i]); }
  knock(args) { this.knocks.push(args); }
}

function fly({ city, start, azimuth, elevation, speed, keys, fps, seconds,
               releaseOnKnock = true,
               // The CLOUD domain, which the camera's world position folds
               // at — independent of the city tile's frame.
               domain = [[-200000, -200000, 0], [200000, 200000, 6000]],
               videoFps = null,
               // Mouse look between frames, deg/s of azimuth.
               turn = 0.0 }) {
  const [BMIN, BMAX] = domain;
  const recorder = new TrackRecorder();
  recorder.start();
  const camera = new FlightCamera(BMIN, BMAX, {
    start: { position: worldToRelative(start, BMIN, BMAX),
             azimuth, elevation, fov: 90 },
    surfaceTileExtent: EXTENT, surfaceOffsetM: [0, 0], speed,
    floorZ: null });
  const collider = new CityCollider(city, EXTENT);
  const flyer = new StandInFlyer();
  for (const k of keys) camera.keys.add(k);
  const dt = 1 / fps;
  const frames = [];
  for (let i = 0; i < Math.round(seconds * fps); i++) {
    camera.azimuth = ((camera.azimuth + turn * dt) %% 360 + 360) %% 360;
    collider.advance(dt, camera);
    camera.move(dt);
    camera.constrain();
    flyer.update(dt, camera);
    const knock = collider.resolve(dt, camera, flyer);
    if (knock) recorder.noteKnock(knock);
    recorder.advance(dt);
    recorder.sample(camera);
    if (releaseOnKnock && flyer.knocks.length) camera.keys.clear();
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
  const out = { frames, knocks: flyer.knocks,
                recorded: { samples: recorder.samples,
                            knocks: recorder.knocks } };
  if (videoFps !== null) {
    // What the video pass hands the flyer, frame by frame.
    const video = resampleTrack(recorder.samples, videoFps);
    out.replayKnocks = video.map((f, i) => knocksDue(
      recorder.knocks, i > 0 ? video[i - 1].t : -Infinity, f.t));
    out.videoT = video.map((f) => f.t);
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
  // at a third of its height from 40 m off its west face.
  let pick = null;
  for (let cy = spec.near[1] - 20; cy < spec.near[1] + 20; cy++) {
    for (let cx = spec.near[0] - 20; cx < spec.near[0] + 20; cx++) {
      const c = cityCell(tile, cx, cy);
      if (c.built && c.tiers === 1 && !c.has_frustum && c.height > 60
          && (pick === null || c.height > pick.height)) pick = c;
    }
  }
  if (pick === null) throw new Error("no slab near the requested cell");
  const y = 0.5 * (pick.b1min[1] + pick.b1max[1]);
  spec.start = [pick.b1min[0] - 40, y, pick.height / 3];
  spec.wallX = pick.b1min[0];
} else {
  city = new BoxCity(spec.boxes);
}
process.stdout.write(JSON.stringify({ ...fly({ ...spec, city }), wallX: spec.wallX }));
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
    # A bounce back off the wall, a metre or two.
    back = hit["cam"][0] - min(f["cam"][0] for f in frames[first:last + 1])
    assert 1.0 < back < 3.5
    # A drop of five to fifteen metres (13 m at this severity).
    drop = hit["cam"][2] - frames[last]["cam"][2]
    assert 5.0 <= drop <= 15.0
    assert drop == pytest.approx(5.0 + 10.0 * knock["severity"], abs=0.05)
    # The controls: nearly dead early, fully back after.
    early = frames[first + 6]
    assert early["control"] < 0.15
    assert frames[last + 1]["control"] == 1.0
    # The view was shaken — and is exactly where it was afterwards.
    shaken = max(abs(f["az"] - EAST) + abs(f["el"])
                 for f in frames[first:last + 1])
    assert 0.3 < shaken < 5.0
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
    payload = {"schema": "cloudyview.track.v2", "header": {},
               "samples": r["recorded"]["samples"],
               "knocks": r["recorded"]["knocks"]}
    path = tmp_path / "crash.json"
    path.write_text(json.dumps(payload))
    _, samples = load_track(path)
    assert samples.shape[1] == 9
    assert len(resample_track(samples, 30.0, periodic=True)) > 100


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
