"""A recorded city flight replays over the district — and buildings — it flew.

The scene has two independent periodic frames: the clouds wrap at the domain
extent, the surface tile (the night city, and the day ocean's wave patch) at
its own. The persistent bug class (the spawn drift of 2026-08-22, the
recording drift of 2026-09-01) is folding one world position at one of the
periods and letting that fold cross a boundary the other frame can see —
replay converts linearly, so the clouds come back identical and the surface
shifts by a fraction of a tile per crossing.

The design under test: the camera tracks each frame in its own coordinates,
wrapped at its own period — `position` folds at the CLOUD period,
`surfacePosition` at the tile, both advanced by the same world-space delta
on every move. Track schema v2 carries both (x/y cloud-folded exactly as
v1, sx/sy tile-relative at period 1.0), and resampleTrack unwraps each
column at its own frame's period. Uniform row 24 is the per-frame surface
offset, folded into [0, tile): cam.xy - surfacePosition, or the static
offset for a static render. The shader derives the water uv and the WHOLE
city frame from it, and the city is a pure function of that frame — every
cell index folds at the tile before it seeds anything — so the fold's
whole-tile ambiguity is invisible and a replayed track (which serializes
only the tile phase) renders the very buildings that were flown past.

Pinned here under node, with the real modules:

 (a) a camera moved two cloud periods east has position folded into the box
     and surfacePosition advanced by the same delta mod the tile;
 (b) a day scene records v2 samples whose first seven columns are
     byte-identical to v1, plus the ocean's sx/sy; and the ocean frame is
     continuous across a cloud crossing;
 (c) record -> resample -> replay of a city crossing preserves the district
     while the cloud position folds exactly as today;
 (d) a v1 7-column track still resamples, with null surface frames;
 (e) resampled sx wraps the short way across the tile boundary and stays in
     [0, 1);
 (f) spawn identities: the cyberpunk surface frame is the pinned city
     position, and row 8 carries the static offset whatever the camera;
 (g) row 24 semantics: day zero-drift writes a bit-exact 0.0 (the golden
     argument), the static city offset folds into the tile, and a crossing
     shifts live camera and replay pose alike;
plus two shader pins: the tile-frame derivation sites as text, and the
whole-tile invariance of the seed inputs in numbers.

Skips only without node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRACK_JS = REPO / "web" / "soar" / "track.js"
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not TRACK_JS.exists(),
    reason="needs node and web/soar/track.js")


_JS = textwrap.dedent("""
    import { TrackRecorder, resampleTrack, TRACK_SCHEMA } from "%s";
    import { FlightCamera, cameraWorldOrigin, worldToRelative } from "%s";
    import { packUniforms } from "%s";

    // A 10 km cloud box over a 2.3 km city tile: incommensurate on purpose,
    // so one cloud crossing shifts the tile phase by 800 m if anything folds
    // wrongly.
    const bmin = [-5000, -5000, 0], bmax = [5000, 5000, 4000];
    const PERIOD = 10000, TILE = 2300;
    const OFFSET = [75330, -17010];

    const cityCamera = (startRel = null) => new FlightCamera(bmin, bmax, {
      periodic: true, start: startRel,
      surfaceTileExtent: TILE, surfaceOffsetM: OFFSET,
    });

    // Fly d metres east through the real move path (azimuth 90 = east).
    const flyEast = (cam, metres, step = 100) => {
      cam.azimuth = 90; cam.elevation = 0;
      cam.speed = step / 0.1;
      cam.keys = new Set(["w"]);
      for (let i = 0; i < metres / step; i++) cam.move(0.1);
      cam.keys = new Set();
    };

    const fold = (v, p) => ((v %% p) + p) %% p;
    const out = { schema: TRACK_SCHEMA };

    // (a) Two cloud periods east: same cloud, same district.
    {
      const cam = cityCamera();
      cam.teleport(1000, 0);
      const sp0 = [...cam.surfacePosition];
      flyEast(cam, 2 * PERIOD);
      out.moved = {
        x: cam.position[0],
        sp: cam.surfacePosition[0],
        spExpected: fold(sp0[0] + 2 * PERIOD, TILE),
        spStart: sp0[0],
      };
    }

    // A day camera as the viewer now builds one: the ocean's tile frame,
    // static offset at the origin.
    const dayCamera = () => new FlightCamera(bmin, bmax, {
      periodic: true, surfaceTileExtent: TILE, surfaceOffsetM: [0.0, 0.0],
    });

    // (b) A day scene records v2 samples whose first seven columns are
    // byte-identical to a v1 sample; sx/sy (the ocean's tile frame) ride
    // behind them.
    {
      const day = dayCamera();
      day.teleport(1000, -2000);
      const rec = new TrackRecorder();
      rec.start();
      rec.advance(0.0); rec.sample(day);
      rec.advance(1.0); rec.sample(day);
      const rel = day.relativePosition();
      const v1 = [1.0, rel[0], rel[1], rel[2],
                  day.azimuth, day.elevation, day.fov];
      out.day = { samples: rec.stop(), v1,
                  surface: [...day.surfacePosition],
                  spExpected: [fold(1000, TILE) / TILE,
                               fold(-2000, TILE) / TILE] };
    }

    // (b2) The ocean-never-jumps invariant, live: flying a day camera
    // across the cloud boundary folds position and leaves surfacePosition
    // continuous — it advances by exactly the metres flown, mod the tile.
    {
      const day = dayCamera();
      day.teleport(4900, 0);
      const sp0 = day.surfacePosition[0];
      const x0 = day.position[0];
      flyEast(day, 400);                    // crosses the box edge at 5000
      out.dayCrossing = {
        x: day.position[0], x0,
        sp: day.surfacePosition[0],
        spExpected: fold(sp0 + 400, TILE),
      };
    }

    // (c) Record a city crossing with the real camera and recorder, then
    // resample: the cloud x folds exactly as a day flight would, while the
    // tile phase survives the crossing.
    {
      const cam = cityCamera();
      cam.teleport(4900, 0);
      const spStart = cam.surfacePosition[0];
      const rec = new TrackRecorder();
      rec.start();
      rec.sample(cam);
      for (let i = 0; i < 4; i++) {         // 400 m east, crossing at 5000
        flyEast(cam, 100);
        rec.advance(0.1);
        rec.sample(cam);
      }
      const samples = rec.stop();
      const frames = resampleTrack(samples, 10.0, { periodic: true });
      const last = frames[frames.length - 1];
      // Replay's surface offset for the last frame (viewer.renderTrackVideo):
      // world xy from the cloud-folded position, minus sp * TILE.
      const world = cameraWorldOrigin(last.position, bmin, bmax);
      out.crossing = {
        sampleWidth: samples[0].length,
        xs: frames.map((f) => f.position[0]),
        sps: frames.map((f) => f.surfacePosition[0]),
        spStartRel: spStart / TILE,
        // The district a static-offset fold of the folded position claims —
        // the pre-fix reconstruction, wrong by one crossed period.
        foldDerivedRel: fold(world[0] - OFFSET[0], TILE) / TILE,
        replayOffset: world[0] - last.surfacePosition[0] * TILE,
      };
    }

    // (d) A v1 7-column track still resamples, surface-frame-less, and the
    // day wrap still goes the short way and folds into [-1, 1).
    {
      const v1 = [
        [0.0, 0.8, 0.0, 0.1, 0, 0, 70],
        [1.0, 0.9, 0.0, 0.1, 0, 0, 70],
        [2.0, -0.9, 0.0, 0.1, 0, 0, 70],
        [3.0, -0.8, 0.0, 0.1, 0, 0, 70],
      ];
      const frames = resampleTrack(v1, 1.0, { periodic: true });
      out.v1 = { xs: frames.map((f) => f.position[0]),
                 surfaces: frames.map((f) => f.surfacePosition) };
    }

    // (e) sx wraps the short way across the tile boundary.
    {
      const v2 = [
        [0.0, 0.0, 0.0, 0.1, 0, 0, 70, 0.94, 0.5],
        [1.0, 0.0, 0.0, 0.1, 0, 0, 70, 0.98, 0.5],
        [2.0, 0.0, 0.0, 0.1, 0, 0, 70, 0.02, 0.5],
        [3.0, 0.0, 0.0, 0.1, 0, 0, 70, 0.06, 0.5],
      ];
      const frames = resampleTrack(v2, 2.0, { periodic: true });
      out.sxWrap = frames.map((f) => f.surfacePosition[0]);
    }

    // (f) The uniforms-offset identity at spawn.
    //
    // Two spawns. One whose world position lies INSIDE the cloud box, where
    // the identity is exact: cam.xy - surfacePosition = OFFSET + k * TILE.
    // And the cyberpunk one, whose pinned world position (OFFSET + cityM,
    // scene.cityStartCamera) folds into the box at construction — there the
    // offset additionally carries the fold shift, which moves every ray
    // origin equally, so what pins the district is surfacePosition = cityM
    // itself (the uv under the camera), not the raw offset.
    {
      const inside = cityCamera();
      inside.teleport(1000, -2000);
      const off = [inside.position[0] - inside.surfacePosition[0],
                   inside.position[1] - inside.surfacePosition[1]];

      const cityM = [610, 1495];
      const world = [OFFSET[0] + cityM[0], OFFSET[1] + cityM[1], 300];
      const spawn = cityCamera({
        position: worldToRelative(world, bmin, bmax),
        azimuth: 0, elevation: 0, fov: 70,
      });

      const state = {
        bmin, bmax, dtView: 40, dtLight: 40, periodic: true,
        oceanZ: 0.0, oceanReflectance: [0.002, 0.0045, 0.0126],
        oceanFifDx: 90.0, oceanTileExtent: TILE, oceanEnabled: true,
        oceanMaxLod: 8, city: true, cityOffsetM: OFFSET,
      };
      const view = {
        outputSize: [64, 36], renderSize: [64, 36],
        sunAzimuth: 20, sunElevation: 55,
      };
      const withFrame = packUniforms(state, { ...view, camera: inside });
      const noFrame = packUniforms(state, { ...view, camera: {
        position: [...inside.position], azimuth: 0, elevation: 0, fov: 70,
      } });
      out.identity = {
        spawnSp: [...spawn.surfacePosition], cityM,
        offModTile: [fold(off[0] - OFFSET[0], TILE),
                     fold(off[1] - OFFSET[1], TILE)],
        row8WithFrame: [withFrame[33], withFrame[34]],
        row8NoFrame: [noFrame[33], noFrame[34]],
        row8Static: [Math.fround(OFFSET[0]), Math.fround(OFFSET[1])],
        tile: TILE,
      };
    }

    // (g) Row 24 is the folded per-frame surface offset. Day zero-drift is
    // exactly 0.0 (fold(origin) is the very expression sp came from);
    // frame-less day writes fold(0) = 0.0; city static writes the static
    // offset folded; a crossing moves the live camera's (and a replay
    // pose's) offset by the crossed period, mod the tile.
    {
      const state = (city) => ({
        bmin, bmax, dtView: 40, dtLight: 40, periodic: true,
        oceanZ: 0.0, oceanReflectance: [0.002, 0.0045, 0.0126],
        oceanFifDx: 90.0, oceanTileExtent: TILE, oceanEnabled: true,
        oceanMaxLod: 8, city, cityOffsetM: OFFSET,
      });
      const view = (camera) => ({
        camera, outputSize: [64, 36], renderSize: [64, 36],
        sunAzimuth: 20, sunElevation: 55,
      });
      const row24 = (u) => [u[96], u[97], u[98], u[99]];

      const parked = dayCamera();
      parked.teleport(1000, -2000);
      const zeroDrift = row24(packUniforms(state(false), view(parked)));

      const flown = dayCamera();
      flown.teleport(4900, 0);
      flyEast(flown, 400);                  // one eastward cloud crossing
      const drifted = row24(packUniforms(state(false), view(flown)));

      const frameless = row24(packUniforms(state(false), view({
        position: [1000, -2000, 500], azimuth: 0, elevation: 0, fov: 70,
      })));
      const cityStatic = row24(packUniforms(state(true), view({
        position: [1000, -2000, 500], azimuth: 0, elevation: 0, fov: 70,
      })));

      // A replay pose carries the tile phase; its offset is the same
      // fold-first subtraction. Zero-drift city pose reproduces the static
      // offset mod the tile; a crossing shifts it by the crossed period.
      const posePosition = [1000, -2000, 500];
      const poseSp = [fold(1000 - OFFSET[0], TILE),
                      fold(-2000 - OFFSET[1], TILE)];
      const poseZero = row24(packUniforms(state(true), view({
        position: posePosition, azimuth: 0, elevation: 0, fov: 70,
        surfacePosition: poseSp,
      })));
      const poseDrifted = row24(packUniforms(state(true), view({
        position: posePosition, azimuth: 0, elevation: 0, fov: 70,
        surfacePosition: [fold(poseSp[0] + PERIOD, TILE), poseSp[1]],
      })));
      out.row24 = {
        zeroDrift, drifted, frameless, cityStatic, poseZero, poseDrifted,
        cityStaticExpected: [Math.fround(fold(OFFSET[0], TILE)),
                             Math.fround(fold(OFFSET[1], TILE))],
        driftedExpected: Math.fround(fold(-PERIOD, TILE)),
      };
    }

    process.stdout.write(JSON.stringify(out));
""") % (TRACK_JS.as_posix(),
        (REPO / "web" / "soar" / "camera.js").as_posix(),
        (REPO / "web" / "soar" / "uniforms.js").as_posix())


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_schema_is_v2(result):
    assert result["schema"] == "cloudyview.track.v2"


def test_two_cloud_periods_east_keeps_the_district(result):
    moved = result["moved"]
    # The cloud frame folds back to where it started; the tile frame advanced
    # by the same 20 km, mod its own period.
    assert moved["x"] == pytest.approx(1000.0)
    assert moved["sp"] == pytest.approx(moved["spExpected"])
    # And the district genuinely moved (20 km is not a whole number of tiles).
    assert moved["sp"] != pytest.approx(moved["spStart"])


def test_a_day_recording_keeps_v1_columns_and_gains_the_ocean_frame(result):
    day = result["day"]
    # The day camera carries the ocean's tile frame now.
    assert day["surface"] == pytest.approx(
        [s * 2300.0 for s in day["spExpected"]])
    assert [len(s) for s in day["samples"]] == [9, 9]
    # The first seven columns are byte-identical to a v1 sample; sx/sy ride
    # behind them, tile-relative.
    assert day["samples"][1][:7] == day["v1"]
    assert day["samples"][1][7:] == pytest.approx(day["spExpected"])


def test_the_day_ocean_never_jumps_across_a_cloud_crossing(result):
    c = result["dayCrossing"]
    # The cloud frame folded (400 m east of 4900 lands at -4700)...
    assert c["x"] == pytest.approx(-4700.0)
    # ...while the ocean frame advanced by exactly the metres flown.
    assert c["sp"] == pytest.approx(c["spExpected"])


def test_a_resampled_city_crossing_preserves_the_district(result):
    c = result["crossing"]
    assert c["sampleWidth"] == 9
    # The cloud x folds exactly as today: 0.98, then the crossing lands at
    # the opposite face and walks on.
    assert c["xs"][0] == pytest.approx(0.98)
    assert c["xs"][1] == pytest.approx(-1.0)
    assert c["xs"][-1] == pytest.approx(-0.94)
    assert all(-1.0 <= x < 1.0 for x in c["xs"])
    # The tile phase is continuous through the crossing: 100 m per frame, no
    # tile-fraction jump anywhere.
    tile = 2300.0
    deltas = [(b - a) % 1.0 for a, b in zip(c["sps"], c["sps"][1:])]
    assert all(min(d, 1.0 - d) * tile == pytest.approx(100.0, abs=1e-3)
               for d in deltas)
    # The last frame is 400 m east of the start, in the SAME tile frame.
    assert c["sps"][-1] == pytest.approx(
        (c["spStartRel"] + 400.0 / tile) % 1.0)
    # Whereas re-deriving the district from the folded position (the pre-fix
    # reconstruction) is off by exactly the crossed cloud period:
    # 10000 mod 2300 = 800 m of tile phase.
    drift = (c["sps"][-1] - c["foldDerivedRel"]) % 1.0
    assert drift * tile == pytest.approx(800.0, abs=1e-3)


def test_a_v1_track_still_resamples_with_null_surface_frames(result):
    v1 = result["v1"]
    assert v1["surfaces"] == [None] * len(v1["surfaces"])
    xs = v1["xs"]
    assert xs[0] == pytest.approx(0.8)
    assert xs[-1] == pytest.approx(-0.8)
    # The 0.9 -> -0.9 jump goes through the boundary (0.2 of travel), not
    # back across the whole box, and every output lies in [-1, 1).
    assert xs[2] == pytest.approx(-0.9)
    assert all(-1.0 <= x < 1.0 for x in xs)


def test_resampled_sx_wraps_the_short_way(result):
    sx = result["sxWrap"]
    assert all(0.0 <= v < 1.0 for v in sx)
    assert sx[0] == pytest.approx(0.94)
    assert sx[2] == pytest.approx(0.98)
    assert sx[4] == pytest.approx(0.02)
    assert sx[-1] == pytest.approx(0.06)
    # Every step is a short forward one — the boundary crossing is no
    # different from any other step. (Catmull-Rom's clamped ends ease the
    # first and last half-steps, so pin the direction and the total, not an
    # exact per-step size.)
    deltas = [((b - a + 0.5) % 1.0) - 0.5 for a, b in zip(sx, sx[1:])]
    assert all(0.0 <= d < 0.1 for d in deltas)
    assert sum(deltas) == pytest.approx(0.12, abs=1e-9)


def test_the_uniforms_offset_identity_at_spawn(result):
    ident = result["identity"]
    # The cyberpunk spawn's surface frame IS the pinned city position, folded
    # at the tile — no un-fold dance, and the district under the camera
    # (uv = surfacePosition / tile) is the one the capture named.
    assert ident["spawnSp"][0] == pytest.approx(ident["cityM"][0], abs=1e-6)
    assert ident["spawnSp"][1] == pytest.approx(ident["cityM"][1], abs=1e-6)
    # An in-box spawn with zero accumulated drift:
    # cam.xy - surfacePosition = staticOffset + k * tileExtent exactly.
    for v in ident["offModTile"]:
        assert min(v, ident["tile"] - v) == pytest.approx(0.0, abs=1e-6)
    # Row 8's city packing is the STATIC offset, camera or no camera — the
    # live frame is row 24's job, and row 8 is the scene's own statement of
    # where the tile was pinned.
    assert ident["row8NoFrame"] == ident["row8Static"]
    assert ident["row8WithFrame"] == ident["row8Static"]


def test_row24_is_the_folded_surface_offset(result):
    r = result["row24"]
    tile = 2300.0
    # Zero drift, day: EXACTLY 0.0 — the water subtraction is then
    # bit-neutral, which is the whole golden-safety argument. Frame-less
    # day (v1 replay, Python) writes the same 0.0.
    assert r["zeroDrift"] == [0.0, 0.0, 0.0, 0.0]
    assert r["frameless"] == [0.0, 0.0, 0.0, 0.0]
    # The static city offset, folded into [0, tile).
    assert r["cityStatic"][:2] == r["cityStaticExpected"]
    assert all(0.0 <= v < tile for v in r["cityStatic"][:2])
    # A zero-drift city pose reproduces the static offset mod the tile —
    # replay equals a fresh static render, exactly.
    assert r["poseZero"][:2] == r["cityStaticExpected"]
    # One eastward crossing shifts the offset by the crossed period, mod the
    # tile — from the day camera's zero base and from the city pose's static
    # base alike, which is the live == replay property in uniform form.
    assert r["drifted"][0] == pytest.approx(r["driftedExpected"])
    assert r["drifted"][1] == 0.0
    assert r["poseDrifted"][0] == pytest.approx(
        (r["cityStaticExpected"][0] - 10000.0) % tile)
    assert r["poseDrifted"][1] == pytest.approx(r["cityStaticExpected"][1])



def test_the_shader_derives_water_and_city_from_the_tile_frame():
    """Textual pin on raymarch.wgsl: the water samples subtract the row-24
    offset, the city enters its frame exactly once (city_camera_origin),
    the glow is a plain tile read of that frame, and every cell-index draw
    folds at the tile before it seeds anything — the property that makes a
    building a function of its tile coordinate, so a replayed track (which
    serializes only the tile phase) renders the very buildings that were
    flown past. GPU renders cannot run here; this catches the edit that
    quietly reverts one site — a regenerated city block included, since the
    composer rewrites the GENERATED section wholesale."""
    src = SHADER.read_text()
    # The two water-normal reads sample the offset-corrected frame.
    assert src.count(
        "(world_xy - u.surface_offset.xy) / u.ocean_params.y") == 2
    # The city frame has ONE entry, and the trace and the dome probe both
    # use it; nothing calls city_trace with a raw world origin.
    assert "fn city_camera_origin()" in src
    assert src.count("city_camera_origin()") >= 4   # def + trace + probe + adscreens
    assert src.count("city_trace(city_o, dir)") == 1
    assert "city_trace(u.cam_origin.xyz" not in src
    # The glow is a plain tile read of the city frame — no offset algebra.
    assert "let uv = xy / u.ocean_params.y;" in src
    assert "(u.ocean.yz - u.surface_offset.xy)" not in src
    # Every cell-index seed folds at the tile first: the core's two
    # city_cell seeds and the components' draws all go through
    # city_tile_cell, and no bare bitcast of an unwrapped cell index
    # remains (the star field's cid is a view-direction cell, not a city
    # cell, and is the one legitimate exception).
    assert "fn city_tile_cell(" in src
    for needle in ["bitcast<u32>(ci.", "bitcast<u32>(anchor.",
                   "bitcast<u32>(clo."]:
        assert needle not in src, f"an unwrapped cell seed survives: {needle}"
    # The march loop's uplight converts its WORLD cloud sample through the
    # same canonical fold; the tau probe beside it stays world.
    assert "city_glow_sample(city_fold_xy(p.xy)" in src
    assert "city_uplight_probe_tau(p, probe_jitter)" in src


def test_a_whole_tile_offset_shift_names_the_same_buildings():
    """The tile-periodicity property in numbers: the seed inputs pinned
    above are mod(floor(p_city / cell), n), so shifting the city frame by
    whole tiles — the only ambiguity row 24's fold leaves — changes no
    texel and no seed. Mirrors city_tile_cell with the shipped bake's
    numbers (n = 1024 blocks of 90 m; tile = n * cell exactly)."""
    n, cell = 1024, 90.0
    tile = n * cell
    assert tile == 92160.0                      # the bake's tile_extent_m
    points = [(12.5, -33333.3), (75330.0, -17010.0), (-1.0, 92159.9)]
    for x, y in points:
        for k in [-3, 1, 7]:
            a = (int(x // cell) % n, int(y // cell) % n)
            b = (int((x + k * tile) // cell) % n,
                 int((y - k * tile) // cell) % n)
            assert a == b
