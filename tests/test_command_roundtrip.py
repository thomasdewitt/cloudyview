"""The browser's copy-command output must parse in the CLI, always.

The standing goal (Thomas, 2026-08-22): *"the 'render in terminal' option can
actually reproduce the view, always."* Two halves make that true. The loader
mirror in io.py handles files the browser resolved silently; for everything
the browser inferred or ASKED — variables, axis assignment, coordinate names,
units, timestep, an attached ice file — `viewer._selectionFlags` writes the
answers out as flags, and this test proves the CLI accepts every one of them.

The emission is driven under node (viewer.js imports clean without a DOM);
the parse is the real witness argparse with `main` stubbed out.
"""

import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import cloudyview as cv

# The package exports witness the FUNCTION; the CLI lives on the module.
witness = importlib.import_module("cloudyview.witness")

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="needs node")

# A scene as the loader leaves it after inferring some things and asking
# others: DALES split-file ice with a space in the ice file's name and its
# own answered units, a multi-step time dimension, units the user supplied,
# an answered coordinate-units question, SAM-spelled axes. The ice units
# deliberately differ from the main --units so a parse that conflated the
# two would show.
SCENE = {
    "liquidVar": "clw",
    "iceVar": None,
    "iceFrom": {"filename": "DALES cli.nc", "iceVar": "cli", "group": None,
                "units": "g/kg"},
    "timeDim": {"name": "time", "size": 4},
    "timestep": 2,
    "unitsAssumed": "kg/kg",
    "coordUnitsAssumed": "km",
    "dimNames": {"x": "xt", "y": "yt", "z": "zt"},
    "coordNames": {"x": "xt", "y": "yt", "z": "zt"},
}

_JS = textwrap.dedent("""
    import { Viewer } from "%s";
    const q = (v) => (/\\s/.test(String(v))
      ? JSON.stringify(String(v)) : String(v));
    const flags = (scene) =>
      Viewer.prototype._selectionFlags.call({ scene }, q).join(" ");
    const scene = %s;
    const noIce = { ...scene, iceFrom: null };
    const bare = { ...scene, iceFrom: null, iceVar: "qi" };
    process.stdout.write(JSON.stringify({
      full: flags(scene),
      noIce: flags(noIce),
      sameFileIce: flags(bare),
      demo: flags({}),            // a demo scene carries no selection
      coarseDemo: flags({ coarsen: 2 }),   // ...but a coarsened bake's grid
      // A metric-less regular grid, placed by the grid-spacing answer: x
      // and z asked (y had meters), z with its first level at the top.
      spaced: flags({ liquidVar: "qc",
                      gridSpacingAssumed: { spacing: { x: 250, z: 40.5 },
                                            zFirst: "top" } }),
    }));
""") % ((REPO / "web" / "soar" / "viewer.js").as_posix(), json.dumps(SCENE))


@pytest.fixture(scope="module")
def emitted():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def _parse_with_witness_cli(monkeypatch, flag_string):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(witness, "main", fake_main)
    argv = ["witness", "input.nc"] + shlex.split(flag_string)
    monkeypatch.setattr(sys, "argv", argv)
    witness.cli()
    return captured["kwargs"]


def test_full_selection_round_trips(monkeypatch, emitted):
    kwargs = _parse_with_witness_cli(
        monkeypatch, emitted["full"] + " --quality max")
    assert kwargs["liquid_water_var"] == "clw"
    assert kwargs["ice"] == "DALES cli.nc"          # quoting survived shlex
    assert kwargs["ice_water_var"] == "cli"
    assert kwargs["timestep"] == 2
    assert kwargs["fallback_units"] == "kg/kg"
    assert kwargs["x_dim"] == "xt" and kwargs["z_dim"] == "zt"
    assert kwargs["x_coord_name"] == "xt" and kwargs["z_coord_name"] == "zt"
    assert kwargs["quality"] == "max"
    assert kwargs["no_ice"] is False
    # The answered ice and coordinate units reach the CLI. Membership rather
    # than a dest name: the contract fixes the FLAG spellings (--ice-units,
    # --coord-units); the argparse dest is the CLI's own business.
    assert "g/kg" in kwargs.values()          # --ice-units, distinct from
    assert kwargs["fallback_units"] == "kg/kg"  # the main --units
    assert "km" in kwargs.values()            # --coord-units


def test_answered_units_are_emitted_at_all(emitted):
    """The browser-asked ice units lived only in iceFrom.units and were
    dropped from the copied command; the coord-units answer is new. Both
    must appear, under the contract's exact flag names."""
    assert "--ice-units g/kg" in emitted["full"]
    assert "--coord-units km" in emitted["full"]
    # And only when there is something to say: no split-file ice, no
    # answered ice units.
    assert "--ice-units" not in emitted["noIce"]
    assert "--ice-units" not in emitted["sameFileIce"]


def test_a_view_without_ice_stays_without_ice(monkeypatch, emitted):
    # --no-ice is a fact about the view, not a default: the CLI must not
    # re-infer ice the browser did not show.
    assert "--no-ice" in emitted["noIce"]
    kwargs = _parse_with_witness_cli(monkeypatch, emitted["noIce"])
    assert kwargs["no_ice"] is True
    assert kwargs["ice"] is None


def test_same_file_ice_names_the_variable(monkeypatch, emitted):
    kwargs = _parse_with_witness_cli(monkeypatch, emitted["sameFileIce"])
    assert kwargs["ice_water_var"] == "qi"
    assert kwargs["ice"] is None
    assert kwargs["no_ice"] is False


def test_the_grid_spacing_answer_round_trips(monkeypatch, emitted):
    """Exactly the axes the browser asked about, and the vertical order
    with them — the CLI refuses a spacing on an axis with meters, so an
    extra flag would break the reproduction rather than be ignored."""
    assert "--dy" not in emitted["spaced"]
    kwargs = _parse_with_witness_cli(monkeypatch, emitted["spaced"])
    assert kwargs["grid_spacing"] == {"x": 250.0, "z": 40.5}
    assert kwargs["z_first"] == "top"
    plain = _parse_with_witness_cli(monkeypatch, emitted["noIce"])
    assert plain["grid_spacing"] is None and plain["z_first"] is None


def test_a_demo_scene_emits_no_selection(emitted):
    assert emitted["demo"] == ""


def test_a_coarse_demo_says_how_its_grid_was_made(monkeypatch, emitted):
    """The coarse marine demo names its FINE source file, so its command
    carries the block average that made the shipped grid from it."""
    assert emitted["coarseDemo"] == "--coarsen 2"
    assert _parse_with_witness_cli(
        monkeypatch, emitted["coarseDemo"])["coarsen"] == 2
    assert _parse_with_witness_cli(monkeypatch, "")["coarsen"] is None


# --- the surface: what the world stands on ---------------------------------
#
# A cyberpunk still's command used to carry no surface at all, so `witness`
# rendered the day ocean under it. This half drives the browser's real
# witnessCommand for each surface and runs what comes out through the real
# witness CLI all the way to pack_uniforms — only the GPU is stubbed — and
# checks that the rows which say where the world stands are the browser's:
# the specialization, row 8 (the city tile's world offset), row 9 (the bound
# tile's cell, extent and top mip), row 24 (the surface offset the view
# rendered with) and row 0 (the camera, through the relative round trip).

witness_module = witness

# The field the fake loader hands the CLI: 8 x 8 x 4 cells of 100 m, cloud in
# every cell so the z crop keeps the whole box.
_NX, _NZ, _DX = 8, 4, 100.0

CITY_OFFSET = [8100.0, -44190.0]          # K.CITY_TILE_OFFSET_M

# (name, scene flags, tile extent, the period the surface frame folds at,
# the surface's clock period (null: it has none), whether the flight's folds
# moved the surface frame off the static offset — which is when
# --surface-origin must appear and carry it). The city and the desert fold
# at their tile, the sea at 16 tiles (SURFACE_FRAME_TILES); only the city
# has a clock (its steam).
SURFACE_CASES = [
    ("ocean", {}, 102.4, 1638.4, None, False),
    ("ocean-drifted", {}, 102.4, 1638.4, None, True),
    ("city", {"city": True, "cityOffsetM": CITY_OFFSET}, 92160.0, 92160.0,
     2600.0, False),
    ("city-drifted", {"city": True, "cityOffsetM": CITY_OFFSET}, 92160.0,
     92160.0, 2600.0, True),
    ("desert", {"desert": True}, 4096.0, 4096.0, None, False),
    ("desert-drifted", {"desert": True}, 4096.0, 4096.0, None, True),
]
# The scene clock the fake views were held at: where the city's steam was.
SCENE_TIME = 37.625

_SURFACE_JS = textwrap.dedent("""
    import { Viewer } from "%s";
    const cases = %s;
    const P = Viewer.prototype;
    const fold = (v, t) => { const r = v %% t; return r < 0 ? r + t : r; };
    const out = {};
    for (const c of cases) {
      const t = c.period;
      const position = [c.bmin[0] + 312.5, c.bmin[1] + 587.25, 1234.5];
      const stat = c.scene.city ? c.scene.cityOffsetM : [0.0, 0.0];
      // The camera's surface frame: where the static offset puts it, or
      // three blocks along when the flight's folds have moved it.
      const sp = [0, 1].map((i) =>
        fold(fold(position[i], t) - fold(stat[i], t) + (c.drift ? 270.0 : 0.0), t));
      const self = {
        scene: { ...c.scene, bmin: c.bmin, bmax: c.bmax,
                 sourceName: "field.nc", oceanTileExtent: c.tile,
                 surfacePeriod: t, clockPeriod: c.clock },
        camera: { position, azimuth: 301.25, elevation: 12.5, fov: 90.0,
                  surfacePosition: sp, surfaceTileExtent: t },
        canvas: { width: 1920, height: 961 },
        renderer: { periodic: true },
        sunAzimuth: 181.0, sunElevation: 20.0, sceneTime: c.sceneTime,
        toneMapGamma: 1.66, toneMapWhitePoint: 15.0, contrast: 1.0,
        _selectionFlags: P._selectionFlags,
        _surfaceOffsetM: P._surfaceOffsetM,
        _surfaceOriginFlags: P._surfaceOriginFlags,
      };
      out[c.name] = {
        command: P.witnessCommand.call(self, {
          size: [640, 360], lodStrength: 1.0, haze: 0.35,
          hazeHeightDependent: false, exposure: 2.5, quality: "high" }),
        surfaceOffsetM: P._surfaceOffsetM.call(self),
        position,
      };
    }
    process.stdout.write(JSON.stringify(out));
""")


def _surface_field():
    from cloudyview.cloudfield import CloudField
    x = (np.arange(_NX) + 0.5) * _DX
    z = (np.arange(_NZ) + 0.5) * _DX
    return CloudField(lwc=np.full((_NX, _NX, _NZ), 0.5, np.float32),
                      x=x, y=x.copy(), z=z)


@pytest.fixture(scope="module")
def surface_commands():
    bmin, bmax = witness_module._volume_aabb(_surface_field())
    cases = [dict(name=name, scene=scene, tile=tile, period=period,
                  clock=clock, drift=drift, sceneTime=SCENE_TIME,
                  bmin=list(bmin), bmax=list(bmax))
             for name, scene, tile, period, clock, drift in SURFACE_CASES]
    js = _SURFACE_JS % ((REPO / "web" / "soar" / "viewer.js").as_posix(),
                        json.dumps(cases))
    proc = subprocess.run(["node", "--input-type=module", "-e", js],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


class _FakeRenderer:
    """What render_nested needs of a SoarRenderer, minus the GPU: the bound
    tile's meta, and a render that keeps the uniforms it would have drawn."""

    def __init__(self, surface):
        from cloudyview.soar_host import pack_uniforms
        self._pack = pack_uniforms
        self.surface = surface
        self.surface_meta = json.loads(
            (REPO / "cloudyview" / "soar" / surface / "meta.json").read_text())
        self.uniforms = None

    def bake_light_cache(self, state, view, divisor):
        pass

    def render(self, state, view, frames):
        self.uniforms = self._pack(state, view)
        w, h = view.output_size
        return np.zeros((h, w, 3))


@pytest.mark.parametrize("name", [c[0] for c in SURFACE_CASES])
def test_the_surface_round_trips_to_the_uniforms(
        monkeypatch, tmp_path, surface_commands, name):
    emitted = surface_commands[name]
    made = []

    def fake_renderer_for(levels, *, periodic, tone_mapped, surface):
        made.append(_FakeRenderer(surface))
        return made[-1]

    monkeypatch.setattr(witness_module, "_renderer_for", fake_renderer_for)
    monkeypatch.setattr(witness_module, "_load_field",
                        lambda *a, **k: _surface_field())
    argv = shlex.split(emitted["command"])
    monkeypatch.setattr(sys, "argv",
                        argv + ["--output", str(tmp_path)])
    witness_module.cli()

    (renderer,) = made
    u = renderer.uniforms
    scene, tile, drift = next((s, t, d) for n, s, t, _, _c, d in SURFACE_CASES
                              if n == name)
    kind = ("city" if scene.get("city")
            else "desert" if scene.get("desert") else "ocean")
    meta = renderer.surface_meta
    assert renderer.surface == kind
    # Row 9: cell size, tile extent, surface on, top mip — the bound tile's.
    cell = meta["dx_m"] if kind == "ocean" else meta["cell_m"]
    assert tuple(u[9]) == pytest.approx(
        (cell, tile, 1.0, meta["mips"] - 1), rel=0, abs=0)
    # Row 24: the offset the browser view rendered with, float32-exact, and
    # the scene clock it was held at.
    assert tuple(u[24][:2]) == tuple(
        np.float32(emitted["surfaceOffsetM"]))
    # The clock runs where something reads it: a city still carries it (its
    # period far exceeds it, so it packs unwrapped); an ocean or a desert
    # still neither writes --time nor packs one.
    if kind == "city":
        assert u[24][2] == np.float32(SCENE_TIME)
    else:
        assert u[24][2] == 0.0
        assert "--time" not in argv
    assert ("--surface-origin" in argv) == drift
    # Row 0: the camera, through worldToRelative and back.
    assert u[0][:3] == pytest.approx(emitted["position"], rel=1e-9, abs=1e-6)
    if kind == "desert":
        # Row 8 under DESERT: the datum and the tile's relief, from its meta.
        assert tuple(u[8]) == tuple(np.float32(
            (0.0, meta["height_min_m"], meta["height_max_m"],
             meta["band_max_m"])))
    if kind == "city":
        assert tuple(u[8][1:3]) == tuple(np.float32(CITY_OFFSET))
        # Under CITY the sun row is the moon the browser flew under.
        assert "--sun-elevation 20" in emitted["command"]


def test_a_city_command_says_which_district(surface_commands):
    """The offset travels with the surface: the CLI's own default is the
    night-city harness's district, not cyberpunk's."""
    city = surface_commands["city"]["command"]
    assert "--surface city --city-offset 8100 -44190" in city
    assert "--surface ocean" in surface_commands["ocean"]["command"]
    assert "--city-offset" not in surface_commands["ocean"]["command"]


def test_a_still_command_carries_the_capture_size(surface_commands):
    # The capture size, not the canvas's (1920 x 961 in the fake viewer).
    assert "--size 640 360" in surface_commands["ocean"]["command"]


def test_witness_refuses_a_city_without_a_district():
    from cloudyview.witness import render_nested
    with pytest.raises(ValueError, match="city_offset"):
        render_nested([None], (0, 0, 0), azimuth=0, elevation=0,
                      sun_azimuth=0, sun_elevation=20, surface="city")
    with pytest.raises(ValueError, match="surface='city'"):
        render_nested([None], (0, 0, 0), azimuth=0, elevation=0,
                      sun_azimuth=0, sun_elevation=20, surface="ocean",
                      city_offset=(0, 0))


# --- a coarse demo: the command remakes the grid the bake shipped ----------
#
# marine-congestus-coarse is demo_marine-congestus.nc block-averaged 2x2x2 by
# tools/prebake_demos.py, and the browser can only name that fine file. Here
# a small fine field goes through the bake's own load_demo, the browser's
# witnessCommand is built over the box the bake would ship, and the command
# runs through the real witness CLI and loader (only the GPU is stubbed): the
# field it marches must be the bake's to the fp16 bit, in the same box, with
# the camera where the browser had it. Before `--coarsen` it marched the fine
# field in a box ~10 m taller.

_COARSE_JS = textwrap.dedent("""
    import { Viewer } from "%s";
    const c = %s;
    const P = Viewer.prototype;
    const position = [c.bmin[0] + 312.5, c.bmin[1] + 587.25, 433.0];
    const self = {
      scene: { coarsen: 2, bmin: c.bmin, bmax: c.bmax, sourceName: c.source,
               oceanTileExtent: 102.4, surfacePeriod: 1638.4,
               clockPeriod: null },
      camera: { position, azimuth: 301.25, elevation: 12.5, fov: 90.0,
                surfacePosition: null, surfaceTileExtent: 1638.4 },
      canvas: { width: 640, height: 360 },
      renderer: { periodic: true },
      sunAzimuth: 181.0, sunElevation: 20.0, sceneTime: 0.0,
      toneMapGamma: 1.66, toneMapWhitePoint: 15.0, contrast: 1.0,
      _selectionFlags: P._selectionFlags,
      _surfaceOffsetM: P._surfaceOffsetM,
      _surfaceOriginFlags: P._surfaceOriginFlags,
    };
    process.stdout.write(JSON.stringify({
      command: P.witnessCommand.call(self, {
        size: [64, 36], lodStrength: 1.0, haze: 0.35,
        hazeHeightDependent: false, exposure: 2.5, quality: "high" }),
      position,
    }));
""")


def _write_fine_field(path):
    """8 x 8 x 12 cells of 100 x 100 x 50 m, kg/kg as STEAM writes it. Cloud
    in planes 3-9 only: the band starts on an odd plane and is an odd 7
    planes deep, so the bake's blocks start at plane 3 and the band grows
    to 3-10, where coarsening the whole column would pair 2-3, 4-5, ..."""
    from netCDF4 import Dataset
    rng = np.random.default_rng(7)
    qc = np.zeros((8, 8, 12), np.float32)
    qi = np.zeros((8, 8, 12), np.float32)
    # Six decades of mixing ratio, as a cascade field has.
    qc[:, :, 3:10] = 10.0 ** rng.uniform(-9, -3, (8, 8, 7))
    qi[:, :, 6:10] = 10.0 ** rng.uniform(-9, -4, (8, 8, 4))
    with Dataset(path, "w") as ds:
        for name, n, d in (("x", 8, 100.0), ("y", 8, 100.0), ("z", 12, 50.0)):
            ds.createDimension(name, n)
            v = ds.createVariable(name, "f8", (name,))
            v.units = "m"
            v[:] = (np.arange(n) + 0.5) * d
        for name, a in (("qc", qc), ("qi", qi)):
            v = ds.createVariable(name, "f4", ("x", "y", "z"))
            v.units = "kg/kg"
            v[:] = a


def _bake(path):
    """The coarse demo's extinction and box, by the bake's own code."""
    sys.path.insert(0, str(REPO / "tools"))
    import prebake_demos as pb
    spec = dict(id="coarse-test", liquid=(path.name, "qc"),
                ice=(path.name, "qi"), src=path.parent, dims="xyz",
                crop=dict(x=(0, 8), y=(0, 8), z="auto"), scale=1e3,
                coarsen=2)
    field = pb.load_demo(spec)
    sigma = pb._extinction(field.lwc, field.z, field.iwc)
    bmin, bmax = pb._volume_aabb(field)
    return sigma.astype(np.float16), bmin, bmax


def test_a_coarse_demo_command_marches_the_baked_grid(monkeypatch, tmp_path):
    fine = tmp_path / "fine.nc"
    _write_fine_field(fine)
    baked, bmin, bmax = _bake(fine)
    assert baked.shape == (4, 4, 4)           # z band 3-10, in 2-plane blocks

    js = _COARSE_JS % ((REPO / "web" / "soar" / "viewer.js").as_posix(),
                       json.dumps(dict(bmin=list(bmin), bmax=list(bmax),
                                       source=str(fine))))
    proc = subprocess.run(["node", "--input-type=module", "-e", js],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    emitted = json.loads(proc.stdout)
    assert "--coarsen 2" in emitted["command"]

    made, levels = [], []

    def fake_renderer_for(lv, *, periodic, tone_mapped, surface):
        levels.extend(lv)
        made.append(_FakeRenderer(surface))
        return made[-1]

    monkeypatch.setattr(witness_module, "_renderer_for", fake_renderer_for)
    monkeypatch.setattr(sys, "argv", shlex.split(emitted["command"])
                        + ["--output", str(tmp_path)])
    witness_module.cli()

    (level,) = levels
    # The bake's field, bit for bit as the r16float texture holds it...
    assert level.sigma.shape == baked.shape
    assert np.array_equal(level.sigma.astype(np.float16).view(np.uint16),
                          baked.view(np.uint16))
    # ...in the bake's box, so the relative camera lands where it flew.
    assert list(level.bmin) == list(bmin) and list(level.bmax) == list(bmax)
    (renderer,) = made
    assert renderer.uniforms[0][:3] == pytest.approx(
        emitted["position"], rel=1e-9, abs=1e-6)

    # The control: blocks that start at the bottom of the file's column
    # instead of the band's make a different field, so the test can fail.
    from cloudyview.coarsen import axis_mean, block_mean
    field = cv.load(fine)
    whole = cv.CloudField(lwc=block_mean(field.lwc, 2),
                          iwc=block_mean(field.iwc, 2),
                          x=axis_mean(field.x, 2), y=axis_mean(field.y, 2),
                          z=axis_mean(field.z, 2))
    other = witness_module._field_level(whole, "single").sigma
    assert other.shape != baked.shape or not np.array_equal(
        other.astype(np.float16).view(np.uint16), baked.view(np.uint16))


def test_coarsen_refuses_what_it_cannot_average(tmp_path):
    from cloudyview.coarsen import coarsened, grow_band
    fine = tmp_path / "fine.nc"
    _write_fine_field(fine)
    field = cv.load(fine)
    with pytest.raises(ValueError, match="x axis has 8 cells"):
        coarsened(field, 3)
    with pytest.raises(ValueError, match="whole number"):
        coarsened(field, 0)
    assert coarsened(field, 1) is field
    # Up first, then down when the top is reached; never inward.
    assert grow_band(3, 10, 12, 2) == (3, 11)
    assert grow_band(5, 12, 12, 2) == (4, 12)
    with pytest.raises(ValueError, match="no z band"):
        grow_band(0, 7, 7, 2)


def test_witness_refuses_a_coarse_field_under_a_nest(capsys):
    # Refused before the load: the file need not exist for the pair to be
    # wrong, and the message is what tells this exit from a missing file's.
    with pytest.raises(SystemExit):
        witness_module.main("missing.nc", nest_group="/nest", coarsen=2)
    assert "--coarsen and --nest-group" in capsys.readouterr().err
