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

# (name, scene flags, tile extent, whether the flight's folds moved the
# surface frame off the static offset — which is when --surface-origin must
# appear and carry it).
SURFACE_CASES = [
    ("ocean", {}, 102.4, False),
    ("city", {"city": True, "cityOffsetM": CITY_OFFSET}, 92160.0, False),
    ("city-drifted", {"city": True, "cityOffsetM": CITY_OFFSET}, 92160.0,
     True),
    ("desert", {"desert": True}, 4096.0, False),
    ("desert-drifted", {"desert": True}, 4096.0, True),
]

_SURFACE_JS = textwrap.dedent("""
    import { Viewer } from "%s";
    const cases = %s;
    const P = Viewer.prototype;
    const fold = (v, t) => { const r = v %% t; return r < 0 ? r + t : r; };
    const out = {};
    for (const c of cases) {
      const t = c.tile;
      const position = [c.bmin[0] + 312.5, c.bmin[1] + 587.25, 1234.5];
      const stat = c.scene.city ? c.scene.cityOffsetM : [0.0, 0.0];
      // The camera's surface frame: where the static offset puts it, or
      // three blocks along when the flight's folds have moved it.
      const sp = [0, 1].map((i) =>
        fold(fold(position[i], t) - fold(stat[i], t) + (c.drift ? 270.0 : 0.0), t));
      const self = {
        scene: { ...c.scene, bmin: c.bmin, bmax: c.bmax,
                 sourceName: "field.nc", oceanTileExtent: t },
        camera: { position, azimuth: 301.25, elevation: 12.5, fov: 90.0,
                  surfacePosition: sp, surfaceTileExtent: t },
        canvas: { width: 1920, height: 961 },
        renderer: { periodic: true },
        sunAzimuth: 181.0, sunElevation: 20.0,
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
    cases = [dict(name=name, scene=scene, tile=tile, drift=drift,
                  bmin=list(bmin), bmax=list(bmax))
             for name, scene, tile, drift in SURFACE_CASES]
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
    scene, tile, drift = next((s, t, d) for n, s, t, d in SURFACE_CASES
                              if n == name)
    kind = ("city" if scene.get("city")
            else "desert" if scene.get("desert") else "ocean")
    meta = renderer.surface_meta
    assert renderer.surface == kind
    # Row 9: cell size, tile extent, surface on, top mip — the bound tile's.
    cell = meta["dx_m"] if kind == "ocean" else meta["cell_m"]
    assert tuple(u[9]) == pytest.approx(
        (cell, tile, 1.0, meta["mips"] - 1), rel=0, abs=0)
    # Row 24: the offset the browser view rendered with, float32-exact.
    assert tuple(u[24][:2]) == tuple(
        np.float32(emitted["surfaceOffsetM"]))
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
