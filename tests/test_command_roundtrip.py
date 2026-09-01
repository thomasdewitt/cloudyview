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


def test_a_demo_scene_emits_no_selection(emitted):
    assert emitted["demo"] == ""
