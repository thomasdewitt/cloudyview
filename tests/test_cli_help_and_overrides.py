"""CLI parsing/help coverage for dataset override flags."""

import importlib
import subprocess
import sys

import pytest

# The package attributes `cloudyview.glimpse` / `.witness` / `.behold` are
# the public render functions (they shadow the submodules); import the CLI
# modules themselves explicitly.
behold = importlib.import_module("cloudyview.behold")
glimpse = importlib.import_module("cloudyview.glimpse")
witness = importlib.import_module("cloudyview.witness")


def test_glimpse_cli_passes_dataset_override_args(monkeypatch):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(glimpse, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "glimpse",
            "input.nc",
            "--group",
            "physics/clouds",
            "--liquid-water-var",
            "qc_cloud",
            "--ice-water-var",
            "qi_cloud",
            "--coords-group",
            "grid",
            "--x-coord",
            "xh",
            "--y-coord",
            "yh",
            "--z-coord",
            "zh",
            "--x-dim",
            "ni",
            "--y-dim",
            "nj",
            "--z-dim",
            "nk",
        ],
    )

    glimpse.cli()

    assert captured["args"] == ("input.nc", ".", False)
    assert captured["kwargs"]["dataset_group"] == "physics/clouds"
    assert captured["kwargs"]["liquid_water_var"] == "qc_cloud"
    assert captured["kwargs"]["ice_water_var"] == "qi_cloud"
    assert captured["kwargs"]["coords_group"] == "grid"
    assert captured["kwargs"]["x_coord_name"] == "xh"
    assert captured["kwargs"]["y_coord_name"] == "yh"
    assert captured["kwargs"]["z_coord_name"] == "zh"
    assert captured["kwargs"]["x_dim"] == "ni"
    assert captured["kwargs"]["y_dim"] == "nj"
    assert captured["kwargs"]["z_dim"] == "nk"


def test_witness_cli_passes_dataset_override_args(monkeypatch):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(witness, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "witness",
            "input.nc",
            "high",
            "--liquid-water-group",
            "state/liquid",
            "--ice-water-group",
            "state/ice",
            "--coords-group",
            "grid",
            "--x-dim",
            "ni",
            "--y-dim",
            "nj",
            "--z-dim",
            "nk",
        ],
    )

    witness.cli()

    assert captured["args"][:2] == ("input.nc", None)
    assert captured["kwargs"]["liquid_water_group"] == "state/liquid"
    assert captured["kwargs"]["ice_water_group"] == "state/ice"
    assert captured["kwargs"]["coords_group"] == "grid"
    assert captured["kwargs"]["custom_size"] == witness.QUALITY_PRESETS["high"]
    assert captured["kwargs"]["x_dim"] == "ni"
    assert captured["kwargs"]["y_dim"] == "nj"
    assert captured["kwargs"]["z_dim"] == "nk"


def test_witness_cli_passes_image_control_args(monkeypatch):
    """The browser's `copy command` button emits exactly these flag names."""
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(witness, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "witness",
            "input.nc",
            "--gamma",
            "1.9",
            "--white-point",
            "12.5",
            "--contrast",
            "1.15",
            "--haze",
            "0.4",
            "--periodic",
            "--nest-group",
            "nest",
        ],
    )

    witness.cli()

    assert captured["kwargs"]["tone_map_gamma"] == pytest.approx(1.9)
    assert captured["kwargs"]["tone_map_white_point"] == pytest.approx(12.5)
    assert captured["kwargs"]["contrast"] == pytest.approx(1.15)
    assert captured["kwargs"]["haze"] == pytest.approx(0.4)
    assert captured["kwargs"]["periodic"] is True
    assert captured["kwargs"]["nest_group"] == "nest"


def test_witness_cli_image_controls_default_to_none(monkeypatch):
    """Unset knobs stay None so the library defaults are the only source."""
    captured = {}

    monkeypatch.setattr(witness, "main", lambda *a, **k: captured.update(k))
    monkeypatch.setattr(sys, "argv", ["witness", "input.nc"])

    witness.cli()

    for key in ("tone_map_gamma", "tone_map_white_point", "contrast",
                "haze", "nest_group"):
        assert captured[key] is None
    assert captured["periodic"] is False


def test_witness_main_forwards_image_controls_to_witness(monkeypatch, tmp_path):
    """main() passes only the knobs that were given through to witness()."""
    captured = {}

    monkeypatch.setattr(witness, "_load_field", lambda *a, **k: object())
    monkeypatch.setattr(
        witness, "witness",
        lambda field, **kwargs: captured.update(kwargs) or _dummy_image())
    monkeypatch.setattr(
        "cloudyview.basic_render.save_image", lambda *a, **k: None)

    witness.main("input.nc", str(tmp_path), tone_map_gamma=2.0, haze=0.25,
                 periodic=True)

    assert captured["tone_map_gamma"] == pytest.approx(2.0)
    assert captured["haze"] == pytest.approx(0.25)
    assert captured["periodic"] is True
    # Not given on the command line: left out entirely.
    assert "tone_map_white_point" not in captured
    assert "contrast" not in captured


def test_witness_main_nest_group_builds_finest_first(monkeypatch, tmp_path):
    """--nest-group renders two levels, nest first, camera on the outer box."""
    import numpy as np

    loads = []

    def fake_load(filename, **kwargs):
        loads.append((filename, kwargs))
        return f"field{len(loads)}"

    levels_seen = {}

    def fake_field_level(field, name, verbose=False):
        scale = 1.0 if field == "field1" else 0.25
        return witness.NestedLevel(
            sigma=np.zeros((2, 2, 2)),
            bmin=np.array([-1000.0, -1000.0, 0.0]) * scale,
            bmax=np.array([1000.0, 1000.0, 2000.0]) * scale,
            name=name,
        )

    def fake_render_nested(levels, position, **kwargs):
        levels_seen["levels"] = levels
        levels_seen["position"] = position
        levels_seen["kwargs"] = kwargs
        return _dummy_image()

    monkeypatch.setattr(witness, "_load_field", fake_load)
    monkeypatch.setattr(witness, "_field_level", fake_field_level)
    monkeypatch.setattr(witness, "render_nested", fake_render_nested)
    monkeypatch.setattr(
        "cloudyview.basic_render.save_image", lambda *a, **k: None)

    witness.main("input.nc", str(tmp_path), nest_group="fine",
                 liquid_water_group="state/liquid", liquid_water_var="qc",
                 x_dim="ni", tone_map_white_point=11.0, periodic=True)

    # Outer load keeps the group-specific overrides; the nest gets the nest
    # group as its dataset group and only the NAME overrides.
    outer_kwargs, nest_kwargs = loads[0][1], loads[1][1]
    assert loads[1][0] == "input.nc"
    assert outer_kwargs["liquid_water_group"] == "state/liquid"
    assert nest_kwargs["dataset_group"] == "fine"
    assert "liquid_water_group" not in nest_kwargs
    assert nest_kwargs["liquid_water_var"] == "qc"
    assert nest_kwargs["x_dim"] == "ni"

    names = [lvl.name for lvl in levels_seen["levels"]]
    assert names == ["fine", "outer"], "finest level must come first"
    assert levels_seen["kwargs"]["periodic"] is True
    assert levels_seen["kwargs"]["tone_map_white_point"] == pytest.approx(11.0)
    # Relative z=-0.999 resolves against the OUTER box (2 km deep here).
    assert abs(levels_seen["position"][2]) < 100.0


def test_witness_main_nest_group_load_failure_is_loud(monkeypatch, tmp_path):
    """A nest that will not load must never degrade to a single-field image."""
    rendered = []

    def fake_load(filename, **kwargs):
        if kwargs.get("dataset_group") == "missing":
            raise KeyError("no such group")
        return "outer"

    monkeypatch.setattr(witness, "_load_field", fake_load)
    monkeypatch.setattr(witness, "witness",
                        lambda *a, **k: rendered.append("single"))
    monkeypatch.setattr(witness, "render_nested",
                        lambda *a, **k: rendered.append("nested"))

    with pytest.raises(SystemExit) as excinfo:
        witness.main("input.nc", str(tmp_path), nest_group="missing")

    assert excinfo.value.code == 1
    assert rendered == []


def _dummy_image():
    import numpy as np
    return np.zeros((4, 4, 3))


def test_behold_cli_passes_dataset_override_args(monkeypatch):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(behold, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "behold",
            "input.nc",
            "medium",
            "--cpu",
            "--group",
            "physics/clouds",
            "--liquid-water-var",
            "qc_cloud",
            "--x-coord",
            "xh",
            "--y-coord",
            "yh",
            "--z-coord",
            "zh",
        ],
    )

    behold.cli()

    assert captured["args"][:4] == ("input.nc", "llvm", "medium", None)
    assert captured["kwargs"]["dataset_group"] == "physics/clouds"
    assert captured["kwargs"]["liquid_water_var"] == "qc_cloud"
    assert captured["kwargs"]["x_coord_name"] == "xh"
    assert captured["kwargs"]["y_coord_name"] == "yh"
    assert captured["kwargs"]["z_coord_name"] == "zh"


@pytest.mark.parametrize(
    ("module_name", "required_text"),
    [
        ("cloudyview.glimpse", "--liquid-water-group"),
        ("cloudyview.witness", "--coords-group"),
        ("cloudyview.behold", "--x-dim"),
    ],
)
def test_cli_help_lists_dataset_override_flags(module_name: str, required_text: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert required_text in result.stdout
    assert "Input dataset selection:" in result.stdout


def test_witness_help_lists_image_control_flags():
    result = subprocess.run(
        [sys.executable, "-m", "cloudyview.witness", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for flag in ("--gamma", "--white-point", "--contrast", "--haze",
                 "--periodic", "--nest-group"):
        assert flag in result.stdout
