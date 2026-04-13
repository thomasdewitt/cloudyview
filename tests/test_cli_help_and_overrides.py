"""CLI parsing/help coverage for dataset override flags."""

import subprocess
import sys

import pytest

from cloudyview import behold, glimpse, witness


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


def test_witness_cli_passes_gpu_flag(monkeypatch):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(witness, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["witness", "input.nc", "--gpu"],
    )

    witness.cli()

    assert captured["kwargs"]["gpu"] is True


def test_witness_cli_gpu_defaults_false(monkeypatch):
    captured = {}

    def fake_main(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(witness, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        ["witness", "input.nc"],
    )

    witness.cli()

    assert captured["kwargs"]["gpu"] is False


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
