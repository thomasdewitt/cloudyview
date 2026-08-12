"""Tests for explicit variable/group/coordinate override handling."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cloudyview import io


def _write_grouped_dataset(path: Path) -> None:
    xr.Dataset().to_netcdf(path, mode="w")

    coords = xr.Dataset(
        data_vars={
            "xh": ("ni", np.array([0.0, 1000.0], dtype=np.float64)),
            "yh": ("nj", np.array([0.0, 2000.0, 4000.0], dtype=np.float64)),
            "zh": ("nk", np.array([100.0, 300.0, 800.0, 1600.0], dtype=np.float64)),
        }
    )
    coords.to_netcdf(path, mode="a", group="grid")

    liquid = xr.Dataset(
        data_vars={
            "qc_cloud": (
                ("ni", "nj", "nk"),
                np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10.0,
                {"units": "g/kg"},
            )
        }
    )
    liquid.to_netcdf(path, mode="a", group="physics/liquid")

    ice = xr.Dataset(
        data_vars={
            "qi_cloud": (
                ("ni", "nj", "nk"),
                np.full((2, 3, 4), 0.0002, dtype=np.float64),
                {"units": "kg/kg"},
            )
        }
    )
    ice.to_netcdf(path, mode="a", group="physics/ice")


def test_load_and_validate_supports_group_and_name_overrides(tmp_path: Path):
    path = tmp_path / "grouped.nc"
    _write_grouped_dataset(path)

    data = io.load_and_validate(
        str(path),
        liquid_water_var="qc_cloud",
        ice_water_var="qi_cloud",
        liquid_water_group="physics/liquid",
        ice_water_group="physics/ice",
        coords_group="grid",
        x_coord_name="xh",
        y_coord_name="yh",
        z_coord_name="zh",
        x_dim="ni",
        y_dim="nj",
        z_dim="nk",
    )

    assert data["liquid_water_var"] == "qc_cloud"
    assert data["ice_water_var"] == "qi_cloud"
    assert data["liquid_water_data"].dims == ("x", "y", "z")
    assert data["ice_water_data"].dims == ("x", "y", "z")
    assert data["liquid_water_data"].shape == (2, 3, 4)
    assert data["ice_water_data"].shape == (2, 3, 4)
    np.testing.assert_allclose(data["x_coord"], [0.0, 1000.0])
    np.testing.assert_allclose(data["y_coord"], [0.0, 2000.0, 4000.0])
    np.testing.assert_allclose(data["z_coord"], [100.0, 300.0, 800.0, 1600.0])
    np.testing.assert_allclose(
        data["ice_water_data"].values,
        np.full((2, 3, 4), 0.2, dtype=np.float64),
    )


def _write_nested_render_file(path: Path, *, with_units: bool = True) -> None:
    """Two sibling groups each holding a full field, and an empty root.

    The shape STEAM writes its render nests in.
    """
    xr.Dataset().to_netcdf(path, mode="w")
    attrs = {"units": "kg/kg"} if with_units else {}
    for group in ("render_a", "render_b"):
        xr.Dataset(
            data_vars={
                "qc": (
                    ("x", "y", "z"),
                    np.full((2, 3, 4), 0.001, dtype=np.float64),
                    dict(attrs),
                ),
                "qi": (
                    ("x", "y", "z"),
                    np.full((2, 3, 4), 0.0002, dtype=np.float64),
                    dict(attrs),
                ),
            },
            coords={
                "x": np.array([0.0, 1000.0]),
                "y": np.array([0.0, 2000.0, 4000.0]),
                "z": np.array([100.0, 300.0, 800.0, 1600.0]),
            },
        ).to_netcdf(path, mode="a", group=group)


def test_find_liquid_water_groups_reports_root_and_nested_fields(tmp_path: Path):
    nested = tmp_path / "nests.nc"
    _write_nested_render_file(nested)
    assert io.find_liquid_water_groups(str(nested)) == ["render_a", "render_b"]

    flat = tmp_path / "flat.nc"
    xr.Dataset(
        data_vars={
            "qc": (
                ("x", "y", "z"),
                np.zeros((2, 3, 4), dtype=np.float64),
                {"units": "g/kg"},
            )
        }
    ).to_netcdf(flat)
    # The root group is reported as "" — the loader's own default.
    assert io.find_liquid_water_groups(str(flat)) == [""]

    grouped = tmp_path / "grouped.nc"
    _write_grouped_dataset(grouped)
    # Non-standard variable names are not candidates without an override.
    assert io.find_liquid_water_groups(str(grouped)) == []


def test_condensate_vars_missing_units_and_fallback(tmp_path: Path):
    path = tmp_path / "nests_no_units.nc"
    _write_nested_render_file(path, with_units=False)

    assert io.condensate_vars_missing_units(str(path), group="render_a") == [
        "qc", "qi",
    ]

    data = io.load_and_validate(
        str(path),
        liquid_water_group="render_a",
        ice_water_group="render_a",
        fallback_units="kg/kg",
    )
    # kg/kg -> g/kg is the same x1000 conversion a units attribute would get.
    np.testing.assert_allclose(data["liquid_water_data"].values, 1.0)
    np.testing.assert_allclose(data["ice_water_data"].values, 0.2)


def test_units_attribute_present_needs_no_fallback(tmp_path: Path):
    path = tmp_path / "nests.nc"
    _write_nested_render_file(path)

    assert io.condensate_vars_missing_units(str(path), group="render_b") == []

    data = io.load_and_validate(str(path), liquid_water_group="render_b")
    np.testing.assert_allclose(data["liquid_water_data"].values, 1.0)


# --- Nest detection: fineness is a per-axis relation -----------------------
#
# The ordinary atmospheric nest refines horizontally while SHARING its
# parent's vertical levels. Collapsing each grid to one scalar spacing made
# that pair tie on the vertical (the finest axis of both) and silently drop
# out of the group picker (docs/soar-bugs.md entry 5). Finer now means: no
# axis coarser, at least one axis strictly finer.

PARENT_COORDS = {
    "x": np.arange(0.0, 8001.0, 2000.0),        # dx 2000
    "y": np.arange(0.0, 8001.0, 2000.0),        # dy 2000
    "z": np.array([100.0, 300.0, 800.0, 1600.0]),  # min dz 200
}
# Refines x and y 2x, shares the parent's z levels exactly.
NEST_COORDS = {
    "x": np.arange(2000.0, 4001.0, 1000.0),
    "y": np.arange(2000.0, 4001.0, 1000.0),
    "z": PARENT_COORDS["z"],
}
# Finer y and z, but COARSER x — the scalar gate would rank it "finer"
# (its minimum spacing 100 beats the parent's 200); the per-axis gate must
# refuse it.
MIXED_COORDS = {
    "x": np.array([2000.0, 6000.0]),               # dx 4000: coarser
    "y": np.arange(2000.0, 4001.0, 1000.0),        # finer
    "z": np.array([100.0, 200.0, 300.0, 400.0]),   # dz 100: finer
}


def _write_refinement_file(path: Path, groups: dict) -> None:
    xr.Dataset().to_netcdf(path, mode="w")
    for name, coords in groups.items():
        shape = tuple(len(coords[a]) for a in ("x", "y", "z"))
        xr.Dataset(
            data_vars={
                "qc": (("x", "y", "z"),
                       np.full(shape, 0.001), {"units": "kg/kg"}),
            },
            coords={a: np.asarray(coords[a]) for a in ("x", "y", "z")},
        ).to_netcdf(path, mode="a", group=name)


def test_nest_sharing_vertical_levels_is_offered(tmp_path: Path):
    path = tmp_path / "shared_z.nc"
    _write_refinement_file(path, {"parent": PARENT_COORDS, "nest": NEST_COORDS})
    assert io.find_nestable_group_pairs(str(path)) == [("parent", "nest")]


def test_candidate_coarser_on_one_axis_is_refused(tmp_path: Path):
    path = tmp_path / "mixed.nc"
    _write_refinement_file(path, {"parent": PARENT_COORDS, "mixed": MIXED_COORDS})
    assert io.find_nestable_group_pairs(str(path)) == []


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_nestable_pairs_agree_with_browser(tmp_path: Path):
    """field.js must reach the same verdicts on the same grids."""
    repo = Path(__file__).resolve().parents[1]
    script = """
    import { domainExtent, nestablePairs } from %s;
    const chunks = [];
    for await (const c of process.stdin) chunks.push(c);
    const domains = JSON.parse(Buffer.concat(chunks).toString()).map((d) => {
      const { bmin, bmax, spacing } = domainExtent(d.x, d.y, d.z);
      return { name: d.name, bmin, bmax, spacing };
    });
    process.stdout.write(JSON.stringify(nestablePairs(domains)));
    """ % json.dumps(str(repo / "web" / "soar" / "field.js"))
    cases = {
        "shared_z": {"parent": PARENT_COORDS, "nest": NEST_COORDS},
        "mixed": {"parent": PARENT_COORDS, "mixed": MIXED_COORDS},
    }
    for label, groups in cases.items():
        path = tmp_path / f"{label}.nc"
        _write_refinement_file(path, groups)
        expected = [list(p) for p in io.find_nestable_group_pairs(str(path))]
        payload = json.dumps([
            {"name": n, **{a: list(c[a]) for a in ("x", "y", "z")}}
            for n, c in groups.items()])
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=payload, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"node failed:\n{proc.stderr[-2000:]}")
        assert json.loads(proc.stdout) == expected, label
