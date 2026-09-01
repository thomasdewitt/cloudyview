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


def test_fallback_units_convert_like_an_attribute_would(tmp_path: Path):
    path = tmp_path / "nests_no_units.nc"
    _write_nested_render_file(path, with_units=False)

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

    data = io.load_and_validate(str(path), liquid_water_group="render_b")
    np.testing.assert_allclose(data["liquid_water_data"].values, 1.0)


def test_explicit_units_beat_the_empty_string_heuristic(tmp_path: Path):
    """units='' plus --units kg/kg must convert, not assume SAM g/kg.

    The empty attribute says nothing; letting it outrank an explicit answer
    loaded such files 1000x too thin.
    """
    path = tmp_path / "empty_units.nc"
    xr.Dataset(
        data_vars={"qc": (("x", "y", "z"),
                          np.full((2, 3, 4), 0.001), {"units": ""})},
        coords={"x": ("x", [0.0, 1000.0], {"units": "m"}),
                "y": ("y", [0.0, 2000.0, 4000.0], {"units": "m"}),
                "z": ("z", [100.0, 300.0, 800.0, 1600.0], {"units": "m"})},
    ).to_netcdf(path)

    with_flag = io.load_and_validate(str(path), fallback_units="kg/kg")
    np.testing.assert_allclose(with_flag["liquid_water_data"].values, 1.0)
    # No flag: the SAM empty-string convention still stands, unconverted.
    without = io.load_and_validate(str(path))
    np.testing.assert_allclose(without["liquid_water_data"].values, 0.001)


def _write_pair(tmp_path: Path, ice_coords: dict, ice_coord_attrs=None,
                ice_units="g/kg"):
    """A liquid file and a split ice file on (z, y, x), 2x3x4 voxels."""
    lw_coords = {"z": ("z", [100.0, 300.0], {"units": "m"}),
                 "y": ("y", [0.0, 2000.0, 4000.0], {"units": "m"}),
                 "x": ("x", [0.0, 1000.0, 2000.0, 3000.0], {"units": "m"})}
    lw_path = tmp_path / "liquid.nc"
    xr.Dataset(
        data_vars={"qc": (("z", "y", "x"), np.full((2, 3, 4), 0.1),
                          {"units": "g/kg"})},
        coords=lw_coords).to_netcdf(lw_path)

    attrs = {"units": ice_units} if ice_units is not None else {}
    ice_path = tmp_path / "ice.nc"
    xr.Dataset(
        data_vars={"qi": (("z", "y", "x"), np.full((2, 3, 4), 0.2), attrs)},
        coords={name: (name, values,
                       (ice_coord_attrs or {}).get(name, {"units": "m"}))
                for name, values in ice_coords.items()},
    ).to_netcdf(ice_path)
    return str(lw_path), str(ice_path)


def test_split_ice_matching_grid_loads(tmp_path: Path):
    lw, ice = _write_pair(tmp_path, {
        "z": [100.0, 300.0], "y": [0.0, 2000.0, 4000.0],
        "x": [0.0, 1000.0, 2000.0, 3000.0]})
    data = io.load_and_validate(lw, ice_filepath=ice)
    assert data["ice_water_var"] == "qi"
    np.testing.assert_allclose(data["ice_water_data"].values, 0.2)


def test_split_ice_mismatch_caught_even_when_another_axis_fails(tmp_path):
    """A missing z coordinate must not blind the x comparison.

    The old check extracted all three ice coordinates in one call inside a
    blanket except: any one axis failing threw the other two away and let a
    provably mismatched grid in by shape alone.
    """
    lw, ice = _write_pair(tmp_path, {
        # z coordinate absent entirely; x shifted off the liquid grid.
        "y": [0.0, 2000.0, 4000.0],
        "x": [500.0, 1500.0, 2500.0, 3500.0]})
    with pytest.raises(ValueError, match="different x-coordinate"):
        io.load_and_validate(lw, ice_filepath=ice)


def test_split_ice_missing_axis_is_recorded_not_silent(tmp_path: Path):
    lw, ice = _write_pair(tmp_path, {
        "y": [0.0, 2000.0, 4000.0],
        "x": [0.0, 1000.0, 2000.0, 3000.0]})
    data = io.load_and_validate(lw, ice_filepath=ice)
    assert any("no z coordinate" in a for a in data["assumptions"])


def test_split_ice_non_length_coordinate_refuses(tmp_path: Path):
    """Ice coordinates in degrees are a different grid, not a shape match."""
    lw, ice = _write_pair(
        tmp_path,
        {"z": [100.0, 300.0], "y": [0.0, 2000.0, 4000.0],
         "x": [0.0, 1000.0, 2000.0, 3000.0]},
        ice_coord_attrs={"x": {"units": "degrees_east"}})
    with pytest.raises(ValueError, match="unit of length|not a unit"):
        io.load_and_validate(lw, ice_filepath=ice)


def test_ice_units_flag_answers_the_ice_file(tmp_path: Path):
    """--ice-units mirrors --units, for the ice variable specifically."""
    lw, ice = _write_pair(
        tmp_path,
        {"z": [100.0, 300.0], "y": [0.0, 2000.0, 4000.0],
         "x": [0.0, 1000.0, 2000.0, 3000.0]},
        ice_units=None)
    with pytest.raises(ValueError, match="units"):
        io.load_and_validate(lw, ice_filepath=ice)
    data = io.load_and_validate(lw, ice_filepath=ice,
                                fallback_ice_units="kg/kg")
    np.testing.assert_allclose(data["ice_water_data"].values, 200.0)
    # The liquid keeps its own attribute: --ice-units touches ice only.
    np.testing.assert_allclose(data["liquid_water_data"].values, 0.1)


def test_ice_units_attribute_beats_the_flag(tmp_path: Path):
    lw, ice = _write_pair(
        tmp_path,
        {"z": [100.0, 300.0], "y": [0.0, 2000.0, 4000.0],
         "x": [0.0, 1000.0, 2000.0, 3000.0]},
        ice_units="g/kg")
    data = io.load_and_validate(lw, ice_filepath=ice,
                                fallback_ice_units="kg/kg")
    np.testing.assert_allclose(data["ice_water_data"].values, 0.2)


def test_same_file_ice_may_use_its_own_dimension_names(tmp_path: Path):
    """qc(zt, yt, xt) + qi(z, y, x) on one logical grid must load."""
    path = tmp_path / "two_spellings.nc"
    xr.Dataset(
        data_vars={
            "qc": (("zt", "yt", "xt"), np.full((2, 3, 4), 0.1),
                   {"units": "g/kg"}),
            "qi": (("z", "y", "x"), np.full((2, 3, 4), 0.2),
                   {"units": "g/kg"}),
        },
        coords={
            "zt": ("zt", [100.0, 300.0], {"units": "m"}),
            "yt": ("yt", [0.0, 2000.0, 4000.0], {"units": "m"}),
            "xt": ("xt", [0.0, 1000.0, 2000.0, 3000.0], {"units": "m"}),
            "z": ("z", [100.0, 300.0], {"units": "m"}),
            "y": ("y", [0.0, 2000.0, 4000.0], {"units": "m"}),
            "x": ("x", [0.0, 1000.0, 2000.0, 3000.0], {"units": "m"}),
        },
    ).to_netcdf(path)
    data = io.load_and_validate(str(path))
    assert data["ice_water_var"] == "qi"
    assert data["ice_water_data"].shape == data["liquid_water_data"].shape
    assert any("independently" in a for a in data["assumptions"])


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


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_nestable_pairs_per_axis_fineness_gate(tmp_path: Path):
    """field.js offers the shared-z nest and refuses the mixed one."""
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
        "shared_z": ({"parent": PARENT_COORDS, "nest": NEST_COORDS},
                     [["parent", "nest"]]),
        "mixed": ({"parent": PARENT_COORDS, "mixed": MIXED_COORDS}, []),
    }
    for label, (groups, expected) in cases.items():
        payload = json.dumps([
            {"name": n, **{a: list(c[a]) for a in ("x", "y", "z")}}
            for n, c in groups.items()])
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            input=payload, capture_output=True, text=True)
        if proc.returncode != 0:
            pytest.fail(f"node failed:\n{proc.stderr[-2000:]}")
        assert json.loads(proc.stdout) == expected, label
