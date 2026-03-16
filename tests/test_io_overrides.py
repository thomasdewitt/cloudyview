"""Tests for explicit variable/group/coordinate override handling."""

from pathlib import Path

import numpy as np
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
