"""The CLI loader resolves what the browser resolves.

io.py's dimension/coordinate rules are a port of web/soar/ingest/netcdf.js
(the reference copy, tested in test_soar_ingest_dims.py). These tests pin the
Python side on the same cases: the SAM/DALES dimension spellings, the
coordinate-metadata rules, the stated positional last resort, the UM
not-a-length override, kilometre conversion, timestep selection, and the
refusals that replaced silent nonsense (ICON's degree/level-index
coordinates; QN inferred as liquid).
"""

import logging

import numpy as np
import pytest
import xarray as xr

from cloudyview import io


def _write(path, dims, coords, name="qc", data_units="g/kg", attrs=None):
    """A tiny file: one 3-D variable on `dims`, 1-D coords from `coords`."""
    shape = tuple(len(coords[d][0]) if d in coords else 2 for d in dims)
    ds = xr.Dataset(
        data_vars={name: (dims, np.full(shape, 0.1), {"units": data_units})},
        coords={cname: (dim, values, cattrs)
                for cname, (values, cattrs, dim) in coords.items()},
    )
    if attrs:
        for var, a in attrs.items():
            ds[var].attrs.update(a)
    ds.to_netcdf(path)
    return str(path)


def test_sam_dales_spellings_resolve(tmp_path):
    """(zt, yt, xt) — the DALES file that used to die on inference."""
    path = _write(
        tmp_path / "dales.nc", ("zt", "yt", "xt"),
        {"zt": (np.arange(4) * 100.0, {"units": "m"}, "zt"),
         "yt": (np.arange(3) * 50.0, {"units": "m"}, "yt"),
         "xt": (np.arange(2) * 50.0, {"units": "m"}, "xt")},
        name="clw")
    r = io.load_and_validate(path)
    assert r["dims"] == {"x": "xt", "y": "yt", "z": "zt"}
    assert r["liquid_water_data"].shape == (2, 3, 4)
    assert r["assumptions"] == []


def test_metadata_axis_attribute_settles_unknown_names(tmp_path):
    path = _write(
        tmp_path / "attrs.nc", ("celo", "cela", "celi"),
        {"celo": (np.arange(4) * 100.0, {"units": "m", "axis": "Z"}, "celo"),
         "cela": (np.arange(3) * 50.0, {"units": "m", "axis": "Y"}, "cela"),
         "celi": (np.arange(2) * 50.0, {"units": "m", "axis": "X"}, "celi")})
    r = io.load_and_validate(path)
    assert r["dims"] == {"x": "celi", "y": "cela", "z": "celo"}
    assert any("axis attribute" in a for a in r["assumptions"])


def test_position_is_last_resort_and_stated(tmp_path, caplog):
    """Nothing named, nothing declared: (slow, mid, fast) = (z, y, x)."""
    path = _write(
        tmp_path / "bare.nc", ("i", "j", "k"),
        {"i": (np.arange(4) * 100.0, {}, "i"),
         "j": (np.arange(3) * 50.0, {}, "j"),
         "k": (np.arange(2) * 50.0, {}, "k")})
    with caplog.at_level(logging.WARNING, logger="cloudyview.io"):
        r = io.load_and_validate(path)
    assert r["dims"] == {"z": "i", "y": "j", "x": "k"}
    assert any("storage position" in a for a in r["assumptions"])
    assert any("storage position" in m for m in caplog.messages)


def test_icon_degree_and_index_coordinates_are_refused(tmp_path):
    """Declared non-length units cannot place a field in space."""
    path = _write(
        tmp_path / "icon.nc", ("height", "lat", "lon"),
        {"height": (np.arange(4) * 1.0, {}, "height"),
         "lat": (np.linspace(-10, 10, 3), {"units": "degrees_north"}, "lat"),
         "lon": (np.linspace(-180, 179, 2), {"units": "degrees_east"}, "lon")},
        name="clw")
    with pytest.raises(ValueError, match="unit of length"):
        io.load_and_validate(path)


def test_um_dimensionless_vertical_loses_to_the_length(tmp_path):
    """The 0.99 m pancake: eta (0..1) beaten by the metres beside it."""
    path = _write(
        tmp_path / "um.nc", ("eta", "y", "x"),
        {"eta": (np.linspace(0, 1, 4), {"units": "1"}, "eta"),
         "zsea": (np.linspace(20, 33000, 4), {"units": "m"}, "eta"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["coord_names"]["z"] == "zsea"
    assert r["z_coord"].max() == pytest.approx(33000.0)


def test_kilometre_coordinates_arrive_in_metres(tmp_path):
    path = _write(
        tmp_path / "km.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 0.1, {"units": "km"}, "z"),
         "y": (np.arange(3) * 0.05, {"units": "km"}, "y"),
         "x": (np.arange(2) * 0.05, {"units": "km"}, "x")})
    r = io.load_and_validate(path)
    assert r["z_coord"].max() == pytest.approx(300.0)
    assert r["x_coord"].max() == pytest.approx(50.0)


def _write_stepped(path):
    ds = xr.Dataset(
        data_vars={"qc": (("time", "z", "y", "x"),
                          np.arange(4 * 2 * 2 * 2, dtype=float)
                          .reshape(4, 2, 2, 2),
                          {"units": "g/kg"})},
        coords={"z": ("z", [0.0, 100.0], {"units": "m"}),
                "y": ("y", [0.0, 50.0], {"units": "m"}),
                "x": ("x", [0.0, 50.0], {"units": "m"})})
    ds.to_netcdf(path)
    return str(path)


def test_multi_timestep_needs_the_flag(tmp_path):
    path = _write_stepped(tmp_path / "steps.nc")
    with pytest.raises(ValueError, match="--timestep"):
        io.load_and_validate(path)


def test_timestep_selects_the_step(tmp_path):
    path = _write_stepped(tmp_path / "steps.nc")
    r = io.load_and_validate(path, timestep=2)
    # Step 2 of the arange: values 16..23, transposed to (x, y, z).
    assert float(r["liquid_water_data"].min()) == 16.0
    with pytest.raises(ValueError, match="out of range"):
        io.load_and_validate(path, timestep=9)


def test_no_ice_skips_an_inferrable_ice_variable(tmp_path):
    path = tmp_path / "iced.nc"
    coords = {"z": ("z", [0.0, 100.0], {"units": "m"}),
              "y": ("y", [0.0, 50.0], {"units": "m"}),
              "x": ("x", [0.0, 50.0], {"units": "m"})}
    xr.Dataset(
        data_vars={"qc": (("z", "y", "x"), np.full((2, 2, 2), 0.1),
                          {"units": "g/kg"}),
                   "qi": (("z", "y", "x"), np.full((2, 2, 2), 0.2),
                          {"units": "g/kg"})},
        coords=coords).to_netcdf(path)
    assert io.load_and_validate(str(path))["ice_water_var"] == "qi"
    r = io.load_and_validate(str(path), no_ice=True)
    assert r["ice_water_var"] is None and r["ice_water_data"] is None


def test_qualified_metre_spellings_load(tmp_path):
    """'m AGL' and 'meters above sea level' are metres with a datum note."""
    path = _write(
        tmp_path / "agl.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "m AGL"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "meters above sea level"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["z_coord"].max() == pytest.approx(300.0)
    assert r["y_coord"].max() == pytest.approx(100.0)


def test_centimetre_coordinates_convert(tmp_path):
    path = _write(
        tmp_path / "cm.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "cm"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["z_coord"].max() == pytest.approx(3.0)


def test_unrecognized_units_refuse_and_name_the_flag(tmp_path):
    path = _write(
        tmp_path / "odd.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "furlongs"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    with pytest.raises(ValueError, match=r"furlongs.*--coord-units") :
        io.load_and_validate(path)


def test_coord_units_fallback_answers_unrecognized_strings(tmp_path):
    """--coord-units applies to unrecognized units and is recorded."""
    path = _write(
        tmp_path / "odd_km.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 0.1, {"units": "km_agl"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path, fallback_coord_units="km")
    assert r["z_coord"].max() == pytest.approx(300.0)
    assert any("--coord-units" in a for a in r["assumptions"])


def test_coord_units_fallback_cannot_rescue_degrees(tmp_path):
    """A declared non-length is refused whatever the caller asserts."""
    path = _write(
        tmp_path / "deg.nc", ("height", "lat", "lon"),
        {"height": (np.arange(4) * 1.0, {}, "height"),
         "lat": (np.linspace(-10, 10, 3), {"units": "degrees_north"}, "lat"),
         "lon": (np.linspace(-180, 179, 2), {"units": "degrees_east"}, "lon")},
        name="clw")
    with pytest.raises(ValueError, match="unit of length"):
        io.load_and_validate(path, fallback_coord_units="m")


def test_missing_units_still_assume_metres(tmp_path):
    path = _write(
        tmp_path / "bare_units.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["z_coord"].max() == pytest.approx(300.0)
    assert any("assumed meters" in a for a in r["assumptions"])


def test_timestep_on_a_single_step_file_is_refused(tmp_path):
    """--timestep N must not be silently ignored when only step 0 exists."""
    coords = {"z": (np.arange(4) * 100.0, {"units": "m"}, "z"),
              "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
              "x": (np.arange(2) * 50.0, {"units": "m"}, "x")}
    no_time = _write(tmp_path / "notime.nc", ("z", "y", "x"), coords)
    with pytest.raises(ValueError, match="no time dimension"):
        io.load_and_validate(no_time, timestep=2)
    assert io.load_and_validate(no_time, timestep=0)["liquid_water_var"] == "qc"

    one_step = tmp_path / "onestep.nc"
    xr.Dataset(
        data_vars={"qc": (("time", "z", "y", "x"),
                          np.full((1, 4, 3, 2), 0.1), {"units": "g/kg"})},
        coords={"z": ("z", np.arange(4) * 100.0, {"units": "m"}),
                "y": ("y", np.arange(3) * 50.0, {"units": "m"}),
                "x": ("x", np.arange(2) * 50.0, {"units": "m"})},
    ).to_netcdf(one_step)
    with pytest.raises(ValueError, match="only one step"):
        io.load_and_validate(str(one_step), timestep=1)
    assert io.load_and_validate(str(one_step), timestep=0)


def test_qn_is_not_inferred_as_liquid(tmp_path):
    """SAM's QN is total condensate; it must be chosen, never inferred."""
    path = _write(
        tmp_path / "qn.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "m"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")},
        name="QN")
    with pytest.raises(ValueError, match="QN"):
        io.load_and_validate(path)
    assert io.load_and_validate(path, liquid_water_var="QN")[
        "liquid_water_var"] == "QN"
