"""How a file becomes a field: io.load_and_validate and cv.load.

Four groups, each under the docstring of the file it was merged from: the
condensate value contract; explicit variable, group, coordinate and unit
overrides (and nest detection); the dimension and coordinate resolution
ladder the CLI shares with the browser; and a stated grid spacing for an
axis with no coordinate in meters.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cloudyview import io
from cloudyview.field import CloudField, load as load_field
from cloudyview.witness import crop_empty_z
from tests.conftest import REPO


# =============================================================================
# Merged from test_condensate_values
#
# The condensate value contract: fills, NaNs, negatives, and grid order.
#
# Fill values (_FillValue/missing_value) decode to NaN and mean cloud-free
# air, so they load as zero condensate with a stated assumption. Any other
# non-finite value refuses the load — it would otherwise propagate into
# glimpse's tau and the witness/behold textures and render as silent holes.
# Small negative condensate is normal LES numerics and clamps to zero,
# stated. Coordinates must end up strictly ascending or the voxels have no
# defined place in space.
# =============================================================================


def _condensate_write(path, values, encoding=None):
    ds = xr.Dataset(
        data_vars={"qc": (("z", "y", "x"), values, {"units": "g/kg"})},
        coords={"z": ("z", [100.0, 300.0], {"units": "m"}),
                "y": ("y", [0.0, 2000.0, 4000.0], {"units": "m"}),
                "x": ("x", [0.0, 1000.0, 2000.0, 3000.0], {"units": "m"})})
    ds.to_netcdf(path, encoding=encoding or {})
    return str(path)


def test_fill_halo_loads_as_cloud_free(tmp_path):
    values = np.full((2, 3, 4), 0.1)
    values[0, 0, :] = np.nan   # the halo the fill marks
    path = _condensate_write(tmp_path / "halo.nc", values,
                  encoding={"qc": {"_FillValue": -999.0}})
    r = io.load_and_validate(path)
    out = r["liquid_water_data"].values
    assert np.isfinite(out).all()
    assert float(out.max()) == pytest.approx(0.1)
    # (x, y, z) order after standardization; the halo was z=0, y=0, all x.
    np.testing.assert_allclose(out[:, 0, 0], 0.0)
    assert any("cloud-free" in a for a in r["assumptions"])
    # And the whole load survives to a CloudField.
    field = load_field(path)
    assert np.isfinite(field.lwc).all()


def test_nan_without_declared_fill_refuses(tmp_path):
    values = np.full((2, 3, 4), 0.1)
    values[1, 2, 3] = np.nan
    # _FillValue: None suppresses xarray's default fill attribute, so the
    # NaN reaches the reader with nothing to explain it.
    path = _condensate_write(tmp_path / "corrupt.nc", values,
                  encoding={"qc": {"_FillValue": None}})
    with pytest.raises(ValueError, match="qc.*non-finite"):
        io.load_and_validate(path)


def test_negative_condensate_clamps_and_is_recorded(tmp_path):
    values = np.full((2, 3, 4), 0.1)
    values[0, 1, 2] = -1e-9    # ordinary LES advection undershoot
    path = _condensate_write(tmp_path / "neg.nc", values)
    r = io.load_and_validate(path)
    out = r["liquid_water_data"].values
    assert float(out.min()) == 0.0
    assert any("negative" in a.lower() for a in r["assumptions"])


def _cube(n=2):
    return np.full((n, n, n), 0.1, dtype=np.float32)


def test_cloudfield_refuses_non_finite_condensate():
    bad = _cube()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        CloudField(lwc=bad, x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0])


def test_cloudfield_clamps_negatives(caplog):
    field = CloudField(lwc=[[[0.1, -1e-8], [0.1, 0.1]], [[0.1, 0.1], [0.1, 0.1]]],
                       x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0])
    assert float(field.lwc.min()) == 0.0
    assert any("negative" in r.message.lower() for r in caplog.records)


def test_cloudfield_refuses_folded_coordinates():
    with pytest.raises(ValueError, match="y coordinate"):
        CloudField(lwc=np.full((2, 3, 2), 0.1),
                   x=[0.0, 1.0], y=[0.0, 2.0, 1.0], z=[0.0, 1.0])


def test_cloudfield_refuses_repeated_coordinates():
    with pytest.raises(ValueError, match="x coordinate"):
        CloudField(lwc=_cube(),
                   x=[1.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0])


def test_cloudfield_refuses_non_finite_coordinates():
    with pytest.raises(ValueError, match="z coordinate"):
        CloudField(lwc=_cube(),
                   x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, np.nan])


def test_cloudfield_still_flips_descending_coordinates():
    lwc = np.zeros((2, 2, 2), dtype=np.float32)
    lwc[0, 0, 0] = 1.0
    field = CloudField(lwc=lwc, x=[0.0, 1.0], y=[0.0, 1.0], z=[100.0, 0.0])
    assert field.z[0] == 0.0 and field.z[1] == 100.0
    assert field.lwc[0, 0, 1] == 1.0


# --- z-crop widening at the domain boundaries -------------------------------

def _column(n, occupied_plane):
    sigma = np.zeros((2, 2, n))
    sigma[:, :, occupied_plane] = 1e-3
    z = np.arange(n, dtype=np.float64) * 100.0
    return sigma, z


@pytest.mark.parametrize("plane,expected", [
    (0, (0, 1)),      # bottom: widen upward, as before
    (2, (2, 3)),      # interior: widen upward
    (4, (3, 4)),      # top: nowhere up to go — widen DOWNWARD, not crash
])
def test_crop_single_occupied_plane_always_keeps_two(plane, expected):
    sigma, z = _column(5, plane)
    cropped, z_crop, (lo, hi) = crop_empty_z(sigma, z)
    assert (lo, hi) == expected
    assert cropped.shape[2] == 2 and z_crop.size == 2


# =============================================================================
# Merged from test_io_overrides
#
# Tests for explicit variable/group/coordinate override handling.
# =============================================================================


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
    script = """
    import { domainExtent, nestablePairs } from %s;
    const chunks = [];
    for await (const c of process.stdin) chunks.push(c);
    const domains = JSON.parse(Buffer.concat(chunks).toString()).map((d) => {
      const { bmin, bmax, spacing } = domainExtent(d.x, d.y, d.z);
      return { name: d.name, bmin, bmax, spacing };
    });
    process.stdout.write(JSON.stringify(nestablePairs(domains)));
    """ % json.dumps(str(REPO / "web" / "soar" / "field.js"))
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


# =============================================================================
# Merged from test_io_resolution_ladder
#
# The CLI loader resolves what the browser resolves.
#
# io.py's dimension/coordinate rules are a port of web/soar/ingest/netcdf.js
# (the reference copy, tested in tests/app/test_soar_ingest_dims.py). These tests pin the
# Python side on the same cases: the SAM/DALES dimension spellings, the
# coordinate-metadata rules, the stated positional last resort, the UM
# not-a-length override, kilometre conversion, timestep selection, and the
# refusals that replaced silent nonsense (ICON's degree/level-index
# coordinates; QN inferred as liquid).
# =============================================================================


def _ladder_write(path, dims, coords, name="qc", data_units="g/kg",
                  attrs=None):
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
    path = _ladder_write(
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
    path = _ladder_write(
        tmp_path / "attrs.nc", ("celo", "cela", "celi"),
        {"celo": (np.arange(4) * 100.0, {"units": "m", "axis": "Z"}, "celo"),
         "cela": (np.arange(3) * 50.0, {"units": "m", "axis": "Y"}, "cela"),
         "celi": (np.arange(2) * 50.0, {"units": "m", "axis": "X"}, "celi")})
    r = io.load_and_validate(path)
    assert r["dims"] == {"x": "celi", "y": "cela", "z": "celo"}
    assert any("axis attribute" in a for a in r["assumptions"])


def test_position_is_last_resort_and_stated(tmp_path, caplog):
    """Nothing named, nothing declared: (slow, mid, fast) = (z, y, x)."""
    path = _ladder_write(
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
    path = _ladder_write(
        tmp_path / "icon.nc", ("height", "lat", "lon"),
        {"height": (np.arange(4) * 1.0, {}, "height"),
         "lat": (np.linspace(-10, 10, 3), {"units": "degrees_north"}, "lat"),
         "lon": (np.linspace(-180, 179, 2), {"units": "degrees_east"}, "lon")},
        name="clw")
    with pytest.raises(ValueError, match="unit of length"):
        io.load_and_validate(path)


def test_um_dimensionless_vertical_loses_to_the_length(tmp_path):
    """The 0.99 m pancake: eta (0..1) beaten by the metres beside it."""
    path = _ladder_write(
        tmp_path / "um.nc", ("eta", "y", "x"),
        {"eta": (np.linspace(0, 1, 4), {"units": "1"}, "eta"),
         "zsea": (np.linspace(20, 33000, 4), {"units": "m"}, "eta"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["coord_names"]["z"] == "zsea"
    assert r["z_coord"].max() == pytest.approx(33000.0)


def test_kilometre_coordinates_arrive_in_metres(tmp_path):
    path = _ladder_write(
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
    path = _ladder_write(
        tmp_path / "agl.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "m AGL"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "meters above sea level"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["z_coord"].max() == pytest.approx(300.0)
    assert r["y_coord"].max() == pytest.approx(100.0)


def test_centimetre_coordinates_convert(tmp_path):
    path = _ladder_write(
        tmp_path / "cm.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "cm"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path)
    assert r["z_coord"].max() == pytest.approx(3.0)


def test_unrecognized_units_refuse_and_name_the_flag(tmp_path):
    path = _ladder_write(
        tmp_path / "odd.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "furlongs"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    with pytest.raises(ValueError, match=r"furlongs.*--coord-units") :
        io.load_and_validate(path)


def test_coord_units_fallback_answers_unrecognized_strings(tmp_path):
    """--coord-units applies to unrecognized units and is recorded."""
    path = _ladder_write(
        tmp_path / "odd_km.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 0.1, {"units": "km_agl"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")})
    r = io.load_and_validate(path, fallback_coord_units="km")
    assert r["z_coord"].max() == pytest.approx(300.0)
    assert any("--coord-units" in a for a in r["assumptions"])


def test_coord_units_fallback_cannot_rescue_degrees(tmp_path):
    """A declared non-length is refused whatever the caller asserts."""
    path = _ladder_write(
        tmp_path / "deg.nc", ("height", "lat", "lon"),
        {"height": (np.arange(4) * 1.0, {}, "height"),
         "lat": (np.linspace(-10, 10, 3), {"units": "degrees_north"}, "lat"),
         "lon": (np.linspace(-180, 179, 2), {"units": "degrees_east"}, "lon")},
        name="clw")
    with pytest.raises(ValueError, match="unit of length"):
        io.load_and_validate(path, fallback_coord_units="m")


def test_missing_units_still_assume_metres(tmp_path):
    path = _ladder_write(
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
    no_time = _ladder_write(tmp_path / "notime.nc", ("z", "y", "x"), coords)
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
    path = _ladder_write(
        tmp_path / "qn.nc", ("z", "y", "x"),
        {"z": (np.arange(4) * 100.0, {"units": "m"}, "z"),
         "y": (np.arange(3) * 50.0, {"units": "m"}, "y"),
         "x": (np.arange(2) * 50.0, {"units": "m"}, "x")},
        name="QN")
    with pytest.raises(ValueError, match="QN"):
        io.load_and_validate(path)
    assert io.load_and_validate(path, liquid_water_var="QN")[
        "liquid_water_var"] == "QN"


# =============================================================================
# Merged from test_grid_spacing
#
# A stated grid spacing is the only way past "no coordinate in meters".
#
# Both loaders refuse an axis with no coordinate in a unit of length — none at
# all (index-only output), or degrees (a regridded lat-lon file), or a level
# index — because using the numbers anyway renders a plausible box with
# nonsense proportions (ICON RCEMIP, 2026-08-31). The override (2026-09-23) is
# the person stating the spacing of a REGULAR grid: --dx/--dy/--dz in the CLI,
# the grid-spacing question in the browser. Its rules, pinned here on
# synthetic metric-less files, in both languages:
#
# - never guessed: without the answer the refusal stands, and names the flag;
# - only where the refusal fires: a spacing for an axis with meters refuses;
# - the vertical order goes with a vertical spacing, always asked, because a
#   level index does not say which end is the ground (ICON counts down);
# - uniform, and said out loud;
# - native unstructured output (height x ncells) still refuses on its
#   dimensionality, and the refusal does not suggest the override.
# =============================================================================


NETCDF_JS = REPO / "web" / "soar" / "ingest" / "netcdf.js"

NX, NY, NZ = 5, 6, 4


def _field():
    # Distinct per level so an order can be checked: level k holds k + 1.
    qc = np.zeros((NZ, NY, NX), dtype=np.float32)
    for k in range(NZ):
        qc[k] = k + 1.0
    return qc


def _spacing_write(path, coords=None, dims=("lev", "lat", "lon")):
    ds = xr.Dataset({"qc": (dims, _field(), {"units": "g/kg"})},
                    coords=coords or {})
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def index_only(tmp_path):
    """No coordinate variables at all: dimension names and nothing else."""
    return _spacing_write(tmp_path / "index_only.nc")


@pytest.fixture
def degrees(tmp_path):
    """A regridded lat-lon file: degrees on x/y, a level index on z."""
    return _spacing_write(tmp_path / "degrees.nc", coords={
        "lon": ("lon", np.linspace(0, 0.4, NX), {"units": "degrees_east"}),
        "lat": ("lat", np.linspace(0, 0.5, NY), {"units": "degrees_north"}),
        "lev": ("lev", np.arange(1, NZ + 1, dtype=float), {"units": "1"}),
    })


def _load(path, **kw):
    return io.load_and_validate(path, **kw)


def test_the_refusal_stands_and_names_the_flags(index_only, degrees):
    with pytest.raises(io.NoLengthCoordinate, match="--dx, --dy, --dz"):
        _load(index_only)
    with pytest.raises(io.NoLengthCoordinate, match="--dx"):
        _load(degrees)


def test_a_stated_spacing_places_the_cells(index_only):
    r = _load(index_only, grid_spacing={"x": 100.0, "y": 200.0, "z": 50.0},
              z_first="bottom")
    np.testing.assert_allclose(r["x_coord"], (np.arange(NX) + 0.5) * 100.0)
    np.testing.assert_allclose(r["y_coord"], (np.arange(NY) + 0.5) * 200.0)
    np.testing.assert_allclose(r["z_coord"], (np.arange(NZ) + 0.5) * 50.0)


def test_degrees_and_a_level_index_are_rescued_by_the_same_answer(degrees):
    r = _load(degrees, grid_spacing={"x": 90.0, "y": 90.0, "z": 40.0},
              z_first="bottom")
    assert r["x_coord"][0] == 45.0 and r["z_coord"][-1] == 140.0


def test_top_first_turns_the_field_the_right_way_up(tmp_path, index_only):
    from cloudyview.field import load
    up = load(index_only, grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0},
              z_first="bottom")
    down = load(index_only, grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0},
                z_first="top")
    # Level k holds k + 1. Bottom-first keeps that order upward; top-first
    # means the file's first level is the top, so it lands at the top.
    col_up = up.lwc[0, 0, :]
    col_down = down.lwc[0, 0, :]
    np.testing.assert_array_equal(col_up, [1, 2, 3, 4])
    np.testing.assert_array_equal(col_down, [4, 3, 2, 1])
    np.testing.assert_allclose(up.z, down.z)


def test_the_vertical_order_is_always_asked(index_only):
    with pytest.raises(ValueError, match="--z-first"):
        _load(index_only, grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0})
    with pytest.raises(ValueError, match="--z-first only goes with --dz"):
        _load(index_only, grid_spacing={"x": 1.0, "y": 1.0}, z_first="top")


def test_a_spacing_never_replaces_real_meters(tmp_path):
    path = _spacing_write(tmp_path / "x_in_m.nc", coords={
        "lon": ("lon", (np.arange(NX) + 0.5) * 100.0, {"units": "m"}),
    })
    with pytest.raises(ValueError, match="already has meters in 'lon'"):
        _load(path, grid_spacing={"x": 100.0, "y": 1.0, "z": 1.0},
              z_first="bottom")


def test_a_spacing_and_a_named_coordinate_conflict(degrees):
    with pytest.raises(ValueError, match="--dx and --x-coord"):
        _load(degrees, x_coord_name="lon",
              grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0}, z_first="bottom")


def test_the_spacing_must_be_a_positive_length(index_only):
    with pytest.raises(ValueError, match="positive number of meters"):
        _load(index_only, grid_spacing={"x": 0.0, "y": 1.0, "z": 1.0},
              z_first="bottom")


def test_native_unstructured_output_still_refuses(tmp_path):
    """ICON's native grid: (time, height, ncells). The override is for a
    regular grid that lacks a metric, and must not look like a rescue."""
    ds = xr.Dataset({"qc": (("time", "height", "ncells"),
                            np.ones((1, NZ, 40), np.float32),
                            {"units": "g/kg"})})
    path = str(tmp_path / "icon_native.nc")
    ds.to_netcdf(path)
    for kw in ({}, {"grid_spacing": {"x": 1.0, "y": 1.0, "z": 1.0},
                    "z_first": "top"}):
        with pytest.raises(ValueError, match="regular grid") as err:
            _load(path, **kw)
        assert "Unstructured output" in str(err.value)


def test_the_cli_carries_the_answer(monkeypatch):
    witness = __import__("importlib").import_module("cloudyview.witness")
    captured = {}
    monkeypatch.setattr(witness, "main",
                        lambda *a, **k: captured.update(k))
    monkeypatch.setattr(sys, "argv", [
        "witness", "f.nc", "--dx", "100", "--dy", "100", "--dz", "50",
        "--z-first", "top"])
    witness.cli()
    assert captured["grid_spacing"] == {"x": 100.0, "y": 100.0, "z": 50.0}
    assert captured["z_first"] == "top"
    monkeypatch.setattr(sys, "argv", ["witness", "f.nc"])
    witness.cli()
    assert captured["grid_spacing"] is None and captured["z_first"] is None


# --- the browser's half: the same rules in netcdf.js -------------------------

_JS = textwrap.dedent("""
    import { describeGroup } from "%s";
    const dataset = (shape, dims, attrs = {}) => ({
      shape, attrs,
      get_attached_scales: (axis) => (dims[axis] ? [dims[axis]] : []),
      metadata: { chunks: null }, filters: [], dtype: "<f4",
    });
    const coord = (values, attrs = {}) => ({
      shape: [values.length], attrs, value: Float64Array.from(values),
      get_attached_scales: () => [],
    });
    const group = (entries) => ({
      keys: () => Object.keys(entries), get: (k) => entries[k],
      type: "Group",
    });
    const dims = ["lev", "lat", "lon"];
    const qc = dataset([%d, %d, %d], dims, { units: "g/kg" });
    const indexOnly = group({ qc });
    const degrees = group({
      qc,
      lon: coord([0, 0.1, 0.2, 0.3, 0.4], { units: "degrees_east" }),
      lat: coord([0, 0.1, 0.2, 0.3, 0.4, 0.5], { units: "degrees_north" }),
      lev: coord([1, 2, 3, 4], { units: "1" }),
    });
    const xInM = group({ qc, lon: coord([50, 150, 250, 350, 450],
                                        { units: "m" }) });
    const attempt = (fn) => {
      try { return { ok: true, value: fn() }; }
      catch (err) { return { ok: false, message: String(err.message) }; }
    };
    const pick = (d) => ({
      needs: d.needsGridSpacing, needed: d.gridSpacingNeeded,
      coords: d.coords, coordNames: d.coordNames,
      assumptions: d.assumptions,
    });
    const all = { gridSpacing: { x: 100, y: 200, z: 50 } };
    process.stdout.write(JSON.stringify({
      asks: attempt(() => pick(describeGroup(indexOnly, "", null))),
      asksDegrees: attempt(() => pick(describeGroup(degrees, "", null))),
      placed: attempt(() => pick(describeGroup(
        indexOnly, "", { ...all, zFirst: "bottom" }))),
      topFirst: attempt(() => pick(describeGroup(
        degrees, "", { ...all, zFirst: "top" }))),
      noZFirst: attempt(() => pick(describeGroup(indexOnly, "", all))),
      hasMeters: attempt(() => pick(describeGroup(
        xInM, "", { ...all, zFirst: "bottom" }))),
      nonPositive: attempt(() => pick(describeGroup(
        indexOnly, "", { gridSpacing: { x: -1, y: 1, z: 1 },
                         zFirst: "top" }))),
    }));
""") % (NETCDF_JS.as_posix(), NZ, NY, NX)


@pytest.fixture(scope="module")
def browser():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    proc = subprocess.run(["node", "--input-type=module", "-e", _JS],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_browser_asks_instead_of_refusing(browser):
    for case in ("asks", "asksDegrees"):
        d = browser[case]
        assert d["ok"], d
        assert d["value"]["needs"] is True
        assert d["value"]["coords"] is None      # no box: not nestable yet
        assert [a["axis"] for a in d["value"]["needed"]] == ["x", "y", "z"]
    found = {a["axis"]: a["found"] for a in browser["asksDegrees"]["value"]["needed"]}
    assert "degrees_east" in found["x"] and "(1)" in found["z"]


def test_browser_places_the_cells_as_python_does(browser):
    v = browser["placed"]["value"]
    assert v["needs"] is False
    np.testing.assert_allclose(v["coords"]["x"], (np.arange(NX) + 0.5) * 100)
    np.testing.assert_allclose(v["coords"]["y"], (np.arange(NY) + 0.5) * 200)
    np.testing.assert_allclose(v["coords"]["z"], (np.arange(NZ) + 0.5) * 50)
    assert v["coordNames"] == {"x": None, "y": None, "z": None}
    assert any("as entered" in a and "uniform" in a for a in v["assumptions"])
    # Top-first is written descending; the read's own flip rights it.
    z = browser["topFirst"]["value"]["coords"]["z"]
    np.testing.assert_allclose(z, ((np.arange(NZ) + 0.5) * 50)[::-1])


def test_browser_keeps_the_same_rules(browser):
    assert not browser["noZFirst"]["ok"]
    assert "first level" in browser["noZFirst"]["message"]
    assert not browser["hasMeters"]["ok"]
    assert "already has meters in 'lon'" in browser["hasMeters"]["message"]
    assert not browser["nonPositive"]["ok"]
    assert "positive number of meters" in browser["nonPositive"]["message"]
