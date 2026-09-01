"""The condensate value contract: fills, NaNs, negatives, and grid order.

Fill values (_FillValue/missing_value) decode to NaN and mean cloud-free
air, so they load as zero condensate with a stated assumption. Any other
non-finite value refuses the load — it would otherwise propagate into
glimpse's tau and the witness/behold textures and render as silent holes.
Small negative condensate is normal LES numerics and clamps to zero,
stated. Coordinates must end up strictly ascending or the voxels have no
defined place in space.
"""

import numpy as np
import pytest
import xarray as xr

from cloudyview import io
from cloudyview.cloudfield import CloudField, load as load_field
from cloudyview.witness import crop_empty_z


def _write(path, values, encoding=None):
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
    path = _write(tmp_path / "halo.nc", values,
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
    path = _write(tmp_path / "corrupt.nc", values,
                  encoding={"qc": {"_FillValue": None}})
    with pytest.raises(ValueError, match="qc.*non-finite"):
        io.load_and_validate(path)


def test_negative_condensate_clamps_and_is_recorded(tmp_path):
    values = np.full((2, 3, 4), 0.1)
    values[0, 1, 2] = -1e-9    # ordinary LES advection undershoot
    path = _write(tmp_path / "neg.nc", values)
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
