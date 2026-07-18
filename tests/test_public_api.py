"""Tests for the library-first public API: cv.load / CloudField, cv.Camera,
and the render functions cv.glimpse / cv.witness / cv.behold."""

import importlib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cloudyview as cv

# The package attributes glimpse/witness/behold are the render functions;
# the CLI modules are imported explicitly where needed.
glimpse_mod = importlib.import_module("cloudyview.glimpse")
witness_mod = importlib.import_module("cloudyview.witness")

try:
    from .conftest import TEST_DATA_FILES
except ImportError:
    from conftest import TEST_DATA_FILES


def _twpice_path() -> Path:
    data_file = TEST_DATA_FILES["TWPICE_128"]
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")
    return data_file


# =============================================================================
# Synthetic SAM LPT-style split-file pair (one variable per file, singleton
# time dimension, empty units attribute — mirrors real SAM 3D output).
# =============================================================================

def _write_sam_style(path: Path, var: str, values: np.ndarray,
                     z_offset: float = 0.0) -> None:
    nx, ny, nz = values.shape
    ds = xr.Dataset(
        {
            var: (
                ("time", "x", "y", "z"),
                values[np.newaxis].astype(np.float32),
                {"units": "", "long_name": ""},
            )
        },
        coords={
            "time": ("time", np.array([20.01], dtype=np.float32)),
            "x": ("x", (100.0 + 200.0 * np.arange(nx)).astype(np.float32)),
            "y": ("y", (100.0 + 200.0 * np.arange(ny)).astype(np.float32)),
            "z": ("z", (25.0 + z_offset + 50.0 * np.arange(nz)).astype(np.float32)),
        },
    )
    ds.to_netcdf(path)


@pytest.fixture
def sam_split_pair(tmp_path):
    rng = np.random.default_rng(0)
    qc = rng.random((4, 3, 5)).astype(np.float32)
    qi = (0.1 * rng.random((4, 3, 5))).astype(np.float32)
    qc_path = tmp_path / "TEST_LPT_3D_QC_0000000600.nc"
    qi_path = tmp_path / "TEST_LPT_3D_QI_0000000600.nc"
    _write_sam_style(qc_path, "QC", qc)
    _write_sam_style(qi_path, "QI", qi)
    return qc_path, qi_path, qc, qi


# =============================================================================
# cv.load / CloudField
# =============================================================================

def test_load_single_file():
    field = cv.load(str(_twpice_path()))

    assert isinstance(field, cv.CloudField)
    assert field.lwc.dtype == np.float32
    assert field.lwc.ndim == 3
    nx, ny, nz = field.shape
    assert (len(field.x), len(field.y), len(field.z)) == (nx, ny, nz)
    assert field.liquid_var == "QC"
    assert field.source.endswith(_twpice_path().name)
    # TWPICE subvolume has no ice variable
    assert field.iwc is None
    assert field.ice_source is None


def test_load_split_files(sam_split_pair):
    qc_path, qi_path, qc, qi = sam_split_pair

    field = cv.load(qc_path, ice=qi_path)

    assert field.liquid_var == "QC"
    assert field.ice_var == "QI"
    assert field.lwc.dtype == np.float32
    assert field.iwc is not None and field.iwc.dtype == np.float32
    np.testing.assert_array_equal(field.lwc, qc)
    np.testing.assert_array_equal(field.iwc, qi)
    # singleton time dropped, dims standardized
    assert field.shape == (4, 3, 5)
    np.testing.assert_allclose(field.z, 25.0 + 50.0 * np.arange(5))
    assert field.source.endswith("QC_0000000600.nc")
    assert field.ice_source.endswith("QI_0000000600.nc")


def test_load_split_ice_is_required_in_ice_file(sam_split_pair, tmp_path):
    qc_path, _, qc, _ = sam_split_pair
    # A file with no recognizable ice variable
    not_ice = tmp_path / "TEST_LPT_3D_U_0000000600.nc"
    _write_sam_style(not_ice, "U", qc)

    with pytest.raises(ValueError, match="ice water variable"):
        cv.load(qc_path, ice=not_ice)


def test_load_split_grid_mismatch_raises(sam_split_pair, tmp_path):
    qc_path, _, _, qi = sam_split_pair
    shifted = tmp_path / "TEST_LPT_3D_QI_shifted.nc"
    _write_sam_style(shifted, "QI", qi, z_offset=10.0)  # different z grid

    with pytest.raises(ValueError, match="z-coordinate"):
        cv.load(qc_path, ice=shifted)


def test_load_explicit_var_overrides(sam_split_pair):
    qc_path, qi_path, _, _ = sam_split_pair

    field = cv.load(qc_path, ice=qi_path,
                    liquid_water_var="QC", ice_water_var="QI")
    assert field.liquid_var == "QC" and field.ice_var == "QI"

    with pytest.raises(ValueError):
        cv.load(qc_path, liquid_water_var="NOT_A_VAR")


def test_cloudfield_direct_construction_validates():
    lwc = np.zeros((2, 3, 4), dtype=np.float64)  # coerced to float32
    x = np.arange(2.0)
    y = np.arange(3.0)
    z = np.arange(4.0)
    field = cv.CloudField(lwc=lwc, x=x, y=y, z=z)
    assert field.lwc.dtype == np.float32

    with pytest.raises(ValueError, match="iwc shape"):
        cv.CloudField(lwc=lwc, x=x, y=y, z=z, iwc=np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="coordinate"):
        cv.CloudField(lwc=lwc, x=np.arange(5.0), y=y, z=z)


def test_cloudfield_normalizes_descending_coordinates():
    lwc = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    iwc = lwc + 100.0
    field = cv.CloudField(
        lwc=lwc,
        iwc=iwc,
        x=np.array([10.0, 0.0]),
        y=np.array([20.0, 10.0, 0.0]),
        z=np.array([300.0, 200.0, 100.0, 0.0]),
    )

    np.testing.assert_array_equal(field.x, [0.0, 10.0])
    np.testing.assert_array_equal(field.y, [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(field.z, [0.0, 100.0, 200.0, 300.0])
    np.testing.assert_array_equal(field.lwc, lwc[::-1, ::-1, ::-1])
    np.testing.assert_array_equal(field.iwc, iwc[::-1, ::-1, ::-1])


# =============================================================================
# cv.Camera
# =============================================================================

def test_camera_defaults_match_config():
    from cloudyview import config
    cam = cv.Camera()
    default = config.DEFAULT_WITNESS_CONFIG['camera']
    assert cam.position == tuple(default['position'])
    assert cam.azimuth == default['azimuth']
    assert cam.elevation == default['elevation']
    assert cam.fov == default['fov']


def test_camera_validation():
    with pytest.raises(ValueError):
        cv.Camera(position=(0.0, 0.0))
    with pytest.raises(ValueError):
        cv.Camera(elevation=120.0)
    with pytest.raises(ValueError):
        cv.Camera(fov=0.0)


def test_camera_basis_orthonormal():
    cam = cv.Camera(azimuth=37.0, elevation=20.0)
    forward, right, up = cam.basis()
    for v in (forward, right, up):
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(forward, right), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(forward, up), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(right, up), 0.0, atol=1e-12)


def test_camera_basis_continuous_through_vertical():
    # Regression (2026-07-17): the old up-reference flip within ~2.5° of
    # vertical snapped the view (and flight frame) when looking straight
    # up or down. The right vector must stay the analytic horizontal
    # (cos az, -sin az, 0) at every elevation, including exactly ±90.
    for az in (0.0, 45.0, 200.0):
        expected = np.array(
            [np.cos(np.deg2rad(az)), -np.sin(np.deg2rad(az)), 0.0]
        )
        prev = None
        for el in (0.0, 60.0, 87.0, 89.0, 89.9, 90.0, -90.0):
            _, right, _ = cv.Camera(azimuth=az, elevation=el).basis()
            np.testing.assert_allclose(right, expected, atol=1e-12)
            if prev is not None:
                assert np.linalg.norm(right - prev) < 1e-9
            prev = right


# =============================================================================
# cv.glimpse
# =============================================================================

def test_glimpse_shape_and_range(sam_split_pair):
    qc_path, qi_path, _, _ = sam_split_pair
    field = cv.load(qc_path, ice=qi_path)
    albedo = cv.glimpse(field)
    assert albedo.shape == (3, 4)  # (ny, nx)
    assert albedo.dtype == np.float32
    assert np.all(albedo >= 0) and np.all(albedo < 1)


def test_glimpse_matches_cli_albedo(monkeypatch, tmp_path):
    """The CLI's plotted albedo must be exactly cv.glimpse(field)."""
    data_file = _twpice_path()
    captured = {}

    def fake_plot_optical_depth(optical_depth_2d, **kwargs):
        captured["albedo"] = np.array(optical_depth_2d, copy=True)
        Path(kwargs["output_path"]).write_bytes(b"fake")
        return None, None

    monkeypatch.setattr(glimpse_mod.basic_render, "plot_optical_depth",
                        fake_plot_optical_depth)

    glimpse_mod.main(str(data_file), output=str(tmp_path))

    library_albedo = cv.glimpse(cv.load(str(data_file)))
    np.testing.assert_array_equal(captured["albedo"], library_albedo)


# =============================================================================
# cv.witness
# =============================================================================

@pytest.mark.slow
def test_witness_returns_image():
    field = cv.load(str(_twpice_path()))
    img = cv.witness(field, camera=cv.Camera(), size=(30, 20))
    assert img.shape == (20, 30, 3)
    assert np.all(np.isfinite(img))
    assert img.min() >= 0.0 and img.max() <= 1.0


# =============================================================================
# cv.behold
# =============================================================================

@pytest.mark.slow
def test_behold_returns_image():
    pytest.importorskip("mitsuba")
    field = cv.load(str(_twpice_path()))
    img = cv.behold(field, camera=cv.Camera(), quality="min")
    assert img.shape == (100, 150, 3)
    assert np.all(np.isfinite(img))


# =============================================================================
# cv.save_image
# =============================================================================

def test_save_image_roundtrip(tmp_path):
    from PIL import Image
    img = np.zeros((4, 6, 3))
    img[..., 0] = 1.0   # pure red
    out = tmp_path / "img.png"
    cv.save_image(img, str(out))
    loaded = np.array(Image.open(out))
    assert loaded.shape == (4, 6, 3)
    assert loaded[..., 0].min() == 255 and loaded[..., 1].max() == 0
