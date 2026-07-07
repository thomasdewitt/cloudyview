"""Sanity checks for glimpse labeling and camera overlay behavior."""

import os
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

# The package attribute `cloudyview.glimpse` is the public render function
# (it shadows the submodule); import the module itself explicitly.
import importlib

glimpse = importlib.import_module("cloudyview.glimpse")

try:
    from .conftest import TEST_DATA_FILES
except ImportError:
    from conftest import TEST_DATA_FILES


def _twpice_dataset() -> Path:
    data_file = TEST_DATA_FILES["TWPICE_128"]
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")
    return data_file


def _output_png_path(output_dir: Path, data_file: Path) -> Path:
    return output_dir / f"cloudyview_glimpse_top_view_{data_file.stem}.png"


def _read_rgb(image_path: Path) -> np.ndarray:
    img = iio.imread(image_path)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[-1] >= 3:
        return img[..., :3]
    raise ValueError(f"Unexpected image shape: {img.shape}")


def _count_red_pixels(image: np.ndarray) -> int:
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    red_mask = (r > 180) & (g < 120) & (b < 120)
    return int(np.count_nonzero(red_mask))


def test_glimpse_no_label_writes_png(tmp_path: Path):
    data_file = _twpice_dataset()
    glimpse.main(str(data_file), output=str(tmp_path), label_dirs=False, label=False)

    output_path = _output_png_path(tmp_path, data_file)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    img = _read_rgb(output_path)
    assert img.shape[0] == img.shape[1]


def test_glimpse_label_writes_png_and_changes_pixels(tmp_path: Path):
    data_file = _twpice_dataset()

    plain_dir = tmp_path / "plain"
    labeled_dir = tmp_path / "labeled"
    plain_dir.mkdir()
    labeled_dir.mkdir()

    glimpse.main(str(data_file), output=str(plain_dir), label=False)
    glimpse.main(str(data_file), output=str(labeled_dir), label=True)

    plain_img = _read_rgb(_output_png_path(plain_dir, data_file))
    labeled_img = _read_rgb(_output_png_path(labeled_dir, data_file))

    assert plain_img.size > 0
    assert labeled_img.size > 0
    assert plain_img.shape[0] == plain_img.shape[1]
    assert labeled_img.shape[0] == labeled_img.shape[1]
    assert plain_img.shape[:2] == labeled_img.shape[:2]
    assert _count_red_pixels(labeled_img) > _count_red_pixels(plain_img) + 50


def test_overlay_default_camera_centerish():
    overlay = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[0.0, 0.0, -0.999],
        camera_azimuth=0.0,
        camera_elevation=35.0,
        camera_fov=100.0,
        render_aspect=600 / 400,
    )
    cam_x, cam_y = overlay["camera_xy"]
    assert abs(cam_x - 299.5) < 2.0
    assert abs(cam_y - 199.5) < 2.0


def test_overlay_camera_position_mapping():
    lower_left = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[-1.0, -1.0, -0.5],
        camera_azimuth=0.0,
        camera_elevation=20.0,
        camera_fov=60.0,
        render_aspect=600 / 400,
    )
    upper_right = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[1.0, 1.0, -0.5],
        camera_azimuth=0.0,
        camera_elevation=20.0,
        camera_fov=60.0,
        render_aspect=600 / 400,
    )
    x1, y1 = lower_left["camera_xy"]
    x2, y2 = upper_right["camera_xy"]
    assert x1 < 1.0 and y1 < 1.0
    assert x2 > 598.0 and y2 > 398.0


def test_fov_rays_direction_north():
    overlay = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[0.0, 0.0, -0.5],
        camera_azimuth=0.0,
        camera_elevation=25.0,
        camera_fov=70.0,
        render_aspect=600 / 400,
    )
    cam_x, cam_y = overlay["camera_xy"]
    endpoints = overlay["fov_endpoints"]
    assert len(endpoints) == 2
    for end_x, end_y in endpoints:
        assert end_y > cam_y
        assert abs(end_x - cam_x) < 900.0


def test_fov_rays_direction_east():
    overlay = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[0.0, 0.0, -0.5],
        camera_azimuth=90.0,
        camera_elevation=25.0,
        camera_fov=70.0,
        render_aspect=600 / 400,
    )
    cam_x, _ = overlay["camera_xy"]
    endpoints = overlay["fov_endpoints"]
    assert len(endpoints) == 2
    for end_x, _ in endpoints:
        assert end_x > cam_x


def test_zenith_switches_to_circle():
    overlay = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[0.0, 0.0, -0.5],
        camera_azimuth=0.0,
        camera_elevation=80.0,
        camera_fov=30.0,
        render_aspect=600 / 400,
    )
    assert "circle_radius" in overlay
    assert overlay["circle_radius"] == pytest.approx(60.0)
    assert "fov_endpoints" not in overlay


def test_nadir_switches_to_circle():
    overlay = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[0.0, 0.0, -0.5],
        camera_azimuth=0.0,
        camera_elevation=-80.0,
        camera_fov=30.0,
        render_aspect=600 / 400,
    )
    assert "circle_radius" in overlay
    assert overlay["circle_radius"] == pytest.approx(60.0)
    assert "fov_endpoints" not in overlay


def test_non_zenith_uses_rays_not_circle():
    overlay = glimpse._build_camera_overlay(
        image_shape=(400, 600),
        camera_position=[0.0, 0.0, -0.5],
        camera_azimuth=0.0,
        camera_elevation=35.0,
        camera_fov=100.0,
        render_aspect=600 / 400,
    )
    assert "circle_radius" not in overlay
    assert "fov_endpoints" in overlay
    assert len(overlay["fov_endpoints"]) == 2


def test_cli_accepts_camera_args(tmp_path: Path):
    data_file = _twpice_dataset()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "cloudyview.glimpse",
        str(data_file),
        "--output",
        str(tmp_path),
        "--label",
        "--camera-position",
        "0",
        "0",
        "-0.9",
        "--camera-azimuth",
        "0",
        "--camera-elevation",
        "35",
        "--fov",
        "100",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert _output_png_path(tmp_path, data_file).exists()


def test_descending_coords_keep_east_right_north_up(monkeypatch, tmp_path: Path):
    class DummyArray:
        def __init__(self, values):
            self.values = values

    captured = {}

    def fake_load_and_validate(_filename, **_kwargs):
        lw = np.array(
            [
                [[0.2, 0.2], [0.4, 0.4]],
                [[0.6, 0.6], [0.8, 0.8]],
            ],
            dtype=np.float64,
        )
        return {
            "liquid_water_var": "qc",
            "liquid_water_data": DummyArray(lw),
            "ice_water_var": None,
            "ice_water_data": None,
            "x_coord": np.array([1000.0, 0.0]),   # descending x
            "y_coord": np.array([1000.0, 0.0]),   # descending y
            "z_coord": np.array([0.0, 100.0]),
        }

    def fake_vertically_integrated_optical_depth(lwc, z_coord, iwc=None):
        assert iwc is None
        assert lwc.shape == (2, 2, 2)
        assert z_coord.shape == (2,)
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

    def fake_plot_optical_depth(optical_depth_2d, **kwargs):
        captured["albedo"] = np.array(optical_depth_2d, copy=True)
        captured["overlay"] = kwargs.get("camera_overlay")
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")
        return None, None

    monkeypatch.setattr(glimpse.io, "load_and_validate", fake_load_and_validate)
    monkeypatch.setattr(
        glimpse.optical_depth, "vertically_integrated_optical_depth", fake_vertically_integrated_optical_depth
    )
    monkeypatch.setattr(glimpse.basic_render, "plot_optical_depth", fake_plot_optical_depth)

    glimpse.main(
        "dummy.nc",
        output=str(tmp_path),
        label=True,
        camera_position=[0.0, 0.0, -0.9],
        camera_azimuth=90.0,
        camera_elevation=25.0,
        camera_fov=70.0,
    )

    raw_tau = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    # Two-stream visual albedo, g=0.85 (matches glimpse.main)
    raw_albedo = raw_tau / (raw_tau + np.float32(2.0 / (1.0 - 0.85)))
    expected = raw_albedo[::-1, ::-1].T
    np.testing.assert_allclose(captured["albedo"], expected, atol=1e-6)

    cam_x, _ = captured["overlay"]["camera_xy"]
    for end_x, _ in captured["overlay"]["fov_endpoints"]:
        assert end_x > cam_x


def test_label_dirs_and_label_together(tmp_path: Path):
    data_file = _twpice_dataset()
    glimpse.main(str(data_file), output=str(tmp_path), label_dirs=True, label=True)

    out_path = _output_png_path(tmp_path, data_file)
    assert out_path.exists()
    img = _read_rgb(out_path)
    assert _count_red_pixels(img) > 50
