"""
Pixel-by-pixel regression tests for behold renderer.

Compares new renders against precomputed reference images with tolerance
for Monte Carlo variance.

Run with: pytest tests/test_behold_renders.py -v
Skip slow tests: pytest tests/test_behold_renders.py -v -m "not slow"
"""

import importlib
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

# The package attribute `cloudyview.behold` is the public render function
# (it shadows the submodule); import the CLI module itself explicitly.
behold = importlib.import_module("cloudyview.behold")

try:
    # Works when run as a test module (e.g., pytest / python -m tests.test_behold_renders)
    from .conftest import (
        CAMERA_CONFIGS,
        MAX_DIFF_THRESHOLD,
        REFERENCE_DIR,
        RENDER_SETTINGS,
        RMSE_THRESHOLD,
        TEST_DATA_FILES,
    )
except ImportError:
    # Works when run directly as a script (python tests/test_behold_renders.py)
    from conftest import (
        CAMERA_CONFIGS,
        MAX_DIFF_THRESHOLD,
        REFERENCE_DIR,
        RENDER_SETTINGS,
        RMSE_THRESHOLD,
        TEST_DATA_FILES,
    )


def load_image_as_float(path: Path) -> np.ndarray:
    """
    Load a PNG image as a float32 array normalized to [0, 1].

    Parameters
    ----------
    path : Path
        Path to the PNG file.

    Returns
    -------
    np.ndarray
        Image as float32 array with values in [0, 1], shape (H, W, C).
    """
    img = iio.imread(path)
    return img.astype(np.float32) / 255.0


def compare_images(
    test_image: np.ndarray, reference_image: np.ndarray
) -> tuple[float, float, bool]:
    """
    Compare two images and compute error metrics.

    Parameters
    ----------
    test_image : np.ndarray
        Newly rendered image, float32 in [0, 1].
    reference_image : np.ndarray
        Reference image, float32 in [0, 1].

    Returns
    -------
    rmse : float
        Root mean square error across all pixels and channels.
    max_diff : float
        Maximum absolute difference across all pixels and channels.
    passed : bool
        True if both RMSE and max_diff are within thresholds.
    """
    diff = np.abs(test_image - reference_image)
    rmse = np.sqrt(np.mean(diff**2))
    max_diff = np.max(diff)

    passed = rmse <= RMSE_THRESHOLD and max_diff <= MAX_DIFF_THRESHOLD
    return rmse, max_diff, passed


def render_view(
    data_file: Path, view_name: str, camera_config: dict, output_dir: Path
) -> Path:
    """
    Render a single view using behold.

    Parameters
    ----------
    data_file : Path
        Path to the NetCDF data file.
    view_name : str
        Name of the view (used for output filename).
    camera_config : dict
        Camera configuration (position, azimuth, elevation, fov).
    output_dir : Path
        Directory to save the rendered image.

    Returns
    -------
    Path
        Path to the rendered image.
    """
    behold.main(
        filename=str(data_file),
        backend=RENDER_SETTINGS["backend"],
        quality="custom",
        output=str(output_dir),
        custom_spp=RENDER_SETTINGS["spp"],
        custom_size=RENDER_SETTINGS["resolution"],
        custom_max_depth=RENDER_SETTINGS["max_depth"],
        custom_rr_depth=RENDER_SETTINGS["rr_depth"],
        camera_position=camera_config["position"],
        camera_azimuth=camera_config["azimuth"],
        camera_elevation=camera_config["elevation"],
        camera_fov=camera_config["fov"],
        sun_azimuth=RENDER_SETTINGS["sun_azimuth"],
        sun_elevation=RENDER_SETTINGS["sun_elevation"],
        progress_interval=RENDER_SETTINGS["progress_interval"],
    )

    # behold outputs filename based on max_depth and rr_depth
    output_filename = (
        f"behold_ground_view_max_depth={RENDER_SETTINGS['max_depth']}"
        f"_rr_depth={RENDER_SETTINGS['rr_depth']}.png"
    )
    return output_dir / output_filename


def save_diff_image(
    test_image: np.ndarray, reference_image: np.ndarray, output_path: Path
) -> None:
    """
    Save a difference image for debugging failed tests.

    The diff image shows absolute differences scaled to be visible.
    """
    diff = np.abs(test_image - reference_image)
    # Scale to make differences visible (10x amplification)
    diff_scaled = np.clip(diff * 10, 0, 1)
    diff_uint8 = (diff_scaled * 255).astype(np.uint8)
    iio.imwrite(output_path, diff_uint8)


# Generate test cases: all combinations of (dataset, view)
TEST_CASES = [
    (dataset_name, view_name)
    for dataset_name in TEST_DATA_FILES
    for view_name in CAMERA_CONFIGS
]


@pytest.mark.slow
@pytest.mark.parametrize("dataset_name,view_name", TEST_CASES)
def test_render_matches_reference(dataset_name: str, view_name: str, tmp_path: Path):
    """
    Test that a rendered view matches its reference image within tolerance.

    This test:
    1. Renders the specified view for the specified dataset
    2. Loads the corresponding reference image
    3. Compares pixel-by-pixel with tolerance for Monte Carlo noise
    4. Saves a diff image on failure for debugging

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "TWPICE_256").
    view_name : str
        Name of the view (e.g., "ground_view").
    tmp_path : Path
        Pytest fixture providing a temporary directory.
    """
    # Get paths
    data_file = TEST_DATA_FILES[dataset_name]
    camera_config = CAMERA_CONFIGS[view_name]
    reference_path = REFERENCE_DIR / dataset_name / f"{view_name}.png"

    # Check prerequisites
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")

    if not reference_path.exists():
        pytest.skip(
            f"Reference image not found: {reference_path}. "
            f"Run 'python tests/generate_references.py' to generate it."
        )

    # Render the view
    rendered_path = render_view(data_file, view_name, camera_config, tmp_path)

    # Load images
    test_image = load_image_as_float(rendered_path)
    reference_image = load_image_as_float(reference_path)

    # Compare
    rmse, max_diff, passed = compare_images(test_image, reference_image)

    # Save diff image on failure
    if not passed:
        diff_path = tmp_path / f"diff_{dataset_name}_{view_name}.png"
        save_diff_image(test_image, reference_image, diff_path)
        pytest.fail(
            f"Render mismatch for {dataset_name}/{view_name}:\n"
            f"  RMSE: {rmse:.4f} (threshold: {RMSE_THRESHOLD})\n"
            f"  Max diff: {max_diff:.4f} (threshold: {MAX_DIFF_THRESHOLD})\n"
            f"  Diff image saved to: {diff_path}"
        )

    # Log metrics even on success
    print(f"\n  {dataset_name}/{view_name}: RMSE={rmse:.4f}, max_diff={max_diff:.4f}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
