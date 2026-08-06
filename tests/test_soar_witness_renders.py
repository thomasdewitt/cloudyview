"""
Look regression for the witness / soar renderer.

`witness` and the browser's soar app drive the same WGSL shader, so these
eight golden images pin the look of both: the lighting, the sky and ocean
models, the phase function, the accumulation, and the dithered 8-bit encode.
The cameras are the frozen judge views of the lighting loop -- see
conftest.SOAR_VIEWS for what that contract means.

Needs a real GPU; skips cleanly without one. Regenerate the references with

    uv run python tests/generate_references.py --target soar --overwrite

Run with: uv run --extra dev python -m pytest tests/test_soar_witness_renders.py -v
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

# wgpu drives the renderer; skip (don't fail) when it isn't installed, so a
# Mitsuba-only checkout still runs the rest of the suite.
pytest.importorskip("wgpu")

import cloudyview as cv

try:
    from .conftest import (
        SOAR_DATA_FILE,
        SOAR_MAX_DIFF_THRESHOLD,
        SOAR_REFERENCE_DIR,
        SOAR_RENDER_SETTINGS,
        SOAR_RMSE_THRESHOLD,
        SOAR_VIEWS,
        build_soar_level,
        render_soar_view,
        soar_gpu_adapter,
    )
except ImportError:
    from conftest import (
        SOAR_DATA_FILE,
        SOAR_MAX_DIFF_THRESHOLD,
        SOAR_REFERENCE_DIR,
        SOAR_RENDER_SETTINGS,
        SOAR_RMSE_THRESHOLD,
        SOAR_VIEWS,
        build_soar_level,
        render_soar_view,
        soar_gpu_adapter,
    )


@pytest.fixture(scope="module")
def soar_level():
    """The uploaded field, built once for all eight views.

    Both halves are expensive and shared: reading the volume and computing
    extinction takes seconds, and holding one NestedLevel alive lets the
    renderer's session cache keep the ~400 MB volume on the GPU across views
    instead of re-uploading it per test.
    """
    if soar_gpu_adapter() is None:
        pytest.skip(
            "No GPU adapter available (a software rasterizer does not count); "
            "the witness/soar references are baked on real hardware."
        )
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    return build_soar_level()


def encode(image: np.ndarray) -> np.ndarray:
    """Put a float render through the library's 8-bit encode, as [0, 1] floats.

    Comparing after the encode rather than before means the dither is applied
    to both sides and cancels, and it puts the encode itself under test.
    """
    return cv.quantize_uint8(image).astype(np.float32) / 255.0


def load_reference(path: Path) -> np.ndarray:
    img = iio.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img.astype(np.float32) / 255.0


def save_diff_image(test_image, reference_image, output_path: Path) -> None:
    """Absolute difference, amplified 10x so a subtle drift is visible."""
    diff = np.clip(np.abs(test_image - reference_image) * 10, 0, 1)
    iio.imwrite(output_path, (diff * 255).astype(np.uint8))


@pytest.mark.gpu
@pytest.mark.parametrize("view_name", list(SOAR_VIEWS))
def test_soar_view_matches_reference(view_name, soar_level, tmp_path):
    """Render one frozen view and hold it against its golden image."""
    reference_path = SOAR_REFERENCE_DIR / f"{view_name}.png"
    if not reference_path.exists():
        pytest.skip(
            f"Reference image not found: {reference_path}. Run "
            "'uv run python tests/generate_references.py --target soar'."
        )

    test_image = encode(render_soar_view(soar_level, SOAR_VIEWS[view_name]))
    reference_image = load_reference(reference_path)

    if test_image.shape != reference_image.shape:
        pytest.fail(
            f"Render/reference shape mismatch for {view_name}: "
            f"{test_image.shape} vs {reference_image.shape}. The reference was "
            f"baked at a different size than SOAR_RENDER_SETTINGS "
            f"({SOAR_RENDER_SETTINGS['size']}) asks for; regenerate it."
        )

    diff = np.abs(test_image - reference_image)
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_diff = float(np.max(diff))

    if rmse > SOAR_RMSE_THRESHOLD or max_diff > SOAR_MAX_DIFF_THRESHOLD:
        diff_path = tmp_path / f"diff_{view_name}.png"
        save_diff_image(test_image, reference_image, diff_path)
        pytest.fail(
            f"Look regression for {view_name}:\n"
            f"  RMSE: {rmse:.5f} (threshold: {SOAR_RMSE_THRESHOLD})\n"
            f"  Max diff: {max_diff:.5f} (threshold: {SOAR_MAX_DIFF_THRESHOLD})\n"
            f"  Diff image (10x): {diff_path}\n"
            f"If the change was intended, re-bake with "
            f"'uv run python tests/generate_references.py --target soar --overwrite'."
        )

    print(f"\n  soar/{view_name}: RMSE={rmse:.5f}, max_diff={max_diff:.5f}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
