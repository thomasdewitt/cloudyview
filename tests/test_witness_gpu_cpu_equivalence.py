"""Test that witness GPU and CPU rendering produce visually equivalent output."""

import glob
import importlib
from pathlib import Path

import numpy as np
import pytest

# The package attribute `cloudyview.witness` is the public render function
# (it shadows the submodule); import the module itself explicitly.
witness = importlib.import_module("cloudyview.witness")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_FILE = str(DATA_DIR / "TWPICE_subvolume_128x128_5km.nc")


def _cuda_backend_available():
    """Check the actual witness CUDA import path, not just raw numba.cuda."""
    try:
        from cloudyview.witness import _CUDA_AVAILABLE
        return _CUDA_AVAILABLE
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_backend_available(),
                    reason="witness CUDA backend not available")
def test_gpu_cpu_visual_equivalence(tmp_path):
    """GPU and CPU renders of the same scene should be nearly identical."""
    from PIL import Image

    cpu_dir = tmp_path / "cpu"
    gpu_dir = tmp_path / "gpu"

    # Render on CPU
    witness.main(DATA_FILE, output=str(cpu_dir), custom_size=(30, 20), gpu=False)

    # Render on GPU
    witness.main(DATA_FILE, output=str(gpu_dir), custom_size=(30, 20), gpu=True)

    # Load output images
    cpu_files = glob.glob(str(cpu_dir / "witness_*.png"))
    gpu_files = glob.glob(str(gpu_dir / "witness_*.png"))
    assert len(cpu_files) == 1
    assert len(gpu_files) == 1

    cpu_img = np.array(Image.open(cpu_files[0]))
    gpu_img = np.array(Image.open(gpu_files[0]))

    assert cpu_img.shape == gpu_img.shape

    # float32 vs float64 may cause small per-pixel differences after
    # tone mapping and 8-bit quantization
    diff = np.abs(cpu_img.astype(int) - gpu_img.astype(int))
    max_diff = np.max(diff)
    rmse = np.sqrt(np.mean(diff.astype(float) ** 2))

    assert max_diff <= 5, f"Max pixel difference {max_diff} > 5"
    assert rmse < 2.0, f"RMSE {rmse:.2f} >= 2.0"
