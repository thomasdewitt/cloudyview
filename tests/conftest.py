"""Pytest configuration and fixtures for CloudyView render tests."""

from pathlib import Path

import pytest

# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = Path(__file__).parent / "reference_images"

# =============================================================================
# Test Data Files
# =============================================================================

TEST_DATA_FILES = {
    "TWPICE_256": DATA_DIR / "TWPICE_subvolume_256x256_5km.nc",
    "TWPICE_128": DATA_DIR / "TWPICE_subvolume_128x128_5km.nc",
}

# =============================================================================
# Camera Configurations (4 views per dataset)
# =============================================================================

CAMERA_CONFIGS = {
    "ground_view": {
        "position": [0, -0.99, -0.99],
        "azimuth": 0.0,
        "elevation": 35.0,
        "fov": 100.0,
    },
    "overhead_view": {
        "position": [0, -.5, 4.0],
        "azimuth": 0.0,
        "elevation": -85.0,
        "fov": 80.0,
    },
    "side_east": {
        "position": [1.5, 0, -.99],
        "azimuth": 270.0,
        "elevation": 35.0,
        "fov": 90.0,
    },
    "side_north": {
        "position": [0, 1.5, -.99],
        "azimuth": 180.0,
        "elevation": 35.0,
        "fov": 90.0,
    },
}

# =============================================================================
# Render Settings
# =============================================================================

RENDER_SETTINGS = {
    "resolution": (10,6),
    "spp": 1024,
    "max_depth": 128,
    "rr_depth": 64,
    # "spp": 2,
    # "max_depth": 4,
    # "rr_depth": 2,
    "backend": "llvm",
    "sun_azimuth": 20.0,
    "sun_elevation": 55.0,
    "progress_interval": 128,
}

# =============================================================================
# Tolerance Thresholds
# =============================================================================

# With 1024 spp, Monte Carlo noise is ~3%, so these thresholds give ~2x margin
RMSE_THRESHOLD = 0.05  # 5% average difference on 0-1 scale
MAX_DIFF_THRESHOLD = 0.15  # 15% max per-pixel difference


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
