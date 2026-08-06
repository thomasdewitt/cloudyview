"""Pytest configuration and fixtures for CloudyView render tests."""

from pathlib import Path

import pytest

# Importing cloudyview first wires DRJIT_LIBLLVM_PATH (behold module scope)
# before any test module's `pytest.importorskip("mitsuba")` can initialize
# Dr.Jit without it — the LLVM backend availability is decided at that
# first import.
import cloudyview  # noqa: F401

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
# Soar / witness look regression
# =============================================================================
#
# The witness CLI and the browser's soar app run the same WGSL shader, so one
# set of golden images pins the look of both. These eight cameras are the
# frozen judge views of the lighting loop (cloudyview-lighting-harness/
# views.json). They are copied here rather than read from that repo so the
# test suite stands on its own; the two lists are a contract and must be
# edited together, or renders stop being comparable across the loop.

SOAR_REFERENCE_DIR = REFERENCE_DIR / "soar_witness"
SOAR_DATA_FILE = DATA_DIR / "TWPICE_subvolume_256x256_5km.nc"

SOAR_VIEWS = {
    "v1_thick_backlit":  {"camera_position": [-0.1, -0.2, 0.1],  "azimuth": 20,  "elevation": 8,   "fov": 70},
    "v2_under_base":     {"camera_position": [0.2, 0.55, -0.85], "azimuth": 20,  "elevation": 25,  "fov": 80},
    "v3_thin_field":     {"camera_position": [-0.45, -0.5, 0.0], "azimuth": 300, "elevation": 8,   "fov": 70},
    "v4_overview_south": {"camera_position": [0.0, -1.0, 0.7],   "azimuth": 0,   "elevation": 2,   "fov": 90},
    "v5_high_oblique":   {"camera_position": [-0.5, -0.8, 2.2],  "azimuth": 30,  "elevation": -25, "fov": 70},
    "v6_thick_lit":      {"camera_position": [0.3, 1.7, 0.35],   "azimuth": 180, "elevation": 2,   "fov": 75},
    "v7_low_sun":        {"camera_position": [1.6, 0.2, 0.5],    "azimuth": 270, "elevation": -2,  "fov": 75,
                          "sun_azimuth": 250, "sun_elevation": 9},
    "v8_ocean_lod":      {"camera_position": [0.0, 0.0, 3.0],    "azimuth": 180, "elevation": -55, "fov": 70},
}

# The harness renders these at 960x540; the references are half that in each
# axis. RMSE is essentially resolution-independent for a look change, so the
# smaller images discriminate just as well while keeping eight PNGs that get
# re-baked from time to time down to ~1.8 MB of git history per bake.
SOAR_RENDER_SETTINGS = {
    "size": (640, 360),
    "accumulate": 64,
    "periodic": True,
    "sun_azimuth": 20.0,
    "sun_elevation": 55.0,
}

# Measured on the bake machine (RTX 5080, Vulkan, driver 595.80) at the
# settings above, worst view of the eight:
#
#   two independent renders, separate processes ..  RMSE 0.00000  max 0.00000
#   accumulate 64 -> 32 (same look, resampled) ...  RMSE 0.00401  max 0.06667
#   exposure +2% .................................  RMSE 0.00341  max 0.00392
#   exposure +5% .................................  RMSE 0.00748  max 0.01176
#   tone-map gamma +1% ...........................  RMSE 0.20949  max 0.23137
#
# There is no run-to-run Monte Carlo noise to absorb. The shader seeds its
# sampling on the frame index and the dither on a fixed seed, so a rerun on
# this machine is bit-identical -- the top row is a measurement, not a
# rounding. The tolerance is instead sized against the resampling row, the
# closest available stand-in for another GPU accumulating the same passes in
# another order, and set below the smallest look change worth catching.
#
# The gap between those two is real but narrow, so be honest about what this
# cannot see: a change smaller than roughly a 3% exposure nudge is
# indistinguishable here from a sampling difference. Gamma, sun angle, phase
# function and ocean changes all land orders of magnitude above the line;
# a hairline exposure trim would not.
SOAR_RMSE_THRESHOLD = 0.006
# RMSE is the real gate. This catches the localized change RMSE dilutes: a few
# hundred pixels moving a long way (the sun disc, the horizon line) rather
# than every pixel moving a little.
SOAR_MAX_DIFF_THRESHOLD = 0.12


def soar_gpu_adapter():
    """Return an adapter that can actually render these views, else None.

    A software rasterizer (llvmpipe, SwiftShader) is deliberately reported as
    None rather than used: the references are baked on real hardware, and a
    CPU adapter would produce a slow render of a subtly different image, which
    is worse than an honest skip.

    The probe goes as far as requesting a device, because enumerating an
    adapter is not evidence that one can be used. With Vulkan unavailable,
    wgpu falls back to an OpenGL adapter that reports the real GPU and then
    fails at device creation ("parent device is lost") -- which, checked any
    later than here, surfaces as a pile of confusing test failures instead of
    a skip.
    """
    try:
        import wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:
        return None
    if adapter is None:
        return None
    info = adapter.info
    if str(info.get("adapter_type", "")).lower() in ("cpu", "software"):
        return None
    if "llvmpipe" in str(info.get("device", "")).lower():
        return None
    try:
        adapter.request_device_sync()
    except Exception:
        return None
    return adapter


def build_soar_level(nc_path=None):
    """Build the NestedLevel the reference views are rendered from.

    This is the harness recipe (cloudyview-lighting-harness/render_views.py)
    and must stay identical to it.
    """
    import numpy as np

    import cloudyview as cv
    from cloudyview import optical_depth
    from cloudyview.witness import (
        ICE_NEGLIGIBLE_G_KG, RE_ICE_UM, RE_LIQUID_UM, NestedLevel, _volume_aabb,
    )

    nc_path = Path(nc_path or SOAR_DATA_FILE)
    field = cv.load(str(nc_path))
    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < ICE_NEGLIGIBLE_G_KG:
        iwc = None
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=RE_LIQUID_UM, iwc=iwc, re_ice=RE_ICE_UM)
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)
    bmin, bmax = _volume_aabb(field)
    return NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name=nc_path.stem)


def render_soar_view(level, view):
    """Render one frozen view off `level`, returning float RGB in [0, 1]."""
    from cloudyview.soar_host import camera_world_origin
    from cloudyview.witness import render_nested

    position = camera_world_origin(view["camera_position"], level.bmin, level.bmax)
    return render_nested(
        [level], position,
        azimuth=view["azimuth"],
        elevation=view["elevation"],
        sun_azimuth=view.get("sun_azimuth", SOAR_RENDER_SETTINGS["sun_azimuth"]),
        sun_elevation=view.get("sun_elevation", SOAR_RENDER_SETTINGS["sun_elevation"]),
        fov_degrees=view.get("fov", 70.0),
        image_size=SOAR_RENDER_SETTINGS["size"],
        periodic=SOAR_RENDER_SETTINGS["periodic"],
        accumulate=SOAR_RENDER_SETTINGS["accumulate"],
        verbose=False,
    )


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
