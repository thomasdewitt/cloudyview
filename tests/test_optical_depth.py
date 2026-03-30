"""Tests for optical depth calculation correctness."""

import numpy as np
import pytest

from cloudyview.optical_depth import vertically_integrated_optical_depth


def test_optical_depth_linearity():
    """Optical depth should scale linearly with the number of uniform cloud layers.

    Uses 2, 4, and 6 z-levels (each with dz=0.1m at ~1000m altitude) so that
    rho_air is effectively constant across all cases. The effective vertical
    paths are then proportional to the number of levels, so tau should scale
    exactly 1:2:3.

    Note: a minimum of 2 z-levels is required by the implementation (np.diff).
    Each pair of levels at equal spacing contributes the same water path, so
    2/4/6 levels give 1x/2x/3x the optical depth.
    """
    lwc_val = 1.0   # g/kg, uniform
    dz = 0.1        # m, tiny so height range is negligible → constant rho_air
    base_z = 1000.0  # m

    taus = []
    for n_levels in [2, 4, 6]:
        z = base_z + np.arange(n_levels) * dz
        lwc = np.full((1, 1, n_levels), lwc_val, dtype=np.float64)
        tau = vertically_integrated_optical_depth(lwc, z)
        taus.append(float(tau[0, 0]))

    tau_1x, tau_2x, tau_3x = taus

    assert tau_1x > 0, "Optical depth should be positive for nonzero LWC"
    # rtol=1e-4: the internal float32 arithmetic introduces ~1e-5 relative error,
    # so we allow 1e-4 which still catches any real linearity violations.
    np.testing.assert_allclose(tau_2x, 2.0 * tau_1x, rtol=1e-4,
                               err_msg="2x layers should give 2x optical depth")
    np.testing.assert_allclose(tau_3x, 3.0 * tau_1x, rtol=1e-4,
                               err_msg="3x layers should give 3x optical depth")


def test_optical_depth_zero_padding_invariance():
    """Surrounding a cloud layer with zero-content levels should not change tau.

    Compares [0,0,0,1,0,0,0] vs [0,1,0] where the cloudy level is at the same
    height (1000m) with the same dz (50m) in both cases. Only the nonzero level
    contributes to the water path integral, so the results must be identical.
    """
    lwc_val = 1.0  # g/kg
    dz = 50.0      # m
    cloud_z = 1000.0  # m, height of the cloudy level in both cases

    # Long column: cloud at index 3, heights 850–1150m in 50m steps
    z_long = cloud_z + np.arange(-3, 4) * dz   # [850, 900, 950, 1000, 1050, 1100, 1150]
    lwc_long = np.zeros((1, 1, 7))
    lwc_long[0, 0, 3] = lwc_val  # cloud at index 3 → z=1000m, dz=50m

    # Short column: cloud at index 1, heights 950–1050m in 50m steps
    z_short = cloud_z + np.arange(-1, 2) * dz  # [950, 1000, 1050]
    lwc_short = np.zeros((1, 1, 3))
    lwc_short[0, 0, 1] = lwc_val  # cloud at index 1 → z=1000m, dz=50m

    tau_long = float(vertically_integrated_optical_depth(lwc_long, z_long)[0, 0])
    tau_short = float(vertically_integrated_optical_depth(lwc_short, z_short)[0, 0])

    assert tau_long > 0, "Optical depth should be positive for nonzero LWC"
    np.testing.assert_allclose(tau_long, tau_short, rtol=1e-4,
                               err_msg="Zero padding around cloud layer should not change tau")
