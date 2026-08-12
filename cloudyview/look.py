"""Shared, dependency-light constants for CloudyView's rendered look.

This module is the single source of truth for the tuning values shared by the
Numba Witness renderer and the wgpu Soar renderer.  Keep it free of NumPy,
Numba, Mitsuba, and other renderer-specific dependencies so Soar can import
the look without pulling the offline renderers into a desktop bundle.
"""

import math
from typing import Tuple


SUN_COLOR = (20.2, 21.0, 22.4)

AMBIENT_TINT_R = 0.18
AMBIENT_TINT_G = 0.225
AMBIENT_TINT_B = 0.33

SPECTRAL_LIGHTING_STRENGTH = 1.0
ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG = 55.0
ATMOSPHERE_MAX_AIRMASS = 20.0
ATMOSPHERE_RAYLEIGH_OD_550 = 0.10
ATMOSPHERE_AEROSOL_OD_550 = 0.12
ATMOSPHERE_AEROSOL_ANGSTROM = 1.3
ATMOSPHERE_RGB_WAVELENGTHS_NM = (680.0, 550.0, 460.0)
SUNSET_HORIZON_RADIANCE = (0.42, 0.20, 0.055)

LOW_SUN_SKY_FIELD_STRENGTH = 1.0
LOW_SUN_SKY_WARM_ELEVATION_DEG = 32.0
LOW_SUN_SKY_HORIZON_AZIMUTH_DEG = 105.0
LOW_SUN_SKY_UPPER_AZIMUTH_DEG = 45.0
LOW_SUN_SKY_NEUTRAL_RADIANCE = (0.27, 0.30, 0.32)
_LOW_SUN_SKY_MAX_WARM_DZ = math.sin(math.radians(
    LOW_SUN_SKY_WARM_ELEVATION_DEG
))
_LOW_SUN_SKY_HORIZON_AZIMUTH_COS = math.cos(math.radians(
    LOW_SUN_SKY_HORIZON_AZIMUTH_DEG
))
_LOW_SUN_SKY_UPPER_AZIMUTH_COS = math.cos(math.radians(
    LOW_SUN_SKY_UPPER_AZIMUTH_DEG
))

CONE_STENCIL_THETA_DEG = 2.0

LIGHT_TRANSFER_SPLIT_STRENGTH = 1.0
LIGHT_TRANSFER_DIRECT_BOOST = 0.25
LIGHT_TRANSFER_SHADOW_SKYLIGHT = 0.26
LIGHT_TRANSFER_FULL_ELEVATION_DEG = 45.0
LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG = 55.0

AERIAL_PERSPECTIVE_STRENGTH = 1.0
AERIAL_BETA_PER_KM = 0.035
AERIAL_SCALE_HEIGHT_M = 2500.0

OCEAN_REALISM = 1.0
OCEAN_MIP_BIAS = -0.5
OCEAN_GLINT_STRENGTH = 0.85
OCEAN_GLINT_ROUGHNESS = 0.28
# How much of the slope variance the normal-mip filter removed is drawn
# stochastically per pixel per frame (lighting-loop iter_008). The remainder
# stays as extra microfacet-lobe width. 1 = fully sampled, 0 = fully analytic.
OCEAN_SLOPE_DRAW_FRACTION = 0.5
OCEAN_SKY_SHADOW_FLOOR = 0.75
OCEAN_HAZE_EXTINCTION_PER_KM = 0.012

# --- haze: one knob for the whole aerosol story ----------------------------
#
# Four separate sliders for aerial extinction, horizon whitening, the
# circumsolar lobe and the haze over the sea would let a viewer build a sky
# no photograph contains — a razor-sharp far field under a milky horizon.
# They are all the same aerosol, so they get one number, and every term is
# ANCHORED at HAZE_ANCHOR: at 0.35 each expression returns its tuned
# constant exactly. The anchor is a calibration point, not the default —
# Thomas flies at 1.0 ("slammed to 1 looks best", 2026-08-11), so that is
# the default; the slider runs to 2 for genuinely soupy days.
HAZE_ANCHOR = 0.35
DEFAULT_HAZE = 1.0
HAZE_MAX = 2.0
# Even at haze 0 the air is not vacuum. 0.015/km is a clean Rayleigh-limited
# atmosphere: Koschmieder's 3.912/beta puts the visual range at ~260 km,
# which is the "you can see the far range from the pass" day, not vacuum.
AERIAL_BETA_FLOOR_PER_KM = 0.015
# Extinction rises faster than linearly with the slider because aerosol
# loading does: haze 1 lands at 0.11/km, a visual range of ~36 km, which is
# a genuinely thick summer haze rather than a slightly softer version of the
# default. A linear ramp spent most of its travel in looks nobody wants.
_AERIAL_BETA_HAZE_COEFFICIENT = (
    (AERIAL_BETA_PER_KM - AERIAL_BETA_FLOOR_PER_KM)
    / (HAZE_ANCHOR * math.sqrt(HAZE_ANCHOR))
)
# The sea's haze was tuned against the sky's, not independently: 0.343x the
# column beta. Holding the ratio is what keeps a hazy horizon and a hazy
# ocean the same weather.
_OCEAN_HAZE_BETA_RATIO = OCEAN_HAZE_EXTINCTION_PER_KM / AERIAL_BETA_PER_KM


def aerial_beta_per_km(haze: float) -> float:
    """Sea-level clear-air extinction for a haze setting in [0, HAZE_MAX].

    h**1.5 is written h*sqrt(h) so the browser cannot disagree: sqrt is
    correctly rounded by IEEE-754 and pow is not, and this number is packed
    into a uniform block that a test diffs byte for byte against JS.
    """
    return (AERIAL_BETA_FLOOR_PER_KM
            + _AERIAL_BETA_HAZE_COEFFICIENT * (haze * math.sqrt(haze)))


def ocean_haze_extinction_per_km(haze: float) -> float:
    """Haze extinction along the sea-surface sight line, tracking the sky."""
    return _OCEAN_HAZE_BETA_RATIO * aerial_beta_per_km(haze)


def _spectral_lighting_colors(
    sun_direction: Tuple[float, float, float],
    sun_color: Tuple[float, float, float],
    strength: float,
):
    """Precompute low-sun cloud and sky spectra from relative air mass."""
    legacy_ambient = (AMBIENT_TINT_R, AMBIENT_TINT_G, AMBIENT_TINT_B)
    legacy_horizon = (0.10, 0.18, 0.38)
    legacy_bloom = (0.8, 0.6, 0.3)
    legacy_disc = (50.0, 45.0, 35.0)
    if strength == 0.0:
        return (sun_color, legacy_ambient, legacy_horizon,
                legacy_bloom, legacy_disc)

    def air_mass(elevation_deg):
        elevation_deg = max(0.0, elevation_deg)
        denom = (math.sin(math.radians(elevation_deg))
                 + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)
        return min(ATMOSPHERE_MAX_AIRMASS, 1.0 / denom)

    sun_z = max(-1.0, min(1.0, sun_direction[2]))
    elevation_deg = math.degrees(math.asin(sun_z))
    extra_air_mass = max(
        0.0,
        air_mass(elevation_deg)
        - air_mass(ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG),
    )

    optical_depths = []
    for wavelength_nm in ATMOSPHERE_RGB_WAVELENGTHS_NM:
        wavelength_ratio = 550.0 / wavelength_nm
        rayleigh_od = ATMOSPHERE_RAYLEIGH_OD_550 * wavelength_ratio ** 4.0
        aerosol_od = (ATMOSPHERE_AEROSOL_OD_550
                      * wavelength_ratio ** ATMOSPHERE_AEROSOL_ANGSTROM)
        optical_depths.append(rayleigh_od + aerosol_od)

    beam_scale = tuple(
        1.0 - strength * (1.0 - math.exp(-extra_air_mass * tau))
        for tau in optical_depths
    )
    cloud_sun = tuple(sun_color[c] * beam_scale[c] for c in range(3))

    scattered = tuple(1.0 - math.exp(-tau) for tau in optical_depths)
    legacy_luma = (0.2126 * legacy_ambient[0]
                   + 0.7152 * legacy_ambient[1]
                   + 0.0722 * legacy_ambient[2])
    scattered_luma = (0.2126 * scattered[0]
                      + 0.7152 * scattered[1]
                      + 0.0722 * scattered[2])
    fill_target = tuple(c * legacy_luma / scattered_luma for c in scattered)
    fill_mix = strength * (1.0 - math.exp(-0.6 * extra_air_mass))
    ambient = tuple(
        legacy_ambient[c] + fill_mix * (fill_target[c] - legacy_ambient[c])
        for c in range(3)
    )

    horizon_mix = strength * (1.0 - math.exp(-0.45 * extra_air_mass))
    horizon = tuple(
        legacy_horizon[c]
        + horizon_mix * (SUNSET_HORIZON_RADIANCE[c] - legacy_horizon[c])
        for c in range(3)
    )

    bloom = tuple(legacy_bloom[c] * beam_scale[c] for c in range(3))
    disc = tuple(legacy_disc[c] * beam_scale[c] for c in range(3))
    return cloud_sun, ambient, horizon, bloom, disc
