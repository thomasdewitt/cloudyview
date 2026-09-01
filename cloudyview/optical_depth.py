"""Optical depth and extinction coefficient calculations for CloudyView.

Uses physical models for cloud radiative properties.
"""

import logging

import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# --- bulk optics -----------------------------------------------------------
#
# Liquid is the geometric-optics limit with Q_ext = 2 exactly:
# tau = 3 Q_ext LWP / (4 rho_w r_e) = 1.5 LWP / r_e for LWP in g/m^2 and
# r_e in microns (rho_w = 1e6 g/m^3 folds in). Petty (2006), "A First Course
# in Atmospheric Radiation", 2nd ed., Eq. 7.86. compute_extinction_field
# uses the same coefficients for both species, so the column integral and
# the volume renderers agree.
TAU_PER_LWP_UM = np.float32(1.5)          # m^2 um / g

# Ice is Ebert & Curry (1992), JGR 97(D4), 3831-3836, doi:10.1029/91JD02472,
# Table 2 band 1 (0.25-0.7 um), used exactly as published:
# tau / IWP = a + b / r_e  [IWP g/m^2, r_e um].
EC92_BAND1_A = np.float32(3.448e-3)       # m^2 / g
EC92_BAND1_B = np.float32(2.431)          # m^2 um / g


def liquid_mass_extinction(re_um: float) -> np.float32:
    """Liquid mass-extinction coefficient (m^2/g): 1.5 / r_e, Q_ext = 2."""
    return TAU_PER_LWP_UM / np.float32(re_um)


def ice_mass_extinction(re_um: float) -> np.float32:
    """Ice mass-extinction coefficient (m^2/g): Ebert & Curry (1992) band 1.

    THE ice coefficient, everywhere: the volume renderers used to convert
    ice with the geometric-optics sphere at bulk ice density
    (1.5 / (rho_ice r_e) = 0.0545 m^2/g at 30 um) while the column optical
    depth used this fit (0.0845 m^2/g at 30 um) — a 1.55x disagreement
    between glimpse and witness/behold from the same file. Ebert & Curry is
    the one with a physical case for the visible band: it is fitted to real
    (nonspherical) ice cloud optics, whose area per unit mass exceeds the
    mass-equivalent sphere's, which is exactly what the geometric prefactor
    understates.
    """
    return EC92_BAND1_A + EC92_BAND1_B / np.float32(re_um)


def compute_extinction_field(lwc: np.ndarray, z: np.ndarray, re: float = 10.0,
                            iwc: np.ndarray = None, re_ice: float = 30.0) -> np.ndarray:
    """
    Compute extinction coefficient field from liquid and ice water content.

    Uses standard relationships for cloud optics with effective radius parameter.
    Total extinction is the sum of liquid and ice contributions.

    Parameters
    ----------
    lwc : ndarray (nx, ny, nz)
        Liquid water content (g/kg)
    z : ndarray (nz,)
        Heights (m)
    re : float, optional
        Liquid effective radius (microns, default: 10.0)
    iwc : ndarray (nx, ny, nz), optional
        Ice water content (g/kg). If None, only liquid extinction is computed.
    re_ice : float, optional
        Ice effective radius (microns, default: 30.0)

    Returns
    -------
    sigma_ext : ndarray (nx, ny, nz)
        Total extinction coefficient (m^-1) from liquid + ice
    """
    # Atmospheric properties
    R, T = 287.05, 280.0
    scale_height = 7000.0
    p0 = 101300.0

    # Pressure and density at each level
    pressures = p0 * np.exp(-z / scale_height)
    rho_air = (pressures / (R * T)).astype(np.float32)

    # Liquid water contribution. The 1.5 is 3 Q_ext / 4 with Q_ext = 2, the
    # geometric-optics limit — Petty (2006), Eq. 7.86. For r_e in um this is
    # 1.5 / r_e m^2/g, the same coefficient the column optical depth uses.
    lwc_g_m3 = lwc * rho_air[np.newaxis, np.newaxis, :]
    sigma_ext_liquid = liquid_mass_extinction(re) * lwc_g_m3

    # Ice water contribution (if present): Ebert & Curry (1992) band 1, the
    # same coefficient the column optical depth uses — see
    # ice_mass_extinction for why the geometric sphere it replaced is wrong.
    if iwc is not None:
        iwc_g_m3 = iwc * rho_air[np.newaxis, np.newaxis, :]
        sigma_ext_ice = ice_mass_extinction(re_ice) * iwc_g_m3
        sigma_ext = sigma_ext_liquid + sigma_ext_ice
    else:
        logger.info("No ice water content detected; using liquid-only extinction.")
        sigma_ext = sigma_ext_liquid

    return sigma_ext


def optical_depth_from_water_paths(
    iwp: np.ndarray,
    lwp: np.ndarray,
    swp: np.ndarray = None,
    liquid_re: float = 10.0,
    ice_re: float = 30.0,
    snow_re: float = 300.0
) -> np.ndarray:
    """
    Calculate optical depth from liquid, ice and snow water paths.

    Parameters
    ----------
    iwp : ndarray
        Ice water path (g/m²) - already vertically integrated
    lwp : ndarray
        Liquid water path (g/m²) - already vertically integrated
    swp : ndarray, optional
        Snow/precipitation water path (g/m²) - already vertically integrated (default: None)
    liquid_re : float
        Effective radius for liquid cloud droplets (microns, default: 10)
    ice_re : float
        Effective radius for ice crystals (microns, default: 30)
    snow_re : float
        Effective radius for snow particles (microns, default: 300)

    Returns
    -------
    tau : ndarray
        Total optical depth (unitless)

    Notes
    -----
    Liquid is the geometric-optics limit with Q_ext = 2 (Petty 2006, Eq. 7.86);
    ice is Ebert & Curry (1992) Table 2 band 1 exactly. See the module header.
    """
    # Liquid: tau = 1.5 * LWP / r_e — Petty (2006), Eq. 7.86 with Q_ext = 2.
    tau_liquid = lwp * liquid_mass_extinction(liquid_re)

    # Ice: tau = IWP * (a + b / r_e) — Ebert & Curry (1992), Table 2 band 1.
    tau_ice = iwp * ice_mass_extinction(ice_re)

    # Snow: same relationship as ice. r_e = 300 um is well outside the range
    # Ebert & Curry fit (ice cloud, r_e ~ 5-130 um); extrapolating rather than
    # splitting snow onto its own parameterization is a deliberate choice.
    tau_snow = np.zeros_like(tau_ice)
    if swp is not None:
        tau_snow = swp * ice_mass_extinction(snow_re)
    else:
        logger.info("No snow water path detected; using liquid+ice optical depth only.")

    tau_total = tau_liquid + tau_ice + tau_snow

    return tau_total


def vertically_integrated_optical_depth(lwc: np.ndarray, z: np.ndarray,
                           iwc: Optional[np.ndarray] = None,
                           swc: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate optical depth from 3D water content fields using SAM relationships.

    This integrates vertical water content and applies SAM optical depth formulas.

    Parameters
    ----------
    lwc : ndarray (nx, ny, nz)
        Liquid water content (g/kg)
    z : ndarray (nz,)
        Heights (m)
    iwc : ndarray, optional
        Ice water content (g/kg)
    swc : ndarray, optional
        Snow water content (g/kg)

    Returns
    -------
    tau : ndarray (nx, ny)
        Optical depth (unitless, 2D field)
    """
    # Atmospheric properties
    R, T = 287.05, 280.0
    scale_height = 7000.0
    p0 = 101300.0

    # Pressure and density at each level
    pressures = p0 * np.exp(-z / scale_height)
    rho_air = (pressures / (R * T)).astype(np.float32)

    # Cell thickness for cell-centred coordinates.
    # Interior cells: half-distance to neighbour below + half-distance above.
    # Boundary cells: half-distance to sole neighbour + that same half outward.
    spacing = np.diff(z)
    dz = np.empty_like(z, dtype=np.float32)
    dz[0] = spacing[0]                              # bottom boundary
    dz[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])   # interior
    dz[-1] = spacing[-1]                             # top boundary

    # Integrate water content vertically to get water paths
    water_path_liquid = (lwc * rho_air[np.newaxis, np.newaxis, :] *
                        dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    water_path_ice = np.zeros_like(water_path_liquid)
    water_path_snow = np.zeros_like(water_path_liquid)

    if iwc is not None:
        water_path_ice = (iwc * rho_air[np.newaxis, np.newaxis, :] *
                         dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    else:
        logger.info("No ice water content detected; optical depth uses liquid water only.")

    if swc is not None:
        water_path_snow = (swc * rho_air[np.newaxis, np.newaxis, :] *
                          dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    else:
        logger.info("No snow water content detected; optical depth excludes snow.")

    # Use generic optical depth relationships
    tau = optical_depth_from_water_paths(water_path_ice, water_path_liquid, water_path_snow)

    return tau
