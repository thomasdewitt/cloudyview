"""Optical depth and extinction coefficient calculations for CloudyView.

Uses physical models for cloud radiative properties.
"""

import numpy as np
from typing import Optional


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

    # Liquid water contribution
    lwc_g_m3 = lwc * rho_air[np.newaxis, np.newaxis, :]
    rho_water = 1e6  # g/m³
    r_eff_liquid_m = re * 1e-6  # Convert μm to m
    sigma_ext_liquid = np.float32(1.5 / (rho_water * r_eff_liquid_m)) * lwc_g_m3

    # Ice water contribution (if present)
    if iwc is not None:
        iwc_g_m3 = iwc * rho_air[np.newaxis, np.newaxis, :]
        rho_ice = 917e3  # g/m³ (ice density ~917 kg/m³)
        r_eff_ice_m = re_ice * 1e-6  # Convert μm to m
        sigma_ext_ice = np.float32(1.5 / (rho_ice * r_eff_ice_m)) * iwc_g_m3
        sigma_ext = sigma_ext_liquid + sigma_ext_ice
    else:
        print("  No ice water content detected; using liquid-only extinction.")
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
    Calculate optical depth from water paths using relationships from Steve Krueger.

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
    Uses empirical relationships between water path, effective radius, and optical depth.
    Default values derived from standard cloud optics literature.
    """
    # Liquid water: LWP = 0.6292 * tau * re (g/m²)
    tau_liquid = lwp / np.float32(0.6292 * liquid_re)

    # Ice: IWP = 0.350 * tau * re (g/m²)
    tau_ice = iwp / np.float32(0.350 * ice_re)

    # Snow: same relationship as ice
    tau_snow = np.zeros_like(tau_ice)
    if swp is not None:
        tau_snow = swp / np.float32(0.350 * snow_re)
    else:
        print("  No snow water path detected; using liquid+ice optical depth only.")

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

    # Calculate dz between levels
    dz = np.diff(z)
    # Pad dz to match z dimensions
    dz = np.concatenate([dz, [dz[-1]]]).astype(np.float32)

    # Integrate water content vertically to get water paths
    water_path_liquid = (lwc * rho_air[np.newaxis, np.newaxis, :] *
                        dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    water_path_ice = np.zeros_like(water_path_liquid)
    water_path_snow = np.zeros_like(water_path_liquid)

    if iwc is not None:
        water_path_ice = (iwc * rho_air[np.newaxis, np.newaxis, :] *
                         dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    else:
        print("  No ice water content detected; optical depth uses liquid water only.")

    if swc is not None:
        water_path_snow = (swc * rho_air[np.newaxis, np.newaxis, :] *
                          dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    else:
        print("  No snow water content detected; optical depth excludes snow.")

    # Use generic optical depth relationships
    tau = optical_depth_from_water_paths(water_path_ice, water_path_liquid, water_path_snow)

    return tau
