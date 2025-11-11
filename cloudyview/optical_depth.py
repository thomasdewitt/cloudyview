"""Optical depth and extinction coefficient calculations for CloudyView.

Uses physical models for cloud radiative properties.
"""

import xarray as xr
import numpy as np
from typing import Optional, Tuple


def calculate_optical_depth(
    liquid_water: xr.DataArray,
    ice_water: Optional[xr.DataArray] = None,
    effective_radius_liquid: float = 10.0,
    effective_radius_ice: float = 50.0,
) -> xr.DataArray:
    """
    Calculate optical depth from cloud water content.

    Parameters
    ----------
    liquid_water : xr.DataArray
        Liquid water mixing ratio (g/kg)
    ice_water : xr.DataArray, optional
        Ice water mixing ratio (g/kg)
    effective_radius_liquid : float
        Effective radius of liquid droplets (microns)
    effective_radius_ice : float
        Effective radius of ice crystals (microns)

    Returns
    -------
    xr.DataArray
        Optical depth field

    Notes
    -----
    This is a placeholder. Replace with your actual optical depth calculation.
    """
    # Placeholder: simple approximation
    # TODO: Replace with actual optical depth calculations from your code

    tau = np.zeros_like(liquid_water.values)

    if liquid_water is not None:
        # Simple approximation: tau ≈ constant * LWC * path_length
        tau += 0.1 * liquid_water.values

    if ice_water is not None:
        # Ice contribution (typically smaller)
        tau += 0.05 * ice_water.values

    # Create output with same coordinates/dims as input
    optical_depth = liquid_water.copy(data=tau)
    optical_depth.name = "optical_depth"
    optical_depth.attrs['long_name'] = "Optical Depth"

    return optical_depth


def column_optical_depth(
    optical_depth_3d: xr.DataArray,
    axis: int = 0,
) -> xr.DataArray:
    """
    Integrate optical depth vertically to get column values.

    Parameters
    ----------
    optical_depth_3d : xr.DataArray
        3D optical depth field
    axis : int
        Axis along which to integrate (default: 0, vertical)

    Returns
    -------
    xr.DataArray
        2D column optical depth field
    """
    # Simple integration along specified axis
    col_od = optical_depth_3d.sum(axis=axis)
    col_od.name = "column_optical_depth"
    col_od.attrs['long_name'] = "Column Optical Depth"

    return col_od


def compute_extinction_field(lwc: np.ndarray, z: np.ndarray, re: float = 10.0) -> np.ndarray:
    """
    Compute extinction coefficient field from liquid water content.

    Uses standard relationships for cloud optics with effective radius parameter.

    Parameters
    ----------
    lwc : ndarray (nx, ny, nz)
        Liquid water content (g/kg)
    z : ndarray (nz,)
        Heights (m)
    re : float, optional
        Effective radius (microns, default: 10.0)

    Returns
    -------
    sigma_ext : ndarray (nx, ny, nz)
        Extinction coefficient (m^-1)
    """
    # Atmospheric properties
    g, R, T = 9.81, 287.05, 280.0
    scale_height = 7000.0
    p0 = 101300.0

    # Pressure and density at each level
    pressures = p0 * np.exp(-z / scale_height)
    rho_air = pressures / (R * T)

    # LWC in g/m^3
    lwc_g_m3 = lwc * rho_air[np.newaxis, np.newaxis, :]

    # Extinction coefficient
    rho_water = 1e6  # g/m³
    r_eff_m = re * 1e-6  # Convert μm to m
    sigma_ext = 1.5 * lwc_g_m3 / (rho_water * r_eff_m)

    return sigma_ext


def integrate_water_content(lwc: np.ndarray, z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Integrate liquid water content vertically to get liquid water path (LWP).

    Parameters
    ----------
    lwc : ndarray
        Liquid water content (g/kg)
    z : ndarray
        Heights (m)
    axis : int
        Axis along which to integrate (default: -1, last axis = vertical)

    Returns
    -------
    lwp : ndarray
        Liquid water path (g/m²)
    """
    # Atmospheric properties
    R, T = 287.05, 280.0
    scale_height = 7000.0
    p0 = 101300.0

    # Pressure and density at each level
    pressures = p0 * np.exp(-z / scale_height)
    rho_air = pressures / (R * T)

    # Calculate dz between levels
    dz = np.diff(z, axis=0)
    # Pad dz to match z dimensions
    dz = np.concatenate([dz, [dz[-1]]])

    # Water path: integrate rho * q * dz along vertical axis
    water_path = lwc * rho_air[np.newaxis, np.newaxis, :] * dz[np.newaxis, np.newaxis, :]
    lwp = np.sum(water_path, axis=axis)

    return lwp


def optical_depth_from_water_paths(
    iwp: np.ndarray,
    lwp: np.ndarray,
    swp: np.ndarray = None,
    liquid_re: float = 10.0,
    ice_re: float = 30.0,
    snow_re: float = 300.0
) -> np.ndarray:
    """
    Calculate optical depth from water paths using generic relationships.

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
    tau_liquid = lwp / (0.6292 * liquid_re)

    # Ice: IWP = 0.350 * tau * re (g/m²)
    tau_ice = iwp / (0.350 * ice_re)

    # Snow: same relationship as ice
    tau_snow = np.zeros_like(tau_ice)
    if swp is not None:
        tau_snow = swp / (0.350 * snow_re)

    tau_total = tau_liquid + tau_ice + tau_snow

    return tau_total


def opacity_field_3d(od_3d: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calculate 3D opacity field showing opacity above each point.

    For each grid point, calculates the cumulative opacity from that point
    to the top of the domain.

    Parameters
    ----------
    od_3d : ndarray (nx, ny, nz)
        3D optical depth field
    axis : int
        Axis along which to integrate (default: -1, vertical/z)

    Returns
    -------
    opacity_3d : ndarray (nx, ny, nz)
        3D opacity field (0=transparent, 1=opaque)
    """
    # Cumulative sum from top downwards
    cumsum = np.cumsum(od_3d[:, :, ::-1], axis=axis)[:, :, ::-1]
    # Convert optical depth to opacity
    opacity_3d = 1.0 - np.exp(-cumsum)
    return opacity_3d


def optical_depth_from_lwc(lwc: np.ndarray, z: np.ndarray,
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
    rho_air = pressures / (R * T)

    # Calculate dz between levels
    dz = np.diff(z)
    # Pad dz to match z dimensions
    dz = np.concatenate([dz, [dz[-1]]])

    # Integrate water content vertically to get water paths
    water_path_liquid = (lwc * rho_air[np.newaxis, np.newaxis, :] *
                        dz[np.newaxis, np.newaxis, :]).sum(axis=-1)
    water_path_ice = np.zeros_like(water_path_liquid)
    water_path_snow = np.zeros_like(water_path_liquid)

    if iwc is not None:
        water_path_ice = (iwc * rho_air[np.newaxis, np.newaxis, :] *
                         dz[np.newaxis, np.newaxis, :]).sum(axis=-1)

    if swc is not None:
        water_path_snow = (swc * rho_air[np.newaxis, np.newaxis, :] *
                          dz[np.newaxis, np.newaxis, :]).sum(axis=-1)

    # Use generic optical depth relationships
    tau = optical_depth_from_water_paths(water_path_ice, water_path_liquid, water_path_snow)

    return tau
