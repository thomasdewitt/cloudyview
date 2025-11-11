"""NetCDF I/O utilities and variable inference for CloudyView."""

import xarray as xr
from pathlib import Path
from typing import Tuple, Dict, Any


# Common variable names for liquid water
LIQUID_WATER_NAMES = ["qc", "QC", "ql", "QL", "QN", "qn", "LWC",
                       "cloud_liquid_water_mixing_ratio",
                       "liquid_water_content", "q_liquid", "lwc"]

# Common variable names for ice water
ICE_WATER_NAMES = ["qi", "QI", "qice", "QICE", "IWC",
                    "cloud_ice_mixing_ratio",
                    "ice_water_content", "q_ice", "iwc"]


def load_data(filepath: str) -> xr.Dataset:
    """
    Load NetCDF file using xarray.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file

    Returns
    -------
    xr.Dataset
        Loaded dataset

    Raises
    ------
    FileNotFoundError
        If file does not exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    return xr.open_dataset(filepath)


def infer_variable(ds: xr.Dataset, candidate_names: list) -> Tuple[str, xr.DataArray]:
    """
    Infer variable from dataset by trying common naming conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    candidate_names : list
        List of variable names to try

    Returns
    -------
    str, xr.DataArray
        Variable name found and the data array

    Raises
    ------
    ValueError
        If no matching variable found
    """
    available_vars = set(ds.data_vars)

    for name in candidate_names:
        if name in available_vars:
            return name, ds[name]

    raise ValueError(f"Could not find variable from {candidate_names}. "
                     f"Available variables: {sorted(available_vars)}")


def infer_liquid_water(ds: xr.Dataset) -> Tuple[str, xr.DataArray]:
    """Infer liquid water variable from dataset."""
    return infer_variable(ds, LIQUID_WATER_NAMES)


def infer_ice_water(ds: xr.Dataset) -> Tuple[str, xr.DataArray]:
    """Infer ice water variable from dataset (optional)."""
    try:
        return infer_variable(ds, ICE_WATER_NAMES)
    except ValueError:
        # Ice water is optional
        return None, None


def validate_data(ds: xr.Dataset, data_var: xr.DataArray, var_name: str) -> None:
    """
    Validate dataset properties.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    data_var : xr.DataArray
        Data variable to validate
    var_name : str
        Name of variable being validated

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check single timestep
    time_dims = [d for d in data_var.dims if 'time' in d.lower()]
    if time_dims:
        time_dim = time_dims[0]
        if data_var.sizes[time_dim] > 1:
            raise ValueError(f"Data has {data_var.sizes[time_dim]} timesteps. "
                           "Only single-timestep files are supported.")

    # Check 3D spatial data (should have at least 3 non-time dimensions)
    spatial_dims = [d for d in data_var.dims if 'time' not in d.lower()]
    if len(spatial_dims) < 3:
        raise ValueError(f"Data has {len(spatial_dims)} spatial dimensions. "
                       "3D spatial data is required (e.g., x, y, z).")


def standardize_dims(data_array: xr.DataArray) -> xr.DataArray:
    """
    Standardize dimension names to (x, y, z).

    Maps common dimension names to standard (x, y, z) format:
    - Horizontal: (x, y) from (x/lon/nx/longitude, y/lat/ny/latitude)
    - Vertical: (z) from (z/height/nz/altitude/level)
    - Removes time dimension (single timestep already validated)

    Parameters
    ----------
    data_array : xr.DataArray
        Input data array with arbitrary dimension names

    Returns
    -------
    xr.DataArray
        Data array with standardized (x, y, z) dimensions

    Raises
    ------
    ValueError
        If dimensions cannot be inferred
    """
    # Drop time dimension if present (single timestep already validated)
    time_dims = [d for d in data_array.dims if 'time' in d.lower()]
    for time_dim in time_dims:
        data_array = data_array.isel({time_dim: 0}, drop=True)

    # Get spatial dimensions
    dims = list(data_array.dims)

    if len(dims) != 3:
        raise ValueError(f"Expected 3 spatial dimensions, got {len(dims)}: {dims}")

    # Map dimension names
    x_candidates = ['x', 'lon', 'longitude', 'nx']
    y_candidates = ['y', 'lat', 'latitude', 'ny']
    z_candidates = ['z', 'height', 'altitude', 'level', 'nz']

    x_dim = None
    y_dim = None
    z_dim = None

    for dim in dims:
        dim_lower = dim.lower()
        if x_dim is None and any(cand.lower() == dim_lower for cand in x_candidates):
            x_dim = dim
        elif y_dim is None and any(cand.lower() == dim_lower for cand in y_candidates):
            y_dim = dim
        elif z_dim is None and any(cand.lower() == dim_lower for cand in z_candidates):
            z_dim = dim

    if x_dim is None or y_dim is None or z_dim is None:
        raise ValueError(f"Could not infer x, y, z dimensions from {dims}. "
                        f"Found: x={x_dim}, y={y_dim}, z={z_dim}")

    # Transpose to (x, y, z) order and rename
    data_array = data_array.transpose(x_dim, y_dim, z_dim)
    data_array = data_array.rename({x_dim: 'x', y_dim: 'y', z_dim: 'z'})

    return data_array


def load_and_validate(filepath: str) -> Dict[str, Any]:
    """
    Load NetCDF file and validate it, inferring variable names.

    Parameters
    ----------
    filepath : str
        Path to NetCDF file

    Returns
    -------
    dict
        Dictionary with keys:
        - 'dataset': xr.Dataset
        - 'liquid_water_var': str (variable name)
        - 'liquid_water_data': xr.DataArray (with standardized (x, y, z) dims)
        - 'ice_water_var': str or None
        - 'ice_water_data': xr.DataArray or None (with standardized dims)
        - 'filepath': str
        - 'x_coord': ndarray (x coordinates)
        - 'y_coord': ndarray (y coordinates)
        - 'z_coord': ndarray (z coordinates)

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If validation fails
    """
    # Load dataset
    ds = load_data(filepath)

    # Infer liquid water variable (required)
    lw_var, lw_data = infer_liquid_water(ds)
    validate_data(ds, lw_data, lw_var)

    # Standardize dimensions to (x, y, z)
    lw_data = standardize_dims(lw_data)

    # Infer ice water variable (optional)
    iw_var, iw_data = infer_ice_water(ds)
    if iw_data is not None:
        validate_data(ds, iw_data, iw_var)
        # Standardize dimensions
        iw_data = standardize_dims(iw_data)

    # Extract coordinate arrays
    x_coord = lw_data.coords['x'].values if 'x' in lw_data.coords else None
    y_coord = lw_data.coords['y'].values if 'y' in lw_data.coords else None
    z_coord = lw_data.coords['z'].values if 'z' in lw_data.coords else None

    return {
        'dataset': ds,
        'liquid_water_var': lw_var,
        'liquid_water_data': lw_data,
        'ice_water_var': iw_var,
        'ice_water_data': iw_data,
        'filepath': str(filepath),
        'x_coord': x_coord,
        'y_coord': y_coord,
        'z_coord': z_coord,
    }
