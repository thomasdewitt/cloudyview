"""NetCDF I/O utilities and variable inference for CloudyView."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import xarray as xr


# Common variable names for liquid water
LIQUID_WATER_NAMES = ["qc", "QC", "ql", "QL", "QN", "qn", "LWC", "clw",
                       "cloud_liquid_water_mixing_ratio",
                       "liquid_water_content", "q_liquid", "lwc"]

# Common variable names for ice water
ICE_WATER_NAMES = ["qi", "QI", "qice", "QICE", "IWC", "cli",
                    "cloud_ice_mixing_ratio",
                    "ice_water_content", "q_ice", "iwc"]

AXIS_CANDIDATES = {
    "x": ["x", "lon", "longitude", "nx", "ni"],
    "y": ["y", "lat", "latitude", "ny", "nj"],
    "z": ["z", "height", "altitude", "level", "nz", "nk"],
}


def _normalize_group(group: Optional[str]) -> Optional[str]:
    """Normalize user-provided NetCDF group names."""
    if group in (None, "", ".", "/"):
        return None
    return str(group)


def _describe_group(group: Optional[str]) -> str:
    """Return a human-readable group label for errors and logging."""
    group = _normalize_group(group)
    return "the root group" if group is None else f"group '{group}'"


def load_data(filepath: str, group: Optional[str] = None) -> xr.Dataset:
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

    return xr.open_dataset(filepath, group=_normalize_group(group))


def infer_variable(
    ds: xr.Dataset,
    candidate_names: list,
    explicit_name: Optional[str] = None,
    variable_role: str = "variable",
    group: Optional[str] = None,
) -> Tuple[str, xr.DataArray]:
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

    if explicit_name is not None:
        if explicit_name in available_vars:
            return explicit_name, ds[explicit_name]
        raise ValueError(
            f"Could not find {variable_role} '{explicit_name}' in {_describe_group(group)}. "
            f"Available variables: {sorted(available_vars)}"
        )

    for name in candidate_names:
        if name in available_vars:
            return name, ds[name]

    raise ValueError(
        f"Could not find {variable_role} from {candidate_names} in {_describe_group(group)}. "
        f"Available variables: {sorted(available_vars)}"
    )


def infer_liquid_water(
    ds: xr.Dataset,
    explicit_name: Optional[str] = None,
    group: Optional[str] = None,
) -> Tuple[str, xr.DataArray]:
    """Infer liquid water variable from dataset."""
    return infer_variable(
        ds,
        LIQUID_WATER_NAMES,
        explicit_name=explicit_name,
        variable_role="liquid water variable",
        group=group,
    )


def infer_ice_water(
    ds: xr.Dataset,
    explicit_name: Optional[str] = None,
    group: Optional[str] = None,
) -> Tuple[str, xr.DataArray]:
    """Infer ice water variable from dataset (optional)."""
    try:
        return infer_variable(
            ds,
            ICE_WATER_NAMES,
            explicit_name=explicit_name,
            variable_role="ice water variable",
            group=group,
        )
    except ValueError:
        if explicit_name is not None:
            raise
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


def check_and_convert_units(data_array: xr.DataArray, var_name: str) -> xr.DataArray:
    """
    Check and convert water content units to g/kg.

    Parameters
    ----------
    data_array : xr.DataArray
        Data variable to check
    var_name : str
        Name of variable for error messages

    Returns
    -------
    xr.DataArray
        Data array with values in g/kg

    Raises
    ------
    ValueError
        If units are not g/kg, g/g, or kg/kg, or if units attribute is missing
    """
    # Get units attribute
    units = data_array.attrs.get('units', None)

    if units is None:
        raise ValueError(f"Variable {var_name} has no 'units' attribute. "
                        "Units must be specified as 'g/kg', 'g/g', or 'kg/kg'.")

    # Normalize units string (strip whitespace, handle case variations)
    units_normalized = units.strip().lower()

    if units_normalized == 'g/kg':
        # Already in correct units
        return data_array
    elif units_normalized == 'g/g' or units_normalized == 'kg/kg':
        # Convert from g/g or kg/kg to g/kg (multiply by 1000)
        data_array = data_array * 1000.0
        data_array.attrs['units'] = 'g/kg'
        return data_array
    else:
        raise ValueError(f"Variable {var_name} has unsupported units: {units}. "
                        "Expected 'g/kg', 'g/g', or 'kg/kg'.")


def _drop_time_dims(data_array: xr.DataArray) -> xr.DataArray:
    """Remove any single-length time dimensions from a data array."""
    time_dims = [d for d in data_array.dims if "time" in d.lower()]
    for time_dim in time_dims:
        data_array = data_array.isel({time_dim: 0}, drop=True)
    return data_array


def _resolve_spatial_dims(
    data_array: xr.DataArray,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    z_dim: Optional[str] = None,
) -> Tuple[xr.DataArray, Dict[str, str]]:
    """Resolve x/y/z dimensions, using explicit overrides when provided."""
    data_array = _drop_time_dims(data_array)
    dims = list(data_array.dims)

    if len(dims) != 3:
        raise ValueError(f"Expected 3 spatial dimensions, got {len(dims)}: {dims}")

    resolved = {}
    used_dims = set()

    for axis, explicit_dim in {"x": x_dim, "y": y_dim, "z": z_dim}.items():
        if explicit_dim is None:
            continue
        if explicit_dim not in dims:
            raise ValueError(
                f"Requested {axis}-dimension '{explicit_dim}' was not found in {dims}."
            )
        if explicit_dim in used_dims:
            raise ValueError(
                f"Dimension '{explicit_dim}' was assigned to more than one axis."
            )
        resolved[axis] = explicit_dim
        used_dims.add(explicit_dim)

    for axis, candidates in AXIS_CANDIDATES.items():
        if axis in resolved:
            continue
        for dim in dims:
            if dim in used_dims:
                continue
            dim_lower = dim.lower()
            if any(candidate.lower() == dim_lower for candidate in candidates):
                resolved[axis] = dim
                used_dims.add(dim)
                break

    unresolved_axes = [axis for axis in ("x", "y", "z") if axis not in resolved]
    remaining_dims = [dim for dim in dims if dim not in used_dims]
    if len(unresolved_axes) == 1 and len(remaining_dims) == 1:
        resolved[unresolved_axes[0]] = remaining_dims[0]
        used_dims.add(remaining_dims[0])
        unresolved_axes = []

    if unresolved_axes:
        raise ValueError(
            f"Could not infer x, y, z dimensions from {dims}. "
            f"Resolved so far: {resolved}. "
            "Use explicit x/y/z dimension overrides."
        )

    return data_array, resolved


def standardize_dims(
    data_array: xr.DataArray,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    z_dim: Optional[str] = None,
) -> xr.DataArray:
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
    data_array, dims_map = _resolve_spatial_dims(
        data_array,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )

    # Transpose to (x, y, z) order and rename
    data_array = data_array.transpose(dims_map["x"], dims_map["y"], dims_map["z"])
    data_array = data_array.rename(
        {
            dims_map["x"]: "x",
            dims_map["y"]: "y",
            dims_map["z"]: "z",
        }
    )

    return data_array


def _unique_datasets(datasets: Iterable[xr.Dataset]) -> list:
    """Deduplicate datasets while preserving order."""
    unique = []
    seen_ids = set()
    for dataset in datasets:
        if dataset is None:
            continue
        dataset_id = id(dataset)
        if dataset_id in seen_ids:
            continue
        unique.append(dataset)
        seen_ids.add(dataset_id)
    return unique


def _matches_axis(var: xr.DataArray, axis_dim: str, axis_size: int) -> bool:
    """Return True when a 1D variable can describe the requested axis."""
    if var.ndim != 1 or not var.dims:
        return False
    dim_name = var.dims[0]
    return dim_name == axis_dim or var.sizes.get(dim_name) == axis_size


def _lookup_named_coord(
    data_array: xr.DataArray,
    datasets: Iterable[xr.Dataset],
    coord_name: str,
    axis_dim: str,
):
    """Look up a specific coordinate/1D variable by name."""
    axis_size = data_array.sizes[axis_dim]

    if coord_name in data_array.coords:
        coord = data_array.coords[coord_name]
        if _matches_axis(coord, axis_dim, axis_size):
            return coord.values

    for dataset in datasets:
        if coord_name in dataset:
            coord = dataset[coord_name]
            if _matches_axis(coord, axis_dim, axis_size):
                return coord.values

    return None


def _lookup_axis_coord(
    data_array: xr.DataArray,
    datasets: Iterable[xr.Dataset],
    axis: str,
    axis_dim: str,
):
    """Auto-detect a coordinate array for one axis."""
    for candidate in AXIS_CANDIDATES[axis]:
        coord = _lookup_named_coord(data_array, datasets, candidate, axis_dim)
        if coord is not None:
            return coord

    axis_size = data_array.sizes[axis_dim]

    if axis_dim in data_array.coords:
        coord = data_array.coords[axis_dim]
        if _matches_axis(coord, axis_dim, axis_size):
            return coord.values

    for dataset in datasets:
        if axis_dim in dataset.coords:
            coord = dataset.coords[axis_dim]
            if _matches_axis(coord, axis_dim, axis_size):
                return coord.values

        for collection in (dataset.coords, dataset.data_vars):
            for var_name in collection:
                coord = collection[var_name]
                if _matches_axis(coord, axis_dim, axis_size) and coord.dims == (axis_dim,):
                    return coord.values

    return None


def _available_1d_names(datasets: Iterable[xr.Dataset]) -> list:
    """Collect the names of all 1D coordinates/data variables for error messages."""
    names = set()
    for dataset in datasets:
        for collection in (dataset.coords, dataset.data_vars):
            for var_name in collection:
                if collection[var_name].ndim == 1:
                    names.add(var_name)
    return sorted(names)


def _extract_coords(
    data_array: xr.DataArray,
    datasets: Iterable[xr.Dataset],
    x_coord_name: Optional[str] = None,
    y_coord_name: Optional[str] = None,
    z_coord_name: Optional[str] = None,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    z_dim: Optional[str] = None,
):
    """
    Extract x, y, z coordinate arrays from a data array, handling cases where
    coordinate variable names differ from dimension names (e.g., dim 'ni' with
    coord 'x(ni)').
    """
    data_array, dims_map = _resolve_spatial_dims(
        data_array,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )
    datasets = _unique_datasets(datasets)

    coord_overrides = {
        "x": x_coord_name,
        "y": y_coord_name,
        "z": z_coord_name,
    }
    coords = {}

    for axis in ("x", "y", "z"):
        axis_dim = dims_map[axis]
        coord_name = coord_overrides[axis]
        if coord_name is not None:
            coord = _lookup_named_coord(data_array, datasets, coord_name, axis_dim)
            if coord is None:
                raise ValueError(
                    f"Could not find {axis}-coordinate '{coord_name}' for dimension "
                    f"'{axis_dim}'. Available 1D coordinates/variables: "
                    f"{_available_1d_names(datasets)}"
                )
        else:
            coord = _lookup_axis_coord(data_array, datasets, axis, axis_dim)
        coords[axis] = coord

    x_coord = coords["x"]
    y_coord = coords["y"]
    z_coord = coords["z"]
    return x_coord, y_coord, z_coord


def load_and_validate(
    filepath: str,
    liquid_water_var: Optional[str] = None,
    ice_water_var: Optional[str] = None,
    dataset_group: Optional[str] = None,
    liquid_water_group: Optional[str] = None,
    ice_water_group: Optional[str] = None,
    coords_group: Optional[str] = None,
    x_coord_name: Optional[str] = None,
    y_coord_name: Optional[str] = None,
    z_coord_name: Optional[str] = None,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    z_dim: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load NetCDF file and validate it, inferring variable names and checking units.

    Performs the following checks and conversions:
    1. Loads NetCDF file and infers liquid/ice water variables
    2. Validates 3D single-timestep data
    3. Standardizes dimension names to (x, y, z)
    4. Checks units and converts g/g to g/kg if needed

    Parameters
    ----------
    filepath : str
        Path to NetCDF file
    liquid_water_var : str, optional
        Explicit liquid water variable name override
    ice_water_var : str, optional
        Explicit ice water variable name override
    dataset_group : str, optional
        Default NetCDF group to search for variables and coordinates
    liquid_water_group : str, optional
        NetCDF group containing the liquid water variable
    ice_water_group : str, optional
        NetCDF group containing the ice water variable
    coords_group : str, optional
        NetCDF group containing x/y/z coordinate arrays
    x_coord_name, y_coord_name, z_coord_name : str, optional
        Explicit coordinate variable names
    x_dim, y_dim, z_dim : str, optional
        Explicit dimension names for x/y/z axes

    Returns
    -------
    dict
        Dictionary with keys:
        - 'dataset': xr.Dataset
        - 'liquid_water_var': str (variable name)
        - 'liquid_water_data': xr.DataArray (with standardized (x, y, z) dims, units in g/kg)
        - 'ice_water_var': str or None
        - 'ice_water_data': xr.DataArray or None (with standardized dims, units in g/kg)
        - 'filepath': str
        - 'x_coord': ndarray (x coordinates)
        - 'y_coord': ndarray (y coordinates)
        - 'z_coord': ndarray (z coordinates)

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If validation fails or units are missing/unsupported
    """
    dataset_group = _normalize_group(dataset_group)
    liquid_water_group = _normalize_group(liquid_water_group)
    ice_water_group = _normalize_group(ice_water_group)
    coords_group = _normalize_group(coords_group)

    liquid_water_group = (
        liquid_water_group if liquid_water_group is not None else dataset_group
    )
    ice_water_group = (
        ice_water_group if ice_water_group is not None else dataset_group
    )
    coords_group = coords_group if coords_group is not None else liquid_water_group

    dataset_cache = {}

    def get_dataset(group: Optional[str]) -> xr.Dataset:
        group = _normalize_group(group)
        if group not in dataset_cache:
            dataset_cache[group] = load_data(filepath, group=group)
        return dataset_cache[group]

    root_ds = get_dataset(None)
    lw_ds = get_dataset(liquid_water_group)
    iw_ds = get_dataset(ice_water_group)
    coord_ds = get_dataset(coords_group)

    # Infer liquid water variable (required)
    lw_var, lw_data = infer_liquid_water(
        lw_ds,
        explicit_name=liquid_water_var,
        group=liquid_water_group,
    )
    validate_data(lw_ds, lw_data, lw_var)

    # Extract coordinate arrays before standardizing dims (dim names may differ from coord names)
    x_coord, y_coord, z_coord = _extract_coords(
        lw_data,
        datasets=[coord_ds, lw_ds, root_ds],
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        z_coord_name=z_coord_name,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )

    # Standardize dimensions to (x, y, z)
    lw_data = standardize_dims(lw_data, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)

    # Check and convert units to g/kg
    lw_data = check_and_convert_units(lw_data, lw_var)

    # Infer ice water variable (optional)
    iw_var, iw_data = infer_ice_water(
        iw_ds,
        explicit_name=ice_water_var,
        group=ice_water_group,
    )
    if iw_data is not None:
        validate_data(iw_ds, iw_data, iw_var)
        # Standardize dimensions
        iw_data = standardize_dims(iw_data, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
        # Check and convert units to g/kg
        iw_data = check_and_convert_units(iw_data, iw_var)
        if iw_data.shape != lw_data.shape:
            raise ValueError(
                "Liquid water and ice water arrays must have identical spatial shapes. "
                f"Got liquid={lw_data.shape}, ice={iw_data.shape}."
            )

    if x_coord is None or y_coord is None or z_coord is None:
        raise ValueError(
            "Could not determine x/y/z coordinate arrays from the input dataset. "
            "Coordinate variables for all three dimensions are required."
        )

    return {
        'dataset': lw_ds,
        'liquid_water_var': lw_var,
        'liquid_water_data': lw_data,
        'ice_water_var': iw_var,
        'ice_water_data': iw_data,
        'filepath': str(filepath),
        'x_coord': x_coord,
        'y_coord': y_coord,
        'z_coord': z_coord,
    }
