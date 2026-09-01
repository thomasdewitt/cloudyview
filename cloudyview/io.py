"""NetCDF I/O utilities and variable inference for CloudyView."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# Common variable names for liquid water. Order matters: first hit wins, and
# matching is exact-case — `qc` really does beat `QC`.
#
# `QN`/`qn` are deliberately ABSENT. SAM writes QN for total non-precipitating
# condensate — cloud water AND cloud ice together — so it is not the liquid
# variable, and inferring it as one renders every ice cloud as water. A SAM
# run whose only condensate is QN fails to infer and the error lists what the
# file holds, so it can still be chosen with --liquid-water-var by someone who
# knows what their run wrote. (Mirrors web/soar/ingest/netcdf.js.)
LIQUID_WATER_NAMES = ["qc", "QC", "ql", "QL", "LWC", "clw",
                       "cloud_liquid_water_mixing_ratio",
                       "liquid_water_content", "q_liquid", "lwc"]

# Common variable names for ice water
ICE_WATER_NAMES = ["qi", "QI", "qice", "QICE", "IWC", "cli",
                    "cloud_ice_mixing_ratio",
                    "ice_water_content", "q_ice", "iwc"]

# Dimension matching, unlike variable matching, is case-insensitive.
#
# Beyond the bare axis letters this covers:
#   - the staggered/centred suffixes t, s, h and c. SAM writes cell centres
#     as xt/yt/zt and cell edges as xs/ys/zs; MPAS/CM1 use xh/zh for centres
#     and some tools write xc/yc. All twelve combinations are listed rather
#     than derived from a regex, so that adding a suffix is a visible edit
#     and not a rule that quietly starts matching something new.
#   - WRF's spelled-out dimensions and their _stag variants.
# A name absent from here is not a failure — it falls to the coordinate
# metadata rules below (see _axis_from_attrs), and only then to position.
# Kept in step with web/soar/ingest/netcdf.js, which is the reference copy.
AXIS_CANDIDATES = {
    "x": ["x", "xt", "xs", "xh", "xc", "lon", "longitude", "nx", "ni",
          "west_east", "west_east_stag"],
    "y": ["y", "yt", "ys", "yh", "yc", "lat", "latitude", "ny", "nj",
          "south_north", "south_north_stag"],
    "z": ["z", "zt", "zs", "zh", "zc", "height", "altitude", "level", "lev",
          "nz", "nk", "bottom_top", "bottom_top_stag", "plev", "pressure",
          "model_level_number"],
}

_STANDARD_NAME_AXIS = {
    "longitude": "x", "grid_longitude": "x", "projection_x_coordinate": "x",
    "latitude": "y", "grid_latitude": "y", "projection_y_coordinate": "y",
    "height": "z", "altitude": "z", "air_pressure": "z",
    "model_level_number": "z",
    "atmosphere_hybrid_sigma_pressure_coordinate": "z",
    "atmosphere_sigma_coordinate": "z",
    "height_above_mean_sea_level": "z",
    "height_above_reference_ellipsoid": "z",
}

# Units that name an axis on their own.
#
# The horizontal ones are unambiguous: nothing but a longitude is in
# degrees_east. The vertical ones are NOT — a SAM x axis is in metres too —
# which is why units are consulted only for axes still unclaimed after names
# and after the `axis`/`standard_name` attributes, and why two dimensions
# both claiming z by units is left unresolved rather than settled by
# whichever came first.
_UNITS_AXIS = {
    "degrees_east": "x", "degree_east": "x", "degrees_e": "x", "degree_e": "x",
    "degrees_north": "y", "degree_north": "y", "degrees_n": "y",
    "degree_n": "y",
    "m": "z", "metre": "z", "metres": "z", "meter": "z", "meters": "z",
    "km": "z", "kilometre": "z", "kilometres": "z", "kilometer": "z",
    "kilometers": "z",
    "pa": "z", "hpa": "z", "mb": "z", "millibar": "z", "millibars": "z",
    "level": "z", "levels": "z", "sigma": "z", "1": "z",
}

# Metres per unit of a coordinate, or None when it is not a length at all.
_LENGTH_UNITS = {
    "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
    "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimetre": 0.01, "centimetres": 0.01,
    "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimetre": 0.001, "millimetres": 0.001,
    "millimeter": 0.001, "millimeters": 0.001,
}

# Units that are recognized and definitely NOT lengths: they can never place
# a field in space, so no --coord-units override can rescue them.
_NON_LENGTH_UNITS = {
    "degrees_east", "degree_east", "degrees_e", "degree_e",
    "degrees_north", "degree_north", "degrees_n", "degree_n",
    "degrees", "degree",
    "pa", "hpa", "mb", "millibar", "millibars",
    "level", "levels", "sigma", "1", "index",
}


def _metres_per_unit(units) -> Optional[float]:
    """Metres per unit for a recognized length units string, else None.

    A qualifier after the unit — 'm AGL', 'meters above sea level' — is
    allowed as long as it carries no digits or slashes, which would make it
    an exponent or a rate ('m s-1') rather than a datum qualifier.
    """
    if units is None:
        return None
    text = str(units).strip().lower()
    direct = _LENGTH_UNITS.get(text)
    if direct is not None:
        return direct
    head, _, tail = text.partition(" ")
    tail = tail.strip()
    if tail and not any(ch.isdigit() or ch == "/" for ch in tail):
        return _LENGTH_UNITS.get(head)
    return None


def _is_non_length_unit(units) -> bool:
    """True when a units string is recognized as definitely not a length."""
    text = str(units).strip().lower()
    return text in _NON_LENGTH_UNITS or text.startswith("degree")


def _axis_from_attrs(attrs) -> Optional[Tuple[str, str]]:
    """Which axis a coordinate variable's own metadata claims, or None.

    CF gives three ways of saying it and they are not equally trustworthy, so
    the answer is tagged with which rule fired: "axis" (the `axis` attribute,
    says exactly this), "standard_name" (a CF name only one axis can carry),
    then "units" (weakest — metres suggest the vertical, but a Cartesian x is
    in metres too).
    """
    axis = attrs.get("axis")
    if isinstance(axis, str) and axis.strip().lower() in ("x", "y", "z"):
        return axis.strip().lower(), "axis"
    standard = attrs.get("standard_name")
    if isinstance(standard, str):
        a = _STANDARD_NAME_AXIS.get(standard.strip().lower())
        if a:
            return a, "standard_name"
    units = attrs.get("units")
    if units is not None:
        a = _UNITS_AXIS.get(str(units).strip().lower())
        if a:
            return a, "units"
    return None


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


def validate_data(ds: xr.Dataset, data_var: xr.DataArray, var_name: str,
                  timestep: Optional[int] = None) -> None:
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
    timestep : int, optional
        The caller's chosen step for a multi-timestep file; without it a
        multi-step file is an error (range checking happens at selection,
        in `_drop_time_dims`)

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check single timestep
    time_dims = [d for d in data_var.dims if 'time' in d.lower()]
    if time_dims:
        time_dim = time_dims[0]
        if data_var.sizes[time_dim] > 1 and timestep is None:
            raise ValueError(f"Data has {data_var.sizes[time_dim]} timesteps "
                           f"in '{time_dim}'. Pass --timestep to choose one.")

    # Check 3D spatial data (should have at least 3 non-time dimensions)
    spatial_dims = [d for d in data_var.dims if 'time' not in d.lower()]
    if len(spatial_dims) < 3:
        raise ValueError(f"Data has {len(spatial_dims)} spatial dimensions. "
                       "3D spatial data is required (e.g., x, y, z).")

    # Each spatial dimension must have at least 2 points for grid spacing
    for dim in spatial_dims:
        if data_var.sizes[dim] < 2:
            raise ValueError(
                f"Dimension '{dim}' has only {data_var.sizes[dim]} point. "
                "At least 2 points per spatial dimension are required."
            )


_UNITS_EXPONENT_RE = re.compile(r"\s*(?:\*\*|\^)\s*-\s*1\b")
_UNITS_RATIO_RE = re.compile(r"^(\S+)\s+(\S+?)\s*-\s*1$")


def _normalize_units(units: str) -> str:
    """Fold the CF spellings of a mixing-ratio unit onto the slash form.

    CF writes a ratio as a product of powers — 'kg kg-1', 'g kg**-1' — so
    the same unit reaches us half a dozen ways. Rewrite the negative-exponent
    denominator as a division and the rest is a plain string compare.
    """
    normalized = _UNITS_EXPONENT_RE.sub("-1", units.strip().lower())
    normalized = _UNITS_RATIO_RE.sub(r"\1/\2", normalized)
    return re.sub(r"\s+", "", normalized)


def check_and_convert_units(
    data_array: xr.DataArray,
    var_name: str,
    fallback_units: Optional[str] = None,
) -> xr.DataArray:
    """
    Check and convert water content units to g/kg.

    Parameters
    ----------
    data_array : xr.DataArray
        Data variable to check
    var_name : str
        Name of variable for error messages
    fallback_units : str, optional
        Units to assume when the variable has no 'units' attribute at all
        (the --units flag; the browser asks this as a question). Without it
        a missing attribute stays an error.

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

    # An empty attribute says nothing, so an explicit caller-supplied
    # fallback beats it just as it beats a missing one — the SAM empty-string
    # heuristic below applies only when the caller stated nothing.
    attr_is_empty = units is not None and _normalize_units(str(units)) == ''
    if (units is None or attr_is_empty) and fallback_units is not None:
        logger.warning(
            "Variable %s has %s 'units' attribute; using the caller-supplied "
            "'%s'.", var_name,
            "no" if units is None else "an empty", fallback_units,
        )
        units = fallback_units

    if units is None:
        raise ValueError(f"Variable {var_name} has no 'units' attribute. "
                        "Units must be specified as 'g/kg', 'g/g', or 'kg/kg'.")

    # Normalize units string (strip whitespace, handle case variations)
    units_normalized = _normalize_units(units)

    if units_normalized == '':
        # SAM LPT 3D output writes an empty units attribute on QC/QI even
        # though the values are g/kg (SAM convention). Assume g/kg, loudly.
        logger.warning(
            "Variable %s has an empty 'units' attribute; assuming 'g/kg' "
            "(SAM convention).", var_name,
        )
        return data_array

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


def _drop_time_dims(
    data_array: xr.DataArray,
    timestep: Optional[int] = None,
) -> xr.DataArray:
    """Remove time dimensions, selecting `timestep` on a multi-step one.

    A file with one timestep needs no choice and none is asked for. A file
    with several is refused unless the caller says which — the browser asks
    this as a question on screen; a terminal cannot, so the error names the
    flag that answers it. Two multi-step time dimensions is not a file this
    can describe: there is no single index to ask for.
    """
    time_dims = [d for d in data_array.dims if "time" in d.lower()]
    stepped = [d for d in time_dims if data_array.sizes[d] > 1]
    if len(stepped) > 1:
        raise ValueError(
            "Data has more than one time dimension with several steps ("
            + ", ".join(f"{d}={data_array.sizes[d]}" for d in stepped)
            + ").")
    if stepped:
        if timestep is None:
            raise ValueError(
                f"Data has {data_array.sizes[stepped[0]]} timesteps in "
                f"'{stepped[0]}'. Pass --timestep to choose one.")
        if not 0 <= timestep < data_array.sizes[stepped[0]]:
            raise ValueError(
                f"Timestep {timestep} is out of range for '{stepped[0]}' "
                f"({data_array.sizes[stepped[0]]} steps).")
    elif timestep not in (None, 0):
        # Only step 0 exists (or there is no time dimension at all).
        # Accepting any other index and quietly rendering step 0 would show
        # the wrong moment with no sign anything was ignored.
        raise ValueError(
            f"Timestep {timestep} was requested but the data has "
            + (f"only one step in '{time_dims[0]}'" if time_dims
               else "no time dimension")
            + "; only timestep 0 exists.")
    for time_dim in time_dims:
        index = timestep if time_dim in stepped else 0
        data_array = data_array.isel({time_dim: index}, drop=True)
    return data_array


def _dim_hints(
    dims: list,
    datasets: Iterable[xr.Dataset],
) -> Dict[str, Tuple[str, str]]:
    """What each dimension's own coordinate variable says about itself.

    Keyed by dimension name. A dimension with no like-named 1-D variable, or
    one whose attributes say nothing, has no entry — the common case for
    files the name rule already settles.
    """
    hints = {}
    for dim in dims:
        for dataset in datasets:
            for collection in (dataset.coords, dataset.data_vars):
                if dim not in collection or collection[dim].ndim != 1:
                    continue
                hint = _axis_from_attrs(collection[dim].attrs)
                if hint:
                    hints[dim] = hint
                break
            if dim in hints:
                break
    return hints


def _resolve_spatial_dims(
    data_array: xr.DataArray,
    datasets: Iterable[xr.Dataset] = (),
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    z_dim: Optional[str] = None,
    timestep: Optional[int] = None,
    assumptions: Optional[list] = None,
) -> Tuple[xr.DataArray, Dict[str, str]]:
    """Map the storage dimensions onto x, y and z.

    Time dimensions are dropped first, leaving exactly three. The rest is
    tried in this order, and the order is the whole point — each rule is
    weaker than the one above it, and the first one that settles an axis
    keeps it (mirroring web/soar/ingest/netcdf.js resolveSpatialDims):

      1. Explicit overrides (--x-dim/--y-dim/--z-dim).
      2. NAME, case-insensitively, against AXIS_CANDIDATES.
      3. COORDINATE METADATA for axes still unclaimed — `axis` = X/Y/Z
         first, then `standard_name`, then units. Applied strongest-rule-
         first across all axes at once; a rule that would give one axis to
         two dimensions is not applied at all.
      4. The leftover pair: two axes down and one dimension left over, that
         dimension is the missing axis.
      5. POSITION — (z, y, x) in C order — and only when NOTHING else
         settled anything. Never silent: it is appended to `assumptions`
         and the load states it.

    Guesses (rules 3 and 5) are appended to `assumptions` as sentences.
    """
    data_array = _drop_time_dims(data_array, timestep)
    dims = list(data_array.dims)

    if len(dims) != 3:
        raise ValueError(f"Expected 3 spatial dimensions, got {len(dims)}: {dims}")
    for dim in dims:
        if data_array.sizes[dim] < 2:
            raise ValueError(
                f"Dimension '{dim}' has only {data_array.sizes[dim]} point. "
                "At least 2 points per spatial dimension are required.")

    if assumptions is None:
        assumptions = []
    resolved: Dict[str, str] = {}
    used_dims = set()

    # 1. The user's own assignment, if there is one.
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

    # 2. Name.
    for axis, candidates in AXIS_CANDIDATES.items():
        if axis in resolved:
            continue
        for dim in dims:
            if dim in used_dims:
                continue
            if dim.lower() in candidates:
                resolved[axis] = dim
                used_dims.add(dim)
                break

    # 3. Coordinate metadata, strongest rule first. A rule that would give
    # one axis to two dimensions is not applied at all — picking the first
    # would be a guess dressed as a detection.
    if len(used_dims) < 3:
        hints = _dim_hints([d for d in dims if d not in used_dims], datasets)
        for rule in ("axis", "standard_name", "units"):
            by_axis: Dict[str, list] = {}
            for dim in dims:
                if dim in used_dims:
                    continue
                hint = hints.get(dim)
                if not hint or hint[1] != rule or hint[0] in resolved:
                    continue
                by_axis.setdefault(hint[0], []).append(dim)
            for axis, matches in by_axis.items():
                if len(matches) != 1:
                    continue        # ambiguous: leave it unclaimed
                resolved[axis] = matches[0]
                used_dims.add(matches[0])
                assumptions.append(
                    f"Took '{matches[0]}' as the {axis} axis from its "
                    f"coordinate's {rule} attribute.")

    # 4. The leftover pair.
    missing = [axis for axis in ("x", "y", "z") if axis not in resolved]
    spare = [dim for dim in dims if dim not in used_dims]
    if len(missing) == 1 and len(spare) == 1:
        resolved[missing[0]] = spare[0]
        used_dims.add(spare[0])
        missing = []
        spare = []

    # 5. Position, and only when NOTHING was settled. A partial name match
    # plus positional filling of the rest would be the worst of both: the
    # C-order convention is about the whole tuple, and applying it to the
    # leftovers of a different rule is not the convention, it is a coin toss.
    if len(missing) == 3 and len(spare) == 3:
        for axis, dim in zip(("z", "y", "x"), spare):   # C order, slowest first
            resolved[axis] = dim
        assumptions.append(
            f"No dimension is named or declared as an axis; took "
            f"({', '.join(spare)}) as (z, y, x) by storage position, the "
            "netCDF convention.")
        missing = []

    if missing:
        raise ValueError(
            f"Could not tell which dimensions are {', '.join(missing)} from "
            f"{dims}. Names recognized directly are "
            f"{sorted(set(sum(AXIS_CANDIDATES.values(), [])))}; failing "
            "that, a coordinate variable's axis, standard_name or units "
            "attribute is used. Pass --x-dim/--y-dim/--z-dim to assign "
            "them yourself."
        )

    return data_array, resolved


def standardize_dims(
    data_array: xr.DataArray,
    x_dim: Optional[str] = None,
    y_dim: Optional[str] = None,
    z_dim: Optional[str] = None,
    datasets: Iterable[xr.Dataset] = (),
    timestep: Optional[int] = None,
) -> xr.DataArray:
    """
    Standardize dimension names to (x, y, z).

    Maps dimension names to standard (x, y, z) format via the resolution
    ladder in `_resolve_spatial_dims` (names, coordinate metadata, leftover
    pair, storage position) and removes time dimensions, selecting
    `timestep` on a multi-step one.

    Parameters
    ----------
    data_array : xr.DataArray
        Input data array with arbitrary dimension names
    datasets : iterable of xr.Dataset, optional
        Where to look for the coordinate variables whose attributes the
        metadata rules read.
    timestep : int, optional
        Which step to take from a multi-timestep file.

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
        datasets=datasets,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        timestep=timestep,
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


def _named_coord_var(
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
            return coord

    for dataset in datasets:
        if coord_name in dataset:
            coord = dataset[coord_name]
            if _matches_axis(coord, axis_dim, axis_size):
                return coord

    return None


def _candidate_coords(
    data_array: xr.DataArray,
    datasets: Iterable[xr.Dataset],
    axis: str,
    axis_dim: str,
) -> list:
    """Every 1-D variable that could be this axis's coordinate, best first.

    Collected in order of how well it claims the axis (mirroring
    web/soar/ingest/netcdf.js findCoordinate):

      1. The dimension's own variable — the CF coordinate-variable
         convention, and the strongest signal there is.
      2. THIS axis's candidate names (the sweep used to run the x, y and z
         lists concatenated, which on a cubic grid could hand z the variable
         called `x`).
      3. Any 1-D variable of the right length. Loose, and last.

    Returns [(name, DataArray)], deduplicated by name.
    """
    axis_size = data_array.sizes[axis_dim]
    found = []
    seen = set()

    def consider(name, var):
        if name in seen or not _matches_axis(var, axis_dim, axis_size):
            return
        seen.add(name)
        found.append((name, var))

    def lookup(name):
        if name in data_array.coords:
            consider(name, data_array.coords[name])
        for dataset in datasets:
            if name in dataset.coords:
                consider(name, dataset.coords[name])
            if name in dataset.data_vars:
                consider(name, dataset.data_vars[name])

    lookup(axis_dim)
    for candidate in AXIS_CANDIDATES[axis]:
        lookup(candidate)
    for dataset in datasets:
        for collection in (dataset.coords, dataset.data_vars):
            for var_name in collection:
                var = collection[var_name]
                if var.dims == (axis_dim,):
                    consider(var_name, var)
    return found


def _coordinate_in_metres(name, var, axis, assumptions: list,
                          fallback_coord_units: Optional[str] = None):
    """A chosen coordinate variable's values, converted to metres.

    Everything downstream — the bounding box, the voxel sizes, the march —
    is in metres. A coordinate in km left unconverted is a domain a
    thousand times too small. The units ladder, in full:

      * no units attribute (or an empty one): assume metres — the
        long-standing convention most LES output relies on — and say so.
      * a recognized length: convert.
      * a recognized NON-length (degrees, a level index): refuse. No
        override rescues these; they cannot place a field in space.
      * anything else: refuse, naming the string — unless the caller stated
        what the values are with --coord-units, which is then used and
        recorded.
    """
    units = var.attrs.get("units")
    values = np.asarray(var.values, dtype=np.float64)
    if units in (None, ""):
        assumptions.append(
            f"The {axis} coordinate '{name}' has no units attribute; "
            "assumed meters.")
        return values
    scale = _metres_per_unit(units)
    if scale is None:
        if _is_non_length_unit(units):
            raise ValueError(
                f"The {axis} coordinate '{name}' is in '{units}', which is "
                "not a unit of length and cannot place the field in space. "
                "Cell-center coordinates in meters are required.")
        if fallback_coord_units is None:
            raise ValueError(
                f"The {axis} coordinate '{name}' has unrecognized units "
                f"'{units}'. Pass --coord-units m|km to state what the "
                "values are, or write coordinates in meters into the file.")
        scale = _LENGTH_UNITS[fallback_coord_units]
        assumptions.append(
            f"The {axis} coordinate '{name}' has unrecognized units "
            f"'{units}'; read as {fallback_coord_units} (--coord-units).")
    if scale != 1.0:
        values = values * scale
        assumptions.append(
            f"Converted the {axis} coordinate '{name}' to meters.")
    return values


def _lookup_axis_coord(
    data_array: xr.DataArray,
    datasets: Iterable[xr.Dataset],
    axis: str,
    axis_dim: str,
    assumptions: list,
    fallback_coord_units: Optional[str] = None,
):
    """Auto-detect a coordinate array for one axis, in metres.

    One override on the candidate order: a first choice that is NOT usable
    as a length loses to one that is, further down the list. UM output
    dimensions its fields by `rholev_eta_rho`, a dimensionless hybrid-height
    coordinate running 0 to 1, and carries the actual height in
    `rholev_zsea_rho` — so the CF rule alone produced a domain 6000 km wide
    and 0.99 m tall. A coordinate that cannot be a distance cannot place a
    field in space, whatever its name says.
    """
    def usable(v):
        units = v.attrs.get("units")
        if units in (None, "") or _metres_per_unit(units) is not None:
            return True
        # An unrecognized string is usable exactly when the caller has said
        # what it means; a declared non-length never is.
        return (fallback_coord_units is not None
                and not _is_non_length_unit(units))

    found = _candidate_coords(data_array, datasets, axis, axis_dim)
    if not found:
        return None, None

    name, var = found[0]
    if not usable(var):
        for other_name, other_var in found[1:]:
            if usable(other_var):
                assumptions.append(
                    f"Took '{other_name}' for the {axis} coordinate: it is "
                    f"in a unit of length and '{name}' is not.")
                name, var = other_name, other_var
                break

    # A declared unit that is NOT a length (degrees of longitude, a bare
    # level index) cannot place the field in space, and using it anyway
    # renders a plausible-looking box with nonsense proportions — an ICON
    # RCEMIP file with lon/lat in degrees came out 360 m wide and taller
    # than it was deep. No usable candidate is a dead end, said out loud;
    # _coordinate_in_metres separates the unrecognized-string case (which
    # --coord-units answers) from the definitely-not-a-length one.
    if not usable(var) and _is_non_length_unit(var.attrs.get("units")):
        described = ", ".join(
            f"'{n}' ({v.attrs.get('units') or 'no units'})" for n, v in found)
        raise ValueError(
            f"No {axis} coordinate in a unit of length: found {described}. "
            "Cell-center coordinates in meters are required — they are what "
            "places the field in space. If the grid spacing is known, write "
            "coordinate variables in meters into the file."
        )

    return name, _coordinate_in_metres(
        name, var, axis, assumptions, fallback_coord_units)


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
    timestep: Optional[int] = None,
    assumptions: Optional[list] = None,
    fallback_coord_units: Optional[str] = None,
):
    """
    Extract x, y, z coordinate arrays (in metres) from a data array, handling
    cases where coordinate variable names differ from dimension names (e.g.,
    dim 'ni' with coord 'x(ni)').

    Returns ``(coords, dims_map, coord_names)`` where `coords` maps axis to
    a value array or None, `dims_map` maps axis to the storage dimension
    chosen for it, and `coord_names` maps axis to the coordinate variable
    the values came from (None when nothing was found).
    """
    if assumptions is None:
        assumptions = []
    datasets = _unique_datasets(datasets)
    data_array, dims_map = _resolve_spatial_dims(
        data_array,
        datasets=datasets,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        timestep=timestep,
        assumptions=assumptions,
    )

    coord_overrides = {
        "x": x_coord_name,
        "y": y_coord_name,
        "z": z_coord_name,
    }
    coords = {}
    coord_names = {}

    for axis in ("x", "y", "z"):
        axis_dim = dims_map[axis]
        coord_name = coord_overrides[axis]
        if coord_name is not None:
            var = _named_coord_var(data_array, datasets, coord_name, axis_dim)
            if var is None:
                raise ValueError(
                    f"Could not find {axis}-coordinate '{coord_name}' for dimension "
                    f"'{axis_dim}'. Available 1D coordinates/variables: "
                    f"{_available_1d_names(datasets)}"
                )
            coords[axis] = _coordinate_in_metres(
                coord_name, var, axis, assumptions, fallback_coord_units)
            coord_names[axis] = coord_name
        else:
            coord_names[axis], coords[axis] = _lookup_axis_coord(
                data_array, datasets, axis, axis_dim, assumptions,
                fallback_coord_units)

    return coords, dims_map, coord_names


def _clean_condensate(
    data_array: xr.DataArray,
    var_name: str,
    assumptions: list,
) -> xr.DataArray:
    """Fill values become cloud-free; other bad values refuse the load.

    xarray decodes a declared _FillValue/missing_value to NaN, recording the
    original in ``.encoding`` — so a fill halo reaches here as NaN with the
    declaration attached. Those voxels are cloud-free by construction and
    become zero condensate, stated in `assumptions`. NaN with NO declared
    fill, or any infinity, is corrupt data and raises rather than rendering
    a plausible cloud with silent holes in it. Small negative condensate is
    normal LES numerics and clamps to zero, stated.

    Must run before unit conversion: arithmetic drops ``.encoding``.
    """
    values = np.asarray(data_array.values)
    finite = np.isfinite(values)
    if not finite.all():
        n_nan = int(np.isnan(values).sum())
        n_other = int((~finite).sum()) - n_nan
        fill_declared = any(
            key in source
            for key in ("_FillValue", "missing_value")
            for source in (data_array.encoding, data_array.attrs))
        if n_other or not fill_declared:
            raise ValueError(
                f"Variable {var_name} has {n_nan + n_other} non-finite "
                "values" + ("" if fill_declared else
                            " and declares no _FillValue/missing_value that "
                            "would explain them")
                + ". Refusing to render a field with undefined voxels.")
        values = np.where(np.isnan(values), 0.0, values)
        assumptions.append(
            f"Treated {n_nan} fill value(s) in '{var_name}' as cloud-free "
            "(zero condensate).")
        data_array = data_array.copy(data=values)
    negative = values < 0
    if negative.any():
        n_neg = int(negative.sum())
        worst = float(values.min())
        values = np.where(negative, 0.0, values)
        data_array = data_array.copy(data=values)
        assumptions.append(
            f"Clamped {n_neg} negative value(s) in '{var_name}' to zero "
            f"(most negative: {worst:g}).")
    return data_array


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
    ice_filepath: Optional[str] = None,
    fallback_units: Optional[str] = None,
    fallback_ice_units: Optional[str] = None,
    fallback_coord_units: Optional[str] = None,
    timestep: Optional[int] = None,
    no_ice: bool = False,
) -> Dict[str, Any]:
    """
    Load NetCDF file and validate it, inferring variable names and checking units.

    Performs the following checks and conversions:
    1. Loads NetCDF file and infers liquid/ice water variables
    2. Validates 3D data, selecting one timestep on multi-step files
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
    ice_filepath : str, optional
        Path to a second NetCDF file containing the ice water variable
        (SAM LPT-style output writes one variable per file). When given,
        the ice variable is REQUIRED and is looked up in this file only;
        any ice variable in `filepath` is ignored. The ice grid must match
        the liquid grid (shape, and coordinates when both files carry them).
    fallback_units : str, optional
        Units to assume for condensate variables whose 'units' attribute is
        missing or empty. Without it a missing attribute stays an error.
    fallback_ice_units : str, optional
        Same, for the ice variable specifically (--ice-units). When absent
        the ice variable uses `fallback_units`, as it always has.
    fallback_coord_units : str, optional
        'm' or 'km': units to assume for a spatial coordinate whose units
        attribute is present but unrecognized (--coord-units). Without it an
        unrecognized units string stays an error.
    timestep : int, optional
        Which step to take from a multi-timestep file. A multi-step file
        without it is an error; a single-step file needs no choice.
    no_ice : bool, optional
        Skip the ice variable even when one could be inferred — the CLI
        spelling of the browser's "No ice" answer.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'dataset': xr.Dataset
        - 'liquid_water_var': str (variable name)
        - 'liquid_water_data': xr.DataArray (with standardized (x, y, z) dims, units in g/kg)
        - 'liquid_water_group': str or None (group the liquid came from, root = None)
        - 'ice_water_var': str or None
        - 'ice_water_data': xr.DataArray or None (with standardized dims, units in g/kg)
        - 'ice_water_group': str or None (group the ice came from, root = None)
        - 'filepath': str
        - 'ice_filepath': str or None
        - 'x_coord': ndarray (x coordinates)
        - 'y_coord': ndarray (y coordinates)
        - 'z_coord': ndarray (z coordinates)

    Raises
    ------
    FileNotFoundError
        If a file does not exist
    ValueError
        If validation fails or units are missing/unsupported
    """
    if (fallback_coord_units is not None
            and fallback_coord_units not in _LENGTH_UNITS):
        raise ValueError(
            f"fallback_coord_units must be a recognized length unit "
            f"('m' or 'km'), got '{fallback_coord_units}'.")

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

    def get_dataset(group: Optional[str], path: str = filepath) -> xr.Dataset:
        group = _normalize_group(group)
        key = (path, group)
        if key not in dataset_cache:
            dataset_cache[key] = load_data(path, group=group)
        return dataset_cache[key]

    root_ds = get_dataset(None)
    lw_ds = get_dataset(liquid_water_group)
    if ice_filepath is not None:
        iw_ds = get_dataset(ice_water_group, path=ice_filepath)
    else:
        iw_ds = get_dataset(ice_water_group)
    coord_ds = get_dataset(coords_group)

    # Infer liquid water variable (required)
    lw_var, lw_data = infer_liquid_water(
        lw_ds,
        explicit_name=liquid_water_var,
        group=liquid_water_group,
    )
    validate_data(lw_ds, lw_data, lw_var, timestep=timestep)

    # Guesses the load makes are stated, never silent: a field whose axes
    # were taken by position renders a perfectly plausible cloud with x and
    # z swapped, and the only defence is that the person was told.
    assumptions: list = []

    # Extract coordinate arrays before standardizing dims (dim names may differ from coord names)
    coords, dims_map, coord_names = _extract_coords(
        lw_data,
        datasets=[coord_ds, lw_ds, root_ds],
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        z_coord_name=z_coord_name,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        timestep=timestep,
        assumptions=assumptions,
        fallback_coord_units=fallback_coord_units,
    )
    x_coord, y_coord, z_coord = coords["x"], coords["y"], coords["z"]

    # Standardize dimensions to (x, y, z)
    lw_data = standardize_dims(
        lw_data, x_dim=dims_map["x"], y_dim=dims_map["y"],
        z_dim=dims_map["z"], timestep=timestep)

    # Fill/NaN/negative policy, then units: cleaning reads .encoding, which
    # the unit conversion's arithmetic would drop.
    lw_data = _clean_condensate(lw_data, lw_var, assumptions)
    lw_data = check_and_convert_units(lw_data, lw_var, fallback_units)

    if no_ice:
        iw_var, iw_data = None, None
    elif ice_filepath is not None:
        # Ice explicitly requested from a separate file: it is required there.
        iw_var, iw_data = infer_variable(
            iw_ds,
            ICE_WATER_NAMES,
            explicit_name=ice_water_var,
            variable_role="ice water variable",
            group=ice_water_group,
        )
    else:
        # Infer ice water variable (optional)
        iw_var, iw_data = infer_ice_water(
            iw_ds,
            explicit_name=ice_water_var,
            group=ice_water_group,
        )
    if iw_data is not None:
        validate_data(iw_ds, iw_data, iw_var, timestep=timestep)
        if ice_filepath is not None:
            # Cross-check the ice file's own coordinates against the liquid
            # file's before standardizing — PER AXIS. An axis whose
            # coordinate resolves must be compared even when another axis's
            # cannot; a mismatched grid accepted by shape alone renders
            # liquid and ice from two different worlds in one picture.
            ice_dims_map = None
            try:
                iw_spatial, ice_dims_map = _resolve_spatial_dims(
                    iw_data, datasets=[iw_ds],
                    x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                    timestep=timestep, assumptions=[])
            except ValueError as exc:
                assumptions.append(
                    f"Could not resolve the ice file's dimensions ({exc}); "
                    "the ice grid was checked by shape alone.")
            ice_coord_overrides = {
                "x": x_coord_name, "y": y_coord_name, "z": z_coord_name}
            for axis in ("x", "y", "z") if ice_dims_map else ():
                lw_c = {"x": x_coord, "y": y_coord, "z": z_coord}[axis]
                axis_dim = ice_dims_map[axis]
                # Conversion notes for the ice file's own coordinates are
                # scratch: the liquid file's coordinates are the ones the
                # load uses, and their notes are already recorded.
                scratch: list = []
                coord_name = ice_coord_overrides[axis]
                if coord_name is not None:
                    var = _named_coord_var(
                        iw_spatial, [iw_ds], coord_name, axis_dim)
                    iw_c = None if var is None else _coordinate_in_metres(
                        coord_name, var, axis, scratch, fallback_coord_units)
                else:
                    # Non-length or unrecognized units raise out of here:
                    # a coordinate that cannot be read cannot be checked,
                    # and pretending otherwise is the old bug.
                    _, iw_c = _lookup_axis_coord(
                        iw_spatial, [iw_ds], axis, axis_dim, scratch,
                        fallback_coord_units)
                if iw_c is None:
                    assumptions.append(
                        f"The ice file carries no {axis} coordinate; its "
                        f"{axis} grid was checked by shape alone.")
                    continue
                if lw_c is None:
                    continue
                if len(lw_c) != len(iw_c) or not np.allclose(
                    np.asarray(lw_c, dtype=np.float64),
                    np.asarray(iw_c, dtype=np.float64),
                ):
                    raise ValueError(
                        f"Ice file '{ice_filepath}' has a different {axis}-coordinate "
                        f"grid than liquid file '{filepath}'. The two files must "
                        "describe the same grid."
                    )
            # A separate file resolves its own dimension names.
            iw_data = standardize_dims(
                iw_data, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim,
                datasets=[iw_ds], timestep=timestep)
        else:
            # Same file: the liquid variable's resolution is the resolution
            # — when the ice variable actually uses those dimension names.
            # Files that put qc on (zt, yt, xt) and qi on (z, y, x) describe
            # one logical grid twice; force-feeding the liquid's names would
            # refuse a valid file, so the ice resolves its own dims and the
            # shape cross-check below still holds the two together.
            iw_dims = set(iw_data.dims)
            if all(dims_map[a] in iw_dims for a in ("x", "y", "z")):
                iw_data = standardize_dims(
                    iw_data, x_dim=dims_map["x"], y_dim=dims_map["y"],
                    z_dim=dims_map["z"], timestep=timestep)
            else:
                assumptions.append(
                    f"Ice variable '{iw_var}' does not use the liquid "
                    "variable's dimension names; resolved its dimensions "
                    "independently and matched the grids by shape.")
                iw_data = standardize_dims(
                    iw_data,
                    x_dim=x_dim if x_dim in iw_dims else None,
                    y_dim=y_dim if y_dim in iw_dims else None,
                    z_dim=z_dim if z_dim in iw_dims else None,
                    datasets=[iw_ds], timestep=timestep)
        iw_data = _clean_condensate(iw_data, iw_var, assumptions)
        # Check and convert units to g/kg. --ice-units answers the ice
        # variable specifically; without it, --units covers both species.
        iw_data = check_and_convert_units(
            iw_data, iw_var,
            fallback_ice_units if fallback_ice_units is not None
            else fallback_units)
        if iw_data.shape != lw_data.shape:
            raise ValueError(
                "Liquid water and ice water arrays must have identical spatial shapes. "
                f"Got liquid={lw_data.shape}, ice={iw_data.shape}."
            )

    if x_coord is None or y_coord is None or z_coord is None:
        missing = [a for a, c in zip("xyz", (x_coord, y_coord, z_coord))
                   if c is None]
        raise ValueError(
            f"No coordinate variable found for the {', '.join(missing)} "
            "dimension(s). Cell-center coordinates in meters are required "
            "for all three axes — they are what places the field in space."
        )

    for sentence in assumptions:
        logger.warning("%s", sentence)

    return {
        'dataset': lw_ds,
        'liquid_water_var': lw_var,
        'liquid_water_data': lw_data,
        'liquid_water_group': liquid_water_group,
        'ice_water_var': iw_var,
        'ice_water_data': iw_data,
        'ice_water_group': ice_water_group if iw_data is not None else None,
        'filepath': str(filepath),
        'ice_filepath': str(ice_filepath) if ice_filepath is not None else None,
        'x_coord': x_coord,
        'y_coord': y_coord,
        'z_coord': z_coord,
        # Which storage dimension and which coordinate variable each axis
        # came from, and what was guessed along the way — the resolved
        # selection, mirroring the browser loader's description.
        'dims': dims_map,
        'coord_names': coord_names,
        'timestep': timestep,
        'assumptions': assumptions,
    }
