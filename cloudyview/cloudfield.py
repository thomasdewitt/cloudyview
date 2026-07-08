"""CloudField: the in-memory cloud volume handed to the render functions.

`load()` is the library entry point for getting data into CloudyView:

    import cloudyview as cv
    field = cv.load("cloud.nc")                       # autodetect qc + qi
    field = cv.load("..._QC_0000000600.nc",
                    ice="..._QI_0000000600.nc")       # SAM LPT split files
    field = cv.load("cloud.nc", liquid_water_var="QC")  # explicit overrides

All validation and variable/coordinate inference lives in
``io.load_and_validate``; this module only shapes the result into a
:class:`CloudField`.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union
from pathlib import Path

import numpy as np

from . import io


@dataclass
class CloudField:
    """A single-timestep 3D cloud water field on a regular (x, y, z) grid.

    Attributes
    ----------
    lwc : ndarray (nx, ny, nz), float32
        Liquid water content in g/kg, dims standardized to (x, y, z).
    iwc : ndarray (nx, ny, nz), float32, or None
        Ice water content in g/kg, or None when the source has no ice.
    x, y, z : ndarray, 1D
        Cell-center coordinates in meters (dtype preserved from the source).
    source : str or None
        Path of the file the liquid water came from.
    ice_source : str or None
        Path of the separate ice file, when ice was loaded split-file style.
    liquid_var, ice_var : str or None
        Variable names the data came from (e.g. "QC", "QI").
    """

    lwc: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    iwc: Optional[np.ndarray] = None
    source: Optional[str] = None
    ice_source: Optional[str] = None
    liquid_var: Optional[str] = None
    ice_var: Optional[str] = None

    def __post_init__(self):
        self.lwc = np.asarray(self.lwc, dtype=np.float32)
        if self.lwc.ndim != 3:
            raise ValueError(
                f"lwc must be 3D (x, y, z); got shape {self.lwc.shape}."
            )
        if self.iwc is not None:
            self.iwc = np.asarray(self.iwc, dtype=np.float32)
            if self.iwc.shape != self.lwc.shape:
                raise ValueError(
                    "iwc shape must match lwc shape; "
                    f"got iwc={self.iwc.shape}, lwc={self.lwc.shape}."
                )
        for axis, (name, coord, n) in enumerate(
            (("x", self.x, self.lwc.shape[0]),
             ("y", self.y, self.lwc.shape[1]),
             ("z", self.z, self.lwc.shape[2]))
        ):
            coord = np.asarray(coord)
            if coord.ndim != 1 or coord.size != n:
                raise ValueError(
                    f"{name} coordinate must be 1D with length {n}; "
                    f"got shape {coord.shape}."
                )
            if coord.size > 1 and np.all(np.diff(coord) < 0):
                coord = coord[::-1].copy()
                self.lwc = np.flip(self.lwc, axis=axis).copy()
                if self.iwc is not None:
                    self.iwc = np.flip(self.iwc, axis=axis).copy()
            setattr(self, name, coord)

    @property
    def shape(self) -> tuple:
        """(nx, ny, nz) grid shape."""
        return self.lwc.shape

    def __repr__(self) -> str:
        nx, ny, nz = self.lwc.shape
        ice = self.ice_var or ("yes" if self.iwc is not None else "none")
        src = Path(self.source).name if self.source else "in-memory"
        return (f"CloudField({nx}x{ny}x{nz}, liquid={self.liquid_var or 'lwc'}, "
                f"ice={ice}, source={src})")


def load(
    filepath: Union[str, Path],
    ice: Optional[Union[str, Path]] = None,
    *,
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
    stage_callback: Optional[Callable[[str], None]] = None,
) -> CloudField:
    """Load a cloud field from NetCDF into a :class:`CloudField`.

    Parameters
    ----------
    filepath : str or Path
        NetCDF file with the liquid water variable (and, unless `ice` is
        given, optionally the ice water variable).
    ice : str or Path, optional
        Second NetCDF file containing the ice water variable, for output
        that writes one variable per file (SAM LPT style). When given, the
        ice variable is required in that file.
    liquid_water_var, ice_water_var : str, optional
        Explicit variable-name overrides (autodetected otherwise).
    dataset_group, liquid_water_group, ice_water_group, coords_group : str, optional
        NetCDF group overrides for variable/coordinate lookup.
    x_coord_name, y_coord_name, z_coord_name : str, optional
        Explicit coordinate variable names.
    x_dim, y_dim, z_dim : str, optional
        Explicit dimension names for the x/y/z axes.
    stage_callback : callable, optional
        Receives coarse loading stage strings for interactive callers.

    Returns
    -------
    CloudField

    Raises
    ------
    FileNotFoundError
        If a file does not exist.
    ValueError
        If validation fails (dims, units, grid mismatch between files, ...).
    """
    if stage_callback is not None:
        stage_callback("loading file")
    result = io.load_and_validate(
        str(filepath),
        liquid_water_var=liquid_water_var,
        ice_water_var=ice_water_var,
        dataset_group=dataset_group,
        liquid_water_group=liquid_water_group,
        ice_water_group=ice_water_group,
        coords_group=coords_group,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        z_coord_name=z_coord_name,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
        ice_filepath=str(ice) if ice is not None else None,
    )

    if stage_callback is not None:
        stage_callback("building CloudField")
    iw_data = result['ice_water_data']
    return CloudField(
        lwc=result['liquid_water_data'].values,
        iwc=iw_data.values if iw_data is not None else None,
        x=result['x_coord'],
        y=result['y_coord'],
        z=result['z_coord'],
        source=result.get('filepath', str(filepath)),
        ice_source=result.get('ice_filepath'),
        liquid_var=result.get('liquid_water_var'),
        ice_var=result.get('ice_water_var'),
    )
