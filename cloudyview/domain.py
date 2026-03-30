"""
Shared domain geometry for CloudyView renderers.

Computes physical domain dimensions and aspect ratios from coordinate arrays,
handling non-uniform vertical spacing correctly. Used by both witness (numba
ray marching) and behold (Mitsuba path tracing) to ensure consistent geometry.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DomainGeometry:
    """Physical domain geometry derived from coordinate arrays.

    Attributes
    ----------
    nx, ny, nz : int
        Grid dimensions.
    dx, dy : float
        Horizontal grid spacings (metres). Assumed uniform within each axis.
    width_x, width_y : float
        Total horizontal extents (metres).
    height_z : float
        Total vertical extent (metres), computed from actual z-coordinate range.
    ar_x : float
        Horizontal-to-vertical aspect ratio in x: width_x / height_z.
    ar_y : float
        Horizontal-to-vertical aspect ratio in y: width_y / height_z.
    """
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    width_x: float
    width_y: float
    height_z: float
    ar_x: float
    ar_y: float


def compute_domain_geometry(x_coord, y_coord, z_coord, nx, ny, nz):
    """Compute domain geometry from coordinate arrays.

    Handles non-uniform vertical spacing by using the actual z-coordinate
    range rather than ``nz * dz``.  Horizontal spacing is taken from the
    first coordinate pair and assumed uniform.

    Parameters
    ----------
    x_coord, y_coord, z_coord : array-like
        1-D coordinate arrays (cell centres, in metres).
    nx, ny, nz : int
        Grid dimensions.

    Returns
    -------
    DomainGeometry
        Dataclass with all scalars — safe to unpack for numba @njit functions.
    """
    if len(x_coord) < 2 or len(y_coord) < 2 or len(z_coord) < 2:
        raise ValueError(
            "x/y/z coordinates must each contain at least 2 points "
            "to determine grid spacing."
        )

    dx = float(x_coord[1] - x_coord[0])
    dy = float(y_coord[1] - y_coord[0])

    width_x = nx * dx
    width_y = ny * dy

    # Vertical extent from actual coordinate range + half-cells at boundaries.
    # z_coord values are cell centres, so the full domain spans from
    # half a cell below the first centre to half a cell above the last.
    dz_first = float(z_coord[1] - z_coord[0])
    dz_last = float(z_coord[-1] - z_coord[-2])
    height_z = float(z_coord[-1] - z_coord[0]) + 0.5 * dz_first + 0.5 * dz_last

    ar_x = width_x / height_z
    ar_y = width_y / height_z

    return DomainGeometry(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy,
        width_x=width_x, width_y=width_y, height_z=height_z,
        ar_x=ar_x, ar_y=ar_y,
    )
