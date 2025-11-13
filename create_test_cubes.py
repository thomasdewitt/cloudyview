#!/usr/bin/env python
"""Create simple test NetCDF file with cube-shaped clouds for testing observe path."""

import numpy as np
import xarray as xr
from pathlib import Path

def create_test_cubes(nx=64, ny=64, nz=64, output_file='data/test_cubes.nc'):
    """
    Create a simple test case with cube-shaped clouds.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions
    output_file : str
        Output NetCDF file path
    """
    # Create coordinate arrays
    dx = dy = dz = 100.0  # 100m grid spacing
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    # Initialize liquid water content field (g/kg)
    qc = np.zeros((nx, ny, nz), dtype=np.float32)

    # Create several cubes at different heights
    # Cube 1: Large cube at mid-level (should be opaque)
    qc[15:35, 15:35, 25:45] = 0.5

    # Cube 2: Small cube at high level
    qc[40:50, 40:50, 50:60] = 0.3

    # Cube 3: Thin layer at low level
    qc[10:50, 10:50, 10:15] = 0.4

    # Cube 4: Another cube to create shadow
    qc[20:30, 45:55, 35:45] = 0.6

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'qc': (['x', 'y', 'z'], qc),
        },
        coords={
            'x': (['x'], x),
            'y': (['y'], y),
            'z': (['z'], z),
        }
    )

    # Add metadata
    ds['qc'].attrs['long_name'] = 'Cloud liquid water mixing ratio'
    ds['qc'].attrs['units'] = 'g/kg'
    ds['x'].attrs['long_name'] = 'x coordinate'
    ds['x'].attrs['units'] = 'm'
    ds['y'].attrs['long_name'] = 'y coordinate'
    ds['y'].attrs['units'] = 'm'
    ds['z'].attrs['long_name'] = 'height'
    ds['z'].attrs['units'] = 'm'

    # Save to NetCDF
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"Created test file: {output_path}")
    print(f"  Grid: {nx}x{ny}x{nz}")
    print(f"  Grid spacing: {dx}m x {dy}m x {dz}m")
    print(f"  Max qc: {qc.max():.3f} g/kg")
    print(f"  Number of cloud points: {np.sum(qc > 0)}")

    return ds

if __name__ == '__main__':
    create_test_cubes()
