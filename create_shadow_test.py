#!/usr/bin/env python
"""Create test NetCDF file with bottom uniform cloud and top cube for shadow testing."""

import numpy as np
import xarray as xr
from pathlib import Path

def create_shadow_test(nx=64, ny=64, nz=64, output_file='data/test_shadow.nc'):
    """
    Create test case with uniform horizontal cloud at bottom and cube cloud at top center.

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

    # Bottom uniform horizontal cloud (thin layer)
    qc[:, :, 5:10] = 0.3

    # Top center cube cloud (to cast shadow on bottom cloud)
    cube_size = 10
    center_x = nx // 2
    center_y = ny // 2
    top_z = nz - 15
    qc[center_x-cube_size:center_x+cube_size,
       center_y-cube_size:center_y+cube_size,
       top_z:top_z+10] = 0.6

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
    print(f"Created shadow test file: {output_path}")
    print(f"  Grid: {nx}x{ny}x{nz}")
    print(f"  Bottom cloud layer: z={5*dz:.0f}-{10*dz:.0f}m")
    print(f"  Top cube: center ({center_x*dx:.0f}, {center_y*dy:.0f}), z={top_z*dz:.0f}-{(top_z+10)*dz:.0f}m")
    print(f"  Max qc: {qc.max():.3f} g/kg")

    return ds

if __name__ == '__main__':
    create_shadow_test()
