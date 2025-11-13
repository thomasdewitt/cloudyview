#!/usr/bin/env python
"""
Create a simple test NetCDF file with cloud condensate cubes for testing observe.py
"""
import numpy as np
import netCDF4 as nc
from pathlib import Path

def create_test_cubes(output_file: str = "test_cubes.nc"):
    """
    Create a test dataset with several cloud cubes at different heights.
    This allows testing of:
    1. Opacity rendering (see-through vs opaque)
    2. Color rendering (white at top, grey below significant cloud)
    3. Cumulative optical depth effects
    """
    # Grid dimensions
    nx, ny, nz = 64, 64, 64

    # Coordinate arrays (in meters)
    x = np.arange(nx) * 100.0  # 100m spacing
    y = np.arange(ny) * 100.0
    z = np.arange(nz) * 50.0   # 50m vertical spacing

    # Initialize LWC field (all zeros = clear sky)
    lwc = np.zeros((nx, ny, nz), dtype=np.float32)

    # Create test cubes with different LWC values
    # Cube 1: Top layer, moderate LWC (should be white and somewhat opaque)
    # Located at x=[10:20], y=[10:20], z=[50:60]
    lwc[10:20, 10:20, 50:60] = 0.5  # g/kg

    # Cube 2: Middle layer, high LWC (should be white on top, grey if below Cube 1)
    # Located at x=[25:35], y=[10:20], z=[30:40]
    lwc[25:35, 10:20, 30:40] = 1.0  # g/kg - higher LWC

    # Cube 3: Lower layer, light LWC (should be darker grey, semi-transparent)
    # Located at x=[40:50], y=[10:20], z=[10:20]
    lwc[40:50, 10:20, 10:20] = 0.3  # g/kg

    # Cube 4: Tall column going through multiple levels
    # Located at x=[10:20], y=[35:45], z=[15:55] - should show gradient from bright to dark
    lwc[10:20, 35:45, 15:55] = 0.8  # g/kg

    # Cube 5: Very thin cloud at top (should be semi-transparent white)
    # Located at x=[40:55], y=[40:55], z=[55:60]
    lwc[40:55, 40:55, 55:60] = 0.1  # g/kg - very light

    # Create NetCDF file
    print(f"Creating test file: {output_file}")
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # Create dimensions
        ds.createDimension('x', nx)
        ds.createDimension('y', ny)
        ds.createDimension('z', nz)

        # Create coordinate variables
        x_var = ds.createVariable('x', 'f4', ('x',))
        x_var[:] = x
        x_var.units = 'm'
        x_var.long_name = 'x coordinate'

        y_var = ds.createVariable('y', 'f4', ('y',))
        y_var[:] = y
        y_var.units = 'm'
        y_var.long_name = 'y coordinate'

        z_var = ds.createVariable('z', 'f4', ('z',))
        z_var[:] = z
        z_var.units = 'm'
        z_var.long_name = 'height'

        # Create LWC variable
        qc_var = ds.createVariable('QC', 'f4', ('x', 'y', 'z'))
        qc_var[:] = lwc
        qc_var.units = 'g/kg'
        qc_var.long_name = 'Cloud water mixing ratio'

        # Add global attributes
        ds.description = 'Test cloud field with cubes at different heights'
        ds.history = 'Created by create_test_cubes.py'

    print(f"✓ Created {output_file}")
    print(f"  Grid: {nx}×{ny}×{nz}")
    print(f"  Domain: {x.max():.0f}m × {y.max():.0f}m × {z.max():.0f}m")
    print(f"  5 test cubes with varying LWC values")
    print(f"\nTest cube summary:")
    print(f"  Cube 1: Top layer (z=2500-3000m), LWC=0.5 g/kg - should be bright white")
    print(f"  Cube 2: Mid layer (z=1500-2000m), LWC=1.0 g/kg - should be white")
    print(f"  Cube 3: Low layer (z=500-1000m), LWC=0.3 g/kg - should be grey/semi-transparent")
    print(f"  Cube 4: Tall column (z=750-2750m), LWC=0.8 g/kg - should show gradient")
    print(f"  Cube 5: Very top (z=2750-3000m), LWC=0.1 g/kg - should be semi-transparent white")

if __name__ == "__main__":
    create_test_cubes("test_cubes.nc")
