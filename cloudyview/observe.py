#!/usr/bin/env python
"""
observe.py: Cloud field visualization with PyVista isosurface rendering.

Usage:
    python observe.py <filename.nc> [--output <path>] [--threshold <value>]

This script provides a simple 3D isosurface view of cloud extinction fields:
1. Loads cloud data and calculates extinction coefficients
2. Renders white isosurface at specified extinction threshold
3. Views from above at an angle with sky blue background
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc
import pyvista as pv

from . import io, optical_depth


def main(filename: str, output: str = None, threshold: float = 0.001) -> None:
    """
    Main function for observe.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output file path for render
    threshold : float
        Extinction coefficient threshold for isosurface (m^-1)
    """
    print(f"CloudyView Observe: Loading {filename}")
    start_time = time.perf_counter()

    try:
        # Load and validate data with xarray
        data_dict = io.load_and_validate(filename)
        ds = data_dict['dataset']
        lw_var = data_dict['liquid_water_var']
        lw_data = data_dict['liquid_water_data']
        iw_data = data_dict['ice_water_data']

        # Get coordinates from data or dataset
        if hasattr(lw_data, 'coords'):
            dims = lw_data.dims
            x_coord = None
            y_coord = None
            z_coord = None

            # Try to find coordinate arrays
            for coord_name in ['x', 'lon', 'longitude']:
                if coord_name in lw_data.coords:
                    x_coord = lw_data.coords[coord_name].values
                    break

            for coord_name in ['y', 'lat', 'latitude']:
                if coord_name in lw_data.coords:
                    y_coord = lw_data.coords[coord_name].values
                    break

            for coord_name in ['z', 'height', 'altitude', 'level']:
                if coord_name in lw_data.coords:
                    z_coord = lw_data.coords[coord_name].values
                    break

        # Fallback: try to load from NetCDF directly for SAM format
        if x_coord is None or y_coord is None or z_coord is None:
            print("  Using NetCDF direct access for coordinates...")
            ds_nc = nc.Dataset(filename, 'r')
            try:
                x_coord = ds_nc.variables.get('x', None)
                y_coord = ds_nc.variables.get('y', None)
                z_coord = ds_nc.variables.get('z', None)

                if x_coord is not None:
                    x_coord = x_coord[:]
                if y_coord is not None:
                    y_coord = y_coord[:]
                if z_coord is not None:
                    z_coord = z_coord[:]
            finally:
                ds_nc.close()

        # Create default coordinates if still missing
        lw_np = lw_data.values
        if 'time' in lw_data.dims:
            lw_np = lw_np[0]  # Remove time dimension if present

        nx, ny, nz = lw_np.shape
        if x_coord is None:
            x_coord = np.arange(nx)
        if y_coord is None:
            y_coord = np.arange(ny)
        if z_coord is None:
            z_coord = np.arange(nz)

        # Calculate grid spacing
        dx = float(x_coord[1] - x_coord[0]) if len(x_coord) > 1 else 1.0
        dy = float(y_coord[1] - y_coord[0]) if len(y_coord) > 1 else 1.0
        dz = float(z_coord[1] - z_coord[0]) if len(z_coord) > 1 else 1.0

        print(f"  Domain: {nx} x {ny} x {nz}")
        print(f"  Grid spacing: dx={dx:.1f} m, dy={dy:.1f} m, dz={dz:.1f} m")

        # Compute extinction coefficient
        print("  Computing extinction coefficients...")
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0)

        print(f"  Extinction range: {sigma_ext.min():.6f} to {sigma_ext.max():.6f} m^-1")
        print(f"  Isosurface threshold: {threshold:.6f} m^-1")

        # Create PyVista structured grid
        # PyVista expects dimensions in (nz, ny, nx) order for structured grids
        grid = pv.ImageData(dimensions=(nx, ny, nz))
        grid.spacing = (dx, dy, dz)
        grid.origin = (x_coord[0], y_coord[0], z_coord[0])

        # Add extinction field as point data
        # Flatten in Fortran order to match PyVista's expected ordering
        grid.point_data['extinction'] = sigma_ext.flatten(order='F')

        # Create isosurface
        print("  Creating isosurface...")
        isosurface = grid.contour([threshold], scalars='extinction')

        # Set up plotter
        plotter = pv.Plotter(off_screen=(output is not None))

        # Sky blue background
        sky_blue = (0.53, 0.81, 0.92)
        plotter.set_background(sky_blue)

        # Add isosurface as white mesh
        plotter.add_mesh(isosurface, color='white', smooth_shading=True)

        # Calculate camera position - view from above at an angle
        # Center of domain
        center_x = (x_coord[0] + x_coord[-1]) / 2
        center_y = (y_coord[0] + y_coord[-1]) / 2
        center_z = (z_coord[0] + z_coord[-1]) / 2

        # Domain extent
        extent_x = x_coord[-1] - x_coord[0]
        extent_y = y_coord[-1] - y_coord[0]
        extent_z = z_coord[-1] - z_coord[0]
        max_extent = max(extent_x, extent_y, extent_z)

        # Camera position: elevated and offset for angled view
        # Position camera above and to the side
        camera_distance = max_extent * 1.5
        camera_pos = [
            center_x + camera_distance * 0.5,  # offset in x
            center_y + camera_distance * 0.5,  # offset in y
            center_z + camera_distance * 1.2   # elevated above
        ]

        plotter.camera_position = [
            camera_pos,  # camera position
            (center_x, center_y, center_z),  # focal point
            (0, 0, 1)  # view up vector
        ]

        # Determine output path
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(f"observe_threshold_{threshold:.6f}.png")

        print(f"  Rendering to {output_path}...")
        plotter.show(screenshot=str(output_path))

        elapsed = time.perf_counter() - start_time
        print("\n✓ Observe complete!")
        print(f"  Total runtime: {elapsed:.1f} s")
        print(f"  Render saved to {output_path}")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"✗ PyVista required but not installed: {e}", file=sys.stderr)
        print("  Install with: pip install pyvista", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cli():
    """Command-line interface for observe.py"""
    parser = argparse.ArgumentParser(
        description="Cloud visualization with PyVista isosurface rendering"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path for saving render (default: observe_threshold_<value>.png)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.001,
        help="Extinction coefficient threshold for isosurface in m^-1 (default: 0.001)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.threshold)


if __name__ == "__main__":
    cli()
