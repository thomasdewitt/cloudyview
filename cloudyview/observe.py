#!/usr/bin/env python
"""
observe.py: Cloud field visualization with PyVista isosurface rendering.

Usage:
    # Multiple surfaces (default):
    python observe.py <filename.nc> [--html] [-n 3] [--min-threshold 0.001] [--max-threshold 0.01]

    # Single surface:
    python observe.py <filename.nc> --threshold 0.005 [--html]

This script provides a 3D isosurface view of cloud extinction fields:
1. Loads cloud data and calculates extinction coefficients
2. Renders white isosurface(s) with varying opacity based on extinction threshold
3. Multiple surfaces: logarithmically spaced thresholds from low (transparent) to high (opaque)
4. Views from above at an angle with sky blue background and directional sun lighting
5. Optional interactive HTML export for portability (open in any browser!)
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc
import pyvista as pv

from . import io, optical_depth


def main(filename: str, output: str = None, threshold: float = None,
         html: bool = False, n_surfaces: int = 10,
         min_threshold: float = 0.001, max_threshold: float = 0.1) -> None:
    """
    Main function for observe.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output file path for render
    threshold : float, optional
        Single extinction coefficient threshold for isosurface (m^-1)
        If None, uses multiple surfaces from min_threshold to max_threshold
    html : bool
        If True, export interactive HTML instead of static PNG
    n_surfaces : int
        Number of isosurfaces to plot (default: 3)
    min_threshold : float
        Minimum extinction threshold (default: 0.001 m^-1)
    max_threshold : float
        Maximum extinction threshold (default: 0.01 m^-1)
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

        # Determine thresholds and opacities
        if threshold is not None:
            # Single surface mode
            thresholds = [threshold]
            opacities = [1.0]
            print(f"  Single isosurface threshold: {threshold:.6f} m^-1")
        else:
            # Multiple surfaces mode
            # Logarithmically spaced thresholds from min to max
            thresholds = np.logspace(np.log10(min_threshold), np.log10(max_threshold), n_surfaces)
            # Linearly spaced opacities from low to fully opaque
            min_opacity = 0.15
            opacities = np.linspace(min_opacity, 1.0, n_surfaces)
            print(f"  Multiple isosurfaces: {n_surfaces} surfaces")
            print(f"  Threshold range: {min_threshold:.6f} to {max_threshold:.6f} m^-1")
            print(f"  Opacity range: {min_opacity:.2f} to 1.00")

        # Create PyVista structured grid
        # PyVista expects dimensions in (nz, ny, nx) order for structured grids
        grid = pv.ImageData(dimensions=(nx, ny, nz))
        grid.spacing = (dx, dy, dz)
        grid.origin = (x_coord[0], y_coord[0], z_coord[0])

        # Add extinction field as point data
        # Flatten in Fortran order to match PyVista's expected ordering
        grid.point_data['extinction'] = sigma_ext.flatten(order='F')

        # Calculate domain geometry (needed for lighting and camera)
        center_x = (x_coord[0] + x_coord[-1]) / 2
        center_y = (y_coord[0] + y_coord[-1]) / 2
        center_z = (z_coord[0] + z_coord[-1]) / 2

        # Domain extent
        extent_x = x_coord[-1] - x_coord[0]
        extent_y = y_coord[-1] - y_coord[0]
        extent_z = z_coord[-1] - z_coord[0]
        max_extent = max(extent_x, extent_y, extent_z)

        # Set up plotter
        plotter = pv.Plotter(off_screen=(output is not None), lighting='none')

        # Sky blue background
        sky_blue = (0.53, 0.81, 0.92)
        plotter.set_background(sky_blue)

        # Add custom lighting - sun at 70 deg azimuth
        # Azimuth: angle from north (0°) going clockwise
        # Elevation: angle above horizon
        sun_azimuth = 70.0  # degrees
        sun_elevation = 45.0  # degrees above horizon

        # Convert to Cartesian coordinates for light position
        # Place light far away to simulate directional sun
        light_distance = max_extent * 10
        az_rad = np.deg2rad(sun_azimuth)
        el_rad = np.deg2rad(sun_elevation)

        light_pos = [
            center_x + light_distance * np.cos(el_rad) * np.sin(az_rad),
            center_y + light_distance * np.cos(el_rad) * np.cos(az_rad),
            center_z + light_distance * np.sin(el_rad)
        ]

        # Add directional sun light
        sun_light = pv.Light(position=light_pos, light_type='scene light')
        sun_light.set_direction_angle(sun_elevation, sun_azimuth)
        sun_light.intensity = 1.0
        plotter.add_light(sun_light)

        # Add subtle ambient fill light so shadows aren't completely black
        ambient_intensity = 0.3
        plotter.add_light(pv.Light(light_type='headlight', intensity=ambient_intensity))

        # Create and add isosurfaces
        print(f"  Creating {len(thresholds)} isosurface(s)...")
        surfaces_added = 0
        for i, (thresh, opacity) in enumerate(zip(thresholds, opacities)):
            isosurface = grid.contour([thresh], scalars='extinction')
            # Skip empty meshes
            if isosurface.n_points == 0:
                print(f"    Surface {i+1}/{len(thresholds)}: threshold={thresh:.6f} m^-1 - skipped (empty)")
                continue
            print(f"    Surface {i+1}/{len(thresholds)}: threshold={thresh:.6f} m^-1, opacity={opacity:.2f}")
            plotter.add_mesh(isosurface, color='white', opacity=opacity, smooth_shading=True)
            surfaces_added += 1

        if surfaces_added == 0:
            print("✗ No surfaces generated - try lower thresholds", file=sys.stderr)
            sys.exit(1)

        # Calculate camera position - view from above at an angle
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
        else:
            if threshold is not None:
                # Single surface mode
                if html:
                    output_path = Path(f"observe_threshold_{threshold:.6f}.html")
                else:
                    output_path = Path(f"observe_threshold_{threshold:.6f}.png")
            else:
                # Multiple surfaces mode
                if html:
                    output_path = Path(f"observe_multi_{n_surfaces}surfaces.html")
                else:
                    output_path = Path(f"observe_multi_{n_surfaces}surfaces.png")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if html:
            print(f"  Exporting interactive HTML to {output_path}...")
            plotter.export_html(str(output_path))
            plotter.close()
        else:
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
        help="Output file path for saving render (default: observe_threshold_<value>.png or .html)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="Single extinction coefficient threshold for isosurface in m^-1. If not set, uses multiple surfaces."
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Export interactive HTML instead of static PNG (fully portable, open in any browser)"
    )
    parser.add_argument(
        "--n-surfaces", "-n", type=int, default=10,
        help="Number of isosurfaces to plot in multiple surface mode (default: 3)"
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.001,
        help="Minimum extinction threshold in m^-1 for multiple surface mode (default: 0.001)"
    )
    parser.add_argument(
        "--max-threshold", type=float, default=1,
        help="Maximum extinction threshold in m^-1 for multiple surface mode (default: 0.01)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.threshold, args.html,
         args.n_surfaces, args.min_threshold, args.max_threshold)


if __name__ == "__main__":
    cli()
