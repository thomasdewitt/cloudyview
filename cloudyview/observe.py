#!/usr/bin/env python
"""
observe.py: Cloud field visualization with PyVista isosurface rendering.

Usage:
    # Multiple surfaces (default):
    python observe.py <filename.nc> [--interactive] [-n 10] [--min-threshold 0.001] [--max-threshold 1.0]

    # Single surface:
    python observe.py <filename.nc> --threshold 0.5 [--interactive]

This script generates 3D isosurface visualizations of cloud optical depth fields:
1. Computes optical depth from extinction coefficients (per-pixel, using geometric mean voxel size)
2. Renders white isosurface(s) with physically-based opacity: α = 1 - exp(-Δτ)
3. Multiple surfaces: logarithmically spaced optical depth thresholds
4. Produces two views: from below and from above
5. Optional interactive HTML export for browser-based 3D exploration
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
         interactive: bool = False, n_surfaces: int = 10,
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
    interactive : bool
        If True, export interactive HTML instead of static PNG
    n_surfaces : int
        Number of isosurfaces to plot (default: 3)
    min_threshold : float
        Minimum extinction threshold (default: 0.001 m^-1)
    max_threshold : float
        Maximum extinction threshold (default: 0.01 m^-1)
    """
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

        # Handle variable dz (common in atmospheric models)
        if len(z_coord) > 1:
            dz_array = np.diff(z_coord)
            # Pad to match nz (use last spacing for top level)
            dz_array = np.concatenate([dz_array, [dz_array[-1]]])
            dz_mean = float(np.mean(dz_array))
        else:
            dz_array = np.array([1.0])
            dz_mean = 1.0

        # Compute extinction coefficient and optical depth
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0)

        # Convert to optical depth (dimensionless) using voxel characteristic length
        # For variable dz, compute voxel_length at each vertical level
        voxel_length = (dx * dy * dz_array[np.newaxis, np.newaxis, :]) ** (1.0 / 3.0)
        tau = sigma_ext * voxel_length

        # Determine thresholds and opacities
        if threshold is not None:
            # Single surface mode
            thresholds = [threshold]
            opacities = [1.0]
        else:
            # Multiple surfaces mode - physically-based opacity
            # Logarithmically spaced thresholds from min to max
            thresholds = np.logspace(np.log10(min_threshold), np.log10(max_threshold), n_surfaces)

            # Calculate physical opacities based on shell thickness
            # Each surface i represents a shell from τᵢ to τᵢ₊₁
            # Opacity: α = 1 - exp(-Δτ) where Δτ = τᵢ₊₁ - τᵢ
            opacities = np.zeros(n_surfaces)
            for i in range(n_surfaces - 1):
                delta_tau = thresholds[i + 1] - thresholds[i]
                opacities[i] = 1.0 - np.exp(-delta_tau)
            # Innermost surface (highest τ) represents dense core - make it opaque
            opacities[-1] = 1.0

        # Create PyVista structured grid
        # PyVista expects dimensions in (nz, ny, nx) order for structured grids
        grid = pv.ImageData(dimensions=(nx, ny, nz))
        grid.spacing = (dx, dy, dz_mean)
        grid.origin = (x_coord[0], y_coord[0], z_coord[0])

        # Add optical depth field as point data
        # Flatten in Fortran order to match PyVista's expected ordering
        grid.point_data['optical_depth'] = tau.flatten(order='F')

        # Calculate domain geometry (needed for lighting and camera)
        center_x = (x_coord[0] + x_coord[-1]) / 2
        center_y = (y_coord[0] + y_coord[-1]) / 2
        center_z = (z_coord[0] + z_coord[-1]) / 2

        # Domain extent
        extent_x = x_coord[-1] - x_coord[0]
        extent_y = y_coord[-1] - y_coord[0]
        extent_z = z_coord[-1] - z_coord[0]
        max_extent = max(extent_x, extent_y, extent_z)

        # Create isosurfaces (skip empty ones)
        isosurfaces = []
        for thresh, opacity in zip(thresholds, opacities):
            isosurface = grid.contour([thresh], scalars='optical_depth')
            if isosurface.n_points > 0:
                isosurfaces.append((isosurface, opacity))

        if len(isosurfaces) == 0:
            print("✗ No surfaces generated - try lower thresholds", file=sys.stderr)
            sys.exit(1)

        # Lighting setup
        sun_azimuth = 70.0  # degrees
        sun_elevation = 45.0  # degrees above horizon
        light_distance = max_extent * 10
        az_rad = np.deg2rad(sun_azimuth)
        el_rad = np.deg2rad(sun_elevation)
        light_pos = [
            center_x + light_distance * np.cos(el_rad) * np.sin(az_rad),
            center_y + light_distance * np.cos(el_rad) * np.cos(az_rad),
            center_z + light_distance * np.sin(el_rad)
        ]

        # Dark blue background
        bg_color = (0.2, 0.3, 0.4)

        # Define camera positions: both from above
        camera_distance = max_extent * 1.5
        camera_positions = {
            'side': {
                'position': [
                    center_x,  # centered in x
                    center_y + camera_distance,  # offset in +y
                    center_z + max_extent * 0.8  # elevated above
                ],
                'focal_point': (center_x, center_y, center_z),
                'up': (0, 0, 1)
            },
            'top': {
                'position': [
                    center_x,  # centered in x
                    center_y,  # centered in y
                    center_z + camera_distance * 1.2  # directly above
                ],
                'focal_point': (center_x, center_y, center_z),
                'up': (0, 1, 0)  # y-axis as up for top-down view
            }
        }

        # Determine base output path
        if output:
            output_base = Path(output).stem
            output_dir = Path(output).parent
        else:
            # Use dataset filename in output
            dataset_name = Path(filename).stem
            output_base = f"cloudyview_observe"
            output_dir = Path(".")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset filename for output naming
        dataset_name = Path(filename).stem

        if interactive:
            # Interactive mode: single view (top), save HTML and show window
            plotter = pv.Plotter(off_screen=False, lighting='none', window_size=[1024, 1024])
            plotter.set_background(bg_color)

            # Add lighting
            sun_light = pv.Light(position=light_pos, light_type='scene light')
            sun_light.set_direction_angle(sun_elevation, sun_azimuth)
            sun_light.intensity = 1.0
            plotter.add_light(sun_light)
            plotter.add_light(pv.Light(light_type='headlight', intensity=0.3))

            # Add isosurfaces
            for isosurface, opacity in isosurfaces:
                plotter.add_mesh(isosurface, color='white', opacity=opacity, smooth_shading=True)

            # Set camera to "top" view
            plotter.camera_position = [
                camera_positions['top']['position'],
                camera_positions['top']['focal_point'],
                camera_positions['top']['up']
            ]

            # Save HTML
            if output:
                output_file = output_dir / f"{output_base}.html"
            else:
                output_file = output_dir / f"{output_base}_top_{dataset_name}.html"
            plotter.export_html(str(output_file))
            print(f"Saved: {output_file}")

            # Show interactive window
            plotter.show()

        else:
            # Static mode: render both views as PNG
            for view_name, camera_config in camera_positions.items():
                plotter = pv.Plotter(off_screen=True, lighting='none', window_size=[1024, 1024])
                plotter.set_background(bg_color)

                # Add lighting
                sun_light = pv.Light(position=light_pos, light_type='scene light')
                sun_light.set_direction_angle(sun_elevation, sun_azimuth)
                sun_light.intensity = 1.0
                plotter.add_light(sun_light)
                plotter.add_light(pv.Light(light_type='headlight', intensity=0.3))

                # Add isosurfaces
                for isosurface, opacity in isosurfaces:
                    plotter.add_mesh(isosurface, color='white', opacity=opacity, smooth_shading=True)

                # Set camera
                plotter.camera_position = [
                    camera_config['position'],
                    camera_config['focal_point'],
                    camera_config['up']
                ]

                # Save PNG
                if output:
                    output_file = output_dir / f"{output_base}_{view_name}.png"
                else:
                    output_file = output_dir / f"{output_base}_{view_name}_{dataset_name}.png"
                plotter.screenshot(str(output_file))
                print(f"Saved: {output_file}")

                plotter.close()

        elapsed = time.perf_counter() - start_time
        print(f"✓ Complete ({elapsed:.1f}s)")

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
        description="Cloud visualization with PyVista isosurface rendering (generates two views: below & above)"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output base name (generates <name>_below.png and <name>_above.png)"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=None,
        help="Single optical depth threshold for isosurface (dimensionless). If not set, uses multiple surfaces."
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Export interactive HTML instead of static PNG (fully portable, open in any browser)"
    )
    parser.add_argument(
        "--n-surfaces", "-n", type=int, default=10,
        help="Number of isosurfaces to plot in multiple surface mode (default: 10)"
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.001,
        help="Minimum optical depth threshold for multiple surface mode (default: 0.001)"
    )
    parser.add_argument(
        "--max-threshold", type=float, default=1,
        help="Maximum optical depth threshold for multiple surface mode (default: 1.0)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.threshold, args.interactive,
         args.n_surfaces, args.min_threshold, args.max_threshold)


if __name__ == "__main__":
    cli()
