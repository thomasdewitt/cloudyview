#!/usr/bin/env python
"""
observe.py: Cloud field surface visualization with pyvista.

Usage:
    python observe.py <filename.nc> [--output <path>] [--tau-threshold <value>]

This script provides a surface rendering view of cloud data using:
1. Optical depth calculation from liquid water content
2. Surface extraction at optical depth thresholds
3. PyVista surface rendering with lighting and shadows
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc

from . import io, optical_depth


def main(filename: str, output: str = None, tau_threshold: float = 0.5, tau_threshold_2: float = None) -> None:
    """
    Main function for observe.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for renders
    tau_threshold : float
        Optical depth threshold for first surface (default: 0.5)
    tau_threshold_2 : float, optional
        Optical depth threshold for second surface (default: None)
    """
    print(f"CloudyView Observe: Loading {filename}")
    start_time = time.perf_counter()

    try:
        # Import pyvista here to allow graceful error handling
        try:
            import pyvista as pv
            # Start Xvfb for offscreen rendering
            pv.start_xvfb()
        except ImportError:
            print("✗ PyVista required but not installed", file=sys.stderr)
            print("  Install with: pip install pyvista", file=sys.stderr)
            sys.exit(1)

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

        print(f"  Grid: {nx}x{ny}x{nz}, spacing: {dx:.1f}x{dy:.1f}x{dz:.1f}m")

        # Compute extinction coefficient
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0)

        # Convert extinction to optical depth per layer
        optical_depth_per_layer = sigma_ext * dz

        # Calculate cumulative optical depth from top down
        print(f"  Calculating optical depth from top down...")
        tau_from_top = np.cumsum(optical_depth_per_layer[:, :, ::-1], axis=2)[:, :, ::-1]

        print(f"  Max optical depth from top: {tau_from_top.max():.3f}")
        print(f"  Creating surface at tau = {tau_threshold}...")

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Create structured grid for pyvista
        xx, yy, zz = np.meshgrid(x_coord, y_coord, z_coord, indexing='ij')
        grid = pv.StructuredGrid(xx, yy, zz)
        grid['tau_from_top'] = tau_from_top.ravel(order='F')
        grid['local_optical_depth'] = optical_depth_per_layer.ravel(order='F')

        # Extract isosurface at tau threshold
        surface = grid.contour([tau_threshold], scalars='tau_from_top')

        # Filter surface to only keep points where there's actual cloud nearby
        # This prevents vertical columns through clear air
        if surface.n_points > 0:
            # Get the local optical depth at each surface point
            surface = surface.compute_cell_sizes()
            # Threshold: keep only surface regions where local optical depth is significant
            cloud_threshold = 1e-6
            surface = surface.threshold(cloud_threshold, scalars='local_optical_depth', preference='point')

        if surface.n_points == 0:
            print(f"  Warning: No surface found at tau={tau_threshold}")
            print(f"  Try a different threshold value")
            sys.exit(1)

        print(f"  Surface extracted: {surface.n_points} points, {surface.n_cells} cells")

        # Extract second surface if requested
        surface_2 = None
        if tau_threshold_2 is not None:
            print(f"  Creating second surface at tau = {tau_threshold_2}...")
            surface_2 = grid.contour([tau_threshold_2], scalars='tau_from_top')

            # Filter second surface the same way
            if surface_2.n_points > 0:
                surface_2 = surface_2.compute_cell_sizes()
                surface_2 = surface_2.threshold(cloud_threshold, scalars='local_optical_depth', preference='point')

            if surface_2.n_points > 0:
                print(f"  Second surface: {surface_2.n_points} points, {surface_2.n_cells} cells")
            else:
                print(f"  Warning: No second surface found at tau={tau_threshold_2}")
                surface_2 = None

        # Create plotter with sky blue background
        print("  Rendering surface view...")
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))

        # Sky blue background
        sky_blue = (0.53, 0.81, 0.92)  # Light blue
        plotter.background_color = sky_blue

        # Add first surface - opaque white with shadows
        if surface_2 is None:
            # Single surface - opaque white
            plotter.add_mesh(
                surface,
                color='white',
                smooth_shading=True,
                show_edges=False,
                opacity=1.0,
                lighting=True,
                specular=0.3,
                specular_power=15,
            )
        else:
            # Two surfaces - first one semi-transparent, second one opaque
            plotter.add_mesh(
                surface,
                color='white',
                smooth_shading=True,
                show_edges=False,
                opacity=0.6,
                lighting=True,
                specular=0.3,
                specular_power=15,
            )
            plotter.add_mesh(
                surface_2,
                color='white',
                smooth_shading=True,
                show_edges=False,
                opacity=1.0,
                lighting=True,
                specular=0.3,
                specular_power=15,
            )

        # Add a directional light (like the sun) to create shadows
        light = pv.Light(position=(np.max(x_coord)*2, np.max(y_coord)*2, np.max(z_coord)*3),
                        focal_point=(np.mean(x_coord), np.mean(y_coord), np.mean(z_coord)),
                        color='white',
                        intensity=0.8)
        plotter.add_light(light)

        # Enable shadows for realistic perspective
        plotter.enable_shadows()

        # Set view angle - looking at clouds from above at an angle
        domain_center = [np.mean(x_coord), np.mean(y_coord), np.mean(z_coord)]
        domain_size = [np.ptp(x_coord), np.ptp(y_coord), np.ptp(z_coord)]

        # Camera positioned above and to the side
        plotter.camera.position = [
            domain_center[0] + 0.8 * domain_size[0],
            domain_center[1] + 0.8 * domain_size[1],
            domain_center[2] + 1.2 * domain_size[2]
        ]
        plotter.camera.focal_point = domain_center

        # Output file
        if tau_threshold_2 is None:
            output_file = output_dir / f"observe_surface_tau={tau_threshold:.2f}.png"
        else:
            output_file = output_dir / f"observe_surfaces_tau={tau_threshold:.2f}_{tau_threshold_2:.2f}.png"

        plotter.screenshot(str(output_file))
        print(f"  ✓ Saved {output_file}")
        plotter.close()

        elapsed = time.perf_counter() - start_time
        print("\n✓ Observe complete!")
        print(f"  Total runtime: {elapsed:.1f} s")
        if output:
            print(f"  Renders saved to {output_dir}")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"✗ Import error: {e}", file=sys.stderr)
        print("  Make sure all required packages are installed", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cli():
    """Command-line interface for observe.py"""
    parser = argparse.ArgumentParser(
        description="Cloud visualization with surface rendering based on optical depth thresholds"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for saving renders"
    )
    parser.add_argument(
        "--tau-threshold", type=float, default=0.5,
        help="Optical depth threshold for first surface (default: 0.5)"
    )
    parser.add_argument(
        "--tau-threshold-2", type=float, default=None,
        help="Optical depth threshold for second surface (optional)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.tau_threshold, args.tau_threshold_2)


if __name__ == "__main__":
    cli()
