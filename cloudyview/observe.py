#!/usr/bin/env python
"""
observe.py: Cloud field volumetric visualization with pyvista.

Usage:
    python observe.py <filename.nc> [--output <path>] [--threshold <value>]

This script provides a volumetric rendering view of cloud data using:
1. Opacity calculation from liquid water content
2. Illumination based on overhead sun (cumulative opacity above each point)
3. PyVista volumetric rendering with sky blue background and path tracing
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc

from . import io, optical_depth


def main(filename: str, output: str = None, threshold: float = 0.001) -> None:
    """
    Main function for observe.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for renders
    threshold : float
        Opacity threshold for rendering (default: 0.001)
    """
    print(f"CloudyView Observe: Loading {filename}")
    start_time = time.perf_counter()

    try:
        # Import pyvista here to allow graceful error handling
        try:
            import pyvista as pv
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

        # Compute extinction coefficient
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0)

        # Convert extinction to optical depth per layer
        optical_depth_3d = sigma_ext * dz

        # Local opacity drives transparency directly (no remapping)
        local_opacity = 1.0 - np.exp(-optical_depth_3d)
        local_opacity = np.clip(local_opacity, 0.0, 1.0)

        # Calculate illumination: brightness = 1 - opacity_above
        # For overhead sun, we need opacity ABOVE each point
        # Cumsum from top gives us opacity above
        cumsum_above = np.cumsum(optical_depth_3d[:, :, ::-1], axis=2)[:, :, ::-1]
        opacity_above = 1.0 - np.exp(-cumsum_above)

        target_tau_half = 500.0
        tau_above = np.maximum(cumsum_above - optical_depth_3d, 0.0)
        illumination = np.exp(-tau_above / target_tau_half)
        illumination = np.clip(illumination, 0.0, 1.0)
        illumination_gamma = np.power(illumination, 0.85)

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Create structured grid for pyvista
        print("  Creating volumetric grid...")
        xx, yy, zz = np.meshgrid(x_coord, y_coord, z_coord, indexing='ij')

        # Create structured grid
        grid = pv.StructuredGrid(xx, yy, zz)
        grid['illumination'] = illumination_gamma.ravel(order='F')
        opacity_flat = local_opacity.ravel(order='F')
        # Build 256-sample piecewise opacity transfer based on raw opacity values
        lut_bins = 256
        edges = np.linspace(0.0, 1.0, lut_bins + 1)
        indices = np.clip(np.digitize(opacity_flat, edges) - 1, 0, lut_bins - 1)
        counts = np.bincount(indices, minlength=lut_bins)
        opacity_tf = np.bincount(indices, weights=opacity_flat, minlength=lut_bins)
        opacity_tf = np.divide(opacity_tf, counts, out=np.zeros_like(opacity_tf), where=counts > 0)
        # Fill gaps by nearest previous value
        for i in range(1, lut_bins):
            if counts[i] == 0:
                opacity_tf[i] = opacity_tf[i-1]
        opacity_tf = np.clip(opacity_tf, 0.0, 1.0)

        # Create plotter with sky blue background
        print("  Rendering volumetric view...")
        plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))

        # Sky blue background
        sky_blue = (58/255, 74/255, 166/255)
        plotter.background_color = sky_blue

        # Add volumetric rendering
        # Use combined field (illumination * local_opacity) for both color and transparency
        # This gives: white opaque (sunlit cloud), black opaque (shadowed cloud),
        # transparent clear regions
        volume = plotter.add_volume(
            grid,
            scalars='illumination',
            cmap='gray',
            opacity=opacity_tf.tolist(),
            shade=False,
            show_scalar_bar=False,
            clim=[0, 1],
        )

        # Set view angle - looking down at clouds from above at an angle
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
        output_file = output_dir / f"observe_volumetric_threshold={threshold:.3f}.png"
        plotter.screenshot(str(output_file))
        print(f"  ✓ Saved {output_file}")
        plotter.close()

        elapsed = time.perf_counter() - start_time
        print("\n✓ Observe complete!")
        print(f"  Total runtime: {elapsed:.1f} s ({elapsed/60:.1f} min)")
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
        description="Cloud visualization with volumetric pyvista rendering"
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
        "--threshold", type=float, default=0.001,
        help="Opacity threshold for rendering (default: 0.001)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.threshold)


if __name__ == "__main__":
    cli()
