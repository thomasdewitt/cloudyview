#!/usr/bin/env python
"""
expedition.py: Cloud field visualization with optical depth and 3D radiative transfer (Mitsuba).

Usage:
    python expedition.py <filename.nc> [--output <path>] [--sza <angle>]

This script provides a realistic 3D view of your cloud data using:
1. Optical depth calculation via extinction coefficient
2. Mitsuba 3 Monte Carlo path tracing with physically-based sky
3. Ground-looking-up perspective with 16 samples per pixel
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc

from . import io, optical_depth, radiative_transfer, basic_render


def main(filename: str, output: str = None, sza: float = 70.0) -> None:
    """
    Main function for expedition.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for renders
    sza : float
        Solar zenith angle in degrees
    """
    print(f"CloudyView Expedition: Loading {filename}")
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


        # Compute extinction coefficient
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0)

        # Domain dimensions
        width_x = nx * dx
        width_y = ny * dy
        height_z = nz * dz
        aspect_ratio = width_x / height_z

        # Domain center at origin
        domain_center = [0, 0, 0]
        ar = aspect_ratio

        # Camera position scaling based on FOV and domain width
        # Cube spans [-ar, ar] x [-ar, ar] x [-1, 1], so width = 2*ar
        # For a perspective camera: visible_width = 2 * distance * tan(fov/2)
        # We want: domain_width = visible_width / margin
        # So: distance = (margin * domain_width) / (2 * tan(fov/2))
        fov_for_full_domain = 70.0  # degrees
        boa_distance = (2 * ar) / (2 * np.tan(np.deg2rad(fov_for_full_domain / 2)))
        # print(boa_distance)

        # Place camera just above ocean plane but offset in +y to frame the scene
        camera_height = -0.5  # inside the cube but above reflective ocean
        # camera_origin = [0.0, ar , camera_height]
        # camera_origin = [0.0, ar + boa_distance, camera_height]
        camera_origin = [0.0, - ar * 3/4, camera_height]

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        import mitsuba as mi
        mi.set_variant('llvm_ad_rgb')

        # Render ground-looking-up view with 16 SPP
        print(f"  Camera offset: x={camera_origin[0]:.1f}, y={camera_origin[1]:.1f}, z={camera_origin[2]:.1f}")
        view_config = {
            'name': 'Ground-Looking-Up (16 SPP)',
            'width': 300,
            'height': 150,
            'fov': 100,
            'transform': radiative_transfer.look_at_world_up(
                origin=camera_origin,
                target=domain_center
            ),
            'camera_origin': camera_origin,
            'spp': 2,
            'exposure': 4.0,
            'extinction_multiplier': 1.0,
            'sky_type': 'sunsky',  # Physically-based sky
            'turbidity': 3.0,
            'sun_azimuth': 270.0,
            'sun_elevation': 90.0 - sza,  # Convert zenith angle to elevation
            'ground_albedo': 0.5,
            'add_ocean': True,
            'ocean_reflectance': [0.2, 0.3, 0.45],
            'ocean_height': -.99,
            'integrator': 'volpathmis',
            'max_depth': 512,
            'rr_depth': 512,
            'sampler': {'type': 'independent'},
            'seed': 0,
        }

        # Sobol prefers power-of-two spp; adjust if necessary
        samples = view_config['spp']
        next_pow2 = 1 << (samples - 1).bit_length()
        if next_pow2 != samples:
            print(f"  Adjusting spp from {samples} to {next_pow2} for Sobol sampler")
            view_config['spp'] = next_pow2

        view_config['sampler']['sample_count'] = view_config['spp']

        output_file = output_dir / f"expedition_ground_view_max_depth={view_config['max_depth']}_rr_depth={view_config['rr_depth']}_spp={view_config['spp']}.png"
        radiative_transfer.render_view(sigma_ext, dx, dy, dz, view_config, str(output_file))

        elapsed = time.perf_counter() - start_time
        print("\n✓ Expedition complete!")
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
        print(f"✗ Mitsuba 3 required but not installed: {e}", file=sys.stderr)
        print("  Install with: pip install mitsuba drjit", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cli():
    """Command-line interface for expedition.py"""
    parser = argparse.ArgumentParser(
        description="Cloud visualization with 3D Mitsuba radiative transfer (ground-looking-up view)"
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
        "--sza", type=float, default=45.0,
        help="Solar zenith angle in degrees (default: 45 for realistic perspective)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.sza)


if __name__ == "__main__":
    cli()
