#!/usr/bin/env python
"""
appreciate.py: Cloud field visualization with optical depth and 3D radiative transfer (Mitsuba).

Usage:
    python appreciate.py <filename.nc> [quality] [options]

    quality: low (400x200), medium (800x400, default), high (1600x800)

Options:
    --output <path>           Output directory for renders
    --sza <angle>             Solar zenith angle in degrees (default: 45)
    --camera-azimuth <angle>  Camera look azimuth in degrees (default: 0)
    --camera-elevation <angle> Camera look elevation in degrees (default: 45)
    --fov <angle>             Field of view in degrees (default: 100)
    --mode <mode>             mono/rgb/chromatic (default: rgb)

This script provides a realistic 3D view of your cloud data using:
1. Optical depth calculation via extinction coefficient
2. Mitsuba 3 Monte Carlo path tracing with physically-based sky
3. Accurate Mie scattering phase functions from Bouthors (2008)
4. Optional RGB wavelength-dependent rendering for chromatic effects
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc
import mitsuba as mi

from . import io, radiative_transfer, optical_depth


def main(filename: str, output: str = None, sza: float = 50.0, quality: str = 'medium',
         camera_azimuth: float = 0.0, camera_elevation: float = 45.0, fov: float = 100.0,
         render_mode: str = 'rgb') -> None:
    """
    Main function for appreciate.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for renders
    sza : float
        Solar zenith angle in degrees
    quality : str
        Render quality: 'low' (400x200), 'medium' (800x400, default), 'high' (1600x800)
    camera_azimuth : float
        Camera look direction azimuth in degrees (0=+X, 90=+Y, 180=-X, 270=-Y)
    camera_elevation : float
        Camera look direction elevation in degrees (0=horizontal, 90=up, -90=down)
    fov : float
        Camera field of view in degrees (default: 100)
    render_mode : str
        Rendering mode: 'mono', 'rgb', or 'chromatic'
        - 'mono': Single grayscale render, fastest
        - 'rgb': Single RGB render with sunsky (default)
        - 'chromatic': 3-channel render for chromatic effects (coronas, halos, glories)
    """
    print(f"CloudyView Appreciate: Loading {filename}")
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
        domain_center = [0, 0, 0.5]
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

        # Use CUDA if available, otherwise LLVM
        variant = radiative_transfer.get_best_variant('rgb')
        mi.set_variant(variant)
        print(f"  Using Mitsuba variant: {variant}")

        # Map quality to resolution
        quality_map = {
            'low': (400, 200),
            'medium': (800, 400),
            'high': (1600, 800)
        }
        width, height = quality_map[quality]

        # Compute camera target from azimuth/elevation
        # Convert to radians
        az_rad = np.deg2rad(camera_azimuth)
        el_rad = np.deg2rad(camera_elevation)

        # Compute look direction vector
        # Azimuth: 0=+X, 90=+Y, 180=-X, 270=-Y
        # Elevation: 0=horizontal, 90=zenith, -90=nadir
        look_dir = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])

        # Target point is origin + look direction
        camera_target = camera_origin + look_dir

        # Render ground-looking-up view with progressive checkpoints
        print(f"  Camera offset: x={camera_origin[0]:.1f}, y={camera_origin[1]:.1f}, z={camera_origin[2]:.1f}")
        print(f"  Camera azimuth: {camera_azimuth:.1f}°, elevation: {camera_elevation:.1f}°")
        print(f"  Field of view: {fov:.1f}°")
        print(f"  Render quality: {quality} ({width}x{height})")
        view_config = {
            'name': 'Ground-Looking-Up (Progressive Rendering)',
            'width': width,
            'height': height,
            'fov': fov,
            'transform': radiative_transfer.look_at_world_up(
                origin=camera_origin,
                target=camera_target
            ),
            'camera_origin': camera_origin,
            'spp': 256, 
            'exposure': 4.0,
            'extinction_multiplier': 1.0,
            'sky_type': 'sunsky',  # Physically-based sky
            'turbidity': 3.0,
            'sun_azimuth': 90.0,
            'sun_elevation': 90.0 - sza,  # Convert zenith angle to elevation
            'ground_albedo': 0.5,
            'add_ocean': True,
            'ocean_reflectance': [0.0392, 0.1098, 0.1490],  # #0A1C26 = RGB(10, 28, 38)
            'ocean_height': -.99,
            'integrator': 'volpathmis',
            'max_depth': 128,
            'rr_depth': 64,
            'sampler': {'type': 'independent'},
            'seed': 0,
            'render_mode': render_mode,
        }

        # Sobol prefers power-of-two spp; adjust if necessary
        samples = view_config['spp']
        next_pow2 = 1 << (samples - 1).bit_length()
        if next_pow2 != samples:
            print(f"  Adjusting spp from {samples} to {next_pow2} for Sobol sampler")
            view_config['spp'] = next_pow2

        view_config['sampler']['sample_count'] = view_config['spp']

        # Define checkpoint SPP values for progressive rendering
        checkpoint_spp = [2, 32, 128, 512, 1028, 2048, 4096]

        output_file = output_dir / f"appreciate_ground_view_max_depth={view_config['max_depth']}_rr_depth={view_config['rr_depth']}_spp={view_config['spp']}.png"
        radiative_transfer.render_view(
            sigma_ext, dx, dy, dz, view_config, str(output_file),
            checkpoint_spp=checkpoint_spp
        )

        elapsed = time.perf_counter() - start_time
        print("\n✓ Appreciate complete!")
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
    """Command-line interface for appreciate.py"""
    parser = argparse.ArgumentParser(
        description="Cloud visualization with 3D Mitsuba radiative transfer (ground-looking-up view)"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "quality",
        nargs='?',
        default='medium',
        choices=['low', 'medium', 'high'],
        help="Render quality: low (400x200), medium (800x400, default), high (1600x800)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for saving renders"
    )
    parser.add_argument(
        "--sza", type=float, default=45.0,
        help="Solar zenith angle in degrees (default: 45 for realistic perspective)"
    )
    parser.add_argument(
        "--camera-azimuth", type=float, default=0.0,
        help="Camera look direction azimuth in degrees (0=+X/east, 90=+Y/north, default: 0)"
    )
    parser.add_argument(
        "--camera-elevation", type=float, default=45.0,
        help="Camera look direction elevation in degrees (0=horizontal, 90=up, default: 45)"
    )
    parser.add_argument(
        "--fov", type=float, default=100.0,
        help="Camera field of view in degrees (default: 100)"
    )
    parser.add_argument(
        "--mode", type=str, default='rgb',
        choices=['mono', 'rgb', 'chromatic'],
        help="Rendering mode: mono (grayscale), rgb (default, full color), chromatic (coronas/halos)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.sza, args.quality,
         args.camera_azimuth, args.camera_elevation, args.fov, args.mode)


if __name__ == "__main__":
    cli()
