#!/usr/bin/env python
"""
behold.py: Cloud field visualization with optical depth and 3D radiative transfer (Mitsuba).

Usage:
    python behold.py <filename.nc> <backend> [quality] [--output <path>]

    backend: llvm or cuda (required)
    quality: min (200x400), low (400x200), medium (800x400, default), high (1600x800)

This script provides a realistic 3D view of your cloud data using:
1. Optical depth calculation via extinction coefficient
2. Mitsuba 3 Monte Carlo path tracing with physically-based sky
3. Accurate Mie scattering phase functions from Bouthors (2008)
4. Configurable camera and sun positions via config file

Configuration is loaded from cloudyview.yaml (current dir) or ~/.cloudyview.yaml
Settings include camera position, sun angle, rendering parameters, etc.

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import netCDF4 as nc
import mitsuba as mi

from . import io, radiative_transfer, optical_depth, config


def main(filename: str, backend: str, quality: str = 'medium', output: str = None) -> None:
    """
    Main function for behold.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    backend : str
        Mitsuba backend: 'llvm' or 'cuda'
    quality : str
        Render quality: 'min' (200x400, spp=1), 'low' (400x200),
        'medium' (800x400, default), 'high' (1600x800)
    output : str, optional
        Output directory for renders
    """
    print(f"CloudyView Behold: Loading {filename}")
    start_time = time.perf_counter()

    # Load configuration
    behold_config = config.get_behold_config()
    camera_config = behold_config['camera']
    sun_config = behold_config['sun']
    rendering_config = behold_config['rendering']

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

        # Process ice water content if present
        iw_np = None
        ice_fraction = None
        has_ice = False

        if iw_data is not None:
            iw_np = iw_data.values
            if 'time' in iw_data.dims:
                iw_np = iw_np[0]  # Remove time dimension if present

            # Check if there's actually ice in the volume
            if np.max(iw_np) > 1e-6:
                has_ice = True
                print(f"  Ice water content detected (max: {np.max(iw_np):.6f} g/kg)")

                # Compute ice fraction (0 = liquid, 1 = ice)
                # Avoid division by zero
                total_water = lw_np + iw_np
                ice_fraction = np.divide(iw_np, total_water,
                                        out=np.zeros_like(iw_np),
                                        where=total_water > 1e-10)
            else:
                print("  Ice water content negligible, using liquid-only rendering")
                iw_np = None
        else:
            print("  No ice water content in dataset")

        # Compute extinction coefficient (liquid + ice if present)
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0,
                                                          iwc=iw_np, re_ice=30.0)

        # Domain dimensions
        width_x = nx * dx
        width_y = ny * dy
        height_z = nz * dz
        aspect_ratio = width_x / height_z


        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Set Mitsuba backend explicitly based on CLI argument
        # Validate backend choice
        if backend.lower() not in ['llvm', 'cuda']:
            raise ValueError(f"Invalid backend '{backend}'. Must be 'llvm' or 'cuda'.")

        # Determine render mode from config (default to 'rgb')
        render_mode = 'rgb'  # Fixed to RGB mode for now
        variant = f'{backend.lower()}_ad_{render_mode}'
        mi.set_variant(variant)
        print(f"  Using Mitsuba variant: {variant}")

        # Map quality to resolution and spp
        quality_map = {
            'min': {'resolution': (150, 100), 'spp': 1},
            'low': {'resolution': (300, 200), 'spp': 32},
            'medium': {'resolution': (600, 400), 'spp': 512},
            'high': {'resolution': (1200, 800), 'spp': 4096}
        }
        width, height = quality_map[quality]['resolution']
        spp = quality_map[quality]['spp']

        # Get camera settings from config (relative coordinates)
        camera_azimuth = camera_config['azimuth']
        camera_elevation = camera_config['elevation']
        fov = camera_config['fov']

        # Convert relative camera position to absolute
        # Relative coords: ±1.0 = domain edge
        rel_pos = camera_config['position']
        camera_origin = [
            rel_pos[0] * aspect_ratio,  # x in Mitsuba normalized cube (±ar)
            rel_pos[1] * aspect_ratio,  # y in Mitsuba normalized cube (±ar)
            rel_pos[2]        # z in Mitsuba normalized cube (±1)
        ]

        # Compute camera target from azimuth/elevation
        # Convert to radians
        az_rad = np.deg2rad(camera_azimuth)
        el_rad = np.deg2rad(camera_elevation)

        # Compute look direction vector (meteorological convention)
        # Azimuth: 0°=East(+x), 90°=North(+y)
        # Elevation: angle above horizon
        look_dir = np.array([
            np.cos(el_rad) * np.cos(az_rad),  # x (East)
            np.cos(el_rad) * np.sin(az_rad),  # y (North)
            np.sin(el_rad)  # z (Up)
        ])

        # Target point is origin + look direction
        camera_target = camera_origin + look_dir

        # Get sun configuration
        sun_azimuth = sun_config['azimuth']
        sun_elevation = sun_config['elevation']

        # Render ground-looking-up view with progressive checkpoints
        print(f"  Camera offset: x={camera_origin[0]:.1f}, y={camera_origin[1]:.1f}, z={camera_origin[2]:.1f}")
        print(f"  Camera azimuth: {camera_azimuth:.1f}°, elevation: {camera_elevation:.1f}°")
        print(f"  Sun azimuth: {sun_azimuth:.1f}°, elevation: {sun_elevation:.1f}°")
        print(f"  Field of view: {fov:.1f}°")
        print(f"  Render quality: {quality} ({width}x{height}, spp={spp})")
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
            'spp': spp,
            'exposure': rendering_config['exposure'],
            'extinction_multiplier': rendering_config['extinction_multiplier'],
            'sky_type': 'sunsky',  # Physically-based sky
            'turbidity': rendering_config['turbidity'],
            'sun_azimuth': sun_azimuth,
            'sun_elevation': sun_elevation,
            'ground_albedo': rendering_config['ground_albedo'],
            'add_ocean': rendering_config['ocean']['enabled'],
            'ocean_reflectance': rendering_config['ocean']['reflectance'],
            'ocean_height': rendering_config['ocean']['height'],
            'integrator': rendering_config['integrator'],
            'max_depth': rendering_config['max_depth'],
            'rr_depth': rendering_config['rr_depth'],
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
        checkpoint_spp = [2, 32, 128, 512, 1024, 2048, 4096, 8192]

        output_file = output_dir / f"behold_ground_view_max_depth={view_config['max_depth']}_rr_depth={view_config['rr_depth']}.png"
        radiative_transfer.render_view(
            sigma_ext, dx, dy, dz, view_config, str(output_file),
            checkpoint_spp=checkpoint_spp,
            ice_fraction=ice_fraction
        )

        elapsed = time.perf_counter() - start_time
        print("\n✓ Behold complete!")
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
    """Command-line interface for behold.py"""
    parser = argparse.ArgumentParser(
        description="Cloud visualization with 3D Mitsuba radiative transfer (photorealistic ground-looking-up view)"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "backend",
        choices=['llvm', 'cuda'],
        help="Mitsuba backend: llvm (CPU) or cuda (GPU)"
    )
    parser.add_argument(
        "quality",
        nargs='?',
        default='medium',
        choices=['min', 'low', 'medium', 'high'],
        help="Render quality: min (200x400, spp=1), low (400x200, spp=32), medium (800x400, spp=512, default), high (1600x800, spp=4096)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for saving renders (default: current directory)"
    )

    args = parser.parse_args()
    main(args.filename, args.backend, args.quality, args.output)


if __name__ == "__main__":
    cli()
