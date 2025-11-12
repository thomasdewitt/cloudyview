#!/usr/bin/env python
"""
odyssey.py: Comprehensive cloud field visualization with full 3D radiative transfer (Mitsuba).

Usage:
    python odyssey.py <filename.nc> [--output <path>] [--sza <angle>] [--wavelength <wl>]

This script provides comprehensive analysis of your cloud data using:
1. Optical depth calculation via extinction coefficient
2. Mitsuba 3 Monte Carlo path tracing with physically-based sky
3. A high-quality (2048 spp) ground-looking-up view matching Expedition's framing
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc
import time

from . import io, optical_depth, radiative_transfer, basic_render


def main(
    filename: str,
    output: str = None,
    sza: float = 45.0,
    wavelength: float = 0.55
) -> None:
    """
    Main function for odyssey.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for renders
    sza : float
        Solar zenith angle in degrees (default: 45)
    wavelength : float
        Wavelength in microns (for reference, default: 0.55 visible)
    """
    start_time = time.perf_counter()

    print(f"CloudyView Odyssey: Loading {filename}")

    try:
        # Load and validate data with xarray
        data_dict = io.load_and_validate(filename)
        ds = data_dict['dataset']
        lw_var = data_dict['liquid_water_var']
        lw_data = data_dict['liquid_water_data']
        iw_data = data_dict['ice_water_data']

        print(f"✓ Loaded {lw_var} variable")
        print(f"  Shape: {lw_data.shape}")
        print(f"  Range: {lw_data.min().values:.4f} - {lw_data.max().values:.4f} g/kg")

        if iw_data is not None:
            print(f"✓ Also loaded ice water variable")
            print(f"  Range: {iw_data.min().values:.4f} - {iw_data.max().values:.4f} g/kg")

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

        print(f"  Grid spacing: dx={dx:.1f}m, dy={dy:.1f}m, dz={dz:.1f}m")

        # Compute extinction coefficient
        print("\nComputing extinction coefficient...")
        sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0)
        print(f"✓ Extinction range: {sigma_ext.min():.6e} - {sigma_ext.max():.6e} m^-1")

        # Domain dimensions
        width_x = nx * dx
        width_y = ny * dy
        height_z = nz * dz
        aspect_ratio = width_x / height_z

        # Domain center at origin
        domain_center = [0.0, 0.0, 0.0]
        ar = aspect_ratio

        # Camera position scaling based on FOV and domain width
        # Cube spans [-ar, ar] x [-ar, ar] x [-1, 1], so width = 2*ar, height = 2
        # For a perspective camera: visible_width = 2 * distance * tan(fov/2)
        # We want: domain_width = visible_width / margin
        # So: distance = (margin * domain_width) / (2 * tan(fov/2))
        fov_ground = 80.0  # match Expedition
        margin = 1.1  # Domain takes up 1/1.1 ≈ 91% of image width

        # Use distance to offset camera along +y while keeping z near the ocean surface
        boa_distance = (margin * 2 * ar) / (2 * np.tan(np.deg2rad(fov_ground / 2)))
        camera_height = -0.9
        camera_origin = [0.0, ar + boa_distance, camera_height]

        print(f"\nDomain:")
        print(f"  Physical size: {width_x/1000:.1f} x {width_y/1000:.1f} x {height_z/1000:.1f} km")
        print(f"  Aspect ratio (x/z): {aspect_ratio:.2f}")
        print(f"  Scaled width: {ar:.1f}")
        print(f"  Camera (ground view, margin={margin:.1f}x, fov={fov_ground:.0f}°):")
        print(f"    y-offset={camera_origin[1]:.1f}, z={camera_origin[2]:.1f}")

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Initialize Mitsuba
        print("\nSetting up Mitsuba 3...")
        import mitsuba as mi
        mi.set_variant('llvm_ad_rgb')

        # High-quality ground-looking-up view (matches Expedition framing)
        views = [
            {
                'name': 'Ground-Looking-Up (2048 SPP)',
                'width': 800,
                'height': 800,
                'fov': fov_ground,
                'transform': radiative_transfer.look_at_world_up(
                    origin=camera_origin,
                    target=domain_center
                ),
                'camera_origin': camera_origin,
                'spp': 2048,
                'exposure': 4.0,
                'extinction_multiplier': 1.0,
                'sky_type': 'sunsky',
                'turbidity': 3.0,
                'sun_azimuth': 270.0,
                'sun_elevation': 90.0 - sza,
                'ground_albedo': 0.5,
                'add_ocean': True,
                'ocean_reflectance': [0.2, 0.3, 0.45],
                'ocean_height': -0.99,
                'seed': 0,
            }
        ]

        # Render the view
        print(f"\nRendering {len(views)} view with {views[0]['spp']} SPP...")
        print("(This will still take several minutes—grab a coffee!)")
        print("=" * 60)

        for i, view in enumerate(views):
            safe_name = view['name'].replace(' ', '_').replace('(', '').replace(')', '').lower()
            output_file = output_dir / f"odyssey_{i+1}_{safe_name}.png"
            radiative_transfer.render_view(sigma_ext, dx, dy, dz, view, str(output_file))

        # Summary statistics
        print("\n" + "="*60)
        print("ODYSSEY ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nCloud Water Content ({lw_var}):")
        print(f"  Min: {lw_data.min().values:.4f} g/kg")
        print(f"  Max: {lw_data.max().values:.4f} g/kg")
        print(f"  Mean: {lw_data.mean().values:.4f} g/kg")

        print(f"\nExtinction Coefficient:")
        print(f"  Min: {sigma_ext.min():.6e} m^-1")
        print(f"  Max: {sigma_ext.max():.6e} m^-1")
        print(f"  Mean: {sigma_ext.mean():.6e} m^-1")

        print(f"\nRadiative Transfer Configuration:")
        print(f"  Solar zenith angle: {sza}°")
        print(f"  Wavelength: {wavelength} μm (reference)")
        print(f"  Sky model: Physically-based (Hosek-Wilkie)")
        print(f"  Samples per pixel: 2048 (high quality)")

        elapsed = time.perf_counter() - start_time
        print(f"\nTotal runtime: {elapsed:.1f} s ({elapsed/60:.1f} minutes)")
        print("="*60)

        if output:
            print(f"\n✓ Odyssey complete! Renders saved to {output_dir}")
        else:
            print(f"\n✓ Odyssey complete!")

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
    """Command-line interface for odyssey.py"""
    parser = argparse.ArgumentParser(
        description="Comprehensive cloud visualization with full 3D Mitsuba radiative transfer (2048 SPP ground view)"
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
        help="Solar zenith angle in degrees (default: 45)"
    )
    parser.add_argument(
        "--wavelength", type=float, default=0.55,
        help="Wavelength in microns for reference (default: 0.55 for visible)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.sza, args.wavelength)


if __name__ == "__main__":
    cli()
