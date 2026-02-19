#!/usr/bin/env python
"""
behold.py: Cloud field visualization with optical depth and 3D radiative transfer (Mitsuba).

Usage:
    python behold.py <filename.nc> <backend> [quality] [--output <path>]

    backend: llvm or cuda (required)
    quality: min (150x100, spp=1, max_depth=4, rr_depth=2),
             low (300x200, spp=32, max_depth=16, rr_depth=4),
             medium (600x400, spp=512, max_depth=64, rr_depth=16, default),
             high (1200x800, spp=4096, max_depth=128, rr_depth=32),
             custom

    For custom quality, use: --spp N --size W H --max-depth N --rr-depth N

    Camera/sun options (override config file):
        --camera-position X Y Z   Camera position in relative coords (±1.0 = domain edge)
        --camera-azimuth DEG      Camera azimuth (0=North, 90=East, 180=South, 270=West)
        --camera-elevation DEG    Camera elevation (angle above horizon)
        --fov DEG                 Camera field of view
        --sun-azimuth DEG         Sun azimuth (0=North, 90=East, 180=South, 270=West)
        --sun-elevation DEG       Sun elevation (angle above horizon)

This script provides a realistic 3D view of your cloud data using:
1. Optical depth calculation via extinction coefficient
2. Mitsuba 3 Monte Carlo path tracing with physically-based sky
3. Accurate Mie scattering phase functions from Bouthors (2008)
4. Configurable camera and sun positions via defaults + CLI overrides

Rendering uses built-in default settings from config.py.
CLI arguments override those defaults.

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
import mitsuba as mi

from . import io, radiative_transfer, optical_depth, config
from .angles import direction_from_azimuth_elevation


def main(filename: str, backend: str, quality: str = 'medium', output: str = None,
         custom_spp: int = None, custom_size: tuple = None,
         custom_max_depth: int = None, custom_rr_depth: int = None,
         camera_position: list = None, camera_azimuth: float = None,
         camera_elevation: float = None, camera_fov: float = None,
         sun_azimuth: float = None, sun_elevation: float = None,
         progress_interval: int = None) -> None:
    """
    Main function for behold.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    backend : str
        Mitsuba backend: 'llvm' or 'cuda'
    quality : str
        Render quality: 'min' (150x100, spp=1, max_depth=4, rr_depth=2),
        'low' (300x200, spp=32, max_depth=16, rr_depth=4),
        'medium' (600x400, spp=512, max_depth=64, rr_depth=16, default),
        'high' (1200x800, spp=4096, max_depth=128, rr_depth=32),
        or 'custom' (user-specified via --spp, --size, --max-depth, --rr-depth)
    output : str, optional
        Output directory for renders
    camera_position : list, optional
        Camera position [x, y, z] in relative coords (±1.0 = domain edge)
    camera_azimuth : float, optional
        Camera azimuth in degrees (0=North, 90=East, 180=South, 270=West)
    camera_elevation : float, optional
        Camera elevation in degrees (angle above horizon)
    camera_fov : float, optional
        Camera field of view in degrees
    sun_azimuth : float, optional
        Sun azimuth in degrees (0=North, 90=East, 180=South, 270=West)
    sun_elevation : float, optional
        Sun elevation in degrees (angle above horizon)
    progress_interval : int, optional
        Print progress every N samples (default: 2 for rgb/mono, 16 for chromatic)
    """
    print(f"CloudyView Behold: Loading {filename}")
    start_time = time.perf_counter()

    # Load configuration
    behold_config = config.get_behold_config()
    camera_config = behold_config['camera']
    sun_config = behold_config['sun']
    rendering_config = behold_config['rendering']

    # Apply CLI overrides to camera config
    if camera_position is not None:
        camera_config['position'] = list(camera_position)
    if camera_azimuth is not None:
        camera_config['azimuth'] = camera_azimuth
    if camera_elevation is not None:
        camera_config['elevation'] = camera_elevation
    if camera_fov is not None:
        camera_config['fov'] = camera_fov

    # Apply CLI overrides to sun config
    if sun_azimuth is not None:
        sun_config['azimuth'] = sun_azimuth
    if sun_elevation is not None:
        sun_config['elevation'] = sun_elevation

    try:
        # Load and validate data with xarray
        data_dict = io.load_and_validate(filename)
        lw_data = data_dict['liquid_water_data']
        iw_data = data_dict['ice_water_data']
        x_coord = data_dict.get('x_coord')
        y_coord = data_dict.get('y_coord')
        z_coord = data_dict.get('z_coord')
        if x_coord is None or y_coord is None or z_coord is None:
            raise ValueError(
                "Missing x/y/z coordinate arrays in validated dataset; "
                "cannot render behold view."
            )

        lw_np = lw_data.values
        if 'time' in lw_data.dims:
            lw_np = lw_np[0]  # Remove time dimension if present

        nx, ny, nz = lw_np.shape
        if len(x_coord) < 2 or len(y_coord) < 2 or len(z_coord) < 2:
            raise ValueError(
                "x/y/z coordinates must each contain at least 2 points to determine grid spacing."
            )

        # Calculate grid spacing
        dx = float(x_coord[1] - x_coord[0])
        dy = float(y_coord[1] - y_coord[0])
        dz = float(z_coord[1] - z_coord[0])

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
            'min': {'resolution': (150, 100), 'spp': 1, 'rr_depth': 2, 'max_depth': 4},
            'low': {'resolution': (300, 200), 'spp': 32, 'rr_depth': 4, 'max_depth': 16},
            'medium': {'resolution': (600, 400), 'spp': 512, 'rr_depth': 16, 'max_depth': 64},
            'high': {'resolution': (1200, 800), 'spp': 4096, 'rr_depth': 32, 'max_depth': 128}
        }

        if quality == 'custom':
            # Use custom parameters (with sensible defaults)
            width, height = custom_size if custom_size else (600, 400)
            spp = custom_spp if custom_spp else 512
            # Override config max_depth/rr_depth if custom values provided
            if custom_max_depth is not None:
                rendering_config['max_depth'] = custom_max_depth
            if custom_rr_depth is not None:
                rendering_config['rr_depth'] = custom_rr_depth
        else:
            width, height = quality_map[quality]['resolution']
            spp = quality_map[quality]['spp']
            rendering_config['max_depth'] = quality_map[quality]['max_depth']
            rendering_config['rr_depth'] = quality_map[quality]['rr_depth']

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

        # Compute look direction vector from meteorological azimuth/elevation
        look_dir = direction_from_azimuth_elevation(camera_azimuth, camera_elevation)

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
            'sampler': {'type': 'independent', 'sample_count': spp},
            'seed': 0,
            'render_mode': render_mode,
        }

        # Add progress_interval if specified
        if progress_interval is not None:
            view_config['progress_interval'] = progress_interval

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
        choices=['min', 'low', 'medium', 'high', 'custom'],
        help=(
            "Render quality: min (150x100, spp=1, max_depth=4, rr_depth=2), "
            "low (300x200, spp=32, max_depth=16, rr_depth=4), "
            "medium (600x400, spp=512, max_depth=64, rr_depth=16, default), "
            "high (1200x800, spp=4096, max_depth=128, rr_depth=32), "
            "custom (use --spp, --size, --max-depth, --rr-depth)"
        )
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for saving renders (default: current directory)"
    )
    parser.add_argument(
        "--spp",
        type=int,
        help="Samples per pixel (for custom quality)"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help="Image size in pixels (for custom quality)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum ray depth (for custom quality)"
    )
    parser.add_argument(
        "--rr-depth",
        type=int,
        help="Russian roulette depth (for custom quality)"
    )
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        help="Camera position in relative coords (default: 0 0 -0.999). ±1.0 = domain edge"
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        help="Camera azimuth in degrees (default: 0). 0=North, 90=East, 180=South, 270=West"
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        help="Camera elevation in degrees (default: 35). Angle above horizon"
    )
    parser.add_argument(
        "--fov",
        type=float,
        help="Camera field of view in degrees (default: 100)"
    )
    parser.add_argument(
        "--sun-azimuth",
        type=float,
        help="Sun azimuth in degrees (default: 20). 0=North, 90=East, 180=South, 270=West"
    )
    parser.add_argument(
        "--sun-elevation",
        type=float,
        help="Sun elevation in degrees (default: 55). Angle above horizon"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        help="Print progress every N samples (default: 2 for rgb/mono, 16 for chromatic)"
    )

    args = parser.parse_args()
    main(args.filename, args.backend, args.quality, args.output,
         custom_spp=args.spp,
         custom_size=tuple(args.size) if args.size else None,
         custom_max_depth=args.max_depth,
         custom_rr_depth=args.rr_depth,
         camera_position=args.camera_position,
         camera_azimuth=args.camera_azimuth,
         camera_elevation=args.camera_elevation,
         camera_fov=args.fov,
         sun_azimuth=args.sun_azimuth,
         sun_elevation=args.sun_elevation,
         progress_interval=args.progress_interval)


if __name__ == "__main__":
    cli()
