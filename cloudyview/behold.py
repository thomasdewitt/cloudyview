#!/usr/bin/env python
"""
behold.py: Photorealistic cloud rendering with Mitsuba 3 path tracing.

Usage:
    behold <filename.nc> --cpu|--gpu [quality] [options]

    quality: min, low, medium (default), high, custom
    For custom quality: --spp N --size W H --max-depth N --rr-depth N

Renders a 3D cloud field using Monte Carlo volumetric path tracing with
physically-based Preetham sunsky illumination and Mie scattering phase
functions. Supports arbitrary non-square domains and non-uniform vertical
grid spacing.

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy as np
from textwrap import dedent

from . import io, optical_depth, config
from .camera import Camera
from .cloudfield import CloudField, load as _load_field
from .domain import compute_domain_geometry
from .angles import direction_from_azimuth_elevation
from .cli_utils import (
    CloudyViewHelpFormatter,
    DATA_SELECTION_HELP,
    add_dataset_selection_arguments,
    dataset_selection_kwargs,
)


# Quality presets: resolution, samples per pixel, path-depth budgets.
QUALITY_MAP = {
    'min': {'resolution': (150, 100), 'spp': 1, 'rr_depth': 2, 'max_depth': 4},
    'low': {'resolution': (300, 200), 'spp': 32, 'rr_depth': 4, 'max_depth': 16},
    'medium': {'resolution': (600, 400), 'spp': 512, 'rr_depth': 16, 'max_depth': 64},
    'high': {'resolution': (1200, 800), 'spp': 2048, 'rr_depth': 64, 'max_depth': 96},
}


def _prepare_extinction(field: CloudField, verbose: bool = False):
    """Extinction field + ice fraction + geometry from a CloudField.

    Returns (sigma_ext, ice_fraction, geom, dx, dy, dz).
    """
    x_coord, y_coord, z_coord = field.x, field.y, field.z
    lw_np = field.lwc
    nx, ny, nz = lw_np.shape

    # Domain geometry (shared with witness)
    geom = compute_domain_geometry(x_coord, y_coord, z_coord, nx, ny, nz)
    dx, dy = geom.dx, geom.dy
    dz = float(z_coord[1] - z_coord[0])  # first spacing, for interface compat

    if verbose:
        print(f"  Grid: {nx} x {ny} x {nz}, spacing: {dx:.1f} x {dy:.1f} m")
        print(f"  Domain: {geom.width_x:.0f} x {geom.width_y:.0f} x {geom.height_z:.0f} m, "
              f"aspect ratio: {geom.ar_x:.2f} x {geom.ar_y:.2f}")

    # Process ice water content if present
    iw_np = field.iwc
    ice_fraction = None

    if iw_np is not None:
        # Check if there's actually ice in the volume
        if np.max(iw_np) > 1e-6:
            if verbose:
                print(f"  Ice water content detected (max: {np.max(iw_np):.6f} g/kg)")

            # Compute ice fraction (0 = liquid, 1 = ice)
            # Avoid division by zero
            total_water = lw_np + iw_np
            ice_fraction = np.divide(iw_np, total_water,
                                    out=np.zeros_like(iw_np),
                                    where=total_water > 1e-10)
        else:
            if verbose:
                print("  Ice water content negligible, using liquid-only rendering")
            iw_np = None
    else:
        if verbose:
            print("  No ice water content in dataset")

    if iw_np is None and verbose:
        print("  No ice water content detected; using liquid-only extinction.")

    # Compute extinction coefficient (liquid + ice if present)
    sigma_ext = optical_depth.compute_extinction_field(lw_np, z_coord, re=10.0,
                                                      iwc=iw_np, re_ice=30.0)

    return sigma_ext, ice_fraction, geom, dx, dy, dz


def _build_view_config(geom, camera: Camera, sun_azimuth: float, sun_elevation: float,
                       quality: str, custom_spp: Optional[int],
                       custom_size: Optional[tuple], custom_max_depth: Optional[int],
                       custom_rr_depth: Optional[int],
                       progress_interval: Optional[int] = None,
                       verbose: bool = False) -> dict:
    """Build the Mitsuba view/scene configuration dict.

    Requires the Mitsuba variant to be set already (the camera transform
    is a Mitsuba type).
    """
    from . import radiative_transfer

    rendering_config = config.get_behold_config()['rendering']
    ar_x, ar_y = geom.ar_x, geom.ar_y

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
        width, height = QUALITY_MAP[quality]['resolution']
        spp = QUALITY_MAP[quality]['spp']
        rendering_config['max_depth'] = QUALITY_MAP[quality]['max_depth']
        rendering_config['rr_depth'] = QUALITY_MAP[quality]['rr_depth']

    # Convert relative camera position to absolute
    # Relative coords: ±1.0 = domain edge
    rel_pos = camera.position
    camera_origin = [
        rel_pos[0] * ar_x,  # x in world space (±ar_x)
        rel_pos[1] * ar_y,  # y in world space (±ar_y)
        rel_pos[2]          # z in world space (±1)
    ]

    # Compute look direction vector from meteorological azimuth/elevation
    look_dir = direction_from_azimuth_elevation(camera.azimuth, camera.elevation)

    # Target point is origin + look direction
    camera_target = camera_origin + look_dir

    if verbose:
        print(f"  Camera offset: x={camera_origin[0]:.1f}, y={camera_origin[1]:.1f}, z={camera_origin[2]:.1f}")
        print(f"  Camera azimuth: {camera.azimuth:.1f}°, elevation: {camera.elevation:.1f}°")
        print(f"  Sun azimuth: {sun_azimuth:.1f}°, elevation: {sun_elevation:.1f}°")
        print(f"  Field of view: {camera.fov:.1f}°")
        print(f"  Render quality: {quality} ({width}x{height}, spp={spp})")

    view_config = {
        'name': 'Ground-Looking-Up (Progressive Rendering)',
        'width': width,
        'height': height,
        'fov': camera.fov,
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
        'ar_x': ar_x,
        'ar_y': ar_y,
        'height_z': geom.height_z,
    }

    # Add progress_interval if specified
    if progress_interval is not None:
        view_config['progress_interval'] = progress_interval

    return view_config


def behold(
    field: CloudField,
    camera: Optional[Camera] = None,
    quality: str = 'medium',
    *,
    gpu: bool = False,
    spp: Optional[int] = None,
    size: Optional[Tuple[int, int]] = None,
    max_depth: Optional[int] = None,
    rr_depth: Optional[int] = None,
    sun_azimuth: Optional[float] = None,
    sun_elevation: Optional[float] = None,
    seed: int = 0,
    progress_callback: Optional[Callable[[dict], None]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Render a cloud field with the Mitsuba 3 path tracer.

    Parameters
    ----------
    field : CloudField
        Loaded cloud field (see :func:`cloudyview.load`).
    camera : Camera, optional
        Viewpoint; defaults to the standard behold camera.
    quality : str
        'min', 'low', 'medium', 'high', or 'custom'. With 'custom', the
        `spp`, `size`, `max_depth`, and `rr_depth` overrides apply
        (same semantics as the CLI).
    gpu : bool
        Use the CUDA Mitsuba backend instead of LLVM. Raises if the
        variant is unavailable — no fallback.
    sun_azimuth, sun_elevation : float, optional
        Sun direction in degrees (met bearing / above horizon);
        defaults from config (20 / 55).
    seed : int
        Monte Carlo seed.
    progress_callback : callable, optional
        Called as progressive chunks complete with percent, SPP, elapsed,
        and ETA information from ``render_with_progress``.
    verbose : bool
        Print scene diagnostics and render progress (the CLI uses this).

    Returns
    -------
    ndarray (height, width, 3)
        Tone-mapped RGB image in [0, 1].

    Raises
    ------
    ImportError
        If Mitsuba 3 is not installed.
    """
    import mitsuba as mi
    from . import radiative_transfer

    behold_config = config.get_behold_config()
    sun_config = behold_config['sun']
    rendering_config = behold_config['rendering']

    if camera is None:
        camera = Camera()
    if sun_azimuth is None:
        sun_azimuth = sun_config['azimuth']
    if sun_elevation is None:
        sun_elevation = sun_config['elevation']

    variant = f"{'cuda' if gpu else 'llvm'}_ad_rgb"
    mi.set_variant(variant)

    sigma_ext, ice_fraction, geom, dx, dy, dz = _prepare_extinction(
        field, verbose=verbose)

    view_config = _build_view_config(
        geom, camera, sun_azimuth, sun_elevation,
        quality, spp, size, max_depth, rr_depth,
        verbose=verbose,
    )

    scene = radiative_transfer.create_mitsuba_scene(
        sigma_ext, dx, dy, dz,
        view_config,
        spp=view_config['spp'],
        ice_fraction=ice_fraction,
        ar_x=view_config['ar_x'],
        ar_y=view_config['ar_y'],
        height_z=view_config['height_z'],
        verbose=verbose,
    )

    image = radiative_transfer.render_with_progress(
        scene,
        spp_total=view_config['spp'],
        step_spp=view_config.get('progress_interval', 2),
        seed=seed,
        progress_callback=progress_callback,
        verbose=verbose,
    )

    img_np = np.array(image)
    return radiative_transfer.tone_map(
        img_np, exposure=view_config.get('exposure', 1.0))


def main(filename: str, backend: str, quality: str = 'medium', output: str = None,
         custom_spp: int = None, custom_size: tuple = None,
         custom_max_depth: int = None, custom_rr_depth: int = None,
         camera_position: list = None, camera_azimuth: float = None,
         camera_elevation: float = None, camera_fov: float = None,
         sun_azimuth: float = None, sun_elevation: float = None,
         progress_interval: int = None,
         liquid_water_var: str = None,
         ice_water_var: str = None,
         dataset_group: str = None,
         liquid_water_group: str = None,
         ice_water_group: str = None,
         coords_group: str = None,
         x_coord_name: str = None,
         y_coord_name: str = None,
         z_coord_name: str = None,
         x_dim: str = None,
         y_dim: str = None,
         z_dim: str = None,
         ice: str = None) -> None:
    """CLI wrapper: load, build the scene, progressively render and save PNGs.

    Composes the same helpers as the library :func:`behold`; the file
    writes (final PNG + spp checkpoints) and printed progress live here.

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    backend : str
        Mitsuba backend: 'llvm' or 'cuda'
    ice : str, optional
        Separate NetCDF file with the ice variable (SAM LPT split-file style)
    quality : str
        Render quality: 'min' (150x100, spp=1, max_depth=4, rr_depth=2),
        'low' (300x200, spp=32, max_depth=16, rr_depth=4),
        'medium' (600x400, spp=512, max_depth=64, rr_depth=16, default),
        'high' (1200x800, spp=2048, max_depth=96, rr_depth=64),
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
    liquid_water_var, ice_water_var : str, optional
        Explicit variable-name overrides for water-content arrays
    dataset_group, liquid_water_group, ice_water_group, coords_group : str, optional
        NetCDF group overrides for variable/coordinate lookup
    x_coord_name, y_coord_name, z_coord_name : str, optional
        Explicit coordinate variable names
    x_dim, y_dim, z_dim : str, optional
        Explicit dimension names for x/y/z
    """
    print(f"CloudyView Behold: Loading {filename}")
    start_time = time.perf_counter()

    # Load configuration
    behold_config = config.get_behold_config()
    camera_config = behold_config['camera']
    sun_config = behold_config['sun']

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
        import mitsuba as mi
        from . import radiative_transfer

        # Load and validate data with xarray
        field = _load_field(
            filename,
            ice=ice,
            liquid_water_var=liquid_water_var,
            ice_water_var=ice_water_var,
            dataset_group=dataset_group,
            liquid_water_group=liquid_water_group,
            ice_water_group=ice_water_group,
            coords_group=coords_group,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            z_coord_name=z_coord_name,
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
        )

        camera = Camera(
            position=camera_config['position'],
            azimuth=camera_config['azimuth'],
            elevation=camera_config['elevation'],
            fov=camera_config['fov'],
        )

        sigma_ext, ice_fraction, geom, dx, dy, dz = _prepare_extinction(
            field, verbose=True)

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Set Mitsuba backend (AD variant required by most Mitsuba builds)
        variant = f'{backend}_ad_rgb'
        mi.set_variant(variant)
        print(f"  Using Mitsuba variant: {variant}")

        view_config = _build_view_config(
            geom, camera,
            sun_config['azimuth'], sun_config['elevation'],
            quality, custom_spp, custom_size, custom_max_depth, custom_rr_depth,
            progress_interval=progress_interval,
            verbose=True,
        )

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
        prog="behold",
        description="Photorealistic cloud rendering with Mitsuba volumetric path tracing.",
        formatter_class=CloudyViewHelpFormatter,
        epilog=dedent(
            f"""
            What `behold` does:
              1. Loads a 3D cloud field from NetCDF.
              2. Converts liquid water and optional ice water to extinction coefficients.
              3. Builds a Mitsuba 3 scene with physically-based sky illumination.
              4. Progressively path-traces a ground-looking-up RGB render and saves a PNG.

            Positional arguments:
              filename  Path to the NetCDF file.
              quality   Preset render budget. `custom` enables the `--spp`, `--size`,
                        `--max-depth`, and `--rr-depth` overrides.

            Backend (required, pick one):
              --cpu     Use LLVM CPU backend.
              --gpu     Use CUDA GPU backend.

            Quality presets:
              - min:    150x100, 1 spp,   max_depth=4,   rr_depth=2
              - low:    300x200, 32 spp,  max_depth=16,  rr_depth=4
              - medium: 600x400, 512 spp, max_depth=64,  rr_depth=16
              - high:   1200x800, 2048 spp, max_depth=96, rr_depth=64

            Camera and sun conventions:
              - Coordinates are meteorological: +x east, +y north, +z up.
              - Camera position uses relative coordinates where +/-1 reaches the domain edge
                in x and y, and z spans the domain height.
              - Azimuth is a meteorological bearing: 0 north, 90 east, 180 south, 270 west.
              - Elevation is degrees above the horizon.

            Output:
              A PNG named `behold_ground_view_max_depth=<N>_rr_depth=<M>.png` written
              to the output directory. Progressive checkpoint images may also be written
              by the renderer while sampling accumulates.

            Dependencies:
              `behold --help` works without Mitsuba installed. Actual rendering requires
              `mitsuba` and `drjit`.

            {DATA_SELECTION_HELP}

            Examples:
              behold cloud.nc --cpu
              behold cloud.nc high --gpu --output renders
              behold cloud_QC.nc low --gpu --ice cloud_QI.nc
              behold cloud.nc custom --gpu --size 1024 768 --spp 256 --max-depth 64 --rr-depth 32
              behold cloud.nc --cpu --camera-position 0 -0.99 -0.99 --camera-azimuth 0 --camera-elevation 35 --sun-azimuth 20 --sun-elevation 55
              behold cloud.nc --help
              behold grouped.nc --gpu --group /physics/clouds --liquid-water-var qc_cloud --ice-water-var qi_cloud
              behold custom.nc --gpu --liquid-water-group /state/liquid --ice-water-group /state/ice --coords-group /grid --x-dim ni --y-dim nj --z-dim nk --x-coord xh --y-coord yh --z-coord zh
            """
        ),
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument(
        "--cpu", action="store_true",
        help="Use LLVM CPU backend"
    )
    backend_group.add_argument(
        "--gpu", action="store_true",
        help="Use CUDA GPU backend"
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
            "high (1200x800, spp=2048, max_depth=96, rr_depth=64), "
            "custom (use --spp, --size, --max-depth, --rr-depth)"
        )
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for saving renders (default: current directory)"
    )
    parser.add_argument(
        "--ice",
        help="separate NetCDF file with the ice variable (SAM LPT split-file style)"
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
        help="Print progress every N samples (default: 2)"
    )
    add_dataset_selection_arguments(parser)

    args = parser.parse_args()
    backend = 'llvm' if args.cpu else 'cuda'
    main(args.filename, backend, args.quality, args.output,
         ice=args.ice,
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
         progress_interval=args.progress_interval,
         **dataset_selection_kwargs(args))


if __name__ == "__main__":
    cli()
