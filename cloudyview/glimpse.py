#!/usr/bin/env python
"""
glimpse.py: Quick optical depth calculation for cloud fields.

Usage:
    python glimpse.py <filename.nc> [--output <path>]

This script provides a quick glimpse of your cloud data using:
1. Column optical depth calculation
2. Top-view visualization (matplotlib)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from textwrap import dedent

from . import io, optical_depth, basic_render, config
from .angles import direction_from_azimuth_elevation, azimuth_met_to_internal_deg
from .cli_utils import (
    CloudyViewHelpFormatter,
    DATA_SELECTION_HELP,
    add_dataset_selection_arguments,
    dataset_selection_kwargs,
)


def _unit_xy(direction_3d: np.ndarray, fallback_angle_rad: float) -> np.ndarray:
    """Project 3D direction to unit XY direction with azimuth fallback."""
    xy = direction_3d[:2]
    norm_xy = np.linalg.norm(xy)
    if norm_xy < 1e-10:
        return np.array([np.cos(fallback_angle_rad), np.sin(fallback_angle_rad)])
    return xy / norm_xy


def _build_camera_overlay(
    image_shape: tuple,
    camera_position: list,
    camera_azimuth: float,
    camera_elevation: float,
    camera_fov: float,
    render_aspect: float,
) -> dict:
    """
    Build camera marker/FOV overlay for top-down optical depth image.

    Returns dict for basic_render.plot_optical_depth(camera_overlay=...).
    """
    ny, nx = image_shape
    cam_x = ((camera_position[0] + 1.0) * 0.5) * (nx - 1)
    cam_y = ((camera_position[1] + 1.0) * 0.5) * (ny - 1)

    half_vfov_deg = 0.5 * float(camera_fov)
    includes_zenith = (90.0 - float(camera_elevation)) <= half_vfov_deg
    includes_nadir = (90.0 + float(camera_elevation)) <= half_vfov_deg

    # If straight up or down is in view, top-down FOV rays are ambiguous.
    if includes_zenith or includes_nadir:
        return {
            'camera_xy': (cam_x, cam_y),
            'circle_radius': nx / 10.0,
        }

    az_internal_rad = np.deg2rad(azimuth_met_to_internal_deg(camera_azimuth))
    half_vfov = np.deg2rad(camera_fov * 0.5)
    half_hfov = np.arctan(np.tan(half_vfov) * render_aspect)

    forward = direction_from_azimuth_elevation(camera_azimuth, camera_elevation)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        right = np.array([-np.sin(az_internal_rad), np.cos(az_internal_rad), 0.0])
    else:
        right /= right_norm

    left_dir = forward - np.tan(half_hfov) * right
    right_dir = forward + np.tan(half_hfov) * right
    left_dir /= np.linalg.norm(left_dir)
    right_dir /= np.linalg.norm(right_dir)

    left_xy = _unit_xy(left_dir, az_internal_rad - half_hfov)
    right_xy = _unit_xy(right_dir, az_internal_rad + half_hfov)

    ray_length = 1.5 * max(nx, ny)
    left_end = (cam_x + ray_length * left_xy[0], cam_y + ray_length * left_xy[1])
    right_end = (cam_x + ray_length * right_xy[0], cam_y + ray_length * right_xy[1])

    return {
        'camera_xy': (cam_x, cam_y),
        'fov_endpoints': [left_end, right_end],
    }


def main(
    filename: str,
    output: str = None,
    label_dirs: bool = False,
    label: bool = False,
    camera_position: list = None,
    camera_azimuth: float = None,
    camera_elevation: float = None,
    camera_fov: float = None,
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
) -> None:
    """
    Main function for glimpse.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for plots (default: current directory)
    label_dirs : bool, optional
        If True, label N/S/W/E sections of domain (default: False)
    label : bool, optional
        If True, draw camera marker and FOV overlay (default: False)
    camera_position : list, optional
        Camera position [x, y, z] in relative coords (±1.0 = domain edge)
    camera_azimuth : float, optional
        Camera azimuth in degrees (0=North, 90=East, 180=South, 270=West)
    camera_elevation : float, optional
        Camera elevation in degrees (angle above horizon)
    camera_fov : float, optional
        Camera field of view in degrees (vertical FOV)
    liquid_water_var, ice_water_var : str, optional
        Explicit variable-name overrides for water-content arrays
    dataset_group, liquid_water_group, ice_water_group, coords_group : str, optional
        NetCDF group overrides for variable/coordinate lookup
    x_coord_name, y_coord_name, z_coord_name : str, optional
        Explicit coordinate variable names
    x_dim, y_dim, z_dim : str, optional
        Explicit dimension names for x/y/z
    """
    print(f"CloudyView Glimpse: Loading {filename}")

    try:
        witness_config = config.get_witness_config()
        default_camera = witness_config['camera']
        render_aspect = (
            witness_config['rendering']['width'] / witness_config['rendering']['height']
        )

        cam_position = list(default_camera['position'])
        cam_azimuth = float(default_camera['azimuth'])
        cam_elevation = float(default_camera['elevation'])
        cam_fov = float(default_camera['fov'])

        if camera_position is not None:
            cam_position = list(camera_position)
        if camera_azimuth is not None:
            cam_azimuth = float(camera_azimuth)
        if camera_elevation is not None:
            cam_elevation = float(camera_elevation)
        if camera_fov is not None:
            cam_fov = float(camera_fov)

        # Get base filename without path and extension
        base_filename = Path(filename).stem

        # Load and validate data
        data_dict = io.load_and_validate(
            filename,
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
        lw_var = data_dict['liquid_water_var']
        lw_data = data_dict['liquid_water_data']
        iw_var = data_dict['ice_water_var']
        iw_data = data_dict['ice_water_data']

        print(f"✓ Loaded {lw_var} variable (liquid water)")

        if iw_data is not None:
            print(f"✓ Loaded {iw_var} variable (ice water)")
        else:
            print(f"⚠ No ice water variable found (only liquid water will be used)")

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Get z-coordinates (already standardized by load_and_validate)
        z_coord = data_dict['z_coord']

        # Convert to numpy
        lw_np = lw_data.values

        iw_np = None
        if iw_data is not None:
            iw_np = iw_data.values

        # Calculate column optical depth (2D) from liquid and ice water content
        # Uses empirical relationships: for liquid LWP = 0.6292 * tau * re,
        # for ice IWP = 0.350 * tau * re (consistent with plot_optical_depth.py)
        od_col = optical_depth.vertically_integrated_optical_depth(lw_np, z_coord, iwc=iw_np)

        if iw_np is not None:
            print(f"✓ Optical depth (liquid + ice) range: {od_col.min():.4f} - {od_col.max():.4f}")
        else:
            print(f"✓ Optical depth (liquid only) range: {od_col.min():.4f} - {od_col.max():.4f}")

        # Convert optical depth to visual albedo via conservative-scattering
        # two-stream reflectance: A = tau / (tau + 2/(1-g)), g = 0.85.
        # Unlike beam opacity 1-exp(-tau) (saturates by tau~4), this keeps
        # contrast between cirrus (tau~1-5) and deep cores (tau~100).
        two_stream_denom = np.float32(2.0 / (1.0 - 0.85))  # 13.3 for g=0.85
        tau32 = od_col.astype(np.float32)
        albedo = tau32 / (tau32 + two_stream_denom)
        print(f"✓ Visual albedo range: {albedo.min():.4f} - {albedo.max():.4f}")

        # Enforce map orientation compatibility with witness/behold:
        # east-right (+x) and north-up (+y).
        x_coord = data_dict['x_coord']
        y_coord = data_dict['y_coord']
        albedo_oriented = albedo
        if x_coord[1] < x_coord[0]:
            albedo_oriented = albedo_oriented[::-1, :]
        if y_coord[1] < y_coord[0]:
            albedo_oriented = albedo_oriented[:, ::-1]
        albedo_oriented = albedo_oriented.T  # plot expects [y, x]

        camera_overlay = None
        if label:
            camera_overlay = _build_camera_overlay(
                image_shape=albedo_oriented.shape,
                camera_position=cam_position,
                camera_azimuth=cam_azimuth,
                camera_elevation=cam_elevation,
                camera_fov=cam_fov,
                render_aspect=render_aspect,
            )

        # Plot visual albedo
        od_path = output_dir / f"cloudyview_glimpse_top_view_{base_filename}.png"
        basic_render.plot_optical_depth(
            albedo_oriented,
            output_path=str(od_path),
            label_dirs=label_dirs,
            camera_overlay=camera_overlay,
            print_save=False,
        )
        saved_path = str(od_path)
        if not od_path.is_absolute() and not saved_path.startswith("./"):
            saved_path = f"./{saved_path}"
        print(f"✓ Saved to {saved_path}")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cli():
    """Command-line interface for glimpse.py"""
    parser = argparse.ArgumentParser(
        prog="glimpse",
        description="Quick top-down cloud overview from column optical depth.",
        formatter_class=CloudyViewHelpFormatter,
        epilog=dedent(
            f"""
            What `glimpse` does:
              1. Loads one 3D single-timestep cloud field from a NetCDF file.
              2. Finds a liquid water mixing ratio/content array and, if present, an
                 ice water array.
              3. Converts supported units (`g/kg`, `g/g`, or `kg/kg`) to `g/kg`.
              4. Computes vertically integrated optical depth and writes a top-view PNG.

            Input requirements:
              - The selected liquid water array is required.
              - The ice water array is optional.
              - The selected cloud array must be 3D after any single time dimension is removed.
              - Physical x/y/z coordinate arrays are required so vertical spacing is known.

            Output:
              A PNG named `cloudyview_glimpse_top_view_<input-stem>.png` written to the
              output directory.

            Camera overlay:
              `--label` draws the current `witness` camera position and field-of-view on
              the top view. Camera coordinates are relative to the model domain:
              - x: east-west, where -1 is west edge and +1 is east edge
              - y: south-north, where -1 is south edge and +1 is north edge
              - z: relative height, where -1 is bottom and +1 is top
              Azimuth uses meteorological bearings: 0 north, 90 east, 180 south, 270 west.

            {DATA_SELECTION_HELP}

            Examples:
              glimpse cloud.nc
              glimpse cloud.nc --output renders --label-dirs
              glimpse cloud.nc --label --camera-position 0 -0.8 -0.95 --camera-azimuth 0 --camera-elevation 35 --fov 100
              glimpse cloud.nc --group /physics/clouds --liquid-water-var qc_cloud --ice-water-var qi_cloud
              glimpse custom.nc --liquid-water-group /state/liquid --coords-group /grid --x-dim ni --y-dim nj --z-dim nk --x-coord xh --y-coord yh --z-coord zh
            """
        ),
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable; qi/QI/IWC optional; must be 3D single-timestep)"
    )
    parser.add_argument(
        "--output", "-o", default=".",
        help="Output directory for saving plots (default: current directory)"
    )
    parser.add_argument(
        "--label-dirs", action="store_true", default=False,
        help="Label N/S/W/E sections of the domain (default: False)"
    )
    parser.add_argument(
        "--label", action="store_true", default=False,
        help="Overlay camera marker and field-of-view on top view (default: False)"
    )
    parser.add_argument(
        "--camera-position", type=float, nargs=3, metavar=('X', 'Y', 'Z'),
        help="Camera position in relative coords (default: 0 0 -0.999). ±1.0 = domain edge"
    )
    parser.add_argument(
        "--camera-azimuth", type=float,
        help="Camera azimuth in degrees (default: 0). 0=North, 90=East, 180=South, 270=West"
    )
    parser.add_argument(
        "--camera-elevation", type=float,
        help="Camera elevation in degrees (default: 35). Angle above horizon"
    )
    parser.add_argument(
        "--fov", type=float,
        help="Camera field of view in degrees (default: 100)"
    )
    add_dataset_selection_arguments(parser)

    args = parser.parse_args()
    main(
        args.filename,
        args.output,
        args.label_dirs,
        label=args.label,
        camera_position=args.camera_position,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        camera_fov=args.fov,
        **dataset_selection_kwargs(args),
    )


if __name__ == "__main__":
    cli()
