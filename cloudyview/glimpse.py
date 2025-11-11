#!/usr/bin/env python
"""
glimpse.py: Quick visualization of cloud fields using optical depth and matplotlib 3D.

Usage:
    python glimpse.py <filename.nc> [--output <path>]

This script provides a quick glimpse of your cloud data using:
1. Optical depth calculation
2. 3D isosurface plot (matplotlib)
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from . import io, optical_depth, basic_render


def main(filename: str, output: str = None) -> None:
    """
    Main function for glimpse.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for plots (default: current directory)
    """
    print(f"CloudyView Glimpse: Loading {filename}")

    try:
        # Get base filename without path and extension
        base_filename = Path(filename).stem

        # Load and validate data
        data_dict = io.load_and_validate(filename)
        ds = data_dict['dataset']
        lw_var = data_dict['liquid_water_var']
        lw_data = data_dict['liquid_water_data']

        print(f"✓ Loaded {lw_var} variable")
        print(f"  Shape: {lw_data.shape}")
        print(f"  Range: {lw_data.min().values:.4f} - {lw_data.max().values:.4f} g/kg")

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Calculate column optical depth (2D)
        print("\nCalculating column optical depth...")

        # Get z-coordinates (already standardized by load_and_validate)
        z_coord = data_dict['z_coord']
        if z_coord is None:
            # Fallback to indices if no coordinates available
            z_coord = np.arange(lw_data.shape[-1])

        # Convert to numpy
        lw_np = lw_data.values

        iw_np = None
        if data_dict['ice_water_data'] is not None:
            iw_np = data_dict['ice_water_data'].values

        # Calculate column optical depth (2D)
        od_col = optical_depth.optical_depth_from_lwc(lw_np, z_coord, iwc=iw_np)
        print(f"✓ Optical depth range: {od_col.min():.4f} - {od_col.max():.4f}")

        # Convert optical depth to opacity (1 - exp(-tau))
        opacity = 1.0 - np.exp(-od_col)
        print(f"✓ Opacity range: {opacity.min():.4f} - {opacity.max():.4f}")

        # Plot opacity
        print("\nRendering top view...")
        od_path = output_dir / f"cloudyview_glimpse_top_view_{base_filename}.png"
        basic_render.plot_optical_depth(opacity, output_path=str(od_path))

        # Calculate 3D optical depth field for coloring
        print("Calculating 3D optical depth...")
        # Create 3D optical depth field from water content
        od_3d = optical_depth.compute_extinction_field(lw_np, z_coord)

        # Convert to opacity field (opacity of everything above each point)
        opacity_3d = optical_depth.opacity_field_3d(od_3d)
        print(f"✓ 3D opacity range: {opacity_3d.min():.4f} - {opacity_3d.max():.4f}")

        # Plot isosurface with opacity coloring
        print("Rendering 3D isosurface...")
        iso_path = output_dir / f"cloudyview_glimpse_3D_{base_filename}.png"
        fig, ax = basic_render.plot_isosurface(
            lw_data,
            threshold=0.01,
            output_path=str(iso_path),
            color_by_opacity=True,
            opacity_field=opacity_3d
        )

        print("\n✓ Glimpse complete!")
        print(f"  Saved to {output_dir}")

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
        description="Quick 3D cloud visualization with optical depth and matplotlib"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "--output", "-o", default=".",
        help="Output directory for saving plots (default: current directory)"
    )

    args = parser.parse_args()
    main(args.filename, args.output)


if __name__ == "__main__":
    cli()
