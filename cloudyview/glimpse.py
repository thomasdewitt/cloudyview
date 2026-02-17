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

from . import io, optical_depth, basic_render


def main(filename: str, output: str = None, label_dirs: bool = False) -> None:
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
        iw_var = data_dict['ice_water_var']
        iw_data = data_dict['ice_water_data']

        print(f"✓ Loaded {lw_var} variable (liquid water)")
        print(f"  Shape: {lw_data.shape}")
        print(f"  Range: {lw_data.min().values:.4f} - {lw_data.max().values:.4f} g/kg")

        if iw_data is not None:
            print(f"✓ Loaded {iw_var} variable (ice water)")
            print(f"  Shape: {iw_data.shape}")
            print(f"  Range: {iw_data.min().values:.4f} - {iw_data.max().values:.4f} g/kg")
        else:
            print(f"⚠ No ice water variable found (only liquid water will be used)")

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
        if iw_data is not None:
            iw_np = iw_data.values

        # Calculate column optical depth (2D) from liquid and ice water content
        # Uses empirical relationships: for liquid LWP = 0.6292 * tau * re,
        # for ice IWP = 0.350 * tau * re (consistent with plot_optical_depth.py)
        od_col = optical_depth.optical_depth_from_lwc(lw_np, z_coord, iwc=iw_np)

        if iw_np is not None:
            print(f"✓ Optical depth (liquid + ice) range: {od_col.min():.4f} - {od_col.max():.4f}")
        else:
            print(f"✓ Optical depth (liquid only) range: {od_col.min():.4f} - {od_col.max():.4f}")

        # Convert optical depth to opacity (1 - exp(-tau))
        opacity = 1.0 - np.exp(-od_col)
        print(f"✓ Opacity range: {opacity.min():.4f} - {opacity.max():.4f}")

        # Plot opacity
        print("\nRendering top view...")
        od_path = output_dir / f"cloudyview_glimpse_top_view_{base_filename}.png"
        basic_render.plot_optical_depth(opacity, output_path=str(od_path), label_dirs=label_dirs)

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
        description="Quick optical depth calculation and top-view visualization"
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

    args = parser.parse_args()
    main(args.filename, args.output, args.label_dirs)


if __name__ == "__main__":
    cli()
