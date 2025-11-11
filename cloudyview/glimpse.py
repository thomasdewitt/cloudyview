#!/usr/bin/env python
"""
glimpse.py: Quick visualization of cloud fields using optical depth and matplotlib.

Usage:
    python glimpse.py <filename.nc> [--output <path>]

This script provides a quick glimpse of your cloud data using:
1. Optical depth calculation
2. 3D isosurface plot (matplotlib)
3. 2D slice views
"""

import argparse
import sys
from pathlib import Path

from . import io, optical_depth, basic_render


def main(filename: str, output: str = None) -> None:
    """
    Main function for glimpse.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for plots
    """
    print(f"CloudyView Glimpse: Loading {filename}")

    try:
        # Load and validate data
        data_dict = io.load_and_validate(filename)
        ds = data_dict['dataset']
        lw_var = data_dict['liquid_water_var']
        lw_data = data_dict['liquid_water_data']

        print(f"✓ Loaded {lw_var} variable")
        print(f"  Shape: {lw_data.shape}")
        print(f"  Range: {lw_data.min().values:.4f} - {lw_data.max().values:.4f} g/kg")

        # Calculate optical depth
        print("\nCalculating optical depth...")
        od = optical_depth.calculate_optical_depth(
            lw_data,
            ice_water=data_dict['ice_water_data']
        )
        print(f"✓ Optical depth range: {od.min().values:.4f} - {od.max().values:.4f}")

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Plot isosurface
        print("\nPlotting 3D isosurface...")
        iso_path = output_dir / "isosurface.png" if output else None
        fig, ax = basic_render.plot_isosurface(
            lw_data,
            threshold=0.01,
            title=f"Cloud Field Isosurface (qn >= 0.01 g/kg)",
            output_path=str(iso_path) if iso_path else None
        )
        if not output:
            print("(View in matplotlib window)")

        # Plot slices
        print("Plotting 2D slices...")
        slice_path = output_dir / "slices.png" if output else None
        fig_slices, axes = basic_render.plot_slices(
            lw_data,
            title=f"{lw_var} Field Slices",
            output_path=str(slice_path) if slice_path else None
        )
        if not output:
            print("(View in matplotlib window)")

        print("\n✓ Glimpse complete!")

        # Show plots if not saving
        if not output:
            import matplotlib.pyplot as plt
            plt.show()

    except FileNotFoundError as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
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
        "--output", "-o",
        help="Output directory for saving plots (if not set, plots are displayed)"
    )

    args = parser.parse_args()
    main(args.filename, args.output)


if __name__ == "__main__":
    cli()
