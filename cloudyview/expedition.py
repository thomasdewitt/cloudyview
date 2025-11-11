#!/usr/bin/env python
"""
expedition.py: Cloud field visualization with optical depth and simple 3D radiative transfer.

Usage:
    python expedition.py <filename.nc> [--output <path>] [--sza <angle>]

This script provides a more detailed view of your cloud data using:
1. Optical depth calculation
2. Simple 3D radiative transfer (two-stream approximation)
3. 3D isosurface visualization (matplotlib)
4. Radiative transfer field plots (reflectance, transmission)
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from . import io, optical_depth, radiative_transfer, basic_render


def main(filename: str, output: str = None, sza: float = 0.0) -> None:
    """
    Main function for expedition.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for plots
    sza : float
        Solar zenith angle in degrees (default: 0)
    """
    print(f"CloudyView Expedition: Loading {filename}")

    try:
        # Load and validate data
        data_dict = io.load_and_validate(filename)
        ds = data_dict['dataset']
        lw_var = data_dict['liquid_water_var']
        lw_data = data_dict['liquid_water_data']
        iw_data = data_dict['ice_water_data']

        print(f"✓ Loaded {lw_var} variable")
        print(f"  Shape: {lw_data.shape}")
        print(f"  Range: {lw_data.min().values:.4f} - {lw_data.max().values:.4f} g/kg")

        # Calculate optical depth
        print("\nCalculating optical depth...")
        od = optical_depth.calculate_optical_depth(lw_data, ice_water=iw_data)
        print(f"✓ Optical depth range: {od.min().values:.4f} - {od.max().values:.4f}")

        # Simple 3D radiative transfer
        print(f"\nPerforming 3D radiative transfer (SZA={sza}°)...")
        rt_results = radiative_transfer.simple_3d_radiative_transfer(
            od, solar_zenith_angle=sza
        )
        print(f"✓ Reflectance range: {rt_results['reflectance'].min().values:.4f} - {rt_results['reflectance'].max().values:.4f}")
        print(f"✓ Transmission range: {rt_results['transmission'].min().values:.4f} - {rt_results['transmission'].max().values:.4f}")

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

        # Plot optical depth slices
        print("Plotting optical depth slices...")
        od_slice_path = output_dir / "optical_depth_slices.png" if output else None
        fig_od, _ = basic_render.plot_slices(
            od,
            title="Optical Depth Slices",
            output_path=str(od_slice_path) if od_slice_path else None
        )

        # Plot reflectance
        print("Plotting reflectance field...")
        ref_slice_path = output_dir / "reflectance_slices.png" if output else None
        fig_ref, _ = basic_render.plot_slices(
            rt_results['reflectance'],
            title="Reflectance Slices",
            output_path=str(ref_slice_path) if ref_slice_path else None
        )

        # Plot transmission
        print("Plotting transmission field...")
        trans_slice_path = output_dir / "transmission_slices.png" if output else None
        fig_trans, _ = basic_render.plot_slices(
            rt_results['transmission'],
            title="Transmission Slices",
            output_path=str(trans_slice_path) if trans_slice_path else None
        )

        print("\n✓ Expedition complete!")

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
    """Command-line interface for expedition.py"""
    parser = argparse.ArgumentParser(
        description="Cloud visualization with optical depth and simple 3D radiative transfer"
    )
    parser.add_argument(
        "filename",
        help="NetCDF file with cloud data (must contain qc/ql/LWC variable and be 3D single-timestep)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for saving plots (if not set, plots are displayed)"
    )
    parser.add_argument(
        "--sza", type=float, default=0.0,
        help="Solar zenith angle in degrees (default: 0)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.sza)


if __name__ == "__main__":
    cli()
