#!/usr/bin/env python
"""
odyssey.py: Comprehensive cloud field visualization with full 3D radiative transfer.

Usage:
    python odyssey.py <filename.nc> [--output <path>] [--sza <angle>] [--wavelength <wl>]

This script provides a comprehensive analysis of your cloud data using:
1. Optical depth calculation
2. Advanced 3D radiative transfer (with multiple scattering)
3. Multiple viewing angles for 3D isosurfaces
4. Full radiative transfer diagnostics
5. Heating rates and energy budgets
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from . import io, optical_depth, radiative_transfer, basic_render


def main(
    filename: str,
    output: str = None,
    sza: float = 0.0,
    wavelength: float = 0.55
) -> None:
    """
    Main function for odyssey.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for plots
    sza : float
        Solar zenith angle in degrees (default: 0)
    wavelength : float
        Wavelength in microns (default: 0.55 for visible)
    """
    print(f"CloudyView Odyssey: Loading {filename}")

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

        if iw_data is not None:
            print(f"✓ Also loaded ice water variable")
            print(f"  Range: {iw_data.min().values:.4f} - {iw_data.max().values:.4f} g/kg")

        # Calculate optical depth
        print("\nCalculating optical depth...")
        od = optical_depth.calculate_optical_depth(lw_data, ice_water=iw_data)
        print(f"✓ Optical depth range: {od.min().values:.4f} - {od.max().values:.4f}")

        # Column optical depth
        col_od = optical_depth.column_optical_depth(od)
        print(f"✓ Column optical depth range: {col_od.min().values:.4f} - {col_od.max().values:.4f}")

        # Advanced 3D radiative transfer
        print(f"\nPerforming advanced 3D radiative transfer...")
        print(f"  Solar zenith angle: {sza}°")
        print(f"  Wavelength: {wavelength} μm")
        rt_results = radiative_transfer.advanced_3d_radiative_transfer(
            od,
            lw_data,
            ice_water=iw_data,
            solar_zenith_angle=sza,
            wavelength=wavelength
        )
        print(f"✓ Reflectance range: {rt_results['reflectance'].min().values:.4f} - {rt_results['reflectance'].max().values:.4f}")
        print(f"✓ Transmission range: {rt_results['transmission'].min().values:.4f} - {rt_results['transmission'].max().values:.4f}")

        # Create output directory if needed
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        # Plot cloud field with multiple views
        print("\nGenerating visualizations...")

        # Main isosurface
        iso_path = output_dir / "isosurface.png" if output else None
        fig, ax = basic_render.plot_isosurface(
            lw_data,
            threshold=0.01,
            title=f"Cloud Field Isosurface (qn >= 0.01 g/kg)",
            output_path=str(iso_path) if iso_path else None
        )

        # Cloud field slices
        print("  → Cloud field slices")
        lw_slice_path = output_dir / "cloud_slices.png" if output else None
        fig_cloud, _ = basic_render.plot_slices(
            lw_data,
            title=f"{lw_var} Field Slices",
            output_path=str(lw_slice_path) if lw_slice_path else None
        )

        # Optical depth
        print("  → Optical depth")
        od_slice_path = output_dir / "optical_depth_slices.png" if output else None
        fig_od, _ = basic_render.plot_slices(
            od,
            title="Optical Depth Slices",
            output_path=str(od_slice_path) if od_slice_path else None
        )

        # Reflectance
        print("  → Reflectance")
        ref_slice_path = output_dir / "reflectance_slices.png" if output else None
        fig_ref, _ = basic_render.plot_slices(
            rt_results['reflectance'],
            title="Reflectance Slices",
            output_path=str(ref_slice_path) if ref_slice_path else None
        )

        # Transmission
        print("  → Transmission")
        trans_slice_path = output_dir / "transmission_slices.png" if output else None
        fig_trans, _ = basic_render.plot_slices(
            rt_results['transmission'],
            title="Transmission Slices",
            output_path=str(trans_slice_path) if trans_slice_path else None
        )

        # Heating rates
        if 'heating_rate' in rt_results:
            print("  → Heating rates")
            hr_slice_path = output_dir / "heating_rate_slices.png" if output else None
            fig_hr, _ = basic_render.plot_slices(
                rt_results['heating_rate'],
                title="Heating Rate Slices",
                output_path=str(hr_slice_path) if hr_slice_path else None
            )

        # Summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"\nCloud Water Content ({lw_var}):")
        print(f"  Min: {lw_data.min().values:.4f} g/kg")
        print(f"  Max: {lw_data.max().values:.4f} g/kg")
        print(f"  Mean: {lw_data.mean().values:.4f} g/kg")

        print(f"\nOptical Depth:")
        print(f"  Min: {od.min().values:.4f}")
        print(f"  Max: {od.max().values:.4f}")
        print(f"  Mean: {od.mean().values:.4f}")
        print(f"  Column range: {col_od.min().values:.4f} - {col_od.max().values:.4f}")

        print(f"\nRadiative Transfer (SZA={sza}°, λ={wavelength}μm):")
        print(f"  Reflectance: {rt_results['reflectance'].min().values:.4f} - {rt_results['reflectance'].max().values:.4f}")
        print(f"  Transmission: {rt_results['transmission'].min().values:.4f} - {rt_results['transmission'].max().values:.4f}")

        if output:
            print(f"\n✓ All outputs saved to {output_dir}")
        else:
            print("\n✓ Odyssey visualization generation complete!")
            print("  (View plots in matplotlib windows)")

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
    """Command-line interface for odyssey.py"""
    parser = argparse.ArgumentParser(
        description="Comprehensive 3D cloud visualization with full radiative transfer"
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
    parser.add_argument(
        "--wavelength", type=float, default=0.55,
        help="Wavelength in microns (default: 0.55 for visible)"
    )

    args = parser.parse_args()
    main(args.filename, args.output, args.sza, args.wavelength)


if __name__ == "__main__":
    cli()
