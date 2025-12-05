"""Reformat ice phase function tables for RGB wavelengths.

This script reads the ice NetCDF file containing P11 and writes two text files:
- IceMie0_normalized.txt  : forward-peak part (θ < split angle) per wavelength
- IceMiePF3_normalized.txt: remainder (θ ≥ split angle) per wavelength

Defaults target an effective radius of 30 µm (diameter 60 µm) and RGB
wavelengths near 0.47, 0.55, and 0.64 µm. Only P11 is used (no polarization).

If the forward peak at 0° is not greater than the value at 90° by the given
ratio threshold, the tables are not split: IceMie0 is all zeros and
IceMiePF3 is the fully normalized phase function.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import netCDF4
import numpy as np


def _find_nearest(arr: np.ndarray, target: float) -> Tuple[int, float]:
    """Return (index, value) of the element in arr closest to target."""
    idx = int(np.abs(arr - target).argmin())
    return idx, float(arr[idx])


def _normalize_phase_function(
    values: np.ndarray, angles_rad: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Scale values so that 2π ∫ P(θ) sinθ dθ = 4π (i.e., ∫ P sinθ dθ = 2)."""
    integral = float(np.trapz(values * np.sin(angles_rad), angles_rad))
    if integral <= 0:
        return np.zeros_like(values), 0.0, 0.0
    scale = 2.0 / integral
    return values * scale, scale, integral


def _split_peak(
    p11: np.ndarray,
    angles_deg: np.ndarray,
    split_angle_deg: float,
    forward_ratio_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split into forward peak and remainder; return (peak, rest, ratio)."""
    peak_idx = int(np.abs(angles_deg - 0.0).argmin())
    ninety_idx = int(np.abs(angles_deg - 90.0).argmin())
    p0 = float(p11[peak_idx])
    p90 = float(p11[ninety_idx]) if float(p11[ninety_idx]) != 0 else 1e-30
    ratio = p0 / p90

    if ratio <= forward_ratio_threshold:
        return np.zeros_like(p11), p11.copy(), ratio

    forward_mask = angles_deg < split_angle_deg
    peak = np.where(forward_mask, p11, 0.0)
    rest = np.where(forward_mask, 0.0, p11)
    return peak, rest, ratio


def reformat_tables(
    nc_path: Path,
    output_dir: Path,
    eff_radius_um: float,
    wavelengths_um: Iterable[float],
    split_angle_deg: float,
    forward_ratio_threshold: float,
) -> None:
    ds = netCDF4.Dataset(nc_path)

    wavelengths = ds.variables["wavelengths"][:]
    effective_diameter = ds.variables["effective_diameter"][:]
    angles_deg = ds.variables["phase_angles"][:]
    angles_rad = np.deg2rad(angles_deg)
    p11 = ds.variables["p11_phase_function"]

    eff_diam_target = eff_radius_um * 2.0
    eff_idx, eff_val = _find_nearest(effective_diameter, eff_diam_target)

    wavelength_indices: List[int] = []
    wavelength_vals: List[float] = []
    for wl in wavelengths_um:
        idx, val = _find_nearest(wavelengths, wl)
        wavelength_indices.append(idx)
        wavelength_vals.append(val)

    peak_cols: List[np.ndarray] = []
    rest_cols: List[np.ndarray] = []
    ratios: List[float] = []

    for idx in wavelength_indices:
        original = p11[:, eff_idx, idx]
        peak, rest, ratio = _split_peak(
            original, angles_deg, split_angle_deg, forward_ratio_threshold
        )
        peak_norm, peak_scale, peak_int = _normalize_phase_function(
            peak, angles_rad
        )
        rest_norm, rest_scale, rest_int = _normalize_phase_function(
            rest, angles_rad
        )

        # If no split happened, peak_norm will be zeros; ensure rest is normalized.
        peak_cols.append(peak_norm)
        rest_cols.append(rest_norm)
        ratios.append(ratio)

        print(
            f"Wavelength {wavelengths[idx]:.4g} µm (requested {wavelength_vals[len(ratios)-1]:.4g} µm): "
            f"Deff={eff_val:.1f} µm, peak ratio={ratio:.2e}, "
            f"peak scale={peak_scale:.3g}, rest scale={rest_scale:.3g}, "
            f"peak integral={peak_int:.3g}, rest integral={rest_int:.3g}"
        )

    peak_matrix = np.column_stack(peak_cols)
    rest_matrix = np.column_stack(rest_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "IceMie0_normalized.txt", peak_matrix, fmt="%.10f")
    np.savetxt(output_dir / "IceMiePF3_normalized.txt", rest_matrix, fmt="%.10f")

    print(f"Written {output_dir / 'IceMie0_normalized.txt'} "
          f"and {output_dir / 'IceMiePF3_normalized.txt'} "
          f"with {peak_matrix.shape[0]} angles and {peak_matrix.shape[1]} wavelengths.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reformat ice P11 data into forward-peak (Mie0) and remainder (MiePF3) text tables."
    )
    parser.add_argument(
        "--nc-path",
        type=Path,
        default=Path("GeneralHabitMixture_SeverelyRough_AllWavelengths_FullPhaseMatrix.nc"),
        help="Path to the ice NetCDF file with full phase matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Mie_tables"),
        help="Directory to write IceMie0_normalized.txt and IceMiePF3_normalized.txt.",
    )
    parser.add_argument(
        "--eff-radius",
        type=float,
        default=30.0,
        help="Effective radius in microns (converted to effective diameter by multiplying by 2).",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="+",
        default=[0.47, 0.55, 0.64],
        help="Target wavelengths (µm); nearest available values are used.",
    )
    parser.add_argument(
        "--split-angle",
        type=float,
        default=10.0,
        help="Angle threshold in degrees to define the forward peak.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=10.0,
        help="Split only if P(0°)/P(90°) exceeds this ratio.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reformat_tables(
        nc_path=args.nc_path,
        output_dir=args.output_dir,
        eff_radius_um=args.eff_radius,
        wavelengths_um=args.wavelengths,
        split_angle_deg=args.split_angle,
        forward_ratio_threshold=args.ratio_threshold,
    )
