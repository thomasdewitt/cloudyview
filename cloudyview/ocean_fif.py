"""FIF-based ocean heightfield / normal-map generator.

Generates a 2D multifractal heightfield via scaleinvariance.FIF_ND, boosts
one random direction in Fourier space to give a dominant wave-propagation
direction, and returns analytic normals via centered finite differences.
Intended to be sampled with periodic wrap at ocean hits during rendering.

Defaults tuned on phone photos of cumulus-over-ocean scenes: H=0.9 gives a
smooth rolling swell; outer scale 10 m + 1 m max-min gives a gentle sea.
"""
from typing import Tuple
import numpy as np
import scaleinvariance


DEFAULT_N = 8192
DEFAULT_H = 0.9
DEFAULT_ALPHA = 2.0
DEFAULT_C1 = 0.001
DEFAULT_OUTER_SCALE_M = 10.0
DEFAULT_TARGET_RANGE_M = 1.0
DEFAULT_BOOST = 8.0
DEFAULT_BOOST_BANDWIDTH = 0.15
DEFAULT_DX_M = 0.05   # fixed; tile repeats every N*dx = 410 m by default


def generate_fif_normals(
    dx_m: float = DEFAULT_DX_M,
    N: int = DEFAULT_N,
    H: float = DEFAULT_H,
    alpha: float = DEFAULT_ALPHA,
    C1: float = DEFAULT_C1,
    outer_scale_m: float = DEFAULT_OUTER_SCALE_M,
    target_range_m: float = DEFAULT_TARGET_RANGE_M,
    boost: float = DEFAULT_BOOST,
    boost_bandwidth: float = DEFAULT_BOOST_BANDWIDTH,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (nx, ny, nz, dx_m) as float32 N×N arrays + the dx used.

    nx/ny/nz are components of the unit surface normal at grid cell centers.
    Sample them with periodic wrap + bilinear interpolation at any world (x,y).
    """
    scaleinvariance.set_numerical_precision('float32')

    if verbose:
        print(f"  FIF: {N}x{N} alpha={alpha} C1={C1} H={H} "
              f"outer={outer_scale_m}m dx={dx_m:.3f}m "
              f"(tile={N * dx_m / 1000:.2f} km)")

    fif = scaleinvariance.FIF_ND(
        size=(N, N),
        alpha=alpha, C1=C1, H=H,
        outer_scale=outer_scale_m / dx_m,
        periodic=True,
    )
    h = fif.astype(np.float64) - fif.mean()

    if boost != 1.0:
        if rng is None:
            rng = np.random.default_rng()
        theta = rng.uniform(0.0, 2.0 * np.pi)
        target_k = 1.0 / outer_scale_m   # cycles/m
        k0x = np.cos(theta) * target_k
        k0y = np.sin(theta) * target_k
        if verbose:
            print(f"  FIF boost: direction {np.degrees(theta):6.1f}° ×{boost}")
        F = np.fft.fft2(h)
        kx = np.fft.fftfreq(N, d=dx_m)
        ky = np.fft.fftfreq(N, d=dx_m)
        KX, KY = np.meshgrid(kx, ky, indexing='xy')
        sigma_k = target_k * boost_bandwidth
        d1 = (KX - k0x) ** 2 + (KY - k0y) ** 2
        d2 = (KX + k0x) ** 2 + (KY + k0y) ** 2
        bump = np.clip(np.exp(-d1 / (2 * sigma_k ** 2))
                       + np.exp(-d2 / (2 * sigma_k ** 2)), 0.0, 1.0)
        F = F * (1.0 + (boost - 1.0) * bump)
        h = np.real(np.fft.ifft2(F))

    h_range = h.max() - h.min()
    h = h / h_range * target_range_m

    dhdx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2.0 * dx_m)
    dhdy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2.0 * dx_m)
    inv_len = 1.0 / np.sqrt(dhdx ** 2 + dhdy ** 2 + 1.0)
    nx = np.ascontiguousarray((-dhdx * inv_len).astype(np.float32))
    ny = np.ascontiguousarray((-dhdy * inv_len).astype(np.float32))
    nz = np.ascontiguousarray(inv_len.astype(np.float32))

    if verbose:
        slope = np.sqrt(dhdx ** 2 + dhdy ** 2)
        print(f"  FIF stats: h∈[{h.min():.3f},{h.max():.3f}] m  "
              f"|slope| med={np.median(slope):.3f} p99={np.quantile(slope, 0.99):.3f}")
    return nx, ny, nz, dx_m


def dummy_fif_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1×1 placeholders for when use_fif=False, so the kernel signature is stable."""
    z = np.zeros((1, 1), dtype=np.float32)
    return z, z.copy(), np.ones((1, 1), dtype=np.float32)
