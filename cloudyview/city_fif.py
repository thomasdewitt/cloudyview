"""FIF-based city density tile generator (the night-city surface).

One periodic 2D multifractal drives the whole city: building height,
building presence, and the aggregate ground glow the clouds see from above
are all functions of this density field, read at different mip levels.
One texel is one city block, so the tile repeats every N * cell_m meters —
far outside the visible range at the shipped values.

Parameters follow the cyberpunk brief rather than any ocean: H=0 keeps the
field rough at every scale (a dense block can sit next to an empty lot),
C1=0.07 gives real intermittency — the "crazy clusters" are the cascade's
extreme excursions — and alpha=2 keeps the tails lognormal rather than
Levy-wild, so the tallest district is dramatic but not a single spike.
"""
from typing import Tuple

import numpy as np
import scaleinvariance
from scaleinvariance.simulation.FIF import extremal_levy

DEFAULT_N = 1024
DEFAULT_H = 0.0
DEFAULT_ALPHA = 2.0
# Lowered 0.1 -> 0.07 (Thomas, 2026-08-22). Less extreme clustering: the
# districts stay legible instead of collapsing onto a few runaway spikes.
DEFAULT_C1 = 0.07
# Districts: the largest coherent structure the cascade builds, in cells.
DEFAULT_OUTER_SCALE_CELLS = 256.0
DEFAULT_CELL_M = 90.0


def generate_city_density(
    N: int = DEFAULT_N,
    H: float = DEFAULT_H,
    alpha: float = DEFAULT_ALPHA,
    C1: float = DEFAULT_C1,
    outer_scale_cells: float = DEFAULT_OUTER_SCALE_CELLS,
    rng: np.random.Generator = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (height, rank), both float32 (N, N) in [0, 1].

    height: the cascade normalized by its 99.5th percentile, UNCLIPPED —
        the building-height modulator with its whole lognormal tail. The
        extremes are the point: the rare 5x-8x excursions are the megatowers
        that reach the cloud deck (Thomas, 2026-08-20: the tallest should
        "poke into clouds", the distribution "not really clipped").
    rank: the same field rank-transformed to exactly uniform [0, 1] — the
        thresholdable channel (building presence, lit fraction), so a cut at
        q keeps exactly (1-q) of the city whatever the cascade drew.
    """
    scaleinvariance.set_numerical_precision('float32')
    if rng is None:
        rng = np.random.default_rng()

    if verbose:
        print(f"  city FIF: {N}x{N} alpha={alpha} C1={C1} H={H} "
              f"outer={outer_scale_cells:.0f} cells")

    # Same discipline as ocean_fif: hand the noise in, or the rng steers
    # nothing and two calls with the same generator differ.
    levy = extremal_levy(
        alpha, size=N * N, seed=int(rng.integers(0, 2 ** 32))
    ).reshape(N, N)
    fif = scaleinvariance.FIF_ND(
        size=(N, N),
        alpha=alpha, C1=C1, H=H,
        levy_noise=levy,
        outer_scale=outer_scale_cells,
        periodic=True,
    ).astype(np.float64)

    p995 = np.quantile(fif, 0.995)
    height = (fif / p995).astype(np.float32)

    order = np.argsort(fif, axis=None)
    rank = np.empty(N * N, dtype=np.float32)
    rank[order] = np.arange(N * N, dtype=np.float32) / float(N * N - 1)
    rank = rank.reshape(N, N)

    if verbose:
        print(f"  city stats: mean={fif.mean():.3f} p99.5={p995:.3f} "
              f"max/p99.5={fif.max() / p995:.2f}")
    return height, rank
