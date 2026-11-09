"""cloudyview's own reading of docs/desert_tile_format.md: the meta.json
derivations, written from the spec, to check tiles against their bytes.

The desert's generator lives in desert-surface now (desert_surface.tile is
its reference writer). This is a second implementation of the same spec on
purpose: the shipped tile and the conformance fixtures are checked against
it here, and against the writer there, so a change on either side that the
spec does not carry fails a test.
"""

from __future__ import annotations

import numpy as np

# TILE_FORMAT "Channels": the albedo endmembers at lightness 0, 1/2, 1.
ALBEDO_PAVEMENT = (0.17, 0.145, 0.12)
ALBEDO_SOIL = (0.31, 0.235, 0.155)
ALBEDO_WASH = (0.33, 0.285, 0.22)


def albedo_of(lightness: np.ndarray) -> np.ndarray:
    """Piecewise linear in lightness, the knot at soil."""
    l = np.clip(lightness, 0.0, 1.0)[..., None]
    p, s, w = (np.asarray(c) for c in (ALBEDO_PAVEMENT, ALBEDO_SOIL, ALBEDO_WASH))
    return np.where(l < 0.5, p + (s - p) * 2.0 * l, s + (w - s) * (2.0 * l - 1.0))


def _round_up(v: float, digits: int = 3) -> float:
    q = 10 ** digits
    return float(np.ceil(v * q) / q)


def _bilerp_at_level0_centres(level: np.ndarray, n0: int) -> np.ndarray:
    """A level read bilinearly (periodic) at every level-0 texel centre."""
    side = level.shape[0]
    c = (np.arange(n0) + 0.5) * side / n0 - 0.5
    i0 = np.floor(c).astype(int)
    f = c - i0
    a, b = i0 % side, (i0 + 1) % side
    rows = level[a] * (1 - f)[:, None] + level[b] * f[:, None]
    return rows[:, a] * (1 - f)[None, :] + rows[:, b] * f[None, :]


def relief_tables(heights: list, cell: float) -> dict:
    """slope_max, envelope_m, slope_variance_removed and band_max_m, as the
    spec's meta.json table defines them, from the stored heights."""
    n = heights[0].shape[0]
    slope_max, msq = [], []
    for k, h in enumerate(heights):
        dx = cell * 2 ** k
        gx0 = (np.roll(h, -1, 1) - h) / dx
        gy0 = (np.roll(h, -1, 0) - h) / dx
        pairs = [(gx, gy) for gx in (gx0, np.roll(gx0, -1, 0))
                 for gy in (gy0, np.roll(gy0, -1, 1))]
        slope_max.append(_round_up(1.25 * max(float(np.hypot(*q).max())
                                               for q in pairs)))
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * dx)
        gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * dx)
        msq.append(float((gx * gx + gy * gy).mean()))
    envelope = [_round_up(1.1 * float((heights[0]
                                       - _bilerp_at_level0_centres(h, n)).max()))
                for h in heights]
    lod = np.log2(5.0)
    lo = int(np.floor(lod))
    f = lod - lo
    coarse = ((1 - f) * _bilerp_at_level0_centres(heights[lo], n)
              + f * _bilerp_at_level0_centres(heights[lo + 1], n))
    return {"slope_max": slope_max,
            "envelope_m": envelope,
            "slope_variance_removed": [round(msq[0] - v, 6) for v in msq],
            "band_max_m": round(float(np.abs(heights[0] - coarse).max()), 4)}


def albedo_correction(lights: list) -> list:
    truth = albedo_of(lights[0]).reshape(-1, 3).mean(0)
    return [[round(float(v), 4) for v in truth / albedo_of(l).reshape(-1, 3).mean(0)]
            for l in lights]


def derived_meta(levels: list, cell: float) -> dict:
    """Every meta.json field derived from the stored levels."""
    heights = [lv[..., 0].astype(np.float64) for lv in levels]
    lights = [lv[..., 1].astype(np.float64) for lv in levels]
    return {"height_min_m": float(heights[0].min()),
            "height_max_m": float(heights[0].max()),
            "mean_lightness": round(float(lights[-1][0, 0]), 4),
            **relief_tables(heights, cell),
            "albedo_correction": albedo_correction(lights),
            "bounce_albedo": [round(float(v), 3)
                              for v in albedo_of(lights[0]).reshape(-1, 3).mean(0)]}


# TILE_FORMAT "Conformance": the closed-form field the fixtures hold
# (desert-surface tests/fixtures/make_fixtures.py writes them).
FIXTURE_N = 64
FIXTURE_EXTENT_M = 640.0


def fixture_field():
    x = (np.arange(FIXTURE_N) + 0.5) / FIXTURE_N
    X, Y = np.meshgrid(x, x)
    height = (12.0 * np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y)
              + 3.0 * np.sin(6 * np.pi * (X + Y)) + 0.37)
    light = np.round(0.5 * (np.sin(2 * np.pi * X) + 1.0)
                     * np.cos(2 * np.pi * Y) ** 2 * 64.0) / 64.0
    return height, light


def box_mips(level0: np.ndarray) -> list:
    """Exact 2x2 box means in double precision, down to 1x1."""
    mips = [level0]
    while mips[-1].shape[0] > 1:
        c = mips[-1]
        mips.append(0.25 * (c[0::2, 0::2] + c[1::2, 0::2]
                            + c[0::2, 1::2] + c[1::2, 1::2]))
    return mips
