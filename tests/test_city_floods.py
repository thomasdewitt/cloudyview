"""The industrial belt's flood pools: the far-field mean is the near field's.

The shader (city/floods.wgsl) paints each floodlight's pool as an
anisotropic Gaussian on the
ground of its own lot (city_flood_pool) — lopsided along its aim, and cut at
the wall for a wall pack —
and, once the pixel footprint outgrows a pool, hands over to the pools' mean
over the ground the near field shows (city_flood_ground). That mean is a
closed form: a separable product of erf differences over the lot rectangle
(city_flood_visible), with erf from Abramowitz & Stegun 7.1.26 (city_erf).
If the closed form disagrees with the painted pools, the hand-off is a
visible seam. These tests pin the closed form against a brute-force integral
of the painted pool, and pin the Python copy of the erf coefficients to the
WGSL's.

Needs no GPU: the WGSL functions are mirrored line for line below.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest
from .conftest import ShaderModule

WGSL = ShaderModule("city").read_text()

# city_erf's coefficients, as the WGSL states them.
P = 0.3275911
A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def test_erf_coefficients_match_wgsl():
    body = re.search(r"fn city_erf\(x: f32\) -> f32 \{(.*?)\n\}", WGSL,
                     re.S).group(1)
    nums = [float(v) for v in re.findall(r"-?\d+\.\d+", body)]
    assert nums[:3] == [1.0, 1.0, P]
    assert nums[3:8] == list(A)


def city_erf(x):
    a = np.abs(x)
    t = 1.0 / (1.0 + P * a)
    poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))))
    return np.sign(x) * (1.0 - poly * np.exp(-a * a))


def test_erf_accuracy():
    x = np.linspace(-4, 4, 2001)
    exact = np.array([math.erf(v) for v in x])
    assert np.max(np.abs(city_erf(x) - exact)) < 2e-7


def gauss_int(x0, x1, sigma):
    k = 0.70710678 / sigma
    return sigma * 1.2533141 * (city_erf(x1 * k) - city_erf(x0 * k))


def flood_visible(pool, aim, sigma, lo, hi):
    """Mirror of city_flood_visible. sigma = (ahead, behind, across)."""
    rlo = np.subtract(lo, pool)
    rhi = np.subtract(hi, pool)
    along_x = abs(aim[0]) > 0.5
    s = aim[0] if along_x else aim[1]
    u0, u1 = (rlo[0], rhi[0]) if along_x else (rlo[1], rhi[1])
    a0, a1 = (u0, u1) if s > 0 else (-u1, -u0)
    along = gauss_int(max(a0, 0.0), max(a1, 0.0), sigma[0])
    if sigma[1] > 0.0:
        along += gauss_int(min(a0, 0.0), min(a1, 0.0), sigma[1])
    b0, b1 = (rlo[1], rhi[1]) if along_x else (rlo[0], rhi[0])
    return along * gauss_int(b0, b1, sigma[2])


def flood_pool(pool, aim, sigma, X, Y):
    """Mirror of city_flood_pool, on a grid."""
    dx, dy = X - pool[0], Y - pool[1]
    a = dx * aim[0] + dy * aim[1]
    b = aim[0] * dy - aim[1] * dx
    sa = np.where(a >= 0, sigma[0], sigma[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.exp(-0.5 * ((a / sa) ** 2 + (b / sigma[2]) ** 2))
    return np.where(sa <= 0, 0.0, g)


# (pool, aim, (sigma ahead, behind, across)) over a 180 x 90 m lot: a wall
# pack on a +y court wall and one on a -x wall near a lot corner (tails cut
# by the lot edge), and yard poles thrown inward from the lot's edges — one
# slewed so its pool runs off the lot's side, one close enough to the lot
# edge behind it that its back tail is cut.
LOT_LO, LOT_HI = (1000.0, 2000.0), (1180.0, 2090.0)
CASES = [
    ((1060.0, 2030.0), (0.0, 1.0), (7.0, 0.0, 7.0)),
    ((1004.0, 2080.0), (-1.0, 0.0), (8.4, 0.0, 5.6)),
    ((1030.0, 2015.0), (0.0, 1.0), (9.0, 3.5, 5.5)),
    ((1176.0, 2050.0), (-1.0, 0.0), (10.8, 4.2, 6.6)),
    ((1003.0, 2004.0), (1.0, 0.0), (7.2, 2.8, 4.4)),
]


@pytest.mark.parametrize("case", CASES)
def test_visible_integral_matches_brute_force(case):
    pool, aim, sigma = case
    h = 0.05
    xs = np.arange(LOT_LO[0] + h / 2, LOT_HI[0], h)
    ys = np.arange(LOT_LO[1] + h / 2, LOT_HI[1], h)
    X, Y = np.meshgrid(xs, ys)
    brute = flood_pool(pool, aim, sigma, X, Y).sum() * h * h
    closed = flood_visible(pool, aim, sigma, LOT_LO, LOT_HI)
    assert closed == pytest.approx(brute, rel=2e-4)


def test_cut_pools_lose_their_tails():
    # The case the old infinite-plane mean got wrong: a corner wall pack
    # keeps well under half of a whole Gaussian's 2 pi sa sb.
    pool, aim, sigma = CASES[1]
    whole = 2.0 * math.pi * sigma[0] * sigma[2]
    assert flood_visible(pool, aim, sigma, LOT_LO, LOT_HI) < 0.4 * whole
