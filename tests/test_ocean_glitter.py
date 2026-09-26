"""The sea's glitter: the sun's reflection drawn as the facets that catch it.

The glint is the smooth Cox-Munk lobe times 1 + core (draw - 1), the draw
K ~ Poisson(lambda) lognormal facets over lambda (raymarch.wgsl
sea_poisson, sea_glitter_brightness). Its mean is 1, so the glitter moves
no energy: the tests compute that mean exactly from the shader's own
source, and show that breaking the sampler would move it.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"


def _wgsl_scalar(src, name):
    m = re.search(rf"const {name}: f32 = ([0-9.eE+-]+);", src)
    assert m, f"{name} not found in raymarch.wgsl"
    return float(m.group(1))


def _fn_body(src, name):
    m = re.search(rf"(?ms)^fn {name}\(.*?^}}", src)
    assert m, f"{name} not found in raymarch.wgsl"
    return m.group(0)


def _glitter_mean(lam, src):
    """E[sea_glitter_brightness(K) / lambda], K = sea_poisson(lambda), as
    the given raymarch.wgsl computes them — exactly, not sampled.

    The port takes from the source everything the mean depends on: the
    normal branch's threshold; the inversion's start, step, pmf divisor,
    comparison and step cap, stepped in float32 with its stall guard over
    the 2^24 values the 24-bit uniform takes; and the facets' count, cutoff
    and lognormal correction (the -s2/2 term, 0 if it is gone), with the
    uniform's 1e-7 clamp. So a WGSL edit that moves the mean moves this."""
    from scipy.special import i0e, ndtr
    poisson = _fn_body(src, "sea_poisson")
    bright = _fn_body(src, "sea_glitter_brightness")
    assert ">> vec2<u32>(8u)) * (1.0 / 16777216.0)" in _fn_body(src, "sea_uniforms")
    threshold = float(re.search(r"if \(lambda > ([0-9.]+)\)", poisson).group(1))
    assert "max(round(lambda + sqrt(lambda) * z), 0.0)" in poisson
    k0 = float(re.search(r"var k = ([0-9.]+);", poisson).group(1))
    step = float(re.search(r"k = k \+ ([0-9.]+);", poisson).group(1))
    divisor = re.search(r"pmf = pmf \* lambda / ([^;]+);", poisson).group(1)
    assert re.fullmatch(r"[k0-9. ()+-]+", divisor), divisor
    comparison = re.search(r"if \(r\.x (<=|<) cdf\)", poisson).group(1)
    cap = int(re.search(r"i < (\d+); i = i \+ 1\)", poisson).group(1))
    cutoff = float(re.search(r"if \(k >= ([0-9.]+) \|\| k <= 0\.0\)",
                             bright).group(1))
    facets = int(re.search(r"i < (\d+)u;", bright).group(1))
    assert "sqrt(-2.0 * log(max(r.x, 1e-7)))" in bright
    corr = re.search(r"exp\(SEA_GLITTER_SPREAD \* z(?: - ([0-9.]+) \* s2)?\)",
                     bright)
    assert corr, "the facet brightness's form changed"
    half = float(corr.group(1)) if corr.group(1) else 0.0
    spread = _wgsl_scalar(src, "SEA_GLITTER_SPREAD")
    # One facet's mean brightness over the shader's uniforms: the angle
    # averages exp(b cos) to I0(b); the radius is the 24-bit grid, clamped.
    rx = np.maximum(np.arange(2 ** 24, dtype=np.float64) / 2 ** 24, 1e-7)
    b = spread * np.sqrt(-2.0 * np.log(rx))
    facet = float(np.mean(i0e(b) * np.exp(b - half * spread * spread)))

    def brightness(k):
        if k >= cutoff or k <= 0:
            return k
        return min(facets, int(np.ceil(k))) * facet
    if lam > threshold:
        ks = np.arange(0, int(lam + 20 * np.sqrt(lam)) + 1)
        sd = np.sqrt(lam)
        pk = ndtr((ks + 0.5 - lam) / sd) - ndtr((ks - 0.5 - lam) / sd)
        pk[0] = ndtr((0.5 - lam) / sd)              # max(round(...), 0)
        return float(sum(pk[k] * brightness(k) for k in ks)) / lam
    f32 = np.float32
    lam32 = f32(lam)
    pmf = cdf = f32(np.exp(-lam32))
    thresholds, values, k, exit_k = [cdf], [k0], k0, None
    for _ in range(cap):
        k = k + step
        pmf = f32(pmf * lam32 / f32(eval(divisor, {}, {"k": k})))
        nxt = f32(cdf + pmf)
        if nxt <= cdf:
            exit_k = k
            break
        cdf = nxt
        thresholds.append(cdf)
        values.append(k)
    if exit_k is None:
        exit_k = k
    # K(r): the value at the first cdf that r reaches (<=, or strictly <);
    # past the last, the loop's own exit value.
    r = np.arange(2 ** 24, dtype=np.float64) / 2 ** 24
    idx = np.searchsorted(np.asarray(thresholds, np.float64), r,
                          side="left" if comparison == "<=" else "right")
    kr = np.where(idx >= len(thresholds), exit_k,
                  np.asarray(values)[np.minimum(idx, len(values) - 1)])
    ks, counts = np.unique(kr, return_counts=True)
    return float(sum(c * brightness(k) for k, c in zip(ks, counts))
                 / 2 ** 24) / lam


# From the rarest cell the core admits to the normal branch, both sides of
# the switch between them.
GLITTER_LAMBDAS = (5e-5, 1e-4, 1e-3, 1e-2, 0.3, 1.0, 3.5, 10.0, 15.99,
                   16.01, 40.0)


def _glitter_tolerance(lam):
    """1e-3, plus what a 24-bit uniform can resolve at a rare lambda: the
    chance of any glint, ~lambda, is counted in steps of 2^-24, and exp(-lambda)
    is a float32 near 1 whose ulp is 2^-24 — together 2^-23 of lambda."""
    return 1e-3 + 2.0 ** -23 / lam


def test_the_glitter_keeps_the_smooth_specular_mean():
    """The glint is the smooth Cox-Munk lobe times 1 + core (draw - 1): the
    draw's mean is 1 — the facets' expected count is lambda and each facet's
    brightness has mean 1 — so whatever the core weight the energy is the
    specular's. Computed exactly from the shader's own sampler."""
    src = SHADER.read_text()
    assert "sparkle = 1.0 + core * (draw - 1.0);" in src
    for lam in GLITTER_LAMBDAS:
        assert _glitter_mean(lam, src) == pytest.approx(
            1.0, abs=_glitter_tolerance(lam)), lam


@pytest.mark.parametrize("mutation", [
    ("exp(SEA_GLITTER_SPREAD * z - 0.5 * s2)", "exp(SEA_GLITTER_SPREAD * z)"),
    ("var k = 0.0;", "var k = 1.0;"),
    ("pmf = pmf * lambda / k;", "pmf = pmf * lambda / (k + 1.0);"),
])
def test_the_glitter_mean_test_catches_a_broken_sampler(mutation):
    """Negative controls: the port reads the sampler from the source, so a
    lost lognormal correction or an off-by-one in the inversion moves the
    mean it computes off 1 — the test above would fail on either."""
    src = SHADER.read_text()
    old, new = mutation
    assert src.count(old) == 1
    broken = src.replace(old, new)
    assert any(abs(_glitter_mean(lam, broken) - 1.0) > 10 * _glitter_tolerance(lam)
               for lam in GLITTER_LAMBDAS)


def test_the_glitter_cells_fold_with_the_tile():
    """A glitter cell is the facet times a power of two, folded at the tile:
    a whole number of facets spans the tile, so every level's cells fold
    with it and the sparkles have no seam at the tile's edge."""
    src = SHADER.read_text()
    meta = json.loads((REPO / "cloudyview" / "soar" / "ocean"
                       / "meta.json").read_text())
    per_tile = meta["tile_extent_m"] / _wgsl_scalar(src, "SEA_GLITTER_FACET_M")
    assert per_tile == pytest.approx(round(per_tile), abs=1e-6)
    assert int(round(per_tile)) & (int(round(per_tile)) - 1) == 0
