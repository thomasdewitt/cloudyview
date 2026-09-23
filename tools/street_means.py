#!/usr/bin/env python
"""Measure the night city's street-surface means for raymarch.wgsl.

THE STREET SURFACE section of raymarch.wgsl shades the ground from a
cross-section (sidewalk, kerb, gutter, lanes, markings) under the lamp pools,
darkened where it is wet. Near the camera every term is resolved; far away the
lamp term is replaced by its area mean, and that mean cannot be written down
analytically: the pools pass through a knee, and they fall hardest on the
sidewalk (the lamps stand on it), whose albedo is not the road's. So it is
measured here, by integrating the shader's own formulas over a block, and
fitted as A s / (1 + B s) in the district lamp scale s:

    CITY_GROUND_FAR_STREET   the mean over the street (everything outside
                             the plot)
    CITY_GROUND_FAR_PLOT     the mean over the plot interior (a plaza at
                             sidewalk grade, which only the pools' tails reach)

It also fits CITY_PUDDLE_FIT, the expected puddle coverage at a given bias:
the far-field stand-in for the thresholded two-octave noise.

A port, not a copy of the renderer: noise textures whose mean is 1 by
construction (flag tone, stains, aggregate, patching) are left at 1, and the
manhole covers and grates, a fraction of a percent of the area, are left out.
Everything that moves the mean is here. Keep it in step with the WGSL.

    uv run python tools/street_means.py
"""

from __future__ import annotations

import numpy as np

# --- constants, as raymarch.wgsl states them ---------------------------------
CELL = 90.0
STREET_HALF = 6.0
AVENUE_EXTRA = 10.0
AVENUE_PERIOD = 8
LAMP_SPACING = 26.0
LAMP_OFFSET = 2.5
LAMP_SIGMA = 3.2
POOL_KNEE = 0.30

SIDEWALK_W = 3.0
SIDEWALK_W_AVE = 4.5
KERB_R = 3.0
KERB_TOP_W = 0.30
GUTTER_W = 0.45
PARK_W = 2.4
FLAG = 1.5
JOINT_W = 0.018

REL_WALK = 1.5
REL_KERB = 2.6
REL_GUTTER = 1.7
REL_PAINT = 4.2
PAINT_YELLOW = np.array([1.0, 0.78, 0.22])
PAINT_WEAR_MEAN = 0.8

WET_ROAD = 0.90
WET_WALK = 0.85
WET_DARKEN = 0.93
PUDDLE_DARKEN = 0.85
PUDDLE_THR = 0.67
PUDDLE_EDGE = 0.015

# The district lamp scale s = CITY_LAMP_RADIANCE (0.55) x (0.20 .. 1.50).
S_GRID = 0.55 * np.linspace(0.20, 1.50, 14)


def is_avenue(k):
    return (np.mod(k, AVENUE_PERIOD) == 0)


def half(k):
    return STREET_HALF + np.where(is_avenue(k), AVENUE_EXTRA, 0.0)


def lamp_inset(k):
    return half(k) - LAMP_OFFSET


def pcg2d(x, y):
    """raymarch.wgsl pcg2d, in uint32 arithmetic."""
    m = np.uint64(0xFFFFFFFF)
    x = np.asarray(x, np.uint64) & m
    y = np.asarray(y, np.uint64) & m
    x = (x * 1664525 + 1013904223) & m
    y = (y * 1664525 + 1013904223) & m
    x = (x + y * 1664525) & m
    y = (y + x * 1664525) & m
    x ^= x >> np.uint64(16)
    y ^= y >> np.uint64(16)
    x = (x + y * 1664525) & m
    y = (y + x * 1664525) & m
    x ^= x >> np.uint64(16)
    y ^= y >> np.uint64(16)
    return x, y


TILE_N = 1024   # blocks per tile: city_line_seed folds a centreline index


def line_seed(k):
    """city_line_seed: the centreline index folded at the tile."""
    k = np.mod(np.asarray(k).astype(np.int64), TILE_N)
    return k.astype(np.uint64)


def park_side_bits(k):
    """The low bit city_park_side reads."""
    k = line_seed(k)
    hx, _ = pcg2d(k ^ np.uint64(0x6c8e9cf5), np.full_like(k, 0x2c1b3c6d))
    return (hx & np.uint64(1)).astype(np.int64)


def lamp_station(k, a):
    """city_lamp_station: minor streets draw a spacing (22-31 m) and phase."""
    k = np.asarray(k)
    seed = line_seed(k)
    hx, hy = pcg2d(seed ^ np.uint64(0x3c6ef372), np.full_like(seed, 0xa54ff53a))
    sp = 22.0 + 3.0 * (hx & np.uint64(3)).astype(np.float64)
    ph = sp * hy.astype(np.float64) * 2.3283064365386963e-10
    sp = np.where(is_avenue(k), LAMP_SPACING, sp)
    ph = np.where(is_avenue(k), 0.0, ph)
    return ph + np.round((a - ph) / sp) * sp


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --- the shader's noise, in float64 -------------------------------------------
def hash12(px, py):
    p3x = np.mod(px * 0.1031, 1.0)
    p3y = np.mod(py * 0.1031, 1.0)
    p3z = np.mod(px * 0.1031, 1.0)
    d = p3x * (p3y + 33.33) + p3y * (p3z + 33.33) + p3z * (p3x + 33.33)
    p3x, p3y, p3z = p3x + d, p3y + d, p3z + d
    return np.mod((p3x + p3y) * p3z, 1.0)


def vnoise(px, py):
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy
    wx, wy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = hash12(ix, iy)
    b = hash12(ix + 1, iy)
    c = hash12(ix, iy + 1)
    d = hash12(ix + 1, iy + 1)
    return (a + (b - a) * wx) * (1 - wy) + (c + (d - c) * wx) * wy


# --- the ground, one cell --------------------------------------------------------
def pools(x, y):
    """city_street_pools, verbatim."""
    out = np.zeros_like(x)
    bxi = np.round(x / CELL)
    bx = bxi * CELL
    amp = np.where(is_avenue(bxi), 1.6, 1.0)
    ly = lamp_station(bxi, y)
    for s in (-1.0, 1.0):
        lx = bx + s * lamp_inset(bxi)
        out += amp * np.exp(-((x - lx) ** 2 + (y - ly) ** 2)
                            / (2 * LAMP_SIGMA ** 2))
    byi = np.round(y / CELL)
    by = byi * CELL
    amp = np.where(is_avenue(byi), 1.6, 1.0)
    lx = lamp_station(byi, x)
    for s in (-1.0, 1.0):
        lyy = by + s * lamp_inset(byi)
        out += amp * np.exp(-((x - lx) ** 2 + (y - lyy) ** 2)
                            / (2 * LAMP_SIGMA ** 2))
    return out


def ground(ci, x, y, noise_off):
    """(albedo rgb relative to asphalt, inside-plot mask) at points of cell ci,
    every edge a street (the stub city-layout's predicate replaces)."""
    cmin = np.array(ci, float) * CELL
    pmin = cmin + np.array([half(ci[0]), half(ci[1])])
    pmax = cmin + CELL - np.array([half(ci[0] + 1), half(ci[1] + 1)])

    def sw(edge):
        gap = abs(edge - np.round(edge / CELL) * CELL)
        return SIDEWALK_W_AVE if gap > STREET_HALF + 1.0 else SIDEWALK_W

    kmin = pmin - np.array([sw(pmin[0]), sw(pmin[1])])
    kmax = pmax + np.array([sw(pmax[0]), sw(pmax[1])])
    c = 0.5 * (kmin + kmax)
    hs = 0.5 * (kmax - kmin)
    r = min(KERB_R, hs.min())
    qx = np.abs(x - c[0]) - hs[0] + r
    qy = np.abs(y - c[1]) - hs[1] + r
    d = (np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
         + np.minimum(np.maximum(qx, qy), 0) - r)

    w_road = d > GUTTER_W
    w_gutter = (d > 0) & ~w_road
    w_kerb = (d > -KERB_TOP_W) & (d <= 0)
    w_in = d <= -KERB_TOP_W
    joint_duty = 1 - (1 - JOINT_W / FLAG) ** 2
    rel_walk = REL_WALK * (1 - 0.6 * joint_duty)

    # Markings (city_road, city_ground).
    kx, ky = np.round(x / CELL), np.round(y / CELL)
    sx, sy = x - kx * CELL, y - ky * CELL
    ex, ey = np.abs(sx), np.abs(sy)
    hx, hy = half(kx), half(ky)
    ax, ay = is_avenue(kx), is_avenue(ky)
    swx = np.where(ax, SIDEWALK_W_AVE, SIDEWALK_W)
    swy = np.where(ay, SIDEWALK_W_AVE, SIDEWALK_W)
    on_x, on_y = ex < hx, ey < hy

    def band(v, a, w):
        return ((v >= a) & (v < a + w)).astype(float)

    def stripes(v, a, w, period):
        return (np.mod(v - a, period) < w).astype(float)

    white = np.zeros_like(x)
    yellow = np.zeros_like(x)
    m = on_x & (ey > hy)
    yellow += np.where(m & ax, band(ex, 0.10, 0.12), 0)
    white += np.where(m & ax, (band(ex, 3.05, 0.12) + band(ex, 6.15, 0.12))
                      * stripes(ey, 0.0, 3.0, 9.0)
                      + band(ex, hx - swx - PARK_W - 0.12, 0.12), 0)
    white += np.where(m, band(ey, hy + 1.2, 0.40), 0)
    m = on_y & (ex > hx)
    yellow += np.where(m & ay, band(ey, 0.10, 0.12), 0)
    white += np.where(m & ay, (band(ey, 3.05, 0.12) + band(ey, 6.15, 0.12))
                      * stripes(ex, 0.0, 3.0, 9.0)
                      + band(ey, hy - swy - PARK_W - 0.12, 0.12), 0)
    white += np.where(m, band(ex, hx + 1.2, 0.40), 0)
    white += np.where(on_x, band(ey, hy - swy + 0.3, swy - 0.1)
                      * stripes(sx, 0.25, 0.5, 1.0), 0)
    white += np.where(on_y, band(ex, hx - swx + 0.3, swx - 0.1)
                      * stripes(sy, 0.25, 0.5, 1.0), 0)
    white = np.minimum(white, 1) * PAINT_WEAR_MEAN
    yellow = np.minimum(yellow, 1) * PAINT_WEAR_MEAN
    road = (np.ones((3,) + x.shape)
            * (1 - white - yellow)
            + REL_PAINT * white
            + REL_PAINT * PAINT_YELLOW[:, None, None] * yellow)

    dry = (w_road * road
           + (w_gutter * REL_GUTTER + w_kerb * REL_KERB + w_in * rel_walk))

    film = w_road * WET_ROAD + w_gutter * 1.0 + (w_kerb | w_in) * WET_WALK
    px, py = x + noise_off[0], y + noise_off[1]
    n = 0.62 * vnoise(px * 0.38, py * 0.38) \
        + 0.38 * vnoise(px * 1.1 + 37.1, py * 1.1 + 37.1)
    region = vnoise(px * 0.025 + 5.3, py * 0.025 + 5.3) - 0.5
    crown = 1 - smoothstep(0, 2.5, np.minimum(np.where(on_x, ex, 1e3),
                                               np.where(on_y, ey, 1e3)))
    gutter = np.where(d > 0, 1 - smoothstep(0.10, 0.70, d), 0.0)
    park_side = lambda k: np.where(
        park_side_bits(k) == 1, 1.0, -1.0)
    lane_c = np.full_like(x, 1e3)
    m = on_x & (ey > hy)
    lane_c = np.where(m, np.where(ax, ex - 1.6 - np.round((ex - 1.6) / 3.1) * 3.1,
                                  sx + park_side(kx) * 1.2), lane_c)
    m = on_y & (ex > hx)
    lane_c = np.where(m, np.where(ay, ey - 1.6 - np.round((ey - 1.6) / 3.1) * 3.1,
                                  sy + park_side(ky) * 1.2), lane_c)
    rut = 1 - smoothstep(0.10, 0.30, np.abs(np.abs(lane_c) - 0.85))
    bias = (0.38 * gutter + 0.14 * rut * w_road
            - 0.12 * crown - 0.35 * (w_kerb | w_in) + 0.12 * region)
    puddle = smoothstep(PUDDLE_THR - PUDDLE_EDGE, PUDDLE_THR + PUDDLE_EDGE,
                        n + bias)
    alb = dry * (1 - WET_DARKEN * film * (1 - puddle)
                 - PUDDLE_DARKEN * puddle)
    inside = ((x > pmin[0]) & (x < pmax[0]) & (y > pmin[1]) & (y < pmax[1]))
    return alb.mean(axis=0), inside


def fit_a_over_1_plus_bs(s, m):
    """Least squares for m = A s / (1 + B s): linear in (A, B) as
    m = A s - B s m."""
    a = np.stack([s, -s * m], axis=1)
    (A, B), *_ = np.linalg.lstsq(a, m, rcond=None)
    return A, B


def main():
    rng = np.random.default_rng(7)
    step = 0.04
    g = (np.arange(int(CELL / step)) + 0.5) * step
    # Cell types by which of their low boundaries is an avenue: the high
    # boundary is one with the same odds, and a cell's mean is symmetric
    # under the swap. Weights are the block frequencies (avenue every 8).
    types = {(1, 1): 36, (0, 1): 12, (1, 0): 12, (0, 0): 4}
    street_acc = np.zeros_like(S_GRID)
    plot_acc = np.zeros_like(S_GRID)
    street_w = plot_w = 0.0
    for (ix, iy), weight in types.items():
        # Spacing and phase are per street, so each type is sampled over
        # several blocks (moving by whole avenue periods keeps the type).
        for blk in range(4):
            ci = (8 * (10 + 3 * blk) + ix * 3, 8 * (7 + 5 * blk) + iy * 3)
            x, y = np.meshgrid(ci[0] * CELL + g, ci[1] * CELL + g,
                               indexing="ij")
            pl = pools(x, y)
            for rep in range(2):
                off = rng.uniform(0, 5000, 2)
                alb, inside = ground(ci, x, y, off)
                for k, s in enumerate(S_GRID):
                    raw = s * pl
                    lamp = raw / (1 + POOL_KNEE * raw) * alb
                    street_acc[k] += weight * lamp[~inside].mean()
                    plot_acc[k] += weight * lamp[inside].mean()
                street_w += weight
                plot_w += weight
    street = street_acc / street_w
    plot = plot_acc / plot_w
    for name, m in (("CITY_GROUND_FAR_STREET", street),
                    ("CITY_GROUND_FAR_PLOT", plot)):
        A, B = fit_a_over_1_plus_bs(S_GRID, m)
        err = np.max(np.abs(A * S_GRID / (1 + B * S_GRID) - m) / m)
        print(f"{name}: vec2<f32>({A:.4f}, {B:.4f})   "
              f"max rel. fit error {err:.2%}")

    # Puddle coverage against the bias, over many noise positions.
    px = rng.uniform(0, 2e4, 2_000_000)
    py = rng.uniform(0, 2e4, 2_000_000)
    n = 0.62 * vnoise(px * 0.38, py * 0.38) \
        + 0.38 * vnoise(px * 1.1 + 37.1, py * 1.1 + 37.1)
    b = np.linspace(-0.4, 0.6, 51)
    cov = np.array([smoothstep(PUDDLE_THR - PUDDLE_EDGE,
                               PUDDLE_THR + PUDDLE_EDGE, n + bb).mean()
                    for bb in b])
    # The coverage is 1 / (1 + exp(-(b + 0.5 - x0) / s)): minimax over the
    # whole curve, because a least-squares fit in logit space weights the
    # tails, where the coverage is nothing, as heavily as the middle.
    best = (np.inf, 0.0, 0.0)
    for x0 in np.linspace(0.60, 0.75, 151):
        for s_fit in np.linspace(0.03, 0.12, 181):
            e = np.max(np.abs(1 / (1 + np.exp(-(b + 0.5 - x0) / s_fit))
                              - cov))
            if e < best[0]:
                best = (e, x0, s_fit)
    print(f"CITY_PUDDLE_FIT: vec2<f32>({best[1]:.4f}, {best[2]:.4f})   "
          f"max abs. coverage error {best[0]:.3f}")


if __name__ == "__main__":
    main()
