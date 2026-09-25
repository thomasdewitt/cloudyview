"""The desert's bounds pyramid: local upper bounds on the ground the trace
meets, built from the tile's own mip chain.

raymarch.wgsl desert_terrain_trace steps a ray over the desert surface

    ground(xy, fp) = datum + tile(xy, lod(fp)) + sum_k fade_k(fp) amp_k band(q_k)

(desert_surface_of with `traced`): the tile read trilinearly at the
footprint's mip, plus octaves that each read the tile's band,
band(q) = tile(q, 0) - tile(q, log2 5), at q_k = M^k xy, a similarity that
scales by 5^k. It needs to know, without reading the ground itself, where
the ground cannot be. A global bound (the tile's steepest slope anywhere)
made a ray skimming a few metres over the ground take metre steps, and the
192-step budget ran out a few hundred metres out: the ground ended short of
the horizon (2026-09-25). This pyramid makes the bounds local.

Every quantity is a max (or min) of texel values over an explicit window of
texels, or one IEEE operation on such maxima, so any algorithm that forms
the same windows gives the same bits; web/soar/desert_bounds.js builds it in
the browser, and tests/test_desert_bounds.py pins the two byte-equal on the
shipped tile.

GRID. Pyramid level k is the tile's level T = BASE_LEVEL + k: m = n >> T
cells of S_k = cell_m 2^T metres, k = 0 .. mips - 1 - BASE_LEVEL. Cell I of
level k covers level-l texels [I f, I f + f - 1], f = 2^(T - l), for l <= T,
and the one texel I >> (l - T) for l > T. Everything is periodic: indices
wrap at the tile's edge, as the repeat sampler does.

WINDOWS. win(Q_l, c) is the max of Q_l over the cell's level-l texels
widened by one texel each way (a point in texel i reads the bilinear
footprint of centres i - 1 .. i + 1), and with c = 1 over the cells I - 1 ..
I + 1 first (so a point may move up to S_k from anywhere in cell I and stay
inside the window). winmin likewise with min.

A bilinear read of level l at a point in the cell is a convex combination
of texels in win(h_l, 0), and a trilinear read a convex combination of two
levels, so max_l win(h_l, 0) bounds the tile at every footprint; a bilinear
patch between centres i .. i + 1 has |d/dx| <= max of its two x-differences
over s_l (likewise y), so P_l, the root sum of squares of those over s_l,
bounds its gradient, and a trilinear read's gradient is bounded by the
larger level's.

MIPS A LEVEL SERVES. The trace reads mip lod(fp) and lod(fp) + 1, and the
footprint grows along a ray; raymarch.wgsl steps against pyramid level k
only while lod <= T and never past twice its distance, so level k covers
the tile's mips l <= T + 1 and no coarser (coarse means over wide windows
would lift the bound over every valley).

THE BAND. band_w = tile(q, 0) - ((1 - w) tile(q, 2) + w tile(q, 3)), w the
mip weight of the octave's second read (the hardware's, which only the
hardware knows: every bound here holds for all w in [0, 1]). Its reads are
bilinear between the centre lines of level 0, 2 and 3, so between the union
of those lines (the LATTICE: per axis, (i + 1/2) cell_m 2^l for l in 0, 2,
3) band_w is one bilinear function per rectangle — a SUB-PATCH — and a
bilinear function on a rectangle takes its extremes, and the extremes of
each partial derivative, on the corners. And band_w is linear in w, so its
extremes over w are at w = 0 and 1. Hence, exactly, over a sub-patch:
  sup band      = max over corners of  tile0 - min(tile2, tile3)       BH
  sup |band|    = max over corners of  max(that, max(tile2, tile3) - tile0)
  sup |grad|   <= max over w in {0, 1} of the root sum of squares of the
                  largest |x-difference| / width and |y-difference| / height
                  of band_w across its edges                              GB
each value evaluated at the lattice point by the shader's bilinear formula
in float64. A sub-patch counts toward every cell it touches.

CHANNELS, as rgba16float in a two-layer 2D array (the WGSL binding 9 slot).
Layer 0 is the tile and layer 1 the octaves' band, so the simplified desert
(DESERT_LITE: no octaves) reads layer 0 alone.

  layer 0: the tile
    r  H    max_{l <= T+1} win(h_l, 1): its highest.
    g  G    max_{l <= T+1} win(P_l, 1): its steepest.
    b  E    max_{l <= T+1} win(h_l, 1) - min_{l <= T+1} winmin(h_l, 1): its
            range, which bounds how far a mip step can move it (the mip
            read grows with the footprint along a ray).
    a  H0   max_{l <= T+1} win(h_l, 0): its highest over the cell alone,
            for a step that stays inside the cell.
  layer 1: the band (sub-patches of cells I-1..I+1)
    r  BH   its highest, and never below 0: an octave's fade only falls
            along a ray, to 0 at the trace's cut, so a negative band rises
            to 0 within a step, and the octave's bound has to hold there.
    g  GB   its steepest.
    b  BA   its largest magnitude.
    a  G23  max(win(P_2, 1), win(P_3, 1)): the steepest of the band's
            mip-2/3 read alone (a filter's snapped position moves that read
            by a sixteenth of a metre where level 0's moves by a centimetre,
            so the two are bounded apart).

Every channel but H0 is a c = 1 window: a point anywhere in cell I may move up to
S_k and stay inside it. The top level's cell is the whole tile, so it holds
the global bounds.

Values are formed in float64 and stored as the smallest float16 at or above
them (rounded up, never to nearest), so every stored bound is at least the
exact one.
"""

import numpy as np

BASE_LEVEL = 1
LAYERS = 2
# The mips of the octave band (raymarch.wgsl DESERT_OCTAVE_LOD = log2 5 mixes
# levels 2 and 3).
BAND_MIPS = (2, 3)


def ceil_half(x):
    """The smallest float16 at or above each float64 in `x`; a value that has
    none (not finite, or past float16's largest) is an error."""
    x = np.asarray(x, np.float64)
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 65504.0):
        raise ValueError("desert bounds: a value outside float16's range")
    y = x.astype(np.float16)
    low = y.astype(np.float64) < x
    y[low] = np.nextafter(y[low], np.float16(np.inf))
    y[y == 0] = np.float16(0.0)           # no negative zeros
    return y


def dilate(a, reduce=np.maximum):
    """The 3 x 3 periodic max (or min) of a square array."""
    rows = reduce(reduce(np.roll(a, 1, 0), a), np.roll(a, -1, 0))
    return reduce(reduce(np.roll(rows, 1, 1), rows), np.roll(rows, -1, 1))


def to_grid(a, level, t, reduce=np.maximum):
    """A level-`level` array onto level t's grid: 2 x 2 pooled per level
    down, each texel repeated per level up."""
    for _ in range(level, t):
        a = reduce(reduce(a[0::2, 0::2], a[0::2, 1::2]),
                   reduce(a[1::2, 0::2], a[1::2, 1::2]))
    d = level - t
    if d > 0:
        a = np.repeat(np.repeat(a, 1 << d, 0), 1 << d, 1)
    return a


def windows(q, t, levels, reduce=np.maximum):
    """win(q_l, 0) on level t's grid, for each of `levels`.

    The WINDOWS of the module docstring, formed the cheap way (a max is
    exact, so any order of it gives the same bits): widening by one level-l
    texel is a 3 x 3 dilation on that level's grid, and the cell's texels
    are that pooled (or, for a coarser level, repeated) onto level t.
    Widening by the neighbouring cells, c = 1, is a 3 x 3 dilation on t's
    grid, and a max over levels commutes with it, so the caller takes it
    once after the max.
    """
    return {lv: to_grid(dilate(q[lv], reduce), lv, t, reduce) for lv in levels}


def patch_slopes(h, cell):
    """P: each bilinear patch's gradient bound (centres i..i+1, j..j+1)."""
    right = np.roll(h, -1, axis=1)
    down = np.roll(h, -1, axis=0)
    dx = np.abs(right - h)
    dy = np.abs(down - h)
    dx = np.maximum(dx, np.roll(dx, -1, axis=0))
    dy = np.maximum(dy, np.roll(dy, -1, axis=1))
    return np.sqrt(dx * dx + dy * dy) / cell


def lattice(n, cell_m):
    """The band's lattice on one axis: the centre lines of levels 0 and
    BAND_MIPS, sorted, within one tile period."""
    lines = set()
    for lv in (0,) + BAND_MIPS:
        s = cell_m * (1 << lv)
        lines.update((i + 0.5) * s for i in range(n >> lv))
    return np.array(sorted(lines))


def bilinear(a, x, y, texel):
    """The shader's bilinear read of periodic level `a` (texel m) at the
    points (x, y) — the formula desert_tile's sampler evaluates, in float64."""
    n = a.shape[0]
    gx = x / texel - 0.5
    gy = y / texel - 0.5
    i = np.floor(gx)
    j = np.floor(gy)
    fx = gx - i
    fy = gy - j
    i = i.astype(np.int64) % n
    j = j.astype(np.int64) % n
    i1 = (i + 1) % n
    j1 = (j + 1) % n
    return ((a[j, i] * (1.0 - fx) + a[j, i1] * fx) * (1.0 - fy)
            + (a[j1, i] * (1.0 - fx) + a[j1, i1] * fx) * fy)


def band_patches(h, cell_m):
    """BH, BA and GB on each sub-patch (see THE BAND), as arrays over the
    lattice: entry (b, a) is the sub-patch from lattice line a to a + 1 in x
    and b to b + 1 in y, the last wrapping to the first plus the period."""
    n = h[0].shape[0]
    xs = lattice(n, cell_m)
    X, Y = np.meshgrid(xs, xs)
    t0 = bilinear(h[0], X, Y, cell_m)
    b2, b3 = (bilinear(h[lv], X, Y, cell_m * (1 << lv)) for lv in BAND_MIPS)
    hi = t0 - np.minimum(b2, b3)
    mag = np.maximum(hi, np.maximum(b2, b3) - t0)

    def corners(v):
        return np.maximum(np.maximum(v, np.roll(v, -1, 1)),
                          np.maximum(np.roll(v, -1, 0), np.roll(v, -1, (0, 1))))

    widths = np.diff(np.append(xs, xs[0] + n * cell_m))
    slope = np.zeros_like(t0)
    for w in (b2, b3):
        band = t0 - w
        ddx = np.abs(np.roll(band, -1, 1) - band) / widths[None, :]
        ddy = np.abs(np.roll(band, -1, 0) - band) / widths[:, None]
        ddx = np.maximum(ddx, np.roll(ddx, -1, 0))
        ddy = np.maximum(ddy, np.roll(ddy, -1, 1))
        slope = np.maximum(slope, np.sqrt(ddx * ddx + ddy * ddy))
    return corners(hi), corners(mag), slope


def _pool_axis(v, xs, extent, m, axis):
    """Max of sub-patch values along one axis onto m cells: a sub-patch from
    xs[a] to the next line counts toward the cells of both its ends."""
    v = np.moveaxis(v, axis, -1)
    s = extent / m
    ends = np.append(xs[1:], xs[0] + extent)
    out = np.full(v.shape[:-1] + (m,), -np.inf)
    for edge in (xs, ends):
        cell = np.floor(edge / s).astype(np.int64)
        wrap = cell >= m                     # only the last sub-patch's end
        inside = ~wrap
        starts = np.flatnonzero(np.diff(cell[inside], prepend=-1))
        group = np.maximum.reduceat(v[..., inside], starts, axis=-1)
        out[..., cell[inside][starts]] = np.maximum(
            out[..., cell[inside][starts]], group)
        if wrap.any():
            out[..., 0] = np.maximum(out[..., 0], v[..., wrap].max(axis=-1))
    return np.moveaxis(out, -1, axis)


def pool_patches(v, xs, extent, m):
    """The max of sub-patch values over each of m x m cells, a sub-patch
    counting toward every cell its closed rectangle touches."""
    return _pool_axis(_pool_axis(v, xs, extent, m, 1), xs, extent, m, 0)


def build(levels, cell_m):
    """The pyramid from the tile's height mips.

    `levels` is the tile's height channel per mip, level 0 first (n x n,
    float16 or exactly-representable float64), `cell_m` its level-0 texel.
    Returns one float16 array per pyramid level, (LAYERS, m, m, 4).
    """
    h = [np.asarray(lv, np.float64) for lv in levels]
    mips = len(h)
    n = h[0].shape[0]
    for lv, a in enumerate(h):
        if a.shape != (max(1, n >> lv),) * 2:
            raise ValueError(f"tile level {lv} is {a.shape}, not "
                             f"{max(1, n >> lv)} square")
    if mips <= max(BAND_MIPS) or mips <= BASE_LEVEL:
        raise ValueError(f"the tile has {mips} mips; the pyramid needs the "
                         f"band's levels {BAND_MIPS} and its base "
                         f"{BASE_LEVEL}")
    p = [patch_slopes(a, cell_m * (1 << lv)) for lv, a in enumerate(h)]
    extent = n * cell_m
    xs = lattice(n, cell_m)
    bh, ba, gb = band_patches(h, cell_m)
    out = []
    for t in range(BASE_LEVEL, mips):
        served = range(min(t + 2, mips))              # mips l <= T + 1
        hi0 = windows(h, t, served)
        lo0 = windows(h, t, served, np.minimum)
        p0 = windows(p, t, sorted(set(served) | set(BAND_MIPS)))
        # a max over levels commutes with the dilation onto cells I-1..I+1
        top = np.max([hi0[lv] for lv in served], axis=0)
        low = np.min([lo0[lv] for lv in served], axis=0)
        m = top.shape[0]
        level = np.zeros((LAYERS, m, m, 4), np.float64)
        level[0, ..., 0] = dilate(top)
        level[0, ..., 1] = dilate(np.max([p0[lv] for lv in served], axis=0))
        level[0, ..., 2] = dilate(top) - dilate(low, np.minimum)
        level[0, ..., 3] = top
        level[1, ..., 0] = np.maximum(dilate(pool_patches(bh, xs, extent, m)), 0.0)
        level[1, ..., 1] = dilate(pool_patches(gb, xs, extent, m))
        level[1, ..., 2] = dilate(pool_patches(ba, xs, extent, m))
        level[1, ..., 3] = dilate(np.maximum(p0[BAND_MIPS[0]], p0[BAND_MIPS[1]]))
        out.append(ceil_half(level))
    return out
