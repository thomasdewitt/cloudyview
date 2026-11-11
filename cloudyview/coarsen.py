"""A field block-averaged onto a coarser grid, by the rule the demo bake uses.

Soar ships one demo, marine-congestus-coarse, that has no source file of its
own: tools/prebake_demos.py makes it from demo_marine-congestus.nc by
averaging the mixing ratios over 2x2x2 cells. The browser's reproduction
commands can only name the fine file, so the CLI has to make the same coarse
grid from it, or the terminal render is a different field in a different box
(`--coarsen`). The bake and the CLI share this module so there is one
definition of that grid, not two that agree today.

The coarse grid depends on WHERE the blocks start, not only on their size. z
is cut to the occupied band first (the planes that hold cloud, by witness's
fp16 rule) and the band is grown outward until the factor divides it; blocks
start at its bottom. Coarsening the file's whole column instead pairs the
planes with the other parity — a different field. After averaging, an edge
plane averaged with the empty plane the growth added can store as zero;
witness and behold cut the averaged field to its own occupied band as they
cut any field, and the bake makes the same cut before it ships.

The mixing ratios are averaged after the loader's value policy (negative
condensate clamped to zero), and x and y are averaged whole. The bake does
the same (tools/prebake_demos.py), so `--coarsen` remakes a coarsened demo
whose x/y crop is the whole file and whose z ascends in the file, as STEAM's
do; a spec that crops x or y, or a file written top-down, is not one the
flag can name.
"""

import numpy as np

from . import optical_depth


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Average (x, y, z) over f x f x f blocks, accumulating in float64.

    float64 for the sum even though the inputs and the output are float32:
    a block is only f**3 terms, but the terms are mixing ratios that span
    many orders of magnitude across a cascade field, and a float32 sum of
    eight of those loses the small one entirely. Cheap insurance at this size.

    Requires every axis to divide exactly. A partial edge block would be an
    average over fewer cells than its neighbours — a quietly different
    quantity at the domain edge, which then wraps against the opposite face
    in a periodic field. Better to refuse the crop.
    """
    if f == 1:
        return a
    if any(n % f for n in a.shape):
        raise ValueError(
            f"cannot block-average {a.shape} by {f}: every axis must divide "
            "exactly, or the edge blocks would average fewer cells than the "
            "interior and the periodic wrap would meet a seam")
    nx, ny, nz = (n // f for n in a.shape)
    out = np.empty((nx, ny, nz), dtype=np.float32)
    # Slabbed over x so the float64 view of the field never exists whole: the
    # marine case is 2048 x 2048 x 385 float32 (6.5 GB), and .astype(float64)
    # on all of it is 13 GB on top of the copy already in memory. 128 output
    # planes at a time is under a gigabyte and x is the slowest axis, so the
    # slabs are exactly the whole-array result.
    for o0 in range(0, nx, 128):
        o1 = min(o0 + 128, nx)
        block = a[o0 * f:o1 * f].astype(np.float64)
        out[o0:o1] = (block.reshape(o1 - o0, f, ny, f, nz, f)
                           .mean(axis=(1, 3, 5))
                           .astype(np.float32))
    return out


def axis_mean(a: np.ndarray, f: int) -> np.ndarray:
    """One coordinate axis under the same block average.

    The centre of a merged block is the mean of the centres it merges, which
    is exact for a uniform grid and the right answer for a stretched one too:
    the coarse cell spans the fine cells' union either way, and its centre is
    where the march should sample it. Already float64 — coordinates are read
    that way — so no accumulator question arises.
    """
    if f == 1:
        return a
    if a.size % f:
        raise ValueError(
            f"cannot block-average a {a.size}-point axis by {f}")
    return a.reshape(-1, f).mean(axis=1)


def grow_band(lo: int, hi: int, top: int, f: int) -> tuple:
    """The z band [lo, hi) grown until `f` divides it, out of [0, top).

    Outward from the occupied band, never inward, because trimming to fit
    would drop planes that hold cloud: up while there is room, then down.
    """
    while (hi - lo) % f:
        if hi < top:
            hi += 1
        elif lo > 0:
            lo -= 1
        else:
            raise ValueError(
                f"the whole {top}-plane column is not a multiple of {f}, so "
                "no z band can be")
    return lo, hi


def extinction_peaks(lwc: np.ndarray, z: np.ndarray, iwc=None) -> np.ndarray:
    """Per-plane maximum of the extinction a band is judged on (float64).

    What the renderers march, at witness's effective radii. Ice counts even
    where witness would drop it as negligible, as the bake never drops it.
    The bake calls this on x-slabs read from the file and folds them with
    np.maximum; occupied_band does the same in memory.
    """
    from .witness import RE_ICE_UM, RE_LIQUID_UM
    sigma = optical_depth.compute_extinction_field(
        lwc, np.asarray(z, dtype=np.float64), re=RE_LIQUID_UM, iwc=iwc,
        re_ice=RE_ICE_UM)
    return np.nanmax(sigma, axis=(0, 1)).astype(np.float64)


def band_of_peaks(peaks: np.ndarray, z: np.ndarray) -> tuple:
    """The occupied z planes [lo, hi) of a per-plane peak profile.

    crop_empty_z's decision on a 1x1xnz array: the same function witness
    crops with, so there is one definition of "occupied".
    """
    from .witness import crop_empty_z
    _, _, (lo, hi) = crop_empty_z(np.asarray(peaks).reshape(1, 1, -1),
                                  np.asarray(z, dtype=np.float64))
    return lo, hi + 1


def occupied_band(lwc: np.ndarray, z: np.ndarray, iwc=None,
                  slab: int = 256) -> tuple:
    """The z planes [lo, hi) that hold cloud, by witness's crop_empty_z rule.

    Over x-slabs, so a field the size of the marine case does not need a
    float64 extinction copy of itself.
    """
    peak = np.zeros(np.asarray(z).size, dtype=np.float64)
    for i0 in range(0, lwc.shape[0], slab):
        sl = slice(i0, i0 + slab)
        peak = np.maximum(peak, extinction_peaks(
            lwc[sl], z, None if iwc is None else iwc[sl]))
    return band_of_peaks(peak, z)


def coarsened(field, f: int):
    """`field` block-averaged by `f` on every axis, as the demo bake does it.

    z is cut to the occupied band grown to a multiple of `f` (see the module
    docstring); x and y must divide as they are, since nothing crops them.
    The mixing ratios are averaged, not the extinction: sigma is linear in
    each at fixed r_eff, but the ice fraction is a ratio of the two and is
    not.
    """
    from .field import CloudField

    if f < 1 or int(f) != f:
        raise ValueError(f"the coarsening factor must be a whole number "
                         f"of at least 1; got {f!r}")
    f = int(f)
    if f == 1:
        return field
    nx, ny, nz = field.lwc.shape
    for axis, n in (("x", nx), ("y", ny)):
        if n % f:
            raise ValueError(
                f"cannot coarsen by {f}: the {axis} axis has {n} cells, "
                f"which {f} does not divide")
    lo, hi = grow_band(*occupied_band(field.lwc, field.z, field.iwc),
                       nz, f)
    z = slice(lo, hi)
    return CloudField(
        lwc=block_mean(field.lwc[:, :, z], f),
        iwc=None if field.iwc is None else block_mean(field.iwc[:, :, z], f),
        x=axis_mean(np.asarray(field.x, np.float64), f),
        y=axis_mean(np.asarray(field.y, np.float64), f),
        z=axis_mean(np.asarray(field.z, np.float64)[z], f),
        source=field.source, ice_source=field.ice_source,
        liquid_var=field.liquid_var, ice_var=field.ice_var,
        liquid_group=field.liquid_group, ice_group=field.ice_group)
