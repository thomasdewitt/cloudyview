"""The moving sea's texture: shipped bytes in, the GPU's mip chain out.

The sea (cloudyview/ocean_fif.py) ships as one gzipped file of 8-bit planes,
[channel][t][y][x] — slope x, slope y, foam rank, crest — because float16
noise hardly compresses and the tile rides to every visitor. Each host
decodes it to float32, box-filters every frame's own spatial mip chain, and
uploads rgba16float into a 2D array texture, one layer per frame (binding 9,
raymarch.wgsl `ocean_waves`). Mips are spatial only: a 3D texture would
average neighbouring frames together at every level, which is not a coarser
view of the same instant.

This module is that recipe for Python (soar_host, the exporter and the
tests); web/soar/scene.js loadSea is the browser's copy of the same
arithmetic, all in float32, and tests/test_sea_tile.py holds the two to each
other.
"""
import gzip
import json
from pathlib import Path

import numpy as np

# The schema this layout is registered under is soar_host.SURFACE_TILE_SCHEMAS
# ["ocean"] (mirrored in web/soar/constants.js).
FILE = "sea.bin.gz"
CHANNELS = 4
# Slopes are stored signed about 128 over +-127 steps of their own scale;
# foam and crest are fractions over 255.
SLOPE_OFFSET = 128.0
SLOPE_STEPS = 127.0


def channel_decode(meta: dict) -> tuple:
    """Per-channel (offset, multiplier), the multiplier rounded to float32.

    Rounded once, from the double quotient, which is what Math.fround(a / b)
    does in the browser — so both hosts multiply by the same float32.
    """
    sx, sy = (float(v) for v in meta["slope_scale"])
    offsets = np.array([SLOPE_OFFSET, SLOPE_OFFSET, 0.0, 0.0], np.float32)
    mults = np.array([np.float32(sx / SLOPE_STEPS), np.float32(sy / SLOPE_STEPS),
                      np.float32(1.0 / 255.0), np.float32(1.0 / 255.0)],
                     np.float32)
    return offsets, mults


def encode(channels: dict, slope_scale) -> np.ndarray:
    """float32 channels (nt, n, n) -> uint8 planes (4, nt, n, n)."""
    sx, sy = (float(v) for v in slope_scale)

    def slope(a, s):
        return np.clip(np.round(a / s * SLOPE_STEPS) + SLOPE_OFFSET,
                       SLOPE_OFFSET - SLOPE_STEPS,
                       SLOPE_OFFSET + SLOPE_STEPS).astype(np.uint8)

    def frac(a):
        return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)

    return np.ascontiguousarray(np.stack([
        slope(channels["slope_x"], sx), slope(channels["slope_y"], sy),
        frac(channels["foam"]), frac(channels["crest"]),
    ]))


def decode_frame(planes: np.ndarray, t: int, meta: dict) -> np.ndarray:
    """Frame t of the uint8 planes as float32 (n, n, 4)."""
    offsets, mults = channel_decode(meta)
    frame = np.moveaxis(planes[:, t], 0, -1).astype(np.float32)
    return (frame - offsets) * mults


def mip_chain(level0: np.ndarray) -> list:
    """Box-filter one frame (n, n, 4) float32 down to 1x1, in float32.

    ((a + b) + c) + d, then * 0.25 — the order scene.js adds them in.
    """
    mips = [np.ascontiguousarray(level0, dtype=np.float32)]
    cur = mips[0]
    while cur.shape[0] > 1:
        down = (cur[0::2, 0::2] + cur[1::2, 0::2] + cur[0::2, 1::2]
                + cur[1::2, 1::2]) * np.float32(0.25)
        mips.append(np.ascontiguousarray(down, dtype=np.float32))
        cur = down
    return mips


def read(tile_dir: Path, schema: str) -> tuple:
    """(meta, uint8 planes (4, nt, n, n)) from a sea directory, checked
    against the schema the caller decodes."""
    tile_dir = Path(tile_dir)
    meta = json.loads((tile_dir / "meta.json").read_text())
    if meta.get("schema") != schema:
        raise RuntimeError(
            f"{tile_dir}/meta.json has schema {meta.get('schema')!r}; this "
            f"reader knows {schema!r}. Re-run tools/export_web_assets.py.")
    n, nt = int(meta["n"]), int(meta["frames"])
    raw = gzip.decompress((tile_dir / meta["file"]).read_bytes())
    expected = CHANNELS * nt * n * n
    if len(raw) != expected:
        raise RuntimeError(
            f"{tile_dir / meta['file']} unpacks to {len(raw)} bytes; "
            f"expected {expected} for {CHANNELS} planes of {nt}x{n}x{n}.")
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(CHANNELS, nt, n, n)
    return meta, planes


def foam_far_field(lace_cover, opacity_old: float, opacity_fresh: float,
                   tint_old) -> tuple:
    """The whitecaps' far-field means (raymarch.wgsl SEA_FOAM_FAR_*).

    Freshness f is uniform over the foam a gate keeps; the lace keeps
    lace_cover[k] at freshness 1 - k/4, linearly between (as the shader
    reads it), and opacity and tint run linearly from old (f = 0) to fresh
    (f = 1, white). Returns (mean of lace x opacity, the cover-weighted mean
    tint) — Simpson on a fine grid of an integrand that is piecewise cubic
    between the five points, so exact to rounding.
    """
    f = np.linspace(0.0, 1.0, 4 * 256 + 1)
    k = 4.0 * (1.0 - f)
    mask = np.interp(k, np.arange(5.0), np.asarray(lace_cover, np.float64))
    opacity = opacity_old + (opacity_fresh - opacity_old) * f
    tint = (np.asarray(tint_old, np.float64)[None, :] * (1.0 - f)[:, None]
            + f[:, None])
    w = np.ones_like(f)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w *= (f[1] - f[0]) / 3.0
    cover = float(np.sum(w * mask * opacity))
    tint_mean = (w * mask * opacity) @ tint / cover
    return cover, tuple(float(v) for v in tint_mean)
