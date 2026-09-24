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
                   tint_old, fresh_k: float) -> tuple:
    """The whitecaps' far-field means (raymarch.wgsl SEA_FOAM_FAR_*).

    Freshness f is uniform over the foam a gate keeps; the lace keeps
    lace_cover[k] at index k = fresh_k + (4 - fresh_k)(1 - f), linearly
    between the five measured points (as the shader reads it), and opacity and tint run linearly from old (f = 0) to fresh
    (f = 1, white). Returns (mean of lace x opacity, the cover-weighted mean
    tint) — Simpson on a fine grid of an integrand that is piecewise cubic
    between the five points, so exact to rounding.
    """
    f = np.linspace(0.0, 1.0, 4 * 256 + 1)
    k = fresh_k + (4.0 - fresh_k) * (1.0 - f)
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


def foam_resolved_cover(theta: float, lace_cover, opacity_old: float,
                        opacity_fresh: float, fresh_k: float,
                        edge: float) -> float:
    """What the resolved whitecaps average to at a gate cut theta, per unit
    of foam cap: the mean over the foam rank v in (0, 1] of
    smoothstep(theta - edge, theta + edge, v) times lace x opacity at the
    freshness clamp((v - theta) / (1 - theta), 0, 1) — the shader's near
    field, integrated. raymarch.wgsl carries it at five cuts near a closed
    gate (SEA_FOAM_TAIL_*), where the ramp is all there is.
    """
    v = np.linspace(0.0, 1.0, 400001)
    t = np.clip((v - (theta - edge)) / (2.0 * edge), 0.0, 1.0)
    cut = t * t * (3.0 - 2.0 * t)
    f = np.clip((v - theta) / max(1.0 - theta, 1e-9), 0.0, 1.0)
    k = fresh_k + (4.0 - fresh_k) * (1.0 - f)
    g = (np.interp(k, np.arange(5.0), np.asarray(lace_cover, np.float64))
         * (opacity_old + (opacity_fresh - opacity_old) * f))
    y = cut * g
    return float(np.sum(0.5 * (y[1:] + y[:-1])) * (v[1] - v[0]))


# --- the whitecaps' near field, mirrored -----------------------------------
#
# raymarch.wgsl sea_surface draws resolved whitecaps by cutting the foam rank
# as the pixel reads it — bilinear between 0.4 m texels, a B-spline closer
# in, a mip level farther out, scaled by the detail octave's crest — and the
# pixel's rank is diluted by all of those reads: a foam texel averaged with
# its empty neighbours falls under the cut and reads older than it is. The
# far field (foam_resolved_cover) is the undiluted texel mean, so without a
# correction the whitecaps thin out wherever the near field is drawn. What
# follows is the shader's near field in numpy, on the shipped bytes, and the
# rank gain per (cut, footprint) that gives it the far field's mean
# (raymarch.wgsl sea_foam_near_gain; measured by tools/export_web_assets.py
# into meta.json foam_near_gain).

# The shader constants the near field depends on, by their WGSL names;
# the exporter reads them from raymarch.wgsl and records them beside the
# table it measured with them.
FOAM_NEAR_CONSTANTS = (
    "SEA_FOAM_EDGE", "SEA_FOAM_BREAKUP", "SEA_FOAM_OPACITY_OLD",
    "SEA_FOAM_OPACITY_FRESH", "SEA_LACE_FRESH_K", "SEA_DETAIL_SCALE",
    "SEA_LACE_SCALE", "SEA_LACE_STRETCH")


def shader_scalars(src: str, names) -> dict:
    """`const NAME: f32 = value;` from WGSL source, for each name."""
    import re
    out = {}
    for name in names:
        m = re.search(rf"const {name}: f32 = ([0-9.eE+-]+);", src)
        if not m:
            raise ValueError(f"{name} is not an f32 const in the shader.")
        out[name] = float(m.group(1))
    return out


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear(img, uv):
    """A repeat-wrapped linear fetch at texel-centred uv, as the sampler."""
    side = img.shape[0]
    st = uv * side - 0.5
    i = np.floor(st).astype(np.int64)
    f = st - i
    x0, y0 = i[:, 0] % side, i[:, 1] % side
    x1, y1 = (x0 + 1) % side, (y0 + 1) % side
    return ((1 - f[:, 1]) * ((1 - f[:, 0]) * img[y0, x0] + f[:, 0] * img[y0, x1])
            + f[:, 1] * ((1 - f[:, 0]) * img[y1, x0] + f[:, 0] * img[y1, x1]))


def _trilinear(chain, uv, lod: float):
    l0 = int(np.floor(lod))
    w = lod - l0
    v = _bilinear(chain[l0], uv)
    if w == 0.0:
        return v
    return (1 - w) * v + w * _bilinear(chain[min(l0 + 1, len(chain) - 1)], uv)


def _bspline(img, uv):
    """sea_rank_smooth's cubic B-spline over level 0 (the 16 taps it
    reaches with its four bilinear fetches)."""
    n = img.shape[0]
    st = uv * n - 0.5
    i = np.floor(st).astype(np.int64)
    f = st - i

    def weights(f):
        return ((1 - f) ** 3 / 6, (4 - 6 * f * f + 3 * f ** 3) / 6,
                (1 + 3 * f + 3 * f * f - 3 * f ** 3) / 6, f ** 3 / 6)
    wx, wy = weights(f[:, 0]), weights(f[:, 1])
    out = 0.0
    for a in range(4):
        for b in range(4):
            out = out + wx[a] * wy[b] * img[(i[:, 1] + b - 1) % n,
                                            (i[:, 0] + a - 1) % n]
    return out


def _frame_blend(fetch, chains, uv, a, f):
    """mix(fetch(frame a), fetch(frame a + 1), f), per sample."""
    out = np.empty(len(uv))
    nt = len(chains)
    for t in np.unique(a):
        m = a == t
        out[m] = ((1 - f[m]) * fetch(chains[t], uv[m])
                  + f[m] * fetch(chains[(t + 1) % nt], uv[m]))
    return out


class NearFoam:
    """The shader's near-field whitecaps over the shipped tile.

    `consts` are the FOAM_NEAR_CONSTANTS values and `mip_bias` the ocean's
    (look.OCEAN_MIP_BIAS). Each channel's mips are the hosts' own
    (mip_chain, in float32).
    """

    def __init__(self, meta: dict, planes: np.ndarray, consts: dict,
                 mip_bias: float):
        self.meta, self.c, self.mip_bias = meta, consts, float(mip_bias)
        frames = [mip_chain(decode_frame(planes, t, meta))
                  for t in range(int(meta["frames"]))]
        self.foam = [[lv[..., 2] for lv in ch] for ch in frames]
        self.crest = [[lv[..., 3] for lv in ch] for ch in frames]
        self.dx = float(meta["dx_m"])
        self.extent = float(meta["tile_extent_m"])
        self.lace = np.asarray(meta["lace_cover"], np.float64)

    def samples(self, along_m: float, n: int, seed: int,
                chunk: int = 250_000) -> dict:
        """n pixels of footprint `along_m`, uniform over the tile and the
        loop; kept are only those with any foam rank (the rest are empty
        for every cut and gain). Per kept pixel: the rank the cut sees
        (before the gain), tile.z, and the lace's crest read."""
        c, dx = self.c, self.dx
        nt = len(self.foam)
        rng = np.random.default_rng(seed)
        lod = float(np.clip(np.log2(max(along_m / dx, 1.0)) + self.mip_bias,
                            0.0, int(self.meta["mips"]) - 1))
        smooth_w = 1.0 - float(_smoothstep(0.25, 0.75, along_m / dx))
        dtexel = dx / c["SEA_DETAIL_SCALE"]
        dlod = float(np.clip(np.log2(max(along_m / dtexel, 1.0))
                             + self.mip_bias, 0.0, 3.0))
        ltexel = dx / c["SEA_LACE_SCALE"]
        llod = float(np.log2(max(along_m / ltexel, 1.0)))
        lace_w = (float(np.clip(1.0 - 0.5 * llod, 0.0, 1.0))
                  * float(_smoothstep(0.25, 1.0, along_m
                                      / (ltexel * c["SEA_LACE_STRETCH"]))))
        keep = {"rank": [], "z": [], "lace_crest": []}
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            p = rng.uniform(0.0, self.extent, (m, 2))
            a = rng.integers(0, nt, m)
            f = rng.uniform(0.0, 1.0, m)
            da = rng.integers(0, nt, m)
            df = rng.uniform(0.0, 1.0, m)
            uv = p / self.extent
            z = _frame_blend(lambda ch, q: _trilinear(ch, q, lod),
                             self.foam, uv, a, f)
            on = z > 0.0
            p, uv, a, f, da, df, z = p[on], uv[on], a[on], f[on], \
                da[on], df[on], z[on]
            rank = z
            if smooth_w > 0.0:
                bs = _frame_blend(lambda ch, q: _bspline(ch[0], q),
                                  self.foam, uv, a, f)
                rank = (1 - smooth_w) * z + smooth_w * bs
            if dlod < 3.0:
                fine = _frame_blend(
                    lambda ch, q: _trilinear(ch, q, dlod), self.crest,
                    np.mod(uv * c["SEA_DETAIL_SCALE"], 1.0), da, df)
                crest = (fine - self.meta["crest_mean"]) * min(3.0 - dlod, 1.0)
                rank = rank * np.maximum(
                    1.0 + c["SEA_FOAM_BREAKUP"] * crest, 0.0)
            lace_crest = np.zeros(0)
            if lace_w > 0.0:
                luv = np.stack([p[:, 1], p[:, 0] / c["SEA_LACE_STRETCH"]], 1) \
                    * (c["SEA_LACE_SCALE"] / self.extent)
                lace_crest = _frame_blend(
                    lambda ch, q: _trilinear(ch, q, llod), self.crest,
                    np.mod(luv, 1.0), a, f)
            keep["rank"].append(rank)
            keep["z"].append(z)
            keep["lace_crest"].append(lace_crest)
        out = {k: np.concatenate(v) for k, v in keep.items()}
        out.update(n=n, lace_w=lace_w, lod=lod)
        return out

    def cover(self, s: dict, theta: float, gain: float = 1.0) -> float:
        """The near field's mean whitecap cover per unit of foam cap, at
        cut theta, with the rank (the cut's and freshness's) times gain."""
        c = self.c
        e = c["SEA_FOAM_EDGE"]
        z = s["z"] * gain
        fresh = np.clip((z - theta) / max(1.0 - theta, 1e-6), 0.0, 1.0)
        k = c["SEA_LACE_FRESH_K"] + (4.0 - c["SEA_LACE_FRESH_K"]) * (1.0 - fresh)
        lace = np.interp(k, np.arange(5.0), self.lace)
        if s["lace_w"] > 0.0:
            reach, ledge = self.meta["lace_reach"], self.meta["lace_edge"]
            cut = -ledge + 0.25 * k * reach
            lace = lace + (_smoothstep(cut - ledge, cut + ledge,
                                       s["lace_crest"]) - lace) * s["lace_w"]
        opacity = (c["SEA_FOAM_OPACITY_OLD"]
                   + (c["SEA_FOAM_OPACITY_FRESH"]
                      - c["SEA_FOAM_OPACITY_OLD"]) * fresh)
        near = _smoothstep(theta - e, theta + e, s["rank"] * gain) * lace * opacity
        return float(near.sum()) / s["n"] / self.meta["foam_max_coverage"]

    def gain(self, s: dict, theta: float, target: float) -> float:
        """The rank gain at which cover(s, theta) is `target` (bisection;
        the cover rises with the gain)."""
        lo, hi = 0.5, 4.0
        if not (self.cover(s, theta, lo) <= target <= self.cover(s, theta, hi)):
            raise RuntimeError(
                f"no rank gain in [{lo}, {hi}] gives the far field's cover "
                f"{target:.5f} at theta {theta}.")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.cover(s, theta, mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
