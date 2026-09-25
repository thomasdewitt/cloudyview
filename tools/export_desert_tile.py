"""Bake the Sonoran desert tile: an eroded bajada, exactly periodic.

    uv run python tools/export_desert_tile.py            # the shipped tile
    uv run python tools/export_desert_tile.py --n 512 --out /tmp/desert  # quick look

Writes cloudyview/soar/desert/ (web/soar/desert symlinks it): a mip chain of
rg16float texels and a meta.json. Each level ships as its two channel planes,
byte-shuffled (every texel's low bytes, then its high bytes) and gzipped —
the arrangement that lets gzip see the smoothness of a height field, which
interleaved half floats hide from it (13.6 MiB -> 5.7 for level 0). The
loaders undo it: cloudyview.soar_host.decode_surface_level and scene.js.
Channel r is the
ground's height in metres about its own mean; channel g is its LIGHTNESS, the
surface's age from varnished pavement (0) through soil (1/2) to fresh wash
sand (1). Albedo is piecewise linear in it (the knot at soil), so its box
means are the far field's mean albedo up to a per-level colour factor the
bake measures and ships (albedo_correction): a distant bajada is the colour
its resolved washes average to.

THE SLOPE, AND WHY THE TILE HAS NONE. A bajada falls a few percent away from
its mountain front, and the washes are what that fall organizes: they run
down it, split and rejoin across it. A periodic world cannot carry a net
slope — two percent over a hundred kilometres of flight is two kilometres of
climb into the clouds — so the regional slope enters only the FLOW LAW. The
domain is helical: row i drains into row i+1 and the last row into the first,
and every row-to-row height difference carries the ramp's s*dx. The stored
field is the detrended surface, exactly periodic, and the washes on it are
the ones a sloping bajada would cut. On the ground, one or two percent of
tilt is invisible to the eye; the organization it leaves behind is not.

HOW THE WASHES ARE CUT. Water is routed down the helical domain (each cell
shares its discharge among its three downslope neighbours in proportion to
slope^NEXP), losing a fixed fraction per metre to the bed — the transmission
loss that makes desert washes lose flow downstream, and the reason discharge
stays finite on a domain with no outlet. Where the discharge concentrates
beyond the sheetflow a uniform surface would carry, a wash is cut to the
hydraulic geometry that discharge asks for: depth saturating with
discharge, width growing as a power of it (WIDTH_EXPONENT), a flat
sandy floor and steep banks. The carved surface re-routes the water, which
deepens the washes that captured it, and the loop is iterated to
self-consistency.

WHAT MAKES THEM A BAJADA'S. A regional slope alone cuts a comb: every wash
parallel, one width, the same bright thread repeated (the 2026-09-23
review read them from 300 m as white contour lines). Three things break
it. The fans' cross-slope turns the fall across the tile, so washes
radiate over a fan and gather between fans. A wash's width wanders along
it, and past BRAID_WIDTH_M its floor braids around low gravel bars. And
lightness goes with size: a rill's floor is barely paler than its soil, a
braided bed is clean sand, and the active tract either side of a braided
bed never ages into pavement. Banks are drawn from the blurred footprint of
the routed centreline, whose cell staircase otherwise beads every bank.

Initial surface: a periodic FIF cascade for the ground's roughness and a
smooth fBm for its broad undulation — the "FIF is not enough" of the brief is
exactly why the washes are cut rather than read off the cascade.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
DESERT_OUT = REPO / "cloudyview" / "soar" / "desert"

SEED = 20260923
TILE_EXTENT_M = 4096.0

# The bajada.
REGIONAL_SLOPE = 0.02          # the fall the flow law sees, and nothing else does
LOSS_LENGTH_M = 400.0         # transmission-loss e-folding length
ROUTING_EXPONENT = 2.5         # discharge shared as slope^this
# A bajada is coalescing fans: across it the fall turns, radiating from each
# fan's apex and converging between fans. The flow law sees that as a
# cross-slope (the fall toward +x per metre) varying on the scale of a fan,
# zero-mean so the tile stays periodic, and — like the regional slope —
# stored nowhere in the heights.
FAN_CROSS_SLOPE = 0.018
FAN_SCALE_M = 700.0
ROUGHNESS_M = 0.6              # std of the FIF roughness
UNDULATION_M = 5.0             # std of the broad fBm undulation
# The washes.
CHANNEL_THRESHOLD = 1.2        # channelized beyond this many sheetflows
DEPTH_MAX_M = 2.0              # the deepest wash's banks
DEPTH_SCALE = 30.0             # sheetflows past threshold to reach ~0.76 DEPTH_MAX
# Two cells is the narrowest bed the 2 m tile draws with a smooth bank;
# the rills below it are the octaves' (raymarch.wgsl desert_surface).
WIDTH_MIN_M = 4.0
WIDTH_MAX_M = 24.0
# Width goes as a power of discharge. Leopold and Maddock's at-a-station
# 0.5 is for perennial rivers; ephemeral washes on a fan split rather than
# widen, and the references' threads stay a few metres wide for kilometres.
WIDTH_EXPONENT = 0.35
CARVE_ITERATIONS = 12
# Along a wash its width wanders: a bed widens where the banks are soft and
# pinches where they are not, by a lognormal factor of this spread,
# correlated over WIDTH_WANDER_M; a widening bed is shallower for it.
WIDTH_SPREAD = 0.45
WIDTH_WANDER_M = 40.0
PINCH_MIN = 0.6                # a pinch deepens a bed by at most 1/sqrt of this
# How soft a bank's contour is: the blur its footprint is drawn from.
BANK_SOFTNESS_M = 2.0
# A wide wash braids: low gravel bars, lens-shaped and drawn out along the
# flow, stand up out of its floor and split it into threads. They rise to
# this fraction of the wash's depth, in beds wider than BRAID_WIDTH_M.
BAR_RELIEF = 0.45
BAR_LENGTH_M = 40.0
BAR_BREADTH_M = 8.0
BRAID_WIDTH_M = (6.0, 14.0)
# The surface's age, as lightness (see generate).
SOIL_LIGHTNESS = 0.5
SOIL_PATCHINESS = 0.08
PAVEMENT_LIGHTNESS = 0.08
PAVEMENT_DRAINAGE_RADIUS_M = 40.0   # "the local drainage" a surface stands above
BAR_DARKENING = 0.25
# A wash's floor is as fresh as the wash is big: a rill's is barely paler
# than the soil it cuts, a braided wash's is clean sand (lightness 1).
RILL_LIGHTNESS = 0.62
FRESH_WIDTH_M = (3.0, 12.0)
# The active tract: the braided washes rework the ground either side of
# them, so no pavement forms there and the soil is younger and paler.
TRACT_M = 20.0
TRACT_GAIN = 1.5
TRACT_LIGHTENING = 0.08


def _round_up(v: float, digits: int = 3) -> float:
    """A bound, written to `digits` places without ever shrinking it."""
    q = 10 ** digits
    return float(np.ceil(v * q) / q)


# The shader's albedo endmembers (raymarch.wgsl DESERT_ALBEDO_*; the tile
# test holds the two to each other): pavement, soil, wash sand, at
# lightness 0, 1/2 and 1.
ALBEDO_PAVEMENT = (0.17, 0.145, 0.12)
ALBEDO_SOIL = (0.31, 0.235, 0.155)
ALBEDO_WASH = (0.33, 0.285, 0.22)


def albedo_of(lightness: np.ndarray) -> np.ndarray:
    """raymarch.wgsl desert_albedo: piecewise linear, the knot at soil."""
    l = np.clip(lightness, 0.0, 1.0)[..., None]
    p, s, w = (np.asarray(c) for c in (ALBEDO_PAVEMENT, ALBEDO_SOIL,
                                        ALBEDO_WASH))
    return np.where(l < 0.5, p + (s - p) * 2.0 * l,
                    s + (w - s) * (2.0 * l - 1.0))


def albedo_correction(lights: list) -> list:
    """Per mip level, the RGB factor that restores the far field's mean.

    Lightness is box-averaged exactly, but albedo is piecewise linear in it,
    and a footprint that mixes wash sand with pavement straddles the knot:
    the albedo OF its mean lightness is redder and duller than the MEAN of
    its albedos (by ~10% in blue at the top of the chain). The factor is the
    tile-wide ratio of the two at each level — the far field's average
    brought back to the near field's, which is the rule that keeps a seam
    from appearing where detail drops out.
    """
    truth = albedo_of(lights[0]).reshape(-1, 3).mean(0)
    return [[round(float(v), 4)
             for v in truth / albedo_of(l).reshape(-1, 3).mean(0)]
            for l in lights]


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _disc(radius_cells: float) -> np.ndarray:
    r = max(radius_cells, 0.5)
    k = int(np.ceil(r))
    y, x = np.mgrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-6


def route(h: np.ndarray, dx: float, cross: np.ndarray,
          sweeps: int = 2) -> np.ndarray:
    """Discharge per cell, in units of one cell's rain.

    Rows drain i -> i+1 (mod n), each cell to its three neighbours in the
    next row. The regional slope enters every drop as s*dx, and the fans'
    cross-slope `cross` every diagonal one as +-cross*dx; nothing else about
    the domain knows either exists. Two sweeps: the first carries the flow
    that wraps round from the last row, the second is the answer.
    """
    n = h.shape[0]
    loss = dx / LOSS_LENGTH_M
    diag = np.sqrt(2.0)
    discharge = np.zeros_like(h)
    q_in = np.zeros(n)
    for _ in range(sweeps):
        for i in range(n):
            drop = h[i] + REGIONAL_SLOPE * dx
            hn = h[(i + 1) % n]
            s0 = (drop - hn) / dx
            sl = (drop - cross[i] * dx - np.roll(hn, 1)) / (dx * diag)
            sr = (drop + cross[i] * dx - np.roll(hn, -1)) / (dx * diag)
            q = q_in + 1.0
            discharge[i] = q
            wl = np.maximum(sl, 0.0) ** ROUTING_EXPONENT
            w0 = np.maximum(s0, 0.0) ** ROUTING_EXPONENT
            wr = np.maximum(sr, 0.0) ** ROUTING_EXPONENT
            ws = wl + w0 + wr
            # A cell with no downslope neighbour (a pit) drains straight on.
            pit = ws <= 0.0
            w0 = np.where(pit, 1.0, w0)
            ws = np.where(pit, 1.0, ws)
            q_in = (1.0 - loss) * (np.roll(q * wl / ws, -1) + q * w0 / ws
                                   + np.roll(q * wr / ws, 1))
    return discharge


def carve(discharge: np.ndarray, dx: float, wander: np.ndarray,
          bars: np.ndarray):
    """Wash depth (m) from discharge by hydraulic geometry, and the width
    (m) of the wash each cell lies in (0 outside one).

    Channelization is discharge over the sheetflow a uniform surface would
    carry (1 / loss per cell). Depth saturates with it; width goes as its
    WIDTH_EXPONENT power, times the lognormal `wander` (a wider stretch of
    the same wash is shallower, holding its cross-section). Each width band
    is laid down by a disc-shaped grey dilation of its own depth, which is
    what gives a wash a flat floor and banks; `bars` (0..1) then raise the
    braid bars out of the floors of the wide ones.
    """
    sheetflow = LOSS_LENGTH_M / dx
    ch = discharge / sheetflow
    depth = np.where(ch > CHANNEL_THRESHOLD,
                     DEPTH_MAX_M * np.tanh((ch - CHANNEL_THRESHOLD) / DEPTH_SCALE)
                     / np.sqrt(np.maximum(wander, PINCH_MIN)), 0.0)
    width = np.clip(WIDTH_MIN_M * (np.maximum(ch, CHANNEL_THRESHOLD)
                                   / CHANNEL_THRESHOLD) ** WIDTH_EXPONENT
                    * wander, WIDTH_MIN_M, WIDTH_MAX_M)
    bands = [4.0, 8.0, 16.0, 32.0]
    edges = [0.0] + [0.5 * (a + b) for a, b in zip(bands[:-1], bands[1:])] + [1e9]
    out = np.zeros_like(discharge)
    wide = np.zeros_like(discharge)
    soft = BANK_SOFTNESS_M / dx
    for wb, lo, hi in zip(bands, edges[:-1], edges[1:]):
        sel = (width >= lo) & (width < hi) & (depth > 0.0)
        if not sel.any():
            continue
        disc = _disc(0.5 * wb / dx)
        inside = ndimage.grey_dilation(sel.astype(np.float64), footprint=disc,
                                       mode="wrap")
        # The bank is the half-way contour of the BLURRED footprint: the
        # routed centreline is a staircase of cells, and the union of discs
        # along it is a string of beads — scallops a cell apart that a low
        # sun lit as a ladder along every bank. A blurred staircase's
        # contours are smooth.
        bank = _smoothstep(0.25, 0.75, ndimage.gaussian_filter(
            inside, soft, mode="wrap"))
        # The floor's depth, smoothed inside the footprint only (a
        # normalized convolution), so it does not sag toward the banks.
        wet = ndimage.gaussian_filter(inside, 2.0 * soft, mode="wrap")
        d = ndimage.gaussian_filter(ndimage.grey_dilation(
            np.where(sel, depth, 0.0), footprint=disc, mode="wrap"),
            2.0 * soft, mode="wrap") / np.maximum(wet, 1e-6)
        w = ndimage.grey_dilation(np.where(sel, width, 0.0), footprint=disc,
                                  mode="wrap")
        out = np.maximum(out, bank * d)
        wide = np.maximum(wide, bank * w)
    out = out * (1.0 - BAR_RELIEF * bars * _smoothstep(*BRAID_WIDTH_M, wide))
    # Banks are steep, not vertical — and not soft: the smoothstep on the
    # blurred footprint puts a bank across about one cell. A plain blur of
    # the depth wide enough to hide the staircase (0.6 cell was tried)
    # turned the cut banks into gentle swales that read as painted lines.
    return out, wide


def _smooth_noise(rng, n: int, sigma_cells) -> np.ndarray:
    """Periodic gaussian-filtered white noise, zero mean, unit std."""
    f = ndimage.gaussian_filter(rng.standard_normal((n, n)), sigma_cells,
                                mode="wrap")
    return (f - f.mean()) / f.std()


def generate(n: int, seed: int = SEED, verbose: bool = True):
    """(height, lightness), each (n, n) float64, exactly periodic."""
    import scaleinvariance as si
    from scaleinvariance.simulation.FIF import extremal_levy

    dx = TILE_EXTENT_M / n
    rng = np.random.default_rng(seed)
    noise = extremal_levy(1.8, size=n * n, seed=int(rng.integers(2**31)))
    rough = si.FIF_ND((n, n), alpha=1.8, C1=0.05, H=0.7, periodic=True,
                      levy_noise=noise.reshape(n, n)).astype(np.float64)
    np.random.seed(int(rng.integers(2**31)))
    und = si.fBm_ND_circulant((n, n), H=0.95, periodic=True).astype(np.float64)
    base = (ROUGHNESS_M * (rough - rough.mean()) / rough.std()
            + UNDULATION_M * (und - und.mean()) / und.std())

    cross = FAN_CROSS_SLOPE * _smooth_noise(rng, n, FAN_SCALE_M / dx)
    wander = np.exp(WIDTH_SPREAD * _smooth_noise(rng, n, WIDTH_WANDER_M / dx))
    # Rows are the flow direction (axis 0), so the bars are long in it.
    bars = _smoothstep(0.0, 1.5, _smooth_noise(
        rng, n, (0.5 * BAR_LENGTH_M / dx, 0.5 * BAR_BREADTH_M / dx)))
    h = base.copy()
    depth = np.zeros_like(h)
    for k in range(CARVE_ITERATIONS):
        new_depth, wide = carve(route(h, dx, cross), dx, wander, bars)
        depth = 0.5 * depth + 0.5 * new_depth
        h = base - depth
        if verbose:
            print(f"  carve {k}: channelized {(depth > 0.1).mean():.3f} of the "
                  f"ground, deepest {depth.max():.2f} m", flush=True)

    # Lightness, by the age of the surface. The active wash floors are fresh
    # sand (1). The interfluves are tan soil, patchy on a cascade of its own.
    # The oldest surfaces — standing highest above the local drainage and
    # farthest from a wash — have had the time to armour into varnished
    # pavement (0), and they do so in patches, not everywhere at once.
    np.random.seed(int(rng.integers(2**31)))
    patch = si.fBm_ND_circulant((n, n), H=0.4, periodic=True).astype(np.float64)
    patch = (patch - patch.mean()) / patch.std()
    # Sand is the wash FLOOR: the blurred shoulder of a bank is not bed.
    wash = _smoothstep(0.12, 0.3, depth / DEPTH_MAX_M)
    dist = ndimage.distance_transform_edt(
        np.tile(wash < 0.5, (3, 3)))[n:2 * n, n:2 * n] * dx
    ring = _disc(PAVEMENT_DRAINAGE_RADIUS_M / dx)
    above = h - ndimage.minimum_filter(h, footprint=ring, mode="wrap")
    old = (_smoothstep(0.15, 0.7, above)
           * _smoothstep(6.0, 30.0, dist)
           * _smoothstep(-0.9, 0.1, patch))
    braided = _smoothstep(*BRAID_WIDTH_M, wide)
    tract = np.clip(TRACT_GAIN * ndimage.gaussian_filter(
        braided, TRACT_M / dx, mode="wrap"), 0.0, 1.0)
    old = old * (1.0 - tract)
    soil = (SOIL_LIGHTNESS + SOIL_PATCHINESS * np.tanh(patch)
            + TRACT_LIGHTENING * tract)
    ground = soil + (PAVEMENT_LIGHTNESS - soil) * old
    fresh = RILL_LIGHTNESS + (1.0 - RILL_LIGHTNESS) * _smoothstep(
        *FRESH_WIDTH_M, wide)
    # The braid bars are the coarser, darker gravel a flood leaves.
    bar_gravel = BAR_DARKENING * bars * braided
    light = wash * fresh * (1.0 - bar_gravel) + (1.0 - wash) * ground
    # 1/64 steps: far finer than any albedo difference an eye resolves, and
    # coarse enough that the channel compresses to almost nothing.
    return h - h.mean(), np.round(light * 64.0) / 64.0


def box_mips(level0: np.ndarray) -> list:
    """Exact periodic 2x2 box means, down to 1x1."""
    mips = [level0]
    cur = level0
    while cur.shape[0] > 1:
        cur = 0.25 * (cur[0::2, 0::2] + cur[1::2, 0::2]
                      + cur[0::2, 1::2] + cur[1::2, 1::2])
        mips.append(cur)
    return mips


def _bilinear_periodic(level: np.ndarray, n0: int) -> np.ndarray:
    """A mip level read at every level-0 texel centre, GPU-style (periodic)."""
    side = level.shape[0]
    c = (np.arange(n0) + 0.5) * side / n0 - 0.5
    i0 = np.floor(c).astype(int)
    f = c - i0
    a, b = i0 % side, (i0 + 1) % side
    rows = level[a] * (1 - f)[:, None] + level[b] * f[:, None]
    return rows[:, a] * (1 - f)[None, :] + rows[:, b] * f[None, :]


def relief_stats(heights: list) -> dict:
    """What the shader's slab tests and clearance stepping are built from.

    slope_max: the steepest one-texel slope at each mip level, a quarter
    again for safety (raymarch.wgsl desert_slope_max_at copies it).
    band_max_m and band_slope_max: the largest excursion and slope of the
    band an octave adds — the tile minus itself read log2(5) levels up, trilinear,
    exactly as desert_surface computes it.
    """
    n = heights[0].shape[0]
    cell = TILE_EXTENT_M / n
    slope_max = []
    for k, h in enumerate(heights):
        # The bilinear cell's gradient is (x-difference on one of its two
        # rows, y-difference on one of its two columns), so its maximum is
        # over all four pairings, not only the pair at one corner.
        dx = cell * 2 ** k
        gx0 = (np.roll(h, -1, 1) - h) / dx
        gx1 = np.roll(gx0, -1, 0)
        gy0 = (np.roll(h, -1, 0) - h) / dx
        gy1 = np.roll(gy0, -1, 1)
        worst = max(float(np.hypot(a, b).max())
                    for a in (gx0, gx1) for b in (gy0, gy1))
        slope_max.append(_round_up(1.25 * worst))
    lod = np.log2(5.0)
    lo = int(np.floor(lod))
    f = lod - lo
    coarse = ((1 - f) * _bilinear_periodic(heights[lo], n)
              + f * _bilinear_periodic(heights[lo + 1], n))
    band = heights[0] - coarse
    bg = np.hypot((np.roll(band, -1, 1) - band) / cell,
                  (np.roll(band, -1, 0) - band) / cell)
    # envelope_m: how far the full surface can stand ABOVE mip k read at the
    # same point — the lid the trace can step against from far above, with
    # mip k's gentle slope bound in place of level 0's (a tenth again for
    # the bilinear corners between texel centres).
    envelope = [_round_up(1.1 * float((heights[0]
                                       - _bilinear_periodic(h, n)).max()))
                for h in heights]
    # slope_variance_removed: mean squared slope (central differences) at
    # level 0 minus the same at level k — what the far field's filtered
    # normal no longer carries, and what the shader puts back as shadowing
    # on the direct beam (raymarch.wgsl desert_slope_variance_removed_at).
    msq = []
    for k, h in enumerate(heights):
        dx = cell * 2 ** k
        gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * dx)
        gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * dx)
        msq.append(float((gx * gx + gy * gy).mean()))
    removed = [round(msq[0] - v, 6) for v in msq]
    return {"slope_max": slope_max,
            "envelope_m": envelope,
            "slope_variance_removed": removed,
            "band_max_m": round(float(np.abs(band).max()), 4),
            "band_slope_max": _round_up(1.25 * float(bg.max()))}


def encode_level(level: np.ndarray) -> bytes:
    """(side, side, channels) -> planar, byte-shuffled float16 bytes."""
    planes = []
    for c in range(level.shape[-1]):
        b = np.ascontiguousarray(level[..., c], dtype=np.float16).view(np.uint8)
        b = b.reshape(-1, 2)
        planes += [b[:, 0].tobytes(), b[:, 1].tobytes()]
    return b"".join(planes)


def export(n: int, out: Path, seed: int = SEED) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    print(f"Desert tile {n}^2 over {TILE_EXTENT_M:g} m (seed {seed})")
    height, light = generate(n, seed)
    rg = np.stack([height, light], axis=-1)
    mips = box_mips(rg)
    for stale in list(out.glob("fif_mip*.bin*")) + list(out.glob("mip*.bin*")):
        stale.unlink()
    wire = 0
    files = []
    for i, level in enumerate(mips):
        packed = gzip.compress(encode_level(level), compresslevel=9, mtime=0)
        files.append(f"mip{i}.bin.gz")
        (out / files[-1]).write_bytes(packed)
        wire += len(packed)
    heights = [m[..., 0].astype(np.float16).astype(np.float64) for m in mips]
    lights = [m[..., 1].astype(np.float16).astype(np.float64) for m in mips]
    h16 = heights[0]
    meta = {
        "schema": "cloudyview.web.desert.v2",
        "n": n,
        "cell_m": TILE_EXTENT_M / n,
        "mips": len(mips),
        "format": "rg16float",
        "encoding": "planar-byte-shuffle",
        "compression": "gzip",
        "files": files,
        "tile_extent_m": TILE_EXTENT_M,
        "height_min_m": float(h16.min()),
        "height_max_m": float(h16.max()),
        "seed": seed,
        # From the stored (half-precision) top mip, which is what a reader
        # of the tile sees.
        "mean_lightness": round(float(np.float16(mips[-1][0, 0, 1])), 4),
        **relief_stats(heights),
        "albedo_correction": albedo_correction(lights),
        # The regional albedo the far field and the clouds' surface bounce
        # see: the mean of the level-0 albedos (not the albedo of the mean
        # lightness — see albedo_correction).
        "bounce_albedo": [round(float(v), 3) for v in
                          albedo_of(lights[0]).reshape(-1, 3).mean(0)],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(mips)} mips, heights {meta['height_min_m']:.2f} to "
          f"{meta['height_max_m']:.2f} m, {wire / 2**20:.1f} MiB gzipped")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=DESERT_OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    export(args.n, args.out, args.seed)


if __name__ == "__main__":
    main()
