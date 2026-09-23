"""Export the browser build's binary assets.

Destinations (name the ones to write):

  web/soar/ocean/   the moving sea: a periodic 100 m, 25.6 s patch of sea
                    surface (cloudyview/ocean_fif.py). Field-independent,
                    generated from a multifractal that only exists in
                    Python, so it ships WITH the tool and works offline.
                    Committed, and the seed is fixed, so a re-run must
                    reproduce the same bytes.

Cloud fields are not exported here. Each demo field is baked separately by
tools/prebake_demos.py into web/soar/demos/<id>/, which is what the browser
fetches; this module keeps the two array helpers that bake shares.

The WGSL under web/soar/ is edited in place. It used to be copied from the
desktop engine, which was the source; the desktop app is gone and the browser
copy is now the original.

    uv run python tools/export_web_assets.py ocean
"""

import gzip
import json
from pathlib import Path

import numpy as np
import scaleinvariance

from cloudyview.cloudfield import CloudField
from cloudyview.city_fif import (
    DEFAULT_CELL_M as CITY_CELL_M,
    DEFAULT_N as CITY_N,
    generate_city_density,
)
from cloudyview import ocean_fif, sea_tile

REPO = Path(__file__).resolve().parents[1]
SOAR_OUT = REPO / "web" / "soar"
# The ocean tiles live in the package (cloudyview/soar/ocean); web/soar/ocean
# is a symlink to them. Write to the real location.
OCEAN_OUT = REPO / "cloudyview" / "soar" / "ocean"
# The night-city density tile, same arrangement (web/soar/city symlinks it).
CITY_OUT = REPO / "cloudyview" / "soar" / "city"

CITY_SEED = 20260820

SEA_SEED = 20260923
# Quantization range of the two slopes. The sea's crests run across the wind
# (+x), so its x slopes reach further; each is set just past its 99.999th
# percentile, and the clip fraction is printed on export.
SEA_SLOPE_SCALE = (0.75, 0.40)
# The surface frame folds at this many tiles: the swell's period.
SEA_SURFACE_TILES = 8


def _volume_aabb(field: CloudField):
    """Absolute-meter AABB with half-cell padding (matches witness())."""
    x = np.asarray(field.x, dtype=np.float64)
    y = np.asarray(field.y, dtype=np.float64)
    z = np.asarray(field.z, dtype=np.float64)
    dx_half = 0.5 * abs(x[1] - x[0])
    dy_half = 0.5 * abs(y[1] - y[0])
    dz_lo_half = 0.5 * abs(z[1] - z[0])
    dz_hi_half = 0.5 * abs(z[-1] - z[-2])
    bmin = np.array([x.min() - dx_half, y.min() - dy_half, z.min() - dz_lo_half])
    bmax = np.array([x.max() + dx_half, y.max() + dy_half, z.max() + dz_hi_half])
    return bmin, bmax


def _slope_variance_tables(planes: np.ndarray, meta: dict):
    """The two variance budgets the shader's microfacet lobe puts back.

    `removed[k]`: mean(sx^2 + sy^2) at level 0 minus the same at level k —
    what box-filtering to level k takes out of the slope distribution.
    `temporal[k]`: at level k, mean(s(t) . s(t)) minus mean(s(t) . s(t+1)).
    The shader blends two frames, and a blend at fraction f of two fields
    with variance v and covariance c has variance v - 2 f (1 - f) (v - c):
    the finest scales, which decorrelate fastest, dim between frames unless
    the lobe carries what the blend removed. Both are measured on the bytes
    the hosts decode, through the mip recipe they run.
    """
    nt = planes.shape[1]
    levels = None
    power = None
    lag = None
    first = sea_tile.mip_chain(sea_tile.decode_frame(planes, 0, meta))
    prev = first
    for t in range(nt):
        cur = prev if t == 0 else sea_tile.mip_chain(
            sea_tile.decode_frame(planes, t, meta))
        nxt = first if t == nt - 1 else sea_tile.mip_chain(
            sea_tile.decode_frame(planes, t + 1, meta))
        if levels is None:
            levels = len(cur)
            power = np.zeros(levels)
            lag = np.zeros(levels)
        for k in range(levels):
            a = cur[k][..., :2].astype(np.float64)
            b = nxt[k][..., :2].astype(np.float64)
            power[k] += np.mean(np.sum(a * a, axis=-1)) / nt
            lag[k] += np.mean(np.sum(a * b, axis=-1)) / nt
        prev = nxt
    removed = power[0] - power
    temporal = power - lag
    return levels, removed, temporal


def export_ocean() -> dict:
    """The moving sea: generate, quantize, and measure what the shader needs.

    The two variance tables and the loop length also live in raymarch.wgsl as
    constants (SEA_SLOPE_VARIANCE_REMOVED, SEA_SLOPE_VARIANCE_TEMPORAL,
    SEA_LOOP_S); tests/test_sea_tile.py fails until the shader is updated to
    match a re-export.
    """
    OCEAN_OUT.mkdir(parents=True, exist_ok=True)
    print(f"sea {ocean_fif.N}^2 x {ocean_fif.NT} frames (seeded {SEA_SEED})")
    sea = ocean_fif.generate_sea(SEA_SEED)
    planes = sea_tile.encode(sea, SEA_SLOPE_SCALE)
    clipped = [float(np.mean(np.abs(planes[c].astype(np.int16) - 128) == 127))
               for c in (0, 1)]
    print(f"  slope clip fraction x {clipped[0]:.2e}, y {clipped[1]:.2e}")
    for old in OCEAN_OUT.glob("fif_mip*.bin"):
        old.unlink()                   # the static tile this replaces
    payload = gzip.compress(planes.tobytes(), compresslevel=9, mtime=0)
    (OCEAN_OUT / sea_tile.FILE).write_bytes(payload)
    meta = {
        "schema": sea_tile.SCHEMA,
        "file": sea_tile.FILE,
        "encoding": "gzip, uint8 planes [slope x, slope y, foam, crest][t][y][x]",
        "n": ocean_fif.N,
        "frames": ocean_fif.NT,
        "dx_m": ocean_fif.DX_M,
        "dt_s": ocean_fif.DT_S,
        "loop_s": ocean_fif.NT * ocean_fif.DT_S,
        "tile_extent_m": ocean_fif.N * ocean_fif.DX_M,
        # The swell trains are periodic here, not at the tile; the surface
        # frame folds at this period (raymarch.wgsl SWELL_*).
        "surface_period_m": SEA_SURFACE_TILES * ocean_fif.N * ocean_fif.DX_M,
        "slope_scale": list(SEA_SLOPE_SCALE),
        "foam_max_coverage": ocean_fif.FOAM_MAX_COVERAGE,
        "seed": SEA_SEED,
        "scaleinvariance": scaleinvariance.__version__,
    }
    levels, removed, temporal = _slope_variance_tables(planes, meta)
    meta["mips"] = levels
    meta["slope_variance_removed"] = [round(float(v), 7) for v in removed]
    meta["slope_variance_temporal"] = [round(float(v), 7) for v in temporal]
    (OCEAN_OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(payload) / 1e6:.2f} MB gzipped, {levels} mips, loop "
          f"{meta['loop_s']:.1f} s, surface period {meta['surface_period_m']:.1f} m")
    print(f"  removed  {np.array2string(removed, precision=6)}")
    print(f"  temporal {np.array2string(temporal, precision=6)}")
    return meta


def _build_city_mips(base: np.ndarray) -> list:
    """Plain box-average mip chain — the channels are scalars, not normals.

    The coarse levels ARE the ground-glow field: mip k of the height channel
    is the mean building density over 2^k blocks, which is exactly the
    aggregate light a cloud base (or a distant sightline) integrates over.
    """
    mips = [np.ascontiguousarray(base, dtype=np.float32)]
    cur = mips[0]
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h, w, _ = cur.shape
        nh = max(1, h // 2)
        nw = max(1, w // 2)
        down = 0.25 * (
            cur[0:2 * nh:2, 0:2 * nw:2]
            + cur[1:2 * nh:2, 0:2 * nw:2]
            + cur[0:2 * nh:2, 1:2 * nw:2]
            + cur[1:2 * nh:2, 1:2 * nw:2]
        )
        mips.append(np.ascontiguousarray(down, dtype=np.float32))
        cur = mips[-1]
    return mips


def export_city() -> dict:
    CITY_OUT.mkdir(parents=True, exist_ok=True)
    print(f"city density tile {CITY_N}^2 at {CITY_CELL_M} m/block "
          f"(seeded {CITY_SEED})")
    height, rank = generate_city_density(
        rng=np.random.default_rng(CITY_SEED)
    )
    base = np.empty((CITY_N, CITY_N, 4), dtype=np.float32)
    base[..., 0] = height
    base[..., 1] = rank
    base[..., 2] = 0.0
    base[..., 3] = 1.0
    mips = _build_city_mips(base)
    for i, mip in enumerate(mips):
        (CITY_OUT / f"fif_mip{i}.bin").write_bytes(
            np.ascontiguousarray(mip, dtype=np.float16).tobytes()
        )
    meta = {
        "schema": "cloudyview.web.city.v1",
        "n": CITY_N,
        "cell_m": float(CITY_CELL_M),
        "mips": len(mips),
        "format": "rgba16float",
        "tile_extent_m": float(CITY_N * CITY_CELL_M),
        "seed": CITY_SEED,
    }
    (CITY_OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(mips)} mips, tile extent {meta['tile_extent_m'] / 1000:.1f} km")
    return meta


EXPORTS = {"ocean": export_ocean, "city": export_city}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Named, not all-by-default: each tile is committed, and re-exporting one
    # under a newer scaleinvariance than it was baked with is a change to
    # that scene, not a no-op (0.14's N-D normalization fix scaled every N-D
    # FIF by ~1.57).
    parser.add_argument("tiles", nargs="+", choices=sorted(EXPORTS))
    args = parser.parse_args()
    SOAR_OUT.mkdir(parents=True, exist_ok=True)
    for name in args.tiles:
        EXPORTS[name]()


if __name__ == "__main__":
    main()
