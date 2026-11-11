"""Export the browser build's binary assets.

Destinations (name the ones to write):

  web/soar/ocean/   the FIF ocean normal tile. Field-independent (it is a
                    periodic 100 m patch of sea surface, not anything about
                    your data), generated from a multifractal that only
                    exists in Python, so it ships WITH the tool and works
                    offline. Committed, and the seed is fixed; the shipped
                    tile was baked under scaleinvariance 0.14.0, and a
                    re-run under any other version is refused unless
                    --rebake says a new sea is meant (another FIF may draw
                    another field from the same seed). A new tile also
                    needs ocean.wgsl's fif_slope_variance table, the
                    @domain bound on its normals and OCEAN_GLINT_ROUGHNESS
                    re-measured (tests/test_ocean_tile_bytes.py holds the
                    first two to the bytes).

Cloud fields are not exported here. Each demo field is baked separately by
tools/prebake_demos.py into web/soar/demos/<id>/, which is what the browser
fetches; this module keeps the two array helpers that bake shares.

The shader is not exported: its files are cloudyview/soar/shader/, which
web/soar/ reaches through a symlink.

    uv run python tools/export_web_assets.py ocean
"""

import gzip
import json
from pathlib import Path

import numpy as np

from cloudyview.field import CloudField
from cloudyview.soar.surfaces import (SURFACE_TILE_SCHEMAS,
                                      encode_surface_level,
                                      surface_level_files)
# The tile generators sit beside this file in tools/. They need
# scaleinvariance and scipy, which the installed package does not.
from city_fif import (
    DEFAULT_CELL_M as CITY_CELL_M,
    DEFAULT_N as CITY_N,
    generate_city_density,
)
from city_lots import bake_lots, glow_weight, pack_channels
from ocean_fif import generate_fif_normals

REPO = Path(__file__).resolve().parents[1]
SOAR_OUT = REPO / "web" / "soar"
# The ocean tiles live in the package (cloudyview/soar/ocean); web/soar/ocean
# is a symlink to them. Write to the real location.
OCEAN_OUT = REPO / "cloudyview" / "soar" / "ocean"
# The night-city density tile, same arrangement (web/soar/city symlinks it).
CITY_OUT = REPO / "cloudyview" / "soar" / "city"

CITY_SEED = 20260820
# The city's clock period (surfaces.surface_clock_period_s): where its
# animation — the manhole steam, components/manholesteam.wgsl — repeats
# exactly. The steam's noise rises 1.3 * 1.2 = 1.56 lattice cells a second
# and its lattice wraps at 4056 cells, so 2600 s is a whole number of every
# octave's cycles and the hosts' wrap of the clock there shows nothing.
CITY_CLOCK_PERIOD_S = 2600.0

# 1024 texels of 0.1 m: the 102.4 m tile, twice as fine as 512 of 0.2 m,
# so a low pass over the water still resolves wave
# texture a few hundred metres out. Stored byte-shuffled and gzipped
# (5.2 MiB on the wire, 10.7 MiB on the card).
FIF_N = 1024
FIF_DX_M = 0.1
FIF_SEED = 20260717
# The scaleinvariance release (PyPI) the shipped tile was baked with.
OCEAN_BAKED_WITH = "0.14.0"


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


def _build_fif_normal_mips(base: np.ndarray) -> list:
    """Average and renormalize a periodic normal-map mip chain on the CPU."""
    mips = [np.ascontiguousarray(base, dtype=np.float32)]
    cur = mips[0]
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h, w, _ = cur.shape
        nh = max(1, h // 2)
        nw = max(1, w // 2)
        y0 = (np.arange(nh) * 2) % h
        y1 = (y0 + 1) % h
        x0 = (np.arange(nw) * 2) % w
        x1 = (x0 + 1) % w
        down = (
            cur[y0[:, None], x0[None, :], :3]
            + cur[y1[:, None], x0[None, :], :3]
            + cur[y0[:, None], x1[None, :], :3]
            + cur[y1[:, None], x1[None, :], :3]
        ) * np.float32(0.25)
        length = np.linalg.norm(down, axis=-1, keepdims=True)
        down = down / np.maximum(length, np.float32(1e-12))
        level = np.empty((nh, nw, 4), dtype=np.float32)
        level[..., :3] = down
        level[..., 3] = 1.0
        mips.append(np.ascontiguousarray(level))
        cur = mips[-1]
    return mips


def remove_tile(out: Path) -> None:
    """Delete the tile in `out` by the level files its meta.json names: a
    level the old tile had and the new one does not would otherwise ship
    alongside it. A delete driven by names read from a file, so every name
    is checked before anything is removed: each must be a bare file name in
    `out` itself — no separator, no "." or "..", nothing absolute — or the
    whole delete is refused. The meta is left for the export to overwrite.
    """
    meta_path = out / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if "files" in meta and not isinstance(meta["files"], list):
        raise ValueError(f"{meta_path}: 'files' is not a list: "
                         f"{meta['files']!r}")
    names = surface_level_files(meta)
    bad = [n for n in names
           if not isinstance(n, str) or n in ("", ".", "..")
           or "/" in n or "\\" in n or n != Path(n).name]
    if bad:
        raise ValueError(f"{meta_path} names files that are not bare names "
                         f"in {out}: {bad!r}; nothing was deleted.")
    for name in names:
        (out / name).unlink(missing_ok=True)


def export_ocean() -> dict:
    OCEAN_OUT.mkdir(parents=True, exist_ok=True)
    print(f"FIF ocean tile {FIF_N}^2 at dx={FIF_DX_M} m (seeded {FIF_SEED})")
    nx, ny, nz, dx = generate_fif_normals(
        N=FIF_N, dx_m=FIF_DX_M, rng=np.random.default_rng(FIF_SEED)
    )
    base = np.empty((FIF_N, FIF_N, 4), dtype=np.float32)
    base[..., 0], base[..., 1], base[..., 2] = nx, ny, nz
    base[..., 3] = 1.0
    mips = _build_fif_normal_mips(base)
    remove_tile(OCEAN_OUT)
    files, wire = [], 0
    for i, mip in enumerate(mips):
        # Planar and byte-shuffled (planar-byte-shuffle): gzip then sees
        # each normal component's smoothness, and the constant alpha plane
        # costs nothing.
        packed = gzip.compress(encode_surface_level(mip), compresslevel=9,
                               mtime=0)
        files.append(f"mip{i}.bin.gz")
        (OCEAN_OUT / files[-1]).write_bytes(packed)
        wire += len(packed)
    meta = {
        "schema": SURFACE_TILE_SCHEMAS["ocean"],
        "n": FIF_N,
        "dx_m": float(dx),
        "mips": len(mips),
        "format": "rgba16float",
        "encoding": "planar-byte-shuffle",
        "compression": "gzip",
        "files": files,
        "tile_extent_m": float(FIF_N * dx),
        # The sea has no clock: nothing on it moves (surfaces.
        # surface_clock_period_s; null, not a period of 0).
        "clock_period_s": None,
        "seed": FIF_SEED,
    }
    (OCEAN_OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(mips)} mips, tile extent {meta['tile_extent_m']:.1f} m, "
          f"{wire / 2**20:.2f} MiB gzipped")
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
    # Lots, zoning and streets (tools/city_lots.py), baked from the
    # channels as the renderer will see them — float16, the stored precision
    # — so a lot boundary is decided on exactly the numbers city_cell reads.
    lot_word, street_word = bake_lots(
        height.astype(np.float16).astype(np.float32),
        rank.astype(np.float16).astype(np.float32),
    )[:2]
    # r is two things. At mip 0 it is the density city_cell raises buildings
    # from; from mip 1 up it is the GLOW, the density weighted by what its
    # land use lights (city_lots.GLOW_WEIGHT), so the fog, the sky dome and
    # the clouds' uplight follow the city that is actually drawn rather than
    # a fringe that is now dark. Every glow read is at lod >= 1.
    base = np.empty((CITY_N, CITY_N, 4), dtype=np.float32)
    base[..., 0] = height * glow_weight(street_word)
    base[..., 1] = rank
    base[..., 2] = 0.0
    base[..., 3] = 0.0
    mips = _build_city_mips(base)
    mips[0][..., 0] = height
    # b and a are packed integer words at mip 0 ONLY: every packed field is
    # at most 12 bits, stored offset to [-2048, 2047] where float16 is exact
    # (pack_channels asserts the round trip). Their coarse mips stay zero:
    # an average of bitfields means nothing, and nothing samples it — the
    # renderer reads b and a with textureLoad at mip 0 and filters only r.
    mips[0][..., 2], mips[0][..., 3] = pack_channels(lot_word, street_word)
    for i, mip in enumerate(mips):
        (CITY_OUT / f"fif_mip{i}.bin").write_bytes(
            np.ascontiguousarray(mip, dtype=np.float16).tobytes()
        )
    meta = {
        "schema": "cloudyview.web.city.v2",
        "n": CITY_N,
        "cell_m": float(CITY_CELL_M),
        "mips": len(mips),
        "format": "rgba16float",
        "tile_extent_m": float(CITY_N * CITY_CELL_M),
        "clock_period_s": CITY_CLOCK_PERIOD_S,
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
    parser.add_argument("--rebake", action="store_true",
                        help="write the ocean under a scaleinvariance other "
                             f"than the {OCEAN_BAKED_WITH} it was baked with: "
                             "a new sea, not a reproduction")
    args = parser.parse_args()
    if "ocean" in args.tiles and not args.rebake:
        import scaleinvariance
        if scaleinvariance.__version__ != OCEAN_BAKED_WITH:
            parser.error(
                f"the shipped ocean tile was baked with scaleinvariance "
                f"{OCEAN_BAKED_WITH}; this is {scaleinvariance.__version__}, "
                "whose FIF may draw another field from the same seed. Pass "
                "--rebake to write a new sea, then re-measure ocean.wgsl's "
                "fif_slope_variance and @domain bound and re-fit "
                "OCEAN_GLINT_ROUGHNESS.")
    SOAR_OUT.mkdir(parents=True, exist_ok=True)
    for name in args.tiles:
        EXPORTS[name]()


if __name__ == "__main__":
    main()
