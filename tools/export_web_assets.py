"""Export the browser build's binary assets.

One destination:

  web/soar/ocean/   the FIF ocean normal tile. Field-independent (it is a
                    periodic 100 m patch of sea surface, not anything about
                    your data), generated from a multifractal that only
                    exists in Python, so it ships WITH the tool and works
                    offline. Committed, and the seed is fixed, so a re-run
                    must reproduce the same bytes.

Cloud fields are not exported here. Each demo field is baked separately by
tools/prebake_demos.py into web/soar/demos/<id>/, which is what the browser
fetches; this module keeps the two array helpers that bake shares.

The WGSL under web/soar/ is edited in place. It used to be copied from the
desktop engine, which was the source; the desktop app is gone and the browser
copy is now the original.

    uv run python tools/export_web_assets.py
"""

import json
from pathlib import Path

import numpy as np

from cloudyview.cloudfield import CloudField
from cloudyview.ocean_fif import generate_fif_normals

REPO = Path(__file__).resolve().parents[1]
SOAR_OUT = REPO / "web" / "soar"
# The ocean tiles live in the package (cloudyview/soar/ocean); web/soar/ocean
# is a symlink to them. Write to the real location.
OCEAN_OUT = REPO / "cloudyview" / "soar" / "ocean"

# 512 rather than the app's 2048: a quarter the texel count for the same
# physical tile (dx scales to match), which keeps the download honest.
FIF_N = 512
FIF_DX_M = 0.2
FIF_SEED = 20260717


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
    for i, mip in enumerate(mips):
        (OCEAN_OUT / f"fif_mip{i}.bin").write_bytes(
            np.ascontiguousarray(mip, dtype=np.float16).tobytes()
        )
    meta = {
        "schema": "cloudyview.web.ocean.v1",
        "n": FIF_N,
        "dx_m": float(dx),
        "mips": len(mips),
        "format": "rgba16float",
        "tile_extent_m": float(FIF_N * dx),
        "seed": FIF_SEED,
    }
    (OCEAN_OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(mips)} mips, tile extent {meta['tile_extent_m']:.1f} m")
    return meta


def main() -> None:
    SOAR_OUT.mkdir(parents=True, exist_ok=True)
    export_ocean()


if __name__ == "__main__":
    main()
