"""Export the browser build's binary assets.

Two destinations, for two different reasons:

  web/soar/ocean/   the FIF ocean normal tile. Field-independent (it is a
                    periodic 100 m patch of sea surface, not anything about
                    your data), generated from a multifractal that only
                    exists in Python, so it ships WITH the tool and works
                    offline.
  web/demo/         the demo cloud field. Fetched from the cloudyview repo
                    at run time rather than copied to the website, so the
                    deployed folder stays small.

Also copies raymarch.wgsl into web/soar/ — the shader is the shared artifact
and the browser must run it verbatim, so it is never hand-edited there.

    uv run --extra interactive python tools/export_web_assets.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

from cloudyview.cloudfield import load
from cloudyview.ocean_fif import generate_fif_normals
from cloudyview.soar import engine as soar_engine
from cloudyview.soar.engine import _build_fif_normal_mips, _ghost_face_arrays

REPO = Path(__file__).resolve().parents[1]
DEMO_NC = REPO / "data" / "TWPICE_subvolume_256x256_5km.nc"
SOAR_OUT = REPO / "web" / "soar"
OCEAN_OUT = SOAR_OUT / "ocean"
DEMO_OUT = REPO / "web" / "demo"

# 512 rather than the app's 2048: a quarter the texel count for the same
# physical tile (dx scales to match), which keeps the download honest.
FIF_N = 512
FIF_DX_M = 0.2
FIF_SEED = 20260717

DEMO_SUN = {"azimuth": 235.0, "elevation": 25.0}


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


def export_demo() -> None:
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    field = load(str(DEMO_NC))
    print(f"demo field: {field.lwc.shape} from {DEMO_NC.name}")

    sigma = soar_engine.optical_depth.compute_extinction_field(
        field.lwc, field.z, re=10.0, iwc=field.iwc, re_ice=30.0
    )
    sigma = np.ascontiguousarray(sigma, dtype=np.float16)
    nx, ny, nz = sigma.shape

    # Ghost-padded, but with a ZERO border: the browser writes the periodic
    # faces itself so that toggling periodic off and on again does not need
    # another download. Original voxel i lands on padded texel i+1, which is
    # what makes the shader's (g + 1.5) / (N + 2) mapping exact.
    padded = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float16)
    padded[1:-1, 1:-1, 1:-1] = sigma
    (DEMO_OUT / "volume.bin").write_bytes(padded.tobytes())
    print(f"  volume.bin: {padded.nbytes / 1e6:.1f} MB {padded.shape}")

    faces = _ghost_face_arrays(sigma)
    blob = b"".join(
        np.ascontiguousarray(faces[name], dtype=np.float16).tobytes()
        for name in ("x_lo", "x_hi", "y_lo", "y_hi")
    )
    (DEMO_OUT / "faces.bin").write_bytes(blob)
    print(f"  faces.bin: {len(blob) / 1e6:.2f} MB")

    bmin, bmax = soar_engine._volume_aabb(field)
    meta = {
        "schema": "cloudyview.web.demo.v2",
        "source": DEMO_NC.name,
        "title": "TWP-ICE, Darwin 2006",
        "description": (
            "A large-eddy simulation of tropical convection near Darwin, "
            "Australia, from the TWP-ICE field campaign."
        ),
        "volume": {
            "shape_xyz": [int(nx), int(ny), int(nz)],
            "padded_dims_xyz": [nx + 2, ny + 2, nz + 2],
            "format": "r16float",
            "bmin": [float(v) for v in bmin],
            "bmax": [float(v) for v in bmax],
        },
        # Face planes, in file order, as (rows, cols) of r16float.
        "faces": {
            "order": ["x_lo", "x_hi", "y_lo", "y_hi"],
            "x_shape": [ny + 2, nz + 2],
            "y_shape": [nx + 2, nz + 2],
        },
        "sun": DEMO_SUN,
    }
    (DEMO_OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    total = sum(f.stat().st_size for f in DEMO_OUT.iterdir())
    print(f"  {DEMO_OUT} ({total / 1e6:.1f} MB total)")


def main() -> None:
    SOAR_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "cloudyview" / "soar" / "raymarch.wgsl",
                SOAR_OUT / "raymarch.wgsl")
    print(f"copied raymarch.wgsl -> {SOAR_OUT / 'raymarch.wgsl'}")
    export_ocean()
    export_demo()


if __name__ == "__main__":
    main()
