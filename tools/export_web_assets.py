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

The WGSL under web/soar/ is edited in place. It used to be copied from the
desktop engine, which was the source; the desktop app is gone and the browser
copy is now the original.

    uv run python tools/export_web_assets.py
"""

import json
from pathlib import Path

import numpy as np

from cloudyview import optical_depth
from cloudyview.cloudfield import CloudField, load
from cloudyview.glimpse import glimpse
from cloudyview.ocean_fif import generate_fif_normals

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


def _ghost_face_arrays(sigma: np.ndarray) -> dict:
    """Periodic x/y ghost-border faces for the padded volume texture.

    The padded texture is (w=nz+2, h=ny+2, d=nx+2); texture depth indexes x
    and texture rows index y. The x faces are single depth slices of shape
    (ny+2, nz+2); the y faces span every depth slice with shape
    (nx+2, 1, nz+2) so one upload covers the whole row. Corner texels wrap in
    both x and y (they are the trilinear support for samples near a domain
    corner); the z ghost columns stay zero — the vertical taper is not
    periodic.
    """
    nx, ny, nz = sigma.shape
    dtype = sigma.dtype
    x_lo = np.zeros((ny + 2, nz + 2), dtype=dtype)  # texture depth 0
    x_hi = np.zeros((ny + 2, nz + 2), dtype=dtype)  # texture depth nx+1
    x_lo[1:-1, 1:-1] = sigma[-1]
    x_lo[0, 1:-1] = sigma[-1, -1]
    x_lo[-1, 1:-1] = sigma[-1, 0]
    x_hi[1:-1, 1:-1] = sigma[0]
    x_hi[0, 1:-1] = sigma[0, -1]
    x_hi[-1, 1:-1] = sigma[0, 0]
    y_lo = np.zeros((nx + 2, 1, nz + 2), dtype=dtype)  # texture row 0
    y_hi = np.zeros((nx + 2, 1, nz + 2), dtype=dtype)  # texture row ny+1
    y_lo[1:-1, 0, 1:-1] = sigma[:, -1]
    y_lo[0, 0, 1:-1] = sigma[-1, -1]
    y_lo[-1, 0, 1:-1] = sigma[0, -1]
    y_hi[1:-1, 0, 1:-1] = sigma[:, 0]
    y_hi[0, 0, 1:-1] = sigma[-1, 0]
    y_hi[-1, 0, 1:-1] = sigma[0, 0]
    return {"x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi}


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


def export_demo() -> None:
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    field = load(str(DEMO_NC))
    print(f"demo field: {field.lwc.shape} from {DEMO_NC.name}")

    sigma = optical_depth.compute_extinction_field(
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

    # The minimap image. glimpse integrates water paths rather than the sigma
    # above — ice is weighted nearly twice as heavily — so this is a genuinely
    # separate quantity, not a reduction of volume.bin. Shipped as float32 and
    # colorized in the browser, so the demo and an opened file go through the
    # same colour ramp instead of two that could drift.
    albedo = np.ascontiguousarray(glimpse(field), dtype=np.float32)
    (DEMO_OUT / "map.bin").write_bytes(albedo.tobytes())
    print(f"  map.bin: {albedo.nbytes / 1e6:.2f} MB {albedo.shape} (ny, nx)")

    faces = _ghost_face_arrays(sigma)
    blob = b"".join(
        np.ascontiguousarray(faces[name], dtype=np.float16).tobytes()
        for name in ("x_lo", "x_hi", "y_lo", "y_hi")
    )
    (DEMO_OUT / "faces.bin").write_bytes(blob)
    print(f"  faces.bin: {len(blob) / 1e6:.2f} MB")

    bmin, bmax = _volume_aabb(field)
    meta = {
        "schema": "cloudyview.web.demo.v3",
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
        # Two-stream albedo for the minimap, float32, (ny, nx), east right
        # and north up.
        "map": {"shape_yx": [int(albedo.shape[0]), int(albedo.shape[1])]},
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
    export_ocean()
    export_demo()


if __name__ == "__main__":
    main()
