"""Export a self-contained browser demo payload for soar's WebGPU build.

Produces web/demo/: the fp16 ghost-padded extinction volume, an fp16 FIF
ocean normal mip chain, the raymarch WGSL (copied verbatim — the shader IS
the shared artifact), and meta.json holding the full 21-row uniform
template dumped from a real InteractiveRenderer, so the JS host only
rewrites the camera/size/sampling rows per frame and can never drift from
the Python renderer's look constants.

Run on a machine with the GPU available (the template dump instantiates
the real renderer): uv run python tools/export_web_demo.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

from cloudyview.camera import Camera
from cloudyview.cloudfield import load
from cloudyview.ocean_fif import generate_fif_normals
from cloudyview.soar import engine as soar_engine
from cloudyview.soar.engine import InteractiveRenderer, _build_fif_normal_mips

REPO = Path(__file__).resolve().parents[1]
DEMO_NC = REPO / "data" / "TWPICE_subvolume_256x256_5km.nc"
OUT = REPO / "web" / "demo"
FIF_N = 512          # 2048 in the app; 512 keeps the download reasonable
SUN_AZIMUTH = 235.0
SUN_ELEVATION = 25.0
LOD = dict(light_march_lod_degrees=1.4, view_step_lod_degrees=0.6)

OUT.mkdir(parents=True, exist_ok=True)

field = load(str(DEMO_NC))
print(f"field: {field.lwc.shape}")

print(f"FIF ocean tile {FIF_N}^2 (seeded)")
# dx=0.2 m keeps the app's ~100 m tile extent at 1/4 the texel count; the
# physical outer scale is unchanged (outer_scale_m is in meters).
fif = generate_fif_normals(
    N=FIF_N, dx_m=0.2, rng=np.random.default_rng(20260717)
)

renderer = InteractiveRenderer(
    field, periodic=True, volume_fp16=True, fif_normals=fif
)
renderer.set_quality_tier("high", camera_moving=True)
renderer.write_uniforms(
    Camera(), (1280, 720), jitter=True, subpixel=True, jitter_scale=0.65,
    sun_azimuth=SUN_AZIMUTH, sun_elevation=SUN_ELEVATION, **LOD,
)
template = renderer._current_uniform.astype(np.float32)

# --- Volume: fp16, ghost-padded, periodic x/y faces baked (identical to
# the engine's upload; see engine.__init__). C-order (nx+2, ny+2, nz+2),
# so the texture upload is width=nz+2, height=ny+2, depth=nx+2.
sigma = soar_engine.optical_depth.compute_extinction_field(
    field.lwc, field.z, re=10.0, iwc=field.iwc, re_ice=30.0
)
sigma = np.ascontiguousarray(sigma, dtype=np.float16)
nx, ny, nz = sigma.shape
padded = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float16)
padded[1:-1, 1:-1, 1:-1] = sigma
faces = soar_engine._ghost_face_arrays(sigma)
padded[0, :, :] = faces["x_lo"]
padded[-1, :, :] = faces["x_hi"]
padded[:, 0, :] = faces["y_lo"][:, 0, :]
padded[:, -1, :] = faces["y_hi"][:, 0, :]
(OUT / "volume.bin").write_bytes(padded.tobytes())
print(f"volume.bin: {padded.nbytes / 1e6:.1f} MB ({padded.shape})")

# --- FIF normal mips: rgba16float per level.
base = np.empty((FIF_N, FIF_N, 4), dtype=np.float32)
base[..., 0], base[..., 1], base[..., 2] = fif[0], fif[1], fif[2]
base[..., 3] = 1.0
mips = _build_fif_normal_mips(base)
for i, mip in enumerate(mips):
    (OUT / f"fif_mip{i}.bin").write_bytes(
        np.ascontiguousarray(mip, dtype=np.float16).tobytes()
    )
print(f"fif: {len(mips)} mips, base {mips[0].shape[0]}^2")

shutil.copy(REPO / "cloudyview" / "soar" / "raymarch.wgsl",
            REPO / "web" / "raymarch.wgsl")

bmin = [float(v) for v in renderer.bmin]
bmax = [float(v) for v in renderer.bmax]
meta = {
    "schema": "cloudyview.webdemo.v1",
    "source": DEMO_NC.name,
    "volume": {
        "padded_dims_xyz": [nx + 2, ny + 2, nz + 2],
        "format": "r16float",
        "bmin": bmin,
        "bmax": bmax,
    },
    "fif": {"n": FIF_N, "mips": len(mips), "format": "rgba16float"},
    "uniform_template": template.tolist(),
    # Rows the JS host owns per frame; everything else must come from the
    # template verbatim (spectral constants, LOD, realism gates, ocean).
    "dynamic_rows": {
        "0": "camera origin xyz + tan(fov/2)",
        "1": "forward xyz + aspect",
        "2": "right xyz + exposure(keep template w)",
        "3": "up xyz + jitter enable",
        "4": "sun xyz(keep template) + frame index",
        "7": "render w, h + keep template zw",
        "10": "subpixel enable, jitter scale + zeros",
    },
    "sun": {"azimuth": SUN_AZIMUTH, "elevation": SUN_ELEVATION},
    "camera_default": {
        "position": list(Camera().position),
        "azimuth": Camera().azimuth,
        "elevation": Camera().elevation,
        "fov": Camera().fov,
    },
    "cloudyview_version": soar_engine.__version__
    if hasattr(soar_engine, "__version__") else None,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
total = sum(f.stat().st_size for f in OUT.iterdir())
print(f"exported {OUT} ({total / 1e6:.1f} MB total)")
