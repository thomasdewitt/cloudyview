"""Render r2 subdomain of refine_test_000.nc for cloud-scattering iteration.

One folder per viewing geometry under temp/cloud-iter/<view>/.
Each run writes {iter:02d}_{shorthash}.png into every view's folder.
Iter number is derived from file count in the first geometry folder.

FIF normals are cached to temp/cloud-iter/_fif_cache.npz and reused across
iterations (fixed seed) so cloud-scattering changes are the only thing that
varies between renders.
"""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import netCDF4

from cloudyview.witness import NestedLevel, render_nested
from cloudyview.ocean_fif import generate_fif_normals
from cloudyview.optical_depth import compute_extinction_field
from cloudyview.angles import direction_from_azimuth_elevation


NC_PATH = '/home/thomas/code-and-data/cloudyview/temp/refine_test_000.nc'
BASE_OUT = Path('temp/cloud-iter')
FIF_CACHE = BASE_OUT / '_fif_cache.npz'
IMAGE_SIZE = (900, 675)

SUN_AZ, SUN_EL = 20.0, 55.0


def load_r2():
    with netCDF4.Dataset(NC_PATH) as ds:
        g = ds.groups['refinements'].groups['r2']
        x = g.variables['x'][:]
        y = g.variables['y'][:]
        z = g.variables['z'][:]
        qc = g.variables['qc'][:]
        qc_units = g.variables['qc'].getncattr('units')
    if qc_units.strip().lower() in ('kg/kg', 'g/g'):
        qc = qc * 1000.0
    elif qc_units.strip().lower() != 'g/kg':
        raise ValueError(f"Unexpected qc units: {qc_units}")
    sigma = compute_extinction_field(qc, z, re=10.0)
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    dz0 = z[1] - z[0]
    dzN = z[-1] - z[-2]
    bmin = np.array([x[0] - dx/2, y[0] - dy/2, z[0] - dz0/2], dtype=np.float64)
    bmax = np.array([x[-1] + dx/2, y[-1] + dy/2, z[-1] + dzN/2], dtype=np.float64)
    return sigma.astype(np.float64), bmin, bmax


def camera_basis(az_deg, el_deg):
    fwd = direction_from_azimuth_elevation(az_deg, el_deg)
    wu = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(fwd, wu)) > 0.999:
        wu = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, wu); right /= np.linalg.norm(right)
    up = np.cross(right, fwd); up /= np.linalg.norm(up)
    return tuple(fwd), tuple(right), tuple(up)


def get_fif():
    if FIF_CACHE.exists():
        d = np.load(FIF_CACHE)
        return d['nx'], d['ny'], d['nz'], float(d['dx'])
    print("Generating FIF normals (first run only)...")
    rng = np.random.default_rng(seed=42)  # fixed seed for iteration comparability
    nx, ny, nz, fif_dx = generate_fif_normals(rng=rng, verbose=True)
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    np.savez(FIF_CACHE, nx=nx, ny=ny, nz=nz, dx=np.array(fif_dx))
    return nx, ny, nz, fif_dx


def get_commit_hash():
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()


def next_iter_num(base_dir, first_view):
    d = base_dir / first_view
    if not d.exists():
        return 1
    return len(list(d.glob('*.png'))) + 1


def main():
    sigma, bmin, bmax = load_r2()
    print(f"sigma shape: {sigma.shape}  max={sigma.max():.3f}  nonzero mean={sigma[sigma>0].mean():.3f} m^-1")

    level = NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name="r2")
    fif_nx, fif_ny, fif_nz, fif_dx = get_fif()
    fif_normals = (fif_nx, fif_ny, fif_nz, fif_dx)

    ocean_z = 3.0
    top_z = bmax[2] + 1000.0
    x0, x1 = bmin[0], bmax[0]
    y0, y1 = bmin[1], bmax[1]
    corners = [
        ("SW", (x0, y0),  45.0),
        ("SE", (x1, y0), 315.0),
        ("NE", (x1, y1), 225.0),
        ("NW", (x0, y1), 135.0),
    ]
    views = []
    for lbl, (cx_, cy_), az in corners:
        views.append((f"ocean_{lbl}", (cx_, cy_, ocean_z), az,   5.0))
    for lbl, (cx_, cy_), az in corners:
        views.append((f"top_{lbl}",   (cx_, cy_, top_z),   az, -30.0))

    sha = get_commit_hash()
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    iter_num = next_iter_num(BASE_OUT, views[0][0])
    print(f"\n=== iteration {iter_num:02d}  commit {sha} ===\n")

    sun = tuple(direction_from_azimuth_elevation(SUN_AZ, SUN_EL))
    for name, cam_pos, cam_az, cam_el in views:
        out_dir = BASE_OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{iter_num:02d}_{sha}.png"
        print(f"-- {name}  az={cam_az} el={cam_el}  -> {out_path}")
        fwd, right, up = camera_basis(cam_az, cam_el)
        img = render_nested(
            [level],
            camera_position=cam_pos,
            camera_forward=fwd, camera_right=right, camera_up=up,
            sun_direction=sun,
            image_size=IMAGE_SIZE, fov_degrees=50.0,
            ocean_enabled=True,
            ocean_z=0.0,
            fif_normals=fif_normals,
            verbose=True,
        )
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(u8).save(out_path)


if __name__ == "__main__":
    main()
