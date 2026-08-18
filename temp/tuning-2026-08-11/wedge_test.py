"""Monotonicity harness: brightness under a linear-tau wedge slab.

A 1 km slab whose vertical optical depth ramps linearly 0 -> ~80 along x;
camera below looking straight up. The center image row is then brightness
as a function of optical depth. Real cloud bases darken monotonically with
tau; any local minimum/maximum here is a lighting-term handover artifact.

Usage: uv run python temp/tuning-2026-08-11/wedge_test.py <out_prefix>
Writes <out_prefix>.png (render) and prints the profile + dip report.
"""
import sys
import numpy as np
from PIL import Image

from cloudyview.soar_host import SceneState, ViewState
import importlib
W = importlib.import_module("cloudyview.witness")

NX, NY, NZ = 512, 64, 60
DX = 60.0            # m, so x spans 30.7 km
Z0, Z1 = 1200.0, 4800.0
SLAB_LO, SLAB_HI = 20, 37   # k-range of the slab -> ~1 km thick
TAU_MAX = 372.0  # over the full 30.7 km; ~12.1 tau/km so the visible span covers 0-80

_cached = None


def get_level():
    global _cached
    if _cached is None:
        dz = (Z1 - Z0) / NZ
        thickness = (SLAB_HI - SLAB_LO) * dz
        sigma = np.zeros((NX, NY, NZ), dtype=np.float64)
        ramp = np.linspace(0.0, TAU_MAX, NX) / thickness  # m^-1 per column
        sigma[:, :, SLAB_LO:SLAB_HI] = ramp[:, None, None]
        bmin = np.array([0.0, 0.0, Z0])
        bmax = np.array([NX * DX, NY * DX, Z1])
        _cached = W.NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name="wedge")
    return _cached


def main():
    prefix = sys.argv[1]
    level = get_level()
    renderer = W._renderer_for([level], periodic=False, tone_mapped=True)
    dt = min(level.dx) * W.STEP_VOXEL_FACTOR
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dt, dt_light=dt, periodic=False,
        ocean_reflectance=W.OCEAN_REFLECTANCE,
    )
    # Under the slab, mid-domain, looking straight up; horizontal fov wide
    # enough that image x spans most of the wedge.
    view = ViewState(
        camera_position=[3500.0, NY * DX * 0.5, 300.0],
        azimuth=0.0, elevation=89.5, fov=120.0,
        output_size=(1024, 256), render_size=(1024, 256),
        sun_azimuth=20.0, sun_elevation=55.0,
        tone_map_gamma=1.66,
    )
    img = renderer.render(state, view, frames=64)
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(f"{prefix}.png")

    # Luminance profile along the center row, smoothed lightly.
    row = arr[arr.shape[0] // 2].astype(float)
    lum = row @ [0.2126, 0.7152, 0.0722]
    k = 9
    smooth = np.convolve(lum, np.ones(k) / k, mode="valid")
    # Report the profile every 32 px and any interior local extrema deeper
    # than 2/255 (dip = local min below both neighbors' running max).
    print("profile (x -> luminance):")
    print("  " + " ".join(f"{v:5.1f}" for v in smooth[::32]))
    # find dips: points where value < running max so far AND a later value
    # exceeds it by margin (i.e. brightness goes back UP after falling)
    margin = 2.0
    run_min = np.minimum.accumulate(smooth)
    later_rise = smooth - run_min
    worst = float(later_rise.max())
    where = int(np.argmax(later_rise))
    print(f"worst re-brightening after a fall: {worst:.1f}/255 at px {where}"
          f" (0 = perfectly monotone decay past the peak)")


if __name__ == "__main__":
    main()
