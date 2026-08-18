"""Render the reference view from cloudyview_soar_20260811_105010_353.png metadata.

Usage: uv run python temp/tuning-2026-08-11/render_view.py <output.png>
           [--size W H] [--set name=value ...]

--set overrides any ViewState field (e.g. --set aerial_perspective_strength=0).
Drives witness's internals directly so every tuning knob is reachable.
"""
import sys
import time
import numpy as np
from PIL import Image

import cloudyview as cv
from cloudyview.camera import Camera
from cloudyview import optical_depth
import importlib
W = importlib.import_module("cloudyview.witness")
from cloudyview.soar_host import SceneState, ViewState, camera_world_origin

FIELD = "/home/thomas/code-and-data/turbulon-analysis/runs/small-domain/small_c002_s0030.nc"

# From the screenshot's cloudyview.render_metadata block.
CAMERA = Camera(
    position=(-0.10143794457919986, 0.42970502165019075, -0.9908945455437893),
    azimuth=148.080000000003,
    elevation=30.560000000000095,
    fov=100.0,
)
SUN_AZ = 20.0
SUN_EL = 55.0
GAMMA = 1.66
ACCUMULATE = 64

_field_cache = None


def get_level():
    global _field_cache
    if _field_cache is None:
        field = cv.load(FIELD, dataset_group="parent")
        iwc = field.iwc
        if iwc is not None and float(np.max(iwc)) < W.ICE_NEGLIGIBLE_G_KG:
            iwc = None
        sigma = optical_depth.compute_extinction_field(
            field.lwc, field.z, re=W.RE_LIQUID_UM,
            iwc=iwc, re_ice=W.RE_ICE_UM)
        sigma = np.ascontiguousarray(sigma, dtype=np.float64)
        bmin, bmax = W._volume_aabb(field)
        _field_cache = W.NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax,
                                     name="single")
    return _field_cache


def render(size=(1920, 1080), overrides=None):
    level = get_level()
    renderer = W._renderer_for([level], periodic=False, tone_mapped=True)
    dt = min(level.dx) * W.STEP_VOXEL_FACTOR
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dt, dt_light=dt, periodic=False,
        ocean_reflectance=W.OCEAN_REFLECTANCE,
    )
    position = camera_world_origin(CAMERA.position, level.bmin, level.bmax)
    kwargs = dict(
        camera_position=[float(v) for v in position],
        azimuth=CAMERA.azimuth, elevation=CAMERA.elevation, fov=CAMERA.fov,
        output_size=size, render_size=size,
        sun_azimuth=SUN_AZ, sun_elevation=SUN_EL,
        tone_map_gamma=GAMMA,
    )
    if overrides:
        kwargs.update(overrides)
    view = ViewState(**kwargs)
    t0 = time.perf_counter()
    image = renderer.render(state, view, frames=ACCUMULATE)
    print(f"  rendered {size[0]}x{size[1]} in {time.perf_counter()-t0:.2f}s"
          + (f"  overrides={overrides}" if overrides else ""))
    return image


def main():
    out = sys.argv[1]
    size = (1920, 1080)
    overrides = {}
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--size":
            size = (int(args[i + 1]), int(args[i + 2]))
            i += 3
        elif args[i] == "--camera":
            global CAMERA
            CAMERA = Camera(
                position=(float(args[i+1]), float(args[i+2]), float(args[i+3])),
                azimuth=float(args[i+4]), elevation=float(args[i+5]),
                fov=float(args[i+6]))
            i += 7
            continue
        elif args[i] == "--set":
            k, v = args[i + 1].split("=")
            overrides[k] = float(v)
            i += 2
        else:
            raise SystemExit(f"unknown arg {args[i]}")
    img = render(size, overrides)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
