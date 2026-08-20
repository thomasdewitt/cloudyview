#!/usr/bin/env python
"""Render the night-city scene from a terminal, a few named views at a time.

This is the iteration loop for soar's CITY mode: it drives the same shader,
the same uniform block and the same accumulation the browser does, off a real
cloud field, so a change to the night look can be looked at rather than
argued about. It is not a test — nothing here asserts an image; it writes
PNGs and prints what each one cost.

The field pipeline is witness's, reused rather than re-derived
(`cloudyview.witness._field_level`): condensate to extinction, the config's
extinction multiplier, the absolute-metre AABB, and the empty-sky z crop. A
harness that built its sigma its own way would be rendering a different scene
from the app and would not know it.

City mode needs a shader that declares `const CITY: bool = false;`. Until that
lands, construction fails with that message and nothing else happens — which
is the point of running `--no-city` first: it renders the ordinary daylight
ocean scene through this same code path, so a failure afterwards is the
shader's and not the harness's.

    uv run python tools/night_city_harness.py --no-city     # sanity
    uv run python tools/night_city_harness.py               # the city
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_OUTDIR = ("/tmp/claude-1000/-home-thomas-code-and-data-cloudyview/"
                  "a89b2561-235a-44d3-a402-4937f515436f/scratchpad/city_views")
DEFAULT_FIELD = "data/TWPICE_subvolume_256x256_5km.nc"

# The moon, packed into the sun's row. Above the horizon because the periodic
# light march exits through the domain top and has no exit below it.
MOON_AZIMUTH = 310.0
MOON_ELEVATION = 22.0
# Night wants far more exposure than day; this is a starting value, not a
# metered one.
NIGHT_EXPOSURE = 6.0

# camera xy as a FRACTION of the domain box, z in absolute metres. The TWP-ICE
# subvolume is about 5.1 km square, so 0.5 is roughly 2.6 km in from either
# edge — but the fractions are resolved against the field's own bmin/bmax, not
# against that number.
CITY_VIEWS = {
    "aerial":  {"xy": (0.50, 0.50), "z": 3300.0, "elevation": -28.0,
                "azimuth": 45.0,  "fov": 65.0},
    "base":    {"xy": (0.35, 0.35), "z": 900.0,  "elevation": -8.0,
                "azimuth": 30.0,  "fov": 70.0},
    # A canyon 2.5 km east of the megatower district, looking back at it.
    "street":  {"xy": (0.60, 0.50), "z": 60.0,   "elevation": 25.0,
                "azimuth": 265.0, "fov": 80.0},
    "horizon": {"xy": (0.80, 0.72), "z": 1600.0, "elevation": -4.0,
                "azimuth": 227.0, "fov": 68.0},
    # Toward the moon (default light azimuth 310): crescent over the skyline.
    "moonrise": {"xy": (0.73, 0.42), "z": 600.0, "elevation": 20.0,
                 "azimuth": 300.0, "fov": 55.0},
    # From above the cloud tops, straight at the moon: nothing between the
    # camera and the crescent, so this view judges the disc itself.
    "moon_check": {"xy": (0.50, 0.50), "z": 4900.0, "elevation": 38.0,
                   "azimuth": 310.0, "fov": 50.0},
}

# --no-city renders one view only, and it is the aerial geometry under a
# daytime sun over the ocean: the question it answers is whether this script
# can load, upload, render and write at all, not what the daylight scene
# looks like.
DAYLIGHT_VIEW = {"daylight_sanity": {**CITY_VIEWS["aerial"]}}
DAY_AZIMUTH = 20.0
DAY_ELEVATION = 55.0
DAY_EXPOSURE = 4.0


def build_level(nc_path: Path, verbose: bool = True):
    """witness's own field preparation, called rather than copied."""
    import cloudyview as cv
    from cloudyview.witness import _field_level

    field = cv.load(str(nc_path))
    return _field_level(field, nc_path.stem, verbose=verbose)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render soar's night-city views to PNG.")
    parser.add_argument("--field", default=DEFAULT_FIELD,
                        help=f"NetCDF cloud field (default: {DEFAULT_FIELD})")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help="where the PNGs go")
    parser.add_argument("--frames", type=int, default=48,
                        help="accumulated passes per view (default: 48)")
    parser.add_argument("--size", type=int, nargs=2, default=(960, 540),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--exposure", type=float, default=None,
                        help=f"tone-map exposure (default: {NIGHT_EXPOSURE} "
                             f"in city mode, {DAY_EXPOSURE} in --no-city)")
    parser.add_argument("--city", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="night city (default). --no-city renders one "
                             "daylight ocean view named daylight_sanity, "
                             "which is how this harness proves itself "
                             "against a shader that has no CITY yet")
    parser.add_argument("--views", nargs="+", default=None,
                        help=f"subset of {sorted(CITY_VIEWS)}")
    parser.add_argument("--camera", nargs=6, type=float, default=None,
                        metavar=("X", "Y", "Z", "AZ", "EL", "FOV"),
                        help="one free view instead of the named ones: "
                             "world meters and degrees. The component "
                             "iteration loop's close-up lens.")
    args = parser.parse_args(argv)

    from cloudyview.basic_render import save_image
    from cloudyview.soar_host import (
        STEP_VOXEL_FACTOR, APP_LIGHT_MARCH_LOD_DEGREES,
        APP_VIEW_STEP_LOD_DEGREES, DEFAULT_LOD_STRENGTH,
        SceneState, SoarRenderer, ViewState,
    )
    from cloudyview.witness import OCEAN_REFLECTANCE

    field_path = Path(args.field)
    if not field_path.is_absolute():
        field_path = Path(__file__).resolve().parents[1] / field_path
    if not field_path.exists():
        print(f"No such field: {field_path}", file=sys.stderr)
        return 1

    views = DAYLIGHT_VIEW if not args.city else dict(CITY_VIEWS)
    if args.views:
        if not args.city:
            print("--views has nothing to choose from in --no-city mode; "
                  "it renders the one sanity view.", file=sys.stderr)
            return 2
        missing = [v for v in args.views if v not in views]
        if missing:
            print(f"unknown view(s) {missing}; have {sorted(views)}",
                  file=sys.stderr)
            return 2
        views = {k: views[k] for k in args.views}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Field: {field_path}")
    level = build_level(field_path, verbose=True)
    bmin, bmax = level.bmin, level.bmax
    print(f"  Grid {level.sigma.shape}, box "
          f"x {bmin[0]:.0f}-{bmax[0]:.0f} m, y {bmin[1]:.0f}-{bmax[1]:.0f} m, "
          f"z {bmin[2]:.0f}-{bmax[2]:.0f} m")

    # The shader is specialized at construction, so a missing CITY declaration
    # fails here, before a GPU has been asked to do anything.
    try:
        renderer = SoarRenderer(periodic=True, city=args.city)
    except RuntimeError as exc:
        print(f"\nCannot build the {'city' if args.city else 'ocean'} "
              f"renderer:\n  {exc}", file=sys.stderr)
        return 3
    renderer.upload_volume(level.sigma)

    meta = renderer.surface_meta
    cell_m = float(meta["cell_m"] if args.city else meta["dx_m"])
    tile_extent_m = float(meta["tile_extent_m"])
    max_lod = int(meta["mips"]) - 1
    print(f"  Surface tile: {'city' if args.city else 'ocean'}, "
          f"n={meta['n']}, cell {cell_m:g} m, extent {tile_extent_m:g} m, "
          f"top mip {max_lod}")

    min_voxel = min(level.dx)
    state = SceneState(
        bmin=[float(v) for v in bmin], bmax=[float(v) for v in bmax],
        dt_view=min_voxel * STEP_VOXEL_FACTOR,
        dt_light=min_voxel * STEP_VOXEL_FACTOR,
        periodic=True,
        ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=cell_m,
        ocean_tile_extent=tile_extent_m,
        ocean_max_lod=max_lod,
        ocean_enabled=True,
        city=args.city,
    )

    light_az = MOON_AZIMUTH if args.city else DAY_AZIMUTH
    light_el = MOON_ELEVATION if args.city else DAY_ELEVATION
    exposure = args.exposure
    if exposure is None:
        exposure = NIGHT_EXPOSURE if args.city else DAY_EXPOSURE
    w, h = args.size
    print(f"  {'Moon' if args.city else 'Sun'}: azimuth {light_az:g}, "
          f"elevation {light_el:g}; exposure {exposure:g}; "
          f"{w}x{h}, {args.frames} accumulated passes\n")

    if args.camera is not None:
        x, y, z, az, el, fov = args.camera
        views = {"camera": {"xy": None, "world_xy": (x, y), "z": z,
                            "azimuth": az, "elevation": el, "fov": fov}}

    for name, v in views.items():
        if v.get("world_xy") is not None:
            position = [float(v["world_xy"][0]), float(v["world_xy"][1]),
                        float(v["z"])]
        else:
            fx, fy = v["xy"]
            position = [float(bmin[0] + fx * (bmax[0] - bmin[0])),
                        float(bmin[1] + fy * (bmax[1] - bmin[1])),
                        float(v["z"])]
        view = ViewState(
            camera_position=position,
            azimuth=v["azimuth"], elevation=v["elevation"], fov=v["fov"],
            output_size=(w, h), render_size=(w, h),
            jitter=True, subpixel=True,
            sun_azimuth=light_az, sun_elevation=light_el,
            exposure=exposure,
            light_march_lod_degrees=(APP_LIGHT_MARCH_LOD_DEGREES
                                     * DEFAULT_LOD_STRENGTH),
            view_step_lod_degrees=(APP_VIEW_STEP_LOD_DEGREES
                                   * DEFAULT_LOD_STRENGTH),
        )
        t0 = time.perf_counter()
        # renderer.render IS the accumulation: `frames` passes averaged in a
        # float32 target, pass 0 unjittered in subpixel, the rest jittered.
        # Averaging returned frames here instead would be a second, different
        # accumulator — witness does not, so neither does this.
        image = renderer.render(state, view, frames=args.frames)
        elapsed = time.perf_counter() - t0
        out = outdir / f"{name}.png"
        save_image(np.asarray(image), str(out))
        print(f"  {name:<16} {elapsed:6.2f} s  "
              f"mean {image.mean():.4f}  max {image.max():.4f}  -> {out}")

    print(f"\n{len(views)} view(s) written to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
