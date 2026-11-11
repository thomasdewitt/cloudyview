#!/usr/bin/env python
"""Time the soar renderer: milliseconds per frame, by scene, tier and view.

    uv run python tools/soar_bench.py
    uv run python tools/soar_bench.py --scenes day --tiers low,medium --repeats 5
    uv run python tools/soar_bench.py --demo web/soar/demos/twpice

Each tier is rendered as it flies: its own step factors, render scale and
lighting method, read from web/soar/constants.js, at the output size given.
The renderer is the Python host, so a number here is the cost of the shader
on this machine's GPU without the browser around it.

Reading the numbers:
- Each view is timed `--repeats` times and the fastest is reported. Anything
  else using the GPU (a browser, another render) makes a run slower, never
  faster, so the minimum is the estimate least disturbed by it. Close the
  browser for numbers that will be compared.
- Compare two checkouts by running this in each, one after the other, more
  than once. The differences between runs of one checkout say how small a
  difference between two is worth believing.
- Below about 100,000 pixels (Minimal and Low at 1280x720) a frame's cost
  is mostly the fixed work of submitting it, not the march.

The default output size is small enough not to tie up a laptop. A sweep of
every tier at 2560x1440 holds the GPU for minutes, and on a Mac that is the
GPU the window server draws with.
"""

import argparse
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "tools"))

from conftest import build_soar_level                         # noqa: E402
from cloudyview.soar.renderer import SoarRenderer             # noqa: E402
from cloudyview.soar.surfaces import (surface_cell_m,         # noqa: E402
                                      surface_clock_period_s,
                                      surface_period_m)
from cloudyview.soar.uniforms import (DEFAULT_OCEAN_REFLECTANCE,  # noqa: E402
                                      SceneState, ViewState,
                                      camera_world_origin)
from cloudyview.witness import NestedLevel                    # noqa: E402
from night_city_harness import (CITY_VIEWS, MOON_AZIMUTH,     # noqa: E402
                                MOON_ELEVATION, NIGHT_EXPOSURE)

# Day views: camera position as a fraction of the field's box (the judge
# views' convention, tests/conftest.py). Thick backlit cloud, under the base,
# a sky-dominated overview, a lit wall, and the sea from above.
DAY_VIEWS = {
    "thick_backlit": dict(camera_position=[-0.1, -0.2, 0.1], azimuth=20, elevation=8, fov=70),
    "under_base": dict(camera_position=[0.2, 0.55, -0.85], azimuth=20, elevation=25, fov=80),
    "overview": dict(camera_position=[0.0, -1.0, 0.7], azimuth=0, elevation=2, fov=90),
    "thick_lit": dict(camera_position=[0.3, 1.7, 0.35], azimuth=180, elevation=2, fov=75),
    "ocean_down": dict(camera_position=[0.0, 0.0, 3.0], azimuth=180, elevation=-55, fov=70),
}
DAY_SUN = dict(sun_azimuth=20.0, sun_elevation=55.0)
TIER_ORDER = ("minimal", "low", "medium", "high", "max")


def flight_presets() -> dict:
    """QUALITY_PRESETS as the browser has them."""
    script = ('import { QUALITY_PRESETS as Q } from "%s";'
              "console.log(JSON.stringify(Q));"
              % (REPO / "web/soar/constants.js").as_uri())
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def level_from_demo(demo_dir: Path) -> NestedLevel:
    """The field of a baked demo (meta.json + volume.bin.gz): the bytes the
    browser is served."""
    meta = json.loads((demo_dir / "meta.json").read_text())
    vol = meta["volume"]
    if vol["format"] != "r16float":
        raise SystemExit(f"unsupported demo volume format {vol['format']!r}")
    raw = gzip.decompress((demo_dir / vol["file"]).read_bytes())
    nx, ny, nz = vol["shape_xyz"]
    # C-order (nx, ny, nz), z fastest: the layout upload_volume takes.
    sigma = np.frombuffer(raw, dtype=np.float16).reshape(nx, ny, nz)
    return NestedLevel(sigma=sigma,
                       bmin=np.asarray(vol["bmin"], dtype=np.float64),
                       bmax=np.asarray(vol["bmax"], dtype=np.float64),
                       name=meta["id"])


def scene_views(scene: str, level) -> dict:
    """name -> ViewState keywords (camera in world metres, the sun)."""
    if scene == "day":
        return {name: dict(camera_position=[float(x) for x in camera_world_origin(
                               v["camera_position"], level.bmin, level.bmax)],
                           azimuth=v["azimuth"], elevation=v["elevation"],
                           fov=v["fov"], **DAY_SUN)
                for name, v in DAY_VIEWS.items()}
    lo, hi = level.bmin, level.bmax
    return {name: dict(camera_position=[float(lo[0] + v["xy"][0] * (hi[0] - lo[0])),
                                        float(lo[1] + v["xy"][1] * (hi[1] - lo[1])),
                                        float(v["z"])],
                       azimuth=v["azimuth"], elevation=v["elevation"],
                       fov=v["fov"], sun_azimuth=MOON_AZIMUTH,
                       sun_elevation=MOON_ELEVATION, exposure=NIGHT_EXPOSURE)
            for name, v in CITY_VIEWS.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="day,city")
    ap.add_argument("--tiers", default="minimal,low,medium,high")
    ap.add_argument("--views", default="", help="comma-separated names; default all")
    ap.add_argument("--output", default="1280x720",
                    help="the window size; each tier renders at its own scale of it")
    ap.add_argument("--frames", type=int, default=8, help="frames per timing")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timings per view; the fastest is reported")
    ap.add_argument("--demo", type=Path, default=None,
                    help="a baked demo directory instead of the test field")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per line instead of the table")
    args = ap.parse_args()

    presets = flight_presets()
    out_w, out_h = (int(v) for v in args.output.lower().split("x"))
    level = level_from_demo(args.demo) if args.demo else build_soar_level()
    voxel = float(min(level.dx))
    if not args.json:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
        print(f"soar bench at {rev}: field {level.name}, window {out_w}x{out_h}, "
              f"fastest of {args.repeats} x {args.frames} frames\n")
        print(f"{'scene':6s}{'tier':9s}{'view':15s}{'render size':>12s}"
              f"{'ms/frame':>10s}{'fps':>8s}")

    for scene in args.scenes.split(","):
        surface = {"day": "ocean", "city": "city"}[scene]
        renderer = SoarRenderer(periodic=True, nested=False, surface=surface)
        renderer.upload_volume(level.sigma)
        meta = renderer.surface_meta
        views = scene_views(scene, level)
        wanted = [v for v in views if not args.views or v in args.views.split(",")]

        def scene_state(tier):
            return SceneState(
                bmin=[float(v) for v in level.bmin],
                bmax=[float(v) for v in level.bmax],
                dt_view=voxel * presets[tier]["stepFactor"],
                dt_light=voxel * presets[tier]["lightStepFactor"],
                periodic=True, ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE,
                surface=surface, ocean_fif_dx=surface_cell_m(surface, meta),
                ocean_tile_extent=float(meta["tile_extent_m"]),
                ocean_max_lod=int(meta["mips"]) - 1,
                surface_period_m=surface_period_m(surface, meta),
                clock_period_s=surface_clock_period_s(surface, meta))

        # One sun per scene, so one light cache: baked at High's light step
        # whatever the tier flies with, as the browser bakes it.
        renderer.bake_light_cache(
            scene_state("high"),
            ViewState(output_size=(out_w, out_h), render_size=(out_w, out_h),
                      **views[wanted[0]]))
        for tier in sorted(args.tiers.split(","), key=TIER_ORDER.index):
            p = presets[tier]
            state = scene_state(tier)
            size = (max(1, round(out_w * p["renderScale"])),
                    max(1, round(out_h * p["renderScale"])))
            for name in wanted:
                view = ViewState(output_size=(out_w, out_h), render_size=size,
                                 jitter=True, light_cache=bool(p["lightCache"]),
                                 **views[name])
                renderer.render(state, view, frames=2)       # pipelines, caches
                ms = float("inf")
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    renderer.render(state, view, frames=args.frames)
                    ms = min(ms, 1e3 * (time.perf_counter() - t0) / args.frames)
                if args.json:
                    print(json.dumps(dict(scene=scene, tier=tier, view=name,
                                          size=list(size), ms=round(ms, 2))),
                          flush=True)
                else:
                    print(f"{scene:6s}{tier:9s}{name:15s}"
                          f"{'%dx%d' % size:>12s}{ms:10.1f}{1e3 / ms:8.1f}",
                          flush=True)


if __name__ == "__main__":
    main()
