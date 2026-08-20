#!/usr/bin/env python
"""A/B stills for the sun-tau light cache: live march vs cache at 1x/2x/4x.

The decision material for whether the cache ships (Thomas, 2026-08-19: "i'm
not yet sure whether i want it, and would want to see some a/b images of
cloud edges"). Renders converged full-resolution stills of the judge views
that stress lit/shadowed cloud edges, once with the live per-sample sun
march and once per cache divisor, and writes full frames plus a side-by-side
edge-crop strip per view.

Usage:
    uv run python tools/light_cache_ab.py
    uv run python tools/light_cache_ab.py --views v1 --divisors 1,2

Output lands in temp/light-cache-ab-<date>/.
"""

import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_FILE = REPO / "data" / "TWPICE_subvolume_256x256_5km.nc"

# The two judge views where sun-shadow structure on cloud edges is the
# picture: the thick backlit mass (silver lining, deep shadow) and the
# overview (shadowed flanks against lit tops at many scales).
VIEWS = {
    "v1_thick_backlit": {"camera_position": [-0.1, -0.2, 0.1],
                         "azimuth": 20, "elevation": 8, "fov": 70},
    "v4_overview_south": {"camera_position": [0.0, -1.0, 0.7],
                          "azimuth": 0, "elevation": 2, "fov": 90},
}
SUN = {"sun_azimuth": 20.0, "sun_elevation": 55.0}
OUTPUT = (1280, 720)
FRAMES = 96          # converged enough that jitter noise is not the diff

# Where each view's edge crop sits (fractions of width/height), chosen to
# land on a lit/shadow cloud edge in these frozen views.
CROPS = {
    "v1_thick_backlit": (0.35, 0.15, 0.30),   # cx, cy, size (of height)
    "v4_overview_south": (0.55, 0.35, 0.30),
}


def render_all(views, divisors, output, frames):
    import sys
    sys.path.insert(0, str(REPO / "tools" / "benchmarking"))
    from soar_frame_bench import build_level
    from cloudyview.soar_host import (
        SceneState, SoarRenderer, ViewState, camera_world_origin,
    )
    from cloudyview.witness import OCEAN_REFLECTANCE

    level = build_level(DATA_FILE)
    renderer = SoarRenderer(periodic=True, nested=False)
    renderer.upload_volume(level.sigma)
    min_voxel = min(level.dx)
    # High-tier sampling: full resolution, fine steps — the configuration a
    # parked view converges to, which is the picture the A/B judges.
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=min_voxel * 2.0, dt_light=min_voxel * 2.0,
        periodic=True, ocean_reflectance=OCEAN_REFLECTANCE,
    )

    results = {}       # (view, variant) -> (h, w, 3) float64
    timings = []
    for variant in ["live"] + [f"cache{d}" for d in divisors]:
        divisor = 0 if variant == "live" else int(variant.removeprefix("cache"))
        baked = False
        for view_name, v in views.items():
            position = camera_world_origin(
                v["camera_position"], level.bmin, level.bmax)
            view = ViewState(
                camera_position=[float(p) for p in position],
                azimuth=v["azimuth"], elevation=v["elevation"], fov=v["fov"],
                output_size=output, render_size=output,
                sun_azimuth=SUN["sun_azimuth"],
                sun_elevation=SUN["sun_elevation"],
                light_cache=divisor > 0,
            )
            if divisor > 0 and not baked:
                t0 = time.perf_counter()
                dims = renderer.bake_light_cache(state, view, divisor=divisor)
                bake_ms = (time.perf_counter() - t0) * 1000.0
                timings.append((variant, "bake",
                                f"{dims[0]}x{dims[1]}x{dims[2]}", bake_ms))
                print(f"{variant}: bake {dims} in {bake_ms:.0f} ms")
                baked = True
            renderer.render(state, view, frames=8)      # pipeline warmup
            t0 = time.perf_counter()
            img = renderer.render(state, view, frames=frames)
            ms = (time.perf_counter() - t0) * 1000.0 / frames
            timings.append((variant, view_name, f"{output[0]}x{output[1]}", ms))
            print(f"{variant}: {view_name} {ms:.2f} ms/frame")
            results[(view_name, variant)] = img
    return results, timings


def to_u8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def crop(img, frac):
    cx, cy, size = frac
    h, w = img.shape[:2]
    s = int(size * h)
    x0 = int(cx * w) - s // 2
    y0 = int(cy * h) - s // 2
    x0 = max(0, min(w - s, x0))
    y0 = max(0, min(h - s, y0))
    return img[y0:y0 + s, x0:x0 + s]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=str, default=None)
    parser.add_argument("--divisors", type=str, default="1,2,4")
    parser.add_argument("--frames", type=int, default=FRAMES)
    args = parser.parse_args()

    from PIL import Image

    views = VIEWS
    if args.views:
        wanted = args.views.split(",")
        views = {k: v for k, v in VIEWS.items()
                 if any(k.startswith(p) for p in wanted)}
    divisors = [int(d) for d in args.divisors.split(",")]

    out_dir = REPO / "temp" / f"light-cache-ab-{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results, timings = render_all(views, divisors, OUTPUT, args.frames)

    variants = ["live"] + [f"cache{d}" for d in divisors]
    for view_name in views:
        for variant in variants:
            img = to_u8(results[(view_name, variant)])
            Image.fromarray(img).save(out_dir / f"{view_name}_{variant}.png")
        # The edge-crop strip: live leftmost, then each divisor, upscaled 2x
        # nearest so the pixels themselves stay visible.
        strip = []
        for variant in variants:
            c = crop(to_u8(results[(view_name, variant)]), CROPS[view_name])
            c = np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)
            strip.append(c)
            strip.append(np.full((c.shape[0], 4, 3), 255, np.uint8))
        strip = np.concatenate(strip[:-1], axis=1)
        Image.fromarray(strip).save(out_dir / f"{view_name}_edge_strip.png")
        # Difference maps against live, amplified 8x, one per divisor.
        live = results[(view_name, "live")]
        for variant in variants[1:]:
            diff = np.abs(results[(view_name, variant)] - live) * 8.0
            Image.fromarray(to_u8(diff)).save(
                out_dir / f"{view_name}_diff8x_{variant}.png")

    with open(out_dir / "TIMINGS.md", "w") as f:
        f.write("# light-cache A/B timings (converged stills, "
                f"{OUTPUT[0]}x{OUTPUT[1]}, {args.frames} frames)\n\n")
        f.write("| variant | view | size | ms |\n|---|---|---|---|\n")
        for row in timings:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.2f} |\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
