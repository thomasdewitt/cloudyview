#!/usr/bin/env python
"""Per-frame timing of the soar/witness WGSL renderer across quality tiers.

Drives web/soar/raymarch.wgsl through cloudyview.soar_host exactly the way
the browser does — same uniform packing, same accumulate chain — so a number
measured here is a statement about the shader, not about the browser host.
Run before and after an optimization and diff the tables.

Usage:
    uv run python benchmarking/soar_frame_bench.py
    uv run python benchmarking/soar_frame_bench.py --frames 64 --views v1,v4,v8

Results are appended to benchmarking/soar_frame_results.md.
"""

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "TWPICE_subvolume_256x256_5km.nc"
RESULTS_FILE = Path(__file__).parent / "soar_frame_results.md"

# Mirrors web/soar/constants.js QUALITY_PRESETS — keep the two in step.
# These are the FLIGHT configurations; the browser's hold ladder converges
# every tier to high's sampling at full scale when the view is held.
TIERS = {
    "high":   {"render_scale": 1.0,   "step_factor": 2.0,
               "light_step_factor": 2.0,  "max_light_steps": 512},
    "medium": {"render_scale": 0.75,  "step_factor": 2.5,
               "light_step_factor": 4.0,  "max_light_steps": 512},
    "low":    {"render_scale": 0.60,  "step_factor": 3.0,
               "light_step_factor": 8.0,  "max_light_steps": 512},
    "potato": {"render_scale": 0.125, "step_factor": 4.0,
               "light_step_factor": 12.0, "max_light_steps": 512},
}

# The frozen judge views this repo already uses (tests/conftest.py), the
# three that stress different regimes: thick backlit cloud, sky-dominated
# overview, and the ocean LOD path.
VIEWS = {
    "v1_thick_backlit":  {"camera_position": [-0.1, -0.2, 0.1],  "azimuth": 20,  "elevation": 8,   "fov": 70},
    "v4_overview_south": {"camera_position": [0.0, -1.0, 0.7],   "azimuth": 0,   "elevation": 2,   "fov": 90},
    "v8_ocean_lod":      {"camera_position": [0.0, 0.0, 3.0],    "azimuth": 180, "elevation": -55, "fov": 70},
}

OUTPUT_SIZE = (960, 540)
SUN = {"sun_azimuth": 20.0, "sun_elevation": 55.0}


def build_level():
    import cloudyview as cv
    from cloudyview import optical_depth
    from cloudyview.witness import (
        ICE_NEGLIGIBLE_G_KG, RE_ICE_UM, RE_LIQUID_UM, NestedLevel, _volume_aabb,
    )

    field = cv.load(str(DATA_FILE))
    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < ICE_NEGLIGIBLE_G_KG:
        iwc = None
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=RE_LIQUID_UM, iwc=iwc, re_ice=RE_ICE_UM)
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)
    bmin, bmax = _volume_aabb(field)
    return NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name=DATA_FILE.stem)


def time_tier(level, tier_name, tier, views, frames, warmup):
    from cloudyview.soar_host import (
        SceneState, SoarRenderer, ViewState, camera_world_origin,
    )
    from cloudyview.witness import _padded, OCEAN_REFLECTANCE

    renderer = SoarRenderer(periodic=True, nested=False,
                            max_light_steps=tier["max_light_steps"])
    renderer.upload_volume(_padded(level.sigma))

    min_voxel = min(level.dx)
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=min_voxel * tier["step_factor"],
        dt_light=min_voxel * tier["light_step_factor"],
        periodic=True,
        ocean_reflectance=OCEAN_REFLECTANCE,
    )

    scale = tier["render_scale"]
    rw = max(1, int(OUTPUT_SIZE[0] * scale + 0.5))
    rh = max(1, int(OUTPUT_SIZE[1] * scale + 0.5))

    rows = []
    for view_name, v in views.items():
        position = camera_world_origin(v["camera_position"], level.bmin, level.bmax)
        view = ViewState(
            camera_position=[float(p) for p in position],
            azimuth=v["azimuth"], elevation=v["elevation"], fov=v["fov"],
            output_size=OUTPUT_SIZE, render_size=(rw, rh),
            sun_azimuth=v.get("sun_azimuth", SUN["sun_azimuth"]),
            sun_elevation=v.get("sun_elevation", SUN["sun_elevation"]),
        )
        renderer.render(state, view, frames=warmup)   # warm caches + pipeline
        t0 = time.perf_counter()
        renderer.render(state, view, frames=frames)
        elapsed = time.perf_counter() - t0
        ms = elapsed / frames * 1000.0
        rows.append((view_name, rw, rh, ms))
        print(f"  {tier_name:7s} {view_name:18s} {rw}x{rh}  "
              f"{ms:7.3f} ms/frame  ({1000.0 / ms:6.1f} fps)")
    return rows


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown GPU"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=64,
                        help="timed frames per view (default 64)")
    parser.add_argument("--warmup", type=int, default=8,
                        help="untimed warmup frames per view (default 8)")
    parser.add_argument("--views", type=str, default=None,
                        help="comma-separated prefixes, e.g. v1,v8")
    parser.add_argument("--tiers", type=str, default=None,
                        help="comma-separated tier names, e.g. high,potato")
    parser.add_argument("--label", type=str, default="",
                        help="note recorded with the results (e.g. git rev)")
    args = parser.parse_args()

    views = VIEWS
    if args.views:
        wanted = args.views.split(",")
        views = {k: v for k, v in VIEWS.items()
                 if any(k.startswith(p) for p in wanted)}
    tiers = TIERS
    if args.tiers:
        tiers = {k: TIERS[k] for k in args.tiers.split(",")}

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=PROJECT_ROOT).stdout.strip()
    print(f"soar frame bench @ {rev} on {gpu_name()}")
    print(f"output {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}, "
          f"{args.frames} frames/view, {args.warmup} warmup")

    level = build_level()
    all_rows = []
    for tier_name, tier in tiers.items():
        rows = time_tier(level, tier_name, tier, views,
                         args.frames, args.warmup)
        all_rows.extend((tier_name, *r) for r in rows)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_file = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a") as f:
        if new_file:
            f.write("# soar per-frame benchmark results\n")
        f.write(f"\n## {stamp} — {rev}"
                + (f" — {args.label}" if args.label else "") + "\n\n")
        f.write(f"GPU: {gpu_name()} · output {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}"
                f" · {args.frames} frames/view\n\n")
        f.write("| tier | view | render size | ms/frame | fps |\n")
        f.write("|------|------|-------------|----------|-----|\n")
        for tier_name, view_name, rw, rh, ms in all_rows:
            f.write(f"| {tier_name} | {view_name} | {rw}x{rh} "
                    f"| {ms:.3f} | {1000.0 / ms:.1f} |\n")
    print(f"appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
