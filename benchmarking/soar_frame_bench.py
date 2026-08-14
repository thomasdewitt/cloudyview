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
# Max is absent on purpose: it is High run 8 times per presented frame, so
# its per-frame cost is 8x the "high" row and nothing here would be learned
# by measuring it. The "hold" rows are the top of each tier's hold ladder —
# High's sampling at the scale that tier is allowed to converge to.
TIERS = {
    "high":   {"render_scale": 1.0,   "step_factor": 2.0,
               "light_step_factor": 2.0,  "max_light_steps": 512},
    "medium": {"render_scale": 0.60,  "step_factor": 2.5,
               "light_step_factor": 4.0,  "max_light_steps": 512},
    "low":    {"render_scale": 0.30,  "step_factor": 3.0,
               "light_step_factor": 8.0,  "max_light_steps": 512},
    "minimal": {"render_scale": 0.125, "step_factor": 4.0,
               "light_step_factor": 12.0, "max_light_steps": 512},
    "hold_low": {"render_scale": 0.75, "step_factor": 2.0,
               "light_step_factor": 2.0,  "max_light_steps": 512},
    "hold_minimal": {"render_scale": 0.50, "step_factor": 2.0,
               "light_step_factor": 2.0,  "max_light_steps": 512},
}

# The frozen judge views this repo already uses (tests/conftest.py), the
# three that stress different regimes: thick backlit cloud, sky-dominated
# overview, and the ocean LOD path.
VIEWS = {
    "v1_thick_backlit":  {"camera_position": [-0.1, -0.2, 0.1],  "azimuth": 20,  "elevation": 8,   "fov": 70},
    "v4_overview_south": {"camera_position": [0.0, -1.0, 0.7],   "azimuth": 0,   "elevation": 2,   "fov": 90},
    "v8_ocean_lod":      {"camera_position": [0.0, 0.0, 3.0],    "azimuth": 180, "elevation": -55, "fov": 70},
}

# The default is small enough to iterate on. Pass --output for the numbers
# that decide anything: below ~1 Mpixel the frame stops being about the march
# and the per-frame CPU/driver floor dominates, which is exactly the regime
# where a tier ratio measured here would lie (Minimal can measure SLOWER than
# Low at 960x540 — it did, 2026-08-14).
OUTPUT_SIZE = (960, 540)
SUN = {"sun_azimuth": 20.0, "sun_elevation": 55.0}


def build_level(data_file=DATA_FILE, ice_file=None, fallback_units=None,
                zcrop=True):
    import cloudyview as cv
    from cloudyview import optical_depth
    from cloudyview.witness import (
        ICE_NEGLIGIBLE_G_KG, RE_ICE_UM, RE_LIQUID_UM, NestedLevel,
        _volume_aabb, crop_empty_z,
    )

    field = cv.load(str(data_file), ice=str(ice_file) if ice_file else None,
                    fallback_units=fallback_units)
    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < ICE_NEGLIGIBLE_G_KG:
        iwc = None
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=RE_LIQUID_UM, iwc=iwc, re_ice=RE_ICE_UM)
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)
    bmin, bmax = _volume_aabb(field)
    # The crop is the feature under test as often as it is a fixed part of the
    # pipeline, so it is switchable here even though the app applies it always.
    if zcrop:
        sigma, z, (lo, hi) = crop_empty_z(sigma, field.z)
        sigma = np.ascontiguousarray(sigma)
        source = np.asarray(field.z).size
        if hi - lo + 1 < source:
            bmin[2] = z.min() - 0.5 * abs(z[1] - z[0])
            bmax[2] = z.max() + 0.5 * abs(z[-1] - z[-2])
            print(f"z-crop: planes {lo}-{hi} of {source} "
                  f"({100 * (1 - (hi - lo + 1) / source):.0f}% empty)")
    return NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name=data_file.stem)


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
    parser.add_argument("--output", type=str, default=None,
                        help="output size WxH (default 960x540); use "
                             "2560x1440 for tier cost ratios")
    parser.add_argument("--frames", type=int, default=64,
                        help="timed frames per view (default 64)")
    parser.add_argument("--warmup", type=int, default=8,
                        help="untimed warmup frames per view (default 8)")
    parser.add_argument("--views", type=str, default=None,
                        help="comma-separated prefixes, e.g. v1,v8")
    parser.add_argument("--tiers", type=str, default=None,
                        help="comma-separated tier names, e.g. high,minimal")
    parser.add_argument("--label", type=str, default="",
                        help="note recorded with the results (e.g. git rev)")
    # Which field is benched is not a detail: an optimization that skips empty
    # space is worth what the field's emptiness is worth. TWPICE is 9.9%
    # occupied by voxel and 29% by 8^3 brick; a STEAM parent is 0.22% and
    # 0.79%. Timing a skip on TWPICE alone would measure the one regime where
    # it cannot pay.
    parser.add_argument("--field", type=Path, default=DATA_FILE,
                        help=f"netCDF field to render (default {DATA_FILE.name})")
    parser.add_argument("--ice", type=Path, default=None,
                        help="separate netCDF file with the ice variable "
                             "(SAM LPT one-variable-per-file style)")
    parser.add_argument("--no-zcrop", action="store_true",
                        help="render the file's whole z extent, including the "
                             "empty sky the app would crop away (for A/B)")
    parser.add_argument("--fallback-units", type=str, default=None,
                        help="units to assume when the file's condensate "
                             "variable carries no units attribute (SAM: g/kg)")
    args = parser.parse_args()
    if not args.field.exists():
        raise SystemExit(f"no such field: {args.field}")
    if args.ice is not None and not args.ice.exists():
        raise SystemExit(f"no such ice file: {args.ice}")

    views = VIEWS
    if args.views:
        wanted = args.views.split(",")
        views = {k: v for k, v in VIEWS.items()
                 if any(k.startswith(p) for p in wanted)}
    global OUTPUT_SIZE
    if args.output:
        OUTPUT_SIZE = tuple(int(v) for v in args.output.lower().split("x"))
    tiers = TIERS
    if args.tiers:
        tiers = {k: TIERS[k] for k in args.tiers.split(",")}

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True,
                         cwd=PROJECT_ROOT).stdout.strip()
    print(f"soar frame bench @ {rev} on {gpu_name()}")
    print(f"output {OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}, "
          f"{args.frames} frames/view, {args.warmup} warmup")

    level = build_level(args.field, args.ice, args.fallback_units,
                        zcrop=not args.no_zcrop)
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
                f" · {args.frames} frames/view"
                f" · field {args.field.name}\n\n")
        f.write("| tier | view | render size | ms/frame | fps |\n")
        f.write("|------|------|-------------|----------|-----|\n")
        for tier_name, view_name, rw, rh, ms in all_rows:
            f.write(f"| {tier_name} | {view_name} | {rw}x{rh} "
                    f"| {ms:.3f} | {1000.0 / ms:.1f} |\n")
    print(f"appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
