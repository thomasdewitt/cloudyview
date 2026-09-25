#!/usr/bin/env python
"""GPU time of the soar ray pass, several shader revisions interleaved.

soar_frame_bench.py times whole frames from the host's side of the queue, so
at small render sizes on a fast card it measures python and the driver as
much as the shader (the 5080 renders minimal's frame in a tenth of the time
the bench reports for it). This times the fs_main pass alone, with GPU
timestamp queries at the pass's begin and end, and runs the revisions under
test in round-robin blocks on one device so that clock drift, thermals and a
neighbour's work land on all of them alike. It is the instrument for "is
this revision faster", which is the question an optimization asks, where
soar_frame_bench answers "how fast is the app".

    uv run python tools/benchmarking/soar_pass_timer.py \\
        --variant base=git:HEAD --variant new=tree
    uv run python tools/benchmarking/soar_pass_timer.py \\
        --variant base=git:HEAD --variant new=tree --cases rce --tiers low

A variant is name=spec, spec as tools/soar_shader_ab.py takes it: `tree`,
`git:<rev>`, or a path to a .wgsl file (a feature-ablated scratch copy is
how cost gets attributed to a feature — measurement only).

Cases are the TWP-ICE judge field and the prebaked demos the site ships
(dist/soar/demos, or --demos), each with its own sun. Output size is
2560x1440, so the tier render sizes are the ones the 2026-08-19 M1 rows were
measured at (low 768x432, minimal 320x180).

Clock-lock handicap (2026-09-23): the 5080 locked to 180 MHz graphics and
405 MHz memory (`nvidia-smi -lgc 180,180 -lmc 405,405`, the floor this driver
allows) runs these frames at ~0.65-0.8x the M1's time, and the memory-heavy
DYCOMS v1 view at ~0.4x. It is a proxy for the Mac, not a substitute; always
apply and reset the lock inside one tools/gpulock command with a trap.

One GPU, many sessions (tools/benchmarking/gpu_idle.py has the policy). The
timer starts only when no other process is on the card and checks again at
the end: a neighbour there exits non-zero, CONTAMINATED. The desktop drawing
(gnome-shell, the Claude app — Thomas's own screen) is a WARNING instead,
and the ratios stand only if the null control holds within NULL_TOLERANCE
and the timed rounds agree within it too; the summary says which. So give
every comparison a null: the baseline again under another name
(--variant base=git:HEAD --variant null=git:HEAD --variant new=tree). Under
the clock lock with the desktop drawing, absolute ms are pessimistic.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "benchmarking"))

from cloudyview.soar_host import (                          # noqa: E402
    SceneState, SoarRenderer, ViewState, camera_world_origin, pack_uniforms,
    DESERT_LITE_TIERS, desert_scene_fields, surface_cell_m, surface_period_m,
)
from cloudyview.witness import OCEAN_REFLECTANCE, NestedLevel  # noqa: E402
from soar_frame_bench import DATA_FILE, TIERS, VIEWS, build_level  # noqa: E402
from soar_shader_ab import source_for                       # noqa: E402
from gpu_idle import check_quiet, clock_mode, require_idle  # noqa: E402
from night_city_harness import (CITY_VIEWS, MOON_AZIMUTH,     # noqa: E402
                                MOON_ELEVATION, NIGHT_EXPOSURE)

OUTPUT = (2560, 1440)
# A null control within this (and rounds agreeing within it) is what lets
# the ratios stand when the desktop drew during the run (gpu_idle.py).
NULL_TOLERANCE = 0.01
LIGHT_CACHE_DIVISOR = 2       # web/soar/constants.js
SHORT_VIEWS = {name.split("_")[0]: v for name, v in VIEWS.items()}


def load_case(name, demos: Path):
    """(level, sun) for the judge field or one prebaked demo."""
    sun = {"sun_azimuth": 20.0, "sun_elevation": 55.0}
    if name == "twpice":
        return build_level(DATA_FILE), sun
    d = demos / name
    meta = json.loads((d / "meta.json").read_text())
    vol = meta["volume"]
    raw = gzip.decompress((d / vol["file"]).read_bytes())
    nx, ny, nz = vol["shape_xyz"]
    sigma = np.frombuffer(raw, dtype=np.float16).reshape(nx, ny, nz)
    level = NestedLevel(sigma=sigma, bmin=np.asarray(vol["bmin"], float),
                        bmax=np.asarray(vol["bmax"], float), name=meta["id"])
    if meta.get("sun"):
        sun = {"sun_azimuth": meta["sun"]["azimuth"],
               "sun_elevation": meta["sun"]["elevation"]}
    return level, sun


def pass_times_ms(renderer, state, view, frames, query_set, query_buf):
    """GPU ms of `frames` consecutive frames as (surface, refine, march)
    arrays, every stage encoded exactly as SoarRenderer.render encodes it:
    the surface pre-pass (fs_surface) — into the first-stage target when the
    surface has a second stage, the final one otherwise — then its
    refinement (fs_surface_refine) into the final surface, then the march
    (fs_main). A stage the scene does not have reads as zeros. One timestamp
    pair per pass actually encoded: resolving a query no pass wrote fails
    validation, and the readback below then never maps."""
    import wgpu
    device = renderer.device
    targets = renderer._ensure_targets(view.render_size)
    bind_group = renderer._ensure_bind_group()
    sample_view = targets["sample"].create_view()
    surface_pipeline = renderer._surface_pipeline_for(view)
    refine_pipeline = renderer._refine_pipeline_for(view)
    surface_view = (targets["surface"].create_view()
                    if targets["surface"] is not None else None)
    plain_view = (targets["plain"].create_view()
                  if targets["plain"] is not None else None)
    stages = []                 # (name, pipeline, group 1, target view)
    if surface_pipeline is not None:
        stages.append(("surface", surface_pipeline, None,
                       plain_view if refine_pipeline is not None
                       else surface_view))
        if refine_pipeline is not None:
            stages.append(("refine", refine_pipeline,
                           targets["refine_group"], surface_view))
    stages.append(("march", renderer._ray_pipeline_for(view),
                   targets["surface_group"], sample_view))
    per = 2 * len(stages)
    for i in range(frames):
        pass_view = ViewState(**{**view.__dict__, "frame_index": i,
                                 "subpixel": i >= 1})
        device.queue.write_buffer(renderer._uniform_buf, 0,
                                  pack_uniforms(state, pass_view).tobytes())
        enc = device.create_command_encoder()
        for k, (name, pipeline, group1, target) in enumerate(stages):
            clear = ((0.0, 0.0, 0.0, 1.0) if name == "march"
                     else (0.0, 0.0, 0.0, 1e30))
            rp = enc.begin_render_pass(
                color_attachments=[{"view": target, "load_op": "clear",
                                    "store_op": "store",
                                    "clear_value": clear}],
                timestamp_writes={"query_set": query_set,
                                  "beginning_of_pass_write_index": per * i + 2 * k,
                                  "end_of_pass_write_index": per * i + 2 * k + 1})
            rp.set_pipeline(pipeline)
            rp.set_bind_group(0, bind_group)
            if group1 is not None:
                rp.set_bind_group(1, group1)
            rp.draw(3)
            rp.end()
        device.queue.submit([enc.finish()])
    enc = device.create_command_encoder()
    enc.resolve_query_set(query_set, 0, per * frames, query_buf, 0)
    readback = device.create_buffer(
        size=8 * per * frames,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc.copy_buffer_to_buffer(query_buf, 0, readback, 0, 8 * per * frames)
    device.queue.submit([enc.finish()])
    readback.map_sync(wgpu.MapMode.READ)
    ts = np.frombuffer(bytearray(readback.read_mapped()), np.uint64)
    readback.unmap()
    ts = ts.astype(np.float64).reshape(frames, per)
    out = {name: np.zeros(frames) for name in ("surface", "refine", "march")}
    for k, (name, *_rest) in enumerate(stages):
        out[name] = (ts[:, 2 * k + 1] - ts[:, 2 * k]) * 1e-6   # ns -> ms
    return out["surface"], out["refine"], out["march"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True,
                    help="name=spec (tree, git:REV, or a .wgsl path); the "
                         "first is the baseline the ratios divide by")
    ap.add_argument("--cases", default="twpice,rce,dycoms")
    ap.add_argument("--demos", type=Path, default=REPO / "dist" / "soar" / "demos",
                    help="prebaked demo directory (default: dist/soar/demos)")
    ap.add_argument("--views", default="v1,v4,v8",
                    help="the judge views (v1..v8); with --surface city also "
                         "the night-city harness's (aerial, base, street, "
                         "horizon, moonrise, moon_check), rendered as it "
                         "renders them: under the moon, at night exposure")
    ap.add_argument("--tiers", default="low,minimal")
    ap.add_argument("--frames", type=int, default=16,
                    help="timed passes per block")
    ap.add_argument("--surface", default="ocean",
                    choices=("ocean", "desert", "city"),
                    help="what the world stands on; desert and city time "
                         "their surface pre-pass (fs_surface) as its own "
                         "column beside the march (fs_main)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="round-robin blocks per variant; the first is warmup")
    args = ap.parse_args()

    # One GPU, many sessions: start only on an idle card (gpu_idle.py).
    warnings = require_idle()
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_features=["timestamp-query"])
    query_set = device.create_query_set(type="timestamp", count=6 * args.frames)
    query_buf = device.create_buffer(
        size=48 * args.frames,
        usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)

    variants = [v.split("=", 1) for v in args.variant]
    names = [n for n, _ in variants]
    print("ray-pass GPU ms, median over "
          f"{args.frames * max(args.rounds - 1, 1)} passes; output "
          f"{OUTPUT[0]}x{OUTPUT[1]}")
    for name, spec in variants:
        print(f"  {name} = {spec}")
    print(f"  clocks: {clock_mode()}")
    header = f"{'case':10s} {'tier':8s} {'view':8s} " + " ".join(
        f"{n:>9s}" for n in names)
    if len(names) > 1:
        header += "  " + " ".join(f"{n + '/' + names[0]:>12s}" for n in names[1:])
    print(header)
    # Nulls: a variant whose spec repeats an earlier one's.
    specs = [spec for _, spec in variants]
    nulls = [(specs.index(sp), j) for j, sp in enumerate(specs)
             if specs.index(sp) != j]
    null_dev, round_dev = 0.0, 0.0

    for case in args.cases.split(","):
        level, sun = load_case(case, args.demos)
        renderers = []
        for _, spec in variants:
            r = SoarRenderer(device=device, periodic=True, nested=False,
                             surface=args.surface,
                             shader_source=source_for(spec))
            r.upload_volume(level.sigma)
            renderers.append(r)
        min_voxel = min(level.dx)
        for tier_name in args.tiers.split(","):
            tier = TIERS[tier_name]
            state = SceneState(
                bmin=[float(v) for v in level.bmin],
                bmax=[float(v) for v in level.bmax],
                dt_view=min_voxel * tier["step_factor"],
                dt_light=min_voxel * tier["light_step_factor"],
                periodic=True, ocean_reflectance=OCEAN_REFLECTANCE,
                surface=args.surface)
            meta = renderers[0].surface_meta
            state.ocean_fif_dx = surface_cell_m(args.surface, meta)
            state.ocean_tile_extent = float(meta["tile_extent_m"])
            state.ocean_max_lod = int(meta["mips"]) - 1
            state.surface_period_m = surface_period_m(args.surface, meta)
            if args.surface == "desert":
                for key, value in desert_scene_fields(meta).items():
                    setattr(state, key, value)
            scale = tier["render_scale"]
            size = (max(1, int(OUTPUT[0] * scale + 0.5)),
                    max(1, int(OUTPUT[1] * scale + 0.5)))
            baked = None       # the light the cache was baked under
            for view_name in args.views.split(","):
                if args.surface == "city" and view_name in CITY_VIEWS:
                    # tools/night_city_harness.py's view: xy a fraction of
                    # the box, z in metres, the moon, night exposure.
                    v = CITY_VIEWS[view_name]
                    fx, fy = v["xy"]
                    pos = (level.bmin[0] + fx * (level.bmax[0] - level.bmin[0]),
                           level.bmin[1] + fy * (level.bmax[1] - level.bmin[1]),
                           v["z"])
                    light = {"sun_azimuth": MOON_AZIMUTH,
                             "sun_elevation": MOON_ELEVATION,
                             "exposure": NIGHT_EXPOSURE}
                else:
                    v = SHORT_VIEWS[view_name]
                    pos = camera_world_origin(v["camera_position"],
                                              level.bmin, level.bmax)
                    light = sun
                view = ViewState(
                    camera_position=[float(p) for p in pos],
                    azimuth=v["azimuth"], elevation=v["elevation"],
                    fov=v["fov"], output_size=OUTPUT, render_size=size,
                    light_cache=tier["light_cache"],
                    sky_probe=tier["sky_probe"],
                    # The simplified desert at the tiers the app flies it.
                    desert_lite=(args.surface == "desert"
                                 and tier_name in DESERT_LITE_TIERS),
                    **light)
                if tier["light_cache"] and baked != light:
                    for r in renderers:
                        r.bake_light_cache(state, view,
                                           divisor=LIGHT_CACHE_DIVISOR)
                    baked = light
                surf = [[] for _ in renderers]
                refine = [[] for _ in renderers]
                march = [[] for _ in renderers]
                per_round = [[] for _ in renderers]
                for rnd in range(args.rounds):
                    for k, r in enumerate(renderers):
                        ds, dr, dm = pass_times_ms(r, state, view, args.frames,
                                                   query_set, query_buf)
                        if rnd > 0 or args.rounds == 1:
                            surf[k].extend(ds)
                            refine[k].extend(dr)
                            march[k].extend(dm)
                            per_round[k].append(float(np.median(
                                np.add(np.add(ds, dr), dm))))
                meds = [float(np.median(np.add(np.add(a, b), c)))
                        for a, b, c in zip(surf, refine, march)]
                for i, j in nulls:
                    null_dev = max(null_dev, abs(meds[j] / meds[i] - 1.0))
                for rounds_k in per_round:
                    if len(rounds_k) > 1:
                        round_dev = max(round_dev,
                                        max(rounds_k) / min(rounds_k) - 1.0)
                line = f"{case:10s} {tier_name:8s} {view_name:8s} " + " ".join(
                    f"{m:9.3f}" for m in meds)
                if len(meds) > 1:
                    line += "  " + " ".join(f"{m / meds[0]:12.3f}"
                                            for m in meds[1:])
                if args.surface != "ocean":
                    line += "   [surface+refine+march " + ", ".join(
                        f"{np.median(a):.3f}+{np.median(b):.3f}"
                        f"+{np.median(c):.3f}"
                        for a, b, c in zip(surf, refine, march)) + "]"
                print(line, flush=True)
        for r in renderers:
            r._vol_tex.destroy()
    warnings += check_quiet()
    null_note = (f"null control within {100 * null_dev:.2f}%" if nulls
                 else "no null control")
    round_note = (f"timed rounds within {100 * round_dev:.2f}%"
                  if args.rounds > 2 else "one timed round")
    print(f"# {null_note}; {round_note}")
    if warnings:
        print("# WARNING, the desktop drew during the run:\n#   "
              + "\n#   ".join(sorted(set(warnings))))
        if (nulls and null_dev <= NULL_TOLERANCE and args.rounds > 2
                and round_dev <= NULL_TOLERANCE):
            print(f"# The null and the rounds hold within "
                  f"{100 * NULL_TOLERANCE:.0f}%: the ratios stand; absolute "
                  f"ms are pessimistic ({clock_mode()}).")
        else:
            sys.exit(f"# Without a null and rounds within "
                     f"{100 * NULL_TOLERANCE:.0f}% the desktop's share cannot "
                     f"be ruled out: discard these numbers (give the run a "
                     f"null variant and --rounds 3 or more).")


if __name__ == "__main__":
    main()
