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
)
from cloudyview.witness import OCEAN_REFLECTANCE, NestedLevel  # noqa: E402
from soar_frame_bench import DATA_FILE, TIERS, VIEWS, build_level  # noqa: E402
from soar_shader_ab import source_for                       # noqa: E402

OUTPUT = (2560, 1440)
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
    """GPU ms of `frames` consecutive ray passes, one timestamp pair each."""
    import wgpu
    device = renderer.device
    targets = renderer._ensure_targets(view.render_size)
    bind_group = renderer._ensure_bind_group()
    sample_view = targets["sample"].create_view()
    for i in range(frames):
        pass_view = ViewState(**{**view.__dict__, "frame_index": i,
                                 "subpixel": i >= 1})
        device.queue.write_buffer(renderer._uniform_buf, 0,
                                  pack_uniforms(state, pass_view).tobytes())
        enc = device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": sample_view, "load_op": "clear",
                                "store_op": "store",
                                "clear_value": (0.0, 0.0, 0.0, 1.0)}],
            timestamp_writes={"query_set": query_set,
                              "beginning_of_pass_write_index": 2 * i,
                              "end_of_pass_write_index": 2 * i + 1})
        rp.set_pipeline(renderer._ray_pipeline_for(view))
        rp.set_bind_group(0, bind_group)
        # Group 1: the surface pre-pass's target, a stand-in in these day
        # scenes (the march reads it only under CITY or DESERT).
        rp.set_bind_group(1, targets["surface_group"])
        rp.draw(3)
        rp.end()
        device.queue.submit([enc.finish()])
    enc = device.create_command_encoder()
    enc.resolve_query_set(query_set, 0, 2 * frames, query_buf, 0)
    readback = device.create_buffer(
        size=16 * frames,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc.copy_buffer_to_buffer(query_buf, 0, readback, 0, 16 * frames)
    device.queue.submit([enc.finish()])
    readback.map_sync(wgpu.MapMode.READ)
    ts = np.frombuffer(bytearray(readback.read_mapped()), np.uint64)
    readback.unmap()
    ts = ts.astype(np.float64)
    return (ts[1::2] - ts[0::2]) * 1e-6          # WebGPU timestamps are ns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", required=True,
                    help="name=spec (tree, git:REV, or a .wgsl path); the "
                         "first is the baseline the ratios divide by")
    ap.add_argument("--cases", default="twpice,rce,dycoms")
    ap.add_argument("--demos", type=Path, default=REPO / "dist" / "soar" / "demos",
                    help="prebaked demo directory (default: dist/soar/demos)")
    ap.add_argument("--views", default="v1,v4,v8")
    ap.add_argument("--tiers", default="low,minimal")
    ap.add_argument("--frames", type=int, default=16,
                    help="timed passes per block")
    ap.add_argument("--rounds", type=int, default=3,
                    help="round-robin blocks per variant; the first is warmup")
    args = ap.parse_args()

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_features=["timestamp-query"])
    query_set = device.create_query_set(type="timestamp", count=2 * args.frames)
    query_buf = device.create_buffer(
        size=16 * args.frames,
        usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)

    variants = [v.split("=", 1) for v in args.variant]
    names = [n for n, _ in variants]
    print("ray-pass GPU ms, median over "
          f"{args.frames * max(args.rounds - 1, 1)} passes; output "
          f"{OUTPUT[0]}x{OUTPUT[1]}")
    for name, spec in variants:
        print(f"  {name} = {spec}")
    header = f"{'case':10s} {'tier':8s} {'view':4s} " + " ".join(
        f"{n:>9s}" for n in names)
    if len(names) > 1:
        header += "  " + " ".join(f"{n + '/' + names[0]:>12s}" for n in names[1:])
    print(header)

    for case in args.cases.split(","):
        level, sun = load_case(case, args.demos)
        renderers = []
        for _, spec in variants:
            r = SoarRenderer(device=device, periodic=True, nested=False,
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
                periodic=True, ocean_reflectance=OCEAN_REFLECTANCE)
            scale = tier["render_scale"]
            size = (max(1, int(OUTPUT[0] * scale + 0.5)),
                    max(1, int(OUTPUT[1] * scale + 0.5)))
            baked = False
            for view_name in args.views.split(","):
                v = SHORT_VIEWS[view_name]
                pos = camera_world_origin(v["camera_position"],
                                          level.bmin, level.bmax)
                view = ViewState(
                    camera_position=[float(p) for p in pos],
                    azimuth=v["azimuth"], elevation=v["elevation"],
                    fov=v["fov"], output_size=OUTPUT, render_size=size,
                    light_cache=tier["light_cache"],
                    sky_probe=tier["sky_probe"], **sun)
                if tier["light_cache"] and not baked:
                    for r in renderers:
                        r.bake_light_cache(state, view,
                                           divisor=LIGHT_CACHE_DIVISOR)
                    baked = True
                samples = [[] for _ in renderers]
                for rnd in range(args.rounds):
                    for k, r in enumerate(renderers):
                        d = pass_times_ms(r, state, view, args.frames,
                                          query_set, query_buf)
                        if rnd > 0 or args.rounds == 1:
                            samples[k].extend(d)
                meds = [float(np.median(s)) for s in samples]
                line = f"{case:10s} {tier_name:8s} {view_name:4s} " + " ".join(
                    f"{m:9.3f}" for m in meds)
                if len(meds) > 1:
                    line += "  " + " ".join(f"{m / meds[0]:12.3f}"
                                            for m in meds[1:])
                print(line, flush=True)
        for r in renderers:
            r._vol_tex.destroy()


if __name__ == "__main__":
    main()
