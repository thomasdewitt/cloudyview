#!/usr/bin/env python
"""Kernel ablations of the soar march: the most each piece of fs_main could
save, and whether a packed-clearance decode costs anything by itself.

    uv run --extra dev python tools/soar_ablate.py --demo twpice
    uv run --extra dev python tools/soar_ablate.py --demo congestus \\
        --tiers low,high --variants no_premarch,clearance_ctl --repeats 7
    uv run --extra dev python tools/soar_ablate.py --demo marine-congestus-coarse \\
        --scene city --variants uplight_const

Each variant is the module text with one patch (soar_variants.patch, which
refuses an anchor that has moved); the tree's shader is not touched. None of
them is meant to look right. They are ceilings and controls:

  no_premarch    premarch_tau_ahead never called; tau_total stays 0
  no_gradient    sigma_gradient never called at cloud samples; grad = 0
  ocean_const    ocean_shade replaced by a constant colour; where the sea is,
                 and what is composited over it, unchanged
  ice_const      ice_on a compile-time false instead of u.ice_mode > 0.5
  clearance_ctl  the control for a packed-clearance scheme: the view march
                 decodes `raw = level.sigma; sigma = max(raw, 0.0)` and
                 widens its step with `dt = max(dt, -raw * 1.0)` before the
                 existing end clamps. The texture is today's, nonnegative, so
                 no jump ever happens: anything it costs is the decode's.
  uplight_const  (city scene only) city_uplight_probe_tau replaced by 1.0
  stock_again    the unpatched module compiled a second time: the A/A
                 control. Its difference from stock is the noise floor any
                 other difference has to clear.

Every tier flies as the browser flies it (render scale, step factors, light
cache, haze, the two LOD angles; soar_variants.tier_look), over the demo's
own periodicity and sun. Day views are soar_bench's five plus five
flight-like views placed from the field's cloud levels
(soar_variants.demo_views); the city scene uses night_city_harness's views
under its moon, over the field it is given (cyberpunk mode flies congestus
or marine-congestus-coarse over the city).

Timing. For each view every arm is warmed, then timed --repeats times, in
an order rotated each repeat so that a slow spell of the GPU falls on all
arms alike. The reported number is the fastest repeat. Primary: the march
pass's GPU time from timestamp queries (the device's timestamp-query
feature). Without that feature, or when the timestamps fail the sanity
check below, the table falls back to wall clock, which is soar_bench's
number: render(frames) / frames including one read-back of the image.

Sanity check of the timestamps. WebGPU defines them in nanoseconds. Each
timed render's GPU total must not exceed its wall-clock time; if any does,
the timestamps are not in nanoseconds on this backend and the table uses
wall clock (say --gpu-scale to correct them instead, if the factor is known).

Running it on an Apple-silicon Mac (wgpu on Metal), from a fresh checkout:

    git fetch origin perf-census && git checkout perf-census   # or the commit
    uv sync --extra dev
    # the baked demos: web/soar/demos is a symlink into the website's tree;
    # if it is not there, point --demo-dir at a directory holding
    # <demo>/meta.json and <demo>/volume.bin.gz
    uv run --extra dev python tools/soar_ablate.py --demo marine-congestus-coarse \\
        --tiers minimal,low,medium,high --out ablate_m1.jsonl

  * Close the browser and anything else drawing: on a Mac the GPU is also
    the window server's, and its work lands in these timings.
  * The render is banded (SoarRenderer.render) so that no single submit runs
    long enough for the macOS GPU watchdog; that holds here too.
  * Memory: the field is uploaded once as half floats (congestus is about
    3.2 GB, marine-congestus-coarse 0.4 GB) plus a light cache of an eighth
    of that and a few image targets.
  * The first line printed says whether GPU timestamps are in use. Metal
    exposes timestamp-query on Apple GPUs; if it does not here, every number
    is wall clock and only differences larger than stock_again's mean much.

Nothing here is specific to one machine: the device is whatever wgpu's
high-performance adapter is, and the paths are relative to the repository.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soar_variants as sv                                    # noqa: E402

DAY_VARIANTS = ("no_premarch", "no_gradient", "ocean_const", "ice_const",
                "clearance_ctl")
CITY_VARIANTS = ("uplight_const",)
CONTROL = "stock_again"


def ablate(src: str, name: str) -> str:
    p = sv.patch
    if name == CONTROL:
        return src + "\n// A/A control: the stock module, compiled again.\n"
    if name == "no_premarch":
        return p(src, """\
    if (t_near >= 0.0 && t_near < t_far) {
        tau_total = premarch_tau_ahead(
            u.cam_origin, dir, t_near, min(t_far, t_surface), ahead_jitter
        );
    }
""", "    // ablation: no pre-march; tau_total stays 0.\n", what=name)
    if name == "no_gradient":
        return p(src, "                let grad_conf_v = sigma_gradient(p, sigma, t, "
                      "level.in_nest);\n",
                 "                let grad_conf_v = vec4<f32>(0.0);  // ablation\n",
                 what=name)
    if name == "ocean_const":
        return p(src, """\
                    * ocean_shade(
                        ocean_hit, dir, sun,
                        sun_cone_dir(sun, sun_frame, sun_cone_seed,
                                     penumbra_tan),
                        t_ocean, shadow_jitter),
""", "                    * vec3<f32>(0.03, 0.05, 0.07),  // ablation\n", what=name)
    if name == "ice_const":
        return p(src, "    let ice_on = u.ice_mode > 0.5;\n",
                 "    let ice_on = false;  // ablation\n", what=name)
    if name == "clearance_ctl":
        src = p(src, """\
            was_in_nest = level.in_nest;
            let sigma = level.sigma;
""", """\
            was_in_nest = level.in_nest;
            // ablation: the packed-clearance decode, on a field with no
            // negative texel.
            let raw = level.sigma;
            let sigma = max(raw, 0.0);
""", what=name + " decode")
        return p(src, """\
            var dt = step_dt_for_sigma(
                sigma, max(level.dt_view, t * u.view_lod_tan)
            );
            if (t + dt > t_far) {
""", """\
            var dt = step_dt_for_sigma(
                sigma, max(level.dt_view, t * u.view_lod_tan)
            );
            dt = max(dt, -raw * 1.0);
            if (t + dt > t_far) {
""", what=name + " step")
    if name == "uplight_const":
        return p(src, "                let tau_dn = city_uplight_probe_tau(p, "
                      "probe_jitter);\n",
                 "                let tau_dn = 1.0;  // ablation\n", what=name)
    raise SystemExit(f"unknown variant {name!r}")


def city_views(level, tier_presets):
    from night_city_harness import (CITY_VIEWS, MOON_AZIMUTH, MOON_ELEVATION,
                                    NIGHT_EXPOSURE)
    lo, hi = level.bmin, level.bmax
    return {f"city_{n}": dict(
        camera_position=[float(lo[0] + v["xy"][0] * (hi[0] - lo[0])),
                         float(lo[1] + v["xy"][1] * (hi[1] - lo[1])), float(v["z"])],
        azimuth=v["azimuth"], elevation=v["elevation"], fov=v["fov"],
        sun_azimuth=MOON_AZIMUTH, sun_elevation=MOON_ELEVATION,
        exposure=NIGHT_EXPOSURE) for n, v in CITY_VIEWS.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True,
                    help="a baked demo: a name under --demo-dir, or a directory")
    ap.add_argument("--demo-dir", default=None,
                    help="where the baked demos are (default web/soar/demos)")
    ap.add_argument("--scene", default="day", choices=("day", "city"))
    ap.add_argument("--tiers", default="minimal,low,medium,high")
    ap.add_argument("--variants", default="",
                    help="comma-separated; default all for the scene. "
                         "stock and stock_again always run.")
    ap.add_argument("--views", default="", help="comma-separated; default all")
    ap.add_argument("--output", default="2560x1440",
                    help="the window; each tier renders at its scale of it")
    ap.add_argument("--frames", type=int, default=8, help="frames per timing")
    ap.add_argument("--repeats", type=int, default=5,
                    help="timings per arm and view; the fastest is reported")
    ap.add_argument("--gpu-scale", type=float, default=1.0,
                    help="factor on the raw timestamp differences, if this "
                         "backend's are known not to be nanoseconds")
    ap.add_argument("--out", default=None, help="JSON lines, appended")
    args = ap.parse_args()

    city = args.scene == "city"
    surface = "city" if city else "ocean"
    allowed = CITY_VARIANTS + ("no_premarch", "no_gradient", "ice_const",
                               "clearance_ctl") if city else DAY_VARIANTS
    names = [v for v in (args.variants.split(",") if args.variants
                         else (CITY_VARIANTS if city else DAY_VARIANTS)) if v]
    for v in names:
        if v not in allowed:
            raise SystemExit(f"variant {v!r} does not apply to the {args.scene} "
                             f"scene; choose from {', '.join(allowed)}")
    presets = sv.flight_presets()
    out_w, out_h = (int(v) for v in args.output.lower().split("x"))
    adapter, device = sv.make_device(timestamps=True, float32_filterable=False)
    info = adapter.info
    level, meta = sv.load_demo(args.demo, args.demo_dir)
    periodic = bool(meta["periodic"])
    stock = sv.shader.assemble(surface)
    sources = {"stock": stock, CONTROL: ablate(stock, CONTROL)}
    for v in names:
        sources[v] = ablate(stock, v)
    r = sv.VariantRenderer(sources, device=device, periodic=periodic,
                           surface=surface)
    for v, s in sources.items():
        if v != "stock":
            r.add_variant(v, s)
    r.upload_volume(level.sigma)
    if city:
        views = city_views(level, presets)
    else:
        views, _ = sv.demo_views(level, meta, periodic)
    if args.views:
        views = {k: v for k, v in views.items() if k in args.views.split(",")}
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sv.REPO,
                         capture_output=True, text=True).stdout.strip()
    print(f"soar ablations at {rev} on {info.get('device', '?')} "
          f"({info.get('backend_type', '?')}): {level.name}, {args.scene}, "
          f"window {out_w}x{out_h}, fastest of {args.repeats} x {args.frames} "
          f"frames; GPU timestamps "
          f"{'on' if r.timestamps else 'OFF (wall clock only)'}", flush=True)

    arms = [sv.Arm("stock", "stock"), sv.Arm(CONTROL, CONTROL)] + [
        sv.Arm(v, v) for v in names]
    out = open(args.out, "a") if args.out else None
    totals = {}
    ts_ok = r.timestamps
    for tier in sorted(args.tiers.split(","), key=sv.TIER_ORDER.index):
        look = sv.tier_look(presets, tier, city=city)
        extra = ({"city_roof_density": float(presets[tier]["cityRoofDensity"])}
                 if city else {})
        state = sv.tier_scene(r, level, presets, tier, periodic)
        size = sv.render_size(presets, tier, out_w, out_h)
        print(f"\n{tier}: {size[0]}x{size[1]}, haze {look['haze']:.4g}, view LOD "
              f"{look['view_step_lod_degrees']:.3g} deg, light LOD "
              f"{look['light_march_lod_degrees']:.3g} deg", flush=True)
        head = "".join(f"{a.label:>15s}" for a in arms[1:])
        print(f"  {'view':20s}{'stock ms':>10s}{head}   (change vs stock)", flush=True)
        for name, vk in views.items():
            view = sv.make_view(vk, look, (out_w, out_h), size, **extra)
            res = sv.time_variants(r, state, view, arms, frames=args.frames,
                                   repeats=args.repeats)
            for a in res.values():
                a["gpu_ms"] = [None if g is None else g * args.gpu_scale
                               for g in a["gpu_ms"]]
                for g, w in zip(a["gpu_ms"], a["wall_ms"]):
                    if g is not None and g > 1.05 * w:
                        ts_ok = False
            summ = {k: {m: sv.summarize(v[m]) for m in v} for k, v in res.items()}
            if out:
                out.write(json.dumps(dict(
                    kind="ablate", rev=rev, device=info.get("device"),
                    backend=info.get("backend_type"), demo=level.name,
                    scene=args.scene, tier=tier, view=name, size=list(size),
                    output=[out_w, out_h], look=look, frames=args.frames,
                    repeats=args.repeats, when=time.strftime("%Y-%m-%d %H:%M:%S"),
                    samples=res, summary=summ)) + "\n")
                out.flush()
            metric = "gpu_ms" if ts_ok else "wall_ms"
            base = summ["stock"][metric]["min"]
            cells = "".join(
                f"{100 * (summ[a.label][metric]['min'] / base - 1):+14.1f}%"
                for a in arms[1:])
            sp = max(summ[a.label][metric]["spread"] or 0 for a in arms)
            print(f"  {name:20s}{base:10.3f}{cells}   spread<= {100 * sp:.0f}%",
                  flush=True)
            for a in arms:
                t = totals.setdefault((tier, a.label), {"gpu_ms": 0.0, "wall_ms": 0.0})
                for m in ("gpu_ms", "wall_ms"):
                    v = summ[a.label][m]["min"]
                    t[m] = None if (v is None or t[m] is None) else t[m] + v
    metric = "gpu_ms" if ts_ok else "wall_ms"
    if r.timestamps and not ts_ok:
        print("\nGPU timestamps exceeded wall-clock time on some render: they "
              "are not nanoseconds on this backend. The summary uses wall "
              "clock; rerun with --gpu-scale if the factor is known.")
    print(f"\nSummary ({'march-pass GPU time' if metric == 'gpu_ms' else 'wall clock'}"
          f", summed over the views): change vs stock")
    print(f"  {'tier':10s}{'stock ms':>10s}" + "".join(f"{a.label:>15s}" for a in arms[1:]))
    for tier in sorted(args.tiers.split(","), key=sv.TIER_ORDER.index):
        base = totals[(tier, "stock")][metric]
        print(f"  {tier:10s}{base:10.2f}" + "".join(
            f"{100 * (totals[(tier, a.label)][metric] / base - 1):+14.1f}%"
            for a in arms[1:]))
    if out:
        out.close()


if __name__ == "__main__":
    main()
