#!/usr/bin/env python
"""What each quality tier's march costs as flown, and the cost ratio from one
tier to the next: the numbers behind web/soar/constants.js
AUTO_TIER_COST_RATIO_TO_NEXT.

    uv run --extra dev python tools/soar_tier_cost.py --out tier_cost.jsonl
    uv run --extra dev python tools/soar_tier_cost.py --report tier_cost.jsonl --md tier_cost.md

Each tier renders as the browser flies it (soar_variants.tier_look and
tier_scene: render scale of the --output window, step factors, light cache,
pre-march switch, haze, LOD angles), over each demo's own periodicity and
sun, from soar_variants.demo_views' ten views. Two arms:

  stock         the march pass as shipped
  nomarch_live  the same kernel with the march not entered (t_far = t_near
                behind a uniform, so nothing is folded away): the per-pixel
                work outside the march loop (ray setup, sky, the surface)

timed as soar_ablate times them: warmed, then --repeats rounds in an order
rotated each round, the fastest kept. The number is the march pass's GPU
time from timestamp queries. Wall clock per frame is recorded beside it;
it includes this Python host's submit, poll and read-back, which are not
the browser's, so it is context only.

A small render does not fill a large GPU: Minimal at 2560x1440 is 320x180,
57,600 pixels, a few waves on an RTX 5080, so part of its time is latency
that a slower GPU, which the small render does fill, would not multiply.
So, unless --no-saturate, each tier's look is timed again at each of
--saturate-size (stock only) and scaled by its pixel count: the time it
would take if the GPU were busy throughout, the throughput a slower GPU is
limited by. The first size is the report's "throughput". The step LOD at
every tier is the tier's angle, not the pixel's (the pixel floor, a quarter
pixel, is under it at every tier), so the steps per pixel do not change
with render size; what can change is how coherent neighbouring pixels'
fetches are, which is worse at a small render. The second size measures
how far the per-pixel cost moves with resolution while the GPU is full.

The report's third measure, "extrapolated", is an estimate, not a
measurement: the throughput time times that per-pixel penalty raised to
the number of halvings from the first saturate size down to the tier's
own render width, i.e. the penalty per halving between the two saturate
sizes assumed to hold all the way down. It assumes the second size still
fills the GPU; where it does not, part of the penalty is fill, and the
estimate over-corrects (makes the small tiers dearer than they are).

Max is timed as one march per frame, which is what this host's frames are
(the browser's Max today presents eight per frame, sppPerFrame; --tiers
high,max times the single-sample Max beside High).

The report gives, for each step up (low/minimal, medium/low, high/medium,
max/high),
the ratio per view and per demo (summed over the demo's views), and their
mean, median and maximum. The gate in escalateQualityTier has to hold on
the worst field and view, so the maximum is the number that sets it.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soar_variants as sv                                    # noqa: E402
from soar_void_census import nomarch_patch                    # noqa: E402

DEMOS = ("twpice,congestus,dycoms,rce,stratified,rce-dales,rce-sam,"
         "marine-congestus-coarse,utah-convection-coarse")
STEPS = (("minimal", "low"), ("low", "medium"), ("medium", "high"), ("high", "max"))
SPREAD_LIMIT = 0.10


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", default=DEMOS)
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--tiers", default="minimal,low,medium,high")
    ap.add_argument("--views", default="", help="comma-separated; default all")
    ap.add_argument("--output", default="2560x1440")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--saturate-size", default="2560x1440,1280x720",
                    help="comma-separated render sizes; the first is 'throughput'")
    ap.add_argument("--no-saturate", action="store_true")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="JSON lines, appended")
    ap.add_argument("--report", nargs="+", default=None, metavar="JSONL")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    if args.report:
        return report(args.report, args.md)
    if not args.out:
        ap.error("--out is required (unless --report)")

    presets = sv.flight_presets()
    out_w, out_h = (int(v) for v in args.output.lower().split("x"))
    sats = [] if args.no_saturate else [
        tuple(int(v) for v in z.lower().split("x"))
        for z in args.saturate_size.split(",")]
    adapter, device = sv.make_device(timestamps=True, float32_filterable=False)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sv.REPO,
                         capture_output=True, text=True).stdout.strip()
    out = open(args.out, "a")
    stock = sv.shader.assemble("ocean")
    for demo in args.demos.split(","):
        level, meta = sv.load_demo(demo, args.demo_dir)
        periodic = bool(meta["periodic"])
        sources = {"stock": stock, "nomarch_live": nomarch_patch(stock, live=True)}
        r = sv.VariantRenderer(sources, device=device, periodic=periodic,
                               surface="ocean")
        r.add_variant("nomarch_live", sources["nomarch_live"])
        if not r.timestamps:
            raise SystemExit("no timestamp-query on this device")
        r.upload_volume(level.sigma)
        views, _ = sv.demo_views(level, meta, periodic)
        arms = [sv.Arm("stock", "stock"), sv.Arm("nomarch_live", "nomarch_live")]
        for tier in sorted(args.tiers.split(","), key=sv.TIER_ORDER.index):
            look = sv.tier_look(presets, tier)
            state = sv.tier_scene(r, level, presets, tier, periodic)
            size = sv.render_size(presets, tier, out_w, out_h)
            for name, vk in views.items():
                if args.views and name not in args.views.split(","):
                    continue
                view = sv.make_view(vk, look, (out_w, out_h), size)
                res = sv.time_variants(r, state, view, arms, frames=args.frames,
                                       repeats=args.repeats)
                rec = dict(kind="tier_cost", label=args.label, rev=rev, demo=demo,
                           tier=tier, view=name, size=list(size),
                           output=[out_w, out_h], look=look, frames=args.frames,
                           repeats=args.repeats,
                           when=time.strftime("%Y-%m-%d %H:%M:%S"), samples=res)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                g = min(res["stock"]["gpu_ms"])
                line = (f"{demo:24s}{tier:8s}{name:18s} stock {g:8.3f} ms  "
                        f"not entered {min(res['nomarch_live']['gpu_ms']):7.3f} ms  "
                        f"wall {min(res['stock']['wall_ms']):7.2f} ms")
                for sat in sats:
                    vs = sv.make_view(vk, look, (out_w, out_h), sat)
                    rs = sv.time_variants(r, state, vs, arms[:1],
                                          frames=args.frames, repeats=args.repeats)
                    scale = size[0] * size[1] / (sat[0] * sat[1])
                    out.write(json.dumps(dict(
                        kind="tier_cost_sat", label=args.label, rev=rev, demo=demo,
                        tier=tier, view=name, size=list(sat), flown_size=list(size),
                        pixel_scale=scale, output=[out_w, out_h], frames=args.frames,
                        repeats=args.repeats, when=time.strftime("%Y-%m-%d %H:%M:%S"),
                        samples=rs)) + "\n")
                    out.flush()
                    line += (f"  at {sat[0]}x{sat[1]} scaled "
                             f"{min(rs['stock']['gpu_ms']) * scale:8.3f} ms")
                print(line, flush=True)
        del r
    out.close()


def _spread(rec):
    """Worst (slowest - fastest) of repeats 2.. over the stock arm's fastest
    (soar_void_census_report.arm_spread's reading)."""
    full = min(rec["samples"]["stock"]["gpu_ms"])
    return max((max(v["gpu_ms"][1:]) - min(v["gpu_ms"][1:])) / full
               for v in rec["samples"].values())


def _cells(recs, kind):
    """(demo, tier, view) -> the records of that cell that pass the spread
    filter (all of them if none does), and the set of cells none passed."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in recs:
        if r.get("kind") == kind:
            by[(r["demo"], r["tier"], r["view"])].append(r)
    cells, noisy = {}, set()
    for key, rs in by.items():
        ok = [r for r in rs if _spread(r) <= SPREAD_LIMIT]
        if not ok:
            noisy.add(key)
            ok = rs
        cells[key] = ok
    return cells, noisy


def _fastest(rs, arm="stock", metric="gpu_ms"):
    return min(min(r["samples"][arm][metric]) for r in rs)


def _ratios(ms, demos, views):
    """Per view and per demo (summed over its views) ratios for each step,
    from ms[(demo, tier, view)]."""
    pv, pd = {}, {}
    for lo, hi in STEPS:
        step = f"{hi}/{lo}"
        pv[step] = [(ms[(d, hi, v)] / ms[(d, lo, v)], d, v)
                    for d in demos for v in views
                    if (d, lo, v) in ms and (d, hi, v) in ms]
        pd[step] = []
        for d in demos:
            ks = [v for v in views if (d, lo, v) in ms and (d, hi, v) in ms]
            if ks:
                pd[step].append((sum(ms[(d, hi, v)] for v in ks)
                                 / sum(ms[(d, lo, v)] for v in ks), d))
    return pv, pd


def report(paths, md_path=None):
    recs = [json.loads(line) for p in paths for line in open(p) if line.strip()]
    flown_recs, noisy = _cells(recs, "tier_cost")
    sat_sizes = list(dict.fromkeys(tuple(r["size"]) for r in recs
                                   if r.get("kind") == "tier_cost_sat"))
    scaled, noisy_sat = [], set()
    for z in sat_sizes:
        cz, nz = _cells([r for r in recs if tuple(r["size"]) == z], "tier_cost_sat")
        scaled.append({k: _fastest(rs) * rs[0]["pixel_scale"] for k, rs in cz.items()})
        noisy_sat |= {k + (f"{z[0]}x{z[1]}",) for k in nz}
    flown = {k: _fastest(rs) for k, rs in flown_recs.items()}
    live = {k: _fastest(rs, "nomarch_live") for k, rs in flown_recs.items()}
    sat = scaled[0] if scaled else {}
    extra = {}
    if len(scaled) > 1:
        import math
        z0, z1 = sat_sizes[0], sat_sizes[1]
        per_halving = math.log2(z0[0] / z1[0])
        for k, v in sat.items():
            if k in scaled[1] and k in flown_recs:
                w = flown_recs[k][0]["size"][0]
                halvings = math.log2(z0[0] / w)
                extra[k] = v * (scaled[1][k] / v) ** (halvings / per_halving)
    demos = list(dict.fromkeys(k[0] for k in flown))
    views = list(dict.fromkeys(k[2] for k in flown))
    tiers = [t for t in sv.TIER_ORDER if any(k[1] == t for k in flown)]
    steps = [f"{hi}/{lo}" for lo, hi in STEPS if lo in tiers and hi in tiers]
    measures = ([("as flown", flown)] + ([("throughput", sat)] if sat else [])
                + ([("extrapolated", extra)] if extra else []))
    ratios = {name: _ratios(ms, demos, views) for name, ms in measures}

    lines = ["March-pass GPU time, ms per frame, fastest of the repeats (and of "
             "the runs). *as flown*: each tier at its render scale of the "
             "window. *throughput*: each tier's look timed at "
             + (f"{sat_sizes[0][0]}x{sat_sizes[0][1]}" if sat_sizes else "-")
             + " and scaled by the tier's pixel count. *extrapolated*: the "
             "throughput time with the per-pixel penalty measured between the "
             "two saturate sizes carried down to the tier's own size (an "
             "estimate; see the docstring). Ratios are the next tier's time "
             "over this tier's.", ""]
    for name, ns in (("as flown", noisy), ("throughput", noisy_sat)):
        if ns:
            lines += [f"{name}: {len(ns)} cells spread by more than "
                      f"{SPREAD_LIMIT:.0%} in every run, kept anyway (their "
                      "fastest): " + ", ".join("/".join(k) for k in sorted(ns)), ""]

    lines += ["## Ratio to the next tier", "",
              "| step | measure | over | n | mean | median | max | min | max at |",
              "|---|---|---|---|---|---|---|---|---|"]
    out_json = {"summary": [], "per_view": [], "per_demo": [], "per_tier": []}
    for step in steps:
        for name, _ in measures:
            pv, pd = ratios[name]
            for over, xs in (("demos", pd[step]), ("views", pv[step])):
                vals = [x[0] for x in xs]
                m, md, mx, mn = (statistics.mean(vals), statistics.median(vals),
                                 max(vals), min(vals))
                at = max(xs)[1:]
                lines.append(f"| {step} | {name} | {over} | {len(xs)} | {m:.2f} | "
                             f"{md:.2f} | {mx:.2f} | {mn:.2f} | {'/'.join(at)} |")
                out_json["summary"].append(dict(step=step, measure=name, over=over,
                                                n=len(xs), mean=m, median=md,
                                                max=mx, min=mn, max_at=at))
            out_json["per_view"] += [dict(step=step, measure=name, demo=d, view=v,
                                          ratio=x) for x, d, v in pv[step]]
            out_json["per_demo"] += [dict(step=step, measure=name, demo=d, ratio=x)
                                     for x, d in pd[step]]
    lines.append("")

    lines += ["## Per tier", "",
              "Means over every (demo, view). *not entered*: the same kernel "
              "with the march not entered, the per-pixel work outside the march "
              "loop. *above throughput*: 1 - throughput / as flown, the part of "
              "the as-flown time the GPU spends not full. *above extrapolated*: "
              "the same against the extrapolated time, the part that is "
              "neither throughput nor fetch coherence. *wall - all passes*: "
              "this Python host's per-frame time outside the timed passes: its "
              "encoding, its wait on each band before the next is submitted, "
              "the untimed accumulate pass at render size and the read-back. "
              "Not the browser's."
              + ("" if len(sat_sizes) < 2 else
                 f" *px cost {sat_sizes[1][0]}x{sat_sizes[1][1]} / "
                 f"{sat_sizes[0][0]}x{sat_sizes[0][1]}*: the tier's per-pixel "
                 "cost at the smaller saturate size over the larger.") , "",
              "| tier | render size | as flown ms | throughput ms | above throughput | "
              "px cost ratio | extrapolated ms | above extrapolated | "
              "not entered ms | not entered share | wall ms | wall - all passes ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tiers:
        ks = [k for k in flown if k[1] == t]
        f = statistics.mean(flown[k] for k in ks)
        lv = statistics.mean(live[k] for k in ks)
        w = statistics.mean(_fastest(flown_recs[k], metric="wall_ms") for k in ks)
        al = statistics.mean(_fastest(flown_recs[k], metric="gpu_all_ms") for k in ks)
        ksat = [k for k in ks if k in sat]
        st = statistics.mean(sat[k] for k in ksat) if ksat else None
        size = flown_recs[ks[0]][0]["size"]
        st_s = f"{st:.3f}" if st is not None else "-"
        above = f"{100 * (1 - st / f):.0f}%" if st is not None else "-"
        k2 = [k for k in ksat if len(scaled) > 1 and k in scaled[1]]
        pxr = (statistics.mean(scaled[1][k] / scaled[0][k] for k in k2)
               if k2 else None)
        pxr_s = f"{pxr:.3f}" if pxr is not None else "-"
        kx = [k for k in ks if k in extra]
        ex = statistics.mean(extra[k] for k in kx) if kx else None
        ex_s = f"{ex:.3f}" if ex is not None else "-"
        ab_x = f"{100 * (1 - ex / f):.0f}%" if ex is not None else "-"
        lines.append(f"| {t} | {size[0]}x{size[1]} | {f:.3f} | {st_s} | {above} | "
                     f"{pxr_s} | {ex_s} | {ab_x} | {lv:.3f} | {100 * lv / f:.0f}% | "
                     f"{w:.2f} | {w - al:.2f} |")
        out_json["per_tier"].append(dict(tier=t, size=size, flown_ms=f,
                                         throughput_ms=st, px_cost_ratio=pxr,
                                         extrapolated_ms=ex,
                                         not_entered_ms=lv, wall_ms=w,
                                         wall_minus_gpu_all_ms=w - al))
    lines.append("")

    lines += ["## Per demo: mean ms per view, and the ratios (demo sums)", ""]
    for name, ms in measures:
        pv, pd = ratios[name]
        lines += [f"*{name}*", "",
                  "| demo | " + " | ".join(tiers) + " | " + " | ".join(steps) + " |",
                  "|" + "---|" * (1 + len(tiers) + len(steps))]
        for d in demos:
            per = [statistics.mean(ms[(d, t, v)] for v in views if (d, t, v) in ms)
                   for t in tiers]
            rat = [dict((dd, x) for x, dd in pd[st]).get(d) for st in steps]
            lines.append(f"| {d} | " + " | ".join(f"{x:.3f}" for x in per) + " | "
                         + " | ".join("-" if x is None else f"{x:.2f}" for x in rat)
                         + " |")
        lines.append("")

    lines += ["## Per view", "",
              "| demo | view | " + " | ".join(tiers) + " | "
              + " | ".join(f"{st} flown" for st in steps)
              + "".join(" | " + " | ".join(f"{st} {tag}" for st in steps)
                        for tag, ms in (("thru", sat), ("extrap", extra)) if ms)
              + " |",
              "|" + "---|" * (2 + len(tiers) + len(steps) * len(measures))]
    for d in demos:
        for v in views:
            if not all((d, t, v) in flown for t in tiers):
                continue
            row = [f"{flown[(d, t, v)]:.3f}" for t in tiers]
            for name, ms in measures:
                for lo, hi in STEPS:
                    if lo in tiers and hi in tiers:
                        a, b = ms.get((d, lo, v)), ms.get((d, hi, v))
                        row.append("-" if a is None or b is None else f"{b / a:.2f}")
            lines.append(f"| {d} | {v} | " + " | ".join(row) + " |")
    md = "\n".join(lines) + "\n"
    if md_path:
        Path(md_path).write_text(md)
        Path(md_path).with_suffix(".json").write_text(json.dumps(out_json, indent=1))
    print(md)


if __name__ == "__main__":
    main()
