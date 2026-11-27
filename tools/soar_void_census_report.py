"""The report of tools/soar_void_census.py: census and timing JSON lines into
markdown tables and one JSON file. Run through `soar_void_census.py report`.

Aggregates. A per-view number is a mean per pixel (counts) or the fastest
time (ms). A demo x tier number is the mean over its views of the per-view
numbers, so each view weighs the same. A tier number over all demos is the
same mean over every (demo, view). Shares of time are ratios of sums, so a
slow view weighs more than a fast one, as it does in frame time.
"""

import json
from collections import defaultdict
from pathlib import Path

TIER_ORDER = ("minimal", "low", "medium", "high", "max")
ARMS = ("full", "zero", "nomarch", "nomarch_live", "oracle_px", "oracle_t16")


def _load(paths):
    census, timing, placement = [], [], {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["kind"] == "census":
                census.append(r)
            elif r["kind"] == "timing":
                timing.append(r)
            elif r["kind"] == "placement":
                placement[r["demo"]] = r
    return census, timing, placement


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _f(x, nd=1):
    if x is None:
        return "–"
    return f"{x:.{nd}f}"


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _tier_key(t):
    return TIER_ORDER.index(t)


def timing_cells(timing):
    """(demo, tier, view) -> {arm: {"gpu": fastest over all runs, "wall": ..,
    "runs": {label: fastest gpu}, "spread_in_run": max over runs}}"""
    cells = defaultdict(lambda: defaultdict(lambda: {"gpu": None, "wall": None,
                                                    "runs": {}, "spread": 0.0}))
    for r in timing:
        key = (r["demo"], r["tier"], r["view"])
        for arm, s in r["summary"].items():
            c = cells[key][arm]
            g, w = s["gpu_ms"]["min"], s["wall_ms"]["min"]
            c["gpu"] = g if c["gpu"] is None else min(c["gpu"], g)
            c["wall"] = w if c["wall"] is None else min(c["wall"], w)
            prev = c["runs"].get(r["label"])
            c["runs"][r["label"]] = g if prev is None else min(prev, g)
            c["spread"] = max(c["spread"], s["gpu_ms"]["spread"] or 0.0)
    return cells


def run_to_run(c):
    vals = [v for v in c["runs"].values() if v is not None]
    if len(vals) < 2:
        return None
    return (max(vals) - min(vals)) / min(vals)


def report_main(args):
    census, timing, placement = _load(args.jsonl)
    lines = []
    out = {"census": census, "timing": timing,
           "placement": list(placement.values())}

    # ---- setup ----
    if census:
        lines += ["## Tiers as flown", "",
                  "| tier | render size | haze | view LOD (deg) | light LOD (deg) "
                  "| view step (voxels) | light cache |", "|---|---|---|---|---|---|---|"]
        seen = set()
        for r in sorted(census, key=lambda r: _tier_key(r["tier"])):
            if r["tier"] in seen:
                continue
            seen.add(r["tier"])
            lk = r["look"]
            dx = placement.get(r["demo"], {}).get("dx")
            vox = r["dt_view"] / min(dx) if dx else None
            lines.append(f"| {r['tier']} | {r['size'][0]}x{r['size'][1]} | "
                         f"{lk['haze']:.4g} | {lk['view_step_lod_degrees']:.3g} | "
                         f"{lk['light_march_lod_degrees']:.3g} | {_f(vox, 2)} | "
                         f"{lk['light_cache']} |")
        lines.append("")

    if placement:
        lines += ["## Flight-view placement", "",
                  "Heights in metres. base/top: lowest/highest level with >= 1% "
                  "of columns at sigma > 1e-3 /m; top_any: highest with >= 1% "
                  "at sigma > 1e-5 /m. cruise peak: the largest sigma within 4 "
                  "voxels of the cruise camera (0 = clear air).", "",
                  "| demo | periodic | shape | base | top | top_any | cruise z "
                  "| under z | above z | cruise peak sigma |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for d, p in placement.items():
            q = p["placement"]
            lines.append(f"| {d} | {p['periodic']} | {'x'.join(map(str, p['shape']))} | "
                         f"{q['base']:.0f} | {q['top']:.0f} | {q['top_any']:.0f} | "
                         f"{q['z_cruise']:.0f} | {q['z_under']:.0f} | {q['z_above']:.0f} | "
                         f"{q['cruise_peak_sigma']:.2g} |")
        lines.append("")

    # ---- census by demo x tier ----
    groups = defaultdict(list)
    for r in census:
        groups[(r["demo"], r["tier"])].append(r)
    demos = list(dict.fromkeys(r["demo"] for r in census))

    def g(rs, var, key):
        return _mean([r[var].get(key) for r in rs])

    if census:
        lines += ["## Step census, per demo and tier", "",
                  "Mean per pixel over the views (each view weighs the same). "
                  "lead/int/trail/cloud are view-march steps; pm = pre-march "
                  "steps (pm void = those wholly before the first or after "
                  "the last cloud sample); no-cloud = fraction of pixels whose "
                  "ray has no cloud sample; zero = all steps with the field "
                  "zeroed; px / t16 = all steps left by the per-pixel / 16x16 "
                  "oracle clamp (march + pre-march); warp = mean over 8x4 "
                  "pixel blocks of the block's largest step count.", "",
                  "| demo | tier | lead | int | trail | cloud | pm | pm void "
                  "| no-cloud | zero | px | t16 | warp stock | warp px |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in demos:
            for t in sorted({k[1] for k in groups if k[0] == d}, key=_tier_key):
                rs = groups[(d, t)]
                pmv = _mean([r["stock"]["premarch_before_first"]
                             + r["stock"]["premarch_after_last"] for r in rs])
                left = lambda v: _mean([r[v]["total"] + r[v]["premarch"]  # noqa: E731
                                        for r in rs])
                lines.append(
                    f"| {d} | {t} | {_f(g(rs, 'stock', 'lead'))} | "
                    f"{_f(g(rs, 'stock', 'interior'))} | {_f(g(rs, 'stock', 'trail'))} | "
                    f"{_f(g(rs, 'stock', 'cloud'))} | {_f(g(rs, 'stock', 'premarch'))} | "
                    f"{_f(pmv)} | {_f(g(rs, 'stock', 'frac_nocloud'), 2)} | "
                    f"{_f(left('zero'))} | {_f(left('oracle_px'))} | "
                    f"{_f(left('oracle_t16'))} | {_f(g(rs, 'stock', 'warp_max_total'))} | "
                    f"{_f(g(rs, 'oracle_px', 'warp_max_total'))} |")
        lines.append("")

        lines += ["## Step census, per tier over all demos and views", "",
                  "| tier | lead | int | trail | cloud | pm | pm void | stock total "
                  "| zero total | px | t8 | t16 | t32 | cloud kept by px "
                  "| no-cloud | subcut | cloud dtau<1e-4 | <1e-3 | >=1e-3 |",
                  "|" + "---|" * 20]
        for t in sorted({r["tier"] for r in census}, key=_tier_key):
            rs = [r for r in census if r["tier"] == t]
            st = lambda v: _mean([r[v]["total"] + r[v]["premarch"] for r in rs])  # noqa: E731
            pmv = _mean([r["stock"]["premarch_before_first"]
                         + r["stock"]["premarch_after_last"] for r in rs])
            kept = (sum(r["oracle_px"]["cloud"] for r in rs)
                    / max(sum(r["stock"]["cloud"] for r in rs), 1e-12))
            lines.append(
                f"| {t} | {_f(g(rs, 'stock', 'lead'))} | {_f(g(rs, 'stock', 'interior'))} "
                f"| {_f(g(rs, 'stock', 'trail'))} | {_f(g(rs, 'stock', 'cloud'))} "
                f"| {_f(g(rs, 'stock', 'premarch'))} | {_f(pmv)} | {_f(st('stock'))} "
                f"| {_f(st('zero'))} | {_f(st('oracle_px'))} | {_f(st('oracle_t8'))} "
                f"| {_f(st('oracle_t16'))} | {_f(st('oracle_t32'))} | {kept:.4f} "
                f"| {_f(g(rs, 'stock', 'frac_nocloud'), 2)} | {_f(g(rs, 'stock', 'subcut'))} "
                f"| {_f(g(rs, 'stock', 'cloud_dtau_lt_1e4'))} | "
                f"{_f(g(rs, 'stock', 'cloud_dtau_lt_1e3'))} | "
                f"{_f(g(rs, 'stock', 'cloud_dtau_ge_1e3'))} |")
        lines.append("")

    # ---- timing ----
    cells = timing_cells(timing)
    if cells:
        cen = {(r["demo"], r["tier"], r["view"]): r for r in census}
        lines += ["## GPU time of the march pass, per demo and tier", "",
                  "Fastest over all repeats and runs, ms, mean over the views. "
                  "full = as is; zero = sigma 0 everywhere; nomarch = t_far = "
                  "t_near (foldable); live = the same behind a uniform; px / "
                  "t16 = the oracle clamp. r2r = largest run-to-run difference "
                  "of any arm in the group; in-run = largest spread of any arm "
                  "within one run's repeats.", "",
                  "| demo | tier | full | zero | nomarch | live | px | t16 | "
                  "px saves | t16 saves | wall full | r2r | in-run |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        tg = defaultdict(list)
        for (d, t, v), c in cells.items():
            tg[(d, t)].append(c)
        for (d, t) in sorted(tg, key=lambda k: (demos.index(k[0]) if k[0] in demos
                                                else 99, _tier_key(k[1]))):
            cs = tg[(d, t)]
            m = {a: _mean([c[a]["gpu"] for c in cs]) for a in ARMS}
            full_sum = sum(c["full"]["gpu"] for c in cs)
            spx = 1 - sum(c["oracle_px"]["gpu"] for c in cs) / full_sum
            s16 = 1 - sum(c["oracle_t16"]["gpu"] for c in cs) / full_sum
            r2r = max((run_to_run(c[a]) or 0) for c in cs for a in ARMS)
            inr = max(c[a]["spread"] for c in cs for a in ARMS)
            lines.append(
                f"| {d} | {t} | {_f(m['full'], 3)} | {_f(m['zero'], 3)} | "
                f"{_f(m['nomarch'], 3)} | {_f(m['nomarch_live'], 3)} | "
                f"{_f(m['oracle_px'], 3)} | {_f(m['oracle_t16'], 3)} | {_pct(spx)} | "
                f"{_pct(s16)} | {_f(_mean([c['full']['wall'] for c in cs]), 2)} | "
                f"{_pct(r2r)} | {_pct(inr)} |")
        lines.append("")

        lines += ["## Per tier: where the march pass's time goes", "",
                  "Sums over every (demo, view) timed. fixed = live (the full "
                  "kernel, march not entered). The linear estimate prices every "
                  "clear-air step (view march and pre-march) at the zero "
                  "field's cost per step, c = (zero - live) / (zero-field "
                  "steps), and is what a step count alone predicts; the oracle "
                  "columns are measured.", "",
                  "| tier | full ms | fixed | zero-field traversal | c (ns per "
                  "pixel-step) | void est. | of which lead | int | trail | pm void "
                  "| px saves | t16 saves |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        derived = {}
        for t in sorted({k[1] for k in cells}, key=_tier_key):
            keys = [k for k in cells if k[1] == t and k in cen]
            if not keys:
                continue
            S = lambda a: sum(cells[k][a]["gpu"] for k in keys)       # noqa: E731
            full, zero, live = S("full"), S("zero"), S("nomarch_live")
            px, t16 = S("oracle_px"), S("oracle_t16")
            # pixel-steps per frame, summed over views
            def steps(var, f):
                return sum(f(cen[k][var]) * cen[k][var]["pixels"] for k in keys)
            zsteps = steps("zero", lambda s: s["total"] + s["premarch"])
            c = (zero - live) / zsteps if zsteps else None        # ms per pixel-step
            lead = steps("stock", lambda s: s["lead"])
            inte = steps("stock", lambda s: s["interior"])
            trail = steps("stock", lambda s: s["trail"])
            pmv = steps("stock", lambda s: s["premarch_before_first"]
                        + s["premarch_after_last"])
            void = (lead + inte + trail + pmv) * c
            d = dict(full=full, fixed=live, zero_traversal=zero - live,
                     c_ns=c * 1e6, void_est=void, lead=lead * c, interior=inte * c,
                     trail=trail * c, pm_void=pmv * c,
                     px_saves=full - px, t16_saves=full - t16, views=len(keys))
            derived[t] = d
            lines.append(
                f"| {t} | {full:.2f} | {_pct(live / full)} | {_pct((zero - live) / full)} "
                f"| {c * 1e6:.3f} | {_pct(void / full)} | {_pct(d['lead'] / full)} | "
                f"{_pct(d['interior'] / full)} | {_pct(d['trail'] / full)} | "
                f"{_pct(d['pm_void'] / full)} | {_pct(d['px_saves'] / full)} | "
                f"{_pct(d['t16_saves'] / full)} |")
        lines.append("")
        out["derived_per_tier"] = derived

    md = "\n".join(lines) + "\n"
    Path(args.md).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
    per_view(census, cells, Path(args.md).with_name(
        Path(args.md).stem + "_per_view.md"))
    print(md)


def per_view(census, cells, path):
    lines = ["# Per view", "",
             "## Census (mean per pixel)", "",
             "| demo | tier | view | pixels | lead | int | trail | cloud | pm "
             "| pm before | pm after | no-cloud | not entered | end: t_far | opaque "
             "| sea | cap | subcut | zero | px | t16 |",
             "|" + "---|" * 21]
    for r in census:
        s, z = r["stock"], r["zero"]
        lines.append(
            f"| {r['demo']} | {r['tier']} | {r['view']} | {s['pixels']} | {_f(s['lead'])} "
            f"| {_f(s['interior'])} | {_f(s['trail'])} | {_f(s['cloud'])} "
            f"| {_f(s['premarch'])} | {_f(s['premarch_before_first'])} "
            f"| {_f(s['premarch_after_last'])} | {_f(s['frac_nocloud'], 2)} "
            f"| {_f(s['frac_not_entered'], 2)} | {_f(s['frac_end_tfar'], 2)} "
            f"| {_f(s['frac_end_opaque'], 2)} | {_f(s['frac_end_sea'], 2)} "
            f"| {_f(s['frac_end_stepcap'], 3)} | {_f(s['subcut'])} "
            f"| {_f(z['total'] + z['premarch'])} "
            f"| {_f(r['oracle_px']['total'] + r['oracle_px']['premarch'])} "
            f"| {_f(r['oracle_t16']['total'] + r['oracle_t16']['premarch'])} |")
    lines += ["", "## GPU march-pass ms (fastest of all repeats and runs)", "",
              "| demo | tier | view | " + " | ".join(ARMS)
              + " | wall full | r2r max | in-run max |",
              "|" + "---|" * (len(ARMS) + 6)]
    for (d, t, v), c in cells.items():
        r2r = max((run_to_run(c[a]) or 0) for a in ARMS)
        inr = max(c[a]["spread"] for a in ARMS)
        lines.append(f"| {d} | {t} | {v} | "
                     + " | ".join(_f(c[a]["gpu"], 3) for a in ARMS)
                     + f" | {_f(c['full']['wall'], 2)} | {_pct(r2r)} | {_pct(inr)} |")
    path.write_text("\n".join(lines) + "\n")
