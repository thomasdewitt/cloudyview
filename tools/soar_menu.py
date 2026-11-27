#!/usr/bin/env python
"""Cheaper kernels that could ship at soar's Low and Minimal tiers: how much
march time each saves, and how far each moves the picture.

    uv run --extra dev python tools/soar_menu.py --demo twpice --out menu.jsonl
    uv run --extra dev python tools/soar_menu.py --demo congestus \\
        --scene city --out menu.jsonl
    uv run --extra dev python tools/soar_menu.py --report menu.jsonl --md menu.md

Each candidate is the stock module with one patch (soar_variants.patch,
which refuses an anchor that has moved); the tree's shader is not touched.

  grad_coarse  sigma_gradient with the coarse stencil alone: no fine taps,
               no fine/coarse blend. Its radius is still the distance-
               dependent one (sample distance x tan 2 deg, floored at the
               fine radius) and its confidence still gates the shading. The
               taps are unpinned, as the coarse ones are today, so under
               NESTED a near sample at the nest boundary reads the parent.
  grad_fine    sigma_gradient with the fine stencil alone (one voxel,
               pinned to the sample's level), gated by its own confidence.
  pm_step8     the pre-march at AHEAD_STEP_SCALE 8 (stock 4): half the steps.
  pm_cap30     the pre-march stopping at AHEAD_TAU_CAP 30 (stock 60).
  pm_vertical  the pre-march skipped for a ray whose ahead_novelty (the sine
               of its angle from the vertical) is below PM_NOVELTY_MIN.
               Everything the pre-march feeds is ahead_factor, mixed toward
               its value by AHEAD_WEIGHT * ahead_novelty * deep_shadow_gate,
               so a skipped ray moves its fills by at most AHEAD_WEIGHT *
               PM_NOVELTY_MIN * (1 - AHEAD_FLOOR). The share of rays it
               skips is computed from the camera (pixel centres) and printed.
  uplight_4    (city scene) city_uplight_probe_tau at 4 taps (stock 12).

Beside them, for scale:

  stock_again  the stock module compiled again: the A/A control for time.
  no_premarch, no_gradient   soar_ablate's ceilings (the pass removed).

Time is measured as soar_ablate measures it: every arm warmed, then timed
--repeats times in an order rotated each repeat, fastest repeat reported,
the march pass's GPU time from timestamp queries.

The picture. Every arm renders the view converged, --image-frames frames
(frame 0 unjittered and the rest jittered, as the browser accumulates), at
the tier's render size, tone-mapped by the shader: the values are display
values in [0, 1], reported in 8-bit levels (x 255). Every arm uses the same
frame indices, so the stratified streams agree and the difference from
stock is the candidate's own. `noise` is stock against stock over another
set of frame indices (offset NOISE_FRAME_OFFSET): the Monte Carlo grain of
a converged render, the floor against which a difference can be seen.
Reported per view: RMS over pixels and channels, the 99.9th percentile of
|difference|, and its maximum.

Pictures saved: for each demo and candidate, the two (tier, view) cells
among --png-tiers whose RMS difference is largest, as one PNG of three
panels: stock | candidate | |candidate - stock| x DIFF_GAIN. None is saved
for a cell whose 8-bit picture is stock's exactly.

Everything is relative to the repository and the device wgpu picks; the
M1 notes in soar_ablate.py apply here unchanged.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soar_variants as sv                                    # noqa: E402
from soar_ablate import ablate, city_views, CONTROL           # noqa: E402
from soar_gradient_stencils import (GRADIENT_CANDIDATES,       # noqa: E402
                                    ALL_GRADIENT_VARIANTS, gradient_candidate)

DAY_CANDIDATES = ("grad_coarse", "grad_fine", "pm_step8", "pm_cap30",
                  "pm_vertical")
CITY_CANDIDATES = ("uplight_4",)
CEILINGS = ("no_premarch", "no_gradient")
# --set gradient: soar_gradient_stencils' variants, beside grad_coarse (one
# stencil dropped) and no_gradient (the whole stencil dropped) for scale.
GRADIENT_SET = GRADIENT_CANDIDATES + ("grad_coarse", "no_gradient")
CROP = (960, 540)            # PNG panels larger than this are cropped to it
PM_NOVELTY_MIN = 0.05        # sin of 2.87 deg from the vertical
NOISE_FRAME_OFFSET = 4096
DIFF_GAIN = 8.0

GRADIENT_HEAD = ("fn sigma_gradient(p: vec3<f32>, sigma: f32, "
                 "sample_distance_m: f32,\n                  in_nest: bool) "
                 "-> vec4<f32> {\n")
GRADIENT_TAIL = "    return vec4<f32>(blended, max(fine_conf, coarse_conf));\n}\n"

GRAD_COARSE = """\
    // menu: the coarse stencil alone.
    let voxel = level_voxel_size(in_nest);
    let fine_h = voxel * GRADIENT_SHADING_RADIUS_VOXELS;
    let coarse_h = max(vec3<f32>(sample_distance_m * CONE_STENCIL_TAN), fine_h);
    let coarse_grad = sigma_gradient_at_radius(p, coarse_h, false, in_nest);
    let coarse_len = length(coarse_grad);
    let coarse_conf = (
        coarse_len * min(min(coarse_h.x, coarse_h.y), coarse_h.z)
    ) / (sigma + 1e-4);
    let coarse_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, coarse_conf
    );
    var n = vec3<f32>(0.0);
    if (coarse_len > 1e-12) {
        n = coarse_gate * coarse_grad / coarse_len;
    }
    return vec4<f32>(n, coarse_conf);
}
"""

GRAD_FINE = """\
    // menu: the fine stencil alone.
    let voxel = level_voxel_size(in_nest);
    let fine_h = voxel * GRADIENT_SHADING_RADIUS_VOXELS;
    let fine_grad = sigma_gradient_at_radius(p, fine_h, true, in_nest);
    let fine_len = length(fine_grad);
    let fine_conf = (
        fine_len * min(min(fine_h.x, fine_h.y), fine_h.z)
    ) / (sigma + 1e-4);
    let fine_gate = smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, fine_conf
    );
    var n = vec3<f32>(0.0);
    if (fine_len > 1e-12) {
        n = fine_gate * fine_grad / fine_len;
    }
    return vec4<f32>(n, fine_conf);
}
"""


def replace_gradient(src: str, body: str) -> str:
    """sigma_gradient's body replaced by `body` (which closes the function)."""
    if src.count(GRADIENT_HEAD) != 1:
        raise RuntimeError("sigma_gradient's signature has moved; update "
                           "GRADIENT_HEAD")
    i = src.index(GRADIENT_HEAD) + len(GRADIENT_HEAD)
    j = src.find(GRADIENT_TAIL, i)
    if j < 0 or src.count(GRADIENT_TAIL) != 1:
        raise RuntimeError("sigma_gradient's return has moved; update "
                           "GRADIENT_TAIL")
    return src[:i] + body + src[j + len(GRADIENT_TAIL):]


def candidate(src: str, name: str) -> str:
    p = sv.patch
    if name == "grad_coarse":
        return replace_gradient(src, GRAD_COARSE)
    if name == "grad_fine":
        return replace_gradient(src, GRAD_FINE)
    if name == "pm_step8":
        return p(src, "const AHEAD_STEP_SCALE: f32 = 4.0;",
                 "const AHEAD_STEP_SCALE: f32 = 8.0;  // menu", what=name)
    if name == "pm_cap30":
        return p(src, "const AHEAD_TAU_CAP: f32 = 60.0;",
                 "const AHEAD_TAU_CAP: f32 = 30.0;  // menu", what=name)
    if name == "pm_vertical":
        return p(src, """\
    var tau_total = 0.0;
    if (t_near >= 0.0 && t_near < t_far) {
        tau_total = premarch_tau_ahead(""", f"""\
    var tau_total = 0.0;
    // menu: no pre-march for a sightline near the vertical.
    if (t_near >= 0.0 && t_near < t_far && ahead_novelty >= {PM_NOVELTY_MIN}) {{
        tau_total = premarch_tau_ahead(""", what=name)
    if name == "uplight_4":
        return p(src, "const CITY_UP_PROBE_STEPS: i32 = 12;",
                 "const CITY_UP_PROBE_STEPS: i32 = 4;  // menu", what=name)
    if name in ALL_GRADIENT_VARIANTS:
        return gradient_candidate(src, name)
    return ablate(src, name)


def crop_window(diff: np.ndarray, size=CROP):
    """(y0, x0) of the size[0] x size[1] window holding the most |diff|,
    or (0, 0) for an image no larger than the window."""
    h, w = diff.shape[:2]
    cw, ch = min(size[0], w), min(size[1], h)
    if (cw, ch) == (w, h):
        return 0, 0
    m = np.abs(diff).sum(axis=-1)
    ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    win = ii[ch:, cw:] - ii[:-ch, cw:] - ii[ch:, :-cw] + ii[:-ch, :-cw]
    y0, x0 = np.unravel_index(np.argmax(win), win.shape)
    return int(y0), int(x0)


def vertical_share(view_kw: dict, size, threshold=PM_NOVELTY_MIN) -> float:
    """The share of pixel-centre rays whose sine from the vertical is below
    `threshold`: camera_ray in camera.wgsl, without the sub-pixel jitter."""
    from cloudyview.soar.uniforms import camera_basis
    w, h = size
    fwd, right, up = camera_basis(view_kw["azimuth"], view_kw["elevation"])
    th = math.tan(math.radians(view_kw["fov"]) * 0.5)
    aspect = w / h
    x = (2.0 * (np.arange(w) + 0.5) / w - 1.0) * th
    y = (1.0 - 2.0 * (np.arange(h) + 0.5) / h) * th / aspect
    X, Y = np.meshgrid(x, y)
    d = fwd[None, None] + X[..., None] * right + Y[..., None] * up
    dz = d[..., 2] / np.linalg.norm(d, axis=-1)
    return float((np.sqrt(np.maximum(1.0 - dz * dz, 0.0)) < threshold).mean())


def picture_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """RMS, p99.9 and max of |a - b| over pixels and channels, in 8-bit
    levels of display value."""
    d = np.abs(a - b) * 255.0
    return dict(rms=float(np.sqrt(np.mean(d * d))),
                p999=float(np.percentile(d, 99.9)), max=float(d.max()))


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_triptych(path: Path, stock: np.ndarray, cand: np.ndarray, title: str):
    from PIL import Image, ImageDraw
    diff = to_u8(np.abs(cand.astype(np.float64) - stock) / 255.0 * DIFF_GAIN)
    h, w, _ = stock.shape
    gap = 4
    band = 16
    canvas = np.full((h + band, 3 * w + 2 * gap, 3), 255, np.uint8)
    for k, im in enumerate((stock, cand, diff)):
        canvas[band:, k * (w + gap):k * (w + gap) + w] = im
    pim = Image.fromarray(canvas)
    dr = ImageDraw.Draw(pim)
    for k, lab in enumerate(("stock", title, f"|diff| x {DIFF_GAIN:g}")):
        dr.text((k * (w + gap) + 3, 2), lab, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pim.save(path, optimize=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", default=None,
                    help="a baked demo: a name under --demo-dir, or a directory")
    ap.add_argument("--demo-dir", default=None,
                    help="where the baked demos are (default web/soar/demos)")
    ap.add_argument("--scene", default="day", choices=("day", "city"))
    ap.add_argument("--tiers", default="minimal,low,medium,high")
    ap.add_argument("--set", default="menu", choices=("menu", "gradient"),
                    help="the default variants: the first menu (pre-march, "
                         "gradient and uplight cuts) or the gradient stencils")
    ap.add_argument("--variants", default="",
                    help="comma-separated; default the --set's candidates and "
                         "ceilings for the scene. stock and stock_again always run.")
    ap.add_argument("--views", default="", help="comma-separated; default all")
    ap.add_argument("--output", default="2560x1440",
                    help="the window; each tier renders at its scale of it")
    ap.add_argument("--frames", type=int, default=8, help="frames per timing")
    ap.add_argument("--repeats", type=int, default=5,
                    help="timings per arm and view; the fastest is reported")
    ap.add_argument("--image-frames", type=int, default=32,
                    help="frames accumulated for each picture")
    ap.add_argument("--png-tiers", default=None,
                    help="tiers whose pictures are candidates for the PNGs "
                         "(default minimal,low; low,medium,high for --set "
                         "gradient). Panels larger than CROP are cropped, at "
                         "full resolution, to the window that differs most.")
    ap.add_argument("--png-dir", default=None,
                    help="where the PNGs go (default: beside --out, menu_png/)")
    ap.add_argument("--out", default=None, help="JSON lines, appended")
    ap.add_argument("--label", default="run")
    ap.add_argument("--report", nargs="+", default=None, metavar="JSONL",
                    help="instead of measuring: join --out files into tables")
    ap.add_argument("--md", default=None, help="with --report: write here")
    args = ap.parse_args()
    if args.report:
        return report(args.report, args.md)
    if not args.demo:
        ap.error("--demo is required (unless --report)")

    city = args.scene == "city"
    surface = "city" if city else "ocean"
    if args.set == "gradient":
        default = GRADIENT_SET
    else:
        default = (CITY_CANDIDATES if city else DAY_CANDIDATES) + CEILINGS
    if args.png_tiers is None:
        args.png_tiers = "low,medium,high" if args.set == "gradient" else "minimal,low"
    names = [v for v in (args.variants.split(",") if args.variants else default) if v]
    presets = sv.flight_presets()
    out_w, out_h = (int(v) for v in args.output.lower().split("x"))
    adapter, device = sv.make_device(timestamps=True, float32_filterable=False)
    info = adapter.info
    level, meta = sv.load_demo(args.demo, args.demo_dir)
    periodic = bool(meta["periodic"])
    stock = sv.shader.assemble(surface)
    sources = {"stock": stock, CONTROL: ablate(stock, CONTROL)}
    for v in names:
        sources[v] = candidate(stock, v)
    r = sv.VariantRenderer(sources, device=device, periodic=periodic,
                           surface=surface)
    for v, s in sources.items():
        if v != "stock":
            r.add_variant(v, s)
    r.upload_volume(level.sigma)
    views = city_views(level, presets) if city else sv.demo_views(
        level, meta, periodic)[0]
    if args.views:
        views = {k: v for k, v in views.items() if k in args.views.split(",")}
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sv.REPO,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "cloudyview/soar/shader"],
                           cwd=sv.REPO, capture_output=True, text=True).stdout.strip()
    print(f"soar menu at {rev}{' (shader dirty)' if dirty else ''} on "
          f"{info.get('device', '?')} ({info.get('backend_type', '?')}): "
          f"{level.name}, {args.scene}, window {out_w}x{out_h}; time fastest of "
          f"{args.repeats} x {args.frames} frames, pictures {args.image_frames} "
          f"frames; GPU timestamps {'on' if r.timestamps else 'OFF'}", flush=True)
    if not r.timestamps:
        raise SystemExit("no timestamp-query on this device; the menu needs "
                         "the march pass's own time")

    arms = [sv.Arm("stock", "stock"), sv.Arm(CONTROL, CONTROL)] + [
        sv.Arm(v, v) for v in names]
    png_tiers = set(args.png_tiers.split(","))
    png_dir = Path(args.png_dir) if args.png_dir else (
        Path(args.out).parent / "menu_png" if args.out else Path("menu_png"))
    best = {}            # candidate -> [(rms, tier, view, stock_u8, cand_u8)]
    out = open(args.out, "a") if args.out else None
    for tier in sorted(args.tiers.split(","), key=sv.TIER_ORDER.index):
        look = sv.tier_look(presets, tier, city=city)
        extra = ({"city_roof_density": float(presets[tier]["cityRoofDensity"])}
                 if city else {})
        state = sv.tier_scene(r, level, presets, tier, periodic)
        size = sv.render_size(presets, tier, out_w, out_h)
        print(f"\n{tier}: {size[0]}x{size[1]}, haze {look['haze']:.4g}, view LOD "
              f"{look['view_step_lod_degrees']:.3g} deg, light LOD "
              f"{look['light_march_lod_degrees']:.3g} deg", flush=True)
        print(f"  {'view':20s}{'stock ms':>10s}" + "".join(
            f"{a.label:>22s}" for a in arms[1:]) + "   (time vs stock / RMS levels)",
            flush=True)
        for name, vk in views.items():
            view = sv.make_view(vk, look, (out_w, out_h), size, **extra)
            res = sv.time_variants(r, state, view, arms, frames=args.frames,
                                   repeats=args.repeats)
            summ = {k: {m: sv.summarize(v[m]) for m in v} for k, v in res.items()}
            pics = {}
            for a in arms:
                sv._select(r, a)
                pics[a.label] = r.render(state, view, frames=args.image_frames)
            sv._select(r, arms[0])
            noise_img = r.render(state, replace(
                view, frame_index=view.frame_index + NOISE_FRAME_OFFSET),
                frames=args.image_frames)
            ref = pics["stock"]
            diffs = {a.label: picture_diff(pics[a.label], ref) for a in arms[1:]}
            diffs["noise"] = picture_diff(noise_img, ref)
            # One frame alone (frame 0, unjittered): what a moving camera
            # shows before any accumulation, which is where an alternating
            # stencil's error is not averaged away.
            single = {}
            for a in arms:
                sv._select(r, a)
                single[a.label] = r.render(state, view, frames=1)
            sv._select(r, arms[0])
            diffs_1 = {a.label: picture_diff(single[a.label], single["stock"])
                       for a in arms[1:]}
            del single
            rec = dict(kind="menu", label=args.label, rev=rev, shader_dirty=bool(dirty),
                       device=info.get("device"), backend=info.get("backend_type"),
                       demo=level.name, scene=args.scene, tier=tier, view=name,
                       size=list(size), output=[out_w, out_h], look=look,
                       frames=args.frames, repeats=args.repeats,
                       image_frames=args.image_frames,
                       when=time.strftime("%Y-%m-%d %H:%M:%S"),
                       samples=res, summary=summ, picture=diffs,
                       picture_one_frame=diffs_1, variant_set=args.set,
                       pm_vertical_share=(vertical_share(vk, size)
                                          if "pm_vertical" in names else None),
                       rays_within_deg_of_vertical={
                           str(dg): vertical_share(vk, size,
                                                   math.sin(math.radians(dg)))
                           for dg in (10, 20, 30)})
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            base = summ["stock"]["gpu_ms"]["min"]
            cells = "".join(
                f"{100 * (summ[a.label]['gpu_ms']['min'] / base - 1):+12.1f}% "
                f"{diffs[a.label]['rms']:7.2f}  " for a in arms[1:])
            sp = max(summ[a.label]["gpu_ms"]["spread"] or 0 for a in arms)
            print(f"  {name:20s}{base:10.3f}{cells}  noise {diffs['noise']['rms']:.2f}"
                  f"  spread<= {100 * sp:.0f}%", flush=True)
            if tier in png_tiers:
                for a in arms[2:]:
                    lst = best.setdefault(a.label, [])
                    if len(lst) == 2 and diffs[a.label]["rms"] <= lst[-1][0]:
                        continue
                    y0, x0 = crop_window(pics[a.label] - ref)
                    ch, cw = min(CROP[1], ref.shape[0]), min(CROP[0], ref.shape[1])
                    win = (slice(y0, y0 + ch), slice(x0, x0 + cw))
                    lst.append((diffs[a.label]["rms"], tier, name, to_u8(ref[win]),
                                to_u8(pics[a.label][win])))
                    lst.sort(key=lambda e: -e[0])
                    del lst[2:]
            del pics, noise_img
    for cand, lst in best.items():
        for rms, tier, name, s8, c8 in lst:
            if not (c8 != s8).any():
                continue          # the picture is stock's to the 8-bit level
            save_triptych(png_dir / f"{level.name}_{args.scene}_{cand}_{tier}_{name}.png",
                          s8, c8, f"{cand} (RMS {rms:.2f} levels)")
    if out:
        out.close()


def report(paths, md_path=None):
    """Per scene and tier, over every (demo, view): each arm's change in
    march-pass GPU time against stock (sums), and its picture difference
    (RMS: root of the mean over views of the per-view mean square; p99.9 and
    max: the largest over views), beside stock_again's time and the
    Monte Carlo noise of a converged render."""
    from collections import defaultdict
    recs = [json.loads(line) for p in paths for line in open(p) if line.strip()]
    recs = [r for r in recs if r.get("kind") == "menu"]
    if not recs:
        raise SystemExit("no menu records in " + ", ".join(paths))
    order = {t: i for i, t in enumerate(sv.TIER_ORDER)}
    scene_arms = {}
    for r in recs:
        scene_arms.setdefault(r["scene"], [a for a in r["summary"] if a != "stock"])
    groups = defaultdict(list)
    for r in recs:
        groups[(r["scene"], r["tier"])].append(r)
    lines = ["Time: change in march-pass GPU time against stock, summed over "
             "every (demo, view) of the scene and tier. Picture: difference "
             "from stock in 8-bit display levels; RMS is the root of the mean "
             "over views of each view's mean square, p99.9 and max the "
             "largest over views. noise = stock against stock over other "
             "frame indices (the converged render's own grain). 1-frame RMS: "
             "the same RMS for a single unjittered frame, where present."]
    for scene in sorted({k[0] for k in groups}):
        arms = scene_arms[scene]
        lines += ["", f"## {scene}", "",
                  "| tier | views | stock ms | arm | time | RMS | p99.9 | max |"
                  " 1-frame RMS | pm_vertical skips |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for tier in sorted({k[1] for k in groups if k[0] == scene}, key=order.get):
            rs = groups[(scene, tier)]
            base = sum(r["summary"]["stock"]["gpu_ms"]["min"] for r in rs)
            for a in arms + ["noise"]:
                if a == "noise":
                    t = "–"
                else:
                    s = sum(r["summary"][a]["gpu_ms"]["min"] for r in rs)
                    t = f"{100 * (s / base - 1):+.1f}%"
                pd = [r["picture"][a] for r in rs]
                rms = math.sqrt(sum(d["rms"] ** 2 for d in pd) / len(pd))
                skip = ""
                if a == "pm_vertical":
                    sh = [r["pm_vertical_share"] for r in rs
                          if r.get("pm_vertical_share") is not None]
                    skip = f"{100 * sum(sh) / len(sh):.2f}% (max {100 * max(sh):.1f}%)"
                    near = [r.get("rays_within_deg_of_vertical") for r in rs]
                    near = [n for n in near if n]
                    if near:
                        skip += "; within 10/20/30 deg: " + " / ".join(
                            f"{100 * sum(n[k] for n in near) / len(near):.1f}%"
                            for k in ("10", "20", "30"))
                one = [r["picture_one_frame"][a] for r in rs
                       if a in r.get("picture_one_frame", {})]
                rms1 = (f"{math.sqrt(sum(d['rms'] ** 2 for d in one) / len(one)):.2f}"
                        if one else "–")
                lines.append(f"| {tier} | {len(rs)} | {base:.2f} | {a} | {t} | "
                             f"{rms:.2f} | {max(d['p999'] for d in pd):.1f} | "
                             f"{max(d['max'] for d in pd):.0f} | {rms1} | {skip} |")
        lines.append("")
    md = "\n".join(lines) + "\n"
    if md_path:
        Path(md_path).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
