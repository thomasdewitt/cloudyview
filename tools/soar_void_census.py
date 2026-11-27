#!/usr/bin/env python
"""Where the soar view march's steps and frame time go in clear air.

    uv run --extra dev python tools/soar_void_census.py census \\
        --demos twpice,congestus,dycoms,rce,stratified --out census.jsonl
    uv run --extra dev python tools/soar_void_census.py timing \\
        --demos twpice,congestus,dycoms,rce,stratified --out timing_run1.jsonl
    uv run --extra dev python tools/soar_void_census.py report \\
        census.jsonl timing_run1.jsonl timing_run2.jsonl --md results.md

Every tier is flown as the browser flies it: its render scale, step factors,
light cache, haze and the two LOD angles scaled by its LOD strength
(soar_variants.tier_look), over each demo's own periodicity and sun. The
views are soar_bench's five day views and five flight-like views placed from
the field's cloud levels (soar_variants.demo_views).

census
    An instrumented copy of the shader (patched module text; the tree is not
    touched) writes per-pixel counts to a float32 target, one frame each:
      lead      view steps before the ray's first cloud sample (all of a
                ray's steps when it meets no cloud)
      interior  clear-air steps between its first and last cloud sample
      trail     clear-air steps after its last cloud sample
      cloud     steps that pass the EMPTY_DTAU_CUTOFF test, split by d_tau
                into [1e-5, 1e-4), [1e-4, 1e-3) and >= 1e-3
      subcut    clear-air steps whose sigma is not zero (d_tau below the
                cutoff): what a sigma > 0 occupancy test would call cloud
      premarch  premarch_tau_ahead's steps, and how many of them lie wholly
                before the first or after the last cloud sample (none after,
                for a ray that went opaque: there is cloud behind it)
      end       why the march stopped: t_far, opacity, the sea, the step cap
    It also runs the same counts with the field zeroed, and with the march
    clamped by an oracle (below), to show what the oracle leaves.

timing
    The uninstrumented shader, its march pass timed by GPU timestamps and
    the whole render by wall clock (soar_bench's number), as variants:
      full          as is
      zero          the same box and uniforms with sigma = 0 everywhere
      nomarch       t_far = t_near after the box test (the compiler may fold
                    the whole march away: fixed work under a smaller kernel)
      nomarch_live  the same through a uniform it cannot fold
                    (u.ice_mode < 0.5), so the full kernel is compiled
      oracle_px     t_near/t_far clamped per pixel to the first/last cloud
                    distance the census found, dilated 3x3 and widened by
                    two steps; a pixel with no cloud near it skips the march
      oracle_t16    the same per 16x16-pixel tile
      oracle_null   the oracle patch reading a texture that clamps nothing
                    (t_near -inf, t_far +inf on every pixel): what the clamp
                    itself costs, its load, branch and live values. The
                    oracle arms' saving is the ceiling less this cost.
    The oracle reads its distances from group 1 binding 0 (the traced
    surface, which the ocean module never reads), so it pays the one
    textureLoad a real pre-pass would also add, and none of the pre-pass's
    own cost. It is the ceiling of a first/last-distance pre-pass.

report
    Joins census and timing records into a markdown page and a JSON file.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soar_variants as sv                                    # noqa: E402

TIERS_DEFAULT = "minimal,low,medium,high"
DEMOS_DEFAULT = "twpice,congestus,dycoms,rce,stratified"
ORACLE_TILES = (8, 16, 32)          # census; timing uses 1 (per pixel) and 16
ORACLE_MARGIN_STEPS = 2.0
WARP = (8, 4)                       # pixel block standing in for a SIMD group

# -- patches ------------------------------------------------------------------

AFTER_BOX = """    } else {
        let hit = ray_box(u.cam_origin, inv_dir);
        t_near = max(hit.x, 0.0);
        t_far = hit.y;
    }
"""

# premarch_tau_ahead as the census copies it. If the tree's function has
# changed, the census refuses to run rather than count a different march.
PREMARCH_SRC = """fn premarch_tau_ahead(origin: vec3<f32>, dir: vec3<f32>,
                      t0: f32, t1: f32, jit: f32) -> f32 {
    var tau = 0.0;
    let dt_base = u.dt_view * AHEAD_STEP_SCALE;
    var t = t0 + clamp(jit, 0.0, 1.0) * dt_base;
    for (var i: i32 = 0; i < AHEAD_MAX_STEPS; i = i + 1) {
        if (t >= t1 || tau > AHEAD_TAU_CAP) {
            break;
        }
        let s = sample_level_at(origin + t * dir);
        var dt = max(s.dt_view * AHEAD_STEP_SCALE, t * u.view_lod_tan);
        if (t + dt > t1) {
            dt = t1 - t;
        }
        tau = tau + s.sigma * dt;
        t = t + dt;
    }
    return tau;
}"""

PREMARCH_CENSUS = """
// census tool: premarch_tau_ahead's loop, counting its steps: all of them,
// those wholly before the first cloud sample (all, for a ray with none), and
// those wholly after the last.
fn cn_premarch_census(origin: vec3<f32>, dir: vec3<f32>,
                      t0: f32, t1: f32, jit: f32,
                      seen: bool, tf: f32, tl: f32) -> vec3<f32> {
    var n = 0.0;
    var nb = 0.0;
    var na = 0.0;
    var tau = 0.0;
    let dt_base = u.dt_view * AHEAD_STEP_SCALE;
    var t = t0 + clamp(jit, 0.0, 1.0) * dt_base;
    for (var i: i32 = 0; i < AHEAD_MAX_STEPS; i = i + 1) {
        if (t >= t1 || tau > AHEAD_TAU_CAP) {
            break;
        }
        let s = sample_level_at(origin + t * dir);
        var dt = max(s.dt_view * AHEAD_STEP_SCALE, t * u.view_lod_tan);
        if (t + dt > t1) {
            dt = t1 - t;
        }
        tau = tau + s.sigma * dt;
        n = n + 1.0;
        if (!seen || t + dt <= tf) {
            nb = nb + 1.0;
        } else if (t >= tl) {
            na = na + 1.0;
        }
        t = t + dt;
    }
    return vec3<f32>(n, nb, na);
}

"""


def oracle_patch(src: str) -> str:
    """Clamp the march to the distances in group 1 binding 0:
    (t_near, t_far, has_cloud, _) per pixel."""
    src = sv.patch(src, AFTER_BOX, AFTER_BOX + """
    // census tool: the oracle clamp.
    let t_near_full = t_near;
    let t_far_full = t_far;
    let oracle_t = textureLoad(surface_buf, vec2<i32>(frag_pos.xy), 0);
    if (oracle_t.z > 0.5) {
        t_near = max(t_near, oracle_t.x);
        t_far = min(t_far, oracle_t.y);
    } else {
        t_far = min(t_far, t_near);
    }
""", what="oracle clamp")
    # The unclamped march would have reached the sea, and reaching it is
    # what exempts the sea from the 50-box-width shading limit. Keep that.
    return sv.patch(src, "    var shade_ocean = ocean_in_march;\n", """\
    if (ocean_on && !ocean_consumed && t_near_full >= 0.0
        && t_near_full < t_far_full && t_ocean <= t_far_full
        && transmittance > TRANSMITTANCE_CUTOFF) {
        ocean_in_march = true;
        ocean_consumed = true;
    }
    var shade_ocean = ocean_in_march;
""", what="oracle sea")


def nomarch_patch(src: str, *, live: bool) -> str:
    line = ("    t_far = select(t_far, t_near, u.ice_mode < 0.5);\n" if live
            else "    t_far = t_near;\n")
    return sv.patch(src, AFTER_BOX, AFTER_BOX + line, what="nomarch")


def census_patch(src: str, out_pass: int) -> str:
    """The instrumented march. `out_pass` picks the four numbers returned:
    0 (lead, interior, trail, cloud), 1 (premarch, premarch before the first
    cloud, premarch after the last, end code), 2 (t_first, t_last, t_near,
    t_far), 3 (subcut, cloud d_tau < 1e-4, cloud d_tau < 1e-3, cloud
    d_tau >= 1e-3)."""
    if PREMARCH_SRC not in src:
        raise RuntimeError("premarch_tau_ahead is not the function this census "
                           "counts; update PREMARCH_SRC and cn_premarch_census")
    src = sv.patch(src, "@fragment\nfn fs_main(",
                   PREMARCH_CENSUS + "@fragment\nfn fs_main(", what="census fn")
    src = sv.patch(src, "    var was_in_nest = false;\n", """\
    var was_in_nest = false;
    // census tool
    var cn_lead = 0.0;
    var cn_gap = 0.0;
    var cn_pending = 0.0;
    var cn_cloud = 0.0;
    var cn_seen = false;
    var cn_t_first = -1.0;
    var cn_t_last = -1.0;
    var cn_end = 0.0;
    var cn_subcut = 0.0;
    var cn_c4 = 0.0;
    var cn_c3 = 0.0;
    var cn_c2 = 0.0;
""", what="census vars")
    src = sv.patch(src, """\
    if (t_near >= 0.0 && t_near < t_far) {
        for (var i: i32 = 0; i < max_view_steps; i = i + 1) {
""", """\
    if (t_near >= 0.0 && t_near < t_far) {
        cn_end = 4.0;
        for (var i: i32 = 0; i < max_view_steps; i = i + 1) {
""", what="census entry")
    src = sv.patch(src, """\
            if (ocean_on && t >= t_ocean) {
                ocean_in_march = true;
""", """\
            if (ocean_on && t >= t_ocean) {
                cn_end = 3.0;
                ocean_in_march = true;
""", what="census sea")
    src = sv.patch(src, """\
            if (t >= t_far || transmittance < TRANSMITTANCE_CUTOFF) {
                break;
""", """\
            if (t >= t_far || transmittance < TRANSMITTANCE_CUTOFF) {
                cn_end = select(2.0, 1.0, t >= t_far);
                break;
""", what="census end")
    src = sv.patch(src, """\
            if (d_tau < EMPTY_DTAU_CUTOFF) {
                tau_depth = 0.0;
""", """\
            if (d_tau < EMPTY_DTAU_CUTOFF) {
                if (cn_seen) {
                    cn_pending = cn_pending + 1.0;
                } else {
                    cn_lead = cn_lead + 1.0;
                }
                if (sigma > 0.0) {
                    cn_subcut = cn_subcut + 1.0;
                }
                tau_depth = 0.0;
""", what="census void")
    src = sv.patch(src, """\
            tau_depth = tau_depth + d_tau;
            tau_view = tau_view + d_tau;
""", """\
            if (cn_seen) {
                cn_gap = cn_gap + cn_pending;
            } else {
                cn_t_first = t;
            }
            cn_pending = 0.0;
            cn_seen = true;
            cn_cloud = cn_cloud + 1.0;
            cn_t_last = t + dt;
            if (d_tau < 1e-4) {
                cn_c4 = cn_c4 + 1.0;
            } else if (d_tau < 1e-3) {
                cn_c3 = cn_c3 + 1.0;
            } else {
                cn_c2 = cn_c2 + 1.0;
            }
            tau_depth = tau_depth + d_tau;
            tau_view = tau_view + d_tau;
""", what="census cloud")
    src = sv.patch(src, """\
    // The water, for rays that leave or miss the box without becoming
""", """\
    var cn_pm = vec3<f32>(0.0);
    if (t_near >= 0.0 && t_near < t_far) {
        // A ray that went opaque has cloud behind its last sample, which
        // the pre-march still integrates: none of its steps count as after
        // the last cloud.
        cn_pm = cn_premarch_census(u.cam_origin, dir, t_near,
                                   min(t_far, t_surface), ahead_jitter,
                                   cn_seen, cn_t_first,
                                   select(cn_t_last, 3.0e38, cn_end == 2.0));
    }
    // The water, for rays that leave or miss the box without becoming
""", what="census premarch")
    outs = {
        0: "vec4<f32>(cn_lead, cn_gap, cn_pending, cn_cloud)",
        1: "vec4<f32>(cn_pm.x, cn_pm.y, cn_pm.z, cn_end)",
        2: "vec4<f32>(cn_t_first, cn_t_last, t_near, t_far)",
        3: "vec4<f32>(cn_subcut, cn_c4, cn_c3, cn_c2)",
    }
    return sv.patch(src, """\
    // Alpha is 1, except under the live steam, where it carries t_front.
""", f"""\
    return {outs[out_pass]};
    // Alpha is 1, except under the live steam, where it carries t_front.
""", what="census out")


def build_sources(surface: str = "ocean") -> dict:
    stock = sv.shader.assemble(surface)
    src = {"stock": stock,
           "oracle": oracle_patch(stock),
           "nomarch": nomarch_patch(stock, live=False),
           "nomarch_live": nomarch_patch(stock, live=True)}
    census = {}
    for k in range(4):
        census[f"census{k}"] = census_patch(stock, k)
    for k in (0, 1):
        census[f"census_oracle{k}"] = census_patch(oracle_patch(stock), k)
    return src, census


# -- the census itself -------------------------------------------------------------

def run_census_pass(r, state, view, variant, group1=None):
    r.set_variant(variant, group1_texture=group1)
    r.render(state, view, frames=1)
    return r.read_sample()


def oracle_inputs(r, state, view, frames):
    """First and last cloud distance per pixel, and whether there is cloud,
    over the entry jitter of frames 0 .. frames-1: the census at each
    frame_index (pixel centres; the 3x3 dilation in oracle_array stands in
    for the sub-pixel jitter), the earliest first, the latest last, cloud
    in any. A thin feature that one frame's sampling phase steps over and
    another's hits is in the oracle."""
    import dataclasses
    tf = tl = seen = opaque = None
    for k in range(frames):
        v = dataclasses.replace(view, frame_index=view.frame_index + k)
        p0 = run_census_pass(r, state, v, "census0")
        p1 = run_census_pass(r, state, v, "census1")
        p2 = run_census_pass(r, state, v, "census2")
        s = p0[..., 3] > 0
        o = p1[..., 3] == 2.0
        f = np.where(s, p2[..., 0], np.inf)
        l = np.where(s, p2[..., 1], -np.inf)
        if seen is None:
            tf, tl, seen, opaque = f, l, s, o
        else:
            tf, tl = np.minimum(tf, f), np.maximum(tl, l)
            seen, opaque = seen | s, opaque | o
    # A ray that went opaque has cloud behind its last sample: the march
    # never gets there, but the pre-march integrates it, and a geometric
    # first/last pre-pass would report that cloud's far end, not where the
    # march stopped. Its far side is left unclamped.
    tl = np.where(opaque, np.inf, tl)
    return tf, tl, seen


def oracle_array(t_first, t_last, seen, tile, dt_view, lod_tan):
    """(h, w, 4) rgba32float: per pixel (t_near, t_far, has_cloud, 0) of the
    oracle clamp. The first/last distances are dilated over 3x3 pixels
    (sub-pixel jitter moves a ray by less than a pixel), then taken over
    tiles of `tile` pixels (1 = per pixel), then widened by
    ORACLE_MARGIN_STEPS steps of the clear-air step at that distance."""
    from scipy.ndimage import maximum_filter, minimum_filter
    big = np.float32(3e38)
    tf = np.where(seen, t_first, big).astype(np.float32)
    tl = np.where(seen, t_last, -big).astype(np.float32)
    tf = np.minimum(tf, big)
    tl = np.maximum(tl, -big)
    tf = minimum_filter(tf, size=3, mode="nearest")
    tl = maximum_filter(tl, size=3, mode="nearest")
    h, w = tf.shape
    if tile > 1:
        H, W = -(-h // tile) * tile, -(-w // tile) * tile
        tfp = np.full((H, W), big, np.float32)
        tlp = np.full((H, W), -big, np.float32)
        tfp[:h, :w], tlp[:h, :w] = tf, tl
        tfb = tfp.reshape(H // tile, tile, W // tile, tile).min(axis=(1, 3))
        tlb = tlp.reshape(H // tile, tile, W // tile, tile).max(axis=(1, 3))
        tf = np.repeat(np.repeat(tfb, tile, 0), tile, 1)[:h, :w]
        tl = np.repeat(np.repeat(tlb, tile, 0), tile, 1)[:h, :w]
    has = tf < big / 2
    step = lambda t: np.maximum(dt_view, t * lod_tan)          # noqa: E731
    near = np.where(has, np.maximum(tf - ORACLE_MARGIN_STEPS * step(tf), 0.0), 0.0)
    far = np.where(has, tl + ORACLE_MARGIN_STEPS * step(np.maximum(tl, 0.0)), 0.0)
    out = np.zeros((h, w, 4), np.float32)
    out[..., 0], out[..., 1], out[..., 2] = near, far, has
    return out


def null_oracle_array(h, w):
    """(h, w, 4) rgba32float for oracle_null: has_cloud everywhere, with a
    range that leaves t_near and t_far as the box test set them."""
    out = np.zeros((h, w, 4), np.float32)
    out[..., 0], out[..., 1], out[..., 2] = -3e38, 3e38, 1.0
    return out


def block_max_mean(a: np.ndarray, block=WARP) -> float:
    """Mean over pixels of the max over each block of pixels (bw x bh): the
    iterations a SIMD group covering that block runs, per pixel."""
    bw, bh = block
    h, w = a.shape
    H, W = h - h % bh, w - w % bw
    b = a[:H, :W].reshape(H // bh, bh, W // bw, bw).max(axis=(1, 3))
    return float(b.mean())


def stats_from(p0, p1=None, p2=None, p3=None) -> dict:
    lead, gap, trail, cloud = (p0[..., k] for k in range(4))
    seen = cloud > 0
    total = lead + gap + trail + cloud
    s = dict(
        pixels=int(lead.size),
        lead=float(lead.mean()), interior=float(gap.mean()),
        trail=float(trail.mean()), cloud=float(cloud.mean()),
        total=float(total.mean()),
        lead_in_nocloud_rays=float(np.where(seen, 0, lead).mean()),
        frac_nocloud=float((~seen).mean()),
        warp_max_total=block_max_mean(total),
        warp_max_interior_cloud=block_max_mean(gap + cloud),
        warp_max_cloud=block_max_mean(cloud),
    )
    if p1 is not None:
        pm, pmb, pma, end = (p1[..., k] for k in range(4))
        s.update(premarch=float(pm.mean()), premarch_before_first=float(pmb.mean()),
                 premarch_after_last=float(pma.mean()),
                 frac_not_entered=float((end == 0).mean()),
                 frac_end_tfar=float((end == 1).mean()),
                 frac_end_opaque=float((end == 2).mean()),
                 frac_end_sea=float((end == 3).mean()),
                 frac_end_stepcap=float((end == 4).mean()),
                 frac_nocloud_entered=float(((~seen) & (end != 0)).mean()))
    if p2 is not None:
        tf, tl = p2[..., 0], p2[..., 1]
        if seen.any():
            s.update(t_first_median_m=float(np.median(tf[seen])),
                     t_last_median_m=float(np.median(tl[seen])))
    if p3 is not None:
        s.update(subcut=float(p3[..., 0].mean()), cloud_dtau_lt_1e4=float(p3[..., 1].mean()),
                 cloud_dtau_lt_1e3=float(p3[..., 2].mean()),
                 cloud_dtau_ge_1e3=float(p3[..., 3].mean()))
    return s


def setup_demo(demo, device, presets, out_w, out_h, demo_dir=None):
    level, meta = sv.load_demo(demo, demo_dir)
    periodic = bool(meta["periodic"])
    stock_and, census = build_sources("ocean")
    r = sv.VariantRenderer(stock_and, device=device, periodic=periodic)
    for name, s in stock_and.items():
        if name != "stock":
            r.add_variant(name, s)
    for name, s in census.items():
        r.add_variant(name, s, f32_output=True)
    r.upload_volume(level.sigma)
    views, placement = sv.demo_views(level, meta, periodic)
    return level, meta, periodic, r, views, placement


def lod_tan(look):
    return float(np.tan(np.radians(look["view_step_lod_degrees"])))


def census_main(args):
    presets = sv.flight_presets()
    out_w, out_h = (int(v) for v in args.output.lower().split("x"))
    _, device = sv.make_device(timestamps=False, float32_filterable=True)
    out = open(args.out, "a")
    for demo in args.demos.split(","):
        t0 = time.time()
        level, meta, periodic, r, views, placement = setup_demo(
            demo, device, presets, out_w, out_h, args.demo_dir)
        print(f"{demo}: periodic={periodic} placement="
              + json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                            for k, v in placement.items()}), flush=True)
        out.write(json.dumps(dict(kind="placement", demo=demo, periodic=periodic,
                                  placement=placement, views=views,
                                  shape=list(level.sigma.shape),
                                  dx=[float(d) for d in level.dx])) + "\n")
        for tier in sorted(args.tiers.split(","), key=sv.TIER_ORDER.index):
            look = sv.tier_look(presets, tier)
            if args.bench_look:
                # soar_bench's ViewState: the tier's sampling and light
                # cache, every look default (haze, both LOD angles) left
                # as ViewState has them.
                d = sv.ViewState(camera_position=[0, 0, 0], azimuth=0,
                                 elevation=0, fov=1, output_size=(1, 1),
                                 render_size=(1, 1))
                look = dict(light_cache=look["light_cache"], haze=d.haze,
                            haze_height_dependent=d.haze_height_dependent,
                            light_march_lod_degrees=d.light_march_lod_degrees,
                            view_step_lod_degrees=d.view_step_lod_degrees)
            state = sv.tier_scene(r, level, presets, tier, periodic)
            size = sv.render_size(presets, tier, out_w, out_h)
            for name in views:
                if args.views and name not in args.views.split(","):
                    continue
                view = sv.make_view(views[name], look, (out_w, out_h), size)
                r.use_zero_field(False)
                if args.bench_look:
                    p = [run_census_pass(r, state, view, f"census{k}") for k in range(2)]
                    out.write(json.dumps(dict(
                        kind="census_bench_look", demo=demo, tier=tier, view=name,
                        size=list(size), look=look, dt_view=state.dt_view,
                        stock=stats_from(*p))) + "\n")
                    out.flush()
                    continue
                p = [run_census_pass(r, state, view, f"census{k}") for k in range(4)]
                rec = dict(kind="census", demo=demo, tier=tier, view=name,
                           size=list(size), look=look,
                           dt_view=state.dt_view, dt_light=state.dt_light)
                rec["stock"] = stats_from(*p)
                tf, tl, seen = oracle_inputs(r, state, view, args.oracle_frames)
                rec["oracle_frames"] = args.oracle_frames
                rec["frac_cloud_any_frame"] = float(seen.mean())
                lt = lod_tan(look)
                for tile in (1,) + ORACLE_TILES:
                    arr = oracle_array(tf, tl, seen, tile, state.dt_view, lt)
                    tex = r.data_texture(arr)
                    q0 = run_census_pass(r, state, view, "census_oracle0", tex)
                    q1 = run_census_pass(r, state, view, "census_oracle1", tex)
                    rec[f"oracle_{'px' if tile == 1 else f't{tile}'}"] = stats_from(q0, q1)
                    tex.destroy()
                r.set_variant("stock")
                r.use_zero_field(True)
                z0 = run_census_pass(r, state, view, "census0")
                z1 = run_census_pass(r, state, view, "census1")
                rec["zero"] = stats_from(z0, z1)
                r.use_zero_field(False)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                s = rec["stock"]
                print(f"  {tier:8s}{name:18s} lead {s['lead']:7.1f} int {s['interior']:6.1f} "
                      f"trail {s['trail']:6.1f} cloud {s['cloud']:6.1f} pm {s['premarch']:5.1f} "
                      f"nocloud {s['frac_nocloud']:.2f} | zero total {rec['zero']['total']:7.1f}"
                      f" | oracle_px total {rec['oracle_px']['total']:6.1f}"
                      f" cloud {rec['oracle_px']['cloud']:6.1f}", flush=True)
        print(f"{demo}: {time.time() - t0:.0f} s", flush=True)
        del r
    out.close()


# -- timing --------------------------------------------------------------------------

def timing_main(args):
    presets = sv.flight_presets()
    out_w, out_h = (int(v) for v in args.output.lower().split("x"))
    _, device = sv.make_device(timestamps=True, float32_filterable=True)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sv.REPO,
                         capture_output=True, text=True).stdout.strip()
    cells = None
    if args.cells:
        cells = {tuple(c) for c in json.loads(Path(args.cells).read_text())}
        args.demos = ",".join(dict.fromkeys(c[0] for c in cells))
        args.tiers = ",".join(dict.fromkeys(c[1] for c in cells))
    out = open(args.out, "a")
    for demo in args.demos.split(","):
        level, meta, periodic, r, views, placement = setup_demo(
            demo, device, presets, out_w, out_h, args.demo_dir)
        if not r.timestamps:
            print("no timestamp-query on this device: GPU times are absent",
                  flush=True)
        for tier in sorted(args.tiers.split(","), key=sv.TIER_ORDER.index):
            look = sv.tier_look(presets, tier)
            state = sv.tier_scene(r, level, presets, tier, periodic)
            size = sv.render_size(presets, tier, out_w, out_h)
            for name in views:
                if args.views and name not in args.views.split(","):
                    continue
                if cells is not None and (demo, tier, name) not in cells:
                    continue
                view = sv.make_view(views[name], look, (out_w, out_h), size)
                r.use_zero_field(False)
                tf, tl, seen = oracle_inputs(r, state, view, args.frames)
                lt = lod_tan(look)
                tex_px = r.data_texture(oracle_array(tf, tl, seen, 1,
                                                     state.dt_view, lt))
                tex_t16 = r.data_texture(oracle_array(tf, tl, seen, 16,
                                                      state.dt_view, lt))
                tex_null = r.data_texture(null_oracle_array(*tf.shape))
                arms = [sv.Arm("full", "stock"),
                        sv.Arm("zero", "stock", zero_field=True),
                        sv.Arm("nomarch", "nomarch"),
                        sv.Arm("nomarch_live", "nomarch_live"),
                        sv.Arm("oracle_px", "oracle", group1_texture=tex_px),
                        sv.Arm("oracle_t16", "oracle", group1_texture=tex_t16),
                        sv.Arm("oracle_null", "oracle", group1_texture=tex_null)]
                res = sv.time_variants(r, state, view, arms, frames=args.frames,
                                       repeats=args.repeats)
                tex_px.destroy()
                tex_t16.destroy()
                tex_null.destroy()
                rec = dict(kind="timing", label=args.label, rev=rev, demo=demo,
                           tier=tier, view=name, size=list(size),
                           output=[out_w, out_h], frames=args.frames,
                           repeats=args.repeats, when=time.strftime("%H:%M:%S"),
                           samples=res,
                           summary={k: {m: sv.summarize(v[m]) for m in v}
                                    for k, v in res.items()})
                out.write(json.dumps(rec) + "\n")
                out.flush()
                sm = rec["summary"]
                g = lambda k: sm[k]["gpu_ms"]["min"]                     # noqa: E731
                sp = max((sm[k]["gpu_ms"]["spread"] or 0) for k in sm)
                print(f"{demo:11s}{tier:8s}{name:18s} gpu ms: full {g('full'):7.3f} "
                      f"zero {g('zero'):7.3f} nomarch {g('nomarch'):6.3f} "
                      f"live {g('nomarch_live'):6.3f} oracle_px {g('oracle_px'):7.3f} "
                      f"t16 {g('oracle_t16'):7.3f} null {g('oracle_null'):7.3f}  wall full "
                      f"{sm['full']['wall_ms']['min']:7.2f}  max spread {sp:.2f}",
                      flush=True)
        del r
    out.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("census", "timing"):
        p = sub.add_parser(cmd)
        p.add_argument("--demos", default=DEMOS_DEFAULT)
        p.add_argument("--demo-dir", default=None,
                       help="where the baked demos are (default web/soar/demos)")
        p.add_argument("--tiers", default=TIERS_DEFAULT)
        p.add_argument("--views", default="", help="comma-separated; default all")
        p.add_argument("--output", default="2560x1440",
                       help="the window; each tier renders at its scale of it")
        p.add_argument("--out", required=True, help="JSON lines, appended")
        if cmd == "census":
            p.add_argument("--bench-look", action="store_true",
                           help="count the stock march only, under "
                                "soar_bench's look (ViewState's defaults "
                                "for haze and the LOD angles)")
            p.add_argument("--oracle-frames", type=int, default=8,
                           help="frames whose cloud distances the oracle "
                                "covers (timing's --frames)")
        if cmd == "timing":
            p.add_argument("--frames", type=int, default=8)
            p.add_argument("--repeats", type=int, default=5)
            p.add_argument("--label", default="run")
            p.add_argument("--cells", default=None,
                           help="a JSON list of [demo, tier, view] to time "
                                "(for rerunning noisy cells); overrides "
                                "--demos and --tiers")
    p = sub.add_parser("report")
    p.add_argument("jsonl", nargs="+")
    p.add_argument("--md", required=True)
    p.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.cmd == "census":
        census_main(args)
    elif args.cmd == "timing":
        timing_main(args)
    else:
        from soar_void_census_report import report_main
        report_main(args)


if __name__ == "__main__":
    main()
