#!/usr/bin/env python
"""Hold two revisions of raymarch.wgsl against each other, in raw float.

The eight golden images (tests/test_soar_witness_renders.py) are the look
regression and they compare 8-bit PNGs at RMSE 0.006 / max 0.12 — by their
own measurement they cannot separate a 3 percent exposure nudge from a
sampling difference. A change that claims to preserve the image exactly needs
a sharper instrument than that, and this is it: both revisions compiled on one
device, the same deterministic frames, the float32 accumulator compared
without ever passing through 8 bits.

    tools/gpulock uv run python tools/soar_shader_ab.py                     # HEAD vs tree
    tools/gpulock uv run python tools/soar_shader_ab.py --a git:HEAD~3 --b git:HEAD
    tools/gpulock uv run python tools/soar_shader_ab.py --frames 64 --jitter
    tools/gpulock uv run python tools/soar_shader_ab.py --periodic off --views repro
    tools/gpulock uv run python tools/soar_shader_ab.py --tier low --frames 4 --jitter
    tools/gpulock uv run python tools/soar_shader_ab.py --scene city --frames 4 --jitter
    tools/gpulock uv run python tools/soar_shader_ab.py --scene nested --frames 4 --jitter

`--tier` renders with a quality preset's flight sampling and lighting methods
(step factors, the baked sun-tau cache, the sky probe) rather than the judge
views' High-style live march, because those are different code paths: every
tier but Max reads the cache, and a change that is neutral on one can move the
other. `--scene city` compiles CITY in and renders the night-city harness's
views under its moon; `--scene nested` binds a synthetic nest (the outer
field's central box, refined 2x) so the NESTED paths run. A claim of
bit-identity should be backed by all three scenes, not just the day one.

One thing to know before reading a result, because it cost an afternoon to
learn: a difference of order 1e-4 on a handful of channels does NOT
necessarily mean the two revisions compute different things. They are two
different programs, and the driver is free to contract `a + b * c` into an FMA
in one and not the other; on this box that shows up as roughly one channel in
270000, always in a long accumulation (the forward pre-march's tau, which then
crosses its own cap at a different step and amplifies). If a change is meant
to be exactly neutral and lands there instead of at zero, the way to settle it
is to make the two runs the SAME compilation and vary the data instead.

Anything larger than that is a real difference and should be explained.

What this cannot compare: a change that moves the HOSTS as well as the
shader — a new binding, a new pass, a changed bind-group layout. Both
revisions run under this tree's one Python host, so the old shader meets the
new layout and fails validation (the surface pre-pass, 2026-09-23, broke it
for the city). For those, compare whole browser builds
(tools/pinned_still.mjs renders pinned stills of each tree in Chrome and
hashes them byte for byte), or run this tool in each tree with its own host:
`--save DIR` in the one writes every view's float render there, and
`--against DIR` in the other renders only its B and holds it against those
(the desert's removal, 2026-11-09, moved the uniform block and two bindings).
A tree from before --save existed takes a copy of this file anywhere inside
it (REPO is found from the file's own path). The run's settings are written
beside the renders (run.json) and an --against run with others is refused.

    cp tools/soar_shader_ab.py ../old-tree/temp/ab.py     # if it lacks --save
    (cd ../old-tree && uv run python temp/ab.py --a tree --b tree --save /tmp/ab-old)
    uv run python tools/soar_shader_ab.py --b tree --against /tmp/ab-old
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from conftest import SOAR_VIEWS, build_soar_level          # noqa: E402
from cloudyview.soar_host import (                          # noqa: E402
    SceneState, SoarRenderer,
    ViewState, camera_world_origin, read_shader,
    surface_cell_m, surface_clock_period_s, surface_period_m,
)
from cloudyview.witness import OCEAN_REFLECTANCE   # noqa: E402

sys.path.insert(0, str(REPO / "tools"))
from night_city_harness import (                            # noqa: E402
    CITY_VIEWS, MOON_AZIMUTH, MOON_ELEVATION, NIGHT_EXPOSURE,
)

# Views the judge set does not cover, for the failure modes that live off it.
EXTRA_VIEWS = {
    # Down a wrap axis from just inside the east face: crosses the periodic
    # seam every domain width.
    "wrap_seam": {"camera_position": [0.95, 0.0, -0.9], "azimuth": 90,
                  "elevation": 0, "fov": 90},
    # docs/soar-bugs.md 6: outside the domain, looking down, periodicity off.
    # Most of these rays MISS the box, which is its own code path.
    "repro": {"camera_position": [1.25, 1.25, 1.2], "azimuth": 225,
              "elevation": -35, "fov": 95},
}
ALL_VIEWS = {**SOAR_VIEWS, **EXTRA_VIEWS}

# Mirrors web/soar/constants.js QUALITY_PRESETS (flight sampling):
# (step_factor, light_step_factor, light cache, sky probe). Max is High with
# the live sun march.
TIERS = {
    "max":     (2.0, 2.0, False, True),
    "high":    (2.0, 2.0, True, True),
    "medium":  (2.5, 4.0, True, False),
    "low":     (3.0, 8.0, True, False),
    "minimal": (4.0, 12.0, True, False),
    # Not a preset: the method a cache tier flies with while its sun-tau bake
    # is still in flight (the renderer reads the live march until the cache
    # is ready). Low's sampling, no cache, no probe — the fourth lighting
    # specialization the browser compiles.
    "baking":  (3.0, 8.0, False, False),
}
LIGHT_CACHE_DIVISOR = 2       # constants.js


def synthetic_nest(level):
    """The outer field's central box, refined 2x by trilinear resampling and
    given a mild multiplicative texture so it is not the parent's own values.
    Only its code paths are under test, not its look."""
    from scipy.ndimage import zoom
    from cloudyview.witness import NestedLevel

    sigma = np.asarray(level.sigma, dtype=np.float64)
    n = np.array(sigma.shape)
    lo = n * np.array([0.3, 0.3, 0.2])
    hi = n * np.array([0.7, 0.7, 0.8])
    lo, hi = lo.astype(int), hi.astype(int)
    sub = sigma[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    fine = zoom(sub, 2, order=1)
    i, j, k = np.meshgrid(*[np.arange(m) for m in fine.shape], indexing="ij")
    fine = fine * (1.0 + 0.3 * np.sin(0.7 * i) * np.cos(0.5 * j + 0.3 * k))
    dx = (np.asarray(level.bmax) - np.asarray(level.bmin)) / n
    bmin = np.asarray(level.bmin) + lo * dx
    bmax = np.asarray(level.bmin) + hi * dx
    return NestedLevel(sigma=np.ascontiguousarray(fine), bmin=bmin, bmax=bmax,
                       name="synthetic_nest")


def source_for(spec: str) -> str:
    """'tree', 'git:<rev>', or a path."""
    if spec == "tree":
        return read_shader()
    if spec.startswith("git:"):
        # The shader moved from web/soar/ into the package 2026-08-18;
        # web/soar/raymarch.wgsl is a symlink since then, and `git show` of a
        # symlink path yields the link target text, not the shader. Try the
        # package path first, then the old location for pre-move revisions.
        errors = []
        for relpath in ("cloudyview/soar/raymarch.wgsl",
                        "web/soar/raymarch.wgsl"):
            out = subprocess.run(
                ["git", "show", f"{spec[4:]}:{relpath}"],
                cwd=REPO, capture_output=True, text=True)
            if out.returncode == 0 and "fn " in out.stdout:
                return out.stdout
            errors.append(out.stderr.strip())
        raise SystemExit(f"git show failed for {spec!r}:\n" + "\n".join(errors))
    return Path(spec).read_text()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default=None,
                    help="baseline revision (default git:HEAD)")
    ap.add_argument("--b", default="tree", help="candidate revision")
    ap.add_argument("--size", type=int, nargs=2, default=(400, 225))
    ap.add_argument("--frames", type=int, default=1,
                    help="accumulated passes (1 = one deterministic frame)")
    ap.add_argument("--jitter", action="store_true",
                    help="enable the stochastic streams (needs --frames > 1 "
                         "to mean anything)")
    ap.add_argument("--periodic", default="on", choices=("on", "off"))
    ap.add_argument("--views", default="",
                    help="comma-separated names, default all")
    ap.add_argument("--scene", default="day", choices=("day", "city", "nested"))
    ap.add_argument("--field", type=Path, default=None,
                    help="netCDF field (default: the judge views' TWP-ICE "
                         "subvolume). An optimization that gates on the cloud "
                         "should be proven on the fields where it fires.")
    ap.add_argument("--demo", type=Path, default=None,
                    help="a prebaked demo directory (meta.json + "
                         "volume.bin.gz) instead of a netCDF field: the "
                         "exact bytes the browser ships")
    ap.add_argument("--tier", default=None, choices=sorted(TIERS),
                    help="a quality preset's flight sampling and lighting "
                         "methods (default: the judge views' 2-voxel live "
                         "march with the sky probe)")
    ap.add_argument("--save", type=Path, default=None,
                    help="write each view's A render (float32 .npy) here, "
                         "for an --against run in another tree")
    ap.add_argument("--against", type=Path, default=None,
                    help="hold B against the renders a --save run wrote "
                         "(another tree, under its own host) instead of "
                         "rendering A here")
    # It opens a GPU device: refuse outside tools/gpulock (gpulock_guard),
    # before the parse: -h/--help still answers unlocked.
    from gpulock_guard import require_gpulock
    require_gpulock("tools/soar_shader_ab.py")
    args = ap.parse_args()
    # What a saved set was rendered under: an --against run under any other
    # settings would compare different pictures.
    run = {"size": list(args.size), "frames": args.frames,
           "jitter": bool(args.jitter), "periodic": args.periodic,
           "scene": args.scene, "tier": args.tier,
           "field": str(args.field) if args.field else None,
           "demo": str(args.demo) if args.demo else None,
           "views": args.views}
    if args.save is not None and args.against is not None:
        ap.error("--save and --against: save in one tree, compare in the other")
    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        (args.save / "run.json").write_text(json.dumps(run, indent=1))
    if args.against is not None:
        saved = json.loads((args.against / "run.json").read_text())
        if saved != run:
            ap.error(f"{args.against} was rendered under {saved}; this run is "
                     f"{run}.")
        if args.a is not None:
            ap.error("--a and --against: A is the saved set")
    elif args.a is None:
        args.a = "git:HEAD"

    import wgpu
    device = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync()

    if args.demo is not None:
        sys.path.insert(0, str(REPO / "tools" / "benchmarking"))
        from soar_frame_bench import build_level_from_demo
        level, _ = build_level_from_demo(args.demo)
    else:
        level = build_soar_level(args.field)
    periodic = args.periodic == "on"
    city = args.scene == "city"
    nested = args.scene == "nested"
    surface = {"day": "ocean", "nested": "ocean", "city": "city"}[args.scene]
    step_factor, light_factor, light_cache, sky_probe = (
        TIERS[args.tier] if args.tier else (2.0, 2.0, False, True))
    state = SceneState(
        bmin=[float(v) for v in level.bmin], bmax=[float(v) for v in level.bmax],
        dt_view=min(level.dx) * step_factor,
        dt_light=min(level.dx) * light_factor, periodic=periodic,
        ocean_reflectance=OCEAN_REFLECTANCE, surface=surface, nested=nested)
    nest = synthetic_nest(level) if nested else None
    if nested:
        state.nest_bmin = [float(v) for v in nest.bmin]
        state.nest_bmax = [float(v) for v in nest.bmax]
        state.dt_view_nest = min(nest.dx) * step_factor
        state.dt_light_nest = min(nest.dx) * light_factor

    renderers = []
    # Against a saved set, A is not rendered here: B alone is built.
    for spec in ((args.b,) if args.against is not None
                 else (args.a, args.b)):
        r = SoarRenderer(device=device, periodic=periodic, nested=nested,
                         surface=surface, shader_source=source_for(spec))
        r.upload_volume(level.sigma)
        if nested:
            r.upload_nest(nest.sigma)
        renderers.append(r)
    meta = renderers[-1].surface_meta
    state.ocean_fif_dx = surface_cell_m(surface, meta)
    state.ocean_tile_extent = float(meta["tile_extent_m"])
    state.ocean_max_lod = int(meta["mips"]) - 1
    state.surface_period_m = surface_period_m(surface, meta)
    state.clock_period_s = surface_clock_period_s(surface, meta)

    views = ALL_VIEWS
    if city:
        bmin, bmax = level.bmin, level.bmax
        views = {
            name: {"world": [float(bmin[0] + v["xy"][0] * (bmax[0] - bmin[0])),
                             float(bmin[1] + v["xy"][1] * (bmax[1] - bmin[1])),
                             float(v["z"])],
                   "azimuth": v["azimuth"], "elevation": v["elevation"],
                   "fov": v["fov"], "sun_azimuth": MOON_AZIMUTH,
                   "sun_elevation": MOON_ELEVATION,
                   "exposure": NIGHT_EXPOSURE}
            for name, v in CITY_VIEWS.items()}
    wanted = args.views.split(",") if args.views else list(views)
    w, h = args.size
    print(f"A = {args.against or args.a}\nB = {args.b}")
    print(f"{w}x{h}, {args.frames} frame(s), jitter "
          f"{'on' if args.jitter else 'off'}, periodic {args.periodic}, "
          f"scene {args.scene}, tier {args.tier or 'judge (live march)'}\n")
    print(f"{'view':22s} {'max |diff|':>12s} {'channels':>10s} {'fraction':>10s}")
    worst = 0.0
    baked_sun = None
    for name in wanted:
        v = views[name]
        pos = v["world"] if "world" in v else camera_world_origin(
            v["camera_position"], level.bmin, level.bmax)
        extra = {"exposure": v["exposure"]} if "exposure" in v else {}
        view = ViewState(
            camera_position=[float(p) for p in pos],
            azimuth=v["azimuth"], elevation=v["elevation"],
            fov=v.get("fov", 70), output_size=(w, h), render_size=(w, h),
            sun_azimuth=v.get("sun_azimuth", 20.0),
            sun_elevation=v.get("sun_elevation", 55.0),
            jitter=args.jitter, light_cache=light_cache,
            sky_probe=sky_probe,
            **extra)
        sun = (view.sun_azimuth, view.sun_elevation)
        if light_cache and sun != baked_sun:
            for r in renderers:
                r.bake_light_cache(state, view, divisor=LIGHT_CACHE_DIVISOR)
            baked_sun = sun
        if args.against is not None:
            a = np.load(args.against / f"{name}.npy")
        else:
            a = renderers[0].render(state, view, frames=args.frames)
        if args.save is not None:
            np.save(args.save / f"{name}.npy", a)
        b = renderers[-1].render(state, view, frames=args.frames)
        d = np.abs(a - b)
        n = int((d > 0).sum())
        worst = max(worst, float(d.max()))
        print(f"{name:22s} {d.max():12.4g} {n:10d} {n / d.size:10.6f}")
    print(f"\nworst = {worst:.6g}"
          + ("   (exactly identical)" if worst == 0.0 else ""))


if __name__ == "__main__":
    main()
