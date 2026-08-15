#!/usr/bin/env python
"""Hold two revisions of raymarch.wgsl against each other, in raw float.

The eight golden images (tests/test_soar_witness_renders.py) are the look
regression and they compare 8-bit PNGs at RMSE 0.006 / max 0.12 — by their
own measurement they cannot separate a 3 percent exposure nudge from a
sampling difference. A change that claims to preserve the image exactly needs
a sharper instrument than that, and this is it: both revisions compiled on one
device, the same deterministic frames, the float32 accumulator compared
without ever passing through 8 bits.

    uv run python tools/soar_shader_ab.py                     # HEAD vs tree
    uv run python tools/soar_shader_ab.py --a git:HEAD~3 --b git:HEAD
    uv run python tools/soar_shader_ab.py --frames 64 --jitter
    uv run python tools/soar_shader_ab.py --periodic off --views repro

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
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from conftest import SOAR_VIEWS, build_soar_level          # noqa: E402
from cloudyview.soar_host import (                          # noqa: E402
    SceneState, SoarRenderer, ViewState, camera_world_origin, read_shader,
)
from cloudyview.witness import OCEAN_REFLECTANCE   # noqa: E402

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


def source_for(spec: str) -> str:
    """'tree', 'git:<rev>', or a path."""
    if spec == "tree":
        return read_shader()
    if spec.startswith("git:"):
        out = subprocess.run(
            ["git", "show", f"{spec[4:]}:web/soar/raymarch.wgsl"],
            cwd=REPO, capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"git show failed for {spec!r}:\n{out.stderr}")
        return out.stdout
    return Path(spec).read_text()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="git:HEAD", help="baseline revision")
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
    args = ap.parse_args()

    import wgpu
    device = wgpu.gpu.request_adapter_sync(
        power_preference="high-performance").request_device_sync()

    level = build_soar_level()
    periodic = args.periodic == "on"
    dt = min(level.dx) * 2.0
    state = SceneState(
        bmin=[float(v) for v in level.bmin], bmax=[float(v) for v in level.bmax],
        dt_view=dt, dt_light=dt, periodic=periodic,
        ocean_reflectance=OCEAN_REFLECTANCE)

    renderers = []
    for spec in (args.a, args.b):
        r = SoarRenderer(device=device, periodic=periodic, nested=False,
                         shader_source=source_for(spec))
        r.upload_volume(level.sigma)
        renderers.append(r)

    wanted = args.views.split(",") if args.views else list(ALL_VIEWS)
    w, h = args.size
    print(f"A = {args.a}\nB = {args.b}")
    print(f"{w}x{h}, {args.frames} frame(s), jitter "
          f"{'on' if args.jitter else 'off'}, periodic {args.periodic}\n")
    print(f"{'view':22s} {'max |diff|':>12s} {'channels':>10s} {'fraction':>10s}")
    worst = 0.0
    for name in wanted:
        v = ALL_VIEWS[name]
        pos = camera_world_origin(v["camera_position"], level.bmin, level.bmax)
        view = ViewState(
            camera_position=[float(p) for p in pos],
            azimuth=v["azimuth"], elevation=v["elevation"],
            fov=v.get("fov", 70), output_size=(w, h), render_size=(w, h),
            sun_azimuth=v.get("sun_azimuth", 20.0),
            sun_elevation=v.get("sun_elevation", 55.0),
            jitter=args.jitter)
        a = renderers[0].render(state, view, frames=args.frames)
        b = renderers[1].render(state, view, frames=args.frames)
        d = np.abs(a - b)
        n = int((d > 0).sum())
        worst = max(worst, float(d.max()))
        print(f"{name:22s} {d.max():12.4g} {n:10d} {n / d.size:10.6f}")
    print(f"\nworst = {worst:.6g}"
          + ("   (exactly identical)" if worst == 0.0 else ""))


if __name__ == "__main__":
    main()
