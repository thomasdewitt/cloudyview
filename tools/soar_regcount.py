#!/usr/bin/env python
"""What the GPU driver compiled a soar shader into: registers and binary size.

fs_main's speed on this renderer is set as much by how many registers the
driver allocates as by what executes (raymarch.wgsl, "FS_MAIN IS ONE
COMPILE"): code that never runs still costs occupancy, and a kernel over the
driver's register budget runs few warps at once. wgpu does not expose that
number. This does: it specializes raymarch.wgsl exactly as the hosts do
(soar_host.specialize), compiles it to SPIR-V with naga (the compiler
wgpu-native uses), builds the pipeline through Vulkan with
VK_KHR_pipeline_executable_properties, and prints the driver's own
statistics per entry point (tools/regcount, a small Rust program, built on
first use with cargo).

    uv run python tools/soar_regcount.py                      # fs_main, day, Low
    uv run python tools/soar_regcount.py --surface desert --entry fs_surface
    uv run python tools/soar_regcount.py --shader my_variant.wgsl --tier max
    uv run python tools/soar_regcount.py --a git:HEAD --b tree   # side by side

Numbers are for THIS driver (NVIDIA on Vulkan: Register Count, Binary Size,
Stack Size). Apple's compiler makes different choices; treat a count as a
statement about this card that usually generalizes, and time on the Mac.
Pure compile — no GPU work runs, so it does not need tools/gpulock.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cloudyview.soar_host import (DESERT_FLORA_CUT_BY_TIER,  # noqa: E402
                                  DESERT_LITE_TIERS, read_shader, specialize)

CRATE = REPO / "tools" / "regcount"
BINARY = CRATE / "target" / "release" / "regcount"
# Entry point -> the color-attachment formats its pipeline renders into, one
# per fragment output: the first and last surface stages also write the veil.
FORMATS = {"fs_main": "rgba16f", "fs_surface": "rgba32f,rgba32ui",
           "fs_surface_refine": "rgba32f",
           "fs_surface_street": "rgba32f,rgba32ui", "fs_bake_light": "r16f"}
# The lighting method per tier (web/soar/constants.js QUALITY_PRESETS):
# (light cache, sky probe).
TIERS = {"max": (False, True), "high": (True, True), "medium": (True, False),
         "low": (True, False), "minimal": (True, False)}


def source_for(spec: str) -> str:
    """'tree', 'git:<rev>', or a path — soar_shader_ab.py's spec, without its
    test-suite imports, so this runs in any venv (no pytest)."""
    if spec == "tree":
        return read_shader()
    if spec.startswith("git:"):
        out = subprocess.run(
            ["git", "show", f"{spec[4:]}:cloudyview/soar/raymarch.wgsl"],
            cwd=REPO, capture_output=True, text=True)
        if out.returncode != 0 or "fn " not in out.stdout:
            raise SystemExit(f"git show failed for {spec!r}: {out.stderr.strip()}")
        return out.stdout
    return Path(spec).read_text()


def build():
    # Always: cargo is a no-op when the binary is current, and a binary
    # built before a change to the crate (the veil's second attachment,
    # "rgba32f,rgba32ui") panics on formats it does not know.
    subprocess.run(["cargo", "build", "--release", "--quiet"],
                   cwd=CRATE, check=True)


def stats(source: str, entry: str) -> dict:
    """The driver's statistics for the fragment stage of `entry`."""
    with tempfile.NamedTemporaryFile("w", suffix=".wgsl", delete=False) as f:
        f.write(source)
        path = f.name
    out = subprocess.run([str(BINARY), path, entry, FORMATS[entry]],
                         capture_output=True, text=True)
    Path(path).unlink()
    if out.returncode != 0:
        raise SystemExit(f"regcount failed for {entry}:\n{out.stderr[-2000:]}")
    frag = out.stdout[out.stdout.index("FS (FRAGMENT)"):]
    got = {k: int(v) for k, v in
           re.findall(r"^\s+(Register Count|Binary Size|Stack Size)\s+(\d+)",
                      frag, re.M)}
    got["compile ms"] = int(re.search(r"pipeline compile (\d+) ms",
                                      out.stdout).group(1))
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shader", default="tree",
                    help="tree, git:REV, or a .wgsl path (default: tree)")
    ap.add_argument("--a", default=None, help="compare: baseline revision")
    ap.add_argument("--b", default=None, help="compare: candidate revision")
    ap.add_argument("--entry", default="fs_main", choices=sorted(FORMATS))
    ap.add_argument("--surface", default="ocean",
                    choices=("ocean", "desert", "city"))
    ap.add_argument("--tier", default="low", choices=sorted(TIERS))
    ap.add_argument("--periodic", default="on", choices=("on", "off"))
    ap.add_argument("--nested", action="store_true")
    args = ap.parse_args()
    build()

    light_cache, sky_probe = TIERS[args.tier]
    # As both hosts do: a desert at a lite tier compiles the simplified desert.
    desert_lite = args.surface == "desert" and args.tier in DESERT_LITE_TIERS
    flora_cut = (DESERT_FLORA_CUT_BY_TIER[args.tier]
                 if args.surface == "desert" else 1.0)
    specs = [args.a, args.b] if args.a and args.b else [args.shader]
    print(f"{args.entry}, surface {args.surface}, tier {args.tier} "
          f"(cache {light_cache}, probe {sky_probe}), periodic {args.periodic}"
          f"{', nested' if args.nested else ''}"
          f"{', desert lite' if desert_lite else ''}")
    for spec in specs:
        src = specialize(source_for(spec), periodic=args.periodic == "on",
                         nested=args.nested, max_light_steps=512,
                         surface=args.surface, light_cache=light_cache,
                         sky_probe=sky_probe, desert_lite=desert_lite,
                         desert_flora_cut=flora_cut)
        s = stats(src, args.entry)
        print(f"  {spec:28s} registers {s['Register Count']:4d}   "
              f"binary {s['Binary Size'] / 1024:7.0f} KB   "
              f"stack {s['Stack Size']:5d} B   compile {s['compile ms']:6d} ms")


if __name__ == "__main__":
    main()
