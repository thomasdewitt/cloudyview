#!/usr/bin/env python
"""Before/after contact sheets of soar's scenes: one labelled PNG per scene.

    tools/gpulock uv run python tools/contact_sheet.py
    tools/gpulock uv run python tools/contact_sheet.py --after 1a2b3c4 --scenes city
    tools/gpulock uv run python tools/contact_sheet.py --list

Every view is rendered twice — at BEFORE (default: the start of the
2026-09-23/24 realism push, 3fc2eea) and at AFTER (default: HEAD) — by each
revision's OWN code: the ref is exported with `git archive` into
temp/contact/trees/<sha>/ and a subprocess puts that tree first on sys.path
(and checks it did) before running the tree's `witness` CLI or its
tools/night_city_harness.py. That is the point of doing it this way rather
than swapping shaders under one host (tools/soar_shader_ab.py): across a day
of work the hosts, bindings and tiles moved too, and only a revision's own
code renders what that revision looked like.

The views are fixed, so a later run compares like with like:

  * ocean — `witness` over the TWP-ICE 256^2 subvolume: glitter toward a low
    sun, mid-altitude whitecaps, an aerial, and one close view at two scene
    clocks (a revision without the clock renders it once, marked so).
  * city — the night-city harness in cyberpunk mode's frame (its tile
    offset and moon, web/soar/constants.js), cameras in city-tile metres so
    they name the same streets whatever field stands over the tile.
  * desert — `witness --surface desert`, the full ground (--quality high)
    beside the LITE ground (--quality low, as the app flies its low tiers).
    A revision without the desert shows "n/a".

Cameras are world metres (x, y as fractions of the field's box, z absolute
metres above sea level) and are converted to each revision's own relative
frame with that revision's own field box. Renders are deterministic (fixed
exposure, frame index from 0) and cached per commit under
temp/contact/renders/<sha>/, so a re-run only renders what changed;
`--after tree` renders the working tree and is never cached. Outputs go to
temp/contact/ on disk, never /tmp. Run it under tools/gpulock: it holds the
GPU for the whole sheet.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "temp" / "contact"
FIELD = REPO / "data" / "TWPICE_subvolume_256x256_5km.nc"
DEFAULT_BEFORE = "3fc2eea"
SIZE = (960, 540)
CELL = (640, 360)          # each render's size on the sheet

# Cyberpunk mode's frame (web/soar/constants.js CITY_TILE_OFFSET_M,
# CITY_MOON, CITY_START_CAMERA): the city a visitor lands in.
CITY_OFFSET = (8100.0, -44190.0)
CITY_MOON = (181.0, 20.0)
LANDING = (17393.0, 61646.0)
# A street 23 m west of the landing, on the grid line x = 17370 m, urban on
# both sides from y 61110 to 62010 m (the city tile's street words): where
# the street-level cameras stand.
STREET_X = 17370.0


@dataclasses.dataclass(frozen=True)
class View:
    name: str
    label: str
    kind: str                       # "witness" or "harness"
    # witness: (x_frac, y_frac, z_m, azimuth, elevation, fov, sun_az, sun_el)
    # harness: (city_x_m, city_y_m, z_m, azimuth, elevation, fov)
    camera: tuple
    flags: tuple = ()               # extra CLI flags
    # CLI flags a revision must have to render this view at all ("n/a"
    # otherwise), and flags it may lack and render without (marked).
    needs: tuple = ()
    optional: tuple = ()
    # witness only: camera x, y are world metres, not fractions of the box,
    # on a surface that repeats every fold_m — folded into the field's box.
    fold_m: Optional[float] = None


def _witness(name, label, cam, flags=(), needs=(), optional=()):
    return View(name, label, "witness", cam, tuple(flags), tuple(needs),
                tuple(optional))


# The desert tile folds at 4096 m under a zero surface offset, so any
# world xy is the same ground 4096 m on — which lets desert-ground's proof
# cameras (flown over the 51 km desert-coarse demo) be folded into the 5.1 km
# subvolume's box, over the same stones. z stays absolute: the ground's own
# height plus the AGL they were placed at.
DESERT_FOLD_M = 4096.0
DESERT_SUN = (0.0, 45.5)            # the desert-coarse demo's sun


def _desert(name, label, x, y, z, az, el, fov, sun=DESERT_SUN):
    return View(name, label, "witness", (x, y, z, az, el, fov, *sun),
                ("--surface", "desert"), needs=("--surface",),
                fold_m=DESERT_FOLD_M)


def _city(name, label, cam):
    return View(name, label, "harness", cam,
                ("--city-offset", *map(str, CITY_OFFSET),
                 "--moon", *map(str, CITY_MOON)))


SCENES = {
    "ocean": [
        _witness("glitter_low_sun", "glitter toward a low sun, 30 m",
                 (0.30, 0.10, 30.0, 250.0, -6.0, 70.0, 250.0, 10.0),
                 ["--quality", "high"]),
        _witness("whitecaps_mid", "whitecaps, 120 m, looking down",
                 (0.30, 0.10, 120.0, 100.0, -35.0, 70.0, 20.0, 55.0),
                 ["--quality", "high"]),
        _witness("aerial", "aerial, 2.5 km",
                 (0.30, 0.10, 2500.0, 20.0, -40.0, 70.0, 20.0, 50.0),
                 ["--quality", "high"]),
        _witness("motion_t0", "close sea, scene clock 0 s",
                 (0.30, 0.10, 12.0, 100.0, -30.0, 70.0, 20.0, 45.0),
                 ["--quality", "high", "--time", "0"], optional=["--time"]),
        _witness("motion_t2", "close sea, scene clock 2 s",
                 (0.30, 0.10, 12.0, 100.0, -30.0, 70.0, 20.0, 45.0),
                 ["--quality", "high", "--time", "2"], optional=["--time"]),
    ],
    "city": [
        _city("skyline", "skyline from 3.5 km, 250 m up",
              (16515.0, 55315.0, 250.0, 0.0, 4.0, 70.0)),
        _city("landing", "the cyberpunk landing view",
              (LANDING[0], LANDING[1], 309.0, 168.6, -1.2, 100.0)),
        _city("aerial_1500", "aerial, 1.5 km",
              (16515.0, 56815.0, 1500.0, 0.0, -35.0, 70.0)),
        _city("street_canyon", "street-level canyon, 20 m",
              (STREET_X, 61200.0, 20.0, 0.0, 12.0, 85.0)),
        _city("industrial", "industrial belt, 400 m",
              (21654.0, 46980.0, 400.0, 0.0, -20.0, 70.0)),
        _city("facade", "facade close-up, from the street at 25 m",
              (STREET_X + 6.0, 61500.0, 25.0, 285.0, 15.0, 55.0)),
    ],
    # desert-ground's proof cameras (2026-09-24): one spot on a wash, the
    # ground there at 2.572 m, seen from 300, 30 and 5 m and under a low
    # sun; and a pavement close-up elsewhere.
    "desert": [
        _desert("alt300", "300 m", 7465.14, 34082.99, 302.572, 250, -20, 80),
        _desert("alt30", "30 m", 7465.14, 34082.99, 32.572, 250, -10, 80),
        _desert("alt5", "5 m", 7465.14, 34082.99, 7.572, 250, -4, 80),
        _desert("pavement", "desert pavement close-up, 1.5 m",
                8421.04, 32708.78, 5.876, 270, -35, 70),
        # desert-ground's low sun came from 240; over this field that puts
        # the spot in a cloud's shadow, so the sun here is at 60 (same 18).
        _desert("low_sun", "low sun (60/18), 12 m",
                7465.14, 34082.99, 14.572, 330, -3, 80, sun=(60.0, 18.0)),
    ],
}
# The desert's two columns: the same view, full ground and LITE ground.
DESERT_VARIANTS = (("full (high)", ["--quality", "high"]),
                   ("LITE (low)", ["--quality", "low"]))


# --- trees -----------------------------------------------------------------

def resolve(ref: str) -> str:
    """A ref -> its full commit sha, or "tree" for the working tree."""
    if ref == "tree":
        return "tree"
    out = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"contact_sheet: no such revision {ref!r}.")
    return out.stdout.strip()


def tree_for(sha: str) -> Path:
    """The revision's files on disk (exported once, then reused)."""
    if sha == "tree":
        return REPO
    root = OUT / "trees" / sha[:12]
    if (root / ".complete").exists():
        return root
    if root.exists():
        shutil.rmtree(root)              # a half-written export, never a link
    root.mkdir(parents=True)
    print(f"contact_sheet: exporting {sha[:12]} to {root}", flush=True)
    archive = subprocess.run(["git", "archive", "--format=tar", sha],
                             cwd=REPO, capture_output=True, check=True).stdout
    tar_path = root.parent / f"{sha[:12]}.tar"
    tar_path.write_bytes(archive)
    with tarfile.open(tar_path) as tar:
        tar.extractall(root, filter="data")
    tar_path.unlink()
    (root / ".complete").write_text(sha + "\n")
    return root


def has_flag(tree: Path, script: str, flag: str) -> bool:
    """Whether the revision's CLI declares `flag`."""
    return f'"{flag}"' in (tree / script).read_text()


# --- one render ------------------------------------------------------------

_WITNESS = textwrap.dedent("""
    import sys, json
    tree, field, out, cam, flags, fold = json.loads(sys.argv[1])
    sys.path.insert(0, tree)
    import cloudyview
    if not cloudyview.__file__.startswith(tree):
        raise SystemExit(f"imported {cloudyview.__file__}, not the tree {tree}")
    from cloudyview.cloudfield import load
    from cloudyview.witness import _field_level, cli
    level = _field_level(load(field), "single", verbose=False)
    bmin, bmax = level.bmin, level.bmax
    fx, fy, z, az, el, fov, saz, sel = cam
    if fold is not None:
        # The same ground, fold metres on at a time, inside the box.
        fx, fy = (float(lo) + (v - float(lo)) %% fold for v, lo in
                  zip((fx, fy), bmin[:2]))
        if any(v > float(hi) for v, hi in zip((fx, fy), bmax[:2])):
            raise SystemExit(f"the field's box is narrower than the "
                             f"surface's {fold} m fold")
        fx = (fx - float(bmin[0])) / (float(bmax[0]) - float(bmin[0]))
        fy = (fy - float(bmin[1])) / (float(bmax[1]) - float(bmin[1]))
    rel = [2.0 * fx - 1.0, 2.0 * fy - 1.0, 2.0 * z / float(bmax[2]) - 1.0]
    sys.argv = ["witness", field, "--output", out,
                "--camera-position", *map(str, rel),
                "--camera-azimuth", str(az), "--camera-elevation", str(el),
                "--fov", str(fov), "--sun-azimuth", str(saz),
                "--sun-elevation", str(sel), "--periodic",
                "--size", *map(str, %s), *flags]
    cli()
""") % (list(SIZE),)

_HARNESS = textwrap.dedent("""
    import sys, json, runpy
    tree, field, out, cam, flags = json.loads(sys.argv[1])
    sys.path.insert(0, tree)
    import cloudyview
    if not cloudyview.__file__.startswith(tree):
        raise SystemExit(f"imported {cloudyview.__file__}, not the tree {tree}")
    sys.argv = ["night_city_harness", "--field", field, "--outdir", out,
                "--size", *map(str, %s), "--city-camera", *map(str, cam),
                *flags]
    runpy.run_path(tree + "/tools/night_city_harness.py", run_name="__main__")
""") % (list(SIZE),)


def render(tree: Path, view: View, flags: list, dest: Path) -> None:
    """Render one view with the revision's own code into dest (PNG)."""
    work = dest.parent / (dest.stem + ".work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    script = _WITNESS if view.kind == "witness" else _HARNESS
    payload = json.dumps([str(tree), str(FIELD), str(work),
                          list(view.camera), flags]
                         + ([view.fold_m] if view.kind == "witness" else []))
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", script, payload],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{view.name} at {tree.name} failed:\n"
                           f"{proc.stderr[-3000:]}")
    made = sorted(work.glob("*.png"))
    if len(made) != 1:
        raise RuntimeError(f"{view.name} at {tree.name}: expected one PNG in "
                           f"{work}, found {[p.name for p in made]}.")
    made[0].replace(dest)
    shutil.rmtree(work)
    print(f"  {dest.relative_to(OUT)}  ({time.perf_counter() - t0:.1f} s)",
          flush=True)


def cell(sha: str, tree: Path, scene: str, view: View, extra=(),
         tag: str = "") -> tuple[Optional[Path], str]:
    """(PNG or None, note) for one view at one revision."""
    script = ("cloudyview/witness.py" if view.kind == "witness"
              else "tools/night_city_harness.py")
    if not (tree / script).exists():
        return None, "n/a (no such renderer)"
    flags = list(view.flags) + list(extra)
    for flag in view.needs:
        if not has_flag(tree, script, flag):
            return None, f"n/a (no {flag})"
    note = ""
    for flag in view.optional:
        if flag in flags and not has_flag(tree, script, flag):
            i = flags.index(flag)
            del flags[i:i + 2]
            note = f"(no {flag}: rendered without it)"
    key = sha[:12] if sha != "tree" else "tree"
    dest = OUT / "renders" / key / scene / f"{view.name}{tag}.png"
    if sha == "tree" or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        render(tree, view, flags, dest)
    return dest, note


# --- the sheet -------------------------------------------------------------

def compose(title: str, columns: list, rows: list, out: Path) -> None:
    """rows: [(label, [(png or None, note), ...]), ...] under `columns`."""
    from PIL import Image, ImageDraw
    cw, ch = CELL
    pad, head, left = 10, 46, 230
    ink, muted, paper = (31, 31, 30), (107, 106, 99), (250, 250, 247)
    sheet = Image.new("RGB", (left + len(columns) * (cw + pad) + pad,
                              head + len(rows) * (ch + pad) + pad), paper)
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), title, fill=ink)
    for j, c in enumerate(columns):
        d.text((left + pad + j * (cw + pad), 28), c, fill=ink)
    for i, (label, cells) in enumerate(rows):
        y = head + pad + i * (ch + pad)
        d.multiline_text((pad, y + ch // 2 - 10), textwrap.fill(label, 30),
                         fill=ink)
        for j, (png, note) in enumerate(cells):
            x = left + pad + j * (cw + pad)
            if png is None:
                d.rectangle([x, y, x + cw, y + ch], outline=muted)
                d.text((x + 12, y + ch // 2 - 6), note, fill=muted)
                continue
            sheet.paste(Image.open(png).convert("RGB").resize((cw, ch),
                                                             Image.LANCZOS),
                        (x, y))
            if note:
                d.rectangle([x, y + ch - 20, x + cw, y + ch], fill=paper)
                d.text((x + 6, y + ch - 16), note, fill=muted)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"contact_sheet: wrote {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", default=DEFAULT_BEFORE)
    ap.add_argument("--after", default="HEAD",
                    help="a ref, or 'tree' for the working tree (uncached)")
    ap.add_argument("--scenes", default="ocean,city,desert")
    ap.add_argument("--list", action="store_true",
                    help="print the views and exit")
    args = ap.parse_args(argv)
    scenes = args.scenes.split(",")
    unknown = [s for s in scenes if s not in SCENES]
    if unknown:
        raise SystemExit(f"contact_sheet: unknown scene(s) {unknown}; "
                         f"known: {sorted(SCENES)}")
    if args.list:
        for s in scenes:
            for v in SCENES[s]:
                print(f"{s:7s} {v.name:18s} {v.label}  {v.kind} {v.camera}")
        return 0
    if not FIELD.exists():
        raise SystemExit(f"contact_sheet: the field is missing: {FIELD}")
    before, after = resolve(args.before), resolve(args.after)
    trees = {before: tree_for(before), after: tree_for(after)}
    short = {sha: sha[:7] if sha != "tree" else "working tree"
             for sha in trees}
    for scene in scenes:
        views = SCENES[scene]
        if not views:
            print(f"contact_sheet: no {scene} views defined; skipped.")
            continue
        print(f"contact_sheet: {scene}", flush=True)
        if scene == "desert":
            columns = [f"{name} at {short[after]}"
                       for name, _ in DESERT_VARIANTS]
            rows = [(v.label, [cell(after, trees[after], scene, v, extra,
                                    tag=f"_{name.split()[0].lower()}")
                               for name, extra in DESERT_VARIANTS])
                    for v in views]
        else:
            columns = [f"before: {short[before]}", f"after: {short[after]}"]
            rows = [(v.label, [cell(sha, trees[sha], scene, v)
                               for sha in (before, after)])
                    for v in views]
        tag = short[after].replace(" ", "_")
        compose(f"soar {scene}: {short[before]} -> {short[after]}"
                if scene != "desert" else f"soar desert at {short[after]}",
                columns, rows, OUT / f"{scene}_{short[before]}_{tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
