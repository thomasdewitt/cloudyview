#!/usr/bin/env python
"""Before/after contact sheets of soar's scenes: one labelled PNG per scene.

    tools/gpulock uv run python tools/contact_sheet.py
    tools/gpulock uv run python tools/contact_sheet.py --after 1a2b3c4 --scenes city
    tools/gpulock uv run python tools/contact_sheet.py --list

Every view is rendered twice — at BEFORE (default: the start of the
2026-09-23/24 realism push, 3fc2eea) and at AFTER (default: HEAD) — by each
revision's OWN code: the ref is exported with `git archive` into
temp/contact/trees/<sha>/ and a subprocess puts that tree first on sys.path
(and checks it did) before calling the tree's own renderer. That is the
point of doing it this way rather than swapping shaders under one host
(tools/soar_shader_ab.py): across a day of work the hosts, bindings and
tiles moved too, and only a revision's own code renders what that revision
looked like.

The clouds are the app's own prebaked demos (web/soar/demos), loaded by each
tree's tools/benchmarking/soar_frame_bench.build_level_from_demo, and
rendered periodic as the app flies them. Only the small periodic demos are
used: a field that is not periodic (the TWP-ICE subvolume) shows its wrap
seams as hard cloud edges.

  * ocean — the tree's witness.render_nested over marine-congestus-coarse:
    glitter toward a low sun, a look-down at 120 m, an aerial, a moderate
    look-down into the sun's glint, and a close view.
  * city — the tree's tools/night_city_harness.py (its NetCDF loader swapped
    for the demo loader) in cyberpunk mode's frame: its tile offset and moon
    (web/soar/constants.js), cameras in city-tile metres so they name the
    same streets whatever field stands over the tile.
  * desert — render_nested over desert-coarse at desert-ground's proof
    cameras, the full ground (quality high) beside the LITE ground (quality
    low, as the app flies its low tiers). A tree without the desert: "n/a".

Cameras are world metres (the city's: city-tile metres). Renders are
deterministic (fixed exposure, frame index from 0) and cached per commit and
per view definition under temp/contact/renders/, so a re-run renders only
what changed; `--after tree` renders the working tree, uncached. Outputs go
to temp/contact/ on disk, never /tmp. Run it under tools/gpulock: it holds
the GPU for the whole sheet.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
DEMOS = REPO / "web" / "soar" / "demos"
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
    kind: str                # "nested" (render_nested) or "harness"
    demo: str                # web/soar/demos/<demo>
    # nested: (x_m, y_m, z_m, azimuth, elevation, fov, sun_az, sun_el)
    # harness: (city_x_m, city_y_m, z_m, azimuth, elevation, fov)
    camera: tuple
    # nested: render_nested keyword arguments, as pairs; harness: CLI flags.
    args: tuple = ()
    # render_nested arguments a tree must take to render the view at all
    # ("n/a" otherwise).
    needs: tuple = ()


OCEAN_DEMO = "marine-congestus-coarse"
DESERT_DEMO = "desert-coarse"
CITY_DEMO = "marine-congestus-coarse"
DESERT_SUN = (0.0, 45.5)     # desert-coarse's own sun
# The clearest 3 km of marine-congestus-coarse (1.5 % of its columns past
# optical depth 0.3; the field as a whole is half cloud): the ocean views
# stand here so the sea is in sun, not under a deck.
OCEAN_CLEAR = (17105.0, 1405.0)


def _ocean(name, label, cam):
    return View(name, label, "nested", OCEAN_DEMO, cam, (("quality", "high"),))


def _desert(name, label, x, y, z, az, el, fov, sun=DESERT_SUN):
    return View(name, label, "nested", DESERT_DEMO,
                (x, y, z, az, el, fov, *sun),
                (("surface", "desert"),), needs=("surface",))


def _city(name, label, cam):
    return View(name, label, "harness", CITY_DEMO, cam,
                ("--city-offset", *map(str, CITY_OFFSET),
                 "--moon", *map(str, CITY_MOON)))


SCENES = {
    "ocean": [
        _ocean("glitter_low_sun", "glitter toward a low sun, 30 m",
               (*OCEAN_CLEAR, 30.0, 250.0, -6.0, 70.0, 250.0, 10.0)),
        _ocean("lookdown_mid", "looking down from 120 m",
               (*OCEAN_CLEAR, 120.0, 100.0, -35.0, 70.0, 20.0, 55.0)),
        # Above the congestus tops (3.9 km), 3.5 km back along its azimuth
        # (20 degrees) so that 55 degrees down it sees the clear patch.
        _ocean("aerial", "aerial, 5 km, above the tops",
               (OCEAN_CLEAR[0] - 1197.0, OCEAN_CLEAR[1] - 3289.0, 5000.0,
                20.0, -55.0, 70.0, 20.0, 50.0)),
        # Thomas's framing of 2026-09-25: a moderate look-down into the sun's
        # glint, the sun's reflection inside the frame.
        _ocean("glint_lookdown", "into the sun's glint, 300 m, 35 deg down",
               (*OCEAN_CLEAR, 300.0, 20.0, -35.0, 70.0, 20.0, 40.0)),
        _ocean("close_sea", "close sea, 12 m",
               (*OCEAN_CLEAR, 12.0, 100.0, -30.0, 70.0, 20.0, 45.0)),
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
    # desert-ground's proof cameras (2026-09-24), over their field: one spot
    # on a wash, the ground there at 2.572 m, seen from 300, 30 and 5 m and
    # under a low sun; and a pavement close-up elsewhere.
    "desert": [
        _desert("alt300", "300 m", 7465.14, 34082.99, 302.572, 250, -20, 80),
        _desert("alt30", "30 m", 7465.14, 34082.99, 32.572, 250, -10, 80),
        _desert("alt5", "5 m", 7465.14, 34082.99, 7.572, 250, -4, 80),
        _desert("pavement", "desert pavement close-up, 1.5 m",
                8421.04, 32708.78, 5.876, 270, -35, 70),
        _desert("low_sun", "low sun (240/18), 12 m",
                7465.14, 34082.99, 14.572, 330, -3, 80, sun=(240.0, 18.0)),
    ],
}
# The desert's two columns: the same view, full ground and LITE ground.
DESERT_VARIANTS = (("full (high)", "high"), ("LITE (low)", "low"))


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
    if root.is_symlink():
        raise SystemExit(f"contact_sheet: {root} is a symlink; not touching it.")
    if root.exists():
        shutil.rmtree(root)              # a half-written export
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


# --- one render ------------------------------------------------------------

# Shared by both workers: the tree first on sys.path, and its own demo loader.
_PRELUDE = textwrap.dedent("""
    import sys, json, importlib.util
    job = json.loads(sys.argv[1])
    tree = job["tree"]
    sys.path.insert(0, tree)
    import cloudyview
    if not cloudyview.__file__.startswith(tree):
        raise SystemExit(f"imported {cloudyview.__file__}, not the tree {tree}")
    spec = importlib.util.spec_from_file_location(
        "soar_frame_bench", tree + "/tools/benchmarking/soar_frame_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
""")

_NESTED = _PRELUDE + textwrap.dedent("""
    import inspect
    import numpy as np
    from PIL import Image
    from cloudyview.witness import render_nested
    takes = inspect.signature(render_nested).parameters
    kwargs = dict(job["args"])
    for name in job["needs"]:
        if name not in takes:
            print(f"NA no {name}")
            sys.exit(0)
    level, _ = bench.build_level_from_demo(job["demo"])
    x, y, z, az, el, fov, saz, sel = job["camera"]
    img = render_nested([level], [x, y, z], azimuth=az, elevation=el,
                        sun_azimuth=saz, sun_elevation=sel,
                        image_size=tuple(job["size"]), fov_degrees=fov,
                        periodic=True, verbose=False, **kwargs)
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5)
                    .astype(np.uint8)).save(job["out"])
""")

# The harness reads a NetCDF field; here its build_level is the demo
# loader, and everything else (camera frames, uniforms, the look) is the
# tree's own harness.
_HARNESS = _PRELUDE + textwrap.dedent("""
    from pathlib import Path
    path = tree + "/tools/night_city_harness.py"
    g = {"__name__": "night_city_harness", "__file__": path}
    exec(compile(Path(path).read_text(), path, "exec"), g)
    g["build_level"] = lambda p, verbose=True: bench.build_level_from_demo(p)[0]
    out = Path(job["out"])
    work = out.parent / (out.stem + ".work")
    work.mkdir(parents=True, exist_ok=True)
    code = g["main"](["--field", job["demo"], "--outdir", str(work),
                      "--size", *map(str, job["size"]),
                      "--city-camera", *map(str, job["camera"]),
                      *job["args"]])
    made = sorted(work.glob("*.png"))
    if code or len(made) != 1:
        raise SystemExit(f"harness exit {code}, made {[p.name for p in made]}")
    made[0].replace(out)
    work.rmdir()
""")


def render(tree: Path, view: View, args: tuple, dest: Path) -> str:
    """Render one view with the revision's own code into dest (PNG).
    Returns the note the sheet shows under it ("" for none)."""
    job = {"tree": str(tree), "demo": str(DEMOS / view.demo),
           "out": str(dest), "camera": list(view.camera), "size": list(SIZE),
           "args": [list(a) for a in args] if view.kind == "nested"
           else list(args),
           "needs": list(view.needs)}
    script = _NESTED if view.kind == "nested" else _HARNESS
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(job)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{view.name} at {tree.name} failed:\n"
                           f"{proc.stderr[-3000:]}")
    lines = proc.stdout.splitlines()
    na = [ln[3:] for ln in lines if ln.startswith("NA ")]
    if na:
        return f"n/a ({na[0]})"
    if not dest.exists():
        raise RuntimeError(f"{view.name} at {tree.name} wrote no image.")
    print(f"  {dest.relative_to(OUT)}  ({time.perf_counter() - t0:.1f} s)",
          flush=True)
    return ""


def cell(sha: str, tree: Path, scene: str, view: View, args=None,
         tag: str = "") -> tuple[Optional[Path], str]:
    """(PNG or None, note) for one view at one revision — from the cache
    when this revision already rendered this exact view definition."""
    args = view.args if args is None else args
    key = sha[:12] if sha != "tree" else "tree"
    digest = hashlib.sha256(repr((view, args)).encode()).hexdigest()[:10]
    dest = OUT / "renders" / key / scene / f"{view.name}{tag}_{digest}.png"
    note_file = dest.with_suffix(".note")
    if sha != "tree" and note_file.exists():
        note = note_file.read_text()
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        note = render(tree, view, args, dest)
        note_file.write_text(note)
    return (None if note.startswith("n/a") else dest), note


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
                print(f"{s:7s} {v.name:16s} {v.label:42s} {v.demo} {v.camera}")
        return 0
    for s in scenes:
        for v in SCENES[s]:
            if not (DEMOS / v.demo / "meta.json").exists():
                raise SystemExit(f"contact_sheet: demo {v.demo!r} is not in "
                                 f"{DEMOS}.")
    before, after = resolve(args.before), resolve(args.after)
    trees = {before: tree_for(before), after: tree_for(after)}
    short = {sha: sha[:7] if sha != "tree" else "working tree"
             for sha in trees}
    for scene in scenes:
        views = SCENES[scene]
        print(f"contact_sheet: {scene}", flush=True)
        if scene == "desert":
            columns = [f"{name} at {short[after]}"
                       for name, _ in DESERT_VARIANTS]
            rows = [(v.label, [cell(after, trees[after], scene, v,
                                    v.args + (("quality", q),), tag=f"_{q}")
                               for _, q in DESERT_VARIANTS])
                    for v in views]
            title = f"soar desert at {short[after]}"
        else:
            columns = [f"before: {short[before]}", f"after: {short[after]}"]
            rows = [(v.label, [cell(sha, trees[sha], scene, v)
                               for sha in (before, after)])
                    for v in views]
            title = f"soar {scene}: {short[before]} -> {short[after]}"
        tag = short[after].replace(" ", "_")
        compose(title, columns, rows,
                OUT / f"{scene}_{short[before]}_{tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
