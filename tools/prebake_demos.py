"""Bake the shippable demo set: a GPU-ready volume and a still per demo.

Two products per demo, both written to web/soar/demos/<id>/:

  volume.bin.gz + map.bin + meta.json
      What the viewer uploads: the (nx, ny, nz) field with no border of
      any kind, one texel per cell, gzipped — the texture is r16float
      either way, so the bytes on the wire are exactly the bytes that reach
      the GPU, and deflate is HDF5/browser builtin territory rather than a
      bespoke codec.

  still.webp
      One converged ground-level frame for the landing page, rendered with
      witness — which is now the same shader the browser runs, so the preview
      is the thing itself rather than an impression of it.

    uv run python tools/prebake_demos.py            # everything
    uv run python tools/prebake_demos.py --only rce --skip-volume
    uv run python tools/prebake_demos.py --index-only   # regroup, no baking
"""

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cloudyview as cv
from cloudyview import optical_depth
from cloudyview.cloudfield import CloudField
from cloudyview.glimpse import glimpse
from cloudyview.look import DEFAULT_HAZE
from cloudyview.soar_host import (
    DEFAULT_CONTRAST, DEFAULT_EXPOSURE, DEFAULT_HAZE_HEIGHT_DEPENDENT,
    DEFAULT_LOD_STRENGTH, DEFAULT_TONE_MAP_GAMMA, DEFAULT_TONE_MAP_WHITE_POINT,
)
from cloudyview.witness import crop_empty_z
from export_web_assets import _volume_aabb

# The tone map the app flies with, imported rather than restated: a still that
# is meant to be the frame you land on cannot hold a private copy of the look,
# or moving the app's default silently makes every preview a lie. A demo's
# `still` block therefore names ONLY what it wants to differ — in practice
# just exposure and haze — and everything else follows the app.
#
# Haze is the exception that has to be stated per demo: in the app it is a
# function of the quality tier, so there is no single value the preview could
# inherit, and the right one is whichever Thomas chose the view under.
LOOK_DEFAULTS = {
    "exposure": DEFAULT_EXPOSURE,
    "gamma": DEFAULT_TONE_MAP_GAMMA,
    "white_point": DEFAULT_TONE_MAP_WHITE_POINT,
    "contrast": DEFAULT_CONTRAST,
    "haze": DEFAULT_HAZE,
    # Angular LOD, recorded like the rest: a still rendered at one strength
    # and a flight opened at another are the same camera marched two ways,
    # and the `look` block is the record of how the picture was made.
    "lod": DEFAULT_LOD_STRENGTH,
    # Same, for the haze profile. A demo whose framing was chosen under the
    # exponential atmosphere says so here rather than inheriting whatever the
    # app defaults to this month.
    "haze_height_dependent": DEFAULT_HAZE_HEIGHT_DEPENDENT,
}
_LOOK_ARG = {"exposure": "exposure", "gamma": "tone_map_gamma",
             "white_point": "tone_map_white_point", "contrast": "contrast",
             "haze": "haze", "lod": "lod",
             "haze_height_dependent": "haze_height_dependent"}

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "demos"
OUT = REPO / "web" / "soar" / "demos"

# The STEAM fields are read out of the repo that produces them rather than
# copied here: they are model output, turbulon-model owns them, and a copy
# under data/demos would be a second thing to keep in step with
# generate_demo_fields.py. A spec names its own source with `src`.
#
# Sibling layout, which is the convention on both of Thomas's machines
# (~/code-and-data/<repo>). If the path is wrong the netCDF open fails and
# says so — there is nothing here to fall back to.
STEAM_SRC = REPO.parent / "turbulon-model" / "demos" / "fields"

# The rail on the landing page is grouped, and the grouping lives here: each
# spec names a group, and every group is declared below whether or not it has
# anything in it yet. A group with no demos ships as "coming-soon", which the
# viewer renders as a placeholder rather than an empty row.
# Order here is the order on the page. STEAM leads: it is the model this work
# is for, and the LES cases are the comparison rather than the headline —
# which means the first thing on the landing page is currently a "coming soon"
# strip, deliberately (Thomas, 2026-08-14).
GROUPS = [
    # STEAM is the cascade/turbulon model. Its fields carry nested refinements
    # (r0/r1/r2 in one file), so a demo here will eventually need a per-level
    # payload — that hangs off the demo object as a `levels` list, alongside
    # `base`/`bytes`/`still`, and nothing in the index schema or in the
    # viewer's group rendering assumes a demo is a single volume.
    dict(id="steam", title="STEAM cascade simulations"),
    dict(id="hydrodynamics", title="Hydrodynamic large-eddy simulations"),
]

# Crops come from measuring where the cloud actually is. Levels are trimmed
# to the occupied range plus headroom; empty sky costs the same per voxel as
# a cumulus tower and renders to nothing.
#
# `still` is the framing AND the look of the preview, and since 2026-08-14 it
# is also where the flight starts: the viewer reads the camera back out of
# meta.json and opens there, so clicking a case puts you exactly where the
# picture you clicked was taken from. Anything omitted falls back to the
# defaults in render_still.
#
# These blocks come from Thomas flying the app and copying the reproduction
# command out of the terminal-render panel, which is why the numbers have
# twelve decimal places — they are a camera he stopped at, not a camera
# anybody typed. `sun` moved with them for the same reason: it is the light
# he chose the view under, and the still and the flight have to agree about
# it or entering flight is a jump cut.
#
# Two optional keys, both prose, both shown only when the landing page's card
# is opened by a hover:
#
#   description   What this case IS, when the title and the regime line do not
#                 already say it. The LES cases had one each and they have
#                 been dropped: "the nocturnal stratocumulus deck off the
#                 California coast" told a reader nothing that "SAM DYCOMS /
#                 Marine stratocumulus" and the still behind the page had not
#                 already (Thomas, 2026-08-15). Write one when a case would
#                 otherwise be a mystery — a STEAM cascade with a particular
#                 set of parameters is the likely first customer.
#
#   warning       What will go wrong before it goes wrong. Reserved for a real
#                 cost to the person clicking; it is set in red on the card and
#                 spending that on "this one is quite pretty" would make the
#                 next one unreadable.
DEMOS = [
    # --- STEAM -------------------------------------------------------------
    # Read straight out of turbulon-model rather than copied into data/demos:
    # see STEAM_SRC. Both are periodic — they are cascade fields on their own
    # root domain, not crops out of a larger run, so the wrap is the field's
    # own (Thomas, 2026-08-15) and `periodic` stays at its default.
    #
    # z="auto" rather than a typed range: the browser and witness both trim
    # empty planes before rendering, so the domain box a camera is normalized
    # against IS the occupied band. See occupied_z_band.
    dict(
        id="desert",
        group="steam",
        title="STEAM high-based convection",
        field="Utah summer cumulus congestus",
        warning="Needs a highly capable GPU",
        liquid=("demo_desert-convection.nc", "qc"),
        ice=("demo_desert-convection.nc", "qi"),
        src=STEAM_SRC,
        dims="xyz",
        crop=dict(y=(0, 2048), x=(0, 2048), z="auto"),
        scale=1e3,                       # kg/kg -> g/kg
        sun=dict(azimuth=65.0, elevation=45.5),
        still=dict(
            size=(1920, 963),
            position=(0.0776818342067, -0.27879259066, -0.998326359833),
            azimuth=44.28, elevation=25.76, fov=100.0,
            exposure=1.17125651993, haze=0.55,
        ),
    ),
    # The same world as the case above — same seed, same outer scale, same
    # 51.2 km box — carried two cascade classes less far, so its cells are
    # 100 m rather than 25 m. That pairing is the point of shipping both:
    # it is one field at two resolutions rather than two fields.
    dict(
        id="desert-coarse",
        group="steam",
        title="STEAM high-based convection (coarse)",
        field="Utah cumulus on a coarser grid",
        liquid=("demo_desert-convection-coarse.nc", "qc"),
        ice=("demo_desert-convection-coarse.nc", "qi"),
        src=STEAM_SRC,
        dims="xyz",
        crop=dict(y=(0, 512), x=(0, 512), z="auto"),
        scale=1e3,
        sun=dict(azimuth=0.0, elevation=45.5),
        still=dict(
            size=(1920, 963),
            position=(0.917919995217, -0.696861762887, -0.673006416282),
            azimuth=358.68, elevation=4.16, fov=100.0,
            exposure=2.98704821995, haze=0.92,
        ),
    ),
    dict(
        id="congestus",
        group="steam",
        title="STEAM marine cumulus congestus",
        field="Maritime cumulus congestus",
        warning="Needs a highly capable GPU",
        liquid=("demo_marine-congestus.nc", "qc"),
        ice=("demo_marine-congestus.nc", "qi"),
        src=STEAM_SRC,
        dims="xyz",
        crop=dict(y=(0, 2048), x=(0, 2048), z="auto"),
        scale=1e3,
        sun=dict(azimuth=100.0, elevation=51.5),
        still=dict(
            size=(1920, 963),
            position=(0.516654156817, -0.285165086182, -0.993548387097),
            azimuth=249.48, elevation=45.92, fov=100.0,
            exposure=3.86833683967, haze=1.0,
        ),
    ),
    dict(
        id="stratified",
        group="steam",
        title="STEAM stratified cirrus",
        field="Thin upper-atmosphere ice clouds",
        warning="Needs a highly capable GPU",
        liquid=("demo_stratified.nc", "qc"),
        ice=("demo_stratified.nc", "qi"),
        src=STEAM_SRC,
        dims="xyz",
        crop=dict(y=(0, 1024), x=(0, 1024), z="auto"),
        scale=1e3,
        sun=dict(azimuth=277.0, elevation=4.5),
        still=dict(
            size=(1920, 963),
            position=(-0.488904823109, 0.653836399367, -0.998325965182),
            azimuth=278.16, elevation=21.8, fov=100.0,
            exposure=3.55906028557, haze=0.0,
        ),
    ),
    # --- hydrodynamic LES --------------------------------------------------
    dict(
        id="twpice",
        group="hydrodynamics",
        title="SAM TWP-ICE",
        field="Tropical deep convection",
        # No warning here despite being the largest of the LES set: it is
        # 432 MB of texture and flies fine in practice. The warning moved to
        # the STEAM cases, which are the ones that actually hurt — 1.2 to
        # 4.2 GB (Thomas, 2026-08-15: "TWPICE is not GPU-difficult but all
        # the steam runs are").
        liquid=("TWPICE_LPT_3D_QC_0000003450.nc", "QC"),
        ice=("TWPICE_LPT_3D_QI_0000003450.nc", "QI"),
        dims="yxz",
        crop=dict(y=(0, 1024), x=(1024, 2048), z=(0, 206)),
        scale=1.0,                       # already g/kg
        sun=dict(azimuth=235.0, elevation=70.0),   # zenith ~20 (Thomas, 2026-08-11)
        # The one case that does not tile. It is a 102 km crop out of a larger
        # run rather than a periodic box, and wrapping it repeats a squall
        # line that has no business repeating (Thomas, 2026-08-14).
        periodic=False,
        still=dict(
            size=(1920, 963),
            position=(-0.169577212547, -0.394411429582, -0.875680961028),
            azimuth=50.16, elevation=23.0, fov=100.0,
            exposure=3.87150512286, haze=0.36,
        ),
    ),
    dict(
        id="dycoms",
        group="hydrodynamics",
        title="SAM DYCOMS",
        field="Marine stratocumulus",
        liquid=("DYCOMS_RF01_640x640x640_dt0.25sec_320_0000043200_W_QN.nc", "QN"),
        ice=None,
        dims="zyx",
        crop=dict(y=(0, 640), x=(0, 640), z=(215, 355)),
        scale=1.0,                       # units attribute says g/kg
        sun=dict(azimuth=211.0, elevation=30.5),
        still=dict(
            size=(1920, 963),
            position=(0.0, 0.0, -0.971830985915),
            azimuth=231.84, elevation=11.84, fov=100.0,
            exposure=2.83618562719, haze=1.43,
        ),
    ),
    dict(
        id="rce",
        group="hydrodynamics",
        title="CM1 RCE",
        field="Radiative–convective equilibrium",
        liquid=("CM1_RCE_small_les300_3D_allvars_hour1200.nc", "clw"),
        ice=("CM1_RCE_small_les300_3D_allvars_hour1200.nc", "cli"),
        dims="zyx",
        crop=dict(y=(0, 540), x=(0, 540), z=(0, 88)),
        scale=1e3,                       # g/g -> g/kg
        sun=dict(azimuth=249.0, elevation=34.5),
        still=dict(
            size=(1920, 963),
            position=(-0.00633639238438, 0.0177418162348, -0.947201859468),
            azimuth=256.08, elevation=18.8, fov=100.0,
            exposure=1.66376086725, haze=1.3,
        ),
    ),
    # The FIF cascade was shipped here until 2026-08-08 and was dropped: it is
    # not a fluid-dynamics simulation, and the synthetic slot on the rail now
    # belongs to STEAM. Its source field stays in data/demos/, so restoring it
    # is a spec here plus a bake.
]

# --- loading ---------------------------------------------------------------

def source_dir(spec: dict) -> Path:
    return spec.get("src", SRC)


def _read(path: Path, var: str, dims: str, crop: dict) -> np.ndarray:
    """Read one variable, cropped, standardized to (x, y, z)."""
    ys, xs, zs = crop["y"], crop["x"], crop["z"]
    with Dataset(path) as ds:
        v = ds.variables[var]
        if dims == "yxz":
            a = v[0, ys[0]:ys[1], xs[0]:xs[1], zs[0]:zs[1]]
            a = np.moveaxis(np.asarray(a, np.float32), 0, 1)      # -> (x, y, z)
        elif dims == "zyx":
            a = v[0, zs[0]:zs[1], ys[0]:ys[1], xs[0]:xs[1]]
            a = np.moveaxis(np.asarray(a, np.float32), 0, 2)      # -> (y, x, z)
            a = np.moveaxis(a, 0, 1)                              # -> (x, y, z)
        elif dims == "xyz":
            a = np.asarray(v[xs[0]:xs[1], ys[0]:ys[1], zs[0]:zs[1]], np.float32)
        else:
            raise ValueError(f"unknown dim order {dims!r}")
    return np.ascontiguousarray(a)


def _coords(path: Path, crop: dict):
    with Dataset(path) as ds:
        x = np.asarray(ds.variables["x"][crop["x"][0]:crop["x"][1]], np.float64)
        y = np.asarray(ds.variables["y"][crop["y"][0]:crop["y"][1]], np.float64)
        z = np.asarray(ds.variables["z"][crop["z"][0]:crop["z"][1]], np.float64)
    return x, y, z


def _extinction(lwc, z, iwc):
    """The one extinction the whole toolkit renders — re 10 um / 30 um ice."""
    return optical_depth.compute_extinction_field(lwc, z, re=10.0, iwc=iwc,
                                                  re_ice=30.0)


def occupied_z_band(spec: dict, slab: int = 256) -> tuple:
    """The z planes that hold cloud, by the rule witness and the browser use.

    Why this is computed rather than typed. A camera position in soar is
    normalized to the domain box, and BOTH the browser's ingest and witness
    trim empty z planes before rendering — so the box a flight is placed in is
    the occupied band, not the file's z extent. Thomas flies a raw netCDF,
    copies the reproduction command, and that camera means "relative to the
    band". If the bake then ships a volume cut anywhere else, the still and
    the flight are two different scenes and entering flight is a jump cut.

    The band is found from a per-plane maximum accumulated over x-slabs, so
    this costs one streamed pass and a few GB rather than a second copy of a
    field that can be tens of gigabytes. The decision itself is handed to
    crop_empty_z on a 1x1xnz array — the same function, so there is one
    definition of "occupied" and not a second one that agrees today.
    """
    src = source_dir(spec)
    path = src / spec["liquid"][0]
    with Dataset(path) as ds:
        z = np.asarray(ds.variables["z"][:], np.float64)
    full_z = (0, int(z.size))
    xs = spec["crop"]["x"]
    scale = np.float32(spec["scale"])
    peak = np.zeros(z.size, dtype=np.float64)
    for i0 in range(xs[0], xs[1], slab):
        window = dict(spec["crop"], x=(i0, min(i0 + slab, xs[1])), z=full_z)
        lwc = _read(path, spec["liquid"][1], spec["dims"], window)
        iwc = (_read(src / spec["ice"][0], spec["ice"][1], spec["dims"], window)
               if spec["ice"] else None)
        if scale != 1.0:
            lwc *= scale
            if iwc is not None:
                iwc *= scale
        sigma = _extinction(lwc, z, iwc)
        peak = np.maximum(peak, np.nanmax(sigma, axis=(0, 1)))
        print(f"    scanning for cloud: x {min(i0 + slab, xs[1])}/{xs[1]}",
              end="\r", flush=True)
    _, _, (lo, hi) = crop_empty_z(peak.reshape(1, 1, -1), z)
    print(f"    occupied z band {lo}:{hi + 1} = {z[lo]:.0f}..{z[hi]:.0f} m "
          f"({hi + 1 - lo} of {z.size} planes, "
          f"{100 * (1 - (hi + 1 - lo) / z.size):.0f}% vacuum dropped)")
    return lo, hi + 1


def resolved_crop(spec: dict) -> dict:
    """The spec's crop with an `auto` z resolved to the occupied band."""
    crop = dict(spec["crop"])
    if crop.get("z") == "auto":
        crop["z"] = occupied_z_band(spec)
    return crop


def load_demo(spec: dict) -> CloudField:
    src = source_dir(spec)
    crop = resolved_crop(spec)
    lwc = _read(src / spec["liquid"][0], spec["liquid"][1], spec["dims"], crop)
    iwc = None
    if spec["ice"]:
        iwc = _read(src / spec["ice"][0], spec["ice"][1], spec["dims"], crop)
    s = np.float32(spec["scale"])
    if s != 1.0:
        lwc *= s
        if iwc is not None:
            iwc *= s
    x, y, z = _coords(src / spec["liquid"][0], crop)
    return CloudField(lwc=lwc, iwc=iwc, x=x, y=y, z=z,
                      source=str(src / spec["liquid"][0]),
                      liquid_var=spec["liquid"][1],
                      ice_var=spec["ice"][1] if spec["ice"] else None)


# --- the shipped volume ----------------------------------------------------

def bake_volume(spec: dict, field: CloudField, out: Path) -> dict:
    sigma = np.ascontiguousarray(
        _extinction(field.lwc, field.z, field.iwc), dtype=np.float16)
    nx, ny, nz = sigma.shape

    # The bare field: no ghost border, and no faces.bin beside it. The
    # browser tapers and wraps in the shader, so toggling periodic still
    # needs no second download and a 2048-cell axis still fits.
    raw = sigma.tobytes()
    with gzip.open(out / "volume.bin.gz", "wb", compresslevel=6) as fh:
        fh.write(raw)
    gz_bytes = (out / "volume.bin.gz").stat().st_size
    print(f"    volume.bin.gz {gz_bytes/1e6:7.1f} MB "
          f"(from {len(raw)/1e6:.1f} MB, {len(raw)/gz_bytes:.1f}x)")
    stale = out / "faces.bin"
    if stale.exists():
        stale.unlink()
        print("    removed faces.bin (the wrap is a sampler mode now)")

    albedo = np.ascontiguousarray(glimpse(field), dtype=np.float32)
    (out / "map.bin").write_bytes(albedo.tobytes())

    bmin, bmax = _volume_aabb(field)
    return {
        "schema": "cloudyview.web.demo.v5",
        "id": spec["id"],
        "title": spec["title"],
        "field": spec["field"],
        # Absent rather than empty when a case has nothing to add: the landing
        # page builds the expansion only for the keys that are here, and an
        # empty string would open a card onto a blank line.
        **{k: spec[k] for k in ("description", "warning") if spec.get(k)},
        "source": Path(spec["liquid"][0]).name,
        "periodic": bool(spec.get("periodic", True)),
        "volume": {
            "shape_xyz": [int(nx), int(ny), int(nz)],
            "format": "r16float",
            "compression": "gzip",
            "file": "volume.bin.gz",
            "bytes": int(gz_bytes),
            "bytes_uncompressed": int(len(raw)),
            "bmin": [float(v) for v in bmin],
            "bmax": [float(v) for v in bmax],
        },
        "map": {"shape_yx": [int(albedo.shape[0]), int(albedo.shape[1])]},
        "sun": spec["sun"],
    }


# --- the preview still ------------------------------------------------------

def render_still(spec: dict, field: CloudField, out: Path, size,
                 accumulate: int, quality: int) -> dict:
    """One converged frame, which is what a hover preview should be.

    This was a video for a while. The arithmetic killed it: a 60 s 1440p60
    loop lands somewhere around 30-110 MB per demo, which for the FIF cascade
    is fifteen times the weight of the 1.9 MB field it is advertising. A 4K
    still is 60-210 kB — two to three orders of magnitude less — renders in
    seconds rather than hours, and cannot judder.

    Motion, where it is wanted, is a slow CSS scale on the image and costs
    nothing.
    """
    still = spec.get("still", {})
    # Default framing looks across the sun rather than into or away from it,
    # which is where cloud sides read best. Override per demo in DEMOS.
    position = tuple(still.get("position", (0.0, 0.0, -0.999)))
    azimuth = still.get("azimuth", (spec["sun"]["azimuth"] + 130.0) % 360.0)
    elevation = still.get("elevation", 20.0)
    fov = still.get("fov", 100.0)
    # A per-demo size wins over --size, and it is given as the size Thomas
    # framed at so the aspect is his; the multiplier is what makes it sharp on
    # a 4K display. Framing is unchanged by the multiplier because the aspect
    # ratio is, which is the only thing the projection cares about.
    if still.get("size"):
        size = (still["size"][0] * still.get("size_multiple", 2),
                still["size"][1] * still.get("size_multiple", 2))

    # The look, resolved: the app's defaults with the demo's overrides on top.
    # Resolved rather than sparse because these numbers are also the record of
    # how the picture was made — witness has to be able to reproduce it from
    # them alone, and "the rest were the defaults" is only a reproduction if
    # you also know which day it was.
    look = {**LOOK_DEFAULTS, **{k: still[k] for k in LOOK_DEFAULTS if k in still}}
    differs = {k: v for k, v in look.items() if v != LOOK_DEFAULTS[k]}

    print(f"    still: {size[0]}x{size[1]}, {accumulate} passes, "
          f"azimuth {azimuth:.2f}, elevation {elevation:.2f}, "
          + (f"{', '.join(f'{k}={v}' for k, v in differs.items())} "
             f"(rest at app defaults)" if differs else "look at app defaults"),
          flush=True)

    t0 = time.time()
    img = cv.witness(
        field,
        cv.Camera(position=position, azimuth=azimuth,
                  elevation=elevation, fov=fov),
        size=size, sun_azimuth=spec["sun"]["azimuth"],
        sun_elevation=spec["sun"]["elevation"],
        periodic=spec.get("periodic", True), accumulate=accumulate,
        **{_LOOK_ARG[k]: v for k, v in look.items()})
    render_s = time.time() - t0

    tmp = Path(tempfile.mkdtemp(prefix=f"soar-{spec['id']}-"))
    try:
        master = tmp / "still.png"
        _write_png(master, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
        # WebP rather than AVIF: supported everywhere WebGPU is, and an AVIF
        # trial came out an order of magnitude worse than JPEG rather than
        # better, so it is not paying for its complexity here.
        subprocess.run(["magick", str(master), "-quality", str(quality),
                        "-define", "webp:method=6", "-strip",
                        str(out / "still.webp")], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    nbytes = (out / "still.webp").stat().st_size
    print(f"    still.webp {nbytes/1e3:.0f} kB (rendered in {render_s:.1f}s)")
    # The camera goes into meta.json because the app opens there: the viewer
    # reads `still.camera` and starts flight from it, so clicking a case puts
    # you where its picture was taken from. `look` rides along as the record
    # of how the picture was tone-mapped — witness reproduces it exactly from
    # these numbers, and the viewer can choose to adopt them.
    return {"file": "still.webp", "size": list(size), "bytes": int(nbytes),
            "camera": {"position": list(position), "azimuth": azimuth,
                       "elevation": elevation, "fov": fov},
            "look": look,
            # Kept flat as well: written before the camera block existed, and
            # something may still be reading them.
            "azimuth": azimuth, "elevation": elevation,
            "accumulate": accumulate}


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer — avoids adding an image dependency for the bake."""
    import struct, zlib
    h, w, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


# --- the index the landing page reads ---------------------------------------

def _index_entry(spec: dict) -> dict:
    """One row of index.json, read back from what is actually on disk.

    Nothing here comes from the run that baked it: the row is rebuilt from
    the demo's meta.json and checked against the files beside it, so
    --index-only produces exactly what a full bake would and a truncated
    download or a half-finished bake is a hard error rather than a demo that
    404s in the browser.
    """
    out = OUT / spec["id"]
    meta_path = out / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} — {spec['id']} has not been baked")
    meta = json.loads(meta_path.read_text())

    volume = meta.get("volume") or {}
    if not volume:
        raise ValueError(f"{meta_path} records no volume")
    vol = out / volume["file"]
    if not vol.exists():
        raise FileNotFoundError(f"{vol} — meta.json names it but it is missing")
    if int(volume["bytes"]) != vol.stat().st_size:
        raise ValueError(f"{vol} is {vol.stat().st_size} bytes, meta.json says "
                         f"{volume['bytes']} — re-bake {spec['id']}")

    still = meta.get("still") or {}
    if still:
        img = out / still["file"]
        if not img.exists():
            raise FileNotFoundError(f"{img} — meta.json names it but it is missing")
        if int(still["bytes"]) != img.stat().st_size:
            raise ValueError(f"{img} is {img.stat().st_size} bytes, meta.json says "
                             f"{still['bytes']} — re-render the still")

    return {k: meta[k] for k in ("id", "title", "field", "description", "warning")
            if k in meta} | {
        "base": spec["id"],
        "bytes": int(volume["bytes"]),
        "still": still.get("file"),
    }


def write_index(strict: bool) -> None:
    """Rebuild web/soar/demos/index.json from the per-demo meta.json files.

    `strict` is off during a bake, where a demo declared in DEMOS but never
    baked is a normal intermediate state; it is on for --index-only, where a
    missing bake would quietly drop a demo off the page.
    """
    groups, total, count = [], 0, 0
    for g in GROUPS:
        rows = []
        for spec in (s for s in DEMOS if s["group"] == g["id"]):
            try:
                rows.append(_index_entry(spec))
            except FileNotFoundError as err:
                if strict:
                    raise
                print(f"    ! {spec['id']} left out of the index: {err}")
        entry = {"id": g["id"], "title": g["title"]}
        if not rows:
            entry["status"] = "coming-soon"
        entry["demos"] = rows
        groups.append(entry)
        total += sum(r["bytes"] for r in rows)
        count += len(rows)

    path = OUT / "index.json"
    path.write_text(json.dumps({"schema": "soar.demos.v2", "groups": groups},
                               indent=1))
    shape = ", ".join(f"{g['title']} {len(g['demos'])}" for g in groups)
    print(f"\n{path} — {count} demos ({shape}), {total/1e6:.0f} MB of volume")


# --- driver ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="demo ids to build")
    ap.add_argument("--index-only", action="store_true",
                    help="rewrite web/soar/demos/index.json from the existing "
                         "per-demo meta.json files and bake nothing; the way "
                         "to reshuffle grouping without the sources or a GPU")
    ap.add_argument("--skip-still", action="store_true")
    ap.add_argument("--skip-volume", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=[3840, 2160],
                    help="still resolution (default 4K)")
    ap.add_argument("--accumulate", type=int, default=192,
                    help="accumulated passes; it is one frame (default 192)")
    ap.add_argument("--quality", type=int, default=82,
                    help="WebP quality (default 82)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.index_only:
        if args.only:
            ap.error("--index-only rewrites the whole index; --only means nothing")
        write_index(strict=True)
        return

    for spec in DEMOS:
        if args.only and spec["id"] not in args.only:
            continue
        out = OUT / spec["id"]
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {spec['id']}: {spec['title']} ===", flush=True)
        # Only when a product needs it. With both --skip-volume and
        # --skip-still this pass writes nothing but text, and reading a
        # gigabyte of netCDF to copy a sentence out of DEMOS would put the
        # source files — which are not in the repo — between anyone and a
        # one-word edit to a card.
        field = None
        if not (args.skip_volume and args.skip_still):
            t0 = time.time()
            field = load_demo(spec)
            print(f"    loaded {field.lwc.shape}"
                  f"{' + ice' if field.iwc is not None else ''} "
                  f"in {time.time()-t0:.0f}s", flush=True)

        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not args.skip_volume:
            previous_still = meta.get("still")
            meta = bake_volume(spec, field, out)
            # bake_volume writes the meta from scratch, which drops the still
            # block. still.webp is on disk either way, and the landing page
            # opens flight on the camera that block names, so carry it over
            # rather than silently shipping a demo that starts nowhere.
            if args.skip_still and previous_still is not None:
                meta["still"] = previous_still
        else:
            # --skip-volume keeps the baked arrays, but everything in the meta
            # that is simply a copy of the spec has to follow the spec anyway
            # — otherwise editing the sun or the description here changes the
            # picture and leaves the file still describing the old one, which
            # is a worse failure than a slow re-bake because nothing reports
            # it. Only volume/map are genuinely derived from the arrays.
            for key in ("title", "field"):
                meta[key] = spec[key]
            # The optional prose has to be *removed* when the spec drops it,
            # not merely overwritten: deleting a description from DEMOS and
            # leaving the old sentence in meta.json is the same failure the
            # paragraph above is about, in its quietest form.
            for key in ("description", "warning"):
                if spec.get(key):
                    meta[key] = spec[key]
                else:
                    meta.pop(key, None)
            meta["sun"] = dict(spec["sun"])
            meta["periodic"] = bool(spec.get("periodic", True))
        if not args.skip_still:
            meta["still"] = render_still(spec, field, out, tuple(args.size),
                                         args.accumulate, args.quality)
        meta_path.write_text(json.dumps(meta, indent=1))
        del field

    # Always from disk, never from this run: a partial bake and a full one
    # then write the same file, and there is no merge step to get wrong.
    write_index(strict=False)


if __name__ == "__main__":
    main()
