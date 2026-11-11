"""What the world stands on: the surfaces, their tiles, and their stages.

A surface is "ocean" (the day scene) or "city" (the night city). Each has its
own shader module (shader.py) and its own tile: a periodic rgba16float mip
chain in soar/ocean/ or soar/city/, described by a meta.json. The ocean's tile
is a field of water normals, the city's a building-density field with the
lots and streets packed into two channels. Both bind at binding 3, so exactly
one is loaded per renderer.

web/soar/constants.js and web/soar/scene.js hold the browser's copies of
these tables and decoders, and tests pin the two together.
"""

from pathlib import Path
from typing import Optional

import numpy as np

SURFACES = ("ocean", "city")

# The tiles ship inside the package; the browser reaches the same files
# through symlinks under web/soar/.
SURFACE_DIRS = {"ocean": Path(__file__).resolve().parent / "ocean",
                "city": Path(__file__).resolve().parent / "city"}

# The meta.json schema each tile must carry: the layout the shader decodes it
# with. A tile of another schema is refused, because an old city tile decoded
# with the new layout draws a different city without any error.
SURFACE_TILE_SCHEMAS = {
    "ocean": "cloudyview.web.ocean.v2",
    "city": "cloudyview.web.city.v2",
}

# How many tiles each surface's frame folds at (surface_period_m). The city
# folds at its tile: its frame is the district, so a recorded flight must
# come back to the same one. The sea folds at 16 tiles (1638.4 m). Its tile
# repeats every 102.4 m, so any whole number of tiles folds invisibly, and a
# long frame is what a recorded track can carry between two samples: the
# track resamplers (track.py, track.js) unwrap the surface position at half
# the period, 819 m here. At one tile that would be 51 m, less than the 75 m
# a flight at 1500 m/s covers between samples recorded at 20 fps.
SURFACE_FRAME_TILES = {"ocean": 16, "city": 1}

# Channels per texel of each tile format.
SURFACE_FORMAT_CHANNELS = {"rgba16float": 4}

# The surface stages each traced surface runs before the march, in order:
# fragment entry points in shader/passes.wgsl. Each stage after the first
# reads the one before at group 1 binding 1, and the last writes the surface
# the march reads. The city: its boxes (fs_surface), the facade shell over
# them (fs_surface_refine), then the street (fs_surface_street: furniture,
# wet reflection, steam). The ocean has none; the march shades it directly.
SURFACE_STAGES = {
    "city": ("fs_surface", "fs_surface_refine", "fs_surface_street"),
}
# The stages that also write the veil, a second attachment: the first, and
# the city's street.
VEIL_STAGES = ("fs_surface", "fs_surface_street")


def check_surface(surface: str) -> str:
    """The surface name, or a ValueError naming the ones that exist."""
    if surface == "desert":
        raise ValueError(
            "the desert surface is no longer part of cloudyview's dev line: it "
            "lives on the `desert` branch (and its terrain generator in the "
            f"desert-surface repo). The surfaces here: {', '.join(SURFACES)}.")
    if surface not in SURFACES:
        raise ValueError(
            f"surface must be one of {', '.join(SURFACES)}; got {surface!r}.")
    return surface


def surface_cell_m(surface: str, meta: dict) -> float:
    """A tile's texel size in metres, from its meta.json.

    The ocean names it "dx_m" (a sampling step of a wave field), the city
    "cell_m" (a city block).
    """
    key = "dx_m" if check_surface(surface) == "ocean" else "cell_m"
    if key not in meta:
        raise KeyError(f"the {surface} tile's meta.json has no {key!r}.")
    return float(meta[key])


def surface_period_m(surface: str, meta: dict) -> float:
    """The period the surface frame folds at, in metres: a whole number of
    tiles (SURFACE_FRAME_TILES). Each surface is a pure function of its tile,
    so the fold is invisible."""
    check_surface(surface)
    if "tile_extent_m" not in meta:
        raise KeyError(f"the {surface} tile's meta.json has no 'tile_extent_m'.")
    return SURFACE_FRAME_TILES[surface] * float(meta["tile_extent_m"])


def surface_clock_period_s(surface: str, meta: dict) -> Optional[float]:
    """A surface's clock, from its meta.json "clock_period_s".

    A number is the period its animation repeats at exactly, where the hosts
    wrap the scene clock (the city's steam: 2600 s). null (None) is a surface
    where nothing moves (the ocean), whose clock packs as 0 at any scene time.

    The key must be there: a tile that does not say is an error, not a
    surface without a clock. A period must be positive, because the wrap is a
    modulo and a modulo by 0 is NaN.
    """
    check_surface(surface)
    if "clock_period_s" not in meta:
        raise KeyError(f"the {surface} tile's meta.json has no "
                       "'clock_period_s' (a period in seconds, or null for "
                       "a surface with no clock).")
    period = meta["clock_period_s"]
    if period is None:
        return None
    if not float(period) > 0:
        raise ValueError(f"the {surface} tile's clock_period_s must be "
                         f"positive or null; got {period!r}.")
    return float(period)


def surface_level_files(meta: dict) -> list:
    """The file of each mip level: the meta's "files", or fif_mip<level>.bin
    for a meta that does not list them."""
    if "files" in meta:
        return list(meta["files"])
    return [f"fif_mip{level}.bin" for level in range(int(meta["mips"]))]


def decode_surface_level(raw: bytes, side: int, meta: dict) -> bytes:
    """One stored mip level -> the interleaved texel bytes the texture wants.

    A level is stored either as the texels themselves, or (encoding
    "planar-byte-shuffle", the ocean's) as each channel's plane with every
    half float's low bytes before its high bytes. The high bytes of a smooth
    field barely change from texel to texel, so gzip compresses that
    arrangement far better. `raw` is already decompressed.
    web/soar/scene.js decodeSurfaceLevel is the browser's copy.
    """
    channels = SURFACE_FORMAT_CHANNELS[meta.get("format", "rgba16float")]
    expected = side * side * channels * 2
    if len(raw) != expected:
        raise RuntimeError(
            f"a {side}x{side} {meta.get('format', 'rgba16float')} level is "
            f"{expected} bytes; this one is {len(raw)}.")
    encoding = meta.get("encoding", "interleaved")
    if encoding == "interleaved":
        return raw
    if encoding != "planar-byte-shuffle":
        raise RuntimeError(f"unknown surface-tile encoding {encoding!r}.")
    b = np.frombuffer(raw, np.uint8).reshape(channels, 2, side * side)
    return np.ascontiguousarray(b.transpose(2, 0, 1)).tobytes()


def encode_surface_level(level: np.ndarray) -> bytes:
    """(side, side, channels) -> planar, byte-shuffled float16 bytes: the
    "planar-byte-shuffle" encoding decode_surface_level undoes. Used by
    tools/export_web_assets.py when it writes a tile."""
    planes = []
    for c in range(level.shape[-1]):
        b = np.ascontiguousarray(level[..., c], dtype=np.float16).view(np.uint8)
        b = b.reshape(-1, 2)
        planes += [b[:, 0].tobytes(), b[:, 1].tobytes()]
    return b"".join(planes)


def surface_stage_entries(surface: str) -> tuple:
    """The surface stages a view of `surface` runs, in order; none over the
    sea."""
    return SURFACE_STAGES.get(check_surface(surface), ())


def surface_stage_plan(n: int) -> list:
    """Where each of `n` chained stages writes and what it reads, as
    (writes, reads): "final" is the surface the march reads, "a" and "b" the
    two intermediate textures the chain alternates between, and None means
    the first stage reads nothing. web/soar/renderer.js surfaceStagePlan is
    the browser's copy."""
    plan = []
    for i in range(n):
        writes = "final" if i == n - 1 else "ab"[i % 2]
        reads = None if i == 0 else "ab"[(i - 1) % 2]
        plan.append((writes, reads))
    return plan
