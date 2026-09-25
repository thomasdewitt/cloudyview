"""The night city's lots, zoning and streets: the bake and its contract.

cloudyview/city_lots.py decides offline which blocks form one lot, what each
lot is used for and which grid edges carry a street, and packs the answers
into the city tile's b and a channels at mip 0. raymarch.wgsl city_cell
decodes them. These tests pin three things: that the shipped tile IS the
bake of its own r and g (nobody hand-edited or half-regenerated it), that
the bake keeps its promises (every member of a lot agrees on the lot, no lot
straddles an avenue, no street runs inside a lot or through open land, the
network is one piece), and that the Python and WGSL copies of the shared
numbers are the same numbers.

Needs no GPU: this is arithmetic on the tile's bytes.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from cloudyview import city_lots
from cloudyview.city_lots import (
    USE_INDUSTRIAL, USE_OPEN, USE_URBAN, USE_YARD, WORD_OFFSET,
    bake_lots, glow_weight,
)

REPO = Path(__file__).resolve().parents[1]
TILE = REPO / "cloudyview" / "soar" / "city"
WGSL = REPO / "cloudyview" / "soar" / "raymarch.wgsl"


def _mip(level):
    n = 1024 >> level
    raw = (TILE / f"fif_mip{level}.bin").read_bytes()
    return np.frombuffer(raw, dtype=np.float16).reshape(n, n, 4).astype(
        np.float32)


@pytest.fixture(scope="module")
def tile():
    t = _mip(0)
    lot = (t[..., 2] + WORD_OFFSET).astype(np.int64)
    st = (t[..., 3] + WORD_OFFSET).astype(np.int64)
    return {"density": t[..., 0], "rank": t[..., 1], "lot": lot, "st": st}


@pytest.fixture(scope="module")
def decoded(tile):
    lot, st = tile["lot"], tile["st"]
    n = lot.shape[0]
    Y, X = np.mgrid[0:n, 0:n]
    back_x, back_y = lot & 3, (lot >> 2) & 3
    return {
        "n": n,
        "x0": X - back_x, "y0": Y - back_y,       # unwrapped lot min
        "w": ((lot >> 4) & 3) + 1, "h": ((lot >> 6) & 3) + 1,
        "ox": (lot >> 8) & 3, "oy": (lot >> 10) & 3,
        "edges": st & 0xF, "use": (st >> 8) & 3,
    }


def test_the_shipped_tile_is_the_bake_of_its_own_channels(tile):
    lot_word, street_word, _ = bake_lots(tile["density"], tile["rank"],
                                         verbose=False)
    assert np.array_equal(lot_word, tile["lot"]), (
        "the tile's lot words are not what city_lots.bake_lots makes of its "
        "r and g; re-run tools/export_web_assets.py's export_city")
    assert np.array_equal(street_word, tile["st"])


def test_every_member_of_a_lot_agrees_on_it(decoded):
    """The DDA assembles one building from whichever member cell it visits,
    so every member must decode the same rectangle and owner."""
    d = decoded
    n = d["n"]
    for dy in range(4):
        for dx in range(4):
            # Cell (x0+dx, y0+dy) of each lot, where the lot is that big.
            inside = (dx < d["w"]) & (dy < d["h"])
            mx = (d["x0"] + dx) % n
            my = (d["y0"] + dy) % n
            for key in ("x0", "y0", "w", "h", "ox", "oy", "use"):
                a = d[key][inside]
                b = d[key][my[inside], mx[inside]]
                if key in ("x0", "y0"):
                    a, b = a % n, b % n
                assert np.array_equal(a, b), f"lot members disagree on {key}"
    assert np.all(d["ox"] < d["w"]) and np.all(d["oy"] < d["h"])


def test_no_lot_swallows_a_rival_tower(tile, decoded):
    """The skyline's cluster of needles is peaks on adjacent blocks: a lot
    that took one of them into another's building turned the cluster into a
    single tower. No member of a multi-block lot may be a block whose own
    tower would reach city_lots.RIVAL_FRAC of the owner's (or a tower lot's
    threshold, whichever is higher)."""
    d = decoded
    n = d["n"]
    h0 = city_lots.potential_height(tile["density"], tile["rank"])
    Y, X = np.mgrid[0:n, 0:n]
    ox = (d["x0"] + d["ox"]) % n
    oy = (d["y0"] + d["oy"]) % n
    member = (d["w"] * d["h"] > 1) & ((X != ox) | (Y != oy))
    bound = np.maximum(city_lots.TOWER_LOT_H[0][1],
                       city_lots.RIVAL_FRAC * h0[oy, ox])
    rivals = member & (h0 >= bound)
    assert not rivals.any(), (
        f"{int(rivals.sum())} lot members are rival towers, e.g. cell "
        f"{tuple(np.argwhere(rivals)[0][::-1])}")


def test_no_lot_straddles_an_avenue(decoded):
    """skyway's decks ride the avenue corridors as guaranteed empty air."""
    period = city_lots.AVENUE_PERIOD
    for start, size in ((decoded["x0"], decoded["w"]),
                        (decoded["y0"], decoded["h"])):
        for k in range(1, 4):
            interior = (k < size) & (((start + k) % period) == 0)
            assert not interior.any()


def test_streets_run_only_between_lots_and_never_through_open_land(decoded):
    d = decoded
    lot_id = (d["y0"] % d["n"]) * d["n"] + (d["x0"] % d["n"])
    west = (d["edges"] & 1) != 0
    south = (d["edges"] & 4) != 0
    # An edge inside a lot carries no street.
    assert not np.any(west & (lot_id == np.roll(lot_id, 1, axis=1)))
    assert not np.any(south & (lot_id == np.roll(lot_id, 1, axis=0)))
    # The +x/+y bits are the neighbour's -x/-y bits, seen from this side.
    assert np.array_equal((d["edges"] >> 1) & 1,
                          np.roll(d["edges"] & 1, -1, axis=1))
    assert np.array_equal((d["edges"] >> 3) & 1,
                          np.roll((d["edges"] >> 2) & 1, -1, axis=0))
    # Open scrub fronts no street unless the network needed a road through
    # it to reach something: then the far side of that edge is not open
    # (a joining road through two open blocks is the only exception, and
    # it must connect).
    both_open = (d["use"] == USE_OPEN) & (np.roll(d["use"], 1, axis=1)
                                          == USE_OPEN)
    frac = np.mean(west[both_open]) if both_open.any() else 0.0
    assert frac < 0.02, f"{frac:.3f} of open-open edges carry a street"


def test_the_street_network_is_one_piece(decoded):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = decoded["n"]
    west = (decoded["edges"] & 1) != 0
    south = (decoded["edges"] & 4) != 0
    Y, X = np.mgrid[0:n, 0:n]
    node = lambda x, y: (y % n) * n + (x % n)
    a = np.concatenate([node(X, Y)[west], node(X, Y)[south]])
    b = np.concatenate([node(X, Y + 1)[west], node(X + 1, Y)[south]])
    g = coo_matrix((np.ones(a.size), (a, b)), shape=(n * n, n * n))
    _, label = connected_components(g, directed=False)
    used = np.unique(np.concatenate([a, b]))
    assert np.unique(label[used]).size == 1


def test_the_land_uses_are_the_documented_ones(decoded):
    use = decoded["use"]
    assert set(np.unique(use)) <= {USE_URBAN, USE_INDUSTRIAL, USE_YARD,
                                   USE_OPEN}
    # The belt is a belt, not the city: most of the tile stays urban.
    assert np.mean(use == USE_URBAN) > 0.5
    assert 0.05 < np.mean(use == USE_INDUSTRIAL) < 0.30


def test_the_glow_is_the_land_use_weighted_density():
    """Mip 0's r is the density; mips 1 and up are its weighted box average."""
    t0 = _mip(0)
    st = (t0[..., 3] + WORD_OFFSET).astype(np.int64)
    glow = t0[..., 0] * glow_weight(st)
    t1 = _mip(1)
    expect = 0.25 * (glow[0::2, 0::2] + glow[1::2, 0::2]
                     + glow[0::2, 1::2] + glow[1::2, 1::2])
    np.testing.assert_allclose(t1[..., 0], expect, rtol=2e-3, atol=2e-4)
    # b and a are words at mip 0 only.
    assert not t1[..., 2].any() and not t1[..., 3].any()


# (Python name, WGSL name)
_SHARED = [
    ("EMPTY_RANK", "CITY_EMPTY_RANK"),
    ("SPRAWL_RANK_FULL", "CITY_SPRAWL_RANK_FULL"),
    ("SPRAWL_MIN_FRAC", "CITY_SPRAWL_MIN_FRAC"),
    ("H_BASE", "CITY_H_BASE"),
    ("H_SCALE", "CITY_H_SCALE"),
    ("H_EXP", "CITY_H_EXP"),
    ("WORD_OFFSET", "CITY_WORD_OFFSET"),
]
_SHARED_INT = [
    ("AVENUE_PERIOD", "CITY_AVENUE_PERIOD"),
    ("USE_URBAN", "CITY_USE_URBAN"),
    ("USE_INDUSTRIAL", "CITY_USE_INDUSTRIAL"),
    ("USE_YARD", "CITY_USE_YARD"),
    ("USE_OPEN", "CITY_USE_OPEN"),
]


@pytest.mark.parametrize("py_name,wgsl_name", _SHARED)
def test_shared_float_constants_agree(py_name, wgsl_name):
    m = re.search(rf"const\s+{wgsl_name}\s*:\s*f32\s*=\s*([-\d.eE+]+)\s*;",
                  WGSL.read_text())
    assert m, f"{wgsl_name} is gone from raymarch.wgsl"
    assert float(m.group(1)) == getattr(city_lots, py_name)


@pytest.mark.parametrize("py_name,wgsl_name", _SHARED_INT)
def test_shared_int_constants_agree(py_name, wgsl_name):
    m = re.search(rf"const\s+{wgsl_name}\s*:\s*i32\s*=\s*(-?\d+)\s*;",
                  WGSL.read_text())
    assert m, f"{wgsl_name} is gone from raymarch.wgsl"
    assert int(m.group(1)) == getattr(city_lots, py_name)


def test_every_surface_tile_carries_the_schema_both_hosts_expect():
    """A new shader must never read an old tile as if it were new."""
    import json

    from cloudyview.soar_host import SURFACE_DIRS, SURFACE_TILE_SCHEMAS

    js = (REPO / "web" / "soar" / "constants.js").read_text()
    block = re.search(r"SURFACE_TILE_SCHEMAS\s*=\s*\{(.*?)\}", js, re.S)
    assert block, "constants.js SURFACE_TILE_SCHEMAS is gone"
    js_schemas = dict(re.findall(r'(\w+):\s*"([^"]+)"', block.group(1)))
    assert js_schemas == SURFACE_TILE_SCHEMAS
    for surface, tile_dir in SURFACE_DIRS.items():
        meta = json.loads((tile_dir / "meta.json").read_text())
        assert meta["schema"] == SURFACE_TILE_SCHEMAS[surface], surface
