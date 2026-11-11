"""The night city's land use: lots, zoning and the street network, baked.

The density tile (city_fif) says how much city stands on each 90 m block.
This module decides what KIND of city, and where the roads run, from the
same two channels — offline, because both questions are non-local: a
tall tower claims the neighbouring blocks around its peak (a lot has to be
agreed by every member cell, and two peaks cannot claim the same block),
and a street is only worth drawing if it connects to the rest of the
network. Neither can be answered from one texel in a shader. The answers
are packed into the tile's two spare channels at mip 0 and read with
textureLoad by the renderer (city_cell in shader/city/layout.wgsl) and by the flight
collision mirror (web/soar/citylots.js) alike, so the city stays a pure
function of the tile frame.

Zoning follows the district, not the block. The cascade is rough at every
scale (H = 0), so single blocks flicker between built and unbuilt even
downtown; what separates the core from the fringe is the rank averaged over
a few blocks. Where that average is high the city is URBAN: every built
block is a building and the street grid is the full 90 m lattice. Where it
is low the city thins into an INDUSTRIAL belt — warehouses on 180 m
superblocks served by a street grid of half the density — and its unbuilt
blocks become YARDS (vacant lots, container and trailer storage, construction
refuse) where they front development, or OPEN scrub where they do not.
OPEN land has no streets of its own: the road network is exactly the one
that serves something (a region with no buildings has no streets), and the
only roads across open scrub are the
connectors that join a stranded works to the network (under 1% of open-open
edges).

Massing that needs agreement lives here too. A tall tower's podium wants
more than one block: the peak cell claims a k x k lot centred on itself (k
up to 4), greedily from the tallest down, never across an avenue line
(skyway's elevated decks rely on avenues being empty air), never across
another claim, and never over a RIVAL — a block whose own tower would reach
RIVAL_FRAC of the claimant's. The tile's tallest peaks stand on adjacent
blocks, and one lot over them turned the skyline's cluster of needles into
a single tower; with the rule the peaks of a cluster keep a block each, and
lots go only to towers standing among lower neighbours. Every member cell
reports the owner's density, rank and seed, so every tower keeps exactly the
height it had.

Packing (both channels, mip 0 only; the coarser mips of b and a are zeroed,
because an average of bitfields means nothing and nothing reads it):

    b = lot word - 2048        a = street word - 2048

    lot word    bits 0-1  cells from this cell back to the lot's min x
                bits 2-3  ... to the lot's min y
                bits 4-5  lot width - 1 (cells)
                bits 6-7  lot height - 1
                bits 8-9  owner cell x within the lot
                bits 10-11 owner cell y within the lot
    street word bits 0-3  edge carries a street: -x, +x, -y, +y
                bits 4-7  corner square paved: (-x,-y) (+x,-y) (-x,+y) (+x,+y)
                bits 8-9  land use: 0 urban, 1 industrial, 2 yard, 3 open

Each word is at most 12 bits, stored offset by -2048 so it spans
[-2048, 2047]: every integer there is exact in float16 (whose integers are
exact to magnitude 2048), which is the whole reason for the offset.
"""
from typing import Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# The city shader has the same numbers; tests/city/test_city_lots.py pins them.
EMPTY_RANK = 0.22
SPRAWL_RANK_FULL = 0.60
SPRAWL_MIN_FRAC = 0.15
H_BASE = 14.0
H_SCALE = 390.0
H_EXP = 1.2
AVENUE_PERIOD = 8

USE_URBAN, USE_INDUSTRIAL, USE_YARD, USE_OPEN = 0, 1, 2, 3

# District zoning: rank smoothed over a few blocks, then cut. Below the cut
# the city is fringe (industrial belt, yards, scrub).
DISTRICT_SIGMA_CELLS = 2.5
DISTRICT_URBAN_RANK = 0.40
# The industrial belt's street lattice: every INDUSTRIAL_STREET_PERIOD-th
# grid line, i.e. 180 m superblocks.
INDUSTRIAL_STREET_PERIOD = 2

# Tower lots: the unjittered height (city_cell's law without its per-building
# jitter) at which a peak claims a k x k lot for its podium (city_cell keeps
# the shaft itself slim; see CITY_TOWER_KNEE_H).
TOWER_LOT_H = ((2, 330.0), (3, 900.0), (4, 2000.0))
# A block whose own tower would reach this fraction of the claimant's (and
# is a tower-lot candidate itself) is a rival, never part of its lot.
RIVAL_FRAC = 0.5
# A few dense mid-rise groups merge into one full-block building anyway —
# the department store, the convention centre — a 2x2 superblock's worth.
MIDRISE_MERGE_FRAC = 0.06
MIDRISE_MERGE_RANK = 0.50

WORD_OFFSET = 2048.0

# The glow (the r channel's mips 1 and up, read by the fog, the sky dome and
# the clouds' city uplight) is density weighted by how much of it the land
# use actually lights. Before zoning the glow was plain density, which is
# right for the core; the fringe now keeps a warehouse's few floods and
# office windows instead of lit low-rise and a full lamp grid, and open
# scrub keeps nothing. The weights are the resolved emission of each use
# relative to an urban block of the same density, estimated from the
# shader's means (window ladder, storefront strip, lamp pools, flood pools).
GLOW_WEIGHT = {USE_URBAN: 1.0, USE_INDUSTRIAL: 0.45, USE_YARD: 0.30,
               USE_OPEN: 0.0}


def glow_weight(street_word: np.ndarray) -> np.ndarray:
    use = (street_word.astype(np.int64) >> 8) & 3
    w = np.zeros(use.shape, dtype=np.float32)
    for u, g in GLOW_WEIGHT.items():
        w[use == u] = g
    return w


def potential_height(density: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """city_cell's height law with the per-building jitter (mean 1) left out."""
    t = np.clip((rank - EMPTY_RANK) / (SPRAWL_RANK_FULL - EMPTY_RANK), 0, 1)
    sprawl = SPRAWL_MIN_FRAC + (1 - SPRAWL_MIN_FRAC) * t * t * (3 - 2 * t)
    h = (H_BASE + H_SCALE * np.maximum(density, 0.0) ** H_EXP) * sprawl
    return np.where(rank > EMPTY_RANK, h, 0.0)


def _is_avenue(k):
    return (k % AVENUE_PERIOD) == 0


class _Lots:
    """Lot rectangles over the torus, one id per cell (-1 = unclaimed)."""

    def __init__(self, n):
        self.n = n
        self.owner = -np.ones((n, n), dtype=np.int64)   # lot id per cell
        self.rects = []     # (x0, y0, w, h, owner_x, owner_y), unwrapped x0/y0

    def free(self, x0, y0, w, h):
        idx = np.ix_((np.arange(y0, y0 + h)) % self.n,
                     (np.arange(x0, x0 + w)) % self.n)
        return bool(np.all(self.owner[idx] < 0))

    def claim(self, x0, y0, w, h, ox, oy):
        lid = len(self.rects)
        idx = np.ix_((np.arange(y0, y0 + h)) % self.n,
                     (np.arange(x0, x0 + w)) % self.n)
        self.owner[idx] = lid
        self.rects.append((x0, y0, w, h, ox, oy))
        return lid


def _no_interior_avenue(x0, w):
    """No avenue line strictly inside [x0, x0 + w)."""
    return not any(_is_avenue(k) for k in range(x0 + 1, x0 + w))


def bake_lots(density: np.ndarray, rank: np.ndarray, seed: int = 20260923,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return (lot_word, street_word, stats); words are uint16 (N, N) [y, x].

    Arrays are indexed [y, x], the layout the tile is uploaded in (texel
    (x, y) of textureLoad is element [y, x]).
    """
    n = density.shape[0]
    assert density.shape == (n, n) and rank.shape == (n, n)
    rng = np.random.default_rng(seed)
    built = rank > EMPTY_RANK
    h0 = potential_height(density, rank)

    district = ndimage.gaussian_filter(rank.astype(np.float64),
                                       DISTRICT_SIGMA_CELLS, mode="wrap")
    urban_district = district >= DISTRICT_URBAN_RANK

    use = np.full((n, n), USE_OPEN, dtype=np.int64)
    use[built & urban_district] = USE_URBAN
    use[built & ~urban_district] = USE_INDUSTRIAL
    # Unbuilt: a yard where it touches development, open scrub where not.
    # Inside an urban district every vacant block fronts the grid anyway.
    dev = built.astype(np.int64)
    touch = (np.roll(dev, 1, 0) | np.roll(dev, -1, 0)
             | np.roll(dev, 1, 1) | np.roll(dev, -1, 1)).astype(bool)
    use[~built & (touch | urban_district)] = USE_YARD

    lots = _Lots(n)

    # 1. Tower lots, tallest first.
    cand = np.argwhere((use == USE_URBAN) & (h0 >= TOWER_LOT_H[0][1]))
    order = np.argsort(-h0[cand[:, 0], cand[:, 1]])
    for y, x in cand[order]:
        if lots.owner[y, x] >= 0:
            continue
        k = max(kk for kk, hh in TOWER_LOT_H if h0[y, x] >= hh)
        placed = False
        while k >= 2 and not placed:
            # Offsets that keep the peak as central as the parity allows.
            offs = sorted(range(k), key=lambda o: abs(o - (k - 1) / 2))
            for oy in offs:
                for ox in offs:
                    x0, y0 = x - ox, y - oy
                    if not (_no_interior_avenue(x0, k)
                            and _no_interior_avenue(y0, k)):
                        continue
                    xs = np.arange(x0, x0 + k) % n
                    ys = np.arange(y0, y0 + k) % n
                    if not np.all(urban_district[np.ix_(ys, xs)]):
                        continue
                    if not lots.free(x0, y0, k, k):
                        continue
                    # Never swallow a rival: a block that would raise a
                    # comparable tower of its own keeps it. The tile's
                    # tallest peaks stand on adjacent blocks, and one lot
                    # over them turned the skyline's cluster of needles
                    # into a single tower.
                    sub = h0[np.ix_(ys, xs)].copy()
                    sub[oy, ox] = 0.0
                    if np.any(sub >= max(TOWER_LOT_H[0][1],
                                         RIVAL_FRAC * h0[y, x])):
                        continue
                    lots.claim(x0, y0, k, k, ox, oy)
                    placed = True
                    break
                if placed:
                    break
            k -= 1

    # 2. Mid-rise superblocks: an aligned 2x2 of urban blocks, sometimes.
    for y0 in range(0, n, 2):
        for x0 in range(0, n, 2):
            ys, xs = [y0, y0 + 1], [x0, x0 + 1]
            if not np.all(use[np.ix_(ys, xs)] == USE_URBAN):
                continue
            if rank[y0, x0] <= MIDRISE_MERGE_RANK:
                continue
            if rng.random() >= MIDRISE_MERGE_FRAC or not lots.free(x0, y0, 2, 2):
                continue
            sub = h0[np.ix_(ys, xs)]
            # Mid-rise only: a block tall enough for a tower lot of its own
            # is never folded into a superblock (the peaks stage 1 leaves
            # free for the rival rule would otherwise merge here). Tested
            # after the draw, so every other superblock is unchanged.
            if sub.max() >= TOWER_LOT_H[0][1]:
                continue
            oy, ox = np.unravel_index(np.argmax(sub), sub.shape)
            lots.claim(x0, y0, 2, 2, int(ox), int(oy))

    # 3. The industrial belt: aligned 2x2 superblocks (the half-density
    # street lattice runs between them). A superblock with three or four
    # built blocks is one big warehouse lot; two built blocks side by side
    # are a 2x1 lot; the rest stay single.
    for y0 in range(0, n, 2):
        for x0 in range(0, n, 2):
            ys, xs = [y0, y0 + 1], [x0, x0 + 1]
            u = use[np.ix_(ys, xs)]
            if np.any(u == USE_URBAN) or not lots.free(x0, y0, 2, 2):
                continue
            ind = (u == USE_INDUSTRIAL)
            if ind.sum() >= 3:
                sub = np.where(ind, h0[np.ix_(ys, xs)], -1.0)
                oy, ox = np.unravel_index(np.argmax(sub), sub.shape)
                lots.claim(x0, y0, 2, 2, int(ox), int(oy))
                use[np.ix_(ys, xs)] = USE_INDUSTRIAL
            elif ind.sum() == 2:
                if ind[0].all():
                    lots.claim(x0, y0, 2, 1, int(np.argmax(h0[y0, xs])), 0)
                elif ind[1].all():
                    lots.claim(x0, y0 + 1, 2, 1,
                               int(np.argmax(h0[y0 + 1, xs])), 0)
                elif ind[:, 0].all():
                    lots.claim(x0, y0, 1, 2, 0, int(np.argmax(h0[ys, x0])))
                elif ind[:, 1].all():
                    lots.claim(x0 + 1, y0, 1, 2, 0,
                               int(np.argmax(h0[ys, x0 + 1])))

    # Everything left is a one-block lot.
    for y, x in np.argwhere(lots.owner < 0):
        lots.claim(int(x), int(y), 1, 1, 0, 0)

    # A lot has one use, its owner's: a tower lot swallows the vacant block
    # beside its peak, a warehouse lot its truck court.
    rects_a = np.array(lots.rects, dtype=np.int64)
    ox = (rects_a[:, 0] + rects_a[:, 4]) % n
    oy = (rects_a[:, 1] + rects_a[:, 5]) % n
    use = use[oy, ox][lots.owner]

    # Per-cell lot geometry.
    lot_id = lots.owner
    rects = np.array(lots.rects, dtype=np.int64)
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    r = rects[lot_id]
    back_x = (X - r[..., 0]) % n
    back_y = (Y - r[..., 1]) % n
    assert back_x.max() <= 3 and back_y.max() <= 3
    assert np.all(back_x < r[..., 2]) and np.all(back_y < r[..., 3])

    # 4. Streets. An edge is a street when it is not inside a lot and it
    # serves something: in urban districts every edge that touches a
    # non-open block; in the fringe only the half-density lattice lines.
    # Vertical edge (x-line) k between cells (k-1, y) and (k, y):
    lot_w = lot_id
    lot_e = np.roll(lot_id, 1, axis=1)          # cell (x-1, y) seen from x
    use_l = np.roll(use, 1, axis=1)
    urb_l = np.roll(urban_district, 1, axis=1)
    serves = (use != USE_OPEN) | (use_l != USE_OPEN)
    xline_lattice = (X % INDUSTRIAL_STREET_PERIOD) == 0
    xline = (lot_w != lot_e) & serves & (
        urban_district | urb_l | xline_lattice)
    # Horizontal edge (y-line) k between cells (x, k-1) and (x, k):
    lot_s = np.roll(lot_id, 1, axis=0)
    use_s = np.roll(use, 1, axis=0)
    urb_s = np.roll(urban_district, 1, axis=0)
    serves = (use != USE_OPEN) | (use_s != USE_OPEN)
    yline_lattice = (Y % INDUSTRIAL_STREET_PERIOD) == 0
    yline = (lot_id != lot_s) & serves & (
        urban_district | urb_s | yline_lattice)

    xline, yline = _connect(xline, yline, lot_id, use, verbose)

    # 5. Pack.
    e_w = xline
    e_e = np.roll(xline, -1, axis=1)
    e_s = yline
    e_n = np.roll(yline, -1, axis=0)
    # A corner square is paved when any of the four edges meeting there is.
    # Corner (x, y) — the grid node at the cell's min corner — meets the
    # x-line segments of cells (x, y-1) and (x, y), and the y-line segments
    # of cells (x-1, y) and (x, y).
    node = (xline | np.roll(xline, 1, axis=0)
            | yline | np.roll(yline, 1, axis=1))
    c_sw = node
    c_se = np.roll(node, -1, axis=1)
    c_nw = np.roll(node, -1, axis=0)
    c_ne = np.roll(np.roll(node, -1, axis=0), -1, axis=1)

    lot_word = (back_x | (back_y << 2) | ((r[..., 2] - 1) << 4)
                | ((r[..., 3] - 1) << 6) | (r[..., 4] << 8) | (r[..., 5] << 10))
    street_word = (e_w.astype(np.int64) | (e_e << 1) | (e_s << 2) | (e_n << 3)
                   | (c_sw << 4) | (c_se << 5) | (c_nw << 6) | (c_ne << 7)
                   | (use << 8))
    assert lot_word.max() < 4096 and street_word.max() < 4096

    stats = {
        "urban": float(np.mean(use == USE_URBAN)),
        "industrial": float(np.mean(use == USE_INDUSTRIAL)),
        "yard": float(np.mean(use == USE_YARD)),
        "open": float(np.mean(use == USE_OPEN)),
        "lots_by_size": {f"{w}x{h}": int(np.sum((rects[:, 2] == w)
                                                & (rects[:, 3] == h)))
                         for w in (1, 2, 3, 4) for h in (1, 2, 3, 4)
                         if np.any((rects[:, 2] == w) & (rects[:, 3] == h))},
        "street_edges": float((xline.sum() + yline.sum()) / (2 * n * n)),
    }
    if verbose:
        print(f"  lots: {stats}")
    return lot_word.astype(np.uint16), street_word.astype(np.uint16), stats


def _connect(xline, yline, lot_id, use, verbose):
    """Join every street fragment to the main network by the shortest road.

    Nodes are grid corners; a street edge joins two corners. Fragments that
    do not reach the largest component get the cheapest path to it along
    grid lines that are not inside a lot and not flanked by OPEN land on
    both sides — no, that would forbid the access road entirely; a road
    through scrub is exactly what an isolated works has. So: any grid-line
    segment not inside a lot may carry the new road, at cost 1 per segment,
    and existing streets cost nothing.
    """
    n = xline.shape[0]
    nid = lambda x, y: (y % n) * n + (x % n)
    Y, X = np.mgrid[0:n, 0:n]
    # x-line segment of cell (x, y): node (x, y) to node (x, y+1).
    xa, xb = nid(X, Y), nid(X, Y + 1)
    # y-line segment of cell (x, y): node (x, y) to node (x+1, y).
    ya, yb = nid(X, Y), nid(X + 1, Y)
    x_open = lot_id != np.roll(lot_id, 1, axis=1)   # not inside a lot
    y_open = lot_id != np.roll(lot_id, 1, axis=0)

    def graph(mask_x, mask_y, w_x, w_y):
        rows = np.concatenate([xa[mask_x], ya[mask_y], xb[mask_x], yb[mask_y]])
        cols = np.concatenate([xb[mask_x], yb[mask_y], xa[mask_x], ya[mask_y]])
        w = np.concatenate([w_x[mask_x], w_y[mask_y], w_x[mask_x], w_y[mask_y]])
        return coo_matrix((w, (rows, cols)), shape=(n * n, n * n)).tocsr()

    streets = graph(xline, yline, np.ones_like(xline, float),
                    np.ones_like(yline, float))
    ncomp, label = connected_components(streets, directed=False)
    touched = np.zeros(n * n, bool)
    touched[np.concatenate([xa[xline], xb[xline], ya[yline], yb[yline]])] = True
    sizes = np.bincount(label[touched], minlength=ncomp)
    main = int(np.argmax(sizes))
    stranded = [c for c in range(ncomp) if sizes[c] > 0 and c != main]
    if verbose:
        print(f"  street components: {len(stranded) + 1}, main has "
              f"{sizes[main]} nodes; connecting {len(stranded)} fragments")
    if not stranded:
        return xline, yline

    # Multi-source shortest paths from the main network, over every open
    # grid segment; existing streets are (nearly) free.
    eps = 1e-3
    w_x = np.where(xline, eps, 1.0)
    w_y = np.where(yline, eps, 1.0)
    g = graph(x_open, y_open, w_x, w_y)
    # A super-source: dijkstra from every main-network node at once.
    main_nodes = np.flatnonzero(touched & (label == main))
    dist, pred, src = dijkstra(g, directed=False, indices=main_nodes,
                               min_only=True, return_predecessors=True)
    xline = xline.copy()
    yline = yline.copy()
    for c in stranded:
        nodes = np.flatnonzero(touched & (label == c))
        start = nodes[np.argmin(dist[nodes])]
        if not np.isfinite(dist[start]):
            continue
        cur = start
        while pred[cur] >= 0:
            p = pred[cur]
            ax, ay = cur % n, cur // n
            bx, by = p % n, p // n
            if ax == bx:          # an x-line segment
                y = ay if (by - ay) % n == 1 else by
                xline[y, ax] = True
            else:                 # a y-line segment
                x = ax if (bx - ax) % n == 1 else bx
                yline[ay, x] = True
            cur = p
    return xline, yline


def pack_channels(lot_word: np.ndarray, street_word: np.ndarray):
    """The two words as the float values the tile stores in b and a."""
    b = lot_word.astype(np.float32) - WORD_OFFSET
    a = street_word.astype(np.float32) - WORD_OFFSET
    assert np.array_equal(b.astype(np.float16).astype(np.float32), b)
    assert np.array_equal(a.astype(np.float16).astype(np.float32), a)
    return b, a
