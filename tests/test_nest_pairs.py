"""Which levels of a multi-level file may be rendered together.

A file with three levels should offer all three of its pairs — coarse+middle,
coarse+fine, middle+fine — and one did not: only its finest pair came back
(Thomas, 2026-08-14, on a turbulon `sq1km` run with parent + nest_a + nest_b).

The cause was the containment tolerance. Both boxes are built from cell EDGES,
so a nest can sit proud of its parent for no reason but the grid, and the
allowance for that was 1% of the parent's span. That stands in for "a cell or
so" only while the parent HAS many cells. A turbulon parent is three cells
tall over 15 km: 1% of the span is 149 m, one cell is 4.9 km, and a middle
level reaching half a parent cell below the parent's floor was therefore
refused as a coordinate error rather than clipped. The allowance is now the
fraction or one parent cell, whichever is larger.

The numbers below are that file's, rounded — a coarse three-cell parent, a
middle level that overhangs it vertically, and a fine level inside the middle.
The file itself is 323 MB and lives outside this repo, so the geometry is
reconstructed rather than read.

Both languages are checked: cloudyview.io decides for witness and behold,
web/soar/field.js decides for the browser, and a file that nests in one and
not the other is the same bug wearing different clothes.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview import io

REPO = Path(__file__).resolve().parents[1]
FIELD_JS = REPO / "web" / "soar" / "field.js"

# parent: 2048 km square, 3 cells tall between 2.56 and 17.49 km.
# nest_a: 32 km square inside it, but spanning the ground to 20 km — so it
#         overhangs the parent's floor and ceiling by ~2.5 km.
# nest_b: 8 km square well inside nest_a, and inside it vertically too.
LEVELS = {
    "parent": ([-500.0, -500.0, 2559.24],
               [2047500.0, 2047500.0, 17488.15],
               [1000.0, 1000.0, 4928.91]),
    "nest_a": ([1007968.75, 1007968.75, -10.16],
               [1039968.75, 1039968.75, 19989.84],
               [62.5, 62.5, 20.324]),
    "nest_b": ([1019996.09, 1019996.09, 1003.16],
               [1027996.09, 1027996.09, 4996.84],
               [7.812, 7.812, 6.4]),
}


def pairs_python():
    """io.find_nestable_group_pairs' rule, on geometry rather than a file."""
    out = []
    for outer, (o_min, o_max, o_dx) in LEVELS.items():
        for inner, (i_min, i_max, i_dx) in LEVELS.items():
            if inner == outer:
                continue
            o_min, o_max, o_dx = map(np.asarray, (o_min, o_max, o_dx))
            i_min, i_max, i_dx = map(np.asarray, (i_min, i_max, i_dx))
            if np.any(i_dx > o_dx) or not np.any(i_dx < o_dx):
                continue
            overhang, allowance = io.nest_overhang(
                o_min, o_max, i_min, i_max, o_dx)
            if np.any(overhang > allowance):
                continue
            tol = 1e-9 * np.maximum(o_max - o_min, 1.0)
            if np.all(i_min <= o_min + tol) and np.all(i_max >= o_max - tol):
                continue
            out.append((outer, inner))
    return out


def test_three_levels_offer_all_three_pairs():
    assert pairs_python() == [
        ("parent", "nest_a"), ("parent", "nest_b"), ("nest_a", "nest_b")]


def test_the_middle_level_overhangs_and_is_clipped_not_refused():
    """The specific rejection: half a parent cell below the parent's floor."""
    o_min, o_max, o_dx = (np.asarray(v) for v in LEVELS["parent"])
    i_min, i_max, _ = (np.asarray(v) for v in LEVELS["nest_a"])
    overhang, allowance = io.nest_overhang(o_min, o_max, i_min, i_max, o_dx)
    assert overhang[2] == pytest.approx(2569.4, abs=0.1)   # a real overhang
    assert allowance[2] == pytest.approx(o_dx[2])          # one parent cell
    assert overhang[2] < allowance[2]
    # The old fraction-only allowance is what refused it.
    assert overhang[2] > io.NEST_OVERHANG_FRACTION * (o_max[2] - o_min[2])


def test_a_wrong_origin_is_still_refused():
    """The tolerance must not have become 'anything goes'."""
    o_min, o_max, o_dx = (np.asarray(v) for v in LEVELS["parent"])
    i_min, i_max, _ = (np.asarray(v) for v in LEVELS["nest_a"])
    overhang, allowance = io.nest_overhang(
        o_min, o_max, i_min - 500_000.0, i_max - 500_000.0, o_dx)
    assert np.any(overhang > allowance)


def test_a_fine_grid_keeps_a_tight_allowance():
    """One cell of a 10 m grid is 10 m, not a licence to miss by kilometres."""
    o_min = np.zeros(3)
    o_max = np.array([20480.0, 10240.0, 4000.0])
    o_dx = np.array([10.0, 10.0, 10.0])
    overhang, allowance = io.nest_overhang(
        o_min, o_max, o_min - 300.0, o_max, o_dx)
    # 1% of the span still governs here, and 300 m clears it on z.
    assert allowance[2] == pytest.approx(40.0)
    assert overhang[2] > allowance[2]


_JS = textwrap.dedent("""
    import { nestablePairs } from "%s";
    const levels = %s;
    process.stdout.write(JSON.stringify(nestablePairs(
      Object.entries(levels).map(([name, [bmin, bmax, spacing]]) =>
        ({ name, bmin, bmax, spacing })))));
""") % (FIELD_JS.as_posix(), json.dumps(LEVELS))


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_agrees_with_python():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    assert [tuple(p) for p in json.loads(proc.stdout)] == pairs_python()
