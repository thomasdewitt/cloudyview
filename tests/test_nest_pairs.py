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

The rule lives in web/soar/field.js (nestOverhang / nestablePairs), which is
the only copy: the Python port in io.py served an interactive chooser that
moved into the browser, and was deleted with it. These tests drive field.js
under node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

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

# One node run answers every question below: the offered pairs, the
# middle level's overhang against its parent, the same nest shifted by a
# wrong origin, and a fine grid's allowance.
_JS = textwrap.dedent("""
    import { nestOverhang, nestablePairs, NEST_OVERHANG_FRACTION }
      from "%s";
    const levels = %s;
    const [pMin, pMax, pDx] = levels.parent;
    const [aMin, aMax] = levels.nest_a;
    const shift = (v) => v.map((x) => x - 500000.0);
    const fineMin = [0, 0, 0], fineMax = [20480.0, 10240.0, 4000.0];
    const fineDx = [10.0, 10.0, 10.0];
    process.stdout.write(JSON.stringify({
      fraction: NEST_OVERHANG_FRACTION,
      pairs: nestablePairs(
        Object.entries(levels).map(([name, [bmin, bmax, spacing]]) =>
          ({ name, bmin, bmax, spacing }))),
      middle: nestOverhang(pMin, pMax, aMin, aMax, pDx),
      shifted: nestOverhang(pMin, pMax, shift(aMin), shift(aMax), pDx),
      fine: nestOverhang(fineMin, fineMax,
                         fineMin.map((v) => v - 300.0), fineMax, fineDx),
    }));
""") % (FIELD_JS.as_posix(), json.dumps(LEVELS))


@pytest.fixture(scope="module")
def verdicts():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_three_levels_offer_all_three_pairs(verdicts):
    assert [tuple(p) for p in verdicts["pairs"]] == [
        ("parent", "nest_a"), ("parent", "nest_b"), ("nest_a", "nest_b")]


def test_the_middle_level_overhangs_and_is_clipped_not_refused(verdicts):
    """The specific rejection: half a parent cell below the parent's floor."""
    overhang, allowance = verdicts["middle"]["overhang"], verdicts["middle"]["allowance"]
    o_dx_z = LEVELS["parent"][2][2]
    span_z = LEVELS["parent"][1][2] - LEVELS["parent"][0][2]
    assert overhang[2] == pytest.approx(2569.4, abs=0.1)   # a real overhang
    assert allowance[2] == pytest.approx(o_dx_z)           # one parent cell
    assert overhang[2] < allowance[2]
    # The old fraction-only allowance is what refused it.
    assert overhang[2] > verdicts["fraction"] * span_z


def test_a_wrong_origin_is_still_refused(verdicts):
    """The tolerance must not have become 'anything goes'."""
    overhang, allowance = verdicts["shifted"]["overhang"], verdicts["shifted"]["allowance"]
    assert any(o > a for o, a in zip(overhang, allowance))


def test_a_fine_grid_keeps_a_tight_allowance(verdicts):
    """One cell of a 10 m grid is 10 m, not a licence to miss by kilometres."""
    overhang, allowance = verdicts["fine"]["overhang"], verdicts["fine"]["allowance"]
    # 1% of the span still governs here, and 300 m clears it on z.
    assert allowance[2] == pytest.approx(40.0)
    assert overhang[2] > allowance[2]
