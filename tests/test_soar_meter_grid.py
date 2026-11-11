"""The auto-exposure meter's pixels are square, whatever the window's shape.

The meter marches a tiny linear render of the live view and reads its
highlight statistic. The shader takes the frame's aspect from the OUTPUT
size (row 1.w) and its pixel angle from the grid's width, so a meter grid
of any other shape than the window's has oblong pixels: the fixed 64x36 did
everywhere but 16:9, and in a portrait window each meter pixel was ~3x
taller than wide. meterSize (uniforms.js) now
gives the grid the output's shape at about AUTO_EXPOSURE_METER_RAYS rays,
and tools/prebake_demos.py's meter (which metered 96 wide) uses the same
rule, pinned here to the browser's.

The same fixture also pins prebake's MAX_TIER_HAZE, its restatement of
constants.js DEFAULT_HAZE_BY_TIER.max, which every demo still is baked at.

Needs node, not a GPU.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"
UNIFORMS_JS = REPO / "web" / "soar" / "uniforms.js"
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"

# Window shapes: 16:9 desktop and capture, 16:10 laptop, 4:3, square,
# portrait phone, and a sliver.
OUTPUTS = [(2560, 1440), (1920, 1080), (1440, 900), (1024, 768), (800, 800),
           (1080, 1920), (390, 844), (3000, 400)]

_JS = textwrap.dedent("""
    import * as K from "%s";
    import { meterSize } from "%s";
    const sizes = JSON.parse(process.argv[1]);
    console.log(JSON.stringify({ rays: K.AUTO_EXPOSURE_METER_RAYS,
                                 maxHaze: K.DEFAULT_HAZE_BY_TIER.max,
                                 grids: sizes.map((s) => meterSize(s)) }));
""") % (CONSTANTS_JS.as_uri(), UNIFORMS_JS.as_uri())


@pytest.fixture(scope="module")
def grids():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _JS,
                          json.dumps(OUTPUTS)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_the_16_9_meter_is_unchanged(grids):
    assert grids["grids"][0] == [64, 36]
    assert grids["rays"] == 64 * 36


@pytest.mark.parametrize("k", range(len(OUTPUTS)),
                         ids=[f"{w}x{h}" for w, h in OUTPUTS])
def test_meter_pixels_are_square(grids, k):
    (w, h), (mw, mh) = OUTPUTS[k], grids["grids"][k]
    # Across, a pixel is 2 tan(hfov/2) / mw; up, the frame's 2 tan(hfov/2) /
    # aspect over mh rows. Square to the grid's rounding: half a pixel on
    # either axis.
    oblong = (w / h) * mh / mw
    assert abs(oblong - 1.0) <= 0.5 / mw + 0.5 / mh + 1e-9, (mw, mh, oblong)
    # About the same number of rays at any shape.
    assert abs(mw * mh / grids["rays"] - 1.0) < 0.1, (mw, mh)


def test_the_meter_is_sized_by_meter_size():
    """The renderer's meter grid comes from meterSize (and its readback rows
    are padded, so any width copies), not from a fixed size."""
    src = RENDERER_JS.read_text()
    assert "const [mw, mh] = meterSize(outputSize);" in src
    assert "AUTO_EXPOSURE_METER_SIZE" not in src
    assert "bytesPerRow: this._meterRowBytes" in src


def test_the_demo_bake_meters_on_the_browsers_grid(grids):
    sys.path.insert(0, str(REPO / "tools"))
    import prebake_demos as pb
    assert pb.AE_METER_RAYS == grids["rays"]
    assert [list(pb.meter_size(s)) for s in OUTPUTS] == grids["grids"]


def test_the_demo_bake_restates_the_max_tiers_haze(grids):
    sys.path.insert(0, str(REPO / "tools"))
    import prebake_demos as pb
    assert pb.MAX_TIER_HAZE == grids["maxHaze"]
