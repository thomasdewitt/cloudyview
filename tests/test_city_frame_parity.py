"""The city coordinate is one mapping, written down in three places.

soar shows a camera's position in the city tile's frame (the minimap caption
and the F3 overlay), records it in a still's metadata, and `night_city_harness
--city-camera` turns it back into world metres. All three have to be the SAME
map as the one the shader samples the tile with, or the readout names a street
the picture does not show.

The shader's map is one line of raymarch.wgsl:

    let uv = (xy - u.ocean.yz) / u.ocean_params.y;

with a repeating sampler, so only the fractional part of uv picks a block.
That line is pinned here as text — it is the definition, and a change to it
has to come here and break this test rather than silently desynchronise the
caption. The JS forward map and the harness's inverse are then checked
against it.

Skips when node is unavailable. Needs no GPU: this is arithmetic.
"""

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCENE_JS = REPO / "web" / "soar" / "scene.js"
WGSL = REPO / "cloudyview" / "soar" / "raymarch.wgsl"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not SCENE_JS.exists(),
    reason="needs node and web/soar/scene.js")

# One tile and one offset, neither of them a round multiple of the other, so a
# fold that is off by a tile or that leaks a sign shows up as a wrong number
# rather than a coincidence.
OFFSET = (8100.0, -44190.0)
EXTENT = 46080.0


def test_shader_samples_the_tile_at_xy_minus_offset_over_extent():
    """The definition, as it stands in the shader."""
    source = WGSL.read_text()
    body = re.search(r"fn city_glow_sample\(.*?\n\}", source, re.S)
    assert body is not None, "city_glow_sample is gone from raymarch.wgsl"
    assert re.search(r"let\s+uv\s*=\s*\(xy\s*-\s*u\.ocean\.yz\)\s*/\s*"
                     r"u\.ocean_params\.y\s*;", body.group(0)), (
        "the city tile's world->tile map changed; scene.js cityFramePosition, "
        "the minimap caption, the capture metadata and night_city_harness "
        "--city-camera all have to change with it")


def _city_frame(points):
    js = textwrap.dedent(f"""
        import {{ cityFramePosition }} from "{SCENE_JS.as_uri()}";
        const pts = {json.dumps(points)};
        process.stdout.write(JSON.stringify(pts.map(
          (p) => cityFramePosition(p, {list(OFFSET)}, {EXTENT}))));
    """)
    out = subprocess.run(["node", "--input-type=module", "--eval", js],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_city_frame_is_the_offset_removed_and_folded_into_the_tile():
    points = [
        [OFFSET[0], OFFSET[1], 12.0],                    # the tile origin
        [OFFSET[0] + 123.0, OFFSET[1] + 45.0, 12.0],
        [OFFSET[0] - 1.0, OFFSET[1] - 1.0, 12.0],        # wraps to the far edge
        [0.0, 0.0, 5400.0],
    ]
    got = _city_frame(points)
    for point, city in zip(points, got):
        for i in range(2):
            assert 0.0 <= city[i] < EXTENT
            # Same block as the world point: the difference is whole tiles.
            tiles = (point[i] - OFFSET[i] - city[i]) / EXTENT
            assert abs(tiles - round(tiles)) < 1e-9
        assert city[2] == point[2], "z is world z; the tile is 2D"


def test_a_whole_tile_of_travel_is_the_same_place():
    """What makes the number worth showing: it is bounded and it is stable."""
    base = [OFFSET[0] + 700.0, OFFSET[1] + 30.0, 60.0]
    shifted = [base[0] + 3 * EXTENT, base[1] - 7 * EXTENT, 60.0]
    a, b = _city_frame([base, shifted])
    assert a == pytest.approx(b, abs=1e-6)


def test_harness_city_camera_inverts_the_forward_map():
    """--city-camera CX CY is world (CX + offset_x, CY + offset_y)."""
    source = (REPO / "tools" / "night_city_harness.py").read_text()
    assert "--city-camera" in source
    assert re.search(r"x\s*=\s*cx\s*\+\s*offset\[0\]", source)
    assert re.search(r"y\s*=\s*cy\s*\+\s*offset\[1\]", source)
    city = _city_frame([[1234.0 + OFFSET[0], 99.0 + OFFSET[1], 20.0]])[0]
    assert city[0] == pytest.approx(1234.0)
    assert city[1] == pytest.approx(99.0)
