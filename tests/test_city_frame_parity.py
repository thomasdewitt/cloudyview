"""The city coordinate is one mapping, written down in three places.

Over a night city, soar draws the minimap in the city tile's frame (and out of
the tile's own cascade), shows the camera's tile position in the F3 overlay,
records it in a still's metadata, and `night_city_harness --city-camera` turns
it back into world metres. All of them have to be the SAME map as the one the
shader samples the tile with, or the marker sits on a district the picture
does not show.

The shader's map is one line of raymarch.wgsl:

    let uv = (xy - u.ocean.yz) / u.ocean_params.y;

with a repeating sampler, so only the fractional part of uv picks a block.
That line is pinned here as text — it is the definition, and a change to it
has to come here and break this test rather than silently desynchronise the
map. The JS forward map, the minimap's city image, and the harness's inverse
are then checked against it.

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
MINIMAP_JS = REPO / "web" / "soar" / "minimap.js"
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
        "the minimap's city image, the capture metadata and "
        "night_city_harness "
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


# --- the city minimap ------------------------------------------------------
#
# Over a night city the minimap draws the city tile instead of the cloud
# field, out of the same mip-0 cascade the CITY shader raises buildings from.
# Its arithmetic runs on the CPU in JavaScript and the shader's runs on the
# GPU in WGSL, so the constants are written down twice; these pin the second
# copy to the first.

# (JS name in minimap.js, WGSL name in raymarch.wgsl)
_CITY_MAP_CONSTANTS = [
    ("CITY_MAP_EMPTY_RANK", "CITY_EMPTY_RANK"),
    ("CITY_MAP_SPRAWL_RANK_FULL", "CITY_SPRAWL_RANK_FULL"),
    ("CITY_MAP_SPRAWL_MIN_FRAC", "CITY_SPRAWL_MIN_FRAC"),
    ("CITY_MAP_H_BASE", "CITY_H_BASE"),
    ("CITY_MAP_H_SCALE", "CITY_H_SCALE"),
    ("CITY_MAP_H_EXP", "CITY_H_EXP"),
]


def _number(source, pattern):
    m = re.search(pattern, source)
    assert m, f"no match for {pattern!r}"
    return float(m.group(1))


@pytest.mark.parametrize("js_name,wgsl_name", _CITY_MAP_CONSTANTS)
def test_the_map_raises_its_buildings_on_the_shaders_numbers(js_name, wgsl_name):
    js = _number(MINIMAP_JS.read_text(),
                 rf"const\s+{js_name}\s*=\s*([-\d.eE+]+)\s*;")
    wgsl = _number(WGSL.read_text(),
                   rf"const\s+{wgsl_name}\s*:\s*f32\s*=\s*([-\d.eE+]+)\s*;")
    assert js == wgsl, (
        f"minimap.js {js_name} is {js} and raymarch.wgsl {wgsl_name} is "
        f"{wgsl}; the map would show a city the renderer does not build")


def _height(density, rank):
    """city_cell's height with the per-building jitter (mean 1) left out."""
    empty, full, floor_ = 0.22, 0.60, 0.15
    if not rank > empty:
        return 0.0
    t = min(max((rank - empty) / (full - empty), 0.0), 1.0)
    sprawl = floor_ + (1.0 - floor_) * (t * t * (3.0 - 2.0 * t))
    return (14.0 + 390.0 * density ** 1.2) * sprawl


def test_the_map_height_is_the_shaders_height_without_the_jitter():
    cells = [(0.02, 0.10), (0.05, 0.30), (0.20, 0.55), (0.60, 0.95),
             (0.90, 0.99), (0.40, 0.22)]
    js = textwrap.dedent(f"""
        import {{ cityBlockHeights }} from "{MINIMAP_JS.as_uri()}";
        const cells = {json.dumps(cells)};
        const n = 1;
        const out = cells.map(([d, r]) => cityBlockHeights({{
          n, density: Float32Array.of(d), rank: Float32Array.of(r) }})[0]);
        process.stdout.write(JSON.stringify(out));
    """)
    got = json.loads(subprocess.run(["node", "--input-type=module", "--eval", js],
                                    capture_output=True, text=True,
                                    check=True).stdout)
    for (density, rank), height in zip(cells, got):
        assert height == pytest.approx(_height(density, rank), rel=1e-6)
    # The rank-0.22 case is exactly the empty threshold: not built, no height.
    assert got[-1] == 0.0


def test_the_marker_is_placed_in_the_frame_the_map_is_drawn_in():
    """Tile origin at the map's lower-left, tile centre at its middle."""
    js = textwrap.dedent(f"""
        import {{ cityRelativePosition }} from "{MINIMAP_JS.as_uri()}";
        import {{ cityFramePosition }} from "{SCENE_JS.as_uri()}";
        const scene = {{
          city: true,
          cityOffsetM: {list(OFFSET)},
          oceanTileExtent: {EXTENT},
          cityPosition(p) {{
            return cityFramePosition(p, this.cityOffsetM, this.oceanTileExtent);
          }},
        }};
        const at = (x, y) => cityRelativePosition(scene, [x, y, 30.0]);
        process.stdout.write(JSON.stringify([
          at({OFFSET[0]}, {OFFSET[1]}),
          at({OFFSET[0]} + {EXTENT} / 2, {OFFSET[1]} + {EXTENT} / 2),
          at({OFFSET[0]} + {EXTENT} * 3.25, {OFFSET[1]} - {EXTENT} * 0.25),
        ]));
    """)
    origin, centre, wrapped = json.loads(
        subprocess.run(["node", "--input-type=module", "--eval", js],
                       capture_output=True, text=True, check=True).stdout)
    assert origin[:2] == pytest.approx([-1.0, -1.0])
    assert centre[:2] == pytest.approx([0.0, 0.0])
    # Whole tiles away is the same spot on the map — the map is one tile.
    assert wrapped[:2] == pytest.approx([-0.5, 0.5])
