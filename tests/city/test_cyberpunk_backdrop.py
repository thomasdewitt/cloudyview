"""Cyberpunk's landing backdrop is baked from the app's own numbers.

tools/prebake_demos.py renders web/soar/main.js's CYBERPUNK_BACKDROP, the view
a cyberpunk flight opens on, as a max-tier capture parked there. Python cannot
import the ES modules those numbers live in, so CYBERPUNK restates them; this
file holds each copy to its original. A backdrop baked off a stale copy would
show a street the flight does not open on, under a moon it does not fly under,
and nothing else would notice.

Needs node for the constants.js half, not a GPU.
"""

import json
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import REPO

CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"
MAIN_JS = REPO / "web" / "soar" / "main.js"


@pytest.fixture(scope="module")
def pb():
    sys.path.insert(0, str(REPO / "tools"))
    import prebake_demos
    return prebake_demos


@pytest.fixture(scope="module")
def app():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    js = textwrap.dedent(f"""
        import * as K from "{CONSTANTS_JS.as_uri()}";
        console.log(JSON.stringify({{
          offset: K.CITY_TILE_OFFSET_M,
          moon: K.CITY_MOON,
          start: K.CITY_START_CAMERA,
          parkedLod: K.CITY_PARKED_LOD_STRENGTH,
          maxRoofs: K.QUALITY_PRESETS.max.cityRoofDensity,
          maxHaze: K.QUALITY_PRESETS.max.haze,
        }}));
    """)
    out = subprocess.run(["node", "--input-type=module", "-e", js],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_the_city_stands_where_cyberpunk_stands_it(pb, app):
    assert list(pb.CYBERPUNK["city_offset"]) == app["offset"]


def test_the_moon_is_cyberpunks(pb, app):
    assert pb.CYBERPUNK["sun"] == app["moon"]


def test_the_camera_is_the_start_camera(pb, app):
    still = pb.CYBERPUNK["still"]
    assert list(pb.CYBERPUNK["camera_city_m"]) == app["start"]["positionCityM"]
    for key in ("azimuth", "elevation", "fov"):
        assert still[key] == app["start"][key], key


def test_the_look_is_a_parked_max_capture(pb, app):
    """A named-tier city capture marches CITY_PARKED_LOD_STRENGTH, the
    tier's haze and roofs, and meters its exposure (capturing.js)."""
    assert pb.CYBERPUNK["still"]["lod"] == app["parkedLod"]
    assert pb.CYBERPUNK["city_roof_density"] == app["maxRoofs"]
    assert pb.CYBERPUNK["still"].get("haze", pb.MAX_TIER_HAZE) == app["maxHaze"]
    assert pb.CYBERPUNK["still"]["exposure"] == "auto"


def test_the_field_is_the_one_cyberpunk_flies_in_detail(pb):
    pairs = re.search(r"const FLY_PAIRS = \{(.*?)\n\};", MAIN_JS.read_text(),
                      re.S)
    assert pairs is not None, "FLY_PAIRS is gone from main.js"
    pair = re.search(r'cyberpunk:\s*\{\s*less:\s*"([^"]+)",\s*more:\s*"([^"]+)"',
                     pairs.group(1))
    assert pair is not None, "main.js FLY_PAIRS has no cyberpunk pair"
    assert pb.CYBERPUNK["field_demo"] == pair.group(2)
    assert pb.CYBERPUNK["field_demo"] in {s["id"] for s in pb.DEMOS}


def test_the_backdrop_is_written_where_the_page_reads_it(pb):
    page = re.search(r'const CYBERPUNK_BACKDROP = "([^"]+)";',
                     MAIN_JS.read_text())
    assert page is not None, "main.js CYBERPUNK_BACKDROP is gone"
    # Resolved on both sides: where web/soar/demos is a symlink into the
    # site's working tree, the bake writes through it.
    served = (REPO / "web" / "soar" / page.group(1)).resolve()
    assert pb.CYBERPUNK["out"].resolve() == served


def test_the_city_camera_is_scene_js_cityStartCamera(pb, app):
    # scene.js imports the GPU modules, so its cityStartCamera is restated
    # here from the two pieces it is made of, which node can import.
    """The same world point and the same relative triple as the browser's
    conversion, over a box that does not contain the camera."""
    bmin, bmax = [-5.0, -5.0, 25.0], [20475.0, 20475.0, 3875.0]
    js = textwrap.dedent(f"""
        import {{ worldToRelative }} from "{(REPO / 'web/soar/camera.js').as_uri()}";
        import * as K from "{CONSTANTS_JS.as_uri()}";
        const [cx, cy, cz] = K.CITY_START_CAMERA.positionCityM;
        const world = [K.CITY_TILE_OFFSET_M[0] + cx,
                       K.CITY_TILE_OFFSET_M[1] + cy, cz];
        console.log(JSON.stringify({{ world,
          rel: worldToRelative(world, {bmin}, {bmax}) }}));
    """)
    out = json.loads(subprocess.run(
        ["node", "--input-type=module", "-e", js],
        capture_output=True, text=True, check=True).stdout)
    world = pb.city_camera_world(pb.CYBERPUNK["camera_city_m"],
                                 pb.CYBERPUNK["city_offset"])
    assert list(world) == out["world"]
    assert pb.world_to_relative(world, bmin, bmax) == pytest.approx(
        out["rel"], abs=1e-12)
