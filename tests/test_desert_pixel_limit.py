"""The full desert refuses a pixel finer than its flora can walk to.

raymarch.wgsl's refine stage walks the near plants out to
DESERT_FLORA_REFINE_FP_M / pixel angle, and the explicit walk is sure of
only DF_EXPLICIT_MAX_T on its cell budget. A pixel finer than their ratio
puts plants inside the refine reach that no walk visits: they would go
undrawn, silently. Both hosts check the pixel angle against
DESERT_MIN_PIXEL_TAN when packing; this pins that constant, on both hosts,
to the shader's own, and checks that both hosts refuse the full desert and
pass the simplified one (DESERT_LITE walks no flora) at the same pixel.

And the refine stage's reach against the far cover: ground inside the reach
is shaded by the refine stage alone, which lays no far cover (decals, the
field) over it. That is safe only while the far cover starts beyond the
reach on every ray: its start, the shrubs' cut (df_small_cut), is at least
the smaller of DF_SMALL_FAR_FP0 (a steep view) and the tiers' smallest
DESERT_FLORA_CUT times DF_FAR_FP0 (a grazing one), as a footprint.
"""

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import soar_host as sh

REPO = Path(__file__).resolve().parents[1]
SHADER = (REPO / "cloudyview" / "soar" / "raymarch.wgsl").read_text()
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"
UNIFORMS_JS = REPO / "web" / "soar" / "uniforms.js"


def num(name, kind="f32"):
    m = re.search(rf"^const {name}: {kind} = ([0-9.]+);", SHADER, re.M)
    assert m, name
    return float(m.group(1))


def shader_min_pixel_tan():
    m = re.search(r"^const DF_EXPLICIT_MAX_T: f32 = ([0-9.]+) \* "
                  r"f32\(DF_MAX_BIG_CELLS\) \* DF_BIG_CELL;", SHADER, re.M)
    assert m, "DF_EXPLICIT_MAX_T's form changed; update this test and the hosts"
    max_t = (float(m.group(1)) * num("DF_MAX_BIG_CELLS", "i32")
             * num("DF_BIG_CELL", "f32"))
    return num("DESERT_FLORA_REFINE_FP_M", "f32") / max_t


# 320 pixels across: a 0.3 degree field is finer than the limit (~0.46
# degrees), a 1 degree field coarser.
FINE_FOV, COARSE_FOV = 0.3, 1.0
SCENE = dict(bmin=[0.0, 0.0, 0.0], bmax=[25600.0, 25600.0, 5000.0],
             dt_view=200.0, dt_light=200.0, surface="desert", ocean_z=12.5,
             ocean_fif_dx=2.0, ocean_tile_extent=4096.0, ocean_max_lod=11,
             desert_relief_m=[-14.0, 16.046875, 1.2621],
             desert_tile_rows=[[0.5] * 4 for _ in range(22)])
VIEW = dict(camera_position=[12800.0, 12800.0, 300.0], azimuth=20.0,
            elevation=-10.0, output_size=(320, 200), render_size=(320, 200))


def test_the_far_cover_starts_past_the_refine_reach():
    full = [c for tier, c in sh.DESERT_FLORA_CUT_BY_TIER.items()
            if tier not in sh.DESERT_LITE_TIERS]
    start = min(num("DF_SMALL_FAR_FP0"), min(full) * num("DF_FAR_FP0"))
    assert start >= num("DESERT_FLORA_REFINE_FP_M")


def test_hosts_match_the_shader():
    assert sh.DESERT_MIN_PIXEL_TAN == pytest.approx(shader_min_pixel_tan(),
                                                    rel=1e-6)


def python_packs(fov, lite):
    try:
        sh.pack_uniforms(sh.SceneState(**SCENE),
                         sh.ViewState(**VIEW, fov=fov, desert_lite=lite))
    except ValueError as e:
        assert "finer than the desert's flora" in str(e)
        return False
    return True


def test_python_refuses_the_fine_pixel_on_the_full_desert_only():
    assert not python_packs(FINE_FOV, lite=False)
    assert python_packs(FINE_FOV, lite=True)
    assert python_packs(COARSE_FOV, lite=False)


_JS = textwrap.dedent("""
    import * as K from "%s";
    import { packUniforms } from "%s";
    const [state, view, cases] = JSON.parse(process.argv[1]);
    const packs = {};
    for (const [label, fov, desertLite] of cases) {
      try {
        packUniforms(state, { ...view, camera: { ...view.camera, fov },
                              desertLite });
        packs[label] = true;
      } catch (e) {
        if (!String(e.message).includes("finer than the desert's flora")) {
          throw e;
        }
        packs[label] = false;
      }
    }
    console.log(JSON.stringify({ limit: K.DESERT_MIN_PIXEL_TAN, packs }));
""") % (CONSTANTS_JS.as_uri(), UNIFORMS_JS.as_uri())


def test_browser_matches_the_shader_and_refuses_alike():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    state = dict(bmin=SCENE["bmin"], bmax=SCENE["bmax"], dtView=200.0,
                 dtLight=200.0, periodic=True, oceanZ=12.5,
                 oceanReflectance=list(sh.DEFAULT_OCEAN_REFLECTANCE),
                 oceanFifDx=2.0, oceanTileExtent=4096.0, oceanEnabled=True,
                 oceanMaxLod=11, nested=False, city=False, desert=True,
                 desertReliefM=SCENE["desert_relief_m"],
                 desertTileRows=SCENE["desert_tile_rows"])
    view = dict(camera={"position": VIEW["camera_position"],
                        "azimuth": VIEW["azimuth"],
                        "elevation": VIEW["elevation"]},
                outputSize=list(VIEW["output_size"]),
                renderSize=list(VIEW["render_size"]),
                toneMapGamma=sh.DEFAULT_TONE_MAP_GAMMA)
    cases = [["fine_full", FINE_FOV, False], ["fine_lite", FINE_FOV, True],
             ["coarse_full", COARSE_FOV, False]]
    out = subprocess.run(["node", "--input-type=module", "-e", _JS,
                          json.dumps([state, view, cases])],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["limit"] == pytest.approx(shader_min_pixel_tan(), rel=1e-6)
    assert got["packs"] == {"fine_full": False, "fine_lite": True,
                            "coarse_full": True}
