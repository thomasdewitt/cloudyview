"""The reprojection pass's host side (web/soar/reproject.js), under node.

What can go wrong without a GPU error to say so:

- The resolve block is filled by index and read by a WGSL struct. A field
  added to one and not the other shifts every value after it, and the pass
  then reprojects with a camera made of the wrong numbers.
- The march writes its guide in units of GUIDE_UNIT_M and the resolve
  multiplies the camera's displacement into the same unit. The two constants
  live in two languages.
- A periodic camera that crosses a lateral face reappears at the opposite
  one. That is a fold, not a flight of one domain width, and the picture
  does not move.

Whether the reprojected picture is any good is not tested here: that is
tools/flight_quality.mjs, which needs a browser and a GPU.

Skips only without node.
"""

import json
import re
import shutil
import subprocess
import textwrap

import pytest

from cloudyview.soar import shader

from tests.conftest import REPO

REPROJECT_JS = REPO / "web" / "soar" / "reproject.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not REPROJECT_JS.exists(),
    reason="needs node and web/soar/reproject.js")

_JS = textwrap.dedent("""
    import * as P from "%s";

    const out = { floats: P.RESOLVE_UNIFORM_FLOATS, unit: P.GUIDE_UNIT_M };
    out.shader = P.resolveShader("// math");

    // A 100 km periodic domain; the camera crosses the +x face.
    const bmin = [0, 0, 0], bmax = [100000, 50000, 15000];
    out.fold = P.cameraDelta([40, 20000, 3000], [99990, 20010, 2990],
                             true, bmin, bmax);
    out.open = P.cameraDelta([40, 20000, 3000], [99990, 20010, 2990],
                             false, bmin, bmax);

    // A block whose every float is its own index, and a previous camera
    // offset by 1000, so each packed value says where it came from.
    const u = Float32Array.from({ length: 84 }, (_, i) => i);
    const prev = Float32Array.from({ length: 16 }, (_, i) => 1000 + i);
    const packed = P.packResolveUniforms(
      new Float32Array(P.RESOLVE_UNIFORM_FLOATS), u, prev, [250, -50, 10], {
        frameIndex: 7, jitterScale: 0.5, subpixel: true, historyValid: true,
        lifeFrames: 30, renderSize: [300, 200], historySize: [600, 400],
        params: { splatSigma: 0.25, clampGamma: 1.5, driftBudgetPx: 2.0,
                  fillWeight: 0.125 },
        countCap: 3,
      });
    out.packed = Array.from(packed);
    process.stdout.write(JSON.stringify(out));
""") % REPROJECT_JS.as_uri()


@pytest.fixture(scope="module")
def host():
    proc = subprocess.run(["node", "--input-type=module", "-e", _JS],
                          capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def _struct_fields(wgsl: str, name: str) -> list:
    """(field, floats) for each member of `struct name`, in order."""
    body = re.search(r"struct %s \{(.*?)\n\};" % name, wgsl, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    sizes = {"f32": 1, "vec2<f32>": 2, "vec3<f32>": 3, "vec4<f32>": 4}
    return [(field, sizes[kind]) for field, kind in
            re.findall(r"(\w+):\s*([\w<>]+),", body)]


def test_the_block_is_as_long_as_the_struct(host):
    fields = _struct_fields(host["shader"], "ResolveUniforms")
    assert sum(n for _, n in fields) == host["floats"]


def test_every_field_is_packed_where_the_struct_reads_it(host):
    """Each field's offset in the struct, against what was packed there.

    The struct is rows of a vec3 and a scalar, then two vec2, then scalars,
    so no member needs padding and a field's offset is the floats before it.
    """
    offset, at = 0, {}
    for field, n in _struct_fields(host["shader"], "ResolveUniforms"):
        at[field] = host["packed"][offset:offset + n]
        offset += n
    # This frame's camera rows come from the march block as they lie, but
    # for the two scalars the resolve replaces.
    assert at["cam_origin"] == [0, 1, 2] and at["tan_half_fov"] == [3]
    assert at["cam_forward"] == [4, 5, 6] and at["aspect"] == [7]
    assert at["cam_right"] == [8, 9, 10] and at["frame_index"] == [7]
    assert at["cam_up"] == [12, 13, 14] and at["jitter_scale"] == [0.5]
    # The displacement, in the guide's unit.
    unit = host["unit"]
    assert at["delta_g"] == pytest.approx([250 / unit, -50 / unit, 10 / unit])
    # The previous camera.
    assert at["prev_tan_half_fov"] == [1003]
    assert at["prev_forward"] == [1004, 1005, 1006]
    assert at["prev_right"] == [1008, 1009, 1010]
    assert at["prev_up"] == [1012, 1013, 1014]
    assert at["history_valid"] == [1.0]
    assert at["life_frames"] == [30]
    assert at["splat_sigma"] == [0.25]
    assert at["render_size"] == [300, 200]
    assert at["history_size"] == [600, 400]
    assert at["clamp_gamma"] == [1.5]
    assert at["drift_budget"] == [2.0]
    assert at["fill_weight"] == [0.125]
    assert at["subpixel"] == [1.0]
    # History pixels per radian at the centre: width / (2 tan(fov / 2)).
    assert at["px_per_rad"] == pytest.approx([600 / (2 * 3)])
    assert at["count_cap"] == [3]


def test_the_guide_unit_is_the_marchs(host):
    march = (shader.SHADER_DIR / "march.wgsl").read_text()
    unit = float(re.search(r"const GUIDE_UNIT_M: f32 = ([0-9.]+);",
                           march).group(1))
    assert host["unit"] == unit


def test_a_fold_at_the_period_is_not_a_displacement(host):
    # 10 m short of the +x face to 40 m past the -x face: 50 m of flight.
    assert host["fold"] == pytest.approx([50.0, -10.0, 10.0])
    # A domain that does not tile has no fold to take out.
    assert host["open"] == pytest.approx([40 - 99990, -10.0, 10.0])
