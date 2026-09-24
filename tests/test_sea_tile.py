"""The moving sea's tile: the bytes, the two hosts' reading of them, and the
numbers raymarch.wgsl carries about them.

  * The shader's SEA_* constants are the measurements the exporter recorded
    in the sea's meta.json — a re-export that moves them fails here until
    the shader is updated.
  * The browser (scene.js decodeSeaFrame / seaMipDown) and Python
    (cloudyview/sea_tile.py) decode and mip-filter the same bytes to the same
    float32 values, bit for bit, on real frames of the shipped tile.
  * The foam rank keeps its promise: a cut at 1 - g covers g times the
    foam cap in every frame, which is what lets the shader know the far
    field's whitecap cover without a sample.
"""
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview import sea_tile
from cloudyview.soar_host import SURFACE_TILE_SCHEMAS

REPO = Path(__file__).resolve().parents[1]
SEA_DIR = REPO / "cloudyview" / "soar" / "ocean"
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"


@pytest.fixture(scope="module")
def sea():
    return sea_tile.read(SEA_DIR, SURFACE_TILE_SCHEMAS["ocean"])


def _wgsl_pairs(src, name):
    m = re.search(rf"const {name} = array<vec2<f32>, (\d+)>\((.*?)\n\);",
                  src, re.S)
    assert m, f"{name} not found in raymarch.wgsl"
    pairs = [(float(a), float(b)) for a, b in re.findall(
        r"vec2<f32>\(([-0-9.eE+]+), ([-0-9.eE+]+)\)", m.group(2))]
    assert len(pairs) == int(m.group(1))
    return pairs


def _wgsl_scalar(src, name):
    m = re.search(rf"const {name}: f32 = ([0-9.eE+-]+);", src)
    assert m, f"{name} not found in raymarch.wgsl"
    return float(m.group(1))


def test_the_shader_carries_the_tiles_own_numbers(sea):
    meta, _ = sea
    src = SHADER.read_text()
    pairs = _wgsl_pairs(src, "SEA_SLOPE_VARIANCE")
    assert [a for a, _ in pairs] == pytest.approx(
        meta["slope_variance_removed"], rel=1e-6, abs=1e-12)
    assert [b for _, b in pairs] == pytest.approx(
        meta["slope_variance_temporal"], rel=1e-6, abs=1e-12)
    assert len(pairs) == meta["mips"]
    assert _wgsl_scalar(src, "SEA_LOOP_S") == pytest.approx(meta["loop_s"])
    assert _wgsl_scalar(src, "SEA_FOAM_MAX_COVERAGE") == pytest.approx(
        meta["foam_max_coverage"])
    # The swell folds where the surface frame folds, and the tile divides it.
    period = _wgsl_scalar(src, "SWELL_PERIOD_M")
    assert period == pytest.approx(meta["surface_period_m"])
    ratio = meta["surface_period_m"] / meta["tile_extent_m"]
    assert ratio == pytest.approx(round(ratio))


def test_every_swell_train_is_whole_cycles_across_the_period():
    src = SHADER.read_text()
    import math
    period = _wgsl_scalar(src, "SWELL_PERIOD_M")
    calls = re.findall(
        r"sea_swell_train\(&s, vec2<f32>\(([-0-9.]+), ([-0-9.]+)\), "
        r"([0-9.]+), ([0-9.]+),", src)
    assert len(calls) == 15
    for kx, ky, kk, omega in calls:
        m, n = (float(v) * period / (2.0 * math.pi) for v in (kx, ky))
        assert m == pytest.approx(round(m), abs=1e-5)
        assert n == pytest.approx(round(n), abs=1e-5)
        assert float(kk) == pytest.approx(math.hypot(float(kx), float(ky)),
                                          rel=1e-6)
        # Deep water, to within the clock period's frequency quantum.
        assert abs(float(omega) - math.sqrt(9.81 * float(kk))) <= (
            math.pi / _wgsl_scalar(src, "SEA_CLOCK_PERIOD_S") + 1e-7)


def test_the_foam_rank_covers_exactly_what_the_gate_asks(sea):
    meta, planes = sea
    n = meta["n"]
    cap = meta["foam_max_coverage"]
    foam = planes[2].astype(np.float64) / 255.0          # (t, y, x)
    for gate in (0.25, 0.5, 0.9):
        cover = (foam > 1.0 - gate).mean(axis=(1, 2))
        # One quantization step of the rank is 1/255 of the cap.
        assert np.all(np.abs(cover - gate * cap) <= cap / 255.0 + 1.0 / n**2)


def test_the_sea_loop_closes(sea):
    """Frame 0 follows the last frame the way any frame follows the one
    before it — the loop is periodic, not a clip."""
    meta, planes = sea
    frames = [sea_tile.decode_frame(planes, t, meta)[..., :2]
              for t in range(meta["frames"])]
    steps = [np.mean((frames[(t + 1) % len(frames)] - frames[t]) ** 2)
             for t in range(len(frames))]
    assert steps[-1] < 2.0 * np.median(steps)


_JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    import { gunzipSync } from "node:zlib";
    import { decodeSeaFrame, seaMipDown } from "%s";
    const dir = %s;
    const meta = JSON.parse(readFileSync(dir + "/meta.json", "utf8"));
    const planes = new Uint8Array(gunzipSync(readFileSync(dir + "/" + meta.file)));
    const out = {};
    for (const t of %s) {
      let level = decodeSeaFrame(planes, t, meta);
      const levels = [];
      for (let side = meta.n; ; side >>= 1) {
        levels.push(Buffer.from(level.buffer).toString("base64"));
        if (side === 1) break;
        level = seaMipDown(level, side);
      }
      out[t] = levels;
    }
    process.stdout.write(JSON.stringify(out));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_decodes_and_mips_the_sea_as_python_does(sea):
    import base64
    meta, planes = sea
    frames = [0, 37, meta["frames"] - 1]
    js = _JS % ((REPO / "web" / "soar" / "scene.js").as_posix(),
                json.dumps(SEA_DIR.as_posix()), json.dumps(frames))
    proc = subprocess.run(["node", "--input-type=module", "-e", js],
                          capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    theirs = json.loads(proc.stdout)
    for t in frames:
        ours = sea_tile.mip_chain(sea_tile.decode_frame(planes, t, meta))
        assert len(theirs[str(t)]) == len(ours) == meta["mips"]
        for k, level in enumerate(ours):
            got = np.frombuffer(base64.b64decode(theirs[str(t)][k]),
                                np.float32).reshape(level.shape)
            assert np.array_equal(got, level), f"frame {t} level {k}"


def test_the_whitecaps_far_field_is_their_resolved_mean(sea):
    """The far field's whitecap cover and tint are the means the resolved
    foam averages to: the lace table the shader reads (measured on the tile)
    integrated against the shader's own opacity and tint ramps."""
    meta, _ = sea
    src = SHADER.read_text()
    for k, v in enumerate(meta["lace_cover"]):
        assert _wgsl_scalar(src, f"SEA_LACE_COVER_{k}") == pytest.approx(v)
    assert _wgsl_scalar(src, "SEA_CREST_MEAN") == pytest.approx(
        meta["crest_mean"])
    # The table was measured at these; the shader must cut at the same.
    assert _wgsl_scalar(src, "SEA_LACE_EDGE") == pytest.approx(meta["lace_edge"])
    assert _wgsl_scalar(src, "SEA_LACE_REACH") == pytest.approx(
        meta["lace_reach"])
    tint = [float(v) for v in re.search(
        r"const SEA_FOAM_TINT_OLD: vec3<f32> = vec3<f32>\(([^)]*)\);",
        src).group(1).split(",")]
    cover, mean_tint = sea_tile.foam_far_field(
        meta["lace_cover"], _wgsl_scalar(src, "SEA_FOAM_OPACITY_OLD"),
        _wgsl_scalar(src, "SEA_FOAM_OPACITY_FRESH"), tint,
        _wgsl_scalar(src, "SEA_LACE_FRESH_K"))
    assert _wgsl_scalar(src, "SEA_FOAM_FAR_COVER") == pytest.approx(
        cover, abs=1e-6)
    far_tint = [float(v) for v in re.search(
        r"const SEA_FOAM_FAR_TINT: vec3<f32> = vec3<f32>\(([^)]*)\);",
        src).group(1).split(",")]
    assert far_tint == pytest.approx(list(mean_tint), abs=1e-6)


def test_the_sea_is_periodic_in_time_at_its_clock_period(sea):
    """The hosts wrap the clock at clock_period_s, which is only invisible if
    everything the sea draws repeats there: the tile's loop, the detail
    octave's (loop / SEA_DETAIL_TIME_SCALE) and every swell frequency."""
    import math
    meta, _ = sea
    src = SHADER.read_text()
    T = meta["clock_period_s"]
    assert _wgsl_scalar(src, "SEA_CLOCK_PERIOD_S") == pytest.approx(T)
    loop = meta["loop_s"]
    scale = _wgsl_scalar(src, "SEA_DETAIL_TIME_SCALE")
    assert scale == int(scale)
    for p in (loop, loop / scale):
        assert T / p == pytest.approx(round(T / p), abs=1e-9)
    omegas = re.findall(
        r"sea_swell_train\(&s, vec2<f32>\([-0-9.]+, [-0-9.]+\), "
        r"[0-9.]+, ([0-9.]+),", src)
    assert len(omegas) == 15
    for om in omegas:
        cycles = float(om) * T / (2.0 * math.pi)
        assert cycles == pytest.approx(round(cycles), abs=1e-4)


def test_the_sea_repeats_where_the_surface_frame_folds(sea):
    """Everything drawn off the tile — the detail octave (1/8 scale), the
    lace (1/8, stretched along the wind) and the glitter cells (1.25 cm
    times a power of two) — repeats a whole number of times across the
    surface period, or a fold would show a seam."""
    meta, _ = sea
    src = SHADER.read_text()
    period = meta["surface_period_m"]
    tile = meta["tile_extent_m"]
    detail = tile / _wgsl_scalar(src, "SEA_DETAIL_SCALE")
    lace = tile / _wgsl_scalar(src, "SEA_LACE_SCALE")
    lace_along = lace * _wgsl_scalar(src, "SEA_LACE_STRETCH")
    facet = _wgsl_scalar(src, "SEA_GLITTER_FACET_M")
    for size in (detail, lace, lace_along, facet):
        assert period / size == pytest.approx(round(period / size), abs=1e-6)


def test_the_whitecaps_far_field_near_a_closed_gate(sea):
    """Past theta = 0.6 the kept foam lies inside the cut's soft ramp; the
    far field there is the resolved mean, tabulated at five cuts."""
    meta, _ = sea
    src = SHADER.read_text()
    args = (meta["lace_cover"], _wgsl_scalar(src, "SEA_FOAM_OPACITY_OLD"),
            _wgsl_scalar(src, "SEA_FOAM_OPACITY_FRESH"),
            _wgsl_scalar(src, "SEA_LACE_FRESH_K"),
            _wgsl_scalar(src, "SEA_FOAM_EDGE"))
    for i, theta in enumerate([0.6, 0.7, 0.8, 0.9, 1.0]):
        assert _wgsl_scalar(src, f"SEA_FOAM_TAIL_{i}") == pytest.approx(
            sea_tile.foam_resolved_cover(theta, *args), rel=1e-5, abs=1e-8)
