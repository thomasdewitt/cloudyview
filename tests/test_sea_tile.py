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


def _wgsl_pairs(src, fn):
    """The vec2 cases of a `fn fn(k: i32) -> vec2<f32> { switch k {...} }`."""
    m = re.search(rf"fn {fn}\(k: i32\) -> vec2<f32> \{{\s*switch k \{{(.*?)\n    \}}\n\}}",
                  src, re.S)
    assert m, f"{fn} not found in raymarch.wgsl"
    rows = re.findall(r"(case (\d+)|default): \{ return vec2<f32>\(([-0-9.eE+]+), "
                      r"([-0-9.eE+]+)\); \}", m.group(1))
    assert [r[1] for r in rows[:-1]] == [str(k) for k in range(len(rows) - 1)]
    assert rows[-1][0] == "default"
    return [(float(a), float(b)) for _, _, a, b in rows]


def _wgsl_scalar(src, name):
    m = re.search(rf"const {name}: f32 = ([0-9.eE+-]+);", src)
    assert m, f"{name} not found in raymarch.wgsl"
    return float(m.group(1))


def test_the_shader_carries_the_tiles_own_numbers(sea):
    meta, _ = sea
    src = SHADER.read_text()
    pairs = _wgsl_pairs(src, "sea_slope_variance")
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


@pytest.mark.parametrize("surface", ["city", "desert"])
def test_the_clock_is_refused_where_nothing_moves(surface):
    """Only the sea reads the clock (row 24.z is 0 elsewhere), so a time
    asked of a city or a desert would be a silent no-op."""
    from cloudyview.witness import render_nested
    with pytest.raises(ValueError, match="does not move"):
        render_nested([], [0.0, 0.0, 100.0], azimuth=0.0, elevation=0.0,
                      sun_azimuth=0.0, sun_elevation=45.0,
                      surface=surface, scene_time=5.0,
                      city_offset=(8100.0, -44190.0) if surface == "city"
                      else None)


def _wgsl_gain_rows(src):
    body = re.search(r"fn sea_foam_gain_row\(k: i32\) -> SeaGainRow \{"
                     r"\s*switch k \{(.*?)\n    \}\n\}", src, re.S)
    assert body, "sea_foam_gain_row not found in raymarch.wgsl"
    num = r"([-0-9.eE+]+)"
    rows = re.findall(
        r"(case (\d+)|default): \{ return SeaGainRow\(vec4<f32>\("
        + ", ".join([num] * 4) + r"\), vec4<f32>\(" + ", ".join([num] * 4)
        + r"\), " + num + r"\); \}", body.group(1))
    assert [r[1] for r in rows[:-1]] == [str(k) for k in range(len(rows) - 1)]
    assert rows[-1][0] == "default"
    return [[float(v) for v in r[2:]] for r in rows]


def _gain_lookup(table, theta, along_over_dx):
    """raymarch.wgsl sea_foam_near_gain, on the meta's table."""
    import math
    g = np.asarray(table["gain"])
    x = min(max(2.0 * (math.log2(along_over_dx) + 3.0), 0.0), len(g) - 1.0)
    k = min(math.floor(x), len(g) - 2)
    c = min(max(10.0 * (theta - 0.2), 0.0), 8.0)
    i = min(math.floor(c), 7)

    def pick(row):
        return row[i] + (row[i + 1] - row[i]) * (c - i)
    return pick(g[k]) + (pick(g[k + 1]) - pick(g[k])) * (x - k)


def test_the_shader_reads_the_near_gain_the_tile_measured(sea):
    """The shader's gain rows are meta.json's foam_near_gain, on the axes
    its lookup assumes, measured with the shader's own near-field constants
    — change one of those and the table has to be measured again."""
    from cloudyview import look
    meta, _ = sea
    table = meta["foam_near_gain"]
    src = SHADER.read_text()
    assert table["x"] == pytest.approx([-3.0 + 0.5 * i for i in range(12)])
    assert table["theta"] == pytest.approx([0.2 + 0.1 * i for i in range(9)])
    # Measured where it is read, but for a closed gate (see the exporter).
    assert table["theta_measured"] == pytest.approx(table["theta"][:-1] + [0.99])
    np.testing.assert_allclose(_wgsl_gain_rows(src), table["gain"], atol=5e-6)
    assert table["constants"] == sea_tile.shader_scalars(
        src, sea_tile.FOAM_NEAR_CONSTANTS)
    assert table["mip_bias"] == look.OCEAN_MIP_BIAS


@pytest.mark.parametrize("x, theta", [
    (-2.0, 0.5), (0.0, 0.3), (0.0, 0.7), (1.5, 0.5), (2.5, 0.4),
    (-2.0, 0.95), (2.5, 0.95),
    # Between rows and columns: the bilinear read, not a measured cell.
    (-1.25, 0.45), (1.25, 0.65), (2.25, 0.35)])
def test_the_near_field_has_the_far_fields_cover(sea, x, theta):
    """With its gain, as the shader reads the table, the near field's
    whitecaps average to the far field's cover (foam_resolved_cover) — the
    no-seam promise, re-measured on pixels the exporter never drew."""
    from cloudyview import look
    meta, planes = sea
    table = meta["foam_near_gain"]
    c = table["constants"]
    near = sea_tile.NearFoam(meta, planes, c, look.OCEAN_MIP_BIAS)
    s = near.samples(meta["dx_m"] * 2.0 ** x, 3_000_000, seed=7)
    far = sea_tile.foam_resolved_cover(
        theta, meta["lace_cover"], c["SEA_FOAM_OPACITY_OLD"],
        c["SEA_FOAM_OPACITY_FRESH"], c["SEA_LACE_FRESH_K"],
        c["SEA_FOAM_EDGE"])
    gain = _gain_lookup(table, theta, 2.0 ** x)
    assert near.cover(s, theta, gain) == pytest.approx(far, rel=0.03)
