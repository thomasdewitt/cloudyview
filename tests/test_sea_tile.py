"""The moving sea's tile: the bytes, the two hosts' reading of them, and the
numbers raymarch.wgsl carries about them.

  * The shader's SEA_* constants are the measurements the exporter recorded
    in the sea's meta.json — a re-export that moves them fails here until
    the shader is updated.
  * The browser (scene.js decodeSeaFrame / seaMipDown) and Python
    (cloudyview/sea_tile.py) decode and mip-filter the same bytes to the same
    float32 values, bit for bit, on real frames of the shipped tile.
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
    """Everything drawn off the tile — the detail octave (1/8 scale) and
    the glitter cells (1.25 cm times a power of two) — repeats a whole
    number of times across the surface period, or a fold would show a
    seam."""
    meta, _ = sea
    src = SHADER.read_text()
    period = meta["surface_period_m"]
    tile = meta["tile_extent_m"]
    detail = tile / _wgsl_scalar(src, "SEA_DETAIL_SCALE")
    facet = _wgsl_scalar(src, "SEA_GLITTER_FACET_M")
    for size in (detail, facet):
        assert period / size == pytest.approx(round(period / size), abs=1e-6)


def test_the_clock_is_refused_where_nothing_moves():
    """The desert reads no clock (row 24.z is 0 there), so a time asked of
    it would be a silent no-op. The sea and the city read it."""
    from cloudyview.soar_host import CLOCK_SURFACES
    from cloudyview.witness import render_nested
    assert CLOCK_SURFACES == ("ocean", "city")
    with pytest.raises(ValueError, match="does not"):
        render_nested([], [0.0, 0.0, 100.0], azimuth=0.0, elevation=0.0,
                      sun_azimuth=0.0, sun_elevation=45.0,
                      surface="desert", scene_time=5.0)


def test_the_city_states_its_clock_period():
    """The city's clock wraps at its tile's period, where city-street's
    steam repeats exactly (tools/export_web_assets.py CITY_CLOCK_PERIOD_S)."""
    from cloudyview.soar_host import surface_clock_period_s
    city = json.loads((REPO / "cloudyview" / "soar" / "city"
                       / "meta.json").read_text())
    period = surface_clock_period_s("city", city)
    assert surface_clock_period_s("desert", {}) is None
    # The steam states the period it repeats at, and the tile the hosts
    # wrap at must be the same number — neither side can drift alone.
    src = SHADER.read_text()
    assert _wgsl_scalar(src, "cc_manholesteam_CLOCK_PERIOD_S") == period
    # Why it repeats there: its noise lattice rises (lattice scale) x
    # RISE_RATE cells a second and wraps at ZWRAP cells, and the period is
    # a whole ZWRAP — so every octave (lacunarity 2) is back on its own
    # lattice point when the hosts wrap the clock.
    scale = float(re.search(r"vec3<f32>\([^;]*, ([0-9.]+) \* zeta\)",
                            src).group(1))
    rise = _wgsl_scalar(src, "cc_manholesteam_RISE_RATE")
    zwrap = _wgsl_scalar(src, "cc_manholesteam_ZWRAP")
    assert zwrap == round(zwrap)
    assert scale * rise * period == pytest.approx(zwrap, abs=1e-6)
