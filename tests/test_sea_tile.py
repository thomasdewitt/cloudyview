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

REPO = Path(__file__).resolve().parents[1]
SEA_DIR = REPO / "cloudyview" / "soar" / "ocean"
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"


@pytest.fixture(scope="module")
def sea():
    return sea_tile.read(SEA_DIR)


def _wgsl_array(src, name):
    m = re.search(rf"const {name} = array<f32, (\d+)>\((.*?)\);", src, re.S)
    assert m, f"{name} not found in raymarch.wgsl"
    vals = [float(v) for v in m.group(2).replace("\n", " ").split(",")
            if v.strip()]
    assert len(vals) == int(m.group(1))
    return vals


def _wgsl_scalar(src, name):
    m = re.search(rf"const {name}: f32 = ([0-9.eE+-]+);", src)
    assert m, f"{name} not found in raymarch.wgsl"
    return float(m.group(1))


def test_the_shader_carries_the_tiles_own_numbers(sea):
    meta, _ = sea
    src = SHADER.read_text()
    assert _wgsl_array(src, "SEA_SLOPE_VARIANCE_REMOVED") == pytest.approx(
        meta["slope_variance_removed"], rel=1e-6, abs=1e-12)
    assert _wgsl_array(src, "SEA_SLOPE_VARIANCE_TEMPORAL") == pytest.approx(
        meta["slope_variance_temporal"], rel=1e-6, abs=1e-12)
    assert len(meta["slope_variance_removed"]) == meta["mips"]
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
    m = re.search(r"const SWELL_MN = array<vec2<f32>, (\d+)>\((.*?)\);",
                  src, re.S)
    pairs = re.findall(r"vec2<f32>\(([-0-9.]+), ([-0-9.]+)\)", m.group(2))
    assert len(pairs) == int(m.group(1))
    for a, b in pairs:
        assert float(a) == int(float(a)) and float(b) == int(float(b))


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
