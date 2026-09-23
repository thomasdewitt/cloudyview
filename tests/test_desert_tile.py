"""The shipped desert tile, and everything that has to agree with its bytes.

Nothing at run time would notice if the tile were re-baked and the numbers
derived from it were not: the shader's clearance stepping trusts a per-mip
slope table, the far field's bounce trusts a mean lightness, and the camera's
floor trusts the octaves' margin. Each is held to the tile here.
"""

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from cloudyview.soar_host import (DESERT_DIR, SHADER_PATH,         # noqa: E402
                                  decode_surface_level, surface_level_files)
import export_desert_tile                                        # noqa: E402

META = json.loads((DESERT_DIR / "meta.json").read_text())
SHADER = SHADER_PATH.read_text()


def _level(k):
    side = META["n"] >> k
    raw = gzip.decompress((DESERT_DIR / surface_level_files(META)[k]).read_bytes())
    return np.frombuffer(decode_surface_level(raw, side, META),
                         np.float16).reshape(side, side, 2).astype(np.float64)


def _const(name):
    m = re.search(rf"const {name}: f32 = ([-0-9.e]+);", SHADER)
    assert m, f"{name} is not declared in raymarch.wgsl"
    return float(m.group(1))


def test_the_shader_slope_table_is_the_tiles():
    m = re.search(r"const DESERT_SLOPE_MAX = array<f32, (\d+)>\(([^)]*)\);",
                  SHADER)
    assert m, "DESERT_SLOPE_MAX is not declared in raymarch.wgsl"
    table = [float(v) for v in m.group(2).replace("\n", " ").split(",")
             if v.strip()]
    assert int(m.group(1)) == len(table) == META["mips"]
    assert table == META["slope_max"]
    assert _const("DESERT_OCTAVE_SLOPE_MAX") == META["band_slope_max"]
    m = re.search(r"const DESERT_ENVELOPE_M = array<f32, (\d+)>\(([^)]*)\);",
                  SHADER)
    assert m, "DESERT_ENVELOPE_M is not declared in raymarch.wgsl"
    env = [float(v) for v in m.group(2).replace("\n", " ").split(",")
           if v.strip()]
    assert env == META["envelope_m"]


def test_the_measured_numbers_are_the_bytes():
    """slope_max and band_max_m, re-measured from the shipped levels."""
    heights = [_level(k)[..., 0] for k in range(META["mips"])]
    stats = export_desert_tile.relief_stats(heights)
    assert stats == {k: META[k] for k in stats}
    assert META["height_min_m"] == heights[0].min()
    assert META["height_max_m"] == heights[0].max()


def test_the_mean_lightness_is_the_top_mip():
    top = _level(META["mips"] - 1)
    assert top.shape == (1, 1, 2)
    assert _const("DESERT_MEAN_LIGHTNESS") == META["mean_lightness"]
    assert META["mean_lightness"] == round(float(top[0, 0, 1]), 4)


def test_the_octave_margin_is_the_sum_of_the_gains():
    g = _const("DESERT_OCTAVE_GAIN")
    n = int(re.search(r"const DESERT_OCTAVES: i32 = (\d+);", SHADER).group(1))
    assert g == pytest.approx(5.0 ** -0.7, abs=1e-6)
    assert _const("DESERT_OCTAVE_MARGIN") == pytest.approx(
        sum(g ** k for k in range(1, n + 1)), abs=5e-4)
    scene = (REPO / "web" / "soar" / "scene.js").read_text()
    js = re.search(r"export const DESERT_OCTAVE_MARGIN = ([0-9.]+);", scene)
    assert js and float(js.group(1)) == _const("DESERT_OCTAVE_MARGIN")


def test_each_mip_is_the_box_mean_of_the_last():
    """The far field's albedo is the mean of the near field's: lightness is
    linear and the mips are exact box means (to half precision)."""
    for k in range(1, 5):
        fine, coarse = _level(k - 1), _level(k)
        mean = 0.25 * (fine[0::2, 0::2] + fine[1::2, 0::2]
                       + fine[0::2, 1::2] + fine[1::2, 1::2])
        assert np.abs(mean - coarse).max() <= 2.0 ** -6 * max(
            1.0, np.abs(coarse).max())


def test_the_encoding_round_trips():
    rng = np.random.default_rng(3)
    level = rng.normal(size=(8, 8, 2)).astype(np.float16)
    raw = export_desert_tile.encode_level(level)
    back = np.frombuffer(decode_surface_level(raw, 8, META),
                         np.float16).reshape(8, 8, 2)
    assert np.array_equal(back, level)


_DECODE_JS = textwrap.dedent("""
    import { decodeSurfaceLevel } from "%s";
    import { readFileSync } from "node:fs";
    const raw = new Uint8Array(readFileSync(process.env.RAW));
    const out = decodeSurfaceLevel(raw, 8, JSON.parse(process.env.META));
    process.stdout.write(Buffer.from(out).toString("base64"));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_decodes_the_same_bytes(tmp_path):
    rng = np.random.default_rng(5)
    level = rng.normal(size=(8, 8, 2)).astype(np.float16)
    raw = export_desert_tile.encode_level(level)
    (tmp_path / "raw.bin").write_bytes(raw)
    js = _DECODE_JS % (REPO / "web" / "soar" / "scene.js").as_posix()
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js], capture_output=True,
        text=True, cwd=REPO,
        env={**os.environ, "RAW": str(tmp_path / "raw.bin"),
             "META": json.dumps(META)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    import base64
    assert base64.b64decode(proc.stdout) == decode_surface_level(raw, 8, META)


_FLOOR_JS = textwrap.dedent("""
    import { desertFloorAt } from "%s";
    import { readFileSync } from "node:fs";
    const meta = JSON.parse(process.env.META);
    const heights = new Uint16Array(readFileSync(process.env.H).buffer);
    const floor = desertFloorAt({ heights, meta });
    const pts = JSON.parse(process.env.PTS);
    process.stdout.write(JSON.stringify(pts.map((p) => floor(p))));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_camera_floor_stands_above_the_drawn_ground(tmp_path):
    """At texel centres the bilinear floor is the texel, plus the octaves'
    largest lift and the clearance — including across the tile's wrap."""
    h0 = _level(0)[..., 0]
    halves = h0.astype(np.float16).view(np.uint16)
    (tmp_path / "h.bin").write_bytes(halves.tobytes())
    cell = META["cell_m"]
    n = META["n"]
    ij = [(0, 0), (17, 1033), (n - 1, n - 1), (1500, 3)]
    pts = [[(i + 0.5) * cell, (j + 0.5) * cell] for i, j in ij]
    pts.append([-0.5 * cell, -0.5 * cell])          # wraps to (n-1, n-1)
    js = _FLOOR_JS % (REPO / "web" / "soar" / "scene.js").as_posix()
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js], capture_output=True,
        text=True, cwd=REPO,
        env={**os.environ, "META": json.dumps(META), "PTS": json.dumps(pts),
             "H": str(tmp_path / "h.bin")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout)
    constants = (REPO / "web" / "soar" / "constants.js").read_text()
    clearance = float(re.search(
        r"DESERT_FLOOR_CLEARANCE_M = ([0-9.]+);", constants).group(1))
    lift = META["band_max_m"] * _const("DESERT_OCTAVE_MARGIN") + clearance
    want = [h0[j, i] + lift for i, j in ij] + [h0[n - 1, n - 1] + lift]
    assert got == pytest.approx(want, abs=1e-9)
