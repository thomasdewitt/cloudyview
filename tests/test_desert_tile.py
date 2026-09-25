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


def _table(fn):
    """The constants a switch accessor `fn(i)` returns for i = 0, 1, ...
    (cases in order, then the default): floats, or rows for vec3s. The
    shader keeps its tables as switches, never as runtime-indexed const
    arrays, which inflate every entry point's registers."""
    m = re.search(rf"fn {fn}\(i: i32\) -> [^{{]+\{{[^\n]*\n\s*switch i \{{"
                  rf"(.*?)\n    \}}\n\}}", SHADER, re.S)
    assert m, f"{fn} is not declared in raymarch.wgsl"
    cases = re.findall(r"(?:case (\d+)|default): \{ return ([^;]+); \}",
                       m.group(1))
    assert [int(k) for k, _ in cases[:-1]] == list(range(len(cases) - 1))
    vals = [v for _, v in cases]
    if vals[0].startswith("vec3"):
        return np.array([[float(x) for x in
                          re.search(r"\(([^)]*)\)", v).group(1).split(",")]
                         for v in vals])
    return np.array([float(v) for v in vals])


def test_the_hosts_carry_the_tiles_tables():
    """Rows 25..46 are the tile's own tables (soar_host.desert_tile_rows),
    at the rows raymarch.wgsl's accessors read them from; the JS twin is
    held to this by tests/test_uniform_parity.py and below."""
    from cloudyview import soar_host as sh
    rows = np.array(sh.desert_tile_rows(META))
    assert rows.shape == (sh.DESERT_TILE_ROWS, 4)
    assert list(rows[0:3].ravel()) == META["slope_max"]
    assert list(rows[3:6].ravel()) == META["envelope_m"]
    assert list(rows[6:9].ravel()) == META["slope_variance_removed"]
    assert rows[9:21, :3].tolist() == META["albedo_correction"]
    assert list(rows[21, :3]) == META["bounce_albedo"]
    assert rows[21, 3] == 0.0
    for pattern in [r"fn desert_slope_max_at\(i: i32\) -> f32 \{\s*"
                    r"return desert_tile_level\(0, i\);",
                    r"fn desert_envelope_m_at\(i: i32\) -> f32 \{\s*"
                    r"return desert_tile_level\(3, i\);",
                    r"fn desert_slope_variance_removed_at\(i: i32\) -> f32 "
                    r"\{\s*return desert_tile_level\(6, i\);",
                    r"return u\.desert_tile\[9 \+ i\]\.xyz;",
                    r"return u\.desert_tile\[21\]\.xyz;"]:
        assert re.search(pattern, SHADER), pattern


_ROWS_JS = textwrap.dedent("""
    import { desertTileRows } from "%s";
    process.stdout.write(JSON.stringify(
        desertTileRows(JSON.parse(process.env.META))));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_packs_the_same_tile_rows():
    from cloudyview import soar_host as sh
    js = _ROWS_JS % (REPO / "web" / "soar" / "scene.js").as_posix()
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js], capture_output=True,
        text=True, cwd=REPO, env={**os.environ, "META": json.dumps(META)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout) == sh.desert_tile_rows(META)


def test_the_measured_numbers_are_the_bytes():
    """slope_max and band_max_m, re-measured from the shipped levels."""
    heights = [_level(k)[..., 0] for k in range(META["mips"])]
    stats = export_desert_tile.relief_stats(heights)
    assert stats == {k: META[k] for k in stats}
    assert META["height_min_m"] == heights[0].min()
    assert META["height_max_m"] == heights[0].max()


def test_the_regional_albedo_is_the_mean_of_the_albedos():
    top = _level(META["mips"] - 1)
    assert top.shape == (1, 1, 2)
    assert META["mean_lightness"] == round(float(top[0, 0, 1]), 4)
    truth = export_desert_tile.albedo_of(_level(0)[..., 1]).reshape(-1, 3)
    assert META["bounce_albedo"] == [round(float(v), 3) for v in truth.mean(0)]


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


def test_the_desert_refuses_to_render_without_its_relief():
    """Without row 8's relief every ray lands on the flat datum and the
    picture still looks plausible — so both hosts refuse it."""
    from cloudyview import soar_host as sh
    base = dict(bmin=[0.0, 0.0, 1000.0], bmax=[1000.0, 1000.0, 5000.0],
                dt_view=50.0, dt_light=50.0, surface="desert",
                ocean_fif_dx=2.0, ocean_tile_extent=4096.0, ocean_max_lod=11)
    view = sh.ViewState(camera_position=[500.0, 500.0, 50.0], azimuth=0.0,
                        elevation=0.0, fov=90.0, output_size=(64, 36),
                        render_size=(64, 36))
    with pytest.raises(ValueError, match="relief"):
        sh.pack_uniforms(sh.SceneState(**base), view)
    rows = sh.desert_tile_rows(META)
    with pytest.raises(ValueError, match="tables"):
        sh.pack_uniforms(sh.SceneState(**base, desert_relief_m=(-1, 2, 3)),
                         view)
    ok = sh.pack_uniforms(sh.SceneState(**base, desert_relief_m=(-1, 2, 3),
                                        desert_tile_rows=rows), view)
    assert np.array_equal(ok[sh.DESERT_TILE_ROW0:], np.float32(rows))
    assert tuple(ok[8]) == (0.0, -1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="desert's"):
        sh.pack_uniforms(sh.SceneState(**{**base, "surface": "ocean"},
                                       desert_relief_m=(0, 0, 0)), view)


def _vec3(name):
    m = re.search(rf"const {name}: vec3<f32> = vec3<f32>\(([^)]*)\);", SHADER)
    assert m, f"{name} is not declared in raymarch.wgsl"
    return tuple(float(v) for v in m.group(1).split(","))


def test_the_far_field_colour_correction_is_the_tiles():
    """The bake's albedo endmembers are the shader's, and the per-level
    correction it measured with them is the one the shader applies."""
    assert _vec3("DESERT_ALBEDO_PAVEMENT") == export_desert_tile.ALBEDO_PAVEMENT
    assert _vec3("DESERT_ALBEDO_SOIL") == export_desert_tile.ALBEDO_SOIL
    assert _vec3("DESERT_ALBEDO_WASH") == export_desert_tile.ALBEDO_WASH
    lights = [_level(k)[..., 1] for k in range(META["mips"])]
    assert export_desert_tile.albedo_correction(lights) == \
        META["albedo_correction"]


def test_the_silt_between_resolved_stones_stays_warm():
    """desert_matrix_albedo, with both stone levels resolved, on each pure
    surface: the silt that keeps the footprint's mean must be a warm soil
    colour (R >= G >= B), and not propped up by the solve's floor. A stone
    mix redder than its surface drove the pavement's silt teal-grey."""
    palette = _table("desert_stone_palette_at")
    area = _const("DESERT_STONE_AREA")
    gravel, pebble = _vec3("DESERT_STONE_P_GRAVEL"), _vec3("DESERT_STONE_P_PEBBLE")
    scale_gain = float(re.search(
        r"fn desert_stone_scale\(m: vec3<f32>\) -> f32 \{\s*return 1\.0 \+ "
        r"([0-9.]+) \* m\.x;", SHADER).group(1))
    surfaces = {"PAVEMENT": (1, 0, 0), "SOIL": (0, 1, 0), "WASH": (0, 0, 1)}
    for name, m in surfaces.items():
        m = np.array(m, float)
        mean = np.array(_vec3(f"DESERT_ALBEDO_{name}"))
        stones = _table(f"desert_stone_mix_{name.lower()}_at") @ palette
        a = area * (1.0 + scale_gain * m[0]) ** 2
        cover = min(1 - (1 - m @ gravel * a) * (1 - m @ pebble * a), 0.9)
        silt = (mean - cover * stones) / (1 - cover)
        assert np.all(silt > 0.35 * mean), (name, silt)
        assert silt[0] >= silt[1] >= silt[2], (name, silt)
