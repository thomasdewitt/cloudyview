"""The shipped desert tile, and everything that has to agree with its bytes.

Nothing at run time would notice if the tile were re-baked and the numbers
derived from it were not: the shader's clearance stepping trusts a per-mip
slope table, the far field's bounce trusts a mean lightness, and the camera's
floor follows the drawn ground. Each is held to the tile here, the meta's
derived numbers through desert_tile_spec (this repo's reading of
docs/desert_tile_format.md; the generator is desert-surface now). The
conformance fixtures (tests/fixtures/desert-tile-64-*) are held to the same
reading, and soar's decoders to the fixtures.
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
sys.path.insert(0, str(REPO / "tests"))

from cloudyview.soar_host import (DESERT_DIR, SHADER_PATH,         # noqa: E402
                                  decode_surface_level, encode_surface_level,
                                  surface_level_files)
import desert_tile_spec as spec                                 # noqa: E402

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
    """Every derived meta.json field, re-measured from the shipped levels
    as the spec defines it."""
    levels = [_level(k) for k in range(META["mips"])]
    derived = spec.derived_meta(levels, META["cell_m"])
    assert derived == {k: META[k] for k in derived}


def test_the_regional_albedo_is_the_mean_of_the_albedos():
    top = _level(META["mips"] - 1)
    assert top.shape == (1, 1, 2)
    assert META["mean_lightness"] == round(float(top[0, 0, 1]), 4)
    truth = spec.albedo_of(_level(0)[..., 1]).reshape(-1, 3)
    assert META["bounce_albedo"] == [round(float(v), 3) for v in truth.mean(0)]


def test_the_octave_gain_is_five_to_the_minus_0_7():
    """H = 0.7 (DESERT_OCTAVE_MARGIN, which sums the gains, is held to the
    band's true reach in test_desert_bounds)."""
    assert _const("DESERT_OCTAVE_GAIN") == pytest.approx(5.0 ** -0.7, abs=1e-6)


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
    raw = encode_surface_level(level)
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
    raw = encode_surface_level(level)
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
    const raw = readFileSync(process.env.H);
    const all = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    const heightLevels = [];
    let at = 0;
    for (let k = 0; k < 4; k++) {
      const side = meta.n >> k;
      heightLevels.push(all.subarray(at, at + side * side));
      at += side * side;
    }
    const floor = desertFloorAt({ heightLevels, meta });
    const pts = JSON.parse(process.env.PTS);
    process.stdout.write(JSON.stringify(pts.map((p) => floor(p))));
""")


def _bilinear(h, x, y, period):
    n = h.shape[0]
    cell = period / n
    gx, gy = x / cell - 0.5, y / cell - 0.5
    i, j = int(np.floor(gx)), int(np.floor(gy))
    fx, fy = gx - i, gy - j
    at = lambda a, b: h[b % n, a % n]                 # noqa: E731
    return ((at(i, j) * (1 - fx) + at(i + 1, j) * fx) * (1 - fy)
            + (at(i, j + 1) * (1 - fx) + at(i + 1, j + 1) * fx) * fy)


def _desert_height(levels, x, y, period):
    """raymarch.wgsl desert_height (desert_surface_of at fp = 0), in numpy."""
    lod = _const("DESERT_OCTAVE_LOD")
    lo, w = int(np.floor(lod)), lod - np.floor(lod)
    h = _bilinear(levels[0], x, y, period)
    amp = 1.0
    for _ in range(int(re.search(r"const DESERT_OCTAVES: i32 = (\d+);",
                                 SHADER).group(1))):
        amp *= _const("DESERT_OCTAVE_GAIN")
        mx, my = (4 * x - 3 * y) / period, (3 * x + 4 * y) / period
        x, y = period * (mx - np.floor(mx)), period * (my - np.floor(my))
        coarse = (_bilinear(levels[lo], x, y, period) * (1 - w)
                  + _bilinear(levels[lo + 1], x, y, period) * w)
        h += amp * (_bilinear(levels[0], x, y, period) - coarse)
    return h


def test_the_floor_constants_are_the_shaders():
    js = (REPO / "web" / "soar" / "scene.js").read_text()
    for name in ("DESERT_OCTAVE_LOD", "DESERT_OCTAVE_GAIN"):
        got = float(re.search(rf"export const {name} = ([0-9.]+);", js).group(1))
        assert got == _const(name)
    assert (int(re.search(r"export const DESERT_OCTAVES = (\d+);", js).group(1))
            == int(re.search(r"const DESERT_OCTAVES: i32 = (\d+);", SHADER).group(1)))


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_camera_floor_is_the_drawn_ground_plus_the_clearance(tmp_path):
    """The floor is desert_height — the tile and its octaves, as the shader
    sums them — plus the clearance, at arbitrary points and across the
    tile's wrap."""
    levels = [_level(k)[..., 0] for k in range(4)]
    (tmp_path / "h.bin").write_bytes(b"".join(
        lv.astype(np.float32).tobytes() for lv in levels))
    period = META["tile_extent_m"]
    rng = np.random.default_rng(3)
    pts = (rng.random((40, 2)) * period).tolist()
    pts += [[-0.3 * META["cell_m"], -0.5 * META["cell_m"]], [period * 1.7, 3.0]]
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
    want = [_desert_height(levels, x, y, period) + clearance for x, y in pts]
    assert got == pytest.approx(want, abs=1e-6)


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
    """The spec's albedo endmembers are the shader's, and the per-level
    correction measured with them is the one the shader applies."""
    assert _vec3("DESERT_ALBEDO_PAVEMENT") == spec.ALBEDO_PAVEMENT
    assert _vec3("DESERT_ALBEDO_SOIL") == spec.ALBEDO_SOIL
    assert _vec3("DESERT_ALBEDO_WASH") == spec.ALBEDO_WASH
    lights = [_level(k)[..., 1] for k in range(META["mips"])]
    assert spec.albedo_correction(lights) == META["albedo_correction"]


SPEC_DOC = (REPO / "docs" / "desert_tile_format.md").read_text()


def test_the_spec_documents_the_shaders_albedo():
    rows = re.findall(r"^\| ([0-9½]+) \| \w[\w ]* \| \(([^)]*)\) \|",
                      SPEC_DOC, re.M)
    table = {r[0]: tuple(float(v) for v in r[1].split(",")) for r in rows}
    assert table == {"0": _vec3("DESERT_ALBEDO_PAVEMENT"),
                     "½": _vec3("DESERT_ALBEDO_SOIL"),
                     "1": _vec3("DESERT_ALBEDO_WASH")}
    assert META["schema"] in SPEC_DOC


FIXTURES = REPO / "tests" / "fixtures"


def _fixture(fmt):
    path = FIXTURES / f"desert-tile-64-{fmt}"
    meta = json.loads((path / "meta.json").read_text())
    return path, meta


def _fixture_levels_want(fmt):
    """The conformance field's box means, stored as the format stores them."""
    height, light = spec.fixture_field()
    mips = spec.box_mips(np.stack([height, light], axis=-1))
    dtype = np.float16 if fmt == "rg16float" else np.float32
    return [m.astype(dtype) for m in mips]


def test_soar_host_decodes_the_conformance_tile():
    path, meta = _fixture("rg16float")
    want = _fixture_levels_want("rg16float")
    assert meta["mips"] == len(want)
    for k, w in enumerate(want):
        side = meta["n"] >> k
        raw = gzip.decompress((path / meta["files"][k]).read_bytes())
        got = np.frombuffer(decode_surface_level(raw, side, meta),
                            np.float16).reshape(side, side, 2)
        assert np.array_equal(got, w)


@pytest.mark.parametrize("fmt", ["rg16float", "rg32float"])
def test_the_conformance_tiles_meta_is_its_bytes(fmt):
    """The fixture desert-surface's writer made, read by this repo's
    reading of the spec: the two implementations agree."""
    path, meta = _fixture(fmt)
    levels = [lv.astype(np.float64) for lv in _fixture_levels_want(fmt)]
    assert meta["n"] == spec.FIXTURE_N
    assert meta["tile_extent_m"] == spec.FIXTURE_EXTENT_M
    assert meta["cell_m"] == spec.FIXTURE_EXTENT_M / spec.FIXTURE_N
    assert meta["clock_period_s"] is None
    derived = spec.derived_meta(levels, meta["cell_m"])
    assert derived == {k: meta[k] for k in derived}


_FIXTURE_JS = textwrap.dedent("""
    import { decodeSurfaceLevel } from "%s";
    import { readFileSync } from "node:fs";
    import { gunzipSync } from "node:zlib";
    const dir = process.env.DIR;
    const meta = JSON.parse(readFileSync(dir + "/meta.json"));
    const out = [];
    for (let k = 0; k < meta.mips; k++) {
      let raw = readFileSync(dir + "/" + meta.files[k]);
      if (meta.compression === "gzip") raw = gunzipSync(raw);
      const bytes = decodeSurfaceLevel(new Uint8Array(raw), meta.n >> k, meta);
      out.push(Buffer.from(bytes).toString("base64"));
    }
    process.stdout.write(JSON.stringify(out));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("fmt", ["rg16float", "rg32float"])
def test_the_browser_decodes_the_conformance_tiles(fmt):
    path, meta = _fixture(fmt)
    proc = subprocess.run(
        ["node", "--input-type=module", "-e",
         _FIXTURE_JS % (REPO / "web" / "soar" / "scene.js").as_posix()],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "DIR": str(path)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    import base64
    dtype = "<f2" if fmt == "rg16float" else "<f4"
    for k, (b64, w) in enumerate(zip(json.loads(proc.stdout),
                                     _fixture_levels_want(fmt))):
        side = meta["n"] >> k
        got = np.frombuffer(base64.b64decode(b64), dtype).reshape(side, side, 2)
        assert np.array_equal(got, w)


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
