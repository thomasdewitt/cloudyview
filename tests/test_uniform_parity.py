"""The Python host and the browser must fill the same 416 bytes.

There is one renderer core — web/soar/raymarch.wgsl — driven by two hosts:
JavaScript in the tab and Python via wgpu. The shader is shared by
construction, so it cannot drift. The uniform block can, and that is the
whole surface where a Python render could silently stop matching soar.

So this diffs cloudyview.soar.uniforms.pack_uniforms against the browser's own
packUniforms, executed under node, byte for byte. docs/architecture.md
records that a test of exactly this shape was deleted with the desktop app on
2026-08-05; the look has been unpinned since.

Skips when node is unavailable. It does NOT skip when wgpu is unavailable —
no GPU is needed to pack a buffer, which is the point.
"""

import base64
import os
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview.soar import tiers, uniforms as sh
from cloudyview.soar.surfaces import SURFACE_DIRS

REPO = Path(__file__).resolve().parents[1]
UNIFORMS_JS = REPO / "web" / "soar" / "uniforms.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not UNIFORMS_JS.exists(),
    reason="needs node and web/soar/uniforms.js")


# (label, scene kwargs, view kwargs) — chosen to exercise the branches that
# actually differ: nesting, the spectral low-sun path, non-square aspect,
# a non-unit render scale, and jitter/subpixel.
CASES = [
    ("default", {}, {}),
    ("low_sun", {}, {"sun_elevation": 6.0, "sun_azimuth": 250.0}),
    ("sun_at_reference", {}, {"sun_elevation": 55.0}),
    ("split_fade", {}, {"sun_elevation": 50.0}),
    ("witness_gamma", {}, {"tone_map_gamma": sh.TONE_MAP_GAMMA_WITNESS}),
    # Haze feeds three rows through a non-linear ramp (10.z raw, 16.w aerial
    # beta, 19.y ocean haze), so both ends and one interior point.
    ("haze_clear", {}, {"haze": 0.0}),
    ("haze_thick", {}, {"haze": 0.7}),
    ("haze_max", {}, {"haze": 1.0}),
    ("haze_soup", {}, {"haze": 2.0}),
    ("white_point_low", {}, {"tone_map_white_point": 8.0}),
    ("contrast_up", {}, {"contrast": 1.3}),
    ("display_combo", {}, {"tone_map_white_point": 24.0, "contrast": 0.8,
                           "haze": 0.5}),
    ("no_lod", {}, {"light_march_lod_degrees": 0.0, "view_step_lod_degrees": 0.0}),
    ("subpixel", {}, {"subpixel": True, "jitter_scale": 0.65, "frame_index": 37}),
    ("no_jitter", {}, {"jitter": False}),
    ("non_periodic", {"periodic": False}, {"sun_elevation": -8.0}),
    ("scaled_render", {}, {"render_size": (240, 150)}),
    ("nested", {"nested": True, "nest_bmin": [90000.0, 50000.0, 500.0],
                "nest_bmax": [95000.0, 55000.0, 3000.0],
                "dt_view_nest": 40.0, "dt_light_nest": 40.0}, {}),
    ("ocean_off", {"ocean_enabled": False}, {}),
    ("exotic_look", {}, {"exposure": 2.5, "g_hg": 0.4, "ambient_strength": 0.3,
                         "ocean_realism": 0.25, "ocean_mip_bias": 0.75,
                         "spectral_lighting_strength": 0.5,
                         "low_sun_sky_field_strength": 0.25,
                         "cone_stencil_theta_deg": 7.5}),
    # The haze profile, both ways. It is packed as a sentinel — row 17.w is
    # the scale height, and 0 means "no height dependence" — so the two hosts
    # can disagree about it without disagreeing about any named value. Both
    # default to off, which would leave the on-state untested.
    ("haze_height_dependent", {}, {"haze_height_dependent": True}),
    ("haze_uniform", {}, {"haze_height_dependent": False}),
    # The night city: five fixed spectral rows replace the daytime air-mass
    # pipeline, row 8 becomes ground z + tile offset, and both hosts must
    # agree on every one of those numbers or the browser flies a different
    # night than the harness tuned.
    ("city_night", {"surface": "city", "city_tile_offset_m": [136080.0, -18630.0],
                    "ocean_fif_dx": 90.0, "ocean_tile_extent": 92160.0,
                    "ocean_max_lod": 10, "periodic": False},
     {"sun_azimuth": 310.0, "sun_elevation": 22.0}),
    # Row 24, the surface offset the water samples and the city frame are
    # derived from (2026-09-01). Both hosts must fold it into
    # [0, tile extent) with the SAME arithmetic: an explicit override whose
    # negative component wraps, and one far beyond the tile (a flight's
    # offset shifts by cloud periods, which dwarf the day tile) — on a day
    # scene and a city scene respectively. The city case also pins the
    # static default: with no override, row 24 is city_tile_offset_m folded.
    ("ocean_surface_offset", {"surface_offset_m": [-18630.2, 136143.7]}, {}),
    # The sea has no clock (its meta.json clock_period_s is null): row 24.z
    # packs 0 at any scene time, in both — so the scene key, which is these
    # bytes, never restarts the sea's accumulation for a running clock.
    ("ocean_no_clock",
     {"surface_offset_m": [-18630.2, 136143.7], "clock_period_s": None},
     {"scene_time": 1234.5678}),
    ("ocean_no_clock_later",
     {"surface_offset_m": [-18630.2, 136143.7], "clock_period_s": None},
     {"scene_time": 987654.321}),
    # The city's clock runs too (2026-09-24: its steam reads it), wrapped
    # at the city tile's own period, the same way in both.
    ("city_clock_running",
     {"surface": "city", "city_tile_offset_m": [75330.0, -17010.0],
      "ocean_fif_dx": 90.0, "ocean_tile_extent": 23040.0,
      "ocean_max_lod": 8, "periodic": False, "clock_period_s": 2600.0},
     {"sun_azimuth": 310.0, "sun_elevation": 22.0, "scene_time": 77.5}),
    ("city_clock_wrapped",
     {"surface": "city", "city_tile_offset_m": [75330.0, -17010.0],
      "ocean_fif_dx": 90.0, "ocean_tile_extent": 23040.0,
      "ocean_max_lod": 8, "periodic": False, "clock_period_s": 2600.0},
     {"sun_azimuth": 310.0, "sun_elevation": 22.0, "scene_time": 98765.4321}),
    ("city_surface_offset",
     {"surface": "city", "city_tile_offset_m": [75330.0, -17010.0],
      "surface_offset_m": [-102400.0, 204800.0],
      "ocean_fif_dx": 90.0, "ocean_tile_extent": 23040.0,
      "ocean_max_lod": 8, "periodic": False},
     {"sun_azimuth": 310.0, "sun_elevation": 22.0}),
    # Row 25.x, the city's roof clutter density: a tier's value packed
    # under CITY, and none on a day scene whatever the view says.
    ("city_roof_density",
     {"surface": "city", "city_tile_offset_m": [75330.0, -17010.0],
      "ocean_fif_dx": 90.0, "ocean_tile_extent": 23040.0,
      "ocean_max_lod": 8, "periodic": False},
     {"sun_azimuth": 310.0, "sun_elevation": 22.0,
      "city_roof_density": 0.7}),
    ("day_ignores_roof_density", {}, {"city_roof_density": 1.4}),
    ("city_surface_static",
     {"surface": "city", "city_tile_offset_m": [75330.0, -17010.0],
      "ocean_fif_dx": 90.0, "ocean_tile_extent": 23040.0,
      "ocean_max_lod": 8, "periodic": False},
     {"sun_azimuth": 310.0, "sun_elevation": 22.0}),
]

BASE_SCENE = dict(
    bmin=[80000.0, 40000.0, 0.0],
    bmax=[105600.0, 65600.0, 4997.1962889999995],
    dt_view=200.0, dt_light=200.0,
)
BASE_VIEW = dict(
    camera_position=[92800.0, 52800.0, 120.0],
    azimuth=20.0, elevation=35.0, fov=100.0,
    output_size=(320, 200), render_size=(320, 200),
)

_JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    import { packUniforms } from "%s";
    const cases = JSON.parse(readFileSync(process.env.CASES_FILE, "utf8"));
    const out = {};
    for (const [label, state, view] of cases) {
      const u = packUniforms(state, view);
      out[label] = Buffer.from(
        new Uint8Array(u.buffer, u.byteOffset, u.byteLength)).toString("base64");
    }
    console.log(JSON.stringify(out));
""") % UNIFORMS_JS


# The browser's row 9 for the sea is the shipped tile's (scene.js
# loadOceanTile: dx from dx_m, maxLod the top mip), so the stand-in reads it.
_OCEAN_META = json.loads((SURFACE_DIRS["ocean"] / "meta.json").read_text())


def _js_state(extra):
    s = dict(
        bmin=BASE_SCENE["bmin"], bmax=BASE_SCENE["bmax"],
        dtView=BASE_SCENE["dt_view"], dtLight=BASE_SCENE["dt_light"],
        periodic=True, oceanZ=0.0,
        oceanReflectance=list(sh.DEFAULT_OCEAN_REFLECTANCE),
        oceanFifDx=_OCEAN_META["dx_m"],
        oceanTileExtent=_OCEAN_META["tile_extent_m"], oceanEnabled=True,
        oceanMaxLod=_OCEAN_META["mips"] - 1,
        nested=False,
    )
    remap = {"periodic": "periodic", "ocean_enabled": "oceanEnabled",
             "nested": "nested", "nest_bmin": "nestBmin",
             "nest_bmax": "nestBmax", "dt_view_nest": "dtViewNest",
             "dt_light_nest": "dtLightNest",
             "ocean_z": "oceanZ", "city_tile_offset_m": "cityOffsetM",
             "surface_offset_m": "surfaceOffsetM",
             "surface_period_m": "surfacePeriodM",
             "clock_period_s": "clockPeriodS",
             "ocean_fif_dx": "oceanFifDx",
             "ocean_tile_extent": "oceanTileExtent",
             "ocean_max_lod": "oceanMaxLod"}
    for k, v in extra.items():
        if k == "surface":
            # Python names the surface; the browser's scene carries one flag
            # per specialization.
            s["city"] = v == "city"
        else:
            s[remap[k]] = v
    return s


def _js_view(extra):
    v = dict(
        camera={"position": BASE_VIEW["camera_position"],
                "azimuth": BASE_VIEW["azimuth"],
                "elevation": BASE_VIEW["elevation"], "fov": BASE_VIEW["fov"]},
        outputSize=list(BASE_VIEW["output_size"]),
        renderSize=list(BASE_VIEW["render_size"]),
        toneMapGamma=sh.DEFAULT_TONE_MAP_GAMMA,
    )
    remap = {
        "sun_azimuth": "sunAzimuth", "sun_elevation": "sunElevation",
        "tone_map_gamma": "toneMapGamma", "frame_index": "frameIndex",
        "subpixel": "subpixel", "jitter_scale": "jitterScale", "jitter": "jitter",
        "light_march_lod_degrees": "lightMarchLodDegrees",
        "view_step_lod_degrees": "viewStepLodDegrees",
        "exposure": "exposure", "g_hg": "gHg",
        "ambient_strength": "ambientStrength", "ocean_realism": "oceanRealism",
        "ocean_mip_bias": "oceanMipBias",
        "spectral_lighting_strength": "spectralLightingStrength",
        "low_sun_sky_field_strength": "lowSunSkyFieldStrength",
        "cone_stencil_theta_deg": "coneStencilThetaDeg",
        "haze": "haze", "haze_height_dependent": "hazeHeightDependent",
        "tone_map_white_point": "toneMapWhitePoint", "contrast": "contrast",
        "scene_time": "sceneTime", "city_roof_density": "cityRoofDensity",
    }
    for k, val in extra.items():
        if k == "render_size":
            v["renderSize"] = list(val)
        else:
            v[remap[k]] = val
    return v


@pytest.fixture(scope="module")
def js_blocks(tmp_path_factory):
    payload = json.dumps([[label, _js_state(s), _js_view(v)]
                          for label, s, v in CASES])
    cases_file = tmp_path_factory.mktemp("parity") / "cases.json"
    cases_file.write_text(payload)
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "CASES_FILE": str(cases_file)})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def _python_block(scene_extra, view_extra):
    state = sh.SceneState(**{**BASE_SCENE, **scene_extra})
    view = sh.ViewState(**{**BASE_VIEW, **view_extra})
    return sh.pack_uniforms(state, view)


@pytest.mark.parametrize("label,scene_extra,view_extra", CASES,
                         ids=[c[0] for c in CASES])
def test_uniform_block_matches_browser(label, scene_extra, view_extra, js_blocks):
    ours = _python_block(scene_extra, view_extra)
    theirs = np.frombuffer(base64.b64decode(js_blocks[label]),
                           np.float32).reshape(sh.UNIFORM_ROWS, 4)
    assert ours.nbytes == sh.UNIFORM_NBYTES

    if np.array_equal(ours, theirs):
        return
    rows = np.nonzero(~np.all(ours == theirs, axis=1))[0]
    detail = "\n".join(
        f"  row {r:2d}: python={ours[r]}  js={theirs[r]}" for r in rows)
    raise AssertionError(
        f"uniform block differs from the browser in {len(rows)} row(s):\n{detail}")


def test_the_browsers_buffer_is_the_size_of_the_browsers_block():
    """constants.js must agree with itself, and with the Python host.

    This is the test for the failure it was written after. A 24th row was
    added, UNIFORM_ROWS went 23 -> 24 on both hosts, and web/soar/constants.js
    was ALSO carrying an independent literal `UNIFORM_NBYTES = 368` that nobody
    moved. The packer then produced 384 bytes and wrote them into a 368-byte
    buffer, so every draw failed validation and the app rendered black —
    without any test noticing, because the packed ARRAYS still matched each
    other perfectly. The parity test above compares what the two hosts pack;
    this compares what the browser allocates to hold it.

    That row has since been reverted along with the rest of the sparse-brick
    work, so the block is 23 rows again. The test stays: the bug was never
    about the row, it was about a derived quantity being written out by hand,
    and the next row to be added will be added by someone who was not here.
    """
    proc = subprocess.run(
        ["node", "--input-type=module", "-e",
         'import * as K from "%s";'
         'process.stdout.write(JSON.stringify('
         '{rows: K.UNIFORM_ROWS, bytes: K.UNIFORM_NBYTES}));'
         % (REPO / "web" / "soar" / "constants.js").as_posix()],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    assert js["bytes"] == js["rows"] * 16, (
        f"constants.js says {js['rows']} rows but {js['bytes']} bytes; the "
        "buffer and the block have come apart")
    assert (js["rows"], js["bytes"]) == (sh.UNIFORM_ROWS, sh.UNIFORM_NBYTES), (
        f"browser has {js['rows']} rows / {js['bytes']} bytes, Python has "
        f"{sh.UNIFORM_ROWS} / {sh.UNIFORM_NBYTES}")


def test_block_is_416_bytes():
    """26 rows of 4 floats: row 24 is the surface offset (2026-09-01); the
    desert tile's rows 25..46 left with the desert (2026-11-09); row 25, the
    last, is the city's look knobs (2026-11-09). Pinned rather than derived
    on purpose, so that a row added on one side and not the other fails here
    as well as in the diff above."""
    assert sh.UNIFORM_NBYTES == 416
    assert _python_block({}, {}).nbytes == 416


def test_roof_density_tables_agree():
    """The tier table that fills row 25.x is written in both hosts; they
    must be the same table."""
    import re
    text = (REPO / "web" / "soar" / "constants.js").read_text()
    m = re.search(r"CITY_ROOF_DENSITY_BY_TIER = \{([^}]*)\}", text)
    assert m, "CITY_ROOF_DENSITY_BY_TIER not found in constants.js"
    js = {k: float(v) for k, v in
          re.findall(r"(\w+):\s*([0-9.]+)", m.group(1))}
    assert js == tiers.CITY_ROOF_DENSITY_BY_TIER
    # One density per quality tier, every one inside the range the far mean
    # is measured over (the packers refuse anything else).
    assert set(tiers.CITY_ROOF_DENSITY_BY_TIER) == set(tiers.QUALITY_PRESETS)
    lo, hi = tiers.CITY_ROOF_DENSITY_RANGE
    assert all(lo <= d <= hi
               for d in tiers.CITY_ROOF_DENSITY_BY_TIER.values())
    m = re.search(r"CITY_ROOF_DENSITY_RANGE = \[([^\]]*)\]", text)
    assert m and [float(v) for v in m.group(1).split(",")] == [lo, hi]


def test_roof_density_outside_the_range_is_refused():
    city = {"surface": "city", "city_tile_offset_m": [75330.0, -17010.0],
            "ocean_fif_dx": 90.0, "ocean_tile_extent": 23040.0,
            "ocean_max_lod": 8, "periodic": False}
    for d in (0.0, 0.49, 1.41, 2.0):
        with pytest.raises(ValueError, match="city_roof_density"):
            _python_block(city, {"sun_elevation": 22.0,
                                 "city_roof_density": d})


@pytest.mark.parametrize("surface", ["ocean", "city"])
def test_a_running_clock_needs_its_surfaces_period(surface):
    """No default: a surface's clock is stated (a period, or None for no
    clock), and a running clock with it unstated is an error — a forgotten
    city period would otherwise freeze the steam silently."""
    extra = ({"surface": "city", "city_tile_offset_m": [75330.0, -17010.0]}
             if surface == "city" else {})
    with pytest.raises(ValueError, match="running clock needs"):
        _python_block(extra, {"scene_time": 5.0})
    with pytest.raises(ValueError, match="positive or None"):
        _python_block({**extra, "clock_period_s": 0.0}, {"scene_time": 5.0})


def test_a_static_sea_packs_the_same_at_any_clock(js_blocks):
    """The sea's block at scene time 1234.6 and at 987654.3 is the same
    block, byte for byte, in both hosts: row 24.z is 0 — so the scene key
    (these bytes) does not restart the sea's accumulation as the clock
    runs."""
    a = base64.b64decode(js_blocks["ocean_no_clock"])
    b = base64.b64decode(js_blocks["ocean_no_clock_later"])
    assert a == b
    ours = _python_block(*[c[1:] for c in CASES
                           if c[0] == "ocean_no_clock"][0])
    assert ours[24][2] == 0.0


def test_periodic_requires_sun_above_horizon():
    """Not a clamp — a periodic light march has no exit below the horizon."""
    with pytest.raises(ValueError, match="above the horizon"):
        _python_block({}, {"sun_elevation": -1.0})


def test_both_hosts_build_the_same_shader_module():
    """The module text a host compiles — manifest.json's files joined behind
    the switch header — is the same string from soar/shader.py and from
    web/soar/shader.js, for every surface and every switch."""
    from cloudyview.soar import shader
    script = textwrap.dedent("""
        import fs from "node:fs";
        import path from "node:path";
        import { assembleShader, shaderFiles, specializeShader }
          from "%s";
        const dir = "%s";
        const manifest = JSON.parse(
          fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
        const out = {};
        for (const surface of Object.keys(manifest.surfaces)) {
          const texts = {};
          for (const f of shaderFiles(manifest, surface)) {
            texts[f] = fs.readFileSync(path.join(dir, f), "utf8");
          }
          const source = assembleShader(manifest, texts, surface);
          out[surface] = [false, true].map((on) => specializeShader(source, {
            periodic: on, nested: !on, toneMap: on,
            lightCache: !on, skyProbe: on }));
        }
        console.log(JSON.stringify(out));
    """) % ((REPO / "web" / "soar" / "shader.js").as_uri(), shader.SHADER_DIR)
    js = json.loads(subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True).stdout)
    assert set(js) == set(shader.surfaces())
    for surface in shader.surfaces():
        source = shader.assemble(surface)
        for k, on in enumerate((False, True)):
            py = shader.specialize(source, periodic=on, nested=not on,
                                   tone_map=on, light_cache=not on,
                                   sky_probe=on)
            assert js[surface][k] == py, surface
