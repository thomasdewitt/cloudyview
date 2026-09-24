"""The simplified desert is a tier-keyed specialization, like the lighting.

DESERT_LITE (raymarch.wgsl) is compiled into a desert scene at the tiers in
DESERT_LITE_TIERS, and under it the surface pass has no second stage. What
this pins is the host side, in both hosts, because a mismatch there is
silent: a key without the axis would serve the full desert's pipeline to a
Low frame (or the reverse), and a refine pass left running under lite would
encode a stage the shader compiled out.

- The two hosts agree on which tiers are lite.
- specialize() / specializeShader() turn the sentinel on only over a desert,
  demand it when turned on, and refuse it anywhere else.
- The renderer derives it from the tier, keys every cache on it, and drops
  the refine stage under it — for a desert only.

The node half skips without node.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import soar_host

REPO = Path(__file__).resolve().parents[1]
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"

# The sentinels specialize() always rewrites, plus the two this is about.
_SOURCE = "\n".join([
    "const PERIODIC_DOMAIN: bool = true;",
    "const NESTED: bool = false;",
    "const MAX_LIGHT_STEPS: i32 = 512;",
    "const TONE_MAP: bool = true;",
    "const DESERT: bool = false;",
    "const DESERT_LITE: bool = false;",
])
_ARGS = dict(periodic=True, nested=False, max_light_steps=512)


def test_lite_turns_on_over_a_desert():
    out = soar_host.specialize(_SOURCE, surface="desert", desert_lite=True,
                               **_ARGS)
    assert "const DESERT_LITE: bool = true;" in out
    assert "const DESERT: bool = true;" in out
    full = soar_host.specialize(_SOURCE, surface="desert", **_ARGS)
    assert "const DESERT_LITE: bool = false;" in full


def test_lite_refused_off_the_desert():
    with pytest.raises(ValueError):
        soar_host.specialize(_SOURCE, surface="ocean", desert_lite=True,
                             **_ARGS)


def test_lite_demands_its_sentinel():
    without = _SOURCE.replace("const DESERT_LITE: bool = false;", "")
    with pytest.raises(RuntimeError):
        soar_host.specialize(without, surface="desert", desert_lite=True,
                             **_ARGS)


_JS = textwrap.dedent("""
    import * as K from "%s";
    import { Renderer, specializeShader, shaderVariantKey } from "%s";

    function renderer(scene, tier) {
      const r = Object.create(Renderer.prototype);
      r.scene = scene; r.qualityTier = tier; r.periodic = true;
      r.maxLightSteps = 512; r.lightCacheMode = "auto";
      r.skyProbeMode = "auto"; r._lightCacheReady = true;
      return r;
    }
    const out = { tiers: K.DESERT_LITE_TIERS, cases: {} };
    for (const [name, scene] of [["desert", { desert: true }],
                                 ["city", { city: true }],
                                 ["ocean", {}]]) {
      for (const tier of K.QUALITY_TIER_NAMES) {
        const r = renderer(scene, tier);
        out.cases[`${name}/${tier}`] = {
          lite: r._desertLiteFor(tier),
          refine: r._hasRefineStage(tier),
          // One fixed lighting method, so the key varies only by this axis.
          key: shaderVariantKey(r._shaderSpec({
            tier, lighting: { lightCache: true, skyProbe: false } })),
        };
      }
    }
    let threw = false;
    try {
      specializeShader("", { periodic: true, nested: false,
                             maxLightSteps: 512, desertLite: true,
                             lightCache: false, skyProbe: true });
    } catch (e) { threw = true; }
    out.liteWithoutDesertThrows = threw;
    console.log(JSON.stringify(out));
""") % (CONSTANTS_JS.as_uri(), RENDERER_JS.as_uri())


@pytest.fixture(scope="module")
def js():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _JS],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_hosts_agree_on_the_tiers(js):
    assert tuple(js["tiers"]) == soar_host.DESERT_LITE_TIERS


def test_renderer_keys_and_stages(js):
    lite = set(soar_host.DESERT_LITE_TIERS)
    for case, got in js["cases"].items():
        scene, tier = case.split("/")
        want_lite = scene == "desert" and tier in lite
        assert got["lite"] == want_lite, case
        want_refine = scene == "city" or (scene == "desert" and not want_lite)
        assert got["refine"] == want_refine, case
    # The axis is in the key: a Low desert and a High desert never share a
    # pipeline, and two lite tiers do.
    desert = {t: js["cases"][f"desert/{t}"]["key"] for t in
              ("high", "low", "minimal")}
    assert desert["low"] != desert["high"]
    assert desert["low"] == desert["minimal"]
    # Off the desert the tier does not reach the key through this axis.
    assert (js["cases"]["city/low"]["key"]
            == js["cases"]["city/high"]["key"])


def test_lite_without_desert_throws(js):
    assert js["liteWithoutDesertThrows"]
