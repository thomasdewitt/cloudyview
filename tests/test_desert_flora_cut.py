"""The flora's explicit cut is a tier-keyed specialization, like DESERT_LITE.

DESERT_FLORA_CUT (raymarch.wgsl) scales how far out the full desert walks
explicit plants before the far forms take over; the hosts set it from the
tier (DESERT_FLORA_CUT_BY_TIER). What this pins is the host side, in both
hosts, because a mismatch there is silent: a key without the axis would
serve High's pipeline to a Medium frame, and a host with its own table
would fly a different flora from the one the other renders.

- The two hosts agree on the table.
- specialize() / specializeShader() rewrite the sentinel only when the cut
  departs from the declared 1.0, demand it then, and refuse it off the
  desert.
- The renderer derives it from the tier and keys every cache on it; off the
  desert the tier does not reach the key through it.

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

_SENTINEL = "const DESERT_FLORA_CUT: f32 = 1.0;"
_SOURCE = "\n".join([
    "const PERIODIC_DOMAIN: bool = true;",
    "const NESTED: bool = false;",
    "const MAX_LIGHT_STEPS: i32 = 512;",
    "const TONE_MAP: bool = true;",
    "const DESERT: bool = false;",
    "const DESERT_LITE: bool = false;",
    _SENTINEL,
])
_ARGS = dict(periodic=True, nested=False, max_light_steps=512)
# renderer.js always rewrites the lighting method too.
_JS_SOURCE = "\n".join([_SOURCE, "const LIGHT_CACHE: bool = false;",
                        "const SKY_PROBE: bool = true;"])


def test_cut_rewrites_over_a_desert():
    out = soar_host.specialize(_SOURCE, surface="desert",
                               desert_flora_cut=0.5, **_ARGS)
    assert "const DESERT_FLORA_CUT: f32 = 0.5;" in out
    full = soar_host.specialize(_SOURCE, surface="desert", **_ARGS)
    assert _SENTINEL in full


def test_cut_refused_off_the_desert():
    with pytest.raises(ValueError):
        soar_host.specialize(_SOURCE, surface="ocean", desert_flora_cut=0.5,
                             **_ARGS)


def test_cut_must_be_positive():
    with pytest.raises(ValueError):
        soar_host.specialize(_SOURCE, surface="desert", desert_flora_cut=0.0,
                             **_ARGS)


def test_cut_demands_its_sentinel_only_when_used():
    without = _SOURCE.replace(_SENTINEL, "")
    # A shader from before the axis still serves every tier at 1.0 ...
    soar_host.specialize(without, surface="desert", **_ARGS)
    # ... and refuses one that asks for anything else.
    with pytest.raises(RuntimeError):
        soar_host.specialize(without, surface="desert", desert_flora_cut=0.5,
                             **_ARGS)


def test_every_tier_has_a_cut():
    tiers = set(soar_host.DESERT_FLORA_CUT_BY_TIER)
    assert tiers == {"max", "high", "medium", "low", "minimal"}
    # The lite tiers ignore it; at 1.0 their key does not move.
    for t in soar_host.DESERT_LITE_TIERS:
        assert soar_host.DESERT_FLORA_CUT_BY_TIER[t] == 1.0


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
    const out = { table: K.DESERT_FLORA_CUT_BY_TIER, cases: {} };
    for (const [name, scene] of [["desert", { desert: true }],
                                 ["city", { city: true }],
                                 ["ocean", {}]]) {
      for (const tier of K.QUALITY_TIER_NAMES) {
        const r = renderer(scene, tier);
        out.cases[`${name}/${tier}`] = {
          cut: r._desertFloraCutFor(tier),
          // One fixed lighting method, so the key varies only by the tier's
          // desert axes.
          key: shaderVariantKey(r._shaderSpec({
            tier, lighting: { lightCache: true, skyProbe: false } })),
        };
      }
    }
    const base = { periodic: true, nested: false, maxLightSteps: 512,
                   lightCache: false, skyProbe: true };
    const src = "%s";
    out.rewritten = specializeShader(src, { ...base, desert: true,
                                            desertFloraCut: 0.5 });
    out.untouched = specializeShader(src, { ...base, desert: true });
    let threw = false;
    try { specializeShader(src, { ...base, desertFloraCut: 0.5 }); }
    catch (e) { threw = true; }
    out.cutWithoutDesertThrows = threw;
    console.log(JSON.stringify(out));
""") % (CONSTANTS_JS.as_uri(), RENDERER_JS.as_uri(),
        _JS_SOURCE.replace("\n", "\\n"))


@pytest.fixture(scope="module")
def js():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _JS],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_hosts_agree_on_the_table(js):
    assert js["table"] == soar_host.DESERT_FLORA_CUT_BY_TIER


def test_renderer_keys_on_the_cut(js):
    for case, got in js["cases"].items():
        scene, tier = case.split("/")
        want = (soar_host.DESERT_FLORA_CUT_BY_TIER[tier]
                if scene == "desert" else 1.0)
        assert got["cut"] == want, case
    # Medium and High deserts differ by the cut (the lighting is held fixed
    # here, so nothing else separates them) ...
    desert = {t: js["cases"][f"desert/{t}"]["key"] for t in
              ("max", "high", "medium")}
    assert desert["medium"] != desert["high"]
    assert desert["max"] == desert["high"]
    # ... and off the desert the tier does not reach the key through it.
    assert (js["cases"]["city/medium"]["key"]
            == js["cases"]["city/high"]["key"])
    assert (js["cases"]["ocean/medium"]["key"]
            == js["cases"]["ocean/high"]["key"])


def test_js_specializes_like_python(js):
    assert "const DESERT_FLORA_CUT: f32 = 0.5;" in js["rewritten"]
    assert _SENTINEL in js["untouched"]
    assert js["cutWithoutDesertThrows"]
