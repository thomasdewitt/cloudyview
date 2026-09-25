"""The in-flight surface switch (G) prepares the variant it will fly.

renderer.js prepareSurface builds the target surface's pipelines before
commitSurface stands the scene on it. The desert's tier axes (DESERT_LITE at
Low/Minimal, DESERT_FLORA_CUT at Medium) belong to the surface being
prepared, not the one the scene still stands on: derived from the scene, a
desert-to-ocean switch at Low or Medium asked for a lite or cut ocean and
threw, and an ocean-to-desert switch prewarmed a full-desert variant that the
first frame then did not use.

For every tier and both directions, with the real shader and the real
specializeShader: prepareSurface succeeds, and the (entry, variant) pairs it
builds are exactly those the first frame after the switch draws with.
Skips without node.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"

_JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    import * as K from "%s";
    import { Renderer, specializeShader, shaderVariantKey } from "%s";

    const source = readFileSync("%s", "utf8");
    function renderer(scene, tier) {
      const r = Object.create(Renderer.prototype);
      r.scene = scene; r.qualityTier = tier; r.periodic = true;
      r.maxLightSteps = 512; r.lightCacheMode = "auto";
      r.skyProbeMode = "auto"; r._lightCacheReady = true;
      r.shaderSource = source;
      return r;
    }
    const out = {};
    for (const tier of K.QUALITY_TIER_NAMES) {
      for (const [from, to] of [["ocean", "desert"], ["desert", "ocean"]]) {
        const r = renderer({ desert: from === "desert" }, tier);
        const built = [];
        r._pipeline = async (entry, format, spec) => {
          specializeShader(r.shaderSource, spec);   // throws as flight would
          built.push([entry, shaderVariantKey(spec)]);
        };
        let error = null;
        try { await r.prepareSurface(to); }
        catch (e) { error = String(e?.message || e); }
        // The first frame after commitSurface: the scene on `to`.
        r.scene = { desert: to === "desert" };
        const key = shaderVariantKey(r._shaderSpec());
        const used = [["fs_main", key]];
        if (r._hasSurfacePass()) {
          for (const entry of r._stageEntries()) used.push([entry, key]);
        }
        out[`${tier}/${from}->${to}`] = { error, built, used };
      }
    }
    console.log(JSON.stringify(out));
""") % (CONSTANTS_JS.as_uri(), RENDERER_JS.as_uri(), SHADER.as_posix())


@pytest.fixture(scope="module")
def js():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    out = subprocess.run(["node", "--input-type=module", "-e", _JS],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_every_switch_prepares(js):
    assert len(js) == 10   # five tiers, two directions
    for case, got in js.items():
        assert got["error"] is None, (case, got["error"])


def test_prepared_is_what_flies(js):
    for case, got in js.items():
        assert sorted(map(tuple, got["built"])) == sorted(
            map(tuple, got["used"])), case
