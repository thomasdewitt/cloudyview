"""The surface stages are one ordered list per surface, in both hosts.

soar_host.SURFACE_STAGES and web/soar/constants.js SURFACE_STAGES name the
raymarch.wgsl entries each traced surface runs before the march; both hosts
chain them the same way (surface_stage_plan / surfaceStagePlan) through two
alternating intermediates. What would go wrong silently is pinned here:

- the two hosts' tables and chain plans agree;
- every chain is well formed: the last stage writes the surface the march
  reads, each later stage reads what the one before wrote, none reads what
  it writes;
- every listed entry is a fragment entry of the shader, with a veil output
  exactly when it is listed in VEIL_STAGES, and the desert's optical depths
  exactly for TAUS_STAGE, which comes right before the stage that reads
  them (TAUS_READER);
- the renderer runs, per scene and tier, the stages the Python host runs.

The node half skips without node.
"""

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import soar_host
from cloudyview.soar_host import (SURFACE_STAGES, SURFACE_STAGES_LITE,
                                  TAUS_READER, TAUS_STAGE, VEIL_STAGES,
                                  surface_stage_entries, surface_stage_plan)

REPO = Path(__file__).resolve().parents[1]
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"


@pytest.mark.parametrize("n", range(0, 6))
def test_plan_chains(n):
    plan = surface_stage_plan(n)
    assert len(plan) == n
    if n:
        assert plan[-1][0] == "final"
        assert plan[0][1] is None
    for k in range(1, n):
        assert plan[k][1] == plan[k - 1][0], (n, k)
    for writes, reads in plan:
        assert writes != reads
        assert writes in ("final", "a", "b")


def _fragment_outputs():
    """{fragment entry: its return type} from the shader."""
    src = soar_host.read_shader()
    return {m.group(1): " ".join(m.group(2).split())
            for m in re.finditer(r"@fragment\s+fn\s+(\w+)\s*\(.*?\)\s*->\s*"
                                 r"([^{]+)\{", src, re.S)}


def test_every_stage_is_a_fragment_entry_with_the_right_outputs():
    outputs = _fragment_outputs()
    listed = {e for v in SURFACE_STAGES.values() for e in v}
    listed |= set(SURFACE_STAGES_LITE) | set(VEIL_STAGES)
    for entry in listed:
        assert entry in outputs, f"{entry} is not a fragment entry"
        veil = outputs[entry].startswith("SurfaceOut")
        assert veil == (entry in VEIL_STAGES), (entry, outputs[entry])
        taus = outputs[entry].startswith("DesertSurfaceOut")
        assert taus == (entry == TAUS_STAGE), (entry, outputs[entry])


def test_the_optical_depths_reach_their_reader():
    # Written by the stage right before the one that reads them: the chain
    # alternates two intermediates, but the depths have a target of their
    # own, so only an intervening writer could clobber them — and none may
    # write it.
    for surface, entries in SURFACE_STAGES.items():
        if TAUS_STAGE in entries:
            k = entries.index(TAUS_STAGE)
            assert entries[k + 1] == TAUS_READER, surface
    assert TAUS_STAGE in SURFACE_STAGES["desert"]
    assert TAUS_STAGE not in SURFACE_STAGES_LITE


def test_every_chain_starts_at_a_first_stage():
    for surface, entries in SURFACE_STAGES.items():
        assert entries[0] == ("fs_surface_desert" if surface == "desert"
                              else "fs_surface"), surface
    assert SURFACE_STAGES_LITE == ("fs_surface",)
    assert surface_stage_entries("ocean") == ()
    assert surface_stage_entries("desert", desert_lite=True) == ("fs_surface",)
    with pytest.raises(ValueError):
        surface_stage_entries("city", desert_lite=True)


_JS = textwrap.dedent("""
    import * as K from "%s";
    import { Renderer, surfaceStagePlan } from "%s";

    function renderer(scene, tier) {
      const r = Object.create(Renderer.prototype);
      r.scene = scene; r.qualityTier = tier;
      return r;
    }
    const out = {
      stages: K.SURFACE_STAGES, lite: K.SURFACE_STAGES_LITE,
      veil: K.VEIL_STAGES, tausStage: K.TAUS_STAGE, tausReader: K.TAUS_READER,
      plans: {}, entries: {},
    };
    for (let n = 0; n < 6; n++) out.plans[n] = surfaceStagePlan(n);
    for (const [name, scene] of [["desert", { desert: true }],
                                 ["city", { city: true }],
                                 ["ocean", {}]]) {
      for (const tier of K.QUALITY_TIER_NAMES) {
        out.entries[`${name}/${tier}`] = renderer(scene, tier)._stageEntries(tier);
      }
    }
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


def test_hosts_agree_on_the_tables(js):
    assert {k: tuple(v) for k, v in js["stages"].items()} == SURFACE_STAGES
    assert tuple(js["lite"]) == SURFACE_STAGES_LITE
    assert tuple(js["veil"]) == VEIL_STAGES
    assert (js["tausStage"], js["tausReader"]) == (TAUS_STAGE, TAUS_READER)


def test_hosts_agree_on_the_plan(js):
    for n in range(6):
        assert [tuple(p) for p in js["plans"][str(n)]] == surface_stage_plan(n)


def test_renderer_runs_the_python_hosts_stages(js):
    for case, entries in js["entries"].items():
        surface, tier = case.split("/")
        lite = surface == "desert" and tier in soar_host.DESERT_LITE_TIERS
        assert tuple(entries) == surface_stage_entries(surface, lite), case
