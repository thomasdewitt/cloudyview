"""The live view's steam (passes.wgsl fs_steam_live): the altitude above
which none can be in view, and so above which a parked city view sleeps
(constants.js LIVE_STEAM_CEILING_M), is the steam's top plus its range."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _const(text, name):
    m = re.search(rf"\b{name}\s*(?::\s*f32)?\s*=\s*([0-9.]+)", text)
    assert m, f"{name} not found"
    return float(m.group(1))


def test_the_sleep_ceiling_is_the_steam_top_plus_its_range():
    steam = (ROOT / "cloudyview/soar/shader/city/components/manholesteam.wgsl").read_text()
    consts = (ROOT / "web/soar/constants.js").read_text()
    assert _const(consts, "LIVE_STEAM_CEILING_M") == (
        _const(steam, "cc_manholesteam_ZTOP")
        + _const(steam, "cc_manholesteam_RANGE"))


def test_live_steam_is_drawn_at_medium_and_up_and_only_in_the_city():
    """The live steam costs ~0.15 ms a frame. Medium, High and Max draw it
    live; Low and Minimal march it, so there it pauses while parked and a parked view
    sleeps. Asked of the renderer's own getter (Renderer.liveSteamActive),
    which drawFrame, the viewer's clock and its sleep test all read."""
    import json
    import shutil
    import subprocess
    import pytest
    if shutil.which("node") is None:
        pytest.skip("needs node")
    js = f"""
        import * as K from "{(ROOT / 'web/soar/constants.js').as_posix()}";
        import {{ Renderer }} from "{(ROOT / 'web/soar/renderer.js').as_posix()}";
        const get = Object.getOwnPropertyDescriptor(Renderer.prototype,
                                                    "liveSteamActive").get;
        const out = {{}};
        for (const tier of K.QUALITY_TIER_NAMES) {{
          out[tier] = [true, false].map((city) =>
            get.call({{ qualityTier: tier, scene: {{ city }} }}));
        }}
        process.stdout.write(JSON.stringify({{ out, tiers: K.QUALITY_TIER_NAMES }}));
    """
    proc = subprocess.run(["node", "--input-type=module", "-e", js],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout)
    assert sorted(got["tiers"]) == ["high", "low", "max", "medium", "minimal"]
    assert {t: v[0] for t, v in got["out"].items()} == {
        "max": True, "high": True, "medium": True,
        "low": False, "minimal": False}
    assert not any(v[1] for v in got["out"].values()), \
        "live steam off the city"
