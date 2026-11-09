"""The live view's steam (raymarch.wgsl fs_steam_live): the altitude above
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
    steam = (ROOT / "cloudyview/soar/city/components/manholesteam.wgsl").read_text()
    consts = (ROOT / "web/soar/constants.js").read_text()
    assert _const(consts, "LIVE_STEAM_CEILING_M") == (
        _const(steam, "cc_manholesteam_ZTOP")
        + _const(steam, "cc_manholesteam_RANGE"))
