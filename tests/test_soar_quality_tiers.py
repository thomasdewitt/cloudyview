"""The per-tier settings tables, and the direction each one runs in.

Four tables are keyed by quality tier — parked samples, motion-smoothing
floor, default haze, default LOD strength — and every one of them encodes a
direction: the cheaper the tier, the fewer samples it parks at, the deeper it
is allowed to smooth, the thicker its air, the coarser its far field. A table
that is merely *present* for every tier is not enough; one written the wrong
way round is exactly the bug this suite exists to catch, because it is what
happened to the motion-smoothing slider (it was labelled "Motion smoothing"
and carried the blend alpha, which runs the other way, so turning it up
smoothed less — Thomas, 2026-08-14).

So: the tables are complete, they are monotone in the direction they claim,
and the two conversions between smoothing and alpha are exact inverses.

Also pins HAZE_MAX across the JS/Python boundary. The uniform parity test
diffs packed VALUES; it cannot see a limit that only one side widened, and a
browser that lets the slider reach 2.5 while soar_host raises above 2.0 would
put the app and the terminal command it prints out of step.

Skips only without node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import look

REPO = Path(__file__).resolve().parents[1]
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"
SPECTRAL_JS = REPO / "web" / "soar" / "spectral.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not CONSTANTS_JS.exists(),
    reason="needs node and web/soar/constants.js")

# Cheapest first, which is the direction every table below is monotone in.
# Max is not on this ladder — it is High marched eight times per frame, it is
# never auto-selected, and it is asserted about separately.
TIERS = ["minimal", "low", "medium", "high"]
ALL_TIERS = TIERS + ["max"]


_JS = textwrap.dedent("""
    import * as K from "%s";
    import { hazeEFoldingKm } from "%s";

    const tiers = %s;
    const smoothingSweep = [0, 0.25, 0.5, 0.75, 1.0];
    const hazeSweep = [0.0, 0.35, 1.0, 1.6, 2.0, 2.5];

    process.stdout.write(JSON.stringify({
      tierNames: K.QUALITY_TIER_NAMES,
      cheapestFirst: K.QUALITY_TIERS_CHEAPEST_FIRST,
      parked: K.PARKED_ACCUM_FRAMES_BY_TIER,
      stillFrames: K.STILL_ACCUMULATE_FRAMES,
      alphaFloor: K.MOTION_ALPHA_FLOOR_BY_TIER,
      alphaAtZero: K.MOTION_ALPHA_AT_ZERO_SMOOTHING,
      defaultSmoothing: K.DEFAULT_MOTION_SMOOTHING_BY_TIER,
      legacyAlpha: K.DEFAULT_MOTION_BLEND_ALPHA,
      defaultHaze: K.DEFAULT_HAZE_BY_TIER,
      hazeMax: K.HAZE_MAX,
      defaultLod: K.DEFAULT_LOD_STRENGTH_BY_TIER,
      lodLimits: K.LOD_STRENGTH_LIMITS,
      lodDegrees: [K.APP_LIGHT_MARCH_LOD_DEGREES, K.APP_VIEW_STEP_LOD_DEGREES],
      holdModes: K.HOLD_MODES,
      defaultHoldMode: K.DEFAULT_HOLD_MODE,
      defaultPreview: K.DEFAULT_QUALITY_PREVIEW,
      presets: K.QUALITY_PRESETS,
      ladders: K.QUALITY_HOLD_LADDERS,
      minRenderScale: K.MIN_RENDER_SCALE,
      alphaFor: Object.fromEntries(tiers.map((t) => [
        t, smoothingSweep.map((s) => K.motionAlphaForSmoothing(s, t))])),
      roundTrip: Object.fromEntries(tiers.map((t) => [
        t, smoothingSweep.map((s) => K.motionSmoothingForAlpha(
          K.motionAlphaForSmoothing(s, t), t))])),
      smoothingSweep,
      hazeSweep,
      eFolding: hazeSweep.map((h) => hazeEFoldingKm(h)),
    }));
""") % (CONSTANTS_JS.as_posix(), SPECTRAL_JS.as_posix(), json.dumps(ALL_TIERS))


@pytest.fixture(scope="module")
def js():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_every_tier_appears_in_every_per_tier_table(js):
    assert sorted(js["tierNames"]) == sorted(ALL_TIERS)
    assert js["cheapestFirst"] == TIERS
    for table in ("parked", "alphaFloor", "defaultSmoothing", "defaultHaze",
                  "defaultLod", "presets", "ladders"):
        assert sorted(js[table]) == sorted(ALL_TIERS), table


def test_max_is_reachable_only_by_hand(js):
    """Eight marches a frame is not a choice to make on someone's behalf."""
    assert "max" not in js["cheapestFirst"]
    assert js["presets"]["max"]["sppPerFrame"] == 8
    assert all(js["presets"][t]["sppPerFrame"] == 1 for t in TIERS)
    # Same picture as High, paid for sooner: one is the other's sampling.
    for key in ("renderScale", "stepFactor", "lightStepFactor"):
        assert js["presets"]["max"][key] == js["presets"]["high"][key], key
    assert js["parked"]["max"] == js["parked"]["high"]


def test_flight_scales_fall_with_the_tier_and_stay_renderable(js):
    scales = [js["presets"][t]["renderScale"] for t in TIERS]
    assert scales == sorted(scales)
    assert min(scales) >= js["minRenderScale"]


def test_hold_ladders_climb_from_the_flight_scale_to_the_tier_ceiling(js):
    """Every rung is dearer than the one below and the top samples like High.

    The ceiling is per tier since 2026-08-14 — Minimal converges at half
    scale, Low at three quarters — so this no longer asserts that every tier
    reaches 1.0. What it does assert is that the climb is monotone and that
    the top rung is sampled like High wherever there is a climb at all,
    because a settled picture must not be stepped like the tier that flew it.
    """
    for tier in ALL_TIERS:
        rungs = js["ladders"][tier]
        scales = [js["presets"][tier]["renderScale"]] + [r["scale"] for r in rungs]
        assert scales == sorted(scales), tier
        assert len(set(scales)) == len(scales), tier
        if rungs:
            assert rungs[-1]["sampling"] == "high", tier
    assert js["ladders"]["minimal"][-1]["scale"] == 0.5
    assert js["ladders"]["low"][-1]["scale"] == 0.75
    assert js["ladders"]["medium"][-1]["scale"] == 1.0
    assert js["ladders"]["high"] == js["ladders"]["max"] == []


def test_parked_samples_are_capped_and_rise_with_the_tier(js):
    parked = js["parked"]
    assert parked["minimal"] == 8
    assert parked["high"] == 32
    counts = [parked[t] for t in TIERS]
    assert counts == sorted(counts)
    # A capture is explicit and waited on; a park happens every time you stop.
    assert max(counts) < js["stillFrames"]


def test_more_smoothing_means_a_smaller_alpha(js):
    """The regression this whole rework exists to prevent."""
    for tier in TIERS:
        alphas = js["alphaFor"][tier]
        assert alphas == sorted(alphas, reverse=True), tier
        assert alphas[0] == pytest.approx(js["alphaAtZero"])
        assert alphas[-1] == pytest.approx(js["alphaFloor"][tier])


def test_smoothing_and_alpha_are_exact_inverses(js):
    for tier in TIERS:
        assert js["roundTrip"][tier] == pytest.approx(js["smoothingSweep"])


def test_cheaper_tiers_may_smooth_deeper(js):
    floors = [js["alphaFloor"][t] for t in TIERS]
    assert floors == sorted(floors)                    # cheapest smooths most
    # Minimal reaches twice High's blend depth, which goes as 1/alpha.
    assert js["alphaFloor"]["high"] / js["alphaFloor"]["minimal"] == pytest.approx(2.0)


def test_high_keeps_the_motion_blend_it_has_always_had(js):
    """Changing the units must not change the look on the reference tier."""
    high = js["alphaFor"]["high"]
    at_default = js["defaultSmoothing"]["high"]
    span = js["alphaAtZero"] - js["alphaFloor"]["high"]
    assert js["alphaAtZero"] - at_default * span == pytest.approx(
        js["legacyAlpha"], abs=0.005)
    assert high[0] > js["legacyAlpha"] > high[-1]


def test_cheaper_tiers_get_thicker_air_and_coarser_distance(js):
    hazes = [js["defaultHaze"][t] for t in TIERS]
    lods = [js["defaultLod"][t] for t in TIERS]
    assert hazes == sorted(hazes, reverse=True)
    assert lods == sorted(lods, reverse=True)
    assert max(hazes) <= js["hazeMax"]
    lo, hi = js["lodLimits"]
    assert all(lo <= v <= hi for v in lods)
    # The shader rejects a LOD angle at or past 45 degrees, and the slider
    # multiplies the tuned ones, so the top of its range has to stay legal.
    assert max(js["lodDegrees"]) * hi < 45.0


def test_haze_reads_out_as_a_shrinking_distance(js):
    e_folding = js["eFolding"]
    assert e_folding == sorted(e_folding, reverse=True)
    # The two anchors quoted in look.py's own comments.
    assert e_folding[js["hazeSweep"].index(1.0)] == pytest.approx(9.0, abs=0.1)
    assert e_folding[js["hazeSweep"].index(0.0)] == pytest.approx(
        1.0 / look.AERIAL_BETA_FLOOR_PER_KM)
    for haze, km in zip(js["hazeSweep"], e_folding):
        assert km == pytest.approx(1.0 / look.aerial_beta_per_km(haze))


def test_haze_ceiling_matches_python(js):
    assert js["hazeMax"] == look.HAZE_MAX


def test_flying_gets_stills_and_the_quality_panel_gets_the_live_view(js):
    """Live is the panel's preview, not the session's mode."""
    assert js["defaultHoldMode"] == "still"
    assert js["defaultPreview"] == "live"
    assert sorted(js["holdModes"]) == ["live", "still"]
