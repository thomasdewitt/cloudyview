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
    import { hazeEFoldingKm, HAZE_MIN } from "%s";
    import { Renderer } from "%s";

    // The real builder, which is pure — no device, no textures.
    function ladderFor(tier, holdScale) {
      const r = Object.create(Renderer.prototype);
      r.qualityTier = tier;
      r._flightRenderScale = K.QUALITY_PRESETS[tier].renderScale;
      r._holdRenderScale = holdScale ?? K.HOLD_RENDER_SCALE_BY_TIER[tier];
      r._buildHoldLadder();
      return r._holdLadder;
    }

    const tiers = %s;
    const smoothingSweep = [0, 0.25, 0.5, 0.75, 1.0];
    const hazeSweep = [0.0, 0.35, 1.0, 1.6, 2.0, 2.5];

    process.stdout.write(JSON.stringify({
      tierNames: K.QUALITY_TIER_NAMES,
      cheapestFirst: K.QUALITY_TIERS_CHEAPEST_FIRST,
      parked: K.PARKED_ACCUM_FRAMES_BY_TIER,
      parkedSppLimits: K.PARKED_SPP_LIMITS,
      alphaFloor: K.MOTION_ALPHA_FLOOR_BY_TIER,
      alphaAtZero: K.MOTION_ALPHA_AT_ZERO_SMOOTHING,
      defaultSmoothing: K.DEFAULT_MOTION_SMOOTHING_BY_TIER,
      legacyAlpha: K.DEFAULT_MOTION_BLEND_ALPHA,
      defaultHaze: K.DEFAULT_HAZE_BY_TIER,
      hazeMax: K.HAZE_MAX,
      hazeMin: HAZE_MIN,
      hazeMaxEFoldingKm: K.HAZE_MAX_E_FOLDING_KM,
      defaultLod: K.DEFAULT_LOD_STRENGTH_BY_TIER,
      lodLimits: K.LOD_STRENGTH_LIMITS,
      lodDegrees: [K.APP_LIGHT_MARCH_LOD_DEGREES, K.APP_VIEW_STEP_LOD_DEGREES],
      holdModes: K.HOLD_MODES,
      defaultHoldMode: K.DEFAULT_HOLD_MODE,
      defaultPreview: K.DEFAULT_QUALITY_PREVIEW,
      presets: K.QUALITY_PRESETS,
      holdRungs: K.QUALITY_HOLD_RUNGS,
      holdScale: K.HOLD_RENDER_SCALE_BY_TIER,
      minRenderScale: K.MIN_RENDER_SCALE,
      alphaFor: Object.fromEntries(tiers.map((t) => [
        t, smoothingSweep.map((s) => K.motionAlphaForSmoothing(s, t))])),
      roundTrip: Object.fromEntries(tiers.map((t) => [
        t, smoothingSweep.map((s) => K.motionSmoothingForAlpha(
          K.motionAlphaForSmoothing(s, t), t))])),
      smoothingSweep,
      hazeSweep,
      eFolding: hazeSweep.map((h) => hazeEFoldingKm(h)),
      built: Object.fromEntries(tiers.map((t) => [t, ladderFor(t)])),
      // Minimal, with the hold slider dragged to and below its flight scale.
      handSet: [ladderFor("minimal", 0.125), ladderFor("minimal", 0.1)]
        .map((rungs) => rungs.map((r) => r.scale)),
    }));
""") % (CONSTANTS_JS.as_posix(), SPECTRAL_JS.as_posix(),
        (REPO / "web" / "soar" / "renderer.js").as_posix(),
        json.dumps(ALL_TIERS))


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
                  "defaultLod", "presets", "holdRungs", "holdScale"):
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


def test_hold_ladders_climb_from_the_flight_scale_to_the_hold_scale(js):
    """The composed ladder, as Renderer._buildHoldLadder actually builds it.

    Every rung is dearer than the one below, the top one samples like High
    wherever there is a climb at all (a settled picture must not be stepped
    like the tier that flew it), and the climb ends at the hold scale — which
    is per tier since 2026-08-14 and a slider since the same day, so this
    asserts a ceiling rather than "everything reaches 1.0".
    """
    for tier in ALL_TIERS:
        rungs = js["built"][tier]
        scales = [r["scale"] for r in rungs]
        assert scales[0] == js["presets"][tier]["renderScale"], tier
        assert scales == sorted(scales), tier
        assert len(set(scales)) == len(scales), tier
        assert scales[-1] == max(
            js["presets"][tier]["renderScale"], js["holdScale"][tier]), tier
        if len(rungs) > 1:
            assert rungs[-1]["stepFactor"] == js["presets"]["high"]["stepFactor"], tier
    # The 2026-08-18 hand-tuned defaults (Thomas's panel values).
    assert [r["scale"] for r in js["built"]["minimal"]] == [0.125, 0.25]
    assert [r["scale"] for r in js["built"]["low"]] == [0.3, 0.5]
    assert [r["scale"] for r in js["built"]["medium"]] == [0.7, 1.0]
    assert len(js["built"]["high"]) == len(js["built"]["max"]) == 1


def test_a_hold_scale_at_or_below_the_flight_scale_collapses_the_ladder(js):
    """Asking for a still no sharper than the flight is a hold that only
    accumulates — not a rung that climbs downward."""
    for scales in js["handSet"]:
        assert scales == [0.125], scales


def test_parked_samples_are_capped_and_rise_with_the_tier(js):
    parked = js["parked"]
    assert parked["minimal"] == 8
    assert parked["high"] == 32
    counts = [parked[t] for t in TIERS]
    assert counts == sorted(counts)
    # A capture's spp IS the tier's parked count now (Thomas, 2026-08-20),
    # and the Advanced override slider must be able to reach every default.
    lo, hi = js["parkedSppLimits"]
    assert lo <= min(counts) and max(counts) <= hi


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


def test_high_smoothing_default_stays_inside_its_own_range(js):
    """High's default was pinned to the pre-slider legacy alpha until
    2026-08-18, when Thomas retuned every tier by hand — the hand-set values
    supersede the legacy-look guarantee. What must still hold: the default
    maps inside the tier's own alpha range, so the slider can move it both
    ways."""
    high = js["alphaFor"]["high"]
    at_default = js["defaultSmoothing"]["high"]
    span = js["alphaAtZero"] - js["alphaFloor"]["high"]
    alpha = js["alphaAtZero"] - at_default * span
    assert high[0] > alpha > high[-1]


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


def test_haze_clear_end_matches_python(js):
    """Both ends of the slider, and the distance the clear one is derived from.

    HAZE_MIN is not typed on either side — it is the inverse of the extinction
    ramp evaluated at HAZE_MAX_E_FOLDING_KM — so this pins the inverse as much
    as the constant. A host that got the 2/3 power or the sign wrong lands
    somewhere plausible rather than somewhere obviously broken.
    """
    assert js["hazeMaxEFoldingKm"] == look.HAZE_MAX_E_FOLDING_KM
    assert js["hazeMin"] == pytest.approx(look.HAZE_MIN, rel=1e-12)
    # And it means what it says: that haze is 200 km of e-folding.
    assert 1.0 / look.aerial_beta_per_km(look.HAZE_MIN) == pytest.approx(
        look.HAZE_MAX_E_FOLDING_KM, rel=1e-12)


def test_flying_gets_stills_and_the_quality_panel_gets_the_live_view(js):
    """Live is the panel's preview, not the session's mode."""
    assert js["defaultHoldMode"] == "still"
    assert js["defaultPreview"] == "live"
    assert sorted(js["holdModes"]) == ["live", "still"]
