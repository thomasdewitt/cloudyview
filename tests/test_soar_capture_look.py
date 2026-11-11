"""A capture at a named tier is a parked view at that tier, full stop.

The ruling (Thomas, 2026-09-01): a capture at tier T must march exactly what
a parked (held) T view converges to, at the capture's explicit pixel size —
the tier's haze default, its parked LOD (the city's quarter-pixel floor over
a city), a once-metered exposure, and the hold ladder's top-rung sampling
(High's step factors for every tier). Hand-set look values no longer silently
mix into named tiers; they live in the capture panels' "Custom" entry, which
carries the session's values verbatim. And over the city only, a parked view
marches finer LOD than a moving one.

This pins those contracts under node against the real viewer and capture
modules, with stub renderers — no GPU, no DOM — plus the soar/tiers.py mirror,
whose --quality presets must carry the same parked stepping.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview.soar.tiers import QUALITY_PRESETS

REPO = Path(__file__).resolve().parents[1]
VIEWER_JS = (REPO / "web" / "soar" / "viewer.js").as_posix()
RENDERER_JS = (REPO / "web" / "soar" / "renderer.js").as_posix()
CONSTANTS_JS = (REPO / "web" / "soar" / "constants.js").as_posix()

needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="needs node")


_JS = textwrap.dedent("""
    import { Viewer, autoExposureTarget } from "%s";
    import { Renderer } from "%s";
    import * as K from "%s";

    // A viewer stub carrying only what the look/metadata paths read. The
    // methods under test are the real prototype's.
    const makeViewer = (over = {}) => {
      const v = Object.create(Viewer.prototype);
      Object.assign(v, {
        scene: { sourceName: "field.nc", bmin: [0, 0, 0], bmax: [1, 1, 1],
                 cityPosition: () => null },
        camera: { position: [0.5, 0.5, 0.5], azimuth: 10, elevation: -5,
                  fov: 100, relativePosition: () => [0, 0, -0.999] },
        canvas: { width: 640, height: 360 },
        renderer: { periodic: true, cameraMoving: false },
        sunAzimuth: 20, sunElevation: 55,
        exposure: 2.5, toneMapGamma: 1.66, toneMapWhitePoint: 15,
        contrast: 1.0,
        haze: 1.7, hazeHeightDependent: true, lodStrength: 1.3,
        qualityTier: "medium",
        frameIndex: 7,
        _holdMode: "still",
        _byHand: new Set(),
        _captureStillTier: null,
        _captureVideoTier: null,
      }, over);
      return v;
    };

    const out = {};

    // (a) A named-tier capture is the pure parked preset, whatever the
    // session's hand-set look says.
    {
      const v = makeViewer({ _byHand: new Set(["haze", "lod"]) });
      const day = v._captureLook("high");
      const kw = v._captureViewKwargs(day);
      out.namedDay = { look: day, haze: kw.haze,
                       hazeHeightDependent: kw.hazeHeightDependent,
                       viewLod: kw.viewStepLodDegrees };
      const vc = makeViewer({ _byHand: new Set(["haze", "lod"]) });
      vc.scene = { ...vc.scene, city: true };
      out.namedCity = vc._captureLook("high");
      out.namedMinimal = makeViewer()._captureLook("minimal");
    }

    // (b) Custom carries the session's hand-set values verbatim, on the
    // live tier's preset.
    {
      const v = makeViewer({ _byHand: new Set(["haze", "lod",
                                               "autoExposure"]) });
      out.custom = v._captureLook("custom");
      out.customSelected = [v.captureStillTier, v.captureVideoTier];
      const untouched = makeViewer();
      out.defaultSelected = [untouched.captureStillTier,
                             untouched.captureVideoTier];
      const picked = makeViewer({ _byHand: new Set(["haze"]) });
      picked.captureStillTier = "low";      // an explicit pick stays pure
      out.pickedOverCustom = picked.captureStillTier;
    }

    // (c) The metered-exposure helper: the one formula, and the minimal
    // tier's refusal to meter.
    {
      out.aeTargets = [30.0, 0.1, 1000.0].map(
        (h) => autoExposureTarget(h, 15.0));
      const v = makeViewer();
      let metered = 0;
      v.renderer = { ...v.renderer,
                     meterLuminance: async () => { metered += 1; return 30.0; } };
      const minimal = v._captureLook("minimal");
      out.minimalExposure = await v._meterCaptureExposure(minimal, [64, 36]);
      out.minimalMetered = metered;
      const high = v._captureLook("high");
      out.highExposure = await v._meterCaptureExposure(high, [64, 36]);
      out.highMetered = metered;
      // A meter that saw no light keeps the default, like the live loop.
      v.renderer.meterLuminance = async () => 0.0;
      out.darkExposure = await v._meterCaptureExposure(
        v._captureLook("high"), [64, 36]);
    }

    // (d) Renderer.beginOffline marches parked (High) sampling for every
    // tier.
    {
      const r = {
        flightRenderScale: 0.125, qualityTier: "minimal",
        lightCacheMode: "auto", parkedSppOverride: null,
        lightBakePending: false,
        setQualityTier(t) {           // the real one resets to flight stepping
          this.qualityTier = t;
          this.stepFactor = K.QUALITY_PRESETS[t].stepFactor;
          this.lightStepFactor = K.QUALITY_PRESETS[t].lightStepFactor;
        },
        setRenderScale(s) { this.renderScale = s; },
        resetAccumulation() {},
      };
      Renderer.prototype.beginOffline.call(r, "minimal");
      out.offline = { stepFactor: r.stepFactor,
                      lightStepFactor: r.lightStepFactor,
                      renderScale: r.renderScale,
                      tier: r.qualityTier };
    }

    // (e) The parked city LOD: flight value while moving, the quarter-pixel
    // floor while parked; a hand-set slider is the user's both ways, and
    // hold mode "live" never sharpens.
    {
      const lod = (over) => {
        const v = makeViewer({ lodStrength: 0.5, ...over });
        v.scene = { ...v.scene, city: true };
        return v._liveLodStrength();
      };
      out.city = {
        moving: lod({ renderer: { cameraMoving: true } }),
        parked: lod({ renderer: { cameraMoving: false } }),
        handSetMoving: lod({ renderer: { cameraMoving: true },
                             _byHand: new Set(["lod"]), lodStrength: 0.2 }),
        handSetParked: lod({ renderer: { cameraMoving: false },
                             _byHand: new Set(["lod"]), lodStrength: 0.2 }),
        liveHold: lod({ renderer: { cameraMoving: false },
                        _holdMode: "live" }),
      };
      const day = makeViewer({ lodStrength: 1.05,
                               renderer: { cameraMoving: false } });
      out.dayParked = day._liveLodStrength();
    }

    // (f) The honesty trail: metadata and the reproduction command say what
    // the capture marched.
    {
      const v = makeViewer({ _byHand: new Set(["haze"]) });
      const named = v._captureLook("high");
      named.exposure = 1.75;                       // as the meter would set it
      out.metaNamed = v.renderMetadata([1920, 1080], named).render;
      out.cmdNamed = v.renderMetadata([1920, 1080], named)
        .reproduction_command;
      const custom = v._captureLook("custom");
      out.metaCustom = v.renderMetadata([1920, 1080], custom).render;
      out.cmdCustom = v.renderMetadata([1920, 1080], custom)
        .reproduction_command;
    }

    // (g) A saved track's header is the video's look, metered where the
    // video meters: at frame 0's pose and clock, not the live camera.
    {
      const metered = [];
      const v = makeViewer({
        videoFps: 30,
        scene: { sourceName: "field.nc", bmin: [0, 0, 0],
                 bmax: [1000, 1000, 1000], cityPosition: () => null,
                 surfacePeriod: 1638.4 },
        renderer: { periodic: true, cameraMoving: false,
                    meterLuminance: async (size, kw) => {
                      metered.push({ size, position: [...kw.camera.position],
                                     sceneTime: kw.sceneTime,
                                     haze: kw.haze });
                      return 4.0;
                    } },
        ui: { say() {} },
      });
      let blob = null;
      globalThis.document = { createElement: () => ({ click() {} }) };
      URL.createObjectURL = (b) => { blob = b; return "blob:track"; };
      URL.revokeObjectURL = () => {};
      const samples = [[0, -0.5, 0.2, 0.3, 90, 0, 70, 0.5, 0.25, 12.5],
                       [1, -0.4, 0.2, 0.3, 90, 0, 70, 0.51, 0.25, 13.5]];
      await v.downloadTrack(samples, []);
      const payload = JSON.parse(await blob.text());
      out.savedTrack = {
        metered,
        pose0: v._trackPose(
          { position: samples[0].slice(1, 4), azimuth: 90, elevation: 0,
            fov: 70, surfacePosition: [0.5, 0.25] }).position,
        surfacePeriod: payload.surface_period_m,
        render: payload.header.render,
        expected: autoExposureTarget(4.0, 15),
        videoHaze: K.DEFAULT_HAZE_BY_TIER.high,
      };
    }

    out.tables = {
      haze: K.DEFAULT_HAZE_BY_TIER, lod: K.DEFAULT_LOD_STRENGTH_BY_TIER,
      cityParked: K.CITY_PARKED_LOD_STRENGTH,
      cityTable: K.CITY_LOD_STRENGTH_BY_TIER,
      defaultExposure: K.DEFAULT_EXPOSURE,
      high: K.QUALITY_PRESETS.high,
      parkedFrames: K.PARKED_ACCUM_FRAMES_BY_TIER,
    };
    process.stdout.write(JSON.stringify(out));
""") % (VIEWER_JS, RENDERER_JS, CONSTANTS_JS)


@pytest.fixture(scope="module")
def js():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


@needs_node
def test_a_named_tier_capture_is_the_pure_parked_preset(js):
    day = js["namedDay"]
    assert day["look"]["haze"] == js["tables"]["haze"]["high"]
    assert day["haze"] == js["tables"]["haze"]["high"]        # not the 1.7
    assert day["look"]["lodStrength"] == js["tables"]["lod"]["high"]
    # The session's hazeHeightDependent rides through: it is not tier-varying.
    assert day["hazeHeightDependent"] is True
    # And the kwargs actually carry the capture LOD, not the slider's 1.3.
    assert day["viewLod"] == pytest.approx(0.6 * js["tables"]["lod"]["high"])


@needs_node
def test_a_city_capture_marches_the_parked_city_lod(js):
    assert js["tables"]["cityParked"] == 0.01
    assert js["tables"]["cityParked"] == js["tables"]["cityTable"]["high"]
    assert js["namedCity"]["lodStrength"] == js["tables"]["cityParked"]
    # Every named tier: the capture parks, so the city floor is not per-tier.
    assert js["namedMinimal"]["lodStrength"] == js["tables"]["lod"]["minimal"]


@needs_node
def test_custom_carries_the_session_verbatim(js):
    custom = js["custom"]
    assert custom["haze"] == 1.7
    assert custom["lodStrength"] == 1.3
    assert custom["exposure"] == 2.5
    assert custom["meter"] is False              # the session already metered
    assert custom["preset"] == "medium"          # the LIVE tier's methods
    # Hand-set look selects Custom on both panels by itself...
    assert js["customSelected"] == ["custom", "custom"]
    # ...nothing hand-set keeps today's defaults...
    assert js["defaultSelected"] == ["max", "high"]
    # ...and an explicit named pick stays pure.
    assert js["pickedOverCustom"] == "low"


@needs_node
def test_the_metered_exposure_formula_and_the_minimal_refusal(js):
    # full = 0.9 * 15 / highlight; target = hi * (full/hi)^0.5 below the
    # ceiling, clamped to [1, 4] — the values _aeTick glides toward.
    assert js["aeTargets"][0] == pytest.approx(4.0 * (0.45 / 4.0) ** 0.5)
    assert js["aeTargets"][1] == 4.0             # ceiling: dim highlight
    assert js["aeTargets"][2] == 1.0             # floor: blinding highlight
    # Minimal's AE default is off: DEFAULT_EXPOSURE, and no meter ran.
    assert js["minimalExposure"] == js["tables"]["defaultExposure"]
    assert js["minimalMetered"] == 0
    # High meters once and applies the target directly.
    assert js["highExposure"] == pytest.approx(js["aeTargets"][0])
    assert js["highMetered"] == 1
    # A meter that saw no light keeps the default, like the live loop.
    assert js["darkExposure"] == js["tables"]["defaultExposure"]


@needs_node
def test_offline_render_marches_parked_sampling(js):
    offline = js["offline"]
    high = js["tables"]["high"]
    assert offline["tier"] == "minimal"          # lighting method is the tier's
    assert offline["stepFactor"] == high["stepFactor"]
    assert offline["lightStepFactor"] == high["lightStepFactor"]
    assert offline["renderScale"] == 1.0         # no hold-scale cap: no frame
                                                 # to protect


@needs_node
def test_a_parked_city_view_sharpens_and_a_hand_set_slider_wins(js):
    city = js["city"]
    assert city["moving"] == 0.5                 # the flight table's value
    assert city["parked"] == js["tables"]["cityParked"]
    assert city["handSetMoving"] == 0.2
    assert city["handSetParked"] == 0.2          # the user's, both ways
    assert city["liveHold"] == 0.5               # live never sharpens
    # Over clouds the day defaults already are the parked numbers.
    assert js["dayParked"] == 1.05


@needs_node
def test_metadata_and_command_record_what_was_marched(js):
    named = js["metaNamed"]
    assert named["tier"] == "high" and named["quality"] == "high"
    assert named["haze"] == js["tables"]["haze"]["high"]
    assert named["exposure"] == 1.75
    assert named["lod_strength"] == js["tables"]["lod"]["high"]
    assert named["step_factor"] == js["tables"]["high"]["stepFactor"]
    assert named["accumulate_frames"] == js["tables"]["parkedFrames"]["high"]
    cmd = js["cmdNamed"]
    assert "--quality high" in cmd
    assert f"--haze {js['tables']['haze']['high']}" in cmd
    assert "--exposure 1.75" in cmd
    assert f"--lod {js['tables']['lod']['high']}" in cmd

    custom = js["metaCustom"]
    assert custom["tier"] == "custom"
    assert custom["quality"] == "medium"         # the live tier's preset
    assert custom["haze"] == 1.7
    assert custom["exposure"] == 2.5
    assert custom["lod_strength"] == 1.3
    # The CLI has no "custom": the command names the preset and carries the
    # hand-set look as explicit flags.
    assert "--quality medium" in js["cmdCustom"]
    assert "--haze 1.7" in js["cmdCustom"]
    assert "--lod 1.3" in js["cmdCustom"]



@needs_node
def test_a_saved_track_holds_the_exposure_its_video_meters(js):
    """`soar TRACK.json` renders at the header's exposure, the browser's
    video at the one it meters on frame 0; they are one reading
    (Viewer._trackLook), taken at the first sample's pose and clock at the
    video tier's look."""
    t = js["savedTrack"]
    assert len(t["metered"]) == 1
    m = t["metered"][0]
    assert m["size"] == [640, 360]
    assert m["position"] == pytest.approx(t["pose0"])
    assert m["sceneTime"] == 12.5
    assert m["haze"] == t["videoHaze"]
    assert t["render"]["tier"] == "high" and t["render"]["quality"] == "high"
    assert t["render"]["exposure"] == pytest.approx(t["expected"])
    assert t["surfacePeriod"] == 1638.4

def test_witness_quality_presets_carry_parked_stepping():
    """The CLI promise — '--quality renders as the in-app capture at that
    soar tier' — now means parked sampling: High's step factors for every
    tier, with the tiers differing in lighting method and spp only."""
    for name, preset in QUALITY_PRESETS.items():
        assert preset["step_factor"] == 2.0, name
        assert preset["light_step_factor"] == 2.0, name
    assert QUALITY_PRESETS["max"]["light_cache"] is False
    assert QUALITY_PRESETS["high"]["light_cache"] is True
    assert QUALITY_PRESETS["minimal"]["accumulate"] == 8
