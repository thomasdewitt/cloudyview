"""Focused performance-tier and fp16-volume tests for soar."""

import numpy as np
import pytest

pytest.importorskip("wgpu", reason="requires the 'interactive' extra")

from cloudyview.soar.engine import (
    QUALITY_PRESETS,
    STEP_VOXEL_FACTOR,
    InteractiveRenderer,
    choose_quality_tier,
    render_target_size,
)


def test_quality_preset_math_and_scaled_target_sizes():
    assert QUALITY_PRESETS["high"].render_scale == 1.0
    assert QUALITY_PRESETS["high"].step_factor == STEP_VOXEL_FACTOR == 2.0
    assert QUALITY_PRESETS["high"].max_light_steps == 512

    assert render_target_size((1280, 720), 1.0) == (1280, 720)
    assert render_target_size((1280, 720), 0.75) == (960, 540)
    assert render_target_size((1280, 720), 0.60) == (768, 432)
    assert render_target_size((1280, 720), 0.25) == (320, 180)
    assert render_target_size((3, 3), 0.50) == (2, 2)

    with pytest.raises(ValueError, match="render_scale"):
        render_target_size((1280, 720), 0.2)
    with pytest.raises(ValueError, match="positive"):
        render_target_size((0, 720), 1.0)


def test_auto_tier_choice_prefers_highest_tier_holding_60_fps():
    assert choose_quality_tier({
        "high": 10.0, "medium": 6.0, "low": 3.0, "potato": 1.0,
    }) == "high"
    assert choose_quality_tier({
        "high": 22.0, "medium": 14.0, "low": 8.0, "potato": 3.0,
    }) == "medium"
    assert choose_quality_tier({
        "high": 40.0, "medium": 28.0, "low": 18.0, "potato": 17.0,
    }) == "potato"


def _tiny_field():
    from cloudyview import CloudField

    rng = np.random.default_rng(7)
    return CloudField(
        lwc=rng.uniform(0.001, 0.03, size=(6, 5, 4)).astype(np.float32),
        x=np.linspace(-3000.0, 3000.0, 6, dtype=np.float32),
        y=np.linspace(-2500.0, 2500.0, 5, dtype=np.float32),
        z=np.linspace(250.0, 3250.0, 4, dtype=np.float32),
    )


def _tiny_fif_normals():
    n = 4
    return (
        np.zeros((n, n), dtype=np.float32),
        np.zeros((n, n), dtype=np.float32),
        np.ones((n, n), dtype=np.float32),
        100.0,
    )


def _gpu_available():
    try:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
    except Exception:
        return False
    return "float32-filterable" in adapter.features


@pytest.fixture(scope="module")
def quality_renderer():
    if not _gpu_available():
        pytest.skip("no wgpu adapter with float32-filterable available")
    return InteractiveRenderer(
        _tiny_field(), fif_normals=_tiny_fif_normals(), periodic=True
    )


@pytest.mark.gpu
def test_tier_setters_pack_expected_step_factors(quality_renderer):
    import cloudyview as cv

    renderer = quality_renderer
    for name, preset in QUALITY_PRESETS.items():
        renderer.set_quality_tier(name, camera_moving=True)
        renderer.write_uniforms(cv.Camera(), (128, 72), jitter=False)
        assert renderer._current_uniform_size == render_target_size(
            (128, 72), preset.render_scale
        )
        assert renderer._current_uniform[5, 3] == pytest.approx(
            renderer._min_voxel_m * preset.step_factor
        )
        assert renderer.max_light_steps == preset.max_light_steps


@pytest.mark.gpu
def test_potato_restores_exact_high_settings_when_stationary(quality_renderer):
    renderer = quality_renderer
    renderer.set_quality_tier("potato", camera_moving=True)
    assert (renderer.render_scale, renderer.step_factor,
            renderer.max_light_steps) == (0.25, 4.0, 128)
    renderer.set_camera_moving(False)
    assert (renderer.render_scale, renderer.step_factor,
            renderer.max_light_steps) == (1.0, 2.0, 512)
    assert renderer.flight_render_scale == 0.25


@pytest.mark.gpu
def test_high_is_default_and_bit_identical_after_tier_round_trip(
    quality_renderer,
):
    import cloudyview as cv

    renderer = quality_renderer
    camera = cv.Camera()
    renderer.set_quality_tier("high", camera_moving=False)
    baseline = renderer.render(camera, (96, 54), jitter=False)
    renderer.set_quality_tier("low", camera_moving=True)
    renderer.set_quality_tier("high", camera_moving=False)
    restored = renderer.render(camera, (96, 54), jitter=False)
    np.testing.assert_array_equal(restored, baseline)


@pytest.mark.gpu
def test_scaled_tiers_upscale_to_requested_output_size(quality_renderer):
    import cloudyview as cv

    renderer = quality_renderer
    renderer.set_quality_tier("medium", camera_moving=True)
    image = renderer.render(cv.Camera(), (101, 57), jitter=False)
    assert renderer._current_uniform_size == (76, 43)
    assert image.shape == (57, 101, 3)


@pytest.mark.gpu
def test_fp16_volume_halves_resident_bytes_and_renders():
    import cloudyview as cv

    field = _tiny_field()
    fif = _tiny_fif_normals()
    fp32 = InteractiveRenderer(field, fif_normals=fif, volume_fp16=False)
    fp16 = InteractiveRenderer(field, fif_normals=fif, volume_fp16=True)
    assert fp16.volume_nbytes * 2 == fp32.volume_nbytes
    assert fp16.volume_texture_format == "r16float"
    image = fp16.render(cv.Camera(), (64, 36), jitter=False)
    assert image.shape == (36, 64, 3)
