"""Witness-vs-soar parity harness for staged renderer ports.

Stage 1 covers the analytic sky and tone map. Cloud scattering and the FIF
ocean are intentionally out of scope for this test; the comparison mask keeps
only above-horizon pixels so witness's ocean pass does not participate.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu", reason="requires the 'interactive' extra")


def _adapter_ok():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:
        return False
    return "float32-filterable" in adapter.features


if not _adapter_ok():  # pragma: no cover
    pytest.skip("no wgpu adapter with float32-filterable available",
                allow_module_level=True)


ARTIFACT_DIR = Path(__file__).parent.parent / "parity_out" / "stage1"
SKY_SIZE = (96, 64)
SKY_ABS_TOL = 3.0 / 255.0


@dataclass(frozen=True)
class ParityCase:
    name: str
    camera: object
    sun_azimuth: float
    sun_elevation: float


@dataclass(frozen=True)
class ParityRender:
    witness: np.ndarray
    soar: np.ndarray
    soar_u8: np.ndarray
    above_horizon: np.ndarray


@dataclass(frozen=True)
class DiffStats:
    mean_abs: float
    max_abs: float
    within_1_255: float
    within_2_255: float
    within_3_255: float
    n_pixels: int

    def as_dict(self) -> dict:
        return {
            "mean_abs": self.mean_abs,
            "mean_abs_255": self.mean_abs * 255.0,
            "max_abs": self.max_abs,
            "max_abs_255": self.max_abs * 255.0,
            "within_1_255": self.within_1_255,
            "within_2_255": self.within_2_255,
            "within_3_255": self.within_3_255,
            "n_pixels": self.n_pixels,
        }


def zero_cloud_field(n: int = 8):
    """Small empty field: both renderers should produce sky plus tone map."""
    from cloudyview import CloudField

    lwc = np.zeros((n, n, n), dtype=np.float32)
    x = np.linspace(-3500.0, 3500.0, n, dtype=np.float32)
    y = np.linspace(-3500.0, 3500.0, n, dtype=np.float32)
    z = np.linspace(250.0, 3750.0, n, dtype=np.float32)
    return CloudField(lwc=lwc, x=x, y=y, z=z)


def ray_directions(camera, size: tuple[int, int]) -> np.ndarray:
    """Per-pixel camera rays using the witness pixel-center convention."""
    w, h = size
    forward, right, up = camera.basis()
    tan_half_fov = np.tan(np.deg2rad(camera.fov) * 0.5)
    aspect = w / h

    px = np.arange(w, dtype=np.float64) + 0.5
    py = np.arange(h, dtype=np.float64) + 0.5
    ndc_x = (2.0 * px / w - 1.0) * aspect * tan_half_fov
    ndc_y = (1.0 - 2.0 * py / h) * tan_half_fov
    xx, yy = np.meshgrid(ndc_x, ndc_y)

    dirs = (forward[None, None, :]
            + xx[..., None] * right[None, None, :]
            + yy[..., None] * up[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs


def above_horizon_mask(camera, size: tuple[int, int]) -> np.ndarray:
    return ray_directions(camera, size)[..., 2] > 0.0


def render_witness_soar_pair(
    field,
    renderer,
    *,
    camera,
    size: tuple[int, int],
    sun_azimuth: float,
    sun_elevation: float,
) -> ParityRender:
    """Render one scene through CPU witness and soar with matched controls."""
    import cloudyview as cv

    witness = cv.witness(
        field,
        camera=camera,
        size=size,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        exposure=4.0,
        verbose=False,
    ).astype(np.float64)
    soar_u8 = renderer.render(
        camera,
        size=size,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        exposure=4.0,
        jitter=False,
    )
    soar = soar_u8.astype(np.float64) / 255.0
    assert witness.shape == soar.shape == (size[1], size[0], 3)
    return ParityRender(
        witness=witness,
        soar=soar,
        soar_u8=soar_u8,
        above_horizon=above_horizon_mask(camera, size),
    )


def diff_statistics(
    witness: np.ndarray,
    soar: np.ndarray,
    mask: np.ndarray,
) -> DiffStats:
    delta = np.abs(witness - soar)
    pixel_delta = delta.max(axis=-1)
    masked_channels = delta[mask]
    masked_pixels = pixel_delta[mask]
    return DiffStats(
        mean_abs=float(masked_channels.mean()),
        max_abs=float(masked_channels.max()),
        within_1_255=float(np.mean(masked_pixels <= 1.0 / 255.0)),
        within_2_255=float(np.mean(masked_pixels <= 2.0 / 255.0)),
        within_3_255=float(np.mean(masked_pixels <= 3.0 / 255.0)),
        n_pixels=int(mask.sum()),
    )


def _to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _diff_heatmap(pair: ParityRender) -> np.ndarray:
    delta = np.abs(pair.witness - pair.soar).max(axis=-1)
    t = np.clip(delta / (6.0 / 255.0), 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = (255.0 * t).astype(np.uint8)
    heat[..., 1] = (180.0 * np.maximum(t - 0.25, 0.0) / 0.75).astype(np.uint8)
    heat[..., 2] = (40.0 * t).astype(np.uint8)
    return heat


def _write_artifact(case: ParityCase, pair: ParityRender) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    triptych = np.concatenate([
        _to_u8(pair.witness),
        pair.soar_u8,
        _diff_heatmap(pair),
    ], axis=1)
    iio.imwrite(ARTIFACT_DIR / f"{case.name}_witness_soar_diff.png", triptych)


@pytest.fixture(scope="module")
def field():
    return zero_cloud_field()


@pytest.fixture(scope="module")
def renderer(field):
    from cloudyview.soar import InteractiveRenderer

    return InteractiveRenderer(field)


def _sky_cases():
    import cloudyview as cv

    return [
        ParityCase(
            name="default_up",
            camera=cv.Camera(position=(0.0, 0.0, -0.999),
                             azimuth=0.0, elevation=35.0, fov=100.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
        ParityCase(
            name="sun_centered",
            camera=cv.Camera(position=(0.0, 0.0, -0.999),
                             azimuth=20.0, elevation=55.0, fov=70.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
        ParityCase(
            name="low_horizon",
            camera=cv.Camera(position=(0.0, 0.0, -0.999),
                             azimuth=200.0, elevation=12.0, fov=80.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
    ]


def test_soar_matches_witness_sky_and_tone_map(field, renderer):
    """Sky-only parity, masking below-horizon pixels until stage 3 ocean port."""
    stats_by_case = {}
    for case in _sky_cases():
        pair = render_witness_soar_pair(
            field,
            renderer,
            camera=case.camera,
            size=SKY_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
        )
        assert pair.above_horizon.any()
        stats = diff_statistics(pair.witness, pair.soar, pair.above_horizon)
        stats_by_case[case.name] = stats.as_dict()
        _write_artifact(case, pair)

        # The zero field removes volume texture sampling and light marching
        # from the compared pixels, so the remaining error should be only WGSL
        # fp32 vs CPU fp64 plus rgba8unorm readback quantization (~0.5/255).
        # 3/255 leaves room for backend rounding while still catching math
        # drift in the sky constants, Lorentzian bloom, sun disk, or tone map.
        assert stats.max_abs <= SKY_ABS_TOL

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "sky_stats.json").write_text(
        json.dumps(stats_by_case, indent=2, sort_keys=True) + "\n"
    )
