"""Witness-vs-soar parity harness for staged renderer ports.

Stage 4 covers the analytic sky, tone map, single-domain cloud scattering,
the ghost-zero boundary taper, and the FIF ocean. The harness generates one
seeded FIF normal realization and passes those exact arrays into both
witness.render_nested() and Soar so ocean pixels are deterministic at
parity-test scale.
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


ARTIFACT_DIR = Path(__file__).parent.parent / "parity_out" / "stage4"
DATA128 = Path(__file__).parent.parent / "data" / "TWPICE_subvolume_128x128_5km.nc"
SKY_SIZE = (96, 64)
CLOUD_SIZE = (96, 64)
SKY_ABS_TOL = 3.0 / 255.0
OCEAN_ABS_TOL = 3.0 / 255.0
OCEAN_SHADOW_MEAN_TOL = 1.5 / 255.0
OCEAN_SHADOW_P99_TOL = 3.0 / 255.0
OCEAN_SHADOW_MAX_TOL = 12.0 / 255.0
FIF_SEED = 20260707


def _realism_off() -> dict[str, float]:
    return {
        "gradient_shading_strength": 0.0,
        "deep_shadow_ms_suppression": 0.0,
        "ambient_occlusion_strength": 0.0,
        "bounce_depth_attenuation": 0.0,
    }


def _realism_on() -> dict[str, float]:
    from cloudyview.witness import (
        AMBIENT_OCCLUSION_FLOOR,
        AMBIENT_OCCLUSION_STRENGTH,
        BOUNCE_DEPTH_ATTENUATION,
        DEEP_SHADOW_MS_SUPPRESSION,
        GRADIENT_SHADING_COARSE_RADIUS_M,
        GRADIENT_SHADING_COARSE_WEIGHT,
        GRADIENT_SHADING_STRENGTH,
    )

    return {
        "gradient_shading_strength": GRADIENT_SHADING_STRENGTH,
        "gradient_coarse_weight": GRADIENT_SHADING_COARSE_WEIGHT,
        "gradient_coarse_radius_m": GRADIENT_SHADING_COARSE_RADIUS_M,
        "deep_shadow_ms_suppression": DEEP_SHADOW_MS_SUPPRESSION,
        "ambient_occlusion_strength": AMBIENT_OCCLUSION_STRENGTH,
        "ambient_occlusion_floor": AMBIENT_OCCLUSION_FLOOR,
        "bounce_depth_attenuation": BOUNCE_DEPTH_ATTENUATION,
    }


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
    p95_abs: float
    p99_abs: float
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
            "p95_abs": self.p95_abs,
            "p95_abs_255": self.p95_abs * 255.0,
            "p99_abs": self.p99_abs,
            "p99_abs_255": self.p99_abs * 255.0,
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


def gaussian_cloud_field(n: int = 40):
    """Smooth centered cloud that avoids all lateral/domain boundaries."""
    from cloudyview import CloudField

    x = np.linspace(-4000.0, 4000.0, n, dtype=np.float32)
    y = np.linspace(-4000.0, 4000.0, n, dtype=np.float32)
    z = np.linspace(250.0, 4250.0, n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Peak LWC is intentionally modest: visible optical depth without saturating
    # the image, and the Gaussian is >5 sigma from the side walls so the
    # boundary taper stays out of this focused scattering case.
    lwc = 0.020 * np.exp(
        -0.5 * (
            (xx / 650.0) ** 2
            + (yy / 650.0) ** 2
            + ((zz - 2200.0) / 520.0) ** 2
        )
    )
    lwc[lwc < 1e-7] = 0.0
    return CloudField(lwc=lwc.astype(np.float32), x=x, y=y, z=z)


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


def sky_tone_mapped(
    camera,
    size: tuple[int, int],
    *,
    sun_azimuth: float,
    sun_elevation: float,
    exposure: float = 4.0,
) -> np.ndarray:
    """Vectorized witness._sky_radiance + witness.tone_map for mask building."""
    from cloudyview.angles import direction_from_azimuth_elevation

    dirs = ray_directions(camera, size)
    sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)

    t = np.maximum(0.0, dirs[..., 2])
    t = 1.0 - (1.0 - t) ** 3
    zenith = np.array([0.0044, 0.035, 0.1156])
    horizon = np.array([0.10, 0.18, 0.38])
    sky = horizon + (zenith - horizon) * t[..., None]

    cos_sun = np.sum(dirs * sun[None, None, :], axis=-1)
    bloom = cos_sun > 0.0
    a = np.zeros_like(cos_sun)
    a[bloom] = 0.002 / ((1.0 - cos_sun[bloom]) + 0.002)
    sky += a[..., None] * np.array([0.8, 0.6, 0.3])
    sky[cos_sun > 0.9998] += np.array([50.0, 45.0, 35.0])

    exposed = sky * exposure
    mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 1.4)


def cloud_mask(
    pair: ParityRender,
    camera,
    size: tuple[int, int],
    *,
    sun_azimuth: float,
    sun_elevation: float,
    threshold: float,
) -> np.ndarray:
    sky = sky_tone_mapped(
        camera,
        size,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
    )
    cloud_signal = np.abs(pair.witness - sky).max(axis=-1)
    return pair.above_horizon & (cloud_signal > threshold)


def _ray_box_np(origin: np.ndarray, direction: np.ndarray,
                bmin: np.ndarray, bmax: np.ndarray) -> tuple[float, float]:
    inv = 1.0 / direction
    t0 = (bmin - origin) * inv
    t1 = (bmax - origin) * inv
    t_near = float(np.maximum.reduce(np.minimum(t0, t1)))
    t_far = float(np.minimum.reduce(np.maximum(t0, t1)))
    return max(t_near, 0.0), t_far


def _sample_sigma_ghost_zero(
    sigma: np.ndarray,
    bmin: np.ndarray,
    voxel: np.ndarray,
    p: np.ndarray,
) -> float:
    """Python mirror of witness._sample_sigma_level for shadow classification."""
    shape = np.array(sigma.shape)
    g = (p - bmin) / voxel
    if np.any(g < -1.0) or np.any(g >= shape):
        return 0.0

    i0 = np.floor(g).astype(int)
    f = g - i0
    value = 0.0
    for dx in (0, 1):
        ix = i0[0] + dx
        if ix < 0 or ix >= shape[0]:
            continue
        wx = f[0] if dx else 1.0 - f[0]
        for dy in (0, 1):
            iy = i0[1] + dy
            if iy < 0 or iy >= shape[1]:
                continue
            wy = f[1] if dy else 1.0 - f[1]
            for dz in (0, 1):
                iz = i0[2] + dz
                if iz < 0 or iz >= shape[2]:
                    continue
                wz = f[2] if dz else 1.0 - f[2]
                value += float(sigma[ix, iy, iz]) * wx * wy * wz
    return value


def ocean_hit_mask(field, camera, size: tuple[int, int]) -> np.ndarray:
    """Pixels whose primary ray reaches the witness ocean extent."""
    from cloudyview.soar.engine import _volume_aabb, camera_world_origin

    bmin, bmax = _volume_aabb(field)
    origin = camera_world_origin(camera, bmin, bmax)
    dirs = ray_directions(camera, size)
    dz = dirs[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ocean = (0.0 - origin[2]) / dz
    mask = (dz < -1e-8) & (t_ocean > 0.0)

    hit_x = origin[0] + t_ocean * dirs[..., 0]
    hit_y = origin[1] + t_ocean * dirs[..., 1]
    extent = bmax - bmin
    center = 0.5 * (bmin + bmax)
    mask &= np.abs(hit_x - center[0]) < extent[0] * 50.0
    mask &= np.abs(hit_y - center[1]) < extent[1] * 50.0
    return mask


def ocean_shadow_mask(
    field,
    camera,
    size: tuple[int, int],
    base_mask: np.ndarray,
    *,
    sun_azimuth: float,
    sun_elevation: float,
    shadow_tau_threshold: float = 0.02,
) -> np.ndarray:
    """Return ocean pixels whose sun ray intersects visible cloud density."""
    from cloudyview import optical_depth
    from cloudyview.angles import direction_from_azimuth_elevation
    from cloudyview.soar.engine import (
        STEP_VOXEL_FACTOR,
        _volume_aabb,
        camera_world_origin,
    )

    bmin, bmax = _volume_aabb(field)
    origin = camera_world_origin(camera, bmin, bmax)
    dirs = ray_directions(camera, size)
    sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)
    shape = np.array(field.shape)
    voxel = (bmax - bmin) / shape
    dt_max = float(voxel.min()) * STEP_VOXEL_FACTOR
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, iwc=field.iwc
    ).astype(np.float64)

    shadow = np.zeros(base_mask.shape, dtype=bool)
    for y, x in zip(*np.nonzero(base_mask)):
        direction = dirs[y, x]
        if direction[2] >= -1e-8:
            continue
        t_ocean = (0.0 - origin[2]) / direction[2]
        if t_ocean <= 0.0:
            continue
        p0 = origin + t_ocean * direction
        t, t_far = _ray_box_np(p0, sun, bmin, bmax)
        if t > t_far or t_far <= 0.0:
            continue
        tau = 0.0
        while t < t_far:
            p = p0 + t * sun
            s = _sample_sigma_ghost_zero(sigma, bmin, voxel, p)
            dt = dt_max
            if t + dt > t_far:
                dt = t_far - t
            d_tau = s * dt
            tau += d_tau
            if tau > shadow_tau_threshold:
                shadow[y, x] = True
            if tau > 80.0:
                break
            t += dt
    return shadow


def render_witness_with_fif(
    field,
    *,
    camera,
    size: tuple[int, int],
    sun_azimuth: float,
    sun_elevation: float,
    fif_normals,
    realism: dict[str, float] | None = None,
) -> np.ndarray:
    """Render witness via render_nested() with caller-supplied FIF normals."""
    from cloudyview import optical_depth
    from cloudyview.angles import direction_from_azimuth_elevation
    from cloudyview.soar.engine import _volume_aabb, camera_world_origin
    from cloudyview.witness import (
        AMBIENT_STRENGTH,
        G_HG,
        MAX_STEPS,
        N_LIGHT_STEPS,
        OCEAN_REFLECTANCE,
        POWDER_COEFF,
        STEP_VOXEL_FACTOR,
        SUN_COLOR,
        NestedLevel,
        render_nested,
    )

    bmin, bmax = _volume_aabb(field)
    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < 1e-6:
        iwc = None
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=10.0, iwc=iwc, re_ice=30.0
    ).astype(np.float64)
    level = NestedLevel(
        sigma=np.ascontiguousarray(sigma),
        bmin=bmin,
        bmax=bmax,
        name="single",
    )
    origin = camera_world_origin(camera, bmin, bmax)
    forward, right, up = camera.basis()
    sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)
    if realism is None:
        realism = _realism_off()

    return render_nested(
        [level],
        tuple(origin),
        tuple(forward),
        tuple(right),
        tuple(up),
        tuple(sun),
        size,
        fov_degrees=camera.fov,
        n_light_steps=N_LIGHT_STEPS,
        step_voxel_factor=STEP_VOXEL_FACTOR,
        max_steps=MAX_STEPS,
        exposure=4.0,
        ocean_enabled=True,
        ocean_z=0.0,
        ocean_reflectance=OCEAN_REFLECTANCE,
        fif_normals=fif_normals,
        sun_color=SUN_COLOR,
        g_hg=G_HG,
        ambient_strength=AMBIENT_STRENGTH,
        powder_coeff=POWDER_COEFF,
        # witness_spp=1 is witness's explicit exact-legacy sampling path
        # (pixel centers, no march-phase jitter). WITNESS_SPP>1 supersampling
        # is witness's still-image analogue of soar's sub-pixel jitter +
        # temporal accumulation (tested in test_soar_render), so the parity
        # harness compares the single-deterministic-sample structure on both
        # sides rather than two different anti-aliasing strategies.
        witness_spp=1,
        **realism,
        verbose=False,
    ).astype(np.float64)


def render_witness_soar_pair(
    field,
    renderer,
    *,
    camera,
    size: tuple[int, int],
    sun_azimuth: float,
    sun_elevation: float,
    fif_normals,
    realism: dict[str, float] | None = None,
) -> ParityRender:
    """Render one scene through CPU witness and soar with matched controls."""
    if realism is None:
        realism = _realism_off()
    witness = render_witness_with_fif(
        field,
        camera=camera,
        size=size,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        fif_normals=fif_normals,
        realism=realism,
    )
    soar_u8 = renderer.render(
        camera,
        size=size,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        exposure=4.0,
        jitter=False,
        **realism,
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
        p95_abs=float(np.percentile(masked_channels, 95)),
        p99_abs=float(np.percentile(masked_channels, 99)),
        within_1_255=float(np.mean(masked_pixels <= 1.0 / 255.0)),
        within_2_255=float(np.mean(masked_pixels <= 2.0 / 255.0)),
        within_3_255=float(np.mean(masked_pixels <= 3.0 / 255.0)),
        n_pixels=int(mask.sum()),
    )


def _to_u8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _diff_heatmap(pair: ParityRender, scale: float = 24.0 / 255.0) -> np.ndarray:
    delta = np.abs(pair.witness - pair.soar).max(axis=-1)
    t = np.clip(delta / scale, 0.0, 1.0)
    heat = np.zeros((*delta.shape, 3), dtype=np.uint8)
    heat[..., 0] = (255.0 * t).astype(np.uint8)
    heat[..., 1] = (180.0 * np.maximum(t - 0.25, 0.0) / 0.75).astype(np.uint8)
    heat[..., 2] = (40.0 * t).astype(np.uint8)
    return heat


def _write_artifact(case: ParityCase, pair: ParityRender) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    side_by_side = np.concatenate([
        _to_u8(pair.witness),
        pair.soar_u8,
    ], axis=1)
    iio.imwrite(ARTIFACT_DIR / f"{case.name}_side_by_side.png", side_by_side)
    iio.imwrite(ARTIFACT_DIR / f"{case.name}_absdiff.png", _diff_heatmap(pair))

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
def fif_normals():
    from cloudyview.ocean_fif import generate_fif_normals

    np.random.seed(FIF_SEED)
    return generate_fif_normals(
        rng=np.random.default_rng(FIF_SEED),
        verbose=False,
    )


def assert_renderer_uses_fif(renderer, fif_normals) -> None:
    for uploaded, expected in zip(renderer.ocean_fif_normals[:3], fif_normals[:3]):
        assert np.array_equal(uploaded, expected)
    assert renderer.ocean_fif_normals[3] == fif_normals[3]


@pytest.fixture(scope="module")
def renderer(field, fif_normals):
    from cloudyview.soar import InteractiveRenderer

    # periodic=False throughout this harness: witness is a finite-box
    # renderer, so parity is defined against the non-tiled march.
    r = InteractiveRenderer(field, fif_normals=fif_normals, periodic=False)
    assert_renderer_uses_fif(r, fif_normals)
    return r


@pytest.fixture(scope="module")
def gaussian_field_fixture():
    return gaussian_cloud_field()


@pytest.fixture(scope="module")
def gaussian_renderer(gaussian_field_fixture, fif_normals):
    from cloudyview.soar import InteractiveRenderer

    r = InteractiveRenderer(
        gaussian_field_fixture, fif_normals=fif_normals, periodic=False
    )
    assert_renderer_uses_fif(r, fif_normals)
    return r


@pytest.fixture(scope="module")
def twpice128_field():
    import cloudyview as cv

    return cv.load(str(DATA128))


@pytest.fixture(scope="module")
def twpice128_renderer(twpice128_field, fif_normals):
    from cloudyview.soar import InteractiveRenderer

    r = InteractiveRenderer(
        twpice128_field, fif_normals=fif_normals, periodic=False
    )
    assert_renderer_uses_fif(r, fif_normals)
    return r


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
        ParityCase(
            name="ocean_dominant_empty",
            camera=cv.Camera(position=(0.0, 0.0, -0.999),
                             azimuth=210.0, elevation=2.0, fov=72.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
    ]


def _gaussian_cloud_cases():
    import cloudyview as cv

    return [
        ParityCase(
            name="gaussian_sun_facing",
            camera=cv.Camera(position=(0.0, 0.0, -0.999),
                             azimuth=20.0, elevation=38.0, fov=76.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
        ParityCase(
            name="gaussian_anti_sun",
            camera=cv.Camera(position=(0.0, 0.0, -0.999),
                             azimuth=200.0, elevation=38.0, fov=76.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
        ParityCase(
            name="gaussian_cross_sun",
            camera=cv.Camera(position=(-0.25, -0.15, -0.999),
                             azimuth=95.0, elevation=32.0, fov=82.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
    ]


def _twpice128_cases():
    import cloudyview as cv

    return [
        ParityCase(
            name="twpice128_default",
            camera=cv.Camera(),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
    ]


def _twpice128_ocean_cases():
    import cloudyview as cv

    return [
        ParityCase(
            name="twpice128_ocean_dominant",
            camera=cv.Camera(position=(-0.65, -0.65, -0.999),
                             azimuth=45.0, elevation=-6.0, fov=90.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
        ParityCase(
            name="twpice128_mixed_cloud_ocean_shadow",
            camera=cv.Camera(position=(0.65, 0.65, -0.999),
                             azimuth=300.0, elevation=0.0, fov=110.0),
            sun_azimuth=20.0,
            sun_elevation=55.0,
        ),
    ]


def test_soar_matches_witness_sky_ocean_and_tone_map(field, renderer, fif_normals):
    """Zero-field parity: sky above the horizon, FIF ocean below it."""
    stats_by_case = {}
    for case in _sky_cases():
        pair = render_witness_soar_pair(
            field,
            renderer,
            camera=case.camera,
            size=SKY_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
            fif_normals=fif_normals,
        )
        mask = np.ones(pair.above_horizon.shape, dtype=bool)
        stats = diff_statistics(pair.witness, pair.soar, mask)
        stats_by_case[case.name] = stats.as_dict()
        _write_artifact(case, pair)

        # The zero field removes volume texture sampling and light marching
        # from sky pixels. Ocean pixels add one hardware-bilinear normal sample
        # and fp32 shading, but no shadow march in this empty scene. 3/255
        # leaves room for rgba8 readback/backend rounding while still catching
        # drift in sky constants, tone map, FIF sampling, or ocean BRDF math.
        assert stats.max_abs <= max(SKY_ABS_TOL, OCEAN_ABS_TOL)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "sky_stats.json").write_text(
        json.dumps(stats_by_case, indent=2, sort_keys=True) + "\n"
    )


def test_soar_matches_witness_gaussian_cloud_scattering(
    gaussian_field_fixture,
    gaussian_renderer,
    fif_normals,
):
    """Smooth cloud parity: exercises scattering without boundary truncation."""
    stats_by_case = {}
    for case in _gaussian_cloud_cases():
        pair = render_witness_soar_pair(
            gaussian_field_fixture,
            gaussian_renderer,
            camera=case.camera,
            size=CLOUD_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
            fif_normals=fif_normals,
        )
        mask = cloud_mask(
            pair,
            case.camera,
            CLOUD_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
            threshold=2.0 / 255.0,
        )
        assert mask.sum() > 100
        stats = diff_statistics(pair.witness, pair.soar, mask)
        stats_by_case[case.name] = stats.as_dict()
        _write_artifact(case, pair)

        # Smooth centered Gaussian: no lateral boundary truncation and the mask
        # focuses on above-horizon cloud-scattering pixels. Dedicated stage-3
        # tests assert ocean pixels. The remaining differences are WGSL
        # fp32/hardware texture interpolation, CPU fp64, and rgba8 readback.
        # Mean cloud-pixel error should stay within a couple of code values;
        # the p99/max limits leave room for fp32/tone-map/readback tails while
        # still catching a missing powder/MS/ambient term.
        assert stats.mean_abs <= 2.0 / 255.0
        assert stats.p99_abs <= 4.0 / 255.0
        assert stats.max_abs <= 8.0 / 255.0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "gaussian_cloud_stats.json").write_text(
        json.dumps(stats_by_case, indent=2, sort_keys=True) + "\n"
    )


def test_soar_matches_witness_twpice128_default_cloud_scattering(
    twpice128_field,
    twpice128_renderer,
    fif_normals,
):
    """Default-camera TWPICE parity against the cloud-scattering model."""
    stats_by_case = {}
    for case in _twpice128_cases():
        pair = render_witness_soar_pair(
            twpice128_field,
            twpice128_renderer,
            camera=case.camera,
            size=CLOUD_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
            fif_normals=fif_normals,
        )
        mask = cloud_mask(
            pair,
            case.camera,
            CLOUD_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
            threshold=3.0 / 255.0,
        )
        assert mask.sum() > 100
        stats = diff_statistics(pair.witness, pair.soar, mask)
        stats_by_case[case.name] = {
            "cloud_pixels_raw": stats.as_dict(),
        }
        _write_artifact(case, pair)

        # TWPICE is turbulent and the 128 subvolume has some nonzero lateral
        # and top-edge voxels. Stage 4 makes those boundary-truncated cloud
        # pixels part of the asserted raw set: the only remaining tolerated
        # tails are fp32/hardware-filtered texture interpolation, CPU fp64,
        # and rgba8 readback.
        assert stats.mean_abs <= 2.0 / 255.0
        assert stats.p99_abs <= 8.0 / 255.0
        assert stats.max_abs <= 16.0 / 255.0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "twpice128_cloud_stats.json").write_text(
        json.dumps(stats_by_case, indent=2, sort_keys=True) + "\n"
    )


def test_soar_matches_witness_twpice128_cb_realism_on(
    twpice128_field,
    twpice128_renderer,
    fif_normals,
):
    """Cumulonimbus realism gates-on parity for the thick TWPICE case."""
    stats_by_case = {}
    realism = _realism_on()
    for case in _twpice128_cases():
        gated_case = ParityCase(
            name=f"{case.name}_cb1111",
            camera=case.camera,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
        )
        pair = render_witness_soar_pair(
            twpice128_field,
            twpice128_renderer,
            camera=gated_case.camera,
            size=CLOUD_SIZE,
            sun_azimuth=gated_case.sun_azimuth,
            sun_elevation=gated_case.sun_elevation,
            fif_normals=fif_normals,
            realism=realism,
        )
        mask = cloud_mask(
            pair,
            gated_case.camera,
            CLOUD_SIZE,
            sun_azimuth=gated_case.sun_azimuth,
            sun_elevation=gated_case.sun_elevation,
            threshold=3.0 / 255.0,
        )
        assert mask.sum() > 100
        stats = diff_statistics(pair.witness, pair.soar, mask)
        stats_by_case[gated_case.name] = {
            "cloud_pixels_raw": stats.as_dict(),
            "realism": realism,
        }
        _write_artifact(gated_case, pair)

        # The blended gates-on case adds twelve gradient texture samples and several
        # saturated-shadow nonlinearities. CPU witness uses fp64 trilinear
        # sampling; Soar uses fp32 hardware filtering and rgba8 readback, so
        # the tail is allowed slightly more room than the all-off TWPICE case
        # while still catching any missing gate or sign mismatch.
        assert stats.mean_abs <= 2.5 / 255.0
        assert stats.p99_abs <= 10.0 / 255.0
        assert stats.max_abs <= 20.0 / 255.0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "twpice128_cb_realism_on_stats.json").write_text(
        json.dumps(stats_by_case, indent=2, sort_keys=True) + "\n"
    )


def test_soar_matches_witness_twpice128_ocean_and_shadow(
    twpice128_field,
    twpice128_renderer,
    fif_normals,
):
    """Ocean parity on real data, including cloud-shadowed water pixels."""
    stats_by_case = {}
    for case in _twpice128_ocean_cases():
        pair = render_witness_soar_pair(
            twpice128_field,
            twpice128_renderer,
            camera=case.camera,
            size=CLOUD_SIZE,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
            fif_normals=fif_normals,
        )
        ocean = ocean_hit_mask(twpice128_field, case.camera, CLOUD_SIZE)
        assert ocean.sum() > 100
        shadow = ocean_shadow_mask(
            twpice128_field,
            case.camera,
            CLOUD_SIZE,
            ocean,
            sun_azimuth=case.sun_azimuth,
            sun_elevation=case.sun_elevation,
        )
        mask = ocean
        assert mask.sum() > 100
        shadow_mask = shadow & mask
        if "mixed_cloud_ocean_shadow" in case.name:
            assert shadow_mask.sum() > 25

        stats = diff_statistics(pair.witness, pair.soar, mask)
        glint_mask = mask & (pair.witness.max(axis=-1) > 0.55)
        case_stats = {
            "ocean_pixels_raw": stats.as_dict(),
            "ocean_pixels": int(ocean.sum()),
            "shadow_pixels": int(shadow_mask.sum()),
            "glint_pixels": int(glint_mask.sum()),
        }
        if shadow_mask.any():
            case_stats["shadowed_ocean_pixels"] = diff_statistics(
                pair.witness, pair.soar, shadow_mask
            ).as_dict()
        if glint_mask.any():
            case_stats["glint_pixels_stats"] = diff_statistics(
                pair.witness, pair.soar, glint_mask
            ).as_dict()
        stats_by_case[case.name] = case_stats
        _write_artifact(case, pair)

        # Real-data ocean pixels exercise the FIF normal texture and, where
        # shadowed, the same cloud light march as cloud scattering. The mean
        # is held near a few code values; p99/max allow narrow glint pixels and
        # fp32/hardware-filtered normal interpolation without masking them out.
        assert stats.mean_abs <= OCEAN_SHADOW_MEAN_TOL
        assert stats.p99_abs <= OCEAN_SHADOW_P99_TOL
        assert stats.max_abs <= OCEAN_SHADOW_MAX_TOL

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "twpice128_ocean_stats.json").write_text(
        json.dumps(stats_by_case, indent=2, sort_keys=True) + "\n"
    )
