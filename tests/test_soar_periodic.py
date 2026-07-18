"""Tests for periodic-domain support in the soar WGSL engine.

SAM LES domains are doubly periodic in x/y, so soar tiles the volume
horizontally by default: density sampling wraps (opposite-face ghost
texels), the view march never exits sideways, the light march exits only
through the domain top, and the app camera wraps modulo the domain.

Coverage:
- ghost-border texel content (periodic fill vs ghost zero, toggling);
- render-level seam correctness (translation invariance by one period);
- periodic=off bit-exactness against pre-change reference frames rendered
  at a1e157d (the commit before the periodic port), including the
  documented realism kill combination;
- app camera wrap + minimap marker, ESC-menu toggle, --no-periodic wiring;
- the behold hand-off "view spans domain edge" predicate.

GPU tests are skip-marked like the other soar suites. The bit-exactness
references are rendered ON THE DEV RTX 5080 at the pre-change commit
(a1e157d); bit-for-bit equality is the contract on that adapter only —
other adapters (including llvmpipe sandboxes) round differently, in which
case regenerate the references at the pre-change commit on the target
adapter. (Originally generated on llvmpipe by a sandboxed agent, which
failed on real hardware — regenerated 2026-07-10.)
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloudyview.soar.menu import (
    ACTION_TOGGLE_PERIODIC,
    MENU_MAIN,
    MENU_RENDER_QUALITY,
    menu_transition,
)

DATA128 = Path(__file__).parent.parent / "data" / "TWPICE_subvolume_128x128_5km.nc"
PRECHANGE_DIR = Path(__file__).parent / "reference_images" / "soar_prechange"

pytestmark = pytest.mark.gpu

wgpu = pytest.importorskip("wgpu", reason="requires the 'interactive' extra")


def _adapter_ok():
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    except Exception:
        return False
    return "float32-filterable" in adapter.features


def _is_rtx_5080_reference_adapter():
    """The checked-in byte references are explicitly RTX-5080-specific."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    identity = " ".join(
        str(info.get(name, ""))
        for name in ("vendor", "device", "description")
    ).lower()
    return "5080" in identity


if not _adapter_ok():  # pragma: no cover
    pytest.skip("no wgpu adapter with float32-filterable available",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_fif_normals():
    """Flat 4x4 ocean tile: keeps renderer construction fast and seeded."""
    n = 4
    nx = np.zeros((n, n), dtype=np.float32)
    ny = np.zeros((n, n), dtype=np.float32)
    nz = np.ones((n, n), dtype=np.float32)
    return (nx, ny, nz, 100.0)


def _random_field(shape=(6, 5, 4), seed=7):
    """Small asymmetric random field (distinct dims catch axis mix-ups)."""
    from cloudyview import CloudField

    rng = np.random.default_rng(seed)
    lwc = rng.uniform(0.001, 0.03, size=shape).astype(np.float32)
    nx, ny, nz = shape
    x = np.linspace(-3000.0, 3000.0, nx, dtype=np.float32)
    y = np.linspace(-2500.0, 2500.0, ny, dtype=np.float32)
    z = np.linspace(250.0, 3250.0, nz, dtype=np.float32)
    return CloudField(lwc=lwc, x=x, y=y, z=z)


def _read_volume_texture(renderer) -> np.ndarray:
    """Read the padded (nx+2, ny+2, nz+2) r32float volume back to the host."""
    nx, ny, nz = renderer.field.shape
    data = renderer.device.queue.read_texture(
        {"texture": renderer._texture},
        {"bytes_per_row": (nz + 2) * 4, "rows_per_image": ny + 2},
        (nz + 2, ny + 2, nx + 2),
    )
    return np.frombuffer(data, dtype=np.float32).reshape(
        nx + 2, ny + 2, nz + 2
    )


@pytest.fixture(scope="module")
def small_periodic_renderer():
    from cloudyview.soar import InteractiveRenderer

    return InteractiveRenderer(
        _random_field(), periodic=True, fif_normals=_tiny_fif_normals()
    )


@pytest.fixture(scope="module")
def twpice128_renderer():
    """TWPICE-128, periodic, ocean off (the FIF tile has its own period and
    would break the exact one-domain-period translation invariance)."""
    import cloudyview as cv
    from cloudyview.soar import InteractiveRenderer

    field = cv.load(str(DATA128))
    return InteractiveRenderer(
        field, periodic=True, ocean_enabled=False,
        fif_normals=_tiny_fif_normals(),
    )


# ---------------------------------------------------------------------------
# (a) Ghost-border wrap correctness
# ---------------------------------------------------------------------------

def _assert_border_periodic(arr: np.ndarray) -> None:
    # x ghost slices come from the opposite x faces...
    np.testing.assert_array_equal(arr[0, 1:-1, 1:-1], arr[-2, 1:-1, 1:-1])
    np.testing.assert_array_equal(arr[-1, 1:-1, 1:-1], arr[1, 1:-1, 1:-1])
    # ...same for y...
    np.testing.assert_array_equal(arr[1:-1, 0, 1:-1], arr[1:-1, -2, 1:-1])
    np.testing.assert_array_equal(arr[1:-1, -1, 1:-1], arr[1:-1, 1, 1:-1])
    # ...corners wrap in both x and y...
    np.testing.assert_array_equal(arr[0, 0, 1:-1], arr[-2, -2, 1:-1])
    np.testing.assert_array_equal(arr[0, -1, 1:-1], arr[-2, 1, 1:-1])
    np.testing.assert_array_equal(arr[-1, 0, 1:-1], arr[1, -2, 1:-1])
    np.testing.assert_array_equal(arr[-1, -1, 1:-1], arr[1, 1, 1:-1])
    # ...and the z taper stays ghost-zero (not periodic vertically).
    assert not arr[:, :, 0].any()
    assert not arr[:, :, -1].any()
    # The wrap is meaningful only if the faces actually differ.
    assert not np.array_equal(arr[1, 1:-1, 1:-1], arr[-2, 1:-1, 1:-1])


def _assert_border_zero(arr: np.ndarray) -> None:
    assert not arr[0].any() and not arr[-1].any()
    assert not arr[:, 0].any() and not arr[:, -1].any()
    assert not arr[:, :, 0].any() and not arr[:, :, -1].any()


def test_ghost_border_wraps_to_opposite_faces(small_periodic_renderer):
    arr = _read_volume_texture(small_periodic_renderer)
    assert arr[1:-1, 1:-1, 1:-1].max() > 0.0
    _assert_border_periodic(arr)


def test_set_periodic_rewrites_only_the_border(small_periodic_renderer):
    r = small_periodic_renderer
    before = _read_volume_texture(r)

    r.set_periodic(False)
    assert r.periodic is False
    off = _read_volume_texture(r)
    _assert_border_zero(off)
    np.testing.assert_array_equal(
        off[1:-1, 1:-1, 1:-1], before[1:-1, 1:-1, 1:-1]
    )

    r.set_periodic(True)
    assert r.periodic is True
    np.testing.assert_array_equal(_read_volume_texture(r), before)


def test_periodic_flag_is_packed_and_scene_identity(small_periodic_renderer):
    import cloudyview as cv

    r = small_periodic_renderer
    r.set_periodic(True)
    r.write_uniforms(cv.Camera(), (32, 32), jitter=False)
    assert r._current_uniform.shape == (21, 4)
    assert r._current_uniform[20, 0] == 1.0
    key_on = r._current_uniform_key
    r.set_periodic(False)
    r.write_uniforms(cv.Camera(), (32, 32), jitter=False)
    assert r._current_uniform[20, 0] == 0.0
    assert r._current_uniform_key != key_on  # toggling resets accumulation
    r.set_periodic(True)


def test_periodic_requires_sun_above_horizon(small_periodic_renderer):
    import cloudyview as cv

    r = small_periodic_renderer
    r.set_periodic(True)
    with pytest.raises(ValueError, match="above the horizon"):
        r.write_uniforms(cv.Camera(), (32, 32), sun_elevation=0.0)
    with pytest.raises(ValueError, match="above the horizon"):
        r.write_uniforms(cv.Camera(), (32, 32), sun_elevation=-10.0)


# ---------------------------------------------------------------------------
# Render-level seam correctness
# ---------------------------------------------------------------------------

def test_render_is_invariant_under_one_period_translation(twpice128_renderer):
    """Cameras one full domain period apart see the identical tiled scene.

    rel x = -1 and rel x = +1 are the same wrapped position; a seam bug
    (wrong ghost fill, wrong wrap phase) breaks this equality at the pixels
    whose rays cross the boundary.
    """
    import cloudyview as cv

    r = twpice128_renderer
    r.set_periodic(True)
    size = (192, 108)
    cam_a = cv.Camera(position=(-1.0, 0.2, -0.5), azimuth=70.0, elevation=8.0)
    cam_b = cv.Camera(position=(1.0, 0.2, -0.5), azimuth=70.0, elevation=8.0)
    img_a = r.render(cam_a, size=size, jitter=False)
    img_b = r.render(cam_b, size=size, jitter=False)
    delta = np.abs(img_a.astype(np.int16) - img_b.astype(np.int16))
    # fp32 wrap arithmetic differs by ULPs between the two origins; anything
    # beyond a couple of code values is a real seam-phase error.
    assert delta.max() <= 2
    assert delta.mean() < 0.05

    # Sanity that the invariance has power: the same pair diverges without
    # tiling (rel x = +1 sits at the opposite wall, seeing different clouds).
    r.set_periodic(False)
    off_a = r.render(cam_a, size=size, jitter=False)
    off_b = r.render(cam_b, size=size, jitter=False)
    assert np.abs(off_a.astype(np.int16) - off_b.astype(np.int16)).mean() > 2.0
    r.set_periodic(True)


def test_render_is_continuous_across_camera_wrap_seam(twpice128_renderer):
    """The image must not jump when the camera crosses an x-domain face.

    The two camera positions are adjacent modulo the domain (0.01% of a
    period apart on either side).  Measure the center image column directly:
    finite-box rendering sees unrelated opposite walls, while periodic
    rendering sees the same local field through the wrap.
    """
    import cloudyview as cv

    r = twpice128_renderer
    size = (192, 108)
    epsilon = 1e-4
    cameras = [
        cv.Camera(
            position=(x, 0.1, -0.8), azimuth=45.0, elevation=5.0, fov=70.0
        )
        for x in (1.0 - epsilon, -1.0 + epsilon)
    ]

    contrasts = {}
    for periodic in (False, True):
        r.set_periodic(periodic)
        left, right = [
            r.render(camera, size=size, jitter=False) for camera in cameras
        ]
        center = size[0] // 2
        contrasts[periodic] = float(np.mean(np.abs(
            left[:, center].astype(np.int16)
            - right[:, center].astype(np.int16)
        )))

    assert contrasts[False] > 3.0
    assert contrasts[True] < 0.75
    assert contrasts[True] < 0.1 * contrasts[False]
    r.set_periodic(True)


def test_periodic_removes_the_domain_wall(twpice128_renderer):
    """Low view across a lateral face: tiling must change what the ray sees
    (wrapped clouds instead of the ghost-zero taper into empty sky/ocean)."""
    import cloudyview as cv

    r = twpice128_renderer
    size = (192, 108)
    cam = cv.Camera(position=(0.9, 0.9, -0.9), azimuth=45.0, elevation=5.0)

    r.set_periodic(True)
    on = r.render(cam, size=size, jitter=False)
    r.set_periodic(False)
    off = r.render(cam, size=size, jitter=False)
    r.set_periodic(True)

    assert np.abs(on.astype(np.int16) - off.astype(np.int16)).mean() > 2.0


# ---------------------------------------------------------------------------
# (c) periodic=off bit-exactness vs the pre-change frames
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prechange_renderer():
    """Renderer matching the reference generation, with periodic off.

    The references were rendered with the seeded FIF realization the parity
    harness uses (the default FIF tile is unseeded and differs per process).
    """
    import cloudyview as cv
    from cloudyview.ocean_fif import generate_fif_normals
    from cloudyview.soar import InteractiveRenderer

    if not _is_rtx_5080_reference_adapter():
        pytest.skip("byte-exact prechange references require the RTX 5080")

    meta = json.loads((PRECHANGE_DIR / "meta.json").read_text())
    seed = meta["fif_seed"]
    np.random.seed(seed)
    fif_normals = generate_fif_normals(
        rng=np.random.default_rng(seed), verbose=False
    )
    field = cv.load(str(DATA128))
    return InteractiveRenderer(field, periodic=False, fif_normals=fif_normals)


def _prechange_case_camera(meta: dict, name: str):
    import cloudyview as cv

    case = meta["cases"][name]
    return cv.Camera(
        position=tuple(case["camera_position"]),
        azimuth=case["camera_azimuth"],
        elevation=case["camera_elevation"],
        fov=case["camera_fov"],
    )


@pytest.mark.parametrize(
    "case_name", ["twpice128_default", "twpice128_edge", "twpice128_killcombo"]
)
def test_periodic_off_is_bit_exact_vs_prechange_frame(
    prechange_renderer, case_name
):
    from cloudyview.soar.engine import PRE_PORT_AMBIENT_TINT

    meta = json.loads((PRECHANGE_DIR / "meta.json").read_text())
    reference = np.load(PRECHANGE_DIR / f"{case_name}.npy")
    camera = _prechange_case_camera(meta, case_name)

    kwargs = {}
    if case_name == "twpice128_killcombo":
        # The documented master kill combination (engine.py): with periodic
        # off it must still reproduce the pre-port arithmetic bit-for-bit.
        kwargs = dict(
            spectral_lighting_strength=0.0,
            low_sun_sky_field_strength=0.0,
            light_transfer_split_strength=0.0,
            aerial_perspective_strength=0.0,
            ocean_realism=0.0,
            cone_stencil_theta_deg=0.0,
            ambient_tint=PRE_PORT_AMBIENT_TINT,
        )

    img = prechange_renderer.render(
        camera, size=tuple(meta["size"]), jitter=False, **kwargs
    )
    np.testing.assert_array_equal(img, reference)


# ---------------------------------------------------------------------------
# (b) App camera wrap + menu/CLI wiring
# ---------------------------------------------------------------------------

def _make_position_app(periodic=True):
    from cloudyview.soar.app import FlyThroughApp

    app = object.__new__(FlyThroughApp)
    app.periodic = periodic
    app.renderer = SimpleNamespace(
        bmin=np.array([100.0, -200.0, 0.0]),
        bmax=np.array([400.0, 300.0, 1000.0]),
    )
    return app


def test_constrain_position_wraps_horizontally_and_clamps_z():
    from cloudyview.soar.app import OCEAN_FLOOR_MARGIN_M, FlyThroughApp

    app = _make_position_app(periodic=True)
    constrained = FlyThroughApp._constrain_position(
        app, np.array([450.0, -230.0, -5.0])
    )
    assert constrained[0] == pytest.approx(150.0)   # 450 wraps past 400
    assert constrained[1] == pytest.approx(270.0)   # -230 wraps below -200
    assert constrained[2] == OCEAN_FLOOR_MARGIN_M

    inside = FlyThroughApp._constrain_position(
        app, np.array([250.0, 50.0, 500.0])
    )
    np.testing.assert_allclose(inside, [250.0, 50.0, 500.0])


def test_constrain_position_does_not_wrap_when_not_periodic():
    from cloudyview.soar.app import FlyThroughApp

    app = _make_position_app(periodic=False)
    constrained = FlyThroughApp._constrain_position(
        app, np.array([450.0, -230.0, 500.0])
    )
    np.testing.assert_allclose(constrained, [450.0, -230.0, 500.0])


def test_minimap_marker_follows_the_wrapped_camera(twpice128_renderer):
    """A camera wrapped to the opposite face gets the opposite-side marker."""
    import cloudyview as cv
    from cloudyview.soar.app import _wrap_position_horizontal
    from cloudyview.soar.engine import camera_world_origin

    r = twpice128_renderer
    hud = r.hud
    size = (480, 270)

    east_edge = cv.Camera(position=(0.999, 0.0, -0.9))
    origin = camera_world_origin(east_edge, r.bmin, r.bmax)
    origin[0] += 3.0 * (r.bmax[0] - r.bmin[0]) / r.field.shape[0]  # fly east
    wrapped = _wrap_position_horizontal(origin, r.bmin, r.bmax)
    assert r.bmin[0] <= wrapped[0] < r.bmax[0]
    assert wrapped[0] - r.bmin[0] < 0.05 * (r.bmax[0] - r.bmin[0])

    rel = 2.0 * (wrapped[0] - r.bmin[0]) / (r.bmax[0] - r.bmin[0]) - 1.0
    west_cam = cv.Camera(position=(rel, 0.0, -0.9))
    west_px = hud.marker_pixel(west_cam, size)
    east_px = hud.marker_pixel(east_edge, size)
    _x, _y, w, _h = hud.rect_for_size(size)
    assert east_px[0] - west_px[0] > 0.9 * w


def test_menu_p_toggles_periodic_only_in_main_pause_menu():
    for key in ("p", "P"):
        transition = menu_transition(True, MENU_MAIN, key)
        assert transition.action == ACTION_TOGGLE_PERIODIC
        assert transition.next_state == MENU_MAIN
    # Not a flight-time action, not a quality-menu action.
    assert menu_transition(False, MENU_MAIN, "p").action is None
    assert menu_transition(True, MENU_RENDER_QUALITY, "p").action is None


def test_toggle_periodic_updates_app_renderer_and_position():
    from cloudyview.soar.app import FlyThroughApp

    calls = []

    class DummyCanvas:
        def __init__(self):
            self.titles = []
            self.draw_requests = 0

        def set_title(self, title):
            self.titles.append(title)

        def request_draw(self, *args):
            self.draw_requests += 1

    app = object.__new__(FlyThroughApp)
    app.periodic = True
    app.canvas = DummyCanvas()
    app.renderer = SimpleNamespace(
        bmin=np.array([0.0, 0.0, 0.0]),
        bmax=np.array([100.0, 100.0, 1000.0]),
        set_periodic=lambda value: calls.append(value),
    )
    app.position = np.array([50.0, 50.0, 500.0])

    FlyThroughApp._toggle_periodic(app)
    assert app.periodic is False
    assert calls == [False]
    assert "periodic domain off" in app.canvas.titles[-1]

    FlyThroughApp._toggle_periodic(app)
    assert app.periodic is True
    assert calls == [False, True]
    assert "periodic domain on" in app.canvas.titles[-1]


def test_cli_no_periodic_flag_reaches_run_app(monkeypatch, tmp_path):
    import cloudyview.soar.__main__ as soar_main

    captured = {}

    def fake_run_app(field, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("cloudyview.soar.app.run_app", fake_run_app)
    monkeypatch.setattr("cloudyview.cloudfield.load", lambda path, ice=None: "field")

    soar_main.main([str(DATA128), "--no-periodic"])
    assert captured["periodic"] is False

    captured.clear()
    soar_main.main([str(DATA128)])
    assert captured["periodic"] is True


def test_cli_tier_and_fp16_volume_reach_run_app(monkeypatch):
    import cloudyview.soar.__main__ as soar_main

    captured = {}

    monkeypatch.setattr(
        "cloudyview.soar.app.run_app",
        lambda field, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "cloudyview.cloudfield.load", lambda path, ice=None: "field"
    )

    soar_main.main([str(DATA128)])
    assert captured["tier"] == "auto"
    # None = auto since 2026-07-17: choose_volume_fp16 resolves per field.
    assert captured["volume_fp16"] is None

    captured.clear()
    soar_main.main([str(DATA128), "--tier", "low", "--fp16-volume"])
    assert captured["tier"] == "low"
    assert captured["volume_fp16"] is True

    captured.clear()
    soar_main.main([str(DATA128), "--fp32-volume"])
    assert captured["volume_fp16"] is False


# ---------------------------------------------------------------------------
# Behold hand-off notice predicate
# ---------------------------------------------------------------------------

def test_view_spans_domain_edge_predicate():
    import cloudyview as cv
    from cloudyview.soar.app import view_spans_domain_edge

    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([25600.0, 25600.0, 5000.0])

    # Center of the domain looking steeply up: every frustum ray leaves
    # through the domain top long before any lateral wall.
    up = cv.Camera(azimuth=0.0, elevation=80.0, fov=40.0)
    assert not view_spans_domain_edge(
        [12800.0, 12800.0, 500.0], up, bmin, bmax, aspect=16 / 9
    )

    # Near a wall looking horizontally: the march crosses the boundary while
    # wrapped volume is still visible -> behold (finite volume) will differ.
    low = cv.Camera(azimuth=0.0, elevation=0.0, fov=60.0)
    assert view_spans_domain_edge(
        [25000.0, 12800.0, 500.0], low, bmin, bmax, aspect=16 / 9
    )


def test_periodic_march_cap_matches_closed_form():
    from cloudyview.soar.engine import (
        PERIODIC_AIR_TAU_CUTOFF,
        PERIODIC_MAX_WRAPS,
        periodic_march_cap_m,
    )
    from cloudyview.witness import AERIAL_BETA_PER_KM

    bmin = np.array([0.0, 0.0, 0.0])
    small = np.array([25600.0, 25600.0, 5000.0])
    large = np.array([204800.0, 204800.0, 27000.0])
    horizontal = [1.0, 0.0, 0.0]

    # Small domain at sea level: the 2-wrap ceiling binds.
    assert periodic_march_cap_m(0.0, horizontal, bmin, small) == pytest.approx(
        PERIODIC_MAX_WRAPS * 25600.0
    )
    # Large domain: the ~2% clear-air transmittance distance binds.
    expected_t = PERIODIC_AIR_TAU_CUTOFF / (AERIAL_BETA_PER_KM * 1e-3)
    assert periodic_march_cap_m(0.0, horizontal, bmin, large) == pytest.approx(
        expected_t
    )
    # High-altitude upward ray: haze is left behind, only the ceiling caps.
    upward = np.array([0.6, 0.0, 0.8])
    assert periodic_march_cap_m(
        12000.0, upward, bmin, large
    ) == pytest.approx(PERIODIC_MAX_WRAPS * 204800.0 / 0.6)
