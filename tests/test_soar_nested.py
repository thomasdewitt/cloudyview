"""Tests for nested (two-level) domains in the soar WGSL engine.

Soar mirrors witness's nested-domain model with exactly two levels: an
outer field that bounds the march, and an optional finer "nest" placed by
its own absolute coordinates inside it. Wherever the nest covers, it wins
and the march refines to its voxel scale.

Coverage:
- placement validation (containment is an error, not a silent clip);
- both volumes resident, sharing one texture format;
- per-level march step sizes in the uniforms;
- shader specialization keys (nested/periodic/light-step variants);
- render level: the nest is actually sampled, and in a periodic domain it
  tiles with the parent rather than appearing once.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

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


# ---------------------------------------------------------------------------
# Synthetic two-level scene
# ---------------------------------------------------------------------------
#
# The outer field is deliberately EMPTY: every cloud pixel in these renders
# can then only have come from the nest, which makes "is the second level
# sampled at all" a one-line assertion instead of a difference test against
# a busy field.

OUTER_N = 32
OUTER_NZ = 16
VOXEL_M = 100.0
REFINE = 8


def _outer_field(lwc=None):
    from cloudyview.cloudfield import CloudField

    x = (np.arange(OUTER_N) + 0.5) * VOXEL_M
    y = (np.arange(OUTER_N) + 0.5) * VOXEL_M
    z = (np.arange(OUTER_NZ) + 0.5) * VOXEL_M
    if lwc is None:
        lwc = np.zeros((OUTER_N, OUTER_N, OUTER_NZ), dtype=np.float32)
    return CloudField(lwc=lwc, x=x, y=y, z=z)


def _nest_field(n=64, x0=1200.0, y0=1200.0, z0=400.0):
    """A REFINE-times finer blob centered in the outer domain."""
    from cloudyview.cloudfield import CloudField

    fine = VOXEL_M / REFINE
    x = x0 + (np.arange(n) + 0.5) * fine
    y = y0 + (np.arange(n) + 0.5) * fine
    z = z0 + (np.arange(n) + 0.5) * fine
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(
        (xx - x.mean()) ** 2 + (yy - y.mean()) ** 2 + (zz - z.mean()) ** 2
    )
    lwc = (np.exp(-((r / (0.35 * n * fine)) ** 2)) * 2.0).astype(np.float32)
    return CloudField(lwc=lwc, x=x, y=y, z=z)


def _tiny_fif_normals():
    flat = np.zeros((8, 8), dtype=np.float32)
    up = np.ones((8, 8), dtype=np.float32)
    return (flat, flat, up, 1.0)


def _renderer(*, nest=None, periodic=False, **kwargs):
    from cloudyview.soar import InteractiveRenderer

    return InteractiveRenderer(
        _outer_field(),
        nest=nest,
        periodic=periodic,
        ocean_enabled=False,
        fif_normals=_tiny_fif_normals(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def nested_renderer():
    return _renderer(nest=_nest_field(), periodic=False)


# ---------------------------------------------------------------------------
# (a) Placement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [(1e6, 0.0, 0.0), (0.0, -1e6, 0.0)])
def test_nest_outside_the_outer_box_is_an_error(shift):
    """Placement comes from the files' own coordinates; a miss must shout."""
    nest = _nest_field()
    moved = type(nest)(
        lwc=nest.lwc,
        x=nest.x + shift[0], y=nest.y + shift[1], z=nest.z + shift[2],
    )
    with pytest.raises(ValueError, match="must lie inside"):
        _renderer(nest=moved)


def test_nest_poking_out_of_the_top_is_an_error():
    nest = _nest_field()
    tall = type(nest)(
        lwc=nest.lwc, x=nest.x, y=nest.y,
        z=nest.z + OUTER_NZ * VOXEL_M,
    )
    with pytest.raises(ValueError, match="must lie inside"):
        _renderer(nest=tall)


def test_nest_sharing_a_face_with_the_parent_is_accepted():
    """Float noise on a coincident face must not read as "outside"."""
    nest = _nest_field()
    flush = type(nest)(
        lwc=nest.lwc,
        x=nest.x - nest.x[0] + 0.5 * (VOXEL_M / REFINE),
        y=nest.y, z=nest.z,
    )
    r = _renderer(nest=flush)
    assert r.nested is True


def test_no_nest_leaves_the_renderer_single_level():
    r = _renderer()
    assert r.nested is False
    assert r.nest is None
    assert r.nest_nbytes == 0
    # The dummy stand-in is still bound so one layout serves both shaders.
    assert r._nest_texture is not None


# ---------------------------------------------------------------------------
# (b) Residency and step sizes
# ---------------------------------------------------------------------------

def test_both_levels_are_resident_and_share_a_format(nested_renderer):
    r = nested_renderer
    assert r.nest_nbytes > 0
    assert r.resident_nbytes == r.volume_nbytes + r.nest_nbytes
    assert r._texture.format == r._nest_texture.format


def test_nest_marches_at_its_own_finer_step(nested_renderer):
    r = nested_renderer
    assert r.dt_view_nest == pytest.approx(r.dt_view / REFINE)
    assert r.dt_light_nest == pytest.approx(r.dt_view_nest)


def test_step_sizes_track_the_quality_tier(nested_renderer):
    r = nested_renderer
    ratios = []
    for tier in ("high", "low"):
        r.set_quality_tier(tier, camera_moving=True)
        ratios.append(r.dt_view / r.dt_view_nest)
    r.set_quality_tier("high", camera_moving=False)
    # Both levels scale with the same preset factor, so the refinement
    # ratio is a property of the grids, not of the tier.
    assert ratios[0] == pytest.approx(ratios[1])


def test_uniforms_carry_the_nest_box_and_steps(nested_renderer):
    import cloudyview as cv
    from cloudyview.soar.engine import _UNIFORM_ROWS

    r = nested_renderer
    r.write_uniforms(cv.Camera(), (64, 64), jitter=False)
    u = r._current_uniform
    assert u.shape == (_UNIFORM_ROWS, 4)
    np.testing.assert_allclose(u[21, :3], r.nest_bmin, rtol=1e-6)
    np.testing.assert_allclose(u[22, :3], r.nest_bmax, rtol=1e-6)
    assert u[21, 3] == pytest.approx(r.dt_view_nest, rel=1e-6)
    assert u[22, 3] == pytest.approx(r.dt_light_nest, rel=1e-6)


def test_uniform_nest_rows_are_zero_without_a_nest():
    import cloudyview as cv

    r = _renderer()
    r.write_uniforms(cv.Camera(), (64, 64), jitter=False)
    np.testing.assert_array_equal(r._current_uniform[21:23], 0.0)


# ---------------------------------------------------------------------------
# (c) Shader specialization
# ---------------------------------------------------------------------------

def test_shader_specializations_are_distinct_and_cached(nested_renderer):
    r = nested_renderer
    first = r._shader_for(True, True, 512)
    assert r._shader_for(True, True, 512) is first
    variants = {
        (periodic, nested, steps): r._shader_for(periodic, nested, steps)
        for periodic in (True, False)
        for nested in (True, False)
        for steps in (512, 64)
    }
    assert len({id(module) for module in variants.values()}) == len(variants)


def test_shader_source_declares_both_specialization_consts():
    from cloudyview.soar import engine

    source = engine.SHADER_PATH.read_text()
    assert source.count("const PERIODIC_DOMAIN: bool = true;") == 1
    assert source.count("const NESTED: bool = false;") == 1


def test_shader_rejects_a_missing_sentinel(nested_renderer, monkeypatch):
    """A renamed const must fail loudly, not compile the wrong variant."""
    r = nested_renderer
    monkeypatch.setattr(
        r, "_shader_source", r._shader_source.replace(
            "const NESTED: bool = false;", "const NESTED : bool = false;"
        )
    )
    monkeypatch.setattr(r, "_shader_modules", {})
    with pytest.raises(RuntimeError, match="sentinel"):
        r._shader_for(True, True, 512)


# ---------------------------------------------------------------------------
# (d) Rendering
# ---------------------------------------------------------------------------

_CAM_KW = dict(azimuth=0.0, elevation=0.0, fov=60.0)


def test_nest_is_sampled_where_the_outer_field_is_empty(nested_renderer):
    """Outer field is all zeros: any cloud in frame came from the nest."""
    import cloudyview as cv

    cam = cv.Camera(position=(0.0, -0.999, 0.0), **_CAM_KW)
    empty = _renderer().render(cam, size=(128, 72), jitter=False)
    nested = nested_renderer.render(cam, size=(128, 72), jitter=False)

    # Sky-only is smooth top-to-bottom; the nest adds a bright, shaded blob.
    assert nested.std() > empty.std() + 2.0
    assert np.abs(nested.astype(int) - empty.astype(int)).max() > 40


def test_nest_tiles_with_the_parent_when_periodic():
    """One scene, one tile: a domain away, the nest is still in front."""
    import cloudyview as cv

    nest = _nest_field()
    finite = _renderer(nest=nest, periodic=False)
    tiled = _renderer(nest=nest, periodic=True)
    # Two full domains south of the box: the finite scene shows nothing,
    # the tiled one shows the neighbouring copy of the nest.
    cam = cv.Camera(position=(0.0, -2.999, 0.0), **_CAM_KW)
    a = finite.render(cam, size=(128, 72), jitter=False)
    b = tiled.render(cam, size=(128, 72), jitter=False)
    assert b.std() > a.std() + 2.0


def test_nested_render_is_stable_across_periodic_toggle(nested_renderer):
    """set_periodic() rewrites only the outer border; the nest is untouched."""
    import cloudyview as cv

    r = nested_renderer
    cam = cv.Camera(position=(0.0, -0.999, 0.0), **_CAM_KW)
    before = r.render(cam, size=(96, 54), jitter=False)
    r.set_periodic(True)
    r.set_periodic(False)
    after = r.render(cam, size=(96, 54), jitter=False)
    np.testing.assert_array_equal(before, after)


# ---------------------------------------------------------------------------
# (e) CLI / app wiring
# ---------------------------------------------------------------------------

def test_cli_exposes_nest_arguments():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "cloudyview.soar", "--help"],
        capture_output=True, text=True, check=True,
        cwd=str(Path(__file__).parent.parent),
    ).stdout
    assert "--nest" in out
    assert "--nest-group" in out
    assert "--nest-ice" in out


def test_reproduction_command_includes_the_nest():
    from types import SimpleNamespace

    import cloudyview as cv
    from cloudyview.soar.app import FlyThroughApp

    app = object.__new__(FlyThroughApp)
    app.renderer = SimpleNamespace(
        field=SimpleNamespace(source="outer.nc", ice_source=None),
        nest=SimpleNamespace(source="fine.nc", ice_source=None),
        quality_tier="high",
        volume_fp16=False,
    )
    command = FlyThroughApp._soar_reproduction_command(app, cv.Camera())
    assert "--nest fine.nc" in command
