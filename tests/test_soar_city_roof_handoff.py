"""
A roof shows the same with its props drawn and with their far mean.

rooftopworks draws its roof props (tanks, stairs, units, runs, their lamps)
inside CITY_PROP_RANGE and, past it, adds what they would have shown as a
mean on the deck (the roof hook: the lamps' far table, the skylight field and
door pool kept off what the props hide). The far mean is pinned to its own
model of the props by tests/test_soar_city_roof_lamps.py; this test pins it
to the RENDERED image instead, which is what the hand-over is seen in.

Two builds of the real shader render the same views through the real host:
NEAR, whose props are drawn at every distance (CITY_PROP_RANGE pushed past
the scene), and FAR, whose props are drawn nowhere, so the far mean stands in
everywhere, keyed on each pixel's own footprint as if the hand-over were at
that pixel's distance. Over the pixels that FAR sees as an urban building's
roof deck, the mean SHOWN luminance of the two must agree within the view's
bar: that is the step a flight would see crossing the range. Shown, not
linear, because the accumulator averages tone-mapped samples, so a roof at
rest shows the mean of what its samples show (the core's flat facade rests
on the same fact, city_windows_seen). At two
tiers' roof clutter densities, since the far mean scales with it. NEAR's
prop pixels that fall off FAR's roof (a stair at a deck's edge seen against
the street) count with NEAR's roof: their light is what FAR spreads on the
deck. (What they hide behind them is the props' geometry, which no mean on
the deck can stand in for.)

The views look down at roofs from a few hundred metres to a couple of
kilometres, at a LOD angle that puts the footprint there at the scale of the
real hand-over (1-5 m). Needs a real GPU; skips cleanly without one.

Measured (2026-11-10, densities 0.7 / 1.4): oblique -2.7% / -3.0% against a
bar of 4%, steep -0.7% / -0.8% against 2%. Each bar is held honest by a
negative control (test_a_far_mean_without_its_lamps_fails): FAR with its
lamp term zeroed comes out -7.5% oblique and -3.1% steep, outside both.
The far lamps are shown as the small bright faces they light up close (their
second moment, cc_rooftopworks_lamps_far); spread linearly over the deck they
showed 1.8 times their near light and the step was +5-6%. Two parts of the
far mean are known and unresolved, and are oblique's shortfall:
(b) the props' own faces stand in for deck up close, at their own radiance
    (a "cover" element in the same mixture);
(c) a parapet's lit coping trim vanishes past the range with nothing in its
    place (NEAR is not credited the parapet or the pipes seen off the roof,
    since the far mean stands in for neither).

Run with: uv run --extra dev python -m pytest tests/test_soar_city_roof_handoff.py -v -s
"""

from pathlib import Path

import numpy as np
import pytest
from .conftest import ShaderModule

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
    from .test_soar_city_shell_continuity import TILE_OFFSET
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
    from test_soar_city_shell_continuity import TILE_OFFSET

SHADER = ShaderModule("city")
RANGE = "const CITY_PROP_RANGE: f32 = 2200.0;"
FP_H = "        let fp_h = lod_pixel_angle() * CITY_PROP_RANGE;"
TRACE = ("    let chit = city_trace(city_o, dir, 0.0, CITY_TRACE_RANGE, "
         "CITY_TRACE_ALL);\n")
LAMPS_FAR = ("        far = cc_rooftopworks_lamps_far(cc, h.cell, fp_h, d, cr, bk, "
             "dir);")
# FAR's mask: the pixel's hit is a core roof (kind 2) of a building whose
# roof rooftopworks furnishes (its own gate in the props trace and the hook).
MASK = """    {
        let mc = city_cell(chit.cell);
        let roof = chit.hit && chit.kind == 2 && mc.built && mc.arch != 4
                   && mc.land_use == CITY_USE_URBAN;
        // rooftopworks' props, but not the parapet ring or the pipes: the
        // far mean stands in for neither (they carry no lamp; the coping
        // trim is residual (c)), so NEAR is not credited their light.
        let prop = chit.hit && chit.kind >= ROOF_KINDS.x
                   && chit.kind < ROOF_KINDS.y
                   && chit.kind != ROOF_KINDS.x + 8
                   && chit.kind != ROOF_KINDS.x + 12;
        return vec4<f32>(select(0.0, 1.0, roof), select(0.0, 1.0, prop), 0.0,
                         1.0);
    }
"""
# rooftopworks' hit kinds (its registry kind_base, sixteen of them).
ROOF_KINDS = "vec2<i32>(600, 616)"
# (x, y, z, azimuth, elevation, fov) in the test city's frame; the LOD angle
# (degrees) that sets the footprint the props hand over at.
# (camera, LOD angle, the bar the view is held to; see the docstring for
# what steep still lacks).
VIEWS = {
    "oblique": ((16500.0, 71500.0, 700.0, 45.0, -24.0, 40.0), 0.12, 0.04),
    "steep": ((17000.0, 72200.0, 900.0, 30.0, -55.0, 35.0), 0.20, 0.02),
}
W, H = 480, 270
FRAMES = 32
# Two tiers' roof clutter densities (the tiers' city_roof_density: low, high).
DENSITIES = (0.7, 1.4)


def _builds():
    source = SHADER.read_text()
    assert source.count(RANGE) == 1, "CITY_PROP_RANGE's declaration moved"
    assert source.count(FP_H) == 1, "the roof hook's fp_h line moved"
    assert source.count(TRACE) == 1, "surface_pass_city's trace moved"
    near = source.replace(RANGE, "const CITY_PROP_RANGE: f32 = 1.0e7;")
    far = source.replace(RANGE, "const CITY_PROP_RANGE: f32 = 1.0;").replace(
        FP_H, "        let fp_h = fp;")
    probe = MASK.replace("ROOF_KINDS", ROOF_KINDS)
    assert far.count(LAMPS_FAR) == 1, "the roof hook's far lamps call moved"
    return {"near": near, "far": far,
            "far_nolamps": far.replace(LAMPS_FAR, ""),
            "mask": far.replace(TRACE, TRACE + probe),
            "near_mask": near.replace(TRACE, TRACE + probe)}


@pytest.fixture(scope="module")
def scene():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar.renderer import SoarRenderer
    from cloudyview.soar.uniforms import SceneState
    from cloudyview.soar.uniforms import DEFAULT_OCEAN_REFLECTANCE

    level = build_soar_level()
    rs = {}
    for name, src in _builds().items():
        r = SoarRenderer(periodic=True, surface="city", shader_source=src)
        r.upload_volume(level.sigma)
        rs[name] = r
    meta = rs["near"].surface_meta
    dx = float(min(level.dx))
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dx * 2.0, dt_light=dx * 2.0, periodic=True,
        ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE,
        ocean_fif_dx=float(meta["cell_m"]),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1, ocean_enabled=True,
        surface="city")
    return rs, state


def _render(scene, name, cam, lod, frames, jitter, density):
    from cloudyview.soar.uniforms import ViewState
    rs, state = scene
    x, y, z, az, el, fov = cam
    v = ViewState(
        camera_position=[x + TILE_OFFSET[0], y + TILE_OFFSET[1], z],
        azimuth=az, elevation=el, fov=fov, output_size=(W, H),
        render_size=(W, H), jitter=jitter, subpixel=jitter,
        sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
        view_step_lod_degrees=lod, city_roof_density=density)
    img = np.asarray(rs[name].render(state, v, frames=frames),
                     dtype=np.float64)
    assert np.isfinite(img).all(), f"{name} produced non-finite values"
    return img


@pytest.mark.gpu
@pytest.mark.parametrize("density", DENSITIES)
@pytest.mark.parametrize("view", list(VIEWS))
def test_roofs_show_the_same_near_and_far(scene, view, density):
    cam, lod, tol = VIEWS[view]
    shown, roof_share = _shown(scene, cam, lod, density, ("near", "far"))
    rel = shown["far"] / shown["near"] - 1.0
    print(f"{view} at density {density:g}: roofs {roof_share:.1%} of the "
          f"frame, shown near "
          f"{shown['near']:.5f} far {shown['far']:.5f} ({rel:+.2%})")
    assert abs(rel) <= tol, (view, density, shown)


def _shown(scene, cam, lod, density, builds):
    """Each build's shown luminance over FAR's roofs (NEAR's with its props
    seen off them), per roof pixel; and the roofs' share of the frame."""
    roof = _render(scene, "mask", cam, lod, 1, False, density)[..., 0] > 0.5
    assert roof.mean() > 0.10, f"too few roof pixels in view ({roof.mean()})"
    # NEAR's props seen OFF the roof: a stair stands at a deck's edge, its
    # lit door facing in, so seen from inside the roof its face is backed by
    # the street below. That light is the props' all the same, and FAR
    # carries it on the deck; it is counted with NEAR's roof.
    near_props = _render(scene, "near_mask", cam, lod, 1, False,
                         density)[..., 1] > 0.5
    off = near_props & ~roof
    lum = np.array([0.2126, 0.7152, 0.0722])
    shown = {}
    for name in builds:
        y = _render(scene, name, cam, lod, FRAMES, True, density)[..., :3] @ lum
        shown[name] = float(y[roof].sum()
                            + (y[off].sum() if name == "near" else 0.0))
    return {k: v / roof.sum() for k, v in shown.items()}, float(roof.mean())


@pytest.mark.gpu
@pytest.mark.parametrize("view", list(VIEWS))
def test_a_far_mean_without_its_lamps_fails(scene, view):
    """The negative control: FAR with its lamp term zeroed must fall out of
    the view's bar, or the bar could not see that term go wrong."""
    cam, lod, tol = VIEWS[view]
    shown, _ = _shown(scene, cam, lod, 1.4, ("near", "far_nolamps"))
    rel = shown["far_nolamps"] / shown["near"] - 1.0
    print(f"{view} without the far lamps: {rel:+.2%} (bar {tol:.0%})")
    assert abs(rel) > tol, (view, shown)
