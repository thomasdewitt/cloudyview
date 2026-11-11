"""One pixel's angle, defined once: 2 tan(hfov/2) / render width.

fov is the HORIZONTAL field of view (camera.py), and camera_ray's pixels are
square (ndc_y carries tan(hfov/2) / aspect), so a pixel subtends
2 tan(hfov/2) / w both across and up. The sea's two LOD rules divided by the
image HEIGHT instead (ocean found it, 2026-11-08), so at 16:9 every sea mip
took a pixel as 16/9 its true size, 0.83 of a mip too coarse, from the first
port of the ocean on. Every other surface already used the width.

Three pins:
  * the shader: the only expression that turns tan(hfov/2) into a pixel
    angle is true_pixel_angle(); every surface and component calls it (or
    lod_pixel_angle(), its max with the LOD slider), so the next one cannot
    grow its own rule; and the image height is read only where camera_ray
    places the pixel's ray;
  * the hosts: the pixel angle both packers floor the view-step LOD at is the
    one the shader computes from the same uniforms (test_uniform_parity pins
    uniforms.js to soar/uniforms.py, so this pins both);
  * the image: a pixel's sea does not depend on how tall the image is. Row
    py of an h-tall frame and row py + k of an (h + 2k)-tall frame cast the
    same ray (ndc_y = (h - 2 py - 1) tan(hfov/2) / w), so with jitter off they
    must shade the same water. They agree to float rounding, not bit for bit
    (ndc_y's float arithmetic differs with h): 3e-5 at most on the fix, where
    the height rule put whole rows of sea 1e-2 to 3e-2 apart. The sea is rendered with
    jitter off, so the one-frame render keeps the rays aligned.
"""

import re

import numpy as np
import pytest

from tests.conftest import REPO, shader_paths

SOAR = REPO / "cloudyview" / "soar"
SOURCES = shader_paths()
PIXEL_ANGLE = "2.0 * u.tan_half_fov / u.image_width"


def _code(path):
    return re.sub(r"//[^\n]*", "", path.read_text())


def test_tan_half_fov_becomes_a_pixel_angle_in_one_place():
    defs, strays = [], []
    for path in SOURCES:
        for n, line in enumerate(_code(path).splitlines(), 1):
            if "u.tan_half_fov" not in line:
                continue
            if PIXEL_ANGLE in line:
                defs.append((path.name, n))
            elif not re.fullmatch(r"\s*let tan_half_fov = u\.tan_half_fov;",
                                  line):
                strays.append((path.name, n, line.strip()))
    assert strays == [], (
        "a pixel angle (or footprint) computed from u.tan_half_fov outside "
        f"true_pixel_angle(); call it instead: {strays}")
    assert len(defs) == 1, defs
    body = re.search(r"fn true_pixel_angle\(\) -> f32 \{([^}]*)\}",
                     _code(SOAR / "shader" / "camera.wgsl"))
    assert body and PIXEL_ANGLE in body.group(1)


def test_the_image_height_is_read_only_to_place_the_ray():
    """A footprint written over the height without u.tan_half_fov would
    slip past the lint above; u.image_height has one reader, camera_ray."""
    readers = []
    for path in SOURCES:
        code = _code(path)
        for m in re.finditer(r"u\.image_height\b", code):
            fn = re.findall(r"\bfn (\w+)\(", code[:m.start()])
            readers.append((path.name, fn[-1] if fn else None))
    assert readers == [("camera.wgsl", "camera_ray")], readers


@pytest.mark.parametrize("render_size", [(1600, 900), (900, 1600), (640, 640)])
def test_the_hosts_floor_the_lod_at_the_shaders_pixel_angle(render_size):
    from cloudyview.soar.uniforms import SceneState, ViewState, pack_uniforms
    state = SceneState(bmin=(0.0, 0.0, 0.0), bmax=(1000.0, 1000.0, 1000.0),
                       dt_view=10.0, dt_light=10.0, periodic=False)
    # A view-step LOD far finer than a pixel, so the quarter-pixel floor is
    # what u.view_lod_tan carries.
    view = ViewState(camera_position=(500.0, 500.0, 500.0), azimuth=0.0,
                     elevation=0.0, fov=70.0, output_size=render_size,
                     render_size=render_size, view_step_lod_degrees=1e-6)
    u = pack_uniforms(state, view).astype(np.float64)
    shader_pixel_angle = 2.0 * u[0, 3] / u[17, 0]    # true_pixel_angle()
    assert u[17, 0] == render_size[0]
    assert u[19, 1] == pytest.approx(0.25 * shader_pixel_angle, rel=1e-6)


def _sea(h, jitter, device=None):
    from cloudyview.soar.renderer import SoarRenderer
    from cloudyview.soar.surfaces import surface_cell_m
    from cloudyview.soar.uniforms import SceneState, ViewState
    from cloudyview.soar.uniforms import DEFAULT_OCEAN_REFLECTANCE
    r = SoarRenderer(periodic=False, surface="ocean", tone_map=False,
                     device=device)
    # No cloud: a clear box well above the camera, so the frame is sea and
    # sky.
    r.upload_volume(np.zeros((8, 8, 4), np.float32))
    meta = r.surface_meta
    state = SceneState(
        bmin=(0.0, 0.0, 3000.0), bmax=(8000.0, 8000.0, 4000.0),
        dt_view=100.0, dt_light=100.0, periodic=False,
        ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE,
        ocean_fif_dx=surface_cell_m("ocean", meta),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1, surface="ocean")
    # 40 m up looking 12 degrees down: the sea runs from a few metres to the
    # horizon, so the rows span many mip levels.
    view = ViewState(camera_position=(4000.0, 4000.0, 40.0), azimuth=30.0,
                     elevation=-12.0, fov=70.0, output_size=(96, h),
                     render_size=(96, h), jitter=jitter, sun_azimuth=210.0,
                     sun_elevation=25.0, exposure=1.0, light_cache=False)
    return r.render(state, view, frames=1), r.device


def test_a_pixels_sea_does_not_depend_on_the_image_height():
    from tests.conftest import soar_gpu_adapter
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count)")
    short, device = _sea(54, jitter=False)
    tall, _ = _sea(96, jitter=False, device=device)
    k = (96 - 54) // 2
    same_rays = tall[k:k + 54]
    sea = np.arange(54) >= 30          # the rows below the horizon
    diff = np.abs(short[sea] - same_rays[sea]).max()
    assert diff < 1e-3, (
        f"the same ray shades the sea differently in a 96x54 and a 96x96 "
        f"frame (max |diff| {diff:.3g}): a footprint depends on the height")
