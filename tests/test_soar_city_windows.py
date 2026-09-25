"""
A resolved pane comes back as itself.

The city's flat facade averages the windows under a pixel's footprint as
they are SHOWN (city_windows_seen) and hands back the radiance that shows as
that average, in the colour of their light (city_windows_resolve). Whatever
else that rule does for a footprint holding many windows, a footprint
holding one element must come back as that element; otherwise a pane
resolved on the flat facade would change its brightness or colour against
the same pane one pixel nearer, where the shell draws it, and panes would
pop across the hand-off.

This drives city_windows_resolve directly: a probe variant of the plain pass
turns each pixel into one element, its radiance log-spaced across the frame
(0.01 to 30, through the display curve's toe, knee and highlight shoulder)
and its colour one of the room palette's bands or grey down it. At the night
exposure and below it, every element must SHOW as itself within
SHOWN_TOL, and, wherever the display has not clipped it to white, come back
as its own radiance within ROUND_TRIP_TOL. (Past the white point every
radiance shows the same white, so none can be told back from it; any
radiance that shows as that white is the same pixel.)

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/test_soar_city_windows.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

SHADER = Path(__file__).resolve().parents[1] / "cloudyview/soar/raymarch.wgsl"
ANCHOR = "    let city_o = city_camera_origin();\n    let chit = city_trace("
# One element per pixel: across the frame (dir.x) its radiance, down it
# (dir.z) its colour. It returns the relative error of the round trip, per
# channel, in place of the surface's colour.
PROBE = """    {
        let a = clamp(0.5 + 0.7 * dir.x, 0.0, 1.0);
        let b = clamp(0.5 + 0.7 * dir.z, 0.0, 0.999);
        let k = i32(floor(b * 7.0));
        var c = vec3<f32>(0.5);
        if (k < 6) {
            c = city_room_band_colour(k);
        }
        // @domain: the base is 10
        let l = c * pow(10.0, -2.0 + 3.5 * a);
        let fog = vec4<f32>(1.0, 0.0, 0.0, 0.0);
        let mean = city_win_add(city_win_zero(), l, 1.0, fog, true);
        let back = city_windows_resolve(mean, fog, true);
        let err = abs(back - l) / max(l, vec3<f32>(1e-6));
        let d0 = city_display_curve(l * u.cam_right.w);
        let d1 = city_display_curve(back * u.cam_right.w);
        let white = select(0.0, 1.0, min(min(d0.x, d0.y), d0.z) >= 0.999);
        return vec4<f32>(max(max(err.x, err.y), err.z),
                         max(max(abs(d1.x - d0.x), abs(d1.y - d0.y)),
                             abs(d1.z - d0.z)), white, 1.0);
    }
"""
ROUND_TRIP_TOL = 0.03
SHOWN_TOL = 0.01
EXPOSURES = (2.0, 4.0, 6.0)
W, H = 256, 144


@pytest.fixture(scope="module")
def probe():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar_host import SceneState, SoarRenderer
    from cloudyview.witness import OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(ANCHOR) == 1, (
        "surface_pass_city's opening moved; update ANCHOR")
    level = build_soar_level()
    r = SoarRenderer(periodic=True, surface="city",
                     shader_source=source.replace(ANCHOR, PROBE + ANCHOR))
    r.upload_volume(level.sigma)
    meta = r.surface_meta
    dx = float(min(level.dx))
    state = SceneState(
        bmin=[float(v) for v in level.bmin],
        bmax=[float(v) for v in level.bmax],
        dt_view=dx * 2.0, dt_light=dx * 2.0, periodic=True,
        ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=float(meta["cell_m"]),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1, ocean_enabled=True,
        surface="city")
    return r, state


def _plain_rgb(r):
    import wgpu
    tex = r._targets["plain"]
    row = math.ceil(W * 16 / 256) * 256
    buf = r.device.create_buffer(
        size=row * H,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc = r.device.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf, "offset": 0, "bytes_per_row": row,
         "rows_per_image": H}, (W, H, 1))
    r.device.queue.submit([enc.finish()])
    buf.map_sync(wgpu.MapMode.READ)
    a = np.frombuffer(bytearray(buf.read_mapped()), np.float32)
    buf.unmap()
    return a.reshape(H, row // 4)[:, :W * 4].reshape(H, W, 4)[..., :3]


@pytest.mark.gpu
@pytest.mark.parametrize("exposure", EXPOSURES)
def test_one_element_round_trips(exposure, probe):
    from cloudyview.soar_host import ViewState
    r, state = probe
    view = ViewState(
        camera_position=[80000.0, 55000.0, 300.0], azimuth=0.0,
        elevation=0.0, fov=90.0, output_size=(W, H), render_size=(W, H),
        jitter=False, subpixel=False, exposure=exposure,
        sun_azimuth=310.0, sun_elevation=22.0, view_step_lod_degrees=0.0)
    r.render(state, view, frames=1)
    out = _plain_rgb(r)
    assert np.isfinite(out).all(), "the round trip produced non-finite values"
    err, shown_err, white = out[..., 0], out[..., 1], out[..., 2] > 0.5
    assert float(shown_err.max()) <= SHOWN_TOL, (
        f"exposure {exposure}: one element shows {shown_err.max():.4f} off "
        f"itself (bound {SHOWN_TOL})")
    assert (~white).mean() > 0.5, "the sweep never reaches below white"
    worst = float(err[~white].max())
    assert worst <= ROUND_TRIP_TOL, (
        f"exposure {exposure}: one element comes back {worst:.3f} off its "
        f"radiance below white (bound {ROUND_TRIP_TOL})")
