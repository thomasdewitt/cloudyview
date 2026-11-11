"""
The flat facade's window averages are exact where they can be checked.

The city's flat facade averages the windows under a pixel's footprint as
they are SHOWN (city_windows_seen) and hands back the radiance that shows as
that average, in the colour of their light (city_windows_resolve). Three
properties of that rule are checked here, each by a probe variant of the
plain pass that evaluates it once per pixel:

1. One element round-trips. Whatever else the rule does for a footprint
   holding many windows, a footprint holding one element must come back as
   that element; otherwise a pane resolved on the flat facade would change
   its brightness or colour against the same pane one pixel nearer, where
   the shell draws it, and panes would pop across the hand-off. Radiance is
   log-spaced across the frame (0.01 to 30, through the display curve's toe,
   knee and highlight shoulder), and down it run the room palette's six
   bands, grey and the three pure primaries, clear, in fog and in haze, at several
   exposures and at the host's gamma and white-point limits. Every element
   must SHOW as itself within SHOWN_TOL and, wherever the display has not
   clipped any of its channels, come back as its own radiance within
   ROUND_TRIP_TOL. (Past the clip, a channel shows 1 whatever its radiance,
   so none can be told back from it.) The primaries run in clear air only:
   in fog a near-primary's displayed luminance can peak and fall as it
   brightens (the shoulder's chroma-preserving blend dims the fog's own
   channels), outside what the rule claims (city_windows_resolve); the room
   palette, every channel lit, runs in fog and in thick haze.
2. A mixture shows as its average, in its light's colour. Two elements
   half and half: the resolved radiance, shown, has the luminance of the
   two shown values' average within SHOWN_TOL, and the chromaticity of
   their mean light.
3. The palette's shares are the draws'. The share each room colour band
   takes of a building's lit windows (city_room_band_share) must sum to 1
   and give the mean colour a brute-force walk of the colour draws through
   city_room_color gives, for mixed and monochrome houses across the
   palette bias. (A share that let a band lying wholly above 1 claim the
   draws clamped onto 1 summed to 1.72 and turned rare towers blue.)

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/test_soar_city_windows.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest
from .conftest import ShaderModule

pytest.importorskip("wgpu")

try:
    from .conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter
except ImportError:
    from conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

SHADER = ShaderModule("city")
ANCHOR = "    let city_o = city_camera_origin();\n    let chit = city_trace("
# The frame's two coordinates, each spanning [0, 1] at fov 90 over 16:9.
COORDS = """        let a = clamp(0.5 + 0.7 * dir.x, 0.0, 1.0);
        let b = clamp(0.5 + 1.4 * dir.z, 0.0, 0.999);
"""
# Down the frame: the room palette's six bands, grey, and the primaries.
COLOUR = """        let k = i32(floor(b * 10.0));
        var c = vec3<f32>(0.5);
        if (k < 6) {
            c = city_room_band_colour(k);
        } else if (k == 7) {
            c = vec3<f32>(1.0, 0.0, 0.0);
        } else if (k == 8) {
            c = vec3<f32>(0.0, 1.0, 0.0);
        } else if (k == 9) {
            c = vec3<f32>(0.0, 0.0, 1.0);
        }
"""
FOGS = {"clear": "vec4<f32>(1.0, 0.0, 0.0, 0.0)",
        "fog": "vec4<f32>(0.55, 0.004, 0.005, 0.008)",
        "haze": "vec4<f32>(0.15, 0.015, 0.02, 0.03)"}
# One element per pixel, its radiance across the frame: the round trip's
# relative radiance error, its display error, and whether any channel is
# clipped plus twice the colour row.
ROUND_TRIP = """    {
""" + COORDS + COLOUR + """        // @domain: the base is 10
        let l = c * pow(10.0, -2.0 + 3.5 * a);
        let fog = FOG;
        let mean = city_win_add(city_win_zero(), l, 1.0, fog, true);
        let back = city_windows_resolve(mean, fog, true);
        let err = abs(back - l) / max(l, vec3<f32>(1e-6));
        let e = u.cam_right.w;
        let d0 = city_display_curve((l * fog.x + fog.yzw) * e);
        let d1 = city_display_curve((back * fog.x + fog.yzw) * e);
        let clip = select(0.0, 1.0, max(max(d0.x, d0.y), d0.z) >= 0.999);
        return vec4<f32>(max(max(err.x, err.y), err.z),
                         max(max(abs(d1.x - d0.x), abs(d1.y - d0.y)),
                             abs(d1.z - d0.z)), clip + 2.0 * f32(k), 1.0);
    }
"""
# Two elements half and half: a dimmer one of the frame's colour and a
# brighter one of the next band's. The display luminance error and the
# chromaticity error of the resolved radiance.
MIXTURE = """    {
""" + COORDS + COLOUR + """        let c2 = city_room_band_colour((k + 1) % 6);
        // @domain: the base is 10
        let r = pow(10.0, -2.0 + 3.5 * a);
        let l1 = c * r;
        let l2 = c2 * (3.0 * r);
        let fog = FOG;
        var mean = city_win_add(city_win_zero(), l1, 0.5, fog, true);
        mean = city_win_add(mean, l2, 0.5, fog, true);
        let back = city_windows_resolve(mean, fog, true);
        let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
        let e = u.cam_right.w;
        let y = dot(city_display_curve((back * fog.x + fog.yzw) * e), lum);
        let ml = 0.5 * (l1 + l2);
        let ch = abs(back / max(back.x + back.y + back.z, 1e-9)
                     - ml / max(ml.x + ml.y + ml.z, 1e-9));
        return vec4<f32>(abs(y - dot(mean.shown, lum)),
                         max(max(ch.x, ch.y), ch.z), 0.0, 1.0);
    }
"""
# The palette bias across the frame; down it, a mixed house above and
# monochrome houses at every coordinate below. |sum of shares - 1| and the
# mean colour's error against a midpoint walk of the draws.
SHARES = """    {
""" + COORDS + """        let bias = a;
        var lo = 0.0;
        var hi = 1.0;
        if (b > 0.2) {
            let mono = (b - 0.2) / 0.799;
            lo = mono - 0.04;
            hi = mono + 0.04;
        }
        var total = 0.0;
        var mean = vec3<f32>(0.0);
        for (var k: i32 = 0; k < 6; k = k + 1) {
            let share = city_room_band_share(k, lo, hi, bias);
            total = total + share;
            mean = mean + city_room_band_colour(k) * share;
        }
        var walk = vec3<f32>(0.0);
        for (var n: i32 = 0; n < 1024; n = n + 1) {
            let d = clamp(lo + (hi - lo) * (f32(n) + 0.5) / 1024.0, 0.0, 1.0);
            walk = walk + city_room_color(d, bias);
        }
        let err = abs(mean - walk / 1024.0);
        return vec4<f32>(abs(total - 1.0), max(max(err.x, err.y), err.z),
                         0.0, 1.0);
    }
"""
PROBES = {
    "round_trip_clear": ROUND_TRIP.replace("FOG", FOGS["clear"]),
    "round_trip_fog": ROUND_TRIP.replace("FOG", FOGS["fog"]),
    "round_trip_haze": ROUND_TRIP.replace("FOG", FOGS["haze"]),
    "mixture_clear": MIXTURE.replace("FOG", FOGS["clear"]),
    "mixture_fog": MIXTURE.replace("FOG", FOGS["fog"]),
    "mixture_haze": MIXTURE.replace("FOG", FOGS["haze"]),
    "shares": SHARES,
}
ROUND_TRIP_TOL = 0.03
SHOWN_TOL = 0.01
CHROMA_TOL = 1e-3
# A midpoint walk of 1024 draws misplaces each band edge by at most half a
# draw's width; five edges, colours at most 1 apart.
SHARE_SUM_TOL = 1e-5
SHARE_COLOUR_TOL = 5 * 0.5 / 1024 + 1e-4
# (exposure, gamma, white point): the night exposures at the host's
# defaults, and the gamma and white-point limits (soar_host) at the middle
# one.
LOOKS = ((2.0, None, None), (4.0, None, None), (6.0, None, None),
         (4.0, 1.0, 40.0), (4.0, 4.0, 4.0), (4.0, 4.0, 40.0))
W, H = 256, 144


@pytest.fixture(scope="module")
def probes():
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
    built = {}

    def get(name):
        if name not in built:
            r = SoarRenderer(periodic=True, surface="city",
                             shader_source=source.replace(
                                 ANCHOR, PROBES[name] + ANCHOR))
            r.upload_volume(level.sigma)
            built[name] = r
        return built[name]

    meta = get("shares").surface_meta
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
    return get, state


def _plain(r):
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
    return a.reshape(H, row // 4)[:, :W * 4].reshape(H, W, 4)


def _run(probes, name, exposure=4.0, gamma=None, white=None):
    from cloudyview.soar_host import ViewState
    get, state = probes
    look = {}
    if gamma is not None:
        look["tone_map_gamma"] = gamma
    if white is not None:
        look["tone_map_white_point"] = white
    view = ViewState(
        camera_position=[80000.0, 55000.0, 300.0], azimuth=0.0,
        elevation=0.0, fov=90.0, output_size=(W, H), render_size=(W, H),
        jitter=False, subpixel=False, exposure=exposure,
        sun_azimuth=310.0, sun_elevation=22.0, view_step_lod_degrees=0.0,
        **look)
    r = get(name)
    r.render(state, view, frames=1)
    out = _plain(r)
    assert np.isfinite(out).all(), f"{name} produced non-finite values"
    return out


def _look_id(look):
    e, g, w = look
    return f"exp{e:g}" + ("" if g is None else f"-gamma{g:g}-white{w:g}")


@pytest.mark.gpu
@pytest.mark.parametrize("fog", list(FOGS))
@pytest.mark.parametrize("look", LOOKS, ids=_look_id)
def test_one_element_round_trips(look, fog, probes):
    out = _run(probes, f"round_trip_{fog}", *look)
    keep = np.ones(out.shape[:2], bool)
    if fog != "clear":
        keep = (out[..., 2] // 2.0) < 7
    err, shown_err = out[..., 0][keep], out[..., 1][keep]
    clip = (out[..., 2][keep] % 2.0) > 0.5
    where = f"{_look_id(look)}, {fog}"
    assert float(shown_err.max()) <= SHOWN_TOL, (
        f"{where}: one element shows {shown_err.max():.4f} off itself "
        f"(bound {SHOWN_TOL})")
    assert (~clip).mean() > 0.4, "the sweep never reaches below the clip"
    worst = float(err[~clip].max())
    assert worst <= ROUND_TRIP_TOL, (
        f"{where}: one element comes back {worst:.3f} off its radiance "
        f"where no channel is clipped (bound {ROUND_TRIP_TOL})")


@pytest.mark.gpu
@pytest.mark.parametrize("fog", list(FOGS))
@pytest.mark.parametrize("look", LOOKS, ids=_look_id)
def test_mixture_shows_as_its_average(look, fog, probes):
    out = _run(probes, f"mixture_{fog}", *look)
    where = f"{_look_id(look)}, {fog}"
    y_err, chroma_err = float(out[..., 0].max()), float(out[..., 1].max())
    assert y_err <= SHOWN_TOL, (
        f"{where}: a two-element mixture shows {y_err:.4f} off its average "
        f"luminance (bound {SHOWN_TOL})")
    assert chroma_err <= CHROMA_TOL, (
        f"{where}: a two-element mixture comes back {chroma_err:.5f} off "
        f"its light's chromaticity (bound {CHROMA_TOL})")


@pytest.mark.gpu
def test_palette_shares_are_the_draws(probes):
    out = _run(probes, "shares")
    sum_err, colour_err = float(out[..., 0].max()), float(out[..., 1].max())
    assert sum_err <= SHARE_SUM_TOL, (
        f"the room bands' shares sum {sum_err:.4f} away from 1")
    assert colour_err <= SHARE_COLOUR_TOL, (
        f"the shares' mean colour is {colour_err:.4f} off the draws' "
        f"(bound {SHARE_COLOUR_TOL:.4f})")
