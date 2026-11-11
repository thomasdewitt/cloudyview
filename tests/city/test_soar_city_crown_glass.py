"""
The crown's glass room shows the same near and far.

The crown (city_crown, core massing) is a glass room: a PENT zone over all
its faces (city_zones on CITY_CROWN_TIER), its own 2.1 m bays, its panes lit
at CITY_PENT_LIT, each at its own level (city_pent_window). The flat facade
averages it as shown: bay by bay while the footprint is small
(city_windows_exact, which the shell's per-bay draw matches), in expectation
past that (city_pent_rows), blended across the exact sum's room. The blend is
only seamless if the two agree, so a probe variant of the plain pass builds a
crown face per pixel (its own building seed, mixed and monochrome houses, in
clear air and in fog), tiles its bays with footprints half a bay by half a
storey (exact only), and sets the tiles' mean beside the expectation over
the whole strip. Pooled over the frame's seeds, shown luminance and light
must agree within MEAN_TOL.

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/city/test_soar_city_crown_glass.py -v
"""

import math

import numpy as np
import pytest
from tests.conftest import ShaderModule

pytest.importorskip("wgpu")

from tests.conftest import SOAR_DATA_FILE, build_soar_level, soar_gpu_adapter

SHADER = ShaderModule("city")
ANCHOR = "    let city_o = city_camera_origin();\n    let chit = city_trace("
FOG_VALUES = {"clear": (1.0, 0.0, 0.0, 0.0),
              "fog": (0.55, 0.004, 0.005, 0.008)}

PROBE = """    {
        let px = vec2<u32>(u32(clamp(0.5 + 0.7 * dir.x, 0.0, 1.0) * 4096.0),
                           u32(clamp(0.5 + 1.4 * dir.z, 0.0, 0.999) * 4096.0));
        var cc = city_cell(vec2<i32>(0, 0));
        cc.seed = pcg2d(px ^ vec2<u32>(0x1f83d9abu, 0x5be0cd19u));
        cc.land_use = CITY_USE_URBAN;
        let hs = city_rand4(cc.seed);
        // Half the houses monochrome, across the palette.
        cc.win_mono = select(-1.0, hs.x, hs.y < 0.5);
        cc.palette_bias = hs.z;
        // A crown three storeys tall on a 36 m roof, 16 m wide.
        let f = city_face_make(cc, vec3<f32>(0.0, 0.0, 36.0),
                               vec3<f32>(16.0, 12.0, 46.8), 0,
                               CITY_CROWN_TIER);
        let zones = city_zones(cc, f);
        let zp = city_zones_pack(zones);
        var g: CityWinGeom;
        g.plo = city_pane_lo(cc, f);
        g.phi = city_pane_hi(cc, f);
        g.olo = g.plo - vec2<f32>(CITY_SHELL_FRAME);
        g.ohi = g.phi + vec2<f32>(CITY_SHELL_FRAME);
        g.off = vec2<f32>(0.0);
        g.i_shift = 0;
        let fog = FOG;
        let l_wall = vec3<f32>(0.010, 0.009, 0.012);
        let refl = vec3<f32>(0.004, 0.005, 0.007);
        let through = 0.92;
        let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
        // Tiles: half a bay by half a storey, over the bays and storeys.
        let hb = vec2<f32>(0.25 * f.pitch, 0.25 * CITY_FLOOR_H);
        let nu = 2 * f.nbays;
        var shown = 0.0;
        var light = 0.0;
        var n = 0.0;
        for (var j: i32 = 0; j < 6; j = j + 1) {
            for (var i: i32 = 0; i < nu; i = i + 1) {
                let c = vec2<f32>(f.margin + (f32(i) + 0.5) * 2.0 * hb.x,
                                  f.z0 + (f32(j) + 0.5) * 2.0 * hb.y);
                let s = city_windows_exact(cc, f, g, zp, c, hb, l_wall,
                                           l_wall, refl, through, fog,
                                           false, true);
                let a = 4.0 * hb.x * hb.y;
                shown = shown + dot(s.shown, lum) / a;
                light = light + dot(s.light, lum) / a;
                n = n + 1.0;
            }
        }
        let wc = vec2<f32>(f.margin + 0.5 * f32(f.nbays) * f.pitch,
                           0.5 * (f.z0 + f.z1));
        let wh = vec2<f32>(0.5 * f32(f.nbays) * f.pitch, 0.5 * (f.z1 - f.z0));
        let m = city_pent_rows(cc, f, city_zone_geom(g, f, CITY_ZONE_PENT),
                               wc, wh, l_wall, refl, through, fog, false,
                               true);
        let wa = 4.0 * wh.x * wh.y;
        return vec4<f32>(shown / n, light / n, dot(m.shown, lum) / wa,
                         select(dot(m.light, lum) / wa, -1.0,
                                !zones.pent || f.nbays < 4));
    }
"""
PROBES = {k: PROBE.replace(
    "FOG", "vec4<f32>({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(*v))
    for k, v in FOG_VALUES.items()}
# Pooled over the frame's seeds, a few dozen bays each: the lit share's
# sampling error is a fraction of this.
MEAN_TOL = 0.01
W, H = 256, 144


@pytest.fixture(scope="module")
def probes():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar.renderer import SoarRenderer
    from cloudyview.soar.uniforms import SceneState
    from cloudyview.soar.uniforms import DEFAULT_OCEAN_REFLECTANCE

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

    meta = get("clear").surface_meta
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


@pytest.mark.gpu
@pytest.mark.parametrize("fog", list(FOG_VALUES))
@pytest.mark.parametrize("exposure", (2.0, 6.0))
def test_glass_room_exact_sum_averages_to_the_expectation(fog, exposure,
                                                          probes):
    from cloudyview.soar.uniforms import ViewState
    get, state = probes
    view = ViewState(
        camera_position=[80000.0, 55000.0, 300.0], azimuth=0.0,
        elevation=0.0, fov=90.0, output_size=(W, H), render_size=(W, H),
        jitter=False, subpixel=False, exposure=exposure,
        sun_azimuth=310.0, sun_elevation=22.0, view_step_lod_degrees=0.0)
    r = get(fog)
    r.render(state, view, frames=1)
    out = _plain(r).astype(np.float64)
    assert np.isfinite(out).all()
    assert (out[..., 3] >= 0.0).all(), (
        "a probe face was not a glass room of four bays or more")
    exact_shown, exact_light = out[..., 0].mean(), out[..., 1].mean()
    mean_shown, mean_light = out[..., 2].mean(), out[..., 3].mean()
    assert abs(exact_light - mean_light) <= MEAN_TOL * mean_light, (
        exact_light, mean_light)
    assert abs(exact_shown - mean_shown) <= MEAN_TOL * mean_shown, (
        exact_shown, mean_shown)
