"""
A roof's bright detail shows the same near and far.

rooftopworks draws a skylight field on the deck (cc_rooftopworks_roof) the
way the flat facade draws its windows: the pixel's footprint is averaged as
it is shown, pane by pane while it spans less than a lattice pitch
(cc_rooftopworks_sky_exact), in expectation past that
(cc_rooftopworks_sky_mean), blended across half a pitch to one. The blend is
only seamless if the two agree, so this checks, by a probe variant of the
plain pass that evaluates it once per pixel:

1. Over a field, the exact sum's average equals the expectation. Each
   pixel takes its own building seed and walks a grid of footprints
   tiling 8 x 8 lattice cells, averaging what the exact sum shows and what
   it emits; beside it, the expectation for one such footprint. Pooled over the frame's seeds the two must agree in
   displayed luminance and in light within MEAN_TOL. This pins the
   lattice's geometry (pane inset and area), the lit rate, the brightness
   distribution and the expectation's three-point quadrature over it,
   in clear air and in fog. And at the field's edges, where footprints
   overlap it only in part, the expectation's lit pane area must equal the
   exact sum's, pooled over places and seeds: the expectation counts the
   panes the overlap actually holds, the lattice in its own phase.
2. A footprint on one pane is that pane. A footprint inside a lit pane
   is flagged as wholly on it at its own brightness (the hook's fast path),
   one off every lit pane holds none of their area, and both sums resolve
   back to the pane's and the roof's radiance, so up close nothing changes
   against the per-pixel draw.
3. The coping trim's lit share is its runs'. The trim is shown as the
   mixture of lit and dark its footprint holds, and the runs' share of a
   footprint along the coping is closed form
   (cc_rooftopworks_trim_run_int); it must equal a dense midpoint count of
   the run pattern the trim is drawn with, for footprints from a few
   centimetres to a few fixture pitches, far out along the coping.
(The penthouse's glass wall had its own check here until 2026-11, when the
crown became core massing, raymarch.wgsl city_crown, and its walls the
core facade's.)

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/test_soar_city_roof_means.py -v
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
FOG_VALUES = {"clear": (1.0, 0.0, 0.0, 0.0),
              "fog": (0.55, 0.004, 0.005, 0.008)}
FOGS = {k: "vec4<f32>({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(*v)
        for k, v in FOG_VALUES.items()}
# The roof's radiance under the field (the probe's `base`, below).
BASE = (0.012, 0.011, 0.013)

# A field of its own per pixel: the pixel's seed, a synthetic lattice at the
# shipped pitch and pane, a dim roof under it.
FIELD = """        let px = vec2<u32>(u32(clamp(0.5 + 0.7 * dir.x, 0.0, 1.0) * 4096.0),
                           u32(clamp(0.5 + 1.4 * dir.z, 0.0, 0.999) * 4096.0));
        var cc = city_cell(vec2<i32>(0, 0));
        cc.seed = pcg2d(px ^ vec2<u32>(0x2545f491u, 0x9e3779b9u));
        var sk: cc_rooftopworks_Sky;
        sk.ok = true;
        sk.flo = vec2<f32>(0.0);
        sk.pitch = cc_rooftopworks_SKY_PITCH;
        sk.pane = cc_rooftopworks_SKY_PANE;
        sk.fhi = 8.0 * sk.pitch;
        let base = vec3<f32>BASE;
        let pane_rad = CITY_PALETTE_MEAN * cc_rooftopworks_SKY_RAD;
        let fog = FOG;
        let lum = vec3<f32>(0.2126, 0.7152, 0.0722);
"""

# 1. A 16 x 16 grid of footprints half a pitch on a side (the blend's first
# span), tiling the field exactly, so their mean is the field's; the exact
# sum's per-area shown luminance and light luminance averaged over them,
# and the expectation for one footprint of the same size inside the field.
AGREE = """    {
""" + FIELD + """        let hb = 0.25 * sk.pitch;
        let box_a = 4.0 * hb.x * hb.y;
        var shown = 0.0;
        var light = 0.0;
        for (var j: i32 = 0; j < 16; j = j + 1) {
            for (var i: i32 = 0; i < 16; i = i + 1) {
                let c = (vec2<f32>(f32(i), f32(j)) + 0.5) / 16.0 * sk.fhi;
                let s = cc_rooftopworks_sky_exact(cc, sk, c - hb, c + hb,
                                                  base, pane_rad, fog, true);
                shown = shown + dot(s.sum.shown, lum) / box_a;
                light = light + dot(s.sum.light, lum) / box_a;
            }
        }
        let m = cc_rooftopworks_sky_mean(sk, sk.flo + 2.0 * hb,
                                         sk.flo + 4.0 * hb, base, pane_rad,
                                         fog, true);
        return vec4<f32>(shown / 256.0, light / 256.0,
                         dot(m.sum.shown, lum) / box_a,
                         dot(m.sum.light, lum) / box_a);
    }
"""

# 2. A small footprint (0.2 x 0.2 m) at the centre of each of the field's
# panes in turn, and one in the gap beside it: the hook's arithmetic on
# each (the exact sum, its whole-pane flag) against the pane's own draw.
ONE = """    {
""" + FIELD + """        let k = i32(px.x % 64u);
        let q = vec2<f32>(f32(k % 8), f32(k / 8));
        let inset = 0.5 * (sk.pitch - sk.pane);
        let mid = sk.flo + q * sk.pitch + inset + 0.5 * sk.pane;
        let hb = vec2<f32>(0.1);
        let lit = cc_rooftopworks_sky_lit(cc, q);
        let on = cc_rooftopworks_sky_exact(cc, sk, mid - hb, mid + hb, base,
                                           pane_rad, fog, true);
        let gap = sk.flo + q * sk.pitch + 0.5 * inset;
        let off = cc_rooftopworks_sky_exact(cc, sk, gap - hb, gap + hb, base,
                                            pane_rad, fog, true);
        var err_on = 0.0;
        if (lit > 0.0) {
            err_on = abs(on.whole - lit);
        } else {
            err_on = on.lit_area + select(0.0, 1.0, on.whole >= 0.0);
        }
        // And the sums themselves resolve back to what they hold: the
        // pane (or the roof, for a dark one) and the roof beside it.
        let a = 4.0 * hb.x * hb.y;
        let want = base + pane_rad * lit;
        let r_on = city_windows_resolve(
            CityWinSum(on.sum.shown / a, on.sum.light / a), fog, true);
        let r_off = city_windows_resolve(
            CityWinSum(off.sum.shown / a, off.sum.light / a), fog, true);
        let rel = max(abs(r_on - want) / want, abs(r_off - base) / base);
        return vec4<f32>(err_on, off.lit_area
                         + select(0.0, 1.0, off.whole >= 0.0),
                         select(0.0, 1.0, lit > 0.0),
                         max(max(rel.x, rel.y), rel.z));
    }
"""

# 1b. Footprints three quarters of a pitch on a side, scattered across the
# field's corner and edges (each pixel its own place and seed): the exact
# sum's lit pane area and the expectation's, which must agree pooled.
EDGE = """    {
""" + FIELD + """        let r = city_rand4(px ^ vec2<u32>(0x3c6ef372u, 0xa54ff53au));
        let hb = 0.375 * sk.pitch;
        let c = sk.flo + (r.xy * 3.0 - 1.0) * sk.pitch;
        let ex = cc_rooftopworks_sky_exact(cc, sk, c - hb, c + hb, base,
                                           pane_rad, fog, true);
        let mn = cc_rooftopworks_sky_mean(sk, c - hb, c + hb, base, pane_rad,
                                          fog, true);
        return vec4<f32>(ex.lit_area, mn.lit_area, 0.0, 1.0);
    }
"""

# 3. Across the frame, footprints 0.05 m to 20 m wide; down it, their start
# out to 10 km along the coping. The closed-form share against a 4096-point
# midpoint count of the drawn pattern.
TRIM = """    {
        let a = clamp(0.5 + 0.7 * dir.x, 0.0, 1.0);
        let b = clamp(0.5 + 1.4 * dir.z, 0.0, 0.999);
        // @domain: the base is 10
        let w = 0.05 * pow(10.0, 2.6 * a);
        let u0 = 1e4 * b + 0.37;
        let share = (cc_rooftopworks_trim_run_int(u0 + w)
                     - cc_rooftopworks_trim_run_int(u0)) / w;
        var n = 0.0;
        for (var i: i32 = 0; i < 4096; i = i + 1) {
            let u = u0 + w * (f32(i) + 0.5) / 4096.0;
            n = n + select(0.0, 1.0, fract(u / cc_rooftopworks_TRIM_PITCH)
                                     < cc_rooftopworks_TRIM_DUTY);
        }
        let err = abs(share - n / 4096.0);
        // The midpoint count misplaces each run edge in the footprint by
        // at most half a sample (a footprint holds at most 2 (w / pitch + 2)
        // of them), and float32 positions 10 km out are good to about a
        // millimetre.
        let edges = 2.0 * (w / cc_rooftopworks_TRIM_PITCH + 2.0);
        let tol = 0.5 * edges / 4096.0 + 2e-3 / w;
        return vec4<f32>(err / tol, share, 0.0, 1.0);
    }
"""
FIELD_BASE = "({:.3f}, {:.3f}, {:.3f})".format(*BASE)
PROBES = {
    "agree_clear": AGREE.replace("FOG", FOGS["clear"]).replace("BASE", FIELD_BASE),
    "agree_fog": AGREE.replace("FOG", FOGS["fog"]).replace("BASE", FIELD_BASE),
    "one_clear": ONE.replace("FOG", FOGS["clear"]).replace("BASE", FIELD_BASE),
    "trim": TRIM,
    "edge": EDGE.replace("FOG", FOGS["clear"]).replace("BASE", FIELD_BASE),
}
# Pooled over the frame's 256 x 144 seeds, 64 cells each: the lit share's
# sampling error is under 0.2% of itself; the footprints tile the field, so
# the walk's mean is the field's exactly.
MEAN_TOL = 0.01
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

    meta = get("one_clear").surface_meta
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


def _run(probes, name, exposure=4.0):
    from cloudyview.soar_host import ViewState
    get, state = probes
    view = ViewState(
        camera_position=[80000.0, 55000.0, 300.0], azimuth=0.0,
        elevation=0.0, fov=90.0, output_size=(W, H), render_size=(W, H),
        jitter=False, subpixel=False, exposure=exposure,
        sun_azimuth=310.0, sun_elevation=22.0, view_step_lod_degrees=0.0)
    r = get(name)
    r.render(state, view, frames=1)
    out = _plain(r)
    assert np.isfinite(out).all(), f"{name} produced non-finite values"
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("fog", list(FOGS))
@pytest.mark.parametrize("exposure", (2.0, 6.0))
def test_exact_sum_averages_to_the_expectation(fog, exposure, probes):
    out = _run(probes, f"agree_{fog}", exposure).astype(np.float64)
    exact_shown, exact_light = out[..., 0].mean(), out[..., 1].mean()
    mean_shown, mean_light = out[..., 2].mean(), out[..., 3].mean()
    # The field must be lit enough for the comparison to mean anything: its
    # light well above the roof's alone, through the same fog.
    lum = np.array([0.2126, 0.7152, 0.0722])
    f = FOG_VALUES[fog]
    roof = lum @ (np.array(BASE) * f[0] + np.array(f[1:]))
    assert mean_light > 1.5 * roof, (mean_light, roof)
    assert abs(exact_light - mean_light) <= MEAN_TOL * mean_light, (
        f"light: exact {exact_light:.5f} vs expectation {mean_light:.5f}")
    assert abs(exact_shown - mean_shown) <= MEAN_TOL * mean_shown, (
        f"shown: exact {exact_shown:.5f} vs expectation {mean_shown:.5f}")


@pytest.mark.gpu
def test_expectation_holds_at_the_field_edges(probes):
    out = _run(probes, "edge").astype(np.float64)
    exact, mean = out[..., 0].mean(), out[..., 1].mean()
    assert mean > 0.0
    # Pooled over the frame's places and seeds (a few percent of a pane's
    # lit-or-not draw each), the sampling error is under 2%.
    assert abs(exact - mean) <= 0.03 * mean, (exact, mean)


@pytest.mark.gpu
def test_a_footprint_on_one_pane_is_that_pane(probes):
    out = _run(probes, "one_clear")
    lit_share = out[..., 2].mean()
    # The draws do light about the rate's share of panes.
    assert 0.25 < lit_share < 0.35, lit_share
    assert out[..., 0].max() <= 1e-6, (
        "a footprint inside a pane did not come back as that pane")
    assert out[..., 1].max() == 0.0, (
        "a footprint in the gap between panes saw a lit pane")
    # The windows' round-trip tolerance (tests/test_soar_city_windows.py).
    assert out[..., 3].max() <= 0.03, (
        "a one-element sum did not resolve back to its element")


@pytest.mark.gpu
def test_trim_share_is_the_runs(probes):
    out = _run(probes, "trim")
    # The error over its tolerance (see the probe).
    assert out[..., 0].max() <= 1.0, out[..., 0].max()
    # And the shares span dark joints to whole runs.
    assert out[..., 1].min() == 0.0 and out[..., 1].max() >= 0.999
