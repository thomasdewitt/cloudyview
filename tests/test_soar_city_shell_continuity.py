"""
The city's facade shell hands off to the flat facade without a step.

The shell (fs_surface_refine) resolves a wall's relief while the pixel
footprint is small and blends into the flat facade (city_facade_flat) as it
grows. If the two disagree in the mean where they meet, a wall brightens or
darkens as the camera backs away from it. This holds the camera still and
sweeps the render width instead: at the view-step LOD 0 the pixel angle is
inversely proportional to the width, so the walls in frame walk through the
hand-off with the geometry, the occlusion and every hash draw fixed.

The flat facade's own window ladder also changes with the footprint (panes
into blocks into its asymptote), so the width dependence of the full render
is not the shell's alone. The same sweep is therefore rendered twice, once
as shipped and once with nothing marked for the shell (the flat facade
everywhere), and what is bounded is how much the difference between the two
moves from one width to the next, per image region. A shell that averaged to
its flat twin would leave that difference constant: the near field keeps its
own look (it is meant to differ), and what must not move is the look where
the shell hands over.

Measured when this was written (2026-09-24, 32 frames), worst region step of
(shell - flat) between adjacent widths: 0.0075, 0.0072 and 0.0067 for the
three views; with the resolved shell forced to half brightness (a mismatch
of the kind this guards) 0.0150, 0.0117 and 0.0153. The views are the ones
where that mismatch shows: an oblique walk-up face, a canyon from 150 m up,
and a flight along a curtain wall. The test refuses to pass vacuously: at the
widest width the plain pass must have marked a real share of the frame for
the shell.

The sweep runs at the view-step LOD 0 on purpose. At the app's LOD the pixel
angle is floored by the view-step angle, which at these widths is larger
than the true pixel angle: the footprint would not change across the sweep
and the hand-off would never be exercised.

Needs a real GPU; skips cleanly without one.

Run with: uv run --extra dev python -m pytest tests/test_soar_city_shell_continuity.py -v
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
# The plain pass's mark for the shell; replaced to render the flat facade
# everywhere.
MARK = """    let near = city_shell_on_wall(chit, t_lim)
               || city_shell_envelope(city_o, dir, t_lim);"""

# Tile-frame cameras on the TWP-ICE harness city: (x, y, z, azimuth,
# elevation, fov). World = tile + TILE_OFFSET for this field.
TILE_OFFSET = (75330.0, -17010.0)
VIEWS = {
    "walkup_oblique": (21380.0, 70540.0, 14.0, 35.0, 3.0, 60.0),
    "canyon_150m": (17300.0, 72700.0, 150.0, 10.0, -12.0, 50.0),
    "curtain_flyby": (19900.0, 69836.5, 20.0, 80.0, 2.0, 70.0),
}
# The widths every tier renders at lie within these (minimal renders 320).
WIDTHS = (1280, 960, 720, 480, 360)
FRAMES = 32
GRID = (8, 4)
MAX_STEP = 0.010
MIN_MARKED = 0.05


@pytest.fixture(scope="module")
def city():
    if soar_gpu_adapter() is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    if not SOAR_DATA_FILE.exists():
        pytest.skip(f"Data file not found: {SOAR_DATA_FILE}")
    from cloudyview.soar_host import SceneState, SoarRenderer
    from cloudyview.witness import OCEAN_REFLECTANCE

    source = SHADER.read_text()
    assert source.count(MARK) == 1, (
        "the plain pass's shell mark moved; update MARK")
    level = build_soar_level()
    renderers = {}
    for name, src in (("shell", source),
                      ("flat", source.replace(MARK, "    let near = false;"))):
        r = SoarRenderer(periodic=True, surface="city", shader_source=src)
        r.upload_volume(level.sigma)
        renderers[name] = r
    meta = renderers["shell"].surface_meta
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
    return renderers, state


def _view(cam, width):
    from cloudyview.soar_host import APP_LIGHT_MARCH_LOD_DEGREES, ViewState
    x, y, z, az, el, fov = cam
    height = int(round(width * 9 / 16))
    return ViewState(
        camera_position=[x + TILE_OFFSET[0], y + TILE_OFFSET[1], z],
        azimuth=az, elevation=el, fov=fov,
        output_size=(width, height), render_size=(width, height),
        jitter=True, subpixel=True,
        sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
        light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES * 0.5,
        view_step_lod_degrees=0.0)


def _marked_fraction(r, width, height):
    """Share of pixels the plain pass marked for the shell (red's sign bit,
    CITY_SHELL_MARK in raymarch.wgsl)."""
    import wgpu
    tex = r._targets["plain"]
    row = math.ceil(width * 16 / 256) * 256
    buf = r.device.create_buffer(
        size=row * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
    enc = r.device.create_command_encoder()
    enc.copy_texture_to_buffer(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)},
        {"buffer": buf, "offset": 0, "bytes_per_row": row,
         "rows_per_image": height}, (width, height, 1))
    r.device.queue.submit([enc.finish()])
    buf.map_sync(wgpu.MapMode.READ)
    a = np.frombuffer(bytearray(buf.read_mapped()), np.uint32)
    buf.unmap()
    red = a.reshape(height, row // 4)[:, 0:width * 4:4]
    return float(((red & 0x80000000) != 0).mean())


def _regions(r, state, cam, width):
    import cloudyview as cv
    img = np.asarray(r.render(state, _view(cam, width), frames=FRAMES))
    shown = cv.quantize_uint8(img).astype(np.float64) / 255.0
    gx, gy = GRID
    h, w = shown.shape[:2]
    return np.array([[shown[j * h // gy:(j + 1) * h // gy,
                            i * w // gx:(i + 1) * w // gx].mean()
                      for i in range(gx)] for j in range(gy)])


@pytest.mark.gpu
@pytest.mark.parametrize("view_name", list(VIEWS))
def test_shell_hand_off_is_continuous(view_name, city):
    renderers, state = city
    cam = VIEWS[view_name]
    diff = []
    for width in WIDTHS:
        shell = _regions(renderers["shell"], state, cam, width)
        if width == WIDTHS[0]:
            marked = _marked_fraction(renderers["shell"],
                                      *_view(cam, width).render_size)
            assert marked >= MIN_MARKED, (
                f"{view_name}: only {marked:.3f} of the frame is marked for "
                "the shell; the view no longer exercises the hand-off")
        diff.append(shell - _regions(renderers["flat"], state, cam, width))
    step = float(np.abs(np.diff(np.array(diff), axis=0)).max())
    assert step <= MAX_STEP, (
        f"{view_name}: the shell's difference from the flat facade moved "
        f"{step:.4f} in a region between adjacent widths (bound {MAX_STEP})")
