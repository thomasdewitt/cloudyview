"""
The city's facade shell hands off to the flat facade without a step.

The shell (fs_surface_refine) resolves a wall's relief while the pixel
footprint is small and blends into the flat facade (city_facade_flat) as it
grows. If the two disagree in the mean where they meet, a wall brightens or
darkens as the camera backs away from it. This holds the camera still and
sweeps the render width instead: at the view-step LOD 0 the pixel angle is
inversely proportional to the width, so the walls in frame walk through the
hand-off with the geometry, the occlusion and every hash draw fixed.

The flat facade's own look also changes with the footprint (each pane
filtered at the pixel, then storey by storey in expectation, then the
building's mean), so the width dependence of the full render
is not the shell's alone. The same sweep is therefore rendered twice, once
as shipped and once with nothing marked for the shell (the flat facade
everywhere), and what is bounded is how much the difference between the two
moves from one width to the next, per image region. A shell that averaged to
its flat twin would leave that difference constant: the near field keeps its
own look (it is meant to differ), and what must not move is the look where
the shell hands over.

Measured (2026-09-25, 32 frames, on the flat facade that averages its
windows as shown, city_windows_seen), worst region step of (shell - flat)
between adjacent widths: 0.0053, 0.0071 and 0.0075 for the walk-up, the
canyon and the curtain wall; with the resolved shell forced to half
brightness (a mismatch of the kind this guards) 0.0103, 0.0131 and 0.0158.
The walk-up and the curtain wall
are found by building type in the city itself (TYPED_VIEWS), so a change of
massing moves them with it; the canyon is a fixed camera. The test refuses
to pass vacuously: at the narrowest width, where the shell's reach is
shortest, the plain pass must still have marked a real share of the frame
for it.

A second test checks the hand-off itself, pixel for pixel, where the blend
has reached 1: a mismatch there is a line, not a band, and the sweep's
region means do not see it.

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
VIEW_NAMES = ("walkup_oblique", "canyon_150m", "curtain_flyby")
# A canyon from 150 m up: it looks down a street between blocks, which any
# massing has, so it stays a fixed camera.
CANYON_150M = (17300.0, 72700.0, 150.0, 10.0, -12.0, 50.0)
# The other two are pinned to a BUILDING, found by type in the city itself
# (_find_views), so a change of massing moves the camera with it. Each looks
# at the building's south face (its tier-0 box's min-y side) from a range
# inside the hand-off's (CITY_SHELL_FP_FLAT over the pixel angle, at a
# 60 degree field of view: 375 m at the narrowest width, 1330 m at the
# widest), off its normal, so the sweep carries the face through the
# hand-off at a slant. The face need only fill one of the sweep's regions.
TYPED_VIEWS = {
    # name: (window style, land use, (min height, max height), min face
    #        width, range (m), angle off the face normal (deg), camera z as
    #        a share of the building's height (+ m), fov)
    "curtain_flyby": (3, 0, (120.0, 400.0), 40.0, 700.0, 50.0, (0.35, 0.0),
                      60.0),
    "walkup_oblique": (4, 0, (18.0, 66.0), 30.0, 500.0, 30.0, (1.5, 0.0),
                       60.0),
}
# Where to look for them: a top-down probe over the dense city.
SEARCH_CENTRE = (19500.0, 71000.0)
SEARCH_ALT = 4000.0
SEARCH_FOV = 60.0
SEARCH_PX = 768
# The candidate whose own walls fill the most of its view is taken, and at
# least this share (the camera must see what it was put there for).
MIN_TARGET_SHARE = 0.005
MAX_CANDIDATES = 60
# The widths every tier renders at lie within these (minimal renders 320).
WIDTHS = (1280, 960, 720, 480, 360)
FRAMES = 32
GRID = (8, 4)
# Per view, between what the branch measures and what its controls do (see
# the module docstring), so every control fails every view.
MAX_STEP = {"walkup_oblique": 0.0075, "canyon_150m": 0.0100,
            "curtain_flyby": 0.0110}
MIN_MARKED = 0.05
# The direct check at the hand-off: pixels whose footprint lies in this band
# are blended at least 0.98 into the flat facade, so there the shell's render
# must equal the flat one, pixel for pixel. Pooled over the sweep's widths.
# Measured (2026-09-25): mean |shell - flat| 0.00079, 0.00037 and 0.00068;
# with the flat twin read at the hit's straight-in projection onto the wall
# (the recess offset applied twice, as 2d8a5a0 did) 0.00358, 0.00083 and
# 0.00191.
HANDOFF_FP = (1.15, 1.20)
HANDOFF_MIN_PIXELS = 1000
MAX_HANDOFF_ABS = {"walkup_oblique": 0.0015, "canyon_150m": 0.00060,
                   "curtain_flyby": 0.0012}


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
    views = _find_views(level, state)
    return renderers, state, views


# The plain pass as a probe: it writes each hit's building (its tier-0 box,
# window style, land use and whether it is built) in place of the colour.
# Two targets' worth, so two variants.
PROBE_ANCHOR = "    let t_lim = min(select(1e30, chit.t, chit.hit),"
PROBES = (
    """    if (chit.hit) {
        let pc = city_cell(chit.cell);
        return vec4<f32>(pc.b1min.x, pc.b1min.y, pc.b1max.x, chit.t);
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
""",
    """    if (chit.hit) {
        let pc = city_cell(chit.cell);
        return vec4<f32>(pc.b1max.y, pc.b1max.z,
                         f32(pc.win_style) + 10.0 * f32(pc.land_use)
                         + select(0.0, 100.0, pc.built), chit.t);
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
""")


def _probe_renderers(level):
    from cloudyview.soar_host import SoarRenderer
    source = SHADER.read_text()
    assert source.count(PROBE_ANCHOR) == 1, (
        "the plain pass's probe anchor moved; update PROBE_ANCHOR")
    out = []
    for probe in PROBES:
        r = SoarRenderer(periodic=True, surface="city",
                         shader_source=source.replace(
                             PROBE_ANCHOR, probe + PROBE_ANCHOR))
        r.upload_volume(level.sigma)
        out.append(r)
    return out


def _probe(renderers, state, cam, w, h):
    """Per pixel: (b1min.x, b1min.y, b1max.x, b1max.y, height, code, t)."""
    from cloudyview.soar_host import ViewState
    x, y, z, az, el, fov = cam
    planes = []
    for r in renderers:
        view = ViewState(
            camera_position=[x + TILE_OFFSET[0], y + TILE_OFFSET[1], z],
            azimuth=az, elevation=el, fov=fov, output_size=(w, h),
            render_size=(w, h), jitter=False, subpixel=False,
            sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
            view_step_lod_degrees=0.0)
        r.render(state, view, frames=1)
        planes.append(_plain(r, w, h).view(np.float32))
    a, b = planes
    return np.stack([a[..., 0], a[..., 1], a[..., 2], b[..., 0], b[..., 1],
                     b[..., 2], a[..., 3]], axis=-1)


def _typed_camera(box, height, rng, off, zs, fov):
    """A camera `rng` metres from the middle of the south face of the
    building whose tier-0 box is (min x, min y, max x, max y), `off` degrees
    west of the face's normal (east if negative), aimed at its middle."""
    x0, y0, x1, _ = (float(v) for v in box)
    cx, cz = 0.5 * (x0 + x1), 0.5 * float(height)
    a = math.radians(off)
    x, y = cx - rng * math.sin(a), y0 - rng * math.cos(a)
    z = zs[0] * float(height) + zs[1]
    el = math.degrees(math.atan2(cz - z, rng))
    return (x, y, z, off, el, fov)


def _inside_any(cam, boxes):
    """Whether the camera stands in any building seen from above, or within
    3 m of one (its relief): a trace starting inside a box does not see it,
    so the probe would call the view clear."""
    x, y, z = cam[:3]
    for (x0, y0, x1, y1), (hgt, _) in boxes.items():
        if (x0 - 3.0 <= x <= x1 + 3.0 and y0 - 3.0 <= y <= y1 + 3.0
                and z <= hgt + 3.0):
            return True
    return False


def _find_views(level, state):
    """Pin the typed views to buildings of their type (TYPED_VIEWS)."""
    probes = _probe_renderers(level)
    top = _probe(probes, state, (SEARCH_CENTRE[0], SEARCH_CENTRE[1],
                                 SEARCH_ALT, 0.0, -89.9, SEARCH_FOV),
                 SEARCH_PX, SEARCH_PX)
    hit = top[..., 6] > 0.0
    boxes = {}
    for px in top[hit]:
        key = tuple(np.round(px[:4], 1))
        boxes[key] = (float(px[4]), int(round(px[5])))
    views = {"canyon_150m": CANYON_150M}
    for name, (style, use, (h0, h1), width, rng, off, zs,
               fov) in TYPED_VIEWS.items():
        want = 100 + style + 10 * use
        found = [(key, hgt) for key, (hgt, code) in boxes.items()
                 if code == want and h0 <= hgt <= h1
                 and key[2] - key[0] >= width]
        if style == 3:
            found.sort(key=lambda kh: (-kh[1], kh[0]))
        else:
            found.sort(key=lambda kh: (
                (0.5 * (kh[0][0] + kh[0][2]) - SEARCH_CENTRE[0]) ** 2
                + (0.5 * (kh[0][1] + kh[0][3]) - SEARCH_CENTRE[1]) ** 2,
                kh[0]))
        # Each candidate from west of the face's normal, east of it, and
        # face on: the view where the building is seen best past whatever
        # stands between (the first of equals, so the pick is stable).
        best, best_share = None, MIN_TARGET_SHARE
        for key, hgt in found[:MAX_CANDIDATES]:
            for a in (off, -off, 0.0):
                cam = _typed_camera(key, hgt, rng, a, zs, fov)
                if _inside_any(cam, boxes):
                    continue
                seen = _probe(probes, state, cam, 320, 180)
                mine = ((np.abs(seen[..., 0] - key[0]) < 0.1)
                        & (np.abs(seen[..., 1] - key[1]) < 0.1)
                        & (seen[..., 6] > 0.0))
                if mine.mean() > best_share:
                    best, best_share = cam, float(mine.mean())
        if best is None:
            raise AssertionError(
                f"{name}: no building of its type ({len(found)} candidates) "
                "fills its view; revisit TYPED_VIEWS or SEARCH_*")
        views[name] = best
    return views


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


def _plain(r, width, height):
    """The plain pass's target as raw words: red carries the shell's mark
    (its sign bit, CITY_SHELL_MARK in raymarch.wgsl), alpha the distance
    (negated where the street stage's ground flag is set)."""
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
    return a.reshape(height, row // 4)[:, :width * 4].reshape(height, width, 4)


def _surface_t(r, width, height):
    """The final surface's distance: the city's last surface stage, the
    street stage (fs_surface_street), which passes the refinement's t on
    unless street furniture stands in front of it."""
    import wgpu
    tex = r._targets["surface"]
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
    a = np.frombuffer(bytearray(buf.read_mapped()), np.float32)
    buf.unmap()
    return a.reshape(height, row // 4)[:, 3:width * 4:4]


def _marked_fraction(r, width, height):
    red = _plain(r, width, height)[..., 0]
    return float(((red & 0x80000000) != 0).mean())


def _shown(r, state, cam, width):
    import cloudyview as cv
    img = np.asarray(r.render(state, _view(cam, width), frames=FRAMES))
    return cv.quantize_uint8(img).astype(np.float64) / 255.0


def _regions(r, state, cam, width):
    shown = _shown(r, state, cam, width)
    gx, gy = GRID
    h, w = shown.shape[:2]
    return np.array([[shown[j * h // gy:(j + 1) * h // gy,
                            i * w // gx:(i + 1) * w // gx].mean()
                      for i in range(gx)] for j in range(gy)])




@pytest.mark.gpu
@pytest.mark.parametrize("view_name", VIEW_NAMES)
def test_shell_hand_off_is_continuous(view_name, city):
    renderers, state, views = city
    cam = views[view_name]
    diff = []
    for width in WIDTHS:
        shell = _regions(renderers["shell"], state, cam, width)
        if width == WIDTHS[-1]:
            marked = _marked_fraction(renderers["shell"],
                                      *_view(cam, width).render_size)
            assert marked >= MIN_MARKED, (
                f"{view_name}: only {marked:.3f} of the frame is marked for "
                "the shell; the view no longer exercises the hand-off")
        diff.append(shell - _regions(renderers["flat"], state, cam, width))
    step = float(np.abs(np.diff(np.array(diff), axis=0)).max())
    bound = MAX_STEP[view_name]
    assert step <= bound, (
        f"{view_name}: the shell's difference from the flat facade moved "
        f"{step:.4f} in a region between adjacent widths (bound {bound})")


@pytest.mark.gpu
@pytest.mark.parametrize("view_name", VIEW_NAMES)
def test_shell_equals_flat_where_it_hands_off(view_name, city):
    """Where the blend reaches 1, the shell's pixel IS the flat facade's.

    The width sweep sees mismatches spread over the band; this sees one
    that sits on the edge of the reach, a line rather than a band (a flat
    twin read at the wrong point on the wall did exactly that). Only pixels
    where the shell kept the plain pass's surface count: a nearer piece of
    relief standing in front of a far wall is meant to differ.
    """
    renderers, state, views = city
    cam = views[view_name]
    diffs = []
    for width in WIDTHS:
        shell = _shown(renderers["shell"], state, cam, width)
        w, h = _view(cam, width).render_size
        t_plain = np.abs(_plain(renderers["shell"], w, h)[..., 3]
                         .view(np.float32))
        t_final = np.abs(_surface_t(renderers["shell"], w, h))
        flat = _shown(renderers["flat"], state, cam, width)
        fp = 2.0 * math.tan(math.radians(cam[5]) / 2.0) / w * t_plain
        band = ((fp >= HANDOFF_FP[0]) & (fp < HANDOFF_FP[1])
                & (np.abs(t_final - t_plain) < 0.02 * t_plain))
        diffs.append((shell - flat)[band])
    d = np.concatenate(diffs, axis=None)
    assert d.size >= HANDOFF_MIN_PIXELS, (
        f"{view_name}: only {d.size} pixels at the hand-off")
    mean_abs = float(np.abs(d).mean())
    bound = MAX_HANDOFF_ABS[view_name]
    assert mean_abs <= bound, (
        f"{view_name}: at the hand-off the shell differs from the flat "
        f"facade by {mean_abs:.5f} on average (bound {bound})")


# --- the shops ---------------------------------------------------------------
#
# A commercial ground floor is a row of lit shop windows (the STORE zone,
# city_store_cell), drawn by the shell and averaged by the flat facade
# through the same window path. It used to be an emitting band painted over
# the wall, which the shell left off its reveals and sills: each jamb was a
# dark sliver in a band that clipped to white, there while the shell drew it
# and gone when it handed over ("the window sides appear and disappear",
# Thomas, on the building with the white first floor). Laying the band on
# every surface instead cured the flicker and left a featureless white wall.
# Two things are held here, up close: the shell's ground floor averages to
# what the flat facade shows there, and its frames and mullions stay
# visible against the glass.
#
# The probe: per pixel where the plain pass hit a building's facade (kind
# 1: never the ground or a roof), that building (its tier-0 box's min x and
# y), the wall-plane height and the distance; elsewhere t < 0. Whether the
# building's ground floor is commercial comes from a top-down pass, which
# reads the storefront glow's mean in z (`_find_storefront`).
STORE_PROBE = """    if (chit.hit && chit.kind == 1) {
        let pc = city_cell(chit.cell);
        return vec4<f32>(pc.b1min.x, pc.b1min.y, chit.pos.z, chit.t);
    }
    if (chit.hit) {
        let pc = city_cell(chit.cell);
        return vec4<f32>(pc.b1min.x, pc.b1min.y,
                         dot(city_storefront_glow(pc), vec3<f32>(1.0 / 3.0)),
                         -1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
"""
# The shop glass's height on the wall (CITY_STORE_PANE_LO/HI: 0.2-3.3 m).
STORE_BAND_Z = (0.25, 3.25)
# A camera STORE_RANGE metres off the face's middle, STORE_OFF degrees off
# its normal so the reveals on one side of every window face it, at
# STORE_Z, looking at STORE_AIM_Z.
STORE_RANGE = 28.0
STORE_OFF = 45.0
STORE_Z = 3.0
STORE_AIM_Z = 1.2
STORE_FOV = 50.0
STORE_SIZE = (960, 540)
STORE_MIN_SHARE = 0.05
STORE_MIN_PIXELS = 5000
# Relief standing more than this in front of the wall plane is an object
# (a canopy, a unit, a pipe, the street's furniture), not the wall.
STORE_IN_FRONT = 0.25
# The shell's band mean against the flat facade's, and the band's dark tail
# (its 10th percentile) against its glass (90th), both as shown.
MAX_STORE_MEAN_ERR = None
MAX_STORE_CONTRAST = None


def _store_probe(level):
    from cloudyview.soar_host import SoarRenderer
    source = SHADER.read_text()
    assert source.count(PROBE_ANCHOR) == 1, (
        "the plain pass's probe anchor moved; update PROBE_ANCHOR")
    r = SoarRenderer(periodic=True, surface="city",
                     shader_source=source.replace(
                         PROBE_ANCHOR, STORE_PROBE + PROBE_ANCHOR))
    r.upload_volume(level.sigma)
    return r


def _store_plane(r, state, cam, w, h):
    from cloudyview.soar_host import ViewState
    x, y, z, az, el, fov = cam
    r.render(state, ViewState(
        camera_position=[x + TILE_OFFSET[0], y + TILE_OFFSET[1], z],
        azimuth=az, elevation=el, fov=fov, output_size=(w, h),
        render_size=(w, h), jitter=False, subpixel=False,
        sun_azimuth=310.0, sun_elevation=22.0, exposure=6.0,
        view_step_lod_degrees=0.0), frames=1)
    return _plain(r, w, h).view(np.float32)


def _store_camera(box, off):
    """STORE_RANGE metres off the middle of the building's south face,
    `off` degrees west of its normal, aimed at the band."""
    x0, y0, x1, _ = (float(v) for v in box)
    a = math.radians(off)
    x = 0.5 * (x0 + x1) - STORE_RANGE * math.sin(a)
    y = y0 - STORE_RANGE * math.cos(a)
    el = math.degrees(math.atan2(STORE_AIM_Z - STORE_Z, STORE_RANGE))
    return (x, y, STORE_Z, off, el, STORE_FOV)


def _store_band(plane, box):
    """The building's own facade in the band's fully lit part, eroded by a
    pixel so that jitter at its edges cannot reach past it."""
    m = ((np.abs(plane[..., 0] - box[0]) < 0.1)
         & (np.abs(plane[..., 1] - box[1]) < 0.1) & (plane[..., 3] > 0.0)
         & (plane[..., 2] > STORE_BAND_Z[0])
         & (plane[..., 2] < STORE_BAND_Z[1]))
    e = m.copy()
    e[1:] &= m[:-1]
    e[:-1] &= m[1:]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def _find_storefront(level, state):
    """The commercial building whose band fills most of its view."""
    probes = _probe_renderers(level)
    store = _store_probe(level)
    top_cam = (SEARCH_CENTRE[0], SEARCH_CENTRE[1], SEARCH_ALT, 0.0, -89.9,
               SEARCH_FOV)
    top = _probe(probes, state, top_cam, SEARCH_PX, SEARCH_PX)
    # Looking straight down, a hit on a roof or the ground hands back the
    # storefront glow of the cell under it (t < 0); a facade seen edge-on
    # hands back a height instead, so it is not read.
    plane = _store_plane(store, state, top_cam, SEARCH_PX, SEARCH_PX)
    glow = np.where(plane[..., 3] < 0.0, plane[..., 2], 0.0)
    hit = top[..., 6] > 0.0
    boxes, lit = {}, set()
    for px, g in zip(top[hit], glow[hit]):
        key = tuple(np.round(px[:4], 1))
        boxes[key] = (float(px[4]), int(round(px[5])))
        if g > 0.0:
            lit.add(key)
    found = sorted((key for key in lit if key[2] - key[0] >= 20.0),
                   key=lambda k: (
                       (0.5 * (k[0] + k[2]) - SEARCH_CENTRE[0]) ** 2
                       + (0.5 * (k[1] + k[3]) - SEARCH_CENTRE[1]) ** 2, k))
    best, best_share = None, STORE_MIN_SHARE
    for key in found[:MAX_CANDIDATES]:
        for off in (STORE_OFF, -STORE_OFF):
            cam = _store_camera(key, off)
            if _inside_any(cam, boxes):
                continue
            share = float(_store_band(
                _store_plane(store, state, cam, 320, 180), key).mean())
            if share > best_share:
                best, best_share = (key, cam), share
    if best is None:
        raise AssertionError(
            f"no storefront ({len(found)} candidates) fills its view; "
            "revisit STORE_* or SEARCH_*")
    return best, store


def _store_measure(renderers, state):
    """(shell band mean / flat band mean - 1, p10 / p90 of the shell's
    band, pixels, building, camera)."""
    level = build_soar_level()
    (box, cam), store = _find_storefront(level, state)
    w, h = STORE_SIZE
    band = _store_band(_store_plane(store, state, cam, w, h), box)
    shell = _shown(renderers["shell"], state, cam, w).mean(axis=-1)
    t_plain = np.abs(_plain(renderers["shell"], w, h)[..., 3]
                     .view(np.float32))
    t_final = np.abs(_surface_t(renderers["shell"], w, h))
    flat = _shown(renderers["flat"], state, cam, w).mean(axis=-1)
    wall = band & (t_final > t_plain - STORE_IN_FRONT)
    err = float(shell[wall].mean() / flat[wall].mean() - 1.0)
    p10, p90 = np.percentile(shell[wall], [10.0, 90.0])
    return err, float(p10 / p90), int(wall.sum()), box, cam


@pytest.mark.gpu
def test_shops_agree_and_keep_their_frames(city):
    """Up close, the shell's shop windows average to what the flat facade
    shows there, and their frames stay darker than their glass."""
    renderers, state, _ = city
    err, contrast, n, box, cam = _store_measure(renderers, state)
    assert n >= STORE_MIN_PIXELS, f"only {n} pixels of shop band in view"
    where = f"building {box}, camera {cam}"
    assert abs(err) <= MAX_STORE_MEAN_ERR, (
        f"the shell's shops show {err:+.3f} against the flat facade's "
        f"(bound {MAX_STORE_MEAN_ERR}); {where}")
    assert contrast <= MAX_STORE_CONTRAST, (
        f"the shop band's dark tail is {contrast:.3f} of its glass (bound "
        f"{MAX_STORE_CONTRAST}): the frames are lost; {where}")
