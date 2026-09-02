"""Flight tracks: the file soar records, resampled and re-rendered as video.

    soar TRACK.json

Recording in the browser captures only the *track* — per-frame (t, camera)
samples plus the render header a still would carry — not pixels. This module
is the other half: resample that track at an exact output frame rate
(non-uniform Catmull-Rom through the hand-flown samples, so irregular
in-flight timing interpolates correctly) and render every frame with the
converged accumulation the live view only reaches when you stop moving.
Frames stream straight into ffmpeg; no intermediate files, and no fallback
encoder — a directory of PNGs is not a video.

The browser has the same pipeline (web/soar/track.js + video.js, "Render it
to video"), and the two exist for different reasons: the in-tab render is
what a visitor to the site gets; this one is what a GPU under your own
control gets — a render that survives the tab, and a terminal to watch it
from. The resampler here and track.js's are one algorithm in two languages,
pinned against each other by tests/test_track_resample.py.

Track schema (shared with the browser, which wrote it):

    {"schema": "cloudyview.track.v2",
     "header": <render metadata: source, camera, sun, [city], render>,
     "samples": [[t, x, y, z, azimuth, elevation, fov, sx, sy], ...]}

x/y/z are the relative-coordinate convention (the CLOUD frame, folded at the
cloud period — exactly what `witness --camera-position` takes) and sx/sy the
camera's position in the scene's surface-tile frame in TILE-RELATIVE units
(period 1.0), so the schema needs no tile size. The two frames wrap at
independent periods, which is why a sample carries both: fold one into the
other and a recorded city flight replays over the wrong district. A v1 track
(7 columns, no surface frame) still renders; its surface then holds the
static offset the header recorded, which is the un-shifted surface those
recordings meant.

Everything about the picture comes from the header — field, sun, look,
quality tier, size — so the command needs nothing but the track. The flags
exist to override one thing at a time (`--size`, `--fps`, `--quality`,
`--field` when the NetCDF is not where the header says).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Optional, Sequence

import numpy as np

TRACK_SCHEMAS = ("cloudyview.track.v1", "cloudyview.track.v2")
HEADER_SCHEMA = "cloudyview.render.v1"

# Relative x/y span the domain as [-1, 1], so a periodic flight-path wrap is
# a near-full-span jump between consecutive samples and nothing else is.
REL_PERIOD = 2.0
WRAP_JUMP_THRESHOLD = 1.0

# The browser's defaults for the same render (web/soar/constants.js
# DEFAULT_VIDEO_FPS). A track carries no frame rate: the samples are
# timestamped and the rate is the render's choice.
DEFAULT_VIDEO_FPS = 60.0
DEFAULT_CRF = 18
# Every video frame starts its jitter sequence this far from the last, so
# neighbouring frames' accumulation passes are decorrelated. The browser's
# renderTrackVideo uses the same stride (frameIndex: i * 1024).
FRAME_INDEX_STRIDE = 1024


# --- the file --------------------------------------------------------------

def load_track(path) -> tuple[dict, np.ndarray]:
    """Read a track file -> (header, samples[n, 7 or 9]).

    The width check is the whole validation: 9 columns is a v2 sample, 7 a
    v1 recording, and a mix is a corrupt track rather than a choice to make
    quietly. Everything else about the header is checked where it is used,
    with the key that was missing named.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    schema = payload.get("schema")
    if schema not in TRACK_SCHEMAS:
        raise ValueError(
            f"{path}: expected schema one of {TRACK_SCHEMAS}, got {schema!r}.")
    if "header" not in payload or "samples" not in payload:
        raise ValueError(f"{path}: a track needs both 'header' and 'samples'.")
    rows = payload["samples"]
    widths = {len(s) for s in rows}
    if len(rows) < 2:
        raise ValueError(
            f"{path}: {len(rows)} sample(s) is not enough to interpolate.")
    if len(widths) != 1 or not (widths <= {7, 9}):
        raise ValueError(
            f"{path}: track samples must uniformly have 7 columns (v1 / no "
            f"surface frame) or 9 (v2 with one); got widths "
            f"{sorted(widths)}.")
    samples = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{path}: track samples contain non-finite values.")
    return payload["header"], samples


# --- resampling: a port of web/soar/track.js, kept step for step -----------

def unwrap_periodic(values, period: float = REL_PERIOD,
                    threshold: float = WRAP_JUMP_THRESHOLD) -> np.ndarray:
    """Make a wrapped coordinate continuous across period jumps."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    correction = 0.0
    for i in range(1, len(out)):
        jump = values[i] - values[i - 1]
        if jump > threshold:
            correction -= period
        elif jump < -threshold:
            correction += period
        out[i] += correction
    return out


def unwrap_degrees(values) -> np.ndarray:
    """Angles in degrees, made continuous — numpy's unwrap, in degrees.

    Spelled as track.js spells it rather than through np.unwrap, so the two
    resamplers agree to the last bit on the same input.
    """
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    correction = 0.0
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        dmod = math.fmod(math.fmod(d + 180.0, 360.0) + 360.0, 360.0) - 180.0
        if dmod == -180.0 and d > 0.0:
            dmod = 180.0
        if abs(d) >= 180.0:
            correction += dmod - d
        out[i] = values[i] + correction
    return out


def catmull_rom(times, values, t_out) -> np.ndarray:
    """Non-uniform (time-parameterized) Catmull-Rom through every sample.

    Barry-Goldman, with the real sample times as knots, so irregular
    in-flight frame timing interpolates correctly instead of being treated
    as evenly spaced — which is what turns a stutter in the recording into
    a lurch in the video. Endpoints clamp their outer control points.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    n = len(times)
    out = np.empty(len(t_out), dtype=np.float64)
    i = 0                       # t_out is ascending, so the knot only marches
    for k, t in enumerate(t_out):
        while i + 1 < n and times[i + 1] <= t:
            i += 1
        i = min(max(i, 0), n - 2)
        i0, i1, i2, i3 = max(i - 1, 0), i, i + 1, min(i + 2, n - 1)
        t0, t1, t2, t3 = times[i0], times[i1], times[i2], times[i3]
        p0, p1, p2, p3 = values[i0], values[i1], values[i2], values[i3]
        # Degenerate knot spacing (clamped ends, duplicate stamps) falls back
        # to linear inside the segment.
        if t2 <= t1:
            out[k] = p1
            continue

        def lerp(pa, pb, ta, tb):
            return pa if tb <= ta else pa + (pb - pa) * ((t - ta) / (tb - ta))

        a1 = lerp(p0, p1, t0, t1)
        a2 = lerp(p1, p2, t1, t2)
        a3 = lerp(p2, p3, t2, t3)
        b1 = lerp(a1, a2, t0, t2)
        b2 = lerp(a2, a3, t1, t3)
        out[k] = lerp(b1, b2, t1, t2)
    return out


def frame_count_between(t0: float, t1: float, fps: float) -> int:
    return int(math.floor((t1 - t0 + 1e-9) * fps)) + 1


@dataclasses.dataclass(frozen=True)
class TrackFrame:
    """One output frame's camera: relative cloud-frame position, angles, and
    the tile-relative surface position (None on a v1 track)."""
    t: float
    position: tuple[float, float, float]
    azimuth: float
    elevation: float
    fov: float
    surface_position: Optional[tuple[float, float]]


def resample_track(samples, fps: float, *,
                   periodic: bool = True) -> list[TrackFrame]:
    """Resample hand-flown samples at exact 1/fps steps.

    Azimuth is unwrapped before interpolation so 359 to 1 goes through 0
    rather than the long way round; in a periodic domain each wrapped
    position column is unwrapped the same way at its own frame's period and
    re-wrapped afterwards — x/y at the cloud period, sx/sy at 1.0.
    """
    if not fps > 0:
        raise ValueError(f"fps must be positive; got {fps}.")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] not in (7, 9):
        raise ValueError(
            "samples must be [n, 7] (v1) or [n, 9] (v2); got shape "
            f"{samples.shape}.")
    order = np.argsort(samples[:, 0], kind="stable")
    samples = samples[order]
    keep = np.ones(len(samples), dtype=bool)
    keep[1:] = np.diff(samples[:, 0]) > 0
    unique = samples[keep]
    if len(unique) < 2:
        raise ValueError(
            f"The track collapses to {len(unique)} sample(s) with distinct "
            "times, which is not enough to interpolate. Fly for longer.")
    has_surface = unique.shape[1] == 9

    times = unique[:, 0]
    x, y = unique[:, 1], unique[:, 2]
    if periodic:
        x, y = unwrap_periodic(x), unwrap_periodic(y)
    az = unwrap_degrees(unique[:, 4])
    sx = sy = None
    if has_surface and periodic:
        sx = unwrap_periodic(unique[:, 7], 1.0, 0.5)
        sy = unwrap_periodic(unique[:, 8], 1.0, 0.5)
    elif has_surface:
        sx, sy = unique[:, 7], unique[:, 8]

    count = frame_count_between(times[0], times[-1], fps)
    t_out = times[0] + np.arange(count, dtype=np.float64) / fps

    cols = {
        "x": catmull_rom(times, x, t_out),
        "y": catmull_rom(times, y, t_out),
        "z": catmull_rom(times, unique[:, 3], t_out),
        "az": catmull_rom(times, az, t_out),
        "el": catmull_rom(times, unique[:, 5], t_out),
        "fov": catmull_rom(times, unique[:, 6], t_out),
    }
    if has_surface:
        cols["sx"] = catmull_rom(times, sx, t_out)
        cols["sy"] = catmull_rom(times, sy, t_out)
    if periodic:
        # Fold back at the same period each unwrap used: x/y centred into
        # [-1, 1), sx/sy into [0, 1). math.fmod, not %, because the browser
        # folds with JavaScript's % (which keeps the dividend's sign) and
        # then adds the period back — the same two steps, the same bits.
        for key in ("x", "y"):
            cols[key] = np.array([
                math.fmod(math.fmod(v + 1.0, REL_PERIOD) + REL_PERIOD,
                          REL_PERIOD) - 1.0 for v in cols[key]])
        for key in (("sx", "sy") if has_surface else ()):
            cols[key] = np.array([
                math.fmod(math.fmod(v, 1.0) + 1.0, 1.0) for v in cols[key]])

    frames = []
    for k in range(count):
        frames.append(TrackFrame(
            t=float(t_out[k]),
            position=(float(cols["x"][k]), float(cols["y"][k]),
                      float(cols["z"][k])),
            azimuth=math.fmod(math.fmod(cols["az"][k], 360.0) + 360.0, 360.0),
            elevation=float(min(90.0, max(-90.0, cols["el"][k]))),
            fov=float(cols["fov"][k]),
            surface_position=((float(cols["sx"][k]), float(cols["sy"][k]))
                              if has_surface else None),
        ))
    return frames


# --- the header, read into render settings --------------------------------

@dataclasses.dataclass
class TrackSettings:
    """What the header says the render is, with the CLI's overrides applied."""
    field_path: Path
    ice_path: Optional[str]
    liquid_var: Optional[str]
    ice_var: Optional[str]
    size: tuple[int, int]               # even, as H.264 needs
    fps: float
    quality: str                        # a QUALITY_PRESETS name
    periodic: bool
    sun_azimuth: float
    sun_elevation: float
    exposure: float
    tone_map_gamma: float
    tone_map_white_point: float
    contrast: float
    haze: float
    haze_height_dependent: bool
    lod_strength: float
    city: bool
    city_tile_offset_m: Optional[tuple[float, float]]
    city_tile_extent_m: Optional[float]
    # The static row-24 surface offset the flight recorded — what a v1 track
    # (no per-sample surface frame) renders every frame with.
    surface_offset_m: Optional[tuple[float, float]]


def even_size(size) -> tuple[int, int]:
    """H.264 has no odd dimensions, anywhere; the browser snaps the same way."""
    w, h = (int(v) for v in size)
    return (w & ~1, h & ~1)


def _need(block: dict, key: str, where: str):
    if key not in block:
        raise ValueError(
            f"the track header's '{where}' block has no '{key}'; this file "
            "was written by a soar older than the schema this command reads.")
    return block[key]


def settings_from_header(header: dict, *, track_path=None,
                         field_path=None, size=None, fps=None,
                         quality=None) -> TrackSettings:
    """Turn the header a still would carry into the render's settings.

    The header is the record of what the browser rendered; every value here
    is read from it, and an override is the caller's explicit choice. The
    field path is the one exception with a rule of its own: the header names
    the file the way `witness` was told to (usually bare, "run this in the
    folder the file is in"), so it resolves against the current directory,
    with `field_path` for when the file lives elsewhere.
    """
    from .witness import QUALITY_PRESETS

    schema = header.get("schema")
    if schema != HEADER_SCHEMA:
        raise ValueError(
            f"the track header has schema {schema!r}; expected {HEADER_SCHEMA!r}.")
    source = _need(header, "source", "header")
    render = _need(header, "render", "header")
    sun = _need(header, "sun", "header")
    city = header.get("city")

    if field_path is None:
        field_path = _need(source, "path", "source")
        if not field_path:
            raise ValueError("the track header names no source field.")
    field_path = Path(field_path)
    if not field_path.exists():
        raise FileNotFoundError(
            f"The cloud field this track was flown over, {field_path}, is "
            "not here. Run the command in the folder the file is in, or "
            "pass --field with its path.")

    if quality is None:
        quality = _need(render, "quality", "render")
    if quality not in QUALITY_PRESETS:
        raise ValueError(
            f"unknown quality tier {quality!r}; expected one of "
            f"{sorted(QUALITY_PRESETS)}.")
    if size is None:
        size = _need(render, "size", "render")
    if len(size) != 2 or min(int(v) for v in size) < 2:
        raise ValueError(f"size must be two integers of at least 2; got {size!r}.")
    if fps is None:
        fps = DEFAULT_VIDEO_FPS
    if not (0 < float(fps) <= 240):
        raise ValueError(f"fps must be in (0, 240]; got {fps!r}.")

    city_block = None
    if city is not None:
        city_block = dict(
            tile_offset_m=tuple(float(v) for v in _need(city, "tile_offset_m", "city")),
            tile_extent_m=float(_need(city, "tile_extent_m", "city")),
            surface_offset_m=tuple(float(v) for v in _need(city, "surface_offset_m", "city")),
        )

    return TrackSettings(
        field_path=field_path,
        ice_path=source.get("ice_path"),
        liquid_var=source.get("liquid_var"),
        ice_var=source.get("ice_var"),
        size=even_size(size),
        fps=float(fps),
        quality=quality,
        periodic=bool(_need(render, "periodic", "render")),
        sun_azimuth=float(_need(sun, "azimuth", "sun")),
        sun_elevation=float(_need(sun, "elevation", "sun")),
        exposure=float(_need(render, "exposure", "render")),
        tone_map_gamma=float(_need(render, "tone_map_gamma", "render")),
        tone_map_white_point=float(_need(render, "tone_map_white_point", "render")),
        contrast=float(_need(render, "contrast", "render")),
        haze=float(_need(render, "haze", "render")),
        haze_height_dependent=bool(_need(render, "haze_height_dependent", "render")),
        lod_strength=float(_need(render, "lod_strength", "render")),
        city=city_block is not None,
        city_tile_offset_m=city_block["tile_offset_m"] if city_block else None,
        city_tile_extent_m=city_block["tile_extent_m"] if city_block else None,
        surface_offset_m=city_block["surface_offset_m"] if city_block else None,
    )


def fold_tile(v: float, tile: float) -> float:
    """The browser's foldTile: one fmod, then a conditional add."""
    r = math.fmod(float(v), tile)
    return r + tile if r < 0 else r


def surface_offset_for(origin_xy, surface_position_m, tile: float
                       ) -> tuple[float, float]:
    """Row 24 for a pose that carries its own tile phase.

    cam.xy minus surfacePosition, each folded into [0, tile) — fold(origin)
    FIRST and the surface position subtracted after, exactly as
    uniforms.js packUniforms does it, so a cloud-period fold of the camera
    moves the offset with it and the surface never jumps.
    """
    return (fold_tile(fold_tile(origin_xy[0], tile) - surface_position_m[0], tile),
            fold_tile(fold_tile(origin_xy[1], tile) - surface_position_m[1], tile))


# --- the render ------------------------------------------------------------

def _ffmpeg_command(out_path: Path, size, fps: float, crf: int) -> list[str]:
    w, h = size
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", f"{fps:g}", "-i", "-",
        "-c:v", "libx264", "-crf", str(int(crf)), "-preset", "medium",
        # RGB -> BT.709 limited range, and say so in the file: an untagged
        # yuv420p stream is matrixed as BT.601 by swscale and then guessed
        # at by every player, which is the washed-out-on-one-player,
        # oversaturated-on-the-next failure video.js documents. The tags
        # ride on the frames (setparams) rather than on the encoder: ffmpeg
        # 8 lets the filter graph's "unknown" override -color_primaries and
        # friends, and the file then carries only the matrix.
        "-vf", ("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                "setparams=color_primaries=bt709:color_trc=bt709:"
                "colorspace=bt709:range=tv"),
        "-movflags", "+faststart",
        str(out_path),
    ]


def render_track(track_path, out_path=None, *, field_path=None, size=None,
                 fps=None, quality=None, crf: int = DEFAULT_CRF,
                 nest_group: Optional[str] = None,
                 load_kwargs: Optional[dict] = None,
                 progress_every_s: float = 5.0) -> Path:
    """Re-render a recorded track into an H.264 mp4. Blocks; prints progress.

    Every frame is rendered with the tier's parked accumulation — the still
    the app converges to when the camera stops — and the look the header
    recorded: sun, exposure, tone map, haze, LOD. The exposure is the one
    the capture metered and HELD, the way the browser's video holds one
    exposure for the whole file; a per-frame meter would flicker.

    `load_kwargs` are `cloudyview.load` overrides (the dataset-selection
    flags); the header's own variable names are the defaults beneath them.
    The nest, when `nest_group` names one, loads from that group of the same
    file with the same name overrides — the rule `witness --nest-group`
    follows.

    Overlays (the bird, the minimap) are the browser's; this render is the
    picture alone.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not on PATH, and it is the encoder: install it "
            "(e.g. `sudo dnf install ffmpeg`). Not falling back to writing "
            "frame files — a directory of PNGs is not a video.")

    from .basic_render import quantize_uint8
    from .cloudfield import load as load_field
    from .soar_host import (SceneState, SoarRenderer, ViewState,
                            APP_LIGHT_MARCH_LOD_DEGREES,
                            APP_VIEW_STEP_LOD_DEGREES, camera_world_origin)
    from .witness import (LIGHT_CACHE_DIVISOR, OCEAN_REFLECTANCE,
                          QUALITY_PRESETS, _field_level)

    track_path = Path(track_path)
    out_path = Path(out_path) if out_path is not None \
        else track_path.with_suffix(".mp4")
    header, samples = load_track(track_path)
    s = settings_from_header(header, track_path=track_path,
                             field_path=field_path, size=size, fps=fps,
                             quality=quality)
    preset = QUALITY_PRESETS[s.quality]
    frames = resample_track(samples, s.fps, periodic=s.periodic)
    w, h = s.size
    print(f"soar: {track_path.name}: {len(samples)} samples over "
          f"{samples[-1, 0] - samples[0, 0]:.1f} s -> {len(frames)} frames "
          f"at {s.fps:g} fps, {w}x{h}, quality {s.quality}"
          f"{', city' if s.city else ''}", flush=True)

    # The field, with the header's variable names beneath the caller's.
    kwargs = dict(load_kwargs or {})
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.setdefault("liquid_water_var", s.liquid_var)
    kwargs.setdefault("ice_water_var", s.ice_var)
    ice = kwargs.pop("ice", None) or s.ice_path
    print(f"soar: loading {s.field_path}", flush=True)
    field = load_field(s.field_path, ice=ice, **kwargs)
    outer = _field_level(field, "outer" if nest_group else "single",
                         verbose=True)
    levels = [outer]
    if nest_group:
        name_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("dataset_group", "liquid_water_group",
                                    "ice_water_group", "coords_group")}
        print(f"soar: loading nest group {nest_group}", flush=True)
        nest_field = load_field(s.field_path, dataset_group=nest_group,
                                **name_kwargs)
        levels = [_field_level(nest_field, nest_group, verbose=True), outer]

    renderer = SoarRenderer(periodic=s.periodic, nested=len(levels) > 1,
                            tone_map=True, city=s.city)
    renderer.upload_volume(outer.sigma)
    if len(levels) > 1:
        renderer.upload_nest(levels[0].sigma)

    # The surface tile the shader will read is the one installed with this
    # package; the header says which one the flight flew over. They have to
    # be the same tile for sx/sy — tile-relative — to mean the same metres.
    meta = renderer.surface_meta
    tile = float(meta["tile_extent_m"])
    if s.city and s.city_tile_extent_m != tile:
        raise RuntimeError(
            f"the track was flown over a city tile of {s.city_tile_extent_m:g} "
            f"m, but the installed city tile is {tile:g} m; the flight's "
            "surface positions would land on the wrong blocks.")

    min_voxel = min(outer.dx)
    state = SceneState(
        bmin=[float(v) for v in outer.bmin],
        bmax=[float(v) for v in outer.bmax],
        dt_view=min_voxel * preset["step_factor"],
        dt_light=min_voxel * preset["light_step_factor"],
        periodic=s.periodic,
        ocean_reflectance=OCEAN_REFLECTANCE,
        ocean_fif_dx=float(meta["cell_m"] if s.city else meta["dx_m"]),
        ocean_tile_extent=tile,
        ocean_max_lod=int(meta["mips"]) - 1,
        nested=len(levels) > 1,
        city=s.city,
        **({"city_tile_offset_m": s.city_tile_offset_m} if s.city else {}),
        surface_offset_m=s.surface_offset_m,
    )
    if len(levels) > 1:
        fine = levels[0]
        state.nest_bmin = [float(v) for v in fine.bmin]
        state.nest_bmax = [float(v) for v in fine.bmax]
        state.dt_view_nest = min(fine.dx) * preset["step_factor"]
        state.dt_light_nest = min(fine.dx) * preset["light_step_factor"]

    def view_for(frame: TrackFrame, index: int) -> ViewState:
        origin = camera_world_origin(frame.position, outer.bmin, outer.bmax)
        return ViewState(
            camera_position=[float(v) for v in origin],
            azimuth=frame.azimuth, elevation=frame.elevation, fov=frame.fov,
            output_size=(w, h), render_size=(w, h),
            sun_azimuth=s.sun_azimuth, sun_elevation=s.sun_elevation,
            exposure=s.exposure, tone_map_gamma=s.tone_map_gamma,
            tone_map_white_point=s.tone_map_white_point, contrast=s.contrast,
            haze=s.haze, haze_height_dependent=s.haze_height_dependent,
            light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES * s.lod_strength,
            view_step_lod_degrees=APP_VIEW_STEP_LOD_DEGREES * s.lod_strength,
            light_cache=bool(preset["light_cache"]),
            sky_probe=bool(preset["sky_probe"]),
            frame_index=index * FRAME_INDEX_STRIDE,
        )

    def state_for(frame: TrackFrame, view: ViewState) -> SceneState:
        # Row 24 per frame: the pose's own tile phase when the track carries
        # one; otherwise the static offset the header recorded, every frame.
        if frame.surface_position is None:
            return state
        sp = (frame.surface_position[0] * tile, frame.surface_position[1] * tile)
        return dataclasses.replace(
            state, surface_offset_m=surface_offset_for(
                view.camera_position[:2], sp, tile))

    # The sun-tau cache depends on the field and the sun, neither of which
    # moves along a track, so it is baked once — at the first frame's view,
    # which carries the same sun as every other.
    if preset["light_cache"]:
        print("soar: baking the sun-tau cache", flush=True)
        renderer.bake_light_cache(state, view_for(frames[0], 0),
                                  divisor=LIGHT_CACHE_DIVISOR)

    accumulate = int(preset["accumulate"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(_ffmpeg_command(out_path, (w, h), s.fps, crf),
                            stdin=subprocess.PIPE)
    t_start = time.perf_counter()
    last_report = t_start
    try:
        for i, frame in enumerate(frames):
            view = view_for(frame, i)
            image = renderer.render(state_for(frame, view), view,
                                    frames=accumulate)
            proc.stdin.write(
                np.ascontiguousarray(quantize_uint8(image, seed=i)).tobytes())
            now = time.perf_counter()
            if now - last_report >= progress_every_s or i + 1 == len(frames):
                rate = (i + 1) / (now - t_start)
                eta = (len(frames) - i - 1) / max(rate, 1e-9)
                print(f"soar: frame {i + 1}/{len(frames)} "
                      f"({rate:.2f} frames/s, {eta / 60:.1f} min left)",
                      flush=True)
                last_report = now
    except BaseException:
        proc.stdin.close()
        proc.terminate()
        proc.wait()
        out_path.unlink(missing_ok=True)   # no half-written video left behind
        raise
    proc.stdin.close()
    code = proc.wait()
    if code != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited with status {code}.")
    print(f"soar: wrote {out_path} ({len(frames)} frames, "
          f"{len(frames) / s.fps:.1f} s at {s.fps:g} fps, "
          f"{(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
    return out_path


# --- the command -----------------------------------------------------------

def cli(argv: Optional[Sequence[str]] = None) -> None:
    from .cli_utils import (CloudyViewHelpFormatter, DATA_SELECTION_HELP,
                            add_dataset_selection_arguments,
                            dataset_selection_kwargs)
    from .witness import QUALITY_PRESETS

    parser = argparse.ArgumentParser(
        prog="soar",
        description="Re-render a flight track recorded in soar into a video.",
        formatter_class=CloudyViewHelpFormatter,
        epilog=dedent(
            f"""
            What `soar` does:
              1. Reads the track soar saved (R to record, then "Save the
                 track"): the camera samples and the render header.
              2. Loads the cloud field the header names.
              3. Resamples the flight at an exact frame rate and renders
                 every frame with the quality tier's converged accumulation,
                 with the same WGSL ray marcher the browser runs.
              4. Streams the frames into ffmpeg as an H.264 mp4 next to the
                 track (or at --output).

            Everything about the picture — field, sun, exposure, tone map,
            haze, LOD, quality tier, size, the night city — comes from the
            header, so the command needs only the track. The flags override
            one thing at a time. Overlays (the bird, the minimap) are the
            browser's; this is the picture alone.

            Dependencies:
              A GPU (through wgpu) and the `ffmpeg` binary on PATH. There is
              no fallback encoder.

            {DATA_SELECTION_HELP}

            Examples:
              soar cloudyview_track_20260901.json
              soar flight.json --output flight_4k.mp4 --size 3840 2160
              soar flight.json --field ~/fields/marine-congestus.nc --fps 30
            """
        ),
    )
    parser.add_argument("track", help="the .json track soar saved")
    parser.add_argument("--output", "-o", metavar="OUT.mp4",
                        help="video path (default: the track's name with .mp4)")
    parser.add_argument("--field", metavar="FILE.nc",
                        help="the cloud field, when it is not where the "
                             "header says (the header names it the way "
                             "witness was told to, relative to the folder "
                             "it is run in)")
    parser.add_argument("--size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="video size (default: the capture size the "
                             "header recorded; odd sizes round down to even)")
    parser.add_argument("--fps", type=float,
                        help=f"frame rate (default {DEFAULT_VIDEO_FPS:g}, the "
                             "browser's)")
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS),
                        help="render at this soar tier instead of the one "
                             "the header recorded")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF,
                        help=f"x264 quality, lower is better (default {DEFAULT_CRF})")
    parser.add_argument("--nest-group", metavar="GROUP",
                        help="NetCDF group in the same file holding a finer "
                             "field to render as a nest inside the outer domain")
    add_dataset_selection_arguments(parser)
    args = parser.parse_args(argv)

    try:
        render_track(args.track, args.output, field_path=args.field,
                     size=tuple(args.size) if args.size else None,
                     fps=args.fps, quality=args.quality, crf=args.crf,
                     nest_group=args.nest_group,
                     load_kwargs=dataset_selection_kwargs(args))
    except KeyboardInterrupt:
        print("soar: cancelled; nothing was saved.", file=sys.stderr)
        sys.exit(130)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
