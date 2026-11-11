#!/usr/bin/env python
"""witness: a still image of a cloud field, ray-marched on the GPU.

Library:
    import cloudyview as cv
    image = cv.witness(cv.load("cloud.nc"), cv.Camera(azimuth=30))

Command line:
    witness cloud.nc [--quality high] [--size W H] ...

witness prepares a field and a camera for soar's shader and runs it through
wgpu (soar/renderer.py). The browser app runs the same shader files, so a
witness image is the picture soar shows. Rendering needs a GPU; there is no
CPU path.

The layers, from the top:
    cli()            parses the command line
    main()           loads the field, calls witness(), saves the PNG
    witness()        a CloudField and a Camera -> an image
    render_nested()  extinction levels and a camera in metres -> an image

Coordinates are meteorological: +x east, +y north, +z up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional, Sequence, Tuple

import numpy as np

from . import optical_depth
from .camera import Camera
from .cli import (
    LOAD_KEYWORDS,
    CloudyViewHelpFormatter,
    DATA_SELECTION_HELP,
    add_dataset_selection_arguments,
    dataset_selection_kwargs,
)
from .field import CloudField, load as _load_field
from .soar.look import (AERIAL_SCALE_HEIGHT_M, DEFAULT_HAZE, HAZE_MAX,
                        HAZE_MIN)
from .soar.renderer import STILL_ACCUMULATE_FRAMES, SoarRenderer
from .soar.surfaces import (
    SURFACE_DIRS,
    SURFACES,
    check_surface,
    surface_cell_m,
    surface_clock_period_s,
    surface_period_m,
)
from .soar.tiers import (
    CITY_ROOF_DENSITY_BY_TIER,
    LIGHT_CACHE_DIVISOR,
    QUALITY_PRESETS,
)
from .soar.uniforms import (
    APP_LIGHT_MARCH_LOD_DEGREES,
    APP_VIEW_STEP_LOD_DEGREES,
    DEFAULT_CONTRAST,
    DEFAULT_EXPOSURE,
    DEFAULT_HAZE_HEIGHT_DEPENDENT,
    DEFAULT_LOD_STRENGTH,
    DEFAULT_OCEAN_REFLECTANCE,
    DEFAULT_SUN_AZIMUTH,
    DEFAULT_SUN_ELEVATION,
    DEFAULT_TONE_MAP_GAMMA,
    DEFAULT_TONE_MAP_WHITE_POINT,
    STEP_VOXEL_FACTOR,
    SceneState,
    ViewState,
    camera_world_origin,
)

__all__ = ["witness", "render_nested", "tone_map", "NestedLevel",
           "crop_empty_z", "main", "cli"]

# The image size when none is given, in pixels (width, height).
DEFAULT_SIZE = (600, 400)

# Effective radii the extinction conversion assumes. web/soar/constants.js has
# the same numbers, so the browser derives the same extinction from a file.
RE_LIQUID_UM = 10.0
RE_ICE_UM = 30.0
# Ice below this (g/kg) everywhere in a field is dropped before rendering.
ICE_NEGLIGIBLE_G_KG = 1e-6
# A factor on the extinction of every field rendered.
EXTINCTION_MULTIPLIER = 1.0


@dataclass
class NestedLevel:
    """One refinement level: extinction field plus its absolute-meter AABB.

    sigma is m^-1. bmin/bmax are absolute world meters.
    """
    # (nx, ny, nz) m^-1, in the extinction's own dtype: float32 for a
    # float32 field (the loaders'), float64 for a float64 one.
    sigma: np.ndarray
    bmin: np.ndarray         # (3,) float64: (x_min, y_min, z_min) meters
    bmax: np.ndarray         # (3,) float64: (x_max, y_max, z_max) meters
    name: str = ""

    @property
    def dx(self) -> Tuple[float, float, float]:
        nx, ny, nz = self.sigma.shape
        return (
            (self.bmax[0] - self.bmin[0]) / nx,
            (self.bmax[1] - self.bmin[1]) / ny,
            (self.bmax[2] - self.bmin[2]) / nz,
        )


def tone_map(image, exposure=4.0, gamma=1.4):
    """Reinhard plus gamma, for linear radiance (return_linear=True).

    This is behold's encode. It is not the shader's tone map, which has a
    white point and a contrast term; and an image rendered without
    return_linear is already tone-mapped, so do not run this over it.
    """
    exposed = image * exposure
    tone_mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)


# One field is usually rendered many times (a demo bake moves a camera
# through hundreds of positions in one volume). Creating a device and
# uploading a several-hundred-megabyte texture per call would cost more than
# the rendering, so the last renderer is kept and reused while the field has
# not changed.
_session = None
_session_key = None
# The arrays the key identifies by id(), held so they stay alive: an id is
# unique only while its object lives, and a new field that reused a freed id
# would be served the previous field's texture.
_session_arrays = None


def _volume_aabb(field: CloudField):
    """Absolute-meter AABB with half-cell padding.

    tools/export_web_assets.py has the same rule for the browser's demos;
    the two must agree, or a baked demo sits in a different place from the
    same file opened directly.
    """
    x = np.asarray(field.x, dtype=np.float64)
    y = np.asarray(field.y, dtype=np.float64)
    z = np.asarray(field.z, dtype=np.float64)
    dx_half = 0.5 * abs(x[1] - x[0])
    dy_half = 0.5 * abs(y[1] - y[0])
    dz_lo_half = 0.5 * abs(z[1] - z[0])
    dz_hi_half = 0.5 * abs(z[-1] - z[-2])
    bmin = np.array([x.min() - dx_half, y.min() - dy_half, z.min() - dz_lo_half])
    bmax = np.array([x.max() + dx_half, y.max() + dy_half, z.max() + dz_hi_half])
    return bmin, bmax


# A value stores as a nonzero fp16 exactly when it exceeds half the smallest
# positive subnormal: round-to-nearest-even sends 2**-25 itself to zero.
# Extinction is non-negative, so this one comparison is the whole test.
_FP16_NONZERO_FLOOR = 2.0 ** -25


def crop_empty_z(sigma: np.ndarray, z: np.ndarray):
    """Trim z planes that hold no cloud, returning (sigma, z, (lo, hi)).

    Cloud fields are mostly empty sky, and the emptiness is overwhelmingly
    vertical: measured across the demo set, 8% of the z extent is vacuum on a
    STEAM parent, 19% on TWP-ICE LPT, 35% on the FIF cascade, 40% on CM1, and
    75% on DYCOMS, whose deck occupies 137 of 531 levels. Sizing the volume to
    the file's z extent pays for all of it twice — in memory, and in a march
    that crosses the vacuum sample by sample because nothing tells it there is
    nothing there.

    Emptiness is judged on the value AS STORED, not on sigma as computed. The
    texture is r16float here and in the browser, so a sigma below the smallest
    fp16 subnormal is zero to the renderer, and defining the crop that way
    lets the browser (web/soar/zcrop.js) reach the same band from the same
    file.

    Cropping moves bmin.z/bmax.z onto the cloud. Cameras are placed relative
    to the domain box, so this is a different scene from the uncropped one;
    it is the scene the browser shows.
    """
    if sigma.ndim != 3:
        raise ValueError(f"crop_empty_z needs a 3D field; got {sigma.shape}.")
    z = np.asarray(z, dtype=np.float64)
    if sigma.shape[2] != z.size:
        raise ValueError(
            f"field has {sigma.shape[2]} z planes but {z.size} z coordinates.")

    # Per-plane max rather than a float16 copy of the whole field: same answer,
    # and it does not need a second array the size of the first.
    occupied = np.nanmax(sigma, axis=(0, 1)) > _FP16_NONZERO_FLOOR
    if not occupied.any():
        raise ValueError(
            "Every z plane of this field stores as zero in fp16, so it would "
            "render as empty sky and there is no occupied band to crop to. "
            "Check the units — this is what a field read as kg/kg when it is "
            "really g/kg looks like.")

    lo = int(np.argmax(occupied))
    hi = int(len(occupied) - 1 - np.argmax(occupied[::-1]))
    # The AABB needs two z coordinates to size its outer half-cells, and a
    # one-plane field is not something the march can do anything with anyway.
    # Widen upward when possible, else downward — the sole occupied plane
    # can be the topmost one. (Mirrored in web/soar/zcrop.js.)
    if hi == lo:
        if lo + 1 < len(occupied):
            hi = lo + 1
        else:
            lo = max(0, lo - 1)
    return sigma[:, :, lo:hi + 1], z[lo:hi + 1], (lo, hi)


def _renderer_for(levels: Sequence[NestedLevel], *, periodic: bool,
                  tone_mapped: bool, surface: str) -> SoarRenderer:
    """Get or build a session for these levels."""
    global _session, _session_key, _session_arrays
    nested = len(levels) > 1
    key = (tuple((id(l.sigma), l.sigma.shape, float(l.bmin[0]), float(l.bmax[0]))
                 for l in levels), periodic, nested, tone_mapped, surface)
    if _session is not None and _session_key == key:
        return _session
    renderer = SoarRenderer(periodic=periodic, nested=nested,
                            tone_map=tone_mapped, surface=surface)
    renderer.upload_volume(levels[-1].sigma)
    if nested:
        renderer.upload_nest(levels[0].sigma)
    _session, _session_key = renderer, key
    _session_arrays = tuple(l.sigma for l in levels)
    return renderer


def render_nested(
    levels: Sequence[NestedLevel],
    camera_position,
    *,
    azimuth: float,
    elevation: float,
    sun_azimuth: float,
    sun_elevation: float,
    image_size=DEFAULT_SIZE,
    fov_degrees: float = Camera.fov,
    exposure: float = DEFAULT_EXPOSURE,
    tone_map_gamma: float = DEFAULT_TONE_MAP_GAMMA,
    tone_map_white_point: float = DEFAULT_TONE_MAP_WHITE_POINT,
    contrast: float = DEFAULT_CONTRAST,
    haze: float = DEFAULT_HAZE,
    haze_height_dependent: bool = DEFAULT_HAZE_HEIGHT_DEPENDENT,
    periodic: bool = False,
    accumulate: int = STILL_ACCUMULATE_FRAMES,
    step_voxel_factor: float = STEP_VOXEL_FACTOR,
    lod: float = DEFAULT_LOD_STRENGTH,
    quality: Optional[str] = None,
    surface_origin: Optional[Sequence[float]] = None,
    surface: str = "ocean",
    city_offset: Optional[Sequence[float]] = None,
    scene_time: float = 0.0,
    return_linear: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Render one or two extinction levels, finest first, from a camera
    position in absolute metres.

    This is the entry below witness(): the caller has already turned its
    fields into NestedLevels and placed the camera. The keywords are
    witness()'s, where they are described; `image_size` and `fov_degrees`
    are its `size` and its camera's `fov`, and `step_voxel_factor` is the
    march step in units of the smallest voxel when no `quality` sets it.

    The shader carries one optional nest (`NESTED` is a compile-time switch,
    and core WebGPU cannot index textures dynamically), so more than two
    levels raises.
    """
    check_surface(surface)
    # Which district stands under the field is the tile's world offset, and
    # there is no neutral value to default to: soar's cyberpunk mode and
    # tools/night_city_harness.py each use a different one. So a city must
    # say which, and an ocean render refuses an offset it would ignore.
    if surface == "city":
        if city_offset is None:
            raise ValueError(
                "surface='city' needs city_offset, the city tile's world "
                "offset in metres (x y): it picks the district under the "
                "field. Soar's cyberpunk mode flies (8100, -44190); a soar "
                "reproduction command always writes the one it flew.")
        city_offset = [float(v) for v in city_offset]
        if len(city_offset) != 2:
            raise ValueError(
                f"city_offset needs exactly two values (x y metres); got "
                f"{city_offset!r}.")
    elif city_offset is not None:
        raise ValueError(
            f"city_offset places the night city's tile, but surface is "
            f"'{surface}'; pass surface='city' or drop the offset.")
    if scene_time != 0.0:
        # Only a surface with a clock reads it (its meta.json clock_period_s:
        # the city's steam). Over one without, --time would silently do
        # nothing, so it is an error.
        tile_meta = json.loads((SURFACE_DIRS[check_surface(surface)]
                                / "meta.json").read_text())
        if surface_clock_period_s(surface, tile_meta) is None:
            raise ValueError(
                f"the {surface} has no clock — nothing on it moves (its "
                f"meta.json clock_period_s is null); drop --time.")
    levels = list(levels)
    if not levels:
        raise ValueError("render_nested needs at least one level.")
    if len(levels) > 2:
        raise ValueError(
            f"render_nested got {len(levels)} levels; the shader supports at "
            "most two (one outer field plus one nest). NESTED is a "
            "compile-time constant because core WebGPU has no dynamic "
            "texture indexing, so extra levels cannot be bound.")

    outer = levels[-1]
    renderer = _renderer_for(levels, periodic=periodic,
                             tone_mapped=not return_linear, surface=surface)
    min_voxel = min(outer.dx)
    # A quality tier sets the step factors, the lighting method and the
    # number of passes together, all from the one preset, so the render is
    # the in-app capture at that tier.
    tier = None
    if quality is not None:
        if quality not in QUALITY_PRESETS:
            raise ValueError(
                f"unknown quality tier '{quality}'; expected one of "
                f"{sorted(QUALITY_PRESETS)}.")
        tier = QUALITY_PRESETS[quality]
        accumulate = tier["accumulate"]
    view_factor = tier["step_factor"] if tier else step_voxel_factor
    light_factor = tier["light_step_factor"] if tier else step_voxel_factor
    # surface_origin is SceneState.surface_offset_m, which says what it is.
    state_kwargs = {}
    if surface_origin is not None:
        origin = [float(v) for v in surface_origin]
        if len(origin) != 2:
            raise ValueError(
                f"surface_origin needs exactly two values (x y metres); "
                f"got {surface_origin!r}.")
        state_kwargs["surface_offset_m"] = tuple(origin)
    if surface == "city":
        state_kwargs["city_tile_offset_m"] = tuple(city_offset)
    # The tile's cell size, extent and top mip describe whichever tile the
    # renderer bound, so they come from that tile's own meta.json.
    meta = renderer.surface_meta
    state = SceneState(
        bmin=[float(v) for v in outer.bmin], bmax=[float(v) for v in outer.bmax],
        dt_view=min_voxel * view_factor, dt_light=min_voxel * light_factor,
        periodic=periodic,
        ocean_reflectance=DEFAULT_OCEAN_REFLECTANCE,
        ocean_fif_dx=surface_cell_m(surface, meta),
        ocean_tile_extent=float(meta["tile_extent_m"]),
        ocean_max_lod=int(meta["mips"]) - 1,
        surface_period_m=surface_period_m(surface, meta),
        clock_period_s=surface_clock_period_s(surface, meta),
        nested=len(levels) > 1,
        surface=surface,
        **state_kwargs,
    )
    dt = min_voxel * view_factor
    if len(levels) > 1:
        fine = levels[0]
        state.nest_bmin = [float(v) for v in fine.bmin]
        state.nest_bmax = [float(v) for v in fine.bmax]
        state.dt_view_nest = min(fine.dx) * view_factor
        state.dt_light_nest = min(fine.dx) * light_factor

    w, h = image_size
    view = ViewState(
        camera_position=[float(v) for v in camera_position],
        azimuth=azimuth, elevation=elevation, fov=fov_degrees,
        output_size=(w, h), render_size=(w, h),
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        exposure=exposure, tone_map_gamma=tone_map_gamma,
        tone_map_white_point=tone_map_white_point, contrast=contrast,
        haze=haze, haze_height_dependent=haze_height_dependent,
        # `lod` scales the app's two angles, so it is the same number as the
        # browser's slider.
        light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES * lod,
        view_step_lod_degrees=APP_VIEW_STEP_LOD_DEGREES * lod,
        light_cache=bool(tier and tier["light_cache"]),
        sky_probe=(tier["sky_probe"] if tier else True),
        scene_time=float(scene_time),
        # The roofs' clutter follows the tier, as in the browser; a render
        # with no tier gets the density the roofs were tuned at.
        city_roof_density=(CITY_ROOF_DENSITY_BY_TIER[quality] if tier
                           else 1.0),
    )
    if tier and tier["light_cache"]:
        renderer.bake_light_cache(state, view, divisor=LIGHT_CACHE_DIVISOR)
    if verbose:
        print(f"  Rendering {w}x{h}, {accumulate} accumulated passes, "
              f"{len(levels)} level(s), dt={dt:.1f} m"
              f"{', periodic' if periodic else ''}")
    t0 = time.perf_counter()
    image = renderer.render(state, view, frames=accumulate)
    if verbose:
        print(f"  Rendered in {time.perf_counter() - t0:.2f}s")
    return image


# The extinction is built in x-slabs of about this many voxels, so the
# temporaries the optics make (condensate times air density, each phase's
# term, their sum) are a slab's and not a field's.
_SLAB_VOXELS = 1 << 24


def _x_slabs(shape):
    """Slices along x covering a field of `shape`, each ~_SLAB_VOXELS."""
    step = max(1, _SLAB_VOXELS // max(1, int(np.prod(shape[1:]))))
    return [slice(i, min(i + step, shape[0])) for i in range(0, shape[0], step)]


def _field_level(field: CloudField, name: str, verbose: bool = False) -> NestedLevel:
    """Turn a loaded CloudField into a renderable NestedLevel.

    Converts condensate to extinction (dropping ice when it is negligible),
    applies EXTINCTION_MULTIPLIER, crops the empty sky above and below the
    cloud, and derives the absolute-meter AABB.

    Memory: the result is one array of the occupied z band, in the
    extinction's own dtype, and nothing field-sized is made on the way to it.
    The extinction is elementwise in x, so it is computed slab by slab
    (_x_slabs) twice: once for each plane's peak, which is all crop_empty_z
    judges the band on, and once into the band itself. Building the whole
    field's extinction first keeps five field-sized arrays alive at the
    liquid + ice sum: on a 2048 x 2048 x 495 field with liquid and ice that
    is about 58 GB over the 16.6 GB field.
    """
    multiplier = EXTINCTION_MULTIPLIER

    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < ICE_NEGLIGIBLE_G_KG:
        if verbose:
            print("  Ice water content is negligible; rendering liquid only.")
        iwc = None
    lwc = field.lwc
    # As given: the air density is computed in z's own dtype, and an upcast
    # here would move it by an ulp.
    z = np.asarray(field.z)

    def extinction(xs, zs=slice(None)):
        sigma = optical_depth.compute_extinction_field(
            lwc[xs, :, zs], z[zs], re=RE_LIQUID_UM,
            iwc=None if iwc is None else iwc[xs, :, zs], re_ice=RE_ICE_UM)
        sigma *= multiplier
        return sigma

    slabs = _x_slabs(lwc.shape)
    peaks = np.full(z.size, -np.inf)
    for xs in slabs:
        with warnings.catch_warnings():
            # An all-NaN plane in one slab is no plane's whole answer; fmax
            # skips its NaN as nanmax over the field would.
            warnings.simplefilter("ignore", RuntimeWarning)
            peaks = np.fmax(peaks, np.nanmax(extinction(xs), axis=(0, 1)))
    # Trim the empty sky above and below the cloud, and move the box with it.
    # Only z: x and y are not cropped, here or in the browser. The rule is
    # crop_empty_z's, applied to the planes' peaks (as coarsen.occupied_band).
    _, z_crop, (lo, hi) = crop_empty_z(peaks.reshape(1, 1, -1), z)
    sigma = None
    for xs in slabs:
        part = extinction(xs, slice(lo, hi + 1))
        if sigma is None:
            sigma = np.empty(lwc.shape[:2] + (hi - lo + 1,), dtype=part.dtype)
        sigma[xs] = part

    bmin, bmax = _volume_aabb(field)
    if hi - lo + 1 < z.size:
        bmin[2] = z_crop.min() - 0.5 * abs(z_crop[1] - z_crop[0])
        bmax[2] = z_crop.max() + 0.5 * abs(z_crop[-1] - z_crop[-2])
        if verbose:
            source = z.size
            print(f"  Cropped to z planes {lo}-{hi} of {source} "
                  f"({100 * (1 - (hi - lo + 1) / source):.0f}% of the "
                  "vertical held no cloud).")
    return NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name=name)


def witness(
    field: CloudField,
    camera: Optional[Camera] = None,
    *,
    nest: Optional[CloudField] = None,
    size: Tuple[int, int] = DEFAULT_SIZE,
    sun_azimuth: float = DEFAULT_SUN_AZIMUTH,
    sun_elevation: float = DEFAULT_SUN_ELEVATION,
    exposure: float = DEFAULT_EXPOSURE,
    tone_map_gamma: float = DEFAULT_TONE_MAP_GAMMA,
    tone_map_white_point: float = DEFAULT_TONE_MAP_WHITE_POINT,
    contrast: float = DEFAULT_CONTRAST,
    haze: float = DEFAULT_HAZE,
    haze_height_dependent: bool = DEFAULT_HAZE_HEIGHT_DEPENDENT,
    periodic: bool = False,
    accumulate: int = STILL_ACCUMULATE_FRAMES,
    lod: float = DEFAULT_LOD_STRENGTH,
    quality: Optional[str] = None,
    surface_origin: Optional[Sequence[float]] = None,
    surface: str = "ocean",
    city_offset: Optional[Sequence[float]] = None,
    scene_time: float = 0.0,
    return_linear: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Render a cloud field with soar's volumetric ray marcher.

    Parameters
    ----------
    field : CloudField
        Loaded cloud field (see :func:`cloudyview.load`).
    camera : Camera, optional
        Viewpoint; Camera() when omitted. Position is in relative
        coordinates, where z = -1 is the physical surface rather than the
        bottom of the data.
    nest : CloudField, optional
        A finer field to render inside `field`. It is tapered to zero at its
        edges, so it blends into the coarser field around it. The camera's
        relative coordinates still refer to `field`'s box.
    size : (width, height)
        Image size in pixels.
    sun_azimuth, sun_elevation : float
        Sun direction in degrees: a meteorological bearing (0 north, 90
        east) and the angle above the horizon.
    exposure : float
        Scales the radiance before the tone map.
    tone_map_gamma, tone_map_white_point, contrast : float
        The tone map: the display gamma, the exposed radiance that maps to
        white, and a contrast about mid-grey applied after the gamma.
    haze : float
        Aerosol amount, the same number as soar's slider. Sets the aerial
        perspective, the sky's horizon whitening, the circumsolar lobe and
        the haze over the sea together.
    haze_height_dependent : bool
        Let the haze thin with height instead of being uniform.
    periodic : bool
        Wrap the domain in x and y, as soar does for LES fields. Needs the
        sun above the horizon.
    accumulate : int
        Passes averaged into the image; more is less noise. A `quality`
        tier sets it instead.
    lod : float
        Angular level of detail: multiplies the angles the far field's march
        step grows at. Smaller is finer and slower, never coarser.
    quality : str, optional
        A soar quality tier (soar/tiers.py QUALITY_PRESETS): render as the
        app's capture at that tier, with its step factors, lighting method
        and number of passes.
    surface_origin : (x, y), optional
        The surface offset of the flight being reproduced, in world metres
        (the shader derives the water pattern and the city's frame from
        world minus this, mod the surface period). Omitted
        means the scene's static offset, which is every fresh render; soar's
        reproduction commands write a flight's own offset here.
    surface : str
        What the world stands on: "ocean", soar's day ground, or "city", the
        night city of soar's cyberpunk mode, under which the sun angles are
        the moon's.
    city_offset : (x, y), optional
        The city tile's world offset in metres, which picks the district
        standing under the field. Required with surface="city" and refused
        otherwise; soar's cyberpunk mode flies (8100, -44190).
    scene_time : float
        The scene clock in seconds: where the city's
        steam is in its motion. Refused, unless 0, over a surface with no
        clock (the sea); soar's reproduction commands write the clock the
        still was taken at.
    return_linear : bool
        Compile the tone map out and return unbounded linear HDR radiance
        instead of an encoded image. This is what soar's auto-exposure
        meters.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    ndarray
        (height, width, 3) float64 in [0, 1], tone mapped; or unbounded
        linear radiance when ``return_linear`` is set.
    """
    if camera is None:
        camera = Camera()

    # Both levels are built by the same rule. Finest first.
    outer = _field_level(field, "single" if nest is None else "outer",
                         verbose=verbose)
    levels = [outer]
    if nest is not None:
        levels = [_field_level(nest, "nest", verbose=verbose), outer]
    # Relative camera coordinates resolve against the OUTER box, as they do
    # in the browser.
    position = camera_world_origin(camera.position, outer.bmin, outer.bmax)

    if verbose:
        bmin, bmax = outer.bmin, outer.bmax
        print(f"  Grid: {outer.sigma.shape}, domain "
              f"{(bmax[0]-bmin[0])/1e3:.1f} x {(bmax[1]-bmin[1])/1e3:.1f} x "
              f"{bmax[2]/1e3:.1f} km")
        if nest is not None:
            print(f"  Nest grid: {levels[0].sigma.shape}")
        print(f"  Camera at {np.round(position, 1)} m, azimuth "
              f"{camera.azimuth:.1f}, elevation {camera.elevation:.1f}, "
              f"fov {camera.fov:.1f} (horizontal)")
        print(f"  Sun: azimuth {sun_azimuth:.1f}, elevation {sun_elevation:.1f}")

    return render_nested(
        levels, position,
        azimuth=camera.azimuth, elevation=camera.elevation,
        fov_degrees=camera.fov, image_size=tuple(size),
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        exposure=exposure, tone_map_gamma=tone_map_gamma,
        tone_map_white_point=tone_map_white_point, contrast=contrast,
        haze=haze, haze_height_dependent=haze_height_dependent, lod=lod,
        periodic=periodic, accumulate=accumulate, quality=quality,
        surface_origin=surface_origin, surface=surface,
        city_offset=city_offset, scene_time=scene_time,
        return_linear=return_linear, verbose=verbose)


def main(filename: str, output: str = None, *, nest_group: str = None,
         custom_size: tuple = None, **options) -> None:
    """What the `witness` command does with its parsed arguments: load the
    field, render it, and save `witness_<input-stem>.png` in `output`.

    `options` are the other arguments, under the names the library uses:
    camera_position, camera_azimuth, camera_elevation and camera_fov for the
    Camera; load()'s keywords (cli.LOAD_KEYWORDS); and witness()'s keywords.
    One that is None was not given on the command line and is left out, so
    that the library's default applies.

    `nest_group` names a NetCDF group of the SAME file holding a finer field
    to render as the nest. The group flags (--group, --liquid-water-group and
    the others) describe the outer field only; the nest is read from
    `nest_group` with the same variable, coordinate and dimension names.

    Exits with status 1 on any error.
    """
    print(f"CloudyView Witness: Loading {filename}")
    start_time = time.perf_counter()

    options = {k: v for k, v in options.items() if v is not None}
    load_kwargs = {k: options.pop(k) for k in LOAD_KEYWORDS if k in options}
    camera_kwargs = {k: options.pop("camera_" + k)
                     for k in ("position", "azimuth", "elevation", "fov")
                     if "camera_" + k in options}
    if custom_size:
        options["size"] = tuple(custom_size)
    # What is left in `options` are witness() keywords.

    try:
        if nest_group and "ice" in load_kwargs:
            # The nest loads from its own group of the MAIN file; a split
            # ice file describes the outer grid and has no counterpart for
            # the nest, which would come out silently ice-free.
            raise ValueError(
                "--ice-file and --nest-group are not supported together: "
                "the nest field has no separate ice file to pair with, so "
                "it would render ice-free without saying so. Merge the ice "
                "into the file's groups, or drop one of the flags.")
        if nest_group and "grid_spacing" in load_kwargs:
            # A nest is a finer grid inside the outer one: the outer spacing
            # does not describe it, and with no meters of its own there is
            # nothing to place it inside the outer box by.
            raise ValueError(
                "--dx/--dy/--dz and --nest-group are not supported together: "
                "the stated spacing describes the outer grid only, and a nest "
                "without meters cannot be placed inside it.")
        if nest_group and load_kwargs.get("coarsen", 1) != 1:
            # The nest is the finer grid inside the outer one, there to be
            # seen at its own resolution; a coarsened outer field under it is
            # a pairing no soar view makes.
            raise ValueError(
                "--coarsen and --nest-group are not supported together: "
                "the coarse grid is the outer field's only, and a nest "
                "inside it at its own resolution is not a view soar has.")

        field = _load_field(filename, **load_kwargs)
        nest = None
        if nest_group:
            print(f"  Nest: loading group {nest_group}")
            name_kwargs = {k: v for k, v in load_kwargs.items()
                           if k not in ("dataset_group", "liquid_water_group",
                                        "ice_water_group", "coords_group")}
            # If the nest will not load, stop: an image of the outer field
            # alone would look plausible and be the wrong picture.
            try:
                nest = _load_field(filename, dataset_group=nest_group,
                                   **name_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load the nest field from group '{nest_group}' "
                    f"of {filename}: {exc}") from exc

        image = witness(field, camera=Camera(**camera_kwargs), nest=nest,
                        verbose=True, **options)

        output_dir = Path(output) if output else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"witness_{Path(filename).stem}.png"

        from .image import save_image
        save_image(image, str(output_file))
        print(f"  Saved: {output_file}")
        print(f"  Complete ({time.perf_counter() - start_time:.1f}s)")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cli():
    """Command-line interface for witness.py"""
    parser = argparse.ArgumentParser(
        prog="witness",
        description="Interactive-style volumetric cloud rendering with fast ray marching.",
        formatter_class=CloudyViewHelpFormatter,
        epilog=dedent(
            f"""
            What `witness` does:
              1. Loads a 3D cloud field from NetCDF.
              2. Converts liquid water and optional ice water to extinction.
              3. Ray-marches the volume with single- and multi-scattering approximations.
              4. Writes one tone-mapped PNG named `witness_<input-stem>.png`.

            Input requirements:
              - A liquid water array is required; an ice water array is optional.
              - The selected field must become 3D after dropping any single time dimension.
              - Physical x/y/z coordinates are required to compute grid spacing and aspect ratio.

            Camera and sun conventions:
              - Coordinates are meteorological: +x east, +y north, +z up.
              - Camera position uses relative coordinates. In x and y, +/-1 reaches
                the domain edge. In z, -1 is the physical surface (z=0) and +1 is
                the top of the data domain, so lifted domains (e.g. data starting
                above ground) keep their real altitude.
              - Azimuth is a meteorological bearing: 0 north, 90 east, 180 south, 270 west.
              - Elevation is degrees above the horizon.

            Quality:
              `--quality` renders as the in-app capture at that soar tier
              (minimal/low/medium/high/max): its step factors, lighting method
              and sample count, all from the one preset the browser uses.
              `--size WIDTH HEIGHT` sets the image size (default {DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}).

            Image controls (the same knobs the browser app exposes):
              `--gamma`, `--white-point`, `--contrast` and `--haze` set the tone
              map and aerosol amount; omit one and the library default stands.
              `--periodic` wraps the domain in x and y for LES fields.

            Nesting:
              `--nest-group GROUP` renders a finer field from another NetCDF group
              of the SAME file inside the outer domain. The group-specific flags
              (`--liquid-water-group` and friends) describe the outer field only;
              the nest inherits the variable and coordinate NAME overrides.
              Relative camera coordinates resolve against the outer domain.

            Dependencies:
              `witness --help` works without a GPU. Rendering needs one: this drives
              the same WGSL shader the browser app runs, through wgpu, so there is a
              single renderer core rather than a CPU copy that drifts from it.

            {DATA_SELECTION_HELP}

            Examples:
              witness cloud.nc
              witness cloud.nc --quality max --output renders
              witness cloud.nc --size 1200 800 --camera-position 0 -0.9 -0.99 --camera-azimuth 0 --camera-elevation 35
              witness cloud.nc --group /physics/clouds --liquid-water-var QCLOUD --ice-water-var QICE
              witness cloud.nc --nest-group /nest --white-point 15 --gamma 1.66 --haze 1.0 --periodic
              witness custom.nc --liquid-water-group /state/liquid --ice-water-group /state/ice --coords-group /grid --x-dim ni --y-dim nj --z-dim nk --x-coord xh --y-coord yh --z-coord zh
            """
        ),
    )
    parser.add_argument("filename",
                        help="NetCDF file with cloud data")
    parser.add_argument("--output", "-o",
                        help="Output directory (default: current directory)")
    parser.add_argument("--camera-position", type=float, nargs=3,
                        metavar=('X', 'Y', 'Z'),
                        help="Camera position in relative coords (default: "
                             f"{' '.join(f'{v:g}' for v in Camera.position)})")
    parser.add_argument("--camera-azimuth", type=float,
                        help=f"Camera azimuth in degrees (default: {Camera.azimuth:g}). 0=North, 90=East, 180=South, 270=West")
    parser.add_argument("--camera-elevation", type=float,
                        help=f"Camera elevation in degrees (default: {Camera.elevation:g})")
    parser.add_argument("--fov", type=float,
                        help=f"Camera horizontal field of view in degrees (default: {Camera.fov:g})")
    parser.add_argument("--sun-azimuth", type=float,
                        help=f"Sun azimuth in degrees (default: {DEFAULT_SUN_AZIMUTH:g}). 0=North, 90=East, 180=South, 270=West")
    parser.add_argument("--sun-elevation", type=float,
                        help=f"Sun elevation in degrees (default: {DEFAULT_SUN_ELEVATION:g})")
    parser.add_argument("--size", type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        help=f"Image size in pixels (default: {DEFAULT_SIZE[0]} "
                             f"{DEFAULT_SIZE[1]})")
    parser.add_argument("--exposure", type=float,
                        help="Tone-map exposure (linear pre-scale; default "
                             f"{DEFAULT_EXPOSURE}). "
                             "Soar's auto-exposure writes its metered value here "
                             "so a terminal render reproduces the flown frame")
    parser.add_argument("--gamma", type=float,
                        help=f"Tone-map gamma (default: {DEFAULT_TONE_MAP_GAMMA})")
    parser.add_argument("--white-point", type=float,
                        help="Extended-Reinhard white point: the exposed radiance "
                             f"mapping to 1.0 (default: {DEFAULT_TONE_MAP_WHITE_POINT})")
    parser.add_argument("--contrast", type=float,
                        help="Display contrast about mid-grey, applied after the "
                             f"gamma encode (default: {DEFAULT_CONTRAST})")
    parser.add_argument("--haze", type=float,
                        help="Aerosol amount, as soar's haze slider: "
                             f"{HAZE_MIN:.2f} (clearest) to {HAZE_MAX} "
                             f"(default: {DEFAULT_HAZE})")
    parser.add_argument("--haze-height-dependent",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Let the haze thin with height on a "
                             f"{int(AERIAL_SCALE_HEIGHT_M)} m scale height. Off "
                             "by default: uniform haze puts the sea-level "
                             "extinction at every altitude, which is "
                             "unphysical and bounds how far every ray marches")
    parser.add_argument("--lod", type=float,
                        help="Angular level of detail: multiplies the step "
                             "angles the far field is marched at "
                             f"({APP_VIEW_STEP_LOD_DEGREES} deg view, "
                             f"{APP_LIGHT_MARCH_LOD_DEGREES} deg sun). The step "
                             "floor grows as t*tan(theta), so smaller is finer "
                             "and slower and never coarser. Soar writes its "
                             f"flown value here (default: {DEFAULT_LOD_STRENGTH})")
    parser.add_argument("--quality", choices=sorted(QUALITY_PRESETS),
                        help="Render as the in-app capture at this soar "
                             "quality tier: parked (High) sampling, the "
                             "tier's lighting method (sun-tau cache, sky "
                             "probe) and its parked sample count, all from "
                             "the one preset")
    parser.add_argument("--surface-origin", type=float, nargs=2,
                        metavar=('X', 'Y'),
                        help="The surface offset of the flight being "
                             "reproduced, in metres. "
                             "Omitted means the scene's static offset, "
                             "which is every fresh render; a soar "
                             "reproduction command writes the flown "
                             "offset so the water phase (and over a "
                             "city, the streets) land where that flight "
                             "rendered them")
    parser.add_argument("--surface", choices=SURFACES,
                        help="What the world stands on: the ocean "
                             "(default), soar's day ground, or the night "
                             "city of cyberpunk "
                             "mode, where the sun angles are the moon's "
                             "(needs --city-offset). Soar writes the flown "
                             "one here")
    parser.add_argument("--city-offset", type=float, nargs=2,
                        metavar=('X', 'Y'),
                        help="With --surface city: the city tile's world "
                             "offset in metres, which picks the district "
                             "under the field. Cyberpunk mode flies "
                             "8100 -44190; soar writes the flown one here")
    parser.add_argument("--time", type=float, metavar="SECONDS",
                        dest="scene_time",
                        help="The scene clock: where the city's steam is in "
                             "its motion (default 0; the sea has none). Soar writes the clock the still was "
                             "taken at")
    parser.add_argument("--periodic", action="store_true",
                        help="Wrap the domain in x and y, as soar does for LES fields")
    parser.add_argument("--nest-group", metavar="GROUP",
                        help="NetCDF group in the same file holding a finer field "
                             "to render as a nest inside the outer domain")
    add_dataset_selection_arguments(parser)

    args = parser.parse_args()
    main(args.filename, args.output,
         nest_group=args.nest_group,
         custom_size=args.size,
         camera_position=args.camera_position,
         camera_azimuth=args.camera_azimuth,
         camera_elevation=args.camera_elevation,
         camera_fov=args.fov,
         sun_azimuth=args.sun_azimuth,
         sun_elevation=args.sun_elevation,
         exposure=args.exposure,
         tone_map_gamma=args.gamma,
         tone_map_white_point=args.white_point,
         contrast=args.contrast,
         haze=args.haze,
         haze_height_dependent=args.haze_height_dependent,
         lod=args.lod,
         quality=args.quality,
         surface_origin=args.surface_origin,
         surface=args.surface,
         city_offset=args.city_offset,
         scene_time=args.scene_time,
         periodic=args.periodic,
         **dataset_selection_kwargs(args))


if __name__ == "__main__":
    cli()
