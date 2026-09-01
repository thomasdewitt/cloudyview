#!/usr/bin/env python
"""witness: volumetric cloud rendering, one renderer core.

This used to be a numba kernel that reimplemented what web/soar/raymarch.wgsl
does. Two implementations of one look diverge, and they did — periodic
domains and distance LOD arrived in the shader and never came back here, so
`witness` could not render what soar renders. The kernel is gone. This module
now drives that same shader through wgpu (see cloudyview/soar_host.py), so
there is exactly one definition of the look and `witness` is a wrapper that
prepares a field and a camera for it.

Consequences worth knowing:
  * Rendering needs a GPU. There is no CPU path any more, deliberately.
  * The default tone-map gamma is soar's 2.66 rather than the old 1.4, so
    images are lighter in the far field than the numba renderer produced.
    Pass tone_map_gamma=TONE_MAP_GAMMA_WITNESS for the old encode.
  * Periodic domains and distance LOD are available here for the first time.

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional, Sequence, Tuple

import numpy as np

from . import optical_depth, config
from .camera import Camera
from .cli_utils import (
    CloudyViewHelpFormatter,
    DATA_SELECTION_HELP,
    add_dataset_selection_arguments,
    dataset_selection_kwargs,
)
from .cloudfield import CloudField, load as _load_field
from .look import AERIAL_SCALE_HEIGHT_M, DEFAULT_HAZE
from .soar_host import (
    APP_LIGHT_MARCH_LOD_DEGREES,
    APP_VIEW_STEP_LOD_DEGREES,
    DEFAULT_HAZE_HEIGHT_DEPENDENT,
    DEFAULT_LOD_STRENGTH,
    DEFAULT_TONE_MAP_GAMMA,
    DEFAULT_TONE_MAP_WHITE_POINT,
    DEFAULT_CONTRAST,
    STILL_ACCUMULATE_FRAMES,
    STEP_VOXEL_FACTOR,
    TONE_MAP_GAMMA_WITNESS,
    SceneState,
    SoarRenderer,
    ViewState,
    camera_world_origin,
)

logger = logging.getLogger(__name__)

# TONE_MAP_GAMMA_WITNESS is re-exported: witness's own default is soar's 2.66
# now, and this is how a caller asks for the pre-2026-08 encode.
__all__ = ["witness", "render_nested", "tone_map", "NestedLevel",
           "crop_empty_z",
           "TONE_MAP_GAMMA_WITNESS", "QUALITY_PRESETS", "main", "cli"]

# Effective radii the extinction conversion assumes, mirrored in
# web/soar/constants.js so the browser derives the same sigma from a file.
RE_LIQUID_UM = 10.0
RE_ICE_UM = 30.0
ICE_NEGLIGIBLE_G_KG = 1e-6

OCEAN_REFLECTANCE = (0.0020, 0.0045, 0.0126)

# ============================================================================
# Level descriptor (public — also used by render_nested callers)
# ============================================================================

@dataclass
class NestedLevel:
    """One refinement level: extinction field plus its absolute-meter AABB.

    sigma is m^-1, already scaled by the caller's extinction_multiplier.
    bmin/bmax are absolute world meters.
    """
    sigma: np.ndarray        # (nx, ny, nz) float64, m^-1
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


# ============================================================================
# Tone mapping
# ============================================================================

def tone_map(image, exposure=4.0, gamma=1.4):
    """Reinhard plus gamma.

    Kept because callers import it and because behold's images are compared
    on this scale. Note the shader applies exactly this inside fs_main, so a
    render that came back through SoarRenderer is already mapped — running
    this over it again would encode twice.
    """
    exposed = image * exposure
    tone_mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)


# ============================================================================
# The renderer, and the session that owns a GPU
# ============================================================================

# One field is usually rendered many times — prebake walks hundreds of camera
# positions through a single volume. Creating a device and re-uploading a
# several-hundred-megabyte texture per call would dominate everything else,
# so the last session is kept and reused when the field has not changed.
_session = None
_session_key = None
# Strong references to the arrays the key identifies by id(): an id is only
# unique while its object lives, so a cache that lets the arrays die can be
# hit by a NEW field that happens to reuse a freed id — and serve it the
# previous field's texture.
_session_arrays = None


def _volume_aabb(field: CloudField):
    """Absolute-meter AABB with half-cell padding.

    tools/export_web_assets.py duplicates this rule for the browser demo;
    the two must agree or a baked demo lands in a different place from the
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

    Emptiness is judged on the value AS STORED, not on the float64 sigma. The
    texture is r16float in both hosts, so a sigma below the smallest fp16
    subnormal is zero as far as any renderer will ever know, and defining the
    crop that way is what lets web/soar/ingest/worker.js reach the same band
    from the same file. The two hosts have silently disagreed about texture
    construction once already (tests/test_soar_texture_parity.py exists
    because of it), so this rule lives in one sentence in two places on
    purpose.

    Cropping moves bmin.z/bmax.z onto the cloud, which is a different scene
    from the uncropped one — cameras place relative to the domain box. That is
    intended: it is what the browser does, so it is what a user sees.
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
                  tone_mapped: bool) -> SoarRenderer:
    """Get or build a session for these levels."""
    global _session, _session_key, _session_arrays
    nested = len(levels) > 1
    key = (tuple((id(l.sigma), l.sigma.shape, float(l.bmin[0]), float(l.bmax[0]))
                 for l in levels), periodic, nested, tone_mapped)
    if _session is not None and _session_key == key:
        return _session
    renderer = SoarRenderer(periodic=periodic, nested=nested,
                            tone_map=tone_mapped)
    # The bare field, both levels. What the edges do — taper into zero, or
    # wrap — is the shader's job now, not a border baked into the upload.
    renderer.upload_volume(levels[-1].sigma)
    if nested:
        renderer.upload_nest(levels[0].sigma)
    _session, _session_key = renderer, key
    _session_arrays = tuple(l.sigma for l in levels)
    return renderer


# Soar's per-tier capture recipe, mirrored from web/soar/constants.js
# (QUALITY_PRESETS + PARKED_ACCUM_FRAMES_BY_TIER, 2026-08-20): flight step
# factors, the lighting method, and the parked sample count that is also a
# capture's spp. --quality applies one of these wholesale so a CLI render
# matches the in-app capture in all ways. Keep in step with the browser.
QUALITY_PRESETS = {
    "max":     {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": False, "sky_probe": True,  "accumulate": 32},
    "high":    {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True,  "sky_probe": True,  "accumulate": 32},
    "medium":  {"step_factor": 2.5, "light_step_factor": 4.0,
                "light_cache": True,  "sky_probe": False, "accumulate": 24},
    "low":     {"step_factor": 3.0, "light_step_factor": 8.0,
                "light_cache": True,  "sky_probe": False, "accumulate": 16},
    "minimal": {"step_factor": 4.0, "light_step_factor": 12.0,
                "light_cache": True,  "sky_probe": False, "accumulate": 8},
}
LIGHT_CACHE_DIVISOR = 2       # the one cache resolution; /1 and /4 retired


def render_nested(
    levels: Sequence[NestedLevel],
    camera_position,
    *,
    azimuth: float,
    elevation: float,
    sun_azimuth: float,
    sun_elevation: float,
    image_size=(600, 400),
    fov_degrees: float = 100.0,
    exposure: float = 4.0,
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
    return_linear: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Render one or two nested levels, finest first.

    The shader carries exactly one optional nest — `NESTED` is a compile-time
    constant and core WebGPU has no dynamic texture indexing — so more than
    two levels raises rather than silently dropping the extra ones.

    Camera and sun are given as meteorological angles; the basis is derived
    in one place (soar_host.camera_basis) so it cannot disagree with the
    browser's.

    With return_linear=True the shader's tone map is compiled out and the
    linear HDR radiance comes back unbounded.
    """
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
                             tone_mapped=not return_linear)
    min_voxel = min(outer.dx)
    # --quality is the whole recipe or none of it: step factors, lighting
    # method and sample count all come from the one preset, so this render
    # is the in-app capture at that tier, made from a terminal.
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
    state = SceneState(
        bmin=[float(v) for v in outer.bmin], bmax=[float(v) for v in outer.bmax],
        dt_view=min_voxel * view_factor, dt_light=min_voxel * light_factor,
        periodic=periodic,
        ocean_reflectance=OCEAN_REFLECTANCE,
        nested=len(levels) > 1,
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
        # Angular LOD: the step floor grows as t*tan(theta), so this only ever
        # buys back far-field steps. Scaled from the app's constants so the
        # browser's slider and this argument are the same number.
        light_march_lod_degrees=APP_LIGHT_MARCH_LOD_DEGREES * lod,
        view_step_lod_degrees=APP_VIEW_STEP_LOD_DEGREES * lod,
        light_cache=bool(tier and tier["light_cache"]),
        sky_probe=(tier["sky_probe"] if tier else True),
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


def _field_level(field: CloudField, name: str, verbose: bool = False) -> NestedLevel:
    """Turn a loaded CloudField into a renderable NestedLevel.

    Converts condensate to extinction (dropping ice when it is negligible),
    applies the config's extinction_multiplier, and derives the absolute-meter
    AABB. Shared by the single-field and nested paths so both levels of a nest
    are built by exactly the same rule.
    """
    rendering = config.get_witness_config()['rendering']

    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < ICE_NEGLIGIBLE_G_KG:
        if verbose:
            print("  Ice water content is negligible; rendering liquid only.")
        iwc = None

    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=RE_LIQUID_UM, iwc=iwc, re_ice=RE_ICE_UM)
    sigma = np.ascontiguousarray(
        sigma * rendering.get('extinction_multiplier', 1.0), dtype=np.float64)

    bmin, bmax = _volume_aabb(field)
    # Trim the empty sky above and below the cloud, and move the box with it.
    # Only z: x and y are cropped by nobody, here or in the browser.
    sigma, z_crop, (lo, hi) = crop_empty_z(sigma, field.z)
    sigma = np.ascontiguousarray(sigma)
    if hi - lo + 1 < np.asarray(field.z).size:
        bmin[2] = z_crop.min() - 0.5 * abs(z_crop[1] - z_crop[0])
        bmax[2] = z_crop.max() + 0.5 * abs(z_crop[-1] - z_crop[-2])
        if verbose:
            source = np.asarray(field.z).size
            print(f"  Cropped to z planes {lo}-{hi} of {source} "
                  f"({100 * (1 - (hi - lo + 1) / source):.0f}% of the "
                  "vertical held no cloud).")
    return NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name=name)


def witness(
    field: CloudField,
    camera: Optional[Camera] = None,
    *,
    size: Optional[Tuple[int, int]] = None,
    sun_azimuth: Optional[float] = None,
    sun_elevation: Optional[float] = None,
    exposure: Optional[float] = None,
    tone_map_gamma: float = DEFAULT_TONE_MAP_GAMMA,
    tone_map_white_point: float = DEFAULT_TONE_MAP_WHITE_POINT,
    contrast: float = DEFAULT_CONTRAST,
    haze: Optional[float] = None,
    haze_height_dependent: bool = DEFAULT_HAZE_HEIGHT_DEPENDENT,
    periodic: bool = False,
    accumulate: int = STILL_ACCUMULATE_FRAMES,
    lod: float = DEFAULT_LOD_STRENGTH,
    quality: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Render a cloud field with soar's volumetric ray marcher.

    Parameters
    ----------
    field : CloudField
        Loaded cloud field (see :func:`cloudyview.load`).
    camera : Camera, optional
        Viewpoint; defaults to the standard witness camera. Position is in
        relative coordinates, where z = -1 is the physical surface rather
        than the bottom of the data.
    size : (width, height), optional
        Image size in pixels (default from config: 600x400).
    sun_azimuth, sun_elevation : float, optional
        Sun direction in degrees (met bearing / above horizon);
        defaults from config (20 / 55).
    exposure : float, optional
        Tone-mapping exposure (default from config: 4.0).
    tone_map_gamma : float, optional
        Display encode. Defaults to soar's 2.66; pass
        TONE_MAP_GAMMA_WITNESS (1.4) for the pre-2026-08 look.
    haze : float, optional
        Aerosol amount in [0, 1], the same slider soar shows. Drives the
        aerial perspective, the sky's horizon whitening, the circumsolar
        lobe and the haze over the sea together. Default 0.35, the tuned
        look.
    periodic : bool, optional
        Wrap the domain in x and y, as soar does for LES fields. Requires
        the sun above the horizon.
    accumulate : int, optional
        Accumulated passes; more is less noise (default 64).
    verbose : bool, optional
        Print diagnostics.

    Returns
    -------
    ndarray
        (height, width, 3) float64 in [0, 1], tone mapped.
    """
    witness_config = config.get_witness_config()
    rendering = witness_config['rendering']

    if camera is None:
        camera = Camera()
    if size is None:
        size = (rendering['width'], rendering['height'])
    if sun_azimuth is None:
        sun_azimuth = witness_config['sun']['azimuth']
    if sun_elevation is None:
        sun_elevation = witness_config['sun']['elevation']
    if exposure is None:
        exposure = rendering['exposure']
    if haze is None:
        haze = DEFAULT_HAZE

    level = _field_level(field, "single", verbose=verbose)
    sigma, bmin, bmax = level.sigma, level.bmin, level.bmax
    position = camera_world_origin(camera.position, bmin, bmax)

    if verbose:
        print(f"  Grid: {sigma.shape}, domain "
              f"{(bmax[0]-bmin[0])/1e3:.1f} x {(bmax[1]-bmin[1])/1e3:.1f} x "
              f"{bmax[2]/1e3:.1f} km")
        print(f"  Camera at {np.round(position, 1)} m, azimuth "
              f"{camera.azimuth:.1f}, elevation {camera.elevation:.1f}, "
              f"fov {camera.fov:.1f} (horizontal)")
        print(f"  Sun: azimuth {sun_azimuth:.1f}, elevation {sun_elevation:.1f}")

    return render_nested(
        [level], position,
        azimuth=camera.azimuth, elevation=camera.elevation,
        fov_degrees=camera.fov, image_size=tuple(size),
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        exposure=exposure, tone_map_gamma=tone_map_gamma,
        tone_map_white_point=tone_map_white_point, contrast=contrast,
        haze=haze, haze_height_dependent=haze_height_dependent, lod=lod,
        periodic=periodic, accumulate=accumulate, quality=quality,
        verbose=verbose)


def _render_with_nest(filename: str, outer_field: CloudField, camera: Camera,
                      nest_group: str, *, size, sun_azimuth, sun_elevation,
                      periodic: bool, look_kwargs: dict,
                      load_kwargs: dict) -> np.ndarray:
    """Render `filename` with a finer field from `nest_group` as the nest.

    The nest is a second field in the SAME file, read from its own NetCDF
    group. The group-specific overrides (--liquid-water-group and friends)
    describe the outer field only, so the nest is loaded with `nest_group` as
    its dataset group and inherits just the variable/coordinate NAME
    overrides. If it will not load, this raises: a quietly single-field image
    would look plausible and be the wrong picture.
    """
    print(f"  Nest: loading group {nest_group}")
    try:
        nest_field = _load_field(filename, dataset_group=nest_group,
                                 **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the nest field from group '{nest_group}' of "
            f"{filename}: {exc}") from exc

    witness_config = config.get_witness_config()
    rendering = witness_config['rendering']
    if size is None:
        size = (rendering['width'], rendering['height'])
    if sun_azimuth is None:
        sun_azimuth = witness_config['sun']['azimuth']
    if sun_elevation is None:
        sun_elevation = witness_config['sun']['elevation']

    outer = _field_level(outer_field, "outer", verbose=True)
    nest = _field_level(nest_field, nest_group, verbose=True)
    # Relative camera coordinates resolve against the OUTER box, as they do
    # in the browser, so the same numbers frame the same view either side.
    position = camera_world_origin(camera.position, outer.bmin, outer.bmax)

    print(f"  Outer grid: {outer.sigma.shape}, nest grid: {nest.sigma.shape}")
    print(f"  Camera at {np.round(position, 1)} m, azimuth "
          f"{camera.azimuth:.1f}, elevation {camera.elevation:.1f}, "
          f"fov {camera.fov:.1f} (horizontal)")
    print(f"  Sun: azimuth {sun_azimuth:.1f}, elevation {sun_elevation:.1f}")

    # `exposure` may arrive in look_kwargs (soar's auto-exposure writes it
    # into the reproduction command); pop it off a copy so it does not
    # collide with the explicit kwarg below.
    look_kwargs = dict(look_kwargs)
    exposure = look_kwargs.pop('exposure', rendering['exposure'])

    return render_nested(
        [nest, outer], position,          # finest first
        azimuth=camera.azimuth, elevation=camera.elevation,
        fov_degrees=camera.fov, image_size=tuple(size),
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        exposure=exposure,
        periodic=periodic, verbose=True,
        **look_kwargs)


def main(filename: str, output: str = None,
         camera_position: list = None, camera_azimuth: float = None,
         camera_elevation: float = None, camera_fov: float = None,
         sun_azimuth: float = None, sun_elevation: float = None,
         custom_size: tuple = None,
         exposure: float = None,
         tone_map_gamma: float = None,
         tone_map_white_point: float = None,
         contrast: float = None,
         haze: float = None,
         haze_height_dependent: bool = None,
         lod: float = None,
         quality: str = None,
         periodic: bool = False,
         nest_group: str = None,
         liquid_water_var: str = None,
         ice_water_var: str = None,
         dataset_group: str = None,
         liquid_water_group: str = None,
         ice_water_group: str = None,
         coords_group: str = None,
         x_coord_name: str = None,
         y_coord_name: str = None,
         z_coord_name: str = None,
         x_dim: str = None,
         y_dim: str = None,
         z_dim: str = None,
         timestep: int = None,
         fallback_units: str = None,
         fallback_ice_units: str = None,
         fallback_coord_units: str = None,
         ice: str = None,
         no_ice: bool = False) -> None:
    """CLI wrapper around :func:`witness`: load, render, save a PNG."""
    print(f"CloudyView Witness: Loading {filename}")
    start_time = time.perf_counter()

    witness_config = config.get_witness_config()
    cam_config = witness_config['camera']

    if camera_position is not None:
        cam_config['position'] = list(camera_position)
    if camera_azimuth is not None:
        cam_config['azimuth'] = camera_azimuth
    if camera_elevation is not None:
        cam_config['elevation'] = camera_elevation
    if camera_fov is not None:
        cam_config['fov'] = camera_fov

    # None means "the library default"; only forward what was actually asked
    # for, so cloudyview.witness / soar_host stay the single source of truth.
    look_kwargs = {}
    if exposure is not None:
        look_kwargs['exposure'] = exposure
    if tone_map_gamma is not None:
        look_kwargs['tone_map_gamma'] = tone_map_gamma
    if tone_map_white_point is not None:
        look_kwargs['tone_map_white_point'] = tone_map_white_point
    if contrast is not None:
        look_kwargs['contrast'] = contrast
    if haze is not None:
        look_kwargs['haze'] = haze
    if lod is not None:
        look_kwargs['lod'] = lod
    if haze_height_dependent is not None:
        look_kwargs['haze_height_dependent'] = haze_height_dependent
    if quality is not None:
        look_kwargs['quality'] = quality

    try:
        if nest_group and ice:
            # The nest loads from its own group of the MAIN file; a split
            # ice file describes the outer grid and has no counterpart for
            # the nest, which would come out silently ice-free.
            raise ValueError(
                "--ice-file and --nest-group are not supported together: "
                "the nest field has no separate ice file to pair with, so "
                "it would render ice-free without saying so. Merge the ice "
                "into the file's groups, or drop one of the flags.")
        field = _load_field(
            filename,
            ice=ice,
            liquid_water_var=liquid_water_var,
            ice_water_var=ice_water_var,
            dataset_group=dataset_group,
            liquid_water_group=liquid_water_group,
            ice_water_group=ice_water_group,
            coords_group=coords_group,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            z_coord_name=z_coord_name,
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            timestep=timestep,
            fallback_units=fallback_units,
            fallback_ice_units=fallback_ice_units,
            fallback_coord_units=fallback_coord_units,
            no_ice=no_ice,
        )

        camera = Camera(
            position=cam_config['position'],
            azimuth=cam_config['azimuth'],
            elevation=cam_config['elevation'],
            fov=cam_config['fov'],
        )

        if nest_group:
            image_tm = _render_with_nest(
                filename, field, camera, nest_group,
                size=tuple(custom_size) if custom_size else None,
                sun_azimuth=sun_azimuth,
                sun_elevation=sun_elevation,
                periodic=periodic,
                look_kwargs=look_kwargs,
                load_kwargs=dict(
                    liquid_water_var=liquid_water_var,
                    ice_water_var=ice_water_var,
                    x_coord_name=x_coord_name,
                    y_coord_name=y_coord_name,
                    z_coord_name=z_coord_name,
                    x_dim=x_dim,
                    y_dim=y_dim,
                    z_dim=z_dim,
                    timestep=timestep,
                    fallback_units=fallback_units,
                    fallback_ice_units=fallback_ice_units,
                    fallback_coord_units=fallback_coord_units,
                    no_ice=no_ice,
                ),
            )
        else:
            image_tm = witness(
                field,
                camera=camera,
                size=tuple(custom_size) if custom_size else None,
                sun_azimuth=sun_azimuth,
                sun_elevation=sun_elevation,
                periodic=periodic,
                verbose=True,
                **look_kwargs,
            )

        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        dataset_name = Path(filename).stem
        output_file = output_dir / f"witness_{dataset_name}.png"

        from .basic_render import save_image
        save_image(image_tm, str(output_file))
        print(f"  Saved: {output_file}")

        elapsed = time.perf_counter() - start_time
        print(f"  Complete ({elapsed:.1f}s)")

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
              `--size WIDTH HEIGHT` sets the image size (default 600x400).

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
                        help="Output directory")
    parser.add_argument("--camera-position", type=float, nargs=3,
                        metavar=('X', 'Y', 'Z'),
                        help="Camera position in relative coords (default: 0 0 -0.999)")
    parser.add_argument("--camera-azimuth", type=float,
                        help="Camera azimuth in degrees (default: 0). 0=North, 90=East, 180=South, 270=West")
    parser.add_argument("--camera-elevation", type=float,
                        help="Camera elevation in degrees (default: 35)")
    parser.add_argument("--fov", type=float,
                        help="Camera horizontal field of view in degrees (default: 100)")
    parser.add_argument("--sun-azimuth", type=float,
                        help="Sun azimuth in degrees (default: 20). 0=North, 90=East, 180=South, 270=West")
    parser.add_argument("--sun-elevation", type=float,
                        help="Sun elevation in degrees (default: 55)")
    parser.add_argument("--size", type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        help="Image size in pixels (overrides quality preset)")
    parser.add_argument("--exposure", type=float,
                        help="Tone-map exposure (linear pre-scale; default 4.0). "
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
                        help=f"Aerosol amount, 0 to 2 (default: {DEFAULT_HAZE})")
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
                             "quality tier: its step factors, its lighting "
                             "method (sun-tau cache, sky probe) and its "
                             "parked sample count, all from the one preset")
    parser.add_argument("--periodic", action="store_true",
                        help="Wrap the domain in x and y, as soar does for LES fields")
    parser.add_argument("--nest-group", metavar="GROUP",
                        help="NetCDF group in the same file holding a finer field "
                             "to render as a nest inside the outer domain")
    add_dataset_selection_arguments(parser)

    args = parser.parse_args()
    size = tuple(args.size) if args.size else None

    main(args.filename, args.output,
         camera_position=args.camera_position,
         camera_azimuth=args.camera_azimuth,
         camera_elevation=args.camera_elevation,
         camera_fov=args.fov,
         sun_azimuth=args.sun_azimuth,
         sun_elevation=args.sun_elevation,
         custom_size=size,
         exposure=args.exposure,
         tone_map_gamma=args.gamma,
         tone_map_white_point=args.white_point,
         contrast=args.contrast,
         haze=args.haze,
         lod=args.lod,
         quality=args.quality,
         haze_height_dependent=args.haze_height_dependent,
         periodic=args.periodic,
         nest_group=args.nest_group,
         **dataset_selection_kwargs(args))


if __name__ == "__main__":
    cli()
