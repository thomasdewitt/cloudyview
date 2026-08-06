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
from typing import List, Optional, Sequence, Tuple

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
from .soar_host import (
    DEFAULT_TONE_MAP_GAMMA,
    STILL_ACCUMULATE_FRAMES,
    STEP_VOXEL_FACTOR,
    TONE_MAP_GAMMA_WITNESS,
    SceneState,
    SoarRenderer,
    ViewState,
    camera_world_origin,
)

logger = logging.getLogger(__name__)

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


def _padded(sigma: np.ndarray) -> np.ndarray:
    """Ghost-pad by one voxel per side; original voxel i lands at i+1.

    The border stays zero so hardware trilinear filtering supplies a linear
    taper out of the field instead of smearing the edge voxel outward.
    """
    nx, ny, nz = sigma.shape
    out = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float16)
    out[1:-1, 1:-1, 1:-1] = sigma
    return out


def _renderer_for(levels: Sequence[NestedLevel], *, periodic: bool,
                  tone_mapped: bool) -> SoarRenderer:
    """Get or build a session for these levels."""
    global _session, _session_key
    nested = len(levels) > 1
    key = (tuple((id(l.sigma), l.sigma.shape, float(l.bmin[0]), float(l.bmax[0]))
                 for l in levels), periodic, nested, tone_mapped)
    if _session is not None and _session_key == key:
        return _session
    renderer = SoarRenderer(periodic=periodic, nested=nested,
                            tone_map=tone_mapped)
    renderer.upload_volume(_padded(levels[-1].sigma))
    if nested:
        renderer.upload_nest(_padded(levels[0].sigma))
    _session, _session_key = renderer, key
    return renderer


def render_nested(
    levels: Sequence[NestedLevel],
    camera_position,
    camera_forward=None,
    camera_right=None,
    camera_up=None,
    sun_direction=None,
    image_size=(600, 400),
    fov_degrees: float = 100.0,
    *,
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
    sun_azimuth: Optional[float] = None,
    sun_elevation: Optional[float] = None,
    exposure: float = 4.0,
    tone_map_gamma: float = DEFAULT_TONE_MAP_GAMMA,
    periodic: bool = False,
    accumulate: int = STILL_ACCUMULATE_FRAMES,
    step_voxel_factor: float = STEP_VOXEL_FACTOR,
    return_linear: bool = False,
    verbose: bool = True,
    **legacy,
) -> np.ndarray:
    """Render one or two nested levels, finest first.

    The shader carries exactly one optional nest — `NESTED` is a compile-time
    constant and core WebGPU has no dynamic texture indexing — so more than
    two levels raises rather than silently dropping the extra ones.

    Angles are preferred over basis vectors now that the camera basis lives
    in one place; pass azimuth/elevation and sun_azimuth/sun_elevation. The
    old forward/right/up and sun_direction arguments are accepted and
    converted.

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
    for unexpected in legacy:
        logger.debug("render_nested: ignoring legacy argument %r", unexpected)

    if azimuth is None or elevation is None:
        if camera_forward is None:
            raise ValueError(
                "render_nested needs azimuth and elevation (or a "
                "camera_forward vector to derive them from).")
        f = np.asarray(camera_forward, dtype=np.float64)
        f = f / np.linalg.norm(f)
        elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1.0, 1.0))))
        azimuth = float((90.0 - np.degrees(np.arctan2(f[1], f[0]))) % 360.0)
    if sun_azimuth is None or sun_elevation is None:
        if sun_direction is None:
            raise ValueError(
                "render_nested needs sun_azimuth and sun_elevation (or a "
                "sun_direction vector).")
        s = np.asarray(sun_direction, dtype=np.float64)
        s = s / np.linalg.norm(s)
        sun_elevation = float(np.degrees(np.arcsin(np.clip(s[2], -1.0, 1.0))))
        sun_azimuth = float((90.0 - np.degrees(np.arctan2(s[1], s[0]))) % 360.0)

    outer = levels[-1]
    renderer = _renderer_for(levels, periodic=periodic,
                             tone_mapped=not return_linear)
    min_voxel = min(outer.dx)
    dt = min_voxel * step_voxel_factor
    state = SceneState(
        bmin=[float(v) for v in outer.bmin], bmax=[float(v) for v in outer.bmax],
        dt_view=dt, dt_light=dt, periodic=periodic,
        ocean_reflectance=OCEAN_REFLECTANCE,
        nested=len(levels) > 1,
    )
    if len(levels) > 1:
        fine = levels[0]
        state.nest_bmin = [float(v) for v in fine.bmin]
        state.nest_bmax = [float(v) for v in fine.bmax]
        state.dt_view_nest = state.dt_light_nest = min(fine.dx) * step_voxel_factor

    w, h = image_size
    view = ViewState(
        camera_position=[float(v) for v in camera_position],
        azimuth=azimuth, elevation=elevation, fov=fov_degrees,
        output_size=(w, h), render_size=(w, h),
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        exposure=exposure, tone_map_gamma=tone_map_gamma,
    )
    if verbose:
        print(f"  Rendering {w}x{h}, {accumulate} accumulated passes, "
              f"{len(levels)} level(s), dt={dt:.1f} m"
              f"{', periodic' if periodic else ''}")
    t0 = time.perf_counter()
    image = renderer.render(state, view, frames=accumulate)
    if verbose:
        print(f"  Rendered in {time.perf_counter() - t0:.2f}s")
    return image


def witness(
    field: CloudField,
    camera: Optional[Camera] = None,
    *,
    size: Optional[Tuple[int, int]] = None,
    sun_azimuth: Optional[float] = None,
    sun_elevation: Optional[float] = None,
    exposure: Optional[float] = None,
    tone_map_gamma: float = DEFAULT_TONE_MAP_GAMMA,
    periodic: bool = False,
    accumulate: int = STILL_ACCUMULATE_FRAMES,
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
    level = NestedLevel(sigma=sigma, bmin=bmin, bmax=bmax, name="single")
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
        periodic=periodic, accumulate=accumulate, verbose=verbose)


def main(filename: str, output: str = None,
         camera_position: list = None, camera_azimuth: float = None,
         camera_elevation: float = None, camera_fov: float = None,
         sun_azimuth: float = None, sun_elevation: float = None,
         custom_size: tuple = None,
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
         z_dim: str = None) -> None:
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

    try:
        field = _load_field(
            filename,
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
        )

        camera = Camera(
            position=cam_config['position'],
            azimuth=cam_config['azimuth'],
            elevation=cam_config['elevation'],
            fov=cam_config['fov'],
        )

        image_tm = witness(
            field,
            camera=camera,
            size=tuple(custom_size) if custom_size else None,
            sun_azimuth=sun_azimuth,
            sun_elevation=sun_elevation,
            verbose=True,
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


QUALITY_PRESETS = {
    'min':    (150, 100),
    'low':    (300, 200),
    'medium': (600, 400),
    'high':   (1600, 1200),
}


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
              Positional `quality` chooses a size preset:
              - min: 150x100
              - low: 300x200
              - medium: 600x400
              - high: 1600x1200
              `--size WIDTH HEIGHT` overrides the preset.

            Dependencies:
              `witness --help` works without a GPU. Rendering needs one: this drives
              the same WGSL shader the browser app runs, through wgpu, so there is a
              single renderer core rather than a CPU copy that drifts from it.

            {DATA_SELECTION_HELP}

            Examples:
              witness cloud.nc
              witness cloud.nc high --output renders
              witness cloud.nc medium --size 1200 800 --camera-position 0 -0.9 -0.99 --camera-azimuth 0 --camera-elevation 35
              witness cloud.nc --group /physics/clouds --liquid-water-var QCLOUD --ice-water-var QICE
              witness custom.nc --liquid-water-group /state/liquid --ice-water-group /state/ice --coords-group /grid --x-dim ni --y-dim nj --z-dim nk --x-coord xh --y-coord yh --z-coord zh
            """
        ),
    )
    parser.add_argument("filename",
                        help="NetCDF file with cloud data")
    parser.add_argument("quality", nargs='?', default='medium',
                        choices=QUALITY_PRESETS.keys(),
                        help="Quality preset (default: medium)")
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
    add_dataset_selection_arguments(parser)

    args = parser.parse_args()
    size = tuple(args.size) if args.size else QUALITY_PRESETS[args.quality]

    main(args.filename, args.output,
         camera_position=args.camera_position,
         camera_azimuth=args.camera_azimuth,
         camera_elevation=args.camera_elevation,
         camera_fov=args.fov,
         sun_azimuth=args.sun_azimuth,
         sun_elevation=args.sun_elevation,
         custom_size=size,
         **dataset_selection_kwargs(args))


if __name__ == "__main__":
    cli()
