"""3D Radiative transfer using Mitsuba 3 for CloudyView.

Professional Monte Carlo path tracing for realistic cloud visualization.
Features:
- Physically-based sky models (Preetham sunsky)
- Optional ocean surface with waves
- Proper atmospheric perspective
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import warnings

try:
    import mitsuba as mi
    import drjit as dr
    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False
    raise RuntimeError("Mitsuba 3 is required for radiative transfer rendering but was not found")


def look_at_world_up(origin, target, fallback_up=(0, 1, 0), world_up=(0, 0, 1)):
    """Return a Mitsuba look_at transform that keeps image-up aligned with world-up."""

    origin = np.array(origin, dtype=float)
    target = np.array(target, dtype=float)
    forward = target - origin
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0:
        raise ValueError("look_at_world_up requires origin != target")
    forward /= forward_norm

    world_up_vec = np.array(world_up, dtype=float)
    world_up_vec /= np.linalg.norm(world_up_vec)

    # If forward is almost parallel to world-up (e.g., top-down view), fall back
    if abs(np.dot(forward, world_up_vec)) > 0.999:
        up = np.array(fallback_up, dtype=float)
    else:
        up = world_up_vec

    return mi.ScalarTransform4f.look_at(
        origin=origin.tolist(),
        target=target.tolist(),
        up=up.tolist()
    )


def _fmt_eta(seconds):
    """Format seconds to HH:MM:SS string."""
    if not seconds or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def render_with_progress(scene, spp_total, step_spp=8, seed=0):
    """
    Minimal progressive render with single-line print and ETA.
    Prints 0% first, then updates AFTER each chunk finishes.
    """
    acc = None
    taken = 0
    start = time.perf_counter()
    warmup_time = None

    # initial line (0%)
    print(f"  Rendering...   0.0%  (0/{spp_total} spp)  ETA --:--", end="", flush=True)

    # iterate in chunks without off-by-one
    for k, _ in enumerate(range(0, spp_total, step_spp)):
        spp_k = min(step_spp, spp_total - taken)

        chunk_start = time.perf_counter()
        # do one chunk
        img_k = mi.render(scene, spp=spp_k, seed=seed + k)

        # ensure the chunk actually finished before we report progress
        dr.eval(img_k)

        # accumulate (weighted by spp)
        acc = img_k * spp_k if acc is None else acc + img_k * spp_k
        dr.eval(acc)

        taken += spp_k
        chunk_elapsed = time.perf_counter() - chunk_start
        if warmup_time is None:
            warmup_time = chunk_elapsed

        # update ETA based on samples completed
        elapsed = time.perf_counter() - start
        eta = None
        if taken > step_spp:
            effective_elapsed = elapsed - warmup_time
            samples_after_warmup = taken - step_spp
            if effective_elapsed > 0 and samples_after_warmup > 0:
                eta = effective_elapsed * (spp_total - taken) / samples_after_warmup

        pct = 100.0 * taken / spp_total
        print(f"\r  Rendering... {pct:5.1f}%  ({taken}/{spp_total} spp)  ETA {_fmt_eta(eta)}",
              end="", flush=True)

    print()  # newline after finishing
    return acc / taken


def sun_direction_to_scene(azimuth_deg=0.0, elevation_deg=90.0):
    """Return unit vector pointing FROM the sun TOWARD the scene (for directional lights)."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cos_el = np.cos(el)
    direction = np.array([
        cos_el * np.cos(az),
        cos_el * np.sin(az),
        -np.sin(el)
    ])
    direction /= np.linalg.norm(direction)
    return direction.tolist()


def sun_direction_to_sun(azimuth_deg=0.0, elevation_deg=90.0):
    """Return unit vector pointing FROM scene TO the sun (for sunsky emitter)."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cos_el = np.cos(el)
    direction = np.array([
        cos_el * np.cos(az),
        cos_el * np.sin(az),
        np.sin(el)  # Positive z points upward to sun
    ])
    direction /= np.linalg.norm(direction)
    return direction.tolist()


def create_mitsuba_scene(sigma_ext, dx, dy, dz, camera_config, spp=256):
    """
    Create Mitsuba scene with volumetric cloud, sky, and optional ocean.

    Parameters
    -----------
    sigma_ext : ndarray (nx, ny, nz)
        Extinction coefficient field
    dx, dy, dz : float
        Grid spacings
    camera_config : dict
        Camera configuration with keys:
        - 'transform': camera transform
        - 'width', 'height': image dimensions
        - 'fov': field of view
        - 'extinction_multiplier': extinction scaling factor
        - 'sky_type': 'sunsky' (physically-based) or 'constant' or None
        - 'turbidity': sky turbidity (2-6, for sunsky)
        - 'add_ocean': bool, whether to add ocean surface
        - 'sun_azimuth': azimuth angle in degrees
        - 'sun_elevation': elevation angle in degrees
    spp : int
        Samples per pixel

    Returns
    --------
    scene : mi.Scene
        Mitsuba scene ready to render
    """
    nx, ny, nz = sigma_ext.shape

    # Calculate physical aspect ratios
    width_x = nx * dx
    width_y = ny * dy
    height_z = nz * dz
    aspect_ratio = width_x / height_z


    # Prepare extinction data for Mitsuba
    extinction_data = (sigma_ext * camera_config['extinction_multiplier'] * height_z)[
        ..., np.newaxis
    ].astype(np.float32)
    extinction_data = np.ascontiguousarray(
        np.transpose(extinction_data, (2, 1, 0, 3))
    )

    # Create volume grid
    volume_grid = mi.VolumeGrid(extinction_data)

    # Cube: scale for aspect ratio, centered at origin
    # World space: [-ar, ar] x [-ar, ar] x [-1, 1]
    cube_transform = mi.ScalarTransform4f.scale([aspect_ratio, aspect_ratio, 1.0])

    # Grid: map grid [0,1]^3 -> cube local [-1,1]^3 (and apply aspect ratio)
    # Mitsuba expects sigma_t.to_world to place the unit grid into world space, so
    # we first remap [0,1] to [-1,1] and then reuse the cube transform to stretch
    # the medium volume exactly like the enclosing shape.
    grid_to_cube = mi.ScalarTransform4f.translate([-1.0, -1.0, -1.0]) @ mi.ScalarTransform4f.scale([2.0, 2.0, 2.0])
    grid_transform = cube_transform @ grid_to_cube

    # Sun direction (for directional light: from sun to scene)
    sun_az = camera_config.get('sun_azimuth', 0.0)
    sun_el = camera_config.get('sun_elevation', 90.0)
    sun_dir_to_scene = sun_direction_to_scene(sun_az, sun_el)
    sun_dir_to_sun = sun_direction_to_sun(sun_az, sun_el)

    # Create scene dictionary
    camera_origin = camera_config.get('camera_origin')
    camera_inside = False
    if camera_origin is not None:
        cam_x, cam_y, cam_z = map(float, camera_origin)
        eps = 1e-4
        if (-aspect_ratio - eps <= cam_x <= aspect_ratio + eps and
                -aspect_ratio - eps <= cam_y <= aspect_ratio + eps and
                -1.0 - eps <= cam_z <= 1.0 + eps):
            camera_inside = True

    integrator_type = camera_config.get('integrator', 'volpath')
    max_depth = camera_config.get('max_depth', 32)
    rr_depth = camera_config.get('rr_depth', 5)
    integrator = {
        'type': integrator_type,
        'max_depth': max_depth,
        'rr_depth': rr_depth,
    }

    if integrator_type == 'volpathmis':
        # same params but rr_depth is named differently
        integrator['max_depth'] = max_depth
    elif integrator_type == 'volpath':
        integrator['max_depth'] = max_depth
    elif integrator_type == 'volpath_simple':
        integrator.pop('rr_depth', None)

    scene_dict = {
        'type': 'scene',
        'integrator': integrator,

        # Camera/sensor
        'sensor': {
            'type': 'perspective',
            'fov': camera_config.get('fov', 45),
            'to_world': camera_config['transform'],
            'sampler': camera_config.get('sampler', {
                'type': 'independent',
                'sample_count': spp,
            }),
            'film': {
                'type': 'hdrfilm',
                'width': camera_config['width'],
                'height': camera_config['height'],
                'rfilter': {'type': 'gaussian'},
            }
        },

        # Cloud volume with proper aspect ratio
        'cloud': {
            'type': 'cube',
            'to_world': cube_transform,
            'bsdf': {'type': 'null'},  # Invisible boundary
            'interior': {
                'type': 'heterogeneous',
                'id': 'cloud_medium',
                'sigma_t': {
                    'type': 'gridvolume',
                    'grid': volume_grid,
                    'to_world': grid_transform,  # Map grid [0,1]^3 to cube local [-1,1]^3
                    'wrap_mode': 'mirror',  # Mirror at boundaries
                },
                'albedo': {'type': 'rgb', 'value': [0.9999, 0.9999, 0.9999]},
                'phase': {
                    'type': 'hg',
                    'g': 0.85  # Forward scattering for clouds
                }
            }
        },
    }

    if camera_inside:
        scene_dict['sensor']['medium'] = {'type': 'ref', 'id': 'cloud_medium'}

    # Add sky/sun based on configuration
    sky_type = camera_config.get('sky_type', 'constant')

    if sky_type == 'sunsky':
        # Physically-based Hosek-Wilkie sun+sky model
        scene_dict['sunsky'] = {
            'type': 'sunsky',
            'sun_direction': sun_dir_to_sun,  # Direction TO the sun (upward)
            'turbidity': camera_config.get('turbidity', 3.0),  # 2=clear, 6=hazy
            'albedo': camera_config.get('ground_albedo', 0.3),  # Default ground albedo
            'sun_scale': camera_config.get('sun_scale', 1.0),
            'sky_scale': camera_config.get('sky_scale', 1.0),
        }
    elif sky_type == 'constant':
        # Simple constant sky + directional sun
        print(f"  Using constant sky + directional sun")
        scene_dict['sun'] = {
            'type': 'directional',
            'direction': sun_dir_to_scene,
            'irradiance': {'type': 'rgb', 'value': [1000.0, 1000.0, 1000.0]}
        }
        sky_color = camera_config.get('sky_color', [0.04231, 0.06848, 0.38133])
        scene_dict['sky'] = {
            'type': 'constant',
            'radiance': {'type': 'rgb', 'value': sky_color}
        }
    elif sky_type is None:
        # No sky, just sun
        print(f"  Using directional sun only (no sky)")
        scene_dict['sun'] = {
            'type': 'directional',
            'direction': sun_dir_to_scene,
            'irradiance': {'type': 'rgb', 'value': [1000.0, 1000.0, 1000.0]}
        }

    # Add ocean surface if requested
    if camera_config.get('add_ocean', False):
        ocean_size_multiplier = camera_config.get('ocean_size_multiplier', 10.0)
        ocean_height = camera_config.get('ocean_height', -0.99)
        ocean_height = float(np.clip(ocean_height, -1.0, 1.0))  # keep within cube bounds
        ocean_size = aspect_ratio * ocean_size_multiplier
        ocean_reflectance = camera_config.get('ocean_reflectance', [0.2, 0.3, 0.4])

        ocean_transform = (
            mi.ScalarTransform4f.translate([0.0, 0.0, ocean_height]) @
            mi.ScalarTransform4f.scale([ocean_size, ocean_size, 1.0])
        )

        scene_dict['ocean'] = {
            'type': 'rectangle',
            'to_world': ocean_transform,
            'bsdf': {
                'type': 'twosided',  # Visible and reflective from above/below
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {'type': 'rgb', 'value': ocean_reflectance},
                }
            }
        }

    scene = mi.load_dict(scene_dict)
    return scene


def tone_map(image, exposure=1.0, gamma=1.4):
    """Reinhard tone mapping with gamma correction."""
    exposed = image * exposure
    tone_mapped = exposed / (1.0 + exposed)
    rgb = np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)
    return rgb


def render_view(
    sigma_ext: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    view_config: Dict,
    output_file: str
) -> np.ndarray:
    """
    Render a single view and save to file.

    Parameters
    ----------
    sigma_ext : ndarray
        Extinction coefficient field
    dx, dy, dz : float
        Grid spacings
    view_config : dict
        View configuration (camera position, sky, etc.)
    output_file : str
        Path to save rendered image

    Returns
    -------
    img_np : ndarray
        Rendered and tone-mapped image
    """

    # Create scene
    scene = create_mitsuba_scene(
        sigma_ext, dx, dy, dz,
        view_config,
        spp=view_config['spp']
    )

    # Render
    step = 2
    image = render_with_progress(scene,
                                spp_total=view_config['spp'],
                                step_spp=step,
                                seed=view_config.get('seed', 0))

    # Convert to numpy
    img_np = np.array(image)

    # Tone mapping
    img_np = tone_map(img_np, exposure=view_config.get('exposure', 1.0))

    # Save
    dpi = 192
    height = view_config['height']
    width = view_config['width']
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_np, origin='upper')
    ax.axis('off')
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)


    return img_np
