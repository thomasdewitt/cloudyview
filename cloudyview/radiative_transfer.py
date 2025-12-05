"""3D Radiative transfer using Mitsuba 3 for CloudyView.

Professional Monte Carlo path tracing for realistic cloud visualization.
Features:
- Physically-based sky models (Preetham sunsky)
- Optional ocean surface with waves
- Proper atmospheric perspective
- Accurate Mie scattering phase functions with dual-table importance sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, Literal
import warnings
import os

try:
    import mitsuba as mi
    import drjit as dr
    MITSUBA_AVAILABLE = True
except ImportError:
    MITSUBA_AVAILABLE = False
    raise RuntimeError("Mitsuba 3 is required for radiative transfer rendering but was not found")


def load_mie_phase_tables(channel: Literal['R', 'G', 'B', 'gray'] = 'gray'):
    """
    Load Mie scattering phase function tables from Bouthors (2008) thesis.

    Returns forward peak (Mie0) and rest of distribution (MiePF3) as comma-separated strings
    for use with Mitsuba's tabphase plugin.

    Parameters
    ----------
    channel : {'R', 'G', 'B', 'gray'}
        Which wavelength channel to use. MiePF3 has RGB values, Mie0 is monochrome.
        'gray' uses luminance-weighted average of RGB.

    Returns
    -------
    mie0_str : str
        Forward peak phase function values (comma-separated)
    mie_pf3_str : str
        Rest of distribution phase function values (comma-separated)

    References
    ----------
    Antoine Bouthors (2008) - "Real-time realistic rendering of clouds"
    PhD Thesis, Université Grenoble I - Joseph Fourier
    https://theses.hal.science/tel-00319974
    """
    # Find the Mie_tables directory relative to this module
    module_dir = Path(__file__).parent
    mie_dir = module_dir / 'Mie_tables'

    # Load normalized tables
    mie0_path = mie_dir / 'Mie0_normalized.txt'
    mie_pf3_path = mie_dir / 'MiePF3_normalized.txt'

    if not mie0_path.exists() or not mie_pf3_path.exists():
        raise FileNotFoundError(
            f"Mie phase function tables not found in {mie_dir}. "
            "Expected Mie0_normalized.txt and MiePF3_normalized.txt"
        )

    mie0 = np.loadtxt(mie0_path)
    mie_pf3 = np.loadtxt(mie_pf3_path)

    # Select channel for MiePF3 (RGB)
    if channel == 'R':
        mie_pf3_values = mie_pf3[:, 0]
    elif channel == 'G':
        mie_pf3_values = mie_pf3[:, 1]
    elif channel == 'B':
        mie_pf3_values = mie_pf3[:, 2]
    elif channel == 'gray':
        # Luminance weights: R=0.2126, G=0.7152, B=0.0722
        mie_pf3_values = (0.2126 * mie_pf3[:, 0] +
                          0.7152 * mie_pf3[:, 1] +
                          0.0722 * mie_pf3[:, 2])
    else:
        raise ValueError(f"channel must be 'R', 'G', 'B', or 'gray', got {channel}")

    # Convert to comma-separated strings for Mitsuba tabphase
    mie0_str = ','.join(map(str, mie0))
    mie_pf3_str = ','.join(map(str, mie_pf3_values))

    return mie0_str, mie_pf3_str


def load_ice_phase_tables(channel: Literal['R', 'G', 'B', 'gray'] = 'gray'):
    """
    Load ice crystal scattering phase function tables.

    PLACEHOLDER: Ice tables not yet implemented. When available, they should be
    placed in Mie_tables/ as IceMie0_normalized.txt and IceMiePF3_normalized.txt
    with identical format to the liquid water tables.

    For now, returns HG approximation parameters instead of full tables.

    Parameters
    ----------
    channel : {'R', 'G', 'B', 'gray'}
        Which wavelength channel to use (for future compatibility)

    Returns
    -------
    ice_mie0_str : str or None
        Forward peak phase function values (comma-separated), or None for HG fallback
    ice_mie_pf3_str : str or None
        Rest of distribution phase function values (comma-separated), or None for HG fallback
    use_hg : bool
        If True, use Henyey-Greenstein approximation instead of tables
    g_ice : float
        HG asymmetry parameter for ice (only used if use_hg=True)

    References
    ----------
    Ice scattering properties from Yang et al. (2000, 2013) databases
    """
    # Find the Mie_tables directory relative to this module
    module_dir = Path(__file__).parent
    mie_dir = module_dir / 'Mie_tables'

    # Check if ice tables exist
    ice_mie0_path = mie_dir / 'IceMie0_normalized.txt'
    ice_mie_pf3_path = mie_dir / 'IceMiePF3_normalized.txt'

    if ice_mie0_path.exists() and ice_mie_pf3_path.exists():
        # Load ice tables
        ice_mie0 = np.loadtxt(ice_mie0_path)
        ice_mie_pf3 = np.loadtxt(ice_mie_pf3_path)

        # Handle channel selection for IceMie0
        # Ice tables may have RGB columns (unlike liquid which is monochrome)
        if ice_mie0.ndim == 2:
            # RGB format - select channel
            if channel == 'R':
                ice_mie0_values = ice_mie0[:, 0]
            elif channel == 'G':
                ice_mie0_values = ice_mie0[:, 1]
            elif channel == 'B':
                ice_mie0_values = ice_mie0[:, 2]
            elif channel == 'gray':
                # Luminance weights: R=0.2126, G=0.7152, B=0.0722
                ice_mie0_values = (0.2126 * ice_mie0[:, 0] +
                                   0.7152 * ice_mie0[:, 1] +
                                   0.0722 * ice_mie0[:, 2])
            else:
                raise ValueError(f"channel must be 'R', 'G', 'B', or 'gray', got {channel}")
        else:
            # Monochrome format - use directly
            ice_mie0_values = ice_mie0

        # Select channel for IceMiePF3 (RGB)
        if channel == 'R':
            ice_mie_pf3_values = ice_mie_pf3[:, 0]
        elif channel == 'G':
            ice_mie_pf3_values = ice_mie_pf3[:, 1]
        elif channel == 'B':
            ice_mie_pf3_values = ice_mie_pf3[:, 2]
        elif channel == 'gray':
            # Luminance weights: R=0.2126, G=0.7152, B=0.0722
            ice_mie_pf3_values = (0.2126 * ice_mie_pf3[:, 0] +
                                  0.7152 * ice_mie_pf3[:, 1] +
                                  0.0722 * ice_mie_pf3[:, 2])
        else:
            raise ValueError(f"channel must be 'R', 'G', 'B', or 'gray', got {channel}")

        # Convert to comma-separated strings for Mitsuba tabphase
        ice_mie0_str = ','.join(map(str, ice_mie0_values))
        ice_mie_pf3_str = ','.join(map(str, ice_mie_pf3_values))

        return ice_mie0_str, ice_mie_pf3_str, False, None
    else:
        # Fallback to Henyey-Greenstein approximation
        print("  Ice Mie tables not found, using Henyey-Greenstein approximation (g=0.78)")
        return None, None, True, 0.78


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


def render_with_progress(scene, spp_total, step_spp=8, seed=0, checkpoint_config=None):
    """
    Minimal progressive render with single-line print and ETA.
    Prints 0% first, then updates AFTER each chunk finishes.

    Parameters
    ----------
    scene : mi.Scene
        Mitsuba scene to render
    spp_total : int
        Total samples per pixel
    step_spp : int
        Samples per rendering step
    seed : int
        Random seed
    checkpoint_config : dict, optional
        Configuration for saving checkpoint images:
        - 'checkpoints': list of SPP values at which to save images
        - 'output_pattern': str with {spp} placeholder for filename
        - 'tone_map_func': function to apply tone mapping
        - 'save_func': function to save image (receives img_np, filepath)
    """
    acc = None
    taken = 0
    start = time.perf_counter()
    warmup_time = None

    # Track which checkpoints we've already saved
    checkpoints = []
    if checkpoint_config is not None:
        checkpoints = sorted(checkpoint_config.get('checkpoints', []))
        checkpoints_saved = set()
    else:
        checkpoints_saved = set()

    # initial line (0%)
    print(f"  0% | 0/{spp_total} spp | Elapsed: --:-- | ETA: --:--", end="", flush=True)

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

        # Check if we've reached any checkpoint
        if checkpoint_config is not None and checkpoints:
            for checkpoint_spp in checkpoints:
                if checkpoint_spp <= taken and checkpoint_spp not in checkpoints_saved:
                    # Save checkpoint image
                    current_img = acc / taken
                    img_np = np.array(current_img)

                    # Apply tone mapping if provided
                    if 'tone_map_func' in checkpoint_config:
                        img_np = checkpoint_config['tone_map_func'](img_np)

                    # Generate output filepath
                    output_pattern = checkpoint_config.get('output_pattern', 'checkpoint_spp={spp}.png')
                    filepath = output_pattern.format(spp=checkpoint_spp)

                    # Save using provided save function
                    if 'save_func' in checkpoint_config:
                        checkpoint_config['save_func'](img_np, filepath)

                    checkpoints_saved.add(checkpoint_spp)

        # update ETA based on samples completed
        elapsed = time.perf_counter() - start
        eta = None
        if taken > step_spp:
            effective_elapsed = elapsed - warmup_time
            samples_after_warmup = taken - step_spp
            if effective_elapsed > 0 and samples_after_warmup > 0:
                eta = effective_elapsed * (spp_total - taken) / samples_after_warmup

        pct = 100.0 * taken / spp_total
        elapsed_str = _fmt_eta(elapsed)
        eta_str = _fmt_eta(eta)
        print(f"\r  {pct:3.0f}% | {taken}/{spp_total} spp | Elapsed: {elapsed_str} | ETA: {eta_str}",
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


def create_mitsuba_scene(sigma_ext, dx, dy, dz, camera_config, spp=256,
                        use_mie_phase=True, mie_channel='gray', mie_blend_weight=0.5,
                        wavelength_nm=None, ice_fraction=None):
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
    use_mie_phase : bool
        If True, use accurate Mie scattering tables (Bouthors 2008) instead of Henyey-Greenstein
    mie_channel : {'R', 'G', 'B', 'gray'}
        Wavelength channel for Mie phase function (only used if use_mie_phase=True)
    mie_blend_weight : float
        Blend weight for dual-table importance sampling (0-1). Controls sampling balance
        between forward peak (weight) and rest of distribution (1-weight). Default 0.5
        ensures equal sampling of both regions despite strong forward peak.
    wavelength_nm : float, optional
        Wavelength in nanometers for spectral rendering. If specified, sets sensor to
        render at this specific wavelength. Used for RGB channel rendering.
    ice_fraction : ndarray (nx, ny, nz), optional
        Ice mass fraction field (0 = liquid, 1 = ice). If provided, creates spatially-varying
        phase function blending between liquid and ice phase functions.

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
                'phase': None,  # Will be set below
            }
        },
    }

    # Configure phase function (Mie tables or Henyey-Greenstein)
    # If ice_fraction is provided, create spatially-varying blend between liquid and ice phases
    if ice_fraction is not None:
        print("  Setting up mixed-phase (liquid/ice) rendering...")

        # Prepare ice fraction grid for Mitsuba (same transform as extinction)
        ice_fraction_data = ice_fraction[..., np.newaxis].astype(np.float32)
        ice_fraction_data = np.ascontiguousarray(
            np.transpose(ice_fraction_data, (2, 1, 0, 3))
        )
        ice_fraction_grid = mi.VolumeGrid(ice_fraction_data)

        if use_mie_phase:
            # Load liquid water Mie tables
            mie0_str, mie_pf3_str = load_mie_phase_tables(channel=mie_channel)

            # Load ice Mie tables (or use HG fallback)
            ice_mie0_str, ice_mie_pf3_str, use_hg_ice, g_ice = load_ice_phase_tables(channel=mie_channel)

            # Create liquid phase (dual-table blended)
            liquid_phase = {
                'type': 'blendphase',
                'weight': mie_blend_weight,
                'phase_0': {
                    'type': 'tabphase',
                    'values': mie_pf3_str
                },
                'phase_1': {
                    'type': 'tabphase',
                    'values': mie0_str
                }
            }

            # Create ice phase (dual-table blended or HG)
            if use_hg_ice:
                ice_phase = {
                    'type': 'hg',
                    'g': g_ice
                }
            else:
                ice_phase = {
                    'type': 'blendphase',
                    'weight': mie_blend_weight,
                    'phase_0': {
                        'type': 'tabphase',
                        'values': ice_mie_pf3_str
                    },
                    'phase_1': {
                        'type': 'tabphase',
                        'values': ice_mie0_str
                    }
                }

            print(f"  Liquid phase: Mie tables (channel={mie_channel})")
            print(f"  Ice phase: {'HG (g=' + str(g_ice) + ')' if use_hg_ice else 'Mie tables (channel=' + mie_channel + ')'}")
            print(f"  Spatially-varying blend via ice fraction grid")

            # Spatially-varying blend between liquid and ice
            scene_dict['cloud']['interior']['phase'] = {
                'type': 'blendphase',
                'phase_0': liquid_phase,  # phase_0 when weight=0 (liquid)
                'phase_1': ice_phase,      # phase_1 when weight=1 (ice)
                'weight': {
                    'type': 'gridvolume',
                    'grid': ice_fraction_grid,
                    'to_world': grid_transform,
                    'wrap_mode': 'mirror'
                }
            }

        else:
            # HG approximation for both liquid and ice
            print("  Liquid phase: HG (g=0.85)")
            print("  Ice phase: HG (g=0.78)")
            print("  Spatially-varying blend via ice fraction grid")

            scene_dict['cloud']['interior']['phase'] = {
                'type': 'blendphase',
                'phase_0': {'type': 'hg', 'g': 0.85},  # Liquid
                'phase_1': {'type': 'hg', 'g': 0.78},  # Ice
                'weight': {
                    'type': 'gridvolume',
                    'grid': ice_fraction_grid,
                    'to_world': grid_transform,
                    'wrap_mode': 'mirror'
                }
            }

    elif use_mie_phase:
        # Liquid-only with Mie scattering tables from Bouthors (2008)
        mie0_str, mie_pf3_str = load_mie_phase_tables(channel=mie_channel)
        print(f"  Using Mie phase function (channel={mie_channel}, blend_weight={mie_blend_weight:.2f})")

        # Dual-table blended phase function for efficient importance sampling
        scene_dict['cloud']['interior']['phase'] = {
            'type': 'blendphase',
            'weight': mie_blend_weight,  # Balance between peak and rest sampling
            'phase_0': {  # Rest of distribution (sideways/backward)
                'type': 'tabphase',
                'values': mie_pf3_str
            },
            'phase_1': {  # Forward peak
                'type': 'tabphase',
                'values': mie0_str
            }
        }
    else:
        # Simple Henyey-Greenstein approximation (liquid-only)
        print(f"  Using Henyey-Greenstein phase function (g=0.85)")
        scene_dict['cloud']['interior']['phase'] = {
            'type': 'hg',
            'g': 0.85  # Forward scattering for clouds
        }

    if camera_inside:
        scene_dict['sensor']['medium'] = {'type': 'ref', 'id': 'cloud_medium'}

    # Add sky/sun based on configuration
    sky_type = camera_config.get('sky_type', 'constant')

    if sky_type == 'sunsky':
        # Physically-based Hosek-Wilkie sun+sky model (works in RGB/spectral)
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
        # Default: #334A6C = RGB(51, 74, 108) = (0.2, 0.2902, 0.4235)
        sky_color = camera_config.get('sky_color', [0.2, 0.2902, 0.4235])
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
        ocean_size_multiplier = camera_config.get('ocean_size_multiplier', 100.0)
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
    output_file: str,
    checkpoint_spp: Optional[list] = None,
    ice_fraction: np.ndarray = None
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
        Should include 'render_mode': str - one of:
        - 'mono': Single monochrome render with gray Mie phase, grayscale output
        - 'rgb': Single RGB render with gray Mie phase (wavelength-averaged), full RGB sky
        - 'chromatic': 3 mono renders with wavelength-specific Mie, enables chromatic effects
          (coronas, halos, glories, etc.) but uses simple constant sky
    output_file : str
        Path to save rendered image
    checkpoint_spp : list of int, optional
        SPP values at which to save checkpoint images (e.g., [2, 32, 128, 512, 2048])
    ice_fraction : ndarray, optional
        Ice mass fraction field (0 = liquid, 1 = ice) for mixed-phase clouds

    Returns
    -------
    img_np : ndarray
        Rendered and tone-mapped image
    """

    # Determine render mode
    render_mode = view_config.get('render_mode', 'rgb')

    if render_mode == 'chromatic':
        # Render 3 mono channels separately with wavelength-dependent phase functions
        print("  Chromatic mode: rendering 3 monochrome channels (R, G, B) with wavelength-specific Mie phase")
        print("  (Enables coronas, halos, glories, and other chromatic scattering effects)")
        img_np = render_view_chromatic(sigma_ext, dx, dy, dz, view_config,
                                        output_file, checkpoint_spp, ice_fraction)
    elif render_mode == 'mono':
        # Single monochrome render
        print("  Mono mode: single grayscale render with luminance-weighted Mie phase")
        img_np = render_view_mono(sigma_ext, dx, dy, dz, view_config,
                                   output_file, checkpoint_spp, ice_fraction)
    else:  # 'rgb'
        # Single RGB render with averaged phase function
        print("  RGB mode: single RGB render with luminance-weighted Mie phase")
        img_np = render_view_single(sigma_ext, dx, dy, dz, view_config,
                                     output_file, checkpoint_spp, ice_fraction)

    return img_np


def render_view_chromatic(
    sigma_ext: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    view_config: Dict,
    output_file: str,
    checkpoint_spp: Optional[list] = None,
    ice_fraction: np.ndarray = None
) -> np.ndarray:
    """
    Render RGB by rendering 3 separate monochrome channels with wavelength-dependent phase functions.

    This enables coronas, halos, glories, and other chromatic scattering effects at approximately
    the same computational cost as a single RGB render (3× mono renders ≈ 1× RGB render).

    Renders progressively: at each SPP step, renders all 3 channels, then shows progress.

    Uses constant sky (wavelength-specific) since sunsky doesn't support monochrome mode.

    Internal function called by render_view when view_config['render_mode'] = 'chromatic'.
    """
    # Switch to mono mode for wavelength-specific rendering
    # Note: Each channel uses wavelength-specific phase function from Mie tables
    original_variant = mi.variant()
    mono_variant = get_best_variant('mono')
    mi.set_variant(mono_variant)
    print(f"  Using Mitsuba variant: {mono_variant}")

    # Create view config with simplified constant sky for mono mode
    view_config_mono = view_config.copy()
    view_config_mono['sky_type'] = 'constant'  # Use simple constant sky

    channels = ['R', 'G', 'B']
    scenes = []
    for channel in channels:
        scene = create_mitsuba_scene(
            sigma_ext, dx, dy, dz,
            view_config_mono,
            spp=view_config['spp'],
            mie_channel=channel,
            ice_fraction=ice_fraction
        )
        scenes.append(scene)

    # Progressive rendering setup
    spp_total = view_config['spp']
    step_spp = 16
    seed = view_config.get('seed', 0)

    # Accumulate samples for each channel
    channel_accumulators = [None, None, None]
    taken = 0
    start = time.perf_counter()
    warmup_time = None

    # Track checkpoints
    checkpoints = []
    checkpoints_saved = set()
    if checkpoint_spp is not None:
        checkpoints = sorted(checkpoint_spp)

    print(f"  0% | 0/{spp_total} spp | Elapsed: --:-- | ETA: --:--")

    while taken < spp_total:
        remaining = spp_total - taken
        step = min(step_spp, remaining)

        # Render all 3 channels for this step
        for i, (channel, scene) in enumerate(zip(channels, scenes)):
            img = mi.render(scene, spp=step, seed=seed + taken + i * 10000)

            if channel_accumulators[i] is None:
                channel_accumulators[i] = img
            else:
                channel_accumulators[i] += img

        taken += step
        elapsed = time.perf_counter() - start

        # Set warmup time after first step
        if warmup_time is None and taken >= step_spp:
            warmup_time = elapsed

        # Estimate time per sample (after warmup)
        if warmup_time is not None and taken > step_spp:
            time_per_spp = (elapsed - warmup_time) / (taken - step_spp)
            eta_seconds = time_per_spp * (spp_total - taken)
        else:
            eta_seconds = None

        pct = 100.0 * taken / spp_total
        elapsed_str = _fmt_eta(elapsed)
        eta_str = _fmt_eta(eta_seconds)
        print(f"\r  {pct:3.0f}% | {taken}/{spp_total} spp | Elapsed: {elapsed_str} | ETA: {eta_str}", end="", flush=True)

        # Save checkpoint if needed
        if checkpoints and taken in checkpoints and taken not in checkpoints_saved:
            # Combine current RGB state
            channel_images = [np.array(acc / taken)[..., 0] for acc in channel_accumulators]
            img_rgb = np.stack(channel_images, axis=-1)
            img_np = tone_map(img_rgb, exposure=view_config.get('exposure', 1.0))

            # Save checkpoint image
            output_path = Path(output_file)
            checkpoint_file = output_path.parent / f"{output_path.stem}_spp{taken}{output_path.suffix}"

            dpi = 192
            height = view_config['height']
            width = view_config['width']
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img_np, origin='upper')
            ax.axis('off')
            fig.savefig(checkpoint_file, dpi=dpi)
            plt.close(fig)

            checkpoints_saved.add(taken)

    print()  # Newline after progress

    # Normalize and combine final images
    channel_images = [np.array(acc / taken)[..., 0] for acc in channel_accumulators]
    img_rgb = np.stack(channel_images, axis=-1)

    # Restore original variant
    mi.set_variant(original_variant)

    # Tone mapping
    img_np = tone_map(img_rgb, exposure=view_config.get('exposure', 1.0))

    # Save final image
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


def render_view_mono(
    sigma_ext: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    view_config: Dict,
    output_file: str,
    checkpoint_spp: Optional[list] = None,
    ice_fraction: np.ndarray = None
) -> np.ndarray:
    """
    Render a single monochrome view with grayscale output.

    Uses luminance-weighted Mie phase function and converts RGB sky to grayscale.

    Internal function called by render_view when render_mode='mono'.
    """
    # Switch to mono mode
    original_variant = mi.variant()
    mono_variant = get_best_variant('mono')
    mi.set_variant(mono_variant)
    print(f"  Using Mitsuba variant: {mono_variant}")

    # Override sky_type for mono mode (sunsky doesn't support mono)
    view_config_mono = view_config.copy()
    view_config_mono['sky_type'] = 'constant'

    # Create scene
    scene = create_mitsuba_scene(
        sigma_ext, dx, dy, dz,
        view_config_mono,
        spp=view_config['spp'],
        ice_fraction=ice_fraction
    )

    # Helper function to save checkpoint images
    def save_checkpoint_image(img_np, filepath):
        dpi = 192
        height = view_config['height']
        width = view_config['width']
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_np, origin='upper', cmap='gray')
        ax.axis('off')
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)

    # Helper function to apply tone mapping
    def apply_tone_map(img_np):
        return tone_map(img_np, exposure=view_config.get('exposure', 1.0))

    # Set up checkpoint configuration
    checkpoint_config = None
    if checkpoint_spp is not None and len(checkpoint_spp) > 0:
        output_path = Path(output_file)
        output_pattern = str(output_path.parent / f"{output_path.stem}_spp={{spp}}{output_path.suffix}")

        checkpoint_config = {
            'checkpoints': checkpoint_spp,
            'output_pattern': output_pattern,
            'tone_map_func': apply_tone_map,
            'save_func': save_checkpoint_image
        }

    # Render
    step = 2
    image = render_with_progress(scene,
                                spp_total=view_config['spp'],
                                step_spp=step,
                                seed=view_config.get('seed', 0),
                                checkpoint_config=checkpoint_config)

    # Convert to numpy and extract mono channel
    img_np = np.array(image)[..., 0]

    # Restore variant
    mi.set_variant(original_variant)

    # Tone mapping
    img_np = tone_map(img_np[..., np.newaxis], exposure=view_config.get('exposure', 1.0))[..., 0]

    # Save final image (grayscale)
    dpi = 192
    height = view_config['height']
    width = view_config['width']
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_np, origin='upper', cmap='gray')
    ax.axis('off')
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)

    return img_np


def render_view_single(
    sigma_ext: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    view_config: Dict,
    output_file: str,
    checkpoint_spp: Optional[list] = None,
    ice_fraction: np.ndarray = None
) -> np.ndarray:
    """
    Render a single RGB view with wavelength-averaged Mie phase function.

    Uses full RGB rendering with physically-based sunsky.

    Internal function called by render_view when render_mode='rgb' (default).
    """
    # Create scene
    scene = create_mitsuba_scene(
        sigma_ext, dx, dy, dz,
        view_config,
        spp=view_config['spp'],
        ice_fraction=ice_fraction
    )

    # Helper function to save checkpoint images
    def save_checkpoint_image(img_np, filepath):
        dpi = 192
        height = view_config['height']
        width = view_config['width']
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_np, origin='upper')
        ax.axis('off')
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)

    # Helper function to apply tone mapping
    def apply_tone_map(img_np):
        return tone_map(img_np, exposure=view_config.get('exposure', 1.0))

    # Set up checkpoint configuration
    checkpoint_config = None
    if checkpoint_spp is not None and len(checkpoint_spp) > 0:
        # Generate output pattern based on output_file
        output_path = Path(output_file)
        output_pattern = str(output_path.parent / f"{output_path.stem}_spp{{spp}}{output_path.suffix}")

        checkpoint_config = {
            'checkpoints': checkpoint_spp,
            'output_pattern': output_pattern,
            'tone_map_func': apply_tone_map,
            'save_func': save_checkpoint_image
        }

    # Render
    step = 2
    image = render_with_progress(scene,
                                spp_total=view_config['spp'],
                                step_spp=step,
                                seed=view_config.get('seed', 0),
                                checkpoint_config=checkpoint_config)

    # Convert to numpy
    img_np = np.array(image)

    # Tone mapping
    img_np = tone_map(img_np, exposure=view_config.get('exposure', 1.0))

    # Save final image
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
