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
from typing import Dict, Optional, Literal
from .angles import direction_from_azimuth_elevation

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

    Ice phase tables must exist in Mie_tables/ as
    IceMie0_normalized.txt and IceMiePF3_normalized.txt.

    Parameters
    ----------
    channel : {'R', 'G', 'B', 'gray'}
        Which wavelength channel to use (for future compatibility)

    Returns
    -------
    ice_mie0_str : str
        Forward peak phase function values (comma-separated)
    ice_mie_pf3_str : str
        Rest of distribution phase function values (comma-separated)

    References
    ----------
    Ice scattering properties from Yang et al. (2000, 2013) databases
    """
    # Find the Mie_tables directory relative to this module
    module_dir = Path(__file__).parent
    mie_dir = module_dir / 'Mie_tables'

    # Check that ice tables exist
    ice_mie0_path = mie_dir / 'IceMie0_normalized.txt'
    ice_mie_pf3_path = mie_dir / 'IceMiePF3_normalized.txt'
    if not ice_mie0_path.exists() or not ice_mie_pf3_path.exists():
        raise FileNotFoundError(
            f"Ice Mie phase tables not found in {mie_dir}. "
            "Expected IceMie0_normalized.txt and IceMiePF3_normalized.txt"
        )

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

    return ice_mie0_str, ice_mie_pf3_str


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
        Samples per rendering step (also controls progress print frequency)
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
        try:
            print(f"\r  {pct:3.0f}% | {taken}/{spp_total} spp | Elapsed: {elapsed_str} | ETA: {eta_str}",
                  end="", flush=True)
        except OSError:
            pass  # Ignore NFS stale file handle errors

    try:
        print()  # newline after finishing
    except OSError:
        pass
    return acc / taken


def sun_direction_to_scene(azimuth_deg=0.0, elevation_deg=90.0):
    """
    Return unit vector pointing FROM the sun TOWARD the scene.

    Azimuth uses meteorological convention: 0°=North, 90°=East, clockwise.
    """
    dir_to_sun = direction_from_azimuth_elevation(azimuth_deg, elevation_deg)
    direction = np.array([dir_to_sun[0], dir_to_sun[1], -dir_to_sun[2]])
    direction /= np.linalg.norm(direction)
    return direction.tolist()


def sun_direction_to_sun(azimuth_deg=0.0, elevation_deg=90.0):
    """
    Return unit vector pointing FROM scene TO the sun.

    Azimuth uses meteorological convention: 0°=North, 90°=East, clockwise.
    """
    return direction_from_azimuth_elevation(azimuth_deg, elevation_deg).tolist()


def create_mitsuba_scene(sigma_ext, dx, dy, dz, camera_config, spp=256,
                        use_mie_phase=True, mie_channel='gray', mie_blend_weight=0.5,
                        wavelength_nm=None, ice_fraction=None,
                        ar_x=None, ar_y=None, height_z=None):
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
        - 'sun_azimuth': azimuth angle in degrees (0=N, 90=E, 180=S, 270=W)
        - 'sun_elevation': elevation angle in degrees
    spp : int
        Samples per pixel
    use_mie_phase : bool
        Must be True. Henyey-Greenstein fallback has been removed.
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
    ar_x : float, optional
        Aspect ratio width_x / height_z. If None, computed from nx*dx / (nz*dz).
    ar_y : float, optional
        Aspect ratio width_y / height_z. If None, computed from ny*dy / (nz*dz).
    height_z : float, optional
        Total vertical extent in metres. If None, computed as nz*dz.

    Returns
    --------
    scene : mi.Scene
        Mitsuba scene ready to render
    """
    nx, ny, nz = sigma_ext.shape
    if not use_mie_phase:
        raise ValueError("Henyey-Greenstein fallback is disabled; Mie phase tables are required.")

    # Domain geometry — use caller-provided values or fall back to uniform-dz estimate
    if height_z is None:
        height_z = nz * dz
    if ar_x is None:
        ar_x = (nx * dx) / height_z
    if ar_y is None:
        ar_y = (ny * dy) / height_z

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
    # World space: [-ar_x, ar_x] x [-ar_y, ar_y] x [-1, 1]
    cube_transform = mi.ScalarTransform4f.scale([ar_x, ar_y, 1.0])

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
        if (-ar_x - eps <= cam_x <= ar_x + eps and
                -ar_y - eps <= cam_y <= ar_y + eps and
                -1.0 - eps <= cam_z <= 1.0 + eps):
            camera_inside = True

    integrator_type = camera_config.get('integrator', 'volpath')
    max_depth = camera_config.get('max_depth', 32)
    rr_depth = camera_config.get('rr_depth', 5)
    sampler_cfg = camera_config.get('sampler', {
        'type': 'independent',
        'sample_count': spp,
    })
    if not isinstance(sampler_cfg, dict):
        raise ValueError(f"camera_config['sampler'] must be a dict, got {type(sampler_cfg).__name__}")
    sampler_type = str(sampler_cfg.get('type', 'independent')).lower()
    if sampler_type == 'sobol':
        raise NotImplementedError(
            "Sobol sampler is not supported yet. "
            "Use sampler type 'independent' for now."
        )
    if 'sample_count' not in sampler_cfg:
        sampler_cfg = dict(sampler_cfg)
        sampler_cfg['sample_count'] = spp

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
            'sampler': sampler_cfg,
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

    # Configure phase function (Mie tables only)
    # If ice_fraction is provided, create spatially-varying blend between liquid and ice phases
    if ice_fraction is not None:
        print("  Setting up mixed-phase (liquid/ice) rendering...")

        # Prepare ice fraction grid for Mitsuba (same transform as extinction)
        ice_fraction_data = ice_fraction[..., np.newaxis].astype(np.float32)
        ice_fraction_data = np.ascontiguousarray(
            np.transpose(ice_fraction_data, (2, 1, 0, 3))
        )
        ice_fraction_grid = mi.VolumeGrid(ice_fraction_data)

        # Load liquid water Mie tables
        mie0_str, mie_pf3_str = load_mie_phase_tables(channel=mie_channel)

        # Load ice Mie tables (required)
        ice_mie0_str, ice_mie_pf3_str = load_ice_phase_tables(channel=mie_channel)

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

        # Create ice phase (dual-table blended)
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
        print(f"  Ice phase: Mie tables (channel={mie_channel})")
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
        ocean_size_x = ar_x * ocean_size_multiplier
        ocean_size_y = ar_y * ocean_size_multiplier
        ocean_reflectance = camera_config.get('ocean_reflectance', [0.2, 0.3, 0.4])

        ocean_transform = (
            mi.ScalarTransform4f.translate([0.0, 0.0, ocean_height]) @
            mi.ScalarTransform4f.scale([ocean_size_x, ocean_size_y, 1.0])
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
        View configuration (camera position, sky, etc.).
        May include 'ar_x', 'ar_y', 'height_z' for proper domain geometry.
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
    print("  RGB mode: single RGB render with luminance-weighted Mie phase")
    return render_view_single(sigma_ext, dx, dy, dz, view_config,
                              output_file, checkpoint_spp, ice_fraction)



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

    Internal function called by render_view.
    """
    # Create scene
    scene = create_mitsuba_scene(
        sigma_ext, dx, dy, dz,
        view_config,
        spp=view_config['spp'],
        ice_fraction=ice_fraction,
        ar_x=view_config.get('ar_x'),
        ar_y=view_config.get('ar_y'),
        height_z=view_config.get('height_z'),
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
    step = view_config.get('progress_interval', 2)
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
