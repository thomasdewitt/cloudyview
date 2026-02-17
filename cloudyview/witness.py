#!/usr/bin/env python
"""
witness.py: Video game-style cloud visualization via volume ray marching.

Fast, visually realistic cloud rendering using techniques from real-time graphics:
- Volume ray marching through 3D extinction coefficient field
- Beer-Lambert transmittance with adaptive step size
- Sun light marching for volumetric shadows
- Henyey-Greenstein phase function for forward scattering
- Multi-scattering approximation for optically thick clouds
- Powder effect for dense cloud interiors
- Procedural sky gradient with sun glow
- Diffuse ocean surface

API matches behold (Mitsuba path tracer) for easy comparison:
    witness <filename.nc> [--camera-position X Y Z] [--camera-azimuth DEG]
        [--camera-elevation DEG] [--fov DEG] [--sun-azimuth DEG]
        [--sun-elevation DEG] [--size W H] [--output dir]

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction
"""

import argparse
import sys
import time
import math as pymath
from pathlib import Path
import numpy as np

from . import io, optical_depth, config

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper
    prange = range


# ============================================================================
# Numba JIT helper functions
# ============================================================================

@njit
def _trilinear(field, gx, gy, gz, nx, ny, nz):
    """Trilinear interpolation of a 3D field at continuous grid coordinates."""
    ix = int(gx)
    iy = int(gy)
    iz = int(gz)

    fx = gx - ix
    fy = gy - iy
    fz = gz - iz

    ix1 = min(ix + 1, nx - 1)
    iy1 = min(iy + 1, ny - 1)
    iz1 = min(iz + 1, nz - 1)

    c000 = field[ix, iy, iz]
    c100 = field[ix1, iy, iz]
    c010 = field[ix, iy1, iz]
    c110 = field[ix1, iy1, iz]
    c001 = field[ix, iy, iz1]
    c101 = field[ix1, iy, iz1]
    c011 = field[ix, iy1, iz1]
    c111 = field[ix1, iy1, iz1]

    return (c000 * (1 - fx) * (1 - fy) * (1 - fz) +
            c100 * fx * (1 - fy) * (1 - fz) +
            c010 * (1 - fx) * fy * (1 - fz) +
            c110 * fx * fy * (1 - fz) +
            c001 * (1 - fx) * (1 - fy) * fz +
            c101 * fx * (1 - fy) * fz +
            c011 * (1 - fx) * fy * fz +
            c111 * fx * fy * fz)


@njit
def _ray_box(ox, oy, oz, dx, dy, dz,
             bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z):
    """Ray-AABB intersection. Returns (t_near, t_far) or (-1, -1) if miss."""
    t_near = -1e30
    t_far = 1e30

    # X slab
    if abs(dx) < 1e-12:
        if ox < bmin_x or ox > bmax_x:
            return -1.0, -1.0
    else:
        t1 = (bmin_x - ox) / dx
        t2 = (bmax_x - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    # Y slab
    if abs(dy) < 1e-12:
        if oy < bmin_y or oy > bmax_y:
            return -1.0, -1.0
    else:
        t1 = (bmin_y - oy) / dy
        t2 = (bmax_y - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    # Z slab
    if abs(dz) < 1e-12:
        if oz < bmin_z or oz > bmax_z:
            return -1.0, -1.0
    else:
        t1 = (bmin_z - oz) / dz
        t2 = (bmax_z - oz) / dz
        if t1 > t2:
            t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)

    if t_near > t_far or t_far < 0:
        return -1.0, -1.0

    return max(t_near, 0.0), t_far


@njit
def _hg_phase(cos_theta, g):
    """Henyey-Greenstein phase function (normalized to integrate to 1 over sphere)."""
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * 3.14159265358979 * denom * pymath.sqrt(denom))


@njit
def _sample_sigma(sigma_world, px, py, pz, ar, nx, ny, nz):
    """Sample extinction at a world-space position via trilinear interpolation."""
    gx = (px / ar + 1.0) * 0.5 * (nx - 1)
    gy = (py / ar + 1.0) * 0.5 * (ny - 1)
    gz = (pz + 1.0) * 0.5 * (nz - 1)

    if gx < 0 or gx > nx - 1.001 or gy < 0 or gy > ny - 1.001 or gz < 0 or gz > nz - 1.001:
        return 0.0

    return _trilinear(sigma_world, gx, gy, gz, nx, ny, nz)


@njit
def _light_march(sigma_world, px, py, pz, sun_dx, sun_dy, sun_dz,
                 ar, nx, ny, nz, n_steps):
    """March from a point toward the sun. Returns accumulated optical depth."""
    t_near, t_far = _ray_box(px, py, pz, sun_dx, sun_dy, sun_dz,
                              -ar, -ar, -1.0, ar, ar, 1.0)
    tau = 0.0
    if t_far <= 0:
        return tau

    dt = t_far / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dt
        sx = px + t * sun_dx
        sy = py + t * sun_dy
        sz = pz + t * sun_dz

        sigma = _sample_sigma(sigma_world, sx, sy, sz, ar, nx, ny, nz)
        tau += sigma * dt
        if tau > 80.0:
            break

    return tau


@njit
def _sky_radiance(dx, dy, dz, sun_dx, sun_dy, sun_dz):
    """Procedural sky color in HDR for a given view direction."""
    # Height above horizon
    t = max(0.0, min(1.0, dz))

    # Base sky gradient (horizon -> zenith), slightly muted
    sky_r = 0.14 * (1.0 - t * 0.4)
    sky_g = 0.19 * (1.0 - t * 0.3)
    sky_b = 0.36 - t * 0.04

    # Sun proximity glow
    cos_sun = dx * sun_dx + dy * sun_dy + dz * sun_dz
    if cos_sun > 0:
        # Broad halo
        halo = cos_sun * cos_sun
        sky_r += halo * 0.08
        sky_g += halo * 0.06
        sky_b += halo * 0.03
        # Sharp glow near sun disk
        if cos_sun > 0.9:
            glow = ((cos_sun - 0.9) / 0.1)
            glow = glow * glow * glow
            sky_r += glow * 0.8
            sky_g += glow * 0.7
            sky_b += glow * 0.5
        # Sun disk
        if cos_sun > 0.9998:
            sky_r += 50.0
            sky_g += 45.0
            sky_b += 35.0

    # Below horizon: fade to dark
    if dz < 0:
        fade = max(0.0, 1.0 + dz * 5.0)
        sky_r *= fade
        sky_g *= fade
        sky_b *= fade

    return sky_r, sky_g, sky_b


# ============================================================================
# Main rendering kernel
# ============================================================================

@njit(parallel=True)
def _render_image(sigma_world, nx, ny, nz, ar,
                  cam_ox, cam_oy, cam_oz,
                  cam_fx, cam_fy, cam_fz,
                  cam_rx, cam_ry, cam_rz,
                  cam_ux, cam_uy, cam_uz,
                  sun_dx, sun_dy, sun_dz,
                  img_w, img_h, tan_half_fov,
                  n_light_steps,
                  sun_r, sun_g, sun_b,
                  g_hg, ambient_strength,
                  ocean_enabled, ocean_z,
                  ocean_rr, ocean_rg, ocean_rb,
                  image):
    """Render all pixels using volume ray marching with adaptive stepping."""

    aspect = img_w / img_h
    iso_phase = 1.0 / (4.0 * 3.14159265358979)

    n_pixels = img_w * img_h
    for pixel_idx in prange(n_pixels):
        py = pixel_idx // img_w
        px = pixel_idx % img_w

        # --- Compute ray direction ---
        ndc_x = (2.0 * (px + 0.5) / img_w - 1.0) * aspect * tan_half_fov
        ndc_y = (1.0 - 2.0 * (py + 0.5) / img_h) * tan_half_fov

        d_x = cam_fx + ndc_x * cam_rx + ndc_y * cam_ux
        d_y = cam_fy + ndc_x * cam_ry + ndc_y * cam_uy
        d_z = cam_fz + ndc_x * cam_rz + ndc_y * cam_uz

        inv_len = 1.0 / pymath.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
        d_x *= inv_len
        d_y *= inv_len
        d_z *= inv_len

        # --- Ray-box intersection ---
        t_near, t_far = _ray_box(cam_ox, cam_oy, cam_oz,
                                  d_x, d_y, d_z,
                                  -ar, -ar, -1.0, ar, ar, 1.0)

        # Ocean intersection
        t_ocean = 1e30
        if ocean_enabled and d_z < -1e-8:
            t_ocean_cand = (ocean_z - cam_oz) / d_z
            if t_ocean_cand > 0:
                t_ocean = t_ocean_cand

        # Phase function (constant per pixel since it depends on view-sun angle)
        cos_theta = d_x * sun_dx + d_y * sun_dy + d_z * sun_dz
        phase_hg = _hg_phase(cos_theta, g_hg)

        # --- Initialize accumulation ---
        col_r = 0.0
        col_g = 0.0
        col_b = 0.0
        transmittance = 1.0

        # --- Volume ray march ---
        if t_near >= 0 and t_near < t_far:
            dt_max = (t_far - t_near) / 300.0
            t = t_near

            for _ in range(2048):
                if t >= t_far or transmittance < 0.002:
                    break

                # Check ocean
                if ocean_enabled and t >= t_ocean:
                    # Compute ocean contribution
                    o_x = cam_ox + t_ocean * d_x
                    o_y = cam_oy + t_ocean * d_y
                    o_z = ocean_z

                    tau_ocean = _light_march(sigma_world, o_x, o_y, o_z,
                                             sun_dx, sun_dy, sun_dz,
                                             ar, nx, ny, nz, n_light_steps)
                    t_sun_ocean = pymath.exp(-tau_ocean)
                    cos_sun_n = max(0.0, sun_dz)  # dot(surface_normal, sun_dir)

                    # Ocean surface lighting (Lambertian BRDF = reflectance/pi)
                    # Scale by 0.3 for atmospheric attenuation matching behold
                    inv_pi = 1.0 / 3.14159265358979
                    ocean_sun_scale = 0.05
                    sun_irr = t_sun_ocean * cos_sun_n * ocean_sun_scale * inv_pi
                    ol_r = sun_irr * sun_r * ocean_rr + 0.004
                    ol_g = sun_irr * sun_g * ocean_rg + 0.004
                    ol_b = sun_irr * sun_b * ocean_rb + 0.006

                    col_r += transmittance * ol_r
                    col_g += transmittance * ol_g
                    col_b += transmittance * ol_b
                    transmittance = 0.0
                    break

                # Sample position
                p_x = cam_ox + t * d_x
                p_y = cam_oy + t * d_y
                p_z = cam_oz + t * d_z

                # Sample extinction
                sigma = _sample_sigma(sigma_world, p_x, p_y, p_z,
                                      ar, nx, ny, nz)

                # Adaptive step size: limit optical depth per step to ~0.5
                if sigma > 0.01:
                    dt = min(dt_max, 0.5 / sigma)
                else:
                    dt = dt_max

                # Don't overshoot box or ocean
                dt = min(dt, t_far - t)
                if ocean_enabled and t + dt > t_ocean:
                    dt = max(0.0001, t_ocean - t)

                if sigma < 0.001:
                    t += dt
                    continue

                d_tau = sigma * dt

                # --- Sun light march ---
                tau_sun = _light_march(sigma_world, p_x, p_y, p_z,
                                       sun_dx, sun_dy, sun_dz,
                                       ar, nx, ny, nz, n_light_steps)

                # --- Multi-scattering in-scattering ---
                # Schneider/Hillaire approximation: each octave represents
                # higher-order scattering with reduced extinction and more
                # isotropic phase, allowing light to penetrate deeper
                ms_r = 0.0
                ms_g = 0.0
                ms_b = 0.0

                ms_atten = 1.0
                for octave in range(6):
                    # Reduced sun transmittance for multi-scattered light
                    t_sun_ms = pymath.exp(-tau_sun * ms_atten)

                    # Blend phase function toward isotropic
                    blend = min(1.0, octave * 0.35)
                    oct_phase = phase_hg * (1.0 - blend) + iso_phase * blend

                    contrib = ms_atten * t_sun_ms * oct_phase
                    ms_r += contrib * sun_r
                    ms_g += contrib * sun_g
                    ms_b += contrib * sun_b

                    ms_atten *= 0.4

                # Powder effect: darkens dense interiors, brightens thin edges
                powder = 1.0 - pymath.exp(-1.5 * d_tau)

                # Scattering contribution
                scatter_weight = sigma * dt * powder * transmittance

                col_r += scatter_weight * ms_r
                col_g += scatter_weight * ms_g
                col_b += scatter_weight * ms_b

                # Ambient illumination (height-dependent)
                # Uses warm gray (multi-scattered sunlight becomes neutral)
                # Higher minimum at cloud base for ground-bounce light
                height_frac = (p_z + 1.0) * 0.5
                amb = ambient_strength * (0.3 + 0.7 * height_frac)
                amb_weight = transmittance * sigma * dt * amb

                col_r += amb_weight * 0.22
                col_g += amb_weight * 0.23
                col_b += amb_weight * 0.28

                # Update transmittance
                transmittance *= pymath.exp(-d_tau)

                t += dt

        # --- Ocean for rays that exit the box ---
        if ocean_enabled and transmittance > 0.002 and t_ocean < 1e29 and t_ocean > t_far:
            o_x = cam_ox + t_ocean * d_x
            o_y = cam_oy + t_ocean * d_y
            o_z = ocean_z

            # Only if not too far away
            if abs(o_x) < ar * 100 and abs(o_y) < ar * 100:
                tau_ocean = _light_march(sigma_world, o_x, o_y, o_z,
                                         sun_dx, sun_dy, sun_dz,
                                         ar, nx, ny, nz, n_light_steps)
                t_sun_ocean = pymath.exp(-tau_ocean)
                cos_sun_n = max(0.0, sun_dz)

                inv_pi = 1.0 / 3.14159265358979
                ocean_sun_scale = 0.05
                sun_irr = t_sun_ocean * cos_sun_n * ocean_sun_scale * inv_pi
                ol_r = sun_irr * sun_r * ocean_rr + 0.004
                ol_g = sun_irr * sun_g * ocean_rg + 0.004
                ol_b = sun_irr * sun_b * ocean_rb + 0.006

                col_r += transmittance * ol_r
                col_g += transmittance * ol_g
                col_b += transmittance * ol_b
                transmittance = 0.0

        # --- Background sky ---
        if transmittance > 0.002:
            sky_r, sky_g, sky_b = _sky_radiance(d_x, d_y, d_z,
                                                 sun_dx, sun_dy, sun_dz)
            col_r += transmittance * sky_r
            col_g += transmittance * sky_g
            col_b += transmittance * sky_b

        image[py, px, 0] = col_r
        image[py, px, 1] = col_g
        image[py, px, 2] = col_b


# ============================================================================
# Tone mapping
# ============================================================================

def tone_map(image, exposure=4.0, gamma=1.4):
    """Reinhard tone mapping with gamma correction (matches behold)."""
    exposed = image * exposure
    tone_mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)


# ============================================================================
# Main function
# ============================================================================

def main(filename: str, output: str = None,
         camera_position: list = None, camera_azimuth: float = None,
         camera_elevation: float = None, camera_fov: float = None,
         sun_azimuth: float = None, sun_elevation: float = None,
         custom_size: tuple = None) -> None:
    """
    Main function for witness.py

    Parameters
    ----------
    filename : str
        Path to NetCDF file
    output : str, optional
        Output directory for renders
    camera_position : list, optional
        Camera position [x, y, z] in relative coords (+-1.0 = domain edge)
    camera_azimuth : float, optional
        Camera azimuth in degrees (0=East, 90=North)
    camera_elevation : float, optional
        Camera elevation in degrees (angle above horizon)
    camera_fov : float, optional
        Camera field of view in degrees
    sun_azimuth : float, optional
        Sun azimuth in degrees
    sun_elevation : float, optional
        Sun elevation in degrees
    custom_size : tuple, optional
        Image size (width, height)
    """
    import netCDF4 as nc

    print(f"CloudyView Witness: Loading {filename}")
    start_time = time.perf_counter()

    # Load configuration
    witness_config = config.get_witness_config()
    cam_config = witness_config['camera']
    sun_config = witness_config['sun']
    render_config = witness_config['rendering']

    # Apply CLI overrides
    if camera_position is not None:
        cam_config['position'] = list(camera_position)
    if camera_azimuth is not None:
        cam_config['azimuth'] = camera_azimuth
    if camera_elevation is not None:
        cam_config['elevation'] = camera_elevation
    if camera_fov is not None:
        cam_config['fov'] = camera_fov
    if sun_azimuth is not None:
        sun_config['azimuth'] = sun_azimuth
    if sun_elevation is not None:
        sun_config['elevation'] = sun_elevation

    img_w = custom_size[0] if custom_size else render_config['width']
    img_h = custom_size[1] if custom_size else render_config['height']

    try:
        # Load and validate data
        data_dict = io.load_and_validate(filename)
        lw_data = data_dict['liquid_water_data']
        iw_data = data_dict['ice_water_data']

        # Get coordinates
        x_coord = data_dict.get('x_coord')
        y_coord = data_dict.get('y_coord')
        z_coord = data_dict.get('z_coord')

        if x_coord is None or y_coord is None or z_coord is None:
            ds_nc = nc.Dataset(filename, 'r')
            try:
                if x_coord is None and 'x' in ds_nc.variables:
                    x_coord = ds_nc.variables['x'][:]
                if y_coord is None and 'y' in ds_nc.variables:
                    y_coord = ds_nc.variables['y'][:]
                if z_coord is None and 'z' in ds_nc.variables:
                    z_coord = ds_nc.variables['z'][:]
            finally:
                ds_nc.close()

        lw_np = lw_data.values
        if 'time' in lw_data.dims:
            lw_np = lw_np[0]

        nx_d, ny_d, nz_d = lw_np.shape
        if x_coord is None:
            x_coord = np.arange(nx_d, dtype=np.float64)
        if y_coord is None:
            y_coord = np.arange(ny_d, dtype=np.float64)
        if z_coord is None:
            z_coord = np.arange(nz_d, dtype=np.float64)

        dx = float(x_coord[1] - x_coord[0]) if len(x_coord) > 1 else 1.0
        dy = float(y_coord[1] - y_coord[0]) if len(y_coord) > 1 else 1.0
        dz = float(z_coord[1] - z_coord[0]) if len(z_coord) > 1 else 1.0

        # Process ice water content
        iw_np = None
        if iw_data is not None:
            iw_np = iw_data.values
            if 'time' in iw_data.dims:
                iw_np = iw_np[0]
            if np.max(iw_np) < 1e-6:
                iw_np = None

        # Compute extinction coefficient
        sigma_ext = optical_depth.compute_extinction_field(
            lw_np, z_coord, re=10.0, iwc=iw_np, re_ice=30.0)

        # Domain geometry
        width_x = nx_d * dx
        width_y = ny_d * dy
        height_z = nz_d * dz
        ar = width_x / height_z  # aspect ratio

        print(f"  Grid: {nx_d} x {ny_d} x {nz_d}, spacing: {dx:.1f} x {dy:.1f} x {dz:.1f} m")
        print(f"  Domain: {width_x:.0f} x {width_y:.0f} x {height_z:.0f} m, aspect ratio: {ar:.2f}")

        # Scale extinction to world space (same as behold)
        ext_mult = render_config['extinction_multiplier']
        sigma_world = (sigma_ext * ext_mult * height_z).astype(np.float64)
        sigma_world = np.ascontiguousarray(sigma_world)

        sigma_max = np.max(sigma_world)
        sigma_mean_nz = np.mean(sigma_world[sigma_world > 0]) if np.any(sigma_world > 0) else 0
        print(f"  Extinction (world): max={sigma_max:.1f}, mean(nonzero)={sigma_mean_nz:.1f}")

        # Camera setup
        rel_pos = cam_config['position']
        cam_origin = np.array([
            rel_pos[0] * ar,
            rel_pos[1] * ar,
            rel_pos[2]
        ])

        az_rad = np.deg2rad(cam_config['azimuth'])
        el_rad = np.deg2rad(cam_config['elevation'])
        forward = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # Sun direction (toward the sun)
        sun_az_rad = np.deg2rad(sun_config['azimuth'])
        sun_el_rad = np.deg2rad(sun_config['elevation'])
        sun_dir = np.array([
            np.cos(sun_el_rad) * np.cos(sun_az_rad),
            np.cos(sun_el_rad) * np.sin(sun_az_rad),
            np.sin(sun_el_rad)
        ])
        sun_dir /= np.linalg.norm(sun_dir)

        fov_rad = np.deg2rad(cam_config['fov'])
        tan_half_fov = np.tan(fov_rad * 0.5)

        # Rendering parameters
        n_light_steps = render_config['n_light_steps']
        exposure = render_config['exposure']
        g_hg = 0.76  # HG asymmetry parameter
        ambient_strength = 0.12
        sun_color = np.array([22.0, 21.0, 17.0])  # HDR sun intensity

        # Ocean (override reflectance to neutral gray for video-game style)
        ocean_config = render_config['ocean']
        ocean_enabled = ocean_config['enabled']
        ocean_ref = [0.04, 0.05, 0.07]  # Neutral dark gray-blue
        ocean_height = ocean_config['height']

        print(f"  Camera: pos=[{cam_origin[0]:.2f}, {cam_origin[1]:.2f}, {cam_origin[2]:.2f}]")
        print(f"          azimuth={cam_config['azimuth']:.1f} elev={cam_config['elevation']:.1f} fov={cam_config['fov']:.1f}")
        print(f"  Sun: azimuth={sun_config['azimuth']:.1f} elev={sun_config['elevation']:.1f}")
        print(f"  Image: {img_w}x{img_h}")

        # Allocate output
        image = np.zeros((img_h, img_w, 3), dtype=np.float64)

        # Warmup numba compilation
        if HAS_NUMBA:
            print("  Compiling render kernel (first run only)...", end="", flush=True)
            warmup = np.zeros((1, 1, 3), dtype=np.float64)
            _render_image(sigma_world, nx_d, ny_d, nz_d, ar,
                          cam_origin[0], cam_origin[1], cam_origin[2],
                          forward[0], forward[1], forward[2],
                          right[0], right[1], right[2],
                          up[0], up[1], up[2],
                          sun_dir[0], sun_dir[1], sun_dir[2],
                          1, 1, tan_half_fov,
                          4, sun_color[0], sun_color[1], sun_color[2],
                          g_hg, ambient_strength,
                          ocean_enabled, ocean_height,
                          ocean_ref[0], ocean_ref[1], ocean_ref[2],
                          warmup)
            print(" done")

        # Render
        print("  Rendering...", end="", flush=True)
        render_start = time.perf_counter()

        _render_image(sigma_world, nx_d, ny_d, nz_d, ar,
                      cam_origin[0], cam_origin[1], cam_origin[2],
                      forward[0], forward[1], forward[2],
                      right[0], right[1], right[2],
                      up[0], up[1], up[2],
                      sun_dir[0], sun_dir[1], sun_dir[2],
                      img_w, img_h, tan_half_fov,
                      n_light_steps,
                      sun_color[0], sun_color[1], sun_color[2],
                      g_hg, ambient_strength,
                      ocean_enabled, ocean_height,
                      ocean_ref[0], ocean_ref[1], ocean_ref[2],
                      image)

        render_elapsed = time.perf_counter() - render_start
        print(f" done ({render_elapsed:.1f}s)")

        # Tone mapping
        image_tm = tone_map(image, exposure=exposure)

        # Save
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(".")

        dataset_name = Path(filename).stem
        output_file = output_dir / f"witness_{dataset_name}.png"

        from PIL import Image as PILImage
        img_uint8 = (np.clip(image_tm, 0, 1) * 255).astype(np.uint8)
        PILImage.fromarray(img_uint8).save(str(output_file))
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
        description="Cloud visualization with volume ray marching (video game-style rendering)"
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
                        help="Camera position in relative coords (default: 0 0 -0.9)")
    parser.add_argument("--camera-azimuth", type=float,
                        help="Camera azimuth in degrees (default: 90)")
    parser.add_argument("--camera-elevation", type=float,
                        help="Camera elevation in degrees (default: 35)")
    parser.add_argument("--fov", type=float,
                        help="Camera field of view in degrees (default: 100)")
    parser.add_argument("--sun-azimuth", type=float,
                        help="Sun azimuth in degrees (default: 70)")
    parser.add_argument("--sun-elevation", type=float,
                        help="Sun elevation in degrees (default: 55)")
    parser.add_argument("--size", type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        help="Image size in pixels (overrides quality preset)")

    args = parser.parse_args()
    if args.size:
        size = tuple(args.size)
    else:
        size = QUALITY_PRESETS[args.quality]

    main(args.filename, args.output,
         camera_position=args.camera_position,
         camera_azimuth=args.camera_azimuth,
         camera_elevation=args.camera_elevation,
         camera_fov=args.fov,
         sun_azimuth=args.sun_azimuth,
         sun_elevation=args.sun_elevation,
         custom_size=size)


if __name__ == "__main__":
    cli()
