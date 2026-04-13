"""
witness_cuda.py: CUDA GPU kernels for witness volume ray marching.

Provides GPU-accelerated rendering using numba.cuda. All device functions
use float32 for performance on consumer NVIDIA GPUs. The public
render_image_cuda() function accepts float64 arrays (matching the CPU
interface) and handles conversion internally.
"""

import math
import os
import numpy as np


def _setup_cuda_env():
    """Configure environment so numba.cuda can find pip-installed CUDA toolkit.

    When CUDA toolkit is installed via pip (nvidia-cuda-nvcc-cu12,
    nvidia-cuda-runtime-cu12), the libraries land inside the venv's
    site-packages but numba doesn't know to look there.  This sets
    CUDA_HOME and extends LD_LIBRARY_PATH before numba.cuda is imported.
    """
    try:
        import nvidia.cuda_nvcc as _nvcc_pkg
        nvcc_root = list(_nvcc_pkg.__path__)[0]
    except (ImportError, IndexError):
        return

    # Set CUDA_HOME if not already configured
    if not os.environ.get('CUDA_HOME') and not os.environ.get('NUMBA_CUDA_HOME'):
        nvvm_lib = os.path.join(nvcc_root, 'nvvm', 'lib64')
        if os.path.isdir(nvvm_lib):
            os.environ['CUDA_HOME'] = nvcc_root

            # Ensure versioned symlinks exist (pip ships libnvvm.so without one)
            for lib in ('libnvvm.so',):
                src = os.path.join(nvvm_lib, lib)
                dst = os.path.join(nvvm_lib, lib + '.4')
                if os.path.isfile(src) and not os.path.exists(dst):
                    try:
                        os.symlink(lib, dst)
                    except OSError:
                        pass

    # Always fix up libcudart symlinks regardless of CUDA_HOME setting,
    # since pip splits cuda-nvcc and cuda-runtime into separate packages.
    # Place symlinks in the CUDA_HOME lib64 dir where numba searches.
    cuda_home = os.environ.get('CUDA_HOME') or nvcc_root
    cudalib_dir = os.path.join(cuda_home, 'lib64')
    os.makedirs(cudalib_dir, exist_ok=True)

    try:
        import nvidia.cuda_runtime as _rt_pkg
        rt_lib = os.path.join(list(_rt_pkg.__path__)[0], 'lib')
    except (ImportError, IndexError):
        rt_lib = None

    if rt_lib and os.path.isdir(rt_lib):
        for f in os.listdir(rt_lib):
            if f.startswith('libcudart.so'):
                src = os.path.join(rt_lib, f)
                dst = os.path.join(cudalib_dir, f)
                if not os.path.exists(dst):
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        pass
        # Also ensure an unversioned symlink
        versioned = [f for f in os.listdir(cudalib_dir)
                     if f.startswith('libcudart.so.') and f != 'libcudart.so']
        if versioned and not os.path.exists(os.path.join(cudalib_dir, 'libcudart.so')):
            try:
                os.symlink(versioned[0], os.path.join(cudalib_dir, 'libcudart.so'))
            except OSError:
                pass


_setup_cuda_env()

from numba import cuda  # noqa: E402

if not cuda.is_available():
    raise ImportError("CUDA is not available")


# ============================================================================
# CUDA device functions (float32)
# ============================================================================

@cuda.jit(device=True)
def _trilinear_d(field, gx, gy, gz, nx, ny, nz):
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


@cuda.jit(device=True)
def _ray_box_d(ox, oy, oz, dx, dy, dz,
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


@cuda.jit(device=True)
def _hg_phase_d(cos_theta, g):
    """Henyey-Greenstein phase function."""
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * 3.14159265358979 * denom * math.sqrt(denom))


@cuda.jit(device=True)
def _sample_sigma_d(sigma_world, px, py, pz, ar_x, ar_y, nx, ny, nz):
    """Sample extinction at a world-space position via trilinear interpolation."""
    gx = (px / ar_x + 1.0) * 0.5 * (nx - 1)
    gy = (py / ar_y + 1.0) * 0.5 * (ny - 1)
    gz = (pz + 1.0) * 0.5 * (nz - 1)

    if gx < 0 or gx > nx - 1.001 or gy < 0 or gy > ny - 1.001 or gz < 0 or gz > nz - 1.001:
        return 0.0

    return _trilinear_d(sigma_world, gx, gy, gz, nx, ny, nz)


@cuda.jit(device=True)
def _light_march_d(sigma_world, px, py, pz, sun_dx, sun_dy, sun_dz,
                   ar_x, ar_y, nx, ny, nz, n_steps):
    """March from a point toward the sun. Returns accumulated optical depth."""
    t_near, t_far = _ray_box_d(px, py, pz, sun_dx, sun_dy, sun_dz,
                                -ar_x, -ar_y, -1.0, ar_x, ar_y, 1.0)
    tau = 0.0
    if t_far <= 0:
        return tau

    dt = t_far / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dt
        sx = px + t * sun_dx
        sy = py + t * sun_dy
        sz = pz + t * sun_dz

        sigma = _sample_sigma_d(sigma_world, sx, sy, sz, ar_x, ar_y, nx, ny, nz)
        tau += sigma * dt
        if tau > 80.0:
            break

    return tau


@cuda.jit(device=True)
def _sky_radiance_d(dx, dy, dz, sun_dx, sun_dy, sun_dz):
    """Procedural sky color in HDR for a given view direction."""
    t = max(0.0, min(1.0, dz))

    sky_r = 0.14 * (1.0 - t * 0.4)
    sky_g = 0.19 * (1.0 - t * 0.3)
    sky_b = 0.36 - t * 0.04

    cos_sun = dx * sun_dx + dy * sun_dy + dz * sun_dz
    if cos_sun > 0:
        halo = cos_sun * cos_sun
        sky_r += halo * 0.08
        sky_g += halo * 0.06
        sky_b += halo * 0.03
        if cos_sun > 0.9:
            glow = ((cos_sun - 0.9) / 0.1)
            glow = glow * glow * glow
            sky_r += glow * 0.8
            sky_g += glow * 0.7
            sky_b += glow * 0.5
        if cos_sun > 0.9998:
            sky_r += 50.0
            sky_g += 45.0
            sky_b += 35.0

    if dz < 0:
        fade = max(0.0, 1.0 + dz * 5.0)
        sky_r *= fade
        sky_g *= fade
        sky_b *= fade

    return sky_r, sky_g, sky_b


# ============================================================================
# Main CUDA rendering kernel
# ============================================================================

@cuda.jit
def _render_image_kernel(sigma_world, nx, ny, nz, ar_x, ar_y,
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

    pixel_idx = cuda.grid(1)
    n_pixels = img_w * img_h
    if pixel_idx >= n_pixels:
        return

    aspect = img_w / img_h
    iso_phase = 1.0 / (4.0 * 3.14159265358979)

    py = pixel_idx // img_w
    px = pixel_idx % img_w

    # --- Compute ray direction ---
    ndc_x = (2.0 * (px + 0.5) / img_w - 1.0) * aspect * tan_half_fov
    ndc_y = (1.0 - 2.0 * (py + 0.5) / img_h) * tan_half_fov

    d_x = cam_fx + ndc_x * cam_rx + ndc_y * cam_ux
    d_y = cam_fy + ndc_x * cam_ry + ndc_y * cam_uy
    d_z = cam_fz + ndc_x * cam_rz + ndc_y * cam_uz

    inv_len = 1.0 / math.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
    d_x *= inv_len
    d_y *= inv_len
    d_z *= inv_len

    # --- Ray-box intersection ---
    t_near, t_far = _ray_box_d(cam_ox, cam_oy, cam_oz,
                                d_x, d_y, d_z,
                                -ar_x, -ar_y, -1.0, ar_x, ar_y, 1.0)

    # Ocean intersection
    t_ocean = 1e30
    if ocean_enabled and d_z < -1e-8:
        t_ocean_cand = (ocean_z - cam_oz) / d_z
        if t_ocean_cand > 0:
            t_ocean = t_ocean_cand

    # Phase function
    cos_theta = d_x * sun_dx + d_y * sun_dy + d_z * sun_dz
    phase_hg = _hg_phase_d(cos_theta, g_hg)

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
                o_x = cam_ox + t_ocean * d_x
                o_y = cam_oy + t_ocean * d_y
                o_z = ocean_z

                tau_ocean = _light_march_d(sigma_world, o_x, o_y, o_z,
                                           sun_dx, sun_dy, sun_dz,
                                           ar_x, ar_y, nx, ny, nz, n_light_steps)
                t_sun_ocean = math.exp(-tau_ocean)
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
                break

            # Sample position
            p_x = cam_ox + t * d_x
            p_y = cam_oy + t * d_y
            p_z = cam_oz + t * d_z

            # Sample extinction
            sigma = _sample_sigma_d(sigma_world, p_x, p_y, p_z,
                                    ar_x, ar_y, nx, ny, nz)

            # Adaptive step size
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
            tau_sun = _light_march_d(sigma_world, p_x, p_y, p_z,
                                     sun_dx, sun_dy, sun_dz,
                                     ar_x, ar_y, nx, ny, nz, n_light_steps)

            # --- Multi-scattering in-scattering ---
            ms_r = 0.0
            ms_g = 0.0
            ms_b = 0.0

            ms_atten = 1.0
            for octave in range(6):
                t_sun_ms = math.exp(-tau_sun * ms_atten)

                blend = min(1.0, octave * 0.35)
                oct_phase = phase_hg * (1.0 - blend) + iso_phase * blend

                contrib = ms_atten * t_sun_ms * oct_phase
                ms_r += contrib * sun_r
                ms_g += contrib * sun_g
                ms_b += contrib * sun_b

                ms_atten *= 0.4

            # Powder effect
            powder = 1.0 - math.exp(-1.5 * d_tau)

            # Scattering contribution
            scatter_weight = sigma * dt * powder * transmittance

            col_r += scatter_weight * ms_r
            col_g += scatter_weight * ms_g
            col_b += scatter_weight * ms_b

            # Ambient illumination
            height_frac = (p_z + 1.0) * 0.5
            amb = ambient_strength * (0.3 + 0.7 * height_frac)
            amb_weight = transmittance * sigma * dt * amb

            col_r += amb_weight * 0.22
            col_g += amb_weight * 0.23
            col_b += amb_weight * 0.28

            # Update transmittance
            transmittance *= math.exp(-d_tau)

            t += dt

    # --- Ocean for rays that exit the box ---
    if ocean_enabled and transmittance > 0.002 and t_ocean < 1e29 and t_ocean > t_far:
        o_x = cam_ox + t_ocean * d_x
        o_y = cam_oy + t_ocean * d_y
        o_z = ocean_z

        if abs(o_x) < ar_x * 100 and abs(o_y) < ar_y * 100:
            tau_ocean = _light_march_d(sigma_world, o_x, o_y, o_z,
                                       sun_dx, sun_dy, sun_dz,
                                       ar_x, ar_y, nx, ny, nz, n_light_steps)
            t_sun_ocean = math.exp(-tau_ocean)
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
        sky_r, sky_g, sky_b = _sky_radiance_d(d_x, d_y, d_z,
                                               sun_dx, sun_dy, sun_dz)
        col_r += transmittance * sky_r
        col_g += transmittance * sky_g
        col_b += transmittance * sky_b

    image[py, px, 0] = col_r
    image[py, px, 1] = col_g
    image[py, px, 2] = col_b


# ============================================================================
# Public launch function
# ============================================================================

def render_image_cuda(sigma_world, nx, ny, nz, ar_x, ar_y,
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
    """Render using CUDA GPU. Same interface as CPU _render_image().

    Accepts float64 arrays (matching the CPU path) and converts to float32
    internally for GPU performance. Results are written back into the
    caller's float64 image array.
    """
    # Transfer volume data to GPU as float32
    d_sigma = cuda.to_device(np.ascontiguousarray(sigma_world.astype(np.float32)))

    # Allocate output on GPU
    d_image = cuda.device_array((img_h, img_w, 3), dtype=np.float32)

    # Launch kernel
    n_pixels = img_w * img_h
    threads_per_block = 256
    blocks = (n_pixels + threads_per_block - 1) // threads_per_block

    _render_image_kernel[blocks, threads_per_block](
        d_sigma, nx, ny, nz,
        np.float32(ar_x), np.float32(ar_y),
        np.float32(cam_ox), np.float32(cam_oy), np.float32(cam_oz),
        np.float32(cam_fx), np.float32(cam_fy), np.float32(cam_fz),
        np.float32(cam_rx), np.float32(cam_ry), np.float32(cam_rz),
        np.float32(cam_ux), np.float32(cam_uy), np.float32(cam_uz),
        np.float32(sun_dx), np.float32(sun_dy), np.float32(sun_dz),
        img_w, img_h, np.float32(tan_half_fov),
        n_light_steps,
        np.float32(sun_r), np.float32(sun_g), np.float32(sun_b),
        np.float32(g_hg), np.float32(ambient_strength),
        ocean_enabled, np.float32(ocean_z),
        np.float32(ocean_rr), np.float32(ocean_rg), np.float32(ocean_rb),
        d_image)

    cuda.synchronize()

    # Copy result back and write into caller's float64 image
    image[:] = d_image.copy_to_host().astype(np.float64)
