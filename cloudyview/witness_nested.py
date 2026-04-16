#!/usr/bin/env python
"""
witness_nested.py: Prototype nested-domain extension of witness.

Ray-marches through N strictly-nested extinction grids (finest..coarsest),
picking the finest level that covers each sample point. Works in absolute
world meters end-to-end so levels at wildly different scales compose naturally.

Seams at level boundaries are accepted for now — the prototype does not feather
or zero-out coarser grids inside finer regions.

Conventions:
  - Coordinates: absolute meters. +x east, +y north, +z up.
  - Level ordering: index 0 is the finest, index N-1 the coarsest.
  - Ocean surface: z=0 (matches the .nc domain_z_min=0 convention).

Public entry point:
  render_nested(levels, camera, sun, image_size, ...) -> np.ndarray (tone-mapped, 0..1)

where `levels` is a list of NestedLevel and `camera`/`sun` are dicts with
absolute-meter camera and unit-vector sun parameters.
"""

from __future__ import annotations

import math as pymath
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator
    prange = range


@dataclass
class NestedLevel:
    """One refinement level: extinction field plus its absolute-meter AABB."""
    sigma: np.ndarray        # (nx, ny, nz) float64, m^-1 (already ext_mult-scaled)
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
# Numba kernel helpers
# ============================================================================

@njit(inline="always")
def _active_level(px, py, pz, n_levels, level_bmin, level_bmax):
    """Return index of finest level containing (px, py, pz), or -1."""
    for k in range(n_levels):
        if (level_bmin[k, 0] <= px <= level_bmax[k, 0] and
            level_bmin[k, 1] <= py <= level_bmax[k, 1] and
            level_bmin[k, 2] <= pz <= level_bmax[k, 2]):
            return k
    return -1


@njit(inline="always")
def _sample_sigma_level(sigma_stacked, offset, nx, ny, nz,
                        bmin_x, bmin_y, bmin_z,
                        dx, dy, dz,
                        px, py, pz):
    """Trilinear sigma sample at level k. Returns 0 if out of bounds."""
    gx = (px - bmin_x) / dx
    gy = (py - bmin_y) / dy
    gz = (pz - bmin_z) / dz

    if gx < 0 or gx > nx - 1.001 or gy < 0 or gy > ny - 1.001 or gz < 0 or gz > nz - 1.001:
        return 0.0

    ix = int(gx); iy = int(gy); iz = int(gz)
    fx = gx - ix; fy = gy - iy; fz = gz - iz
    ix1 = ix + 1 if ix + 1 < nx else nx - 1
    iy1 = iy + 1 if iy + 1 < ny else ny - 1
    iz1 = iz + 1 if iz + 1 < nz else nz - 1

    stride_x = ny * nz
    stride_y = nz
    base = offset + ix * stride_x + iy * stride_y
    base1 = offset + ix1 * stride_x + iy * stride_y
    base10 = offset + ix * stride_x + iy1 * stride_y
    base11 = offset + ix1 * stride_x + iy1 * stride_y

    c000 = sigma_stacked[base + iz]
    c100 = sigma_stacked[base1 + iz]
    c010 = sigma_stacked[base10 + iz]
    c110 = sigma_stacked[base11 + iz]
    c001 = sigma_stacked[base + iz1]
    c101 = sigma_stacked[base1 + iz1]
    c011 = sigma_stacked[base10 + iz1]
    c111 = sigma_stacked[base11 + iz1]

    return (c000 * (1 - fx) * (1 - fy) * (1 - fz) +
            c100 * fx * (1 - fy) * (1 - fz) +
            c010 * (1 - fx) * fy * (1 - fz) +
            c110 * fx * fy * (1 - fz) +
            c001 * (1 - fx) * (1 - fy) * fz +
            c101 * fx * (1 - fy) * fz +
            c011 * (1 - fx) * fy * fz +
            c111 * fx * fy * fz)


@njit(inline="always")
def _sample_sigma_nested(px, py, pz,
                         sigma_stacked, level_offsets, level_dims,
                         level_bmin, level_bmax, level_dxs,
                         n_levels):
    """Find finest level covering (px,py,pz) and sample it. 0 if outside all."""
    k = _active_level(px, py, pz, n_levels, level_bmin, level_bmax)
    if k < 0:
        return 0.0, -1
    sigma = _sample_sigma_level(
        sigma_stacked, level_offsets[k],
        level_dims[k, 0], level_dims[k, 1], level_dims[k, 2],
        level_bmin[k, 0], level_bmin[k, 1], level_bmin[k, 2],
        level_dxs[k, 0], level_dxs[k, 1], level_dxs[k, 2],
        px, py, pz,
    )
    return sigma, k


@njit(inline="always")
def _ray_box(ox, oy, oz, dx, dy, dz,
             bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z):
    """Ray-AABB intersection. Returns (t_near, t_far) or (-1,-1) if miss."""
    t_near = -1e30
    t_far = 1e30

    if abs(dx) < 1e-12:
        if ox < bmin_x or ox > bmax_x:
            return -1.0, -1.0
    else:
        t1 = (bmin_x - ox) / dx
        t2 = (bmax_x - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near: t_near = t1
        if t2 < t_far: t_far = t2

    if abs(dy) < 1e-12:
        if oy < bmin_y or oy > bmax_y:
            return -1.0, -1.0
    else:
        t1 = (bmin_y - oy) / dy
        t2 = (bmax_y - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near: t_near = t1
        if t2 < t_far: t_far = t2

    if abs(dz) < 1e-12:
        if oz < bmin_z or oz > bmax_z:
            return -1.0, -1.0
    else:
        t1 = (bmin_z - oz) / dz
        t2 = (bmax_z - oz) / dz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_near: t_near = t1
        if t2 < t_far: t_far = t2

    if t_near > t_far or t_far < 0:
        return -1.0, -1.0
    if t_near < 0:
        t_near = 0.0
    return t_near, t_far


@njit(inline="always")
def _hg_phase(cos_theta, g):
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * 3.14159265358979 * denom * pymath.sqrt(denom))


@njit
def _light_march_nested(px, py, pz, sun_dx, sun_dy, sun_dz,
                        sigma_stacked, level_offsets, level_dims,
                        level_bmin, level_bmax, level_dxs,
                        n_levels, n_steps,
                        outer_bmin_x, outer_bmin_y, outer_bmin_z,
                        outer_bmax_x, outer_bmax_y, outer_bmax_z):
    """March from a point toward the sun through nested levels."""
    t_near, t_far = _ray_box(px, py, pz, sun_dx, sun_dy, sun_dz,
                              outer_bmin_x, outer_bmin_y, outer_bmin_z,
                              outer_bmax_x, outer_bmax_y, outer_bmax_z)
    tau = 0.0
    if t_far <= 0:
        return tau

    dt = t_far / n_steps
    for i in range(n_steps):
        t = (i + 0.5) * dt
        sx = px + t * sun_dx
        sy = py + t * sun_dy
        sz = pz + t * sun_dz

        sigma, _ = _sample_sigma_nested(
            sx, sy, sz,
            sigma_stacked, level_offsets, level_dims,
            level_bmin, level_bmax, level_dxs, n_levels,
        )
        tau += sigma * dt
        if tau > 80.0:
            break

    return tau


@njit(inline="always")
def _sky_radiance(dx, dy, dz, sun_dx, sun_dy, sun_dz):
    """Procedural sky color in HDR."""
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
# Main render kernel
# ============================================================================

@njit(parallel=True)
def _render_image_nested(
    sigma_stacked, level_offsets, level_dims,
    level_bmin, level_bmax, level_dxs, n_levels,
    outer_bmin_x, outer_bmin_y, outer_bmin_z,
    outer_bmax_x, outer_bmax_y, outer_bmax_z,
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
    step_voxel_factor,   # dt_max = min(level_dx) * this
    max_steps,
    image,
):
    aspect = img_w / img_h
    iso_phase = 1.0 / (4.0 * 3.14159265358979)

    n_pixels = img_w * img_h
    for pixel_idx in prange(n_pixels):
        py = pixel_idx // img_w
        px = pixel_idx % img_w

        ndc_x = (2.0 * (px + 0.5) / img_w - 1.0) * aspect * tan_half_fov
        ndc_y = (1.0 - 2.0 * (py + 0.5) / img_h) * tan_half_fov

        d_x = cam_fx + ndc_x * cam_rx + ndc_y * cam_ux
        d_y = cam_fy + ndc_x * cam_ry + ndc_y * cam_uy
        d_z = cam_fz + ndc_x * cam_rz + ndc_y * cam_uz

        inv_len = 1.0 / pymath.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
        d_x *= inv_len
        d_y *= inv_len
        d_z *= inv_len

        # Entry into outermost (coarsest) volume.
        t_near, t_far = _ray_box(cam_ox, cam_oy, cam_oz,
                                  d_x, d_y, d_z,
                                  outer_bmin_x, outer_bmin_y, outer_bmin_z,
                                  outer_bmax_x, outer_bmax_y, outer_bmax_z)

        t_ocean = 1e30
        if ocean_enabled and d_z < -1e-8:
            t_ocean_cand = (ocean_z - cam_oz) / d_z
            if t_ocean_cand > 0:
                t_ocean = t_ocean_cand

        cos_theta = d_x * sun_dx + d_y * sun_dy + d_z * sun_dz
        phase_hg = _hg_phase(cos_theta, g_hg)

        col_r = 0.0
        col_g = 0.0
        col_b = 0.0
        transmittance = 1.0

        # Cumulative optical depth from the most recent cloud entry.
        # Powder is evaluated as a function of this, not per-step d_tau, so
        # the scatter integral is dt-invariant: same physical cloud gives the
        # same brightness whether sampled at r2's ~10 m or r0's ~1 km step.
        # For a single domain at the old adaptive d_tau~=0.5 per step, this
        # still asymptotes to the old thick-cloud behaviour (~sigma*T per unit
        # tau); the constant differs slightly from the old dt-dependent form
        # and is absorbable into exposure / ext_mult.
        tau_depth = 0.0

        if t_near >= 0 and t_near < t_far:
            t = t_near

            for _ in range(max_steps):
                # Ocean hit. Tested before t_far because the STEAM domain has
                # ocean_z == outer_bmin_z (both at z=0), so t_ocean == t_far for
                # downward rays — checking t_far first would terminate the ray
                # without rendering ocean.
                if ocean_enabled and t >= t_ocean:
                    o_x = cam_ox + t_ocean * d_x
                    o_y = cam_oy + t_ocean * d_y
                    o_z = ocean_z

                    tau_ocean = _light_march_nested(
                        o_x, o_y, o_z, sun_dx, sun_dy, sun_dz,
                        sigma_stacked, level_offsets, level_dims,
                        level_bmin, level_bmax, level_dxs, n_levels,
                        n_light_steps,
                        outer_bmin_x, outer_bmin_y, outer_bmin_z,
                        outer_bmax_x, outer_bmax_y, outer_bmax_z,
                    )
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
                    break

                if t >= t_far or transmittance < 0.002:
                    break

                p_x = cam_ox + t * d_x
                p_y = cam_oy + t * d_y
                p_z = cam_oz + t * d_z

                sigma, k = _sample_sigma_nested(
                    p_x, p_y, p_z,
                    sigma_stacked, level_offsets, level_dims,
                    level_bmin, level_bmax, level_dxs, n_levels,
                )

                if k < 0:
                    # Shouldn't happen while inside outer AABB, but bail safely.
                    break

                # dt_max from current level's finest spacing.
                dx_k = level_dxs[k, 0]
                if level_dxs[k, 1] < dx_k:
                    dx_k = level_dxs[k, 1]
                if level_dxs[k, 2] < dx_k:
                    dx_k = level_dxs[k, 2]
                dt_max = dx_k * step_voxel_factor

                if sigma > 0.01:
                    dt = min(dt_max, 0.5 / sigma)
                else:
                    dt = dt_max

                if t + dt > t_far:
                    dt = t_far - t
                if ocean_enabled and t + dt > t_ocean:
                    dt = max(0.0001, t_ocean - t)

                # Skip pure-empty cells cheaply and reset the cloud-entry depth
                # so the next cloud edge gets the full powder ramp.
                d_tau = sigma * dt
                if d_tau < 1e-5:
                    tau_depth = 0.0
                    t += dt
                    continue

                tau_depth += d_tau

                tau_sun = _light_march_nested(
                    p_x, p_y, p_z, sun_dx, sun_dy, sun_dz,
                    sigma_stacked, level_offsets, level_dims,
                    level_bmin, level_bmax, level_dxs, n_levels,
                    n_light_steps,
                    outer_bmin_x, outer_bmin_y, outer_bmin_z,
                    outer_bmax_x, outer_bmax_y, outer_bmax_z,
                )

                ms_r = 0.0; ms_g = 0.0; ms_b = 0.0
                ms_atten = 1.0
                for octave in range(6):
                    t_sun_ms = pymath.exp(-tau_sun * ms_atten)
                    blend = min(1.0, octave * 0.35)
                    oct_phase = phase_hg * (1.0 - blend) + iso_phase * blend
                    contrib = ms_atten * t_sun_ms * oct_phase
                    ms_r += contrib * sun_r
                    ms_g += contrib * sun_g
                    ms_b += contrib * sun_b
                    ms_atten *= 0.4

                # Powder as a function of depth into the current cloud segment.
                # At cloud entry (tau_depth just incremented past zero) powder
                # starts small, saturating to 1 as we penetrate — gives the
                # classic "dark edge, bright core" cumulus look without tying
                # brightness to the step size.
                powder = 1.0 - pymath.exp(-1.5 * tau_depth)
                scatter_weight = d_tau * powder * transmittance

                col_r += scatter_weight * ms_r
                col_g += scatter_weight * ms_g
                col_b += scatter_weight * ms_b

                # Ambient: height-based. Use p_z / outer_height as fraction.
                height_frac = (p_z - outer_bmin_z) / (outer_bmax_z - outer_bmin_z)
                if height_frac < 0.0: height_frac = 0.0
                if height_frac > 1.0: height_frac = 1.0
                amb = ambient_strength * (0.3 + 0.7 * height_frac)
                amb_weight = transmittance * sigma * dt * amb
                col_r += amb_weight * 0.22
                col_g += amb_weight * 0.23
                col_b += amb_weight * 0.28

                transmittance *= pymath.exp(-d_tau)
                t += dt

        # Ocean for rays that exit the outer box without hitting clouds-to-opacity.
        if ocean_enabled and transmittance > 0.002 and t_ocean < 1e29 and t_ocean > t_far:
            o_x = cam_ox + t_ocean * d_x
            o_y = cam_oy + t_ocean * d_y
            o_z = ocean_z
            # Only nearby ocean (within a few outer-box widths).
            dx_outer = outer_bmax_x - outer_bmin_x
            dy_outer = outer_bmax_y - outer_bmin_y
            cx = 0.5 * (outer_bmin_x + outer_bmax_x)
            cy = 0.5 * (outer_bmin_y + outer_bmax_y)
            if abs(o_x - cx) < dx_outer * 50 and abs(o_y - cy) < dy_outer * 50:
                tau_ocean = _light_march_nested(
                    o_x, o_y, o_z, sun_dx, sun_dy, sun_dz,
                    sigma_stacked, level_offsets, level_dims,
                    level_bmin, level_bmax, level_dxs, n_levels,
                    n_light_steps,
                    outer_bmin_x, outer_bmin_y, outer_bmin_z,
                    outer_bmax_x, outer_bmax_y, outer_bmax_z,
                )
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
# Tone mapping (Reinhard + gamma — matches witness)
# ============================================================================

def tone_map(image, exposure=4.0, gamma=1.4):
    exposed = image * exposure
    tone_mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)


# ============================================================================
# Public entry point
# ============================================================================

def _pack_levels(levels: List[NestedLevel]):
    """Pack N levels into flat arrays for the numba kernel."""
    N = len(levels)
    sizes = [l.sigma.size for l in levels]
    total = sum(sizes)
    sigma_stacked = np.empty(total, dtype=np.float64)
    level_offsets = np.zeros(N, dtype=np.int64)
    level_dims = np.zeros((N, 3), dtype=np.int64)
    level_bmin = np.zeros((N, 3), dtype=np.float64)
    level_bmax = np.zeros((N, 3), dtype=np.float64)
    level_dxs = np.zeros((N, 3), dtype=np.float64)

    offset = 0
    for k, lvl in enumerate(levels):
        arr = np.ascontiguousarray(lvl.sigma, dtype=np.float64)
        sigma_stacked[offset:offset + arr.size] = arr.ravel()
        level_offsets[k] = offset
        level_dims[k, 0], level_dims[k, 1], level_dims[k, 2] = arr.shape
        level_bmin[k, :] = lvl.bmin
        level_bmax[k, :] = lvl.bmax
        nx, ny, nz = arr.shape
        level_dxs[k, 0] = (lvl.bmax[0] - lvl.bmin[0]) / nx
        level_dxs[k, 1] = (lvl.bmax[1] - lvl.bmin[1]) / ny
        level_dxs[k, 2] = (lvl.bmax[2] - lvl.bmin[2]) / nz
        offset += arr.size

    return (sigma_stacked, level_offsets, level_dims,
            level_bmin, level_bmax, level_dxs)


def render_nested(
    levels: List[NestedLevel],
    camera_position: Tuple[float, float, float],
    camera_forward: Tuple[float, float, float],
    camera_right: Tuple[float, float, float],
    camera_up: Tuple[float, float, float],
    sun_direction: Tuple[float, float, float],
    image_size: Tuple[int, int],
    fov_degrees: float = 100.0,
    n_light_steps: int = 32,
    step_voxel_factor: float = 2.0,
    max_steps: int = 4096,
    exposure: float = 4.0,
    ocean_enabled: bool = True,
    ocean_z: float = 0.0,
    ocean_reflectance: Tuple[float, float, float] = (0.04, 0.05, 0.07),
    sun_color: Tuple[float, float, float] = (22.0, 21.0, 17.0),
    g_hg: float = 0.76,
    ambient_strength: float = 0.12,
    return_linear: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Render an image with the nested kernel.

    Levels are provided finest-first (index 0 = highest resolution);
    the outermost (coarsest) level defines the outer AABB the ray is clipped
    against. Coarser levels must strictly contain finer levels.
    """
    if len(levels) == 0:
        raise ValueError("Need at least one level.")

    img_w, img_h = image_size
    (sigma_stacked, level_offsets, level_dims,
     level_bmin, level_bmax, level_dxs) = _pack_levels(levels)

    outer = levels[-1]
    outer_bmin = np.asarray(outer.bmin, dtype=np.float64)
    outer_bmax = np.asarray(outer.bmax, dtype=np.float64)

    fov_rad = pymath.radians(fov_degrees)
    tan_half_fov = pymath.tan(fov_rad * 0.5)

    image = np.zeros((img_h, img_w, 3), dtype=np.float64)

    if verbose:
        print(f"  Nested render: {len(levels)} level(s), image {img_w}x{img_h}")
        for k, lvl in enumerate(levels):
            dx, dy, dz = lvl.dx
            size = lvl.bmax - lvl.bmin
            print(f"    L{k} {lvl.name or '':12s} "
                  f"grid={lvl.sigma.shape} "
                  f"dx~({dx:.2f},{dy:.2f},{dz:.2f}) m "
                  f"size=({size[0]/1000:.2f},{size[1]/1000:.2f},{size[2]/1000:.2f}) km "
                  f"origin=({lvl.bmin[0]/1000:.2f},{lvl.bmin[1]/1000:.2f},{lvl.bmin[2]/1000:.2f}) km")

    # Warmup compile with a 1x1 buffer.
    warmup = np.zeros((1, 1, 3), dtype=np.float64)
    if verbose:
        print("  Compiling nested kernel (first run only)...", end="", flush=True)
    _render_image_nested(
        sigma_stacked, level_offsets, level_dims,
        level_bmin, level_bmax, level_dxs, len(levels),
        outer_bmin[0], outer_bmin[1], outer_bmin[2],
        outer_bmax[0], outer_bmax[1], outer_bmax[2],
        camera_position[0], camera_position[1], camera_position[2],
        camera_forward[0], camera_forward[1], camera_forward[2],
        camera_right[0], camera_right[1], camera_right[2],
        camera_up[0], camera_up[1], camera_up[2],
        sun_direction[0], sun_direction[1], sun_direction[2],
        1, 1, tan_half_fov,
        4,
        sun_color[0], sun_color[1], sun_color[2],
        g_hg, ambient_strength,
        ocean_enabled, ocean_z,
        ocean_reflectance[0], ocean_reflectance[1], ocean_reflectance[2],
        step_voxel_factor, 32,
        warmup,
    )
    if verbose:
        print(" done")
        print("  Rendering...", end="", flush=True)

    t0 = time.perf_counter()
    _render_image_nested(
        sigma_stacked, level_offsets, level_dims,
        level_bmin, level_bmax, level_dxs, len(levels),
        outer_bmin[0], outer_bmin[1], outer_bmin[2],
        outer_bmax[0], outer_bmax[1], outer_bmax[2],
        camera_position[0], camera_position[1], camera_position[2],
        camera_forward[0], camera_forward[1], camera_forward[2],
        camera_right[0], camera_right[1], camera_right[2],
        camera_up[0], camera_up[1], camera_up[2],
        sun_direction[0], sun_direction[1], sun_direction[2],
        img_w, img_h, tan_half_fov,
        n_light_steps,
        sun_color[0], sun_color[1], sun_color[2],
        g_hg, ambient_strength,
        ocean_enabled, ocean_z,
        ocean_reflectance[0], ocean_reflectance[1], ocean_reflectance[2],
        step_voxel_factor, max_steps,
        image,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f" done ({elapsed:.1f}s)")

    if return_linear:
        return image
    return tone_map(image, exposure=exposure)
