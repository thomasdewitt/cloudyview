#!/usr/bin/env python
"""
witness.py: Video game-style cloud visualization via volume ray marching.

Single, unified numba kernel that handles one or many strictly-nested
extinction grids. The single-domain CLI (`witness`) and the programmatic
nested-domain API (`render_nested`) both route through the same
`_render_image` kernel:

- Ray marches in absolute world meters; levels at wildly different scales
  compose naturally without coordinate rescaling.
- At each sample point the finest level covering that point wins.
- Lighting model is dt-invariant: the powder term is a function of
  cumulative optical depth from the most recent cloud entry, not per-step
  d_tau. Renders look the same whether sampled at 10 m or 1 km steps, which
  is what makes variable-grid nesting work.

Coordinate System (Meteorological Convention):
- East  = +x direction
- North = +y direction
- Up    = +z direction
"""

from __future__ import annotations

import argparse
import math as pymath
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import io, optical_depth, config
from .angles import direction_from_azimuth_elevation
from .cli_utils import (
    CloudyViewHelpFormatter,
    DATA_SELECTION_HELP,
    add_dataset_selection_arguments,
    dataset_selection_kwargs,
)
from .domain import compute_domain_geometry

from numba import njit, prange


# ============================================================================
# Cloud-scattering tuning block
# ----------------------------------------------------------------------------
# Physically-motivated knobs that control the look of the clouds (not the
# sky/ocean). Kept at module scope — numba captures them as compile-time
# constants inside the kernel — so each tuning iteration is a single edit.
# ============================================================================

POWDER_COEFF = 1.5          # powder = 1 - exp(-POWDER_COEFF * tau_depth)
G_HG = 0.76                 # Henyey-Greenstein asymmetry (Mie for 10 µm ≈ 0.85)
AMBIENT_STRENGTH = 0.12     # overall weight of the ambient term
SUN_COLOR = (22.0, 21.0, 17.0)   # HDR sun radiance (slightly warm)

# Shadow-ray ("light march") step count. Too low → speckled shadows where
# thin cloud blobs alias between steps.
N_LIGHT_STEPS = 64

# Multi-scattering octave loop. Each octave attenuates tau_sun by MS_ATTEN**k
# and phase-blends from pure HG toward isotropic at rate MS_BLEND_RATE.
MS_OCTAVES = 6
MS_ATTEN = 0.4
MS_BLEND_RATE = 0.35

# Ambient spectrum + vertical ramp. The ambient term stands in for multiply-
# scattered skylight that reaches the cloud after leaving the volume.
AMBIENT_TINT_R = 0.22
AMBIENT_TINT_G = 0.23
AMBIENT_TINT_B = 0.28
AMBIENT_HEIGHT_FLOOR = 0.3  # amb(h) = strength * (floor + (1-floor) * h)

# Numerical integration.
STEP_VOXEL_FACTOR = 2.0     # dt_max = min(active_level_dx) * this
MAX_STEPS = 2048

# Ocean diffuse albedo — calibrated to IMG_6048 (kept here so render_nested
# can use it as a default; ocean tuning itself is not part of this block).
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
# Numba JIT helper functions — all in absolute world meters
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
    base00 = offset + ix * stride_x + iy * stride_y
    base10 = offset + ix1 * stride_x + iy * stride_y
    base01 = offset + ix * stride_x + iy1 * stride_y
    base11 = offset + ix1 * stride_x + iy1 * stride_y

    c000 = sigma_stacked[base00 + iz]
    c100 = sigma_stacked[base10 + iz]
    c010 = sigma_stacked[base01 + iz]
    c110 = sigma_stacked[base11 + iz]
    c001 = sigma_stacked[base00 + iz1]
    c101 = sigma_stacked[base10 + iz1]
    c011 = sigma_stacked[base01 + iz1]
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
def _light_march(px, py, pz, sun_dx, sun_dy, sun_dz,
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
    """Procedural sky: cobalt zenith, hazy horizon, gradual circumsolar bloom.

    Tuned against phone photos of real cumulus scenes. The old shader had a
    cos^2 halo across the whole sun-facing hemisphere (producing a ringed
    bloom visible even when the sun was off-frame) plus a smoothstep bloom
    with a hard outer disc. Replaced with a Lorentzian in 1-cos(theta): soft
    peak, long low-amplitude tail, no visible cutoff.
    """
    # Zenith-to-horizon gradient. Biasing with 1-(1-dz)^3 keeps ~97% zenith
    # down to 45° elevation and fades to horizon mostly within the bottom 20°
    # — deep cobalt dominates most of an up-looking frame. Below-horizon rays
    # clamp to horizon color (ocean normally handles those when enabled).
    t = max(0.0, dz)
    one_minus = 1.0 - t
    t = 1.0 - one_minus * one_minus * one_minus
    # Zenith sampled from top-center patch of IMG_6304 (sRGB (14,57,112))
    # reverse-mapped through tone_map(exposure=4, gamma=1.4).
    zen_r = 0.0044; zen_g = 0.035; zen_b = 0.1156
    hor_r = 0.10;   hor_g = 0.18;  hor_b = 0.38
    sky_r = hor_r + (zen_r - hor_r) * t
    sky_g = hor_g + (zen_g - hor_g) * t
    sky_b = hor_b + (zen_b - hor_b) * t

    cos_sun = dx * sun_dx + dy * sun_dy + dz * sun_dz

    # Circumsolar bloom. Lorentzian in 1-cos(theta) so the shape is a smooth
    # peak with a long, low-amplitude tail — no finite cutoff, no visible
    # disc outline. Half-max at ~3.6°, peak amplitude 1.
    if cos_sun > 0.0:
        sun_half_width = 0.002
        a = sun_half_width / ((1.0 - cos_sun) + sun_half_width)
        sky_r += a * 0.8
        sky_g += a * 0.6
        sky_b += a * 0.3

    # Solar disc.
    if cos_sun > 0.9998:
        sky_r += 50.0
        sky_g += 45.0
        sky_b += 35.0

    return sky_r, sky_g, sky_b


@njit(inline="always")
def _ocean_wave_normal_fif(x, y, fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_N):
    """Sample the precomputed FIF normal map at world (x,y).

    Periodic wrap + bilinear interp; renormalizes after interp so a tilted
    normal doesn't shorten toward the mean. fif_* are float32 N×N arrays
    produced by cloudyview.ocean_fif.generate_fif_normals.
    """
    gx = (x * inv_fif_dx) % fif_N
    gy = (y * inv_fif_dx) % fif_N
    i0 = int(gx); j0 = int(gy)
    i1 = (i0 + 1) % fif_N
    j1 = (j0 + 1) % fif_N
    tx = gx - i0
    ty = gy - j0
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    nx = (fif_nx[j0, i0] * w00 + fif_nx[j0, i1] * w10
          + fif_nx[j1, i0] * w01 + fif_nx[j1, i1] * w11)
    ny = (fif_ny[j0, i0] * w00 + fif_ny[j0, i1] * w10
          + fif_ny[j1, i0] * w01 + fif_ny[j1, i1] * w11)
    nz = (fif_nz[j0, i0] * w00 + fif_nz[j0, i1] * w10
          + fif_nz[j1, i0] * w01 + fif_nz[j1, i1] * w11)
    nl = 1.0 / pymath.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * nl, ny * nl, nz * nl


@njit(inline="always")
def _reflection_sky(dx, dy, dz, sun_dx, sun_dy, sun_dz):
    """Sky sampler for ocean reflection: gradient + wide glint lobe.

    The main _sky_radiance has a 3.6° bloom (+1.15° disc) — when reflected
    through wavy water those sharp features produce "binary" on/off sparkles
    as individual wave facets tilt in and out. This variant replaces the
    narrow features with a single ~11° Lorentzian so the glint spreads
    smoothly across many facets, reading as a soft bright path.
    """
    t = dz if dz > 0.0 else 0.0
    one_minus = 1.0 - t
    t = 1.0 - one_minus * one_minus * one_minus
    zen_r = 0.0044; zen_g = 0.035; zen_b = 0.1156
    hor_r = 0.10;   hor_g = 0.18;  hor_b = 0.38
    sky_r = hor_r + (zen_r - hor_r) * t
    sky_g = hor_g + (zen_g - hor_g) * t
    sky_b = hor_b + (zen_b - hor_b) * t
    cos_rs = dx * sun_dx + dy * sun_dy + dz * sun_dz
    if cos_rs > 0.0:
        glint_w = 0.02   # Lorentzian half-width in (1-cos); half-max ~11°
        a = glint_w / ((1.0 - cos_rs) + glint_w)
        sky_r += a * 1.2
        sky_g += a * 1.0
        sky_b += a * 0.6
    return sky_r, sky_g, sky_b


@njit(inline="always")
def _ocean_shade(o_x, o_y, d_x, d_y, d_z,
                 sun_dx, sun_dy, sun_dz, t_sun_ocean,
                 sun_r, sun_g, sun_b,
                 ocean_rr, ocean_rg, ocean_rb,
                 fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_N):
    """Shade an ocean hit. Subsurface diffuse + Fresnel-weighted sky reflection.

    - Perturbed normal sampled from the FIF normal map (waves).
    - Subsurface ("body") color = ocean_reflectance × Lambertian × sun transmittance.
    - Fresnel (Schlick, F0=0.02 for water) blends specular toward sky.
    - Sky reflection from _reflection_sky (wide glint lobe, no narrow sun).
    """
    nx, ny, nz = _ocean_wave_normal_fif(o_x, o_y,
                                          fif_nx, fif_ny, fif_nz,
                                          inv_fif_dx, fif_N)

    # Reflect the view direction around the surface normal.
    vdotn = d_x * nx + d_y * ny + d_z * nz
    r_x = d_x - 2.0 * vdotn * nx
    r_y = d_y - 2.0 * vdotn * ny
    r_z = d_z - 2.0 * vdotn * nz
    # Guard: if the perturbed normal tips the reflected ray below the
    # horizon, flip it up. Rare at realistic slopes but keeps the sky
    # sampler valid.
    if r_z < 0.0:
        r_z = -r_z

    sky_rr, sky_rg, sky_rb = _reflection_sky(r_x, r_y, r_z,
                                              sun_dx, sun_dy, sun_dz)

    # Fresnel (Schlick): cos_theta is -view · normal (view direction points
    # away from camera into the scene, so we flip to get the incidence cos).
    cos_i = -vdotn
    if cos_i < 0.0:
        cos_i = 0.0
    if cos_i > 1.0:
        cos_i = 1.0
    one_minus = 1.0 - cos_i
    om2 = one_minus * one_minus
    F0 = 0.02
    F = F0 + (1.0 - F0) * om2 * om2 * one_minus

    # Subsurface diffuse: Lambertian against the perturbed normal.
    cos_sun_n = sun_dx * nx + sun_dy * ny + sun_dz * nz
    if cos_sun_n < 0.0:
        cos_sun_n = 0.0
    inv_pi = 1.0 / 3.14159265358979
    # Diffuse uses raw t_sun_ocean (direct sun attenuation).
    diff_irr = t_sun_ocean * cos_sun_n * inv_pi
    diff_r = diff_irr * sun_r * ocean_rr
    diff_g = diff_irr * sun_g * ocean_rg
    diff_b = diff_irr * sun_b * ocean_rb

    one_minus_F = 1.0 - F
    ol_r = F * sky_rr + one_minus_F * diff_r
    ol_g = F * sky_rg + one_minus_F * diff_g
    ol_b = F * sky_rb + one_minus_F * diff_b

    # Global shadow dim on the whole ocean output. At boat-level grazing
    # angles Fresnel dominates, so attenuating only the diffuse term leaves
    # shadows invisible. A single multiplier on the final color stands in
    # for three coupled effects: loss of direct sun under cloud, loss of
    # sun-glint reflection where the sun is blocked, and darker sky-
    # reflection because Fresnel then sees the cloud underside rather than
    # clear sky. Floor 0.35 → shadows are ~35% of fully-lit brightness.
    shadow_floor = 0.35
    t_eff = shadow_floor + (1.0 - shadow_floor) * t_sun_ocean
    return ol_r * t_eff, ol_g * t_eff, ol_b * t_eff


# ============================================================================
# Main render kernel (unified: N=1 is the single-domain case)
# ============================================================================

@njit(parallel=True)
def _render_image(
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
    fif_nx, fif_ny, fif_nz, fif_dx,
    step_voxel_factor,   # dt_max = min(active_level_dx) * this
    max_steps,
    powder_coeff,        # powder = 1 - exp(-powder_coeff * tau_depth)
    image,
):
    aspect = img_w / img_h
    iso_phase = 1.0 / (4.0 * 3.14159265358979)
    inv_fif_dx = 1.0 / fif_dx
    fif_N = fif_nx.shape[0]

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

        # Entry into outermost volume.
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

        # Cumulative optical depth from the most recent cloud entry; drives
        # the powder term so that brightness is dt-invariant across nested
        # grids with very different step sizes.
        tau_depth = 0.0

        if t_near >= 0 and t_near < t_far:
            t = t_near

            for _ in range(max_steps):
                # Ocean hit tested before t_far: if the ocean plane coincides
                # with the outer box floor, t_ocean == t_far and the t_far
                # break would skip ocean shading for downward rays.
                if ocean_enabled and t >= t_ocean:
                    o_x = cam_ox + t_ocean * d_x
                    o_y = cam_oy + t_ocean * d_y
                    o_z = ocean_z

                    tau_ocean = _light_march(
                        o_x, o_y, o_z, sun_dx, sun_dy, sun_dz,
                        sigma_stacked, level_offsets, level_dims,
                        level_bmin, level_bmax, level_dxs, n_levels,
                        n_light_steps,
                        outer_bmin_x, outer_bmin_y, outer_bmin_z,
                        outer_bmax_x, outer_bmax_y, outer_bmax_z,
                    )
                    t_sun_ocean = pymath.exp(-tau_ocean)
                    ol_r, ol_g, ol_b = _ocean_shade(
                        o_x, o_y, d_x, d_y, d_z,
                        sun_dx, sun_dy, sun_dz, t_sun_ocean,
                        sun_r, sun_g, sun_b,
                        ocean_rr, ocean_rg, ocean_rb,
                        fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_N,
                    )

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
                    # Outside all levels despite being inside the outer AABB
                    # (can happen at seams). Advance a voxel and keep going.
                    # Use the outer level's step.
                    outer = n_levels - 1
                    dx_k = level_dxs[outer, 0]
                    if level_dxs[outer, 1] < dx_k:
                        dx_k = level_dxs[outer, 1]
                    if level_dxs[outer, 2] < dx_k:
                        dx_k = level_dxs[outer, 2]
                    t += dx_k * step_voxel_factor
                    tau_depth = 0.0
                    continue

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

                tau_sun = _light_march(
                    p_x, p_y, p_z, sun_dx, sun_dy, sun_dz,
                    sigma_stacked, level_offsets, level_dims,
                    level_bmin, level_bmax, level_dxs, n_levels,
                    n_light_steps,
                    outer_bmin_x, outer_bmin_y, outer_bmin_z,
                    outer_bmax_x, outer_bmax_y, outer_bmax_z,
                )

                ms_r = 0.0; ms_g = 0.0; ms_b = 0.0
                ms_atten = 1.0
                for octave in range(MS_OCTAVES):
                    t_sun_ms = pymath.exp(-tau_sun * ms_atten)
                    blend = min(1.0, octave * MS_BLEND_RATE)
                    oct_phase = phase_hg * (1.0 - blend) + iso_phase * blend
                    contrib = ms_atten * t_sun_ms * oct_phase
                    ms_r += contrib * sun_r
                    ms_g += contrib * sun_g
                    ms_b += contrib * sun_b
                    ms_atten *= MS_ATTEN

                # Powder as a function of depth into the current cloud segment:
                # dark edges, bright cores, invariant to step size.
                powder = 1.0 - pymath.exp(-powder_coeff * tau_depth)
                scatter_weight = d_tau * powder * transmittance

                col_r += scatter_weight * ms_r
                col_g += scatter_weight * ms_g
                col_b += scatter_weight * ms_b

                # Ambient: height-based on the outer box.
                height_frac = (p_z - outer_bmin_z) / (outer_bmax_z - outer_bmin_z)
                if height_frac < 0.0: height_frac = 0.0
                if height_frac > 1.0: height_frac = 1.0
                amb = ambient_strength * (AMBIENT_HEIGHT_FLOOR
                                          + (1.0 - AMBIENT_HEIGHT_FLOOR) * height_frac)
                amb_weight = transmittance * d_tau * amb
                col_r += amb_weight * AMBIENT_TINT_R
                col_g += amb_weight * AMBIENT_TINT_G
                col_b += amb_weight * AMBIENT_TINT_B

                transmittance *= pymath.exp(-d_tau)
                t += dt

        # Ocean for rays that exit the outer box without hitting opacity.
        if ocean_enabled and transmittance > 0.002 and t_ocean < 1e29 and t_ocean > t_far:
            o_x = cam_ox + t_ocean * d_x
            o_y = cam_oy + t_ocean * d_y
            o_z = ocean_z
            dx_outer = outer_bmax_x - outer_bmin_x
            dy_outer = outer_bmax_y - outer_bmin_y
            cx = 0.5 * (outer_bmin_x + outer_bmax_x)
            cy = 0.5 * (outer_bmin_y + outer_bmax_y)
            if abs(o_x - cx) < dx_outer * 50 and abs(o_y - cy) < dy_outer * 50:
                tau_ocean = _light_march(
                    o_x, o_y, o_z, sun_dx, sun_dy, sun_dz,
                    sigma_stacked, level_offsets, level_dims,
                    level_bmin, level_bmax, level_dxs, n_levels,
                    n_light_steps,
                    outer_bmin_x, outer_bmin_y, outer_bmin_z,
                    outer_bmax_x, outer_bmax_y, outer_bmax_z,
                )
                t_sun_ocean = pymath.exp(-tau_ocean)
                ol_r, ol_g, ol_b = _ocean_shade(
                    o_x, o_y, d_x, d_y, d_z,
                    sun_dx, sun_dy, sun_dz, t_sun_ocean,
                    sun_r, sun_g, sun_b,
                    ocean_rr, ocean_rg, ocean_rb,
                    fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_N,
                )
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
# Level packing + kernel driver
# ============================================================================

def _pack_levels(levels: Sequence[NestedLevel]):
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


def _render_levels(
    levels: Sequence[NestedLevel],
    camera_position: Tuple[float, float, float],
    camera_forward: Tuple[float, float, float],
    camera_right: Tuple[float, float, float],
    camera_up: Tuple[float, float, float],
    sun_direction: Tuple[float, float, float],
    image_size: Tuple[int, int],
    fov_degrees: float,
    n_light_steps: int,
    step_voxel_factor: float,
    max_steps: int,
    ocean_enabled: bool,
    ocean_z: float,
    ocean_reflectance: Tuple[float, float, float],
    fif_nx: np.ndarray,
    fif_ny: np.ndarray,
    fif_nz: np.ndarray,
    fif_dx: float,
    sun_color: Tuple[float, float, float],
    g_hg: float,
    ambient_strength: float,
    powder_coeff: float,
    verbose: bool,
) -> np.ndarray:
    """Pack levels, warm up, render, and return the linear HDR buffer."""
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
        print(f"  Levels: {len(levels)}")
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
        print("  Compiling render kernel (first run only)...", end="", flush=True)
    _render_image(
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
        fif_nx, fif_ny, fif_nz, fif_dx,
        step_voxel_factor, 32, powder_coeff,
        warmup,
    )
    if verbose:
        print(" done")
        print("  Rendering...", end="", flush=True)

    t0 = time.perf_counter()
    _render_image(
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
        fif_nx, fif_ny, fif_nz, fif_dx,
        step_voxel_factor, max_steps, powder_coeff,
        image,
    )
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f" done ({elapsed:.1f}s)")

    return image


# ============================================================================
# Tone mapping
# ============================================================================

def tone_map(image, exposure=4.0, gamma=1.4):
    """Reinhard tone mapping with gamma correction (matches behold)."""
    exposed = image * exposure
    tone_mapped = exposed / (1.0 + exposed)
    return np.power(np.clip(tone_mapped, 0, 1), 1.0 / gamma)


# ============================================================================
# Nested-domain public entry point
# ============================================================================

def render_nested(
    levels: Sequence[NestedLevel],
    camera_position: Tuple[float, float, float],
    camera_forward: Tuple[float, float, float],
    camera_right: Tuple[float, float, float],
    camera_up: Tuple[float, float, float],
    sun_direction: Tuple[float, float, float],
    image_size: Tuple[int, int],
    fov_degrees: float = 100.0,
    n_light_steps: int = N_LIGHT_STEPS,
    step_voxel_factor: float = STEP_VOXEL_FACTOR,
    max_steps: int = 4096,
    exposure: float = 4.0,
    ocean_enabled: bool = True,
    ocean_z: float = 0.0,
    ocean_reflectance: Tuple[float, float, float] = OCEAN_REFLECTANCE,
    fif_normals: Tuple[np.ndarray, np.ndarray, np.ndarray, float] = None,
    sun_color: Tuple[float, float, float] = SUN_COLOR,
    g_hg: float = G_HG,
    ambient_strength: float = AMBIENT_STRENGTH,
    powder_coeff: float = POWDER_COEFF,
    return_linear: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Render through N strictly-nested extinction grids.

    Levels are finest-first (index 0 = highest resolution); the outermost
    (last) level defines the outer AABB the ray is clipped against.

    fif_normals: (nx, ny, nz, dx_m) tuple from
        cloudyview.ocean_fif.generate_fif_normals. Required when ocean_enabled
        is True; the kernel samples the FIF normal map with periodic wrap at
        each ocean hit. Pass None (with ocean_enabled=False) for sky-only.
    """
    if ocean_enabled and fif_normals is None:
        from cloudyview.ocean_fif import generate_fif_normals
        fif_normals = generate_fif_normals(verbose=verbose)
    if fif_normals is None:
        from cloudyview.ocean_fif import dummy_fif_arrays
        dz, _, _ = dummy_fif_arrays()
        fif_nx, fif_ny, fif_nz, fif_dx = dz, dz, np.ones_like(dz), 1.0
    else:
        fif_nx, fif_ny, fif_nz, fif_dx = fif_normals

    image = _render_levels(
        levels,
        camera_position, camera_forward, camera_right, camera_up,
        sun_direction, image_size,
        fov_degrees, n_light_steps, step_voxel_factor, max_steps,
        ocean_enabled, ocean_z, ocean_reflectance,
        fif_nx, fif_ny, fif_nz, fif_dx,
        sun_color, g_hg, ambient_strength, powder_coeff,
        verbose,
    )
    if return_linear:
        return image
    return tone_map(image, exposure=exposure)


# ============================================================================
# Single-domain main (CLI entry point)
# ============================================================================

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
    """Render a single NetCDF domain through the unified kernel."""
    print(f"CloudyView Witness: Loading {filename}")
    start_time = time.perf_counter()

    witness_config = config.get_witness_config()
    cam_config = witness_config['camera']
    sun_config = witness_config['sun']
    render_config = witness_config['rendering']

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
        data_dict = io.load_and_validate(
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
        lw_data = data_dict['liquid_water_data']
        iw_data = data_dict['ice_water_data']

        x_coord = data_dict.get('x_coord')
        y_coord = data_dict.get('y_coord')
        z_coord = data_dict.get('z_coord')
        if x_coord is None or y_coord is None or z_coord is None:
            raise ValueError(
                "Missing x/y/z coordinate arrays in validated dataset; "
                "cannot render witness view."
            )

        lw_np = lw_data.values
        if 'time' in lw_data.dims:
            lw_np = lw_np[0]
        nx_d, ny_d, nz_d = lw_np.shape

        iw_np = None
        if iw_data is not None:
            iw_np = iw_data.values
            if 'time' in iw_data.dims:
                iw_np = iw_np[0]
            if np.max(iw_np) < 1e-6:
                iw_np = None

        # Physical extinction in m^-1 (not scaled by height_z: the kernel
        # works in absolute meters, so sigma is already in physical units).
        sigma_ext = optical_depth.compute_extinction_field(
            lw_np, z_coord, re=10.0, iwc=iw_np, re_ice=30.0)

        geom = compute_domain_geometry(x_coord, y_coord, z_coord, nx_d, ny_d, nz_d)

        print(f"  Grid: {nx_d} x {ny_d} x {nz_d}, spacing: {geom.dx:.1f} x {geom.dy:.1f} m")
        print(f"  Domain: {geom.width_x:.0f} x {geom.width_y:.0f} x {geom.height_z:.0f} m")

        ext_mult = render_config['extinction_multiplier']
        sigma_world = (sigma_ext * ext_mult).astype(np.float64)
        sigma_world = np.ascontiguousarray(sigma_world)

        sigma_max = float(np.max(sigma_world))
        sigma_mean_nz = float(np.mean(sigma_world[sigma_world > 0])) if np.any(sigma_world > 0) else 0.0
        print(f"  Extinction: max={sigma_max:.4f} m^-1, mean(nonzero)={sigma_mean_nz:.4f} m^-1")

        # Absolute-meter AABB from coordinate arrays (cell-centred; half-step
        # padding so the AABB encloses the outermost cells' extents).
        x_vals = np.asarray(x_coord).astype(np.float64)
        y_vals = np.asarray(y_coord).astype(np.float64)
        z_vals = np.asarray(z_coord).astype(np.float64)
        dx_half = 0.5 * geom.dx
        dy_half = 0.5 * geom.dy
        # For z, use first/last spacing instead of mean dz to tolerate stretched grids.
        dz_lo_half = 0.5 * abs(z_vals[1] - z_vals[0])
        dz_hi_half = 0.5 * abs(z_vals[-1] - z_vals[-2])
        bmin = np.array([x_vals.min() - dx_half,
                         y_vals.min() - dy_half,
                         z_vals.min() - dz_lo_half], dtype=np.float64)
        bmax = np.array([x_vals.max() + dx_half,
                         y_vals.max() + dy_half,
                         z_vals.max() + dz_hi_half], dtype=np.float64)

        level = NestedLevel(sigma=sigma_world, bmin=bmin, bmax=bmax, name="single")

        # Camera in absolute meters. x,y: rel=[-1,1] spans the AABB. z is
        # anchored to the physical surface (z=0), not the AABB's z-range, so
        # that rel_z=-1 is the ground even for elevated domains (e.g. data
        # starting at z=500m keeps its real altitude instead of being slammed
        # down). Reduces to the old mapping when bmin[2]==0.
        rel_pos = cam_config['position']
        cam_origin = np.empty(3, dtype=np.float64)
        cam_origin[0] = bmin[0] + (rel_pos[0] + 1.0) * 0.5 * (bmax[0] - bmin[0])
        cam_origin[1] = bmin[1] + (rel_pos[1] + 1.0) * 0.5 * (bmax[1] - bmin[1])
        cam_origin[2] = (rel_pos[2] + 1.0) * 0.5 * bmax[2]

        forward = direction_from_azimuth_elevation(
            cam_config['azimuth'], cam_config['elevation']
        )
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, forward); up /= np.linalg.norm(up)

        sun_dir = direction_from_azimuth_elevation(
            sun_config['azimuth'], sun_config['elevation']
        )

        # Ocean sits at the physical surface, not the AABB floor: rel=-1 is
        # z=0 (sea level). Default height=-0.9999 → ~0 for any domain top.
        # Reduces to the old mapping when bmin[2]==0.
        ocean_config = render_config['ocean']
        ocean_enabled = ocean_config['enabled']
        ocean_z = (ocean_config['height'] + 1.0) * 0.5 * bmax[2]

        n_light_steps = render_config['n_light_steps']
        exposure = render_config['exposure']

        print(f"  Camera: abs=({cam_origin[0]:.1f},{cam_origin[1]:.1f},{cam_origin[2]:.1f}) m")
        print(f"          azimuth={cam_config['azimuth']:.1f} elev={cam_config['elevation']:.1f} fov={cam_config['fov']:.1f}")
        print(f"  Sun: azimuth={sun_config['azimuth']:.1f} elev={sun_config['elevation']:.1f}")
        print(f"  Image: {img_w}x{img_h}")

        if ocean_enabled:
            from cloudyview.ocean_fif import generate_fif_normals
            fif_nx, fif_ny, fif_nz, fif_dx = generate_fif_normals()
        else:
            from cloudyview.ocean_fif import dummy_fif_arrays
            _z, _, _o = dummy_fif_arrays()
            fif_nx, fif_ny, fif_nz, fif_dx = _z, _z, _o, 1.0

        image = _render_levels(
            [level],
            (cam_origin[0], cam_origin[1], cam_origin[2]),
            (forward[0], forward[1], forward[2]),
            (right[0], right[1], right[2]),
            (up[0], up[1], up[2]),
            (sun_dir[0], sun_dir[1], sun_dir[2]),
            (img_w, img_h),
            cam_config['fov'], n_light_steps,
            STEP_VOXEL_FACTOR, MAX_STEPS,
            ocean_enabled, ocean_z, OCEAN_REFLECTANCE,
            fif_nx, fif_ny, fif_nz, fif_dx,
            SUN_COLOR, G_HG, AMBIENT_STRENGTH, POWDER_COEFF,
            verbose=True,
        )
        image_tm = tone_map(image, exposure=exposure)

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
              `witness --help` works without optional acceleration packages. Rendering is
              fastest when `numba` is installed.

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
                        help="Camera field of view in degrees (default: 100)")
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
