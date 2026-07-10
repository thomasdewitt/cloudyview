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
import logging
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
from .camera import Camera
from .cli_utils import (
    CloudyViewHelpFormatter,
    DATA_SELECTION_HELP,
    add_dataset_selection_arguments,
    dataset_selection_kwargs,
)
from .cloudfield import CloudField, load as _load_field
from .domain import compute_domain_geometry

from numba import njit, prange

logger = logging.getLogger(__name__)


# ============================================================================
# Lighting and cloud-scattering tuning block
# ----------------------------------------------------------------------------
# Physically-motivated knobs that control the look of the clouds and sky.
# Kept at module scope so each tuning iteration is a single edit. Ocean-only
# tuning remains separate; its direct glint reuses the spectral beam color.
# ============================================================================

POWDER_COEFF = 1.5          # powder = 1 - exp(-POWDER_COEFF * tau_depth)
G_HG = 0.76                 # Henyey-Greenstein asymmetry (Mie for 10 µm ≈ 0.85)
AMBIENT_STRENGTH = 0.12     # overall weight of the ambient term
SUN_COLOR = (22.0, 21.0, 17.0)   # HDR sun radiance (slightly warm)

# Shadow-ray ("light march") iteration cap. Stepping is voxel-adaptive
# (same STEP_VOXEL_FACTOR as the view ray), with early exit when
# tau_sun > 80 (effectively zero sun contribution). This is a safety
# bound; the tau saturation usually terminates long before we reach it.
N_LIGHT_STEPS = 512

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

# Spectral time-of-day lighting. SUN_COLOR is calibrated around the default
# 55-degree sun. At lower elevations, the extra optical air mass selectively
# removes short wavelengths from the direct beam (Rayleigh ~ lambda^-4 plus
# an Angstrom aerosol term). The corresponding diffuse fill becomes bluer,
# while the sun-facing horizon and circumsolar light warm. Strength 0 restores
# the exact legacy fixed-color cloud/sky path. These colors are precomputed
# once per frame; no atmospheric work is done in the pixel loop.
SPECTRAL_LIGHTING_STRENGTH = 1.0
ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG = 55.0
ATMOSPHERE_MAX_AIRMASS = 20.0
ATMOSPHERE_RAYLEIGH_OD_550 = 0.10
ATMOSPHERE_AEROSOL_OD_550 = 0.08
ATMOSPHERE_AEROSOL_ANGSTROM = 1.3
ATMOSPHERE_RGB_WAVELENGTHS_NM = (680.0, 550.0, 460.0)
SUNSET_HORIZON_RADIANCE = (0.42, 0.20, 0.055)

# Angular distribution of the low-sun horizon spectrum. Aerosol path length
# reddens a broad sector at the geometric horizon, but that sector narrows
# rapidly with elevation; the zenith and anti-solar sky remain blue. The
# neutral control color bends the blue-to-gold interpolation through pale
# daylight instead of the mauve produced by a straight RGB blend. Strength 0
# restores the iter_006 azimuth-only spectral sky field exactly.
LOW_SUN_SKY_FIELD_STRENGTH = 1.0
LOW_SUN_SKY_WARM_ELEVATION_DEG = 32.0
LOW_SUN_SKY_HORIZON_AZIMUTH_DEG = 105.0
LOW_SUN_SKY_UPPER_AZIMUTH_DEG = 45.0
LOW_SUN_SKY_NEUTRAL_RADIANCE = (0.27, 0.30, 0.32)
_LOW_SUN_SKY_MAX_WARM_DZ = pymath.sin(pymath.radians(
    LOW_SUN_SKY_WARM_ELEVATION_DEG
))
_LOW_SUN_SKY_HORIZON_AZIMUTH_COS = pymath.cos(pymath.radians(
    LOW_SUN_SKY_HORIZON_AZIMUTH_DEG
))
_LOW_SUN_SKY_UPPER_AZIMUTH_COS = pymath.cos(pymath.radians(
    LOW_SUN_SKY_UPPER_AZIMUTH_DEG
))

# Upward diffuse bounce — stands in for sun/skylight reflected off the
# surface (and low-level multiply-scattered light between cloud base and
# surface) that lights cloud undersides. Mirror image of the ambient ramp:
# full weight at the surface, zero at domain top, so it lifts bases without
# touching sunlit tops. Tint is slightly warm (reflected sunlight), unlike
# the cool skylight ambient. Strength 0 disables the term.
# 0.05 chosen from the 2026-07-07 contact-sheet sweep (conservative pick:
# luminous base, maximum retained sunlit/shaded contrast; >=0.15 washes out).
BOUNCE_STRENGTH = 0.05
BOUNCE_TINT_R = 1.00
BOUNCE_TINT_G = 0.97
BOUNCE_TINT_B = 0.92

# Cumulonimbus realism package.
#
# Diagnosis: once a deep convective cloud becomes optically thick, the visible
# near-shell source term is too spatially smooth. Direct sun is mostly gone,
# the saturated light march leaves a uniform isotropic multi-scattering floor,
# ambient is height-only, and bounce keeps lighting deep cores. These five
# gated terms add local surface orientation and remove fill only where optical
# depth says the sample is truly buried. Setting any strength to 0.0 disables
# that term and restores the pre-package look for that path.
#
# 1) Gradient thick-surface shading. Central-difference sigma samples estimate
# the outward normal N=-normalize(grad sigma). A fine stencil keeps close-range
# surface relief; a coarse stencil recovers the broad lobe orientation that
# survives kilometer-range viewing. In cone mode its radius subtends a fixed
# angle at the camera, so the sampled world-space scale grows with viewing
# distance. Set CONE_STENCIL_THETA_DEG to 0.0 for the legacy fixed-radius
# stencil. The N.sun lobe modulates only the sun/MS contribution, gated by
# tau_depth and gradient confidence so thin fair-weather wisps do not become
# plastic or noisy.
GRADIENT_SHADING_STRENGTH = 1.50
GRADIENT_SHADING_RADIUS_VOXELS = 1.0
GRADIENT_SHADING_COARSE_WEIGHT = 0.65
GRADIENT_SHADING_COARSE_RADIUS_M = 500.0
CONE_STENCIL_THETA_DEG = 2.0
GRADIENT_SHADING_COARSE_MIN_VOXELS = 4.0
GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION = 0.125
GRADIENT_SHADING_TAU_START = 0.25
GRADIENT_SHADING_TAU_FULL = 1.60
GRADIENT_SHADING_CONF_START = 0.06
GRADIENT_SHADING_CONF_FULL = 0.28
GRADIENT_SHADING_SHADOW_SIDE_SCALE = 0.55

# 2) Deep-shadow MS floor suppression. When tau_sun approaches the light-march
# cutoff, damp the late/isotropic MS octaves that otherwise glow as a flat grey
# floor. The first, directional octave is left alone and a nonzero floor remains.
DEEP_SHADOW_MS_SUPPRESSION = 0.90
DEEP_SHADOW_TAU_START = 38.0
DEEP_SHADOW_TAU_FULL = 80.0
DEEP_SHADOW_MS_FLOOR = 0.24

# 3) Directional ambient occlusion. The same saturated sun optical depth is
# used as a cheap directional sky-access proxy. It removes fill in buried
# cavities while retaining a cool blue ambient floor.
AMBIENT_OCCLUSION_STRENGTH = 1.00
AMBIENT_OCCLUSION_FLOOR = 0.24

# 4) Low-sun direct/diffuse split. tau_sun is valid for the direct beam but
# overly aggressive as a proxy for the whole sky dome. Under low sun, boost
# the unoccluded warm sun/MS source modestly while adding a separately colored
# diffuse source only as that directional path saturates. The latter reuses
# the spectral ambient color and height ramp, producing cool luminous shade
# without a global exposure lift. Both additions fade out at the 55-degree
# spectral calibration elevation, preserving the approved high-sun look.
# Master strength 0 is the exact previous arithmetic path at every elevation.
LIGHT_TRANSFER_SPLIT_STRENGTH = 1.0
LIGHT_TRANSFER_DIRECT_BOOST = 0.25
LIGHT_TRANSFER_SHADOW_SKYLIGHT = 0.18
LIGHT_TRANSFER_FULL_ELEVATION_DEG = 45.0
LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG = 55.0

# 5) Depth-attenuated bounce. Ground/ocean bounce should light the visible
# underside skin, not the whole core. k is in optical-depth units.
BOUNCE_DEPTH_ATTENUATION = 0.80

# Numerical integration.
STEP_VOXEL_FACTOR = 2.0     # dt_max = min(active_level_dx) * this
MAX_STEPS = 2048

# Deterministic still-image sampling. Values above 1 decorrelate the camera
# ray within each pixel and the view-march phase within its first step, then
# average the samples in linear HDR space. This turns coherent step-shell
# isophotes into high-frequency error that the spatial average removes.
# WITNESS_SPP=1 is an explicit, exact legacy path: pixel centers, no phase
# jitter, and no averaging. Cost scales approximately linearly with SPP.
WITNESS_SPP = 8

# Ocean surface realism. The legacy ocean point-samples a 5 cm FIF normal
# tile, so distant pixels alias unresolved waves into full-amplitude speckle.
# The realism path box-filters and renormalizes a normal mip chain, chooses a
# level from the projected pixel footprint on the water, replaces the old
# fixed-color reflection lobe with a beam-tinted GGX sun glint, and applies
# Beer-Lambert aerial perspective along the ocean sightline. At 0 the master
# gate takes the untouched legacy shader path exactly.
OCEAN_REALISM = 1.0
OCEAN_MIP_BIAS = -0.5
OCEAN_GLINT_STRENGTH = 0.65
OCEAN_GLINT_ROUGHNESS = 0.10
OCEAN_GLINT_ROUGHNESS_PER_LOD = 0.025
OCEAN_HAZE_EXTINCTION_PER_KM = 0.012

# Ocean diffuse albedo — calibrated to IMG_6048 (kept here so render_nested
# can use it as a default).
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
    """Trilinear sigma sample with ghost-zero boundary taper.

    Treats the grid as if a 1-voxel-thick layer of zeros sat just outside
    each face. Samples in that ghost zone produce a smooth linear fade
    from σ(edge) at the last cell center down to 0 one cell out. Without
    this, a cloud field that reaches the grid boundary has a hard σ
    discontinuity that aliases against the shadow-ray stepping and
    produces concentric ring artifacts on domain-truncated thick clouds.
    Returns 0 for samples more than 1 voxel outside the grid on any axis.
    """
    gx = (px - bmin_x) / dx
    gy = (py - bmin_y) / dy
    gz = (pz - bmin_z) / dz

    if gx < -1.0 or gx >= nx or gy < -1.0 or gy >= ny or gz < -1.0 or gz >= nz:
        return 0.0

    ix = int(pymath.floor(gx))
    iy = int(pymath.floor(gy))
    iz = int(pymath.floor(gz))
    fx = gx - ix; fy = gy - iy; fz = gz - iz
    ix1 = ix + 1; iy1 = iy + 1; iz1 = iz + 1

    # Clamp indices to the valid array range for safe fetching; corners
    # whose true index is out of range will be zeroed below.
    ixs = ix if ix >= 0 else 0
    ix1s = ix1 if ix1 < nx else nx - 1
    iys = iy if iy >= 0 else 0
    iy1s = iy1 if iy1 < ny else ny - 1
    izs = iz if iz >= 0 else 0
    iz1s = iz1 if iz1 < nz else nz - 1

    stride_x = ny * nz
    stride_y = nz
    base00 = offset + ixs * stride_x + iys * stride_y
    base10 = offset + ix1s * stride_x + iys * stride_y
    base01 = offset + ixs * stride_x + iy1s * stride_y
    base11 = offset + ix1s * stride_x + iy1s * stride_y

    c000 = sigma_stacked[base00 + izs]
    c100 = sigma_stacked[base10 + izs]
    c010 = sigma_stacked[base01 + izs]
    c110 = sigma_stacked[base11 + izs]
    c001 = sigma_stacked[base00 + iz1s]
    c101 = sigma_stacked[base10 + iz1s]
    c011 = sigma_stacked[base01 + iz1s]
    c111 = sigma_stacked[base11 + iz1s]

    # Zero corners whose true index is outside the grid (ghost layer).
    if ix < 0:
        c000 = 0.0; c010 = 0.0; c001 = 0.0; c011 = 0.0
    if ix1 >= nx:
        c100 = 0.0; c110 = 0.0; c101 = 0.0; c111 = 0.0
    if iy < 0:
        c000 = 0.0; c100 = 0.0; c001 = 0.0; c101 = 0.0
    if iy1 >= ny:
        c010 = 0.0; c110 = 0.0; c011 = 0.0; c111 = 0.0
    if iz < 0:
        c000 = 0.0; c100 = 0.0; c010 = 0.0; c110 = 0.0
    if iz1 >= nz:
        c001 = 0.0; c101 = 0.0; c011 = 0.0; c111 = 0.0

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
def _smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        if x >= edge1:
            return 1.0
        return 0.0
    t = (x - edge0) / (edge1 - edge0)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(inline="always")
def _sample_sigma_level_k(k, px, py, pz,
                          sigma_stacked, level_offsets, level_dims,
                          level_bmin, level_dxs):
    return _sample_sigma_level(
        sigma_stacked, level_offsets[k],
        level_dims[k, 0], level_dims[k, 1], level_dims[k, 2],
        level_bmin[k, 0], level_bmin[k, 1], level_bmin[k, 2],
        level_dxs[k, 0], level_dxs[k, 1], level_dxs[k, 2],
        px, py, pz,
    )


@njit(inline="always")
def _sigma_gradient_at_radius_level(k, px, py, pz, hx, hy, hz,
                                    sigma_stacked, level_offsets, level_dims,
                                    level_bmin, level_dxs):
    sxp = _sample_sigma_level_k(k, px + hx, py, pz,
                                sigma_stacked, level_offsets, level_dims,
                                level_bmin, level_dxs)
    sxm = _sample_sigma_level_k(k, px - hx, py, pz,
                                sigma_stacked, level_offsets, level_dims,
                                level_bmin, level_dxs)
    syp = _sample_sigma_level_k(k, px, py + hy, pz,
                                sigma_stacked, level_offsets, level_dims,
                                level_bmin, level_dxs)
    sym = _sample_sigma_level_k(k, px, py - hy, pz,
                                sigma_stacked, level_offsets, level_dims,
                                level_bmin, level_dxs)
    szp = _sample_sigma_level_k(k, px, py, pz + hz,
                                sigma_stacked, level_offsets, level_dims,
                                level_bmin, level_dxs)
    szm = _sample_sigma_level_k(k, px, py, pz - hz,
                                sigma_stacked, level_offsets, level_dims,
                                level_bmin, level_dxs)

    gx = (sxp - sxm) / (2.0 * hx)
    gy = (syp - sym) / (2.0 * hy)
    gz = (szp - szm) / (2.0 * hz)
    return gx, gy, gz


@njit(inline="always")
def _sigma_gradient_level(k, px, py, pz, sigma, sample_distance_m,
                          gradient_coarse_weight,
                          gradient_coarse_radius_m,
                          cone_stencil_tan_theta,
                          sigma_stacked, level_offsets, level_dims,
                          level_bmin, level_dxs):
    fine_hx = level_dxs[k, 0] * GRADIENT_SHADING_RADIUS_VOXELS
    fine_hy = level_dxs[k, 1] * GRADIENT_SHADING_RADIUS_VOXELS
    fine_hz = level_dxs[k, 2] * GRADIENT_SHADING_RADIUS_VOXELS
    fine_x, fine_y, fine_z = _sigma_gradient_at_radius_level(
        k, px, py, pz, fine_hx, fine_hy, fine_hz,
        sigma_stacked, level_offsets, level_dims, level_bmin, level_dxs,
    )
    fine_h_min = fine_hx
    if fine_hy < fine_h_min:
        fine_h_min = fine_hy
    if fine_hz < fine_h_min:
        fine_h_min = fine_hz
    fine_len = pymath.sqrt(
        fine_x * fine_x + fine_y * fine_y + fine_z * fine_z
    )
    fine_conf = (fine_len * fine_h_min) / (sigma + 1e-4)

    if gradient_coarse_weight <= 0.0:
        return fine_x, fine_y, fine_z, fine_conf

    dx = level_dxs[k, 0]
    dy = level_dxs[k, 1]
    dz = level_dxs[k, 2]
    extent_x = level_dims[k, 0] * dx
    extent_y = level_dims[k, 1] * dy
    extent_z = level_dims[k, 2] * dz

    # A fixed angular radius follows apparent cloud scale: distant samples use
    # a broader world-space normal while nearby samples converge on the fine
    # stencil. An exact zero explicitly selects the legacy fixed-radius path.
    if cone_stencil_tan_theta > 0.0:
        coarse_radius_m = sample_distance_m * cone_stencil_tan_theta
        coarse_min_voxels = GRADIENT_SHADING_RADIUS_VOXELS
    else:
        coarse_radius_m = gradient_coarse_radius_m
        coarse_min_voxels = GRADIENT_SHADING_COARSE_MIN_VOXELS

    coarse_hx = coarse_radius_m
    min_hx = coarse_min_voxels * dx
    max_hx = GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION * extent_x
    if coarse_hx < min_hx:
        coarse_hx = min_hx
    if cone_stencil_tan_theta == 0.0 and coarse_hx > max_hx:
        coarse_hx = max_hx
    if coarse_hx < fine_hx:
        coarse_hx = fine_hx

    coarse_hy = coarse_radius_m
    min_hy = coarse_min_voxels * dy
    max_hy = GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION * extent_y
    if coarse_hy < min_hy:
        coarse_hy = min_hy
    if cone_stencil_tan_theta == 0.0 and coarse_hy > max_hy:
        coarse_hy = max_hy
    if coarse_hy < fine_hy:
        coarse_hy = fine_hy

    coarse_hz = coarse_radius_m
    min_hz = coarse_min_voxels * dz
    max_hz = GRADIENT_SHADING_COARSE_MAX_DOMAIN_FRACTION * extent_z
    if coarse_hz < min_hz:
        coarse_hz = min_hz
    if cone_stencil_tan_theta == 0.0 and coarse_hz > max_hz:
        coarse_hz = max_hz
    if coarse_hz < fine_hz:
        coarse_hz = fine_hz

    coarse_x, coarse_y, coarse_z = _sigma_gradient_at_radius_level(
        k, px, py, pz, coarse_hx, coarse_hy, coarse_hz,
        sigma_stacked, level_offsets, level_dims, level_bmin, level_dxs,
    )
    coarse_h_min = coarse_hx
    if coarse_hy < coarse_h_min:
        coarse_h_min = coarse_hy
    if coarse_hz < coarse_h_min:
        coarse_h_min = coarse_hz
    coarse_len = pymath.sqrt(
        coarse_x * coarse_x + coarse_y * coarse_y + coarse_z * coarse_z
    )
    coarse_conf = (coarse_len * coarse_h_min) / (sigma + 1e-4)

    if gradient_coarse_weight >= 1.0:
        return coarse_x, coarse_y, coarse_z, coarse_conf

    fine_gate = _smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, fine_conf
    )
    coarse_gate = _smoothstep(
        GRADIENT_SHADING_CONF_START, GRADIENT_SHADING_CONF_FULL, coarse_conf
    )
    coarse_w = gradient_coarse_weight * coarse_gate
    fine_w = (1.0 - gradient_coarse_weight) * fine_gate

    blend_x = 0.0
    blend_y = 0.0
    blend_z = 0.0
    if fine_len > 1e-12:
        fine_inv_len = 1.0 / fine_len
        blend_x += fine_w * fine_x * fine_inv_len
        blend_y += fine_w * fine_y * fine_inv_len
        blend_z += fine_w * fine_z * fine_inv_len
    if coarse_len > 1e-12:
        coarse_inv_len = 1.0 / coarse_len
        blend_x += coarse_w * coarse_x * coarse_inv_len
        blend_y += coarse_w * coarse_y * coarse_inv_len
        blend_z += coarse_w * coarse_z * coarse_inv_len

    grad_conf = fine_conf
    if coarse_conf > grad_conf:
        grad_conf = coarse_conf
    return blend_x, blend_y, blend_z, grad_conf


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
def _sampling_hash(pixel_idx, dimension):
    """Deterministic [0, 1) SplitMix64 hash for per-pixel sample rotations."""
    x = np.uint64(pixel_idx) + np.uint64(dimension + 1) * np.uint64(
        0x9E3779B97F4A7C15
    )
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return float(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(inline="always")
def _hg_phase(cos_theta, g):
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * 3.14159265358979 * denom * pymath.sqrt(denom))


@njit
def _light_march(px, py, pz, sun_dx, sun_dy, sun_dz,
                 sigma_stacked, level_offsets, level_dims,
                 level_bmin, level_bmax, level_dxs,
                 n_levels, max_steps,
                 outer_bmin_x, outer_bmin_y, outer_bmin_z,
                 outer_bmax_x, outer_bmax_y, outer_bmax_z):
    """Adaptive-step shadow march toward the sun through nested levels.

    Steps at ~voxel resolution of the active level (STEP_VOXEL_FACTOR ×
    min voxel size), with early exit when tau > 80 (transmittance well
    below perceivable). max_steps is a safety bound; saturation
    normally terminates first on any cloud-intersecting ray.

    Previously used uniform dt = t_far / N_LIGHT_STEPS, which at 64
    samples across a ~20 km box produced 300 m steps — much coarser
    than the grid voxels. That aliased against σ structure and, when
    combined with hard σ edges at domain truncations, produced
    concentric ring artifacts on thick truncated clouds.
    """
    t_near, t_far = _ray_box(px, py, pz, sun_dx, sun_dy, sun_dz,
                              outer_bmin_x, outer_bmin_y, outer_bmin_z,
                              outer_bmax_x, outer_bmax_y, outer_bmax_z)
    tau = 0.0
    if t_far <= 0:
        return tau

    # Fallback dt when a sample falls outside all levels (nested seams).
    outer = n_levels - 1
    outer_dx = level_dxs[outer, 0]
    if level_dxs[outer, 1] < outer_dx:
        outer_dx = level_dxs[outer, 1]
    if level_dxs[outer, 2] < outer_dx:
        outer_dx = level_dxs[outer, 2]

    t = 0.0
    for _ in range(max_steps):
        if t >= t_far:
            break

        sx = px + t * sun_dx
        sy = py + t * sun_dy
        sz = pz + t * sun_dz

        sigma, k = _sample_sigma_nested(
            sx, sy, sz,
            sigma_stacked, level_offsets, level_dims,
            level_bmin, level_bmax, level_dxs, n_levels,
        )

        if k < 0:
            dt = outer_dx * STEP_VOXEL_FACTOR
        else:
            dx_k = level_dxs[k, 0]
            if level_dxs[k, 1] < dx_k:
                dx_k = level_dxs[k, 1]
            if level_dxs[k, 2] < dx_k:
                dx_k = level_dxs[k, 2]
            dt = dx_k * STEP_VOXEL_FACTOR

        if t + dt > t_far:
            dt = t_far - t

        tau += sigma * dt
        if tau > 80.0:
            break
        t += dt

    return tau


@njit(inline="always")
def _sky_radiance(dx, dy, dz, sun_dx, sun_dy, sun_dz,
                  spectral_hor_r, spectral_hor_g, spectral_hor_b,
                  bloom_r, bloom_g, bloom_b,
                  disc_r, disc_g, disc_b,
                  low_sun_sky_field_strength):
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
    base_hor_r = 0.10; base_hor_g = 0.18; base_hor_b = 0.38

    base_sky_r = base_hor_r + (zen_r - base_hor_r) * t
    base_sky_g = base_hor_g + (zen_g - base_hor_g) * t
    base_sky_b = base_hor_b + (zen_b - base_hor_b) * t
    sky_r = base_sky_r
    sky_g = base_sky_g
    sky_b = base_sky_b

    # Strength=0 spectral lighting and the calibrated 55-degree sun both
    # produce the base horizon exactly. Keeping that case out of the angular
    # work preserves the approved legacy sky arithmetic and color.
    has_spectral_horizon = (
        spectral_hor_r != base_hor_r
        or spectral_hor_g != base_hor_g
        or spectral_hor_b != base_hor_b
    )
    if has_spectral_horizon:
        view_h_len = pymath.sqrt(dx * dx + dy * dy)
        sun_h_len = pymath.sqrt(sun_dx * sun_dx + sun_dy * sun_dy)
        cos_sun_az = -1.0
        if view_h_len > 1e-12 and sun_h_len > 1e-12:
            cos_sun_az = ((dx * sun_dx + dy * sun_dy)
                          / (view_h_len * sun_h_len))
            if cos_sun_az < -1.0:
                cos_sun_az = -1.0
            if cos_sun_az > 1.0:
                cos_sun_az = 1.0

        # Exact iter_006 field for a continuous tuning/bypass path.
        legacy_az_weight = 0.5 + 0.5 * cos_sun_az
        legacy_az_weight = (legacy_az_weight * legacy_az_weight
                            * (3.0 - 2.0 * legacy_az_weight))
        legacy_hor_r = base_hor_r + legacy_az_weight * (
            spectral_hor_r - base_hor_r
        )
        legacy_hor_g = base_hor_g + legacy_az_weight * (
            spectral_hor_g - base_hor_g
        )
        legacy_hor_b = base_hor_b + legacy_az_weight * (
            spectral_hor_b - base_hor_b
        )
        legacy_sky_r = legacy_hor_r + (zen_r - legacy_hor_r) * t
        legacy_sky_g = legacy_hor_g + (zen_g - legacy_hor_g) * t
        legacy_sky_b = legacy_hor_b + (zen_b - legacy_hor_b) * t

        # Warmth is confined vertically, while the azimuthal support widens
        # toward the horizon where the aerosol slant path is longest. Work in
        # direction cosine space to avoid inverse trig in every sky pixel.
        elevation_progress = _smoothstep(
            0.0, _LOW_SUN_SKY_MAX_WARM_DZ, max(0.0, dz)
        )
        azimuth_cutoff = (
            _LOW_SUN_SKY_HORIZON_AZIMUTH_COS
            + elevation_progress * (
                _LOW_SUN_SKY_UPPER_AZIMUTH_COS
                - _LOW_SUN_SKY_HORIZON_AZIMUTH_COS
            )
        )
        azimuth_weight = _smoothstep(azimuth_cutoff, 1.0, cos_sun_az)
        warm_weight = (1.0 - elevation_progress) * azimuth_weight

        # Recover the precomputed horizon's spectral mix so the neutral
        # bridge also vanishes exactly with SPECTRAL_LIGHTING_STRENGTH=0.
        sunset_red_span = SUNSET_HORIZON_RADIANCE[0] - base_hor_r
        horizon_mix = (spectral_hor_r - base_hor_r) / sunset_red_span
        if horizon_mix < 0.0:
            horizon_mix = 0.0
        if horizon_mix > 1.0:
            horizon_mix = 1.0
        neutral_hor_r = base_hor_r + horizon_mix * (
            LOW_SUN_SKY_NEUTRAL_RADIANCE[0] - base_hor_r
        )
        neutral_hor_g = base_hor_g + horizon_mix * (
            LOW_SUN_SKY_NEUTRAL_RADIANCE[1] - base_hor_g
        )
        neutral_hor_b = base_hor_b + horizon_mix * (
            LOW_SUN_SKY_NEUTRAL_RADIANCE[2] - base_hor_b
        )
        neutral_sky_r = neutral_hor_r + (zen_r - neutral_hor_r) * t
        neutral_sky_g = neutral_hor_g + (zen_g - neutral_hor_g) * t
        neutral_sky_b = neutral_hor_b + (zen_b - neutral_hor_b) * t
        warm_sky_r = spectral_hor_r + (zen_r - spectral_hor_r) * t
        warm_sky_g = spectral_hor_g + (zen_g - spectral_hor_g) * t
        warm_sky_b = spectral_hor_b + (zen_b - spectral_hor_b) * t

        # Quadratic Bezier in linear radiance: blue -> neutral -> warm. This
        # avoids the purple midpoint of complementary blue/orange endpoints.
        cool_weight = 1.0 - warm_weight
        cool_weight_2 = cool_weight * cool_weight
        neutral_weight = 2.0 * cool_weight * warm_weight
        warm_weight_2 = warm_weight * warm_weight
        wedge_sky_r = (cool_weight_2 * base_sky_r
                       + neutral_weight * neutral_sky_r
                       + warm_weight_2 * warm_sky_r)
        wedge_sky_g = (cool_weight_2 * base_sky_g
                       + neutral_weight * neutral_sky_g
                       + warm_weight_2 * warm_sky_g)
        wedge_sky_b = (cool_weight_2 * base_sky_b
                       + neutral_weight * neutral_sky_b
                       + warm_weight_2 * warm_sky_b)

        sky_r = legacy_sky_r + low_sun_sky_field_strength * (
            wedge_sky_r - legacy_sky_r
        )
        sky_g = legacy_sky_g + low_sun_sky_field_strength * (
            wedge_sky_g - legacy_sky_g
        )
        sky_b = legacy_sky_b + low_sun_sky_field_strength * (
            wedge_sky_b - legacy_sky_b
        )

    cos_sun = dx * sun_dx + dy * sun_dy + dz * sun_dz

    # Circumsolar bloom. Lorentzian in 1-cos(theta) so the shape is a smooth
    # peak with a long, low-amplitude tail — no finite cutoff, no visible
    # disc outline. Half-max at ~3.6°, peak amplitude 1.
    if cos_sun > 0.0:
        sun_half_width = 0.002
        a = sun_half_width / ((1.0 - cos_sun) + sun_half_width)
        sky_r += a * bloom_r
        sky_g += a * bloom_g
        sky_b += a * bloom_b

    # Solar disc.
    if cos_sun > 0.9998:
        sky_r += disc_r
        sky_g += disc_g
        sky_b += disc_b

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
def _sample_ocean_normal_mip_level(x, y, level,
                                   mip_nx, mip_ny, mip_nz,
                                   mip_offsets, mip_dims,
                                   inv_tile_extent):
    """Periodic bilinear sample from one packed FIF normal-map mip level."""
    nx_dim = mip_dims[level, 0]
    ny_dim = mip_dims[level, 1]
    offset = mip_offsets[level]

    u = x * inv_tile_extent
    v = y * inv_tile_extent
    u -= pymath.floor(u)
    v -= pymath.floor(v)
    gx = u * nx_dim
    gy = v * ny_dim
    i0 = int(gx)
    j0 = int(gy)
    i1 = (i0 + 1) % nx_dim
    j1 = (j0 + 1) % ny_dim
    tx = gx - i0
    ty = gy - j0
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    p00 = offset + j0 * nx_dim + i0
    p10 = offset + j0 * nx_dim + i1
    p01 = offset + j1 * nx_dim + i0
    p11 = offset + j1 * nx_dim + i1
    nx = (mip_nx[p00] * w00 + mip_nx[p10] * w10
          + mip_nx[p01] * w01 + mip_nx[p11] * w11)
    ny = (mip_ny[p00] * w00 + mip_ny[p10] * w10
          + mip_ny[p01] * w01 + mip_ny[p11] * w11)
    nz = (mip_nz[p00] * w00 + mip_nz[p10] * w10
          + mip_nz[p01] * w01 + mip_nz[p11] * w11)
    return nx, ny, nz


@njit(inline="always")
def _ocean_wave_normal_mipped(x, y, lod,
                              mip_nx, mip_ny, mip_nz,
                              mip_offsets, mip_dims, n_mips,
                              inv_tile_extent):
    """Trilinear-in-LOD normal sample, renormalized after interpolation."""
    if lod < 0.0:
        lod = 0.0
    max_lod = n_mips - 1
    if lod > max_lod:
        lod = float(max_lod)
    level0 = int(pymath.floor(lod))
    level1 = level0 + 1
    if level1 > max_lod:
        level1 = max_lod
    f = lod - level0
    n0x, n0y, n0z = _sample_ocean_normal_mip_level(
        x, y, level0,
        mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims,
        inv_tile_extent,
    )
    if level1 == level0:
        nx = n0x
        ny = n0y
        nz = n0z
    else:
        n1x, n1y, n1z = _sample_ocean_normal_mip_level(
            x, y, level1,
            mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims,
            inv_tile_extent,
        )
        nx = n0x + f * (n1x - n0x)
        ny = n0y + f * (n1y - n0y)
        nz = n0z + f * (n1z - n0z)
    inv_len = 1.0 / pymath.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * inv_len, ny * inv_len, nz * inv_len


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
def _ocean_shade_legacy(o_x, o_y, d_x, d_y, d_z,
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


@njit(inline="always")
def _reflection_sky_realism(dx, dy, dz, sun_dx, sun_dy, sun_dz,
                            legacy_glint_weight):
    """Legacy reflected sky with its fixed glint faded out by the master gate."""
    t = dz if dz > 0.0 else 0.0
    one_minus = 1.0 - t
    t = 1.0 - one_minus * one_minus * one_minus
    zen_r = 0.0044; zen_g = 0.035; zen_b = 0.1156
    hor_r = 0.10;   hor_g = 0.18;  hor_b = 0.38
    sky_r = hor_r + (zen_r - hor_r) * t
    sky_g = hor_g + (zen_g - hor_g) * t
    sky_b = hor_b + (zen_b - hor_b) * t
    cos_rs = dx * sun_dx + dy * sun_dy + dz * sun_dz
    if cos_rs > 0.0 and legacy_glint_weight > 0.0:
        glint_w = 0.02
        a = glint_w / ((1.0 - cos_rs) + glint_w)
        sky_r += legacy_glint_weight * a * 1.2
        sky_g += legacy_glint_weight * a * 1.0
        sky_b += legacy_glint_weight * a * 0.6
    return sky_r, sky_g, sky_b


@njit(inline="always")
def _ggx_smith_g1(n_dot_x, alpha_squared):
    """Smith masking for a GGX distribution parameterized by RMS slope."""
    root = pymath.sqrt(
        alpha_squared + (1.0 - alpha_squared) * n_dot_x * n_dot_x
    )
    return (2.0 * n_dot_x) / (n_dot_x + root)


@njit
def _ocean_shade_realism(
    o_x, o_y, d_x, d_y, d_z, t_hit, pixel_angular_span,
    sun_dx, sun_dy, sun_dz, t_sun_ocean,
    sun_r, sun_g, sun_b,
    beam_r, beam_g, beam_b,
    ocean_rr, ocean_rg, ocean_rb,
    fif_dx, fif_N,
    mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
    ocean_realism, ocean_mip_bias,
    ocean_glint_strength, ocean_glint_roughness,
    ocean_glint_roughness_per_lod,
    ocean_haze_extinction_per_km,
    sky_hor_r, sky_hor_g, sky_hor_b,
    sky_bloom_r, sky_bloom_g, sky_bloom_b,
    low_sun_sky_field_strength,
):
    """Footprint-filtered ocean with microfacet sun glint and path haze."""
    # Project one pixel's angular span onto the horizontal water plane. The
    # grazing-angle factor is what drives the horizon to coarser mip levels.
    grazing = abs(d_z)
    if grazing < 0.03:
        grazing = 0.03
    ocean_span = t_hit * pixel_angular_span / grazing
    texel_span = ocean_span / fif_dx
    if texel_span < 1.0:
        texel_span = 1.0
    lod = pymath.log(texel_span) * 1.4426950408889634 + ocean_mip_bias
    if lod < 0.0:
        lod = 0.0
    max_lod = n_mips - 1
    if lod > max_lod:
        lod = float(max_lod)
    # Scaling LOD as well as the light-transfer terms makes intermediate
    # master-gate values a useful, continuous tuning range.
    lod *= ocean_realism

    tile_extent = fif_N * fif_dx
    nx, ny, nz = _ocean_wave_normal_mipped(
        o_x, o_y, lod,
        mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
        1.0 / tile_extent,
    )

    # Reflect the view direction around the filtered surface normal.
    vdotn = d_x * nx + d_y * ny + d_z * nz
    r_x = d_x - 2.0 * vdotn * nx
    r_y = d_y - 2.0 * vdotn * ny
    r_z = d_z - 2.0 * vdotn * nz
    if r_z < 0.0:
        r_z = -r_z
    legacy_glint_weight = 1.0 - ocean_realism
    sky_rr, sky_rg, sky_rb = _reflection_sky_realism(
        r_x, r_y, r_z, sun_dx, sun_dy, sun_dz,
        legacy_glint_weight,
    )

    # Water Fresnel for the resolved sky reflection.
    n_dot_v = -vdotn
    if n_dot_v < 0.0:
        n_dot_v = 0.0
    if n_dot_v > 1.0:
        n_dot_v = 1.0
    one_minus = 1.0 - n_dot_v
    om2 = one_minus * one_minus
    view_fresnel = 0.02 + 0.98 * om2 * om2 * one_minus

    n_dot_l = sun_dx * nx + sun_dy * ny + sun_dz * nz
    if n_dot_l < 0.0:
        n_dot_l = 0.0

    # Direct-sun GGX glint. Mip filtering removes unresolved slope variance;
    # increasing alpha with LOD folds that variance back into a broader,
    # stable highlight rather than reintroducing point-sample sparkles.
    glint_weight = ocean_realism * ocean_glint_strength
    glint_r = 0.0
    glint_g = 0.0
    glint_b = 0.0
    sun_fresnel = 0.02
    if glint_weight > 0.0 and n_dot_l > 0.0 and n_dot_v > 1e-8:
        h_x = sun_dx - d_x
        h_y = sun_dy - d_y
        h_z = sun_dz - d_z
        h_len = pymath.sqrt(h_x * h_x + h_y * h_y + h_z * h_z)
        if h_len > 1e-8:
            inv_h_len = 1.0 / h_len
            h_x *= inv_h_len
            h_y *= inv_h_len
            h_z *= inv_h_len
            n_dot_h = nx * h_x + ny * h_y + nz * h_z
            if n_dot_h > 0.0:
                v_dot_h = (-d_x * h_x - d_y * h_y - d_z * h_z)
                if v_dot_h < 0.0:
                    v_dot_h = 0.0
                if v_dot_h > 1.0:
                    v_dot_h = 1.0
                one_minus_vh = 1.0 - v_dot_h
                vh2 = one_minus_vh * one_minus_vh
                sun_fresnel = 0.02 + 0.98 * vh2 * vh2 * one_minus_vh

                alpha = (ocean_glint_roughness
                         + ocean_glint_roughness_per_lod * lod)
                if alpha < 0.02:
                    alpha = 0.02
                if alpha > 0.75:
                    alpha = 0.75
                alpha_squared = alpha * alpha
                denom = (n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0)
                D = alpha_squared / (
                    3.14159265358979 * denom * denom
                )
                G = (_ggx_smith_g1(n_dot_v, alpha_squared)
                     * _ggx_smith_g1(n_dot_l, alpha_squared))
                spec = (glint_weight * t_sun_ocean * D * G * sun_fresnel
                        / (4.0 * n_dot_v))
                glint_r = spec * beam_r
                glint_g = spec * beam_g
                glint_b = spec * beam_b

    # Direct subsurface light uses the complement of the incident Fresnel
    # allocation, so enabling the specular sun path does not create energy
    # without taking it from the transmitted/diffuse path.
    energy_weight = glint_weight
    if energy_weight > 1.0:
        energy_weight = 1.0
    diffuse_partition = 1.0 - energy_weight * sun_fresnel
    inv_pi = 1.0 / 3.14159265358979
    diff_irr = t_sun_ocean * n_dot_l * inv_pi * diffuse_partition
    diff_r = diff_irr * sun_r * ocean_rr
    diff_g = diff_irr * sun_g * ocean_rg
    diff_b = diff_irr * sun_b * ocean_rb

    one_minus_F = 1.0 - view_fresnel
    ol_r = view_fresnel * sky_rr + one_minus_F * diff_r + glint_r
    ol_g = view_fresnel * sky_rg + one_minus_F * diff_g + glint_g
    ol_b = view_fresnel * sky_rb + one_minus_F * diff_b + glint_b

    shadow_floor = 0.35
    t_eff = shadow_floor + (1.0 - shadow_floor) * t_sun_ocean
    ol_r *= t_eff
    ol_g *= t_eff
    ol_b *= t_eff

    # Ocean-only aerial perspective. t_hit is already the slant path length,
    # so Beer-Lambert extinction naturally increases toward grazing angles.
    haze_tau = (ocean_realism * ocean_haze_extinction_per_km
                * 0.001 * t_hit)
    if haze_tau > 0.0:
        haze = 1.0 - pymath.exp(-haze_tau)
        h_len = pymath.sqrt(d_x * d_x + d_y * d_y)
        if h_len > 1e-8:
            h_x = d_x / h_len
            h_y = d_y / h_len
        else:
            h_x = d_x
            h_y = d_y
        # Use the same angular sky field at this sightline's azimuth and the
        # geometric horizon, so the asymptotic haze color equals the adjacent
        # sky. The solar disc remains excluded as in the legacy haze.
        haze_r, haze_g, haze_b = _sky_radiance(
            h_x, h_y, 0.0,
            sun_dx, sun_dy, sun_dz,
            sky_hor_r, sky_hor_g, sky_hor_b,
            sky_bloom_r, sky_bloom_g, sky_bloom_b,
            0.0, 0.0, 0.0,
            low_sun_sky_field_strength,
        )
        one_minus_haze = 1.0 - haze
        ol_r = one_minus_haze * ol_r + haze * haze_r
        ol_g = one_minus_haze * ol_g + haze * haze_g
        ol_b = one_minus_haze * ol_b + haze * haze_b

    return ol_r, ol_g, ol_b


@njit
def _ocean_shade_dispatch(
    o_x, o_y, d_x, d_y, d_z, t_hit, pixel_angular_span,
    sun_dx, sun_dy, sun_dz, t_sun_ocean,
    sun_r, sun_g, sun_b,
    beam_r, beam_g, beam_b,
    ocean_rr, ocean_rg, ocean_rb,
    fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_dx, fif_N,
    mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
    ocean_realism, ocean_mip_bias,
    ocean_glint_strength, ocean_glint_roughness,
    ocean_glint_roughness_per_lod,
    ocean_haze_extinction_per_km,
    sky_hor_r, sky_hor_g, sky_hor_b,
    sky_bloom_r, sky_bloom_g, sky_bloom_b,
    low_sun_sky_field_strength,
):
    """Keep master-gate zero on the untouched legacy arithmetic path."""
    if ocean_realism == 0.0:
        return _ocean_shade_legacy(
            o_x, o_y, d_x, d_y, d_z,
            sun_dx, sun_dy, sun_dz, t_sun_ocean,
            sun_r, sun_g, sun_b,
            ocean_rr, ocean_rg, ocean_rb,
            fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_N,
        )
    return _ocean_shade_realism(
        o_x, o_y, d_x, d_y, d_z, t_hit, pixel_angular_span,
        sun_dx, sun_dy, sun_dz, t_sun_ocean,
        sun_r, sun_g, sun_b,
        beam_r, beam_g, beam_b,
        ocean_rr, ocean_rg, ocean_rb,
        fif_dx, fif_N,
        mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
        ocean_realism, ocean_mip_bias,
        ocean_glint_strength, ocean_glint_roughness,
        ocean_glint_roughness_per_lod,
        ocean_haze_extinction_per_km,
        sky_hor_r, sky_hor_g, sky_hor_b,
        sky_bloom_r, sky_bloom_g, sky_bloom_b,
        low_sun_sky_field_strength,
    )


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
    cloud_sun_r, cloud_sun_g, cloud_sun_b,
    ambient_tint_r, ambient_tint_g, ambient_tint_b,
    sky_hor_r, sky_hor_g, sky_hor_b,
    sky_bloom_r, sky_bloom_g, sky_bloom_b,
    sky_disc_r, sky_disc_g, sky_disc_b,
    low_sun_sky_field_strength,
    g_hg, ambient_strength,
    ocean_enabled, ocean_z,
    ocean_rr, ocean_rg, ocean_rb,
    fif_nx, fif_ny, fif_nz, fif_dx,
    mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
    ocean_realism, ocean_mip_bias,
    ocean_glint_strength, ocean_glint_roughness,
    ocean_glint_roughness_per_lod,
    ocean_haze_extinction_per_km,
    step_voxel_factor,   # dt_max = min(active_level_dx) * this
    max_steps,
    powder_coeff,        # powder = 1 - exp(-powder_coeff * tau_depth)
    gradient_shading_strength,
    gradient_coarse_weight,
    gradient_coarse_radius_m,
    cone_stencil_tan_theta,
    deep_shadow_ms_suppression,
    ambient_occlusion_strength,
    ambient_occlusion_floor,
    light_transfer_split_strength,
    bounce_depth_attenuation,
    sample_index,
    samples_per_pixel,
    image,
):
    aspect = img_w / img_h
    iso_phase = 1.0 / (4.0 * 3.14159265358979)
    inv_fif_dx = 1.0 / fif_dx
    fif_N = fif_nx.shape[0]
    pixel_angular_span = 2.0 * tan_half_fov / img_h

    n_pixels = img_w * img_h
    for pixel_idx in prange(n_pixels):
        py = pixel_idx // img_w
        px = pixel_idx % img_w

        # SPP=1 is deliberately the exact legacy sampling path. At higher
        # SPP, an R2 low-discrepancy sequence is independently rotated per
        # pixel. It spreads rays across the pixel without RNG state or the
        # clustering of independent white-noise samples.
        if samples_per_pixel == 1:
            sample_x = px + 0.5
            sample_y = py + 0.5
        else:
            subpixel_x = (
                _sampling_hash(pixel_idx, 0)
                + sample_index * 0.7548776662466927
            )
            subpixel_y = (
                _sampling_hash(pixel_idx, 1)
                + sample_index * 0.5698402909980532
            )
            subpixel_x -= pymath.floor(subpixel_x)
            subpixel_y -= pymath.floor(subpixel_y)
            sample_x = px + subpixel_x
            sample_y = py + subpixel_y

        ndc_x = (2.0 * sample_x / img_w - 1.0) * aspect * tan_half_fov
        ndc_y = (1.0 - 2.0 * sample_y / img_h) * tan_half_fov

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
            if samples_per_pixel == 1:
                # Exact legacy phase: first sample lies on the AABB entry.
                t = t_near
            else:
                # Stratify the first-step phase across samples and rotate it
                # per pixel. The entry level selects the physical step scale,
                # including when a fine nested grid touches the outer AABB.
                entry_x = cam_ox + t_near * d_x
                entry_y = cam_oy + t_near * d_y
                entry_z = cam_oz + t_near * d_z
                entry_level = _active_level(
                    entry_x, entry_y, entry_z,
                    n_levels, level_bmin, level_bmax,
                )
                if entry_level < 0:
                    entry_level = n_levels - 1
                entry_dx = level_dxs[entry_level, 0]
                if level_dxs[entry_level, 1] < entry_dx:
                    entry_dx = level_dxs[entry_level, 1]
                if level_dxs[entry_level, 2] < entry_dx:
                    entry_dx = level_dxs[entry_level, 2]
                phase = (
                    _sampling_hash(pixel_idx, 2)
                    + (sample_index + 0.5) / samples_per_pixel
                )
                phase -= pymath.floor(phase)
                t = t_near + phase * entry_dx * step_voxel_factor

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
                    ol_r, ol_g, ol_b = _ocean_shade_dispatch(
                        o_x, o_y, d_x, d_y, d_z,
                        t_ocean, pixel_angular_span,
                        sun_dx, sun_dy, sun_dz, t_sun_ocean,
                        sun_r, sun_g, sun_b,
                        cloud_sun_r, cloud_sun_g, cloud_sun_b,
                        ocean_rr, ocean_rg, ocean_rb,
                        fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_dx, fif_N,
                        mip_nx, mip_ny, mip_nz,
                        mip_offsets, mip_dims, n_mips,
                        ocean_realism, ocean_mip_bias,
                        ocean_glint_strength, ocean_glint_roughness,
                        ocean_glint_roughness_per_lod,
                        ocean_haze_extinction_per_km,
                        sky_hor_r, sky_hor_g, sky_hor_b,
                        sky_bloom_r, sky_bloom_g, sky_bloom_b,
                        low_sun_sky_field_strength,
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

                deep_shadow_gate = 0.0
                if (deep_shadow_ms_suppression > 0.0
                        or ambient_occlusion_strength > 0.0
                        or light_transfer_split_strength > 0.0):
                    deep_shadow_gate = _smoothstep(
                        DEEP_SHADOW_TAU_START, DEEP_SHADOW_TAU_FULL, tau_sun
                    )

                ms_r = 0.0; ms_g = 0.0; ms_b = 0.0
                ms_atten = 1.0
                for octave in range(MS_OCTAVES):
                    t_sun_ms = pymath.exp(-tau_sun * ms_atten)
                    blend = min(1.0, octave * MS_BLEND_RATE)
                    oct_phase = phase_hg * (1.0 - blend) + iso_phase * blend
                    contrib = ms_atten * t_sun_ms * oct_phase
                    if deep_shadow_ms_suppression > 0.0:
                        iso_gate = _smoothstep(0.35, 1.0, blend)
                        ms_floor = 1.0 - (
                            deep_shadow_ms_suppression
                            * deep_shadow_gate
                            * iso_gate
                        )
                        if ms_floor < DEEP_SHADOW_MS_FLOOR:
                            ms_floor = DEEP_SHADOW_MS_FLOOR
                        contrib *= ms_floor
                    ms_r += contrib * cloud_sun_r
                    ms_g += contrib * cloud_sun_g
                    ms_b += contrib * cloud_sun_b
                    ms_atten *= MS_ATTEN

                if light_transfer_split_strength > 0.0:
                    direct_factor = 1.0 + (
                        light_transfer_split_strength
                        * LIGHT_TRANSFER_DIRECT_BOOST
                        * pymath.exp(-tau_sun)
                    )
                    ms_r *= direct_factor
                    ms_g *= direct_factor
                    ms_b *= direct_factor

                if gradient_shading_strength > 0.0:
                    grad_x, grad_y, grad_z, grad_conf = _sigma_gradient_level(
                        k, p_x, p_y, p_z, sigma, t,
                        gradient_coarse_weight,
                        gradient_coarse_radius_m,
                        cone_stencil_tan_theta,
                        sigma_stacked, level_offsets, level_dims,
                        level_bmin, level_dxs,
                    )
                    grad_len = pymath.sqrt(
                        grad_x * grad_x + grad_y * grad_y + grad_z * grad_z
                    )
                    if grad_len > 1e-12:
                        surface_gate = (
                            _smoothstep(
                                GRADIENT_SHADING_TAU_START,
                                GRADIENT_SHADING_TAU_FULL,
                                tau_depth,
                            )
                            * _smoothstep(
                                GRADIENT_SHADING_CONF_START,
                                GRADIENT_SHADING_CONF_FULL,
                                grad_conf,
                            )
                        )
                        n_dot_sun = -(
                            grad_x * sun_dx + grad_y * sun_dy + grad_z * sun_dz
                        ) / grad_len
                        if n_dot_sun < 0.0:
                            n_dot_sun *= GRADIENT_SHADING_SHADOW_SIDE_SCALE
                        gradient_factor = 1.0 + (
                            gradient_shading_strength * surface_gate * n_dot_sun
                        )
                        if gradient_factor < 0.20:
                            gradient_factor = 0.20
                        ms_r *= gradient_factor
                        ms_g *= gradient_factor
                        ms_b *= gradient_factor

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
                if ambient_occlusion_strength > 0.0:
                    amb_factor = 1.0 - ambient_occlusion_strength * deep_shadow_gate
                    if amb_factor < ambient_occlusion_floor:
                        amb_factor = ambient_occlusion_floor
                    amb *= amb_factor
                amb_weight = transmittance * d_tau * amb
                col_r += amb_weight * ambient_tint_r
                col_g += amb_weight * ambient_tint_g
                col_b += amb_weight * ambient_tint_b

                # The directional sun ray cannot estimate visibility of the
                # whole sky hemisphere. Restore a cool diffuse floor only in
                # saturated sun shadow; lit faces retain the existing direct
                # radiance and contrast.
                if light_transfer_split_strength > 0.0:
                    sky_fill = (
                        light_transfer_split_strength
                        * LIGHT_TRANSFER_SHADOW_SKYLIGHT
                        * (AMBIENT_HEIGHT_FLOOR
                           + (1.0 - AMBIENT_HEIGHT_FLOOR) * height_frac)
                        * deep_shadow_gate
                    )
                    sky_fill_weight = transmittance * d_tau * sky_fill
                    col_r += sky_fill_weight * ambient_tint_r
                    col_g += sky_fill_weight * ambient_tint_g
                    col_b += sky_fill_weight * ambient_tint_b

                # Surface bounce: upward diffuse light anchored at z=0, not
                # the data AABB floor, so elevated domains do not receive
                # full bounce at their lowest data voxel.
                if BOUNCE_STRENGTH > 0.0:
                    bounce_frac = 1.0 - p_z / outer_bmax_z
                    if bounce_frac < 0.0: bounce_frac = 0.0
                    if bounce_frac > 1.0: bounce_frac = 1.0
                    bounce = BOUNCE_STRENGTH * bounce_frac
                    if bounce_depth_attenuation > 0.0:
                        bounce *= pymath.exp(-bounce_depth_attenuation * tau_depth)
                    bounce_weight = transmittance * d_tau * bounce
                    col_r += bounce_weight * BOUNCE_TINT_R
                    col_g += bounce_weight * BOUNCE_TINT_G
                    col_b += bounce_weight * BOUNCE_TINT_B

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
                ol_r, ol_g, ol_b = _ocean_shade_dispatch(
                    o_x, o_y, d_x, d_y, d_z,
                    t_ocean, pixel_angular_span,
                    sun_dx, sun_dy, sun_dz, t_sun_ocean,
                    sun_r, sun_g, sun_b,
                    cloud_sun_r, cloud_sun_g, cloud_sun_b,
                    ocean_rr, ocean_rg, ocean_rb,
                    fif_nx, fif_ny, fif_nz, inv_fif_dx, fif_dx, fif_N,
                    mip_nx, mip_ny, mip_nz,
                    mip_offsets, mip_dims, n_mips,
                    ocean_realism, ocean_mip_bias,
                    ocean_glint_strength, ocean_glint_roughness,
                    ocean_glint_roughness_per_lod,
                    ocean_haze_extinction_per_km,
                    sky_hor_r, sky_hor_g, sky_hor_b,
                    sky_bloom_r, sky_bloom_g, sky_bloom_b,
                    low_sun_sky_field_strength,
                )
                col_r += transmittance * ol_r
                col_g += transmittance * ol_g
                col_b += transmittance * ol_b
                transmittance = 0.0

        if transmittance > 0.002:
            sky_r, sky_g, sky_b = _sky_radiance(d_x, d_y, d_z,
                                                  sun_dx, sun_dy, sun_dz,
                                                  sky_hor_r, sky_hor_g, sky_hor_b,
                                                  sky_bloom_r, sky_bloom_g, sky_bloom_b,
                                                  sky_disc_r, sky_disc_g, sky_disc_b,
                                                  low_sun_sky_field_strength)
            col_r += transmittance * sky_r
            col_g += transmittance * sky_g
            col_b += transmittance * sky_b

        if samples_per_pixel == 1:
            # Exact legacy write path: no otherwise-redundant multiply by 1.
            image[py, px, 0] = col_r
            image[py, px, 1] = col_g
            image[py, px, 2] = col_b
        else:
            sample_weight = 1.0 / samples_per_pixel
            image[py, px, 0] += col_r * sample_weight
            image[py, px, 1] += col_g * sample_weight
            image[py, px, 2] += col_b * sample_weight


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


def _prepare_fif_normals(fif_nx, fif_ny, fif_nz, fif_dx):
    """Validate and canonicalize the square periodic FIF normal map."""
    fif_nx = np.ascontiguousarray(fif_nx, dtype=np.float32)
    fif_ny = np.ascontiguousarray(fif_ny, dtype=np.float32)
    fif_nz = np.ascontiguousarray(fif_nz, dtype=np.float32)
    if (fif_nx.ndim != 2 or fif_nx.shape != fif_ny.shape
            or fif_nx.shape != fif_nz.shape):
        raise ValueError(
            "fif_normals must contain matching 2D nx/ny/nz arrays; "
            f"got {fif_nx.shape}, {fif_ny.shape}, {fif_nz.shape}."
        )
    if fif_nx.shape[0] != fif_nx.shape[1]:
        raise ValueError(
            "The witness FIF sampler requires a square periodic tile; "
            f"got {fif_nx.shape}."
        )
    if fif_nx.shape[0] < 1:
        raise ValueError("The witness FIF normal tile must not be empty.")
    if not np.isfinite(fif_dx) or fif_dx <= 0.0:
        raise ValueError(f"FIF dx must be positive and finite; got {fif_dx!r}.")
    if (not np.isfinite(fif_nx).all()
            or not np.isfinite(fif_ny).all()
            or not np.isfinite(fif_nz).all()):
        raise ValueError("FIF normals must be finite.")
    return fif_nx, fif_ny, fif_nz, float(fif_dx)


def _build_fif_normal_mips(fif_nx, fif_ny, fif_nz):
    """Pack a periodic, box-filtered, renormalized FIF normal mip chain."""
    dims = []
    h, w = fif_nx.shape
    while True:
        dims.append((w, h))
        if w == 1 and h == 1:
            break
        w = max(1, (w + 1) // 2)
        h = max(1, (h + 1) // 2)

    sizes = [w_level * h_level for w_level, h_level in dims]
    offsets = np.zeros(len(dims), dtype=np.int64)
    if len(dims) > 1:
        offsets[1:] = np.cumsum(sizes[:-1], dtype=np.int64)
    total = int(sum(sizes))
    mip_nx = np.empty(total, dtype=np.float32)
    mip_ny = np.empty(total, dtype=np.float32)
    mip_nz = np.empty(total, dtype=np.float32)
    mip_dims = np.asarray(dims, dtype=np.int64)

    base_size = sizes[0]
    mip_nx[:base_size] = fif_nx.ravel()
    mip_ny[:base_size] = fif_ny.ravel()
    mip_nz[:base_size] = fif_nz.ravel()

    for level in range(1, len(dims)):
        prev_w, prev_h = dims[level - 1]
        cur_w, cur_h = dims[level]
        prev_start = offsets[level - 1]
        prev_stop = prev_start + sizes[level - 1]
        cur_start = offsets[level]
        cur_stop = cur_start + sizes[level]

        prev_x = mip_nx[prev_start:prev_stop].reshape(prev_h, prev_w)
        prev_y = mip_ny[prev_start:prev_stop].reshape(prev_h, prev_w)
        prev_z = mip_nz[prev_start:prev_stop].reshape(prev_h, prev_w)
        y0 = (np.arange(cur_h, dtype=np.int64) * 2) % prev_h
        y1 = (y0 + 1) % prev_h
        x0 = (np.arange(cur_w, dtype=np.int64) * 2) % prev_w
        x1 = (x0 + 1) % prev_w

        down_x = (
            prev_x[y0[:, None], x0[None, :]]
            + prev_x[y1[:, None], x0[None, :]]
            + prev_x[y0[:, None], x1[None, :]]
            + prev_x[y1[:, None], x1[None, :]]
        ) * np.float32(0.25)
        down_y = (
            prev_y[y0[:, None], x0[None, :]]
            + prev_y[y1[:, None], x0[None, :]]
            + prev_y[y0[:, None], x1[None, :]]
            + prev_y[y1[:, None], x1[None, :]]
        ) * np.float32(0.25)
        down_z = (
            prev_z[y0[:, None], x0[None, :]]
            + prev_z[y1[:, None], x0[None, :]]
            + prev_z[y0[:, None], x1[None, :]]
            + prev_z[y1[:, None], x1[None, :]]
        ) * np.float32(0.25)
        inv_len = 1.0 / np.maximum(
            np.sqrt(down_x * down_x + down_y * down_y + down_z * down_z),
            np.float32(1e-12),
        )
        mip_nx[cur_start:cur_stop] = (down_x * inv_len).ravel()
        mip_ny[cur_start:cur_stop] = (down_y * inv_len).ravel()
        mip_nz[cur_start:cur_stop] = (down_z * inv_len).ravel()

    return mip_nx, mip_ny, mip_nz, offsets, mip_dims


def _spectral_lighting_colors(
    sun_direction: Tuple[float, float, float],
    sun_color: Tuple[float, float, float],
    strength: float,
):
    """Precompute low-sun cloud and sky spectra from relative air mass.

    ``sun_color`` is the renderer's calibrated beam at the reference solar
    elevation. Only the additional air mass below that elevation is applied,
    which keeps the approved high-sun look stable while giving dawn and
    golden-hour shots a physical spectral separation.
    """
    legacy_ambient = (AMBIENT_TINT_R, AMBIENT_TINT_G, AMBIENT_TINT_B)
    legacy_horizon = (0.10, 0.18, 0.38)
    legacy_bloom = (0.8, 0.6, 0.3)
    legacy_disc = (50.0, 45.0, 35.0)
    if strength == 0.0:
        return (sun_color, legacy_ambient, legacy_horizon,
                legacy_bloom, legacy_disc)

    def air_mass(elevation_deg):
        # Kasten-Young relative optical air mass, clamped near the horizon
        # where the plane-parallel approximation otherwise diverges.
        elevation_deg = max(0.0, elevation_deg)
        denom = (pymath.sin(pymath.radians(elevation_deg))
                 + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)
        return min(ATMOSPHERE_MAX_AIRMASS, 1.0 / denom)

    sun_z = max(-1.0, min(1.0, sun_direction[2]))
    elevation_deg = pymath.degrees(pymath.asin(sun_z))
    extra_air_mass = max(
        0.0,
        air_mass(elevation_deg)
        - air_mass(ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG),
    )

    optical_depths = []
    for wavelength_nm in ATMOSPHERE_RGB_WAVELENGTHS_NM:
        wavelength_ratio = 550.0 / wavelength_nm
        rayleigh_od = ATMOSPHERE_RAYLEIGH_OD_550 * wavelength_ratio ** 4.0
        aerosol_od = (ATMOSPHERE_AEROSOL_OD_550
                      * wavelength_ratio ** ATMOSPHERE_AEROSOL_ANGSTROM)
        optical_depths.append(rayleigh_od + aerosol_od)

    # Beer-Lambert transmittance through only the extra low-sun atmosphere.
    # Interpolation in transmittance space makes strength=0 an exact bypass.
    beam_scale = tuple(
        1.0 - strength * (1.0 - pymath.exp(-extra_air_mass * tau))
        for tau in optical_depths
    )
    cloud_sun = tuple(sun_color[c] * beam_scale[c] for c in range(3))

    # Diffuse sky fill follows the spectrum removed by one vertical column.
    # Preserve the legacy fill luminance: this changes shadow chromaticity,
    # while the attenuated direct beam naturally raises its relative weight.
    scattered = tuple(1.0 - pymath.exp(-tau) for tau in optical_depths)
    legacy_luma = (0.2126 * legacy_ambient[0]
                   + 0.7152 * legacy_ambient[1]
                   + 0.0722 * legacy_ambient[2])
    scattered_luma = (0.2126 * scattered[0]
                      + 0.7152 * scattered[1]
                      + 0.0722 * scattered[2])
    fill_target = tuple(c * legacy_luma / scattered_luma for c in scattered)
    fill_mix = strength * (1.0 - pymath.exp(-0.6 * extra_air_mass))
    ambient = tuple(
        legacy_ambient[c] + fill_mix * (fill_target[c] - legacy_ambient[c])
        for c in range(3)
    )

    # Low-altitude scattered light forms a broad warm band toward the solar
    # azimuth. Directional application happens in _sky_radiance; this is the
    # per-frame spectral endpoint only.
    horizon_mix = strength * (1.0 - pymath.exp(-0.45 * extra_air_mass))
    horizon = tuple(
        legacy_horizon[c]
        + horizon_mix * (SUNSET_HORIZON_RADIANCE[c] - legacy_horizon[c])
        for c in range(3)
    )

    bloom = tuple(legacy_bloom[c] * beam_scale[c] for c in range(3))
    disc = tuple(legacy_disc[c] * beam_scale[c] for c in range(3))
    return cloud_sun, ambient, horizon, bloom, disc


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
    ocean_realism: float,
    ocean_mip_bias: float,
    ocean_glint_strength: float,
    ocean_glint_roughness: float,
    ocean_glint_roughness_per_lod: float,
    ocean_haze_extinction_per_km: float,
    sun_color: Tuple[float, float, float],
    spectral_lighting_strength: float,
    low_sun_sky_field_strength: float,
    g_hg: float,
    ambient_strength: float,
    powder_coeff: float,
    gradient_shading_strength: float,
    gradient_coarse_weight: float,
    gradient_coarse_radius_m: float,
    cone_stencil_theta_deg: float,
    deep_shadow_ms_suppression: float,
    ambient_occlusion_strength: float,
    ambient_occlusion_floor: float,
    light_transfer_split_strength: float,
    bounce_depth_attenuation: float,
    witness_spp: int,
    verbose: bool,
) -> np.ndarray:
    """Pack levels, warm up, render, and return the linear HDR buffer."""
    if len(levels) == 0:
        raise ValueError("Need at least one level.")
    if (not pymath.isfinite(cone_stencil_theta_deg) or
            cone_stencil_theta_deg < 0.0 or cone_stencil_theta_deg >= 90.0):
        raise ValueError("cone_stencil_theta_deg must be finite and in [0, 90).")
    if (not pymath.isfinite(spectral_lighting_strength) or
            spectral_lighting_strength < 0.0 or spectral_lighting_strength > 1.0):
        raise ValueError("spectral_lighting_strength must be finite and in [0, 1].")
    if (not pymath.isfinite(low_sun_sky_field_strength) or
            low_sun_sky_field_strength < 0.0 or
            low_sun_sky_field_strength > 1.0):
        raise ValueError(
            "low_sun_sky_field_strength must be finite and in [0, 1]."
        )
    if (not pymath.isfinite(LOW_SUN_SKY_WARM_ELEVATION_DEG) or
            LOW_SUN_SKY_WARM_ELEVATION_DEG <= 0.0 or
            LOW_SUN_SKY_WARM_ELEVATION_DEG > 90.0):
        raise ValueError(
            "low-sun sky warm elevation must be finite and in (0, 90]."
        )
    if (not pymath.isfinite(LOW_SUN_SKY_HORIZON_AZIMUTH_DEG) or
            LOW_SUN_SKY_HORIZON_AZIMUTH_DEG <= 0.0 or
            LOW_SUN_SKY_HORIZON_AZIMUTH_DEG > 180.0):
        raise ValueError(
            "low-sun sky horizon azimuth must be finite and in (0, 180]."
        )
    if (not pymath.isfinite(LOW_SUN_SKY_UPPER_AZIMUTH_DEG) or
            LOW_SUN_SKY_UPPER_AZIMUTH_DEG <= 0.0 or
            LOW_SUN_SKY_UPPER_AZIMUTH_DEG >
            LOW_SUN_SKY_HORIZON_AZIMUTH_DEG):
        raise ValueError(
            "low-sun sky upper azimuth must be finite, positive, and no "
            "wider than its horizon azimuth."
        )
    if any(not pymath.isfinite(c) or c < 0.0
           for c in LOW_SUN_SKY_NEUTRAL_RADIANCE):
        raise ValueError(
            "low-sun sky neutral radiance must be finite and nonnegative."
        )
    if (not pymath.isfinite(ocean_realism) or
            ocean_realism < 0.0 or ocean_realism > 1.0):
        raise ValueError("ocean_realism must be finite and in [0, 1].")
    if not pymath.isfinite(ocean_mip_bias):
        raise ValueError("ocean_mip_bias must be finite.")
    if (not pymath.isfinite(ocean_glint_strength)
            or ocean_glint_strength < 0.0):
        raise ValueError("ocean_glint_strength must be finite and nonnegative.")
    if (not pymath.isfinite(ocean_glint_roughness)
            or ocean_glint_roughness < 0.0):
        raise ValueError("ocean_glint_roughness must be finite and nonnegative.")
    if (not pymath.isfinite(ocean_glint_roughness_per_lod)
            or ocean_glint_roughness_per_lod < 0.0):
        raise ValueError(
            "ocean_glint_roughness_per_lod must be finite and nonnegative."
        )
    if (not pymath.isfinite(ocean_haze_extinction_per_km)
            or ocean_haze_extinction_per_km < 0.0):
        raise ValueError(
            "ocean_haze_extinction_per_km must be finite and nonnegative."
        )
    if (isinstance(witness_spp, (bool, np.bool_)) or
            not isinstance(witness_spp, (int, np.integer)) or witness_spp < 1):
        raise ValueError("witness_spp must be a positive integer.")
    if (not pymath.isfinite(light_transfer_split_strength)
            or light_transfer_split_strength < 0.0
            or light_transfer_split_strength > 1.0):
        raise ValueError(
            "light_transfer_split_strength must be finite and in [0, 1]."
        )
    if (LIGHT_TRANSFER_FULL_ELEVATION_DEG
            >= LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG):
        raise ValueError(
            "light-transfer full elevation must be below its cutoff."
        )

    fif_nx, fif_ny, fif_nz, fif_dx = _prepare_fif_normals(
        fif_nx, fif_ny, fif_nz, fif_dx
    )
    if ocean_enabled and ocean_realism > 0.0:
        (mip_nx, mip_ny, mip_nz,
         mip_offsets, mip_dims) = _build_fif_normal_mips(
            fif_nx, fif_ny, fif_nz
        )
    else:
        mip_nx = fif_nx.ravel()
        mip_ny = fif_ny.ravel()
        mip_nz = fif_nz.ravel()
        mip_offsets = np.zeros(1, dtype=np.int64)
        mip_dims = np.asarray(
            [[fif_nx.shape[1], fif_nx.shape[0]]], dtype=np.int64
        )
    n_mips = len(mip_offsets)

    cone_stencil_tan_theta = pymath.tan(
        pymath.radians(cone_stencil_theta_deg)
    )
    sun_elevation_deg = pymath.degrees(pymath.asin(
        max(-1.0, min(1.0, sun_direction[2]))
    ))
    if (sun_elevation_deg
            >= LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG - 1e-6):
        light_transfer_split_strength = 0.0
    elif sun_elevation_deg > LIGHT_TRANSFER_FULL_ELEVATION_DEG:
        low_sun_mix = (
            (LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG - sun_elevation_deg)
            / (LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG
               - LIGHT_TRANSFER_FULL_ELEVATION_DEG)
        )
        low_sun_mix = low_sun_mix * low_sun_mix * (3.0 - 2.0 * low_sun_mix)
        light_transfer_split_strength *= low_sun_mix
    (cloud_sun, ambient_tint, sky_horizon,
     sky_bloom, sky_disc) = _spectral_lighting_colors(
        sun_direction, sun_color, spectral_lighting_strength
    )

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
        cloud_sun[0], cloud_sun[1], cloud_sun[2],
        ambient_tint[0], ambient_tint[1], ambient_tint[2],
        sky_horizon[0], sky_horizon[1], sky_horizon[2],
        sky_bloom[0], sky_bloom[1], sky_bloom[2],
        sky_disc[0], sky_disc[1], sky_disc[2],
        low_sun_sky_field_strength,
        g_hg, ambient_strength,
        ocean_enabled, ocean_z,
        ocean_reflectance[0], ocean_reflectance[1], ocean_reflectance[2],
        fif_nx, fif_ny, fif_nz, fif_dx,
        mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
        ocean_realism, ocean_mip_bias,
        ocean_glint_strength, ocean_glint_roughness,
        ocean_glint_roughness_per_lod,
        ocean_haze_extinction_per_km,
        step_voxel_factor, 32, powder_coeff,
        gradient_shading_strength,
        gradient_coarse_weight,
        gradient_coarse_radius_m,
        cone_stencil_tan_theta,
        deep_shadow_ms_suppression,
        ambient_occlusion_strength,
        ambient_occlusion_floor,
        light_transfer_split_strength,
        bounce_depth_attenuation,
        0, 1,
        warmup,
    )
    if verbose:
        print(" done")
        print("  Rendering...", end="", flush=True)

    t0 = time.perf_counter()
    for sample_index in range(witness_spp):
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
            cloud_sun[0], cloud_sun[1], cloud_sun[2],
            ambient_tint[0], ambient_tint[1], ambient_tint[2],
            sky_horizon[0], sky_horizon[1], sky_horizon[2],
            sky_bloom[0], sky_bloom[1], sky_bloom[2],
            sky_disc[0], sky_disc[1], sky_disc[2],
            low_sun_sky_field_strength,
            g_hg, ambient_strength,
            ocean_enabled, ocean_z,
            ocean_reflectance[0], ocean_reflectance[1], ocean_reflectance[2],
            fif_nx, fif_ny, fif_nz, fif_dx,
            mip_nx, mip_ny, mip_nz, mip_offsets, mip_dims, n_mips,
            ocean_realism, ocean_mip_bias,
            ocean_glint_strength, ocean_glint_roughness,
            ocean_glint_roughness_per_lod,
            ocean_haze_extinction_per_km,
            step_voxel_factor, max_steps, powder_coeff,
            gradient_shading_strength,
            gradient_coarse_weight,
            gradient_coarse_radius_m,
            cone_stencil_tan_theta,
            deep_shadow_ms_suppression,
            ambient_occlusion_strength,
            ambient_occlusion_floor,
            light_transfer_split_strength,
            bounce_depth_attenuation,
            sample_index, witness_spp,
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
    spectral_lighting_strength: float = SPECTRAL_LIGHTING_STRENGTH,
    g_hg: float = G_HG,
    ambient_strength: float = AMBIENT_STRENGTH,
    powder_coeff: float = POWDER_COEFF,
    gradient_shading_strength: float = GRADIENT_SHADING_STRENGTH,
    gradient_coarse_weight: float = GRADIENT_SHADING_COARSE_WEIGHT,
    gradient_coarse_radius_m: float = GRADIENT_SHADING_COARSE_RADIUS_M,
    cone_stencil_theta_deg: float = CONE_STENCIL_THETA_DEG,
    deep_shadow_ms_suppression: float = DEEP_SHADOW_MS_SUPPRESSION,
    ambient_occlusion_strength: float = AMBIENT_OCCLUSION_STRENGTH,
    ambient_occlusion_floor: float = AMBIENT_OCCLUSION_FLOOR,
    light_transfer_split_strength: float = LIGHT_TRANSFER_SPLIT_STRENGTH,
    bounce_depth_attenuation: float = BOUNCE_DEPTH_ATTENUATION,
    witness_spp: int = WITNESS_SPP,
    return_linear: bool = False,
    verbose: bool = True,
    ocean_realism: float = OCEAN_REALISM,
    ocean_mip_bias: float = OCEAN_MIP_BIAS,
    ocean_glint_strength: float = OCEAN_GLINT_STRENGTH,
    ocean_glint_roughness: float = OCEAN_GLINT_ROUGHNESS,
    ocean_glint_roughness_per_lod: float = OCEAN_GLINT_ROUGHNESS_PER_LOD,
    ocean_haze_extinction_per_km: float = OCEAN_HAZE_EXTINCTION_PER_KM,
    low_sun_sky_field_strength: float = LOW_SUN_SKY_FIELD_STRENGTH,
) -> np.ndarray:
    """Render through N strictly-nested extinction grids.

    Levels are finest-first (index 0 = highest resolution); the outermost
    (last) level defines the outer AABB the ray is clipped against.

    cone_stencil_theta_deg sets the coarse-gradient angular radius; 0 uses
        the legacy fixed world-space radius.

    spectral_lighting_strength blends from the legacy fixed spectra at 0 to
        elevation-dependent direct sun, diffuse fill, and main-sky color at 1.

    low_sun_sky_field_strength blends from the iter_006 azimuth-only warm sky
        at 0 to the elevation-and-azimuth warm wedge at 1.

    light_transfer_split_strength controls the low-sun warm-direct/cool-
        diffuse separation; 0 selects the exact previous cloud shader.

    witness_spp controls deterministic still-image sampling. 1 is the exact
        legacy pixel-center/no-jitter path; higher values average jittered rays.

    fif_normals: (nx, ny, nz, dx_m) tuple from
        cloudyview.ocean_fif.generate_fif_normals. Required when ocean_enabled
        is True; the kernel samples the FIF normal map with periodic wrap at
        each ocean hit. Pass None (with ocean_enabled=False) for sky-only.

    ocean_realism controls the footprint-filtered normal, spectral glint, and
        ocean-haze pass. 0 selects the exact legacy ocean shader.
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
        ocean_realism, ocean_mip_bias,
        ocean_glint_strength, ocean_glint_roughness,
        ocean_glint_roughness_per_lod,
        ocean_haze_extinction_per_km,
        sun_color, spectral_lighting_strength,
        low_sun_sky_field_strength,
        g_hg, ambient_strength, powder_coeff,
        gradient_shading_strength,
        gradient_coarse_weight,
        gradient_coarse_radius_m,
        cone_stencil_theta_deg,
        deep_shadow_ms_suppression,
        ambient_occlusion_strength,
        ambient_occlusion_floor,
        light_transfer_split_strength,
        bounce_depth_attenuation,
        witness_spp,
        verbose,
    )
    if return_linear:
        return image
    return tone_map(image, exposure=exposure)


# ============================================================================
# Library render function (exported as cloudyview.witness)
# ============================================================================

def witness(
    field: CloudField,
    camera: Optional[Camera] = None,
    *,
    size: Optional[Tuple[int, int]] = None,
    sun_azimuth: Optional[float] = None,
    sun_elevation: Optional[float] = None,
    exposure: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Render a cloud field with the fast volumetric ray marcher.

    Parameters
    ----------
    field : CloudField
        Loaded cloud field (see :func:`cloudyview.load`).
    camera : Camera, optional
        Viewpoint; defaults to the standard witness camera.
    size : (width, height), optional
        Image size in pixels (default from config: 600x400).
    sun_azimuth, sun_elevation : float, optional
        Sun direction in degrees (met bearing / above horizon);
        defaults from config (20 / 55).
    exposure : float, optional
        Tone-mapping exposure (default from config: 4.0).
    verbose : bool
        Print render diagnostics (the CLI uses this); default silent.

    Returns
    -------
    ndarray (height, width, 3), float64
        Tone-mapped RGB image in [0, 1].
    """
    witness_config = config.get_witness_config()
    sun_config = witness_config['sun']
    render_config = witness_config['rendering']

    if camera is None:
        camera = Camera()
    if sun_azimuth is None:
        sun_azimuth = sun_config['azimuth']
    if sun_elevation is None:
        sun_elevation = sun_config['elevation']
    if exposure is None:
        exposure = render_config['exposure']

    img_w = size[0] if size else render_config['width']
    img_h = size[1] if size else render_config['height']

    x_coord, y_coord, z_coord = field.x, field.y, field.z
    lw_np = field.lwc
    nx_d, ny_d, nz_d = lw_np.shape

    iw_np = field.iwc
    if iw_np is not None and np.max(iw_np) < 1e-6:
        iw_np = None
    if iw_np is None and verbose:
        print("  No ice water content detected; using liquid-only extinction.")

    # Physical extinction in m^-1 (not scaled by height_z: the kernel
    # works in absolute meters, so sigma is already in physical units).
    sigma_ext = optical_depth.compute_extinction_field(
        lw_np, z_coord, re=10.0, iwc=iw_np, re_ice=30.0)

    geom = compute_domain_geometry(x_coord, y_coord, z_coord, nx_d, ny_d, nz_d)

    if verbose:
        print(f"  Grid: {nx_d} x {ny_d} x {nz_d}, spacing: {geom.dx:.1f} x {geom.dy:.1f} m")
        print(f"  Domain: {geom.width_x:.0f} x {geom.width_y:.0f} x {geom.height_z:.0f} m")

    ext_mult = render_config['extinction_multiplier']
    sigma_world = (sigma_ext * ext_mult).astype(np.float64)
    sigma_world = np.ascontiguousarray(sigma_world)

    if verbose:
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
    rel_pos = camera.position
    cam_origin = np.empty(3, dtype=np.float64)
    cam_origin[0] = bmin[0] + (rel_pos[0] + 1.0) * 0.5 * (bmax[0] - bmin[0])
    cam_origin[1] = bmin[1] + (rel_pos[1] + 1.0) * 0.5 * (bmax[1] - bmin[1])
    cam_origin[2] = (rel_pos[2] + 1.0) * 0.5 * bmax[2]

    forward, right, up = camera.basis()

    sun_dir = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)

    # Ocean sits at the physical surface, not the AABB floor: rel=-1 is
    # z=0 (sea level). Default height=-0.9999 → ~0 for any domain top.
    # Reduces to the old mapping when bmin[2]==0.
    ocean_config = render_config['ocean']
    ocean_enabled = ocean_config['enabled']
    ocean_z = (ocean_config['height'] + 1.0) * 0.5 * bmax[2]

    n_light_steps = render_config['n_light_steps']

    if verbose:
        print(f"  Camera: abs=({cam_origin[0]:.1f},{cam_origin[1]:.1f},{cam_origin[2]:.1f}) m")
        print(f"          azimuth={camera.azimuth:.1f} elev={camera.elevation:.1f} fov={camera.fov:.1f}")
        print(f"  Sun: azimuth={sun_azimuth:.1f} elev={sun_elevation:.1f}")
        print(f"  Image: {img_w}x{img_h}")


    if ocean_enabled:
        from cloudyview.ocean_fif import generate_fif_normals
        fif_nx, fif_ny, fif_nz, fif_dx = generate_fif_normals(verbose=verbose)
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
        camera.fov, n_light_steps,
        STEP_VOXEL_FACTOR, MAX_STEPS,
        ocean_enabled, ocean_z, OCEAN_REFLECTANCE,
        fif_nx, fif_ny, fif_nz, fif_dx,
        OCEAN_REALISM, OCEAN_MIP_BIAS,
        OCEAN_GLINT_STRENGTH, OCEAN_GLINT_ROUGHNESS,
        OCEAN_GLINT_ROUGHNESS_PER_LOD,
        OCEAN_HAZE_EXTINCTION_PER_KM,
        SUN_COLOR, SPECTRAL_LIGHTING_STRENGTH,
        LOW_SUN_SKY_FIELD_STRENGTH,
        G_HG, AMBIENT_STRENGTH, POWDER_COEFF,
        GRADIENT_SHADING_STRENGTH,
        GRADIENT_SHADING_COARSE_WEIGHT,
        GRADIENT_SHADING_COARSE_RADIUS_M,
        CONE_STENCIL_THETA_DEG,
        DEEP_SHADOW_MS_SUPPRESSION,
        AMBIENT_OCCLUSION_STRENGTH,
        AMBIENT_OCCLUSION_FLOOR,
        LIGHT_TRANSFER_SPLIT_STRENGTH,
        BOUNCE_DEPTH_ATTENUATION,
        WITNESS_SPP,
        verbose=verbose,
    )
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
