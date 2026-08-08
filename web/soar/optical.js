// Condensate to extinction. Port of cloudyview/optical_depth.py.
//
// sigma = (3 Q_ext / 4) * q * rho_air(z) / (rho_particle * r_eff), with
// Q_ext = 2 in the geometric-optics limit — hence the 1.5 folded into the
// prefactors in constants.js. q is a mixing ratio in g/kg, sigma is m^-1.
//
// The atmosphere is a fixed isothermal profile, NOT the field's own: the
// same assumption witness and behold make, so all three agree.

"use strict";

import {
  AIR_P0, AIR_R, AIR_T, AIR_SCALE_HEIGHT_M,
  SIGMA_LIQUID_PREFACTOR, SIGMA_ICE_PREFACTOR,
  TAU_PER_LWP_UM, EC92_BAND1_A, EC92_BAND1_B,
  RE_LIQUID_UM, RE_ICE_UM, TWO_STREAM_DENOM,
} from "./constants.js";

/** Air density (kg/m^3) at height z (m). */
export function rhoAirAt(z) {
  return AIR_P0 * Math.exp(-z / AIR_SCALE_HEIGHT_M) / (AIR_R * AIR_T);
}

/** Per-level density table — the reason no voxel needs an exp. */
export function rhoAirTable(zCoords) {
  const table = new Float64Array(zCoords.length);
  for (let k = 0; k < zCoords.length; k++) table[k] = rhoAirAt(zCoords[k]);
  return table;
}

/**
 * Extinction (m^-1) for one voxel, given mixing ratios in g/kg.
 *
 * Nothing is clamped — negative condensate, which some models write, comes
 * through as negative extinction, and NaN propagates. That is what the
 * desktop does; a browser that quietly cleaned up would disagree with every
 * other renderer in the package.
 */
export function sigmaAt(lwcGkg, iwcGkg, rhoAir) {
  return (SIGMA_LIQUID_PREFACTOR * lwcGkg + SIGMA_ICE_PREFACTOR * iwcGkg)
       * rhoAir;
}

/**
 * Cell thickness (m) for cell-centred heights.
 *
 * Interior cells get half the gap below plus half the gap above; the two
 * boundary cells get their sole neighbour's whole gap, which is the same
 * thing under the assumption that the grid does not change spacing right at
 * the edge. Verbatim from optical_depth.py, because a column integral that
 * disagreed with glimpse by a cell would show as a brightness offset.
 */
export function cellThickness(zCoords) {
  const n = zCoords.length;
  const dz = new Float64Array(n);
  if (n === 1) { dz[0] = 1.0; return dz; }
  dz[0] = zCoords[1] - zCoords[0];
  for (let k = 1; k < n - 1; k++) {
    dz[k] = 0.5 * (zCoords[k + 1] - zCoords[k - 1]);
  }
  dz[n - 1] = zCoords[n - 1] - zCoords[n - 2];
  return dz;
}

/** Column optical depth from liquid and ice water paths (g/m^2). */
export function opticalDepthFromWaterPaths(lwpGm2, iwpGm2) {
  return lwpGm2 * (TAU_PER_LWP_UM / RE_LIQUID_UM)
       + iwpGm2 * (EC92_BAND1_A + EC92_BAND1_B / RE_ICE_UM);
}

/** Two-stream visual albedo in [0, 1) from column optical depth. */
export function twoStreamAlbedo(tau) {
  return tau / (tau + TWO_STREAM_DENOM);
}
