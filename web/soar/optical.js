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
