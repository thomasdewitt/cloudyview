// Time-of-day lighting: what colour the sun, sky and ambient fill are for a
// given solar elevation. Port of cloudyview/look.py::_spectral_lighting_colors
// and its neighbours.
//
// The desktop build could bake these into a uniform template because its sun
// never moved between exports. Ours does — the time-of-day panel is a slider —
// so the whole calculation has to run per frame in JS.

"use strict";

import {
  SUN_COLOR, LEGACY_AMBIENT, LEGACY_HORIZON, LEGACY_BLOOM, LEGACY_DISC,
  ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG, ATMOSPHERE_MAX_AIRMASS,
  ATMOSPHERE_RAYLEIGH_OD_550, ATMOSPHERE_AEROSOL_OD_550,
  ATMOSPHERE_AEROSOL_ANGSTROM, ATMOSPHERE_RGB_WAVELENGTHS_NM,
  SUNSET_HORIZON_RADIANCE,
  LIGHT_TRANSFER_FULL_ELEVATION_DEG, LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG,
  HAZE_ANCHOR, AERIAL_BETA_PER_KM, AERIAL_BETA_FLOOR_PER_KM,
  OCEAN_HAZE_EXTINCTION_PER_KM,
} from "./constants.js";

const DEG = Math.PI / 180.0;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Python's floor-mod. JS `%` truncates, which breaks negative azimuths. */
export const mod360 = (a) => ((a % 360.0) + 360.0) % 360.0;

/**
 * Meteorological azimuth/elevation to a unit world vector.
 * +x east, +y north, +z up; azimuth 0 = north, 90 = east, clockwise.
 */
export function directionFromAzimuthElevation(azimuthDeg, elevationDeg) {
  const azInternal = mod360(90.0 - mod360(azimuthDeg));
  const az = azInternal * DEG;
  const el = elevationDeg * DEG;
  const cosEl = Math.cos(el);
  const d = [cosEl * Math.cos(az), cosEl * Math.sin(az), Math.sin(el)];
  const n = Math.hypot(d[0], d[1], d[2]);
  return [d[0] / n, d[1] / n, d[2] / n];
}

/** Kasten-Young relative air mass, capped. Elevation in degrees. */
export function airMass(elevationDeg) {
  const e = Math.max(0.0, elevationDeg);
  const denom = Math.sin(e * DEG) + 0.50572 * Math.pow(e + 6.07995, -1.6364);
  return Math.min(ATMOSPHERE_MAX_AIRMASS, 1.0 / denom);
}

// Per-channel atmospheric optical depth at the RGB wavelengths: Rayleigh
// (lambda^-4) plus an Angstrom aerosol term. Constant, so computed once.
const OPTICAL_DEPTHS = ATMOSPHERE_RGB_WAVELENGTHS_NM.map((nm) => {
  const r = 550.0 / nm;
  return ATMOSPHERE_RAYLEIGH_OD_550 * Math.pow(r, 4.0)
       + ATMOSPHERE_AEROSOL_OD_550 * Math.pow(r, ATMOSPHERE_AEROSOL_ANGSTROM);
});

const luma = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

/**
 * The five per-frame colours the shader needs, as a function of where the sun
 * is. Returns {cloudSun, ambient, horizon, bloom, disc}.
 *
 * A low sun travels through more air, so the direct beam loses blue first
 * (the beam scale) while the diffuse fill gains it. The fill is renormalized
 * against the legacy tint's luma so that turning this on redistributes colour
 * without changing overall brightness.
 *
 * At the reference elevation (55 degrees) — or at strength 0 — every output
 * equals its legacy constant exactly. That identity is what keeps the WGSL's
 * legacy paths bit-exact, and it is the first thing to test.
 */
export function spectralLightingColors(
  sunDirection, sunColor = SUN_COLOR, strength = 1.0,
) {
  if (strength === 0.0) {
    return {
      cloudSun: [...sunColor],
      ambient: [...LEGACY_AMBIENT],
      horizon: [...LEGACY_HORIZON],
      bloom: [...LEGACY_BLOOM],
      disc: [...LEGACY_DISC],
    };
  }

  const sunZ = clamp(sunDirection[2], -1.0, 1.0);
  const elevationDeg = Math.asin(sunZ) / DEG;
  const extraAirMass = Math.max(
    0.0,
    airMass(elevationDeg) - airMass(ATMOSPHERE_REFERENCE_SUN_ELEVATION_DEG),
  );

  const beamScale = OPTICAL_DEPTHS.map(
    (tau) => 1.0 - strength * (1.0 - Math.exp(-extraAirMass * tau)));

  const scattered = OPTICAL_DEPTHS.map((tau) => 1.0 - Math.exp(-tau));
  const fillScale = luma(LEGACY_AMBIENT) / luma(scattered);
  const fillMix = strength * (1.0 - Math.exp(-0.6 * extraAirMass));
  const horizonMix = strength * (1.0 - Math.exp(-0.45 * extraAirMass));

  const out = { cloudSun: [], ambient: [], horizon: [], bloom: [], disc: [] };
  for (let c = 0; c < 3; c++) {
    out.cloudSun.push(sunColor[c] * beamScale[c]);
    const fillTarget = scattered[c] * fillScale;
    out.ambient.push(
      LEGACY_AMBIENT[c] + fillMix * (fillTarget - LEGACY_AMBIENT[c]));
    out.horizon.push(
      LEGACY_HORIZON[c] +
      horizonMix * (SUNSET_HORIZON_RADIANCE[c] - LEGACY_HORIZON[c]));
    out.bloom.push(LEGACY_BLOOM[c] * beamScale[c]);
    out.disc.push(LEGACY_DISC[c] * beamScale[c]);
  }
  return out;
}

/**
 * The light-transfer split fades out as the sun climbs, reaching exactly zero
 * at the cutoff so the approved high-sun look is untouched. The 1e-6 slack is
 * load-bearing: it keeps the default 55-degree sun on the legacy path.
 */
export function effectiveLightTransferSplit(strength, sunElevationDeg) {
  if (!(strength >= 0.0 && strength <= 1.0)) {
    throw new Error(
      `light_transfer_split_strength must be in [0, 1]; got ${strength}.`);
  }
  if (sunElevationDeg >= LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG - 1e-6) return 0.0;
  if (sunElevationDeg > LIGHT_TRANSFER_FULL_ELEVATION_DEG) {
    const m = (LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG - sunElevationDeg)
            / (LIGHT_TRANSFER_CUTOFF_ELEVATION_DEG
               - LIGHT_TRANSFER_FULL_ELEVATION_DEG);
    return strength * (m * m * (3.0 - 2.0 * m));  // smoothstep
  }
  return strength;
}

// Aerosol loading rises faster than linearly with the slider, so haze 1 is a
// real murk (0.11/km, ~36 km visual range by Koschmieder) rather than a
// slightly softer default. Twin of look.py's _AERIAL_BETA_HAZE_COEFFICIENT.
const AERIAL_BETA_HAZE_COEFFICIENT =
  (AERIAL_BETA_PER_KM - AERIAL_BETA_FLOOR_PER_KM)
  / (HAZE_ANCHOR * Math.sqrt(HAZE_ANCHOR));
const OCEAN_HAZE_BETA_RATIO =
  OCEAN_HAZE_EXTINCTION_PER_KM / AERIAL_BETA_PER_KM;

/**
 * Sea-level clear-air extinction for a haze setting in [0, 1].
 *
 * h**1.5 is spelled h*sqrt(h) because sqrt is correctly rounded by IEEE-754
 * and Math.pow is not: this number is packed into a uniform block that
 * tests/test_uniform_parity.py diffs byte for byte against Python.
 */
export function aerialBetaPerKm(haze) {
  return AERIAL_BETA_FLOOR_PER_KM
    + AERIAL_BETA_HAZE_COEFFICIENT * (haze * Math.sqrt(haze));
}

/** Haze over the sea, held at its tuned ratio to the sky's own extinction. */
export function oceanHazeExtinctionPerKm(haze) {
  return OCEAN_HAZE_BETA_RATIO * aerialBetaPerKm(haze);
}
