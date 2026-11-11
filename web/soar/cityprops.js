// What stands in the night city besides its buildings, as solids: the
// elevated freeways, the street lamps, the parked cars, the skybridges and
// the warehouse roofs' ridges. citylots.js is the buildings; this is
// everything the components add that a flight can run into, mirrored from
// their WGSL (cloudyview/soar/shader/city/components/) the way citylots.js mirrors
// city_cell — the same constants by the same names, the same draws in the
// same order — and held to the renderer by tests/city/test_city_props_mirror.py,
// which traces the shader's own city on the GPU and asks this file where
// each of those rays first meets something.
//
// Every solid is a convex polytope (polytope.js).
//
// The arithmetic here is float64 and the shader's float32. Positions agree to
// millimetres, which the mirror test allows for. A yes/no decision whose
// argument lands within a float32 rounding of its threshold — a slot's
// occupancy draw against its cut, a car's hover draw, a skybridge's gap
// against 62 m, a sawtooth's tooth count rounding off a half — can go the
// other way here, at odds of order 1e-7 per decision: the same standing as
// pow() in citylots.js's height law. The thresholds compared against a
// float32 hash draw are written f(...), where that is all it takes.

"use strict";

import { boxSolid, polytope } from "./polytope.js";
import {
  CITY_AVENUE_EXTRA, CITY_FLOOR_H, CITY_STREET_HALF, CITY_USE_INDUSTRIAL,
  CITY_USE_URBAN, cityCell, cityRand4, cityU01, isAvenue, pcg2d, tileCell,
} from "./citylots.js";

// --- skyway: the elevated freeways (components/skyway.wgsl) ----------------
//
// Not a property of any block: two families of routes on the avenue lattice,
// x-running decks on every avenue line y = 0 (mod 24 blocks) and y-running
// ones on x = 8 (mod 24 blocks), the y-family climbing over every crossing.
// Each route is its deck (a ribbon of five pieces on the raised family:
// base, ramp up, high flat, ramp down, base, sharing their end planes), a
// rail along each edge, and on 55 m bays a pier cap carried by two columns.
//
// One difference from the WGSL, and it is the WGSL's economy rather than a
// shape: per ray, the shader raises only the crossing nearest the stretch of
// deck that ray crosses (MC_SPAN), so a ray running kilometres down a deck
// sees a far crossing flat. Here every crossing is raised — the one nearest
// the segment, and a frame's segment is metres long — which is the network
// the shader draws everywhere it can be flown into.
export const cc_skyway_ROUTE_BLOCKS = 24.0;
export const cc_skyway_OFFSET_BLOCKS = 8.0;
export const cc_skyway_HW = 8.0;
export const cc_skyway_THICK = 1.4;
export const cc_skyway_DECK_Z = 26.0;
export const cc_skyway_HIGH_Z = 34.0;
export const cc_skyway_RAMP = 200.0;
export const cc_skyway_FLAT_HALF = 62.0;
export const cc_skyway_RAIL_H = 1.1;
export const cc_skyway_RAIL_W = 0.36;
export const cc_skyway_PYL_SP = 55.0;
export const cc_skyway_PYL_HW = 0.6;
export const cc_skyway_PYL_OFF = 5.2;
export const cc_skyway_CAP_H = 1.05;
export const cc_skyway_CAP_HL = 1.15;
export const cc_skyway_CAP_OS = 0.55;
export const cc_skyway_CROSS_CLEAR = 22.0;
export const cc_skyway_Z_TOP = 35.1;

/** components/skyway.wgsl cc_skyway_swap: the route's local frame (its own inverse). */
const swap = (v, fam) => (fam === 0 ? v : [v[1], v[0], v[2]]);

/** cc_skyway_ztop: the deck's top surface along a route. */
function skywayZtop(a, ac, raised) {
  if (!raised) return cc_skyway_DECK_Z;
  const d = Math.abs(a - ac);
  if (d <= cc_skyway_FLAT_HALF) return cc_skyway_HIGH_Z;
  if (d >= cc_skyway_FLAT_HALF + cc_skyway_RAMP) return cc_skyway_DECK_Z;
  return cc_skyway_HIGH_Z + (cc_skyway_DECK_Z - cc_skyway_HIGH_Z)
    * (d - cc_skyway_FLAT_HALF) / cc_skyway_RAMP;
}

/**
 * A polytope in a route's local frame (x along, y lateral from the route's
 * centreline at world lateral `latC`, z up), given as local planes, moved
 * to world. A local plane (nx, ny, nz, d) about the centreline is
 * (nx, ny, nz, d + ny latC) in the route's swapped world frame.
 */
function localSolid(planes, top, fam, latC) {
  return polytope(planes.map(([nx, ny, nz, d, ...grow]) => {
    const n = swap([nx, ny, nz], fam);
    return [n[0], n[1], n[2], d + ny * latC, ...grow];
  }), top, "skyway");
}

/**
 * A slab of the ribbon: along [a0, a1], lateral [y0, y1], and between h0 and
 * h1 above the sheared plane z = zb + m (a - ab). The sheared faces' unit
 * normal is (-m, 0, 1) / sqrt(1 + m^2), and their offsets scale with it.
 *
 * The two end faces are joints with the next piece of the same deck (or
 * a million metres off), and are not grown by the body's radius: grown, a
 * ramp's end face would stand m r + r (sqrt(1 + m^2) - 1) above the flat's
 * grown top, 1.4 cm for the camera — over the collider's 1 cm clearance, so
 * a camera resting on a raised span would meet a wall at the crest where the
 * road is smooth. Ungrown, the pieces' grown tops still meet,
 * within r (sqrt(1 + m^2) - 1), under a third of a millimetre.
 */
function ribbonSlab(a0, a1, y0, y1, ab, zb, m, h0, h1, top, fam, latC) {
  const k = 1.0 / Math.hypot(1.0, m);
  const c = zb - m * ab;            // z = c + m a on the plane
  return localSolid([
    [1, 0, 0, a1, 0.0], [-1, 0, 0, -a0, 0.0],
    [0, 1, 0, y1], [0, -1, 0, -y0],
    [-m * k, 0, k, (c + h1) * k], [m * k, 0, -k, -(c + h0) * k],
  ], top, fam, latC);
}

/** cc_skyway_ribbon's deck and rails for one piece of route. */
function ribbonPiece(out, a0, a1, ab, zb, m, fam, latC) {
  const zTop = Math.max(zb + m * (a0 - ab), zb + m * (a1 - ab));
  const hw = cc_skyway_HW;
  out.push(ribbonSlab(a0, a1, -hw, hw, ab, zb, m, -cc_skyway_THICK, 0.0,
                      zTop, fam, latC));
  for (const [y0, y1] of [[-hw, -hw + cc_skyway_RAIL_W],
                          [hw - cc_skyway_RAIL_W, hw]]) {
    out.push(ribbonSlab(a0, a1, y0, y1, ab, zb, m, 0.0, cc_skyway_RAIL_H,
                        zTop + cc_skyway_RAIL_H, fam, latC));
  }
}

/** cc_skyway_pylons: the bays whose cap reaches the along range [lo, hi]. */
function pylons(out, lo, hi, ac, raised, fam, latC) {
  const reach = cc_skyway_CAP_HL;
  const n0 = Math.ceil((lo - reach) / cc_skyway_PYL_SP);
  const n1 = Math.floor((hi + reach) / cc_skyway_PYL_SP);
  const capHw = cc_skyway_PYL_OFF + cc_skyway_PYL_HW + cc_skyway_CAP_OS;
  for (let n = n0; n <= n1; n++) {
    const s = n * cc_skyway_PYL_SP;
    if (raised && Math.abs(s - ac) < cc_skyway_CROSS_CLEAR) continue;
    const deck = skywayZtop(s, ac, raised);
    const soffit = deck - cc_skyway_THICK;
    const capBot = soffit - cc_skyway_CAP_H;
    const box = (x0, x1, y0, y1, z0, z1) => localSolid([
      [1, 0, 0, x1], [-1, 0, 0, -x0], [0, 1, 0, y1], [0, -1, 0, -y0],
      [0, 0, 1, z1], [0, 0, -1, -z0],
    ], deck, fam, latC);
    out.push(box(s - cc_skyway_CAP_HL, s + cc_skyway_CAP_HL,
                 -capHw, capHw, capBot, soffit));
    for (const lc of [-cc_skyway_PYL_OFF, cc_skyway_PYL_OFF]) {
      out.push(box(s - cc_skyway_PYL_HW, s + cc_skyway_PYL_HW,
                   lc - cc_skyway_PYL_HW, lc + cc_skyway_PYL_HW,
                   0.0, capBot));
    }
  }
}

/**
 * The freeway solids a sphere of radius r swept from p0 to p1 can meet, in
 * the city frame (`cellM` the block pitch).
 */
export function skywaySolids(p0, p1, r, cellM) {
  const out = [];
  if (Math.min(p0[2], p1[2]) > cc_skyway_Z_TOP + r
      || Math.max(p0[2], p1[2]) < -r) {
    return out;
  }
  const period = cc_skyway_ROUTE_BLOCKS * cellM;
  for (let fam = 0; fam < 2; fam++) {
    const l0 = swap(p0, fam), l1 = swap(p1, fam);
    const phase = fam === 1 ? cc_skyway_OFFSET_BLOCKS * cellM : 0.0;
    const reach = cc_skyway_PYL_OFF + cc_skyway_PYL_HW + cc_skyway_CAP_OS;
    const wLo = Math.min(l0[1], l1[1]) - Math.max(cc_skyway_HW, reach) - r;
    const wHi = Math.max(l0[1], l1[1]) + Math.max(cc_skyway_HW, reach) + r;
    const aLo = Math.min(l0[0], l1[0]) - r, aHi = Math.max(l0[0], l1[0]) + r;
    const raised = fam === 1;
    for (let mi = Math.ceil((wLo - phase) / period);
         mi <= Math.floor((wHi - phase) / period); mi++) {
      const latC = phase + mi * period;
      if (!raised) {
        ribbonPiece(out, -1e6, 1e6, 0.0, cc_skyway_DECK_Z, 0.0, fam, latC);
        pylons(out, aLo, aHi, 0.0, false, fam, latC);
        continue;
      }
      const ac = Math.round(0.5 * (aLo + aHi) / period) * period;
      const e = cc_skyway_FLAT_HALF, rp = cc_skyway_RAMP;
      const slope = (cc_skyway_HIGH_Z - cc_skyway_DECK_Z) / rp;
      ribbonPiece(out, -1e6, ac - e - rp, 0.0, cc_skyway_DECK_Z, 0.0,
                  fam, latC);
      ribbonPiece(out, ac - e - rp, ac - e, ac - e - rp, cc_skyway_DECK_Z,
                  slope, fam, latC);
      ribbonPiece(out, ac - e, ac + e, 0.0, cc_skyway_HIGH_Z, 0.0, fam, latC);
      ribbonPiece(out, ac + e, ac + e + rp, ac + e, cc_skyway_HIGH_Z, -slope,
                  fam, latC);
      ribbonPiece(out, ac + e + rp, 1e6, 0.0, cc_skyway_DECK_Z, 0.0,
                  fam, latC);
      pylons(out, aLo, aHi, ac, true, fam, latC);
    }
  }
  return out;
}

// --- streetlife: the lamp poles and the parked cars --------------------------
// (components/streetlife.wgsl, on the core's lamp lattice in city/layout.wgsl)
//
// Each of a block's four edges that carries a street has a lamp line, a
// mast at every station of the street's lattice standing wholly inside the
// block, its arm reaching out over the roadway to the luminaire; and, where
// the street parks on that side, a parking line with slots on a 7 m world
// lattice, about a third of them holding a car. A pole is three boxes (mast,
// arm, luminaire housing). A car is the two boxes the renderer draws it as
// from half a metre a pixel on (hull and greenhouse); nearer, the shader
// sphere-traces a hull inside the first, and that rounded shape is not
// mirrored: a crash meets the car's boxes, which are axis-aligned about a
// car yawed up to 5 degrees and square where its nose narrows and its glass
// rakes back, so tens of centimetres proud of the hull at a corner. The renderer stops drawing a pole's arm past 0.3 m a
// pixel and the props past a few metres a pixel; as solids they are always
// there.
export const CITY_LAMP_SPACING = 26.0;
export const CITY_LAMP_OFFSET = 2.5;
export const CITY_SIDEWALK_W = 3.0;
export const CITY_SIDEWALK_W_AVE = 4.5;
export const cc_streetlife_POLE_H = 9.0;
export const cc_streetlife_MAST_R = 0.085;
export const cc_streetlife_ARM_Z = 8.72;
export const cc_streetlife_ARM_HZ = 0.065;
export const cc_streetlife_ARM_REACH = 1.35;
export const cc_streetlife_HEAD_Z = 8.56;
export const cc_streetlife_HEAD_HZ = 0.115;
export const cc_streetlife_HEAD_HA = 0.17;
export const cc_streetlife_HEAD_HC = 0.40;
export const cc_streetlife_PARK_OFF = 1.20;
export const cc_streetlife_PARK_CLEAR = 6.0;
export const cc_streetlife_SLOT = 7.0;
export const cc_streetlife_OCC = 0.35;
export const cc_streetlife_DENS_LO = 0.55;
export const cc_streetlife_DENS_HI = 1.35;
export const cc_streetlife_DENS_START = 0.005;
export const cc_streetlife_DENS_FULL = 0.070;
export const cc_streetlife_YAW = 0.0873;
export const cc_streetlife_ALONG_JIT = 0.60;
export const cc_streetlife_LAT_JIT = 0.16;
export const cc_streetlife_BB_A = 2.60;
export const cc_streetlife_BB_C = 1.36;
export const cc_streetlife_ROOF = 1.46;
export const cc_streetlife_HOVER_CUT = 0.65;
export const cc_streetlife_HOVER_LIFT = 0.14;

const f = Math.fround;
const fract = (x) => x - Math.floor(x);
const smoothstep = (e0, e1, x) => {
  const t = Math.min(Math.max((x - e0) / (e1 - e0), 0.0), 1.0);
  return t * t * (3.0 - 2.0 * t);
};

/** city/layout.wgsl city_lamp_inset: the lamp line off centreline k. */
function lampInset(k) {
  return CITY_STREET_HALF + (isAvenue(k) ? CITY_AVENUE_EXTRA : 0.0)
    - CITY_LAMP_OFFSET;
}

/** city_line_seed: centreline k, folded at the tile, as a u32. */
const lineSeed = (k, n) => tileCell(n, k) >>> 0;

/** city_lamp_lattice: [spacing, phase] of the stations on centreline k. */
function lampLattice(k, n) {
  if (isAvenue(k)) return [CITY_LAMP_SPACING, 0.0];
  const h = pcg2d((lineSeed(k, n) ^ 0x3c6ef372) >>> 0, 0xa54ff53a);
  const sp = 22.0 + 3.0 * (h[0] & 3);
  return [sp, f(sp * cityU01(h[1]))];
}

/** city_park_side: which side of minor street k parks, +1 or -1. */
function parkSide(k, n) {
  const h = pcg2d((lineSeed(k, n) ^ 0x6c8e9cf5) >>> 0, 0x2c1b3c6d);
  return (h[0] & 1) === 1 ? 1.0 : -1.0;
}

/** city_sidewalk_w: building line to kerb on the street a plot edge faces. */
function sidewalkW(edge, cellM) {
  const gap = Math.abs(edge - Math.round(edge / cellM) * cellM);
  return gap > CITY_STREET_HALF + 1.0 ? CITY_SIDEWALK_W_AVE : CITY_SIDEWALK_W;
}

/** city_edge_street: does the block's `side` edge carry a street? */
const edgeStreet = (cc, side) => ((cc.streets >>> side) & 1) !== 0;

/** city_kerb_rect: [xmin, ymin, xmax, ymax] of the kerb line. */
function kerbRect(cc, cellM) {
  if (!(cc.built && cc.land_use === CITY_USE_URBAN)) {
    return [cc.plot_min[0], cc.plot_min[1], cc.plot_max[0], cc.plot_max[1]];
  }
  const w = (side, edge) => (edgeStreet(cc, side) ? sidewalkW(edge, cellM) : 0.0);
  return [cc.plot_min[0] - w(0, cc.plot_min[0]),
          cc.plot_min[1] - w(2, cc.plot_min[1]),
          cc.plot_max[0] + w(1, cc.plot_max[0]),
          cc.plot_max[1] + w(3, cc.plot_max[1])];
}

/** cc_streetlife_side: one edge's lamp line and parking line. */
function kerbSide(cx, cy, cc, side, cellM, n) {
  const cmin = [cx * cellM, cy * cellM];
  const cmax = [cmin[0] + cellM, cmin[1] + cellM];
  const kr = kerbRect(cc, cellM);
  const s = {};
  let kerbC;
  if (side === 0) {
    s.ax = 1; s.line = cx; s.lamp_c = cmin[0] + lampInset(cx);
    kerbC = kr[0]; s.plot_sign = 1.0;
  } else if (side === 1) {
    s.ax = 1; s.line = cx + 1; s.lamp_c = cmax[0] - lampInset(cx + 1);
    kerbC = kr[2]; s.plot_sign = -1.0;
  } else if (side === 2) {
    s.ax = 0; s.line = cy; s.lamp_c = cmin[1] + lampInset(cy);
    kerbC = kr[1]; s.plot_sign = 1.0;
  } else {
    s.ax = 0; s.line = cy + 1; s.lamp_c = cmax[1] - lampInset(cy + 1);
    kerbC = kr[3]; s.plot_sign = -1.0;
  }
  s.park_c = kerbC - s.plot_sign * cc_streetlife_PARK_OFF;
  s.a_min = s.ax === 0 ? cmin[0] : cmin[1];
  s.a_max = s.ax === 0 ? cmax[0] : cmax[1];
  s.c_min = s.ax === 0 ? cmin[1] : cmin[0];
  s.c_max = s.ax === 0 ? cmax[1] : cmax[0];
  const k0 = Math.round(s.a_min / cellM), k1 = Math.round(s.a_max / cellM);
  s.p_min = s.a_min + CITY_STREET_HALF
    + (isAvenue(k0) ? CITY_AVENUE_EXTRA : 0.0) + cc_streetlife_PARK_CLEAR;
  s.p_max = s.a_max - CITY_STREET_HALF
    - (isAvenue(k1) ? CITY_AVENUE_EXTRA : 0.0) - cc_streetlife_PARK_CLEAR;
  const sideOk = isAvenue(s.line) || parkSide(s.line, n) === s.plot_sign;
  s.park_ok = sideOk
    && s.park_c > s.c_min + cc_streetlife_BB_C
    && s.park_c < s.c_max - cc_streetlife_BB_C;
  return s;
}

/** A box given along the kerb ([a0, a1]) and across it ([c0, c1]). */
function kerbBox(s, a0, a1, c0, c1, z0, z1, source) {
  return s.ax === 0 ? boxSolid([a0, c0, z0], [a1, c1, z1], source)
                    : boxSolid([c0, a0, z0], [c1, a1, z1], source);
}

/** cc_streetlife_pole: mast, luminaire and arm of the pole at `a`. */
function pole(out, s, a) {
  const headC = s.lamp_c - s.plot_sign * cc_streetlife_ARM_REACH;
  const R = cc_streetlife_MAST_R;
  out.push(kerbBox(s, a - R, a + R, s.lamp_c - R, s.lamp_c + R,
                   0.0, cc_streetlife_POLE_H, "lamp"));
  out.push(kerbBox(s, a - cc_streetlife_HEAD_HA, a + cc_streetlife_HEAD_HA,
                   headC - cc_streetlife_HEAD_HC, headC + cc_streetlife_HEAD_HC,
                   cc_streetlife_HEAD_Z - cc_streetlife_HEAD_HZ,
                   cc_streetlife_HEAD_Z + cc_streetlife_HEAD_HZ, "lamp"));
  out.push(kerbBox(s, a - 0.055, a + 0.055,
                   Math.min(s.lamp_c, headC), Math.max(s.lamp_c, headC),
                   cc_streetlife_ARM_Z - cc_streetlife_ARM_HZ,
                   cc_streetlife_ARM_Z + cc_streetlife_ARM_HZ, "lamp"));
}

/** cc_streetlife_slot_draw: slot j of kerb `side`, by tile coordinate. */
function slotDraw(cx, cy, side, j, n) {
  const wx = tileCell(n, cx), wy = tileCell(n, cy);
  return cityRand4(
    (Math.imul(wx, 0x9e3779b9) + Math.imul(side, 0x2545f491)
     + Math.imul(j, 0x85ebca6b) + 0x51ed270b) >>> 0,
    (Math.imul(wy, 0xc2b2ae35) + Math.imul(side, 0x27d4eb2f)
     + Math.imul(j, 0x165667b1) + 0x9e3779b9) >>> 0);
}

/**
 * cc_streetlife_prop and cc_streetlife_trace_car_box: the car in slot j, as
 * its hull and greenhouse boxes, if the slot holds one.
 */
function car(out, s, cx, cy, cc, side, j, n) {
  const r = slotDraw(cx, cy, side, j, n);
  const dens = cc_streetlife_DENS_LO + (cc_streetlife_DENS_HI - cc_streetlife_DENS_LO)
    * smoothstep(cc_streetlife_DENS_START, cc_streetlife_DENS_FULL, cc.density);
  if (r[0] >= cc_streetlife_OCC * dens) return;
  const along = (j + 0.5) * cc_streetlife_SLOT
    + (r[1] - 0.5) * 2.0 * cc_streetlife_ALONG_JIT;
  if (along - cc_streetlife_BB_A < s.p_min
      || along + cc_streetlife_BB_A > s.p_max) {
    return;
  }
  const across = s.park_c + (r[2] - 0.5) * 2.0 * cc_streetlife_LAT_JIT;
  const base = s.ax === 1 ? [0.0, 1.0] : [1.0, 0.0];
  const perp = [base[1], base[0]];
  const yaw = (r[3] - 0.5) * 2.0 * cc_streetlife_YAW;
  let fwd = [base[0] * Math.cos(yaw) + perp[0] * Math.sin(yaw),
             base[1] * Math.cos(yaw) + perp[1] * Math.sin(yaw)];
  if (fract(f(r[1] * f(17.31))) > 0.5) fwd = [-fwd[0], -fwd[1]];
  const rgt = [fwd[1], -fwd[0]];
  const ctr = [base[0] * along + perp[0] * across,
               base[1] * along + perp[1] * across];
  const hover = fract(r[1] * 43.17 + r[2] * 7.31 + r[3] * 2.53)
    > cc_streetlife_HOVER_CUT;
  const lift = hover ? cc_streetlife_HOVER_LIFT : 0.0;
  const ext = (kf, kr) => [Math.abs(fwd[0]) * kf + Math.abs(rgt[0]) * kr,
                           Math.abs(fwd[1]) * kf + Math.abs(rgt[1]) * kr];
  const ea = ext(2.24, 0.90), eb = ext(1.06, 0.76);
  const top = lift + cc_streetlife_ROOF;
  out.push(boxSolid([ctr[0] - ea[0], ctr[1] - ea[1], hover ? lift + 0.10 : 0.02],
                    [ctr[0] + ea[0], ctr[1] + ea[1], lift + 1.00], "car", top));
  const g = [ctr[0] - 0.24 * fwd[0], ctr[1] - 0.24 * fwd[1]];
  out.push(boxSolid([g[0] - eb[0], g[1] - eb[1], lift + 0.94],
                    [g[0] + eb[0], g[1] + eb[1], top], "car", top));
}

/** Everything streetlife stands in block (cx, cy). */
function streetlifeSolids(out, cc, cx, cy, cellM, n) {
  for (let side = 0; side < 4; side++) {
    if (!edgeStreet(cc, side)) continue;
    const s = kerbSide(cx, cy, cc, side, cellM, n);
    const [sp, ph] = lampLattice(s.line, n);
    const j0 = Math.ceil((s.a_min + 0.45 - ph) / sp);
    const j1 = Math.floor((s.a_max - 0.45 - ph) / sp);
    for (let j = j0; j <= j1; j++) pole(out, s, f(ph + f(j * sp)));
    if (!s.park_ok) continue;
    const k0 = Math.floor(s.p_min / cc_streetlife_SLOT) - 1;
    const k1 = Math.ceil(s.p_max / cc_streetlife_SLOT) + 1;
    for (let j = k0; j <= k1; j++) car(out, s, cx, cy, cc, side, j, n);
  }
}

// --- skybridges (components/skybridges.wgsl) ----------------------------------
//
// A bridge belongs to the EDGE between two blocks, built identically from
// either side out of the two cells and the edge's own draws: a deck, a glass
// band (or a service truss's central web) and a roof, three boxes from wall
// to wall. Only downtown, between two built buildings, under both base
// roofs.
export const cc_skybridges_GATE = 0.118;
export const cc_skybridges_MIN_DENSITY = 0.25;
export const cc_skybridges_GAP_MIN = 8.0;
export const cc_skybridges_GAP_MAX = 62.0;
export const cc_skybridges_Z_MIN = 25.0;
export const cc_skybridges_Z_MAX = 380.0;
export const cc_skybridges_ROOF_MARGIN = 15.0;
export const cc_skybridges_DECK_T = 0.5;
export const cc_skybridges_GLASS_H = 2.6;
export const cc_skybridges_TUBE_H = 3.5;
export const cc_skybridges_HALF_W = 2.0;
export const cc_skybridges_HALF_W_GRAND = 4.0;
export const cc_skybridges_GRAND_FRAC = 0.22;
export const cc_skybridges_GLASS_INSET = 0.12;
export const cc_skybridges_ROOF_OVER = 0.18;
export const cc_skybridges_EMBED = 1.0;
export const cc_skybridges_ROW_MARGIN = 1.0;
export const cc_skybridges_TRUSS_FRAC = 0.18;
export const cc_skybridges_TRUSS_HALF = 0.30;
export const cc_skybridges_CHORD_FRAC = 0.55;

/** cc_skybridges_eid: the edge's id, from its lower cell, tile-wrapped. */
function bridgeEid(clo, axis, n) {
  const wx = tileCell(n, clo[0]), wy = tileCell(n, clo[1]);
  return [((2 * wx + (1 - axis)) ^ 0x5bf03635) >>> 0,
          ((2 * wy + axis) ^ 0x9e3779b9) >>> 0];
}

/** cc_skybridges_gate: one multiply-xorshift round on the unwrapped edge. */
function bridgeGate(clo, axis) {
  let h = (Math.imul(2 * clo[0] + (1 - axis), 0x9e3779b9)
           ^ Math.imul(2 * clo[1] + axis, 0x85ebca6b)) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  return cityU01(h) < f(cc_skybridges_GATE);
}

/** cc_skybridges_build: the bridge on one edge, or null. */
function bridgeSpan(clo, axis, ca, cb, d, cellM) {
  if (!ca.built || !cb.built) return null;
  if (ca.seed[0] === cb.seed[0] && ca.seed[1] === cb.seed[1]) return null;
  if (Math.min(ca.density, cb.density) <= cc_skybridges_MIN_DENSITY) return null;
  const ax = axis, cx = 1 - axis;
  const bnd = (clo[ax] + 1) * cellM;
  const aFace = ca.b1max[ax], bFace = cb.b1min[ax];
  const row = clo[cx];
  const reach = 0.5 * cellM - 2.0;
  if (Math.abs(aFace - bnd) > reach || Math.abs(bFace - bnd) > reach) return null;
  const gap = bFace - aFace;
  if (gap < cc_skybridges_GAP_MIN || gap > cc_skybridges_GAP_MAX) return null;
  const c0 = Math.max(ca.b1min[cx], cb.b1min[cx],
                      row * cellM + cc_skybridges_ROW_MARGIN);
  const c1 = Math.min(ca.b1max[cx], cb.b1max[cx],
                      (row + 1) * cellM - cc_skybridges_ROW_MARGIN);
  const frontage = c1 - c0;
  const truss = d[0] < f(cc_skybridges_TRUSS_FRAC);
  let half = cc_skybridges_HALF_W;
  if (!truss && d[3] < f(cc_skybridges_GRAND_FRAC)
      && frontage > 2.0 * cc_skybridges_HALF_W_GRAND + 3.0) {
    half = cc_skybridges_HALF_W_GRAND;
  }
  const slack = frontage - 2.0 * (half + cc_skybridges_ROOF_OVER) - 1.0;
  if (slack < 0.0) return null;
  const zTop = Math.min(ca.b1max[2], cb.b1max[2]) - cc_skybridges_ROOF_MARGIN;
  const zHi = Math.min(zTop - cc_skybridges_TUBE_H, cc_skybridges_Z_MAX);
  if (zHi < cc_skybridges_Z_MIN) return null;
  let z = f(Math.floor(f(f(cc_skybridges_Z_MIN
                           + f(f(zHi - cc_skybridges_Z_MIN) * d[1]))
                         / f(CITY_FLOOR_H))) * f(CITY_FLOOR_H));
  if (z < cc_skybridges_Z_MIN) z = f(z + f(CITY_FLOOR_H));
  return {
    axis, half, z, truss,
    lo: aFace - cc_skybridges_EMBED,
    hi: bFace + cc_skybridges_EMBED,
    ctr: c0 + half + cc_skybridges_ROOF_OVER + 0.5 + slack * d[2],
  };
}

/** The bridges on block (cx, cy)'s four edges. */
function skybridgeSolids(out, cc, cx, cy, tile) {
  if (!cc.built || cc.density <= cc_skybridges_MIN_DENSITY) return;
  for (let e = 0; e < 4; e++) {
    const axis = e >> 1, neg = (e & 1) === 1;
    const stp = axis === 1 ? [0, 1] : [1, 0];
    const clo = neg ? [cx - stp[0], cy - stp[1]] : [cx, cy];
    if (!bridgeGate(clo, axis)) continue;
    const [px, py] = bridgeEid(clo, axis, tile.n);
    const p0 = pcg2d(px, py);
    const other = neg ? clo : [clo[0] + stp[0], clo[1] + stp[1]];
    const nb = cityCell(tile, other[0], other[1]);
    const s = bridgeSpan(clo, axis, neg ? nb : cc, neg ? cc : nb,
                         cityRand4(p0[0], p0[1]), tile.cellM);
    if (s === null) continue;
    const zg = s.z + cc_skybridges_DECK_T;
    const zr = zg + cc_skybridges_GLASS_H;
    const zt = s.z + cc_skybridges_TUBE_H;
    const parts = [
      [s.half, s.z, zg],
      [s.truss ? cc_skybridges_TRUSS_HALF : s.half - cc_skybridges_GLASS_INSET,
       zg, zr],
      [s.truss ? s.half * cc_skybridges_CHORD_FRAC
               : s.half + cc_skybridges_ROOF_OVER, zr, zt],
    ];
    for (const [w, z0, z1] of parts) {
      const lo = [0, 0, z0], hi = [0, 0, z1];
      lo[axis] = s.lo; hi[axis] = s.hi;
      lo[1 - axis] = s.ctr - w; hi[1 - axis] = s.ctr + w;
      out.push(boxSolid(lo, hi, "bridge", zt));
    }
  }
}

// --- industry: the warehouse roofs' ridges (components/industry.wgsl) --------
//
// A warehouse's roof is the core's flat box top, and on it, by the lot's
// draw: a SAWTOOTH (teeth stepping along one axis, each a sheet sloping up
// to a vertical glazed face, so a triangular prism) or a MONITOR (a raised
// clerestory strip down the long axis, a box). Laid out per lot, so every
// member block builds the same roof. (The flat roofs' skylights, a few tens
// of centimetres proud of the roof, are not mirrored.)
export const cc_industry_SAW_FRAC = 0.40;
export const cc_industry_MONITOR_FRAC = 0.30;

/** cc_industry_draw. */
function industryDraw(cc, salt, i, j) {
  return cityRand4(
    (cc.seed[0] ^ Math.imul(salt, 0x9e3779b9) ^ Math.imul(i, 0x85ebca6b)) >>> 0,
    (cc.seed[1] ^ Math.imul(salt, 0x51ed270b) ^ Math.imul(j, 0xc2b2ae35)) >>> 0);
}

/** cc_industry_lot, the parts the roof reads. */
function industryLot(cc) {
  const bs = [cc.b1max[0] - cc.b1min[0], cc.b1max[1] - cc.b1min[1]];
  const longX = bs[0] >= bs[1];      // an industrial lot: the warehouse's axis
  return {
    longX,
    ba: longX ? [cc.b1min[0], cc.b1max[0]] : [cc.b1min[1], cc.b1max[1]],
    bc: longX ? [cc.b1min[1], cc.b1max[1]] : [cc.b1min[0], cc.b1max[0]],
  };
}

/** cc_industry_roof_of and the sawtooth trace's teeth, as solids. */
function roofSolids(out, cc) {
  if (!cc.built || cc.land_use !== CITY_USE_INDUSTRIAL) return;
  const d = industryDraw(cc, 11, 0, 0);
  const e = industryDraw(cc, 12, 0, 0);
  const l = industryLot(cc);
  const h = cc.height;
  if (d[0] < f(cc_industry_SAW_FRAC)) {
    const stepX = d[1] < f(0.70) ? l.longX : !l.longX;
    const ax = stepX ? 0 : 1, cx = 1 - ax;
    const a0 = cc.b1min[ax], a1 = cc.b1max[ax];
    const n = Math.max(Math.round((a1 - a0) / (8.0 + 4.0 * d[2])), 1);
    const pitch = (a1 - a0) / n;
    const rise = 2.5 + d[3];
    const face = e[0] < f(0.5) ? 1.0 : -1.0;
    const s0 = face > 0.0 ? a0 : a1;
    const slope = rise / pitch;
    const k = 1.0 / Math.hypot(1.0, slope);
    for (let t = 0; t < n; t++) {
      // s = face (a - s0); tooth t: z - h <= slope (s - t pitch) and
      // s <= (t + 1) pitch, on the roof (z >= h), across the building.
      // The sheet meets the roof at a wedge of atan(slope), 12 to 24
      // degrees, whose grown planes would meet up to r / sin(6 deg) out —
      // 3.4 m past the warehouse wall for the camera — so a bevel
      // s >= t pitch closes it at the tooth's own low edge (polytope.js).
      const top = [0, 0, k, 0];
      top[ax] = -slope * face * k;
      top[3] = k * (h - slope * t * pitch - slope * face * s0);
      const glass = [0, 0, 0, 0];
      glass[ax] = face;
      glass[3] = (t + 1) * pitch + face * s0;
      const tail = [0, 0, 0, 0];
      tail[ax] = -face;
      tail[3] = -(t * pitch + face * s0);
      const lo = [0, 0, 0, -cc.b1min[cx]], hi = [0, 0, 0, cc.b1max[cx]];
      lo[cx] = -1; hi[cx] = 1;
      out.push(polytope([top, glass, tail, [0, 0, -1, -h], lo, hi], h + rise,
                        "roof"));
    }
  } else if (d[0] < f(cc_industry_SAW_FRAC + cc_industry_MONITOR_FRAC)) {
    const w = (l.bc[1] - l.bc[0]) * (0.24 + 0.10 * e[1]);
    const cm = 0.5 * (l.bc[0] + l.bc[1]);
    const xy = (a, c) => (l.longX ? [a, c] : [c, a]);
    const lo = xy(l.ba[0] + 3.0, cm - 0.5 * w);
    const hi = xy(l.ba[1] - 3.0, cm + 0.5 * w);
    out.push(boxSolid([lo[0], lo[1], h],
                      [hi[0], hi[1], h + 2.6 + 0.9 * e[2]], "roof"));
  }
}

// --- the provider citylots.js CityLots asks ----------------------------------

/** What stands in the city besides its buildings, for CityLots. */
export class CityProps {
  /** `tile`: the city tile (citylots.js CityTile). */
  constructor(tile) {
    if (!(tile?.cellM > 0 && tile?.n > 0)) {
      throw new Error("CityProps needs the city tile.");
    }
    this.tile = tile;
    this.cellM = tile.cellM;
    this.n = tile.n;
  }

  /**
   * The solids block (cx, cy)'s column holds, and the highest top among
   * them: {solids, top}. Cached on the cell (CityLots caches the cells), and
   * radius-free — a sweep grows a polytope by its body's radius as it goes.
   */
  cellSolids(cc, cx, cy, r) {
    if (cc._props === undefined) {
      const solids = [];
      streetlifeSolids(solids, cc, cx, cy, this.cellM, this.n);
      skybridgeSolids(solids, cc, cx, cy, this.tile);
      roofSolids(solids, cc);
      cc._props = {
        solids,
        top: solids.reduce((m, sd) => Math.max(m, sd.top), -Infinity),
      };
    }
    return cc._props;
  }

  /** The solids no block owns (the freeways) near the segment p0 -> p1. */
  segmentSolids(p0, p1, r) {
    return skywaySolids(p0, p1, r, this.cellM);
  }
}
