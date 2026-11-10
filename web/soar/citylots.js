// The night city's buildings, on the CPU: a mirror of raymarch.wgsl's
// city_cell, and a swept-segment query against what it builds.
//
// The city is a pure function of the tile frame — one rgba16float texel per
// 90 m block (r density, g rank, b the packed lot word, a the street word;
// see cloudyview/city_lots.py) — and every building the renderer draws is
// raised out of that texel by city_cell. Flight collision needs the same
// buildings, not a lookalike: a wall the plane bounces off has to be the wall
// on screen, to the centimetre, or you either bounce off air or fly into a
// facade and watch it vanish (the renderer does not draw a building from the
// inside — which is exactly the bug this module exists to end). So the port
// is literal: the same hashes bit for bit, the same draws in the same order,
// the same variable names, and the storey quantization evaluated in float32
// the way the GPU evaluates it. tests/test_city_lots_mirror.py runs the WGSL
// city_cell on the GPU over a couple of thousand cells and holds this file to
// it.
//
// What is solid here: the building boxes b1..b3 (as many as `tiers` says),
// grown by the facade relief their walls carry (shellDepth), the frustum of
// archetypes 3 and 4, the mast, and the ground plane. What the components
// stand among them — the freeways, lamp posts, parked cars, skybridges and
// warehouse roof ridges — is cityprops.js's, handed to CityLots as `props`.
// (Until 2026-11 those were scenery the flyer passed through; Thomas: "it
// seems like all these should be solid. why not?")
//
// city_cell below is kept line for line with the WGSL so that re-syncing
// after a change there is a diff read, not a re-derivation. It deliberately
// stops at geometry: the window style, palette and occupancy draws that
// follow in the WGSL decide nothing a flyer can hit.

"use strict";

import { fromHalf } from "./half.js";
import { insidePolytope, segmentPolytope } from "./polytope.js";

// --- raymarch.wgsl constants (names identical; pinned by the mirror test) ---
export const CITY_GROUND_Z = 0.0;
export const CITY_PER_STOREY = 0.2777777777777778;
export const CITY_MAX_STOREYS = 1500.0;
export const CITY_MAX_H = 5400.0;
export const CITY_SLAB_TOP = 5600.0;
export const CITY_FLOOR_H = 3.6;
export const CITY_STREET_HALF = 6.0;
export const CITY_AVENUE_EXTRA = 10.0;
export const CITY_AVENUE_PERIOD = 8;
export const CITY_EMPTY_RANK = 0.22;
export const CITY_SPRAWL_RANK_FULL = 0.60;
export const CITY_SPRAWL_MIN_FRAC = 0.15;
export const CITY_H_BASE = 14.0;
export const CITY_H_SCALE = 390.0;
export const CITY_H_EXP = 1.2;
export const CITY_LOWRISE_H = 22.0;
export const CITY_TOWER_H = 60.0;
export const CITY_TOWER_SIDE_K = 4.2;
export const CITY_TOWER_KNEE_H = 600.0;
export const CITY_SPIRE_H = 600.0;
export const CITY_SUPERTALL_H = 1200.0;
export const CITY_SLENDER_MAX = 60.0;
export const CITY_USE_URBAN = 0;
export const CITY_USE_INDUSTRIAL = 1;
export const CITY_USE_YARD = 2;
export const CITY_USE_OPEN = 3;
export const CITY_WORD_OFFSET = 2048.0;
export const CITY_MAST_MIN_H = 150.0;
export const CITY_MAST_CROSS = 1.4;
export const CITY_CROWN_TIER = 3;
export const CITY_CROWN_MIN_H = 105.0;
export const CITY_CROWN_FRAC = 0.20;
export const CITY_CROWN_MIN_DECK = 15.0;
export const CITY_CROWN_INSET = 0.28;
export const CITY_CROWN_H_LO = 7.2;
export const CITY_CROWN_H_HI = 10.8;
export const CITY_CROWN_PAD_FRAC = 0.13;
export const CITY_CROWN_PAD_BIG = 60.0;
export const CITY_CROWN_PAD_MIN_H = 55.0;
export const CITY_CROWN_PAD_T = 0.30;

// --- float32 ---------------------------------------------------------------
//
// Most of city_cell is continuous, and there float64 lands within a few
// micrometres of the GPU's float32 — far inside anything a collision can
// feel. The exceptions are the floors: the height is a whole number of
// storeys ns, and every setback, podium and bud level is
// floor(ns * frac) * CITY_FLOOR_H, whose argument sits EXACTLY on an
// integer whenever frac times the storey count does (every fourth spire's
// 0.75, say). Which side of it float32 rounds to decides a whole storey, so
// those chains are evaluated through f() and storeyZ(), operation by
// operation, as the shader evaluates them — with the constants rounded to
// float32 first, because the shader's literals are.
//
// The shader carries ns and multiplies because it once divided: NVIDIA
// compiles x / 3.6 as x times the rounded reciprocal, which put those
// quarter-integer levels a storey high on one GPU and not on another (found
// by this file's mirror test, 2026-09-23; city_cell changed to products,
// which float32 rounds exactly everywhere). What still differs between GPUs
// is pow() in the height law, at the ulp level: a building whose raw height
// sits within an ulp of a storey boundary can come out a storey different.
const f = Math.fround;
const F32_FLOOR_H = f(CITY_FLOOR_H);

/** WGSL's floor(x) * CITY_FLOOR_H for a float32 storey product x. */
function storeyZ(x) {
  return f(Math.floor(f(x)) * F32_FLOOR_H);
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const mix = (a, b, t) => a * (1.0 - t) + b * t;   // WGSL's definition
function smoothstep(e0, e1, x) {
  const t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// --- hashes ----------------------------------------------------------------
//
// pcg2d on u32, bit for bit: Math.imul is the 32-bit multiply, and `>>> 0`
// brings every sum back to an unsigned 32-bit value (the sums stay below
// 2^53, so the double arithmetic before the wrap is exact).

/** raymarch.wgsl pcg2d. Takes and returns two u32s. */
export function pcg2d(x, y) {
  let vx = (Math.imul(x, 1664525) + 1013904223) >>> 0;
  let vy = (Math.imul(y, 1664525) + 1013904223) >>> 0;
  vx = (vx + Math.imul(vy, 1664525)) >>> 0;
  vy = (vy + Math.imul(vx, 1664525)) >>> 0;
  vx = (vx ^ (vx >>> 16)) >>> 0;
  vy = (vy ^ (vy >>> 16)) >>> 0;
  vx = (vx + Math.imul(vy, 1664525)) >>> 0;
  vy = (vy + Math.imul(vx, 1664525)) >>> 0;
  vx = (vx ^ (vx >>> 16)) >>> 0;
  vy = (vy ^ (vy >>> 16)) >>> 0;
  return [vx, vy];
}

/** raymarch.wgsl city_u01: f32(x) rounds to 24 bits BEFORE the scale. */
export function cityU01(x) {
  return f(x) * 2.3283064365386963e-10;
}

/** raymarch.wgsl city_rand4. */
export function cityRand4(sx, sy) {
  const a = pcg2d(sx, sy);
  const b = pcg2d(a[0], a[1]);
  return [cityU01(a[0]), cityU01(a[1]), cityU01(b[0]), cityU01(b[1])];
}

/** A seed XOR a vec2<u32> constant, kept unsigned. */
const xorSeed = (s, kx, ky) => [(s[0] ^ kx) >>> 0, (s[1] ^ ky) >>> 0];

// --- the tile --------------------------------------------------------------

/**
 * Mip 0 of the city tile: four float32 channels per block, decoded from the
 * rgba16float bytes the renderer uploads (fp16 -> f32 is exact, so these
 * are the values textureLoad returns).
 *
 * The b and a channels carry integer words offset by -2048 (city_lots.py).
 * A texel whose word is not such an integer is a tile baked by something
 * that does not know the packing — the renderer would decode garbage lots
 * out of it and so would this — so it refuses the tile at load rather than
 * building a city nobody baked.
 */
export class CityTile {
  constructor({ n, cellM, density, rank, lot, street }) {
    if (!(n > 0) || !(cellM > 0)) {
      throw new Error(`CityTile needs n and cellM; got n=${n}, cellM=${cellM}.`);
    }
    for (const [name, ch] of [["density", density], ["rank", rank],
                              ["lot", lot], ["street", street]]) {
      if (!ch || ch.length !== n * n) {
        throw new Error(
          `CityTile channel '${name}' has ${ch?.length} texels; the tile ` +
          `is ${n} x ${n}.`);
      }
    }
    for (const [name, ch] of [["lot", lot], ["street", street]]) {
      for (let i = 0; i < ch.length; i++) {
        const w = ch[i] + CITY_WORD_OFFSET;
        if (!(Number.isInteger(w) && w >= 0 && w < 4096)) {
          throw new Error(
            `City tile texel ${i % n},${Math.floor(i / n)} has ${name} ` +
            `word ${ch[i]}: not a packed 12-bit word. The tile predates ` +
            "the baked lots (cloudyview/city_lots.py); re-bake it.");
        }
      }
    }
    this.n = n;
    this.cellM = cellM;
    this.extentM = n * cellM;   // "defined as n * cell — no rounding"
    this.density = density;
    this.rank = rank;
    this.lot = lot;
    this.street = street;
  }

  /** From the raw rgba16float mip-0 bytes (little-endian halves, row-major). */
  static fromHalfBytes(bytes, n, cellM) {
    const halves = new Uint16Array(
      bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    if (halves.length !== 4 * n * n) {
      throw new Error(
        `City tile mip 0 is ${bytes.byteLength} bytes; ${n} x ${n} ` +
        `rgba16float is ${8 * n * n}.`);
    }
    const ch = [0, 1, 2, 3].map(() => new Float32Array(n * n));
    for (let i = 0; i < n * n; i++) {
      for (let c = 0; c < 4; c++) ch[c][i] = fromHalf(halves[4 * i + c]);
    }
    return new CityTile({ n, cellM, density: ch[0], rank: ch[1],
                          lot: ch[2], street: ch[3] });
  }
}

/** raymarch.wgsl city_tile_cell: an index folded into [0, n). */
export function tileCell(n, k) {
  return ((k % n) + n) % n;
}

/** raymarch.wgsl city_is_avenue. */
export function isAvenue(k) {
  return ((k % CITY_AVENUE_PERIOD) + CITY_AVENUE_PERIOD)
         % CITY_AVENUE_PERIOD === 0;
}

/** raymarch.wgsl city_word. */
function cityWord(v) {
  return (v + CITY_WORD_OFFSET) >>> 0;
}

const vec3 = (xy, z) => [xy[0], xy[1], z];
const ZERO3 = () => [0.0, 0.0, 0.0];

/**
 * raymarch.wgsl city_cell, geometry only. `(cx, cy)` is the UNWRAPPED cell
 * index — the geometry comes back in the unwrapped city frame (cell i spans
 * [i * cell, (i + 1) * cell)), exactly as the shader builds it, while
 * texels and seeds read the tile-wrapped index.
 *
 * Field names are the WGSL struct's, snake_case and all, so the two read
 * side by side.
 */
export function cityCell(tile, cx, cy) {
  const c = { crown: noCrown() };
  const cell = tile.cellM;
  const n = tile.n;

  // The lot (see the WGSL for why it is baked and packed).
  const wx = tileCell(n, cx), wy = tileCell(n, cy);
  const ti = wy * n + wx;
  const lot = cityWord(tile.lot[ti]);
  const st = cityWord(tile.street[ti]);
  const back = [lot & 3, (lot >>> 2) & 3];
  const span = [((lot >>> 4) & 3) + 1, ((lot >>> 6) & 3) + 1];
  const own = [(lot >>> 8) & 3, (lot >>> 10) & 3];
  const base = [cx - back[0], cy - back[1]];
  c.merged = span[0] > 1 || span[1] > 1;
  c.streets = st & 0xff;
  c.land_use = (st >>> 8) & 3;
  const wo = [tileCell(n, base[0] + own[0]), tileCell(n, base[1] + own[1])];
  const to = wo[1] * n + wo[0];
  c.density = tile.density[to];
  c.rank = tile.rank[to];
  c.seed = pcg2d(wo[0], wo[1]);

  // The plot: the lot less half a street on every outer edge.
  const cmin = [base[0] * cell, base[1] * cell];
  const cmax = [(base[0] + span[0]) * cell, (base[1] + span[1]) * cell];
  c.lot_min = cmin;
  c.lot_max = cmax;
  const hx0 = CITY_STREET_HALF + (isAvenue(base[0]) ? CITY_AVENUE_EXTRA : 0.0);
  const hx1 = CITY_STREET_HALF
    + (isAvenue(base[0] + span[0]) ? CITY_AVENUE_EXTRA : 0.0);
  const hy0 = CITY_STREET_HALF + (isAvenue(base[1]) ? CITY_AVENUE_EXTRA : 0.0);
  const hy1 = CITY_STREET_HALF
    + (isAvenue(base[1] + span[1]) ? CITY_AVENUE_EXTRA : 0.0);
  c.plot_min = [cmin[0] + hx0, cmin[1] + hy0];
  c.plot_max = [cmax[0] - hx1, cmax[1] - hy1];
  const r4 = cityRand4(c.seed[0], c.seed[1]);
  const r4b = cityRand4(...xorSeed(c.seed, 0x9e3779b9, 0x85ebca6b));
  const r4c = cityRand4(...xorSeed(c.seed, 0xdeadbeef, 0x41c64e6d));

  c.built = c.land_use === CITY_USE_URBAN
         || c.land_use === CITY_USE_INDUSTRIAL;
  c.tiers = 1;
  c.has_mast = false;
  c.height = 0.0;
  c.top_z = 0.0;
  c.b1min = ZERO3(); c.b1max = ZERO3();
  c.b2min = ZERO3(); c.b2max = ZERO3();
  c.b3min = ZERO3(); c.b3max = ZERO3();
  c.mast_min = ZERO3(); c.mast_max = ZERO3();
  c.arch = 0;
  c.has_frustum = false;
  c.fmin = ZERO3(); c.fmax = ZERO3();
  c.fscale = 1.0;
  c.win_style = 0;              // the WGSL's zero-initialised CityCell
  if (!c.built) {
    return c;
  }

  const plot_size = [c.plot_max[0] - c.plot_min[0],
                     c.plot_max[1] - c.plot_min[1]];
  const pc = [0.5 * (c.plot_min[0] + c.plot_max[0]),
              0.5 * (c.plot_min[1] + c.plot_max[1])];
  const r4d = cityRand4(...xorSeed(c.seed, 0x27d4eb2f, 0x165667b1));

  if (c.land_use === CITY_USE_INDUSTRIAL) {
    // A warehouse: one low box across most of the lot.
    const h = f(Math.floor(f(f(9.0 + f(6.0 * r4[0])) * 2.0)) * 0.5);
    c.height = h;
    c.top_z = h;
    const long_x = plot_size[0] > plot_size[1]
      || (plot_size[0] === plot_size[1] && r4[1] < 0.5);
    const along = long_x ? plot_size[0] : plot_size[1];
    const across = long_x ? plot_size[1] : plot_size[0];
    const fa = along * (0.78 + 0.17 * r4[2]);
    const fc = across * (0.58 + 0.22 * r4[3]);
    const slack_a = (along - fa) * r4b[0];
    const slack_c = r4b[1] < 0.5 ? 2.0 : across - fc - 2.0;
    let lo, hi;
    if (long_x) {
      lo = [c.plot_min[0] + slack_a, c.plot_min[1] + slack_c];
      hi = [lo[0] + fa, lo[1] + fc];
    } else {
      lo = [c.plot_min[0] + slack_c, c.plot_min[1] + slack_a];
      hi = [lo[0] + fc, lo[1] + fa];
    }
    c.b1min = vec3(lo, CITY_GROUND_Z);
    c.b1max = vec3(hi, h);
    c.win_style = 4;
    return c;
  }

  // Height: base + the cascade to a power, jittered, sprawl-suppressed, and
  // made a whole number of storeys ns. float32 from end to end.
  const sprawl = f(mix(
    CITY_SPRAWL_MIN_FRAC, 1.0,
    smoothstep(CITY_EMPTY_RANK, CITY_SPRAWL_RANK_FULL, c.rank)));
  const h_raw = f(f(f(CITY_H_BASE
                      + f(CITY_H_SCALE * f(Math.pow(c.density, CITY_H_EXP))))
                    * f(f(0.70) + f(f(0.60) * r4[0]))) * sprawl);
  const ns = clamp(Math.floor(f(h_raw * f(CITY_PER_STOREY))),
                   2.0, CITY_MAX_STOREYS);
  const h = f(ns * F32_FLOOR_H);
  c.height = h;

  // Footprint grows with height (the WGSL tells the story).
  const r4f = cityRand4(...xorSeed(c.seed, 0x6a09e667, 0xbb67ae85));
  let fw = [plot_size[0] * (0.80 + 0.17 * r4[1]),
            plot_size[1] * (0.80 + 0.17 * r4[2])];
  let podium = false;
  let zp = 0.0;
  if (h >= CITY_TOWER_H) {
    const side = Math.max(
      CITY_TOWER_SIDE_K * Math.sqrt(Math.min(h, CITY_TOWER_KNEE_H))
        * (0.80 + 0.40 * r4[1]),
      h / CITY_SLENDER_MAX);
    const aspect = 0.72 + 0.28 * r4[2];
    const sides = r4f[1] < 0.5 ? [side, side * aspect]
                               : [side * aspect, side];
    fw = [Math.min(sides[0], plot_size[0] * 0.94),
          Math.min(sides[1], plot_size[1] * 0.94)];
    if (Math.min(fw[0] / plot_size[0], fw[1] / plot_size[1]) < 0.75) {
      podium = true;
      const storeys = Math.floor(f(mix(3.0, c.merged ? 12.0 : 8.0, r4f[2])));
      zp = Math.min(f(storeys * F32_FLOOR_H),
                    storeyZ(ns * f(0.33333334)));
    }
  }
  const slack = [plot_size[0] - fw[0], plot_size[1] - fw[1]];
  const bmin2 = [c.plot_min[0] + slack[0] * r4[3],
                 c.plot_min[1] + slack[1] * r4b[0]];
  const bmax2 = [bmin2[0] + fw[0], bmin2[1] + fw[1]];
  const bc = [0.5 * (bmin2[0] + bmax2[0]), 0.5 * (bmin2[1] + bmax2[1])];
  const pod_fw = [plot_size[0] * (0.92 + 0.06 * r4f[3]),
                  plot_size[1] * (0.92 + 0.06 * r4f[3])];
  const pod_min = [pc[0] - 0.5 * pod_fw[0], pc[1] - 0.5 * pod_fw[1]];
  const pod_max = [pc[0] + 0.5 * pod_fw[0], pc[1] + 0.5 * pod_fw[1]];

  // The architecture.
  if (h > 60.0) {
    const spire_p = h > 300.0 ? 0.30 : 0.13;
    if (r4d[0] < f(0.20)) {
      c.arch = 2;                   // growth: cantilevered buds
    } else if (r4d[0] < f(0.42)) {
      c.arch = 3;                   // tapered shaft on a podium
    } else if (r4d[0] < f(f(0.42) + f(spire_p))) {
      c.arch = 4;                   // spire crown
    }
  }
  if (h > CITY_SPIRE_H) {
    c.arch = r4d[1] < 0.60 ? 4 : 3;
  }

  if (h < CITY_LOWRISE_H) {
    // Low-rise: a row of frontages, not one box.
    c.arch = 0;
    const long_x = fw[0] >= fw[1];
    const n_up = 1.0 + Math.floor(f(3.0 * r4d[2]));
    const z1 = Math.max(f(h - f(n_up * F32_FLOOR_H)), F32_FLOOR_H);
    c.b1min = vec3(bmin2, CITY_GROUND_Z);
    c.b1max = vec3(bmax2, z1);
    if (z1 < h) {
      c.tiers = 2;
      const run = 0.33 + 0.34 * r4d[3];
      const at_hi = r4b[3] < 0.5;
      const lo = [...bmin2];
      const hi = [...bmax2];
      if (long_x) {
        const len = fw[0] * run;
        lo[0] = at_hi ? bmax2[0] - len : bmin2[0];
        hi[0] = lo[0] + len;
      } else {
        const len = fw[1] * run;
        lo[1] = at_hi ? bmax2[1] - len : bmin2[1];
        hi[1] = lo[1] + len;
      }
      c.b2min = vec3(lo, z1);
      c.b2max = vec3(hi, h);
      if (f(h - z1) >= f(2.0 * F32_FLOOR_H) && r4f[0] < 0.40) {
        // A last storey or two over the far half of the run.
        c.tiers = 3;
        const z2 = f(h - F32_FLOOR_H);
        c.b2max[2] = z2;
        const lo3 = [...lo];
        const hi3 = [...hi];
        if (long_x) {
          const len3 = 0.5 * (hi[0] - lo[0]);
          lo3[0] = at_hi ? hi[0] - len3 : lo[0];
          hi3[0] = lo3[0] + len3;
        } else {
          const len3 = 0.5 * (hi[1] - lo[1]);
          lo3[1] = at_hi ? hi[1] - len3 : lo[1];
          hi3[1] = lo3[1] + len3;
        }
        c.b3min = vec3(lo3, z2);
        c.b3max = vec3(hi3, h);
      }
    }
  } else if (c.arch === 3) {
    // Podium + a shaft whose cross-section closes toward the crown.
    zp = podium ? zp
                : Math.max(storeyZ(ns * f(0.12)), F32_FLOOR_H);
    c.b1min = vec3(podium ? pod_min : bmin2, CITY_GROUND_Z);
    c.b1max = vec3(podium ? pod_max : bmax2, zp);
    const s = podium ? 1.0 : 0.90;
    const fw_shaft = [fw[0] * s, fw[1] * s];
    c.has_frustum = true;
    c.fmin = [bc[0] - 0.5 * fw_shaft[0], bc[1] - 0.5 * fw_shaft[1], zp];
    c.fmax = [bc[0] + 0.5 * fw_shaft[0], bc[1] + 0.5 * fw_shaft[1], h];
    c.fscale = h > CITY_SUPERTALL_H ? 0.40 + 0.20 * r4d[2]
                                    : 0.55 + 0.25 * r4d[2];
  } else if (c.arch === 4) {
    // A straight shaft to seven tenths, then a steep spire.
    const zs = storeyZ(ns * (h > CITY_SUPERTALL_H
      ? f(f(0.62) + f(f(0.10) * r4d[3])) : f(0.70)));
    if (podium) {
      c.tiers = 2;
      c.b1min = vec3(pod_min, CITY_GROUND_Z);
      c.b1max = vec3(pod_max, zp);
      c.b2min = vec3(bmin2, zp);
      c.b2max = vec3(bmax2, zs);
    } else {
      c.b1min = vec3(bmin2, CITY_GROUND_Z);
      c.b1max = vec3(bmax2, zs);
    }
    c.has_frustum = true;
    c.fmin = vec3(bmin2, zs);
    c.fmax = vec3(bmax2, h);
    c.fscale = h > CITY_SUPERTALL_H ? 0.06 + 0.08 * r4d[2]
                                    : 0.12 + 0.10 * r4d[2];
  } else if (c.arch === 2) {
    // Growth: a straight tower with one or two cantilevered buds.
    c.b1min = vec3(bmin2, CITY_GROUND_Z);
    c.b1max = vec3(bmax2, h);
    c.tiers = 2;
    const side = Math.floor(f(r4d[2] * 4.0));   // which flank the bud rides
    const bz0 = storeyZ(ns * f(f(0.42) + f(f(0.20) * r4d[3])));
    const bz1 = Math.min(
      f(f(bz0 + storeyZ(f(f(0.10) + f(f(0.10) * r4b[1])) * ns))
        + F32_FLOOR_H),
      f(h - F32_FLOOR_H));
    const reach = 0.22 * Math.min(fw[0], fw[1]);
    let bud_min, bud_max;
    if (side === 0) {
      bud_min = [bmax2[0] - 1.0, bmin2[1] + 0.2 * fw[1]];
      bud_max = [bmax2[0] + reach, bmax2[1] - 0.2 * fw[1]];
    } else if (side === 1) {
      bud_min = [bmin2[0] - reach, bmin2[1] + 0.2 * fw[1]];
      bud_max = [bmin2[0] + 1.0, bmax2[1] - 0.2 * fw[1]];
    } else if (side === 2) {
      bud_min = [bmin2[0] + 0.2 * fw[0], bmax2[1] - 1.0];
      bud_max = [bmax2[0] - 0.2 * fw[0], bmax2[1] + reach];
    } else {
      bud_min = [bmin2[0] + 0.2 * fw[0], bmin2[1] - reach];
      bud_max = [bmax2[0] - 0.2 * fw[0], bmin2[1] + 1.0];
    }
    // Clamped to the lot minus a hair, so no prim leaves it — and to the
    // plot on avenue edges, whose corridor is skyway's empty air.
    const bud_lo = [cmin[0] + (isAvenue(base[0]) ? hx0 : 0.5),
                    cmin[1] + (isAvenue(base[1]) ? hy0 : 0.5)];
    const bud_hi = [cmax[0] - (isAvenue(base[0] + span[0]) ? hx1 : 0.5),
                    cmax[1] - (isAvenue(base[1] + span[1]) ? hy1 : 0.5)];
    bud_min = [Math.max(bud_min[0], bud_lo[0]),
               Math.max(bud_min[1], bud_lo[1])];
    bud_max = [Math.min(bud_max[0], bud_hi[0]),
               Math.min(bud_max[1], bud_hi[1])];
    c.b2min = vec3(bud_min, bz0);
    c.b2max = vec3(bud_max, bz1);
    if (h > 200.0 && r4b[1] > 0.55) {
      c.tiers = 3;
      const cz0 = storeyZ(ns * f(f(0.68) + f(f(0.12) * r4b[3])));
      const cz1 = Math.min(f(cz0 + f(8.0 * F32_FLOOR_H)),
                           f(h - F32_FLOOR_H));
      // The second bud rides the opposite flank: mirror about bc.
      const m0 = [2.0 * bc[0] - bud_max[0], 2.0 * bc[1] - bud_max[1]];
      const m1 = [2.0 * bc[0] - bud_min[0], 2.0 * bc[1] - bud_min[1]];
      c.b3min = [Math.max(Math.min(m0[0], m1[0]), bud_lo[0]),
                 Math.max(Math.min(m0[1], m1[1]), bud_lo[1]), cz0];
      c.b3max = [Math.min(Math.max(m0[0], m1[0]), bud_hi[0]),
                 Math.min(Math.max(m0[1], m1[1]), bud_hi[1]), cz1];
    }
  } else if (podium) {
    // Podium, tower, crown.
    c.tiers = 3;
    const zc = storeyZ(ns * f(f(0.62) + f(f(0.12) * r4b[3])));
    c.b1min = vec3(pod_min, CITY_GROUND_Z);
    c.b1max = vec3(pod_max, zp);
    c.b2min = vec3(bmin2, zp);
    c.b2max = vec3(bmax2, zc);
    const s3 = 0.55 + 0.15 * r4b[2];
    const fw3 = [fw[0] * s3, fw[1] * s3];
    c.b3min = [bc[0] - 0.5 * fw3[0], bc[1] - 0.5 * fw3[1], zc];
    c.b3max = [bc[0] + 0.5 * fw3[0], bc[1] + 0.5 * fw3[1], h];
  } else {
    // Setback tiers: the wedding cake (the founding archetype).
    let z1 = h;
    if (h > 90.0 && r4b[1] > 0.45) {
      c.tiers = 2;
      z1 = storeyZ(ns * f(0.60));
      if (h > 180.0 && r4b[1] > 0.80) {
        c.tiers = 3;
        z1 = storeyZ(ns * f(0.50));
      }
    }
    c.b1min = vec3(bmin2, CITY_GROUND_Z);
    c.b1max = vec3(bmax2, z1);
    if (c.tiers >= 2) {
      let z2 = h;
      let f2 = 0.68;
      if (c.tiers === 3) {
        z2 = storeyZ(ns * f(0.78));
        f2 = 0.72;
      }
      const fw2 = [fw[0] * f2, fw[1] * f2];
      c.b2min = [bc[0] - 0.5 * fw2[0], bc[1] - 0.5 * fw2[1], z1];
      c.b2max = [bc[0] + 0.5 * fw2[0], bc[1] + 0.5 * fw2[1], z2];
      if (c.tiers === 3) {
        const fw3 = [fw[0] * 0.45, fw[1] * 0.45];
        c.b3min = [bc[0] - 0.5 * fw3[0], bc[1] - 0.5 * fw3[1], z2];
        c.b3max = [bc[0] + 0.5 * fw3[0], bc[1] + 0.5 * fw3[1], h];
      }
    }
  }

  // The window style: geometry after all, since the facade shell's relief
  // (sills, units, balconies, fire escapes) stands off the wall by a depth
  // that follows it (shellDepth). The house palette and occupancy draws that
  // follow it in the WGSL decide nothing a flyer can hit.
  const r4e = cityRand4(...xorSeed(c.seed, 0xb5297a4d, 0x68e31da4));
  let style = 0;
  if (h > 350.0 || c.arch === 3) {
    if (r4e[0] < f(0.45)) style = 3;
    else if (r4e[0] < f(0.70)) style = 0;
    else if (r4e[0] < f(0.90)) style = 1;
    else style = 2;
  } else if (h < 40.0) {
    if (r4e[0] < f(0.50)) style = 4;
    else if (r4e[0] < f(0.80)) style = 0;
    else style = 1;
  } else {
    if (r4e[0] < f(0.30)) style = 0;
    else if (r4e[0] < f(0.55)) style = 1;
    else if (r4e[0] < f(0.75)) style = 2;
    else if (r4e[0] < f(0.90)) style = 3;
    else style = 4;
  }
  c.win_style = style;

  // Roof furniture belongs on roofs; a supertall's tapered shaft always
  // ends in a needle, scaled to the tower and centred on its crown.
  const needle = c.arch === 3 && h > CITY_SUPERTALL_H;
  if (h > CITY_MAST_MIN_H && (r4b[2] < 0.65 || needle) && c.arch !== 4) {
    c.has_mast = true;
    let mast_h = f(12.0 + f(45.0 * r4b[3]));
    let cross = CITY_MAST_CROSS;
    let moff = [0.0, 0.0];
    if (needle) {
      mast_h = Math.min(f(h * f(f(0.06) + f(f(0.05) * r4b[3]))),
                        f(f(CITY_SLAB_TOP - h) - 1.0));
      cross = Math.max(CITY_MAST_CROSS, 0.03 * mast_h);
    } else {
      const crown = c.arch === 3 ? c.fscale * 0.85 : 1.0;
      moff = [(r4c[3] - 0.5) * fw[0] * (0.4 * crown),
              (r4[0] - 0.5) * fw[1] * (0.4 * crown)];
    }
    const mc = [bc[0] + moff[0], bc[1] + moff[1]];
    c.mast_min = [mc[0] - 0.5 * cross, mc[1] - 0.5 * cross, h];
    c.mast_max = [mc[0] + 0.5 * cross, mc[1] + 0.5 * cross, f(h + mast_h)];
    c.top_z = f(h + mast_h);
  } else {
    c.top_z = h;
  }
  // The crown's headroom, as a bound (raymarch.wgsl city_cell).
  if (h > CITY_CROWN_MIN_H && c.arch !== 4) {
    c.top_z = Math.max(c.top_z, f(h + CITY_CROWN_H_HI));
  }
  c.crown = cityCrown(c, cx, cy, cell, n);
  return c;
}

// float32 images of the crown's constants, for the decisions made in f32.
const F32_CROWN_INSET = f(CITY_CROWN_INSET);
const F32_CROWN_H_LO = f(CITY_CROWN_H_LO);
const F32_CROWN_H_HI = f(CITY_CROWN_H_HI);
const F32_CROWN_PAD_T = f(CITY_CROWN_PAD_T);
const F32_0_30 = f(0.30), F32_0_40 = f(0.40);
const f32Mix = (a, b, t) => f(f(a * f(1.0 - t)) + f(b * t));

/**
 * raymarch.wgsl city_crown: the penthouse (kind 1) on a tall tower's summit
 * deck, or the helipad (kind 2) that pre-empts it, per CELL — the crown is
 * clipped to the cell column (cx, cy) and only a checkerboard of a merged
 * lot's cells carry one. Every comparison is the shader's, in float32.
 */
function noCrown() {
  return { kind: 0, lo: [f(1e30), f(1e30)], hi: [f(-1e30), f(-1e30)],
           z0: 0.0, z1: 0.0 };
}

export function cityCrown(c, cx, cy, cell, n) {
  const none = noCrown();
  if (!c.built || c.arch === 4 || c.land_use === CITY_USE_INDUSTRIAL
      || c.height <= CITY_CROWN_PAD_MIN_H) {
    return none;
  }
  let dlo = [c.b1min[0], c.b1min[1]];
  let dhi = [c.b1max[0], c.b1max[1]];
  let dz = c.b1max[2];
  if (c.arch === 3) {
    const ctr = [0, 1].map((k) => f(0.5 * f(c.fmin[k] + c.fmax[k])));
    const hf = [0, 1].map((k) => f(f(0.5 * f(c.fmax[k] - c.fmin[k]))
                                   * c.fscale));
    dlo = [f(ctr[0] - hf[0]), f(ctr[1] - hf[1])];
    dhi = [f(ctr[0] + hf[0]), f(ctr[1] + hf[1])];
    dz = c.fmax[2];
  } else if (c.arch !== 2 && c.tiers === 2) {
    dlo = [c.b2min[0], c.b2min[1]]; dhi = [c.b2max[0], c.b2max[1]];
    dz = c.b2max[2];
  } else if (c.arch !== 2 && c.tiers >= 3) {
    dlo = [c.b3min[0], c.b3min[1]]; dhi = [c.b3max[0], c.b3max[1]];
    dz = c.b3max[2];
  }
  const cmin = [f(cx * cell), f(cy * cell)];
  const rlo = [Math.max(dlo[0], cmin[0]), Math.max(dlo[1], cmin[1])];
  const rhi = [Math.min(dhi[0], f(cmin[0] + cell)),
               Math.min(dhi[1], f(cmin[1] + cell))];
  const size = [f(rhi[0] - rlo[0]), f(rhi[1] - rlo[1])];
  const smin = Math.min(size[0], size[1]);
  if (smin < CITY_CROWN_MIN_DECK) return none;
  const foot0 = Math.max(f(dhi[0] - dlo[0]), f(dhi[1] - dlo[1]));
  const padSize = c.merged || foot0 > CITY_CROWN_PAD_BIG;
  if (c.height <= CITY_CROWN_MIN_H && !padSize) return none;
  const gh = cityRand4(...xorSeed(c.seed, 0x1b56c4e9, 0x0d2f1a37));
  const cw = [tileCell(n, cx), tileCell(n, cy)];
  const lh = cityRand4((Math.imul(cw[0], 0x9e3779b9) + 0x2545f491) >>> 0,
                       (Math.imul(cw[1], 0x85ebca6b) + 0x27d4eb2f) >>> 0);
  // Helipad first. (Which cell holds the pad is a float32 division on the
  // GPU, which NVIDIA computes as a reciprocal multiply; a pick within an
  // ulp of a cell edge could land in the neighbour. The pick is a point in
  // the middle half of the deck, so none is that close in the tile.)
  const padAt = [0, 1].map((k) => f32Mix(dlo[k], dhi[k],
                                         f(0.25 + f(0.5 * gh[2 + k]))));
  const padHere = Math.floor(f(padAt[0] / cell)) === cx
               && Math.floor(f(padAt[1] / cell)) === cy;
  if (padSize && padHere && gh[0] < f(CITY_CROWN_PAD_FRAC)) {
    const half = clamp(f(F32_0_30 * smin), 4.5, 11.0);
    const ctr = [0, 1].map((k) => f32Mix(
      f(f(rlo[k] + half) + 1.0), f(f(rhi[k] - half) - 1.0),
      f(F32_0_30 + f(F32_0_40 * lh[k]))));
    const plo = [f(ctr[0] - half), f(ctr[1] - half)];
    const phi = [f(ctr[0] + half), f(ctr[1] + half)];
    const onMast = c.has_mast
      && plo[0] < f(c.mast_max[0] + 1.0) && phi[0] > f(c.mast_min[0] - 1.0)
      && plo[1] < f(c.mast_max[1] + 1.0) && phi[1] > f(c.mast_min[1] - 1.0);
    if (!onMast) {
      return { kind: 2, lo: plo, hi: phi, z0: dz, z1: f(dz + F32_CROWN_PAD_T) };
    }
  }
  // Penthouse, on a checkerboard of a merged lot's cells.
  const pentHere = !c.merged || ((cx + cy) & 1) === 0;
  if (c.height > CITY_CROWN_MIN_H && pentHere && gh[1] < f(CITY_CROWN_FRAC)) {
    const inset = Math.max(f(F32_CROWN_INSET * smin), 3.0);
    const plo = [f(rlo[0] + inset), f(rlo[1] + inset)];
    const phi = [f(rhi[0] - inset), f(rhi[1] - inset)];
    if (Math.min(f(phi[0] - plo[0]), f(phi[1] - plo[1])) > 6.0) {
      return { kind: 1, lo: plo, hi: phi, z0: dz,
               z1: f(dz + (lh[2] > f(0.45) ? F32_CROWN_H_HI : F32_CROWN_H_LO)) };
    }
  }
  return none;
}

// --- intersection ------------------------------------------------------------
//
// Segments, not rays: p(t) = p0 + t (p1 - p0), t in [0, 1], swept by a
// SPHERE of radius r — the body the collider is moving. Sweeping a sphere
// against a solid is sweeping its centre against the solid grown by r, so
// every surface below is pushed out by r: the boxes grow on every side, the
// frustum's sloped faces move out along their own normals, the ground rises
// to z = r. (A grown box's edges are square where the true grown shape is
// rounded — a few centimetres of extra caution at a corner.)
//
// A building's boxes grow sideways by more than that: the facade shell
// (city-facades) hangs relief off the walls — sills, ledges, air-
// conditioning units, fire escapes, balconies — and a camera inside that
// relief sees it vanish exactly as it saw whole buildings vanish. How far
// it stands out is the building's own (shellDepth: a few centimetres of
// mullion cap on a curtain wall, 0.6 m where units hang under the sills,
// 1.1-1.25 m only where there are balconies or a fire escape), so each
// building grows by its own relief — its deepest feature, on every face
// and between the features too, which is the building's envelope rather
// than the wall behind it — and the flight meets the building where it
// stands out. (It grew every building by the deepest any style
// allows, CITY_SHELL_OUT_MAX, until 2026-11: a curtain wall stopped you
// 1.5 m short of its glass, and Thomas noticed — "it happens a bit before
// the airplane actually contacts the building".) Roofs stay exact (nothing
// stands on them from the shell) and nothing hangs below a box. The
// frustum's sloped skin and the masts carry no shell.
//
// A hit reports the ENTRY — a segment that starts inside does not hit. That
// is the right answer for a sweep (a body already inside cannot be stopped
// at a face it is behind), and the collider never starts one inside: it
// backs every body out a hair beyond the grown surface and asks inside()
// separately.
//
// Grown, a solid can reach out of the columns of the cells that report it
// (a bud is clamped only half a metre inside its lot), so every cell the
// walk visits tests its eight neighbours as well. That is exact as long as
// the growth is under a cell, which checkReach() holds.

// raymarch.wgsl's facade shell: the deepest relief any style may carry, and
// the two deepest features.
export const CITY_SHELL_OUT_MAX = 1.5;
export const CITY_ESCAPE_OUT = 1.25;
export const CITY_BALCONY_OUT = 1.10;
// A core strip's proud front (raymarch.wgsl's facade zones).
export const CITY_CORE_OUT = 0.45;

/** raymarch.wgsl city_escape_building: the building carries a fire escape. */
function escapeBuilding(c) {
  const h = cityRand4(...xorSeed(c.seed, 0x1b873593, 0xcc9e2d51));
  return (c.win_style === 4 || c.win_style === 0)
    && c.land_use === CITY_USE_URBAN
    && c.height >= 18.0 && c.height <= 66.0 && h[0] < f(0.25);
}

/** raymarch.wgsl city_core_building: the building carries core strips. */
function coreBuilding(c) {
  const h = cityRand4(...xorSeed(c.seed, 0x9b05688c, 0x1f83d9ab));
  return c.land_use === CITY_USE_URBAN && c.height > 40.0
    && c.win_style !== 4 && h[0] < f(0.5);
}

/** raymarch.wgsl city_balcony_building: balconies (never with an escape). */
function balconyBuilding(c) {
  const h = cityRand4(...xorSeed(c.seed, 0x27220a95, 0x165667b1));
  return (c.win_style === 0 || c.win_style === 4 || c.win_style === 1)
    && c.land_use === CITY_USE_URBAN
    && c.height >= 12.0 && c.height <= 120.0 && h[0] < f(0.33)
    && !escapeBuilding(c);
}

/**
 * How far the facade shell's relief stands off this building's walls: the
 * depth raymarch.wgsl city_shell_style gives the building's shell layer,
 * draw for draw (only the draws that move the depth are mirrored). Zero on
 * an industrial lot, whose walls the shell leaves to the industry
 * component, which hangs nothing off them.
 */
export function shellDepth(c) {
  if (c.land_use !== CITY_USE_URBAN) return 0.0;
  const r = cityRand4(...xorSeed(c.seed, 0x6a09e667, 0xbb67ae85));
  const q = cityRand4(...xorSeed(c.seed, 0x510e527f, 0x9b05688c));
  const tall = c.height > 150.0;
  const pipesOf = (x) => Math.floor(f(x * f(2.99)));
  let sill = 0.0, ledge = 0.0, cornice = 0.0, cap = 0.0, hood = 0.0;
  let canopy = 0.0, ac = 0.0, pipes = 0;
  if (c.win_style === 4) {
    sill = 0.06;
    hood = r[2] > f(0.3) ? 0.07 + 0.05 * q[1] : 0.0;
    if (r[3] < f(0.40)) ledge = 0.05;
    cornice = c.height < 80.0 ? 0.30 + 0.20 * q[1] : 0.0;
    ac = tall ? 0.0 : 0.10 + 0.20 * q[2];
    pipes = pipesOf(q[3]);
  } else if (c.win_style === 0) {
    sill = r[1] < f(0.5) ? 0.04 : 0.0;
    if (r[3] < f(0.60)) ledge = 0.10 + 0.08 * q[0];
    ac = tall ? 0.0 : 0.08 + 0.25 * q[2];
    pipes = tall ? 0 : pipesOf(q[3]);
    canopy = !tall && c.height < 90.0 && r[2] > f(0.6)
      ? 0.45 + 0.25 * r[2] : 0.0;
  } else if (c.win_style === 1) {
    cap = 0.05;
    if (r[3] < f(0.45)) ledge = 0.35 + 0.20 * q[0];
    ac = tall ? 0.0 : 0.06;
    pipes = !tall && q[3] < f(0.4) ? 1 : 0;
  } else if (c.win_style === 2) {
    cap = 0.25 + 0.20 * r[3];
  } else {
    cap = 0.08 + 0.06 * r[3];
    ledge = 0.04;
  }
  let depth = Math.max(sill, ledge, cornice, cap, hood, canopy);
  if (ac > 0.0) depth = Math.max(depth, 0.58);
  if (pipes > 0) depth = Math.max(depth, 0.24);
  if (escapeBuilding(c)) depth = Math.max(depth, CITY_ESCAPE_OUT);
  if (balconyBuilding(c)) depth = Math.max(depth, CITY_BALCONY_OUT);
  // A core strip may stand on any face of a tower that has them.
  if (coreBuilding(c)) depth = Math.max(depth, CITY_CORE_OUT);
  return Math.min(depth + 0.02, CITY_SHELL_OUT_MAX);
}

const EPS_DIR = 1e-12;

/**
 * Exported for tests: a cell's solids grown for a body of radius r, each {min, max, grow} for a
 * box (grow per axis) or {frustum, grow} for the tapered shaft. Cached on
 * the cell per radius — there are two radii in use.
 */
export function cellSolids(cc, r) {
  cc._solids ??= new Map();
  let list = cc._solids.get(r);
  if (list !== undefined) return list;
  list = [];
  const shell = shellDepth(cc);
  const building = [shell + r, shell + r, r];
  list.push({ min: cc.b1min, max: cc.b1max, grow: building });
  if (cc.tiers >= 2) list.push({ min: cc.b2min, max: cc.b2max, grow: building });
  if (cc.tiers >= 3) list.push({ min: cc.b3min, max: cc.b3max, grow: building });
  // The penthouse crown is massing, with the tower's own relief on its
  // walls; the helipad (kind 2) is a 0.3 m prop slab and stays unsolid.
  if (cc.crown?.kind === 1) {
    const k = cc.crown;
    list.push({ min: [k.lo[0], k.lo[1], k.z0], max: [k.hi[0], k.hi[1], k.z1],
                grow: building });
  }
  if (cc.has_mast) {
    list.push({ min: cc.mast_min, max: cc.mast_max, grow: [r, r, r] });
  }
  if (cc.has_frustum) {
    list.push({ frustum: frustumShape(cc.fmin, cc.fmax, cc.fscale, r) });
  }
  cc._solids.set(r, list);
  return list;
}

/** Segment vs a grown box: {t, normal} of the entry, or null. */
function segmentBox(p0, d, { min, max, grow }) {
  let tLo = -Infinity, tHi = Infinity, axis = -1;
  for (let k = 0; k < 3; k++) {
    const lo = min[k] - grow[k], hi = max[k] + grow[k];
    if (Math.abs(d[k]) < EPS_DIR) {
      if (p0[k] < lo || p0[k] > hi) return null;
      continue;
    }
    let ta = (lo - p0[k]) / d[k];
    let tb = (hi - p0[k]) / d[k];
    if (ta > tb) [ta, tb] = [tb, ta];
    if (ta > tLo) { tLo = ta; axis = k; }
    if (tb < tHi) tHi = tb;
  }
  if (axis < 0 || tLo > tHi || tLo < 0.0 || tLo > 1.0) return null;
  const normal = [0.0, 0.0, 0.0];
  normal[axis] = d[axis] > 0.0 ? -1.0 : 1.0;
  return { t: tLo, normal };
}

/**
 * The rectangular frustum of archetypes 3 and 4, as raymarch.wgsl
 * city_frustum_hit builds it (cross-section centred on the footprint,
 * scale 1 at fmin.z running linearly to fscale at fmax.z), grown by the
 * shell sideways and by r along every face.
 */
function frustumShape(fmin, fmax, fscale, r) {
  const h2 = [Math.max(0.5 * (fmax[0] - fmin[0]), 1e-4),
              Math.max(0.5 * (fmax[1] - fmin[1]), 1e-4)];
  const sp = (fscale - 1.0) / Math.max(fmax[2] - fmin[2], 1e-4);  // ds/dz
  return {
    fz: fmin[2],
    c2: [0.5 * (fmin[0] + fmax[0]), 0.5 * (fmin[1] + fmax[1])],
    h2, sp,
    z0: fmin[2] - r, z1: fmax[2] + r,
    // A sloped face moved out by r along its normal moves out by
    // r sqrt(1 + slope^2) horizontally.
    grow: [r * Math.hypot(1.0, h2[0] * sp), r * Math.hypot(1.0, h2[1] * sp)],
  };
}

/** Segment vs a grown frustum: the z slab and four sloped slabs. */
function segmentFrustum(p0, d, { fz, c2, h2, sp, z0, z1, grow }) {
  // The half-width along the segment: h2 s(t) + grow, s(t) = sA + sB t.
  const sA = 1.0 + sp * (p0[2] - fz);
  const sB = sp * d[2];
  let tLo = -Infinity, tHi = Infinity;
  let nLo = null;
  if (Math.abs(d[2]) > EPS_DIR) {
    const ta = (z0 - p0[2]) / d[2];
    const tb = (z1 - p0[2]) / d[2];
    if (Math.min(ta, tb) > tLo) {
      tLo = Math.min(ta, tb);
      nLo = [0.0, 0.0, d[2] > 0.0 ? -1.0 : 1.0];
    }
    tHi = Math.min(tHi, Math.max(ta, tb));
  } else if (p0[2] < z0 || p0[2] > z1) {
    return null;
  }
  // Face (ax, sgn): sgn (p - c) - (h2 s + grow) <= 0, linear in t: a + b t.
  for (const [ax, sgn] of [[0, 1], [0, -1], [1, 1], [1, -1]]) {
    const a = sgn * (p0[ax] - c2[ax]) - h2[ax] * sA - grow[ax];
    const b = sgn * d[ax] - h2[ax] * sB;
    if (Math.abs(b) < EPS_DIR) {
      if (a > 0.0) return null;
      continue;
    }
    const tb = -a / b;
    if (b > 0.0) {
      tHi = Math.min(tHi, tb);
    } else if (tb > tLo) {
      tLo = tb;
      const nv = [0.0, 0.0, -h2[ax] * sp];
      nv[ax] = sgn;
      const m = Math.hypot(nv[0], nv[1], nv[2]);
      nLo = [nv[0] / m, nv[1] / m, nv[2] / m];
    }
  }
  if (nLo === null || tLo > tHi || tLo < 0.0 || tLo > 1.0) return null;
  return { t: tLo, normal: nLo };
}

function segmentSolid(p0, d, solid) {
  return solid.frustum ? segmentFrustum(p0, d, solid.frustum)
                       : segmentBox(p0, d, solid);
}

/** Whether p is inside one grown solid of cellSolids (exported for tests). */
export function insideSolid(p, solid) {
  if (solid.frustum) {
    const { fz, c2, h2, sp, z0, z1, grow } = solid.frustum;
    if (!(p[2] > z0 && p[2] < z1)) return false;
    const s = 1.0 + sp * (p[2] - fz);
    return Math.abs(p[0] - c2[0]) < h2[0] * s + grow[0]
        && Math.abs(p[1] - c2[1]) < h2[1] * s + grow[1];
  }
  const { min, max, grow } = solid;
  return [0, 1, 2].every(
    (k) => p[k] > min[k] - grow[k] && p[k] < max[k] + grow[k]);
}

/** The 3 x 3 block of cells about (cx, cy), as [cx, cy] pairs. */
function* neighbourhood(cx, cy) {
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) yield [cx + dx, cy + dy];
  }
}

// Cells are rebuilt far more often than the city changes: a sweep of a frame
// touches the same handful of blocks the last one did. The cache is bounded
// by clearing rather than evicting — at 4096 blocks it holds a 5 km square,
// several seconds of flight at any speed a street allows.
const CELL_CACHE_MAX = 4096;
const NO_PROPS = Object.freeze({ solids: [], top: -Infinity });
const SHRINK_MAX = 0.01;

/**
 * The city as a solid: segment sweeps and point containment in the
 * UNWRAPPED city frame (metres; any real x, y — the tile repeats).
 *
 * `props`, if given, is what the components stand in the city besides its
 * buildings (cityprops.js CityProps): `cellSolids(cc, cx, cy, r)`, the
 * polytopes block (cx, cy)'s column holds and the highest top among them
 * ({solids, top}), and
 * `segmentSolids(p0, p1, r)`, those owned by no block (the freeways) near a
 * segment. Every hit names its `source`.
 */
export class CityLots {
  constructor(tile, props = null) {
    if (!(tile instanceof CityTile)) {
      throw new Error("CityLots needs a CityTile.");
    }
    this.tile = tile;
    this.props = props;
    this._cells = new Map();
  }

  /** city_cell at an unwrapped index, cached. */
  cell(cx, cy) {
    const key = `${cx},${cy}`;
    let cc = this._cells.get(key);
    if (cc === undefined) {
      if (this._cells.size >= CELL_CACHE_MAX) this._cells.clear();
      cc = cityCell(this.tile, cx, cy);
      this._cells.set(key, cc);
    }
    return cc;
  }

  /**
   * The first solid surface a sphere of radius r, its centre swept from p0
   * to p1, runs into: {t, point, normal} (t in [0, 1], `point` the centre
   * at contact, `normal` the unit outward normal of the face met), or null.
   *
   * city_trace's DDA over the block grid, with one difference that matters
   * for a lot: a building that spans several blocks is reported by every
   * member cell, so a hit found while visiting one cell may lie beyond it.
   * The nearest hit so far is kept across cells and the walk stops once the
   * current cell's exit is past it — no nearer surface can lie further on.
   * Each visited cell tests its neighbours' solids too (see the growth note
   * above); a surface the segment meets at t lies within reach of the cell
   * the segment is in at t, so the stopping rule still holds.
   */
  sweep(p0, p1, r = 0.0, eye = p0) {
    // Folded into the primary tile by whole tiles at the EYE, as the shader
    // folds its camera (city_fold_xy): the buildings repeat with the tile,
    // but the props' lattices (freeway routes, lamp stations, parking slots,
    // pylon bays) do not divide it, so beyond the first tile they stand
    // where the renderer — which draws from the folded camera — puts them
    // only once the query is folded where the camera is. Within a few
    // metres of the tile's seam that is not where the segment starts (the
    // flyer flies metres ahead), so the collider passes this frame's eye.
    const [sx, sy] = this._fold(eye);
    const hit = this._sweep([p0[0] - sx, p0[1] - sy, p0[2]],
                            [p1[0] - sx, p1[1] - sy, p1[2]], r);
    if (hit !== null) hit.point = [hit.point[0] + sx, hit.point[1] + sy,
                                   hit.point[2]];
    return hit;
  }

  _sweep(p0, p1, r) {
    this._checkReach(r);
    const cell = this.tile.cellM;
    const d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let best = null;
    const take = (hit) => {
      if (hit && (best === null || hit.t < best.t)) best = hit;
    };

    // The ground: a plane, entered from above.
    const floor = CITY_GROUND_Z + r;
    if (d[2] < 0.0 && p0[2] >= floor) {
      const tg = (floor - p0[2]) / d[2];
      if (tg <= 1.0) take({ t: tg, normal: [0.0, 0.0, 1.0], source: "ground" });
    }
    if (this.props !== null) {
      for (const solid of this.props.segmentSolids(p0, p1, r)) {
        const hit = segmentPolytope(p0, d, solid, r);
        if (hit) take({ ...hit, source: solid.source });
      }
    }
    if (Math.min(p0[2], p1[2]) > CITY_SLAB_TOP + r) {
      return finish(best, p0, d);
    }

    let cx = Math.floor(p0[0] / cell), cy = Math.floor(p0[1] / cell);
    const stepX = d[0] > 0.0 ? 1 : -1, stepY = d[1] > 0.0 ? 1 : -1;
    const invX = Math.abs(d[0]) > EPS_DIR ? 1.0 / d[0] : Infinity;
    const invY = Math.abs(d[1]) > EPS_DIR ? 1.0 / d[1] : Infinity;
    let tmaxX = Math.abs(d[0]) > EPS_DIR
      ? ((d[0] > 0.0 ? (cx + 1) * cell : cx * cell) - p0[0]) * invX : Infinity;
    let tmaxY = Math.abs(d[1]) > EPS_DIR
      ? ((d[1] > 0.0 ? (cy + 1) * cell : cy * cell) - p0[1]) * invY : Infinity;
    const tdX = cell * Math.abs(invX), tdY = cell * Math.abs(invY);

    let tCur = 0.0;
    const tested = new Set();
    // A segment crosses at most this many cells; the bound is the walk's
    // termination guarantee, not a range limit.
    const maxCells = 4 + Math.ceil((Math.abs(d[0]) + Math.abs(d[1])) / cell);
    for (let i = 0; i <= maxCells; i++) {
      const tExit = Math.min(tmaxX, tmaxY, 1.0);
      if (best !== null && best.t <= tCur) break;
      const zLow = Math.min(p0[2] + tCur * d[2], p0[2] + tExit * d[2]);
      for (const [nx, ny] of neighbourhood(cx, cy)) {
        const key = `${nx},${ny}`;
        if (tested.has(key)) continue;
        const cc = this.cell(nx, ny);
        const props = this._props(cc, nx, ny, r);
        const top = Math.max(cc.built ? cc.top_z : -Infinity, props.top);
        if (zLow >= top + 1.0 + r) continue;
        tested.add(key);
        if (cc.built) {
          for (const solid of cellSolids(cc, r)) {
            const hit = segmentSolid(p0, d, solid);
            if (hit) take({ ...hit, source: "building" });
          }
        }
        for (const solid of props.solids) {
          const hit = segmentPolytope(p0, d, solid, r);
          if (hit) take({ ...hit, source: solid.source });
        }
      }
      if (tExit >= 1.0 || (best !== null && best.t <= tExit)) break;
      if (tmaxX < tmaxY) {
        cx += stepX; tCur = tmaxX; tmaxX += tdX;
      } else {
        cy += stepY; tCur = tmaxY; tmaxY += tdY;
      }
    }
    return finish(best, p0, d);
  }

  /**
   * Whether a sphere of radius r at p overlaps a solid: null if not, else
   * {top} — the z its underside has to be lifted to in order to stand clear
   * on top (the roof-and-mast z of the cell it overlaps; the street's for
   * the ground).
   */
  inside(p, r = 0.0, eye = p) {
    const [sx, sy] = this._fold(eye);
    return this._inside([p[0] - sx, p[1] - sy, p[2]], r);
  }

  /** The whole-tile shift that folds p into the primary tile. */
  _fold(p) {
    const e = this.tile.n * this.tile.cellM;
    return [Math.floor(p[0] / e) * e, Math.floor(p[1] / e) * e];
  }

  _inside(p, r) {
    this._checkReach(r);
    if (p[2] < CITY_GROUND_Z + r) return { top: CITY_GROUND_Z };
    const cell = this.tile.cellM;
    let top = null;
    for (const [nx, ny] of neighbourhood(Math.floor(p[0] / cell),
                                         Math.floor(p[1] / cell))) {
      const cc = this.cell(nx, ny);
      if (cc.built && p[2] < cc.top_z + r
          && cellSolids(cc, r).some((solid) => insideSolid(p, solid))) {
        top = Math.max(top ?? -Infinity, cc.top_z);
      }
      for (const solid of this._props(cc, nx, ny, r).solids) {
        if (insidePolytope(p, solid, r)) {
          top = Math.max(top ?? -Infinity, solid.top);
        }
      }
    }
    if (this.props !== null) {
      for (const solid of this.props.segmentSolids(p, p, r)) {
        if (insidePolytope(p, solid, r)) {
          top = Math.max(top ?? -Infinity, solid.top);
        }
      }
    }
    return top === null ? null : { top };
  }

  /** The props a block's column holds, or none. */
  _props(cc, cx, cy, r) {
    return this.props === null ? NO_PROPS
                               : this.props.cellSolids(cc, cx, cy, r);
  }

  /**
   * The growth must stay under a cell for the 3 x 3 walk to be exact. A
   * radius may also be a hair negative, the solids shrunk rather than grown
   * (to SHRINK_MAX): tests/test_city_props_mirror.py brackets the GPU's
   * float32 decision at an edge between a grown and a shrunk sweep.
   */
  _checkReach(r) {
    if (!(r >= -SHRINK_MAX && CITY_SHELL_OUT_MAX + r < this.tile.cellM)) {
      throw new Error(
        `A city sweep's body radius ${r} m plus the facade shell ` +
        `(${CITY_SHELL_OUT_MAX} m) must stay under a block ` +
        `(${this.tile.cellM} m), and not below -${SHRINK_MAX} m.`);
    }
  }
}

function finish(best, p0, d) {
  if (best === null) return null;
  return {
    t: best.t,
    point: [p0[0] + best.t * d[0], p0[1] + best.t * d[1],
            p0[2] + best.t * d[2]],
    normal: best.normal,
    source: best.source,
  };
}
