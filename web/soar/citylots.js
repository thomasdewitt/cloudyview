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
// What is solid: the building boxes b1..b3 (as many as `tiers` says), the
// frustum of archetypes 3 and 4, the mast, and the ground plane. Everything
// the components add — street furniture, props, industry roof ridges,
// skyways — is scenery the flyer passes through (decided with the
// city-layout session, 2026-09-23).
//
// city_cell below is kept line for line with the WGSL so that re-syncing
// after a change there is a diff read, not a re-derivation. It deliberately
// stops at geometry: the window style, palette and occupancy draws that
// follow in the WGSL decide nothing a flyer can hit.

"use strict";

import { fromHalf } from "./half.js";

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
export const CITY_TOWER_KNEE_H = 800.0;
export const CITY_SLENDER_MAX = 20.0;
export const CITY_USE_URBAN = 0;
export const CITY_USE_INDUSTRIAL = 1;
export const CITY_USE_YARD = 2;
export const CITY_USE_OPEN = 3;
export const CITY_WORD_OFFSET = 2048.0;
export const CITY_MAST_MIN_H = 150.0;
export const CITY_MAST_CROSS = 1.4;

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
function tileCell(n, k) {
  return ((k % n) + n) % n;
}

/** raymarch.wgsl city_is_avenue. */
function isAvenue(k) {
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
  const c = {};
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
    const h_eff = h > CITY_TOWER_KNEE_H
      ? CITY_TOWER_KNEE_H * Math.pow(h / CITY_TOWER_KNEE_H, 0.6) : h;
    const side = Math.max(
      CITY_TOWER_SIDE_K * Math.sqrt(h_eff) * (0.80 + 0.40 * r4[1]),
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
    if (r4d[0] < 0.20) {
      c.arch = 2;                   // growth: cantilevered buds
    } else if (r4d[0] < 0.42) {
      c.arch = 3;                   // tapered shaft on a podium
    } else if (r4d[0] < 0.55) {
      c.arch = 4;                   // spire crown
    }
  }
  if (h > 600.0 && r4d[1] < (h > 1200.0 ? 0.85 : 0.65)) {
    c.arch = r4d[1] < 0.30 ? 4 : 3;
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
    c.fscale = 0.55 + 0.25 * r4d[2];
  } else if (c.arch === 4) {
    // A straight shaft to three quarters, then a steep spire.
    const zs = storeyZ(ns * f(0.75));
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
    c.fscale = 0.12 + 0.10 * r4d[2];
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
    // Clamped to the lot minus a hair, so no prim leaves it.
    bud_min = [Math.max(bud_min[0], cmin[0] + 0.5),
               Math.max(bud_min[1], cmin[1] + 0.5)];
    bud_max = [Math.min(bud_max[0], cmax[0] - 0.5),
               Math.min(bud_max[1], cmax[1] - 0.5)];
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
      c.b3min = [Math.max(Math.min(m0[0], m1[0]), cmin[0] + 0.5),
                 Math.max(Math.min(m0[1], m1[1]), cmin[1] + 0.5), cz0];
      c.b3max = [Math.min(Math.max(m0[0], m1[0]), cmax[0] - 0.5),
                 Math.min(Math.max(m0[1], m1[1]), cmax[1] - 0.5), cz1];
    }
  } else if (podium) {
    // Podium, tower, crown.
    c.tiers = 3;
    const zc = storeyZ(ns * f(f(0.70) + f(f(0.15) * r4b[3])));
    c.b1min = vec3(pod_min, CITY_GROUND_Z);
    c.b1max = vec3(pod_max, zp);
    c.b2min = vec3(bmin2, zp);
    c.b2max = vec3(bmax2, zc);
    const s3 = 0.62 + 0.18 * r4b[2];
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

  // (The WGSL's window style, house palette and occupancy draws go here.
  // None of them is geometry.)

  // Roof furniture belongs on roofs.
  if (h > CITY_MAST_MIN_H && r4b[2] < 0.65 && c.arch !== 4) {
    c.has_mast = true;
    const mast_h = f(12.0 + f(45.0 * r4b[3]));
    const crown = c.arch === 3 ? c.fscale * 0.85 : 1.0;
    const moff = [(r4c[3] - 0.5) * fw[0] * (0.4 * crown),
                  (r4[0] - 0.5) * fw[1] * (0.4 * crown)];
    const mc = [bc[0] + moff[0], bc[1] + moff[1]];
    c.mast_min = [mc[0] - 0.5 * CITY_MAST_CROSS,
                  mc[1] - 0.5 * CITY_MAST_CROSS, h];
    c.mast_max = [mc[0] + 0.5 * CITY_MAST_CROSS,
                  mc[1] + 0.5 * CITY_MAST_CROSS, f(h + mast_h)];
    c.top_z = f(h + mast_h);
  } else {
    c.top_z = h;
  }
  return c;
}

// --- intersection ------------------------------------------------------------
//
// Segments, not rays: p(t) = p0 + t (p1 - p0), t in [0, 1]. A hit reports
// the ENTRY — a segment that starts inside a solid does not hit it. That is
// the right answer for a sweep (a flyer already inside cannot be stopped at
// a wall it is behind), and the collider never starts one inside: it keeps
// its bodies a skin's width off every surface and asks inside() separately.

const EPS_DIR = 1e-12;

/** Segment vs axis-aligned box: {t, normal} of the entry, or null. */
function segmentBox(p0, d, bmin, bmax) {
  let tLo = -Infinity, tHi = Infinity, axis = -1;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(d[k]) < EPS_DIR) {
      if (p0[k] < bmin[k] || p0[k] > bmax[k]) return null;
      continue;
    }
    let ta = (bmin[k] - p0[k]) / d[k];
    let tb = (bmax[k] - p0[k]) / d[k];
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
 * Segment vs the rectangular frustum of archetypes 3 and 4 —
 * raymarch.wgsl city_frustum_hit over t in [0, 1]: the z slab and four
 * sloped slabs, each linear in t.
 */
function segmentFrustum(p0, d, fmin, fmax, fscale) {
  const c2 = [0.5 * (fmin[0] + fmax[0]), 0.5 * (fmin[1] + fmax[1])];
  const h2 = [Math.max(0.5 * (fmax[0] - fmin[0]), 1e-4),
              Math.max(0.5 * (fmax[1] - fmin[1]), 1e-4)];
  const zspan = Math.max(fmax[2] - fmin[2], 1e-4);
  const sp = (fscale - 1.0) / zspan;          // ds/dz
  const sA = 1.0 + sp * (p0[2] - fmin[2]);
  const sB = sp * d[2];
  let tLo = -Infinity, tHi = Infinity;
  let nLo = null;
  if (Math.abs(d[2]) > EPS_DIR) {
    const ta = (fmin[2] - p0[2]) / d[2];
    const tb = (fmax[2] - p0[2]) / d[2];
    if (Math.min(ta, tb) > tLo) {
      tLo = Math.min(ta, tb);
      nLo = [0.0, 0.0, d[2] > 0.0 ? -1.0 : 1.0];
    }
    tHi = Math.min(tHi, Math.max(ta, tb));
  } else if (p0[2] < fmin[2] || p0[2] > fmax[2]) {
    return null;
  }
  // Face k: sign * (p - c) - h s <= 0 along axis `ax`.
  for (const [ax, sgn] of [[0, 1], [0, -1], [1, 1], [1, -1]]) {
    const a = sgn * (p0[ax] - c2[ax]) - h2[ax] * sA;
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

function insideBox(p, bmin, bmax) {
  return p[0] > bmin[0] && p[0] < bmax[0] && p[1] > bmin[1]
      && p[1] < bmax[1] && p[2] > bmin[2] && p[2] < bmax[2];
}

function insideFrustum(p, fmin, fmax, fscale) {
  if (!(p[2] > fmin[2] && p[2] < fmax[2])) return false;
  const s = 1.0 + (fscale - 1.0) * (p[2] - fmin[2])
                  / Math.max(fmax[2] - fmin[2], 1e-4);
  const cx = 0.5 * (fmin[0] + fmax[0]), cy = 0.5 * (fmin[1] + fmax[1]);
  return Math.abs(p[0] - cx) < 0.5 * (fmax[0] - fmin[0]) * s
      && Math.abs(p[1] - cy) < 0.5 * (fmax[1] - fmin[1]) * s;
}

/** A built cell's solid boxes, as city_trace tests them. */
function cellBoxes(cc) {
  const boxes = [[cc.b1min, cc.b1max]];
  if (cc.tiers >= 2) boxes.push([cc.b2min, cc.b2max]);
  if (cc.tiers >= 3) boxes.push([cc.b3min, cc.b3max]);
  if (cc.has_mast) boxes.push([cc.mast_min, cc.mast_max]);
  return boxes;
}

// Cells are rebuilt far more often than the city changes: a sweep of a frame
// touches the same handful of blocks the last one did. The cache is bounded
// by clearing rather than evicting — at 4096 blocks it holds a 5 km square,
// several seconds of flight at any speed a street allows.
const CELL_CACHE_MAX = 4096;

/**
 * The city as a solid: segment sweeps and point containment in the
 * UNWRAPPED city frame (metres; any real x, y — the tile repeats).
 */
export class CityLots {
  constructor(tile) {
    if (!(tile instanceof CityTile)) {
      throw new Error("CityLots needs a CityTile.");
    }
    this.tile = tile;
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
   * The first solid surface the segment p0 -> p1 enters:
   * {t, point, normal} (t in [0, 1], normal the unit outward normal of the
   * face entered), or null.
   *
   * city_trace's DDA over the block grid, with one difference that matters
   * for a lot: a building that spans several blocks is reported by every
   * member cell, so a hit found while visiting one cell may lie beyond it.
   * The nearest hit so far is kept across cells and the walk stops once the
   * current cell's exit is past it — no nearer surface can lie further on.
   */
  sweep(p0, p1) {
    const cell = this.tile.cellM;
    const d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let best = null;
    const take = (hit) => {
      if (hit && (best === null || hit.t < best.t)) best = hit;
    };

    // The ground: a plane, entered from above.
    if (d[2] < 0.0 && p0[2] >= CITY_GROUND_Z) {
      const tg = (CITY_GROUND_Z - p0[2]) / d[2];
      if (tg <= 1.0) take({ t: tg, normal: [0.0, 0.0, 1.0] });
    }
    if (Math.min(p0[2], p1[2]) > CITY_SLAB_TOP) return finish(best, p0, d);

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
    // A segment crosses at most this many cells; the bound is the walk's
    // termination guarantee, not a range limit.
    const maxCells = 4 + Math.ceil((Math.abs(d[0]) + Math.abs(d[1])) / cell);
    for (let i = 0; i <= maxCells; i++) {
      const tExit = Math.min(tmaxX, tmaxY, 1.0);
      if (best !== null && best.t <= tCur) break;
      const za = p0[2] + tCur * d[2], zb = p0[2] + tExit * d[2];
      const cc = this.cell(cx, cy);
      if (cc.built && Math.min(za, zb) < cc.top_z + 1.0) {
        for (const [bmin, bmax] of cellBoxes(cc)) {
          take(segmentBox(p0, d, bmin, bmax));
        }
        if (cc.has_frustum) {
          take(segmentFrustum(p0, d, cc.fmin, cc.fmax, cc.fscale));
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
   * The solid containing p, or null: {top} — the height a body inside it
   * has to be lifted to in order to stand on it (the cell's roof-and-mast
   * z; the ground's own z below the street).
   */
  inside(p) {
    if (p[2] < CITY_GROUND_Z) return { top: CITY_GROUND_Z };
    const cell = this.tile.cellM;
    const cc = this.cell(Math.floor(p[0] / cell), Math.floor(p[1] / cell));
    if (!cc.built || p[2] >= cc.top_z) return null;
    for (const [bmin, bmax] of cellBoxes(cc)) {
      if (insideBox(p, bmin, bmax)) return { top: cc.top_z };
    }
    if (cc.has_frustum && insideFrustum(p, cc.fmin, cc.fmax, cc.fscale)) {
      return { top: cc.top_z };
    }
    return null;
  }
}

function finish(best, p0, d) {
  if (best === null) return null;
  return {
    t: best.t,
    point: [p0[0] + best.t * d[0], p0[1] + best.t * d[1],
            p0[2] + best.t * d[2]],
    normal: best.normal,
  };
}
