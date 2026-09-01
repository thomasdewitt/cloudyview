// Domain geometry: bounding boxes, voxel sizes, and the rules for nesting one
// field inside another.
//
// Originally ported from Python (soar's engine.py and io.py's nest-pair
// finder, both since deleted — this is now the only copy). The AABB
// arithmetic must still agree with cloudyview/soar_host.py to the float —
// these numbers decide what the shader samples, and a browser that places a
// nest differently from the desktop is worse than one that refuses.
//
// Conventions: coordinates are CELL CENTRES in metres, ascending, and every
// bounding box pads by half a cell. Array layout is (x, y, z) with z fastest.

"use strict";

import {
  AERIAL_PERSPECTIVE_STRENGTH, AERIAL_BETA_PER_KM, AERIAL_SCALE_HEIGHT_M,
  PERIODIC_AIR_TAU_CUTOFF, PERIODIC_MAX_RANGE_M,
} from "./constants.js";

/** io.NEST_OVERHANG_FRACTION — 1% of the parent span, per axis. */
export const NEST_OVERHANG_FRACTION = 0.01;

const AXES = ["x", "y", "z"];

/**
 * Absolute-metre AABB with half-cell padding.
 *
 * The z half-cells come from the FIRST and LAST spacings separately, which is
 * what lets a stretched vertical grid land its boundary in the right place.
 * (The march itself still maps world to grid linearly, so the interior of a
 * stretched grid is rendered as if uniform — same as the desktop.)
 */
export function volumeAABB(x, y, z) {
  const half = (c, i, j) => 0.5 * Math.abs(c[i] - c[j]);
  const min = (c) => c.reduce((a, b) => (b < a ? b : a), Infinity);
  const max = (c) => c.reduce((a, b) => (b > a ? b : a), -Infinity);
  return {
    bmin: [
      min(x) - half(x, 1, 0),
      min(y) - half(y, 1, 0),
      min(z) - half(z, 1, 0),
    ],
    bmax: [
      max(x) + half(x, 1, 0),
      max(y) + half(y, 1, 0),
      max(z) + half(z, z.length - 1, z.length - 2),
    ],
  };
}

/** Per-axis voxel edge (m). */
export function voxelSizes(shape, bmin, bmax) {
  return [0, 1, 2].map((i) => (bmax[i] - bmin[i]) / shape[i]);
}

/** Smallest voxel edge (m) — the scale the march steps in. */
export function minVoxelSize(shape, bmin, bmax) {
  return Math.min(...voxelSizes(shape, bmin, bmax));
}

/**
 * Per-axis (overhang, allowance) in metres for a nest inside a parent.
 *
 * Both boxes are built from cell edges, so a nest whose outermost cell is
 * thicker than its parent's can sit proud of the parent's box for no reason
 * but the grid. That is what the allowance absorbs, and it is why the
 * allowance is a fraction of the parent's span OR one parent cell, whichever
 * is larger: the sliver is measured in parent cells, and a fraction of the
 * span only stands in for that while the parent has many of them. A turbulon
 * parent three cells tall over 15 km has 149 m of fraction against 4.9 km of
 * cell, which refused its own middle level as a coordinate error — see
 * io.nest_overhang for the case.
 */
export function nestOverhang(outerMin, outerMax, nestMin, nestMax,
                             outerSpacing) {
  const overhang = [], allowance = [];
  for (let i = 0; i < 3; i++) {
    overhang.push(Math.max(
      outerMin[i] - nestMin[i], nestMax[i] - outerMax[i], 0.0));
    allowance.push(Math.max(
      NEST_OVERHANG_FRACTION * Math.max(outerMax[i] - outerMin[i], 1.0),
      outerSpacing[i]));
  }
  return { overhang, allowance };
}

/**
 * The nest must lie inside the outer AABB, give or take a cell edge.
 *
 * Returns {clipped} — a human-readable note when a sliver was clipped, empty
 * otherwise. Throws when the miss is real: that is nearly always two fields
 * that were never meant to be composed, or one whose coordinates are in the
 * wrong units or on the wrong origin.
 */
export function validateNestContainment(bmin, bmax, nestMin, nestMax,
                                        spacing) {
  const tol = bmin.map((_, i) => 1e-9 * Math.max(bmax[i] - bmin[i], 1.0));
  const { overhang, allowance } =
    nestOverhang(bmin, bmax, nestMin, nestMax, spacing);

  if (overhang.some((o, i) => o > allowance[i])) {
    const detail = AXES.map((a, i) =>
      `${a}: nest [${nestMin[i].toFixed(1)}, ${nestMax[i].toFixed(1)}] ` +
      `vs outer [${bmin[i].toFixed(1)}, ${bmax[i].toFixed(1)}]`).join(", ");
    let worst = 0;
    for (let i = 1; i < 3; i++) {
      if (overhang[i] - allowance[i] > overhang[worst] - allowance[worst]) {
        worst = i;
      }
    }
    throw new Error(
      "The nested field must lie inside the outer field's bounding box " +
      `(absolute meters); it does not. ${detail}. The worst axis ` +
      `(${AXES[worst]}) overhangs by ${overhang[worst].toFixed(1)} m, past ` +
      `the ${allowance[worst].toFixed(1)} m that gets clipped away. Check ` +
      "that both fields carry absolute coordinates on the same origin and " +
      "in the same units.");
  }

  if (nestMax.some((v, i) => v <= nestMin[i])) {
    throw new Error(
      `The nested field has a degenerate bounding box: ${nestMin} -> ${nestMax}.`);
  }

  // The finest level covering a point always wins, so a nest that fills the
  // outer box on every axis hides the parent completely — two renders of one
  // domain rather than a refinement of part of it.
  if (nestMin.every((v, i) => v <= bmin[i] + tol[i]) &&
      nestMax.every((v, i) => v >= bmax[i] - tol[i])) {
    throw new Error(
      `The nested field covers the entire outer domain (${nestMin} -> ` +
      `${nestMax}), so nesting it would hide the outer field completely. ` +
      "These look like two renders of the same domain rather than a coarse " +
      "field and a refinement of part of it; open the finer one on its own.");
  }

  const clippedAxes = AXES.filter((_, i) => overhang[i] > tol[i]);
  return {
    clipped: clippedAxes.length
      ? "The nest overhangs the outer domain (" +
        clippedAxes.map((a) => {
          const i = AXES.indexOf(a);
          return `${a} by ${overhang[i].toFixed(1)} m`;
        }).join(", ") +
        "); the overhanging sliver is clipped and will not be rendered."
      : "",
  };
}

/**
 * Every (outer, inner) pair among these domains that forms a valid nest.
 *
 * `domains` is [{name, bmin, bmax, spacing}] with `spacing` the per-axis
 * minimum grid spacing. Finer means: no axis coarser, at least one axis
 * strictly finer — per-axis, because the ordinary nest refines horizontally
 * while sharing the parent's vertical levels, and a single scalar would rank
 * such a pair as a tie. Three levels of refinement give
 * several qualifying pairs (coarse+middle, coarse+fine, middle+fine) and all
 * of them are offered — which of the two you want is not ours to guess.
 */
export function nestablePairs(domains) {
  const pairs = [];
  for (const outer of domains) {
    for (const inner of domains) {
      if (inner === outer) continue;
      const noAxisCoarser =
        inner.spacing.every((s, i) => s <= outer.spacing[i]);
      const oneAxisFiner =
        inner.spacing.some((s, i) => s < outer.spacing[i]);
      if (!noAxisCoarser || !oneAxisFiner) continue;
      const tol = outer.bmin.map(
        (_, i) => 1e-9 * Math.max(outer.bmax[i] - outer.bmin[i], 1.0));
      const { overhang, allowance } = nestOverhang(
        outer.bmin, outer.bmax, inner.bmin, inner.bmax, outer.spacing);
      if (overhang.some((o, i) => o > allowance[i])) continue;
      const covers =
        inner.bmin.every((v, i) => v <= outer.bmin[i] + tol[i]) &&
        inner.bmax.every((v, i) => v >= outer.bmax[i] - tol[i]);
      if (covers) continue;
      pairs.push([outer.name, inner.name]);
    }
  }
  return pairs;
}

/**
 * Cell-edge extent of one level, matching io.group_domain_extent so extents
 * are comparable across groups. `spacing` is the smallest spacing per axis —
 * kept per-axis because refinement is a per-axis relation (see nestablePairs).
 *
 * NOT volumeAABB. The two differ, and deliberately: engine._volume_aabb pads
 * x and y by the FIRST spacing at both ends, because the march maps world to
 * grid linearly and a horizontal box that does not match that mapping puts
 * the field in the wrong place. group_domain_extent pads every axis by its
 * own first and last spacings, because its job is to compare one grid's
 * footprint with another's. On a uniform horizontal grid these agree, which
 * is why delegating here went unnoticed; on a stretched one it decided
 * nestability from the wrong box.
 */
export function domainExtent(x, y, z) {
  const bmin = [], bmax = [], spacing = [];
  for (const c of [x, y, z]) {
    const lo = 0.5 * Math.abs(c[1] - c[0]);
    const hi = 0.5 * Math.abs(c[c.length - 1] - c[c.length - 2]);
    let min = Infinity, max = -Infinity;
    for (const v of c) { if (v < min) min = v; if (v > max) max = v; }
    bmin.push(min - lo);
    bmax.push(max + hi);
    let dx = Infinity;
    for (let i = 1; i < c.length; i++) {
      dx = Math.min(dx, Math.abs(c[i] - c[i - 1]));
    }
    spacing.push(dx);
  }
  return { bmin, bmax, spacing };
}

/**
 * How far a periodic march can usefully go before clear-air extinction (or
 * the absolute range ceiling) makes further samples invisible. Mirror of
 * the shader's own cap.
 */
export function periodicMarchCapM(
  camZ, direction, bmin, bmax,
  aerialPerspectiveStrength = AERIAL_PERSPECTIVE_STRENGTH,
) {
  let cap = Infinity;
  const hLen = Math.hypot(direction[0], direction[1]);
  if (hLen > 1e-8) {
    cap = PERIODIC_MAX_RANGE_M / hLen;
  }
  if (aerialPerspectiveStrength > 0.0) {
    const beta0 = AERIAL_BETA_PER_KM * 1e-3;
    const scaleH = AERIAL_SCALE_HEIGHT_M;
    const z0 = Math.max(camZ, 0.0);
    const mu = direction[2];
    const tauCap = PERIODIC_AIR_TAU_CUTOFF / aerialPerspectiveStrength;
    const e0 = Math.exp(-z0 / scaleH);
    if (Math.abs(mu) > 1e-6) {
      const a = e0 - tauCap * mu / (beta0 * scaleH);
      if (a > 0.0) {
        const t = (-scaleH * Math.log(a) - z0) / mu;
        if (t > 0.0) cap = Math.min(cap, t);
      }
    } else {
      cap = Math.min(cap, tauCap / (beta0 * e0));
    }
  }
  return cap;
}

/** Forward distance at which a ray leaves the slab(s) of `axes`. */
function slabExitT(origin, direction, lo, hi, axes) {
  let tExit = Infinity;
  for (const axis of axes) {
    const d = direction[axis];
    if (Math.abs(d) < 1e-12) continue;
    const t0 = (lo[axis] - origin[axis]) / d;
    const t1 = (hi[axis] - origin[axis]) / d;
    tExit = Math.min(tExit, Math.max(t0, t1));
  }
  return tExit;
}

/**
 * True when a periodic view marches past the domain's lateral walls — i.e.
 * when you are seeing wrapped copies of the field.
 *
 * behold's volume is finite and does not tile, so its frame of the same
 * camera will differ. That is worth saying before someone waits an hour for
 * a render that does not match what they framed.
 */
export function viewSpansDomainEdge(origin, basis, fovDeg, aspect, bmin, bmax) {
  const [forward, right, up] = basis;
  const tanHalf = Math.tan(fovDeg * Math.PI / 360.0);
  const directions = [forward];
  for (const sx of [-1.0, 1.0]) {
    for (const sy of [-1.0, 1.0]) {
      const d = [0, 1, 2].map((i) =>
        forward[i] + sx * tanHalf * right[i] + sy * tanHalf / aspect * up[i]);
      const n = Math.hypot(d[0], d[1], d[2]);
      directions.push([d[0] / n, d[1] / n, d[2] / n]);
    }
  }
  for (const direction of directions) {
    const tHorizontal = slabExitT(origin, direction, bmin, bmax, [0, 1]);
    const tVertical = slabExitT(origin, direction, bmin, bmax, [2]);
    const cap = periodicMarchCapM(origin[2], direction, bmin, bmax);
    if (tHorizontal < Math.min(tVertical, cap)) return true;
  }
  return false;
}

/**
 * Coordinates are stored ascending. A strictly descending axis is reversed,
 * along with the field data on that axis — matching CloudField.__post_init__,
 * including its strictness: one flat or ascending step and no flip happens.
 */
export function needsFlip(coord) {
  if (coord.length < 2) return false;
  for (let i = 1; i < coord.length; i++) {
    if (!(coord[i] - coord[i - 1] < 0)) return false;
  }
  return true;
}
