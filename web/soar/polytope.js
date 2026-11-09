// Convex polytopes: the solids the city's props are made of (cityprops.js),
// and what citylots.js sweeps them with.
//
// A polytope is a list of planes n.p <= d, each n a unit outward normal. A
// box is six of them, a ramp of freeway is a box with its top and bottom
// sheared along the deck, a sawtooth tooth is a triangular prism. Sweeping a
// sphere of radius r against one is sweeping its centre against the
// polytope with every plane pushed out by r, and that is a Cyrus-Beck clip
// of the segment against the planes: linear in t, one divide a plane, the
// entry face the plane that raised the near end last.
//
// Pushing the planes out is exact on the faces and conservative elsewhere,
// as the buildings' boxes are: where the true grown shape is rounded, the
// pushed planes meet square, r (sqrt 2 - 1) further out at a right-angled
// edge (14 cm for the camera's 0.35 m) and r (sqrt 3 - 1) at a corner
// (26 cm). At an acute edge they meet much further out — r / sin(half the
// angle) — so a solid with a sharp wedge carries a bevel plane across it
// (cityprops.js, the sawtooth's tail).
//
// A plane may carry a fifth number, how much of r it is pushed out by: 0
// for a face a solid shares with its neighbour (the joints between the
// pieces of a freeway deck). Grown, a shared face would stand proud of the
// neighbour's grown surface wherever the two slopes differ — a false wall
// across a smooth road.
//
// Each solid carries `top`, the height a body found inside it is lifted to
// in order to stand clear (CityLots.inside's answer for a building is its
// roof), and `source`, what it is.

"use strict";

const EPS_DIR = 1e-12;

/** A polytope from its planes ([nx, ny, nz, d], n unit), top and source. */
export function polytope(planes, top, source) {
  if (!(planes.length >= 4)) {
    throw new Error(`a polytope needs at least four planes; got ${planes.length}`);
  }
  return { planes, top, source };
}

/** The offset of plane pl pushed out for a body of radius r. */
const grownD = (pl, r) => pl[3] + r * (pl.length > 4 ? pl[4] : 1.0);

/** The axis-aligned box [lo, hi] as a polytope. */
export function boxSolid(lo, hi, source, top = hi[2]) {
  return polytope([
    [1, 0, 0, hi[0]], [-1, 0, 0, -lo[0]],
    [0, 1, 0, hi[1]], [0, -1, 0, -lo[1]],
    [0, 0, 1, hi[2]], [0, 0, -1, -lo[2]],
  ], top, source);
}

/**
 * Segment p0 + t d, t in [0, 1], swept by a sphere of radius r against a
 * polytope: {t, normal} of the entry, or null. A segment that starts inside
 * does not hit (the sweep reports entries; see citylots.js).
 */
export function segmentPolytope(p0, d, { planes }, r) {
  let tLo = -Infinity, tHi = Infinity, enter = null;
  for (const pl of planes) {
    const a = pl[0] * p0[0] + pl[1] * p0[1] + pl[2] * p0[2] - grownD(pl, r);
    const b = pl[0] * d[0] + pl[1] * d[1] + pl[2] * d[2];
    if (Math.abs(b) < EPS_DIR) {
      if (a > 0.0) return null;
      continue;
    }
    const t = -a / b;
    if (b < 0.0) {
      if (t > tLo) { tLo = t; enter = pl; }
    } else if (t < tHi) {
      tHi = t;
    }
    if (tLo > tHi) return null;
  }
  if (enter === null || tLo < 0.0 || tLo > 1.0) return null;
  return { t: tLo, normal: [enter[0], enter[1], enter[2]] };
}

/** Whether a sphere of radius r at p overlaps the polytope. */
export function insidePolytope(p, { planes }, r) {
  return planes.every(
    (pl) => pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] < grownD(pl, r));
}
