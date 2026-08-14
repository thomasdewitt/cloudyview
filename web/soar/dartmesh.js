// The paper dart's geometry: an A4 sheet, actually folded.
//
// The temptation with a paper aeroplane is to model the finished object —
// place a long triangle, tilt two wings, done. Do that and it looks like a
// triangle, because the things that say "paper" are not the silhouette. They
// are that every panel is a rigid piece of one flat sheet, that the nose is
// eight layers thick and the wing tip is one, and that the edges are cut
// rather than grown. None of that survives being modelled by eye.
//
// So this folds a sheet. The sheet is 210 x 297 mm of ordinary paper in
// coordinates (s, t): s across, signed from the centreline, t along, zero at
// the end that becomes the nose. The classic dart is five folds:
//
//   1. in half along s = 0, and unfold — a reference crease only
//   2. both top corners down to that crease. The fold line leaves the nose at
//      45 degrees, and it lands the top edge exactly on the centreline
//   3. the new slanted edges down to the centreline again. That fold line
//      leaves the nose at 22.5 degrees, and 22.5 degrees is where the whole
//      shape comes from: it is the half-angle of the finished dart, so the
//      leading edge is swept 67.5 degrees from the lateral
//   4. in half along s = 0 again, for real this time
//   5. each wing down along a line 18 mm out from that crease, leaving the
//      keel
//
// Everything below is derived from those five folds and the two numbers a
// sheet of A4 actually has. The span is not chosen: it is 2 x (105 - 18) =
// 174 mm because that is what is left of the sheet's width once the keel is
// taken out of it. The pure-keel nose is not chosen either: it is the first
// 43.5 mm, where the planform is still narrower than the keel is deep.
//
// LAYERS ARE THE PAYOFF. Because the folds are real, the number of sheet
// thicknesses over any point is a countable thing rather than a texture, and
// it is what makes the shading work: backlit, a one-layer wing tip glows and
// an eight-layer nose is dead opaque. `layersAt` counts them by running the
// fold map backwards and asking how many points of the original sheet arrive
// at this one.
//
// Local frame matches the bird's: +x right, +y forward, +z up, metres, with
// the origin at roughly a third of the way back — where a dart's centre of
// mass sits, with all that folded paper up front, and therefore the point the
// thing should rotate about. Vertices carry, besides position and normal:
//
//   span    signed, 0 at the wing root and +/-1 at the tip. Drives the flex
//           and the trailing-edge flutter, exactly as the bird's does.
//   chord   0 at the leading edge, 1 at the trailing edge.
//   part    keel or wing (see PART), so the two can shade and flex apart.
//   layers  how many thicknesses of paper. 1 to 8. Drives transmission.
//   crease  metres to the nearest fold or cut edge, capped. Drives the
//           hairline of shadow every fold on used paper carries.
//   grime   0 clean, 1 handled. Fingers go on the nose and under the keel.

"use strict";

export const PART = {
  KEEL: 0.0,     // the fuselage: two half-sheets face to face, hanging below
  WING: 1.0,
};

export const FLOATS_PER_VERTEX = 12;   // pos3 normal3 span chord part layers crease grime
export const VERTEX_STRIDE = FLOATS_PER_VERTEX * 4;

// --- the sheet -------------------------------------------------------------

const SHEET_W = 0.210;              // A4 short edge
const SHEET_L = 0.297;              // A4 long edge
const HALF_W = SHEET_W / 2;
const KEEL = 0.018;                 // wing fold, out from the centre crease
const PAPER_THICKNESS = 0.00010;    // 80 gsm, near enough

export const SEMI_SPAN = HALF_W - KEEL;        // 0.087 m
export const LENGTH = SHEET_L;

// Where the local origin sits along the sheet. A dart carries its mass in the
// folded nose, so this is well forward of the midpoint.
const NOSE_Y = 0.100;

// A hand-folded dart is never symmetric, and the eye knows it. One side's
// second fold is a little over 22.5 degrees, the other a little under; the
// keel is 0.7 mm off centre; the wings sit at slightly different dihedral.
// This is the single cheapest thing in the file and it does more for "someone
// folded this" than any amount of surface detail.
const FOLD_DEG = { right: 22.16, left: 22.78 };
const KEEL_M = { right: KEEL - 0.0004, left: KEEL + 0.0003 };
const DIHEDRAL_DEG = { right: 6.4, left: 5.1 };

const DEG = Math.PI / 180.0;
const foldSlope = (side) => Math.tan((side > 0 ? FOLD_DEG.right : FOLD_DEG.left) * DEG);
const keelOf = (side) => (side > 0 ? KEEL_M.right : KEEL_M.left);
const dihedralOf = (side) => (side > 0 ? DIHEDRAL_DEG.right : DIHEDRAL_DEG.left) * DEG;

// --- weathering ------------------------------------------------------------
//
// This dart has been thrown. Nothing here is large — the whole point is that
// each is at or just above the threshold of being noticed.

const BOW_M = 0.0035;               // wings bow; paper does not stay flat
// The left tip took a knock and stayed bent. This wants to be spread over a
// third of the semi-span: eleven millimetres of rise crammed into the outer
// fifth turned the tip edge-on to the viewer and rendered as a dark rolled
// lip stuck on the end of the wing. Paper that has been bent and flown since
// relaxes into a long shallow curl.
const TIP_BEND_M = 0.0075;
const TIP_BEND_FROM = 0.62;         // fraction of span where it starts
const DOG_EAR_M = 0.021;            // size of the curled trailing corner
// Radians it lifts. This wants to stay well under a right angle: curl it far
// enough to lie back on itself and at ninety pixels it stops reading as a
// corner and starts reading as a tube stuck to the wing.
const DOG_EAR_TURN = 1.25;
// Beyond this a point is not near a fold. At the size a flyer is drawn, one
// pixel is about a millimetre of paper, so a six-millimetre cap made every
// crease a five-pixel smudge; the line has to be tighter than the thing it is
// drawn on.
const CREASE_CAP_M = 0.0035;

// --- small helpers ---------------------------------------------------------

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length3 = (a) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a) => {
  const n = length3(a);
  return n < 1e-12 ? [0, 0, 1] : [a[0] / n, a[1] / n, a[2] / n];
};
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (lo, hi, x) => {
  const t = clamp01((x - lo) / (hi - lo));
  return t * t * (3.0 - 2.0 * t);
};

/** Rotate `p` about the line through `a` with unit direction `d`, by `ang`. */
function rotateAboutLine(p, a, d, ang) {
  const v = sub(p, a);
  const c = Math.cos(ang), s = Math.sin(ang);
  const dv = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
  const cr = cross(d, v);
  return [
    a[0] + v[0] * c + cr[0] * s + d[0] * dv * (1 - c),
    a[1] + v[1] * c + cr[1] * s + d[1] * dv * (1 - c),
    a[2] + v[2] * c + cr[2] * s + d[2] * dv * (1 - c),
  ];
}

// --- the fold map ----------------------------------------------------------

/** Half-width of the folded-flat planform at `t`: the leading edge, then the
 *  sheet's own side edge once the fold line has run off it. */
export function halfWidth(t, side = 1) {
  return Math.min(foldSlope(side) * Math.max(t, 0.0), HALF_W);
}

/**
 * How many thicknesses of the original sheet lie over (s, t), for ONE half of
 * the sheet. Both corner folds are reflections about lines through the nose,
 * so a point's preimages are found by undoing each fold or not — four
 * branches — and keeping the ones that land inside the sheet on the side the
 * fold would actually have moved.
 *
 * The keel doubles this, because folding in half puts the two halves face to
 * face; the wings do not, because folding the wings back out separates them
 * again. That is why a dart's nose is eight layers and its wing tip is one.
 */
export function layersAt(s, t, side = 1) {
  const a1 = 45.0 * DEG;                                   // first corner fold
  const a2 = (side > 0 ? FOLD_DEG.right : FOLD_DEG.left) * DEG;
  const r = Math.hypot(s, t);
  if (r < 1e-9) return 4;
  const ang = Math.atan2(s, t);
  let n = 0;
  for (const undoSecond of [false, true]) {
    // Undo the 22.5-degree fold: reflect the direction about that line.
    const mid = undoSecond ? 2 * a2 - ang : ang;
    if (mid > a1 + 1e-12) continue;      // could not have come out of fold 1
    for (const undoFirst of [false, true]) {
      const o = undoFirst ? 2 * a1 - mid : mid;
      // A branch that undoes a fold must land where that fold had material,
      // and one that does not must land where it had none.
      if (undoFirst !== (o > a1 + 1e-12)) continue;
      const so = r * Math.sin(o), to = r * Math.cos(o);
      if (so >= -1e-9 && so <= HALF_W + 1e-9
          && to >= -1e-9 && to <= SHEET_L + 1e-9) n++;
    }
  }
  return n;
}

/**
 * The creases and cut edges, as polylines in the folded-flat (s, t) plane.
 *
 * Rather than do the algebra for where each edge of the sheet ends up, this
 * pushes the edges forward through the same fold map and records where they
 * land. `creaseDistance` then just asks for the nearest one. Sampled rather
 * than solved because the answer feeds a hairline of shading a pixel wide,
 * and a millimetre of error in it is invisible.
 */
function creaseLines(side) {
  const a1 = 45.0 * DEG;
  const a2 = (side > 0 ? FOLD_DEG.right : FOLD_DEG.left) * DEG;
  const forward = (s, t) => {
    let ang = Math.atan2(s, t);
    const r = Math.hypot(s, t);
    if (ang > a1) ang = 2 * a1 - ang;      // first corner fold
    if (ang > a2) ang = 2 * a2 - ang;      // second
    return [r * Math.sin(ang), r * Math.cos(ang)];
  };
  const lines = [];
  const sample = (from, to, n) => {
    const line = [];
    for (let i = 0; i <= n; i++) {
      const f = i / n;
      line.push(forward(from[0] + (to[0] - from[0]) * f,
                        from[1] + (to[1] - from[1]) * f));
    }
    lines.push(line);
  };
  // The sheet's own side edge and its top edge, wherever they end up.
  sample([HALF_W, 0.0], [HALF_W, SHEET_L], 160);
  sample([0.0, 0.0], [HALF_W, 0.0], 60);
  // The first corner fold's crease, which the second fold carries onto the
  // centreline — that is what makes a dart's centreline look doubled.
  sample([0.0, 0.0], [HALF_W, HALF_W], 60);
  return lines;
}

/** Metres to the nearest crease or cut edge, capped at CREASE_CAP_M. */
function creaseDistance(lines, s, t) {
  let best = CREASE_CAP_M;
  for (const line of lines) {
    for (let i = 1; i < line.length; i++) {
      const [ax, ay] = line[i - 1], [bx, by] = line[i];
      const dx = bx - ax, dy = by - ay;
      const len2 = dx * dx + dy * dy;
      let f = len2 < 1e-14 ? 0 : ((s - ax) * dx + (t - ay) * dy) / len2;
      f = clamp01(f);
      const d = Math.hypot(s - (ax + dx * f), t - (ay + dy * f));
      if (d < best) best = d;
    }
  }
  return best;
}

/**
 * Handling grime: where fingers go. The nose, because that is what you hold
 * to throw it, and the underside of the keel, because that is what you pinch.
 */
function grimeAt(s, t) {
  const nose = Math.exp(-Math.pow((t - 0.030) / 0.038, 2));
  const pinch = Math.exp(-Math.pow((t - 0.115) / 0.045, 2))
              * Math.exp(-Math.pow(s / 0.012, 2));
  const smudge = 0.45 * Math.exp(-Math.pow((t - 0.215) / 0.050, 2))
               * Math.exp(-Math.pow((s - 0.062) / 0.030, 2));
  return clamp01(0.85 * nose + 0.7 * pinch + smudge);
}

// --- surfaces --------------------------------------------------------------

/**
 * Emit a parametric surface as a triangle grid with analytic normals.
 *
 * Deliberately the same shape of helper as birdmesh.js's, and deliberately
 * not shared with it: that one pushes ten floats in the bird's order and this
 * one pushes twelve in the dart's, and threading an attribute description
 * through both would cost more to read than the thirty lines it saved.
 *
 * Normals come from differences of the surface function, so a panel that bows
 * reads as one curved sheet. The hard edges — keel to wing, and the two sides
 * of the keel — are hard because they are SEPARATE surfaces here and share no
 * vertices. That is the whole trick, and it is the opposite of the bird,
 * where every normal is smooth on purpose.
 */
function surface(rows, nu, nv, f, attrs, { flip = false } = {}) {
  const grid = [];
  const eps = 1e-4;
  for (let iu = 0; iu <= nu; iu++) {
    const u = iu / nu;
    const line = [];
    for (let iv = 0; iv <= nv; iv++) {
      const v = iv / nv;
      const p = f(u, v);
      let du = sub(f(Math.min(1, u + eps), v), f(Math.max(0, u - eps), v));
      let dv = sub(f(u, Math.min(1, v + eps)), f(u, Math.max(0, v - eps)));
      if (length3(du) < 1e-9) du = sub(f(Math.min(1, u + 4 * eps), v), p);
      if (length3(dv) < 1e-9) dv = sub(f(u, Math.min(1, v + 4 * eps)), p);
      let n = normalize(cross(du, dv));
      if (flip) n = [-n[0], -n[1], -n[2]];
      line.push({ p, n, ...attrs(u, v, p) });
    }
    grid.push(line);
  }
  const push = (a) => {
    rows.push(a.p[0], a.p[1], a.p[2], a.n[0], a.n[1], a.n[2],
              a.span, a.chord, a.part, a.layers, a.crease, a.grime);
  };
  for (let iu = 0; iu < nu; iu++) {
    for (let iv = 0; iv < nv; iv++) {
      const a = grid[iu][iv], b = grid[iu + 1][iv];
      const c = grid[iu + 1][iv + 1], d = grid[iu][iv + 1];
      if (length3(sub(a.p, b.p)) < 1e-9 && length3(sub(d.p, c.p)) < 1e-9) continue;
      push(a); push(b); push(c);
      push(a); push(c); push(d);
    }
  }
}

/** Sheet coordinate t -> local y. The nose leads, as the bird's bill does. */
const yOf = (t) => NOSE_Y - t;

/** Chord fraction: the leading edge at this s sits at t = s / slope. */
function chordAt(s, t, side) {
  const tLead = s / foldSlope(side);
  return clamp01((t - tLead) / Math.max(SHEET_L - tLead, 1e-6));
}

/**
 * One wing: the sheet from the keel fold out to the leading edge, opened to
 * its dihedral, bowed, and — on the left — bent at the tip.
 */
function addWing(rows, side, lines) {
  const k = keelOf(side);
  const dih = dihedralOf(side);
  const cosD = Math.cos(dih), sinD = Math.sin(dih);
  const tStart = k / foldSlope(side);        // where the planform reaches the fold
  const bowSign = side > 0 ? 1.0 : 0.88;     // the two wings did not bow alike

  // Dog-ear: the trailing outboard corner has curled back over itself. Its
  // fold line, in 3D, on this wing only.
  const dogEar = side > 0;
  const cornerS = HALF_W, cornerT = SHEET_L;
  const earA = [(cornerS - DOG_EAR_M - k) * cosD, yOf(cornerT),
                (cornerS - DOG_EAR_M - k) * sinD];
  const earB = [(cornerS - k) * cosD, yOf(cornerT - DOG_EAR_M),
                (cornerS - k) * sinD];
  const earDir = normalize(sub(earB, earA));

  const place = (u, v) => {
    const t = tStart + (SHEET_L - tStart) * u;
    const w = halfWidth(t, side);
    const s = k + (w - k) * v;
    const spanFrac = clamp01((s - k) / SEMI_SPAN);
    const chord = chordAt(s, t, side);

    // Paper does not stay flat: a gentle bow, deepest mid-span and mid-chord.
    const bow = BOW_M * bowSign * Math.pow(spanFrac, 1.4)
              * (0.35 + 0.65 * Math.sin(Math.PI * chord));
    // The left tip took a knock and never came back.
    const bend = side < 0
      ? TIP_BEND_M * Math.pow(smoothstep(TIP_BEND_FROM, 1.0, spanFrac), 1.5)
      : 0.0;

    let p = [
      side * (s - k) * cosD,
      yOf(t),
      (s - k) * sinD + bow + bend,
    ];

    if (dogEar) {
      // How far past the fold line, measured across it in the wing plane:
      // 0 at the corner itself, 1 at the crease.
      const over = (HALF_W - s) / DOG_EAR_M + (SHEET_L - t) / DOG_EAR_M;
      // The turn holds CONSTANT over most of the flap and lets go only in the
      // last quarter. Ramping it across the whole flap instead — which is the
      // obvious thing to write — rotates every point by a different amount
      // and rolls the corner into a tube; a dog-ear is a flat piece of paper
      // lying at an angle, with one crease.
      const turn = DOG_EAR_TURN * (1.0 - smoothstep(0.74, 1.0, over));
      if (turn > 1e-4) p = rotateAboutLine(p, earA, earDir, -turn);
    }
    return p;
  };

  surface(rows, 60, 16, place, (u, v, p) => {
    const t = tStart + (SHEET_L - tStart) * u;
    const w = halfWidth(t, side);
    const s = k + (w - k) * v;
    const spanFrac = clamp01((s - k) / SEMI_SPAN);
    let layers = layersAt(s, t, side);
    // Curled-over paper is one thicker where it lies back on itself.
    if (dogEar && (HALF_W - s) / DOG_EAR_M + (SHEET_L - t) / DOG_EAR_M < 1.0) {
      layers += 1;
    }
    return {
      span: side * spanFrac,
      chord: chordAt(s, t, side),
      part: PART.WING,
      layers,
      crease: creaseDistance(lines, s, t),
      grime: grimeAt(s, t),
    };
  }, { flip: side < 0 });
}

/**
 * One half of the keel: the strip inboard of the wing fold, hanging below.
 *
 * The two halves are separate surfaces a tenth of a millimetre apart — they
 * are separate paper, and pressed face to face they are exactly the two sheet
 * thicknesses that the bottom edge of a real dart shows.
 */
function addKeel(rows, side, lines) {
  const k = keelOf(side);
  surface(rows, 60, 6, (u, v) => {
    const t = SHEET_L * u;
    const top = Math.min(k, halfWidth(t, side));
    const s = top * v;
    return [
      side * PAPER_THICKNESS * 0.5,
      yOf(t),
      s - k,
    ];
  }, (u, v) => {
    const t = SHEET_L * u;
    const top = Math.min(k, halfWidth(t, side));
    const s = top * v;
    return {
      span: 0.0,
      chord: chordAt(s, t, side),
      part: PART.KEEL,
      // Face to face, so both halves' layers are over every point of it.
      layers: 2 * layersAt(s, t, side),
      crease: creaseDistance(lines, s, t),
      grime: grimeAt(s, t),
    };
  }, { flip: side < 0 });
}

// --- assembly --------------------------------------------------------------

/**
 * Fold the dart. Returns `{data, vertexCount, stride, attributes}` with `data`
 * a Float32Array of interleaved vertices and no index buffer, matching the
 * bird; `attributes` is the vertex layout its pipeline needs, because this
 * mesh is wider than the bird's and the two share a base class.
 *
 * `scale` multiplies every position. 1.0 is a real sheet of A4: 174 mm across
 * the wings, 297 mm nose to tail.
 */
export function buildDartMesh({ scale = 1.0 } = {}) {
  const rows = [];
  for (const side of [1, -1]) {
    const lines = creaseLines(side);
    addKeel(rows, side, lines);
    addWing(rows, side, lines);
  }

  const data = new Float32Array(rows);
  if (scale !== 1.0) {
    for (let i = 0; i < data.length; i += FLOATS_PER_VERTEX) {
      data[i] *= scale; data[i + 1] *= scale; data[i + 2] *= scale;
    }
  }
  const vertexCount = data.length / FLOATS_PER_VERTEX;
  if (vertexCount % 3 !== 0) {
    throw new Error(
      `The dart mesh came out ${vertexCount} vertices, which is not a whole ` +
      "number of triangles.");
  }
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      throw new Error(
        `The dart mesh contains a non-finite value at float ${i}. A surface ` +
        "derivative degenerated; look for a pinch point.");
    }
  }
  return { data, vertexCount, stride: VERTEX_STRIDE, attributes: ATTRIBUTES };
}

/** The vertex layout, in the order `surface` pushes it. */
export const ATTRIBUTES = [
  { format: "float32x3", offset: 0, shaderLocation: 0 },    // pos
  { format: "float32x3", offset: 12, shaderLocation: 1 },   // normal
  { format: "float32", offset: 24, shaderLocation: 2 },     // span
  { format: "float32", offset: 28, shaderLocation: 3 },     // chord
  { format: "float32", offset: 32, shaderLocation: 4 },     // part
  { format: "float32", offset: 36, shaderLocation: 5 },     // layers
  { format: "float32", offset: 40, shaderLocation: 6 },     // crease
  { format: "float32", offset: 44, shaderLocation: 7 },     // grime
];

/** The mesh's bounding box, for placement checks and tests. */
export function meshBounds(data) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < data.length; i += FLOATS_PER_VERTEX) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], data[i + k]);
      hi[k] = Math.max(hi[k], data[i + k]);
    }
  }
  return { lo, hi };
}
