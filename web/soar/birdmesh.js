// The bird's geometry: a common swift (Apus apus), built from anatomy.
//
// The previous bird was sixteen flat triangles at 1.8 m of wingspan — a
// swift the size of an albatross, faceted, hanging a quarter of the way
// across the screen. This is the real animal at its real size: 40 cm span,
// 17 cm nose to tail, flown close enough to read.
//
// What makes a bird look like a bird at a hundred pixels is not polygon
// count, it is three things:
//
//   - The scythe. A swift's arm is very short and its hand very long — the
//     hand is about two and a half times the arm — which is why the wing
//     looks like a crescent rather than a plank. Get that ratio wrong and no
//     amount of shading rescues it.
//   - Separated primaries. The outer wing is ten individual feathers, and
//     near the tip they are visibly apart, with sky between them. A smooth
//     swept sheet reads as a paper dart every time.
//   - Smooth normals with real camber. Every surface here is generated
//     parametrically and differentiated for its normal, so light sweeps
//     across the wing instead of stepping from facet to facet.
//
// Local frame: +x right, +y forward, +z up. Metres. The shoulder line — the
// flap pivot — is at z = 0. Vertices carry, besides position and normal:
//
//   span     signed, 0 on the centreline and +/-1 at the wingtip; drives the
//            flap bend.
//   chord    0 at a surface's leading edge, 1 at its trailing edge; drives
//            how much light passes through and where the edge softens.
//   part     which structure this is (see PART), so the vertex stage can
//            articulate the hand separately from the arm and the fragment
//            stage can shade a feather differently from a body.
//   feather  0..1 across a fan, constant along one feather. Gives each
//            primary and rectrix its own identity for shading.

"use strict";

export const PART = {
  BODY: 0.0,
  ARM: 1.0,        // secondaries: the membrane from shoulder to wrist
  PRIMARY: 2.0,    // the ten feathers of the hand
  TAIL: 3.0,
};

export const FLOATS_PER_VERTEX = 10;      // pos3 normal3 span chord part feather
export const VERTEX_STRIDE = FLOATS_PER_VERTEX * 4;

// --- the animal ------------------------------------------------------------
//
// Measurements are a common swift's, in metres. Wingspan 0.40, body 0.168
// bill to tail base, tail streamers a further 0.07. Slim: the body is barely
// 3 cm across.

export const SEMI_SPAN = 0.200;

const BILL_TIP = [0.0, 0.062, -0.002];
const SHOULDER = [0.011, 0.016, 0.004];
const WRIST = [0.077, 0.008, 0.012];
const WING_TIP = [0.200, -0.062, -0.004];
const TAIL_BASE = [0.0, -0.048, 0.0];

const BODY_MAX_RADIUS = 0.0122;
const PRIMARY_COUNT = 10;
const RECTRIX_COUNT = 10;

// --- small vector helpers --------------------------------------------------

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length = (a) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a) => {
  const n = length(a);
  return n < 1e-12 ? [0, 0, 1] : [a[0] / n, a[1] / n, a[2] / n];
};
const lerp3 = (a, b, t) => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smoothstep = (lo, hi, x) => {
  const t = clamp01((x - lo) / (hi - lo));
  return t * t * (3.0 - 2.0 * t);
};

/**
 * Emit a parametric surface as a triangle grid with analytic normals.
 *
 * The normal comes from central differences of the surface function rather
 * than from face cross-products, which is what keeps a wing reading as one
 * curved sheet. Degenerate rows — a pole where the surface pinches to a point
 * — fall back to a one-sided difference rather than producing a zero normal.
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
      if (length(du) < 1e-9) du = sub(f(Math.min(1, u + 4 * eps), v), p);
      if (length(dv) < 1e-9) dv = sub(f(u, Math.min(1, v + 4 * eps)), p);
      let n = normalize(cross(du, dv));
      if (flip) n = mul(n, -1);
      line.push({ p, n, ...attrs(u, v, p) });
    }
    grid.push(line);
  }

  const push = (a) => {
    rows.push(a.p[0], a.p[1], a.p[2], a.n[0], a.n[1], a.n[2],
              a.span, a.chord, a.part, a.feather);
  };
  for (let iu = 0; iu < nu; iu++) {
    for (let iv = 0; iv < nv; iv++) {
      const a = grid[iu][iv], b = grid[iu + 1][iv];
      const c = grid[iu + 1][iv + 1], d = grid[iu][iv + 1];
      // Skip fully degenerate quads at a pinch point.
      if (length(sub(a.p, b.p)) < 1e-9 && length(sub(d.p, c.p)) < 1e-9) continue;
      push(a); push(b); push(c);
      push(a); push(c); push(d);
    }
  }
}

// --- body ------------------------------------------------------------------

/**
 * The body: a lofted fusiform with a flat wide head and a very small bill.
 *
 * A swift is a torpedo with no neck. The head bump is deliberate and small —
 * exaggerate it and the silhouette turns into a duck.
 */
function bodyProfile(t) {
  // Zero at both ends so the bill comes to a point and the tail base pinches
  // into the rectrices; fattest a third of the way back.
  const base = Math.pow(Math.sin(Math.PI * Math.pow(clamp01(t), 0.62)), 0.75);
  const head = 0.20 * Math.exp(-Math.pow((t - 0.150) / 0.070, 2));
  // The bill is a spike, not a taper of the head.
  const bill = t < 0.075 ? 0.55 : 1.0;
  return (base + head) * bill;
}

function bodyCentre(t) {
  const y = BILL_TIP[1] + (TAIL_BASE[1] - BILL_TIP[1]) * t;
  // A shallow droop toward the tail, and the head carried a touch high.
  const z = 0.004 + 0.010 * Math.exp(-Math.pow((t - 0.16) / 0.16, 2))
          - 0.008 * Math.pow(t, 1.8);
  return [0.0, y, z];
}

function addBody(rows) {
  surface(rows, 34, 16, (u, v) => {
    const c = bodyCentre(u);
    const r = BODY_MAX_RADIUS * bodyProfile(u);
    const a = v * 2.0 * Math.PI;
    // Deeper than wide, and flat-topped at the head — a swift's crown is
    // noticeably flattened, which is half of why the head reads as small.
    const flat = 1.0 - 0.22 * Math.exp(-Math.pow((u - 0.16) / 0.10, 2));
    return [
      c[0] + r * 0.92 * Math.sin(a),
      c[1],
      c[2] + r * 1.10 * Math.cos(a) * (Math.cos(a) > 0 ? flat : 1.0),
    ];
  }, (u, v) => ({
    span: 0.0,
    // Chord runs belly (0) to back (1) here, so the shader can put the pale
    // throat on the underside.
    chord: 0.5 + 0.5 * Math.cos(v * 2.0 * Math.PI),
    part: PART.BODY,
    feather: u,
  }));
}

// --- wing ------------------------------------------------------------------
//
// ONE leading edge, from the shoulder all the way to the wingtip.
//
// The first version of this built the arm and the hand as two separate
// shapes that met at the wrist, and it showed: a rectangular slab of arm
// stopping dead, a fan of primaries starting behind and below it, and a
// visible step in both the leading and the trailing edge. The wings read as
// bolted on, which is exactly what they were.
//
// A real wing has no such seam. The leading edge is a single continuous
// curve; the covert membrane runs out over the base of the hand and hides
// every quill; the primaries emerge from underneath it and their exposed
// lengths make the outer half of the silhouette. So that is how it is built
// now: one curve, one membrane over the inner two thirds, and ten feathers
// rooted well beneath it.

/**
 * The wing's leading edge, shoulder (s = 0) to wingtip (s = 1).
 *
 * The crescent lives in this curve. The sweep accelerates outboard — the
 * exponent — and the arch peaks near the wrist and falls away to a tip held
 * slightly low, which is what stops the thin outer wing from ever going
 * exactly edge-on and rasterizing into dots.
 */
function wingLeading(s) {
  const t = clamp01(s);
  return [
    SHOULDER[0] + (WING_TIP[0] - SHOULDER[0]) * t,
    0.030 - 0.092 * Math.pow(t, 1.35) + 0.010 * Math.sin(Math.PI * Math.pow(t, 0.8)),
    0.004 + 0.016 * Math.sin(Math.PI * Math.min(1.0, t * 0.85)) - 0.008 * t * t,
  ];
}

// Where the covert membrane gives out and the primaries take over the
// silhouette. Not a joint — the membrane tapers to nothing here.
const MEMBRANE_END = 0.60;

/**
 * The membrane: secondaries over the arm, coverts over the base of the hand,
 * as one sheet.
 *
 * Its inner edge starts inside the body so the junction is buried rather than
 * abutted, and its outer end pinches to a point instead of being cut off, so
 * there is no step for the eye to catch.
 */
function membraneChord(s) {
  const t = clamp01(s / MEMBRANE_END);
  const taper = 0.047 * (1.0 - 0.62 * Math.pow(t, 1.25));
  // Pinch out over the last fifth rather than ending on an edge.
  return taper * Math.pow(1.0 - smoothstep(0.78, 1.0, t), 0.65);
}

function addMembrane(rows, side) {
  surface(rows, 26, 8, (u, v) => {
    const s = u * MEMBRANE_END;
    const le = wingLeading(s);
    // Scallops are the secondary tips, and they only exist over the arm —
    // out on the hand the coverts are smooth and the primaries make the edge.
    const scallop = 0.0016 * Math.sin(u * Math.PI * 7.0) * v
                  * (1.0 - smoothstep(0.35, 0.6, u));
    const chord = membraneChord(s) + scallop;
    const camber = -0.0055 * (membraneChord(s) / 0.047)
                 * Math.sin(Math.PI * Math.pow(v, 0.8));
    // Reach inboard of the shoulder at the root so the body swallows the join.
    const inboard = 1.0 - 0.55 * (1.0 - smoothstep(0.0, 0.10, u));
    return [
      side * le[0] * inboard,
      le[1] - chord * v,
      le[2] + camber,
    ];
  }, (u, v, p) => ({
    // Span is read off the vertex's own x, never invented per structure. The
    // vertex stage bends by span, so if the membrane and the primaries lying
    // over it disagreed by even a little they would visibly come apart at the
    // top of every stroke.
    span: p[0] / SEMI_SPAN,
    chord: v,
    part: PART.ARM,
    feather: u,
  }), { flip: side < 0 });
}

/**
 * The ten primaries.
 *
 * Rooted along the leading curve well inboard of where they become visible,
 * so their quills sit under the membrane exactly as a bird's do. Their tips
 * walk around the outer edge, so the gaps open toward the wingtip and close
 * to nothing where the membrane still covers them.
 */
function addPrimaries(rows, side) {
  // The innermost primary reaches just past where the membrane pinches out,
  // so the two silhouettes overlap instead of meeting.
  const innerTip = [0.108, -0.044, 0.004];
  const outerTip = WING_TIP;

  for (let i = 0; i < PRIMARY_COUNT; i++) {
    const f = i / (PRIMARY_COUNT - 1);
    // Quill roots on the leading curve, from a third of the way out — deep
    // under the membrane — to a little past halfway.
    const root = add(wingLeading(0.30 + 0.22 * Math.pow(f, 1.1)),
                     [0.0, -0.006 - 0.008 * f, -0.0012]);
    // Tips fan around the outer edge, with a slight outward bulge so the
    // silhouette is convex rather than a straight-edged wedge.
    const g = Math.pow(f, 1.22);
    const straight = lerp3(innerTip, outerTip, g);
    const along = normalize(sub(outerTip, innerTip));
    const bulge = normalize(cross(along, [0, 0, 1]));
    const tip = add(straight, mul(bulge, -0.010 * Math.sin(Math.PI * g)));

    const spine = (t) => {
      const p = lerp3(root, tip, t);
      // Each feather bows gently: stiff at the quill, curving aft and down.
      const droop = 0.0055 * (0.35 + 0.65 * f) * Math.sin(Math.PI * Math.pow(t, 0.85));
      return [p[0], p[1] - 0.004 * Math.sin(Math.PI * t), p[2] - droop];
    };
    const halfWidth = (t) =>
      0.0082 * (1.0 - 0.42 * f) *
      Math.pow(Math.sin(Math.PI * Math.pow(clamp01(t), 0.72)), 0.7);

    // The width direction: across the feather, in the wing's plane.
    const axis = normalize(sub(tip, root));
    const across = normalize(cross([0, 0, 1], axis));

    surface(rows, 12, 3, (u, v) => {
      const c = spine(u);
      const w = halfWidth(u);
      // The vane is asymmetric — the outer web is narrower than the inner,
      // which is what makes a flight feather a flight feather.
      const offset = (v - 0.38) * 2.0 * w;
      const cup = -0.0022 * w / 0.008 * Math.sin(Math.PI * v);
      return [
        side * (c[0] + across[0] * offset),
        c[1] + across[1] * offset,
        c[2] + across[2] * offset + cup,
      ];
    }, (u, v, p) => ({
      span: p[0] / SEMI_SPAN,     // see addMembrane: never invented per part
      chord: v,
      part: PART.PRIMARY,
      feather: f,
    }), { flip: side < 0 });

  }
}

// --- tail ------------------------------------------------------------------

/**
 * Ten rectrices, deeply forked.
 *
 * The fork is the swift's other signature after the wing, and it is a fork,
 * not a notch — the outer feathers are half again as long as the inner ones.
 */
function addTail(rows) {
  const half = (RECTRIX_COUNT - 1) / 2;
  for (let i = 0; i < RECTRIX_COUNT; i++) {
    const signed = (i - half) / half;              // -1 .. 1
    const side = signed < 0 ? -1 : 1;
    const f = Math.abs(signed);
    const root = [side * (0.0018 + 0.0085 * f), TAIL_BASE[1] + 0.004 * f, 0.0];
    const len = 0.022 + 0.034 * Math.pow(f, 1.15);
    const tip = [
      side * (0.006 + 0.026 * Math.pow(f, 1.1)),
      TAIL_BASE[1] - len,
      -0.003 - 0.003 * f,
    ];
    const spine = (t) => lerp3(root, tip, t);
    const axis = normalize(sub(tip, root));
    const across = normalize(cross([0, 0, 1], axis));
    const halfWidth = (t) =>
      0.0078 * Math.pow(Math.sin(Math.PI * Math.pow(clamp01(t), 0.62)), 0.6);

    surface(rows, 8, 2, (u, v) => {
      const c = spine(u);
      const w = halfWidth(u);
      const offset = (v - 0.42) * 2.0 * w;
      return [
        c[0] + across[0] * offset,
        c[1] + across[1] * offset,
        c[2] + across[2] * offset,
      ];
    }, (u, v) => ({
      span: 0.0,
      chord: v,
      part: PART.TAIL,
      feather: 0.5 + 0.5 * signed,
    }), { flip: side < 0 });
  }
}

// --- assembly --------------------------------------------------------------

/**
 * Build the swift. Returns `{data, vertexCount, stride}` with `data` a
 * Float32Array of interleaved vertices, no index buffer — the mesh is a few
 * thousand triangles and an index buffer would save less than it costs in
 * moving parts.
 *
 * `scale` multiplies every position. 1.0 is a real swift: 40 cm across.
 */
export const ATTRIBUTES = [
  { format: "float32x3", offset: 0, shaderLocation: 0 },    // pos
  { format: "float32x3", offset: 12, shaderLocation: 1 },   // normal
  { format: "float32", offset: 24, shaderLocation: 2 },     // span
  { format: "float32", offset: 28, shaderLocation: 3 },     // chord
  { format: "float32", offset: 32, shaderLocation: 4 },     // part
  { format: "float32", offset: 36, shaderLocation: 5 },     // feather
];

export function buildBirdMesh({ scale = 1.0 } = {}) {
  const rows = [];
  addBody(rows);
  for (const side of [1, -1]) {
    addMembrane(rows, side);
    addPrimaries(rows, side);
  }
  addTail(rows);

  const data = new Float32Array(rows);
  if (scale !== 1.0) {
    for (let i = 0; i < data.length; i += FLOATS_PER_VERTEX) {
      data[i] *= scale; data[i + 1] *= scale; data[i + 2] *= scale;
    }
  }
  const vertexCount = data.length / FLOATS_PER_VERTEX;
  if (vertexCount % 3 !== 0) {
    throw new Error(
      `The bird mesh came out ${vertexCount} vertices, which is not a whole ` +
      "number of triangles.");
  }
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      throw new Error(
        `The bird mesh contains a non-finite value at float ${i}. A surface ` +
        "derivative degenerated; look for a pinch point.");
    }
  }
  return { data, vertexCount, stride: VERTEX_STRIDE, attributes: ATTRIBUTES };
}

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
