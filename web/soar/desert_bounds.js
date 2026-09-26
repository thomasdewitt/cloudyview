// The desert's bounds pyramid, built in the browser from the tile's own
// height mips: the local upper bounds raymarch.wgsl desert_terrain_trace
// steps on. cloudyview/desert_bounds.py is the spec (its docstring says what
// each channel bounds and why); this builds the same bytes, and
// tests/test_desert_bounds.py pins the two equal on the shipped tile.
//
// Every value is a max (or min) of texels over a window, or one IEEE
// operation on such maxima, so any order of forming them gives the same
// bits. The forms here are the memory-lean ones: each level dilated once and
// pooled a level at a time, the band's lattice walked a row at a time
// (Python forms it whole).

export const BASE_LEVEL = 1;
export const LAYERS = 2;
const BAND_MIPS = [2, 3];

const F64 = new Float64Array(1);
const U32 = new Uint32Array(F64.buffer);

/** The float16 bits of the smallest float16 at or above `x`. */
export function ceilHalfBits(x) {
  if (!(x > -65504.0) || x > 65504.0) {
    throw new Error(`desert bounds: ${x} is outside float16's range.`);
  }
  if (x === 0) return 0;
  const neg = x < 0;
  const y = neg ? -x : x;
  F64[0] = y;
  let e = ((U32[1] >>> 20) & 0x7ff) - 1023;       // y is normal in float64
  // Toward +infinity: a positive magnitude rounds up, a negative one down.
  const round = neg ? Math.floor : Math.ceil;
  let bits;
  if (e < -14) {
    bits = round(y * 16777216);                  // subnormal (1024: the least normal)
  } else {
    let m = round((y * 2 ** -e - 1) * 1024);
    if (m === 1024) {
      m = 0;
      e += 1;
    }
    bits = ((e + 15) << 10) | m;
  }
  if (neg) return bits === 0 ? 0 : 0x8000 | bits;
  return bits;
}

// 3 x 3 periodic max (or min) of an n x n array.
function dilate(a, n, useMax) {
  const rows = new a.constructor(n * n);
  const out = new a.constructor(n * n);
  for (let j = 0; j < n; j++) {
    const up = ((j + n - 1) % n) * n;
    const dn = ((j + 1) % n) * n;
    const at = j * n;
    for (let i = 0; i < n; i++) {
      const u = a[up + i], c = a[at + i], d = a[dn + i];
      rows[at + i] = useMax ? (u > c ? (u > d ? u : d) : (c > d ? c : d))
                            : (u < c ? (u < d ? u : d) : (c < d ? c : d));
    }
  }
  for (let j = 0; j < n; j++) {
    const at = j * n;
    for (let i = 0; i < n; i++) {
      const l = rows[at + (i === 0 ? n - 1 : i - 1)];
      const c = rows[at + i];
      const r = rows[at + (i === n - 1 ? 0 : i + 1)];
      out[at + i] = useMax ? (l > c ? (l > r ? l : r) : (c > r ? c : r))
                           : (l < c ? (l < r ? l : r) : (c < r ? c : r));
    }
  }
  return out;
}

// One level down: each 2 x 2 block's max (or min).
function pool(a, n, useMax) {
  const f = useMax ? Math.max : Math.min;
  const m = n >> 1;
  const out = new a.constructor(m * m);
  for (let j = 0; j < m; j++) {
    for (let i = 0; i < m; i++) {
      const at = 2 * j * n + 2 * i;
      out[j * m + i] = f(f(a[at], a[at + 1]), f(a[at + n], a[at + n + 1]));
    }
  }
  return out;
}

// A coarser level's array repeated onto an m x m grid.
function repeat(a, n, m) {
  const g = m / n;
  const out = new a.constructor(m * m);
  for (let j = 0; j < m; j++) {
    const row = Math.floor(j / g) * n;
    for (let i = 0; i < m; i++) out[j * m + i] = a[row + Math.floor(i / g)];
  }
  return out;
}

// P: each bilinear patch's gradient bound (centres i..i+1, j..j+1).
function patchSlopes(h, n, cell) {
  const out = new Float64Array(n * n);
  const dx = new Float64Array(n * n);
  const dy = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const v = h[j * n + i];
      dx[j * n + i] = Math.abs(h[j * n + (i + 1) % n] - v);
      dy[j * n + i] = Math.abs(h[((j + 1) % n) * n + i] - v);
    }
  }
  for (let j = 0; j < n; j++) {
    const dn = ((j + 1) % n) * n;
    for (let i = 0; i < n; i++) {
      const x = Math.max(dx[j * n + i], dx[dn + i]);
      const y = Math.max(dy[j * n + i], dy[j * n + (i + 1) % n]);
      out[j * n + i] = Math.sqrt(x * x + y * y) / cell;
    }
  }
  return out;
}

// For each level, win(q, 0) on the grids t = BASE..mips-1, a level at a time:
// yields [t, c0 per level] with c0[l] on t's grid.
function* cellWindows(q, sides, mips, useMax, levels) {
  const dil = levels.map((l) => dilate(q[l], sides[l], useMax));
  const cur = dil.slice();
  const curSide = levels.map((l) => sides[l]);
  for (let t = BASE_LEVEL; t < mips; t++) {
    const m = sides[t];
    const c0 = {};
    levels.forEach((l, idx) => {
      while (curSide[idx] > m) {
        cur[idx] = pool(cur[idx], curSide[idx], useMax);
        curSide[idx] >>= 1;
      }
      c0[l] = curSide[idx] === m ? cur[idx] : repeat(dil[idx], sides[l], m);
    });
    yield [t, c0];
  }
}

// The band's lattice on one axis: the centre lines of level 0 and the
// band's mips, sorted, within one tile period.
function lattice(n, cell) {
  const lines = new Set();
  for (const l of [0, ...BAND_MIPS]) {
    const s = cell * (1 << l);
    for (let i = 0; i < n >> l; i++) lines.add((i + 0.5) * s);
  }
  return Float64Array.from([...lines].sort((a, b) => a - b));
}

// The shader's bilinear read, split by axis: per lattice line, the texel
// index and the fraction (the same float64 operations as Python's).
function axisTaps(xs, n, texel) {
  const i0 = new Int32Array(xs.length);
  const i1 = new Int32Array(xs.length);
  const fr = new Float64Array(xs.length);
  for (let a = 0; a < xs.length; a++) {
    const g = xs[a] / texel - 0.5;
    const i = Math.floor(g);
    fr[a] = g - i;
    i0[a] = ((i % n) + n) % n;
    i1[a] = (i0[a] + 1) % n;
  }
  return { i0, i1, fr };
}

function bilinearRow(level, n, taps, j, out) {
  const { i0, i1, fr } = taps.x;
  const fy = taps.y.fr[j];
  const r0 = taps.y.i0[j] * n;
  const r1 = taps.y.i1[j] * n;
  for (let a = 0; a < out.length; a++) {
    const fx = fr[a];
    out[a] = (level[r0 + i0[a]] * (1.0 - fx) + level[r0 + i1[a]] * fx) * (1.0 - fy)
           + (level[r1 + i0[a]] * (1.0 - fx) + level[r1 + i1[a]] * fx) * fy;
  }
}

// BH, BA and GB of every sub-patch, pooled onto the base grid (a sub-patch
// counts toward every cell its closed rectangle touches).
function bandOnBase(h, sides, cell) {
  const n = sides[0];
  const extent = n * cell;
  const xs = lattice(n, cell);
  const L = xs.length;
  const widths = new Float64Array(L);
  for (let a = 0; a < L; a++) widths[a] = (a + 1 < L ? xs[a + 1] : xs[0] + extent) - xs[a];
  const mb = sides[BASE_LEVEL];
  const s = extent / mb;
  const lo = new Int32Array(L);
  const hi = new Int32Array(L);
  for (let a = 0; a < L; a++) {
    lo[a] = Math.floor(xs[a] / s) % mb;
    hi[a] = Math.floor((a + 1 < L ? xs[a + 1] : xs[0] + extent) / s) % mb;
  }
  const taps = [0, ...BAND_MIPS].map((l) => {
    const t = axisTaps(xs, sides[l], cell * (1 << l));
    return { x: t, y: t };
  });
  const row = (j) => {
    const v = [0, 1, 2].map(() => new Float64Array(L));
    [0, ...BAND_MIPS].forEach((l, idx) => bilinearRow(h[l], sides[l], taps[idx], j, v[idx]));
    const [t0, b2, b3] = v;
    const bh = new Float64Array(L);
    const ba = new Float64Array(L);
    for (let a = 0; a < L; a++) {
      bh[a] = t0[a] - Math.min(b2[a], b3[a]);
      ba[a] = Math.max(bh[a], Math.max(b2[a], b3[a]) - t0[a]);
    }
    return { t0, b2, b3, bh, ba };
  };
  const BH = new Float64Array(mb * mb).fill(-Infinity);
  const BA = new Float64Array(mb * mb).fill(-Infinity);
  const GB = new Float64Array(mb * mb).fill(-Infinity);
  const put = (at, bh, ba, gb) => {
    if (bh > BH[at]) BH[at] = bh;
    if (ba > BA[at]) BA[at] = ba;
    if (gb > GB[at]) GB[at] = gb;
  };
  // |grad band_w| over the sub-patch, for w the mip-2 or mip-3 read.
  const slope = (ct, cw, nt, nw, a, a1, dxw, dyw) => {
    const c0 = ct[a] - cw[a], c1 = ct[a1] - cw[a1];
    const n0 = nt[a] - nw[a], n1 = nt[a1] - nw[a1];
    const ddx = Math.max(Math.abs(c1 - c0) / dxw, Math.abs(n1 - n0) / dxw);
    const ddy = Math.max(Math.abs(n0 - c0) / dyw, Math.abs(n1 - c1) / dyw);
    return Math.sqrt(ddx * ddx + ddy * ddy);
  };
  const first = row(0);
  let cur = first;
  for (let b = 0; b < L; b++) {
    const nxt = b + 1 < L ? row(b + 1) : first;
    const dyw = widths[b];
    const r0 = lo[b] * mb, r1 = hi[b] * mb;
    for (let a = 0; a < L; a++) {
      const a1 = a + 1 < L ? a + 1 : 0;
      const bh = Math.max(Math.max(cur.bh[a], cur.bh[a1]),
                          Math.max(nxt.bh[a], nxt.bh[a1]));
      const ba = Math.max(Math.max(cur.ba[a], cur.ba[a1]),
                          Math.max(nxt.ba[a], nxt.ba[a1]));
      const gb = Math.max(
        Math.max(0, slope(cur.t0, cur.b2, nxt.t0, nxt.b2, a, a1, widths[a], dyw)),
        slope(cur.t0, cur.b3, nxt.t0, nxt.b3, a, a1, widths[a], dyw));
      const c0 = lo[a], c1 = hi[a];
      put(r0 + c0, bh, ba, gb);
      if (c1 !== c0) put(r0 + c1, bh, ba, gb);
      if (r1 !== r0) {
        put(r1 + c0, bh, ba, gb);
        if (c1 !== c0) put(r1 + c1, bh, ba, gb);
      }
    }
    cur = nxt;
  }
  return { BH, BA, GB };
}

/**
 * The pyramid from the tile's height mips: `levels[l]` is level l's height
 * channel (n_l x n_l, row-major, exactly as the float16 texels), `cellM` its
 * level-0 texel. Returns one Uint16Array of float16 bits per pyramid level,
 * laid out (layer, row, column, channel) — writeTexture's order for a
 * (m, m, LAYERS) rgba16float array level.
 */
export function buildDesertBounds(levels, cellM) {
  const mips = levels.length;
  const sides = levels.map((a) => Math.round(Math.sqrt(a.length)));
  sides.forEach((s, l) => {
    if (s !== Math.max(1, sides[0] >> l) || s * s !== levels[l].length) {
      throw new Error(`desert bounds: tile level ${l} is not ${Math.max(1, sides[0] >> l)} square.`);
    }
  });
  if (mips <= Math.max(...BAND_MIPS) || mips <= BASE_LEVEL) {
    throw new Error(`desert bounds: the tile has ${mips} mips; the pyramid needs ${Math.max(...BAND_MIPS) + 1}.`);
  }
  // float16 heights are exact in float32, and every operation on them below
  // is formed in float64 all the same.
  const h = levels.map((a) => Float32Array.from(a));
  const p = h.map((a, l) => patchSlopes(a, sides[l], cellM * (1 << l)));
  const all = [...Array(mips).keys()];
  const hiW = cellWindows(h, sides, mips, true, all);
  const loW = cellWindows(h, sides, mips, false, all);
  const pW = cellWindows(p, sides, mips, true, all);
  let band = bandOnBase(h, sides, cellM);
  const out = [];
  for (let t = BASE_LEVEL; t < mips; t++) {
    const [, hi0] = hiW.next().value;
    const [, lo0] = loW.next().value;
    const [, p0] = pW.next().value;
    const m = sides[t];
    if (t > BASE_LEVEL) {
      band = { BH: pool(band.BH, m << 1, true), BA: pool(band.BA, m << 1, true),
               GB: pool(band.GB, m << 1, true) };
    }
    // A max over levels commutes with the dilation onto cells I-1..I+1, so
    // it is taken first and the dilation once.
    const served = all.filter((l) => l <= t + 1);
    const top = new Float64Array(m * m).fill(-Infinity);
    const low = new Float64Array(m * m).fill(Infinity);
    const steep = new Float64Array(m * m).fill(-Infinity);
    for (const l of served) {
      for (let c = 0; c < m * m; c++) {
        if (hi0[l][c] > top[c]) top[c] = hi0[l][c];
        if (lo0[l][c] < low[c]) low[c] = lo0[l][c];
        if (p0[l][c] > steep[c]) steep[c] = p0[l][c];
      }
    }
    const b23 = new Float64Array(m * m);
    for (let c = 0; c < m * m; c++) {
      b23[c] = Math.max(p0[BAND_MIPS[0]][c], p0[BAND_MIPS[1]][c]);
    }
    const top1 = dilate(top, m, true);
    const low1 = dilate(low, m, false);
    const layer0 = [top1, dilate(steep, m, true), null, top];
    // BH never below 0 (desert_bounds.py: the fade falls to 0 along a ray).
    const layer1 = [dilate(band.BH, m, true).map((v) => Math.max(v, 0)),
                    dilate(band.GB, m, true),
                    dilate(band.BA, m, true), dilate(b23, m, true)];
    const bits = new Uint16Array(LAYERS * m * m * 4);
    const l1 = m * m * 4;
    for (let c = 0; c < m * m; c++) {
      bits[c * 4] = ceilHalfBits(layer0[0][c]);
      bits[c * 4 + 1] = ceilHalfBits(layer0[1][c]);
      bits[c * 4 + 2] = ceilHalfBits(top1[c] - low1[c]);
      bits[c * 4 + 3] = ceilHalfBits(layer0[3][c]);
      for (let k = 0; k < 4; k++) bits[l1 + c * 4 + k] = ceilHalfBits(layer1[k][c]);
    }
    out.push(bits);
  }
  return out;
}

/**
 * buildDesertBounds in a module Worker (desert_bounds_worker.js), off the
 * main thread: on the shipped tile it is over a second of arithmetic, and on
 * the main thread that second froze the flight at the first switch to the
 * desert (Thomas, 2026-09-25). Resolves to the very same Uint16Arrays — the
 * worker runs this module's own buildDesertBounds (tests/test_desert_bounds.py
 * pins the worker's bytes to Python's). `levels` are transferred, so the
 * caller must not read them afterwards.
 */
export function buildDesertBoundsInWorker(levels, cellM) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./desert_bounds_worker.js",
                                      import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      worker.terminate();
      if (e.data.error) reject(new Error(`desert bounds: ${e.data.error}`));
      else resolve(e.data.out);
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`desert bounds worker: ${e.message ?? "failed"}`));
    };
    worker.onmessageerror = () => {
      worker.terminate();
      reject(new Error("desert bounds worker: its reply could not be read."));
    };
    worker.postMessage({ levels, cellM }, levels.map((a) => a.buffer));
  });
}
