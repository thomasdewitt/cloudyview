// IEEE binary16 conversion, used by everything that fills an r16float
// texture — today that is the ingest worker.
//
// It lived inside ingest/worker.js until the conversion had to be importable
// outside a Worker, so a node test could run the browser's own rounding
// against numpy's.

"use strict";

// --- fp16 ------------------------------------------------------------------

export const HAS_F16 = typeof Float16Array !== "undefined";
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/**
 * IEEE binary32 to binary16, round-to-nearest-EVEN — the mode numpy and the
 * native Float16Array both use. An earlier version rounded half away from
 * zero, which diverged systematically.
 *
 * Only reached on browsers with WebGPU but without Float16Array (Chrome
 * 113-134; Firefox has had it since well before its WebGPU release), so this
 * is close to dead code already.
 *
 * Residual: measured against the native path over 137k values it differs on
 * 4 of them, always by one fp16 ULP. Those are double-rounding cases — this
 * goes f64 to f32 to f16 where the native path goes f64 to f16 directly.
 * Eliminating them needs a round-to-odd intermediate, which is not worth the
 * code for a 3e-5 disagreement one ULP below a quantization the renderer has
 * already applied. Stated rather than hidden.
 */
export function toHalf(value) {
  _f32[0] = value;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);   // inf / NaN
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;                            // overflow
  if (e <= 0) {
    if (e < -10) return sign;                                     // underflow
    const m = mant | 0x800000;
    const shift = 14 - e;
    let half = m >>> shift;
    const rest = m & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rest > halfway || (rest === halfway && (half & 1))) half += 1;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (half & 1))) half += 1;
  return sign | half;
}

/** IEEE binary16 to binary32 — the readback direction (dart-tip probe). */
export function fromHalf(h) {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0x1f) {
    _u32[0] = sign | 0x7f800000 | (mant << 13);       // inf / NaN
    return _f32[0];
  }
  if (exp === 0) {
    if (mant === 0) { _u32[0] = sign; return _f32[0]; }
    // Subnormal: exact in f32.
    return (sign ? -1 : 1) * mant * 2 ** -24;
  }
  _u32[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  return _f32[0];
}

export function makeHalfWriter(length) {
  if (HAS_F16) {
    const view = new Float16Array(length);
    return { store: view, set: (i, v) => { view[i] = v; },
             bytes: () => new Uint16Array(view.buffer) };
  }
  const view = new Uint16Array(length);
  return { store: view, set: (i, v) => { view[i] = toHalf(v); },
           bytes: () => view };
}
