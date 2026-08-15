// Which z planes of a cloud field hold anything, and the band that follows.
//
// Cloud fields are mostly empty sky, and the emptiness is overwhelmingly
// vertical: measured across the demo set, 7.6% of the z extent is vacuum on a
// STEAM parent, 34.8% on the FIF cascade, 40.4% on CM1, and 74.2% on DYCOMS,
// whose deck occupies planes 216-352 of 531. TWP-ICE is the exception at 0% —
// it is already tight, which is why the eight golden judge views are blind to
// this feature. Sizing the volume texture to the file's z extent pays for all
// that vacuum twice: once in memory, and again in a march that crosses it
// sample by sample because nothing tells it there is nothing there.
//
// This module is pure array-in / array-out and node-testable on purpose: the
// browser's ingest worker and the Python host must reach the SAME band from
// the same file, and a rule duplicated in two hosts is a rule that drifts.
// These two have silently disagreed about texture construction once already —
// the browser wrapped a periodic field's lateral ghost ring and the Python
// host shipped zeros there, so every `witness --periodic` render and all
// eight goldens tapered into a boundary that was not there, for the entire
// life of the periodic renderer, and nothing said a word. (That ring is gone
// — the wrap is a sampler mode now — but the lesson is why this file is
// shaped the way it is.)

"use strict";

// A float stores as a NONZERO fp16 exactly when it exceeds half the smallest
// positive subnormal: round-to-nearest-even sends 2**-25 itself to zero.
// Extinction is non-negative, so this one comparison is the whole test.
//
// Emptiness is judged on the value AS STORED, never on the f64 sigma behind
// it. Both hosts upload r16float, so a sigma below this floor is zero as far
// as any renderer is ever going to know — and defining the crop that way is
// what lets a host that holds fp16 bytes (the browser, which has already
// quantized by the time it can look) and a host that holds f64 (Python, which
// would rather not allocate a second copy of the field to find out) agree
// without either one converting.
export const FP16_NONZERO_FLOOR = Math.pow(2, -25);

/**
 * Mark the occupied planes of one fp16 slab.
 *
 * `values` is a slab of stored fp16 bit patterns in the ingest layout, where z
 * is the fastest axis — index (lx * ny + ly) * nz + lz — so the plane a texel
 * belongs to is its index modulo the slab's depth. Extinction is
 * non-negative, so the only zero bit pattern in play is 0x0000 and a plain
 * `!== 0` is exact.
 *
 * @param {Uint16Array} values  Stored fp16, z fastest.
 * @param {number} z0           Global z of the slab's first plane.
 * @param {number} depth        Slab depth in planes.
 * @param {Uint8Array} occupied Global per-plane flags, written in place.
 */
export function markOccupiedPlanes(values, z0, depth, occupied) {
  if (depth <= 0) throw new Error(`slab depth must be positive, got ${depth}`);
  if (values.length % depth !== 0) {
    throw new Error(
      `slab of ${values.length} values is not a whole number of ${depth}-deep ` +
      "columns; the z axis is not the fastest one it was taken to be");
  }
  for (let i = 0; i < values.length; i++) {
    if (values[i] !== 0) occupied[z0 + (i % depth)] = 1;
  }
  return occupied;
}

/**
 * The lowest and highest occupied planes, as a band to crop to.
 *
 * Returns `{lo, hi, count, cropped}`. Throws when nothing is occupied: that
 * is a field which would render as empty sky, and there is no band to crop
 * it to — the caller has a units problem, not a cropping problem, and saying
 * so beats returning a degenerate range that fails further downstream.
 *
 * A single occupied plane is widened to two, because the domain box takes its
 * outer half-cells from the gap between the last two z coordinates and a
 * one-plane field has no such gap. It is not a case the march can do anything
 * with either.
 */
export function occupiedBand(occupied) {
  const n = occupied.length;
  let lo = 0;
  while (lo < n && !occupied[lo]) lo++;
  if (lo >= n) {
    throw new Error(
      "Every z plane of this field stores as zero in fp16, so it would " +
      "render as empty sky and there is no occupied band to crop to. Check " +
      "the units — this is what a field read as kg/kg when it is really " +
      "g/kg looks like.");
  }
  let hi = n - 1;
  while (hi > lo && !occupied[hi]) hi--;
  if (hi === lo) hi = Math.min(n - 1, lo + 1);
  return { lo, hi, count: hi - lo + 1, cropped: hi - lo + 1 < n };
}
