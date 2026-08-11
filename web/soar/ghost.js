// The periodic ghost ring of the volume texture.
//
// A doubly periodic LES field tiles horizontally, so the texel just outside
// face x=0 is not empty air — it is the field at x=nx-1. Filling the lateral
// ghost ring from the opposite faces is what makes hardware trilinear
// filtering exact across the wrap seam; leaving it zero tapers every ray that
// crosses a domain edge into a boundary that is not there.
//
// This module holds the two halves of that as pure array-in / array-out
// functions, because there are three consumers and they must agree exactly:
//
//   * ingest/worker.js builds the faces from four thin hyperslabs of the file
//     (buildXFace / buildYFace below);
//   * scene.js uploads them into the resident texture (GHOST_PLANES gives the
//     origin and size of each);
//   * cloudyview/soar_host.py does the same thing to a numpy array before it
//     uploads, and tests/test_soar_texture_parity.py runs applyGhostFaces
//     under node and byte-diffs the result against the Python host's.
//
// That last one exists because these two hosts silently disagreed about this
// for the whole life of the periodic renderer, including in the golden
// images. A shape that can drift needs a test that runs both sides.
//
// Layout convention, as everywhere in soar: the padded volume is (nx+2, ny+2,
// nz+2) in x-major order with z fastest, which is exactly the byte order a 3D
// texture of width nz+2, height ny+2, depth nx+2 wants.

"use strict";

import { makeHalfWriter } from "./half.js";

/** Ghost face names, in the order faces.bin packs them. */
export const FACE_NAMES = ["x_lo", "x_hi", "y_lo", "y_hi"];

/**
 * Where each face plane lands in the padded texture, as writeTexture wants
 * it: {origin: [x, y, z], size: [width, height, depth]} in texture axes
 * (width indexes z, height y, depth x).
 *
 * The x planes are written first and span the full height, so they include
 * the y ghost rows; the y planes then span the full depth and overwrite the
 * four corner columns. That ordering is what makes a corner texel wrap in
 * both x and y, which is what the trilinear support of a sample near a
 * domain corner needs, and it is why this is a list rather than a set.
 */
export function ghostPlanes(padded) {
  const [px, py, pz] = padded;
  return [
    { name: "x_lo", origin: [0, 0, 0],       size: [pz, py, 1], rows: py },
    { name: "x_hi", origin: [0, 0, px - 1],  size: [pz, py, 1], rows: py },
    { name: "y_lo", origin: [0, 0, 0],       size: [pz, 1, px], rows: 1 },
    { name: "y_hi", origin: [0, py - 1, 0],  size: [pz, 1, px], rows: 1 },
  ];
}

/** Element count of one face plane. */
export function faceLength(name, padded) {
  const [px, py, pz] = padded;
  return name.startsWith("x") ? py * pz : px * pz;
}

/**
 * Where the values of one lateral slice go in the (n+2, nz+2) ghost plane it
 * becomes: the slice in the interior, the two ghost ROWS wrapped from the
 * far side of the OTHER lateral axis, and the z ghost columns left alone —
 * the vertical is never periodic.
 *
 * Expressed once, over a read/write pair, because it has two callers that
 * must not drift: buildXFace/buildYFace convert values to fp16 on the way in
 * (the worker's path, straight off the file), and placeXFace/placeYFace move
 * bit patterns that are already fp16 (the in-memory path, which must not
 * round a second time).
 */
function facePlacement(n, nz, read, write) {
  for (let iz = 0; iz < nz; iz++) {
    for (let i = 0; i < n; i++) write((i + 1) * (nz + 2) + iz + 1, read(i, iz));
    write(iz + 1, read(n - 1, iz));
    write((n + 1) * (nz + 2) + iz + 1, read(0, iz));
  }
}

/** An x ghost plane from one (ny, nz) slice of the field. */
export function buildXFace(values, ny, nz) {
  const face = makeHalfWriter((ny + 2) * (nz + 2));
  facePlacement(ny, nz, (iy, iz) => values[iy * nz + iz],
                (i, v) => face.set(i, v));
  return face.bytes();
}

/** A y ghost plane from one (nx, nz) slice; the mirror of buildXFace. */
export function buildYFace(values, nx, nz) {
  const face = makeHalfWriter((nx + 2) * (nz + 2));
  facePlacement(nx, nz, (ix, iz) => values[ix * nz + iz],
                (i, v) => face.set(i, v));
  return face.bytes();
}

function placeBits(bits, n, nz) {
  const face = new Uint16Array((n + 2) * (nz + 2));
  facePlacement(n, nz, (i, iz) => bits[i * nz + iz],
                (i, v) => { face[i] = v; });
  return face;
}

/**
 * The four faces of an already-padded volume, read back out of it.
 *
 * The worker builds its faces straight from the file because it never holds
 * the whole volume; this is the same construction for callers that do (the
 * baked demo, and the parity test). `data` is fp16 bit patterns.
 */
export function wrapFacesFromPadded(data, padded) {
  const [px, py, pz] = padded;
  const nx = px - 2, ny = py - 2, nz = pz - 2;
  const at = (x, y, z) => data[((x + 1) * py + (y + 1)) * pz + (z + 1)];
  const gather = (n, m, get) => {
    const out = new Uint16Array(n * m);
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < m; b++) out[a * m + b] = get(a, b);
    }
    return out;
  };
  const xSlice = (ix) => gather(ny, nz, (iy, iz) => at(ix, iy, iz));
  const ySlice = (iy) => gather(nx, nz, (ix, iz) => at(ix, iy, iz));
  return {
    x_lo: placeBits(xSlice(nx - 1), ny, nz),
    x_hi: placeBits(xSlice(0), ny, nz),
    y_lo: placeBits(ySlice(ny - 1), nx, nz),
    y_hi: placeBits(ySlice(0), nx, nz),
  };
}

/**
 * Write the four planes into a padded volume held in memory, exactly where
 * scene.writeGhostBorder writes them into the texture.
 *
 * This is what makes the two hosts testable against each other: the browser
 * never materializes this array (it writes straight to the card), but the
 * texel contents it produces are the ones this returns.
 */
export function applyGhostFaces(data, padded, faces) {
  const [px, py, pz] = padded;
  for (const plane of ghostPlanes(padded)) {
    const src = faces[plane.name];
    const [ox, oy, oz] = plane.origin;
    const [sw, sh, sd] = plane.size;
    let o = 0;
    for (let d = 0; d < sd; d++) {
      for (let h = 0; h < sh; h++) {
        const base = ((oz + d) * py + (oy + h)) * pz + ox;
        for (let w = 0; w < sw; w++) data[base + w] = src[o++];
      }
    }
  }
  return data;
}

/** Zeroed planes of the right shapes, for turning periodicity back off. */
export function zeroFaces(padded) {
  const faces = {};
  for (const name of FACE_NAMES) {
    faces[name] = new Uint16Array(faceLength(name, padded));
  }
  return faces;
}
