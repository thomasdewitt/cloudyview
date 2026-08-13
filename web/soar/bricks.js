// Sparse brick decomposition of a cloud field.
//
// A cloud field at LES or STEAM resolution is mostly vacuum — the STEAM
// small-domain parent is 0.22% occupied, and even the bundled LES demos run
// 10-36%. Storing the whole grid as one dense 3D texture pays full price for
// the vacuum and, worse, ties the texture's dimensions to the field's
// extents, which Chrome and Safari clamp to 2048 whatever the hardware
// supports (gpu.js). Bricking cuts both ties: occupied bricks are packed
// into an atlas whose dimensions come from brick COUNT, and a page table
// maps field position to atlas slot.
//
// Every brick carries a 1-voxel apron on each side holding its neighbours'
// edge values (or the periodic wrap at domain faces), so hardware trilinear
// filtering inside one brick — apron included — is exact everywhere,
// including across brick seams. Bricking is therefore value-preserving by
// construction: same voxels, same filtering, different storage. The node
// round-trip test (tests/test_soar_bricks.py) holds this to exact equality
// against a dense reference.
//
// This module is pure array-in / array-out, node-testable, and does not
// touch WebGPU: it is the single definition of the layout that ingest,
// scene.js and the Python host must eventually share, for the same reason
// ghost.js is (two hosts silently disagreed about ghost texels for months;
// a shape that can drift needs one implementation and a test that runs it).
//
// Layout conventions, matching the rest of soar:
//   * Field arrays are x-major with z fastest: index (x * ny + y) * nz + z.
//   * The atlas is one flat array in the same convention over its own texel
//     dims. Slot s sits at brick-grid position (sx, sy, sz) with
//     s = (sx * gy + sy) * gz + sz, scaled by the padded brick extents.
//   * The page table is x-major over ceil(dims/brick) entries; 0 = empty
//     brick, otherwise 1-based slot id.
//   * Periodic fields wrap in x and y only; z out of range is empty, exactly
//     as ghost.js fills the dense texture's ghost ring.

"use strict";

/** Ceil-divide field extents by brick extents. */
function pageDimsFor(dims, brick) {
  return dims.map((n, i) => Math.ceil(n / brick[i]));
}

/**
 * Streaming brick builder.
 *
 * Tiles may arrive in any order, cover any box, and need not align with the
 * brick grid — ingest's tiles are HDF5-chunk-aligned with ragged edges, and
 * a brick straddling two tiles just gets written twice. Only nonzero values
 * are written (slots allocate zeroed), so a tile of vacuum costs nothing.
 *
 * finalize() fills every apron texel from the finished interiors and packs
 * the slots into one near-cubic atlas. One pass over the file suffices: an
 * apron texel's value lives in some brick's interior (or is zero / a wrap),
 * and by finalize time every interior is present.
 *
 * @param {object} opts
 * @param {number[]} opts.dims   Field extents [nx, ny, nz], unpadded.
 * @param {number[]} [opts.brick] Brick extents, default [8, 8, 8].
 * @param {boolean} [opts.periodic] Wrap x/y aprons at domain faces.
 */
export function createBrickBuilder({ dims, brick = [8, 8, 8], periodic = true }) {
  const [nx, ny, nz] = dims;
  const [bx, by, bz] = brick;
  const [gx, gy, gz] = pageDimsFor(dims, brick);
  const px = bx + 2, py = by + 2, pz = bz + 2;   // padded brick extents
  const slotTexels = px * py * pz;

  const pageTable = new Uint32Array(gx * gy * gz);
  /** @type {Array<Float32Array|Uint16Array|Float64Array>} */
  const slots = [];
  let ValueArray = null;
  let occupiedVoxels = 0;

  const pageIndex = (cx, cy, cz) => (cx * gy + cy) * gz + cz;

  function slotFor(cx, cy, cz, ctor) {
    const pi = pageIndex(cx, cy, cz);
    let id = pageTable[pi];
    if (id === 0) {
      slots.push(new ctor(slotTexels));
      id = slots.length;           // 1-based
      pageTable[pi] = id;
    }
    return slots[id - 1];
  }

  function addTile(base, size, values) {
    const [x0, y0, z0] = base;
    const [sx, sy, sz] = size;
    if (values.length !== sx * sy * sz) {
      throw new Error(`tile ${sx}x${sy}x${sz} expects ${sx * sy * sz} values, got ${values.length}`);
    }
    if (ValueArray === null) ValueArray = values.constructor;
    let i = 0;
    for (let lx = 0; lx < sx; lx++) {
      const x = x0 + lx, cx = (x / bx) | 0, ox = x - cx * bx;
      for (let ly = 0; ly < sy; ly++) {
        const y = y0 + ly, cy = (y / by) | 0, oy = y - cy * by;
        for (let lz = 0; lz < sz; lz++, i++) {
          const v = values[i];
          if (v === 0) continue;
          const z = z0 + lz, cz = (z / bz) | 0, oz = z - cz * bz;
          const slot = slotFor(cx, cy, cz, ValueArray);
          slot[((ox + 1) * py + oy + 1) * pz + oz + 1] = v;
          occupiedVoxels++;
        }
      }
    }
  }

  /** Field value at a global coordinate, wrapped or zero per periodicity. */
  function valueAt(x, y, z) {
    if (periodic) {
      if (x < 0) x += nx; else if (x >= nx) x -= nx;
      if (y < 0) y += ny; else if (y >= ny) y -= ny;
    } else if (x < 0 || x >= nx || y < 0 || y >= ny) {
      return 0;
    }
    if (z < 0 || z >= nz || x < 0 || x >= nx || y < 0 || y >= ny) return 0;
    const id = pageTable[pageIndex((x / bx) | 0, (y / by) | 0, (z / bz) | 0)];
    if (id === 0) return 0;
    return slots[id - 1][((x % bx + 1) * py + y % by + 1) * pz + z % bz + 1];
  }

  function finalize() {
    if (ValueArray === null) ValueArray = Float32Array;
    const n = slots.length;

    // Fill every non-interior texel of every slot: true neighbour values
    // where the coordinate lands in an occupied brick, wrap values at
    // periodic faces, zero elsewhere. Interior texels of a PARTIAL edge
    // brick (coordinates past the field extent) get the same treatment, so
    // the invariant is uniform: every atlas texel equals the field value at
    // its (wrapped) global coordinate.
    for (let cx = 0; cx < gx; cx++) {
      for (let cy = 0; cy < gy; cy++) {
        for (let cz = 0; cz < gz; cz++) {
          const id = pageTable[pageIndex(cx, cy, cz)];
          if (id === 0) continue;
          const slot = slots[id - 1];
          for (let ox = 0; ox < px; ox++) {
            const x = cx * bx + ox - 1;
            const xEdge = ox === 0 || ox === px - 1 || x >= nx;
            for (let oy = 0; oy < py; oy++) {
              const y = cy * by + oy - 1;
              const yEdge = oy === 0 || oy === py - 1 || y >= ny;
              for (let oz = 0; oz < pz; oz++) {
                const z = cz * bz + oz - 1;
                if (!(xEdge || yEdge || oz === 0 || oz === pz - 1 || z >= nz)) {
                  continue;   // interior texel already written by addTile
                }
                const v = valueAt(x, y, z);
                if (v !== 0) slot[(ox * py + oy) * pz + oz] = v;
              }
            }
          }
        }
      }
    }

    // Pack slots into a near-cubic grid of bricks: dimensions come from
    // brick count, never from field extent — this is what dissolves the
    // 2048-per-axis texture clamp for large fields.
    const ax = Math.max(1, Math.ceil(Math.cbrt(n)));
    const ay = Math.max(1, Math.ceil(Math.sqrt(n / ax)));
    const az = Math.max(1, Math.ceil(n / (ax * ay)));
    const atlasDims = [ax * px, ay * py, az * pz];
    const atlas = new ValueArray(atlasDims[0] * atlasDims[1] * atlasDims[2]);
    for (let s = 0; s < n; s++) {
      const sx = (s / (ay * az)) | 0;
      const sy = ((s / az) | 0) % ay;
      const sz = s % az;
      const slot = slots[s];
      for (let ox = 0; ox < px; ox++) {
        for (let oy = 0; oy < py; oy++) {
          const dst = ((sx * px + ox) * atlasDims[1] + sy * py + oy) * atlasDims[2]
                    + sz * pz;
          const src = (ox * py + oy) * pz;
          for (let oz = 0; oz < pz; oz++) atlas[dst + oz] = slot[src + oz];
        }
      }
    }

    return {
      pageDims: [gx, gy, gz],
      pageTable,
      atlasDims,
      atlas,
      atlasBrickGrid: [ax, ay, az],
      brick: [bx, by, bz],
      apron: 1,
      stats: {
        occupiedBricks: n,
        totalBricks: gx * gy * gz,
        occupiedVoxels,
        atlasTexels: atlas.length,
        denseTexels: (nx + 2) * (ny + 2) * (nz + 2),
      },
    };
  }

  return { addTile, finalize };
}
