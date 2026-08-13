// GPU residency for the sparse brick decomposition.
//
// bricks.js does the layout — which voxels land in which atlas slot, and what
// every apron texel holds — and knows nothing about WebGPU so that a node test
// can hold it to exact equality against a dense reference. This module is the
// other half: it feeds a field to that builder and turns the result into the
// two textures the shader reads.
//
//   * the ATLAS, r16float, holding only occupied bricks, each padded by a
//     1-voxel apron so hardware trilinear inside a brick is exact everywhere
//     including across brick seams;
//   * the PAGE TABLE, r32uint, one texel per brick of the field, holding 0 for
//     an empty brick and otherwise a 1-based atlas slot.
//
// Nothing here needs a uniform. Every quantity the shader wants — the page
// grid, the atlas's brick grid — is recoverable from textureDimensions of
// these two, and the brick extent is a compile-time constant it shares with
// this file. That is deliberate: the uniform block is byte-diffed against the
// Python host's packer (tests/test_uniform_parity.py), and a feature that can
// be added without touching it should be.

"use strict";

import { createBrickBuilder } from "./bricks.js";
import { guardAllocation } from "./gpu.js";

// Brick extent, in voxels, before the apron. 8^3 is the survey's default and
// the shape tests/test_soar_bricks.py pins hardest, but the whole point of
// keeping it in one place is that trying 16^3 or 16x16x4 is a one-line
// experiment — the page table, the atlas and the shader's DDA all size
// themselves off it.
export const BRICK = [8, 8, 8];

/**
 * Build the brick payload for a field held as ingest-ordered tiles.
 *
 * `feed` is called with the builder and must hand it every tile of the field
 * exactly once, in any order, via `addTile(base, size, values)`. Both callers
 * already hold the field in that shape — the ingest worker as the slabs it
 * staged, the demo loader as the padded volume it downloaded — so neither has
 * to materialise a second copy to get here.
 */
export function buildBrickPayload(dims, feed, { brick = BRICK,
                                                periodic = true } = {}) {
  const builder = createBrickBuilder({ dims, brick, periodic });
  feed(builder);
  return builder.finalize();
}

/**
 * Feed a ghost-padded dense volume to a brick builder.
 *
 * `padded` is the (nx+2, ny+2, nz+2) fp16 array the dense path uploads, x
 * major with z fastest — original voxel i at index i+1 on every axis. Only the
 * interior is fed: the ghost ring is not data, and bricks.js reconstructs the
 * equivalent apron itself (from real neighbours, or the periodic wrap, or
 * zero) precisely so that the two representations agree at the seams.
 *
 * Fed one x-plane at a time. A whole-field tile would be one allocation the
 * size of the field, which is the thing this feature exists to avoid.
 */
export function feedPaddedVolume(padded, dims) {
  const [nx, ny, nz] = dims;
  const [py, pz] = [ny + 2, nz + 2];
  return (builder) => {
    const plane = new Uint16Array(ny * nz);
    for (let x = 0; x < nx; x++) {
      let o = 0;
      for (let y = 0; y < ny; y++) {
        const row = ((x + 1) * py + (y + 1)) * pz + 1;
        for (let z = 0; z < nz; z++) plane[o++] = padded[row + z];
      }
      builder.addTile([x, 0, 0], [1, ny, nz], plane);
    }
  };
}

/**
 * Upload a payload from buildBrickPayload as the atlas and page-table
 * textures, returning them alongside the stats the panel reports.
 *
 * The atlas is written brick-row by brick-row rather than in one call: a
 * writeTexture of a multi-gigabyte volume is exactly the allocation that took
 * the device down on the dense path (see ingest/index.js on draining), and
 * there is no reason to reintroduce it here.
 */
export async function uploadBrickPayload(device, payload, label) {
  const [ax, ay, az] = payload.atlasDims;
  const [gx, gy, gz] = payload.pageDims;

  // Texture axes are (w=z, h=y, d=x), matching the dense volume — see the
  // raymarch.wgsl header. Both textures follow it so the shader's swizzle is
  // the same one it already uses.
  const atlas = await guardAllocation(
    device, `the brick atlas for ${label}`, payload.atlas.byteLength,
    () => device.createTexture({
      label: `soar-brick-atlas-${label}`,
      size: [az, ay, ax], dimension: "3d", format: "r16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));

  // One x-plane of the atlas per write, for the same reason the dense upload
  // drains: driver staging memory is only reclaimed once the GPU consumes it.
  const planeTexels = ay * az;
  for (let x = 0; x < ax; x++) {
    const plane = payload.atlas.subarray(x * planeTexels, (x + 1) * planeTexels);
    device.queue.writeTexture(
      { texture: atlas, origin: { x: 0, y: 0, z: x } },
      plane, { bytesPerRow: az * 2, rowsPerImage: ay }, [az, ay, 1]);
    if ((x & 63) === 63) await device.queue.onSubmittedWorkDone();
  }

  // r32uint, not a storage buffer: the DDA reads it with textureLoad at
  // integer brick coordinates, which needs no bounds arithmetic of its own and
  // no second binding kind. Page tables are small — one texel per brick, so a
  // 1024x512x231 field at 8^3 is 128x64x29, under a megabyte.
  const page = await guardAllocation(
    device, `the page table for ${label}`, payload.pageTable.byteLength,
    () => device.createTexture({
      label: `soar-brick-pages-${label}`,
      size: [gz, gy, gx], dimension: "3d", format: "r32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    }));
  device.queue.writeTexture(
    { texture: page }, payload.pageTable,
    { bytesPerRow: gz * 4, rowsPerImage: gy }, [gz, gy, gx]);

  return {
    atlas, page,
    atlasView: atlas.createView(),
    pageView: page.createView(),
    brick: payload.brick,
    stats: payload.stats,
    destroy() { atlas.destroy(); page.destroy(); },
  };
}

/**
 * Stand-ins for the non-bricked path.
 *
 * The bind group layout is fixed at pipeline-layout creation and cannot have
 * holes, so a dense field still has to bind something of the right kind. Same
 * device-level reason scene.js keeps a nest dummy; same 1-texel cost.
 */
export function createBrickDummies(device) {
  const atlas = device.createTexture({
    label: "soar-brick-atlas-absent",
    size: [1, 1, 1], dimension: "3d", format: "r16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: atlas }, new Uint16Array(1),
    { bytesPerRow: 2, rowsPerImage: 1 }, [1, 1, 1]);
  const page = device.createTexture({
    label: "soar-brick-pages-absent",
    size: [1, 1, 1], dimension: "3d", format: "r32uint",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture: page }, new Uint32Array(1),
    { bytesPerRow: 4, rowsPerImage: 1 }, [1, 1, 1]);
  return {
    atlas, page,
    atlasView: atlas.createView(),
    pageView: page.createView(),
    destroy() { atlas.destroy(); page.destroy(); },
  };
}
