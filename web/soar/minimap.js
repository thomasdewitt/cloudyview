// The minimap: a top-down glimpse albedo map with the camera drawn on it.
//
// Port of cloudyview/soar/hud.py. The shader (hud.wgsl) is copied verbatim
// from the desktop, so everything here is host-side: colorizing the albedo
// into a resident texture once, and packing six vec4s of layout and overlay
// geometry per frame.
//
//     const map = new Minimap(device, { albedo, shape: [ny, nx] });
//     await map.init(targetFormat);
//     map.update(camera, scene, [canvasWidth, canvasHeight]);
//     map.encodePass(encoder, targetView, targetFormat);
//     map.destroy();
//
// `albedo` is a Float32Array in [0, 1) laid out (ny, nx) with east to the
// right and north up — glimpse's own orientation — because that is what the
// worker already produces on its way through the file.

"use strict";

import {
  MAP_HEIGHT_FRAC, MAP_MAX_WIDTH_FRAC, MAP_MARGIN_FRAC, MAP_OPACITY,
  MAP_SKY_BLUE,
} from "./constants.js";
import { cameraBasis } from "./camera.js";
import { mod360 } from "./spectral.js";

const DEG = Math.PI / 180.0;
const UNIFORM_NBYTES = 6 * 16;   // 6 vec4<f32>
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Meteorological azimuth (clockwise from north) to internal (CCW from east). */
export function azimuthMetToInternalDeg(azimuthDeg) {
  return mod360(90.0 - mod360(azimuthDeg));
}

/** RGBA8 sky-blue -> white, the ramp basic_render uses for glimpse output. */
export function colorizeAlbedo(albedo) {
  const rgba = new Uint8Array(albedo.length * 4);
  const [br, bg, bb] = MAP_SKY_BLUE;
  for (let i = 0; i < albedo.length; i++) {
    const a = clamp01(albedo[i]);
    rgba[i * 4 + 0] = Math.round((br + (1.0 - br) * a) * 255.0);
    rgba[i * 4 + 1] = Math.round((bg + (1.0 - bg) * a) * 255.0);
    rgba[i * 4 + 2] = Math.round((bb + (1.0 - bb) * a) * 255.0);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/** Project a 3D direction to unit top-down XY, falling back to an azimuth. */
function unitXY(direction, fallbackRad) {
  const n = Math.hypot(direction[0], direction[1]);
  if (n < 1e-10) return [Math.cos(fallbackRad), Math.sin(fallbackRad)];
  return [direction[0] / n, direction[1] / n];
}

/**
 * Where the minimap sits, in screen pixels: `(x, y, w, h)`, top-right.
 *
 * Exported for the same reason the desktop exposes it — it is pure arithmetic
 * over the window size and the map's aspect, and it is the piece a test can
 * check without a GPU.
 */
export function rectForSize(size, albedoShape) {
  const [screenW, screenH] = [Number(size[0]), Number(size[1])];
  const [ny, nx] = albedoShape;
  const aspect = nx / ny;
  const margin = Math.max(8.0, Math.round(screenH * MAP_MARGIN_FRAC));

  let mapH = Math.max(24.0, Math.round(screenH * MAP_HEIGHT_FRAC));
  let mapW = Math.round(mapH * aspect);
  const maxW = Math.max(24.0, screenW * MAP_MAX_WIDTH_FRAC);
  if (mapW > maxW) {
    const scale = maxW / mapW;
    mapW *= scale; mapH *= scale;
  }
  const availW = Math.max(24.0, screenW - 2.0 * margin);
  const availH = Math.max(24.0, screenH - 2.0 * margin);
  if (mapW > availW) {
    const scale = availW / mapW;
    mapW *= scale; mapH *= scale;
  }
  if (mapH > availH) {
    const scale = availH / mapH;
    mapW *= scale; mapH *= scale;
  }
  return [screenW - margin - mapW, margin, mapW, mapH];
}

/**
 * The camera marker and its field-of-view wedge, in map UV.
 *
 * Straight up or straight down in view makes a top-down wedge meaningless —
 * every horizontal bearing is in frame — so that case draws a ring instead.
 */
export function cameraOverlayGeometry(
  relativePosition, azimuth, elevation, fov, albedoShape, renderAspect,
) {
  const [ny, nx] = albedoShape;
  const nxm1 = Math.max(nx - 1, 1);
  const nym1 = Math.max(ny - 1, 1);
  const camX = (relativePosition[0] + 1.0) * 0.5 * nxm1;
  const camY = (relativePosition[1] + 1.0) * 0.5 * nym1;
  const cameraUV = [camX / nxm1, camY / nym1];

  // fov is horizontal (see camera.py); the zenith/nadir test is a vertical
  // question, so the height's half-angle has to be derived from it.
  const halfHfov = fov * 0.5 * DEG;
  const halfVfovDeg = Math.atan(Math.tan(halfHfov) / renderAspect) / DEG;
  if ((90.0 - elevation) <= halfVfovDeg || (90.0 + elevation) <= halfVfovDeg) {
    return { cameraUV, circleRadiusPx: nx / 10.0 };
  }

  const azInternalRad = azimuthMetToInternalDeg(azimuth) * DEG;
  // The analytic horizontal right vector, not a cross product: continuous
  // through vertical, which is the whole reason cameraBasis is shaped that way.
  const [forward, right] = cameraBasis(azimuth, elevation);
  const t = Math.tan(halfHfov);

  const ray = (sign) => {
    const d = [forward[0] + sign * t * right[0],
               forward[1] + sign * t * right[1],
               forward[2] + sign * t * right[2]];
    const n = Math.hypot(d[0], d[1], d[2]);
    return [d[0] / n, d[1] / n, d[2] / n];
  };
  const leftXY = unitXY(ray(-1), azInternalRad - halfHfov);
  const rightXY = unitXY(ray(+1), azInternalRad + halfHfov);

  const rayLength = 1.5 * Math.max(nx, ny);
  return {
    cameraUV,
    fovEndpoints: [
      [(camX + rayLength * leftXY[0]) / nxm1,
       (camY + rayLength * leftXY[1]) / nym1],
      [(camX + rayLength * rightXY[0]) / nxm1,
       (camY + rayLength * rightXY[1]) / nym1],
    ],
  };
}

/** A nest's horizontal footprint in map UV, or null when there is no nest. */
export function nestMapUV(scene) {
  if (!scene?.nested) return null;
  const { bmin, bmax } = scene;
  const lo = [], hi = [];
  for (let i = 0; i < 2; i++) {
    const extent = bmax[i] - bmin[i];
    if (!(extent > 0)) return null;
    lo.push(clamp01((scene.nestBmin[i] - bmin[i]) / extent));
    hi.push(clamp01((scene.nestBmax[i] - bmin[i]) / extent));
  }
  return [lo[0], lo[1], hi[0], hi[1]];
}

export class Minimap {
  constructor(device, { albedo, shape }) {
    const [ny, nx] = shape;
    if (albedo.length !== nx * ny) {
      throw new Error(
        `The minimap was given ${albedo.length} albedo values for a ` +
        `${nx}x${ny} map, which needs ${nx * ny}.`);
    }
    const maxDim = device.limits.maxTextureDimension2D;
    if (Math.max(nx, ny) > maxDim) {
      throw new Error(
        `The minimap would be ${nx}x${ny}, past this GPU's 2D texture limit ` +
        `of ${maxDim}. The field is fine; only the overlay cannot be built.`);
    }

    this.device = device;
    this.albedoShape = [ny, nx];
    this.uniforms = new Float32Array(6 * 4);

    this.texture = device.createTexture({
      label: "hud-minimap",
      size: [nx, ny, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.texture }, colorizeAlbedo(albedo),
      { bytesPerRow: nx * 4, rowsPerImage: ny }, [nx, ny, 1]);

    this.sampler = device.createSampler({
      addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge",
      magFilter: "linear", minFilter: "linear",
    });
    this.uniformBuf = device.createBuffer({
      label: "hud-uniforms", size: UNIFORM_NBYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Explicit, never layout: "auto". An automatic layout is derived from the
    // bindings the entry point happens to reference, which has already cost
    // this project one silently blank pass.
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" } },
      ],
    });
    this.bindGroup = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.texture.createView() },
        { binding: 2, resource: this.sampler },
      ],
    });
    this._pipelines = new Map();
    this._rect = null;
  }

  async init(targetFormat, shaderSource) {
    const code = shaderSource ??
      await (await fetch(new URL("./hud.wgsl", import.meta.url))).text();
    this.module = this.device.createShaderModule(
      { label: "hud-minimap", code });
    const info = await this.module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) {
      throw new Error(
        "hud.wgsl did not compile: " +
        errors.map((m) => `line ${m.lineNum}: ${m.message}`).join("; "));
    }
    this.pipelineLayout = this.device.createPipelineLayout(
      { bindGroupLayouts: [this.layout] });
    this._pipelineFor(targetFormat);
    return this;
  }

  _pipelineFor(targetFormat) {
    let pipeline = this._pipelines.get(targetFormat);
    if (!pipeline) {
      pipeline = this.device.createRenderPipeline({
        label: "hud-minimap",
        layout: this.pipelineLayout,
        vertex: { module: this.module, entryPoint: "vs_main" },
        primitive: { topology: "triangle-list" },
        fragment: {
          module: this.module, entryPoint: "fs_main",
          targets: [{
            format: targetFormat,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha",
                       operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha",
                       operation: "add" },
            },
          }],
        },
      });
      this._pipelines.set(targetFormat, pipeline);
    }
    return pipeline;
  }

  /** The marker centre in screen pixels — for tests and diagnostics. */
  markerPixel(camera, size) {
    const rect = rectForSize(size, this.albedoShape);
    const { cameraUV } = cameraOverlayGeometry(
      camera.relativePosition(), camera.azimuth, camera.elevation, camera.fov,
      this.albedoShape, size[0] / size[1]);
    return [rect[0] + cameraUV[0] * rect[2],
            rect[1] + (1.0 - cameraUV[1]) * rect[3]];
  }

  /** Pack the layout and overlay geometry for this frame. */
  update(camera, scene, size) {
    const [screenW, screenH] = [Number(size[0]), Number(size[1])];
    const rect = rectForSize(size, this.albedoShape);
    const [ny, nx] = this.albedoShape;
    const overlay = cameraOverlayGeometry(
      camera.relativePosition(), camera.azimuth, camera.elevation, camera.fov,
      this.albedoShape, screenW / screenH);
    const [camU, camV] = overlay.cameraUV;

    const minSide = Math.min(rect[2], rect[3]);
    const markerRadius = Math.max(3.0, minSide * 0.028);
    const lineWidth = Math.max(1.25, minSide * 0.010);
    const borderWidth = Math.max(1.0, minSide * 0.008);
    const haloWidth = Math.max(0.85, lineWidth * 0.75);

    let mode = 0.0, circleRadius = 0.0;
    let leftU = 0.0, leftV = 0.0, rightU = 0.0, rightV = 0.0;
    if (overlay.fovEndpoints) {
      [[leftU, leftV], [rightU, rightV]] = overlay.fovEndpoints;
    } else {
      mode = 1.0;
      circleRadius = overlay.circleRadiusPx * rect[2] / Math.max(nx - 1, 1);
    }

    const nest = nestMapUV(scene);
    const u = this.uniforms;
    u.set([screenW, screenH, MAP_OPACITY, markerRadius], 0);
    u.set(rect, 4);
    u.set([camU, camV, mode, circleRadius], 8);
    u.set([leftU, leftV, rightU, rightV], 12);
    u.set([lineWidth, borderWidth, haloWidth, nest ? 1.0 : 0.0], 16);
    u.set(nest ?? [0, 0, 0, 0], 20);
    this.device.queue.writeBuffer(this.uniformBuf, 0, u);

    this._rect = rect;
    this._size = [screenW, screenH];
  }

  /**
   * Draw over an already-rendered frame. Scissored to the map's own corner so
   * the pass touches a few thousand pixels rather than the whole screen.
   */
  encodePass(encoder, targetView, targetFormat) {
    if (!this._rect) return;
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view: targetView, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this._pipelineFor(targetFormat));
    pass.setBindGroup(0, this.bindGroup);
    const [x, y, w, h] = this._rect;
    const [screenW, screenH] = this._size;
    const pad = 3.0;
    const x0 = Math.max(0, Math.floor(x - pad));
    const y0 = Math.max(0, Math.floor(y - pad));
    const x1 = Math.min(Math.round(screenW), Math.ceil(x + w + pad));
    const y1 = Math.min(Math.round(screenH), Math.ceil(y + h + pad));
    if (x1 > x0 && y1 > y0) pass.setScissorRect(x0, y0, x1 - x0, y1 - y0);
    pass.draw(3);
    pass.end();
  }

  destroy() {
    this.texture?.destroy();
    this.uniformBuf?.destroy();
    this._pipelines.clear();
  }
}
