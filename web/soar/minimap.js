// The minimap: a top-down map with the camera drawn on it — the cloud field's
// glimpse albedo, or, over a night city, the city itself (see "the city map"
// below).
//
// Port of cloudyview/soar/hud.py. The shader (hud.wgsl) is copied verbatim
// from the desktop, so everything here is host-side: colorizing the albedo
// into a resident texture once, and packing six vec4s of layout and overlay
// geometry per frame.
//
//     const map = new Minimap(device, { albedo, shape: [ny, nx] });
//     // …or, over a night city, the city itself:
//     const map = new Minimap(device, { albedo, shape, cityCells });
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
  MAP_CLOUD_RAMP,
  MAP_CITY_GROUND, MAP_CITY_ROOF,
  MAP_ACCENT,
} from "./constants.js";
import { cameraBasis } from "./camera.js";
import { mod360, hazeEFoldingKm } from "./spectral.js";

const DEG = Math.PI / 180.0;
const UNIFORM_NBYTES = 7 * 16;   // 7 vec4<f32>
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Meteorological azimuth (clockwise from north) to internal (CCW from east). */
export function azimuthMetToInternalDeg(azimuthDeg) {
  return mod360(90.0 - mod360(azimuthDeg));
}

/**
 * RGBA8 over MAP_CLOUD_RAMP — the ramp basic_render uses for glimpse output.
 *
 * Piecewise-linear between the ramp's stops, which is what matplotlib's
 * LinearSegmentedColormap does with the same list, so the two agree to a
 * rounding step. It used to be a single lerp from one blue to white; the
 * intermediate stops are the whole reason the field reads as cloud over
 * water, so they cannot be dropped for a cheaper interpolation here.
 */
export function colorizeAlbedo(albedo) {
  const rgba = new Uint8Array(albedo.length * 4);
  for (let i = 0; i < albedo.length; i++) {
    const [r, g, b] = sampleCloudRamp(clamp01(albedo[i]));
    rgba[i * 4 + 0] = Math.round(r * 255.0);
    rgba[i * 4 + 1] = Math.round(g * 255.0);
    rgba[i * 4 + 2] = Math.round(b * 255.0);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/** The ramp at `t` in [0, 1], linear between the bracketing stops. */
export function sampleCloudRamp(t) {
  const stops = MAP_CLOUD_RAMP;
  if (t <= stops[0][0]) return stops[0][1];
  for (let s = 1; s < stops.length; s++) {
    const [x1, c1] = stops[s];
    if (t <= x1) {
      const [x0, c0] = stops[s - 1];
      // Stops are distinct by construction; guard anyway so a future edit
      // that duplicates one divides by zero loudly rather than silently.
      const span = x1 - x0;
      if (span <= 0) throw new Error("MAP_CLOUD_RAMP has a repeated stop");
      const f = (t - x0) / span;
      return [c0[0] + (c1[0] - c0[0]) * f,
              c0[1] + (c1[1] - c0[1]) * f,
              c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

// --- the city map ----------------------------------------------------------
//
// Over a night city the minimap shows the CITY, not the cloud field above it.
// The cloud map would be the wrong picture in the wrong frame: the tile sits
// at fixed world metres under a domain whose size changes with the field, so
// a marker placed in domain coordinates does not say which district you are
// over, and the district is the only thing there is to navigate by down in
// the streets.
//
// The image is the same cascade the shader raises the buildings out of —
// scene.cityCells, the mip-0 texels of cloudyview/soar/city that the CITY
// specialization samples — so a bright patch on the map is a tower district
// on screen, and not an illustration of one.
//
// These six are raymarch.wgsl's, restated because a shader constant cannot
// be imported and this arithmetic runs on the CPU.
// tests/test_city_frame_parity.py fails if any of them drifts.
const CITY_MAP_EMPTY_RANK = 0.22;
const CITY_MAP_SPRAWL_RANK_FULL = 0.60;
const CITY_MAP_SPRAWL_MIN_FRAC = 0.15;
const CITY_MAP_H_BASE = 14.0;
const CITY_MAP_H_SCALE = 390.0;
const CITY_MAP_H_EXP = 1.2;

// Where the brightness ramp tops out (m). The tile's tallest blocks run to
// CITY_MAX_H and beyond, and scaling to the maximum would put the whole city
// in the bottom tenth of the ramp with three white pixels downtown. This is
// a skyline height: everything above it reads as "tall", which is the only
// distinction a 200 px map can carry anyway.
const CITY_MAP_H_FULL = 700.0;

/**
 * Mean building height per block, in metres, from the city tile's cells.
 *
 * city_cell's height, with its per-building random factor dropped — that
 * factor is `0.70 + 0.60 * rand`, mean exactly 1, so leaving it out gives the
 * expected height of the block rather than a different field. Unbuilt blocks
 * (rank at or below CITY_MAP_EMPTY_RANK) are zero, which is what makes the
 * empty outskirts read as empty.
 *
 * Returns the field in the tile's own texel order, row 0 at y = 0 — the same
 * order the texture was uploaded in, so tile x/y are map x/y and the map is
 * world-aligned and north-up like the cloud one.
 */
export function cityBlockHeights({ n, density, rank }) {
  const heights = new Float32Array(n * n);
  for (let i = 0; i < n * n; i++) {
    if (!(rank[i] > CITY_MAP_EMPTY_RANK)) continue;
    const t = clamp01((rank[i] - CITY_MAP_EMPTY_RANK) /
                      (CITY_MAP_SPRAWL_RANK_FULL - CITY_MAP_EMPTY_RANK));
    const smooth = t * t * (3.0 - 2.0 * t);          // smoothstep, as WGSL's
    const sprawl = CITY_MAP_SPRAWL_MIN_FRAC
      + (1.0 - CITY_MAP_SPRAWL_MIN_FRAC) * smooth;
    heights[i] = (CITY_MAP_H_BASE
                  + CITY_MAP_H_SCALE * Math.pow(Math.max(density[i], 0.0),
                                                CITY_MAP_H_EXP)) * sprawl;
  }
  return heights;
}

/**
 * The city map's colours: height as brightness, on the night city's own
 * sodium-over-ink palette rather than the cloud ramp's sky blue.
 *
 * Same shape of thing as colorizeAlbedo — a scalar field in, RGBA8 out — so
 * the Minimap's texture path, sampler and shader are untouched. The street
 * grid is below one texel here; what the map carries is the district
 * structure, which is what the cascade actually decides.
 */
export function colorizeCityHeights(heights) {
  const rgba = new Uint8Array(heights.length * 4);
  for (let i = 0; i < heights.length; i++) {
    // sqrt rather than linear: block heights are a multiplicative cascade,
    // so a linear map spends most of the ramp on the few tallest blocks and
    // leaves the body of the city one flat dark tone.
    const t = Math.sqrt(clamp01(heights[i] / CITY_MAP_H_FULL));
    const [r, g, b] = [
      MAP_CITY_GROUND[0] + (MAP_CITY_ROOF[0] - MAP_CITY_GROUND[0]) * t,
      MAP_CITY_GROUND[1] + (MAP_CITY_ROOF[1] - MAP_CITY_GROUND[1]) * t,
      MAP_CITY_GROUND[2] + (MAP_CITY_ROOF[2] - MAP_CITY_GROUND[2]) * t,
    ];
    rgba[i * 4 + 0] = Math.round(r * 255.0);
    rgba[i * 4 + 1] = Math.round(g * 255.0);
    rgba[i * 4 + 2] = Math.round(b * 255.0);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/**
 * The camera in the city tile's frame, as the relative triple
 * cameraOverlayGeometry takes: x and y in [-1, 1] across ONE tile.
 *
 * This is the whole reason the city map places the marker correctly — the
 * map image is one tile, so the marker's position on it is the camera's
 * position in the tile, which is scene.cityPosition (raymarch.wgsl's
 * world->tile map, see scene.js cityFramePosition). z is passed through
 * unused: nothing on a top-down map depends on it.
 */
export function cityRelativePosition(scene, position) {
  const city = scene.cityPosition(position);
  if (!city) return null;
  const extent = scene.oceanTileExtent;
  return [2.0 * city[0] / extent - 1.0, 2.0 * city[1] / extent - 1.0, 0.0];
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
export function rectForSize(size, albedoShape, fullscreen = false) {
  const [screenW, screenH] = [Number(size[0]), Number(size[1])];
  const [ny, nx] = albedoShape;
  const aspect = nx / ny;
  const margin = Math.max(8.0, Math.round(screenH * MAP_MARGIN_FRAC));

  if (fullscreen) {
    // As large as the screen allows at the map's own aspect, centred.
    const availW = Math.max(24.0, screenW - 2.0 * margin);
    const availH = Math.max(24.0, screenH - 2.0 * margin);
    let mapH = availH;
    let mapW = mapH * aspect;
    if (mapW > availW) {
      mapH *= availW / mapW;
      mapW = availW;
    }
    return [(screenW - mapW) * 0.5, (screenH - mapH) * 0.5, mapW, mapH];
  }

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
 *
 * `hazeUV`, when given, is the haze e-folding length as a map-UV radius per
 * axis (the two differ when the map is not square in metres): the wedge's
 * rays then TERMINATE there instead of running to the map edge, because past
 * that distance the view holds air, not field, and a ray across the whole
 * map claims you can see across the whole map.
 */
export function cameraOverlayGeometry(
  relativePosition, azimuth, elevation, fov, albedoShape, renderAspect,
  hazeUV = null,
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

  // leftXY/rightXY are unit in world XY, so scaling each component by the
  // per-axis UV radius lands the endpoint exactly on the haze ellipse.
  if (hazeUV) {
    return {
      cameraUV,
      fovEndpoints: [
        [cameraUV[0] + hazeUV[0] * leftXY[0],
         cameraUV[1] + hazeUV[1] * leftXY[1]],
        [cameraUV[0] + hazeUV[0] * rightXY[0],
         cameraUV[1] + hazeUV[1] * rightXY[1]],
      ],
    };
  }
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
  /**
   * `cityCells` builds the CITY map instead of the cloud one — see the city
   * map block above. It replaces the cloud map outright rather than being a
   * second image M cycles through: the minimap holds ONE texture, built at
   * construction, and the marker's frame is a property of which image that
   * is (domain-relative over a cloud field, tile-relative over a city). Two
   * images would mean two frames behind one dot, which is the confusion this
   * whole change exists to remove. M keeps its three states untouched.
   *
   * `albedo`/`shape` stay required either way and stay validated either way,
   * so a caller that gets the pair wrong is told so whichever map it asked
   * for. A city scene whose tile arrived without its cells does NOT quietly
   * get the cloud map — the caller has to pass cityCells or not, and passing
   * something malformed throws.
   */
  constructor(device, { albedo, shape, cityCells = null }) {
    let [ny, nx] = shape;
    if (albedo.length !== nx * ny) {
      throw new Error(
        `The minimap was given ${albedo.length} albedo values for a ` +
        `${nx}x${ny} map, which needs ${nx * ny}.`);
    }
    let image;
    if (cityCells) {
      if (!(cityCells.n > 0)
          || cityCells.density?.length !== cityCells.n * cityCells.n) {
        throw new Error(
          "The city minimap was given a cell field that is not n x n; the " +
          "tile's mip 0 is where it comes from.");
      }
      nx = ny = cityCells.n;
      image = colorizeCityHeights(cityBlockHeights(cityCells));
    } else {
      image = colorizeAlbedo(albedo);
    }
    this.frame = cityCells ? "city" : "domain";
    const maxDim = device.limits.maxTextureDimension2D;
    if (Math.max(nx, ny) > maxDim) {
      throw new Error(
        `The minimap would be ${nx}x${ny}, past this GPU's 2D texture limit ` +
        `of ${maxDim}. The field is fine; only the overlay cannot be built.`);
    }

    this.device = device;
    // The image's shape, whichever image it is: the cloud map's (ny, nx) or
    // the city tile's (n, n). Everything downstream — the rect's aspect, the
    // overlay geometry — reads it from here, so the two maps need no second
    // path. Still called albedoShape because it is still what the map is
    // made of, and renaming it would touch every caller for nothing.
    this.albedoShape = [ny, nx];
    this.uniforms = new Float32Array(7 * 4);

    this.texture = device.createTexture({
      label: "hud-minimap",
      size: [nx, ny, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.texture }, image,
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

  /**
   * The camera's position in THIS map's frame, as the relative triple
   * cameraOverlayGeometry takes.
   *
   * One place decides it, because the marker, the still and the fullscreen
   * click all have to agree about what the picture is a picture of. It
   * throws on a mismatch instead of drawing a plausible dot in the wrong
   * frame: a map built for a city and handed an ocean scene (or the reverse)
   * is a bug in the caller, and a silently wrong minimap is the worst thing
   * a navigation aid can be.
   */
  _relativeIn(camera, scene) {
    if (this.frame === "city") {
      if (!scene.city) {
        throw new Error(
          "The minimap was built as a city map and handed a scene with no " +
          "city tile.");
      }
      // The camera's own surface frame when it carries one (the live
      // FlightCamera, and replay poses): with the cloud and tile frames
      // folding at independent periods, a recompute from the world position
      // is the thing that can drift from what is rendered.
      if (camera.surfacePosition) {
        const e = scene.oceanTileExtent;
        return [2.0 * camera.surfacePosition[0] / e - 1.0,
                2.0 * camera.surfacePosition[1] / e - 1.0, 0.0];
      }
      const rel = cityRelativePosition(scene, camera.position);
      if (!rel) {
        throw new Error(
          "The city scene has no tile extent, so there is no city frame to " +
          "put the camera in.");
      }
      return rel;
    }
    if (scene.city) {
      throw new Error(
        "The minimap was built from the cloud field and handed a city " +
        "scene; the marker would be in the wrong frame.");
    }
    return camera.relativePosition();
  }

  /** The marker centre in screen pixels — for tests and diagnostics. */
  markerPixel(camera, size, scene) {
    const rect = rectForSize(size, this.albedoShape);
    const { cameraUV } = cameraOverlayGeometry(
      this._relativeIn(camera, scene), camera.azimuth, camera.elevation,
      camera.fov, this.albedoShape, size[0] / size[1]);
    return [rect[0] + cameraUV[0] * rect[2],
            rect[1] + (1.0 - cameraUV[1]) * rect[3]];
  }

  /**
   * Map a framebuffer-pixel point to world x/y, or null outside the map.
   * Inverse of the marker placement: map u = (rel_x + 1) / 2, and the map
   * is drawn north-up, so screen y runs against world y.
   */
  worldXYFromPixel(px, py, scene, camera = null) {
    if (!this._rect) return null;
    const [x, y, w, h] = this._rect;
    if (px < x || py < y || px > x + w || py > y + h) return null;
    const u = (px - x) / w;
    const v = 1.0 - (py - y) / h;
    if (this.frame === "city") {
      // The city map is ONE tile of a periodic city, so a point on it names
      // infinitely many world positions. The one meant is the nearest — a
      // click three streets over must not be a jump of a whole tile — so the
      // travel is measured from where the camera already is.
      if (!camera) {
        throw new Error(
          "A city map needs the camera to turn a click into a world " +
          "position: the tile repeats, and the nearest copy is the one " +
          "meant.");
      }
      const extent = scene.oceanTileExtent;
      const offset = scene.cityOffsetM;
      return [0, 1].map((i) => {
        const target = offset[i] + (i === 0 ? u : v) * extent;
        return target + Math.round((camera.position[i] - target) / extent)
               * extent;
      });
    }
    const { bmin, bmax } = scene;
    return [bmin[0] + u * (bmax[0] - bmin[0]),
            bmin[1] + v * (bmax[1] - bmin[1])];
  }

  /** Pack the layout and overlay geometry for this frame. */
  update(camera, scene, size, fullscreen = false, haze = null) {
    if (!Number.isFinite(haze)) {
      throw new Error(
        "The minimap needs the haze setting: the FOV rays end at the haze " +
        "e-folding distance, and without it there is nothing to end them at.");
    }
    const [screenW, screenH] = [Number(size[0]), Number(size[1])];
    const rect = rectForSize(size, this.albedoShape, fullscreen);
    const [ny, nx] = this.albedoShape;
    // The haze e-folding length as a map-UV radius per axis: how far the
    // view actually reaches, in the frame this map is drawn in. On the city
    // map that frame is the tile; on the cloud map it is the domain box.
    const hazeM = hazeEFoldingKm(haze) * 1000.0;
    const extentM = this.frame === "city"
      ? [scene.oceanTileExtent, scene.oceanTileExtent]
      : [scene.bmax[0] - scene.bmin[0], scene.bmax[1] - scene.bmin[1]];
    const hazeUV = [hazeM / extentM[0], hazeM / extentM[1]];
    const overlay = cameraOverlayGeometry(
      this._relativeIn(camera, scene), camera.azimuth, camera.elevation,
      camera.fov, this.albedoShape, screenW / screenH, hazeUV);
    const [camU, camV] = overlay.cameraUV;

    const minSide = Math.min(rect[2], rect[3]);
    // Deliberately understated: the marker is a reference, not a cursor.
    // Halved on 2026-08-14 (both the dot and the rays), which is as small as
    // the antialiasing lets them go and still be found: stroke_coverage fades
    // over one pixel past the half-width, so a 0.5 px half-width still paints
    // a ~2 px line. Going smaller stops shrinking the mark and just makes it
    // fainter. The map's own border is not part of that and is unchanged.
    const markerRadius = Math.max(1.25, minSide * 0.010);
    const lineWidth = Math.max(0.5, minSide * 0.00325);
    const borderWidth = Math.max(1.0, minSide * 0.008);
    const haloWidth = Math.max(0.6, lineWidth * 0.75);

    let mode = 0.0, circleRadius = 0.0;
    let leftU = 0.0, leftV = 0.0, rightU = 0.0, rightV = 0.0;
    if (overlay.fovEndpoints) {
      [[leftU, leftV], [rightU, rightV]] = overlay.fovEndpoints;
    } else {
      mode = 1.0;
      circleRadius = overlay.circleRadiusPx * rect[2] / Math.max(nx - 1, 1);
    }

    // The nest footprint is a rectangle in DOMAIN uv. On the city map that
    // is a rectangle over the wrong picture, so it is simply not drawn —
    // and a nested field over a city does not arise today anyway, since only
    // the demos are ever city-surfaced.
    const nest = this.frame === "city" ? null : nestMapUV(scene);
    const u = this.uniforms;
    u.set([screenW, screenH, fullscreen ? 0.95 : MAP_OPACITY, markerRadius], 0);
    u.set(rect, 4);
    u.set([camU, camV, mode, circleRadius], 8);
    u.set([leftU, leftV, rightU, rightV], 12);
    u.set([lineWidth, borderWidth, haloWidth, nest ? 1.0 : 0.0], 16);
    u.set(nest ?? [0, 0, 0, 0], 20);
    // The haze radius in screen pixels per axis (an ellipse when the map is
    // not square in metres) — what the shader shades the visible region with.
    u.set([hazeUV[0] * rect[2], hazeUV[1] * rect[3], 0.0, 0.0], 24);
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
