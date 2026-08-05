// Node harness for the Python<->JS uniform parity test.
//
// Reads a JSON payload of cases on stdin, runs the browser build's own
// packing code, and writes the results back on stdout. It exists so
// tests/test_web_uniform_parity.py can diff the browser's uniform block
// against engine.write_uniforms without a browser.
//
//   node tests/web_uniform_parity.mjs < cases.json > results.json

import { packUniforms, renderTargetSize, chooseQualityTier } from
  "../web/soar/uniforms.js";
import { spectralLightingColors, effectiveLightTransferSplit,
         directionFromAzimuthElevation } from "../web/soar/spectral.js";
import { cameraBasis, cameraWorldOrigin } from "../web/soar/camera.js";
import { volumeAABB, minVoxelSize, periodicMarchCapM, viewSpansDomainEdge,
         nestOverhang, nestablePairs } from "../web/soar/field.js";
import { resampleTrack } from "../web/soar/track.js";
import { sigmaAt, rhoAirAt, rhoAirTable, cellThickness,
         opticalDepthFromWaterPaths, twoStreamAlbedo } from
  "../web/soar/optical.js";

const input = JSON.parse(await new Promise((resolve, reject) => {
  let data = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => { data += chunk; });
  process.stdin.on("end", () => resolve(data));
  process.stdin.on("error", reject);
}));

const out = { uniforms: [], geometry: [], scalars: {} };

for (const proto of input.cases || []) {
  try {
    const u = packUniforms(proto.state, proto.view);
    out.uniforms.push(Array.from(u));
  } catch (err) {
    out.uniforms.push({ error: String(err.message || err) });
  }
}

for (const g of input.geometry || []) {
  const { bmin, bmax } = volumeAABB(g.x, g.y, g.z);
  out.geometry.push({
    bmin, bmax, minVoxel: minVoxelSize(g.shape, bmin, bmax),
  });
}

if (input.scalars) {
  out.scalars.renderTargetSizes = (input.scalars.renderTargetSizes || []).map(
    ([size, scale]) => renderTargetSize(size, scale));
  out.scalars.spectral = (input.scalars.spectral || []).map(
    ([azimuth, elevation, strength]) => {
      const dir = directionFromAzimuthElevation(azimuth, elevation);
      const c = spectralLightingColors(dir, undefined, strength);
      return { dir, ...c };
    });
  out.scalars.lightTransfer = (input.scalars.lightTransfer || []).map(
    ([strength, elevation]) => effectiveLightTransferSplit(strength, elevation));
  out.scalars.basis = (input.scalars.basis || []).map(
    ([azimuth, elevation]) => cameraBasis(azimuth, elevation));
  out.scalars.worldOrigin = (input.scalars.worldOrigin || []).map(
    ([rel, bmin, bmax]) => cameraWorldOrigin(rel, bmin, bmax));
  out.scalars.tiers = (input.scalars.tiers || []).map(
    (times) => chooseQualityTier(times));
  out.scalars.marchCap = (input.scalars.marchCap || []).map(
    ([camZ, direction, bmin, bmax]) =>
      periodicMarchCapM(camZ, direction, bmin, bmax));
  out.scalars.spansEdge = (input.scalars.spansEdge || []).map(
    ([origin, azimuth, elevation, fov, aspect, bmin, bmax]) =>
      viewSpansDomainEdge(
        origin, cameraBasis(azimuth, elevation), fov, aspect, bmin, bmax));
  out.scalars.nestOverhang = (input.scalars.nestOverhang || []).map(
    ([outerMin, outerMax, nestMin, nestMax]) =>
      nestOverhang(outerMin, outerMax, nestMin, nestMax));
  out.scalars.nestPairs = (input.scalars.nestPairs || []).map(
    (domains) => nestablePairs(domains));
  out.scalars.extinction = (input.scalars.extinction || []).map(
    ([lwc, iwc, z]) => sigmaAt(lwc, iwc, rhoAirAt(z)));
  // The minimap's column integral, exactly as the ingest worker accumulates
  // it: water paths first, then glimpse's empirical path-to-tau relations,
  // then two-stream albedo. NOT a reduction of the extinction above.
  out.scalars.columnAlbedo = (input.scalars.columnAlbedo || []).map(
    ([lwc, iwc, z]) => {
      const rho = rhoAirTable(z);
      const dz = cellThickness(z);
      let lwp = 0.0, iwp = 0.0;
      for (let k = 0; k < z.length; k++) {
        lwp += lwc[k] * rho[k] * dz[k];
        iwp += iwc[k] * rho[k] * dz[k];
      }
      return twoStreamAlbedo(opticalDepthFromWaterPaths(lwp, iwp));
    });
  out.scalars.trackResample = (input.scalars.trackResample || []).map(
    ([samples, fps, periodic]) =>
      resampleTrack(samples, fps, { periodic }).map(
        (f) => [f.t, ...f.position, f.azimuth, f.elevation, f.fov]));
}

process.stdout.write(JSON.stringify(out));
