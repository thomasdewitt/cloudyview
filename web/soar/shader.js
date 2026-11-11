// The soar shader's source: WGSL files under shader/, joined per surface.
//
// shader/manifest.json lists the files every module shares and, for each
// surface the world can stand on, the files that draw it and the `absent`
// stand-ins that take their place in a module compiled for another surface.
// A module therefore holds the code of exactly one surface. The Python host
// builds the same text (cloudyview/soar/shader.py); a test pins the two.

"use strict";

/** The files of `surface`'s module, in the order they are joined. */
export function shaderFiles(manifest, surface) {
  const entry = manifest.surfaces[surface];
  if (!entry) {
    throw new Error(
      `unknown surface "${surface}"; the shader has ` +
      `${Object.keys(manifest.surfaces).join(", ")}.`);
  }
  const files = [...manifest.core, ...entry.files];
  for (const [other, e] of Object.entries(manifest.surfaces)) {
    if (other !== surface) files.push(...e.absent);
  }
  return files;
}

/** `surface`'s module, without the switch header, from `texts[file]`. */
export function assembleShader(manifest, texts, surface) {
  return shaderFiles(manifest, surface).map((f) => {
    if (typeof texts[f] !== "string") {
      throw new Error(`shader file ${f} was not loaded.`);
    }
    return texts[f];
  }).join("\n");
}

/**
 * Fetch the manifest under `baseUrl` and every file it names. Returns each
 * surface's module source and the lens-water source the present pass
 * includes: `{ modules: { ocean: "...", city: "..." }, lens: "..." }`.
 *
 * A deployed manifest carries a `version` (tools/stage_deploy.py stamps a
 * hash of the shader files into it). The manifest itself is revalidated on
 * every load and the version rides each file's URL, so a cached shader file
 * from an earlier deploy is never joined with this one's.
 */
export async function loadShaders(baseUrl) {
  const get = async (name, init) => {
    const response = await fetch(new URL(name, baseUrl), init);
    if (!response.ok) {
      throw new Error(`could not load shader/${name}: HTTP ${response.status}`);
    }
    return response.text();
  };
  const manifest = JSON.parse(await get("manifest.json", { cache: "no-cache" }));
  const version = manifest.version ? `?v=${manifest.version}` : "";
  const surfaces = Object.keys(manifest.surfaces);
  const names = [...new Set(surfaces.flatMap((s) => shaderFiles(manifest, s))),
                 manifest.lens];
  const bodies = await Promise.all(names.map((n) => get(n + version)));
  const texts = Object.fromEntries(names.map((n, i) => [n, bodies[i]]));
  return {
    modules: Object.fromEntries(
      surfaces.map((s) => [s, assembleShader(manifest, texts, s)])),
    lens: texts[manifest.lens],
  };
}

/**
 * A compilable module: the compile-time switch header uniforms.wgsl
 * describes, then `source`.
 *
 * The lighting method has no default on purpose: it is the same pair of
 * flags packUniforms writes into the uniform block, and a caller that forgot
 * to say which method it compiled would render one method under the other's
 * scene key.
 */
export function specializeShader(source,
                                 { periodic, nested, toneMap = true,
                                   lightCache, skyProbe }) {
  if (typeof lightCache !== "boolean" || typeof skyProbe !== "boolean") {
    throw new Error(
      "specializeShader needs the lighting method: lightCache and skyProbe " +
      `as booleans; got ${lightCache} and ${skyProbe}.`);
  }
  const switches = {
    PERIODIC_DOMAIN: periodic, NESTED: nested, TONE_MAP: toneMap,
    LIGHT_CACHE: lightCache, SKY_PROBE: skyProbe,
  };
  const header = Object.entries(switches)
    .map(([name, on]) => `const ${name}: bool = ${on ? "true" : "false"};\n`)
    .join("");
  return header + source;
}

/**
 * The one cache key for a compiled module: its surface and every switch, in
 * a fixed order. Every module and pipeline cache keys on it (plus the entry
 * point and target format).
 */
export function shaderVariantKey(spec) {
  return [spec.periodic, spec.nested, spec.toneMap, spec.surface,
          spec.lightCache, spec.skyProbe].join("|");
}
