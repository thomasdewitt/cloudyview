// HDF5 compression filters that libhdf5 cannot decompress on its own.
//
// This exists because of a genuinely nasty failure. h5wasm's build carries
// exactly one external filter, DEFLATE. Meet a dataset compressed with
// anything else — blosc, zstd, lz4, bitshuffle — and H5Dread fails; but
// h5wasm's Dataset.slice() never checks H5Dread's return value, so it copies
// the unfilled malloc buffer straight out of the heap and hands it back. The
// length is right, so nothing downstream can tell. What you get is a field
// of zeros with occasional garbage bytes reinterpreting as infinities, which
// renders as an empty sky speckled with black pixels.
//
// The plugins are loadable .so blobs written into Emscripten's virtual
// filesystem before the read. They are fetched only when a file actually
// needs them, so the common case downloads nothing.

"use strict";

const PLUGIN_BASE = "../vendor/h5wasm-plugins/plugins";

/** Filters libhdf5 handles itself — no plugin, no download. */
export const BUILTIN_FILTER_IDS = new Set([
  1,   // deflate / gzip
  2,   // shuffle
  3,   // fletcher32
  4,   // szip
  5,   // nbit
  6,   // scaleoffset
]);

/** Registered HDF5 filter id to the plugin that implements it. */
export const PLUGIN_BY_FILTER_ID = new Map([
  [307, "bz2"],
  [32000, "lzf"],
  [32001, "blosc"],
  [32004, "lz4"],
  [32008, "bshuf"],
  [32013, "zfp"],
  [32015, "zstd"],
  [32019, "jpeg"],
  [32022, "bitgroom"],
  [32023, "bitround"],
  [32026, "blosc2"],
]);

export const ALL_PLUGINS = [
  "blosc", "blosc2", "bshuf", "bz2", "jpeg", "lz4", "lzf", "zfp", "zstd",
  "bitgroom", "bitround",
];

const asId = (filter) =>
  typeof filter === "number" ? filter : Number(filter?.id ?? NaN);

/**
 * Which plugins a set of datasets needs. An unrecognized filter id returns
 * the whole set rather than guessing — better a larger download than a
 * silently empty field.
 */
export function pluginsNeeded(filterLists) {
  const needed = new Set();
  let unknown = null;
  for (const filters of filterLists) {
    for (const filter of filters ?? []) {
      const id = asId(filter);
      if (!Number.isFinite(id) || BUILTIN_FILTER_IDS.has(id)) continue;
      const plugin = PLUGIN_BY_FILTER_ID.get(id);
      if (plugin) needed.add(plugin);
      else unknown ??= id;
    }
  }
  if (unknown !== null) return { plugins: ALL_PLUGINS, unknown };
  return { plugins: [...needed], unknown: null };
}

/**
 * Write the needed plugins into the wasm filesystem so libhdf5 can find
 * them. Idempotent; only fetches what is not already installed.
 */
export async function installPlugins(module, plugins, installed = new Set()) {
  const wanted = plugins.filter((name) => !installed.has(name));
  if (!wanted.length) return installed;

  const searchPath = module.get_plugin_search_paths?.()?.[0];
  if (!searchPath) {
    throw new Error(
      "This build of h5wasm has no plugin search path, so compressed data " +
      "using a filter beyond gzip cannot be read.");
  }
  module.FS.mkdirTree(searchPath);

  for (const name of wanted) {
    const url = new URL(`${PLUGIN_BASE}/libH5Z${name}.so`, import.meta.url);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Could not load the '${name}' decompression plugin ` +
        `(HTTP ${response.status}). This file needs it to be read at all.`);
    }
    module.FS.writeFile(`${searchPath}/libH5Z${name}.so`,
                        new Uint8Array(await response.arrayBuffer()));
    installed.add(name);
  }
  return installed;
}

/**
 * Refuse to read a dataset whose filters we cannot honour.
 *
 * Without this the read appears to succeed and returns zeros. Naming the
 * filter turns a mystifying empty render into one sentence.
 */
export function assertFiltersSupported(filters, installed, variableName) {
  for (const filter of filters ?? []) {
    const id = asId(filter);
    if (!Number.isFinite(id) || BUILTIN_FILTER_IDS.has(id)) continue;
    const plugin = PLUGIN_BY_FILTER_ID.get(id);
    if (plugin && installed.has(plugin)) continue;
    const label = filter?.name ? `'${filter.name}' (id ${id})` : `id ${id}`;
    throw new Error(
      `'${variableName}' is compressed with the HDF5 filter ${label}, which ` +
      "this build cannot decompress. Reading it would return zeros rather " +
      "than failing, so it is refused instead.");
  }
}
