// netCDF-4 semantics, ported from cloudyview/io.py.
//
// Deliberately pure: it takes an h5wasm-like handle and returns descriptions,
// so the same code runs in the ingest worker and under node in the tests. No
// DOM, no WebGPU, no h5wasm import.
//
// The rule followed throughout is that the browser must behave like the
// desktop tool, quirks included — a file that opens one way there and another
// way here is worse than one that refuses. Where Python gets a behaviour for
// free from xarray (fill values, scale/offset), it is reimplemented here,
// because h5wasm hands back raw storage.

"use strict";

// Order matters: first hit wins, and matching is exact-case, exactly as
// io.py does it. `qc` really does beat `QC`.
export const LIQUID_WATER_NAMES = [
  "qc", "QC", "ql", "QL", "QN", "qn", "LWC", "clw",
  "cloud_liquid_water_mixing_ratio", "liquid_water_content", "q_liquid", "lwc",
];
export const ICE_WATER_NAMES = [
  "qi", "QI", "qice", "QICE", "IWC", "cli",
  "cloud_ice_mixing_ratio", "ice_water_content", "q_ice", "iwc",
];
// Dimension matching, unlike variable matching, is case-insensitive.
export const AXIS_CANDIDATES = {
  x: ["x", "lon", "longitude", "nx", "ni"],
  y: ["y", "lat", "latitude", "ny", "nj"],
  z: ["z", "height", "altitude", "level", "nz", "nk"],
};

const isTimeDim = (name) => name.toLowerCase().includes("time");

/**
 * Every group in the file, root first, paths without a leading slash.
 * `visit(path, group)` is called for each.
 */
export function walkGroups(root, visit, prefix = "") {
  visit(prefix, root);
  for (const key of root.keys()) {
    const child = root.get(key);
    if (child && child.type === "Group") {
      walkGroups(child, visit, prefix ? `${prefix}/${key}` : key);
    }
  }
}

/**
 * Groups that hold something renderable: a variable named in
 * LIQUID_WATER_NAMES with at least three dimensions. A 1-D `qc` profile does
 * not qualify, and ice-only groups are invisible — same as the desktop.
 */
export function findLiquidWaterGroups(root) {
  const found = [];
  walkGroups(root, (path, group) => {
    const names = new Set(group.keys());
    for (const candidate of LIQUID_WATER_NAMES) {
      if (!names.has(candidate)) continue;
      const dataset = group.get(candidate);
      if (dataset?.shape?.length >= 3) { found.push(path); return; }
    }
  });
  return found;
}

/** First candidate present in the group, or null. Exact-case. */
export function inferVariable(group, candidates) {
  const names = new Set(group.keys());
  for (const candidate of candidates) {
    if (names.has(candidate)) return candidate;
  }
  return null;
}

/**
 * The dimension names of a dataset, in storage order.
 *
 * netCDF-4 records these as HDF5 dimension scales, so this is exact rather
 * than a guess. An axis with no scale attached comes back as null and falls
 * to the positional rules below.
 */
export function datasetDimNames(dataset) {
  const names = [];
  for (let axis = 0; axis < dataset.shape.length; axis++) {
    let name = null;
    try {
      const scales = dataset.get_attached_scales?.(axis) ?? [];
      if (scales.length) name = String(scales[0]).replace(/^.*\//, "");
    } catch { /* no scale on this axis */ }
    names.push(name);
  }
  return names;
}

/**
 * Map the storage dimensions onto x, y and z.
 *
 * Time dimensions are dropped first (any name containing "time"), leaving
 * exactly three. Then each axis takes the first unclaimed dimension whose
 * lowercased name matches one of its candidates, scanning in storage order;
 * a single leftover pair is matched positionally. Anything else is an error
 * rather than a guess — a field rendered on the wrong axes looks plausible
 * and is completely wrong.
 */
export function resolveSpatialDims(dimNames, shape) {
  const dims = [];
  const dropped = [];
  for (let i = 0; i < dimNames.length; i++) {
    const name = dimNames[i] ?? `dim_${i}`;
    if (isTimeDim(name)) {
      if (shape[i] > 1) {
        throw new Error(
          `This file has ${shape[i]} timesteps ('${name}'). Only ` +
          "single-timestep fields are supported — extract one first.");
      }
      dropped.push(i);
    } else {
      dims.push({ axis: i, name, size: shape[i] });
    }
  }
  if (dims.length !== 3) {
    throw new Error(
      `Expected 3 spatial dimensions, found ${dims.length} ` +
      `(${dims.map((d) => d.name).join(", ") || "none"}). ` +
      "Three-dimensional cloud data is required.");
  }
  for (const d of dims) {
    if (d.size < 2) {
      throw new Error(
        `Dimension '${d.name}' has only ${d.size} point. At least 2 points ` +
        "per spatial dimension are required.");
    }
  }

  const resolved = {};
  const claimed = new Set();
  for (const axis of ["x", "y", "z"]) {
    const candidates = AXIS_CANDIDATES[axis];
    for (const d of dims) {
      if (claimed.has(d.axis)) continue;
      if (candidates.includes(d.name.toLowerCase())) {
        resolved[axis] = d;
        claimed.add(d.axis);
        break;
      }
    }
  }
  const missing = ["x", "y", "z"].filter((a) => !(a in resolved));
  const spare = dims.filter((d) => !claimed.has(d.axis));
  if (missing.length === 1 && spare.length === 1) {
    resolved[missing[0]] = spare[0];
  } else if (missing.length) {
    throw new Error(
      `Could not tell which dimensions are x, y and z from ` +
      `(${dims.map((d) => d.name).join(", ")}). Recognized names are ` +
      `${Object.values(AXIS_CANDIDATES).flat().join(", ")}.`);
  }
  return { resolved, dropped };
}

/**
 * The 1-D coordinate array for one axis.
 *
 * Tried in io.py's order: the dimension's own variable, then the axis name
 * candidates, then any 1-D variable of the right length. That last rule is
 * loose enough to pick the wrong thing on a cubic grid — it is loose in the
 * desktop tool too, and diverging here would be a different kind of wrong.
 */
export function findCoordinate(group, root, dim) {
  const seen = [];
  const consider = (container, name) => {
    if (!container) return null;
    let variable = null;
    try { variable = container.get(name); } catch { return null; }
    if (!variable || variable.shape?.length !== 1) return null;
    if (variable.shape[0] !== dim.size) return null;
    return variable;
  };
  for (const container of [group, root]) {
    if (!container || seen.includes(container)) continue;
    seen.push(container);
    const direct = consider(container, dim.name);
    if (direct) return { name: dim.name, values: readNumeric(direct) };
    for (const candidate of AXIS_CANDIDATES.x.concat(
        AXIS_CANDIDATES.y, AXIS_CANDIDATES.z)) {
      const hit = consider(container, candidate);
      if (hit) return { name: candidate, values: readNumeric(hit) };
    }
  }
  for (const container of seen) {
    for (const key of container.keys()) {
      const hit = consider(container, key);
      if (hit) return { name: key, values: readNumeric(hit) };
    }
  }
  throw new Error(
    `No coordinate variable found for dimension '${dim.name}' ` +
    `(${dim.size} points). Cell-centre coordinates in metres are required ` +
    "for all three axes — they are what places the field in space.");
}

function readNumeric(variable) {
  return Float64Array.from(variable.value ?? variable.to_array());
}

/**
 * xarray applies these on open; h5wasm does not, so we must.
 *
 * Without the fill-value step a file that stores -9999 as "missing" renders
 * a wall of enormous negative extinction. Returns a decode function or null
 * when nothing needs doing (the common, fast case).
 */
export function decoderFor(attrs) {
  const value = (name) => {
    const attr = attrs?.[name];
    if (attr === undefined || attr === null) return null;
    const raw = attr.value ?? attr;
    return Array.isArray(raw) || ArrayBuffer.isView(raw)
      ? Number(raw[0]) : Number(raw);
  };
  const fill = value("_FillValue");
  const missing = value("missing_value");
  const scale = value("scale_factor");
  const offset = value("add_offset");
  if (fill === null && missing === null && scale === null && offset === null) {
    return null;
  }
  const s = scale ?? 1.0;
  const o = offset ?? 0.0;
  return (v) => {
    if ((fill !== null && v === fill) || (missing !== null && v === missing)) {
      return NaN;
    }
    return v * s + o;
  };
}

/**
 * The condensate unit conversion, and only the ones io.py accepts.
 *
 * Returns the multiplier taking the stored value to g/kg. An absent units
 * attribute is a question for the user, not a guess — hence null.
 */
export function unitsMultiplier(units) {
  if (units === null || units === undefined) return null;
  const u = String(units).trim().toLowerCase();
  if (u === "") return 1.0;          // SAM convention: empty means g/kg
  if (u === "g/kg") return 1.0;
  if (u === "g/g" || u === "kg/kg") return 1000.0;
  throw new Error(
    `Unsupported units '${units}'. Expected 'g/kg', 'g/g' or 'kg/kg'.`);
}

/** Read a string attribute, tolerating h5wasm's several shapes. */
export function attrString(attrs, name) {
  const attr = attrs?.[name];
  if (attr === undefined || attr === null) return null;
  const raw = attr.value ?? attr;
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw) || ArrayBuffer.isView(raw)) {
    return raw.length ? String(raw[0]) : null;
  }
  return String(raw);
}

/**
 * Everything needed to load one group, without reading any field data.
 * Cheap enough to run on every candidate group when offering a choice.
 */
export function describeGroup(root, path) {
  const group = path ? root.get(path) : root;
  const liquidVar = inferVariable(group, LIQUID_WATER_NAMES);
  if (!liquidVar) {
    throw new Error(
      `No cloud liquid water variable in ${path ? `group '${path}'` : "the root group"}. ` +
      `Looked for ${LIQUID_WATER_NAMES.join(", ")}.`);
  }
  const dataset = group.get(liquidVar);
  const iceVar = inferVariable(group, ICE_WATER_NAMES);
  const { resolved, dropped } = resolveSpatialDims(
    datasetDimNames(dataset), dataset.shape);

  const coords = {};
  for (const axis of ["x", "y", "z"]) {
    coords[axis] = findCoordinate(group, root, resolved[axis]);
  }

  const units = attrString(dataset.attrs, "units");
  return {
    path,
    liquidVar,
    iceVar,
    shape: [resolved.x.size, resolved.y.size, resolved.z.size],
    storageShape: dataset.shape,
    // Which storage axis each of x, y, z lives on — the permutation the
    // upload has to undo.
    storageAxis: { x: resolved.x.axis, y: resolved.y.axis, z: resolved.z.axis },
    droppedAxes: dropped,
    dimNames: { x: resolved.x.name, y: resolved.y.name, z: resolved.z.name },
    coords: {
      x: Array.from(coords.x.values),
      y: Array.from(coords.y.values),
      z: Array.from(coords.z.values),
    },
    units,
    unitsKnown: units !== null,
    chunks: dataset.metadata?.chunks ?? null,
    filters: (dataset.filters ?? []).map((f) => f.name ?? String(f.id)),
    dtype: dataset.dtype ?? null,
  };
}
