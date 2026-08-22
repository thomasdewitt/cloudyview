// netCDF-4 semantics. Originally ported from cloudyview/io.py.
//
// Deliberately pure: it takes an h5wasm-like handle and returns descriptions,
// so the same code runs in the ingest worker and under node in the tests. No
// DOM, no WebGPU, no h5wasm import.
//
// Where Python gets a behaviour for free from xarray (fill values,
// scale/offset), it is reimplemented here, because h5wasm hands back raw
// storage.
//
// This module used to say it must behave like the desktop tool, quirks
// included. It no longer does, and the divergence is deliberate rather than
// drift: this side can ASK — which axis is which, which variable is the
// condensate, which timestep — and io.py, on a terminal, cannot. So the rules
// here are the richer ones, and what the browser resolves is meant to come
// back out as CLI flags (see viewer.beholdCommand / witnessCommand) rather
// than be independently re-derived by a weaker copy of this file.

"use strict";

import { T } from "./strings.js";

// Order matters: first hit wins, and matching is exact-case, exactly as
// io.py does it. `qc` really does beat `QC`.
//
// `QN`/`qn` are deliberately ABSENT. SAM writes QN for total
// non-precipitating condensate — cloud water AND cloud ice together (Thomas,
// 2026-08-22) — so it is not the liquid variable, and inferring it as one
// renders every ice cloud as water. It was in this list, with a special case
// bolted alongside to force a question whenever it turned up; the special
// case is gone and so is the entry, which is the same intent expressed once.
// A SAM run whose only condensate is QN now fails to infer and asks, with QN
// among the variables offered — so it can still be chosen, by someone who
// knows what their run wrote.
export const LIQUID_WATER_NAMES = [
  "qc", "QC", "ql", "QL", "LWC", "clw",
  "cloud_liquid_water_mixing_ratio", "liquid_water_content", "q_liquid", "lwc",
];
export const ICE_WATER_NAMES = [
  "qi", "QI", "qice", "QICE", "IWC", "cli",
  "cloud_ice_mixing_ratio", "ice_water_content", "q_ice", "iwc",
];

// Dimension matching, unlike variable matching, is case-insensitive.
//
// Beyond the bare axis letters this covers:
//   - the staggered/centred suffixes t, s, h and c. SAM writes cell centres
//     as xt/yt/zt and cell edges as xs/ys/zs; MPAS/CM1 use xh/zh for centres
//     and some tools write xc/yc. All twelve combinations are listed rather
//     than derived from a regex, so that adding a suffix is a visible edit
//     and not a rule that quietly starts matching something new.
//   - WRF's spelled-out dimensions and their _stag variants.
// A name absent from here is not a failure — it falls to the coordinate
// metadata rules below (see axisFromAttrs), and only then to position.
export const AXIS_CANDIDATES = {
  x: ["x", "xt", "xs", "xh", "xc", "lon", "longitude", "nx", "ni",
      "west_east", "west_east_stag"],
  y: ["y", "yt", "ys", "yh", "yc", "lat", "latitude", "ny", "nj",
      "south_north", "south_north_stag"],
  z: ["z", "zt", "zs", "zh", "zc", "height", "altitude", "level", "lev",
      "nz", "nk", "bottom_top", "bottom_top_stag", "plev", "pressure",
      "model_level_number"],
};

const STANDARD_NAME_AXIS = {
  longitude: "x", grid_longitude: "x", projection_x_coordinate: "x",
  latitude: "y", grid_latitude: "y", projection_y_coordinate: "y",
  height: "z", altitude: "z", air_pressure: "z",
  model_level_number: "z", atmosphere_hybrid_sigma_pressure_coordinate: "z",
  atmosphere_sigma_coordinate: "z",
  height_above_mean_sea_level: "z",
  height_above_reference_ellipsoid: "z",
};

// Units that name an axis on their own.
//
// The horizontal ones are unambiguous: nothing but a longitude is in
// degrees_east. The vertical ones are NOT — a SAM x axis is in metres too —
// which is why units are consulted only for axes still unclaimed after names
// and after the `axis`/`standard_name` attributes, and why two dimensions
// both claiming z by units is reported as an ambiguity rather than resolved
// by whichever came first.
const UNITS_AXIS = {
  degrees_east: "x", degree_east: "x", degrees_e: "x", degree_e: "x",
  degrees_north: "y", degree_north: "y", degrees_n: "y", degree_n: "y",
  m: "z", metre: "z", metres: "z", meter: "z", meters: "z",
  km: "z", kilometre: "z", kilometres: "z", kilometer: "z", kilometers: "z",
  pa: "z", hpa: "z", mb: "z", millibar: "z", millibars: "z",
  level: "z", levels: "z", sigma: "z", "1": "z",
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
 * How many of a dataset's dimensions are spatial — i.e. not time.
 *
 * The test for "could this be a field" throughout. A (time, y, x, z) variable
 * has four dimensions and three spatial ones; a (time, y, x) variable has
 * three and two, and is not a volume however it is named.
 */
export function spatialRank(dataset) {
  const shape = dataset?.shape;
  if (!shape?.length) return 0;
  const names = datasetDimNames(dataset);
  let n = 0;
  for (let i = 0; i < shape.length; i++) {
    if (!isTimeDim(names[i] ?? `dim_${i}`)) n += 1;
  }
  return n;
}

/**
 * EVERY three-dimensional variable in the group, in file order, with what the
 * chooser needs to describe it.
 *
 * This is what the variable question is asked from, and it is deliberately
 * not filtered by name. The old chooser offered only names already in the
 * condensate lists, which meant a file whose water is called something else
 * had nothing to offer and failed instead of asking — the whole reason a
 * temperature-only file died on an h5wasm error rather than a question.
 */
export function listVariables(group) {
  const found = [];
  for (const key of group.keys()) {
    let dataset = null;
    try { dataset = group.get(key); } catch { continue; }
    if (!dataset || dataset.type === "Group") continue;
    if (spatialRank(dataset) < 3) continue;
    found.push({
      name: key,
      shape: Array.from(dataset.shape),
      units: attrString(dataset.attrs, "units"),
      longName: attrString(dataset.attrs, "long_name")
             ?? attrString(dataset.attrs, "description"),
    });
  }
  return found;
}

/**
 * Groups holding at least one three-dimensional variable.
 *
 * NOT "groups holding a recognized condensate name", which is what this used
 * to be. That test doubled as the test for whether the file was openable at
 * all, so a file whose variables are named something the lists do not carry
 * was refused before anything could be asked about it. Openability and
 * recognition are now separate questions: this one decides whether there is
 * anything here to render, and inference decides whether we know which of it
 * is which.
 */
export function findVolumeGroups(root) {
  const found = [];
  walkGroups(root, (path, group) => {
    if (listVariables(group).length) found.push(path);
  });
  return found;
}

/**
 * First candidate present in the group as a 3-D variable, or null.
 *
 * Exact-case, first hit wins, and a miss is not an error — it is the question
 * the chooser exists to ask. There are no special cases: a name in the list
 * is taken, a name not in the list is not, and anything else is the user's
 * call rather than a rule encoded here.
 */
export function inferCondensate(group, candidates) {
  const names = new Set(group.keys());
  for (const candidate of candidates) {
    if (!names.has(candidate)) continue;
    let dataset = null;
    try { dataset = group.get(candidate); } catch { continue; }
    if (spatialRank(dataset) >= 3) return candidate;
  }
  return null;
}

/**
 * What each spatial dimension's own coordinate variable says about itself.
 *
 * Keyed by STORAGE AXIS INDEX, which is what resolveSpatialDims matches on.
 * A dimension with no like-named variable, or one whose attributes say
 * nothing, simply has no entry — that is the common case for the files that
 * already work, where the name alone settles it.
 */
export function dimHints(group, root, dimNames) {
  const hints = {};
  for (let i = 0; i < dimNames.length; i++) {
    const name = dimNames[i];
    if (!name) continue;
    for (const container of [group, root]) {
      if (!container) continue;
      let variable = null;
      try { variable = container.get(name); } catch { continue; }
      if (!variable || variable.shape?.length !== 1) continue;
      const hint = axisFromAttrs(variable.attrs);
      if (hint) { hints[i] = hint; break; }
    }
  }
  return hints;
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
 * Which axis a coordinate variable's own metadata claims, or null.
 *
 * CF gives three ways of saying it and they are not equally trustworthy, so
 * they are returned tagged with which rule fired and how strong it is:
 *   "axis"          — the `axis` attribute, X/Y/Z. Says exactly this.
 *   "standard_name" — a CF standard name that only one axis can carry.
 *   "units"         — weakest. degrees_east/north pin the horizontal; metres,
 *                     pascals and bare level counts suggest the vertical, but
 *                     a Cartesian x is in metres too.
 * Null when the attributes say nothing about which axis this is.
 */
export function axisFromAttrs(attrs) {
  const axis = attrString(attrs, "axis");
  if (axis) {
    const a = axis.trim().toLowerCase();
    if (a === "x" || a === "y" || a === "z") return { axis: a, rule: "axis" };
  }
  const standard = attrString(attrs, "standard_name");
  if (standard) {
    const a = STANDARD_NAME_AXIS[standard.trim().toLowerCase()];
    if (a) return { axis: a, rule: "standard_name" };
  }
  const units = attrString(attrs, "units");
  if (units) {
    const a = UNITS_AXIS[units.trim().toLowerCase()];
    if (a) return { axis: a, rule: "units" };
  }
  return null;
}

/** The error the manual-assignment panel is built from. */
function axisChoiceError(message, dims, assumptions) {
  const err = new Error(message);
  // The payload that turns a dead end into a question: which dimensions are
  // on offer, in storage order, so the panel can list them by name and size.
  err.axisChoice = {
    dims: dims.map((d) => ({ axis: d.axis, name: d.name, size: d.size })),
    assumptions,
  };
  return err;
}

/**
 * Map the storage dimensions onto x, y and z.
 *
 * Time dimensions are dropped first (any name containing "time"), leaving
 * exactly three. The rest is tried in this order, and the order is the whole
 * point — each rule is weaker than the one above it, and the first one that
 * settles an axis keeps it:
 *
 *   1. `override` — what the user picked in the manual-assignment panel.
 *      Nothing else runs; a person who has said which axis is which is not
 *      then second-guessed by a units attribute.
 *   2. NAME, case-insensitively, against AXIS_CANDIDATES: the bare letters,
 *      the t/s/h/c-suffixed SAM and CM1 spellings (xt/yt/zt, xs/ys/zs, …),
 *      lon/lat, the n-prefixed sizes, and WRF's west_east/south_north/
 *      bottom_top with their _stag variants.
 *   3. COORDINATE METADATA, for the axes still unclaimed — `axis` = X/Y/Z
 *      first, then `standard_name`, then units (see axisFromAttrs). Applied
 *      strongest-rule-first across all three axes at once, so a file where
 *      one dimension declares `axis = "Z"` and another merely has metres does
 *      not let the second one take z out from under the first.
 *   4. The leftover pair: two axes down and one dimension left over, that
 *      dimension is the missing axis. No assumption is involved.
 *   5. POSITION — (z, y, x) in C order, which is what netCDF conventionally
 *      writes. Allowed, because refusing a file whose dimensions are simply
 *      unnamed helps nobody, but never silent: it is recorded in
 *      `assumptions` and the caller states it on screen.
 *
 * Anything still unsettled, or settled two contradictory ways, throws an
 * error carrying `.axisChoice` so the caller can ask instead of giving up.
 *
 * Returns `{ resolved, dropped, timeDims, assumptions }`. `assumptions` is a
 * list of sentences about guesses that were made — empty when every axis was
 * named or declared outright. `timeDims` is what the timestep question is
 * asked from; `dropped` is the same axes as bare indices.
 */
export function resolveSpatialDims(dimNames, shape,
                                   { hints = null, override = null } = {}) {
  const dims = [];
  const dropped = [];
  const timeDims = [];
  for (let i = 0; i < dimNames.length; i++) {
    const name = dimNames[i] ?? `dim_${i}`;
    if (isTimeDim(name)) {
      // Dropped from the spatial mapping either way; WHICH step is dropped
      // to is the caller's question (see describeGroup's `timeDim`). A
      // multi-step file used to be refused outright here.
      dropped.push(i);
      timeDims.push({ axis: i, name, size: shape[i] });
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
  const assumptions = [];
  const take = (axis, d) => { resolved[axis] = d; claimed.add(d.axis); };

  // 1. The user's own assignment, if there is one.
  if (override) {
    for (const axis of ["x", "y", "z"]) {
      const d = dims.find((q) => q.axis === override[axis]);
      if (!d) {
        throw new Error(
          `The manual axis assignment names storage axis ${override[axis]} ` +
          `for ${axis}, which is not one of this variable's spatial ` +
          `dimensions (${dims.map((q) => q.axis).join(", ")}).`);
      }
      if (claimed.has(d.axis)) {
        throw new Error(
          `The manual axis assignment uses '${d.name}' for more than one of ` +
          "x, y and z. Each axis needs its own dimension.");
      }
      take(axis, d);
    }
    return { resolved, dropped, timeDims, assumptions };
  }

  // 2. Name.
  for (const axis of ["x", "y", "z"]) {
    const candidates = AXIS_CANDIDATES[axis];
    for (const d of dims) {
      if (claimed.has(d.axis)) continue;
      if (candidates.includes(d.name.toLowerCase())) { take(axis, d); break; }
    }
  }

  // 3. Coordinate metadata, strongest rule first. A rule that would give one
  // axis to two dimensions is not applied at all — that is exactly the
  // ambiguity the chooser exists for, and picking the first would be a guess
  // dressed as a detection.
  if (hints) {
    for (const rule of ["axis", "standard_name", "units"]) {
      const byAxis = new Map();
      for (const d of dims) {
        if (claimed.has(d.axis)) continue;
        const hint = hints[d.axis];
        if (!hint || hint.rule !== rule) continue;
        if (resolved[hint.axis]) continue;
        if (!byAxis.has(hint.axis)) byAxis.set(hint.axis, []);
        byAxis.get(hint.axis).push(d);
      }
      for (const [axis, matches] of byAxis) {
        if (matches.length !== 1) continue;   // ambiguous: leave it unclaimed
        take(axis, matches[0]);
        assumptions.push(T.axisFromAttribute(axis, matches[0].name, rule));
      }
    }
  }

  // 4. The leftover pair.
  let missing = ["x", "y", "z"].filter((a) => !(a in resolved));
  let spare = dims.filter((d) => !claimed.has(d.axis));
  if (missing.length === 1 && spare.length === 1) {
    take(missing[0], spare[0]);
    missing = [];
    spare = [];
  }

  // 5. Position, and only when NOTHING was settled. A partial name match plus
  // positional filling of the rest would be the worst of both: the C-order
  // convention is about the whole tuple, and applying it to the leftovers of
  // a different rule is not the convention, it is a coin toss.
  if (missing.length === 3 && spare.length === 3) {
    const order = ["z", "y", "x"];            // C order, slowest axis first
    for (let i = 0; i < 3; i++) take(order[i], spare[i]);
    assumptions.push(T.axesByPosition(spare.map((d) => d.name), order));
    missing = [];
  }

  if (missing.length) {
    throw axisChoiceError(
      `Could not tell which dimensions are ${missing.join(", ")} from ` +
      `(${dims.map((d) => d.name).join(", ")}). Names recognized directly ` +
      `are ${Object.values(AXIS_CANDIDATES).flat().join(", ")}; failing ` +
      "that, a coordinate variable's axis, standard_name or units " +
      "attribute is used.",
      dims, assumptions);
  }
  return { resolved, dropped, timeDims, assumptions };
}

/**
 * netCDF-4 writes a placeholder HDF5 dataset for a dimension that has no
 * coordinate variable of its own, and marks it with this NAME. It is all
 * zeros — a dimension scale to hang DIMENSION_LIST references on, not data.
 *
 * Recognizing it is not a nicety. CM1 output dimensions its fields
 * (nk, nj, ni) and puts the real coordinates in x, y and z, so every axis
 * has BOTH a same-named placeholder and a real coordinate elsewhere. Taking
 * the placeholder gives three all-zero coordinate arrays, which is a
 * bounding box of zero size, which renders as an empty sky with no ocean in
 * it (Thomas, 2026-08-22).
 */
export function isPhonyDimensionScale(variable) {
  const name = attrString(variable?.attrs, "NAME");
  return Boolean(name && name.startsWith(
    "This is a netCDF dimension but not a netCDF variable"));
}

/** Metres per unit of a coordinate, or null when it is not a length at all. */
const LENGTH_UNITS = new Map([
  ["m", 1], ["metre", 1], ["metres", 1], ["meter", 1], ["meters", 1],
  ["km", 1000], ["kilometre", 1000], ["kilometres", 1000],
  ["kilometer", 1000], ["kilometers", 1000],
]);

export function metresPerUnit(units) {
  if (units === null || units === undefined) return null;
  return LENGTH_UNITS.get(String(units).trim().toLowerCase()) ?? null;
}

/**
 * The 1-D coordinate array for one axis, in metres.
 *
 * Every 1-D variable of the right length is collected, in order of how well
 * it claims the axis:
 *
 *   1. The dimension's own variable — the CF coordinate-variable convention,
 *      and the strongest signal there is.
 *   2. This axis's candidate names. THIS axis's: the sweep used to run the
 *      x, y and z lists concatenated, which on a cubic grid could hand z the
 *      variable called `x`.
 *   3. Any 1-D variable of the right length. Loose, and last.
 *
 * Placeholder dimension scales are excluded throughout — see
 * isPhonyDimensionScale, which is what CM1 output turns on.
 *
 * Then one override: a first choice that is NOT a length loses to a length
 * further down the list. UM output dimensions its fields by
 * `rholev_eta_rho`, a dimensionless hybrid-height coordinate running 0 to 1,
 * and carries the actual height in `rholev_zsea_rho` — so rule 1 produced a
 * domain 6000 km wide and 0.99 m tall (Thomas, 2026-08-22). A coordinate
 * that cannot be a distance cannot place a field in space, whatever its name
 * says.
 *
 * Returns `{ name, values, note }`; `note` is non-empty when a choice was
 * made that the load should say out loud.
 */
export function findCoordinate(group, root, dim, axis) {
  const containers = [];
  for (const c of [group, root]) if (c && !containers.includes(c)) containers.push(c);

  const found = [];
  const consider = (container, name) => {
    let variable = null;
    try { variable = container.get(name); } catch { return; }
    if (!variable || variable.shape?.length !== 1) return;
    if (variable.shape[0] !== dim.size) return;
    if (isPhonyDimensionScale(variable)) return;
    if (found.some((f) => f.name === name)) return;
    found.push({ name, variable, units: attrString(variable.attrs, "units") });
  };

  for (const container of containers) consider(container, dim.name);
  for (const container of containers) {
    for (const candidate of AXIS_CANDIDATES[axis]) consider(container, candidate);
  }
  for (const container of containers) {
    for (const key of container.keys()) consider(container, key);
  }

  if (!found.length) {
    throw new Error(
      `No coordinate variable found for dimension '${dim.name}' ` +
      `(${dim.size} points). Cell-center coordinates in meters are required ` +
      "for all three axes — they are what places the field in space.");
  }

  let chosen = found[0];
  const notes = [];
  if (metresPerUnit(chosen.units) === null) {
    const measured = found.find((f) => metresPerUnit(f.units) !== null);
    if (measured) {
      notes.push(T.coordChosen(axis, measured.name));
      chosen = measured;
    }
  }

  const scale = metresPerUnit(chosen.units);
  let values = readNumeric(chosen.variable);
  if (scale !== null && scale !== 1) {
    // Everything downstream — the AABB, the voxel sizes, the march — is in
    // metres. A coordinate in km left unconverted is a domain a thousand
    // times too small, which is the same failure as the eta one above.
    values = values.map((v) => v * scale);
    notes.push(T.coordConverted(axis));
  }
  return { name: chosen.name, values, note: notes.join(" ") };
}

function readNumeric(variable) {
  return Float64Array.from(variable.value ?? variable.to_array());
}

/**
 * A named 1-D variable's values, or null when there isn't one.
 *
 * Unlike findCoordinate this never searches and never throws: it is for
 * labelling a question (which timestep is which), not for placing a field in
 * space, and a file with no time variable is perfectly ordinary.
 */
export function readCoordValues(group, root, name, size) {
  for (const container of [group, root]) {
    if (!container) continue;
    let variable = null;
    try { variable = container.get(name); } catch { continue; }
    if (!variable || variable.shape?.length !== 1) continue;
    if (variable.shape[0] !== size) continue;
    try { return Array.from(readNumeric(variable)); } catch { return null; }
  }
  return null;
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

/**
 * Refuse to pair an ice field from a second file with a water field unless
 * the two are the same grid.
 *
 * The read loop takes one hyperslab per tile and uses the SAME ranges,
 * strides and flat index for both variables (see worker.js `extinction`).
 * Within one file describeGroup already enforces that; across two files
 * nothing does, and the failure mode is silent: an ice array of the same
 * total size but a different axis order reads ice from the wrong cells and
 * still renders a completely plausible cloud.
 *
 * So this is an equality test, not a compatibility test. No crop, no
 * transpose, no interpolation is attempted — a mismatched pair is reported
 * with the actual numbers and the user regrids it themselves, because every
 * one of those repairs is a choice about their data that this tool has no
 * standing to make.
 *
 * Coordinates are compared where BOTH files carry them, to a relative
 * tolerance: they are floats that have usually been through a text format.
 */
export function assertSameGrid(water, ice, iceFilename) {
  const differences = [];
  const say = (what, a, b) => differences.push(`${what}: ${a} against ${b}`);
  if (String(ice.shape) !== String(water.shape)) {
    say("spatial shape (x, y, z)", `[${ice.shape}]`, `[${water.shape}]`);
  }
  if (String(ice.storageShape) !== String(water.storageShape)) {
    say("stored shape", `[${ice.storageShape}]`, `[${water.storageShape}]`);
  }
  for (const axis of ["x", "y", "z"]) {
    if (ice.storageAxis[axis] !== water.storageAxis[axis]) {
      say(`which storage axis is ${axis}`,
          ice.storageAxis[axis], water.storageAxis[axis]);
    }
  }
  if (String(ice.droppedAxes) !== String(water.droppedAxes)) {
    say("dropped (time) axes", `[${ice.droppedAxes}]`, `[${water.droppedAxes}]`);
  }
  if (ice.timestep !== water.timestep) {
    say("timestep", ice.timestep, water.timestep);
  }
  for (const axis of ["x", "y", "z"]) {
    const a = ice.coords?.[axis], b = water.coords?.[axis];
    if (!a || !b || a.length !== b.length) continue;   // shape already said so
    // Relative to the axis's own extent, so a 20 km domain and a 20 m one
    // are held to the same standard rather than the same absolute metre.
    const span = Math.abs(b[b.length - 1] - b[0]) || 1;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > 1e-6 * span) {
        say(`the ${axis} coordinate at index ${i}`, a[i], b[i]);
        break;
      }
    }
  }
  if (differences.length) {
    throw new Error(
      `'${iceFilename}' is not on the same grid as the field already ` +
      `loaded (${differences.join("; ")}). cloudyview reads the two ` +
      "variables with one set of indices, so they have to be stored " +
      "identically — regrid the ice file onto the water file's grid first. " +
      "Nothing here is cropped or interpolated to make them fit.");
  }
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
export function describeGroup(root, path, choice = null) {
  const group = path ? root.get(path) : root;
  const where = path ? `group '${path}'` : "the root group";

  // Everything in this group that could be a field, named or not. The
  // chooser is offered from here, so it is not filtered by the condensate
  // lists — those decide what can be INFERRED, not what can be picked.
  const variables = listVariables(group);
  if (!variables.length) {
    throw new Error(`No three-dimensional variable in ${where}.`);
  }
  const has = (name) => variables.some((v) => v.name === name);

  // `choice.liquidVar`/`choice.iceVar` set to null is a real answer — "none
  // of these" — and must not read as "nothing was asked". Hence `in` rather
  // than `??` throughout.
  const answered = (role) => Boolean(choice && role in choice);

  const liquidVar = answered("liquidVar")
    ? choice.liquidVar
    : inferCondensate(group, LIQUID_WATER_NAMES);
  if (liquidVar && !has(liquidVar)) {
    throw new Error(
      `'${liquidVar}' was chosen as the liquid condensate but ${where} has ` +
      "no such three-dimensional variable.");
  }

  // Inference failed and nobody has answered yet. Nothing below this point
  // can be worked out — dimensions, coordinates and units all come off the
  // liquid variable — so the description stops here and says what it needs.
  // This is the path a file of temperature takes, and it used to be an
  // error thrown out of h5wasm rather than a question.
  if (!liquidVar) {
    return { path, variables, liquidVar: null, iceVar: null,
             needsLiquidChoice: true, needsIceChoice: false,
             needsTimestepChoice: false, assumptions: [],
             // Provisional, and for the question's wording only: it says
             // "inferred the ice, could not infer the liquid" rather than
             // the blanker "could not infer variables". Not validated
             // against the liquid variable's layout — there isn't one yet —
             // so nothing may read it as the answer.
             inferredIce: inferCondensate(group, ICE_WATER_NAMES) };
  }
  const dataset = group.get(liquidVar);

  const iceVar = answered("iceVar")
    ? choice.iceVar
    : inferCondensate(group, ICE_WATER_NAMES);
  if (iceVar && !has(iceVar)) {
    throw new Error(
      `'${iceVar}' was chosen as the ice condensate but ${where} has no ` +
      "such three-dimensional variable.");
  }
  // Asked whenever inference did not settle it. There is no list of names
  // that makes a miss safe to take silently: a field rendered without its
  // ice is a different picture, and one rendered with the wrong variable as
  // ice is a plausible-looking lie.
  const needsIceChoice = !answered("iceVar") && iceVar === null;

  const dimNames = datasetDimNames(dataset);
  const { resolved, dropped, timeDims, assumptions } = resolveSpatialDims(
    dimNames, dataset.shape,
    { hints: dimHints(group, root, dimNames), override: choice?.axes ?? null });

  // Which timestep, when there is more than one. Two time dimensions both
  // longer than one step is not a file this can describe — there is no
  // single index to ask for.
  const manyStepped = timeDims.filter((d) => d.size > 1);
  if (manyStepped.length > 1) {
    throw new Error(
      `${where} has more than one time dimension with several steps ` +
      `(${manyStepped.map((d) => `${d.name}=${d.size}`).join(", ")}).`);
  }
  const timeDim = manyStepped[0] ?? null;
  const timestep = answered("timestep") ? choice.timestep : 0;
  if (timeDim && !(timestep >= 0 && timestep < timeDim.size)) {
    throw new Error(
      `Timestep ${timestep} is out of range for '${timeDim.name}' ` +
      `(${timeDim.size} steps).`);
  }
  // Storage axis -> the index the read pins it at. Every dropped axis is in
  // here, so the read never has to know which one was the chosen one.
  const timeSelect = {};
  for (const d of timeDims) timeSelect[d.axis] = d === timeDim ? timestep : 0;

  // The ice variable is read with the liquid variable's ranges, strides and
  // flat index — one sweep, one hyperslab pair, one loop. That is only
  // correct if it is laid out identically, which nothing guarantees: io.py
  // standardizes each variable separately and would simply transpose. Here a
  // mismatch would silently read ice from the wrong cells, so it is checked
  // rather than assumed. (Refused rather than rendered liquid-only: dropping
  // the ice changes the picture, and a cirrus deck vanishing quietly is worse
  // than a message saying why.)
  const iceDataset = iceVar ? group.get(iceVar) : null;
  let iceUnits = null;
  if (iceDataset) {
    const iceDimNames = datasetDimNames(iceDataset);
    // Resolved with the same hints and the same override as the liquid
    // variable: a manual assignment that applied to one and not the other
    // would make the two disagree by construction and report it as the
    // file's fault.
    const ice = resolveSpatialDims(
      iceDimNames, iceDataset.shape,
      { hints: dimHints(group, root, iceDimNames),
        override: choice?.axes ?? null });
    const differences = [];
    if (String(iceDataset.shape) !== String(dataset.shape)) {
      differences.push(
        `stored shape [${iceDataset.shape}] against [${dataset.shape}]`);
    }
    for (const axis of ["x", "y", "z"]) {
      const a = ice.resolved[axis], b = resolved[axis];
      if (a.axis !== b.axis || a.size !== b.size) {
        differences.push(
          `${axis} is storage axis ${a.axis} of length ${a.size} against ` +
          `axis ${b.axis} of length ${b.size}`);
      }
    }
    if (differences.length) {
      throw new Error(
        `'${iceVar}' and '${liquidVar}' in ${where} ` +
        `are not stored the same way (${differences.join("; ")}), so they ` +
        "cannot be read together. cloudyview reads both condensate variables " +
        "in one pass over the file.");
    }
    iceUnits = attrString(iceDataset.attrs, "units");
  }

  const coords = {};
  for (const axis of ["x", "y", "z"]) {
    coords[axis] = findCoordinate(group, root, resolved[axis], axis);
    // Choosing one coordinate variable over another, or rescaling one, is a
    // decision about where the field IS. Stated, like the axis guesses.
    if (coords[axis].note) assumptions.push(coords[axis].note);
  }

  const units = attrString(dataset.attrs, "units");
  return {
    path,
    liquidVar,
    iceVar,
    // Carried out so the caller can put a question on screen. The worker
    // cannot ask anything itself — it has no DOM — so every "we do not know
    // which one" has to travel as data.
    variables,
    needsLiquidChoice: false,          // settled, or this would have returned
    needsIceChoice,
    // Null on a single-step file, which is most of them, and then no
    // question is asked. `values` is the time coordinate where the file has
    // one, so the rows can say when rather than only which index.
    timeDim: timeDim
      ? { name: timeDim.name, size: timeDim.size,
          values: readCoordValues(group, root, timeDim.name, timeDim.size) }
      : null,
    timestep,
    timeSelect,
    needsTimestepChoice: timeDim !== null && !answered("timestep"),
    // Guesses the axis resolution made, in plain sentences. Empty on a file
    // that named or declared all three; non-empty means the load toast has
    // something it MUST say (see loadFileScene) rather than something it may.
    assumptions,
    shape: [resolved.x.size, resolved.y.size, resolved.z.size],
    storageShape: dataset.shape,
    // Which storage axis each of x, y, z lives on — the permutation the
    // upload has to undo.
    storageAxis: { x: resolved.x.axis, y: resolved.y.axis, z: resolved.z.axis },
    droppedAxes: dropped,
    dimNames: { x: resolved.x.name, y: resolved.y.name, z: resolved.z.name },
    // Which VARIABLE each axis's coordinates came from, which is not always
    // the dimension's name — see findCoordinate. Carried out because it is
    // exactly what `--x-coord`/`--y-coord`/`--z-coord` would have to say for
    // a terminal render to place the field where this one did.
    coordNames: { x: coords.x.name, y: coords.y.name, z: coords.z.name },
    coords: {
      x: Array.from(coords.x.values),
      y: Array.from(coords.y.values),
      z: Array.from(coords.z.values),
    },
    units,
    unitsKnown: units !== null,
    // Asked about separately from the liquid variable's. Inheriting them was
    // a factor-of-1000 error waiting to happen: qc in g/kg beside a qi with
    // no units that is really kg/kg renders ice a thousand times too thin,
    // and still looks entirely like a cloud.
    iceUnits,
    iceUnitsKnown: iceVar === null || iceUnits !== null,
    chunks: dataset.metadata?.chunks ?? null,
    filters: (dataset.filters ?? []).map((f) => f.name ?? String(f.id)),
    dtype: dataset.dtype ?? null,
  };
}

/**
 * The same description, for a group read as an ICE source alone.
 *
 * A file written to supply the ice a water run omitted often has no liquid
 * variable at all, so describeGroup — which starts by insisting on one —
 * cannot describe it. This is that path: the ice variable IS the field, and
 * everything else (dimension resolution, coordinates, units, chunking) is
 * derived from it exactly the way describeGroup derives it from the water.
 *
 * `liquidVar` comes back null on purpose. Nothing downstream should be able
 * to mistake one of these for a loadable field on its own; it is only ever
 * paired with a water description through assertSameGrid.
 */
export function describeIceGroup(root, path, choice = null) {
  const group = path ? root.get(path) : root;
  const where = path ? `group '${path}'` : "the root group";

  const variables = listVariables(group);
  if (!variables.length) {
    throw new Error(`No three-dimensional variable in ${where}.`);
  }
  const answered = Boolean(choice && "iceVar" in choice);
  const iceVar = answered
    ? choice.iceVar
    : inferCondensate(group, ICE_WATER_NAMES);
  if (iceVar && !variables.some((v) => v.name === iceVar)) {
    throw new Error(
      `'${iceVar}' was chosen as the ice condensate but ${where} has no ` +
      "such three-dimensional variable.");
  }
  // The ice variable IS the field here, so unlike describeGroup's optional
  // ice there is nothing to describe without one. The caller asks and comes
  // back rather than this failing.
  if (!iceVar) {
    return { path, variables, liquidVar: null, iceVar: null,
             needsIceChoice: true, assumptions: [] };
  }
  const dataset = group.get(iceVar);
  const dimNames = datasetDimNames(dataset);
  const { resolved, dropped, timeDims, assumptions } = resolveSpatialDims(
    dimNames, dataset.shape,
    { hints: dimHints(group, root, dimNames), override: choice?.axes ?? null });

  // The attached file has to be pinned at the same step as the field it is
  // joining; the water description's `timestep` is what says which.
  const manyStepped = timeDims.filter((d) => d.size > 1);
  if (manyStepped.length > 1) {
    throw new Error(
      `${where} has more than one time dimension with several steps ` +
      `(${manyStepped.map((d) => `${d.name}=${d.size}`).join(", ")}).`);
  }
  const timeDim = manyStepped[0] ?? null;
  const timestep = choice && "timestep" in choice ? choice.timestep : 0;
  if (timeDim && !(timestep >= 0 && timestep < timeDim.size)) {
    throw new Error(
      `Timestep ${timestep} is out of range for '${timeDim.name}' ` +
      `(${timeDim.size} steps).`);
  }
  const timeSelect = {};
  for (const d of timeDims) timeSelect[d.axis] = d === timeDim ? timestep : 0;

  const coords = {};
  for (const axis of ["x", "y", "z"]) {
    coords[axis] = findCoordinate(group, root, resolved[axis], axis);
    if (coords[axis].note) assumptions.push(coords[axis].note);
  }
  const units = attrString(dataset.attrs, "units");
  return {
    path,
    liquidVar: null,
    iceVar,
    variables,
    needsIceChoice: false,             // settled, or this would have returned
    timeDim: timeDim
      ? { name: timeDim.name, size: timeDim.size,
          values: readCoordValues(group, root, timeDim.name, timeDim.size) }
      : null,
    timestep,
    timeSelect,
    assumptions,
    shape: [resolved.x.size, resolved.y.size, resolved.z.size],
    storageShape: dataset.shape,
    storageAxis: { x: resolved.x.axis, y: resolved.y.axis, z: resolved.z.axis },
    droppedAxes: dropped,
    dimNames: { x: resolved.x.name, y: resolved.y.name, z: resolved.z.name },
    coords: {
      x: Array.from(coords.x.values),
      y: Array.from(coords.y.values),
      z: Array.from(coords.z.values),
    },
    iceUnits: units,
    iceUnitsKnown: units !== null,
    chunks: dataset.metadata?.chunks ?? null,
    filters: (dataset.filters ?? []).map((f) => f.name ?? String(f.id)),
  };
}
