// Every string the file-loading path puts on screen.
//
// One table, so the wording is edited here rather than hunted through
// netcdf.js, worker.js, index.js, ui.js and viewer.js. Functions where a
// value is interpolated, plain strings otherwise.
//
// DRAFTS — Thomas to rewrite. The structure (which strings exist, and what
// each one is handed) is the part that is load-bearing; the words are not.

"use strict";

export const T = {

  // --- failure screen -------------------------------------------------------
  // Shown as the body under "Could not open this field."

  /** The file is netCDF-3 classic. Sniffed from the magic before h5wasm. */
  notNetcdf4: (filename) => `${filename} is netCDF-3 (classic format).`,
  notNetcdf4Advice:
    "Soar only reads netCDF-4/HDF5.",

  /** Nothing in the file has three spatial dimensions. */
  noVariables: (filename) => `${filename} has no 3D variables.`,

  /** Every group had a 3D variable, but none could be described. */
  noGroupReadable: (filename) => `Nothing in ${filename} could be read.`,

  // --- which group ----------------------------------------------------------

  groupsTitle: "Which group?",
  /** Shown above the pair rows when two levels nest. */
  groupsNestable: "Nested subdomains:",
  groupsNestRow: (outer, inner) => `${outer} + ${inner}`,
  groupsNestNote: "both, nested",
  groupsOrOne: "Just one",
  groupsRootLabel: "(root)",

  // --- which variable -------------------------------------------------------
  // The panel header is the question; the body is the status line, composed
  // from whichever of the three below apply.

  askLiquid: "Which variable is liquid condensate?",
  askIce: "Which variable is ice?",

  inferredLiquid: (name) => `Inferred liquid condensate as ${name}.`,
  inferredIce: (name) => `Inferred ice condensate as ${name}.`,
  noLiquid: "Could not infer liquid condensate variable.",
  noIce: "Could not infer ice condensate variable.",
  /** Neither role inferred — replaces the two lines above. */
  noneInferred: "Could not infer variables.",

  /** Sub-label under each variable row: shape, units, long_name. */
  variableNote: ({ shape, units, longName }) => [
    `[${shape.join(" x ")}]`,
    units ? `units ${units}` : null,
    longName || null,
  ].filter(Boolean).join(" · "),

  /** Ice question only. */
  iceNoneOption: "No ice",
  iceNoneNote: "load the field without ice",
  /** "No ice" picked in the attached file, which leaves nothing to attach. */
  noIceInFile: (filename) => `No ice variable chosen from ${filename}.`,

  // --- which timestep -------------------------------------------------------

  askTimestep: "Which timestep?",
  timestepCount: (n) => `This file has ${n} timesteps.`,
  /** Row label. `value` is the time coordinate, null when there isn't one. */
  timestepRow: (index, value) =>
    value === null ? `${index}` : `${index} — ${value}`,

  // --- which units ----------------------------------------------------------

  askUnits: "Which units?",
  unitsMissing: (variables) => `No units on ${variables.join(", ")}.`,
  unitsGkg: "g/kg",
  unitsKgkg: "kg/kg",

  // --- which coordinate units ----------------------------------------------
  // A coordinate declared units, but nothing recognized the string as a
  // length or as a known non-length. Same shape as the condensate question:
  // the file did not settle it, so the person is asked, never guessed for.

  askCoordUnits: "Which coordinate units?",
  coordUnitsMissing: (coords) =>
    `Unrecognized units on ${coords.join(", ")}.`,
  coordUnitsM: "m",
  coordUnitsKm: "km",

  // --- which dimension is which --------------------------------------------

  askAxes: "Which dimension is which?",
  askAxis: (axis) => `Which dimension is ${axis}?`,
  axisAssigned: (axis, name, size) => `${axis} = ${name} (${size})`,
  axisRow: (name, size, storageAxis) =>
    `${size} points, storage axis ${storageAxis}`,
  axesConfirm: "Load with these axes",
  axesRestart: "Start over",

  // --- ice in another file --------------------------------------------------
  // A row on the ice question rather than a panel of its own: "which variable
  // is the ice" and "the ice is in another file" answer the same question.

  iceFileChoose: "Choose an ice file…",
  iceFileChooseNote: "",

  // --- assumptions ----------------------------------------------------------
  // Guesses and substitutions the load made and stuck with. Shown on the
  // toast when the field appears — never only in the console, because each
  // one renders a perfectly plausible cloud in the wrong place or the wrong
  // shape, and being told is the only defence.

  /** An axis resolved from a coordinate variable's own metadata. */
  axisFromAttribute: (axis, name, rule) =>
    `Took '${name}' as ${axis} from its ${rule} attribute.`,
  /** Nothing in the file identified the axes, so C order was assumed. */
  axesByPosition: (names, order) =>
    `Assumed ${names.join(", ")} are ${order.join(", ")} by position ` +
    "(netCDF C order), because nothing in the file identifies them " +
    "individually. Check the field is not rendered with its axes swapped.",
  /** The manual assignment panel's answer. */
  axesByHand: (pairs) => `Axes assigned by hand: ${pairs.join(", ")}.`,
  /** A coordinate variable was preferred over the dimension's own. */
  coordChosen: (axis, name) => `Took '${name}' as the ${axis} coordinate.`,
  /** The loose last-resort sweep adopted a variable for an axis. */
  coordAdopted: (axis, name, dimName) =>
    `Took '${name}' as the ${axis} coordinate for dimension '${dimName}' — ` +
    "nothing better claimed it.",
  /** A coordinate was a length, but not in meters. */
  coordConverted: (axis) => `Converted the ${axis} coordinate to meters.`,
  /** The coordinate-units answer, applied to a unit nothing recognized. */
  coordUnitsApplied: (axis, name, declared, answer) =>
    `Read the ${axis} coordinate '${name}' (units '${declared}') as ` +
    `${answer}, as answered.`,
  /** Fill-valued voxels read as cloud-free. */
  fillAsZero: (name, count) =>
    `${count} fill-valued voxels of '${name}' read as cloud-free (zero).`,
  /** Negative condensate clamped on the way in. */
  negativeClamped: (name, count) =>
    `${count} negative values of '${name}' clamped to zero.`,
  /** An assumption made about the attached ice file rather than the field. */
  inFile: (filename, note) => `In ${filename}: ${note}`,

  // --- shared ---------------------------------------------------------------

  kicker: "open file",
  back: "Back",
  backNote: "choose a different file",
  /** Second line of every load panel: which file, and which group. */
  source: (filename, group) => group ? `${filename} — group ${group}` : filename,
};
