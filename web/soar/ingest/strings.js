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

  // --- which dimension is which --------------------------------------------

  askAxes: "Which dimension is which?",
  askAxis: (axis) => `Which dimension is ${axis}?`,
  axisAssigned: (axis, name, size) => `${axis} = ${name} (${size})`,
  axisRow: (name, size, storageAxis) =>
    `${size} points, storage axis ${storageAxis}`,
  axesConfirm: "Load with these axes",
  axesRestart: "Start over",

  // --- ice in another file --------------------------------------------------

  askIceFile: "Is there ice in another file?",
  iceFileWhy: (liquidVar) =>
    `${liquidVar} is the only condensate here. A separate ice file on the ` +
    "same grid can be read alongside.",
  iceFileSkip: "Continue without ice",
  iceFileChoose: "Choose an ice file…",
  iceFileChooseNote: "",

  // --- shared ---------------------------------------------------------------

  kicker: "open file",
  back: "Back",
  backNote: "choose a different file",
  /** Second line of every load panel: which file, and which group. */
  source: (filename, group) => group ? `${filename} — group ${group}` : filename,
};
