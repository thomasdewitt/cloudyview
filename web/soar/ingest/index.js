// Opening an arbitrary netCDF file in the browser.
//
// Placeholder until the h5wasm path lands: it fails with a sentence rather
// than a module-not-found, so choosing "open your own file" says something
// true instead of breaking the page.

"use strict";

export async function loadFileScene() {
  const err = new Error("Opening your own file is not finished yet.");
  err.advice =
    "The demo field works today. Reading netCDF-4 straight out of a local " +
    "file is the next piece of work.";
  throw err;
}
