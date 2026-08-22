"""What the upload path recognizes, and what it refuses to guess.

Two failures Thomas hit in one morning (2026-08-22) are what this pins.

The first: a SAM file whose dimensions are `(zt, yt, xt)` — cell centres, the
spelling SAM has always used — died on "Could not tell which dimensions are x,
y and z … Recognized names are x, lon, …". The list had the bare letters and
nothing else. So the name table now carries the t/s/h/c-suffixed spellings and
WRF's west_east/south_north/bottom_top, and behind the names there is a second
rule that reads the coordinate variables' own `axis`, `standard_name` and
units attributes, and behind THAT a positional last resort.

The second is the reason the positional rule is not simply a fix: a field
loaded with x and z swapped renders a completely plausible cloud, and nothing
downstream notices. So position is allowed only when nothing else said
anything at all, and when it fires it must report itself — `assumptions` is
what the load toast states on screen. A silent positional guess would be worse
than the error it replaces.

The third thing here is the variable chooser, which was rebuilt on 2026-08-22
after a UCLA file made the old shape untenable. It used to encode judgements
about names — `QN` was flagged as SAM's total condensate and forced a
question, two plausible candidates forced a question — while a file whose
variables matched no list at all could not be opened to ask anything. That is
backwards: inference either succeeds or it does not, and the case that needs a
question is precisely the one it could not handle. So there are no special
cases now. A name in the list is taken silently, first hit wins, and a miss
raises a chooser offering EVERY three-dimensional variable in the group —
which is the only thing that helps when the water is called `clw` and the ice
is called `ice`.

The fourth is the timestep. A file of several was refused outright; it is a
question, and the answer has to reach the read, where index 0 was hardcoded.

Driven under node against stub handles shaped like h5wasm's, because
netcdf.js is deliberately pure — no DOM, no wasm, no WebGPU. Skips only
without node.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NETCDF_JS = REPO / "web" / "soar" / "ingest" / "netcdf.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not NETCDF_JS.exists(),
    reason="needs node and web/soar/ingest/netcdf.js")


_JS = textwrap.dedent("""
    import {
      resolveSpatialDims, describeGroup, listVariables,
      axisFromAttrs, assertSameGrid,
    } from "%s";

    // --- an h5wasm-shaped stand-in ----------------------------------------
    //
    // Only the surface netcdf.js actually touches: keys(), get(), shape,
    // attrs, and get_attached_scales(axis) for the dimension scales netCDF-4
    // writes. Values are never read here — nothing under test reads field
    // data, which is the point of keeping netcdf.js pure.
    const dataset = (shape, dims, attrs = {}) => ({
      shape, attrs,
      get_attached_scales: (axis) => (dims[axis] ? [dims[axis]] : []),
      metadata: { chunks: null }, filters: [], dtype: "<f4",
    });
    const coord = (n, attrs = {}) => ({
      shape: [n], attrs, value: Float64Array.from({ length: n }, (_, i) => i * 10),
      get_attached_scales: () => [],
    });
    const group = (entries) => ({
      keys: () => Object.keys(entries),
      get: (k) => entries[k],
      type: "Group",
    });

    const out = {};
    const attempt = (fn) => {
      try { return { ok: true, value: fn() }; }
      catch (err) {
        return { ok: false, message: String(err.message || err),
                 axisChoice: err.axisChoice ?? null };
      }
    };

    // 1. SAM cell centres: (zt, yt, xt). The exact failure from 2026-08-22.
    out.samCentres = attempt(() => {
      const r = resolveSpatialDims(["zt", "yt", "xt"], [96, 512, 512]);
      return {
        axes: { x: r.resolved.x.name, y: r.resolved.y.name,
                z: r.resolved.z.name },
        storage: { x: r.resolved.x.axis, y: r.resolved.y.axis,
                   z: r.resolved.z.axis },
        assumptions: r.assumptions,
      };
    });

    // 2. The other spellings, each with the sizes distinct so a wrong
    // assignment cannot hide behind a cubic grid.
    out.spellings = {};
    for (const [label, names] of Object.entries({
      edges: ["zs", "ys", "xs"],
      cm1: ["zh", "yh", "xh"],
      centred_c: ["zc", "yc", "xc"],
      wrf: ["bottom_top", "south_north", "west_east"],
      wrfStag: ["bottom_top_stag", "south_north_stag", "west_east_stag"],
      upper: ["ZT", "YT", "XT"],
      geographic: ["lev", "lat", "lon"],
    })) {
      out.spellings[label] = attempt(() => {
        const r = resolveSpatialDims(names, [96, 256, 512]);
        return { x: r.resolved.x.size, y: r.resolved.y.size,
                 z: r.resolved.z.size, assumptions: r.assumptions };
      });
    }

    // 3. Metadata, when the names say nothing. `axis` beats standard_name
    // beats units, and a units rule that would hand one axis to two
    // dimensions is declined rather than resolved by whoever came first.
    out.byAxisAttr = attempt(() => {
      const hints = {
        0: axisFromAttrs({ axis: "Z" }),
        1: axisFromAttrs({ standard_name: "latitude" }),
        2: axisFromAttrs({ units: "degrees_east" }),
      };
      const r = resolveSpatialDims(["dim_a", "dim_b", "dim_c"],
                                   [96, 256, 512], { hints });
      return { x: r.resolved.x.name, y: r.resolved.y.name,
               z: r.resolved.z.name, assumptions: r.assumptions };
    });
    // Three axes all in metres: the units rule matches all three for z, so
    // it declines outright rather than giving z to the first. Nothing is
    // left claimed, so this lands on the positional rule below — which is
    // the right answer (metres on all three says nothing about order) and
    // is stated.
    out.unitsAmbiguous = attempt(() => {
      const r = resolveSpatialDims(
        ["a", "b", "c"], [96, 256, 512],
        { hints: { 0: axisFromAttrs({ units: "m" }),
                   1: axisFromAttrs({ units: "m" }),
                   2: axisFromAttrs({ units: "m" }) } });
      return { x: r.resolved.x.size, y: r.resolved.y.size,
               z: r.resolved.z.size, assumptions: r.assumptions };
    });

    // 4. Positional last resort — allowed, and never silent.
    out.positional = attempt(() => {
      const r = resolveSpatialDims(["phony", "bogus", "nonsense"],
                                   [96, 256, 512]);
      return { x: r.resolved.x.size, y: r.resolved.y.size,
               z: r.resolved.z.size, assumptions: r.assumptions };
    });

    // 5. A partial match must NOT be finished off positionally: one named
    // axis plus two unknowns is a question, and the question carries the
    // dimensions the manual panel is built from.
    out.partial = attempt(() => resolveSpatialDims(
      ["zt", "bogus", "nonsense"], [96, 256, 512]));

    // 6. The manual assignment itself.
    out.override = attempt(() => {
      const r = resolveSpatialDims(["a", "b", "c"], [96, 256, 512],
                                   { override: { x: 0, y: 1, z: 2 } });
      return { x: r.resolved.x.size, y: r.resolved.y.size,
               z: r.resolved.z.size, assumptions: r.assumptions };
    });
    out.overrideReused = attempt(() => resolveSpatialDims(
      ["a", "b", "c"], [96, 256, 512], { override: { x: 0, y: 0, z: 2 } }));

    // --- the variable chooser ---------------------------------------------

    const dims = { 0: "zt", 1: "yt", 2: "xt" };
    const axesOf = (g) => ({ zt: coord(96), yt: coord(256), xt: coord(512),
                             ...g });

    // Inference succeeds on both roles: no question, whatever else the file
    // carries. The regression that matters most — an ordinary file must not
    // have acquired a panel between it and the sky.
    const plain = group(axesOf({
      qc: dataset([96, 256, 512], dims, { units: "g/kg" }),
      qi: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.plain = attempt(() => {
      const d = describeGroup(plain, "");
      return { liquidVar: d.liquidVar, iceVar: d.iceVar,
               needsLiquidChoice: d.needsLiquidChoice,
               needsIceChoice: d.needsIceChoice,
               needsTimestepChoice: d.needsTimestepChoice,
               shape: d.shape, assumptions: d.assumptions };
    });

    // QN is SAM's TOTAL condensate, water and ice together, so it is not in
    // the liquid list at all. A run whose only condensate is QN asks, and
    // offers QN among the answers.
    const qn = group(axesOf({
      QN: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.qnNotInferred = attempt(() => {
      const d = describeGroup(qn, "");
      return { liquidVar: d.liquidVar,
               needsLiquidChoice: d.needsLiquidChoice,
               variables: d.variables.map((v) => v.name) };
    });
    // And it can still be chosen, by someone who knows their own run.
    out.qnChosen = attempt(() => {
      const d = describeGroup(qn, "", { liquidVar: "QN", iceVar: null });
      return { liquidVar: d.liquidVar,
               needsLiquidChoice: d.needsLiquidChoice };
    });

    // First hit in list order wins when several names match. `qc` precedes
    // `ql`, and that is the whole rule — no question, no ranking.
    const several = group(axesOf({
      ql: dataset([96, 256, 512], dims, { units: "g/kg" }),
      qc: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.firstHitWins = attempt(() => {
      const d = describeGroup(several, "");
      return { liquidVar: d.liquidVar,
               needsLiquidChoice: d.needsLiquidChoice };
    });

    // Nothing recognizable at all: a file of temperature. The old code threw
    // here (and, for a netCDF-3 file, threw out of libhdf5 with "error -
    // name not defined!"). It is a question now, and the question offers
    // EVERY 3-D variable rather than only names the lists know.
    const unnamed = group(axesOf({
      ta: dataset([96, 256, 512], dims, { units: "K" }),
      hus: dataset([96, 256, 512], dims, { units: "kg/kg" }),
    }));
    out.nothingInferred = attempt(() => {
      const d = describeGroup(unnamed, "");
      return { liquidVar: d.liquidVar,
               needsLiquidChoice: d.needsLiquidChoice,
               variables: d.variables.map((v) => v.name) };
    });
    // Answered, and the answer is a name no condensate list contains.
    out.unnamedAnswered = attempt(() => {
      const d = describeGroup(unnamed, "", { liquidVar: "hus", iceVar: null });
      return { liquidVar: d.liquidVar, iceVar: d.iceVar,
               needsLiquidChoice: d.needsLiquidChoice,
               needsIceChoice: d.needsIceChoice, shape: d.shape };
    });

    // Liquid inferred, ice not: the ice question, on its own.
    const warm = group(axesOf({
      clw: dataset([96, 256, 512], dims, { units: "kg/kg" }),
    }));
    out.iceAsked = attempt(() => {
      const d = describeGroup(warm, "");
      return { liquidVar: d.liquidVar, iceVar: d.iceVar,
               needsIceChoice: d.needsIceChoice,
               variables: d.variables.map((v) => v.name) };
    });
    // "No ice" is a real answer, not an absent one, and stops the asking.
    out.iceAnswered = attempt(() => {
      const d = describeGroup(warm, "", { iceVar: null });
      return { iceVar: d.iceVar, needsIceChoice: d.needsIceChoice };
    });

    // Only volumes are on offer: a 1-D profile and a (time, y, x) slice are
    // not fields however they are named.
    out.listing = attempt(() => listVariables(group(axesOf({
      qc: dataset([96, 256, 512], dims, { units: "g/kg" }),
      slice: dataset([1, 256, 512], { 0: "time", 1: "yt", 2: "xt" }),
      profile: coord(96),
    }))).map((v) => v.name));

    // --- which variable the coordinates come from -------------------------

    // netCDF-4's placeholder for a dimension with no coordinate variable.
    const PHONY = {
      NAME: "This is a netCDF dimension but not a netCDF variable.      512",
    };
    const scaled = (n, factor, attrs = {}) => ({
      shape: [n], attrs,
      value: Float64Array.from({ length: n }, (_, i) => i * factor),
      get_attached_scales: () => [],
    });

    // CM1: fields are dimensioned (nk, nj, ni) and every one of those has a
    // placeholder of the right length beside the real coordinate. Taking the
    // placeholder gave three all-zero axes and a zero-size domain.
    out.phonyScales = attempt(() => {
      const d = describeGroup(group({
        ni: coord(512, PHONY), nj: coord(256, PHONY), nk: coord(96, PHONY),
        x: scaled(512, 3000, { units: "m" }),
        y: scaled(256, 3000, { units: "m" }),
        z: scaled(96, 400, { units: "m" }),
        clw: dataset([96, 256, 512], { 0: "nk", 1: "nj", 2: "ni" },
                     { units: "g/g" }),
      }), "");
      return { coordNames: d.coordNames, xEnd: d.coords.x[511],
               assumptions: d.assumptions };
    });

    // UM: the dimension's own coordinate is dimensionless hybrid height, and
    // the metres are in a variable beside it.
    out.dimensionlessVertical = attempt(() => {
      const d = describeGroup(group({
        x: scaled(512, 3000, { units: "m" }),
        y: scaled(256, 3000, { units: "m" }),
        rholev_eta_rho: scaled(96, 0.01),
        rholev_zsea_rho: scaled(96, 400, { units: "m" }),
        clw: dataset([96, 256, 512],
                     { 0: "rholev_eta_rho", 1: "y", 2: "x" }, { units: "g/g" }),
      }), "");
      return { coordNames: d.coordNames, zEnd: d.coords.z[95],
               assumptions: d.assumptions };
    });

    // A length is a length; the numbers just are not metres yet.
    out.kilometres = attempt(() => {
      const d = describeGroup(group({
        x: scaled(512, 3, { units: "km" }),
        y: scaled(256, 3, { units: "km" }),
        z: scaled(96, 0.4, { units: "km" }),
        qc: dataset([96, 256, 512], { 0: "z", 1: "y", 2: "x" },
                    { units: "g/kg" }),
      }), "");
      return { xEnd: d.coords.x[511], zEnd: d.coords.z[95],
               assumptions: d.assumptions };
    });

    // --- the timestep -----------------------------------------------------

    const tdims = { 0: "time", 1: "zt", 2: "yt", 3: "xt" };
    const stepped = group({
      time: coord(3), zt: coord(96), yt: coord(256), xt: coord(512),
      qc: dataset([3, 96, 256, 512], tdims, { units: "g/kg" }),
    });
    out.multiStep = attempt(() => {
      const d = describeGroup(stepped, "");
      return { needsTimestepChoice: d.needsTimestepChoice,
               timeDim: d.timeDim, timestep: d.timestep,
               timeSelect: d.timeSelect, shape: d.shape };
    });
    out.stepChosen = attempt(() => {
      const d = describeGroup(stepped, "", { timestep: 2 });
      return { needsTimestepChoice: d.needsTimestepChoice,
               timestep: d.timestep, timeSelect: d.timeSelect };
    });
    out.stepOutOfRange = attempt(() => describeGroup(stepped, "",
                                                    { timestep: 3 }));
    // One step is not a question.
    out.singleStep = attempt(() => {
      const d = describeGroup(group({
        time: coord(1), zt: coord(96), yt: coord(256), xt: coord(512),
        qc: dataset([1, 96, 256, 512], tdims, { units: "g/kg" }),
      }), "");
      return { needsTimestepChoice: d.needsTimestepChoice,
               timeSelect: d.timeSelect };
    });

    // --- the attached ice file's grid check -------------------------------

    const grid = {
      shape: [512, 256, 96], storageShape: [96, 256, 512],
      storageAxis: { x: 2, y: 1, z: 0 }, droppedAxes: [], timestep: 0,
      coords: { x: [0, 10, 20], y: [0, 10, 20], z: [0, 10, 20] },
    };
    out.stepMismatch = attempt(() => assertSameGrid(
      grid, { ...grid, timestep: 1 }, "ice.nc"));
    out.sameGrid = attempt(() => {
      assertSameGrid(grid, { ...grid }, "ice.nc"); return true;
    });
    out.shapeMismatch = attempt(() => assertSameGrid(
      grid, { ...grid, shape: [512, 256, 97] }, "ice.nc"));
    out.coordMismatch = attempt(() => assertSameGrid(
      grid, { ...grid, coords: { ...grid.coords, z: [0, 10, 21] } },
      "ice.nc"));

    process.stdout.write(JSON.stringify(out));
""") % (NETCDF_JS.as_posix(),)


@pytest.fixture(scope="module")
def result():
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", _JS],
        capture_output=True, text=True, cwd=REPO, env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


# --- names ------------------------------------------------------------------

def test_sam_cell_centre_dimensions_resolve(result):
    """(zt, yt, xt) — the file that produced the error report."""
    got = result["samCentres"]
    assert got["ok"], got.get("message")
    assert got["value"]["axes"] == {"x": "xt", "y": "yt", "z": "zt"}
    # And onto the right storage axes: netCDF's slowest axis is z here.
    assert got["value"]["storage"] == {"x": 2, "y": 1, "z": 0}


def test_a_named_match_assumes_nothing(result):
    """Nothing to state on screen when the file named its own axes."""
    assert result["samCentres"]["value"]["assumptions"] == []


@pytest.mark.parametrize("label", ["edges", "cm1", "centred_c", "wrf",
                                   "wrfStag", "upper", "geographic"])
def test_the_other_spellings_resolve_by_name(result, label):
    got = result["spellings"][label]
    assert got["ok"], got.get("message")
    # Sizes, not names: a wrong assignment on a 96/256/512 grid shows up here
    # and could not on a cubic one.
    assert got["value"]["x"] == 512
    assert got["value"]["y"] == 256
    assert got["value"]["z"] == 96
    assert got["value"]["assumptions"] == []


# --- coordinate metadata ----------------------------------------------------

def test_metadata_resolves_axes_that_names_could_not(result):
    got = result["byAxisAttr"]
    assert got["ok"], got.get("message")
    assert got["value"]["z"] == "dim_a"      # axis = "Z"
    assert got["value"]["y"] == "dim_b"      # standard_name = latitude
    assert got["value"]["x"] == "dim_c"      # units = degrees_east
    # Stated, all three: a metadata match is still a rule the user did not
    # write down, and the toast says which rule fired.
    assert len(got["value"]["assumptions"]) == 3


def test_three_dimensions_all_in_metres_is_not_a_race(result):
    """The weak rule declines rather than handing z to whichever came first.

    Nothing is then claimed, so this reaches the positional rule — correctly:
    "all three are in metres" says nothing whatever about their order, and
    C order is the convention. What it must not do is silently call the first
    one z because it was first."""
    got = result["unitsAmbiguous"]
    assert got["ok"], got.get("message")
    assert got["value"]["z"] == 96 and got["value"]["x"] == 512
    assert len(got["value"]["assumptions"]) == 1
    assert "by position" in got["value"]["assumptions"][0]


# --- position ---------------------------------------------------------------

def test_position_is_the_last_resort_and_is_c_order(result):
    got = result["positional"]
    assert got["ok"], got.get("message")
    assert got["value"]["z"] == 96
    assert got["value"]["y"] == 256
    assert got["value"]["x"] == 512


def test_a_positional_guess_is_always_stated(result):
    """The whole licence for guessing. Silence here would be the bug."""
    said = result["positional"]["value"]["assumptions"]
    assert len(said) == 1
    assert "by position" in said[0]
    assert "phony, bogus, nonsense" in said[0]


def test_a_partly_named_file_asks_instead_of_filling_in_by_position(result):
    """C order is a claim about the whole tuple; applying it to leftovers is
    a coin toss wearing a convention's name."""
    got = result["partial"]
    assert got["ok"] is False
    assert got["axisChoice"] is not None
    assert "x, y" in got["message"]


def test_the_failure_carries_what_the_manual_panel_needs(result):
    dims = result["partial"]["axisChoice"]["dims"]
    assert [d["name"] for d in dims] == ["zt", "bogus", "nonsense"]
    assert [d["size"] for d in dims] == [96, 256, 512]
    assert [d["axis"] for d in dims] == [0, 1, 2]


# --- manual assignment ------------------------------------------------------

def test_a_manual_assignment_is_taken_exactly(result):
    got = result["override"]
    assert got["ok"], got.get("message")
    assert got["value"] == {"x": 96, "y": 256, "z": 512, "assumptions": []}


def test_one_dimension_cannot_be_two_axes(result):
    got = result["overrideReused"]
    assert got["ok"] is False
    assert "more than one" in got["message"]


# --- inference, and the question when it misses ------------------------------

def test_an_ordinary_file_asks_nothing(result):
    """The regression that matters most: a file whose names both infer must
    not have acquired a panel between it and the sky."""
    got = result["plain"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "qc"
    assert got["value"]["iceVar"] == "qi"
    assert got["value"]["needsLiquidChoice"] is False
    assert got["value"]["needsIceChoice"] is False
    assert got["value"]["needsTimestepChoice"] is False
    assert got["value"]["shape"] == [512, 256, 96]
    assert got["value"]["assumptions"] == []


def test_qn_is_not_a_liquid_name(result):
    """SAM's QN is water and ice together, so it is not the liquid variable.
    It was in the list with a special case bolted on to force a question;
    both are gone, which is the same intent said once."""
    got = result["qnNotInferred"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] is None
    assert got["value"]["needsLiquidChoice"] is True
    # Absent from the list is not absent from the offer.
    assert got["value"]["variables"] == ["QN"]


def test_qn_can_still_be_chosen(result):
    got = result["qnChosen"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "QN"
    assert got["value"]["needsLiquidChoice"] is False


def test_several_matching_names_take_the_first_in_list_order(result):
    """Two plausible readings used to be a question. Inference succeeding is
    not a failure, so it does not ask."""
    got = result["firstHitWins"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "qc"
    assert got["value"]["needsLiquidChoice"] is False


def test_nothing_recognizable_is_a_question_offering_every_volume(result):
    """The ta case. Not an error, and not limited to names the lists know."""
    got = result["nothingInferred"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] is None
    assert got["value"]["needsLiquidChoice"] is True
    assert got["value"]["variables"] == ["ta", "hus"]


def test_a_variable_no_list_contains_can_be_chosen(result):
    got = result["unnamedAnswered"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "hus"
    assert got["value"]["iceVar"] is None
    assert got["value"]["needsLiquidChoice"] is False
    assert got["value"]["needsIceChoice"] is False
    assert got["value"]["shape"] == [512, 256, 96]


def test_liquid_inferred_but_ice_not_asks_only_about_ice(result):
    """Thomas's example: 'Inferred liquid condensate as clw. Could not infer
    ice condensate variable. Which variable is ice?'"""
    got = result["iceAsked"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "clw"
    assert got["value"]["iceVar"] is None
    assert got["value"]["needsIceChoice"] is True
    assert got["value"]["variables"] == ["clw"]


def test_no_ice_is_an_answer_not_an_absence(result):
    got = result["iceAnswered"]
    assert got["ok"], got.get("message")
    assert got["value"]["iceVar"] is None
    assert got["value"]["needsIceChoice"] is False


def test_only_volumes_are_offered(result):
    """A 1-D profile and a (time, y, x) slice are not fields."""
    got = result["listing"]
    assert got["ok"], got.get("message")
    assert got["value"] == ["qc"]


# --- which variable the coordinates come from --------------------------------

def test_placeholder_dimension_scales_are_not_coordinates(result):
    """CM1: ni/nj/nk are netCDF-4 placeholders, all zeros, sitting beside the
    real x/y/z. Taking them gave a zero-size domain — an empty sky with no
    ocean in it."""
    got = result["phonyScales"]
    assert got["ok"], got.get("message")
    assert got["value"]["coordNames"] == {"x": "x", "y": "y", "z": "z"}
    assert got["value"]["xEnd"] == 511 * 3000
    # Nothing was overridden, only skipped: there is nothing to report.
    assert got["value"]["assumptions"] == []


def test_a_dimensionless_vertical_loses_to_one_in_metres(result):
    """UM: rholev_eta_rho runs 0 to 1, so the domain came out 0.99 m tall."""
    got = result["dimensionlessVertical"]
    assert got["ok"], got.get("message")
    assert got["value"]["coordNames"]["z"] == "rholev_zsea_rho"
    assert got["value"]["zEnd"] == 95 * 400
    assert "rholev_zsea_rho" in got["value"]["assumptions"][0]


def test_kilometres_are_converted_and_said_so(result):
    got = result["kilometres"]
    assert got["ok"], got.get("message")
    assert got["value"]["xEnd"] == 511 * 3 * 1000
    assert got["value"]["zEnd"] == 95 * 0.4 * 1000
    assert len(got["value"]["assumptions"]) == 3


# --- the timestep ------------------------------------------------------------

def test_several_timesteps_are_a_question_not_a_refusal(result):
    got = result["multiStep"]
    assert got["ok"], got.get("message")
    assert got["value"]["needsTimestepChoice"] is True
    assert got["value"]["timeDim"]["size"] == 3
    assert got["value"]["timeDim"]["values"] == [0, 10, 20]
    # The spatial shape is unaffected by which step is picked.
    assert got["value"]["shape"] == [512, 256, 96]


def test_the_chosen_step_reaches_the_read(result):
    """timeSelect is what the read pins the dropped axis at; 0 was hardcoded
    there, so every multi-step file rendered its first step or nothing."""
    got = result["stepChosen"]
    assert got["ok"], got.get("message")
    assert got["value"]["needsTimestepChoice"] is False
    assert got["value"]["timestep"] == 2
    assert got["value"]["timeSelect"] == {"0": 2}


def test_a_step_outside_the_file_is_refused(result):
    got = result["stepOutOfRange"]
    assert got["ok"] is False
    assert "out of range" in got["message"]


def test_one_timestep_asks_nothing(result):
    got = result["singleStep"]
    assert got["ok"], got.get("message")
    assert got["value"]["needsTimestepChoice"] is False
    assert got["value"]["timeSelect"] == {"0": 0}


# --- the attached ice file --------------------------------------------------

def test_an_identical_grid_is_accepted(result):
    assert result["sameGrid"]["ok"], result["sameGrid"].get("message")


def test_a_different_shape_is_refused_with_both_numbers(result):
    got = result["shapeMismatch"]
    assert got["ok"] is False
    assert "97" in got["message"] and "96" in got["message"]
    # Stated as a refusal, not repaired: cropping or interpolating someone's
    # ice onto someone else's grid is a choice about their data.
    assert "regrid" in got["message"]


def test_a_different_timestep_is_refused(result):
    """The attached file is pinned at the field's step; if it somehow is not,
    the two are read with one set of indices out of different moments."""
    got = result["stepMismatch"]
    assert got["ok"] is False
    assert "timestep" in got["message"]


def test_coordinates_that_disagree_are_refused_too(result):
    """Same shape, different grid — the mismatch that a shape check misses
    and that would read ice from the wrong altitudes."""
    got = result["coordMismatch"]
    assert got["ok"] is False
    assert "z coordinate" in got["message"]
