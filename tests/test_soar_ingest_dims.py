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

The third thing here is the variable chooser. `QN` is in the liquid-water name
list and it is SAM's TOTAL non-precipitating condensate — water and ice
together (Thomas, 2026-08-22) — so taking it as the cloud water variable is
the same class of error: right-looking, entirely wrong, silent. Finding it, or
finding two plausible candidates at all, has to become a question.

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
      resolveSpatialDims, describeGroup, condensateCandidates,
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

    // A file with qc AND QN: two plausible readings of the same field.
    const multi = group(axesOf({
      qc: dataset([96, 256, 512], dims, { units: "g/kg" }),
      QN: dataset([96, 256, 512], dims, { units: "g/kg",
                  long_name: "Non-precipitating condensate" }),
      qi: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.multiVariable = attempt(() => {
      const d = describeGroup(multi, "");
      return {
        liquidVar: d.liquidVar, iceVar: d.iceVar,
        needsLiquidChoice: d.needsLiquidChoice,
        needsIceChoice: d.needsIceChoice,
        candidates: d.liquidCandidates.map(
          (c) => ({ name: c.name, ambiguous: c.ambiguous })),
      };
    });
    // Answered: the choice is honoured and the question stops being asked.
    out.multiAnswered = attempt(() => {
      const d = describeGroup(multi, "", { liquidVar: "QN", iceVar: null });
      return { liquidVar: d.liquidVar, iceVar: d.iceVar,
               needsLiquidChoice: d.needsLiquidChoice };
    });

    // The liquid question answered, the ice question NOT yet asked. The two
    // are asked one after the other, so the ice question is always evaluated
    // on a description that already carries the liquid answer — and reading
    // "there is a choice object" as "everything is settled" would skip it.
    const bothAmbiguous = group(axesOf({
      qc: dataset([96, 256, 512], dims, { units: "g/kg" }),
      QN: dataset([96, 256, 512], dims, { units: "g/kg" }),
      qi: dataset([96, 256, 512], dims, { units: "g/kg" }),
      QICE: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.iceStillAsked = attempt(() => {
      const d = describeGroup(bothAmbiguous, "", { liquidVar: "qc" });
      return { liquidVar: d.liquidVar,
               needsLiquidChoice: d.needsLiquidChoice,
               needsIceChoice: d.needsIceChoice };
    });

    // QN alone. One candidate, no competition — and still a question,
    // because the name does not say which phase it holds.
    const lone = group(axesOf({
      QN: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.loneAmbiguous = attempt(() => {
      const d = describeGroup(lone, "");
      return { liquidVar: d.liquidVar,
               needsLiquidChoice: d.needsLiquidChoice };
    });

    // An unambiguous single candidate asks nothing — the common file must
    // not have acquired a panel.
    const plain = group(axesOf({
      qc: dataset([96, 256, 512], dims, { units: "g/kg" }),
    }));
    out.plain = attempt(() => {
      const d = describeGroup(plain, "");
      return { liquidVar: d.liquidVar, iceVar: d.iceVar,
               needsLiquidChoice: d.needsLiquidChoice,
               needsIceChoice: d.needsIceChoice,
               shape: d.shape, assumptions: d.assumptions };
    });

    // --- the attached ice file's grid check -------------------------------

    const grid = {
      shape: [512, 256, 96], storageShape: [96, 256, 512],
      storageAxis: { x: 2, y: 1, z: 0 }, droppedAxes: [],
      coords: { x: [0, 10, 20], y: [0, 10, 20], z: [0, 10, 20] },
    };
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


# --- the variable chooser ---------------------------------------------------

def test_two_plausible_water_variables_become_a_question(result):
    got = result["multiVariable"]
    assert got["ok"], got.get("message")
    assert got["value"]["needsLiquidChoice"] is True
    assert [c["name"] for c in got["value"]["candidates"]] == ["qc", "QN"]


def test_qn_is_flagged_as_total_condensate(result):
    """Water and ice together — never silently the water variable."""
    flags = {c["name"]: c["ambiguous"]
             for c in result["multiVariable"]["value"]["candidates"]}
    assert flags == {"qc": False, "QN": True}


def test_a_lone_qn_is_still_a_question(result):
    """No competition does not make it unambiguous. The name is the problem."""
    got = result["loneAmbiguous"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "QN"
    assert got["value"]["needsLiquidChoice"] is True


def test_an_answered_choice_is_honoured_and_not_re_asked(result):
    got = result["multiAnswered"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "QN"
    # null is a real answer for ice — "none of these" — not an absent one.
    assert got["value"]["iceVar"] is None
    assert got["value"]["needsLiquidChoice"] is False


def test_answering_the_water_question_does_not_swallow_the_ice_question(result):
    """Both are open; answering one must leave the other open. A blanket
    "a choice exists, so stop asking" would have silently taken the first
    ice candidate after the user picked the water variable."""
    got = result["iceStillAsked"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "qc"
    assert got["value"]["needsLiquidChoice"] is False
    assert got["value"]["needsIceChoice"] is True


def test_an_ordinary_file_asks_nothing(result):
    """The regression that matters most: a plain qc file must not have
    acquired a panel between it and the sky."""
    got = result["plain"]
    assert got["ok"], got.get("message")
    assert got["value"]["liquidVar"] == "qc"
    assert got["value"]["iceVar"] is None
    assert got["value"]["needsLiquidChoice"] is False
    assert got["value"]["needsIceChoice"] is False
    assert got["value"]["shape"] == [512, 256, 96]
    assert got["value"]["assumptions"] == []


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


def test_coordinates_that_disagree_are_refused_too(result):
    """Same shape, different grid — the mismatch that a shape check misses
    and that would read ice from the wrong altitudes."""
    got = result["coordMismatch"]
    assert got["ok"] is False
    assert "z coordinate" in got["message"]
