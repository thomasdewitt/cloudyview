"""A stated grid spacing is the only way past "no coordinate in meters".

Both loaders refuse an axis with no coordinate in a unit of length — none at
all (index-only output), or degrees (a regridded lat-lon file), or a level
index — because using the numbers anyway renders a plausible box with
nonsense proportions (ICON RCEMIP, 2026-08-31). The override (2026-09-23) is
the person stating the spacing of a REGULAR grid: --dx/--dy/--dz in the CLI,
the grid-spacing question in the browser. Its rules, pinned here on
synthetic metric-less files, in both languages:

- never guessed: without the answer the refusal stands, and names the flag;
- only where the refusal fires: a spacing for an axis with meters refuses;
- the vertical order goes with a vertical spacing, always asked, because a
  level index does not say which end is the ground (ICON counts down);
- uniform, and said out loud;
- native unstructured output (height x ncells) still refuses on its
  dimensionality, and the refusal does not suggest the override.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cloudyview import io

REPO = Path(__file__).resolve().parents[1]
NETCDF_JS = REPO / "web" / "soar" / "ingest" / "netcdf.js"

NX, NY, NZ = 5, 6, 4


def _field():
    # Distinct per level so an order can be checked: level k holds k + 1.
    qc = np.zeros((NZ, NY, NX), dtype=np.float32)
    for k in range(NZ):
        qc[k] = k + 1.0
    return qc


def _write(path, coords=None, dims=("lev", "lat", "lon")):
    ds = xr.Dataset({"qc": (dims, _field(), {"units": "g/kg"})},
                    coords=coords or {})
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def index_only(tmp_path):
    """No coordinate variables at all: dimension names and nothing else."""
    return _write(tmp_path / "index_only.nc")


@pytest.fixture
def degrees(tmp_path):
    """A regridded lat-lon file: degrees on x/y, a level index on z."""
    return _write(tmp_path / "degrees.nc", coords={
        "lon": ("lon", np.linspace(0, 0.4, NX), {"units": "degrees_east"}),
        "lat": ("lat", np.linspace(0, 0.5, NY), {"units": "degrees_north"}),
        "lev": ("lev", np.arange(1, NZ + 1, dtype=float), {"units": "1"}),
    })


def _load(path, **kw):
    return io.load_and_validate(path, **kw)


def test_the_refusal_stands_and_names_the_flags(index_only, degrees):
    with pytest.raises(io.NoLengthCoordinate, match="--dx, --dy, --dz"):
        _load(index_only)
    with pytest.raises(io.NoLengthCoordinate, match="--dx"):
        _load(degrees)


def test_a_stated_spacing_places_the_cells(index_only):
    r = _load(index_only, grid_spacing={"x": 100.0, "y": 200.0, "z": 50.0},
              z_first="bottom")
    np.testing.assert_allclose(r["x_coord"], (np.arange(NX) + 0.5) * 100.0)
    np.testing.assert_allclose(r["y_coord"], (np.arange(NY) + 0.5) * 200.0)
    np.testing.assert_allclose(r["z_coord"], (np.arange(NZ) + 0.5) * 50.0)


def test_degrees_and_a_level_index_are_rescued_by_the_same_answer(degrees):
    r = _load(degrees, grid_spacing={"x": 90.0, "y": 90.0, "z": 40.0},
              z_first="bottom")
    assert r["x_coord"][0] == 45.0 and r["z_coord"][-1] == 140.0


def test_top_first_turns_the_field_the_right_way_up(tmp_path, index_only):
    from cloudyview.cloudfield import load
    up = load(index_only, grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0},
              z_first="bottom")
    down = load(index_only, grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0},
                z_first="top")
    # Level k holds k + 1. Bottom-first keeps that order upward; top-first
    # means the file's first level is the top, so it lands at the top.
    col_up = up.lwc[0, 0, :]
    col_down = down.lwc[0, 0, :]
    np.testing.assert_array_equal(col_up, [1, 2, 3, 4])
    np.testing.assert_array_equal(col_down, [4, 3, 2, 1])
    np.testing.assert_allclose(up.z, down.z)


def test_the_vertical_order_is_always_asked(index_only):
    with pytest.raises(ValueError, match="--z-first"):
        _load(index_only, grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0})
    with pytest.raises(ValueError, match="--z-first only goes with --dz"):
        _load(index_only, grid_spacing={"x": 1.0, "y": 1.0}, z_first="top")


def test_a_spacing_never_replaces_real_meters(tmp_path):
    path = _write(tmp_path / "x_in_m.nc", coords={
        "lon": ("lon", (np.arange(NX) + 0.5) * 100.0, {"units": "m"}),
    })
    with pytest.raises(ValueError, match="already has meters in 'lon'"):
        _load(path, grid_spacing={"x": 100.0, "y": 1.0, "z": 1.0},
              z_first="bottom")


def test_a_spacing_and_a_named_coordinate_conflict(degrees):
    with pytest.raises(ValueError, match="--dx and --x-coord"):
        _load(degrees, x_coord_name="lon",
              grid_spacing={"x": 1.0, "y": 1.0, "z": 1.0}, z_first="bottom")


def test_the_spacing_must_be_a_positive_length(index_only):
    with pytest.raises(ValueError, match="positive number of meters"):
        _load(index_only, grid_spacing={"x": 0.0, "y": 1.0, "z": 1.0},
              z_first="bottom")


def test_native_unstructured_output_still_refuses(tmp_path):
    """ICON's native grid: (time, height, ncells). The override is for a
    regular grid that lacks a metric, and must not look like a rescue."""
    ds = xr.Dataset({"qc": (("time", "height", "ncells"),
                            np.ones((1, NZ, 40), np.float32),
                            {"units": "g/kg"})})
    path = str(tmp_path / "icon_native.nc")
    ds.to_netcdf(path)
    for kw in ({}, {"grid_spacing": {"x": 1.0, "y": 1.0, "z": 1.0},
                    "z_first": "top"}):
        with pytest.raises(ValueError, match="regular grid") as err:
            _load(path, **kw)
        assert "Unstructured output" in str(err.value)


def test_the_cli_carries_the_answer(monkeypatch):
    witness = __import__("importlib").import_module("cloudyview.witness")
    captured = {}
    monkeypatch.setattr(witness, "main",
                        lambda *a, **k: captured.update(k))
    monkeypatch.setattr(sys, "argv", [
        "witness", "f.nc", "--dx", "100", "--dy", "100", "--dz", "50",
        "--z-first", "top"])
    witness.cli()
    assert captured["grid_spacing"] == {"x": 100.0, "y": 100.0, "z": 50.0}
    assert captured["z_first"] == "top"
    monkeypatch.setattr(sys, "argv", ["witness", "f.nc"])
    witness.cli()
    assert captured["grid_spacing"] is None and captured["z_first"] is None


# --- the browser's half: the same rules in netcdf.js -------------------------

_JS = textwrap.dedent("""
    import { describeGroup } from "%s";
    const dataset = (shape, dims, attrs = {}) => ({
      shape, attrs,
      get_attached_scales: (axis) => (dims[axis] ? [dims[axis]] : []),
      metadata: { chunks: null }, filters: [], dtype: "<f4",
    });
    const coord = (values, attrs = {}) => ({
      shape: [values.length], attrs, value: Float64Array.from(values),
      get_attached_scales: () => [],
    });
    const group = (entries) => ({
      keys: () => Object.keys(entries), get: (k) => entries[k],
      type: "Group",
    });
    const dims = ["lev", "lat", "lon"];
    const qc = dataset([%d, %d, %d], dims, { units: "g/kg" });
    const indexOnly = group({ qc });
    const degrees = group({
      qc,
      lon: coord([0, 0.1, 0.2, 0.3, 0.4], { units: "degrees_east" }),
      lat: coord([0, 0.1, 0.2, 0.3, 0.4, 0.5], { units: "degrees_north" }),
      lev: coord([1, 2, 3, 4], { units: "1" }),
    });
    const xInM = group({ qc, lon: coord([50, 150, 250, 350, 450],
                                        { units: "m" }) });
    const attempt = (fn) => {
      try { return { ok: true, value: fn() }; }
      catch (err) { return { ok: false, message: String(err.message) }; }
    };
    const pick = (d) => ({
      needs: d.needsGridSpacing, needed: d.gridSpacingNeeded,
      coords: d.coords, coordNames: d.coordNames,
      assumptions: d.assumptions,
    });
    const all = { gridSpacing: { x: 100, y: 200, z: 50 } };
    process.stdout.write(JSON.stringify({
      asks: attempt(() => pick(describeGroup(indexOnly, "", null))),
      asksDegrees: attempt(() => pick(describeGroup(degrees, "", null))),
      placed: attempt(() => pick(describeGroup(
        indexOnly, "", { ...all, zFirst: "bottom" }))),
      topFirst: attempt(() => pick(describeGroup(
        degrees, "", { ...all, zFirst: "top" }))),
      noZFirst: attempt(() => pick(describeGroup(indexOnly, "", all))),
      hasMeters: attempt(() => pick(describeGroup(
        xInM, "", { ...all, zFirst: "bottom" }))),
      nonPositive: attempt(() => pick(describeGroup(
        indexOnly, "", { gridSpacing: { x: -1, y: 1, z: 1 },
                         zFirst: "top" }))),
    }));
""") % (NETCDF_JS.as_posix(), NZ, NY, NX)


@pytest.fixture(scope="module")
def browser():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    proc = subprocess.run(["node", "--input-type=module", "-e", _JS],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ})
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout)


def test_browser_asks_instead_of_refusing(browser):
    for case in ("asks", "asksDegrees"):
        d = browser[case]
        assert d["ok"], d
        assert d["value"]["needs"] is True
        assert d["value"]["coords"] is None      # no box: not nestable yet
        assert [a["axis"] for a in d["value"]["needed"]] == ["x", "y", "z"]
    found = {a["axis"]: a["found"] for a in browser["asksDegrees"]["value"]["needed"]}
    assert "degrees_east" in found["x"] and "(1)" in found["z"]


def test_browser_places_the_cells_as_python_does(browser):
    v = browser["placed"]["value"]
    assert v["needs"] is False
    np.testing.assert_allclose(v["coords"]["x"], (np.arange(NX) + 0.5) * 100)
    np.testing.assert_allclose(v["coords"]["y"], (np.arange(NY) + 0.5) * 200)
    np.testing.assert_allclose(v["coords"]["z"], (np.arange(NZ) + 0.5) * 50)
    assert v["coordNames"] == {"x": None, "y": None, "z": None}
    assert any("as entered" in a and "uniform" in a for a in v["assumptions"])
    # Top-first is written descending; the read's own flip rights it.
    z = browser["topFirst"]["value"]["coords"]["z"]
    np.testing.assert_allclose(z, ((np.arange(NZ) + 0.5) * 50)[::-1])


def test_browser_keeps_the_same_rules(browser):
    assert not browser["noZFirst"]["ok"]
    assert "first level" in browser["noZFirst"]["message"]
    assert not browser["hasMeters"]["ok"]
    assert "already has meters in 'lon'" in browser["hasMeters"]["message"]
    assert not browser["nonPositive"]["ok"]
    assert "positive number of meters" in browser["nonPositive"]["message"]
