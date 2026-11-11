"""web/soar/scene.js loadOceanTile, driven end to end in node.

The loader runs only in the browser, so nothing in the Python suite ever
executed it: a ReferenceError on every successful load (dx declared inside
the try that frees the texture, read after it) shipped with the suite green,
and the city could not open. Here the shipped tiles are
served to it from disk through a stand-in fetch, onto a stand-in device that
records what it is asked to upload.
"""

import json
import os
import shutil
import subprocess
import textwrap

import pytest

from cloudyview.soar.surfaces import SURFACE_FRAME_TILES, surface_period_m

from tests.conftest import REPO

SOAR = REPO / "web" / "soar"

_LOAD_JS = textwrap.dedent("""
    import { readFileSync, existsSync } from "node:fs";
    import * as K from "%(soar)s/constants.js";
    import { loadOceanTile } from "%(soar)s/scene.js";

    globalThis.GPUTextureUsage = { COPY_DST: 2, TEXTURE_BINDING: 4 };
    // Serve web/soar/<path> from disk, the query string (?v=schema) ignored.
    globalThis.fetch = async (url) => {
      const path = "%(soar)s/" + String(url).split("?")[0];
      if (!existsSync(path)) return new Response(null, { status: 404 });
      return new Response(readFileSync(path));
    };
    const writes = [];
    let destroyed = false;
    let created = 0;
    const device = {
      createTexture: (desc) => { created += 1;
                                 return { desc, createView: () => ({}),
                                          destroy: () => { destroyed = true; } }; },
      queue: { writeTexture: (dst, bytes, layout, size) =>
        writes.push({ level: dst.mipLevel, bytes: bytes.byteLength,
                      bytesPerRow: layout.bytesPerRow, size }) },
    };
    const [kind, keep, base] = JSON.parse(process.env.ARGS);
    let out;
    try {
      const tile = await loadOceanTile(device, base, undefined,
        { [keep]: true, schema: K.SURFACE_TILE_SCHEMAS[kind] });
      out = { dx: tile.dx, tileExtent: tile.tileExtent,
              surfacePeriod: tile.surfacePeriod, maxLod: tile.maxLod,
              format: tile.texture.desc.format,
              cells: tile.cells?.density.length ?? null, writes, destroyed,
              frameTiles: K.SURFACE_FRAME_TILES };
    } catch (err) {
      out = { error: String(err), destroyed, created };
    }
    process.stdout.write(JSON.stringify(out));
""")


def _load(kind, keep, base=None):
    js = _LOAD_JS % {"soar": SOAR.as_posix()}
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js], capture_output=True,
        text=True, cwd=REPO,
        env={**os.environ, "ARGS": json.dumps([kind, keep, base or kind])})
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("kind, keep, channels", [
    ("city", "keepCells", 4), ("ocean", "keepNothing", 4)])
def test_a_surface_tile_loads_every_level(kind, keep, channels):
    meta = json.loads((SOAR / kind / "meta.json").read_text())
    got = _load(kind, keep)
    assert "error" not in got, got.get("error")
    assert got["dx"] == meta.get("dx_m", meta.get("cell_m"))
    assert got["tileExtent"] == meta["tile_extent_m"]
    # The frame folds at a whole number of tiles: the sea's 16, the
    # district's one — the same period the Python host folds at.
    assert got["frameTiles"] == SURFACE_FRAME_TILES
    assert got["surfacePeriod"] == pytest.approx(
        surface_period_m(kind, meta), rel=1e-15)
    assert got["maxLod"] == meta["mips"] - 1
    assert got["format"] == meta.get("format", "rgba16float")
    assert [w["level"] for w in got["writes"]] == list(range(meta["mips"]))
    for w in got["writes"]:
        side = max(1, meta["n"] >> w["level"])
        assert w["bytes"] == side * side * channels * 2
        assert w["bytesPerRow"] == side * channels * 2
    if keep == "keepCells":
        assert got["cells"] == meta["n"] ** 2
    assert not got["destroyed"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_tile_with_a_bad_clock_is_refused_before_any_texture(tmp_path):
    """The clock is read before anything is allocated: a meta.json that
    does not state it (or states 0) throws with no texture made, rather
    than after every level downloaded, leaking the chain."""
    meta = json.loads((SOAR / "ocean" / "meta.json").read_text())
    for bad in ({k: v for k, v in meta.items() if k != "clock_period_s"},
                {**meta, "clock_period_s": 0}):
        alias = tmp_path / "ocean"
        alias.mkdir(exist_ok=True)
        (alias / "meta.json").write_text(json.dumps(bad))
        got = _load("ocean", "keepNothing",
                    base=os.path.relpath(alias, SOAR))
        assert "clock_period_s" in got["error"] or "positive" in got["error"]
        assert got["created"] == 0


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_tile_that_fails_to_download_gives_its_texture_back(tmp_path):
    """A level missing part-way: the load rejects and frees the texture."""
    meta = json.loads((SOAR / "ocean" / "meta.json").read_text())
    # A meta.json of its own, naming the shipped levels by relative path
    # (no links into the repo) except the last, which does not exist.
    alias = tmp_path / "ocean"
    alias.mkdir()
    shipped = [os.path.relpath(SOAR / "ocean" / name, alias)
               for name in meta["files"][:-1]]
    (alias / "meta.json").write_text(json.dumps(
        {**meta, "files": shipped + ["missing.bin.gz"]}))
    got = _load("ocean", "keepNothing",
                base=os.path.relpath(alias, SOAR))
    assert "missing.bin.gz" in got["error"]
    assert got["destroyed"]
