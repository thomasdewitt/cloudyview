"""web/soar/scene.js loadOceanTile, driven end to end in node.

The loader runs only in the browser, so nothing in the Python suite ever
executed it: a ReferenceError on every successful load (dx declared inside
the try that frees the texture, read after it) shipped with the suite green,
and neither the city nor the desert could open. Here the shipped tiles are
served to it from disk through a stand-in fetch, onto a stand-in device that
records what it is asked to upload.
"""

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
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
    const device = {
      createTexture: (desc) => ({ desc, createView: () => ({}),
                                  destroy: () => { destroyed = true; } }),
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
              heights: tile.heights?.length ?? null,
              cells: tile.cells?.density.length ?? null, writes, destroyed };
    } catch (err) {
      out = { error: String(err), destroyed };
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
    ("desert", "keepHeights", 2), ("city", "keepCells", 4)])
def test_a_surface_tile_loads_every_level(kind, keep, channels):
    meta = json.loads((SOAR / kind / "meta.json").read_text())
    got = _load(kind, keep)
    assert "error" not in got, got.get("error")
    assert got["dx"] == meta.get("dx_m", meta.get("cell_m"))
    assert got["tileExtent"] == got["surfacePeriod"] == meta["tile_extent_m"]
    assert got["maxLod"] == meta["mips"] - 1
    assert got["format"] == meta.get("format", "rgba16float")
    assert [w["level"] for w in got["writes"]] == list(range(meta["mips"]))
    for w in got["writes"]:
        side = max(1, meta["n"] >> w["level"])
        assert w["bytes"] == side * side * channels * 2
        assert w["bytesPerRow"] == side * channels * 2
    held = got["heights"] if keep == "keepHeights" else got["cells"]
    assert held == meta["n"] ** 2
    assert not got["destroyed"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_tile_that_fails_to_download_gives_its_texture_back(tmp_path):
    """A level missing part-way: the load rejects and frees the texture."""
    meta = json.loads((SOAR / "desert" / "meta.json").read_text())
    # A meta.json of its own, naming the shipped levels by relative path
    # (no links into the repo) except the last, which does not exist.
    alias = tmp_path / "desert"
    alias.mkdir()
    shipped = [os.path.relpath(SOAR / "desert" / name, alias)
               for name in meta["files"][:-1]]
    (alias / "meta.json").write_text(json.dumps(
        {**meta, "files": shipped + ["missing.bin.gz"]}))
    got = _load("desert", "keepHeights",
                base=os.path.relpath(alias, SOAR))
    assert "missing.bin.gz" in got["error"]
    assert got["destroyed"]
