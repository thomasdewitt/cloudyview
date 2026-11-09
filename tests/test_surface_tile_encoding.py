"""A surface tile's stored-level encoding, and the two decoders that read it.

A tile level is stored either as its texels or, encoding "planar-byte-shuffle"
(the ocean's), as each channel's plane with every half float's low bytes
before its high bytes: soar_host.encode_surface_level writes it, and
soar_host.decode_surface_level and web/soar/scene.js decodeSurfaceLevel read
it back. They are held here to the same bytes. (Moved from the desert tile's
tests when the desert left this line, 2026-11-09; the ocean's is the tile
that uses the encoding now.)
"""

import base64
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview.soar_host import decode_surface_level, encode_surface_level

REPO = Path(__file__).resolve().parents[1]
# An ocean-shaped meta: four half-float channels, planar and byte-shuffled.
META = {"format": "rgba16float", "encoding": "planar-byte-shuffle"}


def test_the_encoding_round_trips():
    rng = np.random.default_rng(3)
    level = rng.normal(size=(8, 8, 4)).astype(np.float16)
    raw = encode_surface_level(level)
    back = np.frombuffer(decode_surface_level(raw, 8, META),
                         np.float16).reshape(8, 8, 4)
    assert np.array_equal(back, level)


_DECODE_JS = textwrap.dedent("""
    import { decodeSurfaceLevel } from "%s";
    import { readFileSync } from "node:fs";
    const raw = new Uint8Array(readFileSync(process.env.RAW));
    const out = decodeSurfaceLevel(raw, 8, JSON.parse(process.env.META));
    process.stdout.write(Buffer.from(out).toString("base64"));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_decodes_the_same_bytes(tmp_path):
    rng = np.random.default_rng(5)
    level = rng.normal(size=(8, 8, 4)).astype(np.float16)
    raw = encode_surface_level(level)
    (tmp_path / "raw.bin").write_bytes(raw)
    js = _DECODE_JS % (REPO / "web" / "soar" / "scene.js").as_posix()
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js], capture_output=True,
        text=True, cwd=REPO,
        env={**os.environ, "RAW": str(tmp_path / "raw.bin"),
             "META": json.dumps(META)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert base64.b64decode(proc.stdout) == decode_surface_level(raw, 8, META)
