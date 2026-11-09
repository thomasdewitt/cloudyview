"""raymarch.wgsl's GENERATED city block is exactly what tools/compose_city.py
writes from the component files.

Edit a component and forget to re-run the composer (or have the composer
refuse the edit, and miss its error), and the shader keeps the previous
component while the source says otherwise: every render, gate and review of
that tree then tests code that is not the code on the page. It happened
once (2026-09-24, a name clash the composer rejected). This test needs no
GPU: it only rebuilds the text and compares.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import compose_city  # noqa: E402


def test_generated_block_matches_the_components():
    _, expected = compose_city.composed_text()
    actual = compose_city.SHADER.read_text()
    assert actual == expected, (
        "raymarch.wgsl's GENERATED city block is stale: re-run "
        "`tools/gpulock uv run python tools/compose_city.py` (and read its output — it "
        "refuses a component that does not compile).")
