"""The flora harness's placement scan still compiles against the shader.

tools/desert_flora_harness.py places its cameras on plants it finds by
running the flora section's own placement (df_big_item, df_small_item) in a
compute module of its own: the flora section, lifted from raymarch.wgsl,
plus the shared functions it calls, over stubs of the rest (scan_code). A
shared function the flora section starts calling is missing there until the
harness lifts it too — true_pixel_angle, 2026-09-25, broke every harness run
— so this compiles the module and both scan entries. Any adapter will do:
nothing is dispatched. Skips without one.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _harness():
    spec = importlib.util.spec_from_file_location(
        "desert_flora_harness", REPO / "tools" / "desert_flora_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("entry", ["scan", "scan_small"])
def test_the_scan_module_compiles(entry):
    wgpu = pytest.importorskip("wgpu")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        pytest.skip("no wgpu adapter")
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=_harness().scan_code())
    device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": entry})
