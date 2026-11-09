"""The city's graffiti dissolves into exactly its own mean.

raymarch.wgsl's CITY GRAFFITI section paints a run in three rungs keyed on
the pixel footprint: FINE (the strokes), MID (each element as its box at
its expected density) and FAR (the layer's expectation along the run, in
closed form). For the paint to dissolve without a seam the three must
agree on average, which rests on constants measured from the shader's own
shapes (the painted areas) and on the palettes' means. This holds each to
the shader, through tools/city_graffiti_harness.py's kernels:

  * the palette means in the shader are its palettes' exact means;
  * the area constants are what the shapes paint, within sampling error,
    and FINE keeps that area, filtered, until MID takes over;
  * each layer's FINE, MID and FAR, averaged over many runs, agree: MID
    and FAR at every height of the reach band, FINE and FAR over it; the
    composited paint agrees with FAR over the band to second order; and a
    negative control (one area constant off by a fifth) fails.

Skips without a wgpu adapter. Run under tools/gpulock.
"""

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SHADER = (REPO / "cloudyview" / "soar" / "raymarch.wgsl").read_text()


def _harness():
    spec = importlib.util.spec_from_file_location(
        "city_graffiti_harness", REPO / "tools" / "city_graffiti_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def h():
    wgpu = pytest.importorskip("wgpu")
    if wgpu.gpu.request_adapter_sync(power_preference="high-performance") is None:
        pytest.skip("no wgpu adapter")
    return _harness()


def const(name):
    m = re.search(rf"^const {name}: f32 = ([0-9.]+);", SHADER, re.M)
    assert m, name
    return float(m.group(1))


def const3(name):
    m = re.search(rf"^const {name}: vec3<f32> =\s*vec3<f32>\(([^)]*)\);",
                  SHADER, re.M)
    assert m, name
    return np.array([float(x) for x in m.group(1).split(",")])


def test_palette_means_are_exact(h):
    got = h.palette_means()
    for name in ["HUE", "FILL", "LINE", "INK"]:
        np.testing.assert_allclose(const3(f"CITY_GRAFFITI_{name}_MEAN"),
                                   got[name], atol=2e-6, err_msg=name)


def test_area_constants_are_what_the_shapes_paint(h):
    for name, (val, se) in h.measure_means(n=1 << 19).items():
        assert abs(const(name) - val) < 4.0 * se + 2e-4, (name, val, se)


def test_fine_keeps_its_area_to_the_hand_over(h):
    """FINE's filtered strokes paint what they paint unfiltered, within a
    percent and a half, at every footprint up to where MID has taken over:
    letters and buff to CITY_GRAFFITI_MID1 of their height, tags (their
    junctions' loss put back) to CITY_GRAFFITI_TAG_MID1."""
    fs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    r = h.filter_bias(fs)
    lb = fs <= const("CITY_GRAFFITI_MID1")
    np.testing.assert_array_less(np.abs(r[lb, :2] - 1.0), 0.015)
    tg = fs <= const("CITY_GRAFFITI_TAG_MID1")
    np.testing.assert_array_less(np.abs(r[tg, 2] - 1.0), 0.015)


@pytest.mark.parametrize("layer", range(8))
def test_each_layer_dissolves_into_its_mean(h, layer):
    """MID against FAR at every height (both spread an element evenly over
    its box, so they agree exactly), and FINE against FAR over the band
    (the strokes' paint, wherever in the box it sits)."""
    v, fine, mid, far, se, se_band = h.measure_layer(layer)
    np.testing.assert_array_less(np.abs(mid - far), 4.0 * se + 1e-4)
    np.testing.assert_array_less(np.abs(fine.mean(0) - far.mean(0)),
                                 4.0 * se_band + 1e-4)


# A long street wall and a long container row; then the short runs the
# city has, whose lattices are fitted to them (city_graffiti_lattice): a
# 40 ft container's side and a dock door.
@pytest.mark.parametrize("kind,length", [(1, 200.0), (0, 200.0),
                                         (0, 12.19), (3, 3.0)])
def test_the_whole_paint_dissolves_into_its_mean(h, kind, length):
    """The layers composited: FINE against FAR over the band, per level. The
    composite of the layers' expectations is the expectation of their
    composite (independent draws); what is left is second order — where
    two layers overlap, FINE's paint sits where the strokes are and FAR's
    is spread over the boxes."""
    for lv, (v, fine, far, se) in h.measure_paint(kind,
                                                  length=length).items():
        diff = np.abs(fine.mean(0) - far.mean(0))
        assert np.all(diff < 4.0 * se.mean(0) + 3e-3), (lv, diff)


def test_negative_control(h):
    m = re.search(r"^const CITY_GRAFFITI_A_LINE: f32 = ([0-9.]+);", SHADER,
                  re.M)
    bad = SHADER.replace(m.group(0), "const CITY_GRAFFITI_A_LINE: f32 = "
                         f"{1.2 * float(m.group(1)):.4f};")
    v, fine, mid, far, se, se_band = h.measure_layer(3, src=bad)
    assert np.any(np.abs(fine.mean(0) - far.mean(0)) > 4.0 * se_band + 1e-4)
