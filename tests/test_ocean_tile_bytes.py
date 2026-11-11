"""The shader's numbers about the sea tile, held to the tile's bytes.

The shader (ocean.wgsl) carries two facts measured from cloudyview/soar/ocean/: the
slope variance each mip level removed (fif_slope_variance — the glint lobe
hands it back), and the bound its two normalize() calls rest on (the
@domain notes: every texel's normal has z >= 0.923). Neither is derived at
run time, so a new tile (tools/export_web_assets.py ocean --rebake) would
leave both silently stale. Here they are re-measured on the CPU.

The table is the mean-square slope (slope = -n.xy / n.z, both axes) of
level 0 less that of level k: what box-filtering to level k took away.
"""
import gzip
import json
import re

import numpy as np
import pytest

from cloudyview.soar.surfaces import (SURFACE_DIRS, decode_surface_level,
                                      surface_level_files)
from cloudyview.soar.uniforms import SceneState

from .conftest import ShaderModule

OCEAN_DIR = SURFACE_DIRS["ocean"]

SHADER = ShaderModule("ocean")
Z_BOUND = 0.923


def _levels():
    meta = json.loads((OCEAN_DIR / "meta.json").read_text())
    out = []
    for k, name in enumerate(surface_level_files(meta)):
        n = max(1, meta["n"] >> k)
        raw = gzip.decompress((OCEAN_DIR / name).read_bytes())
        texels = np.frombuffer(decode_surface_level(raw, n, meta), np.float16)
        out.append(texels.reshape(n, n, 4).astype(np.float64))
    return out


def _mean_square_slope(level):
    sx = -level[..., 0] / level[..., 2]
    sy = -level[..., 1] / level[..., 2]
    return float(np.mean(sx * sx + sy * sy))


def _shader_table():
    body = re.search(r"fn fif_slope_variance\(k: i32\) -> f32 \{(.*?)\n\}",
                     SHADER.read_text(), re.S).group(1)
    cases = {int(k): float(v) for k, v in
             re.findall(r"case (\d+): \{ return ([0-9.]+); \}", body)}
    default = float(re.search(r"default: \{ return ([0-9.]+); \}",
                              body).group(1))
    return [cases.get(k, default) for k in range(max(cases) + 2)]


def test_the_slope_variance_table_is_the_tiles():
    levels = _levels()
    table = _shader_table()
    assert len(table) == len(levels)
    top = _mean_square_slope(levels[0])
    measured = [top - _mean_square_slope(lv) for lv in levels]
    # The shader states six decimals.
    np.testing.assert_allclose(table, measured, atol=5.1e-7, rtol=0)
    # And fif_slope_variance_removed clamps its lod at the table's top.
    clamps = re.findall(r"fn fif_slope_variance_removed.*?clamp\(floor\(lod\), "
                        r"0\.0, ([0-9.]+)\);\s*let l1 = min\(l0 \+ 1\.0, ([0-9.]+)\)",
                        SHADER.read_text(), re.S)
    assert clamps and all(float(v) == len(levels) - 1 for v in clamps[0])


def test_the_scene_defaults_are_the_tiles():
    """SceneState's row 9 defaults are the sea tile's numbers (witness and
    the parity tests lean on it), so a new tile must move them too."""
    meta = json.loads((OCEAN_DIR / "meta.json").read_text())
    state = SceneState(bmin=(0, 0, 0), bmax=(1, 1, 1), dt_view=1, dt_light=1)
    assert (state.ocean_fif_dx, state.ocean_tile_extent, state.ocean_max_lod) \
        == (meta["dx_m"], meta["tile_extent_m"], meta["mips"] - 1)


def test_the_normalize_bound_is_the_tiles():
    """Bilinear filtering and the mip mix are convex, so a sampled normal's
    z is at least the smallest texel's — the @domain notes' bound."""
    assert min(float(lv[..., 2].min()) for lv in _levels()) >= Z_BOUND
    notes = re.findall(r"// @domain: every texel of every mip has z >= "
                       r"([0-9.]+)", SHADER.read_text())
    # On the normalize() of the tile's mipped normals.
    assert len(notes) == 1
    assert all(float(v) == pytest.approx(Z_BOUND) for v in notes)
