"""The scene clock, per surface: stated by each tile, understood by each host.

Every surface tile's meta.json says its clock: a period in seconds where
its animation repeats exactly and the hosts wrap the scene clock (the
city's steam, 2600 s), or null for a surface where nothing moves (the sea,
the desert), whose clock packs as 0 at any scene time. A missing key is an
error, not a surface without a clock, and a period of 0 is refused (a wrap
at 0 is NaN). The three places that read it — the tile loaders
(scene.js tileClockPeriod, soar_host.surface_clock_period_s), the uniform
packers (tests/test_uniform_parity.py holds the two to each other,
including the static sea's clock packing 0) and witness's --time — agree.
"""
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview.soar_host import SURFACE_DIRS, surface_clock_period_s

REPO = Path(__file__).resolve().parents[1]
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"


def _meta(surface):
    return json.loads((SURFACE_DIRS[surface] / "meta.json").read_text())


def _wgsl_scalar(src, name):
    m = re.search(rf"const {name}: f32 = ([0-9.eE+-]+);", src)
    assert m, f"{name} not found in raymarch.wgsl"
    return float(m.group(1))


def test_every_surface_states_its_clock():
    assert surface_clock_period_s("ocean", _meta("ocean")) is None
    assert surface_clock_period_s("desert", _meta("desert")) is None
    assert surface_clock_period_s("city", _meta("city")) == 2600.0


def test_a_missing_or_zero_clock_is_refused():
    with pytest.raises(KeyError, match="clock_period_s"):
        surface_clock_period_s("ocean", {})
    with pytest.raises(ValueError, match="positive or null"):
        surface_clock_period_s("city", {"clock_period_s": 0})


def test_the_city_clock_is_the_steams():
    """The city's clock wraps at its tile's period, where city-street's
    steam repeats exactly (tools/export_web_assets.py CITY_CLOCK_PERIOD_S):
    the steam states the same period, and its noise lattice — rising
    (lattice scale) x RISE_RATE cells a second, wrapping at ZWRAP cells —
    is back on its own lattice point after a whole period."""
    period = surface_clock_period_s("city", _meta("city"))
    src = SHADER.read_text()
    assert _wgsl_scalar(src, "cc_manholesteam_CLOCK_PERIOD_S") == period
    scale = float(re.search(r"vec3<f32>\([^;]*, ([0-9.]+) \* zeta\)",
                            src).group(1))
    rise = _wgsl_scalar(src, "cc_manholesteam_RISE_RATE")
    zwrap = _wgsl_scalar(src, "cc_manholesteam_ZWRAP")
    assert zwrap == round(zwrap)
    assert scale * rise * period == pytest.approx(zwrap, abs=1e-6)


@pytest.mark.parametrize("surface", ["ocean", "desert"])
def test_the_clock_is_refused_where_nothing_moves(surface):
    """witness --time over a surface with no clock would be a silent
    no-op, so it is refused; over the city it is taken (and this call then
    fails for having no field, past the clock)."""
    from cloudyview.witness import render_nested
    kw = dict(azimuth=0.0, elevation=0.0, sun_azimuth=0.0, sun_elevation=45.0,
              scene_time=5.0)
    with pytest.raises(ValueError, match="has no clock"):
        render_nested([], [0.0, 0.0, 100.0], surface=surface, **kw)
    with pytest.raises(ValueError, match="needs at least one level"):
        render_nested([], [0.0, 0.0, 100.0], surface="city",
                      city_offset=(8100.0, -44190.0), **kw)


_JS = textwrap.dedent("""
    import { tileClockPeriod } from "%s";
    const out = {};
    for (const [name, meta] of [["period", {clock_period_s: 2600}],
                                ["none", {clock_period_s: null}],
                                ["missing", {}],
                                ["zero", {clock_period_s: 0}]]) {
      try { out[name] = tileClockPeriod(meta, name); }
      catch (err) { out[name] = "throws: " + err.message; }
    }
    process.stdout.write(JSON.stringify(out));
""")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_browser_reads_the_clock_the_same_way():
    js = _JS % (REPO / "web" / "soar" / "scene.js").as_posix()
    proc = subprocess.run(["node", "--input-type=module", "-e", js],
                          capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        pytest.fail(f"node failed:\n{proc.stderr[-2000:]}")
    out = json.loads(proc.stdout)
    assert out["period"] == 2600
    assert out["none"] is None
    assert out["missing"].startswith("throws") and "clock_period_s" in out["missing"]
    assert out["zero"].startswith("throws") and "positive or null" in out["zero"]
