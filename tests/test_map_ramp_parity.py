"""The albedo map is one picture, drawn by three different pieces of code.

A field's column albedo shows up in three places, and they are meant to be
indistinguishable: `glimpse` writes a PNG with matplotlib, soar's minimap
colours the same array into a texture in JavaScript, and turbulon-analysis
plots it for the paper. Before 2026-08-14 they were two different blues, and
the only thing holding the first two together was a comment in constants.js
saying "the same sky-blue -> white ramp basic_render uses" — which was true
when it was written and is precisely the kind of claim that stops being true
without anybody noticing, because nothing compares the two pictures.

So the ramp is pinned here instead. Not the constants — the RENDERED colour,
sampled across the whole range, because that is what a reader of the two
images actually compares. Matching stop lists that are interpolated
differently would still be two different pictures.

The accent is pinned the same way and for a sharper reason: it is written out
three times (Python, JS, and again as a literal in hud.wgsl, since a shader
constant cannot import), and two of those are unreachable from the third.

Skips when node is unavailable. Needs no GPU: this is arithmetic on an array.
"""

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from cloudyview.basic_render import ACCENT, cloud_colors

REPO = Path(__file__).resolve().parents[1]
MINIMAP_JS = REPO / "web" / "soar" / "minimap.js"
CONSTANTS_JS = REPO / "web" / "soar" / "constants.js"
HUD_WGSL = REPO / "web" / "soar" / "hud.wgsl"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not MINIMAP_JS.exists(),
    reason="needs node and web/soar/minimap.js")


_JS = textwrap.dedent("""
    import { colorizeAlbedo } from "%s";
    import { MAP_ACCENT, MAP_SKY_BLUE, MAP_CLOUD_RAMP } from "%s";

    // Drive the real texture path, not the sampler underneath it: what the
    // minimap shows is whatever colorizeAlbedo writes, rounding included.
    const n = 256;
    const albedo = new Float32Array(n);
    for (let i = 0; i < n; i++) albedo[i] = i / (n - 1);
    const rgba = colorizeAlbedo(albedo);

    process.stdout.write(JSON.stringify({
      rgb: Array.from({ length: n }, (_, i) =>
        [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]]),
      alpha: Array.from({ length: n }, (_, i) => rgba[i * 4 + 3]),
      accent: MAP_ACCENT,
      skyBlue: MAP_SKY_BLUE,
      stops: MAP_CLOUD_RAMP.map(([x]) => x),
    }));
""") % (MINIMAP_JS.as_posix(), CONSTANTS_JS.as_posix())


@pytest.fixture(scope="module")
def browser():
    """colorizeAlbedo over a 0..1 sweep, as the minimap texture would hold it."""
    script = REPO / "tests" / "_map_ramp_drive.mjs"
    script.write_text(_JS)
    try:
        out = subprocess.run(
            ["node", str(script)], capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin:/usr/local/bin"})
        if out.returncode != 0:
            raise AssertionError(f"node failed:\n{out.stderr}")
        return json.loads(out.stdout)
    finally:
        script.unlink(missing_ok=True)


def test_browser_ramp_matches_matplotlib(browser):
    """The two colourings of the same albedo agree to a rounding step.

    One step, not zero: matplotlib quantizes through a 256-entry lookup table
    while the JS interpolates continuously, so the two can land either side of
    a .5 boundary. Anything larger is a different ramp.
    """
    n = len(browser["rgb"])
    t = np.linspace(0.0, 1.0, n)
    expected = np.round(np.asarray(cloud_colors(t))[:, :3] * 255.0)
    actual = np.asarray(browser["rgb"], dtype=float)

    worst = np.abs(actual - expected).max()
    where = np.unravel_index(np.abs(actual - expected).argmax(), actual.shape)
    assert worst <= 1.0, (
        f"minimap and glimpse disagree by {worst:.0f}/255 at albedo "
        f"{t[where[0]]:.3f}, channel {'rgb'[where[1]]}: "
        f"browser {actual[where[0]].tolist()} vs "
        f"matplotlib {expected[where[0]].tolist()}")


def test_ramp_ends_are_the_documented_colours(browser):
    """Clear sky is the deep ocean blue; albedo 1 is white, not near-white."""
    assert browser["rgb"][0] == [0x06, 0x1a, 0x3c]
    assert browser["rgb"][-1] == [255, 255, 255]
    assert browser["skyBlue"] == [0x06 / 255, 0x1a / 255, 0x3c / 255]
    # The stops are the point of the ramp; a two-stop list would interpolate
    # through a mid-blue neither end asked for and still pass the ends.
    assert browser["stops"] == [0.0, 0.18, 0.38, 0.60, 0.80, 1.0]
    assert all(a == 255 for a in browser["alpha"])


def test_ramp_is_monotone_in_lightness(browser):
    """Cloud over water only reads as depth if the ramp never doubles back."""
    rgb = np.asarray(browser["rgb"], dtype=float) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = lin @ [0.2126, 0.7152, 0.0722]
    assert np.all(np.diff(lum) >= -1e-9), "the ramp dips in lightness"


def _wgsl_accent():
    """The literal in hud.wgsl, which cannot import the JS constant."""
    text = HUD_WGSL.read_text()
    m = re.search(
        r"const\s+ACCENT\s*:\s*vec3<f32>\s*=\s*vec3<f32>\(([^)]*)\)", text)
    assert m, "hud.wgsl has no ACCENT constant"
    return [eval(part.strip(), {"__builtins__": {}}, {})  # noqa: S307 - literals
            for part in m.group(1).split(",")]


def test_accent_agrees_across_python_js_and_wgsl(browser):
    """One warm colour, written in three languages, none able to see another."""
    expected = [0xe8 / 255, 0x83 / 255, 0x4a / 255]
    assert ACCENT.lower() == "#e8834a"
    assert browser["accent"] == pytest.approx(expected)
    assert _wgsl_accent() == pytest.approx(expected, abs=1e-6)


def test_accent_is_not_on_the_cloud_ramp(browser):
    """The marker has to stay legible wherever the camera happens to sit.

    It is the only warm thing in the frame by design; if it ever drifted
    toward the ramp it would disappear into some albedo or other, which is
    what the old pure red did at the white end.
    """
    ramp = np.asarray(browser["rgb"], dtype=float) / 255.0
    accent = np.asarray(browser["accent"], dtype=float)
    nearest = np.abs(ramp - accent).sum(axis=1).min()
    assert nearest > 0.35, (
        f"the accent comes within {nearest:.2f} of the cloud ramp")
