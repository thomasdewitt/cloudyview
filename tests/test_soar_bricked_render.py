"""A bricked field must render as the same picture as a dense one.

tests/test_soar_bricks.py proves the LAYOUT preserves every value: each atlas
texel equals the dense field at its wrapped global coordinate. That is a
statement about an array, and it was true of the two empty-space skips that
were built, measured and reverted as well. What it cannot say is whether the
shader decodes the layout it was handed, whether the apron really does make
trilinear exact across a brick seam, or whether the DDA skips only air.

So this renders the same field twice through the same WGSL — once dense, once
force-bricked — and compares the images. Force-bricked because no gate would
ever choose bricks for a field this small: bricking is a manual switch, and
this is the test that says what the switch does.

The two are NOT bit-identical, and the reason is worth writing down because
it looks exactly like a bug and is not one.

The march seeds two per-step random draws off its LOOP COUNTER — the
light-march jitter phase advances by the golden ratio per step, and the
sun-cone penumbra seed by a per-step increment. A skip that leaps a hundred
empty samples therefore reaches the cloud with a different counter, and every
draw after it differs. The bricked render is a different Monte Carlo
REALIZATION of the same integral, not a different answer: at 8 accumulated
frames the two differ by 0.19/255 in the mean, and the difference falls as
1/sqrt(frames) — 0.0996 at 32, 0.0465 at 128, 0.0262 at 512 — while the mean
brightness of every horizontal band agrees to four decimals throughout. That
is the signature of noise, and it is why this test accumulates rather than
comparing single frames, and why test_the_difference_is_noise_not_bias below
is the one that actually proves the point.

(Diagnosis took four wrong guesses first: step starvation, the closed-form
lattice snap, the measured run length, and an f32 tolerance. All four were
falsified by measurement in minutes each, which is the same lesson the
2026-07-17 perf pass recorded — cheap experiments beat reasoning here.)

Needs node (to run bricks.js, the one definition of the layout) and a real
GPU. Skips without either rather than pretending.
"""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import SOAR_RENDER_SETTINGS, soar_gpu_adapter
from tests.test_soar_bricks import BRICKS_JS, node_script

REPO = Path(__file__).resolve().parents[1]
FIELD = REPO / "data" / "TWPICE_subvolume_128x128_5km.nc"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not BRICKS_JS.exists() or not FIELD.exists(),
    reason="needs node, web/soar/bricks.js and the 128^2 TWP-ICE subvolume")


def brick_payload(field, brick, periodic, tile, tmp_path):
    """Run the browser's own bricks.js over `field`, in ingest's tile order."""
    nx, ny, nz = field.shape
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({
        "dims": [nx, ny, nz], "brick": list(brick), "periodic": periodic,
        "tile": list(tile),
        "field": base64.b64encode(field.astype("<f4").tobytes()).decode(),
    }))
    proc = subprocess.run(
        ["node", "--input-type=module", "-", str(payload_path)],
        input=node_script(), capture_output=True, text=True, cwd=REPO,
        timeout=600)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    page = np.frombuffer(base64.b64decode(out["pageTable"]),
                         dtype="<u4").reshape(out["pageDims"])
    atlas = np.frombuffer(base64.b64decode(out["atlas"]),
                          dtype="<f4").reshape(out["atlasDims"])
    return atlas, page, out["stats"]


@pytest.fixture(scope="module")
def level():
    """A real field, quantized to fp16 up front.

    Both paths store fp16, so quantizing before either sees it removes the one
    difference that is not what this test is about and leaves the ones that
    are.
    """
    import cloudyview as cv
    from cloudyview import optical_depth
    from cloudyview.witness import (
        ICE_NEGLIGIBLE_G_KG, RE_ICE_UM, RE_LIQUID_UM, _volume_aabb,
        crop_empty_z,
    )
    field = cv.load(str(FIELD))
    iwc = field.iwc
    if iwc is not None and float(np.max(iwc)) < ICE_NEGLIGIBLE_G_KG:
        iwc = None
    sigma = optical_depth.compute_extinction_field(
        field.lwc, field.z, re=RE_LIQUID_UM, iwc=iwc, re_ice=RE_ICE_UM)
    sigma = np.ascontiguousarray(sigma, dtype=np.float64)
    bmin, bmax = _volume_aabb(field)
    sigma, z, (lo, hi) = crop_empty_z(sigma, field.z)
    sigma = np.ascontiguousarray(sigma)
    if hi - lo + 1 < np.asarray(field.z).size:
        bmin[2] = z.min() - 0.5 * abs(z[1] - z[0])
        bmax[2] = z.max() + 0.5 * abs(z[-1] - z[-2])
    sigma = np.asarray(np.asarray(sigma, np.float16), np.float64)
    return sigma, bmin, bmax


def render(sigma, bmin, bmax, *, bricked, payload=None, brick=(8, 8, 8),
           shader_source=None, frames=128):
    """One accumulated still, through the dense or the bricked path."""
    from cloudyview.soar_host import (
        SceneState, ViewState, SoarRenderer, camera_world_origin,
    )
    from cloudyview.witness import _padded

    size = (240, 135)
    renderer = SoarRenderer(periodic=True, bricked=bricked, brick=brick,
                            shader_source=shader_source)
    try:
        if bricked:
            atlas, page, _ = payload
            renderer.upload_bricks(atlas, page)
        else:
            renderer.upload_volume(_padded(sigma))
        shape = sigma.shape
        extent = np.asarray(bmax) - np.asarray(bmin)
        min_voxel = float(np.min(extent / np.asarray(shape)))
        # render_nested's own rule: one step per voxel, both marches.
        dt = min_voxel * 1.0
        state = SceneState(
            bmin=bmin, bmax=bmax, dt_view=dt, dt_light=dt,
            periodic=True, shape=shape,
        )
        view = ViewState(
            camera_position=camera_world_origin([0.0, -1.6, 0.55], bmin, bmax),
            azimuth=0.0, elevation=-8.0, fov=70.0,
            output_size=size, render_size=size,
            sun_azimuth=SOAR_RENDER_SETTINGS["sun_azimuth"],
            sun_elevation=SOAR_RENDER_SETTINGS["sun_elevation"],
        )
        return renderer.render(state, view, frames=frames)
    finally:
        del renderer


@pytest.fixture(scope="module")
def payload(level, tmp_path_factory):
    sigma, _, _ = level
    return brick_payload(sigma, (8, 8, 8), True, (32, 32, sigma.shape[2]),
                         tmp_path_factory.mktemp("bricks"))


def test_the_atlas_is_smaller_than_the_field_it_came_from(payload):
    """Otherwise there is nothing to discuss: bricking would only cost."""
    _, _, stats = payload
    assert stats["occupiedBricks"] < stats["totalBricks"]
    assert stats["atlasTexels"] < stats["denseTexels"]


def test_bricked_render_matches_dense(level, payload):
    if soar_gpu_adapter() is None:
        pytest.skip("no usable GPU adapter")
    sigma, bmin, bmax = level
    dense = render(sigma, bmin, bmax, bricked=False)
    brick = render(sigma, bmin, bmax, bricked=True, payload=payload)

    assert dense.shape == brick.shape
    diff = np.abs(dense - brick)
    # In 8-bit display levels, which is the unit the standard is stated in.
    levels = diff * 255.0
    worst = float(levels.max())
    mean = float(levels.mean())
    over_one = float((levels > 1.0).mean())
    report = (f"max {worst:.3f}/255, mean {mean:.4f}/255, "
              f"{100 * over_one:.3f}% of samples over one level")
    # One display level is the threshold below which the two images are the
    # same picture. A handful of pixels may sit above it from FMA contraction
    # and the lattice snap's f32 residue; a systematic decode error would not
    # look like a handful.
    # At 128 frames the measured values are max 3.8, mean 0.047, 0.67% over a
    # level. The gates sit above that with room for driver-to-driver variation
    # but far below anything a person could see; a decode error would not fit
    # under them at any frame count.
    assert mean < 0.15, f"bricked render differs in the mean: {report}"
    assert over_one < 0.02, f"too many pixels differ visibly: {report}"
    assert worst < 8.0, f"bricked render has outlier pixels: {report}"


def test_bricked_render_is_not_blank(level, payload):
    """The cheapest way to pass the test above would be to render nothing.

    A page table read as all-zero gives empty sky everywhere, which is smooth,
    plausible, and wrong — and against a dense render of a thin field it would
    not be all that far off in the mean. So check the cloud is actually there.
    """
    if soar_gpu_adapter() is None:
        pytest.skip("no usable GPU adapter")
    sigma, bmin, bmax = level
    brick = render(sigma, bmin, bmax, bricked=True, payload=payload)
    dense = render(sigma, bmin, bmax, bricked=False)
    # The cloud is the bright structure; compare how much of it each has.
    assert brick.std() > 0.5 * dense.std(), (
        "the bricked render is far flatter than the dense one — the atlas or "
        "the page table is not being read")
    assert float(brick.max()) > 0.5 * float(dense.max())


def test_the_difference_is_noise_not_bias(level, payload):
    """Quadrupling the frames must roughly halve the difference.

    This is the test that distinguishes "a different random realization" from
    "a systematic error", and it is the only one that can. A decode fault, a
    dropped halo brick, or a skip that leapt real cloud would all survive
    accumulation — they are properties of the field, not of the sampling — so
    a difference that shrinks like 1/sqrt(n) is positive evidence that none of
    them is present.
    """
    if soar_gpu_adapter() is None:
        pytest.skip("no usable GPU adapter")
    sigma, bmin, bmax = level
    means = {}
    for frames in (16, 64, 256):
        d = np.abs(render(sigma, bmin, bmax, bricked=False, frames=frames)
                   - render(sigma, bmin, bmax, bricked=True, payload=payload,
                            frames=frames)) * 255.0
        means[frames] = float(d.mean())
    # Two 4x steps in frame count; each should shrink the mean by ~2x. Bracket
    # it loosely — the point is the trend, and a bias would hold flat at 1.0.
    for lo, hi in ((16, 64), (64, 256)):
        ratio = means[lo] / max(means[hi], 1e-9)
        assert 1.4 < ratio < 3.0, (
            f"difference went {means[lo]:.4f} -> {means[hi]:.4f} over a 4x "
            f"frame increase (ratio {ratio:.2f}); noise halves, bias does not. "
            f"all: {means}")
