"""The browser build must pack the same uniforms as the desktop engine.

`raymarch.wgsl` is shared verbatim between the two, so everything that can
make the web render differ from the desktop render lives in the host: the
camera basis, the spectral time-of-day colours, the step sizes, the row
layout. This test runs the browser's own JavaScript under node and diffs the
368-byte uniform block against `InteractiveRenderer.write_uniforms`.

Without it, "the look cannot drift" is a hope rather than a property.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("wgpu", reason="requires the 'interactive' extra")

from cloudyview import CloudField
from cloudyview.soar.engine import (
    APP_LIGHT_MARCH_LOD_DEGREES,
    APP_VIEW_STEP_LOD_DEGREES,
    DEFAULT_TONE_MAP_GAMMA,
    InteractiveRenderer,
    _min_voxel_size,
    _volume_aabb,
    choose_quality_tier,
    render_target_size,
)
from cloudyview.camera import Camera
from cloudyview.look import _spectral_lighting_colors, SUN_COLOR
from cloudyview.soar.engine import _effective_light_transfer_split
from cloudyview.angles import direction_from_azimuth_elevation

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "tests" / "web_uniform_parity.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is needed to run the browser build's JavaScript",
)


def _run_harness(payload: dict) -> dict:
    result = subprocess.run(
        ["node", str(HARNESS)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"node harness failed ({result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)


def _tiny_field():
    rng = np.random.default_rng(11)
    return CloudField(
        lwc=rng.uniform(0.001, 0.03, size=(6, 5, 4)).astype(np.float32),
        x=np.linspace(-3000.0, 3000.0, 6, dtype=np.float32),
        y=np.linspace(-2500.0, 2500.0, 5, dtype=np.float32),
        z=np.linspace(250.0, 3250.0, 4, dtype=np.float32),
    )


def _tiny_fif_normals():
    n = 4
    zeros = np.zeros((n, n), dtype=np.float32)
    ones = np.ones((n, n), dtype=np.float32)
    return (zeros, zeros.copy(), ones, 0.2)


@pytest.fixture(scope="module")
def renderer():
    return InteractiveRenderer(_tiny_field(), fif_normals=_tiny_fif_normals())


def _state_of(renderer) -> dict:
    """The scene half of the uniform block, as the JS host would hold it."""
    state = {
        "bmin": [float(v) for v in renderer.bmin],
        "bmax": [float(v) for v in renderer.bmax],
        "dtView": float(renderer.dt_view),
        "dtLight": float(renderer.dt_light),
        "periodic": bool(renderer.periodic),
        "oceanZ": float(renderer.ocean_z),
        "oceanReflectance": [float(v) for v in renderer.ocean_reflectance],
        "oceanFifDx": float(renderer.ocean_fif_dx),
        "oceanTileExtent": float(renderer.ocean_tile_extent),
        "oceanEnabled": bool(renderer.ocean_enabled),
        "oceanMaxLod": float(renderer.ocean_max_lod),
        "nested": bool(renderer.nested),
    }
    if renderer.nested:
        state.update(
            nestBmin=[float(v) for v in renderer.nest_bmin],
            nestBmax=[float(v) for v in renderer.nest_bmax],
            dtViewNest=float(renderer.dt_view_nest),
            dtLightNest=float(renderer.dt_light_nest),
        )
    return state


# (label, write_uniforms kwargs, camera kwargs) — chosen to move every part of
# the block that can move: the sun through the whole spectral range, the
# camera through the pole, the gamma across its span, and the sampling flags.
CASES = [
    ("reference sun", {}, {}),
    ("high sun", {"sun_azimuth": 180.0, "sun_elevation": 75.0}, {}),
    ("golden hour", {"sun_azimuth": 255.0, "sun_elevation": 12.0}, {}),
    ("sunset", {"sun_azimuth": 270.0, "sun_elevation": 0.5}, {}),
    ("split fade", {"sun_azimuth": 20.0, "sun_elevation": 50.0}, {}),
    ("negative azimuth", {"sun_azimuth": -70.0, "sun_elevation": 33.0}, {}),
    ("looking up", {}, {"azimuth": 137.0, "elevation": 89.0}),
    ("looking down", {}, {"azimuth": -12.0, "elevation": -89.0}),
    ("narrow fov", {}, {"fov": 30.0}),
    ("witness gamma", {"tone_map_gamma": 1.4}, {}),
    ("as-flown gamma", {"tone_map_gamma": 3.08}, {}),
    ("subpixel on", {"subpixel": True, "jitter_scale": 0.65}, {}),
    ("jitter off", {"jitter": False, "frame_index": 4096}, {}),
    ("legacy spectral", {"spectral_lighting_strength": 0.0}, {}),
    ("no lod", {"light_march_lod_degrees": 0.0, "view_step_lod_degrees": 0.0}, {}),
    ("exposure", {"exposure": 7.5, "g_hg": 0.5, "ambient_strength": 0.3}, {}),
]

# The app never calls write_uniforms with the library defaults — it always
# passes its own LOD angles and gamma. The browser ships the app's values, so
# the parity baseline is the app's call, not the library's.
APP_KWARGS = {
    "light_march_lod_degrees": APP_LIGHT_MARCH_LOD_DEGREES,
    "view_step_lod_degrees": APP_VIEW_STEP_LOD_DEGREES,
    "tone_map_gamma": DEFAULT_TONE_MAP_GAMMA,
}


def test_uniform_block_matches_python(renderer):
    output_size = (1280, 720)
    expected = []
    cases = []
    for _label, kwargs, cam_kwargs in CASES:
        merged = {**APP_KWARGS, **kwargs}
        camera = Camera(**{**dict(position=(0.15, -0.4, -0.2),
                                  azimuth=42.0, elevation=-8.0, fov=100.0),
                           **cam_kwargs})
        renderer.write_uniforms(camera, output_size, **merged)
        expected.append(np.array(renderer._current_uniform, dtype=np.float32))

        origin = renderer._current_uniform[0, :3]
        cases.append({
            "state": _state_of(renderer),
            "view": {
                # The JS host keeps world metres directly rather than relative
                # coordinates, so hand it the origin the engine just derived.
                "camera": {
                    "position": [float(v) for v in origin],
                    "azimuth": float(camera.azimuth),
                    "elevation": float(camera.elevation),
                    "fov": float(camera.fov),
                },
                "outputSize": list(output_size),
                "renderSize": list(
                    render_target_size(output_size, renderer.render_scale)),
                **{_camel(k): v for k, v in merged.items()},
            },
        })

    got = _run_harness({"cases": cases})["uniforms"]
    assert len(got) == len(expected)

    for (label, _k, _c), want, have in zip(CASES, expected, got):
        assert not isinstance(have, dict), f"{label}: JS raised {have}"
        have = np.array(have, dtype=np.float32).reshape(want.shape)
        bad = ~np.isclose(want, have, rtol=1e-6, atol=1e-7)
        if bad.any():
            rows = sorted({int(r) for r in np.argwhere(bad)[:, 0]})
            detail = "\n".join(
                f"  row {r}: python={want[r].tolist()} js={have[r].tolist()}"
                for r in rows
            )
            raise AssertionError(
                f"{label}: browser uniform block differs from the engine on "
                f"rows {rows}:\n{detail}"
            )


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def test_field_geometry_matches_python():
    """bmin/bmax and the march step scale, derived from coordinates alone."""
    fields = [
        _tiny_field(),
        CloudField(
            lwc=np.zeros((4, 4, 5), dtype=np.float32),
            x=np.linspace(0.0, 12000.0, 4),
            y=np.linspace(-6000.0, 6000.0, 4),
            # A stretched vertical grid: the top and bottom half-cells differ.
            z=np.array([50.0, 180.0, 420.0, 900.0, 1900.0]),
        ),
    ]
    payload = {"geometry": [
        {"x": f.x.tolist(), "y": f.y.tolist(), "z": f.z.tolist(),
         "shape": list(f.shape)}
        for f in fields
    ]}
    got = _run_harness(payload)["geometry"]

    for field, have in zip(fields, got):
        bmin, bmax = _volume_aabb(field)
        assert np.allclose(bmin, have["bmin"], rtol=0, atol=1e-9)
        assert np.allclose(bmax, have["bmax"], rtol=0, atol=1e-9)
        assert np.isclose(
            _min_voxel_size(field, bmin, bmax), have["minVoxel"],
            rtol=1e-12, atol=0)


def test_spectral_and_helper_math_matches_python():
    spectral_cases = [
        (20.0, 55.0, 1.0), (180.0, 75.0, 1.0), (255.0, 12.0, 1.0),
        (270.0, 0.5, 1.0), (20.0, 55.0, 0.0), (33.0, 30.0, 0.4),
        (-70.0, 5.0, 1.0), (400.0, 89.9, 1.0),
    ]
    transfer_cases = [
        (1.0, 55.0), (1.0, 45.0), (1.0, 50.0), (1.0, 60.0), (0.5, 47.5),
        (1.0, 0.5),
    ]
    tier_cases = [
        {"high": 10.0, "medium": 6.0, "low": 3.0, "potato": 1.0},
        {"high": 22.0, "medium": 14.0, "low": 8.0, "potato": 3.0},
        {"high": 40.0, "medium": 28.0, "low": 18.0, "potato": 17.0},
    ]
    sizes = [[[1280, 720], 1.0], [[1280, 720], 0.75], [[1280, 720], 0.60],
             [[1280, 720], 0.25], [[3, 3], 0.5]]

    got = _run_harness({"scalars": {
        "spectral": [list(c) for c in spectral_cases],
        "lightTransfer": [list(c) for c in transfer_cases],
        "renderTargetSizes": sizes,
        "tiers": tier_cases,
    }})["scalars"]

    for (azimuth, elevation, strength), have in zip(
            spectral_cases, got["spectral"]):
        direction = direction_from_azimuth_elevation(azimuth, elevation)
        assert np.allclose(direction, have["dir"], rtol=1e-12, atol=1e-12), (
            f"sun direction differs at az={azimuth} el={elevation}")
        want = _spectral_lighting_colors(
            tuple(float(c) for c in direction), SUN_COLOR, strength)
        for name, expected in zip(
                ("cloudSun", "ambient", "horizon", "bloom", "disc"), want):
            assert np.allclose(expected, have[name], rtol=1e-12, atol=1e-14), (
                f"{name} differs at az={azimuth} el={elevation} "
                f"strength={strength}: python={expected} js={have[name]}")

    for (strength, elevation), have in zip(transfer_cases, got["lightTransfer"]):
        assert np.isclose(
            _effective_light_transfer_split(strength, elevation), have,
            rtol=1e-12, atol=1e-14)

    for (size, scale), have in zip(sizes, got["renderTargetSizes"]):
        assert list(render_target_size(tuple(size), scale)) == have

    for times, have in zip(tier_cases, got["tiers"]):
        assert choose_quality_tier(times) == have


def test_browser_refuses_a_below_horizon_sun_in_a_periodic_domain(renderer):
    """The same refusal as the engine's, for the same reason."""
    state = _state_of(renderer)
    state["periodic"] = True
    got = _run_harness({"cases": [{
        "state": state,
        "view": {
            "camera": {"position": [0.0, 0.0, 500.0], "azimuth": 0.0,
                       "elevation": 0.0, "fov": 100.0},
            "outputSize": [640, 360], "renderSize": [640, 360],
            "sunElevation": -3.0,
        },
    }]})["uniforms"][0]
    assert isinstance(got, dict), "JS accepted a below-horizon periodic sun"
    assert "above the horizon" in got["error"]

    with pytest.raises(ValueError, match="above the horizon"):
        renderer.write_uniforms(Camera(), (640, 360), sun_elevation=-3.0)
