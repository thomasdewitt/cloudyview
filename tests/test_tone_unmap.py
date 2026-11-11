"""raymarch.wgsl's tone_unmap is tone_map run backwards, on the GPU.

The live view's steam (fs_steam_live) composites in linear light onto an
image accumulated in display values, so it needs the radiance a displayed
value came from. tone_map is invertible wherever its clamps do not bite;
there tone_unmap must return the radiance, and tone_map of what it returns
must be the displayed value. This runs the two WGSL functions as written —
cut out of raymarch.wgsl by name, over a stub of the two uniforms they read
(u.white_point, the white point; u.contrast, the contrast) — over
radiances from deep shadow through the shoulder, at the ends of every
slider's range. It also holds the two cases codex found broken in the
first version: a highlight in the shoulder where its fixed point
oscillated, and a dark value its quadratic root cancelled to 0.

Skips without a real GPU adapter. Run it through tools/gpulock.
"""

import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import soar_gpu_adapter
from .conftest import ShaderModule

WGSL = ShaderModule("ocean")

# The sliders' ends and defaults (constants.js: TONE_MAP_GAMMA_LIMITS,
# TONE_MAP_WHITE_POINT_LIMITS, CONTRAST_LIMITS; DEFAULT_* in the middle).
GAMMAS = (1.0, 1.66, 4.0)
WHITE_POINTS = (4.0, 15.0, 40.0)
CONTRASTS = (0.5, 1.0, 1.6)
EXPOSURE = 2.0


def _extract(source, name):
    m = re.search(rf"^fn {name}\(.*?^\}}", source, re.S | re.M)
    assert m, f"fn {name} is gone from raymarch.wgsl"
    return m.group(0)


def _module_source():
    source = WGSL.read_text()
    shoulder = re.search(r"^const TONE_MAP_SHOULDER: f32 = [^;]+;", source,
                         re.M)
    assert shoulder, "TONE_MAP_SHOULDER is gone from raymarch.wgsl"
    functions = "\n\n".join(_extract(source, f) for f in (
        "tone_map", "tone_unmap_channels", "tone_unmap"))
    return textwrap.dedent("""
        struct Uniforms { white_point: f32, gamma: f32, contrast: f32,
                          pad: f32, }
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var<storage, read> inp: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
        """) + shoulder.group(0) + "\n\n" + functions + textwrap.dedent("""

        // inp.w: 0 maps the radiance forward first, 1 takes a display value.
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            if (i >= arrayLength(&inp)) { return; }
            let gamma = u.gamma;
            var d = inp[i].xyz;
            if (inp[i].w < 0.5) { d = tone_map(inp[i].xyz, EXPOSURE, gamma); }
            let e = tone_unmap(d, EXPOSURE, gamma);
            out[3u * i] = vec4<f32>(d, 0.0);
            out[3u * i + 1u] = vec4<f32>(e, 0.0);
            out[3u * i + 2u] = vec4<f32>(tone_map(e, EXPOSURE, gamma), 0.0);
        }
        """).replace("EXPOSURE", f"{EXPOSURE:.1f}")


@pytest.fixture(scope="module")
def run():
    adapter = soar_gpu_adapter()
    if adapter is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    import wgpu
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=_module_source())
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"})

    def go(rows, gamma, white_point, contrast):
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        ubuf = device.create_buffer_with_data(
            data=np.array([white_point, gamma, contrast, 0],
                          dtype=np.float32).tobytes(),
            usage=wgpu.BufferUsage.UNIFORM)
        ibuf = device.create_buffer_with_data(
            data=rows.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        nbytes = 3 * rows.nbytes
        obuf = device.create_buffer(
            size=nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        bind = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0), entries=[
                {"binding": 0, "resource": {"buffer": ubuf, "offset": 0,
                                            "size": 16}},
                {"binding": 1, "resource": {"buffer": ibuf, "offset": 0,
                                            "size": rows.nbytes}},
                {"binding": 2, "resource": {"buffer": obuf, "offset": 0,
                                            "size": nbytes}},
            ])
        enc = device.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bind)
        cpass.dispatch_workgroups((len(rows) + 63) // 64)
        cpass.end()
        device.queue.submit([enc.finish()])
        out = np.frombuffer(device.queue.read_buffer(obuf), dtype=np.float32)
        d, e, d2 = out.reshape(len(rows), 3, 4)[..., :3].transpose(1, 0, 2)
        return d, e, d2
    return go


def _radiances(white_point, n=4000, seed=20261108):
    """Colours from deep shadow to past the shoulder: a level log-spaced as
    exposed radiance up to 1.2x the white point, each channel within a
    factor ~3 of it either way (saturated enough to reach the shoulder's
    chroma branch, not so saturated that high contrast clips a channel of
    nearly every one)."""
    rng = np.random.default_rng(seed)
    level = np.exp(rng.uniform(np.log(1e-4), np.log(1.2 * white_point),
                               size=(n, 1)))
    exposed = level * np.exp(rng.uniform(-1.1, 1.1, size=(n, 3)))
    return np.concatenate([exposed / EXPOSURE, np.zeros((n, 1))], axis=1)


@pytest.mark.parametrize("contrast", CONTRASTS)
@pytest.mark.parametrize("white_point", WHITE_POINTS)
@pytest.mark.parametrize("gamma", GAMMAS)
def test_tone_unmap_inverts_tone_map_where_nothing_clipped(
        run, gamma, white_point, contrast):
    rows = _radiances(white_point)
    d, e, d2 = run(rows, gamma, white_point, contrast)
    # Unclipped: every channel strictly inside the display range, and inside
    # (0, 1) before the contrast too, where tone_map clamps the curve's
    # output. That clamp bites below the white point for a saturated colour
    # in the shoulder, whose chroma-preserving branch overshoots 1.
    encoded = (d - 0.5) / contrast + 0.5
    open_ = (np.all((d > 1e-6) & (d < 1.0 - 1e-6), axis=1)
             & np.all((encoded > 1e-4) & (encoded < 1.0 - 1e-4), axis=1))
    assert open_.sum() > 300, "the sample barely reaches the open range"
    assert np.all(np.isfinite(e))
    # Back through tone_map, the displayed value to well under an 8-bit
    # step; and the radiance itself, wherever the curve's slope lets a
    # float32 display value carry it (above deep shadow at high gamma).
    assert np.max(np.abs(d2 - d)[open_]) < 2e-4
    resolved = open_ & np.all(d > 0.02, axis=1)
    rel = (np.abs(e - rows[:, :3]) / rows[:, :3]).max(axis=1)
    worst = np.argmax(np.where(resolved, rel, -1.0))
    assert rel[worst] < 2e-2, (
        f"radiance {rows[worst, :3] * EXPOSURE} (exposed) came back "
        f"{e[worst] * EXPOSURE}, through display {d[worst]}")


def test_the_cases_the_first_version_got_wrong(run):
    # A highlight in the shoulder (its fixed point oscillated there and came
    # back ~10 % dark), and a dark value at gamma 4 (its root cancelled to 0).
    rows = np.array([[0.99993, 0.15576, 0.30703, 1.0],
                     [0.03, 0.03, 0.03, 1.0]])
    for gamma in (1.66, 4.0):
        d, e, d2 = run(rows, gamma, 15.0, 1.0)
        assert np.max(np.abs(d2 - d)) < 2e-4
        assert np.all(e > 0.0)
