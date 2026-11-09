"""The ?surfacecs experiment (web/soar/renderer.js surfaceComputeWorkgroup,
raymarch.wgsl cs_surface) is opt-in and loud.

Off, the renderer builds what it built before the experiment: no layout
visible to compute, no storage-writable surface target, every surface stage
a render pipeline. (Pixel-identity of the flag-off path was shown with
tools/pinned_still.mjs: city and day, high/low/max, byte for byte against
dev, 2026-11-09.) On, the first stage is a compute pipeline at the asked
workgroup, set through the module's overrides; a device whose limits do
not allow the workgroup, or a flag value nobody defined, throws rather than
falling back to the fragment stage.

The renderer runs under node against a device that records its calls. Needs
node, not a GPU.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

COMPUTE = 4
STORAGE_BINDING = 8

_JS = textwrap.dedent("""
    import { readFileSync } from "node:fs";
    const [search, limit] = [process.argv[1], Number(process.argv[2])];
    const log = [];
    const anyObj = () => new Proxy(function () {}, {
      get: (t, k) => (k === "then" ? undefined
                      : k === Symbol.toPrimitive ? () => 0 : anyObj()),
      apply: () => anyObj(),
    });
    globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
    globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4,
                                   STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
    globalThis.GPUBufferUsage = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4,
                                  COPY_DST: 8, INDEX: 16, VERTEX: 32,
                                  UNIFORM: 64, STORAGE: 128, INDIRECT: 256,
                                  QUERY_RESOLVE: 512 };
    globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
    globalThis.location = { search };
    const device = new Proxy({
      limits: { maxComputeInvocationsPerWorkgroup: limit,
                maxComputeWorkgroupSizeX: limit, maxComputeWorkgroupSizeY: limit },
      queue: anyObj(),
      pushErrorScope: () => {},
      popErrorScope: () => Promise.resolve(null),
    }, {
      get: (t, k) => (k in t ? t[k] : (...args) => {
        const d = args[0];
        log.push({ call: k,
                   desc: JSON.parse(JSON.stringify(d ?? null, (key, v) =>
                     (typeof v === "function" ? undefined : v))) });
        return k.endsWith("Async") ? Promise.resolve(anyObj()) : anyObj();
      }),
    });
    const { Renderer } = await import(%s);
    const scene = { city: {}, bmin: [0, 0, 0], bmax: [20000, 20000, 5000],
                    minVoxelM: 50, minVoxelNestM: 50, nested: false,
                    oceanFifDx: 0.1, oceanTileExtent: 100, surfacePeriod: 20000,
                    clockPeriod: 1638.4, oceanMaxLod: 9 };
    let error = null;
    try {
      const r = new Renderer(device, readFileSync(%s, "utf8"), scene,
                             { canvasFormat: "bgra8unorm", lensSource: "" });
      await r._makeSurfaceSet({ w: 8, h: 8 }, 2);
      const stages = r._stagePipelines({ lightCache: true, skyProbe: false });
      await Promise.all(stages);
    } catch (e) {
      error = String(e.message);
    }
    console.log(JSON.stringify({ log, error }));
""")


def run(search="", limit=1024):
    js = _JS % (json.dumps(RENDERER_JS.as_uri()), json.dumps(str(SHADER)))
    out = subprocess.run(["node", "--input-type=module", "-e", js, search,
                          str(limit)], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def calls(result, name):
    return [c["desc"] for c in result["log"] if c["call"] == name]


def visibilities(result):
    return [e["visibility"] for d in calls(result, "createBindGroupLayout")
            for e in d["entries"]]


def test_flag_off_builds_what_it_did_before():
    r = run("")
    assert r["error"] is None
    assert not any(v & COMPUTE for v in visibilities(r))
    assert not calls(r, "createComputePipelineAsync")
    assert not calls(r, "createComputePipeline")
    assert not any(d["usage"] & STORAGE_BINDING
                   for d in calls(r, "createTexture") if d)
    entries = [d["fragment"]["entryPoint"]
               for d in calls(r, "createRenderPipelineAsync")]
    assert entries == ["fs_surface", "fs_surface_refine", "fs_surface_street"]


@pytest.mark.parametrize("flag,wg", [("1", (32, 16)), ("512", (32, 16)),
                                     ("256", (16, 16))])
def test_flag_on_makes_the_first_stage_compute(flag, wg):
    r = run(f"?surfacecs={flag}")
    assert r["error"] is None
    (cs,) = calls(r, "createComputePipelineAsync")
    assert cs["compute"]["entryPoint"] == "cs_surface"
    assert cs["compute"]["constants"] == {"SURFACE_CS_WX": wg[0],
                                          "SURFACE_CS_WY": wg[1]}
    entries = [d["fragment"]["entryPoint"]
               for d in calls(r, "createRenderPipelineAsync")]
    assert entries == ["fs_surface_refine", "fs_surface_street"]
    assert any(d["usage"] & STORAGE_BINDING
               for d in calls(r, "createTexture") if d)


def test_a_device_whose_limits_refuse_the_workgroup_fails_loudly():
    r = run("?surfacecs=1", limit=256)
    assert r["error"] and "over this device's limits" in r["error"]
    assert not calls(r, "createComputePipelineAsync")
    assert not calls(r, "createRenderPipelineAsync")


@pytest.mark.parametrize("flag", ["2", "yes", "1024"])
def test_an_unknown_flag_value_is_refused(flag):
    r = run(f"?surfacecs={flag}")
    assert r["error"] and "expected 1 (or 512), 256, or 0" in r["error"]


def test_flag_zero_is_off():
    r = run("?surfacecs=0")
    assert r["error"] is None
    assert not calls(r, "createComputePipelineAsync")


def test_every_stage_pipeline_goes_through_the_one_helper():
    # The meter once built its own stage pipelines, and under the flag fed a
    # render pipeline to the compute pass (2026-11-09): a stage's render
    # pipeline is asked for in exactly one place, _stagePipeline.
    src = RENDERER_JS.read_text()
    call = "this._pipeline(entry, stageFormats(entry), spec)"
    assert src.count(call) == 1
    assert src.index(call) > src.index("  _stagePipeline(entry, spec) {")
