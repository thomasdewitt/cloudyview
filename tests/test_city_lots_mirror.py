"""The flight collider's buildings are the renderer's buildings.

web/soar/citylots.js mirrors raymarch.wgsl's city_cell on the CPU so that the
paper aeroplane bounces off the wall that is on screen (collide.js). A mirror
that drifts from its original is worse than none — the flyer would bounce off
air beside a tower, or pass into one and watch it vanish, which is the bug the
collider exists to end — so this file runs the WGSL city_cell itself, on the
GPU, over a couple of thousand cells of the shipped tile, and holds the JS to
it: every discrete field exactly, every box coordinate to a millimetre plus
the float32 spacing at that coordinate (the GPU builds the city in float32,
whose spacing near the tile's far edge, 92 km out, is 8 mm). With the boxes
goes each building's facade relief depth (city_shell_style's, which the
collider grows the walls by), so the flight meets the wall the shell draws.

The shader is not compiled whole. The functions city_cell needs are cut out of
raymarch.wgsl by name, with every CITY_ constant, and wrapped in a compute
entry point over stub declarations of the two globals they read: the uniform
block's ocean_params (x = block pitch, y = tile extent) and the city tile at
ocean_normals. So a change to city_cell is tested as written, and a new
function it starts calling shows up here as a compile error naming it.

What float32 cannot promise is pow(): the height law's pow() is the GPU's
approximation, so a building whose raw height lands within an ulp of a
storey boundary can come out a storey different in the mirror. A sweep of
every cell of the shipped tile (1,048,576) on 2026-09-23 found exactly one
such building; the fixed sample here does not contain it, and the check
stays tolerance-free on the discrete fields so that anything more than that
shows up.

Skips when node or a real GPU adapter is missing. Run it through
tools/gpulock.
"""

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import soar_gpu_adapter

REPO = Path(__file__).resolve().parents[1]
WGSL = REPO / "cloudyview" / "soar" / "raymarch.wgsl"
CITYLOTS_JS = REPO / "web" / "soar" / "citylots.js"
TILE_DIR = REPO / "cloudyview" / "soar" / "city"

needs_node = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

# What city_cell calls, in dependency order, then the facade shell's style
# (whose relief depth the collider grows each building by). A function they
# start to call that is not listed here fails the compile with its name in
# the error.
WGSL_FUNCTIONS = ["pcg2d", "city_u01", "city_rand4", "city_tile_cell",
                  "city_is_avenue", "city_word", "city_cell",
                  "city_escape_building", "city_balcony_building",
                  "city_core_building",
                  "city_shell_style", "city_crown_none",
                  "city_crown_overlap", "city_crown"]
WGSL_STRUCTS = ["CityCell", "CityShellStyle", "CityCrown"]

# The fields compared, in the order the compute shader writes them.
DISCRETE = ["built", "tiers", "arch", "has_mast", "has_frustum", "merged",
            "land_use", "win_style"]
VEC3 = ["b1min", "b1max", "b2min", "b2max", "b3min", "b3max",
        "mast_min", "mast_max", "fmin", "fmax"]
VEC2 = ["plot_min", "plot_max", "lot_min", "lot_max"]
SCALAR = ["height", "top_z", "fscale"]
# Derived per building: (the shader's expression, the mirror's). The shell's
# relief depth: the shell runs only on urban walls (city_shell_refine skips
# industrial ones; yard and open lots are unbuilt).
DERIVED = {"shell_depth": (
    "select(0.0, city_shell_style(c).depth, c.land_use == CITY_USE_URBAN)",
    "shellDepth(c)")}
# The crown (city_crown, per CELL): the penthouse the collider makes solid,
# and the helipad that pre-empts it.
for _name, _gpu, _js in [
        ("crown_kind", "f32(k.kind)", "c.crown.kind"),
        ("crown_lo.x", "k.lo.x", "c.crown.lo[0]"),
        ("crown_lo.y", "k.lo.y", "c.crown.lo[1]"),
        ("crown_hi.x", "k.hi.x", "c.crown.hi[0]"),
        ("crown_hi.y", "k.hi.y", "c.crown.hi[1]"),
        ("crown_z0", "k.z0", "c.crown.z0"),
        ("crown_z1", "k.z1", "c.crown.z1")]:
    DERIVED[_name] = (_gpu, _js)
STRIDE = (len(DISCRETE) + 3 * len(VEC3) + 2 * len(VEC2) + len(SCALAR)
          + len(DERIVED))


def _meta():
    return json.loads((TILE_DIR / "meta.json").read_text())


def _tile():
    meta = _meta()
    n = meta["n"]
    raw = (TILE_DIR / "fif_mip0.bin").read_bytes()
    texels = np.frombuffer(raw, dtype=np.float16).reshape(n, n, 4)
    return meta, raw, texels.astype(np.float32)


# --- the constants -----------------------------------------------------------

def _wgsl_constants(source):
    return {m.group(1): float(m.group(3)) for m in re.finditer(
        r"^const\s+(CITY_\w+)\s*:\s*(f32|i32)\s*=\s*([-\d.eE+]+)\s*;",
        source, re.M)}


def _js_constants(source):
    return {m.group(1): float(m.group(2)) for m in re.finditer(
        r"^export\s+const\s+(CITY_\w+)\s*=\s*([-\d.eE+]+)\s*;", source, re.M)}


def test_every_mirrored_constant_is_the_shaders():
    wgsl = _wgsl_constants(WGSL.read_text())
    js = _js_constants(CITYLOTS_JS.read_text())
    assert js, "citylots.js exports no CITY_ constants"
    missing = sorted(set(js) - set(wgsl))
    assert not missing, (
        f"citylots.js mirrors constants raymarch.wgsl no longer has: {missing}")
    differ = {k: (js[k], wgsl[k]) for k in js if js[k] != wgsl[k]}
    assert not differ, f"citylots.js disagrees with raymarch.wgsl: {differ}"


# --- the cells ---------------------------------------------------------------

def _sample_cells(texels, count_per_kind=300, seed=20260923):
    """Unwrapped cell indices covering every kind of block the city makes.

    Tower lots, other multi-cell lots, industrial, yard, open, plain urban and
    tall urban, drawn separately so the rare kinds are not swamped; then a
    few hundred re-expressed a whole tile away in either direction, because
    the geometry lives in the UNWRAPPED frame while texels and seeds wrap.
    """
    n = texels.shape[0]
    rng = np.random.default_rng(seed)
    lot = (texels[..., 2] + 2048).astype(np.int64)
    st = (texels[..., 3] + 2048).astype(np.int64)
    use = (st >> 8) & 3
    span = np.maximum((lot >> 4) & 3, (lot >> 6) & 3) + 1
    density = texels[..., 0]
    kinds = {
        # The tallest peaks keep their own blocks (city_lots' rival rule),
        # so the supertalls are sampled by their density, and the tower
        # lots by span, whatever sizes the bake made.
        "supertall": (use == 0) & (density >= 3.0),
        "tower lot": span >= 3,
        "multi-cell lot": span == 2,
        "industrial": use == 1,
        "yard": use == 2,
        "open": use == 3,
        "plain urban": (use == 0) & (span == 1) & (density < 0.3),
        "tall urban": (use == 0) & (span == 1) & (density >= 0.3),
    }
    cells = []
    for name, mask in kinds.items():
        ys, xs = np.nonzero(mask)
        assert len(xs) > 0, f"the shipped tile has no {name} cells"
        pick = rng.choice(len(xs), size=min(count_per_kind, len(xs)),
                          replace=False)
        cells += [(int(xs[i]), int(ys[i])) for i in pick]
    # The tile's own edges, where a lot can straddle the seam.
    for k in range(0, n, 97):
        cells += [(0, k), (n - 1, k), (k, 0), (k, n - 1)]
    # A whole tile away: same block, geometry shifted by the extent.
    shifted = [cells[i] for i in rng.choice(len(cells), 300, replace=False)]
    for j, (x, y) in enumerate(shifted):
        sx, sy = [(-1, 0), (1, 1), (0, -1)][j % 3]
        cells.append((x + sx * n, y + sy * n))
    return cells


# --- the GPU side ------------------------------------------------------------

def _extract(source, name):
    m = re.search(rf"^fn {name}\(.*?^\}}", source, re.S | re.M)
    assert m, f"fn {name} is gone from raymarch.wgsl"
    return m.group(0)


def _compute_source():
    source = WGSL.read_text()
    consts = "\n".join(re.findall(r"^const\s+CITY_\w+\s*:.*?;", source,
                                  re.S | re.M))
    structs = []
    for name in WGSL_STRUCTS:
        m = re.search(rf"^struct {name} \{{.*?^\}}", source, re.S | re.M)
        assert m, f"struct {name} is gone from raymarch.wgsl"
        structs.append(m.group(0))
    functions = "\n\n".join(_extract(source, f) for f in WGSL_FUNCTIONS)

    def put(expr, k):
        return f"    out[o + {k}u] = {expr};"
    lines, k = [], 0
    for name in DISCRETE:
        v = f"c.{name}"
        expr = f"select(0.0, 1.0, {v})" if name in (
            "built", "has_mast", "has_frustum", "merged") else f"f32({v})"
        lines.append(put(expr, k)); k += 1
    for name in VEC3:
        for comp in "xyz":
            lines.append(put(f"c.{name}.{comp}", k)); k += 1
    for name in VEC2:
        for comp in "xy":
            lines.append(put(f"c.{name}.{comp}", k)); k += 1
    for name in SCALAR:
        lines.append(put(f"c.{name}", k)); k += 1
    for gpu_expr, _ in DERIVED.values():
        lines.append(put(gpu_expr, k)); k += 1
    assert k == STRIDE
    writes = "\n".join(lines)
    return textwrap.dedent("""
        struct Uniforms { ocean_params: vec4<f32>, }
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var ocean_normals: texture_2d<f32>;
        @group(0) @binding(2) var<storage, read> cells: array<vec2<i32>>;
        @group(0) @binding(3) var<storage, read_write> out: array<f32>;
        """) + consts + "\n\n" + "\n\n".join(structs) + "\n\n" + functions + \
        textwrap.dedent(f"""

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
            let i = gid.x;
            if (i >= arrayLength(&cells)) {{ return; }}
            let c = city_cell(cells[i]);
            let k = city_crown(c, c.cell);
            let o = i * {STRIDE}u;
        """) + writes + "\n}\n"


def _gpu_cells(adapter, raw, meta, cells):
    import wgpu
    n = meta["n"]
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=_compute_source())
    tex = device.create_texture(
        size=(n, n, 1), format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    device.queue.write_texture(
        {"texture": tex, "mip_level": 0, "origin": (0, 0, 0)}, raw,
        {"offset": 0, "bytes_per_row": 8 * n, "rows_per_image": n},
        (n, n, 1))
    params = np.array([meta["cell_m"], meta["tile_extent_m"], 0.0, 0.0],
                      dtype=np.float32)
    ubuf = device.create_buffer_with_data(
        data=params.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    cbuf = device.create_buffer_with_data(
        data=np.asarray(cells, dtype=np.int32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE)
    nbytes = 4 * STRIDE * len(cells)
    obuf = device.create_buffer(
        size=nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"})
    bind = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": ubuf, "offset": 0,
                                        "size": 16}},
            {"binding": 1, "resource": tex.create_view()},
            {"binding": 2, "resource": {"buffer": cbuf, "offset": 0,
                                        "size": cbuf.size}},
            {"binding": 3, "resource": {"buffer": obuf, "offset": 0,
                                        "size": nbytes}},
        ])
    enc = device.create_command_encoder()
    cpass = enc.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind)
    cpass.dispatch_workgroups((len(cells) + 63) // 64)
    cpass.end()
    device.queue.submit([enc.finish()])
    out = np.frombuffer(device.queue.read_buffer(obuf), dtype=np.float32)
    return out.reshape(len(cells), STRIDE).astype(np.float64)


# --- the JS side -------------------------------------------------------------

def _js_cells(meta, cells):
    fields = json.dumps({"discrete": DISCRETE, "vec3": VEC3, "vec2": VEC2,
                         "scalar": SCALAR})
    js = textwrap.dedent(f"""
        import {{ readFileSync }} from "node:fs";
        import {{ CityTile, cityCell, shellDepth }} from "{CITYLOTS_JS.as_uri()}";
        const bytes = readFileSync("{(TILE_DIR / 'fif_mip0.bin').as_posix()}");
        const tile = CityTile.fromHalfBytes(
          new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
          {meta["n"]}, {meta["cell_m"]});
        const F = {fields};
        const cells = JSON.parse(readFileSync(0, "utf8"));
        const rows = cells.map(([x, y]) => {{
          const c = cityCell(tile, x, y);
          return [
            ...F.discrete.map((k) => Number(c[k])),
            ...F.vec3.flatMap((k) => c[k]),
            ...F.vec2.flatMap((k) => c[k]),
            ...F.scalar.map((k) => c[k]),
            {", ".join(js for _, js in DERIVED.values())},
          ];
        }});
        process.stdout.write(JSON.stringify(rows));
    """)
    out = subprocess.run(["node", "--input-type=module", "--eval", js],
                         input=json.dumps(cells), capture_output=True,
                         text=True, check=True)
    return np.asarray(json.loads(out.stdout), dtype=np.float64)


def _column_names():
    names = list(DISCRETE)
    names += [f"{v}.{c}" for v in VEC3 for c in "xyz"]
    names += [f"{v}.{c}" for v in VEC2 for c in "xy"]
    names += list(SCALAR)
    names += list(DERIVED)
    return names


@needs_node
@pytest.mark.gpu
def test_js_city_cell_matches_the_wgsl_on_the_gpu():
    adapter = soar_gpu_adapter()
    if adapter is None:
        pytest.skip("no hardware GPU adapter")
    meta, raw, texels = _tile()
    cells = _sample_cells(texels)
    gpu = _gpu_cells(adapter, raw, meta, cells)
    js = _js_cells(meta, cells)
    assert gpu.shape == js.shape

    names = _column_names()
    nd = len(DISCRETE)
    # Discrete fields: exactly.
    bad = np.nonzero(np.any(gpu[:, :nd] != js[:, :nd], axis=1))[0]
    # Geometry: a millimetre plus the float32 spacing at the coordinate —
    # 1e-6 relative is eight ulps, and the GPU's own boxes are no finer.
    tol = 1e-3 + 1e-6 * np.abs(gpu[:, nd:])
    far = np.abs(gpu[:, nd:] - js[:, nd:]) > tol
    bad = np.union1d(bad, np.nonzero(np.any(far, axis=1))[0])
    report = []
    for i in bad[:12]:
        cols = [j for j in range(STRIDE)
                if (j < nd and gpu[i, j] != js[i, j])
                or (j >= nd and abs(gpu[i, j] - js[i, j])
                    > 1e-3 + 1e-6 * abs(gpu[i, j]))]
        report.append(f"cell {cells[i]}: " + ", ".join(
            f"{names[j]} gpu={gpu[i, j]!r} js={js[i, j]!r}" for j in cols))
    assert len(bad) == 0, (
        f"{len(bad)} of {len(cells)} cells differ between raymarch.wgsl "
        "city_cell and citylots.js:\n" + "\n".join(report))
    # And the sample really exercised what it says it did.
    built = gpu[:, DISCRETE.index("built")] == 1
    assert built.sum() > 1000 and (~built).sum() > 300
    for field in ("has_frustum", "has_mast", "merged"):
        assert (gpu[:, DISCRETE.index(field)] == 1).sum() > 20, field
    for tiers in (1, 2, 3):
        assert (gpu[:, DISCRETE.index("tiers")] == tiers).sum() > 20, tiers
    # Supertall needles: a mast up to ~10 m thick, centred on the crown.
    height = gpu[:, len(DISCRETE) + 3 * len(VEC3) + 2 * len(VEC2)]
    needles = ((gpu[:, DISCRETE.index("arch")] == 3) & (height > 1200.0)
               & (gpu[:, DISCRETE.index("has_mast")] == 1))
    assert needles.sum() > 20
    # Crowns of both kinds (the seeded sample holds 56 penthouses and 4
    # pads), and every penthouse standing on the roof.
    kind = gpu[:, names.index("crown_kind")]
    assert (kind == 1).sum() > 20 and (kind == 2).sum() >= 3, (
        (kind == 1).sum(), (kind == 2).sum())
    pent = kind == 1
    assert np.all(gpu[pent, names.index("crown_z0")] == height[pent])
