"""Every plume of the city's steam rises from a fixture that is drawn.

The manhole-steam component (city/components/manholesteam.wgsl) chooses its
sources among the core's manhole sites (city_manhole), and the ground shader
draws a cover at every present site (city_manhole_cover); a steaming site
the component gives a stack draws the stack itself (its cell_props). A
plume with nothing under it is a bug (one stood on a random bit of
street). So this runs the shader's own
functions, on the GPU, over every site of a stretch of streets of the
shipped tile, and holds every steaming site to a cover drawn at its centre
(city_manhole_cover through city_road, the ground's own path, at a
resolved footprint) and kept dry there by its heat — and no other site to a
dry lid: the ground and the component ask one rule (city_manhole_steam), so
a pale dry lid and its plume can never disagree.

The functions are cut out of the city module by name, with everything they
call (found by name, transitively) and every module-scope constant, over
stub declarations of the globals they read: the uniform block's
ocean_params (x the block, y the tile, w the top mip) and the city tile
with its mips and sampler (city_glow_sample's district read).

Skips without a real GPU adapter.
"""

import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import soar_gpu_adapter
from .conftest import ShaderModule

REPO = Path(__file__).resolve().parents[1]
WGSL = ShaderModule("city")
TILE_DIR = REPO / "cloudyview" / "soar" / "city"

ROOTS = ["city_manhole", "cc_manholesteam_site", "city_manhole_cover",
         "city_road"]
# Per site: present, steams, stack, cover at its centre, centre x, y, the
# share kept dry at its centre.
STRIDE = 7


def _functions(source):
    """name -> text of every module-scope fn, and the structs by name. The
    first definition wins: a section's stub twin (@stub city) comes after
    the real one and returns a stand-in."""
    fns = {}
    for m in re.finditer(r"^fn (\w+)\(.*?^\}", source, re.S | re.M):
        fns.setdefault(m.group(1), m.group(0))
    structs = {m.group(1): m.group(0) for m in re.finditer(
        r"^struct (\w+) \{.*?^\}", source, re.S | re.M)}
    return fns, structs


def _closure(fns, roots):
    """The roots and every fn they reach by name, callees first."""
    order, seen = [], set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for callee in re.findall(r"\b(\w+)\(", fns[name]):
            if callee in fns and callee != name:
                visit(callee)
        order.append(name)
    for r in roots:
        assert r in fns, f"fn {r} is gone from raymarch.wgsl"
        visit(r)
    return order


def _compute_source(source):
    fns, structs = _functions(source)
    names = _closure(fns, ROOTS)
    body = "\n\n".join(fns[n] for n in names)
    used_structs = [s for s in structs
                    if re.search(rf"\b{s}\b", body) and s != "Uniforms"]
    consts = "\n".join(re.findall(r"^const\s+\w+\s*:.*?;", source, re.S | re.M))
    return textwrap.dedent("""
        struct Uniforms { tile_cell: f32, tile_extent: f32,
                          surface_enabled: f32, tile_max_lod: f32, }
        @group(0) @binding(0) var<uniform> u: Uniforms;
        @group(0) @binding(1) var ocean_normals: texture_2d<f32>;
        @group(0) @binding(2) var ocean_samp: sampler;
        @group(0) @binding(3) var<storage, read> sites: array<vec4<i32>>;
        @group(0) @binding(4) var<storage, read_write> out: array<f32>;
        """) + consts + "\n\n" + "\n\n".join(structs[s] for s in used_structs) \
        + "\n\n" + body + textwrap.dedent(f"""

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
            let i = id.x;
            if (i >= arrayLength(&sites)) {{ return; }}
            let q = sites[i];
            let along_y = q.z == 1;
            let m = city_manhole(q.x, q.y, along_y);
            let s = cc_manholesteam_site(q.x, q.y, along_y);
            let mh = city_manhole_cover(m.xy, city_road(m.xy), 0.01);
            let cover = mh.x;
            let o = i * {STRIDE}u;
            out[o] = m.z;
            out[o + 1u] = select(0.0, 1.0, s.on);
            out[o + 2u] = select(0.0, 1.0, s.stack);
            out[o + 3u] = cover;
            out[o + 4u] = m.x;
            out[o + 5u] = m.y;
            out[o + 6u] = mh.z;
        }}
        """)


def _run(adapter, sites):
    import wgpu
    meta = json.loads((TILE_DIR / "meta.json").read_text())
    n, mips = meta["n"], meta["mips"]
    device = adapter.request_device_sync()
    module = device.create_shader_module(code=_compute_source(WGSL.read_text()))
    tex = device.create_texture(
        size=(n, n, 1), mip_level_count=mips,
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
    for level in range(mips):
        size = max(n >> level, 1)
        raw = (TILE_DIR / f"fif_mip{level}.bin").read_bytes()
        device.queue.write_texture(
            {"texture": tex, "mip_level": level, "origin": (0, 0, 0)}, raw,
            {"offset": 0, "bytes_per_row": 8 * size, "rows_per_image": size},
            (size, size, 1))
    sampler = device.create_sampler(
        address_mode_u="repeat", address_mode_v="repeat",
        mag_filter="linear", min_filter="linear", mipmap_filter="linear")
    params = np.array([meta["cell_m"], meta["tile_extent_m"], 1.0, mips - 1],
                      dtype=np.float32)
    ubuf = device.create_buffer_with_data(data=params.tobytes(),
                                          usage=wgpu.BufferUsage.UNIFORM)
    # array<vec4<i32>>: each site padded to four ints (16 bytes), or every
    # site after the first is read shifted.
    padded = np.zeros((len(sites), 4), dtype=np.int32)
    padded[:, :3] = np.asarray(sites, dtype=np.int32)
    sbuf = device.create_buffer_with_data(data=padded.tobytes(),
                                          usage=wgpu.BufferUsage.STORAGE)
    nbytes = 4 * STRIDE * len(sites)
    obuf = device.create_buffer(
        size=nbytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"})
    bind = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": 16}},
            {"binding": 1, "resource": tex.create_view()},
            {"binding": 2, "resource": sampler},
            {"binding": 3, "resource": {"buffer": sbuf, "offset": 0,
                                        "size": sbuf.size}},
            {"binding": 4, "resource": {"buffer": obuf, "offset": 0,
                                        "size": nbytes}},
        ])
    enc = device.create_command_encoder()
    cpass = enc.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind)
    cpass.dispatch_workgroups((len(sites) + 63) // 64)
    cpass.end()
    device.queue.submit([enc.finish()])
    out = np.frombuffer(device.queue.read_buffer(obuf), dtype=np.float32)
    return meta, out.reshape(len(sites), STRIDE)


@pytest.mark.gpu
def test_every_steaming_site_has_its_cover_drawn():
    adapter = soar_gpu_adapter()
    if adapter is None:
        pytest.skip("No GPU adapter available (a software rasterizer does "
                    "not count).")
    meta = json.loads((TILE_DIR / "meta.json").read_text())
    cell, tile = meta["cell_m"], meta["tile_extent_m"]
    # Every site on 96 centrelines each way (downtown and outskirts alike:
    # the district only moves the steaming share) along 6 km of each.
    ks = range(0, 96)
    js = range(0, int(6000 // 30.0))
    sites = [(k, j, ay) for ay in (0, 1) for k in ks for j in js]
    meta, out = _run(adapter, sites)
    present, steams, stack, cover = (out[:, c] for c in range(4))
    assert steams.sum() > 200, "the sample holds too few plumes to judge"
    assert np.all(present[steams > 0.5] > 0.5), \
        "a steaming site that is not a present manhole"
    bare = (steams > 0.5) & (stack < 0.5) & (cover < 0.99)
    rows = [f"k={sites[i][0]} j={sites[i][1]} along_y={sites[i][2]} "
            f"centre=({out[i, 4]:.1f}, {out[i, 5]:.1f}) cover={cover[i]:.3f}"
            for i in np.nonzero(bare)[0][:12]]
    assert not bare.any(), (
        f"{bare.sum()} of {int((steams > 0.5).sum())} steaming covers have "
        "no cover drawn at their centre:\n  " + "\n  ".join(rows))
    dried = out[:, 6]
    wet_steam = (steams > 0.5) & (dried < 0.99)
    dry_still = (steams < 0.5) & (dried > 0.0)
    assert not wet_steam.any(), \
        f"{wet_steam.sum()} steaming covers are not dry at their centre"
    assert not dry_still.any(), \
        f"{dry_still.sum()} covers that do not steam are dry at their centre"
